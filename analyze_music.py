import numpy as np
import essentia.standard as es
import os
import logging
import sqlite3
import hashlib
import signal
import gc
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

def timeout(seconds=30, error_message="Processing timed out"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutException(error_message)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

class AudioAnalyzer:
    def __init__(self, cache_file=None, timeout_seconds=30):
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = cache_file or os.path.join(cache_dir, 'audio_analysis.db')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.conn = None
        self._cache_hits = 0
        self._cache_misses = 0

    def _init_db(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.cache_file, timeout=30)
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA cache_size = -8000")

    def _get_file_info(self, filepath):
        try:
            stat = os.stat(filepath)
            return {
                'file_hash': hashlib.md5(f"{filepath}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest(),
                'last_modified': stat.st_mtime,
                'file_path': filepath
            }
        except Exception:
            return {
                'file_hash': hashlib.md5(filepath.encode()).hexdigest(),
                'last_modified': 0,
                'file_path': filepath
            }

    @timeout()
    def _safe_audio_load(self, audio_path):
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
                return None
                
            loader = es.MonoLoader(
                filename=audio_path,
                sampleRate=44100,
                resampleQuality=4,
                downmix='mix'
            )
            audio = loader()
            return audio if isinstance(audio, np.ndarray) and len(audio) > 1024 else None
        except Exception as e:
            logger.warning(f"AudioLoader error for {audio_path}: {str(e)}")
            return None

    @timeout()
    def _extract_rhythm_features(self, audio):
        try:
            if len(audio) < 1024:
                return 0.0, 0.0
                
            rhythm_extractor = es.RhythmExtractor2013(
                method="multifeature",
                minTempo=40,
                maxTempo=208
            )
            bpm, _, beats_confidence, _, _ = rhythm_extractor(audio)
            
            if np.isnan(bpm) or bpm < 40 or bpm > 208:
                return 0.0, 0.0
                
            return float(bpm), float(np.nanmean(beats_confidence))
        except Exception as e:
            logger.warning(f"Rhythm extraction failed: {str(e)}")
            return 0.0, 0.0

    @timeout()
    def _extract_spectral_features(self, audio):
        try:
            if len(audio) < 1024:
                return 0.0
                
            spectral = es.SpectralCentroidTime(sampleRate=44100)
            centroid_values = spectral(audio)
            
            if isinstance(centroid_values, np.ndarray):
                return float(np.nanmean(centroid_values))
            return 0.0
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {str(e)}")
            return 0.0

    def extract_features(self, audio_path):
        try:
            self._init_db()
            file_info = self._get_file_info(audio_path)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT duration, bpm, beat_confidence, centroid
                FROM audio_features
                WHERE file_hash = ? AND last_modified >= ?
                LIMIT 1
                """, (file_info['file_hash'], file_info['last_modified']))
            
            if row := cursor.fetchone():
                self._cache_hits += 1
                if self._cache_hits % 100 == 0:
                    gc.collect()
                return {
                    'duration': float(row[0]),
                    'bpm': float(row[1]),
                    'beat_confidence': float(row[2]),
                    'centroid': float(row[3]),
                    'filepath': audio_path,
                    'filename': os.path.basename(audio_path)
                }, True, file_info['file_hash']

            self._cache_misses += 1
            if self._cache_misses % 50 == 0:
                self.conn.commit()
                gc.collect()

            logger.info(f"Processing: {os.path.basename(audio_path)}")
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                return None, False, None

            features = {
                'duration': float(len(audio) / 44100.0),
                'filepath': audio_path,
                'filename': os.path.basename(audio_path)
            }
            
            bpm, confidence = self._extract_rhythm_features(audio)
            features['bpm'] = float(bpm)
            features['beat_confidence'] = float(confidence)
            
            centroid = self._extract_spectral_features(audio)
            features['centroid'] = float(centroid)

            with self.conn:
                self.conn.execute("""
                INSERT OR REPLACE INTO audio_features VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                """, (
                    file_info['file_hash'],
                    audio_path,
                    features['duration'],
                    features['bpm'],
                    features['beat_confidence'],
                    features['centroid'],
                    file_info['last_modified']
                ))

            return features, False, file_info['file_hash']
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return None, False, None
        finally:
            if self._cache_misses % 200 == 0 and self.conn:
                self.conn.execute("VACUUM")

audio_analyzer = AudioAnalyzer(timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', 30)))

def extract_features(audio_path):
    return audio_analyzer.extract_features(audio_path)

def get_all_features():
    return audio_analyzer.get_all_features()