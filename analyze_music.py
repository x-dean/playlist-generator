import numpy as np
import essentia.standard as es
import os
import logging
import sqlite3
import hashlib
import signal
from functools import wraps

# Setup logging
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
        self.conn = None
        self.timeout_seconds = timeout_seconds

    def _init_db(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.cache_file, timeout=30)
            with self.conn:
                self.conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_features (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    duration REAL,
                    bpm REAL,
                    beat_confidence REAL,
                    centroid REAL,
                    last_modified REAL,
                    last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON audio_features(file_path)")

    def _get_file_info(self, filepath):
        try:
            stat = os.stat(filepath)
            return {
                'file_hash': f"{os.path.basename(filepath)}_{stat.st_size}_{stat.st_mtime}",
                'last_modified': stat.st_mtime,
                'file_path': filepath
            }
        except Exception as e:
            logger.warning(f"Couldn't get file stats for {filepath}: {str(e)}")
            return {
                'file_hash': hashlib.md5(filepath.encode()).hexdigest(),
                'last_modified': 0,
                'file_path': filepath
            }

    @timeout()
    def _safe_audio_load(self, audio_path):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                def _handle_timeout(signum, frame):
                    raise TimeoutException(f"Audio loading timed out after {self.timeout_seconds} seconds")

                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(self.timeout_seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
            return wrapper
        return decorator(lambda: self._load_audio(audio_path))()

    def _load_audio(self, audio_path):
        try:
            loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            return audio if isinstance(audio, np.ndarray) and len(audio) > 0 else None
        except Exception as e:
            logger.warning(f"AudioLoader error for {audio_path}: {str(e)}")
            return None

    @timeout()
    def _extract_rhythm_features(self, audio):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                def _handle_timeout(signum, frame):
                    raise TimeoutException(f"Rhythm extraction timed out after {self.timeout_seconds} seconds")

                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(self.timeout_seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
            return wrapper
        return decorator(lambda: self._extract_rhythm(audio))()

    def _extract_rhythm(self, audio):
        try:
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
            bpm = float(bpm) if isinstance(bpm, (float, int, np.number)) else 0.0
            confidence = float(np.nanmean(beats_confidence)) if isinstance(beats_confidence, np.ndarray) else 0.0
            return bpm, confidence
        except Exception as e:
            logger.warning(f"Rhythm extraction failed: {str(e)}")
            return 0.0, 0.0

    @timeout()
    def _extract_spectral_features(self, audio):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                def _handle_timeout(signum, frame):
                    raise TimeoutException(f"Spectral extraction timed out after {self.timeout_seconds} seconds")

                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(self.timeout_seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result
            return wrapper
        return decorator(lambda: self._extract_spectral(audio))()

    def _extract_spectral(self, audio):
        try:
            spectral = es.SpectralCentroidTime(sampleRate=44100)
            centroid_values = spectral(audio)
            if isinstance(centroid_values, np.ndarray):
                return float(np.nanmean(centroid_values))
            elif isinstance(centroid_values, (float, int, np.number)):
                return float(centroid_values)
            return 0.0
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {str(e)}")
            return 0.0

    def extract_features(self, audio_path):
        try:
            self._init_db()
            file_info = self._get_file_info(audio_path)
            
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute("""
                SELECT duration, bpm, beat_confidence, centroid
                FROM audio_features
                WHERE file_hash = ? AND last_modified >= ?
                """, (file_info['file_hash'], file_info['last_modified']))
                if row := cursor.fetchone():
                    return {
                        'duration': float(row[0]),
                        'bpm': float(row[1]),
                        'beat_confidence': float(row[2]),
                        'centroid': float(row[3]),
                        'filepath': audio_path,
                        'filename': os.path.basename(audio_path)
                    }, True, file_info['file_hash']

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
            features['bpm'] = float(bpm) if bpm else 0.0
            features['beat_confidence'] = float(confidence) if confidence else 0.0
            
            centroid = self._extract_spectral_features(audio)
            features['centroid'] = float(centroid) if centroid else 0.0

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
        except ValueError as ve:
            logger.error(f"Value error processing {audio_path}: {str(ve)}")
            return None, False, None
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return None, False, None
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None

# Singleton instance with configurable timeout
audio_analyzer = AudioAnalyzer(timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', 30)))

def extract_features(audio_path):
    return audio_analyzer.extract_features(audio_path)

def get_all_features():
    return audio_analyzer.get_all_features()