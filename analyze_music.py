import numpy as np
import essentia.standard as es
import os
import logging
import sqlite3
import hashlib
import signal
import json
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

def timeout(seconds=120, error_message="Processing timed out"):
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
    def __init__(self, cache_file=None):
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = cache_file or os.path.join(cache_dir, 'audio_analysis.db')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self.timeout_seconds = 30
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.cache_file, timeout=30)
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_features (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                duration REAL,
                bpm REAL,
                beat_confidence REAL,
                centroid REAL,
                loudness REAL,
                spectral_complexity REAL,
                mfcc_mean TEXT,
                pitch_mean REAL,
                pitch_std REAL,
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
        try:
            loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            return audio if audio.size > 0 else None
        except Exception as e:
            logger.warning(f"AudioLoader error for {audio_path}: {str(e)}")
            return None

    @timeout()
    def _extract_rhythm_features(self, audio):
        try:
            rhythm_extractor = es.RhythmExtractor()
            bpm, _, confidence, _ = rhythm_extractor(audio)
            beat_conf = float(np.nanmean(confidence)) if isinstance(confidence, (list, np.ndarray)) else float(confidence)
            return float(bpm), max(0.0, min(1.0, beat_conf))
        except Exception as e:
            logger.warning(f"Rhythm extraction failed: {str(e)}")
            return 0.0, 0.0

    @timeout()
    def _extract_spectral_features(self, audio):
        try:
            spectral = es.SpectralCentroidTime(sampleRate=44100)
            centroid_values = spectral(audio)
            return float(np.nanmean(centroid_values)) if isinstance(centroid_values, (list, np.ndarray)) else float(centroid_values)
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_loudness(self, audio):
        try:
            loudness = es.Loudness()
            return float(loudness(audio))
        except Exception as e:
            logger.warning(f"Loudness extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_spectral_complexity(self, audio):
        try:
            complexity = es.SpectralComplexity()
            return float(complexity(audio))
        except Exception as e:
            logger.warning(f"Spectral complexity extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_mfcc(self, audio):
        try:
            w = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            mfcc = es.MFCC()
            mfccs = []
            for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
                mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
                mfccs.append(mfcc_coeffs)
            mfccs = np.array(mfccs)
            return mfccs.mean(axis=0).tolist() if mfccs.size > 0 else [0.0] * 13
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {str(e)}")
            return [0.0] * 13

    @timeout()
    def _extract_pitch(self, audio):
        try:
            pitcher = es.PitchYinFFT(frameSize=2048, hopSize=512)
            pitches = [pitcher(frame)[0] for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512, startFromZero=True)]
            pitches = np.array(pitches)
            return float(np.mean(pitches)), float(np.std(pitches))
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {str(e)}")
            return 0.0, 0.0

    def extract_features(self, audio_path):
        try:
            file_info = self._get_file_info(audio_path)
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT duration, bpm, beat_confidence, centroid, loudness, spectral_complexity, mfcc_mean, pitch_mean, pitch_std
            FROM audio_features
            WHERE file_hash = ? AND last_modified >= ?
            """, (file_info['file_hash'], file_info['last_modified']))

            if row := cursor.fetchone():
                return {
                    'duration': row[0], 'bpm': row[1], 'beat_confidence': row[2], 'centroid': row[3],
                    'loudness': row[4], 'spectral_complexity': row[5], 'mfcc_mean': json.loads(row[6]),
                    'pitch_mean': row[7], 'pitch_std': row[8],
                    'filepath': audio_path,
                    'filename': os.path.basename(audio_path)
                }, True, file_info['file_hash']

            logger.info(f"Processing: {os.path.basename(audio_path)}")
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                return None, False, None

            features = {
                'duration': len(audio) / 44100.0,
                'filepath': audio_path,
                'filename': os.path.basename(audio_path),
                'bpm': 0.0, 'beat_confidence': 0.0,
                'centroid': 0.0, 'loudness': 0.0, 'spectral_complexity': 0.0,
                'mfcc_mean': [0.0] * 13, 'pitch_mean': 0.0, 'pitch_std': 0.0
            }

            features['bpm'], features['beat_confidence'] = self._extract_rhythm_features(audio)
            features['centroid'] = self._extract_spectral_features(audio)
            features['loudness'] = self._extract_loudness(audio)
            features['spectral_complexity'] = self._extract_spectral_complexity(audio)
            features['mfcc_mean'] = self._extract_mfcc(audio)
            features['pitch_mean'], features['pitch_std'] = self._extract_pitch(audio)

            with self.conn:
                self.conn.execute("""
                INSERT OR REPLACE INTO audio_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                """, (
                    file_info['file_hash'], audio_path, features['duration'], features['bpm'],
                    features['beat_confidence'], features['centroid'], features['loudness'],
                    features['spectral_complexity'], json.dumps(features['mfcc_mean']),
                    features['pitch_mean'], features['pitch_std'], file_info['last_modified']
                ))

            return features, False, file_info['file_hash']
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return None, False, None

    def get_all_features(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT file_path, duration, bpm, beat_confidence, centroid, loudness, spectral_complexity, mfcc_mean, pitch_mean, pitch_std
            FROM audio_features
            """)
            return [
                {
                    'filepath': row[0], 'duration': row[1], 'bpm': row[2], 'beat_confidence': row[3],
                    'centroid': row[4], 'loudness': row[5], 'spectral_complexity': row[6],
                    'mfcc_mean': json.loads(row[7]), 'pitch_mean': row[8], 'pitch_std': row[9]
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"Error fetching features: {str(e)}")
            return []

audio_analyzer = AudioAnalyzer()

def extract_features(audio_path):
    return audio_analyzer.extract_features(audio_path)

def get_all_features():
    return audio_analyzer.get_all_features()
