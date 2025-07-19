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
    def __init__(self, cache_file=None):
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = cache_file or os.path.join(cache_dir, 'audio_analysis.db')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self.timeout_seconds = 30
        # Initialize DB connection once
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.cache_file, timeout=30)
        with self.conn:
            # Enable WAL mode for better concurrency
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")  # 30-second timeout
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
        try:
            loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            # Return audio only if it has content
            return audio if audio.size > 0 else None
        except Exception as e:
            logger.warning(f"AudioLoader error for {audio_path}: {str(e)}")
            return None

    @timeout()
    def _extract_rhythm_features(self, audio):
        try:
            rhythm_extractor = es.RhythmExtractor()
            bpm, _, beats_confidence, _ = rhythm_extractor(audio)
            
            # Handle case where beats_confidence might be a list
            if isinstance(beats_confidence, (list, np.ndarray)):
                beat_conf = float(np.nanmean(beats_confidence))
            else:
                beat_conf = float(beats_confidence)
                
            return float(bpm), beat_conf
        except Exception as e:
            logger.warning(f"Rhythm extraction failed: {str(e)}")
            return 0.0, 0.0

    @timeout()
    def _extract_spectral_features(self, audio):
        try:
            spectral = es.SpectralCentroidTime(sampleRate=44100)
            centroid_values = spectral(audio)
            
            # Ensure we're dealing with a single value
            if isinstance(centroid_values, (list, np.ndarray)):
                centroid = float(np.nanmean(centroid_values))
            else:
                centroid = float(centroid_values)
                
            return centroid
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {str(e)}")
            return 0.0

    def extract_features(self, audio_path):
        try:
            file_info = self._get_file_info(audio_path)
            
            # Check cache first
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT duration, bpm, beat_confidence, centroid
            FROM audio_features
            WHERE file_hash = ? AND last_modified >= ?
            """, (file_info['file_hash'], file_info['last_modified']))
            if row := cursor.fetchone():
                return {
                    'duration': row[0],
                    'bpm': row[1],
                    'beat_confidence': row[2],
                    'centroid': row[3],
                    'filepath': audio_path,
                    'filename': os.path.basename(audio_path)
                }, True, file_info['file_hash']

            # Process new file
            logger.info(f"Processing: {os.path.basename(audio_path)}")
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                return None, False, None

            features = {
                'duration': len(audio) / 44100.0,
                'filepath': audio_path,
                'filename': os.path.basename(audio_path)
            }
            
            # Extract features with proper error handling
            try:
                # Increase timeout for rhythm feature extraction
                features['bpm'], features['beat_confidence'] = self._extract_rhythm_features(audio)
            except Exception as e:
                logger.error(f"Rhythm extraction error for {audio_path}: {str(e)}")
                features['bpm'] = 0.0
                features['beat_confidence'] = 0.0
                
            try:
                # Increase timeout for spectral feature extraction
                features['centroid'] = self._extract_spectral_features(audio)
            except Exception as e:
                logger.error(f"Spectral extraction error for {audio_path}: {str(e)}")
                features['centroid'] = 0.0

            # Update cache
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

    def get_all_features(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT filepath, duration, bpm, beat_confidence, centroid
            FROM audio_features
            """)
            return [
                {
                    'filepath': row[0],
                    'duration': row[1],
                    'bpm': row[2],
                    'beat_confidence': row[3],
                    'centroid': row[4],
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"Error fetching features: {str(e)}")
            return []

# Singleton instance
audio_analyzer = AudioAnalyzer()

def extract_features(audio_path):
    """Wrapper function for the singleton analyzer"""
    return audio_analyzer.extract_features(audio_path)

def get_all_features():
    """Get all features from the database"""
    return audio_analyzer.get_all_features()