# analyze_music.py (final complete version)
import numpy as np
import essentia.standard as es
import os
import logging
import sqlite3
import hashlib
import signal
from functools import wraps
import traceback

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
                danceability REAL,
                key INTEGER,
                scale INTEGER,
                onset_rate REAL,
                zcr REAL,
                last_modified REAL,
                last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            self._verify_db_schema()
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON audio_features(file_path)")

    def _verify_db_schema(self):
        """Ensure all required columns exist with correct types"""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(audio_features)")
        existing_columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        required_columns = {
            'loudness': 'REAL DEFAULT 0',
            'danceability': 'REAL DEFAULT 0',
            'key': 'INTEGER DEFAULT -1',
            'scale': 'INTEGER DEFAULT 0',
            'onset_rate': 'REAL DEFAULT 0',
            'zcr': 'REAL DEFAULT 0'
        }
        
        for col, col_type in required_columns.items():
            if col not in existing_columns:
                logger.info(f"Adding missing column {col} to database")
                self.conn.execute(f"ALTER TABLE audio_features ADD COLUMN {col} {col_type}")

    def _ensure_float(self, value):
        """Convert value to float safely"""
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _ensure_int(self, value):
        """Convert value to int safely"""
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

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
            return float(es.RMS()(audio))
        except Exception as e:
            logger.warning(f"Loudness extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_danceability(self, audio):
        try:
            danceability, _ = es.Danceability()(audio)
            return float(danceability)
        except Exception as e:
            logger.warning(f"Danceability extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_key(self, audio):
        try:
            min_length = 44100 * 3  # 3 seconds minimum
            if len(audio) < min_length:
                audio = np.pad(audio, (0, max(0, min_length - len(audio))), 'constant')
            
            key, scale, _ = es.KeyExtractor(frameSize=4096, hopSize=2048)(audio)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_index = keys.index(key) if key in keys else -1
            return key_index, 1 if scale == 'major' else 0
        except Exception as e:
            logger.warning(f"Key extraction failed: {str(e)}")
            return -1, 0

    @timeout()
    def _extract_onset_rate(self, audio):
        try:
            # Skip if audio is too short (less than 1 second)
            if len(audio) < 44100:
                return 0.0
                
            # Get the raw result from Essentia
            result = es.OnsetRate()(audio)
            
            # Case 1: Result is already a single number
            if isinstance(result, (int, float)):
                return float(result)
                
            # Case 2: Result is a numpy array
            if isinstance(result, np.ndarray):
                if result.size == 1:
                    return float(result.item())  # Convert single-element array
                return float(result[0])  # Take first element if multiple
                
            # Case 3: Result is a tuple (rate, onset_times)
            if isinstance(result, tuple) and len(result) > 0:
                return float(result[0])
                
            # Case 4: Result is a list
            if isinstance(result, list) and len(result) > 0:
                return float(result[0])
                
            # Default fallback
            return 0.0
            
        except Exception as e:
            logger.warning(f"Onset rate extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_zcr(self, audio):
        try:
            return float(np.mean(es.ZeroCrossingRate()(audio)))
        except Exception as e:
            logger.warning(f"Zero crossing rate extraction failed: {str(e)}")
            return 0.0

    def extract_features(self, audio_path):
        try:
            file_info = self._get_file_info(audio_path)
            
            # Check cache first
            cached_features = self._get_cached_features(file_info)
            if cached_features:
                return cached_features

            # Process new file
            logger.info(f"Processing: {os.path.basename(audio_path)}")
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                return None, False, None

            features = self._extract_all_features(audio_path, audio)
            self._save_features_to_db(file_info, features)
            
            return features, False, file_info['file_hash']
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None, False, None

    def _get_cached_features(self, file_info):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT duration, bpm, beat_confidence, centroid, 
               loudness, danceability, key, scale, onset_rate, zcr
        FROM audio_features
        WHERE file_hash = ? AND last_modified >= ?
        """, (file_info['file_hash'], file_info['last_modified']))

        if row := cursor.fetchone():
            return {
                'duration': row[0],
                'bpm': row[1],
                'beat_confidence': row[2],
                'centroid': row[3],
                'loudness': row[4],
                'danceability': row[5],
                'key': row[6],
                'scale': row[7],
                'onset_rate': row[8],
                'zcr': row[9],
                'filepath': file_info['file_path'],
                'filename': os.path.basename(file_info['file_path'])
            }, True, file_info['file_hash']
        return None

    def _extract_all_features(self, audio_path, audio):
        features = {
            'duration': self._ensure_float(len(audio) / 44100.0),
            'filepath': audio_path,
            'filename': os.path.basename(audio_path)
        }

        # Initialize all features with default values first
        default_features = {
            'bpm': 0.0,
            'beat_confidence': 0.0,
            'centroid': 0.0,
            'loudness': 0.0,
            'danceability': 0.0,
            'key': -1,
            'scale': 0,
            'onset_rate': 0.0,
            'zcr': 0.0
        }
        features.update(default_features)

        if len(audio) >= 44100:  # At least 1 second
            try:
                bpm, confidence = self._extract_rhythm_features(audio)
                features.update({
                    'bpm': self._ensure_float(bpm),
                    'beat_confidence': self._ensure_float(confidence),
                    'centroid': self._ensure_float(self._extract_spectral_features(audio)),
                    'loudness': self._ensure_float(self._extract_loudness(audio)),
                    'danceability': self._ensure_float(self._extract_danceability(audio)),
                    'key': self._ensure_int(self._extract_key(audio)[0]),
                    'scale': self._ensure_int(self._extract_key(audio)[1]),
                    'onset_rate': self._ensure_float(self._extract_onset_rate(audio)),
                    'zcr': self._ensure_float(self._extract_zcr(audio))
                })
            except Exception as e:
                logger.error(f"Feature extraction error for {audio_path}: {str(e)}")
        
        return features

    def _save_features_to_db(self, file_info, features):
        with self.conn:
            self.conn.execute("""
            INSERT OR REPLACE INTO audio_features 
            (file_hash, file_path, duration, bpm, beat_confidence, centroid, 
             loudness, danceability, key, scale, onset_rate, zcr, last_modified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(file_info['file_hash']),
                str(file_info['file_path']),
                self._ensure_float(features['duration']),
                self._ensure_float(features['bpm']),
                self._ensure_float(features['beat_confidence']),
                self._ensure_float(features['centroid']),
                self._ensure_float(features['loudness']),
                self._ensure_float(features['danceability']),
                self._ensure_int(features['key']),
                self._ensure_int(features['scale']),
                self._ensure_float(features['onset_rate']),
                self._ensure_float(features['zcr']),
                self._ensure_float(file_info['last_modified'])
            ))

    def get_all_features(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT file_path, duration, bpm, beat_confidence, centroid, 
                   loudness, danceability, key, scale, onset_rate, zcr
            FROM audio_features
            """)
            return [{
                'filepath': row[0],
                'duration': row[1],
                'bpm': row[2],
                'beat_confidence': row[3],
                'centroid': row[4],
                'loudness': row[5],
                'danceability': row[6],
                'key': row[7],
                'scale': row[8],
                'onset_rate': row[9],
                'zcr': row[10]
            } for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching features: {str(e)}")
            return []

    def cleanup_database(self):
        """Remove entries for files that no longer exist"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path FROM audio_features")
            db_files = [row[0] for row in cursor.fetchall()]
            
            missing_files = [f for f in db_files if not os.path.exists(f)]
            
            if missing_files:
                logger.info(f"Cleaning up {len(missing_files)} missing files from database")
                placeholders = ','.join(['?'] * len(missing_files))
                cursor.execute(
                    f"DELETE FROM audio_features WHERE file_path IN ({placeholders})",
                    missing_files
                )
                self.conn.commit()
            return missing_files
        except Exception as e:
            logger.error(f"Database cleanup failed: {str(e)}")
            return []

audio_analyzer = AudioAnalyzer()

def extract_features(audio_path):
    return audio_analyzer.extract_features(audio_path)

def get_all_features():
    return audio_analyzer.get_all_features()