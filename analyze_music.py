import numpy as np
import essentia.standard as es
import os
import logging
import sqlite3
import hashlib
import signal
from functools import wraps

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Current feature version - increment when adding/removing features
CURRENT_FEATURE_VERSION = 2

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
        self._migrate_db()

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
                dynamics REAL,
                key TEXT,
                scale TEXT,
                key_confidence REAL,
                rhythm_complexity REAL,
                last_modified REAL,
                last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feature_version INTEGER DEFAULT 1
            )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON audio_features(file_path)")

    def _migrate_db(self):
        """Migrate database schema to current version"""
        migrations = [
            # Migration 1: Add new features
            "ALTER TABLE audio_features ADD COLUMN loudness REAL DEFAULT 0",
            "ALTER TABLE audio_features ADD COLUMN dynamics REAL DEFAULT 0",
            "ALTER TABLE audio_features ADD COLUMN key TEXT DEFAULT 'unknown'",
            "ALTER TABLE audio_features ADD COLUMN scale TEXT DEFAULT 'unknown'",
            "ALTER TABLE audio_features ADD COLUMN key_confidence REAL DEFAULT 0",
            "ALTER TABLE audio_features ADD COLUMN rhythm_complexity REAL DEFAULT 0.5",
            "ALTER TABLE audio_features ADD COLUMN feature_version INTEGER DEFAULT 1",
        ]
        
        cursor = self.conn.cursor()
        for migration in migrations:
            try:
                cursor.execute(migration)
                self.conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists
    
    def _get_file_info(self, filepath):
        try:
            stat = os.stat(filepath)
            return {
                'file_hash': f"{os.path.basename(filepath)}_{stat.st_size}_{stat.st_mtime}",
                'last_modified': stat.st_mtime,
                'file_path': filepath
            }
        except Exception as e:
            logger.error(f"Couldn't get file stats for {filepath}: {str(e)}")
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
            logger.error(f"AudioLoader error for {audio_path}: {str(e)}")
            return None

    @timeout()
    def _extract_rhythm_features(self, audio):
        try:
            rhythm_extractor = es.RhythmExtractor()
            bpm, _, confidence, _ = rhythm_extractor(audio)

            if isinstance(confidence, (list, np.ndarray)):
                beat_conf = float(np.nanmean(confidence))
            else:
                beat_conf = float(confidence)

            beat_conf = max(0.0, min(1.0, beat_conf))
            return float(bpm), beat_conf
        except Exception as e:
            logger.error(f"Rhythm extraction failed: {str(e)}")
            return 0.0, 0.0

    @timeout()
    def _extract_spectral_features(self, audio):
        try:
            spectral = es.SpectralCentroidTime(sampleRate=44100)
            centroid_values = spectral(audio)

            if isinstance(centroid_values, (list, np.ndarray)):
                centroid = float(np.nanmean(centroid_values))
            else:
                centroid = float(centroid_values)

            return centroid
        except Exception as e:
            logger.error(f"Spectral extraction failed: {str(e)}")
            return 0.0
            
    @timeout()
    def _extract_loudness(self, audio):
        try:
            loudness = es.Loudness()(audio)
            return float(loudness)
        except Exception as e:
            logger.error(f"Loudness extraction failed: {str(e)}")
            return 0.0
            
    @timeout()
    def _extract_dynamics(self, audio):
        try:
            dynamic_complexity, _ = es.DynamicComplexity()(audio)
            return float(dynamic_complexity)
        except Exception as e:
            logger.error(f"Dynamic complexity extraction failed: {str(e)}")
            return 0.0
            
    @timeout()
    def _extract_harmonic_features(self, audio):
        try:
            key_extractor = es.KeyExtractor()
            key, scale, confidence = key_extractor(audio)
            return key, scale, float(confidence)
        except Exception as e:
            logger.error(f"Harmonic extraction failed: {str(e)}")
            return "unknown", "unknown", 0.0
            
    @timeout()
    def _extract_rhythm_complexity(self, audio):
        """Alternative rhythm feature when onset strength isn't available"""
        try:
            # Using Danceability as a rhythm complexity measure
            danceability, _ = es.Danceability()(audio)
            return float(danceability)
        except Exception as e:
            logger.error(f"Rhythm complexity extraction failed: {str(e)}")
            return 0.5

    def extract_features(self, audio_path):
        try:
            file_info = self._get_file_info(audio_path)

            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT duration, bpm, beat_confidence, centroid, 
                   loudness, dynamics, key, scale, key_confidence, rhythm_complexity,
                   feature_version
            FROM audio_features
            WHERE file_hash = ? AND last_modified >= ?
            """, (file_info['file_hash'], file_info['last_modified']))

            row = cursor.fetchone()
            if row:
                # Check feature version
                db_version = row[10] if len(row) > 10 else 1
                if db_version >= CURRENT_FEATURE_VERSION:
                    return {
                        'duration': row[0],
                        'bpm': row[1],
                        'beat_confidence': row[2],
                        'centroid': row[3],
                        'loudness': row[4],
                        'dynamics': row[5],
                        'key': row[6],
                        'scale': row[7],
                        'key_confidence': row[8],
                        'rhythm_complexity': row[9],
                        'filepath': audio_path,
                        'filename': os.path.basename(audio_path)
                    }, True, file_info['file_hash']

            # If we get here, we need to process the file
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                return None, False, None

            features = {
                'duration': len(audio) / 44100.0,
                'filepath': audio_path,
                'filename': os.path.basename(audio_path)
            }

            # Extract features with individual error handling
            try:
                features['bpm'], features['beat_confidence'] = self._extract_rhythm_features(audio)
            except Exception as e:
                logger.error(f"Rhythm extraction error: {str(e)}")
                features['bpm'] = 0.0
                features['beat_confidence'] = 0.0

            try:
                features['centroid'] = self._extract_spectral_features(audio)
            except Exception as e:
                logger.error(f"Spectral extraction error: {str(e)}")
                features['centroid'] = 0.0
                
            try:
                features['loudness'] = self._extract_loudness(audio)
            except Exception as e:
                logger.error(f"Loudness extraction error: {str(e)}")
                features['loudness'] = 0.0
                
            try:
                features['dynamics'] = self._extract_dynamics(audio)
            except Exception as e:
                logger.error(f"Dynamic complexity error: {str(e)}")
                features['dynamics'] = 0.0
                
            try:
                features['key'], features['scale'], features['key_confidence'] = self._extract_harmonic_features(audio)
            except Exception as e:
                logger.error(f"Harmonic analysis error: {str(e)}")
                features['key'] = "unknown"
                features['scale'] = "unknown"
                features['key_confidence'] = 0.0
                
            try:
                features['rhythm_complexity'] = self._extract_rhythm_complexity(audio)
            except Exception as e:
                logger.error(f"Rhythm complexity error: {str(e)}")
                features['rhythm_complexity'] = 0.5

            with self.conn:
                self.conn.execute("""
                INSERT OR REPLACE INTO audio_features (
                    file_hash, file_path, duration, bpm, beat_confidence, centroid, 
                    loudness, dynamics, key, scale, key_confidence, rhythm_complexity, 
                    last_modified, feature_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    file_info['file_hash'],
                    audio_path,
                    features['duration'],
                    features['bpm'],
                    features['beat_confidence'],
                    features['centroid'],
                    features['loudness'],
                    features['dynamics'],
                    features['key'],
                    features['scale'],
                    features['key_confidence'],
                    features['rhythm_complexity'],
                    file_info['last_modified'],
                    CURRENT_FEATURE_VERSION
                ))

            return features, False, file_info['file_hash']
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return None, False, None

    def get_all_features(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT filepath, duration, bpm, beat_confidence, centroid, 
                   loudness, dynamics, key, scale, key_confidence, rhythm_complexity
            FROM audio_features
            WHERE feature_version >= ?
            """, (CURRENT_FEATURE_VERSION,))
            return [
                {
                    'filepath': row[0],
                    'duration': row[1],
                    'bpm': row[2],
                    'beat_confidence': row[3],
                    'centroid': row[4],
                    'loudness': row[5],
                    'dynamics': row[6],
                    'key': row[7],
                    'scale': row[8],
                    'key_confidence': row[9],
                    'rhythm_complexity': row[10],
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