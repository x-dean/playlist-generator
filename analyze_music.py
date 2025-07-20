import numpy as np
import essentia.standard as es
import os
import logging
import sqlite3
import hashlib
import signal
from functools import wraps
import resource

# Enhanced logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

# Current feature version
CURRENT_FEATURE_VERSION = 3

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
        self._init_db()
        self._migrate_db()

    def _init_db(self):
        try:
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
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    def _migrate_db(self):
        """Add new columns if they don't exist"""
        try:
            cursor = self.conn.cursor()
            migrations = [
                "ALTER TABLE audio_features ADD COLUMN loudness REAL DEFAULT 0",
                "ALTER TABLE audio_features ADD COLUMN dynamics REAL DEFAULT 0",
                "ALTER TABLE audio_features ADD COLUMN key TEXT DEFAULT 'unknown'",
                "ALTER TABLE audio_features ADD COLUMN scale TEXT DEFAULT 'unknown'",
                "ALTER TABLE audio_features ADD COLUMN key_confidence REAL DEFAULT 0",
                "ALTER TABLE audio_features ADD COLUMN rhythm_complexity REAL DEFAULT 0.5",
                "ALTER TABLE audio_features ADD COLUMN feature_version INTEGER DEFAULT 1"
            ]
            
            for migration in migrations:
                try:
                    cursor.execute(migration)
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass  # Column already exists
        except Exception as e:
            logger.error(f"Database migration failed: {str(e)}")
            raise

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

    @timeout(120)
    def _safe_audio_load(self, audio_path):
        try:
            loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            return audio if audio.size > 0 else None
        except Exception as e:
            logger.warning(f"AudioLoader error for {audio_path}: {str(e)}")
            return None

    @timeout(120)
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
            logger.warning(f"Rhythm extraction failed: {str(e)}")
            return 0.0, 0.0

    @timeout(120)
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
            logger.warning(f"Spectral extraction failed: {str(e)}")
            return 0.0

    @timeout(120)
    def _extract_loudness(self, audio):
        try:
            loudness = es.Loudness()(audio)
            return float(loudness)
        except Exception as e:
            logger.warning(f"Loudness extraction failed: {str(e)}")
            return -20.0  # Default value for silence
            
    @timeout(120)
    def _extract_dynamics(self, audio):
        try:
            dynamic_complexity, _ = es.DynamicComplexity()(audio)
            return float(dynamic_complexity)
        except Exception as e:
            logger.warning(f"Dynamic complexity extraction failed: {str(e)}")
            return 0.0
            
    @timeout(120)
    def _extract_harmonic_features(self, audio):
        try:
            key_extractor = es.KeyExtractor()
            key, scale, confidence = key_extractor(audio)
            return key, scale, float(confidence)
        except Exception as e:
            logger.warning(f"Harmonic extraction failed: {str(e)}")
            return "unknown", "unknown", 0.0
            
    @timeout(120)
    def _extract_rhythm_complexity(self, audio):
        try:
            danceability, _ = es.Danceability()(audio)
            return float(danceability)
        except Exception as e:
            logger.warning(f"Rhythm complexity extraction failed: {str(e)}")
            return 0.5

    @timeout(300)  # Increased timeout for full feature extraction
    def extract_features(self, audio_path):
        """Enhanced feature extraction with new audio features"""
        try:
            # Set resource limits
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))  # 4GB memory limit
            resource.setrlimit(resource.RLIMIT_CPU, (120, 120))  # 2 minutes CPU time
            
            file_info = self._get_file_info(audio_path)

            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT duration, bpm, beat_confidence, centroid, 
                   loudness, dynamics, key, scale, key_confidence, rhythm_complexity,
                   feature_version
            FROM audio_features
            WHERE file_hash = ? AND last_modified >= ?
            """, (file_info['file_hash'], file_info['last_modified']))

            if row := cursor.fetchone():
                # Check if cache is valid
                valid_cache = True
                for i, feat in enumerate(['duration', 'bpm', 'centroid', 'loudness']):
                    if i < len(row) and (row[i] is None or row[i] < 0):
                        valid_cache = False
                        break
                
                # Check feature version
                db_version = row[10] if len(row) > 10 else 1
                if valid_cache and db_version >= CURRENT_FEATURE_VERSION:
                    result = {
                        'duration': float(row[0]),
                        'bpm': float(row[1]),
                        'beat_confidence': float(row[2]),
                        'centroid': float(row[3]),
                        'loudness': float(row[4]),
                        'dynamics': float(row[5]),
                        'key': str(row[6]),
                        'scale': str(row[7]),
                        'key_confidence': float(row[8]),
                        'rhythm_complexity': float(row[9]),
                        'filepath': audio_path,
                        'filename': os.path.basename(audio_path)
                    }
                    # Validate all required features are present
                    required_features = ['duration', 'bpm', 'centroid', 'loudness']
                    if all(feat in result and result[feat] is not None for feat in required_features):
                        return result, True, file_info['file_hash']

            # Process file if not in cache or cache is outdated
            logger.info(f"Processing: {os.path.basename(audio_path)}")
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                return None, False, None

            features = {
                'duration': len(audio) / 44100.0,
                'filepath': audio_path,
                'filename': os.path.basename(audio_path)
            }

            # Extract all features with error handling
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
                features['loudness'] = -20.0
                
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

            # Validate all required features before saving
            required_features = ['duration', 'bpm', 'centroid', 'loudness']
            if not all(feat in features and features[feat] is not None for feat in required_features):
                logger.error(f"Missing required features for {audio_path}")
                return None, False, None

            # Save to database
            try:
                with self.conn:
                    self.conn.execute("""
                    INSERT OR REPLACE INTO audio_features VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP,?
                    )
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
            except sqlite3.Error as e:
                logger.error(f"Database save failed for {audio_path}: {str(e)}")
                return None, False, None

            return features, False, file_info['file_hash']
        except TimeoutException:
            logger.warning(f"Timeout processing {audio_path}")
            return None, False, None
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return None, False, None

    def get_all_features(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT file_path, duration, bpm, beat_confidence, centroid, 
                   loudness, dynamics, key, scale, key_confidence, rhythm_complexity
            FROM audio_features
            WHERE feature_version >= ? AND duration > 0 AND bpm > 0
            """, (CURRENT_FEATURE_VERSION,))
            
            results = []
            for row in cursor.fetchall():
                try:
                    results.append({
                        'filepath': str(row[0]),
                        'duration': float(row[1]),
                        'bpm': float(row[2]),
                        'beat_confidence': float(row[3]),
                        'centroid': float(row[4]),
                        'loudness': float(row[5]),
                        'dynamics': float(row[6]),
                        'key': str(row[7]),
                        'scale': str(row[8]),
                        'key_confidence': float(row[9]),
                        'rhythm_complexity': float(row[10]),
                    })
                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(f"Invalid feature data for {row[0]}: {str(e)}")
                    continue
            
            return results
        except Exception as e:
            logger.error(f"Error fetching features: {str(e)}")
            return []
        finally:
            try:
                cursor.close()
            except:
                pass

    def __del__(self):
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except:
            pass

audio_analyzer = AudioAnalyzer()

def extract_features(audio_path):
    return audio_analyzer.extract_features(audio_path)

def get_all_features():
    return audio_analyzer.get_all_features()