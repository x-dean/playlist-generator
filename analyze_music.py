import numpy as np
import essentia.standard as es
import os
import logging
import sqlite3
import hashlib
import signal
import gc
from functools import wraps
from pathlib import Path
from queue import Queue
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout_context(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class ConnectionPool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -8000")
            self.pool.put(conn)
            
    def get_conn(self):
        return self.pool.get()
        
    def release_conn(self, conn):
        self.pool.put(conn)
        
    def close_all(self):
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()

class AudioAnalyzer:
    def __init__(self, cache_file=None, timeout_seconds=30, pool_size=5):
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        db_path = cache_file or os.path.join(cache_dir, 'audio_analysis.db')
        self.cache_file = str(Path(db_path)).replace('\\', '/')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        self.timeout_seconds = timeout_seconds
        self.pool_size = pool_size
        self.conn_pool = ConnectionPool(self.cache_file, pool_size)
        self._cache_hits = 0
        self._cache_misses = 0
        self._init_db()
        
        logger.info(f"Initialized analyzer with {pool_size} DB connections")

    def _init_db(self):
        """Initialize database with proper settings"""
        conn = None
        try:
            conn = self.conn_pool.get_conn()
            conn.execute("""
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON audio_features(file_path)")
            conn.commit()
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
        finally:
            if conn:
                self.conn_pool.release_conn(conn)

    def _get_file_info(self, filepath):
        """Generate unique file hash based on metadata"""
        try:
            stat = os.stat(filepath)
            return {
                'file_hash': hashlib.md5(f"{filepath}_{stat.st_size}_{stat.st_mtime}".encode()).hexdigest(),
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

    def _safe_audio_load(self, audio_path):
        """Load audio file with validation and timeout"""
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1024:
                logger.debug(f"Skipping small/corrupt file: {audio_path}")
                return None
                
            with timeout_context(self.timeout_seconds):
                loader = es.MonoLoader(
                    filename=audio_path,
                    sampleRate=44100,
                    resampleQuality=4,
                    downmix='mix'
                )
                audio = loader()
                return audio if isinstance(audio, np.ndarray) and len(audio) > 1024 else None
        except TimeoutException:
            logger.warning(f"Timeout loading audio file: {audio_path}")
            return None
        except Exception as e:
            logger.warning(f"AudioLoader error for {audio_path}: {str(e)}")
            return None

    def _extract_rhythm_features(self, audio):
        """Extract BPM and beat confidence with validation"""
        try:
            if len(audio) < 1024:
                return 0.0, 0.0
                
            with timeout_context(self.timeout_seconds):
                rhythm_extractor = es.RhythmExtractor2013(
                    method="multifeature",
                    minTempo=40,
                    maxTempo=208
                )
                bpm, _, beats_confidence, _, _ = rhythm_extractor(audio)
                
                if np.isnan(bpm) or bpm < 40 or bpm > 208:
                    return 0.0, 0.0
                    
                return float(bpm), float(np.nanmean(beats_confidence))
        except TimeoutException:
            logger.warning("Timeout during rhythm extraction")
            return 0.0, 0.0
        except Exception as e:
            logger.warning(f"Rhythm extraction failed: {str(e)}")
            return 0.0, 0.0

    def _extract_spectral_features(self, audio):
        """Extract spectral centroid with validation"""
        try:
            if len(audio) < 1024:
                return 0.0
                
            with timeout_context(self.timeout_seconds):
                spectral = es.SpectralCentroidTime(sampleRate=44100)
                centroid_values = spectral(audio)
                
                if isinstance(centroid_values, np.ndarray):
                    return float(np.nanmean(centroid_values))
                return 0.0
        except TimeoutException:
            logger.warning("Timeout during spectral extraction")
            return 0.0
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {str(e)}")
            return 0.0

    def extract_features(self, audio_path):
        """Main feature extraction method with caching"""
        conn = None
        try:
            file_info = self._get_file_info(audio_path)
            conn = self.conn_pool.get_conn()
            cursor = conn.cursor()
            
            with timeout_context(self.timeout_seconds):
                cursor.execute("""
                    SELECT duration, bpm, beat_confidence, centroid
                    FROM audio_features
                    WHERE file_hash = ? AND last_modified >= ?
                    LIMIT 1
                """, (file_info['file_hash'], file_info['last_modified']))
                
                if row := cursor.fetchone():
                    self._cache_hits += 1
                    if self._cache_hits % 100 == 0:
                        logger.info(f"Cache hits: {self._cache_hits}, misses: {self._cache_misses}")
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
            logger.info(f"Processing: {os.path.basename(audio_path)}")
            
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                return None, False, None

            features = {
                'duration': float(len(audio) / 44100.0),
                'filepath': audio_path,
                'filename': os.path.basename(audio_path)
            }
            
            features['bpm'], features['beat_confidence'] = self._extract_rhythm_features(audio)
            features['centroid'] = self._extract_spectral_features(audio)

            with timeout_context(self.timeout_seconds):
                conn.execute("""
                    INSERT OR REPLACE INTO audio_features
                    VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                """, (
                    file_info['file_hash'],
                    file_info['file_path'],
                    features['duration'],
                    features['bpm'],
                    features['beat_confidence'],
                    features['centroid'],
                    file_info['last_modified']
                ))
                conn.commit()

            return features, False, file_info['file_hash']
        except TimeoutException:
            logger.warning(f"Timeout during database operations for {audio_path}")
            return None, False, None
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return None, False, None
        finally:
            if conn:
                self.conn_pool.release_conn(conn)
            if self._cache_misses > 0 and self._cache_misses % 200 == 0:
                self._vacuum_db()

    def _vacuum_db(self):
        """Perform database maintenance"""
        conn = None
        try:
            conn = self.conn_pool.get_conn()
            conn.execute("VACUUM")
            conn.commit()
            gc.collect()
        except Exception as e:
            logger.warning(f"Database maintenance failed: {str(e)}")
        finally:
            if conn:
                self.conn_pool.release_conn(conn)

    def get_all_features(self):
        """Retrieve all features from database"""
        conn = None
        try:
            conn = self.conn_pool.get_conn()
            cursor = conn.cursor()
            with timeout_context(self.timeout_seconds):
                cursor.execute("""
                    SELECT file_path as filepath, duration, bpm, beat_confidence, centroid
                    FROM audio_features
                """)
                return [{
                    'filepath': row[0],
                    'duration': float(row[1]),
                    'bpm': float(row[2]),
                    'beat_confidence': float(row[3]),
                    'centroid': float(row[4]),
                    'filename': os.path.basename(row[0])
                } for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return []
        finally:
            if conn:
                self.conn_pool.release_conn(conn)

    def close(self):
        """Clean up resources"""
        self.conn_pool.close_all()

# Initialize analyzer with environment variables
audio_analyzer = AudioAnalyzer(
    timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', 30)),
    pool_size=int(os.getenv('DB_POOL_SIZE', 5))
)

def extract_features(audio_path):
    return audio_analyzer.extract_features(audio_path)

def get_all_features():
    return audio_analyzer.get_all_features()

if __name__ == "__main__":
    print(f"Database location: {audio_analyzer.cache_file}")
    print(f"Database exists: {os.path.exists(audio_analyzer.cache_file)}")
    try:
        audio_analyzer.close()
    except:
        pass