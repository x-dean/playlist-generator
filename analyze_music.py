import os
import signal
import sqlite3
import hashlib
import essentia.standard as es
import numpy as np
from contextlib import contextmanager

class AudioAnalyzer:
    def __init__(self, db_path='audio_features.db', timeout=30):
        self.db_path = db_path
        self.timeout = timeout
        self._init_db()

    def _init_db(self):
        """Initialize database with proper schema and indexes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    bpm REAL,
                    duration REAL,
                    beat_confidence REAL,
                    spectral_centroid REAL,
                    last_modified REAL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON features(file_hash)")
            conn.commit()

    def _file_hash(self, filepath):
        """Compute SHA-256 hash of file metadata"""
        stat = os.stat(filepath)
        return hashlib.sha256(
            f"{filepath}-{stat.st_size}-{stat.st_mtime}".encode()
        ).hexdigest()

    @contextmanager
    def _time_limit(self):
        signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.alarm(self.timeout)
        try:
            yield
        finally:
            signal.alarm(0)

    def _raise_timeout(self, signum, frame):
        raise TimeoutError(f"Timeout after {self.timeout}s")

    def extract_features(self, filepath):
        """Core analysis with caching"""
        file_hash = self._file_hash(filepath)
        
        # Check cache first
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM features WHERE path = ? AND file_hash = ?",
                (filepath, file_hash)
            )
            if row := cursor.fetchone():
                return dict(row)

        try:
            with self._time_limit():
                loader = es.MonoLoader(
                    filename=filepath,
                    sampleRate=44100,
                    resampleQuality=2
                )
                audio = loader()
                
                if len(audio) < 44100:  # 1 second minimum
                    return None

                # Feature extraction
                rhythm = es.RhythmExtractor2013(method="degara")
                bpm, _, confidence, _, _ = rhythm(audio)
                
                spectral = es.SpectralCentroidTime(sampleRate=44100)
                centroid = spectral(audio[:44100*30])  # First 30s only

                features = {
                    'path': filepath,
                    'file_hash': file_hash,
                    'bpm': float(bpm),
                    'duration': len(audio)/44100,
                    'beat_confidence': float(np.mean(confidence)),
                    'spectral_centroid': float(np.mean(centroid)),
                    'last_modified': os.path.getmtime(filepath)
                }

                # Cache results
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO features 
                        VALUES (:path, :file_hash, :bpm, :duration, 
                                :beat_confidence, :spectral_centroid, 
                                :last_modified, CURRENT_TIMESTAMP)
                        """, features)
                    conn.commit()
                
                return features

        except TimeoutError:
            print(f"Timeout processing {os.path.basename(filepath)}")
            return None
        except Exception as e:
            print(f"Error processing {os.path.basename(filepath)}: {str(e)}")
            return None