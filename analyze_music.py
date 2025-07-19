import os
import sqlite3
import hashlib
import essentia.standard as es
import numpy as np
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self, db_path, timeout=30):
        self.db_path = os.path.abspath(db_path)
        self.timeout = timeout
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.commit()

    def _file_hash(self, filepath):
        stat = os.stat(filepath)
        return hashlib.sha256(
            f"{filepath}-{stat.st_size}-{stat.st_mtime}".encode()
        ).hexdigest()

    @contextmanager
    def _timeout_context(self):
        def handler(signum, frame):
            raise TimeoutError("Analysis timed out")
        
        import signal
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.timeout)
        try:
            yield
        finally:
            signal.alarm(0)

    def extract_features(self, filepath):
        file_hash = self._file_hash(filepath)
        
        # Check cache
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cached = conn.execute(
                "SELECT * FROM tracks WHERE file_hash = ?", 
                (file_hash,)
            ).fetchone()
            if cached:
                return dict(cached)

        try:
            with self._timeout_context():
                # Professional-grade analysis
                audio = es.MonoLoader(
                    filename=filepath,
                    sampleRate=44100,
                    resampleQuality=3
                )()
                
                # Rhythm analysis
                bpm, _, confidence, _, _ = es.RhythmExtractor2013()(audio)
                
                # Spectral analysis
                centroid = es.SpectralCentroidTime()(audio[:44100*30])  # First 30s
                
                features = {
                    'path': filepath,
                    'file_hash': file_hash,
                    'bpm': float(bpm),
                    'duration': len(audio)/44100,
                    'energy': float(np.mean(confidence)),
                    'spectral_centroid': float(np.mean(centroid)),
                    'last_modified': os.path.getmtime(filepath)
                }

                # Cache results
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO tracks 
                        VALUES (:path, :file_hash, :bpm, :duration, 
                                :energy, :spectral_centroid, :last_modified)
                        """, features)
                    conn.commit()
                
                return features
                
        except TimeoutError:
            logger.warning(f"Timeout processing {os.path.basename(filepath)}")
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
        
        return None