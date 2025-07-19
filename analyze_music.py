import os
import sqlite3
import hashlib
import essentia.standard as es
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self, db_path='audio_features.db'):
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -10000")  # 10MB cache
            conn.commit()

    def _file_hash(self, filepath):
        """Fast consistent hash"""
        stat = os.stat(filepath)
        return hashlib.md5(f"{filepath}-{stat.st_size}-{stat.st_mtime}".encode()).hexdigest()

    def extract_features(self, filepath):
        """Fast feature extraction with caching"""
        file_hash = self._file_hash(filepath)
        
        # Check cache first
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM audio_files WHERE file_hash = ?",
                (file_hash,)
            )
            if row := cursor.fetchone():
                return dict(row)

        try:
            # Fast loading with reduced quality
            loader = es.MonoLoader(
                filename=filepath,
                sampleRate=22050,  # Lower sample rate
                resampleQuality=1,   # Faster resampling
                downmix='left'       # Mono analysis
            )
            audio = loader()
            
            # Only analyze first 2 minutes
            max_samples = 22050 * 120
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Fast rhythm analysis
            rhythm = es.RhythmExtractor2013(method="degara")
            bpm, _, confidence, _, _ = rhythm(audio)
            
            # Fast spectral analysis (first 30s only)
            spectral = es.SpectralCentroidTime(sampleRate=22050)
            centroid = spectral(audio[:22050*30])
            
            features = {
                'path': filepath,
                'file_hash': file_hash,
                'bpm': float(bpm),
                'duration': len(audio)/22050,
                'beat_confidence': float(np.mean(confidence)),
                'spectral_centroid': float(np.mean(centroid)),
                'last_modified': os.path.getmtime(filepath)
            }

            # Cache results
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO audio_files 
                    VALUES (:path, :file_hash, :bpm, :duration, 
                            :beat_confidence, :spectral_centroid, :last_modified)
                    """, features)
                conn.commit()
            
            return features
            
        except Exception as e:
            logger.warning(f"Error processing {filepath}: {str(e)}")
            return None