import os
import hashlib
import sqlite3
import essentia.standard as es
import numpy as np
from tqdm import tqdm

class AudioAnalyzer:
    def __init__(self, db_path='audio_features.db', timeout=30):
        self.db_path = db_path
        self.timeout = timeout
        self._init_db()

    def _init_db(self):
        """Initialize database with proper schema and indexes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_features (
                    filepath TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    bpm REAL,
                    duration REAL,
                    beat_confidence REAL,
                    spectral_centroid REAL,
                    last_modified REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON audio_features(file_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON audio_features(filepath)")
            conn.commit()

    def _compute_file_hash(self, filepath):
        """Compute SHA-256 hash of file contents with metadata"""
        try:
            stat = os.stat(filepath)
            file_size = stat.st_size
            mtime = stat.st_mtime
            
            # Hash small files completely, larger files with metadata
            if file_size < 1048576:  # 1MB
                with open(filepath, 'rb') as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()
            else:
                content_hash = hashlib.sha256(
                    f"{filepath}-{file_size}-{mtime}".encode()
                ).hexdigest()
            
            return content_hash
            
        except Exception as e:
            print(f"Could not hash {filepath}: {str(e)}")
            return None

    def _get_cached_features(self, filepath, file_hash):
        """Retrieve features from cache if valid"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT bpm, duration, beat_confidence, spectral_centroid 
                   FROM audio_features 
                   WHERE filepath = ? AND file_hash = ?""",
                (filepath, file_hash)
            )
            return cursor.fetchone()

    def extract_features(self, filepath):
        """Main feature extraction method with caching"""
        file_hash = self._compute_file_hash(filepath)
        if not file_hash:
            return None

        # Check cache first
        cached = self._get_cached_features(filepath, file_hash)
        if cached:
            return dict(cached)

        try:
            # Feature extraction
            loader = es.MonoLoader(
                filename=filepath,
                sampleRate=44100,
                resampleQuality=2
            )
            audio = loader()
            
            # Skip invalid/short files
            if len(audio) < 44100:  # 1 second
                return None

            # Extract features
            rhythm = es.RhythmExtractor2013(method="multifeature")
            bpm, _, confidence, _, _ = rhythm(audio)
            
            spectral = es.SpectralCentroidTime(sampleRate=44100)
            centroid = spectral(audio)
            
            features = {
                'filepath': filepath,
                'file_hash': file_hash,
                'bpm': float(bpm),
                'duration': len(audio)/44100,
                'beat_confidence': float(np.mean(confidence)),
                'spectral_centroid': float(np.mean(centroid)),
                'last_modified': os.path.getmtime(filepath)
            }

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO audio_features 
                    (filepath, file_hash, bpm, duration, beat_confidence, spectral_centroid, last_modified)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    features['filepath'],
                    features['file_hash'],
                    features['bpm'],
                    features['duration'],
                    features['beat_confidence'],
                    features['spectral_centroid'],
                    features['last_modified']
                ))
                conn.commit()
            
            return features
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            return None