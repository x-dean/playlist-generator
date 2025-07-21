import essentia.standard as es
import os
import logging
import sqlite3
import hashlib
import numpy as np

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self, cache_file=None):
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = cache_file or os.path.join(cache_dir, 'audio_analysis.db')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.cache_file, timeout=600)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS audio_features (
            file_path TEXT PRIMARY KEY,
            duration REAL,
            bpm REAL,
            centroid REAL,
            danceability REAL,
            last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

    def _get_file_hash(self, filepath):
        try:
            stat = os.stat(filepath)
            return f"{os.path.basename(filepath)}_{stat.st_size}_{stat.st_mtime}"
        except Exception:
            return hashlib.md5(filepath.encode()).hexdigest()

    def extract_features(self, audio_path):
        try:
            file_hash = self._get_file_hash(audio_path)
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM audio_features WHERE file_path=?", (file_hash,))
            if row := cursor.fetchone():
                return {
                    'filepath': audio_path,
                    'duration': row[1],
                    'bpm': row[2],
                    'centroid': row[3],
                    'danceability': row[4]
                }, True, file_hash

            loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            duration = len(audio) / 44100.0
            
            features = {
                'filepath': audio_path,
                'duration': duration,
                'bpm': 0.0,
                'centroid': 0.0,
                'danceability': 0.0
            }
            
            if duration > 1.0:
                try:
                    bpm, _, _, _ = es.RhythmExtractor()(audio)
                    features['bpm'] = float(bpm)
                except Exception:
                    pass
                    
                try:
                    centroid = es.SpectralCentroidTime()(audio)
                    features['centroid'] = float(np.mean(centroid))
                except Exception:
                    pass
                    
                try:
                    danceability, _ = es.Danceability()(audio)
                    features['danceability'] = float(danceability)
                except Exception:
                    pass

            self.conn.execute("""
            INSERT OR REPLACE INTO audio_features 
            (file_path, duration, bpm, centroid, danceability)
            VALUES (?, ?, ?, ?, ?)
            """, (file_hash, features['duration'], features['bpm'], 
                 features['centroid'], features['danceability']))
            self.conn.commit()
            
            return features, False, file_hash
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            return None, False, None

    def get_all_features(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path, duration, bpm, centroid, danceability FROM audio_features")
            return [{
                'filepath': row[0],
                'duration': row[1],
                'bpm': row[2],
                'centroid': row[3],
                'danceability': row[4]
            } for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching features: {str(e)}")
            return []

audio_analyzer = AudioAnalyzer()

def extract_features(audio_path):
    return audio_analyzer.extract_features(audio_path)

def get_all_features():
    return audio_analyzer.get_all_features()