import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import logging
import traceback

logger = logging.getLogger(__name__)

class KMeansPlaylistGenerator:
    def __init__(self):
        self.playlist_history = {}
        self._verify_db_schema()

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

    
    def generate(self, features_list, num_playlists=5, chunk_size=1000):
        playlists = {}
        try:
            data = []
            for f in features_list:
                if not f or 'filepath' not in f:
                    continue
                try:
                    data.append({
                        'filepath': str(f['filepath']),
                        'bpm': max(0, float(f.get('bpm', 0))),
                        'centroid': max(0, float(f.get('centroid', 0))),
                        'danceability': min(1.0, max(0, float(f.get('danceability', 0))))
                    })
                except Exception as e:
                    logger.debug(f"Skipping track: {str(e)}")
                    continue

            if not data:
                return playlists

            df = pd.DataFrame(data)
            cluster_features = ['bpm', 'centroid', 'danceability']
            weights = np.array([1.5, 1.0, 1.2])

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[cluster_features])
            weighted_features = scaled_features * weights

            kmeans = MiniBatchKMeans(
                n_clusters=min(num_playlists, len(df)),
                random_state=42,
                batch_size=min(500, len(df))
            )
            df['cluster'] = kmeans.fit_predict(weighted_features)

            temp_playlists = {}
            for cluster, group in df.groupby('cluster'):
                centroid = {
                    'bpm': group['bpm'].median(),
                    'centroid': group['centroid'].median(),
                    'danceability': group['danceability'].median(),
                }
                name = self.generate_playlist_name(centroid).replace(" ", "_")
                
                if name not in temp_playlists:
                    temp_playlists[name] = {
                        'tracks': group['filepath'].tolist(),
                        'features': centroid
                    }
                else:
                    temp_playlists[name]['tracks'].extend(group['filepath'].tolist())
            
            playlists = temp_playlists
            logger.info(f"Generated {len(playlists)} playlists from {len(df)} tracks")

        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
        return playlists

    def generate_playlist_name(self, features):
        bpm = features['bpm']
        centroid = features['centroid']
        danceability = features['danceability']
        
        bpm_desc = "Slow" if bpm < 70 else "Medium" if bpm < 100 else "Upbeat" if bpm < 130 else "Fast"
        dance_desc = "Chill" if danceability < 0.3 else "Easy" if danceability < 0.5 else "Groovy" if danceability < 0.7 else "Dance"
        mood_desc = "Relaxing" if centroid < 300 else "Mellow" if centroid < 1000 else "Balanced" if centroid < 2000 else "Bright"
        
        return f"{bpm_desc}_{dance_desc}_{mood_desc}"