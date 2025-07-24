import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import traceback
import sqlite3
import re

logger = logging.getLogger()

# In kmeans.py:

class KMeansPlaylistGenerator:
    """Generate playlists using KMeans clustering on audio features."""
    def __init__(self, cache_file: str = None) -> None:
        """Initialize the KMeans playlist generator.

        Args:
            cache_file (str, optional): Path to the cache database file. Defaults to None.
        """
        self.cache_file = cache_file
        self.playlist_history = {}
        if cache_file:
            self._verify_db_schema()

    def _verify_db_schema(self) -> None:
        """Ensure all required columns exist with correct types in the database."""
        conn = None
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
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
                    conn.execute(f"ALTER TABLE audio_features ADD COLUMN {col} {col_type}")
            conn.commit()
        except Exception as e:
            logger.error(f"Schema verification failed: {str(e)}")
        finally:
            if conn:
                conn.close()

    def _sanitize_file_name(self, name: str) -> str:
        """Sanitize a string to be used as a filename.

        Args:
            name (str): The string to sanitize.

        Returns:
            str: Sanitized filename.
        """
        name = re.sub(r'[^A-Za-z0-9_-]+', '_', name)
        name = re.sub(r'_+', '_', name)
        return name.strip('_')

    def _normalize_features(self, features):
        """Normalize feature values to appropriate ranges"""
        if not features:
            return features

        # Ensure danceability is between 0 and 1
        features['danceability'] = np.clip(features['danceability'], 0, 1)

        # Normalize BPM to a reasonable range (20-200)
        features['bpm'] = np.clip(features['bpm'], 20, 200)

        # Normalize centroid (typical range 0-10000)
        features['centroid'] = np.clip(features['centroid'], 0, 10000)

        # Add normalized loudness (typical range -60 to 0 dB)
        if 'loudness' in features:
            features['loudness'] = np.clip(features['loudness'], -60, 0)
            # Convert to 0-1 range
            features['loudness'] = (features['loudness'] + 60) / 60

        return features

    def generate(self, features_list, num_playlists=5, chunk_size=1000):
        playlists = {}
        try:
            data = []
            for f in features_list:
                if not f or 'filepath' not in f:
                    continue
                try:
                    track_data = {
                        'filepath': str(f['filepath']),
                        'bpm': float(f.get('bpm', 0)),
                        'centroid': float(f.get('centroid', 0)),
                        'danceability': float(f.get('danceability', 0)),
                        'loudness': float(f.get('loudness', -30)),
                        'onset_rate': float(f.get('onset_rate', 0)),
                        'zcr': float(f.get('zcr', 0)),
                        'key': int(f.get('key', -1)),
                        'scale': int(f.get('scale', 0))
                    }
                    # Normalize features
                    track_data = self._normalize_features(track_data)
                    data.append(track_data)
                except Exception as e:
                    logger.debug(f"Skipping track: {str(e)}")
                    continue

            if not data:
                logger.warning("No valid tracks after filtering")
                return playlists

            df = pd.DataFrame(data)
            cluster_features = [
                'bpm', 'centroid', 'danceability', 
                'loudness', 'onset_rate', 'zcr'
            ]
            weights = {
                'bpm': 0.3,
                'danceability': 0.2,
                'centroid': 0.2,
                'loudness': 0.1,
                'onset_rate': 0.1,
                'zcr': 0.1
            }
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(df[cluster_features])
            weighted_features = scaled_features * np.array([weights[f] for f in cluster_features])
            max_clusters = min(num_playlists * 3, len(df))
            kmeans = MiniBatchKMeans(
                n_clusters=max_clusters,
                random_state=42,
                batch_size=min(1000, len(df)),
                init='k-means++',
                max_iter=300,
                n_init=10
            )
            df['cluster'] = kmeans.fit_predict(weighted_features)
            temp_playlists = {}
            for cluster, group in df.groupby('cluster'):
                logger.debug(f"Cluster {cluster} size: {len(group)}")
                if len(group) < 3:
                    logger.debug(f"Cluster {cluster} too small, will be merged into fallback.")
                    continue
                centroid = {
                    'bpm': group['bpm'].median(),
                    'centroid': group['centroid'].median(),
                    'danceability': group['danceability'].median(),
                    'loudness': group['loudness'].median(),
                    'onset_rate': group['onset_rate'].median(),
                    'zcr': group['zcr'].median()
                }
                name = self._generate_descriptive_name(centroid)
                file_name = self._sanitize_file_name(name)
                if name in temp_playlists:
                    base_name = name
                    counter = 1
                    while name in temp_playlists:
                        name = f"{base_name}_Variation{counter}"
                        counter += 1
                    file_name = self._sanitize_file_name(name)
                temp_playlists[name] = {
                    'tracks': group['filepath'].tolist(),
                    'features': centroid,
                    'description': self._generate_description(centroid)
                    , 'file_name': file_name
                }
            # Fallback: merge all small/leftover tracks into Mixed_Collection
            assigned = set()
            for p in temp_playlists.values():
                assigned.update(p['tracks'])
            leftovers = [f['filepath'] for f in data if f['filepath'] not in assigned]
            if leftovers:
                logger.info(f"Merging {len(leftovers)} leftover/small-cluster tracks into Mixed_Collection.")
                temp_playlists['Mixed_Collection'] = {
                    'tracks': leftovers,
                    'features': {'type': 'mixed'},
                    'description': 'Tracks from small or leftover clusters.'
                }
            if not temp_playlists:
                logger.warning("No clusters large enough, all tracks go to Mixed_Collection.")
                temp_playlists['Mixed_Collection'] = {
                    'tracks': [f['filepath'] for f in data],
                    'features': {'type': 'mixed'},
                    'description': 'All tracks (no valid clusters)'
                }
            logger.info(f"Generated {len(temp_playlists)} playlists from {len(df)} tracks (kmeans)")
            return temp_playlists
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            logger.error(traceback.format_exc())
        return playlists

    def _generate_descriptive_name(self, features):
        """Generate a descriptive name based on musical features"""
        # BPM categories
        if features['bpm'] < 70:
            tempo = "Slow"
        elif features['bpm'] < 100:
            tempo = "Medium"
        elif features['bpm'] < 130:
            tempo = "Upbeat"
        else:
            tempo = "Fast"

        # Energy/Intensity based on multiple features
        energy_score = (
            features['danceability'] * 0.4 +
            features['loudness'] * 0.3 +
            features['onset_rate'] * 0.3
        )
        
        if energy_score < 0.3:
            energy = "Ambient"
        elif energy_score < 0.5:
            energy = "Chill"
        elif energy_score < 0.7:
            energy = "Groovy"
        else:
            energy = "Energetic"

        # Mood based on spectral features
        mood_score = (
            features['centroid'] * 0.6 +
            features['zcr'] * 0.4
        )
        
        if mood_score < 0.3:
            mood = "Dark"
        elif mood_score < 0.5:
            mood = "Warm"
        elif mood_score < 0.7:
            mood = "Bright"
        else:
            mood = "Crisp"

        return f"{tempo}_{energy}_{mood}"

    def _generate_description(self, features):
        """Generate a human-readable description of the playlist"""
        desc_parts = []
        
        # Tempo description
        if features['bpm'] < 70:
            desc_parts.append("Relaxing, slow-paced tracks")
        elif features['bpm'] < 100:
            desc_parts.append("Moderate tempo songs")
        elif features['bpm'] < 130:
            desc_parts.append("Upbeat, energetic music")
        else:
            desc_parts.append("High-energy, fast-paced tracks")

        # Energy/Mood description
        energy_level = features['danceability'] * 0.5 + features['loudness'] * 0.5
        if energy_level < 0.3:
            desc_parts.append("perfect for relaxation and meditation")
        elif energy_level < 0.5:
            desc_parts.append("great for focused work or unwinding")
        elif energy_level < 0.7:
            desc_parts.append("ideal for casual listening or light activity")
        else:
            desc_parts.append("perfect for workouts or parties")

        return " ".join(desc_parts)

    def _filter_and_sort_playlists(self, playlists, target_count):
        """Filter and sort playlists to get the most diverse selection"""
        if not playlists:
            return {}

        try:
            # Calculate diversity scores
            playlist_features = []
            for name, data in playlists.items():
                features = data['features']
                playlist_features.append({
                    'name': name,
                    'size': len(data['tracks']),
                    'bpm': features['bpm'],
                    'energy': features['danceability'] * 0.6 + features['loudness'] * 0.4,
                    'complexity': features['onset_rate'] * 0.5 + features['zcr'] * 0.5,
                    'brightness': features['centroid']
                })

            # Sort by size first
            size_sorted = sorted(playlist_features, key=lambda x: x['size'], reverse=True)

            # Take top 50% by size
            size_candidates = size_sorted[:int(len(size_sorted) * 0.5)]

            # Score remaining playlists by feature diversity
            selected = []
            while len(selected) < target_count and size_candidates:
                best_score = -1
                best_candidate = None
                best_idx = -1

                for idx, candidate in enumerate(size_candidates):
                    # Skip if too similar to already selected playlists
                    if any(self._playlist_similarity(candidate, sel) > 0.8 for sel in selected):
                        continue

                    # Calculate diversity score
                    score = self._calculate_diversity_score(candidate, selected)
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                        best_idx = idx

                if best_candidate is None:
                    break

                selected.append(best_candidate)
                size_candidates.pop(best_idx)

            # Return the selected playlists
            return {
                name: playlists[name]
                for name in [p['name'] for p in selected]
            }

        except Exception as e:
            logger.error(f"Error filtering playlists: {str(e)}")
            # Fallback to simple size-based selection
            sorted_playlists = sorted(
                playlists.items(),
                key=lambda x: len(x[1]['tracks']),
                reverse=True
            )
            return dict(sorted_playlists[:target_count])

    def _playlist_similarity(self, p1, p2):
        """Calculate similarity between two playlists based on features"""
        if not p2:  # First playlist
            return 0

        bpm_diff = abs(p1['bpm'] - p2['bpm']) / max(p1['bpm'], p2['bpm'])
        energy_diff = abs(p1['energy'] - p2['energy'])
        complexity_diff = abs(p1['complexity'] - p2['complexity'])
        brightness_diff = abs(p1['brightness'] - p2['brightness']) / max(p1['brightness'], p2['brightness'])

        # Weighted similarity (lower is more different)
        return (bpm_diff * 0.3 + energy_diff * 0.3 + 
                complexity_diff * 0.2 + brightness_diff * 0.2)

    def _calculate_diversity_score(self, candidate, selected):
        """Calculate how much diversity this candidate adds to selection"""
        if not selected:
            return 1.0  # First playlist gets maximum score

        # Calculate average similarity to existing playlists
        similarities = [self._playlist_similarity(candidate, sel) for sel in selected]
        avg_similarity = sum(similarities) / len(similarities)

        # Size bonus (0.0 to 0.2)
        size_score = min(1.0, candidate['size'] / 100) * 0.2

        # Return diversity score (higher is better)
        return (1.0 - avg_similarity) + size_score