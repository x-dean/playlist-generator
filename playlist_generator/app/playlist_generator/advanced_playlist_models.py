import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import os

logger = logging.getLogger(__name__)


class AdvancedPlaylistModels:
    """Advanced playlist generation models using machine learning techniques."""
    
    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file
        self.scaler = StandardScaler()
        self.feature_weights = {
            'bpm': 0.15,
            'danceability': 0.15,
            'centroid': 0.10,
            'loudness': 0.10,
            'valence': 0.10,
            'arousal': 0.10,
            'energy_level': 0.10,
            'complexity_score': 0.05,
            'onset_rate': 0.05,
            'zcr': 0.05,
            'key': 0.05
        }
    
    def generate_hybrid_playlists(self, features_list: List[Dict], 
                                 num_playlists: int = 8,
                                 method: str = 'ensemble') -> Dict[str, Any]:
        """Generate playlists using hybrid approaches combining multiple methods."""
        
        if method == 'ensemble':
            return self._ensemble_playlist_generation(features_list, num_playlists)
        elif method == 'hierarchical':
            return self._hierarchical_playlist_generation(features_list, num_playlists)
        elif method == 'recommendation':
            return self._recommendation_based_playlists(features_list, num_playlists)
        elif method == 'mood_based':
            return self._mood_based_playlists(features_list, num_playlists)
        else:
            logger.warning(f"Unknown method: {method}, using ensemble")
            return self._ensemble_playlist_generation(features_list, num_playlists)
    
    def _ensemble_playlist_generation(self, features_list: List[Dict], 
                                     num_playlists: int) -> Dict[str, Any]:
        """Generate playlists using ensemble of multiple clustering methods."""
        
        logger.info(f"Starting ensemble playlist generation with {len(features_list)} tracks")
        
        # Prepare features
        df = self._prepare_features_dataframe(features_list)
        if df.empty:
            return {}
        
        # Apply multiple clustering methods
        clusterings = {}
        
        # 1. K-means clustering
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(num_playlists, len(df)), random_state=42)
            clusterings['kmeans'] = kmeans.fit_predict(df)
            logger.debug("K-means clustering completed")
        except Exception as e:
            logger.warning(f"K-means clustering failed: {str(e)}")
        
        # 2. DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=0.3, min_samples=3)
            clusterings['dbscan'] = dbscan.fit_predict(df)
            logger.debug("DBSCAN clustering completed")
        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {str(e)}")
        
        # 3. Hierarchical clustering
        try:
            hierarchical = AgglomerativeClustering(n_clusters=min(num_playlists, len(df)))
            clusterings['hierarchical'] = hierarchical.fit_predict(df)
            logger.debug("Hierarchical clustering completed")
        except Exception as e:
            logger.warning(f"Hierarchical clustering failed: {str(e)}")
        
        # Combine clusterings using voting
        final_clusters = self._combine_clusterings(clusterings, df)
        
        # Generate playlists from combined clusters
        playlists = self._create_playlists_from_clusters(final_clusters, features_list)
        
        logger.info(f"Ensemble playlist generation completed: {len(playlists)} playlists")
        return playlists
    
    def _hierarchical_playlist_generation(self, features_list: List[Dict], 
                                         num_playlists: int) -> Dict[str, Any]:
        """Generate hierarchical playlists with nested structure."""
        
        logger.info(f"Starting hierarchical playlist generation with {len(features_list)} tracks")
        
        # Prepare features
        df = self._prepare_features_dataframe(features_list)
        if df.empty:
            return {}
        
        # First level: Group by primary mood
        mood_groups = self._group_by_mood(features_list)
        
        # Second level: Within each mood, cluster by musical features
        playlists = {}
        
        for mood, tracks in mood_groups.items():
            if len(tracks) < 3:
                continue
                
            # Cluster tracks within this mood
            mood_df = self._prepare_features_dataframe(tracks)
            if mood_df.empty:
                continue
            
            # Use fewer clusters for mood subgroups
            n_clusters = min(3, len(mood_df))
            if n_clusters < 2:
                continue
            
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(mood_df)
                
                # Create playlists for each cluster
                for cluster_id in range(n_clusters):
                    cluster_tracks = [tracks[i] for i in range(len(tracks)) if clusters[i] == cluster_id]
                    if len(cluster_tracks) >= 3:
                        playlist_name = f"{mood.capitalize()}_{cluster_id + 1}"
                        playlists[playlist_name] = {
                            'tracks': [t['filepath'] for t in cluster_tracks],
                            'features': self._calculate_playlist_features(cluster_tracks),
                            'mood': mood,
                            'cluster_id': cluster_id
                        }
                        
            except Exception as e:
                logger.warning(f"Clustering failed for mood {mood}: {str(e)}")
        
        logger.info(f"Hierarchical playlist generation completed: {len(playlists)} playlists")
        return playlists
    
    def _recommendation_based_playlists(self, features_list: List[Dict], 
                                       num_playlists: int) -> Dict[str, Any]:
        """Generate playlists using recommendation system approach."""
        
        logger.info(f"Starting recommendation-based playlist generation with {len(features_list)} tracks")
        
        # Prepare features matrix
        df = self._prepare_features_dataframe(features_list)
        if df.empty:
            return {}
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(df)
        
        # Find diverse seed tracks for each playlist
        seed_tracks = self._select_diverse_seeds(similarity_matrix, num_playlists)
        
        playlists = {}
        
        for i, seed_idx in enumerate(seed_tracks):
            # Get similar tracks to this seed
            similar_tracks = self._get_similar_tracks(seed_idx, similarity_matrix, 
                                                    features_list, max_tracks=20)
            
            if len(similar_tracks) >= 3:
                playlist_name = f"Recommended_Playlist_{i + 1}"
                playlists[playlist_name] = {
                    'tracks': [t['filepath'] for t in similar_tracks],
                    'features': self._calculate_playlist_features(similar_tracks),
                    'seed_track': features_list[seed_idx]['filepath'],
                    'similarity_threshold': 0.7
                }
        
        logger.info(f"Recommendation-based playlist generation completed: {len(playlists)} playlists")
        return playlists
    
    def _mood_based_playlists(self, features_list: List[Dict], 
                              num_playlists: int) -> Dict[str, Any]:
        """Generate playlists based on mood classification."""
        
        logger.info(f"Starting mood-based playlist generation with {len(features_list)} tracks")
        
        # Group tracks by mood
        mood_groups = self._group_by_mood(features_list)
        
        playlists = {}
        
        for mood, tracks in mood_groups.items():
            if len(tracks) < 3:
                continue
            
            # Split large mood groups into sub-playlists
            if len(tracks) > 30:
                sub_playlists = self._split_large_mood_group(tracks, mood)
                playlists.update(sub_playlists)
            else:
                playlist_name = f"Mood_{mood.capitalize()}"
                playlists[playlist_name] = {
                    'tracks': [t['filepath'] for t in tracks],
                    'features': self._calculate_playlist_features(tracks),
                    'mood': mood,
                    'size': len(tracks)
                }
        
        logger.info(f"Mood-based playlist generation completed: {len(playlists)} playlists")
        return playlists
    
    def _prepare_features_dataframe(self, features_list: List[Dict]) -> pd.DataFrame:
        """Prepare features for machine learning models."""
        
        data = []
        
        for track in features_list:
            if not track or 'filepath' not in track:
                continue
            
            # Extract numerical features
            track_features = {
                'bpm': float(track.get('bpm', 120)),
                'danceability': float(track.get('danceability', 0.5)),
                'centroid': float(track.get('centroid', 2000)),
                'loudness': float(track.get('loudness', -20)),
                'onset_rate': float(track.get('onset_rate', 0)),
                'zcr': float(track.get('zcr', 0))
            }
            
            # Add emotional features if available
            if 'valence' in track:
                track_features['valence'] = float(track['valence'])
            if 'arousal' in track:
                track_features['arousal'] = float(track['arousal'])
            if 'energy_level' in track:
                track_features['energy_level'] = float(track['energy_level'])
            if 'complexity_score' in track:
                track_features['complexity_score'] = float(track['complexity_score'])
            
            # Add key as numerical feature
            key = track.get('key', 'C')
            if isinstance(key, str):
                key_mapping = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                              'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
                track_features['key'] = key_mapping.get(key, 0)
            else:
                track_features['key'] = int(key) if key is not None else 0
            
            data.append(track_features)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Normalize features
        df_scaled = self.scaler.fit_transform(df)
        
        return pd.DataFrame(df_scaled, columns=df.columns)
    
    def _combine_clusterings(self, clusterings: Dict[str, np.ndarray], 
                            df: pd.DataFrame) -> np.ndarray:
        """Combine multiple clusterings using voting mechanism."""
        
        if not clusterings:
            return np.zeros(len(df))
        
        # Create voting matrix
        n_samples = len(df)
        n_methods = len(clusterings)
        
        # Initialize combined clustering
        combined_clusters = np.zeros(n_samples, dtype=int)
        
        # Simple voting: assign to most common cluster across methods
        for i in range(n_samples):
            votes = []
            for method, clusters in clusterings.items():
                if i < len(clusters):
                    votes.append(clusters[i])
            
            if votes:
                # Use most common cluster
                from collections import Counter
                vote_counts = Counter(votes)
                combined_clusters[i] = vote_counts.most_common(1)[0][0]
        
        return combined_clusters
    
    def _create_playlists_from_clusters(self, clusters: np.ndarray, 
                                       features_list: List[Dict]) -> Dict[str, Any]:
        """Create playlists from clustering results."""
        
        playlists = {}
        
        # Group tracks by cluster
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if i < len(features_list):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(features_list[i])
        
        # Create playlists for each cluster
        for cluster_id, tracks in cluster_groups.items():
            if len(tracks) >= 3:
                playlist_name = f"Ensemble_Cluster_{cluster_id + 1}"
                playlists[playlist_name] = {
                    'tracks': [t['filepath'] for t in tracks],
                    'features': self._calculate_playlist_features(tracks),
                    'cluster_id': cluster_id,
                    'size': len(tracks)
                }
        
        return playlists
    
    def _group_by_mood(self, features_list: List[Dict]) -> Dict[str, List[Dict]]:
        """Group tracks by their primary mood."""
        
        mood_groups = {
            'happy': [],
            'sad': [],
            'energetic': [],
            'calm': [],
            'aggressive': [],
            'melancholic': [],
            'unknown': []
        }
        
        for track in features_list:
            if not track:
                continue
            
            # Get primary mood
            primary_mood = track.get('primary_mood', 'unknown')
            if primary_mood in mood_groups:
                mood_groups[primary_mood].append(track)
            else:
                mood_groups['unknown'].append(track)
        
        # Remove empty groups
        return {k: v for k, v in mood_groups.items() if v}
    
    def _split_large_mood_group(self, tracks: List[Dict], mood: str) -> Dict[str, Any]:
        """Split a large mood group into multiple playlists."""
        
        playlists = {}
        
        # Use clustering to split the group
        df = self._prepare_features_dataframe(tracks)
        if df.empty:
            return {}
        
        try:
            from sklearn.cluster import KMeans
            n_clusters = min(3, len(df) // 10)  # Max 3 sub-playlists
            if n_clusters < 2:
                return {f"Mood_{mood.capitalize()}": {
                    'tracks': [t['filepath'] for t in tracks],
                    'features': self._calculate_playlist_features(tracks),
                    'mood': mood
                }}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(df)
            
            for cluster_id in range(n_clusters):
                cluster_tracks = [tracks[i] for i in range(len(tracks)) if clusters[i] == cluster_id]
                if len(cluster_tracks) >= 3:
                    playlist_name = f"Mood_{mood.capitalize()}_{cluster_id + 1}"
                    playlists[playlist_name] = {
                        'tracks': [t['filepath'] for t in cluster_tracks],
                        'features': self._calculate_playlist_features(cluster_tracks),
                        'mood': mood,
                        'sub_cluster': cluster_id
                    }
                    
        except Exception as e:
            logger.warning(f"Failed to split mood group {mood}: {str(e)}")
            # Fallback to single playlist
            playlists[f"Mood_{mood.capitalize()}"] = {
                'tracks': [t['filepath'] for t in tracks],
                'features': self._calculate_playlist_features(tracks),
                'mood': mood
            }
        
        return playlists
    
    def _select_diverse_seeds(self, similarity_matrix: np.ndarray, 
                              num_playlists: int) -> List[int]:
        """Select diverse seed tracks for recommendation-based playlists."""
        
        n_tracks = len(similarity_matrix)
        if n_tracks == 0:
            return []
        
        # Start with random seed
        seeds = [np.random.randint(0, n_tracks)]
        
        # Add seeds that are most different from existing seeds
        for _ in range(1, min(num_playlists, n_tracks)):
            max_min_similarity = -1
            best_seed = 0
            
            for i in range(n_tracks):
                if i in seeds:
                    continue
                
                # Find minimum similarity to existing seeds
                min_similarity = min(similarity_matrix[i][j] for j in seeds)
                
                if min_similarity > max_min_similarity:
                    max_min_similarity = min_similarity
                    best_seed = i
            
            seeds.append(best_seed)
        
        return seeds
    
    def _get_similar_tracks(self, seed_idx: int, similarity_matrix: np.ndarray,
                           features_list: List[Dict], max_tracks: int = 20) -> List[Dict]:
        """Get tracks similar to the seed track."""
        
        if seed_idx >= len(similarity_matrix):
            return []
        
        # Get similarities to seed track
        similarities = similarity_matrix[seed_idx]
        
        # Get indices of most similar tracks
        similar_indices = np.argsort(similarities)[::-1][:max_tracks]
        
        # Return similar tracks
        similar_tracks = []
        for idx in similar_indices:
            if idx < len(features_list):
                similar_tracks.append(features_list[idx])
        
        return similar_tracks
    
    def _calculate_playlist_features(self, tracks: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate features for a playlist."""
        
        if not tracks:
            return {}
        
        features = {}
        
        # Calculate averages for numerical features
        numerical_features = ['bpm', 'danceability', 'centroid', 'loudness', 
                            'onset_rate', 'zcr', 'valence', 'arousal', 'energy_level', 'complexity_score']
        
        for feature in numerical_features:
            values = [float(track.get(feature, 0)) for track in tracks if track.get(feature) is not None]
            if values:
                features[f'avg_{feature}'] = np.mean(values)
                features[f'std_{feature}'] = np.std(values)
        
        # Calculate most common mood
        moods = [track.get('primary_mood', 'unknown') for track in tracks]
        if moods:
            from collections import Counter
            mood_counts = Counter(moods)
            features['primary_mood'] = mood_counts.most_common(1)[0][0]
            features['mood_confidence'] = mood_counts.most_common(1)[0][1] / len(moods)
        
        # Calculate playlist size
        features['size'] = len(tracks)
        
        return features 