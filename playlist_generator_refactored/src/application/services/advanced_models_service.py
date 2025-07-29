"""
Advanced playlist models service.
Ports the original AdvancedPlaylistModels to the new architecture.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random

# Machine learning libraries
try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("scikit-learn not available - advanced models will be limited")

from domain.entities.audio_file import AudioFile
from domain.entities.playlist import Playlist
from shared.exceptions import PlaylistGenerationError

class AdvancedModelsService:
    """Service for advanced playlist generation using machine learning techniques."""
    
    def __init__(self, cache_file: str = None):
        """Initialize the advanced models service.
        
        Args:
            cache_file: Path to the cache database file
        """
        self.cache_file = cache_file
        self.logger = logging.getLogger(__name__)
        
        # Check for required libraries
        if not ML_AVAILABLE:
            self.logger.warning("scikit-learn not available - advanced models will be limited")
        
        # Initialize ML components
        self.scaler = StandardScaler() if ML_AVAILABLE else None
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
    
    def generate_advanced_playlists(self, audio_files: List[AudioFile], 
                                   method: str = 'ensemble',
                                   num_playlists: int = 8) -> List[Playlist]:
        """Generate playlists using advanced machine learning methods."""
        self.logger.info(f"Generating advanced playlists using {method} method")
        
        try:
            if method == 'ensemble':
                return self._ensemble_playlist_generation(audio_files, num_playlists)
            elif method == 'hierarchical':
                return self._hierarchical_playlist_generation(audio_files, num_playlists)
            elif method == 'recommendation':
                return self._recommendation_based_playlists(audio_files, num_playlists)
            elif method == 'mood_based':
                return self._mood_based_playlists(audio_files, num_playlists)
            else:
                self.logger.warning(f"Unknown method: {method}, using ensemble")
                return self._ensemble_playlist_generation(audio_files, num_playlists)
                
        except Exception as e:
            self.logger.error(f"Advanced playlist generation failed: {e}")
            raise PlaylistGenerationError(f"Advanced generation failed: {e}")
    
    def _ensemble_playlist_generation(self, audio_files: List[AudioFile], 
                                     num_playlists: int) -> List[Playlist]:
        """Generate playlists using ensemble of multiple clustering methods."""
        
        self.logger.info(f"Starting ensemble playlist generation with {len(audio_files)} tracks")
        
        try:
            # Prepare features
            df = self._prepare_features_dataframe(audio_files)
            if df.empty or len(df) < 3:
                self.logger.warning("Insufficient data for ensemble clustering")
                return self._create_simple_playlists(audio_files)
            
            # Apply multiple clustering methods
            clusterings = {}
            
            # 1. K-means clustering
            try:
                n_clusters = min(num_playlists, len(df), 10)  # Limit clusters
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusterings['kmeans'] = kmeans.fit_predict(df)
                    self.logger.debug("K-means clustering completed")
            except Exception as e:
                self.logger.warning(f"K-means clustering failed: {str(e)}")
            
            # 2. DBSCAN clustering
            try:
                if len(df) >= 5:  # DBSCAN needs minimum samples
                    dbscan = DBSCAN(eps=0.3, min_samples=min(3, len(df)//2))
                    clusterings['dbscan'] = dbscan.fit_predict(df)
                    self.logger.debug("DBSCAN clustering completed")
            except Exception as e:
                self.logger.warning(f"DBSCAN clustering failed: {str(e)}")
            
            # 3. Hierarchical clustering
            try:
                n_clusters = min(num_playlists, len(df), 8)  # Limit clusters
                if n_clusters >= 2:
                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                    clusterings['hierarchical'] = hierarchical.fit_predict(df)
                    self.logger.debug("Hierarchical clustering completed")
            except Exception as e:
                self.logger.warning(f"Hierarchical clustering failed: {str(e)}")
            
            # Combine clusterings using voting
            if clusterings:
                final_clusters = self._combine_clusterings(clusterings, df)
                # Generate playlists from combined clusters
                playlists = self._create_playlists_from_clusters(final_clusters, audio_files)
            else:
                self.logger.warning("No clustering methods succeeded, using simple grouping")
                playlists = self._create_simple_playlists(audio_files)
            
            self.logger.info(f"Ensemble playlist generation completed: {len(playlists)} playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"Ensemble generation failed: {e}")
            return self._create_simple_playlists(audio_files)
    
    def _hierarchical_playlist_generation(self, audio_files: List[AudioFile], 
                                         num_playlists: int) -> List[Playlist]:
        """Generate playlists using hierarchical clustering."""
        
        self.logger.info(f"Starting hierarchical playlist generation with {len(audio_files)} tracks")
        
        # Prepare features
        df = self._prepare_features_dataframe(audio_files)
        if df.empty:
            return []
        
        playlists = []
        
        try:
            # Apply hierarchical clustering
            hierarchical = AgglomerativeClustering(n_clusters=min(num_playlists, len(df)))
            clusters = hierarchical.fit_predict(df)
            
            # Create playlists from clusters
            for cluster_id in range(max(clusters) + 1):
                cluster_tracks = [audio_files[i] for i in range(len(audio_files)) if clusters[i] == cluster_id]
                
                if len(cluster_tracks) >= 3:
                    playlist = self._create_playlist_from_tracks(
                        cluster_tracks, f"Hierarchical_Cluster_{cluster_id + 1}"
                    )
                    playlists.append(playlist)
            
        except Exception as e:
            self.logger.warning(f"Hierarchical clustering failed: {str(e)}")
        
        self.logger.info(f"Hierarchical playlist generation completed: {len(playlists)} playlists")
        return playlists
    
    def _recommendation_based_playlists(self, audio_files: List[AudioFile], 
                                       num_playlists: int) -> List[Playlist]:
        """Generate playlists using recommendation system approach."""
        
        self.logger.info(f"Starting recommendation-based playlist generation with {len(audio_files)} tracks")
        
        # Prepare features matrix
        df = self._prepare_features_dataframe(audio_files)
        if df.empty:
            return []
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(df)
        
        # Find diverse seed tracks for each playlist
        seed_tracks = self._select_diverse_seeds(similarity_matrix, num_playlists)
        
        playlists = []
        
        for i, seed_idx in enumerate(seed_tracks):
            # Get similar tracks to this seed
            similar_tracks = self._get_similar_tracks(seed_idx, similarity_matrix, 
                                                    audio_files, max_tracks=20)
            
            if len(similar_tracks) >= 3:
                playlist = self._create_playlist_from_tracks(
                    similar_tracks, f"Recommended_Playlist_{i + 1}"
                )
                playlists.append(playlist)
        
        self.logger.info(f"Recommendation-based playlist generation completed: {len(playlists)} playlists")
        return playlists
    
    def _mood_based_playlists(self, audio_files: List[AudioFile], 
                              num_playlists: int) -> List[Playlist]:
        """Generate playlists based on mood classification."""
        
        self.logger.info(f"Starting mood-based playlist generation with {len(audio_files)} tracks")
        
        # Group tracks by mood
        mood_groups = self._group_by_mood(audio_files)
        
        playlists = []
        
        for mood, tracks in mood_groups.items():
            if len(tracks) < 3:
                continue
            
            # Split large mood groups into sub-playlists
            if len(tracks) > 30:
                sub_playlists = self._split_large_mood_group(tracks, mood)
                playlists.extend(sub_playlists)
            else:
                playlist = self._create_playlist_from_tracks(tracks, f"Mood_{mood.capitalize()}")
                playlists.append(playlist)
        
        self.logger.info(f"Mood-based playlist generation completed: {len(playlists)} playlists")
        return playlists
    
    def _prepare_features_dataframe(self, audio_files: List[AudioFile]) -> pd.DataFrame:
        """Prepare features for machine learning models."""
        
        data = []
        
        for audio_file in audio_files:
            features = audio_file.external_metadata.get('features', {})
            if not features:
                continue
            
            try:
                # Extract numerical features with safe defaults
                track_features = {
                    'bpm': max(0, float(features.get('bpm', 120))),
                    'danceability': max(0, min(1, float(features.get('danceability', 0.5)))),
                    'centroid': max(0, float(features.get('centroid', 2000))),
                    'loudness': max(-60, min(0, float(features.get('loudness', -20)))),
                    'onset_rate': max(0, float(features.get('onset_rate', 0))),
                    'zcr': max(0, min(1, float(features.get('zcr', 0))))
                }
                
                # Add emotional features if available
                if 'valence' in features:
                    track_features['valence'] = max(0, min(1, float(features['valence'])))
                if 'arousal' in features:
                    track_features['arousal'] = max(0, min(1, float(features['arousal'])))
                if 'energy_level' in features:
                    track_features['energy_level'] = max(0, min(1, float(features['energy_level'])))
                if 'complexity_score' in features:
                    track_features['complexity_score'] = max(0, min(1, float(features['complexity_score'])))
                
                data.append(track_features)
                
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error processing features for {audio_file.file_path}: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Ensure all values are non-negative
        for col in df.columns:
            if df[col].min() < 0:
                self.logger.warning(f"Found negative values in {col}, shifting to non-negative")
                df[col] = df[col] - df[col].min()  # Shift to non-negative
        
        # Normalize features only if we have enough data
        if self.scaler and not df.empty and len(df) > 1:
            try:
                # Ensure no infinite or NaN values
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(df.mean())
                
                scaled_data = self.scaler.fit_transform(df)
                df = pd.DataFrame(scaled_data, columns=df.columns)
                
            except Exception as e:
                self.logger.warning(f"Feature scaling failed: {e}, using original features")
        
        return df
    
    def _combine_clusterings(self, clusterings: Dict[str, np.ndarray], 
                             df: pd.DataFrame) -> np.ndarray:
        """Combine multiple clustering results using voting."""
        
        if not clusterings:
            return np.zeros(len(df))
        
        # Create voting matrix
        voting_matrix = np.zeros((len(df), max(len(clusterings), 1)))
        
        for i, (method, clusters) in enumerate(clusterings.items()):
            voting_matrix[:, i] = clusters
        
        # Use majority voting
        final_clusters = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=voting_matrix
        )
        
        return final_clusters
    
    def _create_playlists_from_clusters(self, clusters: np.ndarray, 
                                       audio_files: List[AudioFile]) -> List[Playlist]:
        """Create playlists from clustering results."""
        
        playlists = []
        
        for cluster_id in range(max(clusters) + 1):
            cluster_tracks = [audio_files[i] for i in range(len(audio_files)) if clusters[i] == cluster_id]
            
            if len(cluster_tracks) >= 3:
                playlist = self._create_playlist_from_tracks(
                    cluster_tracks, f"Ensemble_Cluster_{cluster_id + 1}"
                )
                playlists.append(playlist)
        
        return playlists
    
    def _group_by_mood(self, audio_files: List[AudioFile]) -> Dict[str, List[AudioFile]]:
        """Group tracks by mood based on audio features."""
        
        mood_groups = {}
        
        for audio_file in audio_files:
            features = audio_file.external_metadata.get('features', {})
            if not features:
                continue
            
            # Simple mood classification based on features
            bpm = features.get('bpm', 120)
            danceability = features.get('danceability', 0.5)
            valence = features.get('valence', 0.5)
            
            if bpm > 140 and danceability > 0.7:
                mood = "Energetic"
            elif bpm < 80 and danceability < 0.3:
                mood = "Chill"
            elif valence > 0.7:
                mood = "Happy"
            elif valence < 0.3:
                mood = "Melancholic"
            else:
                mood = "Balanced"
            
            if mood not in mood_groups:
                mood_groups[mood] = []
            mood_groups[mood].append(audio_file)
        
        return mood_groups
    
    def _split_large_mood_group(self, tracks: List[AudioFile], mood: str) -> List[Playlist]:
        """Split large mood groups into sub-playlists."""
        
        playlists = []
        chunk_size = 15
        
        for i in range(0, len(tracks), chunk_size):
            chunk = tracks[i:i + chunk_size]
            if len(chunk) >= 3:
                playlist = self._create_playlist_from_tracks(
                    chunk, f"{mood}_Part_{i // chunk_size + 1}"
                )
                playlists.append(playlist)
        
        return playlists
    
    def _select_diverse_seeds(self, similarity_matrix: np.ndarray, 
                              num_playlists: int) -> List[int]:
        """Select diverse seed tracks for recommendation-based playlists."""
        
        if len(similarity_matrix) == 0:
            return []
        
        # Start with a random track
        seeds = [random.randint(0, len(similarity_matrix) - 1)]
        
        # Find tracks that are least similar to existing seeds
        for _ in range(min(num_playlists - 1, len(similarity_matrix) - 1)):
            min_similarity = float('inf')
            best_seed = -1
            
            for i in range(len(similarity_matrix)):
                if i in seeds:
                    continue
                
                # Calculate average similarity to existing seeds
                avg_similarity = np.mean([similarity_matrix[i][j] for j in seeds])
                
                if avg_similarity < min_similarity:
                    min_similarity = avg_similarity
                    best_seed = i
            
            if best_seed != -1:
                seeds.append(best_seed)
        
        return seeds
    
    def _get_similar_tracks(self, seed_idx: int, similarity_matrix: np.ndarray,
                           audio_files: List[AudioFile], max_tracks: int = 20) -> List[AudioFile]:
        """Get tracks similar to a seed track."""
        
        if seed_idx >= len(similarity_matrix):
            return []
        
        # Get similarity scores for the seed track
        similarities = similarity_matrix[seed_idx]
        
        # Sort by similarity (descending)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Return top similar tracks (excluding the seed itself)
        similar_tracks = []
        for idx in similar_indices:
            if idx != seed_idx and len(similar_tracks) < max_tracks:
                similar_tracks.append(audio_files[idx])
        
        return similar_tracks
    
    def _create_playlist_from_tracks(self, tracks: List[AudioFile], name: str) -> Playlist:
        """Create a playlist from a list of tracks."""
        
        track_ids = [track.id for track in tracks]
        track_paths = [str(track.file_path) for track in tracks]
        
        return Playlist(
            name=name,
            description=f"Advanced playlist generated using ML techniques",
            track_ids=track_ids,
            track_paths=track_paths,
            generation_method="advanced",
            playlist_type="ml_generated",
            created_date=datetime.now()
        )
    
    def _create_simple_playlists(self, audio_files: List[AudioFile]) -> List[Playlist]:
        """Create simple playlists when clustering fails."""
        
        if len(audio_files) < 3:
            return []
        
        # Create a single playlist with all tracks
        playlist = self._create_playlist_from_tracks(
            audio_files, "Simple_Advanced_Playlist"
        )
        
        return [playlist] 