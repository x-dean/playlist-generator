"""
Advanced playlist generation algorithms
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from ..core.logging import get_logger, LogContext, log_operation_success
from ..core.config import get_settings

logger = get_logger("playlist.algorithms")
settings = get_settings()


class PlaylistAlgorithms:
    """Collection of intelligent playlist generation algorithms"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        logger.info("PlaylistAlgorithms initialized")
    
    def generate_similar_tracks_playlist(
        self,
        tracks: List[Dict[str, Any]],
        seed_track: Dict[str, Any],
        playlist_size: int = 25,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate playlist based on similarity to a seed track
        """
        with LogContext(
            operation="generate_similar_playlist",
            seed_track=seed_track.get("title", "Unknown"),
            target_size=playlist_size
        ):
            if not tracks:
                return []
            
            # Extract features for similarity calculation
            track_features = self._extract_similarity_features(tracks)
            seed_features = self._extract_similarity_features([seed_track])
            
            if len(track_features) == 0 or len(seed_features) == 0:
                logger.warning("No features available for similarity calculation")
                return tracks[:playlist_size]
            
            # Calculate similarities
            similarities = cosine_similarity(seed_features, track_features)[0]
            
            # Create track-similarity pairs
            track_similarities = list(zip(tracks, similarities))
            
            # Filter by threshold and sort by similarity
            similar_tracks = [
                track for track, sim in track_similarities 
                if sim >= similarity_threshold
            ]
            
            # Sort by similarity (descending)
            similar_tracks.sort(
                key=lambda t: track_similarities[tracks.index(t)][1],
                reverse=True
            )
            
            # Take top tracks
            result = similar_tracks[:playlist_size]
            
            avg_similarity = np.mean([
                track_similarities[tracks.index(t)][1] for t in result
            ]) if result else 0
            
            logger.info(
                "Similar tracks playlist generated",
                result_size=len(result),
                average_similarity=round(avg_similarity, 3),
                threshold=similarity_threshold
            )
            
            return result
    
    def generate_kmeans_playlist(
        self,
        tracks: List[Dict[str, Any]],
        num_clusters: int = 5,
        tracks_per_cluster: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate playlist using K-means clustering
        """
        with LogContext(
            operation="generate_kmeans_playlist",
            num_clusters=num_clusters,
            tracks_per_cluster=tracks_per_cluster
        ):
            if len(tracks) < num_clusters:
                logger.warning(
                    "Not enough tracks for clustering",
                    available_tracks=len(tracks),
                    required_clusters=num_clusters
                )
                return tracks
            
            # Extract features
            features = self._extract_clustering_features(tracks)
            
            if len(features) == 0:
                logger.warning("No features available for clustering")
                return tracks[:num_clusters * tracks_per_cluster]
            
            # Normalize features
            normalized_features = self.scaler.fit_transform(features)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_features)
            
            # Select tracks from each cluster
            playlist = []
            for cluster_id in range(num_clusters):
                cluster_tracks = [
                    tracks[i] for i, label in enumerate(cluster_labels)
                    if label == cluster_id
                ]
                
                # Sort by distance to cluster center
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                if cluster_indices:
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    distances = [
                        np.linalg.norm(normalized_features[i] - cluster_center)
                        for i in cluster_indices
                    ]
                    
                    # Sort tracks by distance to center
                    sorted_tracks = [
                        cluster_tracks[i] for i in np.argsort(distances)
                    ]
                    
                    playlist.extend(sorted_tracks[:tracks_per_cluster])
            
            logger.info(
                "K-means playlist generated",
                clusters=num_clusters,
                tracks_per_cluster=tracks_per_cluster,
                result_size=len(playlist)
            )
            
            return playlist
    
    def generate_energy_flow_playlist(
        self,
        tracks: List[Dict[str, Any]],
        playlist_size: int = 25,
        energy_progression: str = "ascending"  # ascending, descending, wave
    ) -> List[Dict[str, Any]]:
        """
        Generate playlist with specific energy flow
        """
        with LogContext(
            operation="generate_energy_flow_playlist",
            progression=energy_progression,
            target_size=playlist_size
        ):
            # Filter tracks with energy values
            energy_tracks = [
                track for track in tracks
                if track.get("energy") is not None
            ]
            
            if not energy_tracks:
                logger.warning("No tracks with energy values available")
                return tracks[:playlist_size]
            
            # Sort by energy
            if energy_progression == "ascending":
                sorted_tracks = sorted(energy_tracks, key=lambda t: t.get("energy", 0))
            elif energy_progression == "descending":
                sorted_tracks = sorted(energy_tracks, key=lambda t: t.get("energy", 0), reverse=True)
            else:  # wave
                # Create a wave pattern: low -> high -> low
                sorted_tracks = sorted(energy_tracks, key=lambda t: t.get("energy", 0))
                mid_point = len(sorted_tracks) // 2
                wave_tracks = (
                    sorted_tracks[:mid_point:2] +  # Low energy tracks
                    sorted_tracks[::-1][:mid_point:2] +  # High energy tracks
                    sorted_tracks[1:mid_point:2]  # Back to low energy
                )
                sorted_tracks = wave_tracks
            
            result = sorted_tracks[:playlist_size]
            
            avg_energy = np.mean([t.get("energy", 0) for t in result])
            energy_std = np.std([t.get("energy", 0) for t in result])
            
            logger.info(
                "Energy flow playlist generated",
                progression=energy_progression,
                result_size=len(result),
                avg_energy=round(avg_energy, 3),
                energy_variation=round(energy_std, 3)
            )
            
            return result
    
    def generate_mood_journey_playlist(
        self,
        tracks: List[Dict[str, Any]],
        start_mood: str,
        end_mood: str,
        playlist_size: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Generate playlist that transitions between moods
        """
        with LogContext(
            operation="generate_mood_journey_playlist",
            start_mood=start_mood,
            end_mood=end_mood,
            target_size=playlist_size
        ):
            # Define mood mappings
            mood_features = {
                "happy": {"valence": 0.8, "energy": 0.7},
                "sad": {"valence": 0.2, "energy": 0.3},
                "energetic": {"valence": 0.6, "energy": 0.9},
                "calm": {"valence": 0.5, "energy": 0.2},
                "angry": {"valence": 0.2, "energy": 0.8},
                "relaxed": {"valence": 0.7, "energy": 0.3}
            }
            
            start_target = mood_features.get(start_mood, {"valence": 0.5, "energy": 0.5})
            end_target = mood_features.get(end_mood, {"valence": 0.5, "energy": 0.5})
            
            # Filter tracks with mood features
            mood_tracks = [
                track for track in tracks
                if track.get("valence") is not None and track.get("energy") is not None
            ]
            
            if not mood_tracks:
                logger.warning("No tracks with mood features available")
                return tracks[:playlist_size]
            
            # Calculate progression points
            playlist = []
            for i in range(playlist_size):
                progress = i / (playlist_size - 1) if playlist_size > 1 else 0
                
                # Interpolate target mood
                target_valence = start_target["valence"] + progress * (end_target["valence"] - start_target["valence"])
                target_energy = start_target["energy"] + progress * (end_target["energy"] - start_target["energy"])
                
                # Find closest track
                best_track = min(
                    mood_tracks,
                    key=lambda t: abs(t.get("valence", 0.5) - target_valence) + 
                                 abs(t.get("energy", 0.5) - target_energy)
                )
                
                playlist.append(best_track)
                # Remove to avoid duplicates
                if best_track in mood_tracks:
                    mood_tracks.remove(best_track)
                
                # If we run out of tracks, break
                if not mood_tracks:
                    break
            
            logger.info(
                "Mood journey playlist generated",
                start_mood=start_mood,
                end_mood=end_mood,
                result_size=len(playlist)
            )
            
            return playlist
    
    def generate_harmonic_mixing_playlist(
        self,
        tracks: List[Dict[str, Any]],
        playlist_size: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Generate playlist optimized for harmonic mixing (DJ-style)
        """
        with LogContext(
            operation="generate_harmonic_mixing_playlist",
            target_size=playlist_size
        ):
            # Filter tracks with key and BPM
            harmonic_tracks = [
                track for track in tracks
                if track.get("key") is not None and track.get("bpm") is not None
            ]
            
            if not harmonic_tracks:
                logger.warning("No tracks with harmonic features available")
                return tracks[:playlist_size]
            
            # Define key compatibility (simplified Camelot wheel)
            key_compatibility = {
                "C": ["C", "G", "F", "Am", "Em", "Dm"],
                "G": ["G", "D", "C", "Em", "Bm", "Am"],
                "D": ["D", "A", "G", "Bm", "F#m", "Em"],
                "A": ["A", "E", "D", "F#m", "C#m", "Bm"],
                "E": ["E", "B", "A", "C#m", "G#m", "F#m"],
                "B": ["B", "F#", "E", "G#m", "D#m", "C#m"],
                "F#": ["F#", "C#", "B", "D#m", "A#m", "G#m"],
                "C#": ["C#", "G#", "F#", "A#m", "Fm", "D#m"],
                "G#": ["G#", "D#", "C#", "Fm", "Cm", "A#m"],
                "D#": ["D#", "A#", "G#", "Cm", "Gm", "Fm"],
                "A#": ["A#", "F", "D#", "Gm", "Dm", "Cm"],
                "F": ["F", "C", "A#", "Dm", "Am", "Gm"],
            }
            
            # Start with a random track
            playlist = [harmonic_tracks[0]]
            remaining_tracks = harmonic_tracks[1:]
            
            while len(playlist) < playlist_size and remaining_tracks:
                current_track = playlist[-1]
                current_key = current_track.get("key", "C")
                current_bpm = current_track.get("bpm", 120)
                
                # Find compatible tracks
                compatible_tracks = []
                for track in remaining_tracks:
                    track_key = track.get("key", "C")
                    track_bpm = track.get("bpm", 120)
                    
                    # Check key compatibility
                    key_compatible = track_key in key_compatibility.get(current_key, [current_key])
                    
                    # Check BPM compatibility (within 10 BPM or double/half time)
                    bpm_diff = abs(track_bpm - current_bpm)
                    bpm_compatible = (
                        bpm_diff <= 10 or
                        abs(track_bpm - current_bpm * 2) <= 10 or
                        abs(track_bpm * 2 - current_bpm) <= 10
                    )
                    
                    if key_compatible and bpm_compatible:
                        compatible_tracks.append(track)
                
                # Choose best compatible track or any track if none compatible
                if compatible_tracks:
                    next_track = compatible_tracks[0]
                else:
                    next_track = remaining_tracks[0]
                
                playlist.append(next_track)
                remaining_tracks.remove(next_track)
            
            logger.info(
                "Harmonic mixing playlist generated",
                result_size=len(playlist),
                tracks_with_key=len([t for t in playlist if t.get("key")]),
                tracks_with_bpm=len([t for t in playlist if t.get("bpm")])
            )
            
            return playlist
    
    def _extract_similarity_features(self, tracks: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for similarity calculation"""
        features = []
        
        for track in tracks:
            feature_vector = [
                track.get("energy", 0.5),
                track.get("danceability", 0.5),
                track.get("valence", 0.5),
                track.get("acousticness", 0.5),
                track.get("instrumentalness", 0.5),
                track.get("bpm", 120) / 200,  # Normalize BPM
                track.get("loudness", -10) / -60,  # Normalize loudness
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_clustering_features(self, tracks: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for clustering"""
        features = []
        
        for track in tracks:
            feature_vector = [
                track.get("energy", 0.5),
                track.get("danceability", 0.5),
                track.get("valence", 0.5),
                track.get("acousticness", 0.5),
                track.get("instrumentalness", 0.5),
                track.get("speechiness", 0.1),
                track.get("liveness", 0.1),
                track.get("bpm", 120) / 200,
                track.get("loudness", -10) / -60,
                track.get("spectral_centroid", 1000) / 4000,  # Normalize
            ]
            features.append(feature_vector)
        
        return np.array(features)
