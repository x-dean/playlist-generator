"""
Playlist Generator for Playlist Generator Simple.
Provides various playlist generation methods using analyzed audio features.
"""

import logging
import os
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import local modules
from .logging_setup import get_logger, log_function_call
from .database import DatabaseManager

logger = get_logger('playlista.playlist_generator')

# Constants
DEFAULT_PLAYLIST_SIZE = 20
DEFAULT_MIN_TRACKS_PER_GENRE = 10
DEFAULT_MAX_PLAYLISTS = 8
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_DIVERSITY_THRESHOLD = 0.3


class PlaylistGenerationMethod:
    """Enumeration of playlist generation methods."""
    KMEANS = "kmeans"
    SIMILARITY = "similarity"
    RANDOM = "random"
    TIME_BASED = "time_based"
    TAG_BASED = "tag_based"
    CACHE_BASED = "cache_based"
    FEATURE_GROUP = "feature_group"
    MIXED = "mixed"


class Playlist:
    """Represents a generated playlist."""
    
    def __init__(self, name: str, tracks: List[str] = None, features: Dict[str, Any] = None):
        self.name = name
        self.tracks = tracks or []
        self.features = features or {}
        self.created_at = datetime.now()
        self.size = len(self.tracks)
    
    def add_track(self, track: str):
        """Add a track to the playlist."""
        if track not in self.tracks:
            self.tracks.append(track)
            self.size = len(self.tracks)
    
    def remove_track(self, track: str):
        """Remove a track from the playlist."""
        if track in self.tracks:
            self.tracks.remove(track)
            self.size = len(self.tracks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert playlist to dictionary."""
        return {
            'name': self.name,
            'tracks': self.tracks,
            'features': self.features,
            'size': self.size,
            'created_at': self.created_at.isoformat()
        }


class PlaylistGenerator:
    """
    Main playlist generator that coordinates different generation methods.
    
    Supports multiple playlist generation strategies:
    - K-means clustering
    - Similarity-based selection
    - Random selection
    - Time-based scheduling
    - Tag-based selection
    - Cache-based selection
    - Feature group selection
    - Mixed approach
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the playlist generator.
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_playlist_config()
        
        self.config = config
        
        # Playlist settings
        self.default_playlist_size = config.get('DEFAULT_PLAYLIST_SIZE', DEFAULT_PLAYLIST_SIZE)
        self.min_tracks_per_genre = config.get('MIN_TRACKS_PER_GENRE', DEFAULT_MIN_TRACKS_PER_GENRE)
        self.max_playlists = config.get('MAX_PLAYLISTS', DEFAULT_MAX_PLAYLISTS)
        self.similarity_threshold = config.get('SIMILARITY_THRESHOLD', DEFAULT_SIMILARITY_THRESHOLD)
        self.diversity_threshold = config.get('DIVERSITY_THRESHOLD', DEFAULT_DIVERSITY_THRESHOLD)
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        # Initialize generation methods
        self._init_generation_methods()
        
        logger.info(f"Initializing PlaylistGenerator")
        logger.debug(f"Playlist configuration: {config}")
        logger.info(f"PlaylistGenerator initialized successfully")
    
    def _init_generation_methods(self):
        """Initialize all playlist generation methods."""
        try:
            # Initialize K-means generator
            self.kmeans_generator = KMeansPlaylistGenerator(self.db_manager)
            
            # Initialize other generators
            self.similarity_generator = SimilarityPlaylistGenerator(self.db_manager)
            self.random_generator = RandomPlaylistGenerator(self.db_manager)
            self.time_generator = TimeBasedPlaylistGenerator(self.db_manager)
            self.tag_generator = TagBasedPlaylistGenerator(self.db_manager)
            self.cache_generator = CacheBasedPlaylistGenerator(self.db_manager)
            self.feature_group_generator = FeatureGroupPlaylistGenerator(self.db_manager)
            
            logger.debug("All playlist generation methods initialized")
            
        except Exception as e:
            logger.error(f"Error initializing playlist generators: {e}")
            raise
    
    @log_function_call
    def generate_playlists(self, method: str = "all", num_playlists: int = None, 
                          playlist_size: int = None, **kwargs) -> Dict[str, Playlist]:
        """
        Generate playlists using the specified method.
        
        Args:
            method: Playlist generation method
            num_playlists: Number of playlists to generate
            playlist_size: Size of each playlist
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dictionary of playlist names to Playlist objects
        """
        logger.info(f"Starting playlist generation with method: {method}")
        
        try:
            # Get analyzed tracks from database
            tracks = self.db_manager.get_analyzed_tracks()
            
            if not tracks:
                logger.warning("No analyzed tracks found in database")
                return {}
            
            logger.info(f"Found {len(tracks)} analyzed tracks")
            
            # Set defaults
            num_playlists = num_playlists or self.max_playlists
            playlist_size = playlist_size or self.default_playlist_size
            
            # Generate playlists based on method
            if method == PlaylistGenerationMethod.KMEANS:
                playlists = self._generate_kmeans_playlists(tracks, num_playlists, playlist_size)
            elif method == PlaylistGenerationMethod.SIMILARITY:
                playlists = self._generate_similarity_playlists(tracks, num_playlists, playlist_size)
            elif method == PlaylistGenerationMethod.RANDOM:
                playlists = self._generate_random_playlists(tracks, num_playlists, playlist_size)
            elif method == PlaylistGenerationMethod.TIME_BASED:
                playlists = self._generate_time_based_playlists(tracks, num_playlists, playlist_size)
            elif method == PlaylistGenerationMethod.TAG_BASED:
                playlists = self._generate_tag_based_playlists(tracks, num_playlists, playlist_size)
            elif method == PlaylistGenerationMethod.CACHE_BASED:
                playlists = self._generate_cache_based_playlists(tracks, num_playlists, playlist_size)
            elif method == PlaylistGenerationMethod.FEATURE_GROUP:
                playlists = self._generate_feature_group_playlists(tracks, num_playlists, playlist_size)
            elif method == PlaylistGenerationMethod.MIXED:
                playlists = self._generate_mixed_playlists(tracks, num_playlists, playlist_size)
            elif method == "all":
                playlists = self._generate_all_playlists(tracks, num_playlists, playlist_size)
            else:
                raise ValueError(f"Unknown playlist generation method: {method}")
            
            # Optimize and finalize playlists
            final_playlists = self._optimize_playlists(playlists, tracks)
            
            logger.info(f"Generated {len(final_playlists)} playlists")
            return final_playlists
            
        except Exception as e:
            logger.error(f"Playlist generation failed: {e}")
            return {}
    
    def _generate_kmeans_playlists(self, tracks: List[Dict], num_playlists: int, 
                                  playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using K-means clustering."""
        logger.info(f"Generating {num_playlists} K-means playlists")
        return self.kmeans_generator.generate(tracks, num_playlists, playlist_size)
    
    def _generate_similarity_playlists(self, tracks: List[Dict], num_playlists: int, 
                                     playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using similarity-based selection."""
        logger.info(f"Generating {num_playlists} similarity-based playlists")
        return self.similarity_generator.generate(tracks, num_playlists, playlist_size)
    
    def _generate_random_playlists(self, tracks: List[Dict], num_playlists: int, 
                                 playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using random selection."""
        logger.info(f"Generating {num_playlists} random playlists")
        return self.random_generator.generate(tracks, num_playlists, playlist_size)
    
    def _generate_time_based_playlists(self, tracks: List[Dict], num_playlists: int, 
                                     playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using time-based scheduling."""
        logger.info(f"Generating {num_playlists} time-based playlists")
        return self.time_generator.generate(tracks, num_playlists, playlist_size)
    
    def _generate_tag_based_playlists(self, tracks: List[Dict], num_playlists: int, 
                                    playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using tag-based selection."""
        logger.info(f"Generating {num_playlists} tag-based playlists")
        return self.tag_generator.generate(tracks, num_playlists, playlist_size)
    
    def _generate_cache_based_playlists(self, tracks: List[Dict], num_playlists: int, 
                                      playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using cache-based selection."""
        logger.info(f"Generating {num_playlists} cache-based playlists")
        return self.cache_generator.generate(tracks, num_playlists, playlist_size)
    
    def _generate_feature_group_playlists(self, tracks: List[Dict], num_playlists: int, 
                                        playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using feature group selection."""
        logger.info(f"Generating {num_playlists} feature group playlists")
        return self.feature_group_generator.generate(tracks, num_playlists, playlist_size)
    
    def _generate_mixed_playlists(self, tracks: List[Dict], num_playlists: int, 
                                playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using mixed approach."""
        logger.info(f"Generating {num_playlists} mixed playlists")
        
        # Combine multiple methods
        playlists = {}
        
        # Add K-means playlists
        kmeans_playlists = self._generate_kmeans_playlists(tracks, num_playlists // 2, playlist_size)
        playlists.update(kmeans_playlists)
        
        # Add similarity playlists
        similarity_playlists = self._generate_similarity_playlists(tracks, num_playlists // 2, playlist_size)
        playlists.update(similarity_playlists)
        
        return playlists
    
    def _generate_all_playlists(self, tracks: List[Dict], num_playlists: int, 
                               playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using all available methods."""
        logger.info(f"Generating playlists using all methods")
        
        all_playlists = {}
        
        # Generate playlists with each method
        methods = [
            PlaylistGenerationMethod.KMEANS,
            PlaylistGenerationMethod.SIMILARITY,
            PlaylistGenerationMethod.RANDOM,
            PlaylistGenerationMethod.TIME_BASED,
            PlaylistGenerationMethod.TAG_BASED,
            PlaylistGenerationMethod.CACHE_BASED,
            PlaylistGenerationMethod.FEATURE_GROUP
        ]
        
        playlists_per_method = max(1, num_playlists // len(methods))
        
        for method in methods:
            try:
                method_playlists = self.generate_playlists(method, playlists_per_method, playlist_size)
                all_playlists.update(method_playlists)
            except Exception as e:
                logger.warning(f"️ Failed to generate playlists with method {method}: {e}")
        
        return all_playlists
    
    def _optimize_playlists(self, playlists: Dict[str, Playlist], 
                           tracks: List[Dict]) -> Dict[str, Playlist]:
        """
        Optimize generated playlists by balancing sizes and removing duplicates.
        
        Args:
            playlists: Dictionary of generated playlists
            tracks: List of available tracks
            
        Returns:
            Optimized playlists dictionary
        """
        logger.debug("Optimizing playlists")
        
        if not playlists:
            return playlists
        
        # Balance playlist sizes
        self._balance_playlist_sizes(playlists, tracks)
        
        # Remove duplicate tracks within playlists
        self._remove_duplicate_tracks(playlists)
        
        # Ensure minimum playlist sizes
        self._ensure_minimum_sizes(playlists, tracks)
        
        logger.debug(f"Optimized {len(playlists)} playlists")
        return playlists
    
    def _balance_playlist_sizes(self, playlists: Dict[str, Playlist], tracks: List[Dict]):
        """Balance playlist sizes by redistributing tracks."""
        if len(playlists) <= 1:
            return
        
        # Calculate target size
        total_tracks = len(tracks)
        target_size = total_tracks // len(playlists)
        
        # Find playlists that are too small or too large
        small_playlists = []
        large_playlists = []
        
        for name, playlist in playlists.items():
            if playlist.size < target_size * 0.8:
                small_playlists.append(name)
            elif playlist.size > target_size * 1.2:
                large_playlists.append(name)
        
        # Redistribute tracks
        for large_name in large_playlists:
            large_playlist = playlists[large_name]
            excess_tracks = large_playlist.size - target_size
            
            for small_name in small_playlists:
                if excess_tracks <= 0:
                    break
                
                small_playlist = playlists[small_name]
                needed_tracks = target_size - small_playlist.size
                tracks_to_move = min(excess_tracks, needed_tracks)
                
                # Move tracks from large to small playlist
                moved_tracks = large_playlist.tracks[-tracks_to_move:]
                large_playlist.tracks = large_playlist.tracks[:-tracks_to_move]
                small_playlist.tracks.extend(moved_tracks)
                
                # Update sizes
                large_playlist.size = len(large_playlist.tracks)
                small_playlist.size = len(small_playlist.tracks)
                
                excess_tracks -= tracks_to_move
    
    def _remove_duplicate_tracks(self, playlists: Dict[str, Playlist]):
        """Remove duplicate tracks within playlists."""
        for playlist in playlists.values():
            unique_tracks = []
            seen = set()
            
            for track in playlist.tracks:
                if track not in seen:
                    unique_tracks.append(track)
                    seen.add(track)
            
            playlist.tracks = unique_tracks
            playlist.size = len(playlist.tracks)
    
    def _ensure_minimum_sizes(self, playlists: Dict[str, Playlist], tracks: List[Dict]):
        """Ensure playlists meet minimum size requirements."""
        min_size = 5  # Minimum tracks per playlist
        
        for playlist in playlists.values():
            if playlist.size < min_size:
                # Add random tracks to reach minimum size
                available_tracks = [t['filepath'] for t in tracks if t['filepath'] not in playlist.tracks]
                needed_tracks = min_size - playlist.size
                
                if available_tracks:
                    additional_tracks = random.sample(available_tracks, min(needed_tracks, len(available_tracks)))
                    playlist.tracks.extend(additional_tracks)
                    playlist.size = len(playlist.tracks)
    
    def save_playlists(self, playlists: Dict[str, Playlist], output_dir: str = None) -> bool:
        """
        Save generated playlists to files.
        
        Args:
            playlists: Dictionary of playlists to save
            output_dir: Output directory for playlist files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if output_dir is None:
                output_dir = self.config.get('PLAYLIST_OUTPUT_DIR', 'playlists')
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each playlist
            for name, playlist in playlists.items():
                # Create playlist file
                playlist_file = os.path.join(output_dir, f"{name}.m3u")
                
                with open(playlist_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Playlist: {name}\n")
                    f.write(f"# Generated: {playlist.created_at}\n")
                    f.write(f"# Tracks: {playlist.size}\n\n")
                    
                    for track in playlist.tracks:
                        f.write(f"{track}\n")
                
                logger.debug(f"Saved playlist '{name}' to {playlist_file}")
            
            # Save playlist metadata
            metadata_file = os.path.join(output_dir, "playlists_metadata.json")
            metadata = {
                'generated_at': datetime.now().isoformat(),
                'total_playlists': len(playlists),
                'playlists': {name: playlist.to_dict() for name, playlist in playlists.items()}
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(playlists)} playlists to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving playlists: {e}")
            return False
    
    def get_playlist_statistics(self, playlists: Dict[str, Playlist]) -> Dict[str, Any]:
        """
        Get statistics about generated playlists.
        
        Args:
            playlists: Dictionary of playlists to analyze
            
        Returns:
            Dictionary with playlist statistics
        """
        if not playlists:
            return {}
        
        stats = {
            'total_playlists': len(playlists),
            'total_tracks': sum(playlist.size for playlist in playlists.values()),
            'average_playlist_size': sum(playlist.size for playlist in playlists.values()) / len(playlists),
            'min_playlist_size': min(playlist.size for playlist in playlists.values()),
            'max_playlist_size': max(playlist.size for playlist in playlists.values()),
            'playlist_sizes': {name: playlist.size for name, playlist in playlists.items()},
            'generated_at': datetime.now().isoformat()
        }
        
        return stats


# Base class for playlist generators
class BasePlaylistGenerator:
    """Base class for playlist generators."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = get_logger(f'playlista.{self.__class__.__name__.lower()}')
    
    def generate(self, tracks: List[Dict], num_playlists: int, playlist_size: int) -> Dict[str, Playlist]:
        """
        Generate playlists.
        
        Args:
            tracks: List of track dictionaries
            num_playlists: Number of playlists to generate
            playlist_size: Size of each playlist
            
        Returns:
            Dictionary of playlist names to Playlist objects
        """
        raise NotImplementedError("Subclasses must implement generate method")


# K-means playlist generator
class KMeansPlaylistGenerator(BasePlaylistGenerator):
    """Generate playlists using K-means clustering."""
    
    def generate(self, tracks: List[Dict], num_playlists: int, playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using K-means clustering."""
        self.logger.info(f"Generating {num_playlists} K-means playlists")
        
        try:
            # Extract features for clustering
            features = self._extract_features(tracks)
            
            if len(features) < num_playlists:
                self.logger.warning(f"️ Not enough tracks for {num_playlists} playlists")
                num_playlists = max(1, len(features) // 2)
            
            # Perform K-means clustering
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare feature matrix
            feature_matrix = np.array([list(f.values()) for f in features])
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=num_playlists, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Create playlists from clusters
            playlists = {}
            for cluster_id in range(num_playlists):
                cluster_tracks = [tracks[i] for i in range(len(tracks)) if cluster_labels[i] == cluster_id]
                
                if cluster_tracks:
                    playlist_name = f"KMeans_Cluster_{cluster_id}"
                    playlist = Playlist(playlist_name)
                    
                    # Add tracks to playlist
                    for track in cluster_tracks[:playlist_size]:
                        playlist.add_track(track['filepath'])
                    
                    playlists[playlist_name] = playlist
            
            self.logger.info(f"Generated {len(playlists)} K-means playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"K-means playlist generation failed: {e}")
            return {}
    
    def _extract_features(self, tracks: List[Dict]) -> List[Dict]:
        """Extract numerical features from tracks for clustering."""
        features = []
        
        for track in tracks:
            feature_dict = {}
            
            # Extract numerical features
            if 'bpm' in track:
                feature_dict['bpm'] = float(track['bpm'])
            if 'centroid' in track:
                feature_dict['centroid'] = float(track['centroid'])
            if 'danceability' in track:
                feature_dict['danceability'] = float(track['danceability'])
            if 'loudness' in track:
                feature_dict['loudness'] = float(track['loudness'])
            if 'key' in track:
                feature_dict['key'] = int(track['key'])
            
            # Use default values for missing features
            feature_dict.setdefault('bpm', 120.0)
            feature_dict.setdefault('centroid', 5000.0)
            feature_dict.setdefault('danceability', 0.5)
            feature_dict.setdefault('loudness', -20.0)
            feature_dict.setdefault('key', 0)
            
            features.append(feature_dict)
        
        return features


# Similarity-based playlist generator
class SimilarityPlaylistGenerator(BasePlaylistGenerator):
    """Generate playlists using similarity-based selection."""
    
    def generate(self, tracks: List[Dict], num_playlists: int, playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using similarity-based selection."""
        self.logger.info(f"Generating {num_playlists} similarity-based playlists")
        
        try:
            playlists = {}
            
            for i in range(num_playlists):
                playlist_name = f"Similarity_Playlist_{i}"
                playlist = Playlist(playlist_name)
                
                # Start with a random track
                if tracks:
                    start_track = random.choice(tracks)
                    playlist.add_track(start_track['filepath'])
                    
                    # Add similar tracks
                    remaining_tracks = [t for t in tracks if t['filepath'] != start_track['filepath']]
                    
                    while len(playlist.tracks) < playlist_size and remaining_tracks:
                        # Find most similar track
                        similar_track = self._find_most_similar(start_track, remaining_tracks)
                        if similar_track:
                            playlist.add_track(similar_track['filepath'])
                            remaining_tracks.remove(similar_track)
                            start_track = similar_track
                        else:
                            break
                
                playlists[playlist_name] = playlist
            
            self.logger.info(f"Generated {len(playlists)} similarity-based playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"Similarity-based playlist generation failed: {e}")
            return {}
    
    def _find_most_similar(self, reference_track: Dict, candidate_tracks: List[Dict]) -> Optional[Dict]:
        """Find the most similar track to the reference track."""
        if not candidate_tracks:
            return None
        
        best_similarity = -1
        most_similar = None
        
        for track in candidate_tracks:
            similarity = self._calculate_similarity(reference_track, track)
            if similarity > best_similarity:
                best_similarity = similarity
                most_similar = track
        
        return most_similar if best_similarity > 0.3 else None
    
    def _calculate_similarity(self, track1: Dict, track2: Dict) -> float:
        """Calculate similarity between two tracks."""
        similarity = 0.0
        factors = 0
        
        # Compare BPM
        if 'bpm' in track1 and 'bpm' in track2:
            bpm_diff = abs(float(track1['bpm']) - float(track2['bpm']))
            bpm_similarity = max(0, 1 - (bpm_diff / 50))  # 50 BPM difference = 0 similarity
            similarity += bpm_similarity
            factors += 1
        
        # Compare key
        if 'key' in track1 and 'key' in track2:
            key_similarity = 1.0 if track1['key'] == track2['key'] else 0.0
            similarity += key_similarity
            factors += 1
        
        # Compare danceability
        if 'danceability' in track1 and 'danceability' in track2:
            dance_diff = abs(float(track1['danceability']) - float(track2['danceability']))
            dance_similarity = max(0, 1 - dance_diff)
            similarity += dance_similarity
            factors += 1
        
        return similarity / factors if factors > 0 else 0.0


# Random playlist generator
class RandomPlaylistGenerator(BasePlaylistGenerator):
    """Generate playlists using random selection."""
    
    def generate(self, tracks: List[Dict], num_playlists: int, playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using random selection."""
        self.logger.info(f"Generating {num_playlists} random playlists")
        
        try:
            playlists = {}
            
            for i in range(num_playlists):
                playlist_name = f"Random_Playlist_{i}"
                playlist = Playlist(playlist_name)
                
                # Randomly select tracks
                if tracks:
                    selected_tracks = random.sample(tracks, min(playlist_size, len(tracks)))
                    for track in selected_tracks:
                        playlist.add_track(track['filepath'])
                
                playlists[playlist_name] = playlist
            
            self.logger.info(f"Generated {len(playlists)} random playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"Random playlist generation failed: {e}")
            return {}


# Time-based playlist generator
class TimeBasedPlaylistGenerator(BasePlaylistGenerator):
    """Generate playlists using time-based scheduling."""
    
    def generate(self, tracks: List[Dict], num_playlists: int, playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using time-based scheduling."""
        self.logger.info(f"Generating {num_playlists} time-based playlists")
        
        try:
            playlists = {}
            
            # Create playlists for different times of day
            time_slots = [
                ("Morning", 6, 12),
                ("Afternoon", 12, 18),
                ("Evening", 18, 24),
                ("Night", 0, 6)
            ]
            
            for i, (time_name, start_hour, end_hour) in enumerate(time_slots[:num_playlists]):
                playlist_name = f"Time_{time_name}"
                playlist = Playlist(playlist_name)
                
                # Select tracks based on time characteristics
                time_tracks = self._select_tracks_for_time(tracks, start_hour, end_hour)
                
                for track in time_tracks[:playlist_size]:
                    playlist.add_track(track['filepath'])
                
                playlists[playlist_name] = playlist
            
            self.logger.info(f"Generated {len(playlists)} time-based playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"Time-based playlist generation failed: {e}")
            return {}
    
    def _select_tracks_for_time(self, tracks: List[Dict], start_hour: int, end_hour: int) -> List[Dict]:
        """Select tracks appropriate for the given time period."""
        # Simple time-based selection based on BPM and energy
        time_tracks = []
        
        for track in tracks:
            if 'bpm' in track:
                bpm = float(track['bpm'])
                
                # Morning: moderate BPM (80-120)
                if start_hour == 6 and 80 <= bpm <= 120:
                    time_tracks.append(track)
                # Afternoon: higher BPM (100-140)
                elif start_hour == 12 and 100 <= bpm <= 140:
                    time_tracks.append(track)
                # Evening: moderate to high BPM (90-130)
                elif start_hour == 18 and 90 <= bpm <= 130:
                    time_tracks.append(track)
                # Night: lower BPM (60-100)
                elif start_hour == 0 and 60 <= bpm <= 100:
                    time_tracks.append(track)
        
        # If no specific tracks found, use all tracks
        if not time_tracks:
            time_tracks = tracks
        
        return time_tracks


# Tag-based playlist generator
class TagBasedPlaylistGenerator(BasePlaylistGenerator):
    """Generate playlists using tag-based selection."""
    
    def generate(self, tracks: List[Dict], num_playlists: int, playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using tag-based selection."""
        self.logger.info(f"Generating {num_playlists} tag-based playlists")
        
        try:
            playlists = {}
            
            # Group tracks by tags/genres
            tag_groups = self._group_tracks_by_tags(tracks)
            
            # Create playlists for each tag group
            for i, (tag, tag_tracks) in enumerate(tag_groups.items()):
                if i >= num_playlists:
                    break
                
                playlist_name = f"Tag_{tag}"
                playlist = Playlist(playlist_name)
                
                for track in tag_tracks[:playlist_size]:
                    playlist.add_track(track['filepath'])
                
                playlists[playlist_name] = playlist
            
            self.logger.info(f"Generated {len(playlists)} tag-based playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"Tag-based playlist generation failed: {e}")
            return {}
    
    def _group_tracks_by_tags(self, tracks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group tracks by their tags or genres."""
        tag_groups = defaultdict(list)
        
        for track in tracks:
            # Extract tags from track metadata
            tags = self._extract_track_tags(track)
            
            if tags:
                # Use the first tag as the primary tag
                primary_tag = tags[0]
                tag_groups[primary_tag].append(track)
            else:
                # If no tags, use "Unknown"category
                tag_groups["Unknown"].append(track)
        
        return dict(tag_groups)
    
    def _extract_track_tags(self, track: Dict) -> List[str]:
        """Extract tags from track metadata."""
        tags = []
        
        # Check for genre information
        if 'genre' in track and track['genre']:
            tags.append(track['genre'])
        
        # Check for artist information
        if 'artist' in track and track['artist']:
            tags.append(track['artist'])
        
        # Check for album information
        if 'album' in track and track['album']:
            tags.append(track['album'])
        
        return tags


# Cache-based playlist generator
class CacheBasedPlaylistGenerator(BasePlaylistGenerator):
    """Generate playlists using cache-based selection."""
    
    def generate(self, tracks: List[Dict], num_playlists: int, playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using cache-based selection."""
        self.logger.info(f"Generating {num_playlists} cache-based playlists")
        
        try:
            playlists = {}
            
            # Get cached playlists from database
            cached_playlists = self.db_manager.get_cached_playlists()
            
            if cached_playlists:
                # Use cached playlists as templates
                for i, cached_playlist in enumerate(cached_playlists[:num_playlists]):
                    playlist_name = f"Cache_Playlist_{i}"
                    playlist = Playlist(playlist_name)
                    
                    # Select tracks based on cached playlist characteristics
                    selected_tracks = self._select_tracks_for_cached_playlist(tracks, cached_playlist)
                    
                    for track in selected_tracks[:playlist_size]:
                        playlist.add_track(track['filepath'])
                    
                    playlists[playlist_name] = playlist
            else:
                # Fallback to random selection if no cached playlists
                random_generator = RandomPlaylistGenerator(self.db_manager)
                playlists = random_generator.generate(tracks, num_playlists, playlist_size)
            
            self.logger.info(f"Generated {len(playlists)} cache-based playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"Cache-based playlist generation failed: {e}")
            return {}
    
    def _select_tracks_for_cached_playlist(self, tracks: List[Dict], cached_playlist: Dict) -> List[Dict]:
        """Select tracks based on cached playlist characteristics."""
        # Simple implementation: select tracks with similar BPM range
        if not cached_playlist or 'avg_bpm' not in cached_playlist:
            return tracks
        
        target_bpm = cached_playlist['avg_bpm']
        bpm_tolerance = 20
        
        matching_tracks = []
        for track in tracks:
            if 'bpm' in track:
                bpm = float(track['bpm'])
                if abs(bpm - target_bpm) <= bpm_tolerance:
                    matching_tracks.append(track)
        
        return matching_tracks if matching_tracks else tracks


# Feature group playlist generator
class FeatureGroupPlaylistGenerator(BasePlaylistGenerator):
    """Generate playlists using feature group selection."""
    
    def generate(self, tracks: List[Dict], num_playlists: int, playlist_size: int) -> Dict[str, Playlist]:
        """Generate playlists using feature group selection."""
        self.logger.info(f"Generating {num_playlists} feature group playlists")
        
        try:
            playlists = {}
            
            # Group tracks by feature characteristics
            feature_groups = self._group_tracks_by_features(tracks)
            
            # Create playlists for each feature group
            for i, (group_name, group_tracks) in enumerate(feature_groups.items()):
                if i >= num_playlists:
                    break
                
                playlist_name = f"Feature_{group_name}"
                playlist = Playlist(playlist_name)
                
                for track in group_tracks[:playlist_size]:
                    playlist.add_track(track['filepath'])
                
                playlists[playlist_name] = playlist
            
            self.logger.info(f"Generated {len(playlists)} feature group playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"Feature group playlist generation failed: {e}")
            return {}
    
    def _group_tracks_by_features(self, tracks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group tracks by their audio features."""
        feature_groups = defaultdict(list)
        
        for track in tracks:
            group_name = self._determine_feature_group(track)
            feature_groups[group_name].append(track)
        
        return dict(feature_groups)
    
    def _determine_feature_group(self, track: Dict) -> str:
        """Determine the feature group for a track."""
        # Analyze BPM
        if 'bpm' in track:
            bpm = float(track['bpm'])
            if bpm < 80:
                return "Slow_Tempo"
            elif bpm < 120:
                return "Medium_Tempo"
            else:
                return "Fast_Tempo"
        
        # Analyze danceability
        if 'danceability' in track:
            danceability = float(track['danceability'])
            if danceability < 0.3:
                return "Low_Energy"
            elif danceability < 0.7:
                return "Medium_Energy"
            else:
                return "High_Energy"
        
        # Default group
        return "Unknown_Features"


# Global playlist generator instance
playlist_generator = PlaylistGenerator() 