"""
Mixed playlist generation service.
Ports the original PlaylistManager's mixed generation logic to the new architecture.
"""

import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

from domain.entities.audio_file import AudioFile
from domain.entities.playlist import Playlist
from shared.exceptions import PlaylistGenerationError

class MixedPlaylistService:
    """Service for mixed playlist generation combining multiple methods."""
    
    def __init__(self, cache_file: str = None):
        """Initialize the mixed playlist service.
        
        Args:
            cache_file: Path to the cache database file
        """
        self.cache_file = cache_file
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.min_size = 10
        self.max_size = 500
        self.min_tracks_per_playlist = 3
        
        self.logger.debug("Mixed playlist service initialized")
    
    def generate_mixed_playlists(self, audio_files: List[AudioFile], 
                                num_playlists: int = 8,
                                method: str = 'all') -> List[Playlist]:
        """Generate playlists using mixed methods combining multiple approaches."""
        self.logger.info(f"Generating mixed playlists using {method} method")
        
        try:
            # Generate playlists using different methods
            playlists = self._generate_by_method(audio_files, num_playlists, method)
            
            # Finalize and optimize playlists
            final_playlists = self._finalize_playlists(playlists, audio_files)
            
            self.logger.info(f"Generated {len(final_playlists)} mixed playlists")
            return final_playlists
            
        except Exception as e:
            self.logger.error(f"Mixed playlist generation failed: {e}")
            raise PlaylistGenerationError(f"Mixed generation failed: {e}")
    
    def _generate_by_method(self, audio_files: List[AudioFile], 
                           num_playlists: int, method: str) -> List[Playlist]:
        """Generate playlists using the specified method."""
        
        if method == 'all':
            return self._generate_all_methods(audio_files, num_playlists)
        elif method == 'ensemble':
            return self._generate_ensemble_playlists(audio_files, num_playlists)
        elif method == 'hierarchical':
            return self._generate_hierarchical_playlists(audio_files, num_playlists)
        elif method == 'recommendation':
            return self._generate_recommendation_playlists(audio_files, num_playlists)
        elif method == 'mood_based':
            return self._generate_mood_based_playlists(audio_files, num_playlists)
        elif method == 'tags':
            return self._generate_tag_based_playlists(audio_files, num_playlists)
        elif method == 'time':
            return self._generate_time_based_playlists(audio_files, num_playlists)
        elif method == 'kmeans':
            return self._generate_kmeans_playlists(audio_files, num_playlists)
        elif method == 'cache':
            return self._generate_cache_based_playlists(audio_files, num_playlists)
        else:
            self.logger.warning(f"Unknown method: {method}, using all methods")
            return self._generate_all_methods(audio_files, num_playlists)
    
    def _generate_all_methods(self, audio_files: List[AudioFile], 
                             num_playlists: int) -> List[Playlist]:
        """Generate playlists using all available methods."""
        
        all_playlists = []
        used_tracks = set()
        
        # Try different methods and collect results
        methods = [
            ('feature_group', self._generate_feature_group_playlists),
            ('cache_based', self._generate_cache_based_playlists),
            ('tag_based', self._generate_tag_based_playlists),
            ('advanced', self._generate_advanced_playlists),
            ('kmeans', self._generate_kmeans_playlists),
            ('time_based', self._generate_time_based_playlists)
        ]
        
        for method_name, method_func in methods:
            try:
                self.logger.debug(f"Trying {method_name} method")
                method_playlists = method_func(audio_files, num_playlists)
                
                # Add playlists that meet minimum size requirement
                for playlist in method_playlists:
                    if len(playlist.track_ids) >= self.min_tracks_per_playlist:
                        # Check for track overlap
                        new_tracks = set(playlist.track_ids) - used_tracks
                        if len(new_tracks) >= self.min_tracks_per_playlist:
                            # Create new playlist with only unused tracks
                            filtered_tracks = [track_id for track_id in playlist.track_ids 
                                             if track_id not in used_tracks]
                            if filtered_tracks:
                                new_playlist = Playlist(
                                    name=f"{playlist.name}_{method_name}",
                                    description=f"Mixed playlist from {method_name} method",
                                    track_ids=filtered_tracks,
                                    track_paths=[p for p in playlist.track_paths 
                                               if any(track_id in filtered_tracks 
                                                     for track_id in playlist.track_ids)],
                                    generation_method="mixed",
                                    playlist_type="mixed",
                                    created_date=datetime.now()
                                )
                                all_playlists.append(new_playlist)
                                used_tracks.update(filtered_tracks)
                                self.logger.debug(f"Added {method_name} playlist with {len(filtered_tracks)} tracks")
                
            except Exception as e:
                self.logger.warning(f"{method_name} method failed: {e}")
                continue
        
        return all_playlists
    
    def _generate_feature_group_playlists(self, audio_files: List[AudioFile], 
                                         num_playlists: int) -> List[Playlist]:
        """Generate feature group playlists."""
        try:
            from .feature_group_service import FeatureGroupService
            service = FeatureGroupService()
            return service.generate_feature_group_playlists(audio_files)
        except Exception as e:
            self.logger.warning(f"Feature group generation failed: {e}")
            return []
    
    def _generate_cache_based_playlists(self, audio_files: List[AudioFile], 
                                       num_playlists: int) -> List[Playlist]:
        """Generate cache-based playlists."""
        try:
            from .cache_based_service import CacheBasedService
            service = CacheBasedService()
            return service.generate_cache_based_playlists(audio_files)
        except Exception as e:
            self.logger.warning(f"Cache-based generation failed: {e}")
            return []
    
    def _generate_tag_based_playlists(self, audio_files: List[AudioFile], 
                                     num_playlists: int) -> List[Playlist]:
        """Generate tag-based playlists."""
        try:
            from .tag_based_service import TagBasedService
            service = TagBasedService()
            return service.generate_tag_based_playlists(audio_files)
        except Exception as e:
            self.logger.warning(f"Tag-based generation failed: {e}")
            return []
    
    def _generate_advanced_playlists(self, audio_files: List[AudioFile], 
                                    num_playlists: int) -> List[Playlist]:
        """Generate advanced playlists."""
        try:
            from .advanced_models_service import AdvancedModelsService
            service = AdvancedModelsService()
            return service.generate_advanced_playlists(audio_files, method='ensemble', num_playlists=num_playlists)
        except Exception as e:
            self.logger.warning(f"Advanced generation failed: {e}")
            return []
    
    def _generate_kmeans_playlists(self, audio_files: List[AudioFile], 
                                  num_playlists: int) -> List[Playlist]:
        """Generate K-means playlists."""
        try:
            # Create a simple K-means implementation for mixed generation
            if len(audio_files) < 3:
                return []
            
            # Simple clustering by BPM ranges
            bpm_groups = {}
            for audio_file in audio_files:
                features = audio_file.external_metadata.get('features', {})
                bpm = features.get('bpm', 120)
                
                if bpm < 80:
                    group = 'Slow'
                elif bpm < 120:
                    group = 'Medium'
                elif bpm < 160:
                    group = 'Fast'
                else:
                    group = 'VeryFast'
                
                if group not in bpm_groups:
                    bpm_groups[group] = []
                bpm_groups[group].append(audio_file)
            
            # Create playlists from groups
            playlists = []
            for group_name, tracks in bpm_groups.items():
                if len(tracks) >= self.min_tracks_per_playlist:
                    track_ids = [track.id for track in tracks]
                    track_paths = [str(track.file_path) for track in tracks]
                    
                    playlist = Playlist(
                        name=f"KMeans_{group_name}",
                        description=f"K-means cluster based on BPM: {group_name}",
                        track_ids=track_ids,
                        track_paths=track_paths,
                        generation_method="mixed",
                        playlist_type="mixed",
                        created_date=datetime.now()
                    )
                    playlists.append(playlist)
            
            return playlists
            
        except Exception as e:
            self.logger.warning(f"K-means generation failed: {e}")
            return []
    
    def _generate_time_based_playlists(self, audio_files: List[AudioFile], 
                                      num_playlists: int) -> List[Playlist]:
        """Generate time-based playlists."""
        try:
            # Simple time-based grouping by duration
            duration_groups = {}
            for audio_file in audio_files:
                duration = audio_file.duration_seconds or 180  # Default 3 minutes
                
                if duration < 120:
                    group = 'Short'
                elif duration < 240:
                    group = 'Medium'
                else:
                    group = 'Long'
                
                if group not in duration_groups:
                    duration_groups[group] = []
                duration_groups[group].append(audio_file)
            
            # Create playlists from groups
            playlists = []
            for group_name, tracks in duration_groups.items():
                if len(tracks) >= self.min_tracks_per_playlist:
                    track_ids = [track.id for track in tracks]
                    track_paths = [str(track.file_path) for track in tracks]
                    
                    playlist = Playlist(
                        name=f"Time_{group_name}",
                        description=f"Time-based group: {group_name} tracks",
                        track_ids=track_ids,
                        track_paths=track_paths,
                        generation_method="mixed",
                        playlist_type="mixed",
                        created_date=datetime.now()
                    )
                    playlists.append(playlist)
            
            return playlists
            
        except Exception as e:
            self.logger.warning(f"Time-based generation failed: {e}")
            return []
    
    def _generate_ensemble_playlists(self, audio_files: List[AudioFile], 
                                    num_playlists: int) -> List[Playlist]:
        """Generate ensemble playlists."""
        return self._generate_advanced_playlists(audio_files, num_playlists)
    
    def _generate_hierarchical_playlists(self, audio_files: List[AudioFile], 
                                        num_playlists: int) -> List[Playlist]:
        """Generate hierarchical playlists."""
        try:
            from .advanced_models_service import AdvancedModelsService
            service = AdvancedModelsService()
            return service.generate_advanced_playlists(audio_files, method='hierarchical', num_playlists=num_playlists)
        except Exception as e:
            self.logger.warning(f"Hierarchical generation failed: {e}")
            return []
    
    def _generate_recommendation_playlists(self, audio_files: List[AudioFile], 
                                          num_playlists: int) -> List[Playlist]:
        """Generate recommendation-based playlists."""
        try:
            from .advanced_models_service import AdvancedModelsService
            service = AdvancedModelsService()
            return service.generate_advanced_playlists(audio_files, method='recommendation', num_playlists=num_playlists)
        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            return []
    
    def _generate_mood_based_playlists(self, audio_files: List[AudioFile], 
                                      num_playlists: int) -> List[Playlist]:
        """Generate mood-based playlists."""
        try:
            from .advanced_models_service import AdvancedModelsService
            service = AdvancedModelsService()
            return service.generate_advanced_playlists(audio_files, method='mood_based', num_playlists=num_playlists)
        except Exception as e:
            self.logger.warning(f"Mood-based generation failed: {e}")
            return []
    
    def _finalize_playlists(self, playlists: List[Playlist], 
                           audio_files: List[AudioFile]) -> List[Playlist]:
        """Finalize playlists by merging small ones and handling leftovers."""
        
        self.logger.debug(f"Finalizing {len(playlists)} playlists")
        
        # Collect all track IDs
        all_track_ids = {audio_file.id for audio_file in audio_files}
        assigned_track_ids = set()
        for playlist in playlists:
            assigned_track_ids.update(playlist.track_ids)
        
        # Find unassigned tracks
        unassigned_track_ids = all_track_ids - assigned_track_ids
        unassigned_audio_files = [af for af in audio_files if af.id in unassigned_track_ids]
        
        self.logger.debug(f"Found {len(unassigned_audio_files)} unassigned tracks")
        
        # Separate small and large playlists
        small_playlists = [p for p in playlists if len(p.track_ids) < self.min_size]
        large_playlists = [p for p in playlists if len(p.track_ids) >= self.min_size]
        
        # Create mixed collection from small playlists and unassigned tracks
        mixed_track_ids = []
        mixed_track_paths = []
        
        # Add tracks from small playlists
        for playlist in small_playlists:
            mixed_track_ids.extend(playlist.track_ids)
            mixed_track_paths.extend(playlist.track_paths)
            self.logger.debug(f"Added {len(playlist.track_ids)} tracks from '{playlist.name}' to mixed playlist")
        
        # Add unassigned tracks
        if unassigned_audio_files:
            unassigned_track_ids = [af.id for af in unassigned_audio_files]
            unassigned_track_paths = [str(af.file_path) for af in unassigned_audio_files]
            mixed_track_ids.extend(unassigned_track_ids)
            mixed_track_paths.extend(unassigned_track_paths)
            self.logger.debug(f"Added {len(unassigned_audio_files)} unassigned tracks to mixed playlist")
        
        # Create mixed playlist if we have tracks
        if mixed_track_ids:
            mixed_playlist = Playlist(
                name="Mixed_Collection",
                description="A diverse collection of tracks from multiple methods",
                track_ids=mixed_track_ids,
                track_paths=mixed_track_paths,
                generation_method="mixed",
                playlist_type="mixed",
                created_date=datetime.now()
            )
            large_playlists.append(mixed_playlist)
            self.logger.info(f"Created mixed playlist with {len(mixed_track_ids)} tracks")
        
        # Split large playlists if needed
        final_playlists = []
        for playlist in large_playlists:
            if len(playlist.track_ids) > self.max_size:
                # Split large playlist
                chunks = [playlist.track_ids[i:i + self.max_size] 
                         for i in range(0, len(playlist.track_ids), self.max_size)]
                track_paths = playlist.track_paths
                
                for i, chunk in enumerate(chunks, 1):
                    chunk_paths = [track_paths[j] for j in range(len(chunk)) 
                                 if j < len(track_paths)]
                    
                    new_name = f"{playlist.name}_Part{i}" if len(chunks) > 1 else playlist.name
                    split_playlist = Playlist(
                        name=new_name,
                        description=playlist.description,
                        track_ids=chunk,
                        track_paths=chunk_paths,
                        generation_method=playlist.generation_method,
                        playlist_type=playlist.playlist_type,
                        created_date=datetime.now()
                    )
                    final_playlists.append(split_playlist)
                    self.logger.debug(f"Split large playlist '{playlist.name}' into '{new_name}' with {len(chunk)} tracks")
            else:
                final_playlists.append(playlist)
        
        self.logger.info(f"Finalized {len(final_playlists)} playlists after merging/splitting")
        return final_playlists 