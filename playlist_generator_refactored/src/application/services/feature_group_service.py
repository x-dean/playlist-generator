"""
Feature group playlist generation service.
Ports the original FeatureGroupPlaylistGenerator to the new architecture.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from domain.entities.audio_file import AudioFile
from domain.entities.playlist import Playlist
from shared.exceptions import PlaylistGenerationError

class FeatureGroupService:
    """Service for feature group playlist generation using audio feature categorization."""
    
    def __init__(self, cache_file: str = None):
        """Initialize the feature group service.
        
        Args:
            cache_file: Path to the cache database file
        """
        self.cache_file = cache_file
        self.logger = logging.getLogger(__name__)
        
        # Musical keys for categorization
        self.keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Centroid mapping for mood similarity
        self.centroid_mapping = {
            'Warm': 0,
            'Mellow': 1,
            'Balanced': 2,
            'Bright': 3,
            'Crisp': 4
        }
        
        self.logger.debug("Feature group service initialized")
    
    def generate_feature_group_playlists(self, audio_files: List[AudioFile]) -> List[Playlist]:
        """Generate playlists using feature group categorization."""
        self.logger.info(f"Generating feature group playlists from {len(audio_files)} tracks")
        
        try:
            # Generate from features
            playlists = self._generate_from_features(audio_files)
            
            self.logger.info(f"Generated {len(playlists)} feature group playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"Feature group playlist generation failed: {e}")
            raise PlaylistGenerationError(f"Feature group generation failed: {e}")
    
    def _generate_from_features(self, audio_files: List[AudioFile]) -> List[Playlist]:
        """Generate playlists from audio files using feature categorization."""
        self.logger.debug(f"Generating playlists from {len(audio_files)} audio files")
        
        try:
            playlists = {}
            processed_tracks = 0
            skipped_tracks = 0

            for audio_file in audio_files:
                try:
                    # Extract features from audio file
                    features = audio_file.external_metadata.get('features', {})
                    if not features:
                        skipped_tracks += 1
                        self.logger.debug(f"Skipping track without features: {audio_file.file_path}")
                        continue

                    # Extract key features
                    bpm = features.get('bpm', 0)
                    centroid = features.get('centroid', 0)
                    danceability = features.get('danceability', 0)
                    key = features.get('key', None)
                    scale = features.get('scale', None)

                    # Skip tracks with failed BPM extraction (-1.0 marker)
                    if bpm == -1.0:
                        skipped_tracks += 1
                        self.logger.debug(f"Skipping track with failed BPM extraction: {audio_file.file_path}")
                        continue

                    # Skip invalid data
                    if None in (bpm, centroid, danceability):
                        skipped_tracks += 1
                        self.logger.debug(f"Skipping track with invalid data: {audio_file.file_path}")
                        continue

                    # Categorize features
                    bpm_group = self._get_bpm_group(bpm)
                    energy_group = self._get_energy_group(danceability)
                    mood_group = self._get_mood_group(centroid)
                    key_group = self._get_key_group(key, scale)

                    # Create playlist name
                    playlist_name = self._create_playlist_name(bpm_group, energy_group, key_group, mood_group)

                    # Add to playlist
                    if playlist_name not in playlists:
                        playlists[playlist_name] = {
                            'tracks': [],
                            'features': {
                                'type': 'feature_group',
                                'bpm_group': bpm_group,
                                'energy_group': energy_group,
                                'mood_group': mood_group,
                                'key_group': key_group
                            }
                        }
                        self.logger.debug(f"Created new playlist: {playlist_name}")

                    playlists[playlist_name]['tracks'].append(audio_file)
                    processed_tracks += 1

                except Exception as e:
                    skipped_tracks += 1
                    self.logger.warning(f"Error processing track {audio_file.file_path}: {str(e)}")

            # Convert to Playlist entities
            result = []
            for playlist_name, playlist_data in playlists.items():
                if len(playlist_data['tracks']) > 0:
                    track_ids = [track.id for track in playlist_data['tracks']]
                    track_paths = [str(track.file_path) for track in playlist_data['tracks']]
                    
                    playlist = Playlist(
                        name=playlist_name,
                        description=f"Feature group playlist: {playlist_data['features']['bpm_group']} {playlist_data['features']['energy_group']}",
                        track_ids=track_ids,
                        track_paths=track_paths,
                        generation_method="feature_group",
                        playlist_type="feature_based",
                        created_date=datetime.now()
                    )
                    result.append(playlist)
                    
                    self.logger.debug(f"Created playlist '{playlist_name}' with {len(playlist_data['tracks'])} tracks")

            self.logger.info(f"Feature group generation complete: {len(result)} playlists created from {processed_tracks} tracks (skipped {skipped_tracks})")
            return result

        except Exception as e:
            self.logger.error(f"Error generating from features: {str(e)}")
            return []
    
    def _get_energy_group(self, danceability: float) -> str:
        """Get energy group based on danceability."""
        if danceability < 0.3:
            return 'Chill'
        if danceability < 0.5:
            return 'Mellow'
        if danceability < 0.7:
            return 'Groovy'
        if danceability < 0.85:
            return 'Energetic'
        return 'Intense'
    
    def _get_bpm_group(self, bpm: float) -> str:
        """Get BPM group based on tempo."""
        if bpm < 70:
            return 'Slow'
        if bpm < 100:
            return 'Medium'
        if bpm < 130:
            return 'Upbeat'
        if bpm < 160:
            return 'Fast'
        return 'VeryFast'
    
    def _get_mood_group(self, centroid: float) -> str:
        """Get mood group based on spectral centroid."""
        if centroid < 500:
            return 'Warm'
        if centroid < 1500:
            return 'Mellow'
        if centroid < 3000:
            return 'Balanced'
        if centroid < 6000:
            return 'Bright'
        return 'Crisp'
    
    def _get_key_group(self, key: Any, scale: Any) -> str:
        """Get key group based on musical key and scale."""
        if key is not None and isinstance(key, (int, float)) and 0 <= int(key) <= 11:
            key_name = self.keys[int(key)]
            scale_name = 'Major' if scale == 1 else 'Minor'
            return f"{key_name}_{scale_name}"
        return ''
    
    def _create_playlist_name(self, bpm_group: str, energy_group: str, key_group: str, mood_group: str) -> str:
        """Create playlist name from feature groups."""
        
        # Create base playlist name
        if key_group:
            playlist_name = f"{bpm_group}_{energy_group}_{key_group}"
        else:
            playlist_name = f"{bpm_group}_{energy_group}"

        # Add mood for more separation
        if mood_group in ('Bright', 'Crisp'):
            playlist_name = f"{playlist_name}_Bright"
        elif mood_group in ('Warm', 'Mellow'):
            playlist_name = f"{playlist_name}_Warm"

        # Sanitize playlist name
        return self._sanitize_filename(playlist_name)
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string to be used as a filename."""
        self.logger.debug(f"Sanitizing filename: {name}")
        name = re.sub(r'[^\w\-_]', '_', name)
        sanitized = re.sub(r'_+', '_', name).strip('_')
        self.logger.debug(f"Sanitized filename: {name} -> {sanitized}")
        return sanitized 