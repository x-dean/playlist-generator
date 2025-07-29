"""
Cache-based playlist generation service.
Ports the original CacheBasedGenerator to the new architecture.
"""

import logging
import sqlite3
import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import re

from domain.entities.audio_file import AudioFile
from domain.entities.playlist import Playlist
from shared.exceptions import PlaylistGenerationError

class CacheBasedService:
    """Service for cache-based playlist generation using rule-based feature categorization."""
    
    def __init__(self, cache_file: str = None):
        """Initialize the cache-based playlist generator.
        
        Args:
            cache_file: Path to the cache database file
        """
        self.cache_file = cache_file
        self.logger = logging.getLogger(__name__)
        
        # BPM ranges with descriptions
        self.bpm_ranges = {
            'Very_Slow': (0, 60, "Ambient and atmospheric tracks"),
            'Slow': (60, 90, "Relaxing and calming music"),
            'Medium': (90, 120, "Moderate tempo for casual listening"),
            'Upbeat': (120, 150, "Energetic and lively tracks"),
            'Fast': (150, 180, "High-energy dance music"),
            'Very_Fast': (180, float('inf'), "Intense and dynamic tracks")
        }

        # Energy levels with descriptions
        self.energy_levels = {
            'Ambient': (0, 0.3, "Perfect for meditation and deep focus"),
            'Chill': (0.3, 0.45, "Relaxed and laid-back vibes"),
            'Balanced': (0.45, 0.6, "Moderate energy for everyday listening"),
            'Groovy': (0.6, 0.75, "Rhythmic and engaging beats"),
            'Energetic': (0.75, 0.85, "High energy for workouts"),
            'Intense': (0.85, 1.0, "Maximum energy for peak moments")
        }

        # Mood categories based on spectral features
        self.mood_ranges = {
            'Dark': (0, 1000, "Deep and atmospheric"),
            'Warm': (1000, 2000, "Rich and full-bodied sound"),
            'Balanced': (2000, 4000, "Clear and well-defined"),
            'Bright': (4000, 6000, "Crisp and detailed"),
            'Brilliant': (6000, float('inf'), "Sparkling and airy")
        }
        
        self.logger.debug("Cache-based generator initialized with feature ranges")
    
    def generate_cache_based_playlists(self, audio_files: List[AudioFile]) -> List[Playlist]:
        """Generate playlists using cache-based method."""
        self.logger.info(f"Generating cache-based playlists from {len(audio_files)} tracks")
        
        try:
            # Generate from features
            playlists = self._generate_from_features(audio_files)
            
            # Merge small playlists if needed
            if len(playlists) > 1:
                playlists = self._merge_playlists(playlists)
            
            self.logger.info(f"Generated {len(playlists)} cache-based playlists")
            return playlists
            
        except Exception as e:
            self.logger.error(f"Cache-based playlist generation failed: {e}")
            raise PlaylistGenerationError(f"Cache-based generation failed: {e}")
    
    def _generate_from_features(self, audio_files: List[AudioFile]) -> List[Playlist]:
        """Generate playlists from audio files using feature categorization."""
        self.logger.debug(f"Generating playlists from {len(audio_files)} audio files")
        
        try:
            playlists = defaultdict(list)
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

                    # Generate playlist name and description
                    playlist_name = self._generate_playlist_name(features)
                    description = self._generate_description(features)

                    # Add to playlist group
                    playlists[playlist_name].append({
                        'audio_file': audio_file,
                        'description': description,
                        'features': features
                    })
                    processed_tracks += 1
                    self.logger.debug(f"Added track to playlist '{playlist_name}': {audio_file.file_path}")

                except Exception as e:
                    skipped_tracks += 1
                    self.logger.warning(f"Error processing track {audio_file.file_path}: {str(e)}")

            # Convert to Playlist entities
            result = []
            for playlist_name, tracks in playlists.items():
                if len(tracks) > 0:
                    track_ids = [t['audio_file'].id for t in tracks]
                    track_paths = [str(t['audio_file'].file_path) for t in tracks]
                    
                    playlist = Playlist(
                        name=playlist_name,
                        description=tracks[0]['description'],
                        track_ids=track_ids,
                        track_paths=track_paths,
                        generation_method="cache_based",
                        playlist_type="cache",
                        created_date=datetime.now()
                    )
                    result.append(playlist)
                    
                    self.logger.debug(f"Created playlist '{playlist_name}' with {len(tracks)} tracks")

            self.logger.info(f"Cache-based generation complete: {len(result)} playlists created from {processed_tracks} tracks (skipped {skipped_tracks})")
            return result

        except Exception as e:
            self.logger.error(f"Error generating from features: {str(e)}")
            return []
    
    def _generate_from_db(self) -> List[Playlist]:
        """Generate playlists from database."""
        self.logger.debug("Generating playlists from database")

        try:
            if not self.cache_file or not os.path.exists(self.cache_file):
                self.logger.warning(f"Cache file not found: {self.cache_file}")
                return []

            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path, bpm, danceability, centroid, loudness, onset_rate
                    FROM audio_features
                    WHERE bpm IS NOT NULL AND danceability IS NOT NULL AND centroid IS NOT NULL
                """)

                tracks = cursor.fetchall()
                self.logger.debug(f"Retrieved {len(tracks)} tracks from database")

                playlists = defaultdict(list)
                processed_tracks = 0
                skipped_tracks = 0

                for row in tracks:
                    try:
                        file_path, bpm, danceability, centroid, loudness, onset_rate = row

                        track_features = {
                            'bpm': bpm,
                            'danceability': danceability,
                            'centroid': centroid,
                            'loudness': loudness,
                            'onset_rate': onset_rate
                        }

                        playlist_name = self._generate_playlist_name(track_features)
                        description = self._generate_description(track_features)

                        # Create mock audio file for database tracks
                        audio_file = AudioFile(file_path=file_path)
                        audio_file.external_metadata = {'features': track_features}

                        playlists[playlist_name].append({
                            'audio_file': audio_file,
                            'description': description,
                            'features': track_features
                        })
                        processed_tracks += 1
                        self.logger.debug(f"Added track to playlist '{playlist_name}': {file_path}")

                    except Exception as e:
                        skipped_tracks += 1
                        self.logger.warning(f"Error processing database track {row[0] if row else 'unknown'}: {str(e)}")

                # Convert to Playlist entities
                result = []
                for playlist_name, tracks in playlists.items():
                    if len(tracks) > 0:
                        track_ids = [t['audio_file'].id for t in tracks]
                        track_paths = [str(t['audio_file'].file_path) for t in tracks]
                        
                        playlist = Playlist(
                            name=playlist_name,
                            description=tracks[0]['description'],
                            track_ids=track_ids,
                            track_paths=track_paths,
                            generation_method="cache_based",
                            playlist_type="cache",
                            created_date=datetime.now()
                        )
                        result.append(playlist)
                        
                        self.logger.debug(f"Created playlist '{playlist_name}' with {len(tracks)} tracks")

                self.logger.info(f"Cache-based generation from database complete: {len(result)} playlists created from {processed_tracks} tracks (skipped {skipped_tracks})")
                return result

        except Exception as e:
            self.logger.error(f"Error generating from database: {str(e)}")
            return []
    
    def _get_category(self, value: float, ranges: Dict[str, tuple]) -> tuple:
        """Get category and description for a value from defined ranges."""
        self.logger.debug(f"Getting category for value {value} from {len(ranges)} ranges")

        try:
            for name, (min_val, max_val, desc) in ranges.items():
                if min_val <= value < max_val:
                    self.logger.debug(f"Value {value} categorized as '{name}': {desc}")
                    return name, desc
            # Return last category as fallback
            last_name = list(ranges.keys())[-1]
            last_desc = ranges[last_name][2]
            self.logger.debug(f"Value {value} categorized as fallback '{last_name}': {last_desc}")
            return last_name, last_desc
        except Exception as e:
            self.logger.error(f"Error categorizing value {value}: {str(e)}")
            return "Unknown", "Unknown category"
    
    def _get_combined_energy(self, features: Dict[str, Any]) -> float:
        """Calculate combined energy score from multiple features."""
        try:
            danceability = float(features.get('danceability', 0))
            loudness = float(features.get('loudness', -60))
            onset_rate = float(features.get('onset_rate', 0))

            # Normalize loudness from dB to 0-1 range
            loudness_norm = (loudness + 60) / 60  # -60dB -> 0, 0dB -> 1

            # Combine features with weights
            combined_energy = (
                danceability * 0.4 +
                loudness_norm * 0.3 +
                min(1.0, onset_rate / 4) * 0.3  # Normalize onset rate
            )

            self.logger.debug(f"Combined energy calculated: {combined_energy}")
            return combined_energy

        except Exception as e:
            self.logger.error(f"Error calculating combined energy: {str(e)}")
            return 0.5  # Default to middle energy
    
    def _sanitize_file_name(self, name: str) -> str:
        """Sanitize file name for safe file system use."""
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Replace multiple underscores with single
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized
    
    def _generate_playlist_name(self, features: Dict[str, Any]) -> str:
        """Generate playlist name based on feature categorization."""
        try:
            bpm = features.get('bpm', 0)
            centroid = features.get('centroid', 0)
            combined_energy = self._get_combined_energy(features)

            # Get categories
            bpm_category, bpm_desc = self._get_category(bpm, self.bpm_ranges)
            energy_category, energy_desc = self._get_category(combined_energy, self.energy_levels)
            mood_category, mood_desc = self._get_category(centroid, self.mood_ranges)

            # Create playlist name
            playlist_name = f"{bpm_category}_{energy_category}_{mood_category}"
            sanitized_name = self._sanitize_file_name(playlist_name)

            self.logger.debug(f"Generated playlist name: {sanitized_name}")
            return sanitized_name

        except Exception as e:
            self.logger.error(f"Error generating playlist name: {str(e)}")
            return "Unknown_Playlist"
    
    def _generate_description(self, features: Dict[str, Any]) -> str:
        """Generate description based on feature categorization."""
        try:
            bpm = features.get('bpm', 0)
            centroid = features.get('centroid', 0)
            combined_energy = self._get_combined_energy(features)

            # Get categories and descriptions
            bpm_category, bpm_desc = self._get_category(bpm, self.bpm_ranges)
            energy_category, energy_desc = self._get_category(combined_energy, self.energy_levels)
            mood_category, mood_desc = self._get_category(centroid, self.mood_ranges)

            # Create description
            description = f"{bpm_desc}. {energy_desc}. {mood_desc}."

            self.logger.debug(f"Generated description: {description}")
            return description

        except Exception as e:
            self.logger.error(f"Error generating description: {str(e)}")
            return "A collection of music tracks."
    
    def _merge_playlists(self, playlists: List[Playlist]) -> List[Playlist]:
        """Merge small playlists into larger ones."""
        self.logger.debug(f"Merging {len(playlists)} playlists")

        try:
            # Find small playlists (less than 10 tracks)
            small_playlists = [p for p in playlists if len(p.track_ids) < 10]
            large_playlists = [p for p in playlists if len(p.track_ids) >= 10]

            self.logger.debug(f"Found {len(small_playlists)} small playlists and {len(large_playlists)} large playlists")

            if not small_playlists:
                self.logger.debug("No small playlists to merge")
                return playlists

            # Create a "Mixed" playlist for small ones
            mixed_track_ids = []
            mixed_track_paths = []
            merged_from = []

            for playlist in small_playlists:
                mixed_track_ids.extend(playlist.track_ids)
                mixed_track_paths.extend(playlist.track_paths)
                merged_from.append(playlist.name)
                self.logger.debug(f"Added {len(playlist.track_ids)} tracks from '{playlist.name}' to mixed playlist")

            if mixed_track_ids:
                mixed_playlist = Playlist(
                    name="Mixed_Collection",
                    description="A diverse collection of tracks",
                    track_ids=mixed_track_ids,
                    track_paths=mixed_track_paths,
                    generation_method="cache_based",
                    playlist_type="cache",
                    created_date=datetime.now()
                )
                large_playlists.append(mixed_playlist)
                
                self.logger.info(f"Created mixed playlist with {len(mixed_track_ids)} tracks from {len(small_playlists)} small playlists")

            return large_playlists

        except Exception as e:
            self.logger.error(f"Error merging playlists: {str(e)}")
            return playlists 