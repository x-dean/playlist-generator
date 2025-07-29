# app/playlist_generator/cache.py
import sqlite3
import logging
import traceback
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)


class CacheBasedGenerator:
    """Generate playlists using rule-based feature bins (cache method)."""

    def __init__(self, cache_file: str) -> None:
        """Initialize the cache-based playlist generator.

        Args:
            cache_file (str): Path to the cache database file.
        """
        self.cache_file = cache_file
        self.playlist_history = defaultdict(list)
        logger.debug(
            f"Initialized CacheBasedGenerator with cache file: {cache_file}")

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
        logger.debug("Cache-based generator initialized with feature ranges")

    def _get_category(self, value: float, ranges: Dict[str, tuple]) -> tuple:
        """Get category and description for a value from defined ranges"""
        logger.debug(
            f"Getting category for value {value} from {len(ranges)} ranges")

        try:
            for name, (min_val, max_val, desc) in ranges.items():
                if min_val <= value < max_val:
                    logger.debug(
                        f"Value {value} categorized as '{name}': {desc}")
                    return name, desc
            # Return last category as fallback
            last_name = list(ranges.keys())[-1]
            last_desc = ranges[last_name][2]
            logger.debug(
                f"Value {value} categorized as fallback '{last_name}': {last_desc}")
            return last_name, last_desc
        except Exception as e:
            logger.error(f"Error categorizing value {value}: {str(e)}")
            return "Unknown", "Unknown category"

    def _get_combined_energy(self, track: Dict[str, Any]) -> float:
        """Calculate combined energy score from multiple features"""
        logger.debug(
            f"Calculating combined energy for track: {track.get('filepath', 'unknown')}")

        try:
            danceability = float(track.get('danceability', 0))
            loudness = float(track.get('loudness', -60))
            onset_rate = float(track.get('onset_rate', 0))

            # Normalize loudness from dB to 0-1 range
            loudness_norm = (loudness + 60) / 60  # -60dB -> 0, 0dB -> 1

            # Combine features with weights
            combined_energy = (
                danceability * 0.4 +
                loudness_norm * 0.3 +
                min(1.0, onset_rate / 4) * 0.3  # Normalize onset rate
            )

            logger.debug(
                f"Combined energy calculation: danceability={danceability:.3f}, loudness_norm={loudness_norm:.3f}, onset_rate={onset_rate:.3f} -> combined={combined_energy:.3f}")
            return combined_energy

        except Exception as e:
            logger.error(f"Error calculating combined energy: {str(e)}")
            return 0.0

    def _sanitize_file_name(self, name: str) -> str:
        """Sanitize a string to be used as a filename."""
        logger.debug(f"Sanitizing filename: {name}")

        try:
            sanitized = re.sub(r'[^A-Za-z0-9_-]+', '_', name)
            sanitized = re.sub(r'_+', '_', sanitized)
            sanitized = sanitized.strip('_')
            logger.debug(f"Sanitized filename: {name} -> {sanitized}")
            return sanitized
        except Exception as e:
            logger.error(f"Error sanitizing filename {name}: {str(e)}")
            return name

    def _generate_playlist_name(self, features: Dict[str, Any]) -> str:
        """Generate descriptive playlist name based on musical features"""
        logger.debug(
            f"Generating playlist name for track: {features.get('filepath', 'unknown')}")

        try:
            bpm = float(features.get('bpm', 0))
            
            # Skip tracks with failed BPM extraction (-1.0 marker)
            if bpm == -1.0:
                logger.debug(f"Skipping track with failed BPM extraction: {features.get('filepath', 'unknown')}")
                return "Failed_BPM_Playlist"
            
            energy = self._get_combined_energy(features)
            centroid = float(features.get('centroid', 0))

            bpm_category, _ = self._get_category(bpm, self.bpm_ranges)
            energy_category, _ = self._get_category(energy, self.energy_levels)
            mood_category, _ = self._get_category(centroid, self.mood_ranges)

            name = f"{bpm_category}_{energy_category}_{mood_category}"
            sanitized_name = self._sanitize_file_name(name)

            logger.debug(
                f"Generated playlist name: {name} -> {sanitized_name}")
            return sanitized_name

        except Exception as e:
            logger.error(f"Error generating playlist name: {str(e)}")
            return "Unknown_Playlist"

    def _generate_description(self, features: Dict[str, Any]) -> str:
        """Generate human-readable description based on musical features"""
        logger.debug(
            f"Generating description for track: {features.get('filepath', 'unknown')}")

        try:
            bpm = float(features.get('bpm', 0))
            
            # Skip tracks with failed BPM extraction (-1.0 marker)
            if bpm == -1.0:
                logger.debug(f"Skipping track with failed BPM extraction: {features.get('filepath', 'unknown')}")
                return "Track with failed BPM extraction."
            
            energy = self._get_combined_energy(features)
            centroid = float(features.get('centroid', 0))

            bpm_category, bpm_desc = self._get_category(bpm, self.bpm_ranges)
            energy_category, energy_desc = self._get_category(
                energy, self.energy_levels)
            mood_category, mood_desc = self._get_category(
                centroid, self.mood_ranges)

            description = f"{bpm_desc}. {energy_desc}. {mood_desc}."
            logger.debug(f"Generated description: {description}")
            return description

        except Exception as e:
            logger.error(f"Error generating description: {str(e)}")
            return "A collection of music tracks."

    def generate(self, features_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """Generate playlists using cache-based method."""
        logger.debug(
            f"Starting cache-based playlist generation with {len(features_list) if features_list else 'database'} tracks")

        try:
            if features_list:
                return self._generate_from_features(features_list)
            else:
                return self._generate_from_db()
        except Exception as e:
            logger.error(f"Error in cache-based playlist generation: {str(e)}")
            import traceback
            logger.error(
                f"Cache-based generation error traceback: {traceback.format_exc()}")
            return {}

    def _generate_from_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate playlists from a list of feature dictionaries."""
        logger.debug(
            f"Generating playlists from {len(features_list)} feature dictionaries")

        try:
            playlists = defaultdict(list)
            processed_tracks = 0
            skipped_tracks = 0

            for track in features_list:
                if not track or 'filepath' not in track:
                    skipped_tracks += 1
                    logger.debug(f"Skipping track without filepath: {track}")
                    continue

                try:
                    playlist_name = self._generate_playlist_name(track)
                    description = self._generate_description(track)

                    playlists[playlist_name].append({
                        'filepath': track['filepath'],
                        'description': description,
                        'features': track
                    })
                    processed_tracks += 1
                    logger.debug(
                        f"Added track to playlist '{playlist_name}': {track['filepath']}")

                except Exception as e:
                    skipped_tracks += 1
                    logger.warning(
                        f"Error processing track {track.get('filepath', 'unknown')}: {str(e)}")

            # Convert to expected format
            result = {}
            for playlist_name, tracks in playlists.items():
                if len(tracks) > 0:
                    result[playlist_name] = {
                        'tracks': [t['filepath'] for t in tracks],
                        'features': {
                            'type': 'cache_based',
                            'description': tracks[0]['description'],
                            'track_count': len(tracks)
                        }
                    }
                    logger.debug(
                        f"Created playlist '{playlist_name}' with {len(tracks)} tracks")

            logger.info(
                f"Cache-based generation from features complete: {len(result)} playlists created from {processed_tracks} tracks (skipped {skipped_tracks})")
            return result

        except Exception as e:
            logger.error(f"Error generating from features: {str(e)}")
            import traceback
            logger.error(
                f"Generate from features error traceback: {traceback.format_exc()}")
            return {}

    def _generate_from_db(self) -> Dict[str, Dict[str, Any]]:
        """Generate playlists from database."""
        logger.debug("Generating playlists from database")

        try:
            if not self.cache_file or not os.path.exists(self.cache_file):
                logger.warning(f"Cache file not found: {self.cache_file}")
                return {}

            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path, bpm, danceability, centroid, loudness, onset_rate
                    FROM audio_features
                    WHERE bpm IS NOT NULL AND danceability IS NOT NULL AND centroid IS NOT NULL
                """)

                tracks = cursor.fetchall()
                logger.debug(f"Retrieved {len(tracks)} tracks from database")

                playlists = defaultdict(list)
                processed_tracks = 0
                skipped_tracks = 0

                for row in tracks:
                    try:
                        file_path, bpm, danceability, centroid, loudness, onset_rate = row

                        track_features = {
                            'filepath': file_path,
                            'bpm': bpm,
                            'danceability': danceability,
                            'centroid': centroid,
                            'loudness': loudness,
                            'onset_rate': onset_rate
                        }

                        playlist_name = self._generate_playlist_name(
                            track_features)
                        description = self._generate_description(
                            track_features)

                        playlists[playlist_name].append({
                            'filepath': file_path,
                            'description': description,
                            'features': track_features
                        })
                        processed_tracks += 1
                        logger.debug(
                            f"Added track to playlist '{playlist_name}': {file_path}")

                    except Exception as e:
                        skipped_tracks += 1
                        logger.warning(
                            f"Error processing database track {row[0] if row else 'unknown'}: {str(e)}")

                # Convert to expected format
                result = {}
                for playlist_name, tracks in playlists.items():
                    if len(tracks) > 0:
                        result[playlist_name] = {
                            'tracks': [t['filepath'] for t in tracks],
                            'features': {
                                'type': 'cache_based',
                                'description': tracks[0]['description'],
                                'track_count': len(tracks)
                            }
                        }
                        logger.debug(
                            f"Created playlist '{playlist_name}' with {len(tracks)} tracks")

                logger.info(
                    f"Cache-based generation from database complete: {len(result)} playlists created from {processed_tracks} tracks (skipped {skipped_tracks})")
                return result

        except Exception as e:
            logger.error(f"Error generating from database: {str(e)}")
            import traceback
            logger.error(
                f"Generate from database error traceback: {traceback.format_exc()}")
            return {}

    def _merge_playlists(self, playlists: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Merge small playlists into larger ones."""
        logger.debug(f"Merging {len(playlists)} playlists")

        try:
            # Find small playlists (less than 10 tracks)
            small_playlists = {name: data for name, data in playlists.items()
                               if len(data['tracks']) < 10}
            large_playlists = {name: data for name, data in playlists.items()
                               if len(data['tracks']) >= 10}

            logger.debug(
                f"Found {len(small_playlists)} small playlists and {len(large_playlists)} large playlists")

            if not small_playlists:
                logger.debug("No small playlists to merge")
                return playlists

            # Create a "Mixed" playlist for small ones
            mixed_tracks = []
            for name, data in small_playlists.items():
                mixed_tracks.extend(data['tracks'])
                logger.debug(
                    f"Added {len(data['tracks'])} tracks from '{name}' to mixed playlist")

            if mixed_tracks:
                large_playlists['Mixed_Collection'] = {
                    'tracks': mixed_tracks,
                    'features': {
                        'type': 'cache_based',
                        'description': 'A diverse collection of tracks',
                        'track_count': len(mixed_tracks),
                        'merged_from': list(small_playlists.keys())
                    }
                }
                logger.info(
                    f"Created mixed playlist with {len(mixed_tracks)} tracks from {len(small_playlists)} small playlists")

            logger.debug(
                f"Merging complete: {len(large_playlists)} final playlists")
            return large_playlists

        except Exception as e:
            logger.error(f"Error merging playlists: {str(e)}")
            import traceback
            logger.error(
                f"Merge playlists error traceback: {traceback.format_exc()}")
            return playlists

    def _get_mood_category(self, centroid: float) -> str:
        """Get mood category based on spectral centroid."""
        logger.debug(f"Getting mood category for centroid: {centroid}")

        try:
            if centroid < 1000:
                return 'Dark'
            elif centroid < 2000:
                return 'Warm'
            elif centroid < 4000:
                return 'Balanced'
            elif centroid < 6000:
                return 'Bright'
            else:
                return 'Brilliant'
        except Exception as e:
            logger.error(f"Error getting mood category: {str(e)}")
            return 'Unknown'

    def _calculate_merge_score(self, playlist1: Dict[str, Any], playlist2: Dict[str, Any]) -> float:
        """Calculate similarity score between two playlists for merging."""
        logger.debug("Calculating merge score between playlists")

        try:
            # Simple similarity based on feature ranges
            # This is a placeholder - implement more sophisticated similarity if needed
            score = 0.5  # Default neutral score
            logger.debug(f"Merge score calculated: {score}")
            return score
        except Exception as e:
            logger.error(f"Error calculating merge score: {str(e)}")
            return 0.0
