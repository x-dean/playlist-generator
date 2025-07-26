import os
import sqlite3
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class FeatureGroupPlaylistGenerator:
    """Generate playlists by grouping tracks based on extracted audio features."""

    def __init__(self, cache_file: str):
        """Initialize the feature group playlist generator.

        Args:
            cache_file (str): Path to the cache database file.
        """
        self.cache_file = cache_file
        logger.debug(
            f"Initialized FeatureGroupPlaylistGenerator with cache file: {cache_file}")

    def sanitize_filename(self, name: str) -> str:
        """Sanitize a string to be used as a filename.

        Args:
            name (str): The string to sanitize.

        Returns:
            str: Sanitized filename.
        """
        logger.debug(f"Sanitizing filename: {name}")
        name = re.sub(r'[^\w\-_]', '_', name)
        sanitized = re.sub(r'_+', '_', name).strip('_')
        logger.debug(f"Sanitized filename: {name} -> {sanitized}")
        return sanitized

    def generate(self, features_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate playlists from a list of feature dictionaries.

        Args:
            features_list (List[Dict[str, Any]]): List of feature dictionaries.

        Returns:
            dict[str, dict]: Dictionary of playlist names to playlist data.
        """
        logger.debug(
            f"Starting feature group playlist generation with {len(features_list)} tracks")

        try:
            # Helper mapping for mood similarity
            centroid_mapping = {
                'Warm': 0,
                'Mellow': 1,
                'Balanced': 2,
                'Bright': 3,
                'Crisp': 4
            }

            # Define groupers
            def get_energy_group(danceability):
                if danceability < 0.3:
                    return 'Chill'
                if danceability < 0.5:
                    return 'Mellow'
                if danceability < 0.7:
                    return 'Groovy'
                if danceability < 0.85:
                    return 'Energetic'
                return 'Intense'

            def get_bpm_group(bpm):
                if bpm < 70:
                    return 'Slow'
                if bpm < 100:
                    return 'Medium'
                if bpm < 130:
                    return 'Upbeat'
                if bpm < 160:
                    return 'Fast'
                return 'VeryFast'

            def get_mood_group(centroid):
                if centroid < 500:
                    return 'Warm'
                if centroid < 1500:
                    return 'Mellow'
                if centroid < 3000:
                    return 'Balanced'
                if centroid < 6000:
                    return 'Bright'
                return 'Crisp'

            keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
                    'F#', 'G', 'G#', 'A', 'A#', 'B']
            playlists = {}
            processed_tracks = 0
            skipped_tracks = 0

            logger.debug("Processing tracks for feature grouping")

            for f in features_list:
                if not f or 'filepath' not in f:
                    skipped_tracks += 1
                    logger.debug(f"Skipping track without filepath: {f}")
                    continue

                bpm = f.get('bpm', 0)
                centroid = f.get('centroid', 0)
                danceability = f.get('danceability', 0)
                key = f.get('key', None)
                scale = f.get('scale', None)
                file_path = f['filepath']

                # Skip invalid data
                if None in (bpm, centroid, danceability):
                    skipped_tracks += 1
                    logger.debug(
                        f"Skipping track with invalid data: {file_path}")
                    continue

                bpm_group = get_bpm_group(bpm)
                energy_group = get_energy_group(danceability)
                mood_group = get_mood_group(centroid)
                key_group = ''

                if key is not None and 0 <= key <= 11:
                    key_group = f"{keys[int(key)]}_{'Major' if scale == 1 else 'Minor'}"
                    logger.debug(
                        f"Track {file_path}: key={key_group}, bpm={bpm}, energy={energy_group}, mood={mood_group}")

                # Create playlist name
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
                playlist_name = self.sanitize_filename(playlist_name)

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
                    logger.debug(f"Created new playlist: {playlist_name}")

                playlists[playlist_name]['tracks'].append(file_path)
                processed_tracks += 1

            logger.info(
                f"Feature group generation complete: {len(playlists)} playlists created from {processed_tracks} tracks (skipped {skipped_tracks})")
            logger.debug(f"Generated playlists: {list(playlists.keys())}")

            return playlists

        except Exception as e:
            logger.error(f"Error in feature group generation: {str(e)}")
            import traceback
            logger.error(
                f"Feature group generation error traceback: {traceback.format_exc()}")
            return {}
