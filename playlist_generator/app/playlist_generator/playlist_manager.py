import logging
import os
import json
from typing import Dict, List, Any, Optional
from .kmeans import KMeansPlaylistGenerator
from .cache import CacheBasedGenerator
from .time_based import TimeBasedScheduler
from .tag_based import TagBasedPlaylistGenerator
from .feature_group import FeatureGroupPlaylistGenerator
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PlaylistManager:
    """Manage playlist generation using various methods and strategies."""

    def __init__(self, cache_file: str = None, playlist_method: str = 'all', min_tracks_per_genre: int = 10) -> None:
        """Initialize the PlaylistManager.

        Args:
            cache_file (str, optional): Path to the cache database file. Defaults to None.
            playlist_method (str, optional): Playlist generation method. Defaults to 'all'.
            min_tracks_per_genre (int, optional): Minimum tracks per genre for tag-based playlists. Defaults to 10.
        """
        self.cache_file = cache_file
        self.playlist_method = playlist_method
        logger.info(
            f"Initializing PlaylistManager with method: {playlist_method}, min_tracks_per_genre: {min_tracks_per_genre}")
        self.feature_group_generator = FeatureGroupPlaylistGenerator(
            cache_file)
        self.time_scheduler = TimeBasedScheduler()
        self.kmeans_generator = KMeansPlaylistGenerator(cache_file)
        self.cache_generator = CacheBasedGenerator(cache_file)
        self.tag_generator = TagBasedPlaylistGenerator(
            min_tracks_per_genre=min_tracks_per_genre, db_file=cache_file)
        self.playlist_stats = defaultdict(dict)
        logger.debug("PlaylistManager initialization complete")

    def generate_playlists(self, features: List[Dict[str, Any]], num_playlists: int = 8) -> Dict[str, Any]:
        """Generate playlists using the specified method.

        Args:
            features (List[Dict[str, Any]]): List of feature dictionaries.
            num_playlists (int, optional): Number of playlists to generate (for kmeans). Defaults to 8.

        Returns:
            dict[str, any]: Dictionary of playlist names to playlist data.
        """
        """Generate playlists using specified method"""
        logger.info(
            f"Starting playlist generation with {len(features)} tracks, method: {self.playlist_method}, target playlists: {num_playlists}")

        if not features:
            logger.warning("No features provided for playlist generation")
            return {}

        try:
            final_playlists = {}
            used_tracks = set()
            min_size = 10
            max_size = 500

            def _finalize_playlists(playlists: Dict[str, Any], features: List[Dict[str, Any]]) -> Dict[str, Any]:
                logger.debug(
                    f"Finalizing {len(playlists)} playlists with {len(features)} total tracks")
                # Merge small playlists, split large ones, assign leftovers
                all_tracks = set(f['filepath'] for f in features)
                assigned_tracks = set()
                for data in playlists.values():
                    assigned_tracks.update(data['tracks'])
                unassigned_tracks = list(all_tracks - assigned_tracks)
                logger.debug(
                    f"Found {len(unassigned_tracks)} unassigned tracks")

                # Merge small playlists
                merged = {}
                small = []
                for name, data in playlists.items():
                    if len(data['tracks']) < min_size:
                        small.extend(data['tracks'])
                        logger.debug(
                            f"Playlist '{name}' has {len(data['tracks'])} tracks, marking for merge")
                    else:
                        merged[name] = data

                logger.debug(
                    f"Merging {len(small)} tracks from {len(playlists) - len(merged)} small playlists")

                # Add small tracks to mixed
                if small:
                    if 'Mixed_Collection' not in merged:
                        merged['Mixed_Collection'] = {
                            'tracks': [],
                            'features': {'type': 'mixed'}
                        }
                    merged['Mixed_Collection']['tracks'].extend(small)
                    logger.debug(
                        f"Added {len(small)} small playlist tracks to Mixed_Collection")

                # Add unassigned tracks to mixed
                if unassigned_tracks:
                    if 'Mixed_Collection' not in merged:
                        merged['Mixed_Collection'] = {
                            'tracks': [],
                            'features': {'type': 'mixed'}
                        }
                    merged['Mixed_Collection']['tracks'].extend(
                        unassigned_tracks)
                    logger.debug(
                        f"Added {len(unassigned_tracks)} unassigned tracks to Mixed_Collection")

                # Split large playlists
                balanced = {}
                for name, data in merged.items():
                    tracks = data['tracks']
                    if len(tracks) > max_size:
                        chunks = [tracks[i:i + max_size]
                                  for i in range(0, len(tracks), max_size)]
                        logger.debug(
                            f"Splitting large playlist '{name}' ({len(tracks)} tracks) into {len(chunks)} parts")
                        for i, chunk in enumerate(chunks, 1):
                            new_name = f"{name}_Part{i}" if len(
                                chunks) > 1 else name
                            balanced[new_name] = {
                                'tracks': chunk,
                                'features': data.get('features', {})
                            }
                    else:
                        balanced[name] = data

                logger.info(
                    f"Finalized {len(balanced)} playlists after merging/splitting")
                return balanced

            # Tag-based playlists (genre + decade)
            if self.playlist_method == 'tags':
                logger.info("Generating tag-based playlists")
                tag_playlists = self.tag_generator.generate(features)
                logger.debug(
                    f"Generated {len(tag_playlists)} tag-based playlists")
                for name, data in tag_playlists.items():
                    if len(data['tracks']) >= 3:
                        final_playlists[name] = data
                        used_tracks.update(data['tracks'])
                        logger.debug(
                            f"Added tag playlist '{name}' with {len(data['tracks'])} tracks")
                final_playlists = _finalize_playlists(
                    final_playlists, features)
                self._calculate_playlist_stats(final_playlists, features)
                return final_playlists

            # Default: Feature-group-based generation
            if self.playlist_method == 'all' or not self.playlist_method:
                logger.info("Generating feature-group-based playlists")
                fg_playlists = self.feature_group_generator.generate(features)
                logger.debug(
                    f"Generated {len(fg_playlists)} feature-group playlists")
                for name, data in fg_playlists.items():
                    if len(data['tracks']) >= 3:
                        final_playlists[name] = data
                        used_tracks.update(data['tracks'])
                        logger.debug(
                            f"Added feature-group playlist '{name}' with {len(data['tracks'])} tracks")
                final_playlists = _finalize_playlists(
                    final_playlists, features)
                self._calculate_playlist_stats(final_playlists, features)
                return final_playlists

            if self.playlist_method == 'time':
                logger.info("Generating time-based playlists")
                # Only time-based playlists
                time_playlists = self.time_scheduler.generate_time_based_playlists(
                    features)
                logger.debug(
                    f"Generated {len(time_playlists)} time-based playlists")
                for name, data in time_playlists.items():
                    if len(data['tracks']) >= 3:
                        final_playlists[name] = data
                        logger.debug(
                            f"Added time playlist '{name}' with {len(data['tracks'])} tracks")
                final_playlists = _finalize_playlists(
                    final_playlists, features)
                self._calculate_playlist_stats(final_playlists, features)
                return final_playlists

            if self.playlist_method == 'kmeans':
                logger.info(
                    f"Generating kmeans playlists with {num_playlists} clusters")
                # Only kmeans playlists
                kmeans_playlists = self.kmeans_generator.generate(
                    features, num_playlists)
                logger.debug(
                    f"Generated {len(kmeans_playlists)} kmeans playlists")
                for name, data in kmeans_playlists.items():
                    if len(data['tracks']) >= 3:
                        final_playlists[name] = data
                        logger.debug(
                            f"Added kmeans playlist '{name}' with {len(data['tracks'])} tracks")
                final_playlists = _finalize_playlists(
                    final_playlists, features)
                self._calculate_playlist_stats(final_playlists, features)
                return final_playlists

            if self.playlist_method == 'cache':
                logger.info("Generating cache-based playlists")
                # Only cache-based playlists
                cache_playlists = self.cache_generator.generate(features)
                logger.debug(
                    f"Generated {len(cache_playlists)} cache-based playlists")
                for name, data in cache_playlists.items():
                    if len(data['tracks']) >= 3:
                        final_playlists[name] = data
                        logger.debug(
                            f"Added cache playlist '{name}' with {len(data['tracks'])} tracks")
                final_playlists = _finalize_playlists(
                    final_playlists, features)
                self._calculate_playlist_stats(final_playlists, features)
                return final_playlists

            # Fallback: use feature group generator
            logger.info("Using fallback feature-group generation")
            fg_playlists = self.feature_group_generator.generate(features)
            logger.debug(
                f"Generated {len(fg_playlists)} fallback feature-group playlists")
            for name, data in fg_playlists.items():
                if len(data['tracks']) >= 3:
                    final_playlists[name] = data
                    used_tracks.update(data['tracks'])
                    logger.debug(
                        f"Added fallback playlist '{name}' with {len(data['tracks'])} tracks")
            final_playlists = _finalize_playlists(final_playlists, features)
            self._calculate_playlist_stats(final_playlists, features)
            return final_playlists

        except Exception as e:
            logger.error(f"Error generating playlists: {str(e)}")
            return {}

    def _optimize_playlists(self, playlists: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize playlists by merging similar ones and ensuring good distribution"""
        logger.debug(f"Optimizing {len(playlists)} playlists")
        try:
            optimized = {}
            merged_tracks = set()

            # Sort playlists by number of tracks (largest first)
            sorted_playlists = sorted(
                playlists.items(),
                key=lambda x: len(x[1].get('tracks', [])),
                reverse=True
            )
            logger.debug(
                f"Sorted {len(sorted_playlists)} playlists by track count")

            for name, data in sorted_playlists:
                tracks = data.get('tracks', [])
                if not tracks:
                    logger.debug(f"Skipping empty playlist '{name}'")
                    continue

                # Skip if most tracks are already in other playlists
                overlap = len(set(tracks) & merged_tracks) / len(tracks)
                if overlap > 0.7:  # More than 70% overlap
                    logger.debug(
                        f"Skipping playlist '{name}' due to {overlap:.1%} overlap with existing playlists")
                    continue

                # Find similar playlists for potential merging
                similar_playlists = self._find_similar_playlists(
                    name, data, optimized, threshold=0.3
                )

                if similar_playlists:
                    # Merge with most similar playlist
                    target_name = similar_playlists[0][0]
                    logger.debug(
                        f"Merging playlist '{name}' into '{target_name}' (similarity: {similar_playlists[0][1]:.2f})")
                    optimized[target_name]['tracks'].extend(tracks)
                else:
                    # Keep as separate playlist
                    optimized[name] = data
                    logger.debug(
                        f"Keeping playlist '{name}' as separate (no similar playlists found)")

                merged_tracks.update(tracks)

            logger.info(
                f"Optimization complete: {len(optimized)} playlists after merging")
            return optimized

        except Exception as e:
            logger.error(f"Error optimizing playlists: {str(e)}")
            return playlists

    def _find_similar_playlists(self, name: str, data: Dict[str, Any],
                                existing_playlists: Dict[str, Any],
                                threshold: float = 0.3) -> List[tuple]:
        """Find similar playlists based on features and track overlap"""
        similarities = []
        features = data.get('features', {})

        for existing_name, existing_data in existing_playlists.items():
            if existing_name == name:
                continue

            similarity = 0.0
            existing_features = existing_data.get('features', {})

            # Compare feature values
            for key in ['bpm_group', 'energy_group', 'mood_group', 'key_group']:
                if key in features and key in existing_features:
                    if features[key] == existing_features[key]:
                        similarity += 0.25

            # Calculate track overlap
            existing_tracks = set(existing_data.get('tracks', []))
            current_tracks = set(data.get('tracks', []))
            if existing_tracks and current_tracks:
                overlap = len(existing_tracks & current_tracks) / \
                    len(current_tracks)
                similarity += overlap

            if similarity >= threshold:
                similarities.append((existing_name, similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)

    def _balance_playlists(self, playlists: Dict[str, Any]) -> Dict[str, Any]:
        """Balance playlist sizes and ensure good track distribution"""
        try:
            balanced = {}
            min_size = 10  # Minimum playlist size
            max_size = 100  # Maximum playlist size
            unassigned_tracks = []

            # First pass: split large playlists and identify small ones
            for name, data in playlists.items():
                tracks = data.get('tracks', [])
                if len(tracks) > max_size:
                    # Split large playlist
                    chunks = [tracks[i:i + max_size]
                              for i in range(0, len(tracks), max_size)]
                    for i, chunk in enumerate(chunks, 1):
                        new_name = f"{name}_Part{i}" if len(
                            chunks) > 1 else name
                        balanced[new_name] = {
                            'tracks': chunk,
                            'features': data.get('features', {})
                        }
                elif len(tracks) < min_size:
                    unassigned_tracks.extend(tracks)
                else:
                    balanced[name] = data

            # Second pass: handle unassigned tracks
            if unassigned_tracks:
                unassigned_tracks = list(
                    set(unassigned_tracks))  # Remove duplicates
                if len(unassigned_tracks) >= min_size:
                    # Create new playlist for unassigned tracks
                    balanced['Mixed_Collection'] = {
                        'tracks': unassigned_tracks,
                        'features': {'type': 'mixed'}
                    }
                else:
                    # Try to add to existing playlists
                    self._distribute_tracks(balanced, unassigned_tracks)

            return balanced

        except Exception as e:
            logger.error(f"Error balancing playlists: {str(e)}")
            return playlists

    def _distribute_tracks(self, playlists: Dict[str, Any], tracks: List[str]) -> None:
        """Distribute tracks among existing playlists intelligently"""
        if not tracks:
            return

        try:
            # Sort playlists by size (smallest first)
            sorted_playlists = sorted(
                playlists.items(),
                key=lambda x: len(x[1].get('tracks', []))
            )

            for track in tracks:
                # Find best playlist for track
                for name, data in sorted_playlists:
                    if len(data.get('tracks', [])) < 100:  # Check max size
                        data['tracks'].append(track)
                        break

        except Exception as e:
            logger.error(f"Error distributing tracks: {str(e)}")

    def _calculate_playlist_stats(self, playlists: Dict[str, Any],
                                  features: List[Dict[str, Any]]) -> None:
        """Calculate statistics for each playlist"""
        try:
            # Create lookup for quick feature access
            feature_lookup = {f['filepath']: f for f in features}

            for name, data in playlists.items():
                tracks = data.get('tracks', [])
                if not tracks:
                    continue

                track_features = [
                    feature_lookup[t] for t in tracks
                    if t in feature_lookup
                ]

                if not track_features:
                    continue

                # Calculate basic stats with None handling
                bpm_values = [f.get('bpm', 0) for f in track_features if f.get('bpm') is not None]
                danceability_values = [min(1.0, max(0.0, f.get('danceability', 0))) for f in track_features if f.get('danceability') is not None]
                duration_values = [f.get('duration', 0) for f in track_features if f.get('duration') is not None]
                
                stats = {
                    'track_count': len(track_features),
                    'total_duration': sum(duration_values) if duration_values else 0,
                    'avg_bpm': np.mean(bpm_values) if bpm_values else 0,
                    'avg_danceability': np.mean(danceability_values) if danceability_values else 0,
                    'key_distribution': self._get_key_distribution(track_features)
                }

                self.playlist_stats[name] = stats

        except Exception as e:
            logger.error(f"Error calculating playlist stats: {str(e)}")

    def _get_key_distribution(self, features: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of musical keys in playlist"""
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']
        distribution = defaultdict(int)

        for feature in features:
            key_idx = feature.get('key', -1)
            if 0 <= key_idx < len(keys):
                key_name = keys[key_idx]
                scale = 'Major' if feature.get('scale', 0) == 1 else 'Minor'
                distribution[f"{key_name} {scale}"] += 1

        return dict(distribution)

    def get_playlist_stats(self, playlist_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific playlist or all playlists.

        Args:
            playlist_name (str, optional): Name of the playlist. Defaults to None.

        Returns:
            dict: Playlist statistics.
        """
        if playlist_name:
            return self.playlist_stats.get(playlist_name, {})
        return dict(self.playlist_stats)

    def get_recommendations(self, playlist_name: str) -> List[str]:
        """Get track recommendations for a playlist based on its characteristics.

        Args:
            playlist_name (str): Name of the playlist.

        Returns:
            List[str]: List of recommended track file paths.
        """
        try:
            stats = self.playlist_stats.get(playlist_name)
            if not stats:
                return []

            # Implementation of recommendation logic would go here
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []
