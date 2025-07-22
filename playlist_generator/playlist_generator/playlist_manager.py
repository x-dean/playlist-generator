import logging
from typing import Dict, List, Any, Optional
from .time_based import TimeBasedScheduler
from .kmeans import KMeansPlaylistGenerator
from .cache import CacheBasedGenerator
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from .feature_group import FeatureGroupPlaylistGenerator

logger = logging.getLogger(__name__)

class PlaylistManager:
    def __init__(self, cache_file: str = None, playlist_method: str = 'all'):
        self.cache_file = cache_file
        self.playlist_method = playlist_method
        self.feature_group_generator = FeatureGroupPlaylistGenerator(cache_file)
        self.time_scheduler = TimeBasedScheduler()
        self.kmeans_generator = KMeansPlaylistGenerator(cache_file)
        self.cache_generator = CacheBasedGenerator(cache_file)
        self.playlist_stats = defaultdict(dict)

    def generate_playlists(self, features: List[Dict[str, Any]], num_playlists: int = 8) -> Dict[str, Any]:
        """Generate playlists using specified method"""
        if not features:
            logger.warning("No features provided for playlist generation")
            return {}

        try:
            final_playlists = {}
            used_tracks = set()
            min_size = 10
            max_size = 500

            def _finalize_playlists(playlists: Dict[str, Any], features: List[Dict[str, Any]]) -> Dict[str, Any]:
                # Merge small playlists, split large ones, assign leftovers
                all_tracks = set(f['filepath'] for f in features)
                assigned_tracks = set()
                for data in playlists.values():
                    assigned_tracks.update(data['tracks'])
                unassigned_tracks = list(all_tracks - assigned_tracks)
                # Merge small playlists
                merged = {}
                small = []
                for name, data in playlists.items():
                    if len(data['tracks']) < min_size:
                        small.extend(data['tracks'])
                    else:
                        merged[name] = data
                # Add small tracks to mixed
                if small:
                    if 'Mixed_Collection' not in merged:
                        merged['Mixed_Collection'] = {
                            'tracks': [],
                            'features': {'type': 'mixed'}
                        }
                    merged['Mixed_Collection']['tracks'].extend(small)
                # Add unassigned tracks to mixed
                if unassigned_tracks:
                    if 'Mixed_Collection' not in merged:
                        merged['Mixed_Collection'] = {
                            'tracks': [],
                            'features': {'type': 'mixed'}
                        }
                    merged['Mixed_Collection']['tracks'].extend(unassigned_tracks)
                # Split large playlists
                balanced = {}
                for name, data in merged.items():
                    tracks = data['tracks']
                    if len(tracks) > max_size:
                        chunks = [tracks[i:i + max_size] for i in range(0, len(tracks), max_size)]
                        for i, chunk in enumerate(chunks, 1):
                            new_name = f"{name}_Part{i}" if len(chunks) > 1 else name
                            balanced[new_name] = {
                                'tracks': chunk,
                                'features': data.get('features', {})
                            }
                    else:
                        balanced[name] = data
                return balanced

            # Default: Feature-group-based generation
            if self.playlist_method == 'all' or not self.playlist_method:
                fg_playlists = self.feature_group_generator.generate(features)
                for name, data in fg_playlists.items():
                    if len(data['tracks']) >= 3:
                        final_playlists[name] = data
                        used_tracks.update(data['tracks'])
                final_playlists = _finalize_playlists(final_playlists, features)
                self._calculate_playlist_stats(final_playlists, features)
                return final_playlists

            if self.playlist_method == 'time':
                # Only time-based playlists
                time_playlists = self.time_scheduler.generate_time_based_playlists(features)
                for name, data in time_playlists.items():
                    if len(data['tracks']) >= 3:
                        final_playlists[name] = data
                final_playlists = _finalize_playlists(final_playlists, features)
                self._calculate_playlist_stats(final_playlists, features)
                return final_playlists

            if self.playlist_method == 'kmeans':
                # Only kmeans playlists
                kmeans_playlists = self.kmeans_generator.generate(features, num_playlists)
                for name, data in kmeans_playlists.items():
                    if len(data['tracks']) >= 3:
                        final_playlists[name] = data
                final_playlists = _finalize_playlists(final_playlists, features)
                self._calculate_playlist_stats(final_playlists, features)
                return final_playlists

            if self.playlist_method == 'cache':
                # Only cache-based playlists
                cache_playlists = self.cache_generator.generate(features)
                for name, data in cache_playlists.items():
                    if len(data['tracks']) >= 3:
                        final_playlists[name] = data
                final_playlists = _finalize_playlists(final_playlists, features)
                self._calculate_playlist_stats(final_playlists, features)
                return final_playlists

            # Fallback: use feature group generator
            fg_playlists = self.feature_group_generator.generate(features)
            for name, data in fg_playlists.items():
                if len(data['tracks']) >= 3:
                    final_playlists[name] = data
                    used_tracks.update(data['tracks'])
            final_playlists = _finalize_playlists(final_playlists, features)
            self._calculate_playlist_stats(final_playlists, features)
            return final_playlists

        except Exception as e:
            logger.error(f"Error generating playlists: {str(e)}")
            return {}

    def _optimize_playlists(self, playlists: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize playlists by merging similar ones and ensuring good distribution"""
        try:
            optimized = {}
            merged_tracks = set()

            # Sort playlists by number of tracks (largest first)
            sorted_playlists = sorted(
                playlists.items(),
                key=lambda x: len(x[1].get('tracks', [])),
                reverse=True
            )

            for name, data in sorted_playlists:
                tracks = data.get('tracks', [])
                if not tracks:
                    continue

                # Skip if most tracks are already in other playlists
                overlap = len(set(tracks) & merged_tracks) / len(tracks)
                if overlap > 0.7:  # More than 70% overlap
                    continue

                # Find similar playlists for potential merging
                similar_playlists = self._find_similar_playlists(
                    name, data, optimized, threshold=0.3
                )

                if similar_playlists:
                    # Merge with most similar playlist
                    target_name = similar_playlists[0][0]
                    optimized[target_name]['tracks'].extend(tracks)
                    # Deduplicate tracks
                    optimized[target_name]['tracks'] = list(set(optimized[target_name]['tracks']))
                else:
                    # Create new playlist
                    optimized[name] = data

                merged_tracks.update(tracks)

            return self._balance_playlists(optimized)

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
                overlap = len(existing_tracks & current_tracks) / len(current_tracks)
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
                    chunks = [tracks[i:i + max_size] for i in range(0, len(tracks), max_size)]
                    for i, chunk in enumerate(chunks, 1):
                        new_name = f"{name}_Part{i}" if len(chunks) > 1 else name
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
                unassigned_tracks = list(set(unassigned_tracks))  # Remove duplicates
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

                # Calculate basic stats
                stats = {
                    'track_count': len(track_features),
                    'total_duration': sum(f.get('duration', 0) for f in track_features),
                    'avg_bpm': np.mean([f.get('bpm', 0) for f in track_features]),
                    'avg_danceability': np.mean([f.get('danceability', 0) for f in track_features]),
                    'key_distribution': self._get_key_distribution(track_features)
                }

                self.playlist_stats[name] = stats

        except Exception as e:
            logger.error(f"Error calculating playlist stats: {str(e)}")

    def _get_key_distribution(self, features: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of musical keys in playlist"""
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        distribution = defaultdict(int)

        for feature in features:
            key_idx = feature.get('key', -1)
            if 0 <= key_idx < len(keys):
                key_name = keys[key_idx]
                scale = 'Major' if feature.get('scale', 0) == 1 else 'Minor'
                distribution[f"{key_name} {scale}"] += 1

        return dict(distribution)

    def get_playlist_stats(self, playlist_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific playlist or all playlists"""
        if playlist_name:
            return self.playlist_stats.get(playlist_name, {})
        return dict(self.playlist_stats)

    def get_recommendations(self, playlist_name: str) -> List[str]:
        """Get track recommendations for a playlist based on its characteristics"""
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