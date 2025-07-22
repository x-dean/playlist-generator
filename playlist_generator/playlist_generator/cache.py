# playlist_generator/cache.py
import sqlite3
import logging
import traceback
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class CacheBasedGenerator:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.playlist_history = defaultdict(list)

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

    def _get_category(self, value: float, ranges: Dict[str, tuple]) -> tuple:
        """Get category and description for a value from defined ranges"""
        for name, (min_val, max_val, desc) in ranges.items():
            if min_val <= value < max_val:
                return name, desc
        return list(ranges.keys())[-1], ranges[list(ranges.keys())[-1]][2]

    def _get_combined_energy(self, track: Dict[str, Any]) -> float:
        """Calculate combined energy score from multiple features"""
        danceability = float(track.get('danceability', 0))
        loudness = float(track.get('loudness', -60))
        onset_rate = float(track.get('onset_rate', 0))
        
        # Normalize loudness from dB to 0-1 range
        loudness_norm = (loudness + 60) / 60  # -60dB -> 0, 0dB -> 1
        
        # Combine features with weights
        return (
            danceability * 0.4 +
            loudness_norm * 0.3 +
            min(1.0, onset_rate / 4) * 0.3  # Normalize onset rate
        )

    def _generate_playlist_name(self, features: Dict[str, Any]) -> str:
        """Generate descriptive playlist name based on musical features"""
        bpm = float(features.get('bpm', 0))
        energy = self._get_combined_energy(features)
        centroid = float(features.get('centroid', 0))
        
        bpm_category, _ = self._get_category(bpm, self.bpm_ranges)
        energy_category, _ = self._get_category(energy, self.energy_levels)
        mood_category, _ = self._get_category(centroid, self.mood_ranges)
        
        return f"{bpm_category}_{energy_category}_{mood_category}"

    def _generate_description(self, features: Dict[str, Any]) -> str:
        """Generate human-readable description based on musical features"""
        bpm = float(features.get('bpm', 0))
        energy = self._get_combined_energy(features)
        centroid = float(features.get('centroid', 0))
        
        _, bpm_desc = self._get_category(bpm, self.bpm_ranges)
        _, energy_desc = self._get_category(energy, self.energy_levels)
        _, mood_desc = self._get_category(centroid, self.mood_ranges)
        
        return f"{bpm_desc} with {energy_desc}. {mood_desc} characteristics."

    def generate(self, features_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """Generate playlists from either provided features or database"""
        if features_list:
            return self._generate_from_features(features_list)
        return self._generate_from_db()

    def _generate_from_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate playlists from provided feature list"""
        playlists = {}
        for feature in features_list:
            if not feature or 'filepath' not in feature:
                continue

            try:
                # Calculate normalized features
                track_features = {
                    'filepath': feature['filepath'],
                    'bpm': float(feature.get('bpm', 0)),
                    'centroid': float(feature.get('centroid', 0)),
                    'danceability': min(1.0, max(0.0, float(feature.get('danceability', 0)))),
                    'loudness': float(feature.get('loudness', -30)),
                    'onset_rate': float(feature.get('onset_rate', 0)),
                    'key': int(feature.get('key', -1)),
                    'scale': int(feature.get('scale', 0))
                }

                # Generate playlist name and get category descriptions
                name = self._generate_playlist_name(track_features)
                description = self._generate_description(track_features)

                if name not in playlists:
                    playlists[name] = {
                        'tracks': [],
                        'features': track_features,
                        'description': description
                    }

                playlists[name]['tracks'].append(track_features['filepath'])

            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping track due to invalid features: {str(e)}")
                continue

        return self._merge_playlists(playlists)

    def _generate_from_db(self) -> Dict[str, Dict[str, Any]]:
        """Generate playlists from database"""
        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()

            # Verify and update schema if needed
            cursor.execute("PRAGMA table_info(audio_features)")
            existing_columns = {row[1]: row[2] for row in cursor.fetchall()}
            required_columns = {
                'loudness': 'REAL DEFAULT 0',
                'danceability': 'REAL DEFAULT 0',
                'key': 'INTEGER DEFAULT -1',
                'scale': 'INTEGER DEFAULT 0',
                'onset_rate': 'REAL DEFAULT 0',
                'zcr': 'REAL DEFAULT 0'
            }
            
            for col, col_type in required_columns.items():
                if col not in existing_columns:
                    logger.info(f"Adding missing column {col} to database")
                    conn.execute(f"ALTER TABLE audio_features ADD COLUMN {col} {col_type}")

            # Get all tracks with features
            cursor.execute("""
                SELECT file_path, bpm, centroid, danceability, loudness, 
                       onset_rate, key, scale
                FROM audio_features
                WHERE bpm IS NOT NULL 
                AND centroid IS NOT NULL 
                AND danceability IS NOT NULL
            """)
            
            # Convert to list of dictionaries
            features_list = [
                {
                    'filepath': row[0],
                    'bpm': row[1],
                    'centroid': row[2],
                    'danceability': row[3],
                    'loudness': row[4],
                    'onset_rate': row[5],
                    'key': row[6],
                    'scale': row[7]
                }
                for row in cursor.fetchall()
            ]

            return self._generate_from_features(features_list)

        except Exception as e:
            logger.error(f"Database playlist generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
        finally:
            conn.close()

    def _merge_playlists(self, playlists: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Merge and balance playlists"""
        min_size = int(os.getenv('MIN_PLAYLIST_SIZE', 20))
        max_size = int(os.getenv('MAX_PLAYLIST_SIZE', 100))
        final_playlists = {}

        # Sort playlists by size (largest first)
        sorted_playlists = sorted(
            playlists.items(),
            key=lambda x: len(x[1]['tracks']),
            reverse=True
        )

        # First pass: keep substantial playlists and split if needed
        for name, data in sorted_playlists:
            tracks = data['tracks']
            
            # Skip empty playlists
            if not tracks:
                continue

            # Split large playlists by mood
            if len(tracks) > max_size:
                # Get mood from original features
                mood_groups = {}
                for track in tracks:
                    # Get mood from track features
                    track_features = data['features']
                    centroid = float(track_features.get('centroid', 0))
                    mood = self._get_mood_category(centroid)
                    
                    if mood not in mood_groups:
                        mood_groups[mood] = []
                    mood_groups[mood].append(track)

                # Create playlists for each mood group
                for mood, mood_tracks in mood_groups.items():
                    if len(mood_tracks) >= min_size:
                        mood_name = f"{name}_{mood}"
                        final_playlists[mood_name] = {
                            'tracks': mood_tracks,
                            'features': data['features'].copy(),
                            'description': f"{data['description']} ({mood} mood)"
                        }
            
            # Keep medium-sized playlists as is
            elif len(tracks) >= min_size:
                final_playlists[name] = data

        # Second pass: try to merge small playlists
        remaining_playlists = {
            name: data for name, data in sorted_playlists
            if len(data['tracks']) < min_size
        }

        for name, data in remaining_playlists.items():
            tracks = data['tracks']
            if not tracks:
                continue

            # Find best matching large playlist
            best_match = None
            best_score = 0

            for final_name, final_data in final_playlists.items():
                if len(final_data['tracks']) + len(tracks) > max_size:
                    continue

                score = self._calculate_merge_score(data, final_data)
                if score > best_score:
                    best_score = score
                    best_match = final_name

            # Merge if good match found
            if best_match and best_score >= 0.5:
                final_playlists[best_match]['tracks'].extend(tracks)
            else:
                # Create new playlist if enough tracks
                if len(tracks) >= min_size // 2:
                    final_playlists[name] = data
                else:
                    # Add to mixed collection
                    mixed_name = "Mixed_Collection"
                    if mixed_name not in final_playlists:
                        final_playlists[mixed_name] = {
                            'tracks': [],
                            'features': {'type': 'mixed'},
                            'description': "A diverse collection of tracks that don't fit other categories"
                        }
                    final_playlists[mixed_name]['tracks'].extend(tracks)

        return final_playlists

    def _get_mood_category(self, centroid: float) -> str:
        """Get mood category based on spectral centroid"""
        if centroid < 500:
            return "Dark"
        elif centroid < 1500:
            return "Warm"
        elif centroid < 3000:
            return "Balanced"
        elif centroid < 6000:
            return "Bright"
        return "Crisp"

    def _calculate_merge_score(self, playlist1: Dict[str, Any], playlist2: Dict[str, Any]) -> float:
        """Calculate merge compatibility score between two playlists"""
        features1 = playlist1['features']
        features2 = playlist2['features']

        # Compare BPM (30% weight)
        bpm1 = float(features1.get('bpm', 0))
        bpm2 = float(features2.get('bpm', 0))
        bpm_diff = 1.0 - (abs(bpm1 - bpm2) / max(bpm1, bpm2))
        
        # Compare energy (30% weight)
        energy1 = self._get_combined_energy(features1)
        energy2 = self._get_combined_energy(features2)
        energy_diff = 1.0 - abs(energy1 - energy2)
        
        # Compare spectral characteristics (40% weight)
        centroid1 = float(features1.get('centroid', 0))
        centroid2 = float(features2.get('centroid', 0))
        centroid_diff = 1.0 - (abs(centroid1 - centroid2) / max(centroid1, centroid2))

        # Calculate weighted score
        return (bpm_diff * 0.3 + energy_diff * 0.3 + centroid_diff * 0.4)
