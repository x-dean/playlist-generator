# app/playlist_generator/time_based.py
import datetime
import logging
import numpy as np
from typing import Dict, List, Any
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

class TimeBasedScheduler:
    def __init__(self):
        # Define time slots with more granular divisions
        self.time_slots = {
            'Early_Morning': (5, 8),    # 5am-8am: Gentle wake up
            'Morning': (8, 11),         # 8am-11am: Productive morning
            'Midday': (11, 14),         # 11am-2pm: Lunch and early afternoon
            'Afternoon': (14, 17),      # 2pm-5pm: Afternoon work
            'Evening': (17, 20),        # 5pm-8pm: Evening relaxation
            'Night': (20, 23),          # 8pm-11pm: Night time
            'Late_Night': (23, 5)       # 11pm-5am: Late night/early morning
        }

        # Define feature rules for each time slot
        self.feature_rules = {
            'Early_Morning': {
                'min_bpm': 50, 'max_bpm': 100,  # Wider BPM range
                'max_danceability': 0.6,         # Only limit max danceability
                'max_centroid': 3000,            # Only limit max brightness
                'max_loudness': -10,             # Only limit max loudness
                'min_duration': 60,              # Shorter minimum duration
                'max_onset_rate': 1.0,           # More permissive onset rate
                'description': "Gentle, calming tracks for a peaceful start to your day"
            },
            'Morning': {
                'min_bpm': 80, 'max_bpm': 130,   # Wider BPM range
                'min_danceability': 0.3,          # Lower minimum danceability
                'min_centroid': 800,              # Lower minimum brightness
                'min_loudness': -25,              # More permissive loudness
                'min_duration': 60,               # Shorter minimum duration
                'description': "Upbeat and energizing music to boost morning productivity"
            },
            'Midday': {
                'min_bpm': 90, 'max_bpm': 140,    # Wider BPM range
                'min_danceability': 0.4,           # Lower minimum danceability
                'min_centroid': 1000,              # Lower minimum brightness
                'min_loudness': -20,               # More permissive loudness
                'description': "Energetic tracks to keep you going through lunch and early afternoon"
            },
            'Afternoon': {
                'min_bpm': 75, 'max_bpm': 135,    # Wider BPM range
                'min_danceability': 0.3,           # Lower minimum danceability
                'min_centroid': 1000,              # Lower minimum brightness
                'min_loudness': -25,               # More permissive loudness
                'description': "Balanced music to maintain focus and energy in the afternoon"
            },
            'Evening': {
                'min_bpm': 60, 'max_bpm': 120,    # Wider BPM range
                'max_danceability': 0.7,           # Higher max danceability
                'max_centroid': 4000,              # Higher max brightness
                'min_loudness': -30,               # More permissive loudness
                'description': "Relaxing music to help you unwind after work"
            },
            'Night': {
                'min_bpm': 55, 'max_bpm': 110,    # Wider BPM range
                'max_danceability': 0.6,           # Higher max danceability
                'max_centroid': 3500,              # Higher max brightness
                'min_loudness': -30,               # More permissive loudness
                'min_duration': 120,               # Reasonable minimum duration
                'description': "Calming evening music for relaxation and quiet activities"
            },
            'Late_Night': {
                'max_bpm': 95,                     # Higher max BPM
                'max_danceability': 0.5,           # Higher max danceability
                'max_centroid': 3000,              # Higher max brightness
                'min_loudness': -35,               # More permissive loudness
                'min_duration': 120,               # Reasonable minimum duration
                'description': "Ambient and peaceful music for late night listening"
            }
        }

    def get_current_time_slot(self) -> str:
        """Get the current time slot based on time of day"""
        now = datetime.datetime.now().time()
        current_hour = now.hour
        
        for slot, (start, end) in self.time_slots.items():
            if start < end:
                if start <= current_hour < end:
                    return slot
            else:  # Overnight slot
                if current_hour >= start or current_hour < end:
                    return slot
        return 'Afternoon'  # Default

    def filter_tracks_for_slot(self, features_list: List[Dict[str, Any]], slot_name: str) -> List[Dict[str, Any]]:
        """Filter tracks based on time slot rules"""
        rules = self.feature_rules.get(slot_name, {})
        filtered = []
        
        for track in features_list:
            if not track:
                continue
            
            valid = True
            # Helper to safely get values with normalization
            def get_val(key: str, default: float = 0) -> float:
                val = track.get(key, default)
                if val is None:
                    return default
                
                # Normalize certain features
                if key == 'danceability':
                    return min(1.0, max(0.0, float(val)))
                elif key == 'loudness':
                    return min(0.0, max(-60.0, float(val)))
                elif key == 'centroid':
                    return min(10000.0, max(0.0, float(val)))
                
                return float(val)
            
            # Numeric validations with min/max checks
            for feature in ['bpm', 'danceability', 'centroid', 'loudness', 
                          'duration', 'onset_rate', 'zcr']:
                val = get_val(feature)
                
                if f'min_{feature}' in rules and val < rules[f'min_{feature}']:
                    valid = False
                    break
                if valid and f'max_{feature}' in rules and val > rules[f'max_{feature}']:
                    valid = False
                    break
                    
            if not valid:
                continue
                    
            # Key validation
            if 'compatible_keys' in rules:
                key = int(track.get('key', -1))
                if key >= 0 and key not in rules['compatible_keys']:
                    valid = False
                    continue
                        
            # Scale validation
            if 'required_scale' in rules:
                scale = int(track.get('scale', 0))
                req_scale = rules['required_scale']
                if (req_scale == 'major' and scale != 1) or (req_scale == 'minor' and scale != 0):
                    valid = False
                    continue
                    
            if valid:
                filtered.append(track)
                
        logger.debug(f"Filtered {len(filtered)} tracks for {slot_name} time slot")
        return filtered

    def generate_time_based_playlist(self, features_list: List[Dict[str, Any]], slot_name: str = None) -> List[Dict[str, Any]]:
        """Generate a playlist for a specific time slot"""
        slot = slot_name or self.get_current_time_slot()
        return self.filter_tracks_for_slot(features_list, slot)

    def _interleave_by_artist(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Interleave tracks by artist to maximize variety. Fallback to shuffle if artist missing."""
        # Check if artist metadata is present for most tracks
        has_artist = any('artist' in t and t['artist'] for t in tracks)
        if not has_artist:
            random.shuffle(tracks)
            return tracks
        # Group tracks by artist
        groups = defaultdict(list)
        for t in tracks:
            artist = t.get('artist', 'Unknown')
            groups[artist].append(t)
        # Shuffle within each artist group
        for group in groups.values():
            random.shuffle(group)
        # Interleave tracks
        interleaved = []
        group_lists = list(groups.values())
        while any(group_lists):
            for group in group_lists:
                if group:
                    interleaved.append(group.pop(0))
        return interleaved

    def generate_time_based_playlists(self, features_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate playlists for all time slots, splitting if total duration exceeds slot length"""
        playlists = {}
        for slot_name in self.time_slots:
            tracks = self.generate_time_based_playlist(features_list, slot_name)
            if not tracks:
                continue
            # ADVANCED SHUFFLE: interleave by artist if possible, else random
            tracks = self._interleave_by_artist(tracks)
            # Calculate slot duration in seconds
            start, end = self.time_slots[slot_name]
            if start < end:
                slot_hours = end - start
            else:
                slot_hours = (24 - start) + end
            slot_duration_sec = slot_hours * 3600
            # Split tracks into sub-playlists if needed
            sub_playlists = []
            current = []
            current_duration = 0
            for track in tracks:
                track_duration = float(track.get('duration', 0))
                if current_duration + track_duration > slot_duration_sec and current:
                    sub_playlists.append(current)
                    current = []
                    current_duration = 0
                current.append(track)
                current_duration += track_duration
            if current:
                sub_playlists.append(current)
            # Add sub-playlists to output
            for i, sub in enumerate(sub_playlists, 1):
                name = f"TimeSlot_{slot_name}" if len(sub_playlists) == 1 else f"TimeSlot_{slot_name}_Part{i}"
                playlists[name] = {
                    'tracks': [t['filepath'] for t in sub],
                    'features': {
                        'type': 'time_based',
                        'slot': slot_name,
                        'rules': self.feature_rules[slot_name]
                    },
                    'description': self.feature_rules[slot_name].get('description', '')
                }
        return playlists