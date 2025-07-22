# playlist_generator/time_based.py
import datetime
import logging
import numpy as np
from typing import Dict, List, Any

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
                'min_bpm': 60, 'max_bpm': 90,
                'min_danceability': 0.2, 'max_danceability': 0.5,
                'min_centroid': 500, 'max_centroid': 2000,
                'min_loudness': -25, 'max_loudness': -12,
                'compatible_keys': [0, 4, 7],  # C, E, G (major keys)
                'min_duration': 120, 'max_duration': 300,
                'required_scale': 'major',
                'max_onset_rate': 0.8,
                'max_zcr': 0.15,
                'description': "Gentle, calming tracks for a peaceful start to your day"
            },
            'Morning': {
                'min_bpm': 90, 'max_bpm': 120,
                'min_danceability': 0.4, 'max_danceability': 0.7,
                'min_centroid': 1000, 'max_centroid': 4000,
                'min_loudness': -18, 'max_loudness': -8,
                'compatible_keys': [0, 2, 4, 7, 9],  # C, D, E, G, A
                'min_duration': 120, 'max_duration': 300,
                'required_scale': 'major',
                'min_onset_rate': 0.5,
                'description': "Upbeat and energizing music to boost morning productivity"
            },
            'Midday': {
                'min_bpm': 100, 'max_bpm': 130,
                'min_danceability': 0.5, 'max_danceability': 0.8,
                'min_centroid': 2000, 'max_centroid': 6000,
                'min_loudness': -15, 'max_loudness': -5,
                'compatible_keys': [2, 5, 7, 9],  # D, F, G, A
                'min_onset_rate': 0.6,
                'description': "Energetic tracks to keep you going through lunch and early afternoon"
            },
            'Afternoon': {
                'min_bpm': 85, 'max_bpm': 125,
                'min_danceability': 0.4, 'max_danceability': 0.7,
                'min_centroid': 1500, 'max_centroid': 5000,
                'min_loudness': -20, 'max_loudness': -8,
                'compatible_keys': [0, 4, 7, 11],  # C, E, G, B
                'max_onset_rate': 1.5,
                'description': "Balanced music to maintain focus and energy in the afternoon"
            },
            'Evening': {
                'min_bpm': 70, 'max_bpm': 110,
                'max_danceability': 0.6,
                'min_centroid': 800, 'max_centroid': 3000,
                'min_loudness': -22, 'max_loudness': -10,
                'required_scale': 'minor',
                'max_zcr': 0.2,
                'description': "Relaxing music to help you unwind after work"
            },
            'Night': {
                'min_bpm': 65, 'max_bpm': 100,
                'max_danceability': 0.5,
                'min_centroid': 500, 'max_centroid': 2500,
                'min_loudness': -25, 'max_loudness': -12,
                'compatible_keys': [0, 3, 5, 8],  # C, D#, F, G#
                'min_duration': 180,
                'max_onset_rate': 0.7,
                'description': "Calming evening music for relaxation and quiet activities"
            },
            'Late_Night': {
                'max_bpm': 85,
                'max_danceability': 0.4,
                'max_centroid': 2000,
                'min_loudness': -30, 'max_loudness': -15,
                'compatible_keys': [0, 3, 5, 8],  # C, D#, F, G#
                'min_duration': 180, 'max_duration': 600,
                'max_onset_rate': 0.5,
                'max_zcr': 0.1,
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

    def generate_time_based_playlists(self, features_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate playlists for all time slots"""
        playlists = {}
        for slot_name in self.time_slots:
            tracks = self.generate_time_based_playlist(features_list, slot_name)
            if tracks:  # Only create playlist if there are matching tracks
                playlists[f"TimeSlot_{slot_name}"] = {
                    'tracks': [t['filepath'] for t in tracks],
                    'features': {
                        'type': 'time_based',
                        'slot': slot_name,
                        'rules': self.feature_rules[slot_name]
                    },
                    'description': self.feature_rules[slot_name].get('description', '')
                }
        return playlists