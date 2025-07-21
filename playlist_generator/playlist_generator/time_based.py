# playlist_generator/time_based.py
import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TimeBasedScheduler:
    def __init__(self):
        self.time_slots = {
            'Morning': (6, 12),     # 6am-12pm
            'Afternoon': (12, 18),  # 12pm-6pm
            'Evening': (18, 22),    # 6pm-10pm
            'Late_Night': (22, 6)   # 10pm-6am
        }
        self.feature_rules = {
            'Morning': {
                'min_bpm': 90, 'max_bpm': 120,
                'min_danceability': 0.4, 'max_danceability': 0.8,
                'min_centroid': 800, 'max_centroid': 3500,
                'min_loudness': -18,
                'compatible_keys': [0, 2, 4, 7, 9],  # C, D, E, G, A
                'min_duration': 120, 'max_duration': 300,
                'required_scale': 'major',
                'min_onset_rate': 0.5, 'min_zcr': 0.05
            },
            'Afternoon': {
                'min_bpm': 100, 'max_bpm': 130,
                'min_danceability': 0.6, 'max_danceability': 0.9,
                'min_centroid': 1500,
                'compatible_keys': [2, 5, 7, 10],  # D, F, G, A#
                'min_duration': 90, 'max_onset_rate': 2.0
            },
            'Evening': {
                'min_bpm': 80, 'max_bpm': 110,
                'max_danceability': 0.7,
                'min_centroid': 1000, 'max_centroid': 5000,
                'min_loudness': -15,
                'required_scale': 'minor',
                'max_zcr': 0.2
            },
            'Late_Night': {
                'max_bpm': 90, 'max_danceability': 0.4,
                'max_centroid': 2000, 'min_loudness': -25,
                'compatible_keys': [0, 3, 5, 8],  # C, D#, F, G#
                'min_duration': 180, 'max_duration': 600,
                'max_onset_rate': 0.8
            }
        }

    def get_current_time_slot(self):
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

    def filter_tracks_for_slot(self, features_list, slot_name):
        rules = self.feature_rules.get(slot_name, {})
        filtered = []
        
        for track in features_list:
            if not track:
                continue
            
            valid = True
            # Helper to safely get values
            get_val = lambda key, default: track.get(key, default) or default
            
            # Numeric validations
            for feature in ['bpm', 'danceability', 'centroid', 'loudness', 
                            'duration', 'onset_rate', 'zcr']:
                val = get_val(feature, 0)
                
                if f'min_{feature}' in rules and val < rules[f'min_{feature}']:
                    valid = False
                if valid and f'max_{feature}' in rules and val > rules[f'max_{feature}']:
                    valid = False
                    
            # Key validation
            if valid and 'compatible_keys' in rules:
                key = get_val('key', -1)
                if key >= 0 and key not in rules['compatible_keys']:
                    valid = False
                        
            # Scale validation
            if valid and 'required_scale' in rules:
                scale = get_val('scale', 0)
                req_scale = rules['required_scale']
                if (req_scale == 'major' and scale != 1) or (req_scale == 'minor' and scale != 0):
                    valid = False
                    
            if valid:
                filtered.append(track)
                
        logger.debug(f"Filtered {len(filtered)} tracks for {slot_name} time slot")
        return filtered

    def generate_time_based_playlist(self, features_list, slot_name=None):
        slot = slot_name or self.get_current_time_slot()
        return self.filter_tracks_for_slot(features_list, slot)

    def generate_time_based_playlists(self, features_list):
        playlists = {}
        for slot_name in self.time_slots:
            tracks = self.generate_time_based_playlist(features_list, slot_name)
            playlists[f"TimeSlot_{slot_name}"] = {
                'tracks': [t['filepath'] for t in tracks],
                'features': {'type': 'time_based', 'slot': slot_name}
            }
        return playlists