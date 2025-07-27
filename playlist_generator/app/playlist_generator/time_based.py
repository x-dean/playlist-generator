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
        logger.debug("Initializing TimeBasedScheduler")

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
        logger.debug(
            f"TimeBasedScheduler initialized with {len(self.time_slots)} time slots")

    def get_current_time_slot(self) -> str:
        """Get the current time slot based on time of day"""
        logger.debug("Getting current time slot")

        try:
            now = datetime.datetime.now().time()
            current_hour = now.hour
            logger.debug(f"Current hour: {current_hour}")

            for slot, (start, end) in self.time_slots.items():
                if start < end:
                    if start <= current_hour < end:
                        logger.debug(f"Current time slot: {slot}")
                        return slot
                else:  # Overnight slot
                    if current_hour >= start or current_hour < end:
                        logger.debug(f"Current time slot: {slot}")
                        return slot

            logger.debug(
                "No matching time slot found, defaulting to Afternoon")
            return 'Afternoon'  # Default

        except Exception as e:
            logger.error(f"Error getting current time slot: {str(e)}")
            return 'Afternoon'

    def filter_tracks_for_slot(self, features_list: List[Dict[str, Any]], slot_name: str) -> List[Dict[str, Any]]:
        """Filter tracks based on time slot rules"""
        logger.debug(f"Filtering tracks for time slot: {slot_name}")

        try:
            rules = self.feature_rules.get(slot_name, {})
            logger.debug(f"Using rules for {slot_name}: {rules}")

            filtered = []
            total_tracks = len(features_list)
            passed_tracks = 0
            failed_tracks = 0

            for track in features_list:
                if not track:
                    failed_tracks += 1
                    logger.debug("Skipping empty track")
                    continue

                try:
                    def get_val(key: str, default: float = 0) -> float:
                        """Get track value with fallback"""
                        val = track.get(key, default)
                        if val is None:
                            return default
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return default

                    # Check BPM rules
                    bpm = get_val('bpm', 120)
                    # Skip tracks with failed BPM extraction (-1.0 marker)
                    if bpm == -1.0:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} skipped due to failed BPM extraction")
                        continue
                    if 'min_bpm' in rules and bpm < rules['min_bpm']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed BPM min check: {bpm} < {rules['min_bpm']}")
                        continue
                    if 'max_bpm' in rules and bpm > rules['max_bpm']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed BPM max check: {bpm} > {rules['max_bpm']}")
                        continue

                    # Check danceability rules
                    danceability = get_val('danceability', 0.5)
                    if 'min_danceability' in rules and danceability < rules['min_danceability']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed danceability min check: {danceability} < {rules['min_danceability']}")
                        continue
                    if 'max_danceability' in rules and danceability > rules['max_danceability']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed danceability max check: {danceability} > {rules['max_danceability']}")
                        continue

                    # Check centroid rules
                    centroid = get_val('centroid', 2000)
                    if 'min_centroid' in rules and centroid < rules['min_centroid']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed centroid min check: {centroid} < {rules['min_centroid']}")
                        continue
                    if 'max_centroid' in rules and centroid > rules['max_centroid']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed centroid max check: {centroid} > {rules['max_centroid']}")
                        continue

                    # Check loudness rules
                    loudness = get_val('loudness', -20)
                    if 'min_loudness' in rules and loudness < rules['min_loudness']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed loudness min check: {loudness} < {rules['min_loudness']}")
                        continue
                    if 'max_loudness' in rules and loudness > rules['max_loudness']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed loudness max check: {loudness} > {rules['max_loudness']}")
                        continue

                    # Check duration rules
                    duration = get_val('duration', 180)
                    if 'min_duration' in rules and duration < rules['min_duration']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed duration min check: {duration} < {rules['min_duration']}")
                        continue
                    if 'max_duration' in rules and duration > rules['max_duration']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed duration max check: {duration} > {rules['max_duration']}")
                        continue

                    # Check onset rate rules
                    onset_rate = get_val('onset_rate', 0.5)
                    if 'min_onset_rate' in rules and onset_rate < rules['min_onset_rate']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed onset rate min check: {onset_rate} < {rules['min_onset_rate']}")
                        continue
                    if 'max_onset_rate' in rules and onset_rate > rules['max_onset_rate']:
                        failed_tracks += 1
                        logger.debug(
                            f"Track {track.get('filepath', 'unknown')} failed onset rate max check: {onset_rate} > {rules['max_onset_rate']}")
                        continue

                    # Track passed all filters
                    filtered.append(track)
                    passed_tracks += 1
                    logger.debug(
                        f"Track {track.get('filepath', 'unknown')} passed all filters for {slot_name}")

                except Exception as e:
                    failed_tracks += 1
                    logger.warning(
                        f"Error filtering track {track.get('filepath', 'unknown')}: {str(e)}")

            logger.info(
                f"Time slot filtering complete for {slot_name}: {passed_tracks}/{total_tracks} tracks passed, {failed_tracks} failed")
            return filtered

        except Exception as e:
            logger.error(
                f"Error filtering tracks for time slot {slot_name}: {str(e)}")
            import traceback
            logger.error(
                f"Time slot filtering error traceback: {traceback.format_exc()}")
            return []

    def generate_time_based_playlist(self, features_list: List[Dict[str, Any]], slot_name: str = None) -> List[Dict[str, Any]]:
        """Generate a playlist for a specific time slot"""
        logger.debug(f"Generating time-based playlist for slot: {slot_name}")

        try:
            if not slot_name:
                slot_name = self.get_current_time_slot()
                logger.debug(f"Using current time slot: {slot_name}")

            # Filter tracks for the time slot
            filtered_tracks = self.filter_tracks_for_slot(
                features_list, slot_name)

            if not filtered_tracks:
                logger.warning(
                    f"No tracks passed filters for time slot: {slot_name}")
                return []

            # Interleave tracks by artist to avoid repetition
            interleaved_tracks = self._interleave_by_artist(filtered_tracks)

            logger.info(
                f"Generated time-based playlist for {slot_name}: {len(interleaved_tracks)} tracks")
            return interleaved_tracks

        except Exception as e:
            logger.error(
                f"Error generating time-based playlist for {slot_name}: {str(e)}")
            import traceback
            logger.error(
                f"Time-based playlist generation error traceback: {traceback.format_exc()}")
            return []

    def _interleave_by_artist(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Interleave tracks by artist to avoid repetition"""
        logger.debug(f"Interleaving {len(tracks)} tracks by artist")

        try:
            # Group tracks by artist
            artist_groups = defaultdict(list)
            for track in tracks:
                metadata = track.get('metadata', {})
                artist = metadata.get('artist', 'Unknown')
                artist_groups[artist].append(track)

            logger.debug(f"Grouped tracks by {len(artist_groups)} artists")

            # Interleave tracks from different artists
            interleaved = []
            max_tracks_per_artist = max(
                len(tracks) for tracks in artist_groups.values()) if artist_groups else 0

            for i in range(max_tracks_per_artist):
                for artist, artist_tracks in artist_groups.items():
                    if i < len(artist_tracks):
                        interleaved.append(artist_tracks[i])
                        logger.debug(
                            f"Added track {i+1} from artist '{artist}'")

            logger.debug(f"Interleaving complete: {len(interleaved)} tracks")
            return interleaved

        except Exception as e:
            logger.error(f"Error interleaving tracks by artist: {str(e)}")
            import traceback
            logger.error(
                f"Interleaving error traceback: {traceback.format_exc()}")
            return tracks

    def generate_time_based_playlists(self, features_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate playlists for all time slots"""
        logger.debug(
            f"Generating time-based playlists for {len(features_list)} tracks")

        try:
            playlists = {}

            for slot_name in self.time_slots.keys():
                logger.debug(f"Generating playlist for time slot: {slot_name}")

                # Filter tracks for this time slot
                filtered_tracks = self.filter_tracks_for_slot(
                    features_list, slot_name)

                if filtered_tracks:
                    # Interleave tracks
                    interleaved_tracks = self._interleave_by_artist(
                        filtered_tracks)

                    # Create playlist
                    playlist_name = f"Time_{slot_name}"
                    playlists[playlist_name] = {
                        'tracks': [track['filepath'] for track in interleaved_tracks],
                        'features': {
                            'type': 'time_based',
                            'time_slot': slot_name,
                            'description': self.feature_rules[slot_name]['description'],
                            'track_count': len(interleaved_tracks)
                        }
                    }
                    logger.debug(
                        f"Created playlist '{playlist_name}' with {len(interleaved_tracks)} tracks")
                else:
                    logger.warning(
                        f"No tracks available for time slot: {slot_name}")

            logger.info(
                f"Time-based playlist generation complete: {len(playlists)} playlists created")
            logger.debug(f"Generated playlists: {list(playlists.keys())}")

            return playlists

        except Exception as e:
            logger.error(f"Error generating time-based playlists: {str(e)}")
            import traceback
            logger.error(
                f"Time-based playlists generation error traceback: {traceback.format_exc()}")
            return {}
