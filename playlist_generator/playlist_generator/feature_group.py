import os
import sqlite3
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class FeatureGroupPlaylistGenerator:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file

    def sanitize_filename(self, name: str) -> str:
        name = re.sub(r'[^\w\-_]', '_', name)
        return re.sub(r'_+', '_', name).strip('_')

    def _sanitize_file_name(self, name: str) -> str:
        import re
        name = re.sub(r'[^A-Za-z0-9_-]+', '_', name)
        name = re.sub(r'_+', '_', name)
        return name.strip('_')

    def generate(self, features_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        # Helper mapping for mood similarity
        centroid_mapping = {
            'Warm': 0,
            'Mellow': 1,
            'Balanced': 2,
            'Bright': 3,
            'Crisp': 4
        }
        try:
            # Define groupers
            def get_energy_group(danceability):
                if danceability < 0.3: return 'Chill'
                if danceability < 0.5: return 'Mellow'
                if danceability < 0.7: return 'Groovy'
                if danceability < 0.85: return 'Energetic'
                return 'Intense'

            def get_bpm_group(bpm):
                if bpm < 70: return 'Slow'
                if bpm < 100: return 'Medium'
                if bpm < 130: return 'Upbeat'
                if bpm < 160: return 'Fast'
                return 'VeryFast'

            def get_mood_group(centroid):
                if centroid < 500: return 'Warm'
                if centroid < 1500: return 'Mellow'
                if centroid < 3000: return 'Balanced'
                if centroid < 6000: return 'Bright'
                return 'Crisp'

            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            playlists = {}
            for f in features_list:
                if not f or 'filepath' not in f:
                    continue
                bpm = f.get('bpm', 0)
                centroid = f.get('centroid', 0)
                danceability = f.get('danceability', 0)
                key = f.get('key', None)
                scale = f.get('scale', None)
                file_path = f['filepath']
                # Skip invalid data
                if None in (bpm, centroid, danceability):
                    continue
                bpm_group = get_bpm_group(bpm)
                energy_group = get_energy_group(danceability)
                mood_group = get_mood_group(centroid)
                key_group = ''
                if key is not None and 0 <= key <= 11:
                    key_group = f"{keys[int(key)]}_{'Major' if scale == 1 else 'Minor'}"
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
                playlist_name = self.sanitize_filename(playlist_name)
                file_name = self._sanitize_file_name(playlist_name)
                if playlist_name not in playlists:
                    playlists[playlist_name] = {
                        'tracks': [],
                        'hashes': set(),
                        'features': {
                            'bpm_group': bpm_group,
                            'energy_group': energy_group,
                            'mood_group': mood_group,
                            'key_group': key_group
                        },
                        'file_name': file_name
                    }
                file_hash = f.get('file_hash', file_path)  # fallback to path if no hash
                if file_hash not in playlists[playlist_name]['hashes']:
                    playlists[playlist_name]['hashes'].add(file_hash)
                    playlists[playlist_name]['tracks'].append(file_path)
            # Merge small playlists (<20 tracks) into similar larger ones
            final_playlists = {}
            various_tracks = []
            various_hashes = set()
            for name, data in sorted(playlists.items(), key=lambda x: -len(x[1]['tracks'])):
                if len(data['tracks']) >= 20:
                    final_playlists[name] = data
                else:
                    # Find best matching large playlist
                    best_match = None
                    best_score = 0
                    for final_name, final_data in final_playlists.items():
                        score = 0
                        if data['features']['bpm_group'] == final_data['features']['bpm_group']:
                            score += 2
                        if data['features']['energy_group'] == final_data['features']['energy_group']:
                            score += 1.5
                        if (abs(centroid_mapping[data['features']['mood_group']] - centroid_mapping[final_data['features']['mood_group']]) <= 1):
                            score += 1
                        if score > best_score:
                            best_score = score
                            best_match = final_name
                    if best_match and best_score >= 2:
                        final_playlists[best_match]['tracks'].extend(data['tracks'])
                        final_playlists[best_match]['hashes'].update(data['hashes'])
                    else:
                        various_tracks.extend(data['tracks'])
                        various_hashes.update(data['hashes'])
            # Add various playlist if needed
            if various_tracks:
                final_playlists["Various_Tracks"] = {
                    'tracks': various_tracks,
                    'hashes': various_hashes,
                    'features': {
                        'bpm_group': 'Various',
                        'energy_group': 'Various',
                        'mood_group': 'Various'
                    }
                }
            # Split large playlists (>500 tracks) by mood group
            balanced_playlists = {}
            for name, data in final_playlists.items():
                if len(data['tracks']) > 500:
                    mood_groups = {}
                    for track in data['tracks']:
                        for orig_name, orig_data in playlists.items():
                            if track in orig_data['tracks']:
                                mood = orig_data['features']['mood_group']
                                break
                        if mood not in mood_groups:
                            mood_groups[mood] = []
                        mood_groups[mood].append(track)
                    for mood, tracks in mood_groups.items():
                        new_name = f"{name}_{mood}"
                        balanced_playlists[new_name] = {
                            'tracks': tracks,
                            'hashes': set(),
                            'features': data['features'].copy()
                        }
                        balanced_playlists[new_name]['features']['mood_group'] = mood
                else:
                    balanced_playlists[name] = data
            logger.info(f"Generated {len(balanced_playlists)} playlists from {sum(len(p['tracks']) for p in balanced_playlists.values())} tracks")
            # Remove hashes from output for compatibility
            for v in balanced_playlists.values():
                v.pop('hashes', None)
            return balanced_playlists
        except Exception as e:
            logger.error(f"Feature-group playlist generation failed: {str(e)}")
            return {} 