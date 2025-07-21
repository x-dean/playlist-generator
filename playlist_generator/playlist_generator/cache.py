# playlist_generator/cache.py
import sqlite3
import logging
from collections import defaultdict
import traceback

logger = logging.getLogger(__name__)

class CacheBasedGenerator:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.playlist_history = defaultdict(list)

    def generate(self):
        """Generate balanced playlists using only meaningful feature combinations"""
        centroid_mapping = {
            'Warm': 0, 'Mellow': 1, 'Balanced': 2, 'Bright': 3, 'Crisp': 4
        }

        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()
            cursor.execute("""
            SELECT file_path, file_hash, bpm, centroid, danceability, key, scale
            FROM audio_features
            WHERE bpm IS NOT NULL 
            AND centroid IS NOT NULL 
            AND danceability IS NOT NULL
            """)
            
            # Helper functions
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

            # Group tracks into playlists
            playlists = {}
            for row in cursor.fetchall():
                file_path, file_hash, bpm, centroid, danceability, key, scale = row
                bpm_group = get_bpm_group(bpm)
                energy_group = get_energy_group(danceability)
                mood_group = get_mood_group(centroid)

                # Create playlist name
                key_str = ""
                if key is not None and 0 <= key <= 11:
                    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    key_str = keys[int(key)]
                    scale_str = 'Major' if scale == 1 else 'Minor'
                    key_str = f"{key_str}_{scale_str}"
                
                # Build playlist name
                playlist_name = f"{bpm_group}_{energy_group}"
                if key_str:
                    playlist_name += f"_{key_str}"
                
                # Apply mood-based grouping
                if mood_group in ('Bright', 'Crisp'):
                    playlist_name += "_Bright"
                elif mood_group in ('Warm', 'Mellow'):
                    playlist_name += "_Warm"

                if playlist_name not in playlists:
                    playlists[playlist_name] = {
                        'tracks': [],
                        'hashes': set(),
                        'features': {
                            'bpm_group': bpm_group,
                            'energy_group': energy_group,
                            'mood_group': mood_group,
                            'key_group': key_str
                        }
                    }

                if file_hash not in playlists[playlist_name]['hashes']:
                    playlists[playlist_name]['hashes'].add(file_hash)
                    playlists[playlist_name]['tracks'].append(file_path)

            # Merge small playlists (<20 tracks) into similar larger ones
            final_playlists = {}
            for name, data in sorted(playlists.items(), key=lambda x: -len(x[1]['tracks'])):
                if len(data['tracks']) >= 20:
                    final_playlists[name] = data
                    continue
                
                # Find best matching playlist
                best_match = None
                best_score = 0
                for final_name, final_data in final_playlists.items():
                    score = 0
                    if data['features']['bpm_group'] == final_data['features']['bpm_group']:
                        score += 2
                    if data['features']['energy_group'] == final_data['features']['energy_group']:
                        score += 1.5
                    if (abs(centroid_mapping.get(data['features']['mood_group'], 0) - 
                        centroid_mapping.get(final_data['features']['mood_group'], 0))) <= 1:
                        score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_match = final_name
                
                # Merge if decent match found
                if best_match and best_score >= 2:
                    for track in data['tracks']:
                        if track not in final_playlists[best_match]['tracks']:
                            final_playlists[best_match]['tracks'].append(track)
                else:
                    # Create "Various" playlist
                    if "Various_Tracks" not in final_playlists:
                        final_playlists["Various_Tracks"] = {
                            'tracks': [],
                            'features': {
                                'bpm_group': 'Various',
                                'energy_group': 'Various',
                                'mood_group': 'Various'
                            }
                        }
                    final_playlists["Various_Tracks"]['tracks'].extend(data['tracks'])

            return final_playlists

        except Exception as e:
            logger.error(f"Database playlist generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
        finally:
            conn.close()