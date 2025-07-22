# playlist_generator/cache.py
import sqlite3
import logging
import traceback
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import os

logger = logging.getLogger(__name__)


class CacheBasedGenerator:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.playlist_history = defaultdict(list)

    def generate(self, features_list=None):
        """Generate playlists from either provided features or database"""
        if features_list:
            return self._generate_from_features(features_list)
        return self._generate_from_db()

    def _generate_from_features(self, features_list):
        playlists = {}
        for feature in features_list:
            if not feature or 'filepath' not in feature:
                continue

            file_path = feature['filepath']
            bpm = feature.get('bpm', 0)
            centroid = feature.get('centroid', 0)
            danceability = feature.get('danceability', 0)
            key = feature.get('key', -1)
            scale = feature.get('scale', 0)

            bpm_group = self._get_bpm_group(bpm)
            energy_group = self._get_energy_group(danceability)
            mood_group = self._get_mood_group(centroid)

            key_str = ""
            if key is not None and 0 <= int(key) <= 11:
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key_str = f"{keys[int(key)]}_{'Major' if scale == 1 else 'Minor'}"

            playlist_name = f"{bpm_group}_{energy_group}"
            if key_str:
                playlist_name += f"_{key_str}"
            if mood_group in ('Bright', 'Crisp'):
                playlist_name += "_Bright"
            elif mood_group in ('Warm', 'Mellow'):
                playlist_name += "_Warm"

            if playlist_name not in playlists:
                playlists[playlist_name] = {
                    'tracks': [],
                    'features': {
                        'bpm_group': bpm_group,
                        'energy_group': energy_group,
                        'mood_group': mood_group,
                        'key_group': key_str
                    }
                }

            playlists[playlist_name]['tracks'].append(file_path)

        return self._merge_playlists(playlists)

    def _generate_from_db(self):
        centroid_mapping = {
            'Warm': 0, 'Mellow': 1, 'Balanced': 2, 'Bright': 3, 'Crisp': 4
        }

        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()

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

            cursor.execute("""
                SELECT file_path, file_hash, bpm, centroid, danceability, key, scale
                FROM audio_features
                WHERE bpm IS NOT NULL 
                AND centroid IS NOT NULL 
                AND danceability IS NOT NULL
            """)
            rows = cursor.fetchall()

            playlists = {}
            for row in rows:
                file_path, file_hash, bpm, centroid, danceability, key, scale = row

                bpm_group = self._get_bpm_group(bpm)
                energy_group = self._get_energy_group(danceability)
                mood_group = self._get_mood_group(centroid)

                key_str = ""
                if key is not None and 0 <= int(key) <= 11:
                    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    key_str = f"{keys[int(key)]}_{'Major' if scale == 1 else 'Minor'}"

                playlist_name = f"{bpm_group}_{energy_group}"
                if key_str:
                    playlist_name += f"_{key_str}"
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

            return self._merge_playlists(playlists, centroid_mapping)

        except Exception as e:
            logger.error(f"Database playlist generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
        finally:
            conn.close()

    def _merge_playlists(self, playlists, centroid_mapping=None):
        final_playlists = {}
        min_size = int(os.getenv('MIN_PLAYLIST_SIZE', 20))
        max_size = int(os.getenv('MAX_PLAYLIST_SIZE', 100))
        for name, data in sorted(playlists.items(), key=lambda x: -len(x[1]['tracks'])):
            # SPLIT large playlists
            tracks = data['tracks']
            if len(tracks) > max_size:
                for i in range(0, len(tracks), max_size):
                    split_name = f"{name}_Part{i//max_size+1}"
                    final_playlists[split_name] = {
                        'tracks': tracks[i:i+max_size],
                        'features': data['features']
                    }
                continue
            # MERGE small playlists
            if len(tracks) >= min_size:
                final_playlists[name] = data
                continue
            best_match = None
            best_score = 0
            for final_name, final_data in final_playlists.items():
                score = 0
                if data['features']['bpm_group'] == final_data['features']['bpm_group']:
                    score += 2
                if data['features']['energy_group'] == final_data['features']['energy_group']:
                    score += 1.5
                if centroid_mapping:
                    diff = abs(
                        centroid_mapping.get(data['features']['mood_group'], 0) -
                        centroid_mapping.get(final_data['features']['mood_group'], 0)
                    )
                    if diff <= 1:
                        score += 1
                elif data['features']['mood_group'] == final_data['features']['mood_group']:
                    score += 1
                if score > best_score:
                    best_score = score
                    best_match = final_name
            if best_match and best_score >= 2:
                for track in tracks:
                    if track not in final_playlists[best_match]['tracks']:
                        final_playlists[best_match]['tracks'].append(track)
            else:
                if "Various_Tracks" not in final_playlists:
                    final_playlists["Various_Tracks"] = {
                        'tracks': [],
                        'features': {
                            'bpm_group': 'Various',
                            'energy_group': 'Various',
                            'mood_group': 'Various',
                            'key_group': 'Various'
                        }
                    }
                final_playlists["Various_Tracks"]['tracks'].extend(tracks)
        return final_playlists

    def _get_energy_group(self, danceability):
        if danceability < 0.3: return 'Chill'
        if danceability < 0.5: return 'Mellow'
        if danceability < 0.7: return 'Groovy'
        if danceability < 0.85: return 'Energetic'
        return 'Intense'

    def _get_bpm_group(self, bpm):
        if bpm < 70: return 'Slow'
        if bpm < 100: return 'Medium'
        if bpm < 130: return 'Upbeat'
        if bpm < 160: return 'Fast'
        return 'VeryFast'

    def _get_mood_group(self, centroid):
        if centroid < 500: return 'Warm'
        if centroid < 1500: return 'Mellow'
        if centroid < 3000: return 'Balanced'
        if centroid < 6000: return 'Bright'
        return 'Crisp'
