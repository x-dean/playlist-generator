import pandas as pd
import numpy as np
import datetime
import sqlite3
import logging
from collections import defaultdict
from audio_analysis import get_all_features

logger = logging.getLogger(__name__)

class TimeBasedScheduler:
    def __init__(self):
        self.time_slots = {
            'Morning': (6, 12),
            'Afternoon': (12, 18),
            'Evening': (18, 22),
            'Late_Night': (22, 6)
        }
        self.feature_rules = {
            # ... same as original ...
        }
    
    # ... rest of TimeBasedScheduler implementation ...

class PlaylistGenerator:
    def __init__(self):
        self.failed_files = []
        self.container_music_dir = ""
        self.host_music_dir = ""
        self.cache_file = os.path.join(os.getenv('CACHE_DIR', '/app/cache'), 'audio_analysis.db')
        self.scheduler = TimeBasedScheduler()
        self.playlist_history = defaultdict(list)
        self._setup_playlist_db()

    def _setup_playlist_db(self):
        conn = sqlite3.connect(self.cache_file, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS playlists (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS playlist_tracks (
            playlist_id INTEGER,
            file_hash TEXT,
            added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(playlist_id) REFERENCES playlists(id),
            FOREIGN KEY(file_hash) REFERENCES audio_features(file_hash),
            PRIMARY KEY (playlist_id, file_hash)
        )
        """)
        conn.commit()
        conn.close()

    def generate_time_based_playlists(self, features_list):
        """Generate all time-based playlists"""
        playlists = {}
        for slot_name in self.scheduler.time_slots:
            # ... filtering logic same as original ...
            playlists[f"TimeSlot_{slot_name}"] = {
                'tracks': filtered_tracks,
                'features': {'type': 'time_based', 'slot': slot_name}
            }
        return playlists

    def generate_playlists_from_db(self):
        """Generate balanced playlists using only meaningful feature combinations"""

        # Helper mapping for mood similarity
        centroid_mapping = {
            'Warm': 0,
            'Mellow': 1,
            'Balanced': 2,
            'Bright': 3,
            'Crisp': 4
        }

        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()

            # Get all tracks with their features
            cursor.execute("""
            SELECT
                file_path,
                file_hash,
                bpm,
                centroid,
                danceability,
                key,
                scale,
                loudness
            FROM audio_features
            """)

            # Define more balanced energy categories
            def get_energy_group(danceability):
                if danceability < 0.3: return 'Chill'
                if danceability < 0.5: return 'Mellow'
                if danceability < 0.7: return 'Groovy'
                if danceability < 0.85: return 'Energetic'
                return 'Intense'

            # Define better BPM categories
            def get_bpm_group(bpm):
                if bpm < 70: return 'Slow'
                if bpm < 100: return 'Medium'
                if bpm < 130: return 'Upbeat'
                if bpm < 160: return 'Fast'
                return 'VeryFast'

            # Define mood categories based on centroid
            def get_mood_group(centroid):
                if centroid < 500: return 'Warm'
                if centroid < 1500: return 'Mellow'
                if centroid < 3000: return 'Balanced'
                if centroid < 6000: return 'Bright'
                return 'Crisp'

            # Group tracks into playlists
            playlists = {}
            for row in cursor.fetchall():
                file_path, file_hash, bpm, centroid, danceability, key, scale, loudness = row

                # Skip invalid data
                if None in (bpm, centroid, danceability):
                    continue

                # Get feature groups
                bpm_group = get_bpm_group(bpm)
                energy_group = get_energy_group(danceability)
                mood_group = get_mood_group(centroid)

                # Only use key if it's valid (0-11)
                key_group = ''
                if key is not None and 0 <= key <= 11:
                    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    key_group = keys[int(key)]
                    scale_group = 'Major' if scale == 1 else 'Minor'
                    key_group = f"{key_group}_{scale_group}"

                # Create playlist name - only use meaningful differentiators
                if key_group:
                    playlist_name = f"{bpm_group}_{energy_group}_{key_group}"
                else:
                    playlist_name = f"{bpm_group}_{energy_group}"

                # Ensure we don't create too many small playlists
                if mood_group in ('Bright', 'Crisp'):
                    playlist_name = f"{playlist_name}_Bright"
                elif mood_group in ('Warm', 'Mellow'):
                    playlist_name = f"{playlist_name}_Warm"

                if playlist_name not in playlists:
                    playlists[playlist_name] = {
                        'tracks': [],
                        'hashes': set(),
                        'moods': [],
                        'features': {
                            'bpm_group': bpm_group,
                            'energy_group': energy_group,
                            'mood_group': mood_group,
                            'key_group': key_group
                        }
                    }

                if file_hash not in playlists[playlist_name]['hashes']:
                    playlists[playlist_name]['hashes'].add(file_hash)
                    playlists[playlist_name]['tracks'].append(file_path)
                    playlists[playlist_name]['moods'].append(mood_group)

            # Merge small playlists (<20 tracks) into similar larger ones
            final_playlists = {}
            various_tracks = []
            various_hashes = set()

            # Process playlists by size (largest first)
            for name, data in sorted(playlists.items(),
                                key=lambda x: -len(x[1]['tracks'])):
                if len(data['tracks']) >= 20:  # Keep substantial playlists
                    final_playlists[name] = data
                else:
                    # Find best matching large playlist
                    best_match = None
                    best_score = 0

                    for final_name, final_data in final_playlists.items():
                        score = 0
                        # Match on BPM group
                        if data['features']['bpm_group'] == final_data['features']['bpm_group']:
                            score += 2
                        # Match on energy group
                        if data['features']['energy_group'] == final_data['features']['energy_group']:
                            score += 1.5
                        # Match on mood group
                        if (abs(centroid_mapping[data['features']['mood_group']] -
                            centroid_mapping[final_data['features']['mood_group']])) <= 1:
                            score += 1

                        if score > best_score:
                            best_score = score
                            best_match = final_name

                    # Merge if decent match found (score >= 2)
                    if best_match and best_score >= 2:
                        final_playlists[best_match]['tracks'].extend(data['tracks'])
                        final_playlists[best_match]['hashes'].update(data['hashes'])
                        final_playlists[best_match]['moods'].extend(data['moods'])
                    else:
                        various_tracks.extend(data['tracks'])
                        various_hashes.update(data['hashes'])

            # Add various playlist if needed
            if various_tracks:
                final_playlists["Various_Tracks"] = {
                    'tracks': various_tracks,
                    'hashes': various_hashes,
                    'moods': ['Various'] * len(various_tracks),
                    'features': {
                        'bpm_group': 'Various',
                        'energy_group': 'Various',
                        'mood_group': 'Various'
                    }
                }

            # Ensure no playlist is too large (>500 tracks)
            balanced_playlists = {}
            for name, data in final_playlists.items():
                if len(data['tracks']) > 500:
                    # Split large playlist by mood group
                    mood_groups = {}
                    for idx, track in enumerate(data['tracks']):
                        mood = data['moods'][idx]
                        if mood not in mood_groups:
                            mood_groups[mood] = []
                        mood_groups[mood].append(track)
                    
                    for mood, tracks in mood_groups.items():
                        new_name = f"{name}_{mood}"
                        balanced_playlists[new_name] = {
                            'tracks': tracks,
                            'hashes': set(),  # Can't preserve hashes when splitting
                            'moods': [mood] * len(tracks),
                            'features': data['features'].copy()
                        }
                        balanced_playlists[new_name]['features']['mood_group'] = mood
                else:
                    balanced_playlists[name] = data

            conn.commit()
            return balanced_playlists

        except Exception as e:
            logger.error(f"Database playlist generation failed: {str(e)}")
            return {}
        finally:
            conn.close()

    def update_playlists(self, changed_files=None):
        """Update playlists based on changed files"""
        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()

            # Get changed files if not provided
            if changed_files is None:
                cursor.execute("""
                SELECT file_path, file_hash
                FROM audio_features
                WHERE last_analyzed > (
                    SELECT MAX(last_updated) FROM playlists
                )
                """)
                changed_files = [row[0] for row in cursor.fetchall()]

            if not changed_files:
                logger.info("No changed files, playlists up-to-date")
                return

            logger.info(f"Updating playlists for {len(changed_files)} changed files")

            # Remove affected tracks from all playlists
            placeholders = ','.join(['?'] * len(changed_files))
            cursor.execute(f"""
            DELETE FROM playlist_tracks
            WHERE file_hash IN (
                SELECT file_hash FROM audio_features
                WHERE file_path IN ({placeholders})
            )
            """, changed_files)

            # Regenerate playlists
            self.generate_playlists_from_db()

            # Update playlist timestamp
            cursor.execute("UPDATE playlists SET last_updated = CURRENT_TIMESTAMP")
            conn.commit()

        except Exception as e:
            logger.error(f"Playlist update failed: {str(e)}")
        finally:
            conn.close()

    def generate_playlists(self, features_list, num_playlists=5, chunk_size=1000, output_dir=None):
        """Generate playlists with underscored names and merged similar clusters"""
        playlists = {}
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        try:
            # 1. Create DataFrame with all tracks
            data = []
            for f in features_list:
                if not f or 'filepath' not in f:
                    continue
                try:
                    key = f.get('key', None)
                    scale = f.get('scale', None)
                    data.append({
                        'filepath': str(f['filepath']),
                        'bpm': max(0, float(f.get('bpm', 0))),
                        'centroid': max(0, float(f.get('centroid', 0))),
                        'danceability': min(1.0, max(0, float(f.get('danceability', 0)))),
                        'key': key,
                        'scale': scale,
                        'duration': max(0, float(f.get('duration', 0)))
                    })
                except Exception as e:
                    logger.debug(f"Skipping track {f.get('filepath','unknown')}: {str(e)}")
                    continue

            if not data:
                logger.warning("No valid tracks after filtering")
                return playlists

            df = pd.DataFrame(data)

            # 2. Clustering with merging similar playlists
            try:
                cluster_features = ['bpm', 'centroid', 'danceability']
                weights = np.array([1.5, 1.0, 1.2])

                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(df[cluster_features])
                weighted_features = scaled_features * weights

                kmeans = MiniBatchKMeans(
                    n_clusters=min(num_playlists, len(df)),
                    random_state=42,
                    batch_size=min(500, len(df))
                )
                df['cluster'] = kmeans.fit_predict(weighted_features)

                # 3. Generate playlist names and merge similar ones
                temp_playlists = {}
                for cluster, group in df.groupby('cluster'):
                    centroid = {
                        'bpm': group['bpm'].median(),
                        'centroid': group['centroid'].median(),
                        'danceability': group['danceability'].median(),
                    }

                    # Generate name with underscores instead of spaces
                    name = self.generate_playlist_name(centroid).replace(" ", "_")

                    if name not in temp_playlists:
                        temp_playlists[name] = {
                            'tracks': group['filepath'].tolist(),
                            'size': len(group),
                            'features': centroid
                        }
                    else:
                        # Merge with existing playlist
                        temp_playlists[name]['tracks'].extend(group['filepath'].tolist())
                        temp_playlists[name]['size'] += len(group)
                        # Update features with new median values
                        combined = pd.concat([
                            pd.DataFrame([temp_playlists[name]['features']]),
                            pd.DataFrame([centroid])
                        ])
                        temp_playlists[name]['features'] = {
                            'bpm': combined['bpm'].median(),
                            'centroid': combined['centroid'].median(),
                            'danceability': combined['danceability'].median()
                        }

                # Final playlists with merged similar ones
                playlists = temp_playlists

                # Verify all tracks are distributed
                distributed_tracks = sum(len(p['tracks']) for p in playlists.values())
                if distributed_tracks != len(df):
                    logger.warning(f"Track distribution mismatch: {distributed_tracks} vs {len(df)}")

                logger.info(f"Generated {len(playlists)} playlists from {len(df)} tracks")

            except Exception as e:
                logger.error(f"Clustering failed: {str(e)}")
                logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"Playlist generation failed: {str(e)}")
            logger.error(traceback.format_exc())

        return playlists


    def cleanup_database(self):
        """Clean up database entries for missing files"""
        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM audio_features")
            db_files = [row[0] for row in cursor.fetchall()]

            missing_files = []
            for file_path in db_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)

            if missing_files:
                logger.info(f"Cleaning up {len(missing_files)} missing files from database")
                placeholders = ','.join(['?'] * len(missing_files))
                cursor.execute(
                    f"DELETE FROM audio_features WHERE file_path IN ({placeholders})",
                    missing_files
                )
                conn.commit()

            conn.close()
            return missing_files
        except Exception as e:
            logger.error(f"Database cleanup failed: {str(e)}")
            return []

    def save_playlists(self, playlists, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0
        total_tracks = 0

        for name, playlist_data in playlists.items():
            songs = playlist_data.get('tracks', [])
            if not songs:
                continue

            host_songs = [self.convert_to_host_path(song) for song in songs]
            total_tracks += len(host_songs)

            playlist_path = os.path.join(output_dir, f"{name}.m3u")
            with open(playlist_path, 'w') as f:
                f.write("\n".join(host_songs))
            saved_count += 1
            logger.info(f"Saved {name} with {len(host_songs)} tracks")

        # Create All_Successful_Tracks playlist
        all_tracks = []
        for playlist_data in playlists.values():
            all_tracks.extend(playlist_data.get('tracks', []))

        if all_tracks:
            host_all_tracks = [self.convert_to_host_path(song) for song in all_tracks]
            all_path = os.path.join(output_dir, "All_Successful_Tracks.m3u")
            with open(all_path, 'w') as f:
                f.write("\n".join(host_all_tracks))
            logger.info(f"Saved All_Successful_Tracks with {len(host_all_tracks)} tracks")

        # Save failed files
        all_failed = list(set(self.failed_files))
        if all_failed:
            failed_path = os.path.join(output_dir, "Failed_Files.m3u")
            with open(failed_path, 'w') as f:
                host_failed = [self.convert_to_host_path(p) for p in all_failed]
                f.write("\n".join(host_failed))
            logger.info(f"Saved {len(all_failed)} failed/missing files")