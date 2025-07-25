#!/usr/bin/env python3
"""
Optimized Music Playlist Generator with Enhanced Naming
"""

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import argparse
import os
import logging
from colorlog import ColoredFormatter
from tqdm import tqdm
import sys
from analyze_music import get_all_features
import numpy as np
import time
import traceback
import re
import sqlite3
import time
import gc

# Logging setup
def setup_colored_logging():
    """Configure colored logging for the application"""
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger

# Initialize colored logging
logger = setup_colored_logging()

def sanitize_filename(name):
    name = re.sub(r'[^\w\-_]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

def process_file_worker(filepath):
    try:
        # Skip files that are too small to be valid audio files
        if os.path.getsize(filepath) < 1024:  # 1KB minimum
            logger.warning(f"Skipping small file: {filepath}")
            return None, filepath

        # Skip non-audio files
        if not filepath.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
            return None, filepath

        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)

        # If we got a timeout, try again once
        if result is None or (result[0] is None and "timeout" in str(result).lower()):
            logger.warning(f"Retrying {filepath} after timeout")
            result = audio_analyzer.extract_features(filepath)

        if result and result[0] is not None:
            features = result[0]
            # Ensure no None values in critical features
            for key in ['bpm', 'centroid', 'duration']:
                if features.get(key) is None:
                    features[key] = 0.0
            return features, filepath
        return None, filepath
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None, filepath

def setup_playlist_db(cache_file):
    conn = sqlite3.connect(cache_file, timeout=60)
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

class PlaylistGenerator:
    def __init__(self):
        self.failed_files = []
        self.container_music_dir = ""
        self.host_music_dir = ""
        self.cache_file = os.path.join(os.getenv('CACHE_DIR', '/app/cache'), 'audio_analysis.db')
        setup_playlist_db(self.cache_file)

    def analyze_directory(self, music_dir, workers=None, force_sequential=False):
        file_list = []
        self.failed_files = []
        self.container_music_dir = music_dir.rstrip('/')

        for root, _, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
                    file_list.append(os.path.join(root, file))

        logger.info(f"Found {len(file_list)} audio files")
        if not file_list:
            logger.warning("No valid audio files found")
            return []

        if workers is None:
            workers = max(1, mp.cpu_count() // 2)
            logger.info(f"Using automatic worker count: {workers}")

        if force_sequential or workers <= 1:
            logger.info("Using sequential processing")
            return self._process_sequential(file_list)

        return self._process_parallel(file_list, workers)

    def _process_sequential(self, file_list):
        results = []
        with tqdm(file_list, desc="Analyzing files") as pbar:
                try:
                    # Free memory periodically
                    if pbar.n % 10 == 0:
                        gc.collect()

                    features, _ = process_file_worker(filepath)
                    if features:
                        results.append(features)
                    else:
                        self.failed_files.append(filepath)
                except Exception as e:
                    self.failed_files.append(filepath)
                    logger.error(f"Error processing {filepath}: {str(e)}")

                pbar.set_postfix_str(f"OK: {len(results)}, Failed: {len(self.failed_files)}")
        return results

    def _process_parallel(self, file_list, workers):
        results = []
        max_retries = 3
        retries = 0
        batch_size = min(50, len(file_list))  # Process in batches of 50

        while file_list and retries < max_retries:
            try:
                logger.info(f"Starting multiprocessing with {workers} workers (retry {retries})")
                ctx = mp.get_context('spawn')

                # Process in smaller batches to avoid hangs
                for i in range(0, len(file_list), batch_size):
                    batch = file_list[i:i+batch_size]
                    with ctx.Pool(processes=workers) as pool:
                        with tqdm(total=len(batch), desc=f"Processing batch {i//batch_size+1}",
                                 bar_format="{l_bar}{bar:40}{r_bar}",
                                 file=sys.stdout) as pbar:

                            # Use imap_unordered for better performance
                            for features, filepath in pool.imap_unordered(process_file_worker, batch):
                                if features:
                                    results.append(features)
                                else:
                                    self.failed_files.append(filepath)
                                pbar.update(1)
                                pbar.set_postfix_str(f"OK: {len(results)}, Failed: {len(self.failed_files)}")

                    # Clear pool after each batch to prevent resource buildup
                    pool.close()
                    pool.join()

                # Successfully processed all batches
                logger.info(f"Processing completed - {len(results)} successful, {len(self.failed_files)} failed")
                return results

            except (mp.TimeoutError, BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Multiprocessing error: {str(e)}")
                retries += 1
                # Skip the batch that caused the failure
                file_list = file_list[i+batch_size:] if 'i' in locals() else file_list
                logger.warning(f"Retrying with {len(file_list)} remaining files")

        # If we exhausted retries, fall back to sequential
        logger.error(f"Max retries reached, switching to sequential for remaining files")
        results.extend(self._process_sequential(file_list))
        return results

    def get_all_features_from_db(self):
        try:
            features = get_all_features()
            return features
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return []

    def convert_to_host_path(self, container_path):
        if not self.host_music_dir or not self.container_music_dir:
            return container_path

        container_path = os.path.normpath(container_path)
        container_music_dir = os.path.normpath(self.container_music_dir)

        if not container_path.startswith(container_music_dir):
            return container_path

        rel_path = os.path.relpath(container_path, container_music_dir)
        return os.path.join(self.host_music_dir, rel_path)

    def generate_playlist_name(self, features):
        """Generate radio-friendly playlist names"""
        # Safely extract all features with defaults
        try:
            bpm = float(features.get('bpm', 0))
            centroid = float(features.get('centroid', 0))
            danceability = float(features.get('danceability', 0))
        except (TypeError, ValueError) as e:
            logger.warning(f"Feature conversion error: {str(e)}")
            bpm, centroid, danceability = 0, 0, 0

        # New descriptive categories
        bpm_desc = (
            "Slow" if bpm < 70 else
            "Medium" if bpm < 100 else
            "Upbeat" if bpm < 130 else
            "Fast"
        )

        dance_desc = (
            "Chill" if danceability < 0.3 else
            "Easy" if danceability < 0.5 else
            "Groovy" if danceability < 0.7 else
            "Dance" if danceability < 0.85 else
            "Energetic"
        )

        mood_desc = (
            "Relaxing" if centroid < 300 else
            "Mellow" if centroid < 1000 else
            "Balanced" if centroid < 2000 else
            "Bright" if centroid < 4000 else
            "Intense"
        )

        return f"{bpm_desc} {dance_desc} {mood_desc}"

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
                            centroid_mapping[final_data['features']['mood_group']]) <= 1):
                            score += 1

                        if score > best_score:
                            best_score = score
                            best_match = final_name

                    # Merge if decent match found (score >= 2)
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

            # Ensure no playlist is too large (>500 tracks)
            balanced_playlists = {}
            for name, data in final_playlists.items():
                if len(data['tracks']) > 500:
                    # Split large playlist by mood group
                    mood_groups = {}
                    for track in data['tracks']:
                        # Get mood from original features
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
                            'hashes': set(),  # Can't preserve hashes when splitting
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
                SELECT file_path
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

def main():
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True, help='Music directory in container')
    parser.add_argument('--host_music_dir', required=True, help='Host music directory')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory')
    parser.add_argument('--num_playlists', type=int, default=8, help='Number of playlists')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of workers (default: auto)')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Clustering chunk size')
    parser.add_argument('--use_db', action='store_true', help='Use database only')
    parser.add_argument('--force_sequential', action='store_true',
                       help='Force sequential processing')
    parser.add_argument('--update', action='store_true',
                       help='Update existing playlists instead of recreating')
    args = parser.parse_args()

    generator = PlaylistGenerator()
    generator.host_music_dir = args.host_music_dir.rstrip('/')

    start_time = time.time()
    try:
        missing_in_db = generator.cleanup_database()
        if missing_in_db:
            logger.info(f"Removed {len(missing_in_db)} missing files from database")
            generator.failed_files.extend(missing_in_db)

        if args.update:
            logger.info("Updating playlists from database")
            generator.update_playlists()
            playlists = generator.generate_playlists_from_db()
            generator.save_playlists(playlists, args.output_dir)
        elif args.use_db:
            logger.info("Generating playlists from database")
            playlists = generator.generate_playlists_from_db()
            generator.save_playlists(playlists, args.output_dir)
        else:
            logger.info("Analyzing directory")
            features = generator.analyze_directory(
                args.music_dir,
                args.workers,
                args.force_sequential
            )

            logger.info(f"Processed {len(features)} files, {len(generator.failed_files)} failed")

            if features:
                playlists = generator.generate_playlists(
                    features,
                    args.num_playlists,
                    args.chunk_size,
                    args.output_dir
                )
                generator.save_playlists(playlists, args.output_dir)
            else:
                logger.error("No valid audio files available")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f} seconds")

    # Clean up Essentia resources
    os.environ['ESSENTIA_LOGGING_LEVEL'] = 'ERROR'
    os.environ['ESSENTIA_LOG_FILE'] = '/dev/null'

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()