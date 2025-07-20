#!/usr/bin/env python3
"""
Optimized Music Playlist Generator with Colored Logging
"""

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import argparse
import os
import logging
from tqdm import tqdm
from analyze_music import get_all_features
import numpy as np
import time
import traceback
import re
import sqlite3
import json
from datetime import datetime
import hashlib
import coloredlogs
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError


# Colored logging setup
logger = logging.getLogger(__name__)
coloredlogs.install(
    level='INFO',
    logger=logger,
    fmt='%(levelname)s - %(message)s',
    field_styles={
        'levelname': {'color': 'cyan', 'bold': True},
    },
    level_styles={
        'debug': {'color': 'green'},
        'info': {'color': 39},  # Default terminal color (usually white)
        'warning': {'color': 214},  # Orange color
        'error': {'color': 'red', 'bold': True},
        'critical': {'color': 'red', 'bold': True, 'background': 'white'},
    }
)

def sanitize_filename(name):
    name = re.sub(r'[^\w\-_]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

def process_file_worker(filepath):
    try:
        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)
        if result and result[0] is not None:
            features = result[0]
            # Ensure critical features have values
            for key in ['bpm', 'centroid', 'duration']:
                if features.get(key) is None:
                    features[key] = 0.0
            return features, filepath
        return None, filepath
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None, filepath

class PlaylistGenerator:
    def __init__(self):
        self.failed_files = []
        self.container_music_dir = ""
        self.host_music_dir = ""
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = os.path.join(cache_dir, 'audio_analysis.db')
        self.playlist_db = os.path.join(cache_dir, 'playlist_tracker.db')
        self.state_file = os.path.join(cache_dir, 'library_state.json')
        self._init_playlist_tracker()
        
    def _init_playlist_tracker(self):
        """Initialize playlist tracking database with schema migration"""
        os.makedirs(os.path.dirname(self.playlist_db), exist_ok=True)
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()
        
        # Create playlists table with basic columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migration: Check existing columns and add missing ones
        cursor.execute("PRAGMA table_info(playlists)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # List of required centroid columns with their types
        centroid_columns = [
            ('centroid_bpm', 'REAL'),
            ('centroid_centroid', 'REAL'),
            ('centroid_duration', 'REAL'),
            ('centroid_loudness', 'REAL'),
            ('centroid_dynamics', 'REAL'),
            ('centroid_rhythm', 'REAL'),
            ('centroid_key', 'TEXT')
        ]
        
        # Add any missing columns
        for col_name, col_type in centroid_columns:
            if col_name not in columns:
                try:
                    cursor.execute(f'ALTER TABLE playlists ADD COLUMN {col_name} {col_type}')
                except sqlite3.OperationalError:
                    pass  # Column already exists
        
        # Create other tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlist_songs (
                playlist_id INTEGER,
                filepath TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (playlist_id, filepath),
                FOREIGN KEY (playlist_id) REFERENCES playlists(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS library_state (
                filepath TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                checksum TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _get_file_metadata(self, filepath):
        """Get file metadata for state tracking"""
        try:
            stat = os.stat(filepath)
            return {
                'mtime': stat.st_mtime,
                'size': stat.st_size,
                'checksum': self._calculate_checksum(filepath) if stat.st_size < 10000000 else None
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {filepath}: {str(e)}")
            return None

    def _calculate_checksum(self, filepath):
        """Calculate MD5 checksum for file (for small files only)"""
        if not os.path.exists(filepath):
            return None
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {filepath}: {str(e)}")
            return None

    def _save_library_state(self, state):
        """Save current library state to file and database"""
        # Save to JSON file
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save to database for more robust tracking
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()
        
        # Clear existing state
        cursor.execute("DELETE FROM library_state")
        
        # Insert new state
        for filepath, metadata in state.items():
            cursor.execute('''
                INSERT INTO library_state (filepath, mtime, size, checksum)
                VALUES (?, ?, ?, ?)
            ''', (filepath, metadata['mtime'], metadata['size'], metadata.get('checksum')))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved library state with {len(state)} files")

    def _load_library_state(self):
        """Load previous library state from file or database"""
        # Try to load from JSON file first
        state = {}
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded library state from file with {len(state)} files")
                return state
            except Exception as e:
                logger.error(f"Error loading state file: {str(e)}")
        
        # If file loading fails, try database
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT filepath, mtime, size, checksum FROM library_state")
            for row in cursor.fetchall():
                state[row[0]] = {
                    'mtime': row[1],
                    'size': row[2],
                    'checksum': row[3]
                }
            logger.info(f"Loaded library state from database with {len(state)} files")
            return state
        except Exception as e:
            logger.error(f"Error loading state from database: {str(e)}")
            return {}
        finally:
            conn.close()

    def _scan_music_directory(self, music_dir):
        """Scan music directory and build file list with metadata"""
        file_metadata = {}
        for root, _, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                    filepath = os.path.join(root, file)
                    metadata = self._get_file_metadata(filepath)
                    if metadata:
                        file_metadata[filepath] = metadata
        return file_metadata

    def _get_file_changes(self, current_state, previous_state):
        """Identify added, removed, and modified files"""
        added = []
        removed = []
        modified = []
        
        current_files = set(current_state.keys())
        previous_files = set(previous_state.keys())
        
        # Find removed files
        for filepath in previous_files - current_files:
            removed.append(filepath)
        
        # Check existing files for modifications
        for filepath in current_files & previous_files:
            current_meta = current_state[filepath]
            prev_meta = previous_state[filepath]
            
            # Check if modified based on metadata
            if (current_meta['mtime'] != prev_meta['mtime'] or
                current_meta['size'] != prev_meta['size'] or
                (current_meta['checksum'] and prev_meta['checksum'] and 
                 current_meta['checksum'] != prev_meta['checksum'])):
                modified.append(filepath)
        
        # Find added files
        for filepath in current_files - previous_files:
            added.append(filepath)
        
        return {
            'added': added,
            'removed': removed,
            'modified': modified
        }

    def analyze_directory(self, music_dir, workers=None, force_sequential=False):
        file_list = []
        self.failed_files = []
        self.container_music_dir = music_dir.rstrip('/')

        for root, _, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                    file_list.append(os.path.join(root, file))

        logger.info(f"Found {len(file_list)} audio files")
        if not file_list:
            logger.warning("No valid audio files found")
            return []

        if workers is None:
            workers = max(1, mp.cpu_count() // 2)

        if force_sequential or workers <= 1:
            return self._process_sequential(file_list)

        return self._process_parallel(file_list, workers)

    def _process_sequential(self, file_list):
        results = []
        pbar = tqdm(file_list, desc="Processing files")
        for filepath in pbar:
            features, _ = process_file_worker(filepath)
            if features:
                results.append(features)
            else:
                self.failed_files.append(filepath)
        return results

    def _process_parallel(self, file_list, workers):
        """Refactored parallel processing with proper progress tracking"""
        results = []
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(process_file_worker, filepath): filepath
                    for filepath in file_list
                }
                
                # Process results as they complete
                with tqdm(total=len(file_list), desc="Processing files") as pbar:
                    for future in as_completed(future_to_file):
                        filepath = future_to_file[future]
                        try:
                            features, _ = future.result(timeout=600)
                            if features:
                                results.append(features)
                            else:
                                self.failed_files.append(filepath)
                        except TimeoutError:
                            self.failed_files.append(filepath)
                            logger.warning(f"Timeout processing {filepath}")
                        except Exception as e:
                            self.failed_files.append(filepath)
                            logger.error(f"Error processing {filepath}: {str(e)}")
                        finally:
                            pbar.update(1)
            return results
        except Exception as e:
            logger.error(f"Parallel processing failed: {str(e)}")
            # Fallback to sequential processing
            return self._process_sequential(file_list)

    def get_all_features_from_db(self):
        try:
            return get_all_features()
        except Exception:
            return []

    def cleanup_database(self):
        """Remove entries for files that no longer exist"""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM audio_features")
            db_files = [row[0] for row in cursor.fetchall()]
            
            missing_files = []
            for file_path in db_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                logger.info(f"Removed {len(missing_files)} missing files from database")
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
        """Improved playlist name generation with accurate duration"""
        # Ensure critical features have values
        required_features = ['bpm', 'centroid', 'duration']
        for feat in required_features:
            if feat not in features or features[feat] is None:
                features[feat] = 0.0

        bpm = features['bpm']
        centroid = features['centroid']
        duration = features['duration']
        
        # BPM descriptors
        bpm_desc = (
            "VerySlow" if bpm < 55 else
            "Slow" if bpm < 70 else
            "Chill" if bpm < 85 else
            "Medium" if bpm < 100 else
            "Upbeat" if bpm < 115 else
            "Energetic" if bpm < 135 else
            "Fast" if bpm < 155 else
            "VeryFast"
        )
        
        # Timbre descriptors
        timbre_desc = (
            "Dark" if centroid < 800 else
            "Warm" if centroid < 1500 else
            "Mellow" if centroid < 2200 else
            "Bright" if centroid < 3000 else
            "Sharp" if centroid < 4000 else
            "VerySharp"
        )
        
        # Duration descriptors - use the actual duration value
        duration_desc = (
            "Brief" if duration < 120 else  # Less than 2 minutes
            "Medium" if duration < 240 else  # 2-4 minutes
            "Long" if duration < 480 else    # 4-8 minutes
            "Extended"                      # More than 8 minutes
        )
        
        # Mood descriptor based on tempo
        mood = (
            "Ambient" if bpm < 75 else
            "Downtempo" if bpm < 95 else
            "Dance" if bpm > 125 else
            "Dynamic" if centroid > 3000 else
            "Balanced"
        )
        
        return f"{bpm_desc}_{timbre_desc}_{duration_desc}_{mood}"

# Replace the existing generate_playlists method with this improved version
    def generate_playlists(self, features_list, num_playlists=5, chunk_size=1000, output_dir=None):
        if not features_list:
            logger.warning("No features to cluster")
            return {}

        valid_features = []
        for feat in features_list:
            if not feat or 'filepath' not in feat:
                continue
            if os.path.exists(feat['filepath']):
                # Handle None values for critical features
                for key in ['bpm', 'centroid', 'duration']:
                    if feat.get(key) is None:
                        feat[key] = 0.0
                valid_features.append(feat)
            else:
                self.failed_files.append(feat['filepath'])
                
        if not valid_features:
            logger.warning("No valid features after filtering")
            return {}
            
        # Optimized clustering parameters
        n_clusters = min(
            max(5, num_playlists * 2),  # Ensure minimum clusters
            max(20, len(valid_features) // 10)  # Scale with library size
        )
        
        logger.info(f"Clustering {len(valid_features)} tracks into {n_clusters} groups")
        
        df = pd.DataFrame(valid_features)
        
        # Define features for clustering
        cluster_features = ['bpm', 'centroid']
        
        # Fill missing values
        for feat in cluster_features:
            if feat not in df.columns:
                df[feat] = 0.0
                
        df[cluster_features] = df[cluster_features].fillna(0)
        
        # Scale features
        features_array = df[cluster_features].values.astype(np.float32)
        features_scaled = StandardScaler().fit_transform(features_array)
        
        # Cluster using MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(500, len(df)),
            n_init='auto',
            max_iter=50,
            compute_labels=True,
            tol=0.001
        )
        df['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Group clusters by their characteristics
        playlist_groups = {}
        # Adjust minimum cluster size based on library size
        min_cluster_size = max(5, len(df) // 20)  # More flexible threshold
        
        for cluster in df['cluster'].unique():
            cluster_songs = df[df['cluster'] == cluster]
            if len(cluster_songs) < min_cluster_size:  # Skip small clusters
                continue
                
            # Get median duration for accurate naming
            median_duration = cluster_songs['duration'].median()
            centroid = cluster_songs[cluster_features].mean().to_dict()
            centroid['duration'] = median_duration
            
            base_name = sanitize_filename(self.generate_playlist_name(centroid))
            
            # Group clusters with similar characteristics
            if base_name not in playlist_groups:
                playlist_groups[base_name] = []
            playlist_groups[base_name].append(cluster_songs)
        
        # Create playlists from grouped clusters
        playlists = {}
        name_counter = {}
        
        for base_name, clusters in playlist_groups.items():
            # Merge clusters with similar characteristics
            merged_songs = pd.concat(clusters)
            
            # Recalculate centroid with median duration
            median_duration = merged_songs['duration'].median()
            centroid = merged_songs[cluster_features].mean().to_dict()
            centroid['duration'] = median_duration
            
            # Generate final name
            final_name = sanitize_filename(self.generate_playlist_name(centroid))
            
            # Ensure unique name
            if final_name in name_counter:
                name_counter[final_name] += 1
                final_name = f"{final_name}_{name_counter[final_name]}"
            else:
                name_counter[final_name] = 1
                
            playlists[final_name] = merged_songs['filepath'].tolist()
            logger.info(f"Created playlist '{final_name}' with {len(merged_songs)} tracks")
        
        return playlists

    def save_playlists(self, playlists, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0

        logger.info(f"Saving playlists to: {output_dir}")
        
        for name, songs in playlists.items():
            if not songs:
                logger.warning(f"Playlist '{name}' is empty, skipping")
                continue

            try:
                logger.debug(f"Processing playlist: {name} with {len(songs)} songs")
                host_songs = []
                missing_count = 0
                
                for song in songs:
                    host_path = self.convert_to_host_path(song)
                    # Verify path exists before adding to playlist
                    if os.path.exists(song):  # Check container path
                        host_songs.append(host_path)
                    else:
                        missing_count += 1
                        logger.debug(f"File not found: {song} (host path: {host_path})")
                
                if not host_songs:
                    logger.warning(f"No valid songs found for playlist '{name}'")
                    continue
                
                playlist_path = os.path.join(output_dir, f"{sanitize_filename(name)}.m3u")
                with open(playlist_path, 'w') as f:
                    f.write("\n".join(host_songs))
                saved_count += 1
                
                msg = f"Saved playlist '{name}' to {playlist_path} with {len(host_songs)} tracks"
                if missing_count:
                    msg += f" ({missing_count} missing files)"
                logger.info(msg)
            except Exception as e:
                logger.error(f"Error saving playlist '{name}': {str(e)}")
                logger.error(traceback.format_exc())

        if saved_count:
            logger.info(f"Saved {saved_count} playlists to {output_dir}")
        else:
            logger.warning("No playlists saved")

        all_failed = list(set(self.failed_files))
        
        if all_failed:
            failed_path = os.path.join(output_dir, "Failed_Files.m3u")
            try:
                host_failed = [self.convert_to_host_path(p) for p in all_failed]
                with open(failed_path, 'w') as f:
                    f.write("\n".join(host_failed))
                logger.info(f"Saved {len(all_failed)} failed/missing files to {failed_path}")
            except Exception as e:
                logger.error(f"Failed to save failed files list: {str(e)}")

    def _update_playlist_tracker(self, playlists, centroids=None):
        """Update playlist tracker with new playlist assignments"""
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()
        
        # Get existing playlists
        cursor.execute("SELECT id, name FROM playlists")
        existing_playlists = {name: id for id, name in cursor.fetchall()}
        
        # Process each playlist
        for name, filepaths in playlists.items():
            # Get or create playlist
            if name in existing_playlists:
                playlist_id = existing_playlists[name]
                cursor.execute(
                    "UPDATE playlists SET last_updated = ? WHERE id = ?",
                    (datetime.now(), playlist_id))
            else:
                centroid_data = centroids.get(name, {}) if centroids else {}
                cursor.execute(
                    """INSERT INTO playlists (name, centroid_bpm, centroid_centroid, 
                    centroid_duration, centroid_loudness, centroid_dynamics, 
                    centroid_rhythm, centroid_key) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (name, 
                     centroid_data.get('bpm', 0),
                     centroid_data.get('centroid', 0),
                     centroid_data.get('duration', 0),
                     centroid_data.get('loudness', -20),
                     centroid_data.get('dynamics', 0),
                     centroid_data.get('rhythm_complexity', 0.5),
                     centroid_data.get('key', 'C') + '-' + centroid_data.get('scale', 'major')))
                playlist_id = cursor.lastrowid
                existing_playlists[name] = playlist_id
            
            # Get current songs in playlist
            cursor.execute(
                "SELECT filepath FROM playlist_songs WHERE playlist_id = ?",
                (playlist_id,))
            current_files = {row[0] for row in cursor.fetchall()}
            new_files = set(filepaths)
            
            # Add new files
            for filepath in new_files - current_files:
                cursor.execute(
                    "INSERT OR IGNORE INTO playlist_songs (playlist_id, filepath) VALUES (?, ?)",
                    (playlist_id, filepath))
            
            # Remove deleted files
            for filepath in current_files - new_files:
                cursor.execute(
                    "DELETE FROM playlist_songs WHERE playlist_id = ? AND filepath = ?",
                    (playlist_id, filepath))
        
        conn.commit()
        conn.close()

    def incremental_update(self, changes, output_dir):
        """Update playlists incrementally based on file changes"""
        logger.info("Performing incremental update")
        start_time = time.time()
        
        # Process removed files
        if changes['removed']:
            self._remove_files_from_playlists(changes['removed'])
        
        # Process added and modified files
        files_to_process = changes['added'] + changes['modified']
        if not files_to_process:
            logger.info("No files to process in incremental update")
            return
        
        # Analyze new/modified files
        features = []
        if len(files_to_process) > 50:  # Use parallel for large batches
            logger.info(f"Processing {len(files_to_process)} files in parallel")
            features = self._process_parallel(files_to_process, workers=max(1, mp.cpu_count() // 2))
        else:
            logger.info(f"Processing {len(files_to_process)} files sequentially")
            features = []
            for filepath in tqdm(files_to_process, desc="Processing files"):
                feat, _ = process_file_worker(filepath)
                if feat:
                    features.append(feat)
        
        # Assign files to existing playlists
        assigned_count = self._assign_to_existing_playlists(features)
        logger.info(f"Assigned {assigned_count} files to existing playlists")
        
        # Save updated playlists
        self.save_playlists_from_db(output_dir)
        
        # Update library state
        self._update_library_state()
        
        elapsed = time.time() - start_time
        logger.info(f"Incremental update completed in {elapsed:.2f} seconds")

    def _remove_files_from_playlists(self, filepaths):
        """Remove files from all playlists"""
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()
        for filepath in filepaths:
            cursor.execute(
                "DELETE FROM playlist_songs WHERE filepath = ?", 
                (filepath,))
        conn.commit()
        conn.close()
        logger.info(f"Removed {len(filepaths)} files from playlists")

    def _assign_to_existing_playlists(self, features):
        """Assign files to existing playlists based on similarity"""
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()
        
        # Get all playlists with their centroids
        cursor.execute("""
            SELECT id, name, centroid_bpm, centroid_centroid
            FROM playlists
        """)
        playlists = cursor.fetchall()
        
        if not playlists:
            logger.warning("No existing playlists found for assignment")
            return 0
        
        assigned_count = 0
        for feature in features:
            filepath = feature['filepath']
            min_distance = float('inf')
            best_playlist = None
            
            # Find closest playlist
            for playlist in playlists:
                playlist_id, name, bpm, centroid = playlist
                
                # Skip playlists with missing data
                if None in (bpm, centroid):
                    continue
                
                # Calculate distance using weighted features
                distance = (
                    0.7 * abs(feature['bpm'] - bpm) +
                    0.3 * abs(feature['centroid'] - centroid)
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_playlist = (playlist_id, name)
            
            # Assign to best playlist
            if best_playlist:
                playlist_id, name = best_playlist
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO playlist_songs (playlist_id, filepath) VALUES (?, ?)",
                        (playlist_id, filepath))
                    assigned_count += 1
                    logger.debug(f"Assigned {os.path.basename(filepath)} to playlist {name}")
                except sqlite3.IntegrityError:
                    # Already assigned, skip
                    pass
        
        conn.commit()
        conn.close()
        return assigned_count

    def save_playlists(self, playlists, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0

        for name, songs in playlists.items():
            if not songs:
                continue

            try:
                host_songs = []
                for song in songs:
                    host_path = self.convert_to_host_path(song)
                    # Verify path exists before adding to playlist
                    if os.path.exists(song):  # Check container path
                        host_songs.append(host_path)
                
                if not host_songs:
                    continue
                
                playlist_path = os.path.join(output_dir, f"{sanitize_filename(name)}.m3u")
                with open(playlist_path, 'w') as f:
                    f.write("\n".join(host_songs))
                saved_count += 1
                logger.info(f"Saved playlist '{name}' with {len(host_songs)} tracks")
            except Exception:
                pass

        if saved_count:
            logger.info(f"Saved {saved_count} playlists")
        else:
            logger.warning("No playlists saved")

        all_failed = list(set(self.failed_files))
        
        if all_failed:
            failed_path = os.path.join(output_dir, "Failed_Files.m3u")
            try:
                host_failed = [self.convert_to_host_path(p) for p in all_failed]
                with open(failed_path, 'w') as f:
                    f.write("\n".join(host_failed))
                logger.info(f"Saved {len(all_failed)} failed/missing files")
            except Exception:
                pass

    def _update_library_state(self):
        """Update library state after processing"""
        current_state = self._scan_music_directory(self.container_music_dir)
        self._save_library_state(current_state)

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
    parser.add_argument('--incremental', action='store_true', 
                       help='Incremental update mode (only process changes)')
    args = parser.parse_args()

    logger.info("Starting playlist generation")
    logger.info(f"Music directory: {args.music_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    generator = PlaylistGenerator()
    generator.host_music_dir = args.host_music_dir.rstrip('/')

    start_time = time.time()
    try:
        missing_in_db = generator.cleanup_database()
        if missing_in_db:
            logger.warning(f"Removed {len(missing_in_db)} missing files from database")
            generator.failed_files.extend(missing_in_db)

        if args.incremental:
            logger.info("Running in incremental mode")
            # Load previous state
            previous_state = generator._load_library_state()
            if not previous_state:
                logger.warning("No previous state found. Doing full run.")
            else:
                # Scan current directory
                current_state = generator._scan_music_directory(args.music_dir)
                
                # Compute changes
                changes = generator._get_file_changes(current_state, previous_state)
                
                # Process incremental update if changes found
                if any(changes.values()):
                    logger.info(f"Changes detected: {len(changes['added'])} added, "
                               f"{len(changes['removed'])} removed, "
                               f"{len(changes['modified'])} modified")
                    generator.incremental_update(changes, args.output_dir)
                    return
                else:
                    logger.info("No changes detected")
                    return

        # Full processing mode
        if args.use_db:
            features = generator.get_all_features_from_db()
        else:
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
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()