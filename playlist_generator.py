#!/usr/bin/env python3
"""
Optimized Music Playlist Generator with Enhanced Musical Analysis
"""

import pandas as pd
from sklearn.cluster import MiniBatchKMeans, DBSCAN
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

# Improved logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("playlist_generator.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

def sanitize_filename(name):
    name = re.sub(r'[^\w\-_]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

def process_file_worker(filepath):
    try:
        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)
        if result and result[0] is not None:
            features = result[0]
            # Ensure all critical features have values
            required_features = {
                'duration': 180,  # Default to 3 minutes
                'bpm': 100, 
                'centroid': 2000,
                'loudness': -15,
                'dynamics': 0,
                'rhythm_complexity': 0.5,
                'key': 'unknown',
                'scale': 'unknown',
                'key_confidence': 0
            }
            for key, default in required_features.items():
                if features.get(key) is None:
                    features[key] = default
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
        """Initialize playlist tracking database"""
        os.makedirs(os.path.dirname(self.playlist_db), exist_ok=True)
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                centroid_bpm REAL,
                centroid_centroid REAL,
                centroid_duration REAL,
                centroid_loudness REAL,
                centroid_dynamics REAL,
                centroid_rhythm REAL,
                centroid_key TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
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
            logger.info(f"Using automatic worker count: {workers}")

        if force_sequential or workers <= 1:
            logger.info("Using sequential processing")
            return self._process_sequential(file_list)

        return self._process_parallel(file_list, workers)

    def _process_sequential(self, file_list):
        results = []
        pbar = tqdm(file_list, desc="Processing files")
        for filepath in pbar:
            pbar.set_postfix(file=os.path.basename(filepath)[:20])
            features, _ = process_file_worker(filepath)
            if features:
                results.append(features)
            else:
                self.failed_files.append(filepath)
            pbar.update(1)  # Ensure progress bar updates per file
        return results

    def _process_parallel(self, file_list, workers):
        results = []
        try:
            logger.info(f"Starting multiprocessing pool with {workers} workers")
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=workers) as pool:
                # Use imap_unordered for better progress tracking
                pbar = tqdm(total=len(file_list), desc="Processing files")
                for features, filepath in pool.imap_unordered(process_file_worker, file_list):
                    if features:
                        results.append(features)
                    else:
                        self.failed_files.append(filepath)
                    pbar.update(1)  # Update for each processed file
                pbar.close()
            logger.info("Processing completed")
            return results
        except Exception as e:
            logger.error(f"Multiprocessing failed: {str(e)}")
            logger.error(traceback.format_exc())
            return self._process_sequential(file_list)

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

    def _categorize_mood(self, features):
        """Categorize mood based on harmonic and dynamic features"""
        scale = features.get('scale', 'unknown')
        bpm = features.get('bpm', 0)
        loudness = features.get('loudness', -20)
        
        if scale == 'minor':
            if bpm < 90: 
                return "Melancholic"
            return "Intense"
        else:
            if loudness < -15: 
                return "Calm"
            if bpm > 120: 
                return "Euphoric"
            return "Upbeat"

    def generate_playlist_name(self, features):
        """Generate meaningful playlist name using multiple features"""
        # Default values for missing features
        defaults = {
            'bpm': 100, 'centroid': 2000, 'duration': 180,
            'loudness': -15, 'rhythm_complexity': 0.5
        }
        for key, default in defaults.items():
            if features.get(key) is None:
                features[key] = default
                logger.warning(f"Using default value for missing feature: {key}")

        bpm = features['bpm']
        centroid = features['centroid']
        duration = features['duration']
        loudness = features['loudness']
        rhythm_complexity = features.get('rhythm_complexity', 0.5)
        scale = features.get('scale', 'major')
        
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
        
        # Duration descriptors
        duration_desc = (
            "Brief" if duration < 60 else
            "Short" if duration < 120 else
            "Medium" if duration < 180 else
            "Long" if duration < 240 else
            "VeryLong"
        )
        
        # Loudness descriptors
        loudness_desc = (
            "Whisper" if loudness < -30 else
            "Soft" if loudness < -20 else
            "Balanced" if loudness < -10 else
            "Loud" if loudness < -5 else 
            "EarBlasting"
        )
        
        # Rhythm complexity descriptors
        rhythm_desc = (
            "Steady" if rhythm_complexity < 0.3 else
            "Grooving" if rhythm_complexity < 0.6 else
            "Complex"
        )
        
        # Mood based on harmonic features
        mood = self._categorize_mood(features)
        
        # Key scale descriptor
        key_desc = "Major" if scale == "major" else "Minor"
        
        return f"{bpm_desc}_{timbre_desc}_{key_desc}_{rhythm_desc}_{mood}"

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
        
        # Process added and modified files - prioritize high-energy tracks
        files_to_process = changes['added'] + changes['modified']
        if not files_to_process:
            logger.info("No files to process in incremental update")
            return
        
        # Sort by likely energy (duration * bpm) for processing priority
        files_to_process.sort(key=lambda x: os.path.getsize(x), reverse=True)
        
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
            SELECT id, name, centroid_bpm, centroid_centroid, centroid_duration,
                   centroid_loudness, centroid_dynamics, centroid_rhythm
            FROM playlists
        """)
        playlists = cursor.fetchall()
        
        if not playlists:
            logger.warning("No existing playlists found for assignment")
            return 0
        
        # Circle of fifths mapping for key similarity
        circle_of_fifths = {'C':0, 'G':1, 'D':2, 'A':3, 'E':4, 'B':5,
                            'F#':6, 'C#':7, 'F':-1, 'Bb':-2, 'Eb':-3,
                            'Ab':-4, 'Db':-5, 'Gb':-6}
        
        assigned_count = 0
        for feature in features:
            filepath = feature['filepath']
            min_distance = float('inf')
            best_playlist = None
            
            # Extract key from centroid if available
            file_key = feature.get('key', 'C')
            file_key_num = circle_of_fifths.get(file_key.split('/')[0], 0)
            
            # Find closest playlist
            for playlist in playlists:
                playlist_id, name, bpm, centroid, duration, loudness, dynamics, rhythm = playlist
                
                # Skip playlists with missing data
                if None in (bpm, centroid, duration, loudness, dynamics, rhythm):
                    continue
                
                # Calculate distance using weighted features
                distance = (
                    0.4 * abs(feature['bpm'] - bpm) +
                    0.3 * abs(feature['centroid'] - centroid) +
                    0.1 * abs(feature['loudness'] - loudness) +
                    0.1 * abs(feature['dynamics'] - dynamics) +
                    0.1 * abs(feature.get('rhythm_complexity', 0.5) - rhythm)
                )
                
                # Add key distance if available
                if file_key != 'unknown':
                    # Extract base key from centroid_key format (e.g., "C-major")
                    centroid_key = name.split('_')[2] if '_' in name else 'C'
                    centroid_key_num = circle_of_fifths.get(centroid_key, 0)
                    key_distance = min(
                        abs(file_key_num - centroid_key_num),
                        12 - abs(file_key_num - centroid_key_num)
                    ) / 12.0
                    distance += 0.2 * key_distance
                
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

    def save_playlists_from_db(self, output_dir):
        """Save playlists based on current database state"""
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()
        
        # Get all playlists
        cursor.execute("SELECT id, name FROM playlists")
        playlists = cursor.fetchall()
        
        if not playlists:
            logger.warning("No playlists found in database")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0
        
        for playlist_id, name in playlists:
            # Get files for this playlist
            cursor.execute(
                "SELECT filepath FROM playlist_songs WHERE playlist_id = ?",
                (playlist_id,))
            filepaths = [row[0] for row in cursor.fetchall()]
            
            if not filepaths:
                logger.info(f"Skipping empty playlist: {name}")
                continue
            
            # Convert paths and save
            host_paths = [self.convert_to_host_path(p) for p in filepaths]
            playlist_path = os.path.join(output_dir, f"{sanitize_filename(name)}.m3u")
            with open(playlist_path, 'w') as f:
                f.write("\n".join(host_paths))
            
            saved_count += 1
            logger.info(f"Updated playlist {name} with {len(host_paths)} tracks")
        
        conn.close()
        
        if saved_count:
            logger.info(f"Saved {saved_count} updated playlists")
        else:
            logger.warning("No playlists saved")

    def _update_library_state(self):
        """Update library state after processing"""
        current_state = self._scan_music_directory(self.container_music_dir)
        self._save_library_state(current_state)

    def generate_playlists(self, features_list, num_playlists=5, chunk_size=1000, output_dir=None):
        if not features_list:
            logger.warning("No features to cluster")
            return {}

        valid_features = []
        for feat in features_list:
            if not feat or 'filepath' not in feat:
                continue
            if os.path.exists(feat['filepath']):
                # Ensure all required features exist
                required = ['bpm', 'centroid', 'duration', 'loudness', 
                        'dynamics', 'rhythm_complexity', 'key', 'scale']
                for key in required:
                    if feat.get(key) is None:
                        # Set default values for missing features
                        defaults = {
                            'duration': 180,
                            'bpm': 100,
                            'centroid': 2000,
                            'loudness': -15,
                            'dynamics': 0,
                            'rhythm_complexity': 0.5,
                            'key': 'unknown',
                            'scale': 'unknown'
                        }
                        feat[key] = defaults.get(key, 0)
                        logger.warning(f"Using default for missing {key} in {feat['filepath']}")
                valid_features.append(feat)
            else:
                logger.warning(f"File not found: {feat['filepath']}")
                self.failed_files.append(feat['filepath'])
                
        if not valid_features:
            logger.warning("No valid features after filtering")
            return {}
            
        logger.info(f"Clustering {len(valid_features)} tracks")
        
        df = pd.DataFrame(valid_features)
        
        # Circle of fifths mapping for key similarity
        circle_of_fifths = {'C':0, 'G':1, 'D':2, 'A':3, 'E':4, 'B':5,
                            'F#':6, 'C#':7, 'F':-1, 'Bb':-2, 'Eb':-3,
                            'Ab':-4, 'Db':-5, 'Gb':-6}
        
        # Convert key to numerical representation
        df['key_num'] = df['key'].apply(
            lambda k: circle_of_fifths.get(str(k).split('/')[0], 0) if k != 'unknown' else 0
        )
        
        # Convert scale to numerical (0 = minor, 1 = major)
        df['scale_num'] = df['scale'].apply(
            lambda s: 1 if s == 'major' else 0
        )
        
        # Define features for clustering
        cluster_features = [
            'bpm', 'centroid', 'loudness', 
            'dynamics', 'rhythm_complexity', 'key_num', 'scale_num'
        ]
        
        # Fill missing values
        for feat in cluster_features:
            if feat not in df.columns:
                df[feat] = 0.0
                logger.warning(f"Added missing cluster feature column: {feat}")
                
        df[cluster_features] = df[cluster_features].fillna(0)
        
        # Scale features
        features_array = df[cluster_features].values.astype(np.float32)
        features_scaled = StandardScaler().fit_transform(features_array)
        
        # Determine optimal cluster count
        n_clusters = min(max(5, num_playlists), len(df), 50)
        logger.info(f"Clustering {len(df)} tracks into {n_clusters} clusters")
        
        # Cluster using MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(500, len(df)),
            n_init=3,
            max_iter=50
        )
        df['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Collect all clusters including small ones
        all_clusters = {}
        for cluster in df['cluster'].unique():
            cluster_songs = df[df['cluster'] == cluster]
            all_clusters[cluster] = cluster_songs
            
        # Merge small clusters with their nearest neighbors
        merged_playlists = {}
        centroids = {}

        # Sort clusters by size descending
        sorted_clusters = sorted(all_clusters.items(), key=lambda x: len(x[1]), reverse=True)

        # Create a list to store clusters that couldn't be merged initially
        unmerged_clusters = []

        for cluster_id, cluster_songs in sorted_clusters:
            if len(cluster_songs) < min_playlist_size:
                # Find nearest cluster regardless of size
                closest_cluster = None
                min_distance = float('inf')
                
                # First check existing merged playlists
                for target_id, target_songs in merged_playlists.items():
                    dist = np.linalg.norm(
                        cluster_songs[cluster_features].mean().values -
                        target_songs[cluster_features].mean().values
                    )
                    if dist < min_distance:
                        min_distance = dist
                        closest_cluster = target_id
                
                # Then check other unmerged clusters
                if not closest_cluster:
                    for other_id, other_songs in unmerged_clusters:
                        if len(other_songs) >= min_playlist_size:  # Only merge into viable clusters
                            dist = np.linalg.norm(
                                cluster_songs[cluster_features].mean().values -
                                other_songs[cluster_features].mean().values
                            )
                            if dist < min_distance:
                                min_distance = dist
                                closest_cluster = other_id
                
                if closest_cluster is not None:
                    # Merge with closest cluster
                    if closest_cluster in merged_playlists:
                        merged_playlists[closest_cluster] = pd.concat([
                            merged_playlists[closest_cluster],
                            cluster_songs
                        ])
                    else:
                        # Create new merged cluster
                        merged_playlists[closest_cluster] = pd.concat([
                            all_clusters[closest_cluster],
                            cluster_songs
                        ])
                    logger.info(f"Merged cluster {cluster_id} ({len(cluster_songs)} tracks) "
                                f"into cluster {closest_cluster}")
                    continue
                
                # If no suitable cluster found, add to unmerged list for later processing
                unmerged_clusters.append((cluster_id, cluster_songs))
                continue

            # Create new playlist for this cluster
            centroid = cluster_songs[cluster_features].mean().to_dict()
            centroid['key'] = cluster_songs['key'].mode().iloc[0] if not cluster_songs['key'].mode().empty else 'C'
            centroid['scale'] = cluster_songs['scale'].mode().iloc[0] if not cluster_songs['scale'].mode().empty else 'major'
            
            name = sanitize_filename(self.generate_playlist_name(centroid))
            merged_playlists[name] = cluster_songs
            centroids[name] = centroid
            logger.info(f"Created playlist '{name}' with {len(cluster_songs)} tracks")

        # Process any remaining unmerged clusters
        for cluster_id, cluster_songs in unmerged_clusters:
            if not merged_playlists:
                # If no playlists exist yet, create one
                centroid = cluster_songs[cluster_features].mean().to_dict()
                centroid['key'] = cluster_songs['key'].mode().iloc[0] if not cluster_songs['key'].mode().empty else 'C'
                centroid['scale'] = cluster_songs['scale'].mode().iloc[0] if not cluster_songs['scale'].mode().empty else 'major'
                
                name = sanitize_filename(self.generate_playlist_name(centroid))
                merged_playlists[name] = cluster_songs
                centroids[name] = centroid
                continue
            
            # Find closest playlist among merged ones
            closest_playlist = None
            min_distance = float('inf')
            
            for playlist_name, playlist_songs in merged_playlists.items():
                dist = np.linalg.norm(
                    cluster_songs[cluster_features].mean().values -
                    playlist_songs[cluster_features].mean().values
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_playlist = playlist_name
            
            if closest_playlist:
                merged_playlists[closest_playlist] = pd.concat([
                    merged_playlists[closest_playlist],
                    cluster_songs
                ])
                logger.info(f"Merged unmerged cluster {cluster_id} ({len(cluster_songs)} tracks) "
                            f"into playlist '{closest_playlist}'")

        # Create final playlists
        playlists = {}
        min_playlist_size = 3  # Minimum tracks per playlist
        
        # First pass: Create playlists from merged clusters
        for name, cluster_songs in merged_playlists.items():
            # Final check to ensure minimum size
            if len(cluster_songs) < min_playlist_size:
                logger.info(f"Skipping small playlist '{name}' with {len(cluster_songs)} tracks")
                continue
                
            playlists[name] = cluster_songs['filepath'].tolist()
            logger.info(f"Final playlist '{name}' with {len(playlists[name])} tracks")
        
        # Track all assigned filepaths
        all_assigned = set()
        for tracks in playlists.values():
            all_assigned.update(tracks)

        # Get all filepaths from the original DataFrame
        all_tracks = set(df['filepath'].tolist())
        missing_tracks = all_tracks - all_assigned

        if missing_tracks:
            logger.warning(f"{len(missing_tracks)} tracks were not assigned to any playlist. Adding to 'Miscellaneous_Uncategorized'.")
            misc_playlist = "Miscellaneous_Uncategorized"
            playlists[misc_playlist] = list(missing_tracks)
            # Create a default centroid for the miscellaneous playlist
            centroids[misc_playlist] = {
                'bpm': 100,
                'centroid': 2000,
                'duration': 180,
                'loudness': -15,
                'dynamics': 0,
                'rhythm_complexity': 0.5,
                'key': 'unknown',
                'scale': 'unknown'
            }
        
        # Update playlist tracker
        self._update_playlist_tracker(playlists, centroids)
        
        # Update library state
        self._update_library_state()
        
        return playlists

    def cleanup_database(self):
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

        for name, songs in playlists.items():
            if not songs:
                continue

            host_songs = [self.convert_to_host_path(song) for song in songs]
            
            playlist_path = os.path.join(output_dir, f"{sanitize_filename(name)}.m3u")
            with open(playlist_path, 'w') as f:
                f.write("\n".join(host_songs))
            saved_count += 1
            logger.info(f"Saved {name} with {len(host_songs)} tracks")

        if saved_count:
            logger.info(f"Saved {saved_count} playlists")
        else:
            logger.warning("No playlists saved")

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
    parser.add_argument('--incremental', action='store_true', 
                       help='Incremental update mode (only process changes)')
    args = parser.parse_args()

    generator = PlaylistGenerator()
    generator.host_music_dir = args.host_music_dir.rstrip('/')

    start_time = time.time()
    try:
        missing_in_db = generator.cleanup_database()
        if missing_in_db:
            logger.info(f"Removed {len(missing_in_db)} missing files from database")
            generator.failed_files.extend(missing_in_db)

        if args.incremental:
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
                    
                    # Update state after processing
                    generator._update_library_state()
                    return
                else:
                    logger.info("No changes detected. Playlists are up to date.")
                    return

        # Full processing mode
        if args.use_db:
            logger.info("Using database features")
            features = generator.get_all_features_from_db()
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

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()