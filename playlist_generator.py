#!/usr/bin/env python3
"""
Optimized Music Playlist Generator with Enhanced Features
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
import colorlog
import sys
import multiprocessing as mp
from multiprocessing import TimeoutError
import queue
import signal
from functools import partial

# === Color logger ===
def setup_logger(level=logging.INFO):
    logger = colorlog.getLogger('playlist_generator')
    logger.propagate = False  # Prevent duplicate logs
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)s] %(name)s: %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    ))
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler("playlist_generator.log", mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

# === Worker Initialization ===
def init_worker():
    """Initialize worker processes to ignore SIGINT"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def sanitize_filename(name):
    name = re.sub(r'[^\w\-_]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

def process_file_worker(filepath):
    """Global worker function with timeout handling"""
    def handler(signum, frame):
        raise TimeoutError(f"Timeout processing {filepath}")
    
    try:
        # Set timeout (3 minutes)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(180)
        
        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)
        signal.alarm(0)  # Disable the alarm
        
        if result and result[0] is not None:
            features = result[0]
            defaults = {'bpm': 0.0, 'centroid': 0.0, 'duration': 0.0,
                       'loudness': -20.0, 'dynamics': 0.0}
            features.update({k: features.get(k, v) for k, v in defaults.items()})
            return features, filepath
        return None, filepath
        
    except TimeoutError:
        logger.warning(f"Timeout processing {filepath}")
        return None, filepath
    except Exception as e:
        logger.error(f"Worker error for {filepath}: {str(e)}")
        return None, filepath
    
def _analyze_worker(filepath):
    """Standalone worker function that can be pickled"""
    try:
        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)
        if result and result[0] is not None:
            features = result[0]
            # Set defaults for missing features
            defaults = {'bpm': 0.0, 'centroid': 0.0, 'duration': 0.0,
                       'loudness': -20.0, 'dynamics': 0.0}
            for k, v in defaults.items():
                if features.get(k) is None:
                    features[k] = v
            return features, filepath
        return None, filepath
    except Exception as e:
        return None, filepath

class PlaylistGenerator:
    def __init__(self):
        self.failed_files = []
        self.container_music_dir = ""
        self.host_music_dir = ""
        self.cache_file = os.path.join(os.getenv('CACHE_DIR', '/app/cache'), 'audio_analysis.db')

    def _validate_audio_file(self, filepath):
        """Check if file exists and has valid audio extension"""
        valid_extensions = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
        return (os.path.isfile(filepath) and \
               (os.path.getsize(filepath) > 1024) and \
               (filepath.lower().endswith(valid_extensions)))
    
    def _check_stuck_process(self, processes, timeout=300):
        """Monitor processes for hangs"""
        start_time = time.time()
        while True:
            if all(not p.is_alive() for p in processes):
                return False
            if time.time() - start_time > timeout:
                return True
            time.sleep(5)

    def _cleanup_processes(self, processes):
        """Ensure all processes are terminated"""
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()

    def cleanup_database(self):
        """Remove entries for missing files from database"""
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

    def analyze_directory(self, music_dir, workers=None, force_sequential=False):
        """Scan directory for valid audio files"""
        self.container_music_dir = music_dir.rstrip('/')
        
        # Build file list with validation
        file_list = []
        for root, _, files in os.walk(music_dir):
            for file in files:
                filepath = os.path.join(root, file)
                if self._validate_audio_file(filepath):
                    file_list.append(filepath)

        logger.info(f"Found {len(file_list)} valid audio files")
        if not file_list:
            logger.error("No valid audio files found")
            return []

        # Determine worker count
        if workers is None:
            workers = max(1, mp.cpu_count() // 2)
            logger.info(f"Using {workers} workers")

        return (self._process_sequential(file_list) if force_sequential or workers <= 1
                else self._process_parallel(file_list, workers))

    def _process_sequential(self, file_list):
        results = []
        pbar = tqdm(file_list, desc="Processing files")
        for filepath in pbar:
            pbar.set_postfix(file=os.path.basename(filepath)[:20])
            features, _ = process_file_worker(filepath)  # Use the same worker function
            if features:
                results.append(features)
            else:
                self.failed_files.append(filepath)
        return results

    def validate_audio_file(filepath):
        """Quick check if file is valid before processing"""
        try:
            return os.path.isfile(filepath) and os.path.getsize(filepath) > 1024
        except:
            return False

    def _process_parallel(self, file_list, workers):
        """Robust parallel processing with proper error handling"""
        results = []
        self.failed_files = []
        
        try:
            ctx = mp.get_context('spawn')
            with ctx.Pool(
                processes=workers,
                maxtasksperchild=50,
                initializer=init_worker
            ) as pool:
                # Process in chunks for better performance
                chunk_size = min(50, len(file_list)//workers + 1)
                
                with tqdm(total=len(file_list), desc="Processing files") as pbar:
                    for features, filepath in pool.imap_unordered(
                        process_file_worker,  # Use the global worker function
                        file_list,
                        chunksize=chunk_size
                    ):
                        if features:
                            results.append(features)
                        else:
                            self.failed_files.append(filepath)
                        pbar.update()
                        
            return results
        except Exception as e:
            logger.error(f"Parallel processing failed: {str(e)}")
            return self._process_sequential(file_list)

    def get_all_features_from_db(self):
        try:
            features = get_all_features()
            return features
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return []

    def convert_to_host_path(self, container_path):
        """Convert container path to host path for playlist files"""
        if not self.host_music_dir or not self.container_music_dir:
            return container_path
            
        # Simple path replacement
        if container_path.startswith(self.container_music_dir):
            rel_path = os.path.relpath(container_path, self.container_music_dir)
            return os.path.join(self.host_music_dir, rel_path)
            
        return container_path

    def generate_playlist_name(self, features):
        """Generate descriptive playlist name from features"""
        # Default values for missing features
        bpm = features.get('bpm', 0)
        centroid = features.get('centroid', 0)
        duration = features.get('duration', 0)
        loudness = features.get('loudness', -20)
        dynamics = features.get('dynamics', 0)
        
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
        
        # Mood descriptor based on multiple features
        mood = (
            "Ambient" if bpm < 75 and centroid < 1200 else
            "Downtempo" if bpm < 95 else
            "Dance" if bpm > 125 else
            "Energetic" if dynamics > 4 else
            "Balanced"
        )
        
        return f"{bpm_desc}_{timbre_desc}_{mood}"

    def generate_playlists(self, features_list, num_playlists=8, output_dir=None):
        if not features_list:
            logger.warning("No features to cluster")
            return {}

        # Create dataframe with validated features
        df = pd.DataFrame([
            {**f, **{k: 0.0 for k in ['bpm', 'centroid', 'loudness', 'dynamics'] if k not in f}}
            for f in features_list
        ])
        
        # Cluster features
        cluster_features = ['bpm', 'centroid', 'loudness', 'dynamics']
        features_scaled = StandardScaler().fit_transform(
            df[cluster_features].values.astype(np.float32)
        )
        
        # Smart cluster sizing
        min_clusters = min(8, len(df))
        max_clusters = min(20, len(df) // 10)
        n_clusters = max(min_clusters, min(max_clusters, num_playlists))
        
        # Efficient clustering
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(500, len(df)),
            n_init=3
        )
        df['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Create playlists
        playlists = {}
        for cluster_id in range(n_clusters):
            cluster_songs = df[df['cluster'] == cluster_id]
            if len(cluster_songs) > 0:
                centroid = cluster_songs[cluster_features].mean().to_dict()
                name = sanitize_filename(self.generate_playlist_name(centroid))
                playlists[name] = cluster_songs['filepath'].tolist()
        
        return playlists

    def save_playlists(self, playlists, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0

        for name, songs in playlists.items():
            if not songs:
                continue

            host_songs = []
            for song in songs:
                host_path = self.convert_to_host_path(song)
                host_songs.append(host_path)
                
            playlist_path = os.path.join(output_dir, f"{name}.m3u")
            with open(playlist_path, 'w') as f:
                f.write("\n".join(host_songs))
            
            saved_count += 1
            logger.info(f"Saved {name} with {len(host_songs)} tracks")

        logger.info(f"Saved {saved_count} playlists total")
        
        # Save failed files list
        if self.failed_files:
            failed_path = os.path.join(output_dir, "Failed_Files.m3u")
            with open(failed_path, 'w') as f:
                host_failed = [self.convert_to_host_path(p) for p in self.failed_files]
                f.write("\n".join(host_failed))
            logger.info(f"Saved {len(self.failed_files)} failed/missing files")

def main():
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--log_level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--music_dir', required=True, help='Music directory in container')
    parser.add_argument('--host_music_dir', required=True, help='Host music directory')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory')
    parser.add_argument('--num_playlists', type=int, default=8, help='Number of playlists')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of workers (default: auto)')
    parser.add_argument('--use_db', action='store_true', help='Use database only')
    parser.add_argument('--force_sequential', action='store_true', 
                       help='Force sequential processing')
    args = parser.parse_args()

    # Set up logger with requested level
    global logger
    logger = setup_logger(getattr(logging, args.log_level.upper(), logging.INFO))
    
    generator = PlaylistGenerator()
    generator.host_music_dir = args.host_music_dir.rstrip('/')
    logger.info(f"Starting playlist generation with {args.workers} workers")

    start_time = time.time()
    try:
        generator.cleanup_database()

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
            playlists = generator.generate_playlists(features, args.num_playlists)
            generator.save_playlists(playlists, args.output_dir)
        else:
            logger.error("No valid audio files available")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()