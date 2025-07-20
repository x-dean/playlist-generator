#!/usr/bin/env python3
"""
Optimized Music Playlist Generator with Enhanced Playlists
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

def process_file_worker(filepath):
    try:
        # Skip files that are too small
        if os.path.getsize(filepath) < 1024:
            logger.warning(f"Skipping small file: {os.path.basename(filepath)}")
            return None, filepath
            
        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)
        
        if result and result[0] is not None:
            return result[0], filepath
            
        logger.warning(f"Failed to process: {os.path.basename(filepath)}")
        return None, filepath
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        return None, filepath

class PlaylistGenerator:
    def __init__(self):
        self.failed_files = []
        self.container_music_dir = ""
        self.host_music_dir = ""
        self.cache_file = os.path.join(os.getenv('CACHE_DIR', '/app/cache'), 'audio_analysis.db')

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
            logger.info(f"Using {workers} workers (half of {mp.cpu_count()} cores)")

        if force_sequential or workers <= 1:
            logger.info("Using sequential processing")
            return self._process_sequential(file_list)

        return self._process_parallel(file_list, workers)

    def _process_sequential(self, file_list):
        results = []
        with tqdm(file_list, desc="Analyzing files") as pbar:
            for filepath in pbar:
                try:
                    features, _ = process_file_worker(filepath)
                    if features:
                        results.append(features)
                    else:
                        self.failed_files.append(filepath)
                    pbar.set_postfix_str(f"OK: {len(results)}, Failed: {len(self.failed_files)}")
                except Exception as e:
                    self.failed_files.append(filepath)
                    logger.error(f"Error processing {os.path.basename(filepath)}: {str(e)}")
        return results

    def _process_parallel(self, file_list, workers):
        results = []
        try:
            logger.info(f"Starting multiprocessing with {workers} workers")
            ctx = mp.get_context('spawn')
            
            # Process in batches
            batch_size = min(50, len(file_list))
            
            for i in range(0, len(file_list), batch_size):
                batch = file_list[i:i+batch_size]
                
                with ctx.Pool(processes=workers) as pool:
                    with tqdm(total=len(batch), desc=f"Processing batch {i//batch_size+1}",
                             bar_format="{l_bar}{bar:40}{r_bar}",
                             file=sys.stdout) as pbar:
                        
                        for features, filepath in pool.imap_unordered(process_file_worker, batch):
                            if features:
                                results.append(features)
                            else:
                                self.failed_files.append(filepath)
                            pbar.update(1)
                            pbar.set_postfix_str(f"OK: {len(results)}, Failed: {len(self.failed_files)}")
            
            logger.info(f"Processing completed - {len(results)} successful, {len(self.failed_files)} failed")
            return results
            
        except Exception as e:
            logger.error(f"Multiprocessing failed: {str(e)}")
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

    def generate_detailed_playlists(self, features_list):
        """Generate playlists with detailed feature combinations"""
        playlists = {}
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        energy_levels = ["Low", "Medium", "High"]
        mood_levels = ["Calm", "Moderate", "Energetic"]
        
        for features in features_list:
            if not features:
                continue
                
            try:
                # Feature extraction
                bpm = features.get('bpm', 0)
                centroid = features.get('centroid', 0)
                danceability = features.get('danceability', 0)
                key_idx = features.get('key', -1)
                scale = features.get('scale', 0)
                
                # BPM category
                bpm_cat = (
                    "Slow" if bpm < 80 else
                    "Medium" if bpm < 120 else
                    "Fast"
                )
                
                # Energy category
                energy_cat = energy_levels[min(2, int(danceability * 3))]
                
                # Mood category
                mood_cat = mood_levels[min(2, int(centroid / 1500))]
                
                # Key and scale
                key_name = keys[key_idx] if 0 <= key_idx < len(keys) else "Unknown"
                scale_name = "Major" if scale == 1 else "Minor"
                
                # Create multiple playlist combinations
                playlists.setdefault(f"{bpm_cat}_Tempo", []).append(features['filepath'])
                playlists.setdefault(f"{energy_cat}_Energy", []).append(features['filepath'])
                playlists.setdefault(f"{mood_cat}_Mood", []).append(features['filepath'])
                playlists.setdefault(f"{key_name}_{scale_name}", []).append(features['filepath'])
                playlists.setdefault(f"{bpm_cat}_{energy_cat}_{key_name}", []).append(features['filepath'])
                
            except Exception as e:
                logger.warning(f"Couldn't categorize features: {str(e)}")
        
        # Convert to playlist format
        return {
            name: {'tracks': tracks, 'size': len(tracks)}
            for name, tracks in playlists.items()
        }

    def cleanup_database(self):
        """Clean up database entries for missing files"""
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
    args = parser.parse_args()

    generator = PlaylistGenerator()
    generator.host_music_dir = args.host_music_dir.rstrip('/')

    start_time = time.time()
    try:
        missing_in_db = generator.cleanup_database()
        if missing_in_db:
            logger.info(f"Removed {len(missing_in_db)} missing files from database")
            generator.failed_files.extend(missing_in_db)

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
            # Generate both clustered and detailed playlists
            clustered_playlists = generator.generate_playlists(features, args.num_playlists)
            detailed_playlists = generator.generate_detailed_playlists(features)
            
            # Combine both playlist types
            all_playlists = {**clustered_playlists, **detailed_playlists}
            generator.save_playlists(all_playlists, args.output_dir)
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