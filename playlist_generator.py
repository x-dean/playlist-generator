#!/usr/bin/env python3
"""
Optimized Music Playlist Generator with Enhanced Naming and Configurable Timeout
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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("playlist_generator.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

def sanitize_filename(name):
    """Convert to Linux-friendly filename"""
    name = re.sub(r'[^\w\-_]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

class PlaylistGenerator:
    def __init__(self, timeout_seconds=60):  # Increased default timeout
        self.failed_files = []
        self.container_music_dir = ""
        self.host_music_dir = ""
        self.timeout_seconds = timeout_seconds

    def analyze_directory(self, music_dir, workers=4, force_sequential=False):
        file_list = []
        self.failed_files = []
        self.container_music_dir = music_dir.rstrip('/')

        for root, _, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                    filepath = os.path.join(root, file)
                    try:
                        # Skip small or invalid files
                        if os.path.getsize(filepath) > 1024:
                            file_list.append(filepath)
                    except OSError:
                        continue

        logger.info(f"Found {len(file_list)} valid audio files")
        if not file_list:
            logger.warning("No valid audio files found")
            return []

        if force_sequential or workers <= 1:
            logger.info("Using sequential processing")
            return self._process_sequential(file_list)

        return self._process_parallel(file_list, workers)

    def _process_sequential(self, file_list):
        """Process files sequentially"""
        results = []
        for filepath in tqdm(file_list, desc="Processing files"):
            features, _ = self.process_file_worker(filepath)
            if features:
                results.append(features)
            else:
                self.failed_files.append(filepath)
        return results

    def _process_parallel(self, file_list, workers):
        """Process files in parallel"""
        results = []
        try:
            logger.info(f"Starting multiprocessing pool with {workers} workers")
            pool = mp.Pool(processes=min(workers, mp.cpu_count() // 2 or 1))
            for features, filepath in tqdm(
                pool.imap_unordered(self.process_file_worker, file_list),
                total=len(file_list),
                desc="Processing files"
            ):
                if features:
                    results.append(features)
                else:
                    self.failed_files.append(filepath)
            pool.close()
            pool.join()
            logger.info("Processing completed")
            return results
        except Exception as e:
            logger.error(f"Multiprocessing failed: {str(e)}")
            logger.error(traceback.format_exc())
            if 'pool' in locals():
                pool.terminate()
                pool.join()
            return self._process_sequential(file_list)

    def process_file_worker(self, filepath):
        """Worker function for processing individual files"""
        try:
            from analyze_music import audio_analyzer
            audio_analyzer.timeout_seconds = self.timeout_seconds
            result = audio_analyzer.extract_features(filepath)
            if result:  # Returns (features, from_cache, file_hash)
                return result[0], filepath
            return None, filepath
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return None, filepath

    def get_all_features_from_db(self):
        """Get all features from database cache"""
        try:
            return get_all_features()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return []

    def convert_to_host_path(self, container_path):
        """Convert container path to host path"""
        if not self.host_music_dir or not self.container_music_dir:
            return container_path

        container_path = os.path.normpath(container_path)
        container_music_dir = os.path.normpath(self.container_music_dir)

        if not container_path.startswith(container_music_dir):
            return container_path

        rel_path = os.path.relpath(container_path, container_music_dir)
        return os.path.join(self.host_music_dir, rel_path)

    def generate_playlist_name(self, features):
        """Generate descriptive playlist name based on features"""
        bpm = float(features['bpm'])
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
        
        centroid = float(features['centroid'])
        timbre_desc = (
            "Dark" if centroid < 800 else
            "Warm" if centroid < 1500 else
            "Mellow" if centroid < 2200 else
            "Bright" if centroid < 3000 else
            "Sharp" if centroid < 4000 else
            "VerySharp"
        )
        
        duration = float(features['duration'])
        duration_desc = (
            "Brief" if duration < 60 else
            "Short" if duration < 120 else
            "Medium" if duration < 180 else
            "Long" if duration < 240 else
            "VeryLong"
        )
        
        energy_level = min(10, int(float(features['beat_confidence']) * 12))
        
        mood = (
            "Ambient" if bpm < 75 and centroid < 1200 else
            "Downtempo" if bpm < 95 and energy_level < 4 else
            "Dance" if bpm > 125 and energy_level > 7 else
            "Dynamic" if centroid > 3000 and energy_level > 5 else
            "Balanced"
        )
        
        return f"{bpm_desc}_{timbre_desc}_{duration_desc}_E{energy_level}_{mood}"

    def generate_playlists(self, features_list, num_playlists=5, chunk_size=1000):
        """Generate playlists using clustering"""
        if not features_list:
            logger.warning("No features to cluster")
            return {}

        df = pd.DataFrame(features_list)
        
        # Prepare features
        numeric_cols = ['bpm', 'beat_confidence', 'centroid', 'duration']
        df[numeric_cols] = df[numeric_cols].fillna(0).astype(float)
        
        # Enhanced features
        df['energy'] = df['beat_confidence'] * 1.5
        df['tempo_timbre'] = (df['bpm']/200) * (df['centroid']/4000) * 2
        cluster_features = numeric_cols + ['energy', 'tempo_timbre']
        
        # Clustering
        kmeans = MiniBatchKMeans(
            n_clusters=min(num_playlists*2, len(df)),
            random_state=42,
            batch_size=min(100, len(df)),
            n_init=5,
            max_iter=100
        )
        
        features_scaled = StandardScaler().fit_transform(df[cluster_features])
        df['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Group into playlists
        playlists = {}
        for cluster in df['cluster'].unique():
            cluster_songs = df[df['cluster'] == cluster]
            centroid = cluster_songs[cluster_features].mean().to_dict()
            name = sanitize_filename(self.generate_playlist_name(centroid))
            
            if name not in playlists:
                playlists[name] = []
            playlists[name].extend(cluster_songs['filepath'].tolist())
        
        return {k:v for k,v in playlists.items() if len(v) >= 5}

    def save_playlists(self, playlists, output_dir):
        """Save playlists to files"""
        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0

        for name, songs in playlists.items():
            if not songs:
                continue

            playlist_path = os.path.join(output_dir, f"{name}.m3u")
            with open(playlist_path, 'w') as f:
                f.write("\n".join(songs))
            saved_count += 1
            logger.info(f"Saved {name} with {len(songs)} tracks")

        if saved_count:
            logger.info(f"Saved {saved_count} playlists")
        else:
            logger.warning("No playlists saved")

        if self.failed_files:
            failed_path = os.path.join(output_dir, "Failed_Files.m3u")
            with open(failed_path, 'w') as f:
                host_failed = [self.convert_to_host_path(p) for p in self.failed_files]
                f.write("\n".join(host_failed))
            logger.info(f"Saved {len(self.failed_files)} failed files")

def main():
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True, help='Music directory in container')
    parser.add_argument('--host_music_dir', required=True, help='Host music directory')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory')
    parser.add_argument('--num_playlists', type=int, default=8, help='Number of playlists')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() // 2), 
                       help='Number of workers')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Clustering chunk size')
    parser.add_argument('--timeout', type=int, default=60, 
                       help='Timeout for audio analysis in seconds')
    parser.add_argument('--use_db', action='store_true', help='Use database only')
    parser.add_argument('--force_sequential', action='store_true', 
                       help='Force sequential processing')
    args = parser.parse_args()

    generator = PlaylistGenerator(timeout_seconds=args.timeout)
    generator.host_music_dir = args.host_music_dir.rstrip('/')

    start_time = time.time()
    try:
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
                args.chunk_size
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