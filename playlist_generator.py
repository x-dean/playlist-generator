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
from tqdm import tqdm
from analyze_music import get_all_features
import numpy as np
import time
import traceback
import re
import sqlite3  # Added for database cleanup

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
    def __init__(self):
        self.failed_files = []
        self.container_music_dir = ""
        self.host_music_dir = ""
        self.cache_file = os.path.join(os.getenv('CACHE_DIR', '/app/cache'), 'audio_analysis.db')  # Added cache file path

    def analyze_directory(self, music_dir, workers=4, force_sequential=False):
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

        if force_sequential or workers <= 1:
            logger.info("Using sequential processing")
            return self._process_sequential(file_list)

        return self._process_parallel(file_list, workers)

    def _process_sequential(self, file_list):
        results = []
        for filepath in tqdm(file_list, desc="Processing files"):
            features, _ = process_file_worker(filepath)
            if features:
                results.append(features)
            else:
                self.failed_files.append(filepath)
        return results

    def _process_parallel(self, file_list, workers):
        results = []
        try:
            logger.info("Starting multiprocessing pool")
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=min(workers, mp.cpu_count() // 2 or 1)) as pool:
                async_results = [
                    pool.apply_async(process_file_worker, (filepath,))
                    for filepath in file_list
                ]
                
                for async_result in tqdm(
                    async_results,
                    total=len(file_list),
                    desc="Processing files"
                ):
                    try:
                        # Add 5-minute timeout per file
                        features, filepath = async_result.get(timeout=300)
                        if features:
                            results.append(features)
                        else:
                            self.failed_files.append(filepath)
                    except mp.TimeoutError:
                        logger.error(f"Timeout processing {filepath}")
                        self.failed_files.append(filepath)
                    except Exception as e:
                        logger.error(f"Error retrieving result: {str(e)}")
                        self.failed_files.append(filepath)
                        
            logger.info("Processing completed")
            return results
        except Exception as e:
            logger.error(f"Multiprocessing failed: {str(e)}")
            logger.error(traceback.format_exc())
            return self._process_sequential(file_list)

    def get_all_features_from_db(self):
        try:
            return get_all_features()
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
        # More granular BPM classification (8 levels)
        bpm = features['bpm']
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
        
        # More detailed timbre classification (6 levels)
        centroid = features['centroid']
        timbre_desc = (
            "Dark" if centroid < 800 else
            "Warm" if centroid < 1500 else
            "Mellow" if centroid < 2200 else
            "Bright" if centroid < 3000 else
            "Sharp" if centroid < 4000 else
            "VerySharp"
        )
        
        # Duration with more ranges (5 levels)
        duration = features['duration']
        duration_desc = (
            "Brief" if duration < 60 else
            "Short" if duration < 120 else
            "Medium" if duration < 180 else
            "Long" if duration < 240 else
            "VeryLong"
        )
        
        # More sensitive energy scaling (0-10)
        energy_level = min(10, int(features['beat_confidence'] * 12))
        
        # Enhanced mood detection
        mood = (
            "Ambient" if bpm < 75 and centroid < 1200 else
            "Downtempo" if bpm < 95 and energy_level < 4 else
            "Dance" if bpm > 125 and energy_level > 7 else
            "Dynamic" if centroid > 3000 and energy_level > 5 else
            "Balanced"
        )
        
        return f"{bpm_desc}_{timbre_desc}_{duration_desc}_E{energy_level}_{mood}"

    def generate_playlists(self, features_list, num_playlists=5, chunk_size=1000):
        if not features_list:
            logger.warning("No features to cluster")
            return {}

        # Filter out files that no longer exist
        valid_features = []
        for feat in features_list:
            if os.path.exists(feat['filepath']):
                valid_features.append(feat)
            else:
                logger.warning(f"File not found: {feat['filepath']}")
                self.failed_files.append(feat['filepath'])
                
        if not valid_features:
            logger.warning("No valid features after filtering")
            return {}
            
        df = pd.DataFrame(valid_features)
        
        # Create enhanced features
        numeric_cols = ['bpm', 'beat_confidence', 'centroid', 'duration']
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Add weighted combination features
        df['energy'] = df['beat_confidence'] * 1.5
        df['tempo_timbre'] = (df['bpm']/200) * (df['centroid']/4000) * 2
        cluster_features = numeric_cols + ['energy', 'tempo_timbre']
        
        # Improved clustering with more iterations
        kmeans = MiniBatchKMeans(
            n_clusters=min(num_playlists*2, len(df)),  # Start with more clusters
            random_state=42,
            batch_size=min(100, len(df)),
            n_init=5,    # More initializations
            max_iter=100  # More iterations
        )
        
        features_scaled = StandardScaler().fit_transform(df[cluster_features])
        df['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Group by playlist name instead of cluster number
        playlists = {}
        for cluster in df['cluster'].unique():
            cluster_songs = df[df['cluster'] == cluster]
            centroid = cluster_songs[cluster_features].mean().to_dict()
            name = sanitize_filename(self.generate_playlist_name(centroid))
            
            # Merge clusters with the same name
            if name not in playlists:
                playlists[name] = []
            playlists[name].extend(cluster_songs['filepath'].tolist())
        
        # Filter out very small playlists
        return {k:v for k,v in playlists.items() if len(v) >= 5}  # Minimum 5 tracks

    def cleanup_database(self):
        """Remove missing files from database"""
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

            # Convert container paths to host paths while preserving original paths
            host_songs = [self.convert_to_host_path(song) for song in songs]
            
            playlist_path = os.path.join(output_dir, f"{name}.m3u")
            with open(playlist_path, 'w') as f:
                f.write("\n".join(host_songs))
            saved_count += 1
            logger.info(f"Saved {name} with {len(host_songs)} tracks")

        if saved_count:
            logger.info(f"Saved {saved_count} playlists")
        else:
            logger.warning("No playlists saved")

        # Combine processing failures and missing DB entries
        all_failed = list(set(self.failed_files))
        
        if all_failed:
            failed_path = os.path.join(output_dir, "Failed_Files.m3u")
            with open(failed_path, 'w') as f:
                host_failed = [self.convert_to_host_path(p) for p in all_failed]
                f.write("\n".join(host_failed))
            logger.info(f"Saved {len(all_failed)} failed/missing files")

def process_file_worker(filepath):
    try:
        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)
        if result:  # Returns (features, from_cache, file_hash)
            return result[0], filepath  # Return only features and filepath
        return None, filepath
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None, filepath

def main():
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True, help='Music directory in container')
    parser.add_argument('--host_music_dir', required=True, help='Host music directory')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory')
    parser.add_argument('--num_playlists', type=int, default=8, help='Number of playlists')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() // 2), 
                       help='Number of workers')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Clustering chunk size')
    parser.add_argument('--use_db', action='store_true', help='Use database only')
    parser.add_argument('--force_sequential', action='store_true', 
                       help='Force sequential processing')
    args = parser.parse_args()

    generator = PlaylistGenerator()
    generator.host_music_dir = args.host_music_dir.rstrip('/')

    start_time = time.time()
    try:
        # Clean up database before processing
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