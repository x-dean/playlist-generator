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
import sqlite3

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
    name = re.sub(r'[^\w\-_]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

def process_file_worker(filepath):
    try:
        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)
        if result and result[0] is not None:  # FIX: Added check for None features
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
        return results

    def _process_parallel(self, file_list, workers):
        results = []
        try:
            logger.info(f"Starting multiprocessing pool with {workers} workers")
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=workers) as pool:
                async_results = [
                    pool.apply_async(process_file_worker, (filepath,))
                    for filepath in file_list
                ]
                
                pbar = tqdm(total=len(file_list), desc="Processing files")
                for i, async_result in enumerate(async_results):
                    try:
                        features, filepath = async_result.get(timeout=600)
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
                    
                    if i % 10 == 0 or not features:
                        pbar.update(1)
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

    def generate_playlist_name(self, features):
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Safely get all features with proper type conversion
        key_idx = int(features.get('key', -1))
        scale = int(features.get('scale', 0))
        bpm = float(features.get('bpm', 0))
        centroid = float(features.get('centroid', 0))
        danceability = float(features.get('danceability', 0))
        duration = float(features.get('duration', 0))
        
        # Key and scale
        key_str = keys[key_idx] if 0 <= key_idx < len(keys) else 'Keyless'
        scale_str = "Maj" if scale == 1 else "Min"
        
        # BPM categories (more musical terms)
        bpm_desc = (
            "Largo" if bpm < 50 else
            "Adagio" if bpm < 70 else
            "Moderato" if bpm < 90 else
            "Allegro" if bpm < 120 else
            "Vivace" if bpm < 150 else
            "Presto"
        )
        
        # Timbre descriptions (more accurate)
        timbre_desc = (
            "SubBass" if centroid < 150 else
            "Dark" if centroid < 300 else
            "Warm" if centroid < 1000 else
            "Neutral" if centroid < 2000 else
            "Bright" if centroid < 4000 else
            "Crisp" if centroid < 6000 else
            "Shrill"
        )
        
        # Danceability (more descriptive)
        dance_desc = (
            "Static" if danceability < 0.3 else
            "Pulse" if danceability < 0.5 else
            "Groove" if danceability < 0.7 else
            "Dance" if danceability < 0.85 else
            "Energetic"
        )
        
        # Duration categories
        duration_desc = (
            "Short" if duration < 120 else
            "Medium" if duration < 240 else
            "Long" if duration < 360 else
            "Epic"
        )
        
        return f"{key_str}{scale_str}_{bpm_desc}_{timbre_desc}_{dance_desc}_{duration_desc}"

    def generate_playlists(self, features_list, num_playlists=5, chunk_size=1000, output_dir=None):
        if not features_list:
            return {}

        # Convert to DataFrame with proper types
        df = pd.DataFrame([{
            'filepath': f['filepath'],
            'bpm': float(f.get('bpm', 0)),
            'centroid': float(f.get('centroid', 0)),
            'danceability': float(f.get('danceability', 0)),
            'key': int(f.get('key', -1)),
            'scale': int(f.get('scale', 0)),
            'duration': float(f.get('duration', 0))
        } for f in features_list if f and 'filepath' in f])

        if df.empty:
            return {}

        # Feature weights - adjust these based on importance
        feature_weights = np.array([1.5, 1.0, 1.2])  # bpm, centroid, danceability
        cluster_features = ['bpm', 'centroid', 'danceability']
        
        # 1. Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[cluster_features])
        
        # 2. Apply weights while preserving standardization
        weighted_features = features_scaled * feature_weights
        
        # Determine optimal cluster count
        min_clusters = min(5, len(df))
        max_clusters = min(50, len(df))
        optimal_clusters = max(min_clusters, min(max_clusters, len(df)//10))

        # Perform clustering
        kmeans = MiniBatchKMeans(
            n_clusters=optimal_clusters,
            random_state=42,
            batch_size=min(500, len(df))
        df['cluster'] = kmeans.fit_predict(weighted_features)

        from sklearn.metrics import silhouette_score
        if len(df['cluster'].unique()) > 1:
            score = silhouette_score(features_scaled, df['cluster'])
            logger.info(f"Clustering quality (silhouette score): {score:.2f}")

        # Merge small clusters (less than 5% of total files)
        MIN_CLUSTER_SIZE = max(5, len(df) * 0.05)
        cluster_counts = df['cluster'].value_counts()
        small_clusters = cluster_counts[cluster_counts < MIN_CLUSTER_SIZE].index.tolist()

        if small_clusters:
            # Find nearest large cluster for each small cluster
            cluster_centers = kmeans.cluster_centers_
            for small_cluster in small_clusters:
                # Find nearest large cluster
                distances = np.linalg.norm(
                    cluster_centers[small_cluster] - cluster_centers, 
                    axis=1
                )
                # Set distance to self as infinity
                distances[small_cluster] = np.inf
                nearest_cluster = np.argmin(distances)
                # Reassign
                df.loc[df['cluster'] == small_cluster, 'cluster'] = nearest_cluster

        # Generate playlists
        playlists = {}
        for cluster in df['cluster'].unique():
            cluster_songs = df[df['cluster'] == cluster]
            
            # Get centroid features
            centroid = {
                'bpm': cluster_songs['bpm'].median(),
                'centroid': cluster_songs['centroid'].median(),
                'danceability': cluster_songs['danceability'].median(),
                'duration': cluster_songs['duration'].median(),
                'key': cluster_songs['key'].mode()[0] if not cluster_songs['key'].mode().empty else -1,
                'scale': cluster_songs['scale'].mode()[0] if not cluster_songs['scale'].mode().empty else 0
            }

            name = self.generate_playlist_name(centroid)
            playlists[name] = cluster_songs['filepath'].tolist()

        return {k: v for k, v in playlists.items() if len(v) >= MIN_CLUSTER_SIZE}

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
            
            playlist_path = os.path.join(output_dir, f"{name}.m3u")
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