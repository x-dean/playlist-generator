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
        with tqdm(file_list, desc="Analyzing files", 
                bar_format="{l_bar}{bar:40}{r_bar}", 
                file=sys.stdout) as pbar:
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
                    logger.error(f"Error processing {filepath}: {str(e)}")
        return results

    def _process_parallel(self, file_list, workers):
        results = []
        try:
            logger.info(f"Starting multiprocessing with {workers} workers")
            ctx = mp.get_context('spawn')
            
            with ctx.Pool(processes=workers) as pool:
                chunksize = min(10, len(file_list)//workers + 1)
                with tqdm(total=len(file_list), desc="Processing files",
                         bar_format="{l_bar}{bar:40}{r_bar}",
                         file=sys.stdout) as pbar:
                    
                    # Process in chunks for better progress tracking
                    for i in range(0, len(file_list), chunksize):
                        chunk = file_list[i:i+chunksize]
                        for features, filepath in pool.map(process_file_worker, chunk):
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

    def generate_playlists(self, features_list, num_playlists=5, chunk_size=1000, output_dir=None):
        """Generate playlists with comprehensive error handling for musical keys"""
        playlists = {}
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        try:
            if not features_list:
                logger.warning("No features provided for playlist generation")
                return playlists

            # All logic inside this try block
            data = []
            valid_tracks = 0
            skipped_tracks = 0

            for f in features_list:
                if not f or 'filepath' not in f:
                    skipped_tracks += 1
                    continue

                try:
                    raw_key = f.get('key', -1)
                    key = -1
                    if isinstance(raw_key, str):
                        try:
                            key = keys.index(raw_key.strip().upper().replace('B', 'Bb').replace('FL', '#'))
                        except ValueError:
                            if raw_key.upper() == 'DB':
                                key = keys.index('C#')
                            elif raw_key.upper() == 'EB':
                                key = keys.index('D#')
                            elif raw_key.upper() == 'GB':
                                key = keys.index('F#')
                            elif raw_key.upper() == 'AB':
                                key = keys.index('G#')
                            elif raw_key.upper() == 'BB':
                                key = keys.index('A#')
                            else:
                                key = -1
                    else:
                        try:
                            key = int(raw_key) if raw_key is not None else -1
                        except (TypeError, ValueError):
                            key = -1

                    key = key if 0 <= key < len(keys) else -1

                    raw_scale = f.get('scale', 0)
                    scale = 0
                    if isinstance(raw_scale, str):
                        scale = 1 if raw_scale.lower() in ['major', 'maj', '1'] else 0
                    else:
                        try:
                            scale = int(raw_scale) if raw_scale is not None else 0
                        except (TypeError, ValueError):
                            scale = 0

                    data.append({
                        'filepath': str(f['filepath']),
                        'bpm': max(0, float(f.get('bpm', 0))),
                        'centroid': max(0, float(f.get('centroid', 0))),
                        'danceability': min(1.0, max(0, float(f.get('danceability', 0)))),
                        'key': key,
                        'scale': 1 if scale == 1 else 0,
                        'duration': max(0, float(f.get('duration', 0))),
                        'loudness': float(f.get('loudness', 0)),
                        'zcr': float(f.get('zcr', 0))
                    })
                except Exception as e:
                    logger.debug(f"Skipping track {f.get('filepath','unknown')}: {str(e)}")
                    continue

            if not data:
                logger.warning("No valid tracks after filtering")
                return playlists

            df = pd.DataFrame(data)

            # 2. Enhanced feature processing with more musical characteristics
            try:
                cluster_features = ['bpm', 'centroid', 'danceability', 'loudness', 'zcr']
                weights = np.array([1.5, 1.0, 1.2, 0.8, 0.7])  # Adjusted weights
                
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(df[cluster_features])
                weighted_features = scaled_features * weights

                # Dynamic cluster sizing based on dataset size and variance
                variance_explained = 0.9  # Aim to explain 90% of variance
                max_clusters = min(50, len(df))
                
                # Find optimal cluster count using explained variance
                distortions = []
                cluster_range = range(2, max_clusters+1)
                for k in cluster_range:
                    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=min(500, len(df)))
                    kmeans.fit(weighted_features)
                    distortions.append(kmeans.inertia_)
                
                # Find elbow point where adding clusters doesn't help much
                optimal_clusters = 2
                for i in range(1, len(distortions)):
                    if (distortions[i-1] - distortions[i])/distortions[i-1] < 0.1:
                        optimal_clusters = cluster_range[i]
                        break

                logger.info(f"Using {optimal_clusters} clusters based on variance analysis")
                
                # Final clustering
                kmeans = MiniBatchKMeans(
                    n_clusters=optimal_clusters,
                    random_state=42,
                    batch_size=min(500, len(df))
                )
                df['cluster'] = kmeans.fit_predict(weighted_features)

                # 3. Improved cluster merging with musical similarity
                MIN_TRACKS = max(5, len(df)//15)  # More flexible minimum size
                
                while True:
                    cluster_counts = df['cluster'].value_counts()
                    small_clusters = cluster_counts[cluster_counts < MIN_TRACKS].index.tolist()
                    
                    if not small_clusters:
                        break
                        
                    # For each small cluster, find most similar cluster to merge with
                    for cluster in small_clusters:
                        # Get centroid of current small cluster
                        small_center = kmeans.cluster_centers_[cluster]
                        
                        # Calculate similarity to all other clusters using multiple features
                        similarities = []
                        for other_cluster in set(df['cluster'].unique()) - {cluster}:
                            other_center = kmeans.cluster_centers_[other_cluster]
                            
                            # Weighted similarity score (higher is more similar)
                            similarity = 1 / (1 + np.linalg.norm(
                                (small_center - other_center) * weights
                            ))
                            similarities.append((other_cluster, similarity))
                        
                        if not similarities:
                            continue
                            
                        # Find most similar cluster
                        most_similar = max(similarities, key=lambda x: x[1])[0]
                        
                        # Merge clusters
                        df.loc[df['cluster'] == cluster, 'cluster'] = most_similar
                        logger.debug(f"Merged cluster {cluster} (size {cluster_counts[cluster]}) "
                                f"with {most_similar} (similarity {max(similarities, key=lambda x: x[1])[1]:.2f})")

                # Generate final playlists with quality checks
                for cluster, group in df.groupby('cluster'):
                    if len(group) < MIN_TRACKS:
                        continue
                        
                    # Calculate median features for naming
                    centroid = {
                        'bpm': group['bpm'].median(),
                        'centroid': group['centroid'].median(),
                        'danceability': group['danceability'].median(),
                        'duration': group['duration'].sum(),
                        'key': group['key'].mode()[0] if not group['key'].mode().empty else -1,
                        'scale': group['scale'].mode()[0] if not group['scale'].mode().empty else 0
                    }
                    
                    # Calculate cluster quality score
                    quality = min(1.0, len(group)/50) * 0.5  # Size component
                    quality += 1 - (group['bpm'].std()/50) * 0.3  # BPM consistency
                    quality += 1 - (group['danceability'].std()/0.3) * 0.2  # Danceability consistency
                    
                    name = self.generate_playlist_name(centroid)
                    
                    # Only keep high quality playlists
                    if quality > 0.6:
                        playlists[name] = {
                            'tracks': group['filepath'].tolist(),
                            'size': len(group),
                            'quality': quality,
                            'features': centroid
                        }
                        logger.info(f"Created playlist {name} with {len(group)} tracks (quality: {quality:.2f})")
                    else:
                        logger.debug(f"Skipping low quality cluster {name} (quality: {quality:.2f})")

                logger.info(f"Generated {len(playlists)} high-quality playlists from {len(df)} tracks")
                
            except Exception as e:
                logger.error(f"Clustering failed: {str(e)}")
                logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"Playlist generation failed: {str(e)}")
            logger.error(traceback.format_exc())
        
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

        for name, playlist_data in playlists.items():
            songs = playlist_data['tracks']
            if not songs:
                continue

            host_songs = [self.convert_to_host_path(song) for song in songs]
            
            playlist_path = os.path.join(output_dir, f"{name}.m3u")
            with open(playlist_path, 'w') as f:
                f.write("\n".join(host_songs))
            saved_count += 1
            logger.info(f"Saved {name} with {len(host_songs)} tracks (quality: {playlist_data['quality']:.2f})")

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