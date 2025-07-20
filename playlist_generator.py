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
import resource
import coloredlogs

# Enhanced logging with colors
logger = logging.getLogger(__name__)
coloredlogs.install(
    level='INFO',
    logger=logger,
    fmt='%(asctime)s %(levelname)s %(message)s',
    field_styles={
        'levelname': {'color': 'cyan', 'bold': True},
    },
    level_styles={
        'debug': {'color': 'green'},
        'info': {'color': 39},
        'warning': {'color': 214},
        'error': {'color': 'red', 'bold': True},
        'critical': {'color': 'red', 'bold': True, 'background': 'white'},
    }
)

def sanitize_filename(name):
    name = re.sub(r'[^\w\-_]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

def safe_analyze(filepath):
    """Robust analysis with resource limits"""
    try:
        # Set resource limits to prevent crashes
        try:
            # 4GB memory limit, 60s CPU time
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
            resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        except:
            pass  # Not available on all systems

        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)
        if result and result[0] is not None:
            features = result[0]
            # Ensure critical features have values
            for key in ['bpm', 'centroid', 'duration', 'loudness', 'dynamics']:
                if features.get(key) is None:
                    features[key] = 0.0
            return features
        return None
    except Exception as e:
        logger.error(f"Critical error processing {filepath}: {str(e)}")
        return None
    except:
        logger.error(f"Unhandled exception in worker for {filepath}")
        return None

def process_file_worker(filepath):
    try:
        features = safe_analyze(filepath)
        return features, filepath
    except Exception as e:
        logger.error(f"Worker crashed for {filepath}: {str(e)}")
        return None, filepath

class PlaylistGenerator:
    def __init__(self):
        self.failed_files = []
        self.container_music_dir = ""
        self.host_music_dir = ""
        self.cache_file = os.path.join(os.getenv('CACHE_DIR', '/app/cache'), 'audio_analysis.db')
        self.blacklisted_files = self.load_blacklist()

    def load_blacklist(self):
        """Load files that consistently cause failures"""
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        blacklist_file = os.path.join(cache_dir, 'blacklist.txt')
        if os.path.exists(blacklist_file):
            try:
                with open(blacklist_file, 'r') as f:
                    return set(f.read().splitlines())
            except Exception as e:
                logger.error(f"Error loading blacklist: {str(e)}")
        return set()

    def analyze_directory(self, music_dir, workers=None, force_sequential=False):
        file_list = []
        self.failed_files = []
        self.container_music_dir = music_dir.rstrip('/')

        # Build file list while skipping blacklisted files
        for root, _, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                    filepath = os.path.join(root, file)
                    if filepath not in self.blacklisted_files:
                        file_list.append(filepath)
                    else:
                        logger.info(f"Skipping blacklisted file: {filepath}")

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
        """Robust parallel processing with proper progress tracking"""
        results = []
        try:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=workers) as pool:
                # Submit all tasks
                async_results = [
                    pool.apply_async(process_file_worker, (filepath,))
                    for filepath in file_list
                ]
                
                # Create progress bar
                pbar = tqdm(total=len(file_list), desc="Processing files")
                processed_count = 0
                
                # Process results as they come in
                while processed_count < len(async_results):
                    for i, async_result in enumerate(async_results):
                        if async_result.ready():
                            try:
                                features, filepath = async_result.get(timeout=1)
                                if features:
                                    results.append(features)
                                else:
                                    self.failed_files.append(filepath)
                                
                                # Remove processed result
                                del async_results[i]
                                pbar.update(1)
                                processed_count += 1
                                break
                            
                            except mp.TimeoutError:
                                logger.error(f"Timeout processing {filepath}")
                                self.failed_files.append(filepath)
                                del async_results[i]
                                pbar.update(1)
                                processed_count += 1
                                break
                            
                            except Exception as e:
                                logger.error(f"Error retrieving result: {str(e)}")
                                self.failed_files.append(filepath)
                                del async_results[i]
                                pbar.update(1)
                                processed_count += 1
                                break
                    
                    # Small sleep to prevent busy waiting
                    time.sleep(0.1)
                
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
        """Enhanced playlist naming with more features"""
        # Ensure required features exist
        required_features = ['bpm', 'centroid', 'duration', 'loudness', 'dynamics']
        for feat in required_features:
            if feat not in features or features[feat] is None:
                logger.warning(f"Missing or None feature '{feat}' in centroid data")
                features[feat] = 0.0

        bpm = features['bpm']
        centroid = features['centroid']
        duration = features['duration']
        loudness = features['loudness']
        dynamics = features['dynamics']
        
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
            "Quiet" if loudness < -30 else
            "Moderate" if loudness < -20 else
            "Loud" if loudness < -10 else
            "VeryLoud"
        )
        
        # Dynamics descriptors
        dynamics_desc = (
            "Smooth" if dynamics < 2 else
            "Dynamic" if dynamics < 5 else
            "Intense"
        )
        
        # Mood descriptor based on multiple features
        mood = (
            "Ambient" if bpm < 75 and centroid < 1200 else
            "Downtempo" if bpm < 95 else
            "Dance" if bpm > 125 else
            "Energetic" if dynamics > 4 else
            "Balanced"
        )
        
        return f"{bpm_desc}_{timbre_desc}_{duration_desc}_{loudness_desc}_{dynamics_desc}_{mood}"

    def generate_playlists(self, features_list, num_playlists=8, output_dir=None):
        if not features_list:
            logger.warning("No features to cluster")
            return {}

        valid_features = []
        for feat in features_list:
            if not feat or 'filepath' not in feat:
                continue
                
            if os.path.exists(feat['filepath']):
                # Handle None values for all features
                for key in ['bpm', 'centroid', 'duration', 'loudness', 'dynamics']:
                    if feat.get(key) is None:
                        feat[key] = 0.0
                valid_features.append(feat)
            else:
                logger.warning(f"File not found: {feat['filepath']}")
                self.failed_files.append(feat['filepath'])
                
        if not valid_features:
            logger.warning("No valid features after filtering")
            return {}
            
        df = pd.DataFrame(valid_features)
        
        # Cluster features - using more dimensions for better grouping
        cluster_features = ['bpm', 'centroid', 'loudness', 'dynamics']
        
        # Fill missing values
        for feat in cluster_features:
            if feat not in df.columns:
                df[feat] = 0.0
                
        df[cluster_features] = df[cluster_features].fillna(0)
        
        # Scale features
        features_scaled = StandardScaler().fit_transform(df[cluster_features].values.astype(np.float32))
        
        # Dynamic cluster count based on library size
        n_clusters = min(max(8, num_playlists * 2), max(20, len(df) // 20))
        
        # Cluster with MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(500, len(df)),
            n_init='auto',
            max_iter=50
        )
        df['cluster'] = kmeans.fit_predict(features_scaled)
        
        playlists = {}
        for cluster in df['cluster'].unique():
            cluster_songs = df[df['cluster'] == cluster]
            
            # Skip very small clusters
            if len(cluster_songs) < 30:
                continue
                
            # Calculate cluster centroid for naming
            centroid = cluster_songs[cluster_features].mean().to_dict()
            name = sanitize_filename(self.generate_playlist_name(centroid))
            
            if name not in playlists:
                playlists[name] = []
            playlists[name].extend(cluster_songs['filepath'].tolist())
        
        # Handle orphaned tracks (small clusters)
        orphaned = df[~df['filepath'].isin([song for songs in playlists.values() for song in songs])]
        if not orphaned.empty:
            logger.info(f"Assigning {len(orphaned)} orphaned tracks to nearest playlists")
            for _, track in orphaned.iterrows():
                min_distance = float('inf')
                best_playlist = None
                
                for playlist_name, songs in playlists.items():
                    # Calculate distance to playlist centroid
                    playlist_tracks = df[df['filepath'].isin(songs)]
                    centroid = playlist_tracks[cluster_features].mean().to_dict()
                    
                    distance = sum(
                        abs(track[feat] - centroid[feat]) for feat in cluster_features
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_playlist = playlist_name
                
                if best_playlist:
                    playlists[best_playlist].append(track['filepath'])
        
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

            host_songs = []
            missing_count = 0
            
            for song in songs:
                host_path = self.convert_to_host_path(song)
                if os.path.exists(song):  # Verify file exists
                    host_songs.append(host_path)
                else:
                    missing_count += 1
                    logger.warning(f"File not found: {song}")
            
            if not host_songs:
                logger.warning(f"Skipping empty playlist: {name}")
                continue
                
            playlist_path = os.path.join(output_dir, f"{name}.m3u")
            with open(playlist_path, 'w') as f:
                f.write("\n".join(host_songs))
            
            saved_count += 1
            logger.info(f"Saved {name} with {len(host_songs)} tracks")
            
            if missing_count:
                logger.warning(f"  Missing {missing_count} files in playlist")

        logger.info(f"Saved {saved_count} playlists total")

        # Save failed files list
        if self.failed_files:
            failed_path = os.path.join(output_dir, "Failed_Files.m3u")
            with open(failed_path, 'w') as f:
                host_failed = [self.convert_to_host_path(p) for p in self.failed_files]
                f.write("\n".join(host_failed))
            logger.info(f"Saved {len(self.failed_files)} failed/missing files")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Music Playlist Generator')
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

    generator = PlaylistGenerator()
    generator.host_music_dir = args.host_music_dir.rstrip('/')

    start_time = time.time()
    try:
        missing_in_db = generator.cleanup_database()
        if missing_in_db:
            logger.info(f"Removed {len(missing_in_db)} missing files from database")

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
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()