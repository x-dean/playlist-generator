#!/usr/bin/env python3
"""
Optimized Music Playlist Generator with Enhanced Features
"""

# Declare logger as global at module level
global logger
logger = None  # Initialize as None

# Imports
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
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# === Color logger ===
def setup_logger(level=logging.INFO) -> logging.Logger:
    """Configure and return a color logger with both console and file handlers."""
    global logger  # Declare we're using the global logger
    
    logger = colorlog.getLogger('playlist_generator')
    logger.propagate = False  # Prevent duplicate logs
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler with colors
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
    
    # File handler with detailed format
    file_handler = logging.FileHandler("playlist_generator.log", mode='w')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    ))
    logger.addHandler(file_handler)
    
    logger.setLevel(level)
    return logger

# Initialize the logger
logger = setup_logger()

# === Worker Initialization ===
def init_worker():
    """Initialize worker processes to ignore SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for filenames."""
    name = re.sub(r'[^\w\-_]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

def process_file_worker(filepath: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Worker function to process a single audio file with robust error handling."""
    try:
        from analyze_music import audio_analyzer
        result = audio_analyzer.extract_features(filepath)
        if result and result[0] is not None:
            features = result[0]
            # Ensure required features have defaults
            defaults = {
                'bpm': 0.0, 
                'centroid': 0.0, 
                'duration': 0.0,
                'loudness': -20.0, 
                'dynamics': 0.0,
                'filepath': filepath
            }
            for k, v in defaults.items():
                if features.get(k) is None:
                    features[k] = v
            return features, filepath
        return None, filepath
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None, filepath

class PlaylistGenerator:
    def __init__(self):
        self.failed_files = []
        self.host_music_dir = ""   # Will be set from CLI args
        self.host_output_dir = ""  # Will be set from CLI args
        self.host_cache_dir = ""   # Will be set from CLI args
        self.cache_file = "/app/cache/audio_analysis.db"  # Fixed container path

    def _validate_audio_file(self, filepath: str) -> bool:
        """Check if file exists and has valid audio extension."""
        valid_extensions = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
        filepath = str(filepath)
        return (Path(filepath).is_file() and 
                Path(filepath).stat().st_size > 1024 and
                filepath.lower().endswith(valid_extensions))

    def _check_stuck_process(self, processes: List[mp.Process], timeout: int = 300) -> bool:
        """Monitor processes for hangs."""
        start_time = time.time()
        while True:
            if all(not p.is_alive() for p in processes):
                return False
            if time.time() - start_time > timeout:
                return True
            time.sleep(5)

    def _cleanup_processes(self, processes: List[mp.Process]):
        """Ensure all processes are terminated properly."""
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()

    def cleanup_database(self) -> List[str]:
        """Remove entries for missing files from database."""
        try:
            conn = sqlite3.connect(self.cache_file)
            cursor = conn.cursor()
            
            # Get all files from database
            cursor.execute("SELECT file_path FROM audio_features")
            db_files = {row[0] for row in cursor.fetchall()}
            
            # Check which files exist
            missing_files = [path for path in db_files if not Path(path).exists()]
            
            if missing_files:
                logger.info(f"Cleaning up {len(missing_files)} missing files from database")
                # Delete in batches to avoid SQL parameter limits
                batch_size = 500
                for i in range(0, len(missing_files), batch_size):
                    batch = missing_files[i:i+batch_size]
                    placeholders = ','.join(['?'] * len(batch))
                    cursor.execute(
                        f"DELETE FROM audio_features WHERE file_path IN ({placeholders})",
                        batch
                    )
                conn.commit()
                
            conn.close()
            return missing_files
        except Exception as e:
            logger.error(f"Database cleanup failed: {str(e)}")
            try:
                conn.close()
            except:
                pass
            return []

    def analyze_directory(self, music_dir: str, workers: Optional[int] = None, 
                         force_sequential: bool = False) -> List[Dict[str, Any]]:
        """Scan directory for valid audio files and extract features."""
        self.container_music_dir = str(Path(music_dir).resolve())
        
        # Build file list with validation
        file_list = []
        for root, _, files in os.walk(music_dir):
            for file in files:
                filepath = str(Path(root) / file)
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

    def _process_sequential(self, file_list: List[str]) -> List[Dict[str, Any]]:
        """Process files sequentially with progress bar."""
        results = []
        pbar = tqdm(file_list, desc="Processing files")
        for filepath in pbar:
            pbar.set_postfix(file=Path(filepath).name[:20])
            features, _ = process_file_worker(filepath)
            if features:
                results.append(features)
            else:
                self.failed_files.append(filepath)
                logger.debug(f"Failed to process: {filepath}")
        return results

    def _process_parallel(self, file_list: List[str], workers: int) -> List[Dict[str, Any]]:
        """Robust parallel processing with watchdog."""
        results = []
        self.failed_files = []
        
        try:
            ctx = mp.get_context('spawn')
            with ctx.Pool(
                processes=workers,
                initializer=init_worker,
                maxtasksperchild=50,  # Recycle workers periodically
            ) as pool:
                # Process in chunks for better performance
                chunk_size = min(100, len(file_list)//workers + 1)
                
                # Use imap_unordered for better progress tracking
                with tqdm(total=len(file_list), desc="Processing files") as pbar:
                    for features, filepath in pool.imap_unordered(
                        process_file_worker,
                        file_list,
                        chunksize=chunk_size
                    ):
                        if features:
                            results.append(features)
                        else:
                            self.failed_files.append(filepath)
                            logger.debug(f"Failed to process: {filepath}")
                        pbar.update()
                        
            return results
        except Exception as e:
            logger.error(f"Parallel processing failed: {str(e)}")
            return self._process_sequential(file_list)

    def get_all_features_from_db(self) -> List[Dict[str, Any]]:
        """Get all features from the database."""
        try:
            features = get_all_features()
            return features
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return []

    def convert_to_host_path(self, container_path: str) -> str:
        """Convert container path to host path using the mounted volumes mapping."""
        try:
            # Check standard mount points
            mount_points = {
                '/music': self.host_music_dir,
                '/output': self.host_output_dir,
                '/cache': self.host_cache_dir
            }
            
            for container_mount, host_path in mount_points.items():
                if container_path.startswith(container_mount):
                    relative_path = Path(container_path).relative_to(container_mount)
                    return str(Path(host_path) / relative_path)
            
            return container_path  # Return unchanged if not in mounted volumes
        except Exception as e:
            logger.warning(f"Path conversion failed for {container_path}: {str(e)}")
            return container_path

    def generate_playlist_name(self, features: Dict[str, Any]) -> str:
        """Generate descriptive playlist name from features."""
        # Get features with defaults
        bpm = features.get('bpm', 0)
        centroid = features.get('centroid', 0)
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

    def generate_playlists(self, features_list: List[Dict[str, Any]], 
                          num_playlists: int = 8, 
                          output_dir: Optional[str] = None) -> Dict[str, List[str]]:
        """Generate playlists by clustering audio features."""
        if not features_list:
            logger.warning("No features to cluster")
            return {}

        # Validate and filter features
        valid_features = []
        for f in features_list:
            if not isinstance(f, dict):
                continue
            if not f.get('filepath') or not Path(f['filepath']).exists():
                continue
            valid_features.append(f)
        
        if not valid_features:
            logger.warning("No valid features to cluster")
            return {}

        # Create dataframe with validated features
        df = pd.DataFrame([
            {**f, **{k: 0.0 for k in ['bpm', 'centroid', 'loudness', 'dynamics'] if k not in f}}
            for f in valid_features
        ])
        
        # Cluster features
        cluster_features = ['bpm', 'centroid', 'loudness', 'dynamics']
        features_scaled = StandardScaler().fit_transform(
            df[cluster_features].values.astype(np.float32))
        
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

    def save_playlists(self, playlists: Dict[str, List[str]], output_dir: str):
        """Save playlists to files in the specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_count = 0

        for name, songs in playlists.items():
            if not songs:
                continue

            host_songs = []
            for song in songs:
                host_path = self.convert_to_host_path(song)
                host_songs.append(host_path)
                
            playlist_path = output_path / f"{name}.m3u"
            try:
                with open(playlist_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(host_songs))
                saved_count += 1
                logger.info(f"Saved {name} with {len(host_songs)} tracks")
            except Exception as e:
                logger.error(f"Failed to save playlist {name}: {str(e)}")

        logger.info(f"Saved {saved_count} playlists total")
        
        # Save failed files list
        if self.failed_files:
            failed_path = output_path / "Failed_Files.m3u"
            try:
                with open(failed_path, 'w', encoding='utf-8') as f:
                    host_failed = [self.convert_to_host_path(p) for p in self.failed_files]
                    f.write("\n".join(host_failed))
                logger.info(f"Saved {len(self.failed_files)} failed/missing files")
            except Exception as e:
                logger.error(f"Failed to save failed files list: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--host_music_dir', required=True,
                      help='Host path corresponding to /music in container')
    parser.add_argument('--host_output_dir', required=True,
                      help='Host path corresponding to /app/playlists in container')
    parser.add_argument('--host_cache_dir', required=False,
                      help='Host path corresponding to /app/cache in container')
    parser.add_argument('--num_playlists', type=int, default=8, 
                       help='Number of playlists')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Number of workers (default: auto)')
    parser.add_argument('--use_db', action='store_true', 
                       help='Use database only')
    parser.add_argument('--force_sequential', action='store_true', 
                       help='Force sequential processing')
    parser.add_argument('--log_level', default='INFO', 
                       help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    args = parser.parse_args()

    # Initialize logger with the requested level
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # ===== FIXED CONTAINER PATHS =====
    container_music_dir = "/music"
    container_output_dir = "/app/playlists"
    container_cache_dir = "/app/cache"

    # Validate host path exists
    if not Path(args.host_music_dir).exists():
        logger.error(f"Host music directory does not exist: {args.host_music_dir}")
        logger.info(f"Expected container path: {container_music_dir}")
        logger.info("Verify your Docker volume mount: -v [host_path]:/music")
        sys.exit(1)

    # Validate container path exists (should be mounted from host)
    if not Path(container_music_dir).exists():
        logger.error(f"Container music directory missing: {container_music_dir}")
        logger.info(f"Mounted from host path: {args.host_music_dir}")
        logger.info("Check your Docker volume mounts")
        sys.exit(1)

    # Create output and cache directories if they don't exist
    Path(container_output_dir).mkdir(parents=True, exist_ok=True)
    Path(container_cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize with host paths
    generator = PlaylistGenerator()
    generator.host_music_dir = str(Path(args.host_music_dir).resolve())
    generator.host_output_dir = str(Path(args.host_output_dir).resolve())
    generator.host_cache_dir = str(Path(args.host_cache_dir).resolve()) if args.host_cache_dir else ""
    
    # Validate host paths exist
    required_dirs = {
        "Music": generator.host_music_dir,
        "Output": generator.host_output_dir
    }
    
    for name, path in required_dirs.items():
        if not Path(path).exists():
            logger.error(f"{name} directory not found: {path}")
            sys.exit(1)

    start_time = time.time()
    try:
        generator.cleanup_database()

        if args.use_db:
            logger.info("Using database features")
            features = generator.get_all_features_from_db()
        else:
            logger.info("Analyzing directory")
            features = generator.analyze_directory(
                container_music_dir,  # Use fixed container path
                args.workers,
                args.force_sequential
            )
        
        logger.info(f"Processed {len(features)} files, {len(generator.failed_files)} failed")
        
        if features:
            playlists = generator.generate_playlists(features, args.num_playlists)
            generator.save_playlists(playlists, container_output_dir)  # Use fixed container path
        else:
            logger.error("No valid audio files available")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
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