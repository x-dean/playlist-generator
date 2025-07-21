#!/usr/bin/env python3
"""Optimized Music Playlist Generator with Time-Based Scheduling"""

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
import numpy as np
import time
import traceback
import re
import sqlite3
import datetime
from collections import defaultdict
from analyze_music import get_all_features

# Logging setup
def setup_colored_logging():
    """Configure colored logging"""
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

logger = setup_colored_logging()

def sanitize_filename(name):
    name = re.sub(r'[^\w\-_]', '_', name)
    return re.sub(r'_+', '_', name).strip('_')

def process_file_worker(filepath):
    try:
        if os.path.getsize(filepath) < 1024:
            logger.warning(f"Skipping small file: {filepath}")
            return None, filepath

        if not filepath.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
            return None, filepath

        from analyze_music import audio_analyzer
        features, _, _ = audio_analyzer.extract_features(filepath)
        return features, filepath
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None, filepath

def setup_playlist_db(cache_file):
    conn = sqlite3.connect(cache_file, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS playlists (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS playlist_tracks (
        playlist_id INTEGER,
        file_hash TEXT,
        added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(playlist_id) REFERENCES playlists(id),
        FOREIGN KEY(file_hash) REFERENCES audio_features(file_hash),
        PRIMARY KEY (playlist_id, file_hash)
    )""")
    conn.commit()
    conn.close()

class TimeBasedScheduler:
    def __init__(self):
        self.time_slots = {
            'Morning': (6, 12),
            'Afternoon': (12, 18),
            'Evening': (18, 22),
            'Late_Night': (22, 6)
        }
        self.feature_rules = {
            'Morning': {'min_bpm': 90, 'max_bpm': 120, 'min_danceability': 0.4},
            'Afternoon': {'min_bpm': 100, 'max_bpm': 130, 'min_danceability': 0.6},
            'Evening': {'min_bpm': 80, 'max_bpm': 110, 'max_danceability': 0.7},
            'Late_Night': {'max_bpm': 90, 'max_danceability': 0.4}
        }

    def get_current_time_slot(self):
        now = datetime.datetime.now().time()
        for slot, (start, end) in self.time_slots.items():
            if start < end:
                if start <= now.hour < end:
                    return slot
            else:
                if now.hour >= start or now.hour < end:
                    return slot
        return 'Afternoon'

    def filter_tracks(self, features_list, slot_name):
        rules = self.feature_rules.get(slot_name, {})
        filtered = []
        
        for track in features_list:
            if not track:
                continue
                
            valid = True
            bpm = track.get('bpm', 0)
            danceability = track.get('danceability', 0)
            
            if 'min_bpm' in rules and bpm < rules['min_bpm']:
                valid = False
            if valid and 'max_bpm' in rules and bpm > rules['max_bpm']:
                valid = False
            if valid and 'min_danceability' in rules and danceability < rules['min_danceability']:
                valid = False
            if valid and 'max_danceability' in rules and danceability > rules['max_danceability']:
                valid = False
                
            if valid:
                filtered.append(track)
                
        return filtered

    def generate_time_based_playlists(self, features_list):
        playlists = {}
        for slot_name in self.time_slots:
            tracks = self.filter_tracks(features_list, slot_name)
            playlists[f"TimeSlot_{slot_name}"] = {
                'tracks': [t['filepath'] for t in tracks],
                'features': {'type': 'time_based', 'slot': slot_name}
            }
        return playlists

class PlaylistGenerator:
    def __init__(self):
        self.failed_files = []
        self.cache_file = os.path.join(os.getenv('CACHE_DIR', '/app/cache'), 'audio_analysis.db')
        self.scheduler = TimeBasedScheduler()
        setup_playlist_db(self.cache_file)

    def analyze_directory(self, music_dir, workers=None):
        file_list = []
        self.failed_files = []

        for root, _, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
                    file_list.append(os.path.join(root, file))

        logger.info(f"Found {len(file_list)} audio files")
        if not file_list:
            return []

        workers = workers or max(1, mp.cpu_count() // 2)
        return self._process_parallel(file_list, workers) if workers > 1 else self._process_sequential(file_list)

    def _process_sequential(self, file_list):
        results = []
        with tqdm(file_list, desc="Analyzing files") as pbar:
            for filepath in pbar:
                features, _ = process_file_worker(filepath)
                if features:
                    results.append(features)
                else:
                    self.failed_files.append(filepath)
                pbar.set_postfix_str(f"OK: {len(results)}, Failed: {len(self.failed_files)}")
        return results

    def _process_parallel(self, file_list, workers):
        results = []
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=workers) as pool:
            with tqdm(total=len(file_list), desc="Processing files") as pbar:
                for features, filepath in pool.imap_unordered(process_file_worker, file_list):
                    if features:
                        results.append(features)
                    else:
                        self.failed_files.append(filepath)
                    pbar.update(1)
                    pbar.set_postfix_str(f"OK: {len(results)}, Failed: {len(self.failed_files)}")
        return results

    def get_all_features_from_db(self):
        try:
            return get_all_features()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return []

    def generate_playlists_from_db(self):
        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()
            cursor.execute("SELECT file_path, bpm, centroid, danceability FROM audio_features")
            
            playlists = {}
            for row in cursor.fetchall():
                file_path, bpm, centroid, danceability = row
                if None in (bpm, centroid, danceability):
                    continue
                    
                bpm_group = "Slow" if bpm < 70 else "Medium" if bpm < 100 else "Upbeat" if bpm < 130 else "Fast"
                dance_group = "Chill" if danceability < 0.3 else "Easy" if danceability < 0.5 else "Groovy" if danceability < 0.7 else "Energetic"
                playlist_name = f"{bpm_group}_{dance_group}"
                
                if playlist_name not in playlists:
                    playlists[playlist_name] = []
                playlists[playlist_name].append(file_path)
                
            conn.close()
            return {name: {'tracks': tracks} for name, tracks in playlists.items()}
        except Exception as e:
            logger.error(f"Database playlist generation failed: {str(e)}")
            return {}

    def generate_playlists(self, features_list, num_playlists=5):
        data = []
        for f in features_list:
            if not f or 'filepath' not in f:
                continue
            try:
                data.append({
                    'filepath': f['filepath'],
                    'bpm': float(f.get('bpm', 0)),
                    'centroid': float(f.get('centroid', 0)),
                    'danceability': float(f.get('danceability', 0))
                })
            except Exception:
                continue

        if not data:
            return {}

        df = pd.DataFrame(data)
        cluster_features = ['bpm', 'centroid', 'danceability']
        
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[cluster_features])
            
            kmeans = MiniBatchKMeans(
                n_clusters=min(num_playlists, len(df)),
                random_state=42,
                batch_size=min(500, len(df))
            )
            df['cluster'] = kmeans.fit_predict(scaled_features)
            
            playlists = {}
            for cluster, group in df.groupby('cluster'):
                centroid = group[cluster_features].median().to_dict()
                name = f"Cluster_{cluster}"
                playlists[name] = {
                    'tracks': group['filepath'].tolist(),
                    'features': centroid
                }
                
            return playlists
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            return {}

    def save_playlists(self, playlists, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for name, data in playlists.items():
            songs = data.get('tracks', [])
            if not songs:
                continue
                
            with open(os.path.join(output_dir, f"{name}.m3u"), 'w') as f:
                f.write("\n".join(songs))
            logger.info(f"Saved {name} with {len(songs)} tracks")

def main():
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True, help='Music directory')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory')
    parser.add_argument('--num_playlists', type=int, default=8, help='Number of playlists')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--use_db', action='store_true', help='Use database only')
    parser.add_argument('--update', action='store_true', help='Update existing playlists')
    parser.add_argument('--analyze_only', action='store_true', help='Only run audio analysis')
    parser.add_argument('--generate_only', action='store_true', help='Only generate playlists')
    args = parser.parse_args()

    generator = PlaylistGenerator()
    start_time = time.time()
    
    try:
        if args.update:
            logger.info("Updating playlists from database")
            playlists = generator.generate_playlists_from_db()
            generator.save_playlists(playlists, args.output_dir)
        elif args.analyze_only:
            logger.info("Running audio analysis only")
            features = generator.analyze_directory(args.music_dir, args.workers)
            logger.info(f"Processed {len(features)} files")
        elif args.generate_only:
            logger.info("Generating playlists from database")
            features = generator.get_all_features_from_db()
            clustered = generator.generate_playlists_from_db()
            time_based = generator.scheduler.generate_time_based_playlists(features)
            generator.save_playlists({**clustered, **time_based}, args.output_dir)
        else:
            features = generator.analyze_directory(args.music_dir, args.workers)
            clustered = generator.generate_playlists(features, args.num_playlists)
            time_based = generator.scheduler.generate_time_based_playlists(features)
            generator.save_playlists({**clustered, **time_based}, args.output_dir)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"Completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()