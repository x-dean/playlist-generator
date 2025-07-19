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
import json
from datetime import datetime

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

class PlaylistGenerator:
    def __init__(self):
        self.failed_files = []
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = os.path.join(cache_dir, 'audio_analysis.db')
        self.playlist_db = os.path.join(cache_dir, 'playlist_tracker.db')
        self._init_playlist_tracker()

    def _init_playlist_tracker(self):
        os.makedirs(os.path.dirname(self.playlist_db), exist_ok=True)
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
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
        conn.commit()
        conn.close()

    def generate_playlist_name(self, features):
        bpm = features.get('bpm', 0.0)
        centroid = features.get('centroid', 0.0)
        loudness = features.get('loudness', 0.0)
        spectral_complexity = features.get('spectral_complexity', 0.0)
        pitch_mean = features.get('pitch_mean', 0.0)

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

        timbre_desc = (
            "Dark" if centroid < 800 else
            "Warm" if centroid < 1500 else
            "Mellow" if centroid < 2200 else
            "Bright" if centroid < 3000 else
            "Sharp" if centroid < 4000 else
            "VerySharp"
        )

        loudness_desc = (
            "Silent" if loudness < -40 else
            "Quiet" if loudness < -20 else
            "Moderate" if loudness < -10 else
            "Loud"
        )

        complexity_desc = (
            "Simple" if spectral_complexity < 10 else
            "Moderate" if spectral_complexity < 20 else
            "Complex"
        )

        pitch_desc = (
            "LowPitch" if pitch_mean < 100 else
            "MidPitch" if pitch_mean < 300 else
            "HighPitch"
        )

        return f"{bpm_desc}_{timbre_desc}_{loudness_desc}_{complexity_desc}_{pitch_desc}"

    def generate_playlists(self, features_list, num_playlists=5, chunk_size=1000, output_dir=None):
        if not features_list:
            logger.warning("No features to cluster")
            return {}

        valid_features = []
        for feat in features_list:
            if not feat or 'filepath' not in feat:
                continue
            if os.path.exists(feat['filepath']):
                if feat.get('beat_confidence', 0.0) < 0.3 or feat.get('duration', 0.0) < 30:
                    continue  # Filter out low-quality or too-short audio
                for key in ['bpm', 'centroid', 'duration', 'loudness', 'spectral_complexity', 'pitch_mean']:
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

        naming_features = ['bpm', 'centroid', 'duration', 'loudness', 'spectral_complexity', 'pitch_mean']
        cluster_features = ['bpm', 'centroid', 'loudness', 'spectral_complexity', 'pitch_mean']

        for feat in naming_features:
            if feat not in df.columns:
                logger.warning(f"Adding missing feature column: {feat}")
                df[feat] = 0.0

        df[naming_features] = df[naming_features].fillna(0)

        features_array = df[cluster_features].values.astype(np.float32)
        features_scaled = StandardScaler().fit_transform(features_array)

        kmeans = MiniBatchKMeans(
            n_clusters=min(num_playlists, len(df)),
            random_state=42,
            batch_size=min(500, len(df)),
            n_init=3,
            max_iter=50
        )
        df['cluster'] = kmeans.fit_predict(features_scaled)

        playlists = {}
        for cluster in df['cluster'].unique():
            cluster_songs = df[df['cluster'] == cluster].copy()
            centroid = cluster_songs[naming_features].mean().to_dict()
            name = sanitize_filename(self.generate_playlist_name(centroid))
            cluster_songs['distance_to_center'] = np.linalg.norm(
                features_scaled[cluster_songs.index] - kmeans.cluster_centers_[cluster], axis=1
            )
            sorted_paths = cluster_songs.sort_values(by='distance_to_center')['filepath'].tolist()
            playlists[name] = sorted_paths
            if output_dir:
                self._export_m3u_playlist(name, sorted_paths, output_dir)

        self._update_playlist_tracker(playlists)
        return {k: v for k, v in playlists.items() if len(v) >= 5}

    def _export_m3u_playlist(self, name, paths, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        playlist_path = os.path.join(output_dir, f"{name}.m3u")
        with open(playlist_path, 'w', encoding='utf-8') as f:
            for track in paths:
                f.write(track + '\n')
        logger.info(f"Exported playlist: {playlist_path}")

    def _update_playlist_tracker(self, playlists):
        conn = sqlite3.connect(self.playlist_db)
        cursor = conn.cursor()

        for name, songs in playlists.items():
            cursor.execute("SELECT id FROM playlists WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                playlist_id = row[0]
                cursor.execute("UPDATE playlists SET last_updated = CURRENT_TIMESTAMP WHERE id = ?", (playlist_id,))
            else:
                cursor.execute("INSERT INTO playlists (name) VALUES (?)", (name,))
                playlist_id = cursor.lastrowid

            for song in songs:
                cursor.execute("""
                    INSERT OR IGNORE INTO playlist_songs (playlist_id, filepath) VALUES (?, ?)
                """, (playlist_id, song))

        conn.commit()
        conn.close()
