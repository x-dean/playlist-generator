#!/usr/bin/env python3
import os
import sqlite3
import hashlib
import multiprocessing as mp
import signal
import time
from tqdm import tqdm
import logging
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from analyze_music import AudioAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PlaylistGenerator:
    def __init__(self, db_path='audio_features.db', workers=4, timeout=30):
        """
        Initialize with:
        - db_path: Path to SQLite database
        - workers: Number of parallel processes
        - timeout: Analysis timeout per file (seconds)
        """
        self.db_path = os.path.abspath(db_path)
        self.workers = workers
        self.timeout = timeout
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize analyzer with shared DB
        self.analyzer = AudioAnalyzer(db_path=self.db_path, timeout=timeout)
        
        # Initialize database schema
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    bpm REAL,
                    duration REAL,
                    beat_confidence REAL,
                    spectral_centroid REAL,
                    last_modified REAL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON audio_files(file_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON audio_files(path)")
            conn.commit()

    def _get_audio_files(self, music_dir):
        """Generator yielding all valid audio files"""
        valid_ext = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
        for root, _, files in os.walk(music_dir):
            for f in files:
                if f.lower().endswith(valid_ext):
                    filepath = os.path.join(root, f)
                    try:
                        if os.path.getsize(filepath) > 1024:  # 1KB minimum
                            yield filepath
                    except OSError as e:
                        logger.warning(f"Skipping {filepath}: {str(e)}")
                        continue

    def process_directory(self, music_dir):
        """
        Process all audio files in directory with:
        - Parallel processing
        - Hash-based change detection
        - Progress tracking
        """
        files = list(self._get_audio_files(music_dir))
        if not files:
            logger.warning("No audio files found in directory")
            return []

        logger.info(f"Found {len(files)} audio files to process")

        # Process files in parallel with progress bar
        with mp.Pool(self.workers) as pool:
            results = []
            with tqdm(total=len(files), desc="Processing files") as pbar:
                for result in pool.imap_unordered(self.analyzer.extract_features, files):
                    if result:  # Only count successful analyses
                        results.append(result)
                    pbar.update(1)

        processed = len(results)
        skipped = len(files) - processed
        logger.info(f"Processing complete: {processed} processed, {skipped} skipped")
        return results

    def generate_playlist_name(self, features):
        """Generate descriptive name from audio features"""
        bpm = features['bpm']
        energy = features['beat_confidence']
        centroid = features['spectral_centroid']

        # BPM classification
        if bpm < 60: bpm_desc = "Glacial"
        elif bpm < 80: bpm_desc = "Chill"
        elif bpm < 100: bpm_desc = "Medium"
        elif bpm < 120: bpm_desc = "Upbeat"
        elif bpm < 140: bpm_desc = "Energetic"
        else: bpm_desc = "Intense"

        # Energy classification
        energy_desc = "Mellow" if energy < 0.3 else "Moderate" if energy < 0.6 else "HighEnergy"

        # Spectral classification
        spectral_desc = (
            "Dark" if centroid < 1000 else
            "Warm" if centroid < 2000 else
            "Bright" if centroid < 3000 else
            "Crisp"
        )

        return f"{bpm_desc}_{energy_desc}_{spectral_desc}"

    def generate_playlists(self, output_dir, min_tracks=5):
        """
        Generate playlists from analyzed tracks with:
        - K-means clustering
        - Automatic naming
        - Minimum tracks threshold
        """
        # Load features from database
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT path, bpm, beat_confidence, spectral_centroid
                FROM audio_files
                WHERE bpm > 0 AND duration > 0
            """)
            features = [dict(row) for row in cursor.fetchall()]

        if not features:
            logger.error("No audio features found in database")
            return 0

        logger.info(f"Generating playlists from {len(features)} tracks")

        # Prepare data for clustering
        df = pd.DataFrame(features)
        X = df[['bpm', 'beat_confidence', 'spectral_centroid']]
        
        # Normalize features
        X_scaled = StandardScaler().fit_transform(X)

        # Dynamic cluster count
        num_clusters = min(15, max(5, len(features) // 50))
        
        # Cluster tracks
        df['cluster'] = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(X_scaled)

        # Generate playlists
        os.makedirs(output_dir, exist_ok=True)
        playlist_count = 0

        for cluster_id in df['cluster'].unique():
            cluster_tracks = df[df['cluster'] == cluster_id]
            if len(cluster_tracks) < min_tracks:
                continue

            # Get median features for naming
            median_features = cluster_tracks.median(numeric_only=True)
            playlist_name = self.generate_playlist_name(median_features)
            
            # Sanitize filename
            safe_name = "".join(c if c.isalnum() else "_" for c in playlist_name)
            playlist_path = os.path.join(output_dir, f"{safe_name}.m3u")

            # Write playlist
            with open(playlist_path, 'w') as f:
                f.write("\n".join(cluster_tracks['path'].tolist()))

            logger.info(f"Created {playlist_name} with {len(cluster_tracks)} tracks")
            playlist_count += 1

        logger.info(f"Generated {playlist_count} playlists in {output_dir}")
        return playlist_count

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True, help='Directory containing music files')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory for playlists')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1), 
                       help='Number of worker processes')
    parser.add_argument('--timeout', type=int, default=30, 
                       help='Timeout per file analysis in seconds')
    parser.add_argument('--db_path', default='audio_features.db',
                       help='Path to analysis database')
    args = parser.parse_args()

    # Initialize generator
    generator = PlaylistGenerator(
        db_path=args.db_path,
        workers=args.workers,
        timeout=args.timeout
    )

    # Process files
    features = generator.process_directory(args.music_dir)
    
    # Generate playlists if we got results
    if features:
        generator.generate_playlists(args.output_dir)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()