#!/usr/bin/env python3
import os
import sqlite3
import multiprocessing as mp
import logging
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from analyze_music import AudioAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('playlist_generator.log')
    ]
)
logger = logging.getLogger(__name__)

class PlaylistGenerator:
    def __init__(self, db_path, workers=4, timeout=30):
        self.db_path = os.path.abspath(db_path)
        self.workers = min(workers, mp.cpu_count())
        self.timeout = timeout
        self.analyzer = AudioAnalyzer(db_path=db_path, timeout=timeout)
        self._init_db()

    def _init_db(self):
        """Initialize database with proper schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    bpm REAL,
                    duration REAL,
                    energy REAL,
                    spectral_centroid REAL,
                    last_modified REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON tracks(file_hash)")
            conn.commit()

    def process_directory(self, music_dir):
        """Robust parallel processing with caching"""
        from glob import glob
        files = glob(f"{music_dir}/**/*.[mM][pP]3", recursive=True) + \
                glob(f"{music_dir}/**/*.[fF][lL][aA][cC]", recursive=True)
        
        with mp.Pool(self.workers) as pool:
            results = []
            with tqdm(total=len(files), desc="Processing") as pbar:
                for result in pool.imap_unordered(self.analyzer.extract_features, files):
                    if result:
                        results.append(result)
                    pbar.update(1)
        return results

    def generate_playlists(self, output_dir):
        """Professional playlist generation"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("SELECT * FROM tracks WHERE bpm > 0", conn)
        
        if len(df) < 10:
            logger.warning("Insufficient tracks for clustering")
            self._create_fallback_playlist(df, output_dir)
            return

        # Professional clustering
        features = df[['bpm', 'energy', 'spectral_centroid']]
        X = StandardScaler().fit_transform(features)
        
        kmeans = MiniBatchKMeans(
            n_clusters=min(20, len(df)//50),
            batch_size=1000,
            random_state=42
        )
        df['cluster'] = kmeans.fit_predict(X)
        
        # Generate playlists
        os.makedirs(output_dir, exist_ok=True)
        for cluster_id in df['cluster'].unique():
            cluster = df[df['cluster'] == cluster_id]
            if len(cluster) < 5:
                continue
                
            name = f"BPM_{cluster['bpm'].median():.0f}_Energy_{cluster['energy'].median():.2f}"
            with open(f"{output_dir}/{name}.m3u", 'w') as f:
                f.write("\n".join(cluster['path'].tolist()))

    def _create_fallback_playlist(self, df, output_dir):
        """Guaranteed playlist creation"""
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/All_Tracks.m3u", 'w') as f:
            f.write("\n".join(df['path'].tolist()))
        logger.info("Created fallback playlist with all tracks")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Professional Playlist Generator')
    parser.add_argument('--music_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--db_path', required=True)
    args = parser.parse_args()

    generator = PlaylistGenerator(
        db_path=args.db_path,
        workers=args.workers,
        timeout=args.timeout
    )
    
    # Process files
    logger.info("Starting processing...")
    features = generator.process_directory(args.music_dir)
    
    # Generate playlists
    logger.info("Generating playlists...")
    generator.generate_playlists(args.output_dir)
    logger.info("Operation completed successfully")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()