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

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('playlist_generator.log')
    ]
)
logger = logging.getLogger(__name__)

class PlaylistGenerator:
    def __init__(self, db_path='audio_features.db', workers=4):
        self.db_path = os.path.abspath(db_path)
        self.workers = min(workers, mp.cpu_count())
        self.analyzer = AudioAnalyzer(db_path=self.db_path)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database with proper indexes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    bpm REAL,
                    duration REAL,
                    beat_confidence REAL,
                    spectral_centroid REAL,
                    last_modified REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON audio_files(file_hash)")
            conn.commit()

    def _get_audio_files(self, music_dir):
        """Fast directory scanning with error handling"""
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

    def _process_batch(self, filepaths):
        """Process a batch of files with error isolation"""
        results = []
        for filepath in filepaths:
            try:
                result = self.analyzer.extract_features(filepath)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed on {filepath}: {str(e)}")
        return results

    def process_directory(self, music_dir):
        """Robust parallel processing with progress tracking"""
        files = list(self._get_audio_files(music_dir))
        if not files:
            logger.warning("No audio files found")
            return []

        # Process in small batches for better progress tracking
        batch_size = min(50, len(files)//self.workers + 1)
        batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
        
        with mp.Pool(self.workers) as pool:
            results = []
            with tqdm(total=len(files), desc="Processing files") as pbar:
                for batch_result in pool.imap_unordered(self._process_batch, batches):
                    results.extend(batch_result)
                    pbar.update(len(batch_result))
        
        logger.info(f"Successfully processed {len(results)}/{len(files)} files")
        return results

    def generate_playlists(self, output_dir, min_tracks=5):
        """Guaranteed playlist generation with fallbacks"""
        try:
            # Load features from database
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql("""
                    SELECT path, bpm, beat_confidence, spectral_centroid
                    FROM audio_files
                    WHERE bpm > 0 AND duration > 0
                """, conn)

            if len(df) < min_tracks:
                logger.error(f"Not enough tracks ({len(df)}) to generate playlists")
                return 0

            # Fallback clustering if too few tracks
            n_clusters = min(20, max(5, len(df)//50))
            
            # Use MiniBatchKMeans for speed and reliability
            features = df[['bpm', 'beat_confidence', 'spectral_centroid']]
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=1000,
                random_state=42
            )
            df['cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(features))

            # Generate playlists
            os.makedirs(output_dir, exist_ok=True)
            playlist_count = 0
            
            for cluster_id in df['cluster'].unique():
                cluster_tracks = df[df['cluster'] == cluster_id]
                if len(cluster_tracks) < min_tracks:
                    continue
                
                # Create simple playlist name
                median_bpm = cluster_tracks['bpm'].median()
                playlist_name = f"BPM_{int(median_bpm)}"
                playlist_path = os.path.join(output_dir, f"{playlist_name}.m3u")
                
                with open(playlist_path, 'w') as f:
                    f.write("\n".join(cluster_tracks['path'].tolist()))
                
                logger.info(f"Created playlist {playlist_name} with {len(cluster_tracks)} tracks")
                playlist_count += 1

            logger.info(f"Successfully created {playlist_count} playlists")
            return playlist_count

        except Exception as e:
            logger.error(f"Playlist generation failed: {str(e)}")
            return 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--db_path', default='audio_features.db')
    args = parser.parse_args()

    generator = PlaylistGenerator(
        db_path=args.db_path,
        workers=args.workers
    )
    
    # Process files
    features = generator.process_directory(args.music_dir)
    
    # Generate playlists if we got results
    if features:
        generator.generate_playlists(args.output_dir)
    else:
        logger.error("No features extracted - cannot generate playlists")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()