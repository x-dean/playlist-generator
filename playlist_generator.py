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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PlaylistGenerator:
    def __init__(self, db_path='/app/cache/audio_features.db', workers=4, timeout=30):
        self.db_path = db_path
        self.workers = workers
        self.timeout = timeout
        self.analyzer = AudioAnalyzer(timeout=timeout)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize database with proper schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    bpm REAL,
                    duration REAL,
                    beat_confidence REAL,
                    centroid REAL,
                    last_modified REAL,
                    processed_time REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON audio_files(file_hash)")
            conn.commit()

    def _file_hash(self, filepath):
        """Generate hash based on file metadata"""
        try:
            stat = os.stat(filepath)
            return hashlib.md5(f"{filepath}-{stat.st_size}-{stat.st_mtime}".encode()).hexdigest()
        except Exception as e:
            logger.error(f"Couldn't get file stats for {filepath}: {str(e)}")
            return hashlib.md5(filepath.encode()).hexdigest()

    def _get_audio_files(self, music_dir):
        """Find all audio files in directory"""
        valid_ext = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
        for root, _, files in os.walk(music_dir):
            for f in files:
                if f.lower().endswith(valid_ext):
                    filepath = os.path.join(root, f)
                    try:
                        if os.path.getsize(filepath) > 1024:  # Skip small files
                            yield filepath
                    except OSError as e:
                        logger.warning(f"Skipping {filepath}: {str(e)}")

    def _process_file(self, filepath):
        """Process single file and return features"""
        try:
            current_hash = self._file_hash(filepath)
            
            # Check if we can skip processing
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_hash FROM audio_files WHERE path = ?", 
                    (filepath,)
                )
                if row := cursor.fetchone():
                    if row[0] == current_hash:
                        return None  # Skip unchanged files
            
            # Only proceed if analyzer is properly initialized
            if not hasattr(self, 'analyzer'):
                self.analyzer = AudioAnalyzer(timeout=self.timeout)
                
            features = self.analyzer.extract_features(filepath)
            if not features:
                return None
                
            record = {
                'path': filepath,
                'file_hash': current_hash,
                'bpm': features.get('bpm', 0),
                'duration': features.get('duration', 0),
                'beat_confidence': features.get('beat_confidence', 0),
                'centroid': features.get('centroid', 0),
                'last_modified': os.path.getmtime(filepath),
                'processed_time': time.time()
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO audio_files 
                    VALUES (:path, :file_hash, :bpm, :duration, 
                            :beat_confidence, :centroid, :last_modified, :processed_time)
                    """, record)
                conn.commit()
            
            return record
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return None

    def process_directory(self, music_dir):
        """Process all files in directory with parallel workers"""
        files = list(self._get_audio_files(music_dir))
        if not files:
            logger.warning("No audio files found")
            return []

        # Process files in parallel
        with mp.Pool(self.workers) as pool:
            results = list(tqdm(
                pool.imap(self._process_file, files),
                total=len(files),
                desc="Processing files",
                unit="file"
            ))
        
        processed_count = sum(1 for r in results if r is not None)
        logger.info(f"Processed {processed_count} files ({len(files) - processed_count} skipped)")
        return [r for r in results if r is not None]

    def generate_playlists(self, output_dir, min_tracks=5):
        """Generate playlists from analyzed features"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT path, bpm, duration, beat_confidence, centroid 
                FROM audio_files
                WHERE bpm > 0 AND duration > 0
                ORDER BY RANDOM()
            """)
            features = [dict(row) for row in cursor.fetchall()]
        
        if not features:
            logger.warning("No features available for playlist generation")
            return
        
        # Prepare features for clustering
        df = pd.DataFrame(features)
        X = df[['bpm', 'beat_confidence', 'centroid']].values
        
        # Dynamic cluster count based on feature diversity
        num_clusters = min(15, max(5, len(features) // 50))
        
        try:
            # Normalize and cluster
            X_scaled = StandardScaler().fit_transform(X)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(X_scaled)
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            df['cluster'] = 0  # Fallback to single playlist
        
        # Create playlists
        os.makedirs(output_dir, exist_ok=True)
        playlist_count = 0
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_tracks = df[df['cluster'] == cluster_id]
            if len(cluster_tracks) < min_tracks:
                continue
                
            # Get median features for naming
            median_features = {
                'bpm': cluster_tracks['bpm'].median(),
                'beat_confidence': cluster_tracks['beat_confidence'].median(),
                'centroid': cluster_tracks['centroid'].median()
            }
            
            playlist_name = self._generate_playlist_name(median_features)
            safe_name = "".join(c if c.isalnum() else "_" for c in playlist_name)
            playlist_path = os.path.join(output_dir, f"{safe_name}.m3u")
            
            # Save playlist
            with open(playlist_path, 'w') as f:
                f.write("\n".join(cluster_tracks['path'].tolist()))
            
            logger.info(f"Created playlist '{playlist_name}' with {len(cluster_tracks)} tracks")
            playlist_count += 1
        
        logger.info(f"Generated {playlist_count} playlists in {output_dir}")
        return playlist_count

    def _generate_playlist_name(self, features):
        """Generate descriptive playlist name"""
        bpm = features.get('bpm', 0)
        if bpm < 60: bpm_desc = "Glacial"
        elif bpm < 80: bpm_desc = "Chill"
        elif bpm < 100: bpm_desc = "Medium"
        elif bpm < 120: bpm_desc = "Upbeat"
        elif bpm < 140: bpm_desc = "Energetic"
        else: bpm_desc = "Intense"

        energy = features.get('beat_confidence', 0) * 10
        energy_desc = "Mellow" if energy < 3 else "Moderate" if energy < 6 else "HighEnergy"

        centroid = features.get('centroid', 0)
        spectral_desc = "Dark" if centroid < 1000 else "Warm" if centroid < 2000 else "Bright" if centroid < 3000 else "Crisp"

        return f"{bpm_desc}_{energy_desc}_{spectral_desc}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True)
    parser.add_argument('--output_dir', default='/app/playlists')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument('--timeout', type=int, default=30)
    args = parser.parse_args()

    generator = PlaylistGenerator(
        workers=args.workers,
        timeout=args.timeout
    )
    
    # Process files
    features = generator.process_directory(args.music_dir)
    
    # Generate playlists
    if features:
        generator.generate_playlists(args.output_dir)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()