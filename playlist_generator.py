#!/usr/bin/env python3
import os
import sqlite3
import hashlib
import multiprocessing as mp
import signal  # Added missing import
from tqdm import tqdm
import logging
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PlaylistGenerator:
    def __init__(self, db_path='audio_features.db', workers=4, timeout=30):
        self.db_path = db_path
        self.workers = workers
        self.timeout = timeout
        self._init_db()
        
    def _init_db(self):
        """Initialize database with file hashes and features"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    bpm REAL,
                    duration REAL,
                    beat_confidence REAL,
                    centroid REAL,
                    last_modified REAL
                )
            """)
            conn.commit()

    def _file_hash(self, filepath):
        """Generate hash based on file metadata"""
        stat = os.stat(filepath)
        return hashlib.md5(f"{filepath}-{stat.st_size}-{stat.st_mtime}".encode()).hexdigest()

    def _get_audio_files(self, music_dir):
        """Find all audio files in directory"""
        valid_ext = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
        for root, _, files in os.walk(music_dir):
            for f in files:
                if f.lower().endswith(valid_ext):
                    filepath = os.path.join(root, f)
                    if os.path.getsize(filepath) > 1024:  # Skip small files
                        yield filepath

    def _process_file(self, filepath):
        """Process single file and return features"""
        try:
            # Skip if already processed (hash matches)
            file_hash = self._file_hash(filepath)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT file_hash FROM audio_files WHERE path = ?", 
                    (filepath,)
                )
                if row := cursor.fetchone():
                    if row[0] == file_hash:
                        return None  # Skip processing
                    
            # Simulate feature extraction (replace with actual analysis)
            features = {
                'path': filepath,
                'file_hash': file_hash,
                'bpm': 120.0,  # Replace with actual BPM extraction
                'duration': 180.0,  # Replace with actual duration
                'beat_confidence': 0.8,  # Replace with actual confidence
                'centroid': 2000.0,  # Replace with actual spectral centroid
                'last_modified': os.path.getmtime(filepath)
            }
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO audio_files 
                    VALUES (:path, :file_hash, :bpm, :duration, 
                            :beat_confidence, :centroid, :last_modified)
                    """, features)
                conn.commit()
            
            return features
            
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
                pool.imap_unordered(self._process_file, files),
                total=len(files),
                desc="Processing files",
                unit="file"
            ))
        
        return [r for r in results if r is not None]

    def generate_playlist_name(self, features):
        """Generate descriptive playlist name based on audio features"""
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

    def generate_playlists(self, output_dir, min_tracks=5):
        """Generate playlists from analyzed features"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT path, bpm, duration, beat_confidence, centroid 
                FROM audio_files
                WHERE bpm > 0 AND duration > 0
            """)
            features = [dict(row) for row in cursor.fetchall()]
        
        if not features:
            logger.warning("No features available for playlist generation")
            return
        
        # Prepare features for clustering
        df = pd.DataFrame(features)
        X = df[['bpm', 'beat_confidence', 'centroid']]
        
        # Normalize and cluster
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        num_clusters = min(max(5, len(features)//20), 15)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create playlists
        os.makedirs(output_dir, exist_ok=True)
        playlist_count = 0
        
        for cluster_id in df['cluster'].unique():
            cluster_tracks = df[df['cluster'] == cluster_id]
            if len(cluster_tracks) < min_tracks:
                continue
                
            # Get centroid features for naming
            centroid_features = {
                'bpm': cluster_tracks['bpm'].median(),
                'beat_confidence': cluster_tracks['beat_confidence'].median(),
                'centroid': cluster_tracks['centroid'].median()
            }
            
            playlist_name = self.generate_playlist_name(centroid_features)
            safe_name = "".join(c if c.isalnum() else "_" for c in playlist_name)
            
            # Save playlist
            with open(os.path.join(output_dir, f"{safe_name}.m3u"), 'w') as f:
                f.write("\n".join(cluster_tracks['path'].tolist()))
            
            logger.info(f"Created playlist '{safe_name}' with {len(cluster_tracks)} tracks")
            playlist_count += 1
        
        logger.info(f"Generated {playlist_count} playlists in {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True, help='Directory containing music files')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory for playlists')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1), 
                       help='Number of worker processes')
    parser.add_argument('--timeout', type=int, default=30, 
                       help='Timeout per file analysis in seconds')
    args = parser.parse_args()

    generator = PlaylistGenerator(
        workers=args.workers,
        timeout=args.timeout
    )
    
    # Process files
    features = generator.process_directory(args.music_dir)
    logger.info(f"Processed {len(features)} audio files")
    
    # Generate playlists
    generator.generate_playlists(args.output_dir)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()