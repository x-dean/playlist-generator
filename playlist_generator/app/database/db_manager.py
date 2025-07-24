import sqlite3
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.connection_pool = {}
        self._init_db()

    def _get_connection(self):
        """Get a connection from the pool or create a new one"""
        pid = os.getpid()
        if pid not in self.connection_pool:
            conn = sqlite3.connect(self.db_file, timeout=60)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.row_factory = sqlite3.Row  # Enable dictionary access
            self.connection_pool[pid] = conn
        return self.connection_pool[pid]

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_file, timeout=60)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")
            
            # Audio features table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_features (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                duration REAL,
                bpm REAL,
                beat_confidence REAL,
                centroid REAL,
                loudness REAL,
                danceability REAL,
                key INTEGER,
                scale INTEGER,
                onset_rate REAL,
                zcr REAL,
                last_modified REAL,
                last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
            """)
            
            # Playlists table with description and features
            conn.execute("""
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                features TEXT,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                stats TEXT
            )
            """)
            
            # Add missing columns if they don't exist
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(playlists)")
            columns = {row[1] for row in cursor.fetchall()}
            
            if 'description' not in columns:
                conn.execute("ALTER TABLE playlists ADD COLUMN description TEXT")
            if 'features' not in columns:
                conn.execute("ALTER TABLE playlists ADD COLUMN features TEXT")
            if 'stats' not in columns:
                conn.execute("ALTER TABLE playlists ADD COLUMN stats TEXT")
            
            # Playlist tracks table with ordering
            conn.execute("""
            CREATE TABLE IF NOT EXISTS playlist_tracks (
                playlist_id INTEGER,
                file_hash TEXT,
                position INTEGER,
                added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(playlist_id) REFERENCES playlists(id),
                FOREIGN KEY(file_hash) REFERENCES audio_features(file_hash),
                PRIMARY KEY (playlist_id, file_hash)
            )
            """)
            
            # Add position column if it doesn't exist
            cursor.execute("PRAGMA table_info(playlist_tracks)")
            playlist_track_columns = {row[1] for row in cursor.fetchall()}
            if 'position' not in playlist_track_columns:
                conn.execute("ALTER TABLE playlist_tracks ADD COLUMN position INTEGER")
            
            # Track tags table for additional metadata
            conn.execute("""
            CREATE TABLE IF NOT EXISTS track_tags (
                file_hash TEXT,
                tag TEXT,
                value TEXT,
                FOREIGN KEY(file_hash) REFERENCES audio_features(file_hash),
                PRIMARY KEY (file_hash, tag)
            )
            """)
            
            # Cache table for expensive computations
            conn.execute("""
            CREATE TABLE IF NOT EXISTS computation_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires TIMESTAMP,
                metadata TEXT
            )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON audio_features(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_playlist_name ON playlists(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_track_tags ON track_tags(tag)")
            
            conn.commit()
        finally:
            conn.close()

    def save_playlist(self, name: str, tracks: List[str], features: Dict[str, Any] = None,
                     description: str = None) -> bool:
        """Save or update a playlist"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Insert or update playlist
            cursor.execute("""
            INSERT INTO playlists (name, description, features, last_updated)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET
                description = excluded.description,
                features = excluded.features,
                last_updated = CURRENT_TIMESTAMP
            """, (name, description, json.dumps(features) if features else None))
            
            playlist_id = cursor.lastrowid or cursor.execute(
                "SELECT id FROM playlists WHERE name = ?", (name,)
            ).fetchone()[0]
            
            # Clear existing tracks
            cursor.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
            
            # Insert new tracks with positions
            for position, track_path in enumerate(tracks):
                file_hash = self._get_file_hash(cursor, track_path)
                if file_hash:
                    cursor.execute("""
                    INSERT INTO playlist_tracks (playlist_id, file_hash, position)
                    VALUES (?, ?, ?)
                    """, (playlist_id, file_hash, position))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error saving playlist {name}: {str(e)}")
            conn.rollback()
            return False

    def get_playlist(self, name: str) -> Optional[Dict[str, Any]]:
        """Get playlist by name with all related data"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Get playlist info
            cursor.execute("""
            SELECT id, name, description, features, created, last_updated, stats
            FROM playlists WHERE name = ?
            """, (name,))
            
            if playlist := cursor.fetchone():
                result = dict(playlist)
                
                # Get tracks with features
                cursor.execute("""
                SELECT af.*, pt.position
                FROM playlist_tracks pt
                JOIN audio_features af ON pt.file_hash = af.file_hash
                WHERE pt.playlist_id = ?
                ORDER BY pt.position
                """, (playlist['id'],))
                
                tracks = []
                for track in cursor.fetchall():
                    track_dict = dict(track)
                    
                    # Get track tags
                    cursor.execute("""
                    SELECT tag, value FROM track_tags
                    WHERE file_hash = ?
                    """, (track['file_hash'],))
                    
                    track_dict['tags'] = {
                        row['tag']: row['value']
                        for row in cursor.fetchall()
                    }
                    
                    tracks.append(track_dict)
                
                result['tracks'] = tracks
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting playlist {name}: {str(e)}")
            return None

    def _get_file_hash(self, cursor: sqlite3.Cursor, file_path: str) -> Optional[str]:
        """Get file hash from file path"""
        cursor.execute(
            "SELECT file_hash FROM audio_features WHERE file_path = ?",
            (file_path,)
        )
        if result := cursor.fetchone():
            return result[0]
        return None

    def cache_computation(self, key: str, value: Any, expires_in: int = 86400,
                         metadata: Dict[str, Any] = None) -> bool:
        """Cache computation result"""
        conn = self._get_connection()
        try:
            expires = datetime.now().timestamp() + expires_in
            conn.execute("""
            INSERT OR REPLACE INTO computation_cache
            (key, value, expires, metadata)
            VALUES (?, ?, datetime(?), ?)
            """, (
                key,
                json.dumps(value),
                expires,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error caching computation {key}: {str(e)}")
            conn.rollback()
            return False

    def get_cached_computation(self, key: str) -> Optional[Any]:
        """Get cached computation result if not expired"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT value, metadata FROM computation_cache
            WHERE key = ? AND (expires IS NULL OR expires > datetime(?))
            """, (key, datetime.now().timestamp()))
            
            if result := cursor.fetchone():
                value = json.loads(result['value'])
                metadata = json.loads(result['metadata']) if result['metadata'] else None
                return {'value': value, 'metadata': metadata}
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached computation {key}: {str(e)}")
            return None

    def cleanup_cache(self, max_age: int = 86400) -> int:
        """Clean up expired cache entries"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            DELETE FROM computation_cache
            WHERE expires < datetime(?)
            """, (datetime.now().timestamp() - max_age,))
            
            deleted = cursor.rowcount
            conn.commit()
            return deleted
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
            conn.rollback()
            return 0

    def add_track_tags(self, file_path: str, tags: Dict[str, str]) -> bool:
        """Add or update tags for a track"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            file_hash = self._get_file_hash(cursor, file_path)
            if not file_hash:
                return False
            
            for tag, value in tags.items():
                cursor.execute("""
                INSERT OR REPLACE INTO track_tags (file_hash, tag, value)
                VALUES (?, ?, ?)
                """, (file_hash, tag, value))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error adding tags for {file_path}: {str(e)}")
            conn.rollback()
            return False

    def get_track_tags(self, file_path: str) -> Dict[str, str]:
        """Get all tags for a track"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            file_hash = self._get_file_hash(cursor, file_path)
            if not file_hash:
                return {}
            
            cursor.execute("""
            SELECT tag, value FROM track_tags
            WHERE file_hash = ?
            """, (file_hash,))
            
            return {row['tag']: row['value'] for row in cursor.fetchall()}
            
        except Exception as e:
            logger.error(f"Error getting tags for {file_path}: {str(e)}")
            return {} 

    def get_library_statistics(self) -> Dict[str, Any]:
        """Return statistics about the music library and playlists."""
        conn = self._get_connection()
        stats = {}
        try:
            cursor = conn.cursor()
            # Total tracks
            cursor.execute("SELECT COUNT(*) FROM audio_features")
            stats['total_tracks'] = cursor.fetchone()[0]

            # Tracks with tags (track_tags table)
            cursor.execute("SELECT COUNT(DISTINCT file_hash) FROM track_tags")
            stats['tracks_with_tags'] = cursor.fetchone()[0]

            # Tracks with genre in metadata
            cursor.execute("SELECT metadata FROM audio_features")
            genre_count = 0
            year_count = 0
            genre_counter = {}
            for row in cursor.fetchall():
                try:
                    meta = json.loads(row['metadata']) if row['metadata'] else {}
                    genre = meta.get('genre')
                    year = meta.get('year') or meta.get('date')
                    # Count for stats['tracks_with_genre']
                    if genre and (isinstance(genre, str) or (isinstance(genre, list) and genre)):
                        genre_count += 1
                    if year and str(year).strip():
                        year_count += 1
                    # Count for genre_counts
                    if genre:
                        if isinstance(genre, str):
                            genre_list = [genre]
                        elif isinstance(genre, list):
                            genre_list = genre
                        else:
                            genre_list = []
                        for g in genre_list:
                            g_norm = g if g else "UnknownGenre"
                            genre_counter[g_norm] = genre_counter.get(g_norm, 0) + 1
                except Exception:
                    continue
            stats['tracks_with_genre'] = genre_count
            stats['tracks_with_year'] = year_count
            stats['genre_counts'] = genre_counter

            # Total playlists and playlist details
            cursor.execute("SELECT name, features FROM playlists")
            playlists = cursor.fetchall()
            stats['total_playlists'] = len(playlists)
            # Count playlists per mode
            mode_counts = {}
            playlist_sizes = []
            for row in playlists:
                mode = None
                # Try to parse mode from features JSON
                try:
                    features = json.loads(row['features']) if row['features'] else {}
                    mode = features.get('mode') or features.get('method')
                except Exception:
                    mode = None
                # Fallback: parse from name
                if not mode and row['name']:
                    name = row['name'].lower()
                    for m in ['tags', 'kmeans', 'time', 'cache', 'all']:
                        if m in name:
                            mode = m
                            break
                if not mode:
                    mode = 'unknown'
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                # Playlist size
                cursor2 = conn.cursor()
                cursor2.execute("SELECT COUNT(*) FROM playlist_tracks pt JOIN playlists p ON pt.playlist_id = p.id WHERE p.name = ?", (row['name'],))
                size = cursor2.fetchone()[0]
                playlist_sizes.append(size)
            stats['playlists_per_mode'] = mode_counts
            if playlist_sizes:
                stats['avg_playlist_size'] = sum(playlist_sizes) / len(playlist_sizes)
                stats['largest_playlist_size'] = max(playlist_sizes)
                stats['smallest_playlist_size'] = min(playlist_sizes)
                stats['playlists_with_0_tracks'] = playlist_sizes.count(0)
            # Tracks not in any playlist
            cursor.execute("SELECT COUNT(*) FROM audio_features WHERE file_hash NOT IN (SELECT file_hash FROM playlist_tracks)")
            stats['tracks_not_in_any_playlist'] = cursor.fetchone()[0]
            # Tracks in multiple playlists
            cursor.execute("SELECT COUNT(*) FROM (SELECT file_hash FROM playlist_tracks GROUP BY file_hash HAVING COUNT(playlist_id) > 1)")
            stats['tracks_in_multiple_playlists'] = cursor.fetchone()[0]
            # Top 5 genres
            if genre_counter:
                sorted_genres = sorted([(g, c) for g, c in genre_counter.items() if g not in ("Other", "UnknownGenre", "", None)], key=lambda x: -x[1])
                stats['top_5_genres'] = sorted_genres[:5]
            # Tracks with real genre and others (count each track only once)
            cursor.execute("SELECT metadata FROM audio_features")
            tracks_with_real_genre = 0
            tracks_with_no_real_genre = 0
            for row in cursor.fetchall():
                try:
                    meta = json.loads(row['metadata']) if row['metadata'] else {}
                    genre = meta.get('genre')
                    found_real = False
                    if genre:
                        genre_list = [genre] if isinstance(genre, str) else genre if isinstance(genre, list) else []
                        for g in genre_list:
                            if g not in ("Other", "UnknownGenre", "", None):
                                found_real = True
                                break
                    if found_real:
                        tracks_with_real_genre += 1
                    else:
                        tracks_with_no_real_genre += 1
                except Exception:
                    tracks_with_no_real_genre += 1
            stats['tracks_with_real_genre'] = tracks_with_real_genre
            stats['tracks_with_no_real_genre'] = tracks_with_no_real_genre
            # Unique file extensions and tracks per extension
            cursor.execute("SELECT file_path FROM audio_features")
            ext_counter = {}
            for row in cursor.fetchall():
                ext = os.path.splitext(row['file_path'])[1].lower()
                if ext:
                    ext_counter[ext] = ext_counter.get(ext, 0) + 1
            stats['unique_file_extensions'] = list(ext_counter.keys())
            stats['tracks_per_extension'] = ext_counter
            # Last analysis date
            cursor.execute("SELECT MAX(last_analyzed) FROM audio_features")
            stats['last_analysis_date'] = cursor.fetchone()[0]
            # Last playlist update date
            cursor.execute("SELECT MAX(last_updated) FROM playlists")
            stats['last_playlist_update_date'] = cursor.fetchone()[0]

            return stats
        except Exception as e:
            logger.error(f"Error getting library statistics: {str(e)}")
            return stats 