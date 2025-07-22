# database/playlist_db.py
import sqlite3
import logging
import os

logger = logging.getLogger(__name__)

class PlaylistDatabase:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self._init_db()
        self.connection_pool = {}

    def _get_connection(self):
        """Get a connection from the pool or create a new one"""
        pid = os.getpid()
        if pid not in self.connection_pool:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")
            self.connection_pool[pid] = conn
        return self.connection_pool[pid]

    def _close_all_connections(self):
        """Close all connections in the pool"""
        for conn in self.connection_pool.values():
            try:
                conn.close()
            except Exception:
                pass
        self.connection_pool.clear()

    def __del__(self):
        """Cleanup connections on object destruction"""
        self._close_all_connections()

    def playlists_exist(self):
        """Check if any playlists exist in the database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM playlists")
            count = cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"Error checking playlists: {str(e)}")
            return False

    def save_playlists(self, playlists):
        """Save playlists to the database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Clear existing playlists
            cursor.execute("DELETE FROM playlist_tracks")
            cursor.execute("DELETE FROM playlists")
            
            # Insert new playlists
            for name, data in playlists.items():
                cursor.execute(
                    "INSERT INTO playlists (name) VALUES (?)",
                    (name,)
                )
                playlist_id = cursor.lastrowid
                
                for track in data.get('tracks', []):
                    # Get file hash for the track
                    cursor.execute(
                        "SELECT file_hash FROM audio_features WHERE file_path = ?",
                        (track,)
                    )
                    result = cursor.fetchone()
                    if result:
                        file_hash = result[0]
                        cursor.execute(
                            "INSERT INTO playlist_tracks (playlist_id, file_hash) VALUES (?, ?)",
                            (playlist_id, file_hash)
                        )
            
            conn.commit()
            logger.info(f"Saved {len(playlists)} playlists to database")
        except Exception as e:
            logger.error(f"Error saving playlists: {str(e)}")
            conn.rollback()
    
    def _init_db(self):
        conn = sqlite3.connect(self.cache_file, timeout=60)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("""
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS playlist_tracks (
                playlist_id INTEGER,
                file_hash TEXT,
                added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(playlist_id) REFERENCES playlists(id),
                FOREIGN KEY(file_hash) REFERENCES audio_features(file_hash),
                PRIMARY KEY (playlist_id, file_hash)
            )
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
        finally:
            conn.close()

    def update_playlists(self, changed_files=None):
        """Update playlists based on changed files. Returns number of changed files."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            if changed_files is None:
                cursor.execute("""
                SELECT file_path, file_hash
                FROM audio_features
                WHERE last_analyzed > (
                    SELECT MAX(last_updated) FROM playlists
                )
                """)
                changed_files = [row[0] for row in cursor.fetchall()]

            if not changed_files:
                logger.info("No changed files, playlists up-to-date")
                return 0

            logger.info(f"Updating playlists for {len(changed_files)} changed files")
            placeholders = ','.join(['?'] * len(changed_files))
            cursor.execute(f"""
            DELETE FROM playlist_tracks
            WHERE file_hash IN (
                SELECT file_hash FROM audio_features
                WHERE file_path IN ({placeholders})
            )
            """, changed_files)

            cursor.execute("UPDATE playlists SET last_updated = CURRENT_TIMESTAMP")
            conn.commit()
            return len(changed_files)
            
        except Exception as e:
            logger.error(f"Playlist update failed: {str(e)}")
            conn.rollback()
            return 0

    def get_changed_files(self):
        """Get files that have changed since last playlist update"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT file_path
            FROM audio_features
            WHERE last_analyzed > (
                SELECT MAX(last_updated) FROM playlists
            )
            """)
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting changed files: {str(e)}")
            return []