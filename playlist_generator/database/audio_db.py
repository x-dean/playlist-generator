import sqlite3
import logging

logger = logging.getLogger(__name__)

class PlaylistDatabase:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.cache_file, timeout=60)
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
        conn.close()
    
    def playlists_exist(self):
        """Check if any playlists exist in the database"""
        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM playlists")
            count = cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"Error checking playlists: {str(e)}")
            return False
        finally:
            conn.close()
    
    def get_changed_files(self):
        """Get files that have changed since last playlist update"""
        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
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
        finally:
            conn.close()
    
    def save_playlists(self, playlists):
        """Save playlists to the database"""
        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
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
        finally:
            conn.close()
    
    def update_playlists(self, changed_files):
        """Update playlists based on changed files"""
        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()
            
            # Remove tracks from changed files
            placeholders = ','.join(['?'] * len(changed_files))
            cursor.execute(f"""
            DELETE FROM playlist_tracks
            WHERE file_hash IN (
                SELECT file_hash FROM audio_features
                WHERE file_path IN ({placeholders})
            )
            """, changed_files)
            
            # Update playlist timestamp
            cursor.execute("UPDATE playlists SET last_updated = CURRENT_TIMESTAMP")
            conn.commit()
            logger.info(f"Updated playlists for {len(changed_files)} changed files")
        except Exception as e:
            logger.error(f"Playlist update failed: {str(e)}")
        finally:
            conn.close()