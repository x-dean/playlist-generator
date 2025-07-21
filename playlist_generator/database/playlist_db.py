# database/playlist_db.py
import sqlite3
import logging

logger = logging.getLogger(__name__)

class PlaylistDatabase:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self._init_db()
    
    def _init_db(self):
        # Move this method outside of update_playlists
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

    def update_playlists(self, changed_files=None):
        """Update playlists based on changed files"""
        try:
            conn = sqlite3.connect(self.cache_file, timeout=60)
            cursor = conn.cursor()

            # Get changed files if not provided
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
                return

            logger.info(f"Updating playlists for {len(changed_files)} changed files")

            # Remove affected tracks from all playlists
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

        except Exception as e:
            logger.error(f"Playlist update failed: {str(e)}")
        finally:
            conn.close()