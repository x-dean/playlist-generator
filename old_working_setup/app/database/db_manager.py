import sqlite3
import json
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database operations for playlists and caching."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        logger.debug(f"Initializing DatabaseManager with path: {db_path}")
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables."""
        logger.debug("Initializing database tables")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create playlists table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS playlists (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        tracks TEXT NOT NULL,
                        features TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.debug("Created playlists table")

                # Create cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.debug("Created cache table")

                # Create tags table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tags (
                        file_path TEXT PRIMARY KEY,
                        tags TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.debug("Created tags table")

                conn.commit()
                logger.info("Database initialization completed successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            import traceback
            logger.error(
                f"Database init error traceback: {traceback.format_exc()}")
            raise

    def save_playlist(self, name: str, tracks: List[str], features: Dict[str, Any] = None) -> bool:
        """Save a playlist to the database."""
        logger.debug(f"Saving playlist '{name}' with {len(tracks)} tracks")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                tracks_json = json.dumps(tracks)
                features_json = json.dumps(features) if features else None

                cursor.execute("""
                    INSERT OR REPLACE INTO playlists (name, tracks, features, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (name, tracks_json, features_json))

                conn.commit()
                logger.info(
                    f"Successfully saved playlist '{name}' to database")
                return True
        except Exception as e:
            logger.error(f"Error saving playlist {name}: {str(e)}")
            import traceback
            logger.error(
                f"Save playlist error traceback: {traceback.format_exc()}")
            return False

    def get_playlist(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a playlist from the database."""
        logger.debug(f"Retrieving playlist '{name}' from database")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT tracks, features, created_at, updated_at
                    FROM playlists WHERE name = ?
                """, (name,))

                row = cursor.fetchone()
                if row:
                    playlist = {
                        'name': name,
                        'tracks': json.loads(row[0]),
                        'features': json.loads(row[1]) if row[1] else {},
                        'created_at': row[2],
                        'updated_at': row[3]
                    }
                    logger.debug(
                        f"Successfully retrieved playlist '{name}' with {len(playlist['tracks'])} tracks")
                    return playlist
                else:
                    logger.debug(f"Playlist '{name}' not found in database")
                    return None
        except Exception as e:
            logger.error(f"Error getting playlist {name}: {str(e)}")
            import traceback
            logger.error(
                f"Get playlist error traceback: {traceback.format_exc()}")
            return None

    def save_cache(self, key: str, value: Any) -> bool:
        """Save a value to the cache."""
        logger.debug(f"Saving cache entry with key: {key}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                value_json = json.dumps(value)

                cursor.execute("""
                    INSERT OR REPLACE INTO cache (key, value, created_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, value_json))

                conn.commit()
                logger.debug(f"Successfully saved cache entry for key: {key}")
                return True
        except Exception as e:
            logger.error(f"Error caching computation {key}: {str(e)}")
            import traceback
            logger.error(
                f"Cache save error traceback: {traceback.format_exc()}")
            return False

    def get_cache(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""
        logger.debug(f"Retrieving cache entry with key: {key}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT value FROM cache WHERE key = ?
                """, (key,))

                row = cursor.fetchone()
                if row:
                    value = json.loads(row[0])
                    logger.debug(
                        f"Successfully retrieved cache entry for key: {key}")
                    return value
                else:
                    logger.debug(f"Cache entry not found for key: {key}")
                    return None
        except Exception as e:
            logger.error(f"Error getting cached computation {key}: {str(e)}")
            import traceback
            logger.error(
                f"Cache get error traceback: {traceback.format_exc()}")
            return None

    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """Clean up old cache entries."""
        logger.debug(
            f"Cleaning up cache entries older than {max_age_hours} hours")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM cache 
                    WHERE created_at < datetime('now', '-{} hours')
                """.format(max_age_hours))

                deleted_count = cursor.rowcount
                conn.commit()
                logger.info(f"Cleaned up {deleted_count} old cache entries")
                return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
            import traceback
            logger.error(
                f"Cache cleanup error traceback: {traceback.format_exc()}")
            return 0

    def save_tags(self, file_path: str, tags: Dict[str, Any]) -> bool:
        """Save tags for a file."""
        logger.debug(f"Saving tags for file: {file_path}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                tags_json = json.dumps(tags)

                cursor.execute("""
                    INSERT OR REPLACE INTO tags (file_path, tags, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (file_path, tags_json))

                conn.commit()
                logger.debug(f"Successfully saved tags for file: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Error adding tags for {file_path}: {str(e)}")
            import traceback
            logger.error(
                f"Save tags error traceback: {traceback.format_exc()}")
            return False

    def get_tags(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get tags for a file."""
        logger.debug(f"Retrieving tags for file: {file_path}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT tags FROM tags WHERE file_path = ?
                """, (file_path,))

                row = cursor.fetchone()
                if row:
                    tags = json.loads(row[0])
                    logger.debug(
                        f"Successfully retrieved tags for file: {file_path}")
                    return tags
                else:
                    logger.debug(f"No tags found for file: {file_path}")
                    return None
        except Exception as e:
            logger.error(f"Error getting tags for {file_path}: {str(e)}")
            import traceback
            logger.error(f"Get tags error traceback: {traceback.format_exc()}")
            return None

    def get_all_playlists(self) -> List[Dict[str, Any]]:
        """Get all playlists from the database."""
        logger.debug("Retrieving all playlists from database")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name, tracks, features, created_at, updated_at
                    FROM playlists ORDER BY updated_at DESC
                """)

                playlists = []
                for row in cursor.fetchall():
                    playlist = {
                        'name': row[0],
                        'tracks': json.loads(row[1]),
                        'features': json.loads(row[2]) if row[2] else {},
                        'created_at': row[3],
                        'updated_at': row[4]
                    }
                    playlists.append(playlist)

                logger.debug(
                    f"Retrieved {len(playlists)} playlists from database")
                return playlists
        except Exception as e:
            logger.error(f"Error getting all playlists: {str(e)}")
            import traceback
            logger.error(
                f"Get all playlists error traceback: {traceback.format_exc()}")
            return []

    def delete_playlist(self, name: str) -> bool:
        """Delete a playlist from the database."""
        logger.debug(f"Deleting playlist '{name}' from database")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM playlists WHERE name = ?", (name,))

                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(
                        f"Successfully deleted playlist '{name}' from database")
                    return True
                else:
                    logger.warning(f"Playlist '{name}' not found for deletion")
                    return False
        except Exception as e:
            logger.error(f"Error deleting playlist {name}: {str(e)}")
            import traceback
            logger.error(
                f"Delete playlist error traceback: {traceback.format_exc()}")
            return False

    def get_library_statistics(self) -> Dict[str, Any]:
        """Get library statistics from the database."""
        logger.debug("Retrieving library statistics from database")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get total playlists
                cursor.execute("SELECT COUNT(*) FROM playlists")
                total_playlists = cursor.fetchone()[0]

                # Get total tracks across all playlists
                cursor.execute("SELECT tracks FROM playlists")
                total_tracks = 0
                unique_tracks = set()
                for row in cursor.fetchall():
                    tracks = json.loads(row[0])
                    total_tracks += len(tracks)
                    unique_tracks.update(tracks)

                # Get cache statistics
                cursor.execute("SELECT COUNT(*) FROM cache")
                cache_entries = cursor.fetchone()[0]

                # Get tags statistics
                cursor.execute("SELECT COUNT(*) FROM tags")
                tagged_files = cursor.fetchone()[0]

                stats = {
                    'total_playlists': total_playlists,
                    'total_tracks': total_tracks,
                    'unique_tracks': len(unique_tracks),
                    'cache_entries': cache_entries,
                    'tagged_files': tagged_files
                }

                logger.debug(f"Retrieved library statistics: {stats}")
                return stats
        except Exception as e:
            logger.error(f"Error getting library statistics: {str(e)}")
            import traceback
            logger.error(
                f"Library statistics error traceback: {traceback.format_exc()}")
            return {}
