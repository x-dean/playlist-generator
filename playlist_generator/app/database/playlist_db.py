# database/playlist_db.py
import sqlite3
import json
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PlaylistDatabase:
    """Database manager for playlist operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        logger.debug(f"Initializing PlaylistDatabase with path: {db_path}")
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with required tables."""
        logger.debug("Initializing playlist database tables")
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
                
                conn.commit()
                logger.info("Playlist database initialization completed successfully")
        except Exception as e:
            logger.error(f"Playlist database initialization failed: {str(e)}")
            import traceback
            logger.error(f"Playlist database init error traceback: {traceback.format_exc()}")
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
                logger.info(f"Successfully saved playlist '{name}' to database")
                return True
        except Exception as e:
            logger.error(f"Error saving playlist {name}: {str(e)}")
            import traceback
            logger.error(f"Save playlist error traceback: {traceback.format_exc()}")
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
                    logger.debug(f"Successfully retrieved playlist '{name}' with {len(playlist['tracks'])} tracks")
                    return playlist
                else:
                    logger.debug(f"Playlist '{name}' not found in database")
                    return None
        except Exception as e:
            logger.error(f"Error getting playlist {name}: {str(e)}")
            import traceback
            logger.error(f"Get playlist error traceback: {traceback.format_exc()}")
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
                
                logger.debug(f"Retrieved {len(playlists)} playlists from database")
                return playlists
        except Exception as e:
            logger.error(f"Error getting all playlists: {str(e)}")
            import traceback
            logger.error(f"Get all playlists error traceback: {traceback.format_exc()}")
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
                    logger.info(f"Successfully deleted playlist '{name}' from database")
                    return True
                else:
                    logger.warning(f"Playlist '{name}' not found for deletion")
                    return False
        except Exception as e:
            logger.error(f"Error deleting playlist {name}: {str(e)}")
            import traceback
            logger.error(f"Delete playlist error traceback: {traceback.format_exc()}")
            return False
    
    def get_library_statistics(self) -> Dict[str, Any]:
        """Get library statistics from the database."""
        logger.debug("Retrieving library statistics from playlist database")
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
                
                stats = {
                    'total_playlists': total_playlists,
                    'total_tracks': total_tracks,
                    'unique_tracks': len(unique_tracks)
                }
                
                logger.debug(f"Retrieved playlist library statistics: {stats}")
                return stats
        except Exception as e:
            logger.error(f"Error getting library statistics: {str(e)}")
            import traceback
            logger.error(f"Library statistics error traceback: {traceback.format_exc()}")
            return {}