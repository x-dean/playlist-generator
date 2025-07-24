# database/playlist_db.py
import logging
from typing import Dict, List, Any, Optional
from .db_manager import DatabaseManager
import json

logger = logging.getLogger(__name__)

class PlaylistDatabase:
    def __init__(self, db_file: str):
        self.db = DatabaseManager(db_file)

    def playlists_exist(self) -> bool:
        """Check if any playlists exist in the database"""
        conn = self.db._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM playlists")
            count = cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"Error checking playlists: {str(e)}")
            return False

    def save_playlists(self, playlists: Dict[str, Dict[str, Any]]) -> bool:
        """Save multiple playlists to the database"""
        success = True
        for name, data in playlists.items():
            if not self.db.save_playlist(
                name=name,
                tracks=data.get('tracks', []),
                features=data.get('features'),
                description=data.get('description')
            ):
                success = False
        return success

    def get_playlist(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a playlist by name"""
        return self.db.get_playlist(name)

    def get_all_playlists(self) -> List[Dict[str, Any]]:
        """Get all playlists with their tracks"""
        conn = self.db._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM playlists")
            return [
                self.get_playlist(row['name'])
                for row in cursor.fetchall()
                if row['name']
            ]
        except Exception as e:
            logger.error(f"Error getting all playlists: {str(e)}")
            return []

    def update_playlists(self, changed_files: Optional[List[str]] = None) -> int:
        """Update playlists based on changed files"""
        conn = self.db._get_connection()
        try:
            cursor = conn.cursor()
            
            if changed_files is None:
                cursor.execute("""
                SELECT file_path
                FROM audio_features
                WHERE last_analyzed > (
                    SELECT MAX(last_updated) FROM playlists
                )
                """)
                changed_files = [row[0] for row in cursor.fetchall()]

            if not changed_files:
                logger.info("No changed files, playlists up-to-date")
                return 0

            # Get all playlists that contain changed files
            placeholders = ','.join(['?' for _ in changed_files])
            cursor.execute(f"""
            SELECT DISTINCT p.name, p.features
            FROM playlists p
            JOIN playlist_tracks pt ON pt.playlist_id = p.id
            JOIN audio_features af ON af.file_hash = pt.file_hash
            WHERE af.file_path IN ({placeholders})
            """, changed_files)

            affected_playlists = cursor.fetchall()
            
            # Regenerate affected playlists
            for playlist in affected_playlists:
                # Get current tracks
                cursor.execute("""
                SELECT af.file_path
                FROM playlist_tracks pt
                JOIN audio_features af ON af.file_hash = pt.file_hash
                WHERE pt.playlist_id = (
                    SELECT id FROM playlists WHERE name = ?
                )
                ORDER BY pt.position
                """, (playlist['name'],))
                
                tracks = [row[0] for row in cursor.fetchall()]
                
                # Save updated playlist
                self.db.save_playlist(
                    name=playlist['name'],
                    tracks=tracks,
                    features=playlist['features']
                )

            return len(affected_playlists)
            
        except Exception as e:
            logger.error(f"Error updating playlists: {str(e)}")
            return 0

    def get_changed_files(self) -> List[str]:
        """Get files that have changed since last playlist update"""
        conn = self.db._get_connection()
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

    def add_track_tags(self, file_path: str, tags: Dict[str, str]) -> bool:
        """Add or update tags for a track"""
        return self.db.add_track_tags(file_path, tags)

    def get_track_tags(self, file_path: str) -> Dict[str, str]:
        """Get all tags for a track"""
        return self.db.get_track_tags(file_path)

    def get_playlist_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a playlist or all playlists"""
        conn = self.db._get_connection()
        try:
            cursor = conn.cursor()
            
            if name:
                cursor.execute("""
                SELECT stats FROM playlists WHERE name = ?
                """, (name,))
                if result := cursor.fetchone():
                    return json.loads(result['stats']) if result['stats'] else {}
                return {}
            
            cursor.execute("SELECT name, stats FROM playlists")
            return {
                row['name']: json.loads(row['stats']) if row['stats'] else {}
                for row in cursor.fetchall()
            }
            
        except Exception as e:
            logger.error(f"Error getting playlist stats: {str(e)}")
            return {}

    def get_library_statistics(self) -> Dict[str, Any]:
        """Return statistics about the music library and playlists."""
        return self.db.get_library_statistics()