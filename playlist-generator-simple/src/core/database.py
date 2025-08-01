"""
Database manager for Playlist Generator Simple.
Handles playlist storage, caching, analysis results, discovery, and web UI optimization.
"""

import sqlite3
import json
import os
import time
import threading
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

# Import configuration and logging
from .config_loader import config_loader
from .logging_setup import get_logger, log_function_call, log_universal

logger = get_logger('playlista.database')


class DatabaseManager:
    """
    Comprehensive database manager optimized for web UI performance.
    
    Handles:
    - Playlist storage and retrieval
    - Analysis results storage
    - File discovery tracking
    - Tags and enrichment data
    - Advanced caching system
    - Statistics for web UI dashboards
    """

    def __init__(self, db_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Database file path (uses fixed Docker path if None)
            config: Optional configuration dictionary (uses global config if None)
        """
        if db_path is None:
            # Use local path for development, Docker path for production
            if os.path.exists('/app/cache'):
                db_path = '/app/cache/playlista.db'  # Docker internal path
            else:
                # Local development path
                import sys
                if hasattr(sys, '_MEIPASS'):  # PyInstaller
                    base_path = sys._MEIPASS
                else:
                    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                db_path = os.path.join(base_path, 'cache', 'playlista.db')
        
        self.db_path = db_path
        
        # Load configuration
        if config is None:
            config = config_loader.get_database_config()
        
        self.config = config
        
        # Database configuration settings
        self.cache_default_expiry_hours = config.get('DB_CACHE_DEFAULT_EXPIRY_HOURS', 24)
        self.connection_timeout_seconds = config.get('DB_CONNECTION_TIMEOUT_SECONDS', 30)
        self.max_retry_attempts = config.get('DB_MAX_RETRY_ATTEMPTS', 3)
        self.batch_size = config.get('DB_BATCH_SIZE', 100)
        
        log_universal('INFO', 'Database', f'Initializing DatabaseManager with path: {db_path}')
        
        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        log_universal('INFO', 'Database', f'Database directory ready: {db_dir}')
        
        # Initialize database tables
        start_time = time.time()
        self._init_database()
        init_time = time.time() - start_time
        
        # Log performance
        log_universal('INFO', 'Database', f"DatabaseManager initialization completed in {init_time:.2f}s")

    def _init_database(self):
        """Initialize the database with optimized schema."""
        log_universal('INFO', 'Database', "Initializing optimized database schema...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if database already has tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'")
                if cursor.fetchone():
                    log_universal('INFO', 'Database', "Database schema already exists")
                    return
                
                # Read and execute schema
                schema_path = os.path.join(os.path.dirname(self.db_path), 'database_schema.sql')
                if not os.path.exists(schema_path):
                    # Try relative path
                    schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database_schema.sql')
                
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schema_sql = f.read()
                    
                    cursor.executescript(schema_sql)
                    log_universal('INFO', 'Database', "Database schema created successfully")
                else:
                    log_universal('ERROR', 'Database', f"Schema file not found: {schema_path}")
                    raise FileNotFoundError(f"Schema file not found: {schema_path}")
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to initialize database: {e}")
            raise

    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.connection_timeout_seconds)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            log_universal('ERROR', 'Database', f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    # =============================================================================
    # DISCOVERY METHODS
    # =============================================================================

    @log_function_call
    def save_discovery_result(self, directory_path: str, file_count: int, 
                            scan_duration: float, status: str = 'completed',
                            error_message: str = None) -> bool:
        """Save file discovery results."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO discovery_cache 
                    (directory_path, file_count, last_scan_date, scan_duration, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (directory_path, file_count, datetime.now(), scan_duration, status, error_message))
                
                conn.commit()
                log_universal('INFO', 'Database', f"Discovery result saved for: {directory_path}")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to save discovery result: {e}")
            return False

    @log_function_call
    def get_discovery_status(self, directory_path: str) -> Optional[Dict[str, Any]]:
        """Get discovery status for a directory."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM discovery_cache WHERE directory_path = ?
                """, (directory_path,))
                
                result = cursor.fetchone()
                return dict(result) if result else None
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get discovery status: {e}")
            return None

    # =============================================================================
    # ANALYSIS METHODS
    # =============================================================================

    @log_function_call
    def save_analysis_result(self, file_path: str, filename: str, file_size_bytes: int,
                           file_hash: str, analysis_data: Dict[str, Any],
                           metadata: Dict[str, Any] = None, discovery_source: str = 'file_system') -> bool:
        """Save analysis result to database."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Extract core data
                title = metadata.get('title', 'Unknown') if metadata else 'Unknown'
                artist = metadata.get('artist', 'Unknown') if metadata else 'Unknown'
                album = metadata.get('album')
                track_number = metadata.get('track_number')
                genre = metadata.get('genre')
                year = metadata.get('year')
                duration = analysis_data.get('duration')
                
                # Extract audio features
                bpm = analysis_data.get('bpm')
                key = analysis_data.get('key')
                mode = analysis_data.get('mode')
                loudness = analysis_data.get('loudness')
                danceability = analysis_data.get('danceability')
                energy = analysis_data.get('energy')
                
                # Determine analysis type and category
                analysis_type = analysis_data.get('analysis_type', 'full')
                long_audio_category = analysis_data.get('long_audio_category')
                
                # Insert or update track
                cursor.execute("""
                    INSERT OR REPLACE INTO tracks (
                        file_path, file_hash, filename, file_size_bytes, analysis_date,
                        title, artist, album, track_number, genre, year, duration,
                        bpm, key, mode, loudness, danceability, energy,
                        analysis_type, long_audio_category, discovery_date, discovery_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_path, file_hash, filename, file_size_bytes, datetime.now(),
                    title, artist, album, track_number, genre, year, duration,
                    bpm, key, mode, loudness, danceability, energy,
                    analysis_type, long_audio_category, datetime.now(), discovery_source
                ))
                
                track_id = cursor.lastrowid
                
                # Save tags if available
                if metadata and 'tags' in metadata:
                    self._save_tags(cursor, track_id, metadata['tags'])
                
                conn.commit()
                log_universal('INFO', 'Database', f"Analysis result saved for: {filename}")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to save analysis result: {e}")
            return False

    @log_function_call
    def save_track_to_normalized_schema(self, file_path: str, filename: str, file_size_bytes: int,
                                      file_hash: str, analysis_data: Dict[str, Any],
                                      metadata: Dict[str, Any] = None) -> bool:
        """Save track to normalized schema (alias for save_analysis_result)."""
        return self.save_analysis_result(file_path, filename, file_size_bytes, file_hash, analysis_data, metadata)

    # =============================================================================
    # PLAYLIST METHODS
    # =============================================================================

    @log_function_call
    def save_playlist(self, name: str, tracks: List[str], description: str = None,
                     generation_method: str = 'manual', generation_params: Dict[str, Any] = None) -> bool:
        """Save playlist to database."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Insert playlist
                params_json = json.dumps(generation_params) if generation_params else None
                cursor.execute("""
                    INSERT INTO playlists (name, description, generation_method, generation_params)
                    VALUES (?, ?, ?, ?)
                """, (name, description, generation_method, params_json))
                
                playlist_id = cursor.lastrowid
                
                # Insert playlist tracks
                for position, file_path in enumerate(tracks, 1):
                    # Get track_id from file_path
                    cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                    track_result = cursor.fetchone()
                    
                    if track_result:
                        track_id = track_result['id']
                        cursor.execute("""
                            INSERT INTO playlist_tracks (playlist_id, track_id, position)
                            VALUES (?, ?, ?)
                        """, (playlist_id, track_id, position))
                
                conn.commit()
                log_universal('INFO', 'Database', f"Playlist '{name}' saved with {len(tracks)} tracks")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to save playlist: {e}")
            return False

    @log_function_call
    def get_playlist(self, name: str) -> Optional[Dict[str, Any]]:
        """Get playlist by name with track details."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get playlist details
                cursor.execute("""
                    SELECT * FROM playlists WHERE name = ?
                """, (name,))
                
                playlist_result = cursor.fetchone()
                if not playlist_result:
                    return None
                
                # Get playlist tracks with full track data
                cursor.execute("""
                    SELECT t.*, pt.position 
                    FROM playlist_tracks pt
                    JOIN tracks t ON pt.track_id = t.id
                    WHERE pt.playlist_id = ?
                    ORDER BY pt.position
                """, (playlist_result['id'],))
                
                tracks = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'id': playlist_result['id'],
                    'name': playlist_result['name'],
                    'description': playlist_result['description'],
                    'generation_method': playlist_result['generation_method'],
                    'generation_params': json.loads(playlist_result['generation_params']) if playlist_result['generation_params'] else None,
                    'track_count': playlist_result['track_count'],
                    'total_duration': playlist_result['total_duration'],
                    'created_at': playlist_result['created_at'],
                    'updated_at': playlist_result['updated_at'],
                    'tracks': tracks
                }
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get playlist: {e}")
            return None

    @log_function_call
    def get_all_playlists(self) -> List[Dict[str, Any]]:
        """Get all playlists with summary information."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM playlist_summary ORDER BY created_at DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get playlists: {e}")
            return []

    # =============================================================================
    # CACHE METHODS
    # =============================================================================

    @log_function_call
    def save_cache(self, key: str, value: Any, cache_type: str = 'general', 
                  expires_hours: int = None) -> bool:
        """Save data to cache with type classification."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                expires_at = None
                if expires_hours:
                    expires_at = datetime.now() + timedelta(hours=expires_hours)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache (cache_key, cache_value, cache_type, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (key, json.dumps(value), cache_type, expires_at))
                
                conn.commit()
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to save cache: {e}")
            return False

    @log_function_call
    def get_cache(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT cache_value, expires_at FROM cache WHERE cache_key = ?
                """, (key,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                # Check if expired
                if result['expires_at']:
                    expires_at = datetime.fromisoformat(result['expires_at'])
                    if datetime.now() > expires_at:
                        cursor.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
                        conn.commit()
                        return None
                
                return json.loads(result['cache_value'])
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get cache: {e}")
            return None

    @log_function_call
    def get_cache_by_type(self, cache_type: str) -> List[Dict[str, Any]]:
        """Get all cache entries of a specific type."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM cache WHERE cache_type = ? ORDER BY created_at DESC
                """, (cache_type,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get cache by type: {e}")
            return []

    # =============================================================================
    # STATISTICS METHODS
    # =============================================================================

    @log_function_call
    def save_statistic(self, category: str, metric_name: str, metric_value: float,
                      metric_data: Dict[str, Any] = None) -> bool:
        """Save a statistic for web UI dashboards."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                data_json = json.dumps(metric_data) if metric_data else None
                cursor.execute("""
                    INSERT INTO statistics (category, metric_name, metric_value, metric_data)
                    VALUES (?, ?, ?, ?)
                """, (category, metric_name, metric_value, data_json))
                
                conn.commit()
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to save statistic: {e}")
            return False

    @log_function_call
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get statistics summary for web UI dashboard."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM statistics_summary")
                stats = [dict(row) for row in cursor.fetchall()]
                
                # Group by category
                summary = {}
                for stat in stats:
                    category = stat['category']
                    if category not in summary:
                        summary[category] = []
                    summary[category].append(stat)
                
                return summary
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get statistics summary: {e}")
            return {}

    # =============================================================================
    # WEB UI OPTIMIZED QUERIES
    # =============================================================================

    @log_function_call
    def get_tracks_for_web_ui(self, limit: int = 50, offset: int = 0,
                             artist: str = None, genre: str = None, 
                             year: int = None, sort_by: str = 'analysis_date') -> List[Dict[str, Any]]:
        """Get tracks optimized for web UI display."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Build query with filters
                query = "SELECT * FROM track_summary WHERE 1=1"
                params = []
                
                if artist:
                    query += " AND artist LIKE ?"
                    params.append(f"%{artist}%")
                
                if genre:
                    query += " AND genre = ?"
                    params.append(genre)
                
                if year:
                    query += " AND year = ?"
                    params.append(year)
                
                # Add sorting
                valid_sort_fields = ['analysis_date', 'title', 'artist', 'album', 'year', 'bpm']
                if sort_by in valid_sort_fields:
                    query += f" ORDER BY {sort_by}"
                else:
                    query += " ORDER BY analysis_date"
                
                query += " DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get tracks for web UI: {e}")
            return []

    @log_function_call
    def get_web_ui_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for web UI."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                dashboard_data = {}
                
                # Track counts
                cursor.execute("SELECT COUNT(*) as total_tracks FROM tracks")
                dashboard_data['total_tracks'] = cursor.fetchone()['total_tracks']
                
                cursor.execute("SELECT COUNT(DISTINCT artist) as unique_artists FROM tracks")
                dashboard_data['unique_artists'] = cursor.fetchone()['unique_artists']
                
                cursor.execute("SELECT COUNT(DISTINCT album) as unique_albums FROM tracks")
                dashboard_data['unique_albums'] = cursor.fetchone()['unique_albums']
                
                cursor.execute("SELECT COUNT(DISTINCT genre) as unique_genres FROM tracks")
                dashboard_data['unique_genres'] = cursor.fetchone()['unique_genres']
                
                # Playlist counts
                cursor.execute("SELECT COUNT(*) as total_playlists FROM playlists")
                dashboard_data['total_playlists'] = cursor.fetchone()['total_playlists']
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) as recent_analyses 
                    FROM tracks 
                    WHERE analysis_date > datetime('now', '-7 days')
                """)
                dashboard_data['recent_analyses'] = cursor.fetchone()['recent_analyses']
                
                # Genre distribution
                cursor.execute("""
                    SELECT genre, COUNT(*) as count 
                    FROM tracks 
                    WHERE genre IS NOT NULL 
                    GROUP BY genre 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                dashboard_data['genre_distribution'] = [dict(row) for row in cursor.fetchall()]
                
                # Year distribution
                cursor.execute("""
                    SELECT year, COUNT(*) as count 
                    FROM tracks 
                    WHERE year IS NOT NULL 
                    GROUP BY year 
                    ORDER BY year DESC 
                    LIMIT 10
                """)
                dashboard_data['year_distribution'] = [dict(row) for row in cursor.fetchall()]
                
                return dashboard_data
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get dashboard data: {e}")
            return {}

    def _save_tags(self, cursor, track_id: int, tags: Dict[str, Any]):
        """Save tags for a track."""
        for source, tag_data in tags.items():
            if isinstance(tag_data, dict):
                for tag_name, tag_value in tag_data.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO tags (track_id, source, tag_name, tag_value)
                        VALUES (?, ?, ?, ?)
                    """, (track_id, source, tag_name, str(tag_value)))
            elif isinstance(tag_data, list):
                for tag_name in tag_data:
                    cursor.execute("""
                        INSERT OR REPLACE INTO tags (track_id, source, tag_name)
                        VALUES (?, ?, ?)
                    """, (track_id, source, tag_name))


def get_db_manager() -> 'DatabaseManager':
    """Get database manager instance."""
    return DatabaseManager()
