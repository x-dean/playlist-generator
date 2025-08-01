"""
Database manager for Playlist Generator Simple.
Handles playlist storage, caching, analysis results, and metadata.
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
    Comprehensive database manager for playlist operations, caching, and analysis results.
    
    Handles:
    - Playlist storage and retrieval
    - Analysis results caching
    - File metadata storage
    - Tags and enrichment data
    - Statistics and reporting
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
        self.cache_cleanup_frequency_hours = config.get('DB_CACHE_CLEANUP_FREQUENCY_HOURS', 24)
        self.cache_max_size_mb = config.get('DB_CACHE_MAX_SIZE_MB', 100)
        self.cleanup_retention_days = config.get('DB_CLEANUP_RETENTION_DAYS', 30)
        self.failed_analysis_retention_days = config.get('DB_FAILED_ANALYSIS_RETENTION_DAYS', 7)
        self.statistics_retention_days = config.get('DB_STATISTICS_RETENTION_DAYS', 90)
        self.connection_timeout_seconds = config.get('DB_CONNECTION_TIMEOUT_SECONDS', 30)
        self.max_retry_attempts = config.get('DB_MAX_RETRY_ATTEMPTS', 3)
        self.batch_size = config.get('DB_BATCH_SIZE', 100)
        self.statistics_collection_frequency_hours = config.get('DB_STATISTICS_COLLECTION_FREQUENCY_HOURS', 24)
        self.auto_cleanup_enabled = config.get('DB_AUTO_CLEANUP_ENABLED', True)
        self.auto_cleanup_frequency_hours = config.get('DB_AUTO_CLEANUP_FREQUENCY_HOURS', 168)
        self.backup_enabled = config.get('DB_BACKUP_ENABLED', True)
        self.backup_frequency_hours = config.get('DB_BACKUP_FREQUENCY_HOURS', 168)
        self.backup_retention_days = config.get('DB_BACKUP_RETENTION_DAYS', 30)
        self.performance_monitoring_enabled = config.get('DB_PERFORMANCE_MONITORING_ENABLED', True)
        self.query_timeout_seconds = config.get('DB_QUERY_TIMEOUT_SECONDS', 60)
        self.max_connections = config.get('DB_MAX_CONNECTIONS', 10)
        self.wal_mode_enabled = config.get('DB_WAL_MODE_ENABLED', True)
        self.synchronous_mode = config.get('DB_SYNCHRONOUS_MODE', 'NORMAL')
        
        # Connection pooling
        self._connection_lock = threading.Lock()
        self._active_connections = 0
        self._max_connections = config.get('DB_MAX_CONNECTIONS', 10)
        
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

    def _check_and_migrate_schema(self, cursor):
        """Check and migrate database schema if needed."""
        try:
            log_universal('INFO', 'Database', "Checking database schema...")
            
            # Check if schema_version table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_version'
            """)
            
            if not cursor.fetchone():
                # Create schema_version table
                cursor.execute("""
                    CREATE TABLE schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("INSERT INTO schema_version (version) VALUES (1)")
                log_universal('INFO', 'Database', "Created schema_version table")
                return
            
            # Check current schema version
            cursor.execute("SELECT MAX(version) FROM schema_version")
            current_version = cursor.fetchone()[0] or 0
            log_universal('INFO', 'Database', f"Current schema version: {current_version}")
            
            if current_version < 2:
                # Migrate to version 2 (add missing columns)
                log_universal('INFO', 'Database', "Migrating database schema to version 2...")
                
                # Check if analysis_results table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='analysis_results'
                """)
                
                if cursor.fetchone():
                    log_universal('INFO', 'Database', "analysis_results table exists, adding missing columns...")
                    # Add missing columns if they don't exist
                    try:
                        cursor.execute("ALTER TABLE analysis_results ADD COLUMN artist TEXT")
                        log_universal('INFO', 'Database', "Added artist column")
                    except sqlite3.OperationalError as e:
                        log_universal('INFO', 'Database', f"artist column already exists: {e}")
                    
                    try:
                        cursor.execute("ALTER TABLE analysis_results ADD COLUMN album TEXT")
                        log_universal('INFO', 'Database', "Added album column")
                    except sqlite3.OperationalError as e:
                        log_universal('INFO', 'Database', f"album column already exists: {e}")
                    
                    try:
                        cursor.execute("ALTER TABLE analysis_results ADD COLUMN title TEXT")
                        log_universal('INFO', 'Database', "Added title column")
                    except sqlite3.OperationalError as e:
                        log_universal('INFO', 'Database', f"title column already exists: {e}")
                    
                    try:
                        cursor.execute("ALTER TABLE analysis_results ADD COLUMN genre TEXT")
                        log_universal('INFO', 'Database', "Added genre column")
                    except sqlite3.OperationalError as e:
                        log_universal('INFO', 'Database', f"genre column already exists: {e}")
                    
                    try:
                        cursor.execute("ALTER TABLE analysis_results ADD COLUMN year INTEGER")
                        log_universal('INFO', 'Database', "Added year column")
                    except sqlite3.OperationalError as e:
                        log_universal('INFO', 'Database', f"year column already exists: {e}")
                else:
                    log_universal('INFO', 'Database', "analysis_results table does not exist yet")
                
                cursor.execute("INSERT INTO schema_version (version) VALUES (2)")
                log_universal('INFO', 'Database', "Database schema migrated to version 2")
            else:
                log_universal('INFO', 'Database', "Database schema is up to date")
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error during schema migration: {e}")
            # Continue with normal initialization

    @log_function_call
    def _init_database(self):
        """Initialize the database with all required tables."""
        log_universal('INFO', 'Database', "Initializing database tables...")
        
        tables_created = 0
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if database needs migration
                self._check_and_migrate_schema(cursor)
                
                # Create playlists table
                log_universal('DEBUG', 'Database', "Creating playlists table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS playlists (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        tracks TEXT NOT NULL,
                        features TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                tables_created += 1
                log_universal('DEBUG', 'Database', "Playlists table ready")
                
                # Create analysis_results table
                log_universal('DEBUG', 'Database', "Creating analysis_results table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        file_path TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_size_bytes INTEGER NOT NULL,
                        file_hash TEXT NOT NULL,
                        analysis_data TEXT NOT NULL,
                        metadata TEXT,
                        artist TEXT,
                        album TEXT,
                        title TEXT,
                        genre TEXT,
                        year INTEGER,
                        analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                tables_created += 1
                log_universal('DEBUG', 'Database', "Analysis results table ready")
                
                # Create cache table
                log_universal('DEBUG', 'Database', "Creating cache table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                tables_created += 1
                log_universal('DEBUG', 'Database', "Cache table ready")
                
                # Create tags table
                log_universal('DEBUG', 'Database', "Creating tags table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tags (
                        file_path TEXT PRIMARY KEY,
                        tags TEXT NOT NULL,
                        source TEXT,
                        confidence REAL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                tables_created += 1
                log_universal('DEBUG', 'Database', "Tags table ready")
                
                # Create failed_analysis table
                log_universal('DEBUG', 'Database', "Creating failed_analysis table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS failed_analysis (
                        file_path TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        failed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_retry TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                tables_created += 1
                log_universal('DEBUG', 'Database', "Failed analysis table ready")
                
                # Create statistics table
                log_universal('DEBUG', 'Database', "Creating statistics table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(category, key, timestamp)
                    )
                """)
                tables_created += 1
                log_universal('DEBUG', 'Database', "Statistics table ready")
                
                # Create indexes for better query performance
                log_universal('DEBUG', 'Database', "Creating database indexes...")
                
                # Indexes for analysis_results table - with error handling for missing columns
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_artist ON analysis_results(artist)")
                except sqlite3.OperationalError:
                    log_universal('DEBUG', 'Database', "Skipping artist index - column may not exist yet")
                
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_album ON analysis_results(album)")
                except sqlite3.OperationalError:
                    log_universal('DEBUG', 'Database', "Skipping album index - column may not exist yet")
                
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_genre ON analysis_results(genre)")
                except sqlite3.OperationalError:
                    log_universal('DEBUG', 'Database', "Skipping genre index - column may not exist yet")
                
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_year ON analysis_results(year)")
                except sqlite3.OperationalError:
                    log_universal('DEBUG', 'Database', "Skipping year index - column may not exist yet")
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_date ON analysis_results(analysis_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_filename ON analysis_results(filename)")
                
                # Indexes for cache table
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created_at)")
                
                # Indexes for tags table
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_source ON tags(source)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_updated ON tags(updated_at)")
                
                # Indexes for failed_analysis table
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_failed_retry_count ON failed_analysis(retry_count)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_failed_date ON failed_analysis(failed_date)")
                
                # Indexes for statistics table
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_category ON statistics(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON statistics(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_category_key ON statistics(category, key)")
                
                # Indexes for playlists table
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_playlists_created ON playlists(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_playlists_updated ON playlists(updated_at)")
                
                conn.commit()
                init_time = time.time() - start_time
                log_universal('INFO', 'Database', f"Database initialization completed successfully")
                log_universal('INFO', 'Database', f"Created {tables_created} tables and indexes in {init_time:.2f}s")
                
                # Log performance
                log_universal('INFO', 'Database', f"Database table creation completed in {init_time:.2f}s")
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Database initialization failed: {e}")
            raise

    @contextmanager
    def _get_db_connection(self):
        """
        Get a database connection with connection pooling.
        
        Yields:
            SQLite connection object
        """
        with self._connection_lock:
            if self._active_connections >= self._max_connections:
                log_universal('WARNING', 'Database', f"Maximum database connections reached ({self._max_connections})")
            
            self._active_connections += 1
            log_universal('DEBUG', 'Database', f"Database connection acquired ({self._active_connections}/{self._max_connections})")
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.connection_timeout_seconds)
            
            # Configure connection
            if self.wal_mode_enabled:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA synchronous={self.synchronous_mode}")
            
            yield conn
        finally:
            conn.close()
            with self._connection_lock:
                self._active_connections -= 1
                log_universal('DEBUG', 'Database', f"Database connection released ({self._active_connections}/{self._max_connections})")

    # =============================================================================
    # PLAYLIST OPERATIONS
    # =============================================================================

    @log_function_call
    def save_playlist(self, name: str, tracks: List[str], description: str = None, 
                     features: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Save a playlist to the database.
        
        Args:
            name: Playlist name
            tracks: List of track file paths
            description: Optional playlist description
            features: Optional playlist features/characteristics
            metadata: Optional additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('INFO', 'Database', f"Saving playlist '{name}' with {len(tracks)} tracks")
        
        start_time = time.time()
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                tracks_json = json.dumps(tracks)
                features_json = json.dumps(features) if features else None
                metadata_json = json.dumps(metadata) if metadata else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO playlists 
                    (name, description, tracks, features, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (name, description, tracks_json, features_json, metadata_json))
                
                conn.commit()
                save_time = time.time() - start_time
                log_universal('INFO', 'Database', f"Successfully saved playlist '{name}' to database in {save_time:.2f}s")
                
                # Log performance
                log_universal('INFO', 'Database', f"Playlist save completed in {save_time:.2f}s")
                return True
                
        except sqlite3.Error as e:
            log_universal('ERROR', 'Database', f"Database error saving playlist '{name}': {e}")
            return False
        except Exception as e:
            log_universal('ERROR', 'Database', f"Unexpected error saving playlist '{name}': {e}")
            return False

    @log_function_call
    def get_playlist(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a playlist from the database.
        
        Args:
            name: Playlist name
            
        Returns:
            Playlist data dictionary or None if not found
        """
        log_universal('DEBUG', 'Database', f"Retrieving playlist '{name}' from database")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name, description, tracks, features, metadata, 
                           created_at, updated_at
                    FROM playlists WHERE name = ?
                """, (name,))
                
                row = cursor.fetchone()
                if row:
                    playlist = {
                        'name': row[0],
                        'description': row[1],
                        'tracks': json.loads(row[2]),
                        'features': json.loads(row[3]) if row[3] else None,
                        'metadata': json.loads(row[4]) if row[4] else None,
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                    log_universal('DEBUG', 'Database', f"Retrieved playlist '{name}' with {len(playlist['tracks'])} tracks")
                    return playlist
                else:
                    log_universal('DEBUG', 'Database', f"Playlist '{name}' not found in database")
                    return None
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error retrieving playlist '{name}': {e}")
            return None

    @log_function_call
    def get_all_playlists(self) -> List[Dict[str, Any]]:
        """
        Get all playlists from the database.
        
        Returns:
            List of playlist dictionaries
        """
        log_universal('DEBUG', 'Database', "Retrieving all playlists from database")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name, description, tracks, features, metadata, 
                           created_at, updated_at
                    FROM playlists ORDER BY updated_at DESC
                """)
                
                playlists = []
                for row in cursor.fetchall():
                    playlist = {
                        'name': row[0],
                        'description': row[1],
                        'tracks': json.loads(row[2]),
                        'features': json.loads(row[3]) if row[3] else None,
                        'metadata': json.loads(row[4]) if row[4] else None,
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                    playlists.append(playlist)
                
                log_universal('INFO', 'Database', f"Retrieved {len(playlists)} playlists from database")
                return playlists
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error retrieving all playlists: {e}")
            return []

    @log_function_call
    def delete_playlist(self, name: str) -> bool:
        """
        Delete a playlist from the database.
        
        Args:
            name: Playlist name to delete
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('INFO', 'Database', f"Deleting playlist '{name}' from database")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM playlists WHERE name = ?", (name,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    log_universal('INFO', 'Database', f"Successfully deleted playlist '{name}' from database")
                    return True
                else:
                    log_universal('WARNING', 'Database', f"Playlist '{name}' not found for deletion")
                    return False
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error deleting playlist '{name}': {e}")
            return False

    @log_function_call
    def get_analyzed_tracks(self) -> List[Dict[str, Any]]:
        """
        Get all analyzed tracks from the database.
        
        Returns:
            List of track dictionaries with analysis data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all analyzed tracks with their features
                cursor.execute("""
                    SELECT filepath, filename, file_size_bytes, file_hash,
                           bpm, centroid, danceability, loudness, key, scale,
                           onset_rate, zcr, analysis_timestamp, metadata
                    FROM audio_features
                    WHERE analysis_status = 'completed'
                    ORDER BY analysis_timestamp DESC
                """)
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'filepath': row[0],
                        'filename': row[1],
                        'file_size_bytes': row[2],
                        'file_hash': row[3],
                        'bpm': row[4],
                        'centroid': row[5],
                        'danceability': row[6],
                        'loudness': row[7],
                        'key': row[8],
                        'scale': row[9],
                        'onset_rate': row[10],
                        'zcr': row[11],
                        'analysis_timestamp': row[12],
                        'metadata': json.loads(row[13]) if row[13] else {}
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} analyzed tracks")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting analyzed tracks: {e}")
            return []
    
    @log_function_call
    def get_cached_playlists(self) -> List[Dict[str, Any]]:
        """
        Get cached playlists from the database.
        
        Returns:
            List of cached playlist dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get cached playlists
                cursor.execute("""
                    SELECT name, tracks, description, features, metadata, created_at
                    FROM playlists
                    WHERE features IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 50
                """)
                
                cached_playlists = []
                for row in cursor.fetchall():
                    playlist = {
                        'name': row[0],
                        'tracks': json.loads(row[1]) if row[1] else [],
                        'description': row[2],
                        'features': json.loads(row[3]) if row[3] else {},
                        'metadata': json.loads(row[4]) if row[4] else {},
                        'created_at': row[5]
                    }
                    cached_playlists.append(playlist)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(cached_playlists)} cached playlists")
                return cached_playlists
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting cached playlists: {e}")
            return []
    
    @log_function_call
    def get_track_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get features for a specific track.
        
        Args:
            file_path: Path to the track file
            
        Returns:
            Dictionary with track features or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT bpm, centroid, danceability, loudness, key, scale,
                           onset_rate, zcr, metadata
                    FROM audio_features
                    WHERE filepath = ? AND analysis_status = 'completed'
                """, (file_path,))
                
                row = cursor.fetchone()
                if row:
                    features = {
                        'bpm': row[0],
                        'centroid': row[1],
                        'danceability': row[2],
                        'loudness': row[3],
                        'key': row[4],
                        'scale': row[5],
                        'onset_rate': row[6],
                        'zcr': row[7],
                        'metadata': json.loads(row[8]) if row[8] else {}
                    }
                    return features
                else:
                    return None
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting track features for {file_path}: {e}")
            return None
    
    @log_function_call
    def get_tracks_by_artist(self, artist: str) -> List[Dict[str, Any]]:
        """
        Get all tracks by a specific artist.
        
        Args:
            artist: Artist name to search for
            
        Returns:
            List of track dictionaries
        """
        log_universal('DEBUG', 'Database', f"Getting tracks by artist: {artist}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT file_path, filename, artist, album, title, genre, year, metadata
                    FROM analysis_results 
                    WHERE artist LIKE ? AND artist IS NOT NULL
                    ORDER BY album, title
                """, (f'%{artist}%',))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'file_path': row[0],
                        'filename': row[1],
                        'artist': row[2],
                        'album': row[3],
                        'title': row[4],
                        'genre': row[5],
                        'year': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {}
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks by artist: {artist}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by artist: {e}")
            return []

    @log_function_call
    def get_tracks_by_album(self, album: str) -> List[Dict[str, Any]]:
        """
        Get all tracks from a specific album.
        
        Args:
            album: Album name to search for
            
        Returns:
            List of track dictionaries
        """
        log_universal('DEBUG', 'Database', f"Getting tracks by album: {album}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT file_path, filename, artist, album, title, genre, year, metadata
                    FROM analysis_results 
                    WHERE album LIKE ? AND album IS NOT NULL
                    ORDER BY tracknumber, title
                """, (f'%{album}%',))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'file_path': row[0],
                        'filename': row[1],
                        'artist': row[2],
                        'album': row[3],
                        'title': row[4],
                        'genre': row[5],
                        'year': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {}
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks from album: {album}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by album: {e}")
            return []

    @log_function_call
    def get_tracks_by_genre(self, genre: str) -> List[Dict[str, Any]]:
        """
        Get all tracks of a specific genre.
        
        Args:
            genre: Genre to search for
            
        Returns:
            List of track dictionaries
        """
        log_universal('DEBUG', 'Database', f"Getting tracks by genre: {genre}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT file_path, filename, artist, album, title, genre, year, metadata
                    FROM analysis_results 
                    WHERE genre LIKE ? AND genre IS NOT NULL
                    ORDER BY artist, album, title
                """, (f'%{genre}%',))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'file_path': row[0],
                        'filename': row[1],
                        'artist': row[2],
                        'album': row[3],
                        'title': row[4],
                        'genre': row[5],
                        'year': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {}
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks of genre: {genre}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by genre: {e}")
            return []

    @log_function_call
    def get_all_artists(self) -> List[str]:
        """
        Get all unique artists in the database.
        
        Returns:
            List of artist names
        """
        log_universal('DEBUG', 'Database', "Getting all unique artists...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT artist 
                    FROM analysis_results 
                    WHERE artist IS NOT NULL AND artist != ''
                    ORDER BY artist
                """)
                
                artists = [row[0] for row in cursor.fetchall()]
                log_universal('DEBUG', 'Database', f"Found {len(artists)} unique artists")
                return artists
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting all artists: {e}")
            return []

    @log_function_call
    def get_all_albums(self) -> List[str]:
        """
        Get all unique albums in the database.
        
        Returns:
            List of album names
        """
        log_universal('DEBUG', 'Database', "Getting all unique albums...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT album 
                    FROM analysis_results 
                    WHERE album IS NOT NULL AND album != ''
                    ORDER BY album
                """)
                
                albums = [row[0] for row in cursor.fetchall()]
                log_universal('DEBUG', 'Database', f"Found {len(albums)} unique albums")
                return albums
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting all albums: {e}")
            return []

    @log_function_call
    def get_all_genres(self) -> List[str]:
        """
        Get all unique genres in the database.
        
        Returns:
            List of genre names
        """
        log_universal('DEBUG', 'Database', "Getting all unique genres...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT genre 
                    FROM analysis_results 
                    WHERE genre IS NOT NULL AND genre != ''
                    ORDER BY genre
                """)
                
                genres = [row[0] for row in cursor.fetchall()]
                log_universal('DEBUG', 'Database', f"Found {len(genres)} unique genres")
                return genres
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting all genres: {e}")
            return []

    @log_function_call
    def get_tracks_by_year(self, year: int) -> List[Dict[str, Any]]:
        """
        Get all tracks from a specific year.
        
        Args:
            year: Year to search for
            
        Returns:
            List of track dictionaries
        """
        log_universal('DEBUG', 'Database', f"Getting tracks from year: {year}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT file_path, filename, artist, album, title, genre, year, metadata
                    FROM analysis_results 
                    WHERE year = ? AND year IS NOT NULL
                    ORDER BY artist, album, title
                """, (year,))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'file_path': row[0],
                        'filename': row[1],
                        'artist': row[2],
                        'album': row[3],
                        'title': row[4],
                        'genre': row[5],
                        'year': row[6],
                        'metadata': json.loads(row[7]) if row[7] else {}
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks from year: {year}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by year: {e}")
            return []

    @log_function_call
    def get_tracks_by_feature_range(self, feature: str, min_value: float, max_value: float) -> List[Dict[str, Any]]:
        """
        Get tracks within a specific feature range.
        
        Args:
            feature: Feature name (bpm, danceability, etc.)
            min_value: Minimum feature value
            max_value: Maximum feature value
            
        Returns:
            List of track dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Validate feature name to prevent SQL injection
                valid_features = ['bpm', 'centroid', 'danceability', 'loudness', 'key', 'scale', 'onset_rate', 'zcr']
                if feature not in valid_features:
                    log_universal('ERROR', 'Database', f"Invalid feature name: {feature}")
                    return []
                
                cursor.execute(f"""
                    SELECT filepath, filename, {feature}, metadata
                    FROM audio_features
                    WHERE {feature} BETWEEN ? AND ? AND analysis_status = 'completed'
                    ORDER BY {feature}
                """, (min_value, max_value))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'filepath': row[0],
                        'filename': row[1],
                        feature: row[2],
                        'metadata': json.loads(row[3]) if row[3] else {}
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks with {feature} between {min_value} and {max_value}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by feature range: {e}")
            return []

    # =============================================================================
    # ANALYSIS RESULTS OPERATIONS
    # =============================================================================

    @log_function_call
    def save_analysis_result(self, file_path: str, filename: str, file_size_bytes: int,
                           file_hash: str, analysis_data: Dict[str, Any],
                           metadata: Dict[str, Any] = None) -> bool:
        """
        Save analysis results for a file.
        
        Args:
            file_path: Path to the analyzed file
            filename: Name of the file
            file_size_bytes: File size in bytes
            file_hash: File hash for change detection
            analysis_data: Analysis results data
            metadata: Optional additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('DEBUG', 'Database', f"Saving analysis results for: {filename}")
        
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Ensure analysis status is set to 'analyzed' when saving complete analysis results
                if analysis_data and 'status' not in analysis_data:
                    analysis_data['status'] = 'analyzed'
                
                # Convert numpy arrays and other non-serializable objects to JSON-compatible format
                analysis_data_serializable = self._convert_to_json_serializable(analysis_data)
                metadata_serializable = self._convert_to_json_serializable(metadata) if metadata else None
                
                analysis_json = json.dumps(analysis_data_serializable)
                metadata_json = json.dumps(metadata_serializable) if metadata_serializable else None
                
                # Extract metadata fields for efficient querying
                artist = metadata.get('artist') if metadata else None
                album = metadata.get('album') if metadata else None
                title = metadata.get('title') if metadata else None
                genre = metadata.get('genre') if metadata else None
                year = metadata.get('year') if metadata else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_results 
                    (file_path, filename, file_size_bytes, file_hash, 
                     analysis_data, metadata, artist, album, title, genre, year, analysis_date, last_checked)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (file_path, filename, file_size_bytes, file_hash, 
                     analysis_json, metadata_json, artist, album, title, genre, year))
                
                conn.commit()
                save_time = time.time() - start_time
                log_universal('DEBUG', 'Database', f"Successfully saved analysis results for: {filename} in {save_time:.2f}s")
                
                # Log performance
                log_universal('INFO', 'Database', f"Analysis result save completed in {save_time:.2f}s")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error saving analysis results for {filename}: {e}")
            return False

    @log_function_call
    def get_analysis_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis results for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Analysis results dictionary or None if not found
        """
        log_universal('DEBUG', 'Database', f"Retrieving analysis results for: {file_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT filename, file_size_bytes, file_hash, analysis_data, 
                           metadata, artist, album, title, genre, year, analysis_date, last_checked
                    FROM analysis_results WHERE file_path = ?
                """, (file_path,))
                
                row = cursor.fetchone()
                if row:
                    result = {
                        'file_path': file_path,
                        'filename': row[0],
                        'file_size_bytes': row[1],
                        'file_hash': row[2],
                        'analysis_data': json.loads(row[3]),
                        'metadata': json.loads(row[4]) if row[4] else None,
                        'artist': row[5],
                        'album': row[6],
                        'title': row[7],
                        'genre': row[8],
                        'year': row[9],
                        'analysis_date': row[10],
                        'last_checked': row[11]
                    }
                    log_universal('DEBUG', 'Database', f"Retrieved analysis results for: {row[0]}")
                    return result
                else:
                    log_universal('DEBUG', 'Database', f"No analysis results found for: {file_path}")
                    return None
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error retrieving analysis results for {file_path}: {e}")
            return None

    @log_function_call
    def get_all_analysis_results(self) -> List[Dict[str, Any]]:
        """
        Get all analysis results from the database.
        
        Returns:
            List of analysis result dictionaries
        """
        log_universal('DEBUG', 'Database', "Retrieving all analysis results from database")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path, filename, file_size_bytes, file_hash, 
                           analysis_data, metadata, artist, album, title, genre, year, analysis_date, last_checked
                    FROM analysis_results ORDER BY analysis_date DESC
                """)
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        'file_path': row[0],
                        'filename': row[1],
                        'file_size_bytes': row[2],
                        'file_hash': row[3],
                        'analysis_data': json.loads(row[4]),
                        'metadata': json.loads(row[5]) if row[5] else None,
                        'artist': row[6],
                        'album': row[7],
                        'title': row[8],
                        'genre': row[9],
                        'year': row[10],
                        'analysis_date': row[11],
                        'last_checked': row[12]
                    }
                    results.append(result)
                
                log_universal('INFO', 'Database', f"Retrieved {len(results)} analysis results from database")
                return results
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error retrieving all analysis results: {e}")
            return []

    @log_function_call
    def delete_analysis_result(self, file_path: str) -> bool:
        """
        Delete analysis results for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('DEBUG', 'Database', f"Deleting analysis results for: {file_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM analysis_results WHERE file_path = ?", (file_path,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    log_universal('DEBUG', 'Database', f"Successfully deleted analysis results for: {file_path}")
                    return True
                else:
                    log_universal('DEBUG', 'Database', f"No analysis results found for: {file_path}")
                    return False
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error deleting analysis results for {file_path}: {e}")
            return False

    @log_function_call
    def delete_failed_analysis(self, file_path: str) -> bool:
        """
        Delete failed analysis entry for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('DEBUG', 'Database', f"Deleting failed analysis for: {file_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM failed_analysis WHERE file_path = ?", (file_path,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    log_universal('DEBUG', 'Database', f"Successfully deleted failed analysis for: {file_path}")
                    return True
                else:
                    log_universal('DEBUG', 'Database', f"No failed analysis found for: {file_path}")
                    return False
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error deleting failed analysis for {file_path}: {e}")
            return False

    # =============================================================================
    # CACHE OPERATIONS
    # =============================================================================

    @log_function_call
    def save_cache(self, key: str, value: Any, expires_hours: int = None) -> bool:
        """
        Save a value to the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expires_hours: Hours until cache expires (uses config default if None)
            
        Returns:
            True if successful, False otherwise
        """
        # Use configurable default expiration
        if expires_hours is None:
            expires_hours = self.config.get('DB_CACHE_DEFAULT_EXPIRY_HOURS', 24)
        
        log_universal('DEBUG', 'Database', f"Saving cache entry: {key} (expires in {expires_hours}h)")
        
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                value_json = json.dumps(value)
                expires_at = datetime.now() + timedelta(hours=expires_hours)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                """, (key, value_json, expires_at))
                
                conn.commit()
                save_time = time.time() - start_time
                log_universal('DEBUG', 'Database', f"Successfully saved cache entry: {key} in {save_time:.2f}s")
                
                # Log performance
                log_universal('INFO', 'Database', f"Cache save completed in {save_time:.2f}s")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error saving cache entry {key}: {e}")
            return False

    @log_function_call
    def get_cache(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        log_universal('DEBUG', 'Database', f"Retrieving cache entry: {key}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT value, expires_at FROM cache 
                    WHERE key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """, (key,))
                
                row = cursor.fetchone()
                if row:
                    value = json.loads(row[0])
                    log_universal('DEBUG', 'Database', f"Retrieved cache entry: {key}")
                    return value
                else:
                    log_universal('DEBUG', 'Database', f"Cache entry not found or expired: {key}")
                    return None
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error retrieving cache entry {key}: {e}")
            return None

    @log_function_call
    def cleanup_cache(self, max_age_hours: int = None) -> int:
        """
        Clean up expired cache entries.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup (uses config default if None)
            
        Returns:
            Number of entries cleaned up
        """
        # Use configurable cleanup frequency
        if max_age_hours is None:
            max_age_hours = self.config.get('DB_CACHE_CLEANUP_FREQUENCY_HOURS', 24)
        
        log_universal('INFO', 'Database', f"Cleaning up cache entries older than {max_age_hours} hours")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete expired entries
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                cursor.execute("""
                    DELETE FROM cache 
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                """, (cutoff_time,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                log_universal('INFO', 'Database', f"Cleaned up {deleted_count} expired cache entries")
                return deleted_count
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error cleaning up cache: {e}")
            return 0

    # =============================================================================
    # TAGS OPERATIONS
    # =============================================================================

    @log_function_call
    def save_tags(self, file_path: str, tags: Dict[str, Any], 
                 source: str = None, confidence: float = None) -> bool:
        """
        Save tags for a file.
        
        Args:
            file_path: Path to the file
            tags: Tags dictionary
            source: Source of the tags (e.g., 'musicbrainz', 'lastfm')
            confidence: Confidence score for the tags
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('DEBUG', 'Database', f"Saving tags for: {file_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                tags_json = json.dumps(tags)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO tags 
                    (file_path, tags, source, confidence, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (file_path, tags_json, source, confidence))
                
                conn.commit()
                log_universal('DEBUG', 'Database', f"Successfully saved tags for: {file_path}")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error saving tags for {file_path}: {e}")
            return False

    @log_function_call
    def get_tags(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get tags for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tags dictionary or None if not found
        """
        log_universal('DEBUG', 'Database', f"Retrieving tags for: {file_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT tags, source, confidence, updated_at
                    FROM tags WHERE file_path = ?
                """, (file_path,))
                
                row = cursor.fetchone()
                if row:
                    tags_data = {
                        'file_path': file_path,
                        'tags': json.loads(row[0]),
                        'source': row[1],
                        'confidence': row[2],
                        'updated_at': row[3]
                    }
                    log_universal('DEBUG', 'Database', f"Retrieved tags for: {file_path}")
                    return tags_data
                else:
                    log_universal('DEBUG', 'Database', f"No tags found for: {file_path}")
                    return None
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error retrieving tags for {file_path}: {e}")
            return None

    # =============================================================================
    # FAILED ANALYSIS OPERATIONS
    # =============================================================================

    @log_function_call
    def mark_analysis_failed(self, file_path: str, filename: str, 
                           error_message: str) -> bool:
        """
        Mark a file as having failed analysis.
        
        Args:
            file_path: Path to the failed file
            filename: Name of the file
            error_message: Error message describing the failure
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('INFO', 'Database', f"Marking analysis as failed for: {filename}")
        log_universal('DEBUG', 'Database', f"  Error: {error_message}")
        
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO failed_analysis 
                    (file_path, filename, error_message, retry_count, 
                     failed_date, last_retry)
                    VALUES (?, ?, ?, 
                           COALESCE((SELECT retry_count FROM failed_analysis WHERE file_path = ?), 0) + 1,
                           CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (file_path, filename, error_message, file_path))
                
                conn.commit()
                save_time = time.time() - start_time
                log_universal('INFO', 'Database', f"Successfully marked analysis as failed for: {filename} in {save_time:.2f}s")
                
                # Log performance
                log_universal('INFO', 'Database', f"Failed analysis mark completed in {save_time:.2f}s")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error marking analysis as failed for {filename}: {e}")
            return False

    @log_function_call
    def get_failed_analysis_files(self, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Get files that have failed analysis.
        
        Args:
            max_retries: Maximum retry count to include
            
        Returns:
            List of failed analysis file dictionaries
        """
        log_universal('DEBUG', 'Database', f"Retrieving failed analysis files (max retries: {max_retries})")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path, filename, error_message, retry_count, 
                           failed_date, last_retry
                    FROM failed_analysis 
                    WHERE retry_count <= ?
                    ORDER BY last_retry ASC
                """, (max_retries,))
                
                failed_files = []
                for row in cursor.fetchall():
                    failed_file = {
                        'file_path': row[0],
                        'filename': row[1],
                        'error_message': row[2],
                        'retry_count': row[3],
                        'failed_date': row[4],
                        'last_retry': row[5]
                    }
                    failed_files.append(failed_file)
                
                log_universal('INFO', 'Database', f"Retrieved {len(failed_files)} failed analysis files")
                return failed_files
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error retrieving failed analysis files: {e}")
            return []

    # =============================================================================
    # STATISTICS OPERATIONS
    # =============================================================================

    @log_function_call
    def save_statistic(self, category: str, key: str, value: Any) -> bool:
        """
        Save a statistic to the database.
        
        Args:
            category: Statistic category
            key: Statistic key
            value: Statistic value
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('DEBUG', 'Database', f"Saving statistic: {category}.{key}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                value_json = json.dumps(value)
                
                cursor.execute("""
                    INSERT INTO statistics (category, key, value, timestamp)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (category, key, value_json))
                
                conn.commit()
                log_universal('DEBUG', 'Database', f"Successfully saved statistic: {category}.{key}")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error saving statistic {category}.{key}: {e}")
            return False

    @log_function_call
    def get_statistics(self, category: str = None, hours: int = None) -> Dict[str, Any]:
        """
        Get statistics from the database.
        
        Args:
            category: Optional category filter
            hours: Hours of history to include (uses config default if None)
            
        Returns:
            Dictionary of statistics
        """
        # Use configurable statistics collection frequency
        if hours is None:
            hours = self.config.get('DB_STATISTICS_COLLECTION_FREQUENCY_HOURS', 24)
        
        log_universal('DEBUG', 'Database', f"Retrieving statistics (category: {category}, hours: {hours})")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if category:
                    cursor.execute("""
                        SELECT key, value, timestamp FROM statistics 
                        WHERE category = ? AND timestamp > datetime('now', '-{} hours')
                        ORDER BY timestamp DESC
                    """.format(hours), (category,))
                else:
                    cursor.execute("""
                        SELECT category, key, value, timestamp FROM statistics 
                        WHERE timestamp > datetime('now', '-{} hours')
                        ORDER BY timestamp DESC
                    """.format(hours))
                
                stats = {}
                for row in cursor.fetchall():
                    if category:
                        key, value_json, timestamp = row
                        stats[key] = {
                            'value': json.loads(value_json),
                            'timestamp': timestamp
                        }
                    else:
                        cat, key, value_json, timestamp = row
                        if cat not in stats:
                            stats[cat] = {}
                        stats[cat][key] = {
                            'value': json.loads(value_json),
                            'timestamp': timestamp
                        }
                
                log_universal('INFO', 'Database', f"Retrieved statistics: {len(stats)} categories")
                return stats
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error retrieving statistics: {e}")
            return {}

    # =============================================================================
    # UTILITY OPERATIONS
    # =============================================================================

    @log_function_call
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        log_universal('DEBUG', 'Database', "Generating database statistics")
        
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                tables = ['playlists', 'analysis_results', 'cache', 'tags', 'failed_analysis', 'statistics']
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[f'{table}_count'] = count
                    log_universal('DEBUG', 'Database', f"{table}: {count} records")
                
                # Get database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                stats['database_size_bytes'] = db_size
                stats['database_size_mb'] = db_size / (1024 * 1024)
                log_universal('DEBUG', 'Database', f"Database size: {stats['database_size_mb']:.2f} MB")
                
                # Get recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM analysis_results 
                    WHERE analysis_date > datetime('now', '-24 hours')
                """)
                stats['recent_analysis_count'] = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) FROM playlists 
                    WHERE updated_at > datetime('now', '-24 hours')
                """)
                stats['recent_playlist_updates'] = cursor.fetchone()[0]
                
                gen_time = time.time() - start_time
                log_universal('INFO', 'Database', f"Database statistics generated successfully in {gen_time:.2f}s")
                
                # Log performance
                log_universal('INFO', 'Database', f"Database statistics generation completed in {gen_time:.2f}s")
                return stats
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error generating database statistics: {e}")
            return {}

    @log_function_call
    def cleanup_old_data(self, days: int = None) -> Dict[str, int]:
        """
        Clean up old data from the database.
        
        Args:
            days: Number of days of data to keep (uses config defaults if None)
            
        Returns:
            Dictionary with cleanup results
        """
        # Use configurable retention periods
        if days is None:
            days = self.config.get('DB_CLEANUP_RETENTION_DAYS', 30)
        
        failed_retention_days = self.config.get('DB_FAILED_ANALYSIS_RETENTION_DAYS', 7)
        stats_retention_days = self.config.get('DB_STATISTICS_RETENTION_DAYS', 90)
        
        log_universal('INFO', 'Database', f"Cleaning up data older than {days} days")
        log_universal('DEBUG', 'Database', f"  Failed analysis retention: {failed_retention_days} days")
        log_universal('DEBUG', 'Database', f"  Statistics retention: {stats_retention_days} days")
        
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                results = {}
                
                # Clean up old cache entries
                cutoff_time = datetime.now() - timedelta(days=days)
                cursor.execute("""
                    DELETE FROM cache 
                    WHERE created_at < ?
                """, (cutoff_time,))
                results['cache_cleaned'] = cursor.rowcount
                log_universal('DEBUG', 'Database', f"Cleaned {results['cache_cleaned']} cache entries")
                
                # Clean up old statistics
                cursor.execute("""
                    DELETE FROM statistics 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(stats_retention_days))
                results['statistics_cleaned'] = cursor.rowcount
                log_universal('DEBUG', 'Database', f"Cleaned {results['statistics_cleaned']} statistics entries")
                
                # Clean up old failed analysis (keep for shorter time)
                cursor.execute("""
                    DELETE FROM failed_analysis 
                    WHERE failed_date < datetime('now', '-{} days')
                """.format(failed_retention_days))
                results['failed_analysis_cleaned'] = cursor.rowcount
                log_universal('DEBUG', 'Database', f"Cleaned {results['failed_analysis_cleaned']} failed analysis entries")
                
                conn.commit()
                
                cleanup_time = time.time() - start_time
                total_cleaned = sum(results.values())
                log_universal('INFO', 'Database', f"Cleanup completed in {cleanup_time:.2f}s: {results}")
                log_universal('INFO', 'Database', f"Total cleaned: {total_cleaned} entries")
                
                # Log performance
                log_universal('INFO', 'Database', f"Database cleanup completed in {cleanup_time:.2f}s")
                return results
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error during cleanup: {e}")
            return {}

    @log_function_call
    def export_database(self, export_path: str) -> bool:
        """
        Export database to a file.
        
        Args:
            export_path: Path to export the database to
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('INFO', 'Database', f"Exporting database to: {export_path}")
        
        start_time = time.time()
        
        try:
            # Get source database size
            source_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            log_universal('DEBUG', 'Database', f"Source database size: {source_size / (1024 * 1024):.2f} MB")
            
            with sqlite3.connect(self.db_path) as source_conn:
                with sqlite3.connect(export_path) as dest_conn:
                    source_conn.backup(dest_conn)
            
            # Get exported database size
            export_size = os.path.getsize(export_path) if os.path.exists(export_path) else 0
            export_time = time.time() - start_time
            
            log_universal('INFO', 'Database', f"Successfully exported database to: {export_path}")
            log_universal('INFO', 'Database', f"Export size: {export_size / (1024 * 1024):.2f} MB in {export_time:.2f}s")
            
            # Log performance
            log_universal('INFO', 'Database', f"Database export completed in {export_time:.2f}s")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error exporting database: {e}")
            return False


    def get_database_config(self) -> Dict[str, Any]:
        """
        Get current database configuration.
        
        Returns:
            Dictionary with current database settings
        """
        return self.config.copy()
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert objects to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if obj is None:
            return None
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            # Handle numpy arrays and other non-serializable objects
            try:
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return str(obj)
            except ImportError:
                return str(obj)

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update database configuration.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config.update(new_config)
            log_universal('INFO', 'Database', f"Updated database configuration: {new_config}")
            return True
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error updating database configuration: {e}")
            return False


# Global database manager instance - created lazily to avoid circular imports
_db_manager_instance = None

def get_db_manager() -> 'DatabaseManager':
    """Get the global database manager instance, creating it if necessary."""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
    return _db_manager_instance 
