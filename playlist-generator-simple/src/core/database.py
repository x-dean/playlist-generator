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
        """Schema migration removed - using fresh database approach."""
        pass

    def _init_database(self):
        """Initialize the database with comprehensive normalized schema."""
        log_universal('INFO', 'Database', "Initializing comprehensive database schema...")
        
        tables_created = 0
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if database already has tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'")
                has_tables = cursor.fetchone() is not None
                
                if has_tables:
                    log_universal('INFO', 'Database', "Database already has tables, skipping schema creation")
                    tables_created = 0
                else:
                    # Read and apply comprehensive schema
                    schema_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'database_schema.sql')
                    
                    # Also try current directory and parent directory
                    if not os.path.exists(schema_file):
                        schema_file = 'database_schema.sql'
                    if not os.path.exists(schema_file):
                        schema_file = os.path.join('..', 'database_schema.sql')
                    
                    if os.path.exists(schema_file):
                        log_universal('INFO', 'Database', f"Applying comprehensive schema from: {schema_file}")
                        with open(schema_file, 'r', encoding='utf-8') as f:
                            schema_sql = f.read()
                        
                        # Execute comprehensive schema
                        cursor.executescript(schema_sql)
                        tables_created = 19  # Our schema has 19 tables
                        log_universal('INFO', 'Database', "Comprehensive schema applied successfully")
                    else:
                        log_universal('WARNING', 'Database', f"Schema file not found: {schema_file}")
                        log_universal('INFO', 'Database', "Falling back to simple schema")
                        
                        # Fallback to simple schema
                        self._create_simple_schema(cursor)
                        tables_created = 8  # Simple schema has 8 tables
                
                conn.commit()
                init_time = time.time() - start_time
                log_universal('INFO', 'Database', f"Database initialization completed successfully")
                log_universal('INFO', 'Database', f"Created {tables_created} tables and indexes in {init_time:.2f}s")
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Database initialization failed: {e}")
            raise

    def _create_simple_schema(self, cursor):
        """Create simple schema as fallback."""
        log_universal('INFO', 'Database', "Creating simple schema as fallback...")
        
        # Create playlists table
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
        
        # Create analysis_results table
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
                long_audio_category TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        # Create tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                file_path TEXT PRIMARY KEY,
                tags TEXT NOT NULL,
                source TEXT,
                confidence REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create failed_analysis table
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
        
        # Create statistics table
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
        
        # Create indexes for simple schema
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_artist ON analysis_results(artist)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_album ON analysis_results(album)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_genre ON analysis_results(genre)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_year ON analysis_results(year)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_long_audio_category ON analysis_results(long_audio_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_date ON analysis_results(analysis_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_filename ON analysis_results(filename)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_source ON tags(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_updated ON tags(updated_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_failed_retry_count ON failed_analysis(retry_count)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_failed_date ON failed_analysis(failed_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_category ON statistics(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON statistics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_category_key ON statistics(category, key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_playlists_created ON playlists(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_playlists_updated ON playlists(updated_at)")
        
        log_universal('INFO', 'Database', "Simple schema created as fallback")

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
        Get features for a specific track from the normalized schema.
        
        Args:
            file_path: Path to the track file
            
        Returns:
            Dictionary with track features or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT t.bpm, t.key, t.mode, t.loudness, t.danceability, t.energy,
                           sf.spectral_centroid, sf.spectral_rolloff,
                           lf.integrated_loudness, lf.loudness_range,
                           af.onset_rate, af.zero_crossing_rate
                    FROM tracks t
                    LEFT JOIN spectral_features sf ON t.id = sf.track_id
                    LEFT JOIN loudness_features lf ON t.id = lf.track_id
                    LEFT JOIN advanced_features af ON t.id = af.track_id
                    WHERE t.file_path = ?
                """, (file_path,))
                
                row = cursor.fetchone()
                if row:
                    features = {
                        'bpm': row[0],
                        'key': row[1],
                        'mode': row[2],
                        'loudness': row[3],
                        'danceability': row[4],
                        'energy': row[5],
                        'spectral_centroid': row[6],
                        'spectral_rolloff': row[7],
                        'integrated_loudness': row[8],
                        'loudness_range': row[9],
                        'onset_rate': row[10],
                        'zero_crossing_rate': row[11]
                    }
                    return features
                else:
                    return None
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting track features for {file_path}: {e}")
            return None
    








    # =============================================================================
    # ANALYSIS RESULTS OPERATIONS
    # =============================================================================

    @log_function_call
    def save_analysis_result(self, file_path: str, filename: str, file_size_bytes: int,
                           file_hash: str, analysis_data: Dict[str, Any],
                           metadata: Dict[str, Any] = None) -> bool:
        """
        Save analysis results to database (LEGACY - uses old analysis_results table).
        
        Args:
            file_path: Path to the analyzed file
            filename: Name of the file
            file_size_bytes: File size in bytes
            file_hash: File hash for change detection
            analysis_data: Analysis results dictionary
            metadata: Metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('DEBUG', 'Database', f"Saving analysis results (LEGACY) for: {filename}")
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert to JSON-serializable format
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
                long_audio_category = metadata.get('long_audio_category') if metadata else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_results 
                    (file_path, filename, file_size_bytes, file_hash, 
                     analysis_data, metadata, artist, album, title, genre, year, long_audio_category, analysis_date, last_checked)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (file_path, filename, file_size_bytes, file_hash, 
                     analysis_json, metadata_json, artist, album, title, genre, year, long_audio_category))
                
                conn.commit()
                save_time = time.time() - start_time
                log_universal('DEBUG', 'Database', f"Successfully saved analysis results (LEGACY) for: {filename} in {save_time:.2f}s")
                
                # Log performance
                log_universal('INFO', 'Database', f"Analysis result save (LEGACY) completed in {save_time:.2f}s")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error saving analysis results (LEGACY) for {filename}: {e}")
            return False

    @log_function_call
    def save_track_to_normalized_schema(self, file_path: str, filename: str, file_size_bytes: int,
                                      file_hash: str, analysis_data: Dict[str, Any],
                                      metadata: Dict[str, Any] = None) -> bool:
        """
        Save analysis results to normalized database schema (NEW - uses tracks table).
        
        Args:
            file_path: Path to the analyzed file
            filename: Name of the file
            file_size_bytes: File size in bytes
            file_hash: File hash for change detection
            analysis_data: Analysis results dictionary
            metadata: Metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        log_universal('DEBUG', 'Database', f"Saving track to normalized schema for: {filename}")
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Extract basic file info
                analysis_date = datetime.now()
                analysis_version = "1.0.0"
                analysis_type = analysis_data.get('analysis_type', 'full')
                long_audio_category = metadata.get('long_audio_category') if metadata else None
                
                # Extract metadata fields
                title = metadata.get('title', 'Unknown Title')
                artist = metadata.get('artist', 'Unknown Artist')
                album = metadata.get('album')
                track_number = metadata.get('track_number')
                genre = metadata.get('genre')
                year = metadata.get('year')
                duration = metadata.get('duration')
                bitrate = metadata.get('bitrate')
                sample_rate = metadata.get('sample_rate')
                channels = metadata.get('channels')
                
                # Extended metadata from file tags
                composer = metadata.get('composer')
                lyricist = metadata.get('lyricist')
                band = metadata.get('band')
                conductor = metadata.get('conductor')
                remixer = metadata.get('remixer')
                subtitle = metadata.get('subtitle')
                grouping = metadata.get('grouping')
                publisher = metadata.get('publisher')
                copyright = metadata.get('copyright')
                encoded_by = metadata.get('encoded_by')
                language = metadata.get('language')
                mood = metadata.get('mood')
                style = metadata.get('style')
                quality = metadata.get('quality')
                original_artist = metadata.get('original_artist')
                original_album = metadata.get('original_album')
                original_year = metadata.get('original_year')
                original_filename = metadata.get('original_filename')
                content_group = metadata.get('content_group')
                encoder = metadata.get('encoder')
                file_type = metadata.get('file_type')
                playlist_delay = metadata.get('playlist_delay')
                recording_time = metadata.get('recording_time')
                tempo = metadata.get('tempo')
                length = metadata.get('length')
                
                # ReplayGain metadata
                replaygain_track_gain = metadata.get('replaygain_track_gain')
                replaygain_album_gain = metadata.get('replaygain_album_gain')
                replaygain_track_peak = metadata.get('replaygain_track_peak')
                replaygain_album_peak = metadata.get('replaygain_album_peak')
                
                # Audio features from analysis
                bpm = analysis_data.get('bpm')
                key = analysis_data.get('key')
                mode = analysis_data.get('mode')
                scale = analysis_data.get('scale')
                key_strength = analysis_data.get('key_strength')
                loudness = analysis_data.get('loudness')
                danceability = analysis_data.get('danceability')
                energy = analysis_data.get('energy')
                rhythm_confidence = analysis_data.get('rhythm_confidence')
                key_confidence = analysis_data.get('key_confidence')
                
                # Insert into tracks table (excluding auto-increment id)
                cursor.execute("""
                    INSERT OR REPLACE INTO tracks (
                        file_path, file_hash, filename, file_size_bytes, analysis_date, analysis_version, 
                        analysis_type, long_audio_category, title, artist, album, track_number, genre, year,
                        duration, bitrate, sample_rate, channels, composer, lyricist, band, conductor, remixer,
                        subtitle, grouping, publisher, copyright, encoded_by, language, mood, style, quality,
                        original_artist, original_album, original_year, original_filename, content_group, encoder,
                        file_type, playlist_delay, recording_time, tempo, length, replaygain_track_gain,
                        replaygain_album_gain, replaygain_track_peak, replaygain_album_peak, bpm, key, mode,
                        scale, key_strength, loudness, danceability, energy, rhythm_confidence, key_confidence,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (file_path, file_hash, filename, file_size_bytes, analysis_date, analysis_version,
                     analysis_type, long_audio_category, title, artist, album, track_number, genre, year,
                     duration, bitrate, sample_rate, channels, composer, lyricist, band, conductor, remixer,
                     subtitle, grouping, publisher, copyright, encoded_by, language, mood, style, quality,
                     original_artist, original_album, original_year, original_filename, content_group, encoder,
                     file_type, playlist_delay, recording_time, tempo, length, replaygain_track_gain,
                     replaygain_album_gain, replaygain_track_peak, replaygain_album_peak, bpm, key, mode,
                     scale, key_strength, loudness, danceability, energy, rhythm_confidence, key_confidence,
                     analysis_date, analysis_date))  # created_at and updated_at use analysis_date
                
                # Get the track ID for related tables
                track_id = cursor.lastrowid
                
                # Save external metadata if available
                if metadata and (metadata.get('musicbrainz_id') or metadata.get('lastfm_url')):
                    self._save_external_metadata(cursor, track_id, metadata)
                
                # Save tags if available
                if metadata and metadata.get('tags'):
                    self._save_tags(cursor, track_id, metadata.get('tags'), metadata.get('tag_source', 'file'))
                
                # Save audio features to specialized tables
                if analysis_data:
                    self._save_audio_features(cursor, track_id, analysis_data)
                
                conn.commit()
                
                # Invalidate related caches
                self._invalidate_related_caches('insert', {
                    'artist': artist,
                    'album': album,
                    'genre': genre
                })
                
                save_time = time.time() - start_time
                log_universal('DEBUG', 'Database', f"Successfully saved track to normalized schema for: {filename} in {save_time:.2f}s")
                
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error saving track to normalized schema for {filename}: {e}")
            return False

    def _save_external_metadata(self, cursor, track_id: int, metadata: Dict[str, Any]):
        """Save external API metadata to external_metadata table."""
        try:
            # MusicBrainz data
            if metadata.get('musicbrainz_id'):
                cursor.execute("""
                    INSERT OR REPLACE INTO external_metadata 
                    (track_id, source, musicbrainz_id, musicbrainz_artist_id, musicbrainz_album_id, 
                     release_date, disc_number, duration_ms, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (track_id, 'musicbrainz', metadata.get('musicbrainz_id'), 
                     metadata.get('musicbrainz_artist_id'), metadata.get('musicbrainz_album_id'),
                     metadata.get('release_date'), metadata.get('disc_number'), metadata.get('duration_ms'), 1.0))
            
            # Last.fm data
            if metadata.get('lastfm_url'):
                cursor.execute("""
                    INSERT OR REPLACE INTO external_metadata 
                    (track_id, source, lastfm_url, play_count, listeners, rating, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (track_id, 'lastfm', metadata.get('lastfm_url'), metadata.get('play_count'),
                     metadata.get('listeners'), metadata.get('rating'), 1.0))
                     
        except Exception as e:
            log_universal('WARNING', 'Database', f"Error saving external metadata: {e}")

    def _save_tags(self, cursor, track_id: int, tags: Dict[str, Any], source: str):
        """Save tags to normalized tags table."""
        try:
            for tag_name, tag_value in tags.items():
                if tag_value is not None:
                    cursor.execute("""
                        INSERT OR REPLACE INTO tags 
                        (track_id, source, tag_name, tag_value, confidence)
                        VALUES (?, ?, ?, ?, ?)
                    """, (track_id, source, tag_name, str(tag_value), 1.0))
        except Exception as e:
            log_universal('WARNING', 'Database', f"Error saving tags: {e}")

    def _save_audio_features(self, cursor, track_id: int, analysis_data: Dict[str, Any]):
        """Save audio features to specialized tables."""
        try:
            # Spectral features
            if 'spectral_features' in analysis_data:
                sf = analysis_data['spectral_features']
                cursor.execute("""
                    INSERT OR REPLACE INTO spectral_features 
                    (track_id, spectral_centroid, spectral_rolloff, spectral_flatness, 
                     spectral_bandwidth, spectral_contrast_mean, spectral_contrast_std)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (track_id, sf.get('spectral_centroid'), sf.get('spectral_rolloff'), 
                     sf.get('spectral_flatness'), sf.get('spectral_bandwidth'),
                     sf.get('spectral_contrast_mean'), sf.get('spectral_contrast_std')))
            
            # Loudness features
            if 'loudness_features' in analysis_data:
                lf = analysis_data['loudness_features']
                cursor.execute("""
                    INSERT OR REPLACE INTO loudness_features 
                    (track_id, integrated_loudness, loudness_range, momentary_loudness_mean,
                     momentary_loudness_std, short_term_loudness_mean, short_term_loudness_std,
                     dynamic_complexity, dynamic_range)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (track_id, lf.get('integrated_loudness'), lf.get('loudness_range'),
                     lf.get('momentary_loudness_mean'), lf.get('momentary_loudness_std'),
                     lf.get('short_term_loudness_mean'), lf.get('short_term_loudness_std'),
                     lf.get('dynamic_complexity'), lf.get('dynamic_range')))
            
            # Rhythm features
            if 'rhythm_features' in analysis_data:
                rf = analysis_data['rhythm_features']
                cursor.execute("""
                    INSERT OR REPLACE INTO rhythm_features 
                    (track_id, bpm_estimates, bpm_intervals, external_bpm)
                    VALUES (?, ?, ?, ?)
                """, (track_id, json.dumps(rf.get('bpm_estimates', [])), 
                     json.dumps(rf.get('bpm_intervals', [])), rf.get('external_bpm')))
            
            # Advanced features
            if 'advanced_features' in analysis_data:
                af = analysis_data['advanced_features']
                cursor.execute("""
                    INSERT OR REPLACE INTO advanced_features 
                    (track_id, onset_rate, zero_crossing_rate, harmonic_complexity)
                    VALUES (?, ?, ?, ?)
                """, (track_id, af.get('onset_rate'), af.get('zero_crossing_rate'), 
                     af.get('harmonic_complexity')))
            
            # MFCC features
            if 'mfcc_features' in analysis_data:
                mf = analysis_data['mfcc_features']
                cursor.execute("""
                    INSERT OR REPLACE INTO mfcc_features 
                    (track_id, mfcc_coefficients, mfcc_bands, mfcc_std)
                    VALUES (?, ?, ?, ?)
                """, (track_id, json.dumps(mf.get('mfcc_coefficients', [])),
                     json.dumps(mf.get('mfcc_bands', [])), json.dumps(mf.get('mfcc_std', []))))
            
            # MusiCNN features
            if 'musicnn_features' in analysis_data:
                mn = analysis_data['musicnn_features']
                cursor.execute("""
                    INSERT OR REPLACE INTO musicnn_features 
                    (track_id, embedding, tags)
                    VALUES (?, ?, ?)
                """, (track_id, json.dumps(mn.get('embedding', [])),
                     json.dumps(mn.get('tags', {}))))
            
            # Chroma features
            if 'chroma_features' in analysis_data:
                cf = analysis_data['chroma_features']
                cursor.execute("""
                    INSERT OR REPLACE INTO chroma_features 
                    (track_id, chroma_mean, chroma_std)
                    VALUES (?, ?, ?)
                """, (track_id, json.dumps(cf.get('chroma_mean', [])),
                     json.dumps(cf.get('chroma_std', []))))
                     
        except Exception as e:
            log_universal('WARNING', 'Database', f"Error saving audio features: {e}")

    @log_function_call
    def get_analysis_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis results for a file from the normalized schema.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Analysis results dictionary or None if not found
        """
        log_universal('DEBUG', 'Database', f"Retrieving analysis results for: {file_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Query the tracks table with all related data
                cursor.execute("""
                    SELECT t.*, 
                           em.musicbrainz_id, em.musicbrainz_artist_id, em.musicbrainz_album_id,
                           em.release_date, em.disc_number, em.duration_ms,
                           em.lastfm_url, em.play_count, em.listeners, em.rating,
                           sf.spectral_centroid, sf.spectral_rolloff, sf.spectral_flatness,
                           sf.spectral_bandwidth, sf.spectral_contrast_mean, sf.spectral_contrast_std,
                           lf.integrated_loudness, lf.loudness_range, lf.momentary_loudness_mean,
                           lf.momentary_loudness_std, lf.short_term_loudness_mean, lf.short_term_loudness_std,
                           lf.dynamic_complexity, lf.dynamic_range,
                           rf.bpm_estimates, rf.bpm_intervals, rf.external_bpm,
                           af.onset_rate, af.zero_crossing_rate, af.harmonic_complexity,
                           mf.mfcc_coefficients, mf.mfcc_bands, mf.mfcc_std,
                           mn.embedding, mn.tags,
                           cf.chroma_mean, cf.chroma_std
                    FROM tracks t
                    LEFT JOIN external_metadata em ON t.id = em.track_id
                    LEFT JOIN spectral_features sf ON t.id = sf.track_id
                    LEFT JOIN loudness_features lf ON t.id = lf.track_id
                    LEFT JOIN rhythm_features rf ON t.id = rf.track_id
                    LEFT JOIN advanced_features af ON t.id = af.track_id
                    LEFT JOIN mfcc_features mf ON t.id = mf.track_id
                    LEFT JOIN musicnn_features mn ON t.id = mn.track_id
                    LEFT JOIN chroma_features cf ON t.id = cf.track_id
                    WHERE t.file_path = ?
                """, (file_path,))
                
                row = cursor.fetchone()
                if row:
                    # Get column names for proper mapping
                    columns = [description[0] for description in cursor.description]
                    result = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    if result.get('bpm_estimates'):
                        result['bpm_estimates'] = json.loads(result['bpm_estimates'])
                    if result.get('bpm_intervals'):
                        result['bpm_intervals'] = json.loads(result['bpm_intervals'])
                    if result.get('mfcc_coefficients'):
                        result['mfcc_coefficients'] = json.loads(result['mfcc_coefficients'])
                    if result.get('mfcc_bands'):
                        result['mfcc_bands'] = json.loads(result['mfcc_bands'])
                    if result.get('mfcc_std'):
                        result['mfcc_std'] = json.loads(result['mfcc_std'])
                    if result.get('embedding'):
                        result['embedding'] = json.loads(result['embedding'])
                    if result.get('tags'):
                        result['tags'] = json.loads(result['tags'])
                    if result.get('chroma_mean'):
                        result['chroma_mean'] = json.loads(result['chroma_mean'])
                    if result.get('chroma_std'):
                        result['chroma_std'] = json.loads(result['chroma_std'])
                    
                    # Get tags
                    cursor.execute("""
                        SELECT tag_name, tag_value, source, confidence
                        FROM tags WHERE track_id = ?
                    """, (result['id'],))
                    
                    tags = {}
                    for tag_row in cursor.fetchall():
                        tags[tag_row[0]] = {
                            'value': tag_row[1],
                            'source': tag_row[2],
                            'confidence': tag_row[3]
                        }
                    result['tags'] = tags
                    
                    log_universal('DEBUG', 'Database', f"Retrieved analysis results for: {result['filename']}")
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
        Get all analysis results from the normalized schema.
        
        Returns:
            List of analysis result dictionaries
        """
        log_universal('DEBUG', 'Database', "Retrieving all analysis results from database")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Query all tracks with basic info
                cursor.execute("""
                    SELECT id, file_path, filename, file_size_bytes, file_hash,
                           title, artist, album, genre, year, bpm, key, mode,
                           loudness, danceability, energy, analysis_date
                    FROM tracks
                    ORDER BY analysis_date DESC
                """)
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'file_size_bytes': row[3],
                        'file_hash': row[4],
                        'title': row[5],
                        'artist': row[6],
                        'album': row[7],
                        'genre': row[8],
                        'year': row[9],
                        'bpm': row[10],
                        'key': row[11],
                        'mode': row[12],
                        'loudness': row[13],
                        'danceability': row[14],
                        'energy': row[15],
                        'analysis_date': row[16]
                    }
                    results.append(result)
                
                log_universal('INFO', 'Database', f"Retrieved {len(results)} analysis results from database")
                return results
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error retrieving all analysis results: {e}")
            return []

    @log_function_call
    def get_analyzed_tracks(self) -> List[Dict[str, Any]]:
        """
        Get all analyzed tracks from the normalized schema.
        
        Returns:
            List of track dictionaries with analysis data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all tracks with their features from the normalized schema
                cursor.execute("""
                    SELECT t.id, t.file_path, t.filename, t.file_size_bytes, t.file_hash,
                           t.title, t.artist, t.album, t.genre, t.year, t.bpm, t.key, t.mode,
                           t.loudness, t.danceability, t.energy, t.analysis_date,
                           sf.spectral_centroid, sf.spectral_rolloff,
                           lf.integrated_loudness, lf.loudness_range,
                           af.onset_rate, af.zero_crossing_rate
                    FROM tracks t
                    LEFT JOIN spectral_features sf ON t.id = sf.track_id
                    LEFT JOIN loudness_features lf ON t.id = lf.track_id
                    LEFT JOIN advanced_features af ON t.id = af.track_id
                    ORDER BY t.analysis_date DESC
                """)
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'file_size_bytes': row[3],
                        'file_hash': row[4],
                        'title': row[5],
                        'artist': row[6],
                        'album': row[7],
                        'genre': row[8],
                        'year': row[9],
                        'bpm': row[10],
                        'key': row[11],
                        'mode': row[12],
                        'loudness': row[13],
                        'danceability': row[14],
                        'energy': row[15],
                        'analysis_date': row[16],
                        'spectral_centroid': row[17],
                        'spectral_rolloff': row[18],
                        'integrated_loudness': row[19],
                        'loudness_range': row[20],
                        'onset_rate': row[21],
                        'zero_crossing_rate': row[22]
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} analyzed tracks")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting analyzed tracks: {e}")
            return []

    @log_function_call
    def get_tracks_by_artist(self, artist: str) -> List[Dict[str, Any]]:
        """
        Get all tracks by a specific artist from the normalized schema.
        
        Args:
            artist: Artist name to search for
            
        Returns:
            List of track dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, file_path, filename, title, album, genre, year, bpm, key, mode,
                           loudness, danceability, energy, analysis_date
                    FROM tracks 
                    WHERE artist LIKE ? 
                    ORDER BY title
                """, (f'%{artist}%',))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'title': row[3],
                        'album': row[4],
                        'genre': row[5],
                        'year': row[6],
                        'bpm': row[7],
                        'key': row[8],
                        'mode': row[9],
                        'loudness': row[10],
                        'danceability': row[11],
                        'energy': row[12],
                        'analysis_date': row[13]
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks for artist: {artist}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by artist {artist}: {e}")
            return []

    @log_function_call
    def get_tracks_by_album(self, album: str) -> List[Dict[str, Any]]:
        """
        Get all tracks from a specific album from the normalized schema.
        
        Args:
            album: Album name to search for
            
        Returns:
            List of track dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, file_path, filename, title, artist, genre, year, bpm, key, mode,
                           loudness, danceability, energy, analysis_date
                    FROM tracks 
                    WHERE album LIKE ? 
                    ORDER BY track_number, title
                """, (f'%{album}%',))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'title': row[3],
                        'artist': row[4],
                        'genre': row[5],
                        'year': row[6],
                        'bpm': row[7],
                        'key': row[8],
                        'mode': row[9],
                        'loudness': row[10],
                        'danceability': row[11],
                        'energy': row[12],
                        'analysis_date': row[13]
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks for album: {album}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by album {album}: {e}")
            return []

    @log_function_call
    def get_tracks_by_genre(self, genre: str) -> List[Dict[str, Any]]:
        """
        Get all tracks of a specific genre from the normalized schema.
        
        Args:
            genre: Genre to search for
            
        Returns:
            List of track dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, file_path, filename, title, artist, album, year, bpm, key, mode,
                           loudness, danceability, energy, analysis_date
                    FROM tracks 
                    WHERE genre LIKE ? 
                    ORDER BY artist, title
                """, (f'%{genre}%',))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'title': row[3],
                        'artist': row[4],
                        'album': row[5],
                        'year': row[6],
                        'bpm': row[7],
                        'key': row[8],
                        'mode': row[9],
                        'loudness': row[10],
                        'danceability': row[11],
                        'energy': row[12],
                        'analysis_date': row[13]
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks for genre: {genre}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by genre {genre}: {e}")
            return []

    @log_function_call
    def get_all_artists(self) -> List[str]:
        """
        Get all unique artists from the normalized schema.
        
        Returns:
            List of artist names
        """
        # Check cache first
        cache_key = "query:all_artists"
        cached_result = self.get_cache(cache_key)
        if cached_result:
            log_universal('DEBUG', 'Database', 'Using cached all artists result')
            return cached_result
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT artist 
                    FROM tracks 
                    WHERE artist IS NOT NULL AND artist != ''
                    ORDER BY artist
                """)
                
                artists = [row[0] for row in cursor.fetchall()]
                log_universal('DEBUG', 'Database', f"Retrieved {len(artists)} unique artists")
                
                # Cache result for 1 hour
                self.save_cache(cache_key, artists, expires_hours=1)
                
                return artists
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting all artists: {e}")
            return []

    @log_function_call
    def get_all_albums(self) -> List[str]:
        """
        Get all unique albums from the normalized schema.
        
        Returns:
            List of album names
        """
        # Check cache first
        cache_key = "query:all_albums"
        cached_result = self.get_cache(cache_key)
        if cached_result:
            log_universal('DEBUG', 'Database', 'Using cached all albums result')
            return cached_result
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT album 
                    FROM tracks 
                    WHERE album IS NOT NULL AND album != ''
                    ORDER BY album
                """)
                
                albums = [row[0] for row in cursor.fetchall()]
                log_universal('DEBUG', 'Database', f"Retrieved {len(albums)} unique albums")
                
                # Cache result for 1 hour
                self.save_cache(cache_key, albums, expires_hours=1)
                
                return albums
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting all albums: {e}")
            return []

    @log_function_call
    def get_all_genres(self) -> List[str]:
        """
        Get all unique genres from the normalized schema.
        
        Returns:
            List of genre names
        """
        # Check cache first
        cache_key = "query:all_genres"
        cached_result = self.get_cache(cache_key)
        if cached_result:
            log_universal('DEBUG', 'Database', 'Using cached all genres result')
            return cached_result
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT genre 
                    FROM tracks 
                    WHERE genre IS NOT NULL AND genre != ''
                    ORDER BY genre
                """)
                
                genres = [row[0] for row in cursor.fetchall()]
                log_universal('DEBUG', 'Database', f"Retrieved {len(genres)} unique genres")
                
                # Cache result for 1 hour
                self.save_cache(cache_key, genres, expires_hours=1)
                
                return genres
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting all genres: {e}")
            return []

    @log_function_call
    def get_tracks_by_year(self, year: int) -> List[Dict[str, Any]]:
        """
        Get all tracks from a specific year from the normalized schema.
        
        Args:
            year: Year to search for
            
        Returns:
            List of track dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, file_path, filename, title, artist, album, genre, bpm, key, mode,
                           loudness, danceability, energy, analysis_date
                    FROM tracks 
                    WHERE year = ? 
                    ORDER BY artist, title
                """, (year,))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'title': row[3],
                        'artist': row[4],
                        'album': row[5],
                        'genre': row[6],
                        'bpm': row[7],
                        'key': row[8],
                        'mode': row[9],
                        'loudness': row[10],
                        'danceability': row[11],
                        'energy': row[12],
                        'analysis_date': row[13]
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks for year: {year}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by year {year}: {e}")
            return []

    @log_function_call
    def get_tracks_by_long_audio_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all tracks of a specific long audio category from the normalized schema.
        
        Args:
            category: Category to search for
            
        Returns:
            List of track dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, file_path, filename, title, artist, album, genre, year, bpm, key, mode,
                           loudness, danceability, energy, analysis_date
                    FROM tracks 
                    WHERE long_audio_category = ? 
                    ORDER BY artist, title
                """, (category,))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'title': row[3],
                        'artist': row[4],
                        'album': row[5],
                        'genre': row[6],
                        'year': row[7],
                        'bpm': row[8],
                        'key': row[9],
                        'mode': row[10],
                        'loudness': row[11],
                        'danceability': row[12],
                        'energy': row[13],
                        'analysis_date': row[14]
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks for category: {category}")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by category {category}: {e}")
            return []

    @log_function_call
    def get_all_long_audio_categories(self) -> List[str]:
        """
        Get all unique long audio categories from the normalized schema.
        
        Returns:
            List of category names
        """
        # Check cache first
        cache_key = "query:all_categories"
        cached_result = self.get_cache(cache_key)
        if cached_result:
            log_universal('DEBUG', 'Database', 'Using cached all categories result')
            return cached_result
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT long_audio_category 
                    FROM tracks 
                    WHERE long_audio_category IS NOT NULL AND long_audio_category != ''
                    ORDER BY long_audio_category
                """)
                
                categories = [row[0] for row in cursor.fetchall()]
                log_universal('DEBUG', 'Database', f"Retrieved {len(categories)} unique categories")
                
                # Cache result for 1 hour
                self.save_cache(cache_key, categories, expires_hours=1)
                
                return categories
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting all categories: {e}")
            return []

    @log_function_call
    def get_tracks_by_feature_range(self, feature: str, min_value: float, max_value: float) -> List[Dict[str, Any]]:
        """
        Get tracks within a specific feature range from the normalized schema.
        
        Args:
            feature: Feature name to filter by
            min_value: Minimum value
            max_value: Maximum value
            
        Returns:
            List of track dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Map feature names to column names
                feature_columns = {
                    'bpm': 'bpm',
                    'loudness': 'loudness',
                    'danceability': 'danceability',
                    'energy': 'energy',
                    'key_strength': 'key_strength',
                    'rhythm_confidence': 'rhythm_confidence',
                    'key_confidence': 'key_confidence'
                }
                
                if feature not in feature_columns:
                    log_universal('ERROR', 'Database', f"Unknown feature: {feature}")
                    return []
                
                column = feature_columns[feature]
                
                cursor.execute(f"""
                    SELECT id, file_path, filename, title, artist, album, genre, year, bpm, key, mode,
                           loudness, danceability, energy, analysis_date
                    FROM tracks 
                    WHERE {column} BETWEEN ? AND ?
                    ORDER BY {column}
                """, (min_value, max_value))
                
                tracks = []
                for row in cursor.fetchall():
                    track = {
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'title': row[3],
                        'artist': row[4],
                        'album': row[5],
                        'genre': row[6],
                        'year': row[7],
                        'bpm': row[8],
                        'key': row[9],
                        'mode': row[10],
                        'loudness': row[11],
                        'danceability': row[12],
                        'energy': row[13],
                        'analysis_date': row[14]
                    }
                    tracks.append(track)
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(tracks)} tracks for {feature} range [{min_value}, {max_value}]")
                return tracks
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting tracks by feature range {feature}: {e}")
            return []

    @log_function_call
    def delete_analysis_result(self, file_path: str) -> bool:
        """
        Delete analysis results for a file from the normalized schema.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # First get the track ID
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                track_row = cursor.fetchone()
                
                if track_row:
                    track_id = track_row[0]
                    
                    # Delete from related tables first (due to foreign key constraints)
                    cursor.execute("DELETE FROM external_metadata WHERE track_id = ?", (track_id,))
                    cursor.execute("DELETE FROM tags WHERE track_id = ?", (track_id,))
                    cursor.execute("DELETE FROM spectral_features WHERE track_id = ?", (track_id,))
                    cursor.execute("DELETE FROM loudness_features WHERE track_id = ?", (track_id,))
                    cursor.execute("DELETE FROM rhythm_features WHERE track_id = ?", (track_id,))
                    cursor.execute("DELETE FROM advanced_features WHERE track_id = ?", (track_id,))
                    cursor.execute("DELETE FROM mfcc_features WHERE track_id = ?", (track_id,))
                    cursor.execute("DELETE FROM musicnn_features WHERE track_id = ?", (track_id,))
                    cursor.execute("DELETE FROM chroma_features WHERE track_id = ?", (track_id,))
                    
                    # Finally delete from tracks table
                    cursor.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
                    
                    conn.commit()
                    
                    # Invalidate related caches
                    self._invalidate_related_caches('delete')
                    
                    log_universal('DEBUG', 'Database', f"Successfully deleted analysis result for: {file_path}")
                    return True
                else:
                    log_universal('DEBUG', 'Database', f"No analysis result found to delete for: {file_path}")
                    return False
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error deleting analysis result for {file_path}: {e}")
            return False

    @log_function_call
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics from the normalized schema.
        
        Returns:
            Dictionary with database statistics
        """
        # Check cache first
        cache_key = "query:database_statistics"
        cached_result = self.get_cache(cache_key)
        if cached_result:
            log_universal('DEBUG', 'Database', 'Using cached database statistics')
            return cached_result
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM tracks")
                stats['total_tracks'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT artist) FROM tracks WHERE artist IS NOT NULL")
                stats['unique_artists'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT album) FROM tracks WHERE album IS NOT NULL")
                stats['unique_albums'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT genre) FROM tracks WHERE genre IS NOT NULL")
                stats['unique_genres'] = cursor.fetchone()[0]
                
                # Feature statistics
                cursor.execute("SELECT AVG(bpm), MIN(bpm), MAX(bpm) FROM tracks WHERE bpm IS NOT NULL")
                bpm_stats = cursor.fetchone()
                if bpm_stats[0]:
                    stats['bpm'] = {
                        'average': round(bpm_stats[0], 2),
                        'min': bpm_stats[1],
                        'max': bpm_stats[2]
                    }
                
                cursor.execute("SELECT AVG(loudness), MIN(loudness), MAX(loudness) FROM tracks WHERE loudness IS NOT NULL")
                loudness_stats = cursor.fetchone()
                if loudness_stats[0]:
                    stats['loudness'] = {
                        'average': round(loudness_stats[0], 2),
                        'min': loudness_stats[1],
                        'max': loudness_stats[2]
                    }
                
                cursor.execute("SELECT AVG(danceability), MIN(danceability), MAX(danceability) FROM tracks WHERE danceability IS NOT NULL")
                danceability_stats = cursor.fetchone()
                if danceability_stats[0]:
                    stats['danceability'] = {
                        'average': round(danceability_stats[0], 2),
                        'min': danceability_stats[1],
                        'max': danceability_stats[2]
                    }
                
                # Year distribution
                cursor.execute("""
                    SELECT year, COUNT(*) 
                    FROM tracks 
                    WHERE year IS NOT NULL 
                    GROUP BY year 
                    ORDER BY year
                """)
                stats['year_distribution'] = dict(cursor.fetchall())
                
                # Genre distribution
                cursor.execute("""
                    SELECT genre, COUNT(*) 
                    FROM tracks 
                    WHERE genre IS NOT NULL 
                    GROUP BY genre 
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """)
                stats['top_genres'] = dict(cursor.fetchall())
                
                # Artist distribution
                cursor.execute("""
                    SELECT artist, COUNT(*) 
                    FROM tracks 
                    WHERE artist IS NOT NULL 
                    GROUP BY artist 
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """)
                stats['top_artists'] = dict(cursor.fetchall())
                
                # External metadata counts
                cursor.execute("SELECT COUNT(*) FROM external_metadata WHERE source = 'musicbrainz'")
                stats['musicbrainz_entries'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM external_metadata WHERE source = 'lastfm'")
                stats['lastfm_entries'] = cursor.fetchone()[0]
                
                # Tags count
                cursor.execute("SELECT COUNT(*) FROM tags")
                stats['total_tags'] = cursor.fetchone()[0]
                
                log_universal('DEBUG', 'Database', f"Retrieved comprehensive database statistics")
                
                # Cache result for 30 minutes
                self.save_cache(cache_key, stats, expires_hours=0.5)
                
                return stats
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error getting database statistics: {e}")
            return {}

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
                    INSERT OR REPLACE INTO cache (cache_key, cache_value, created_at, expires_at)
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
                    SELECT cache_value, expires_at FROM cache 
                    WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
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
                     failed_date, last_retry_date)
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
                           failed_date, last_retry_date
                    FROM failed_analysis 
                    WHERE retry_count <= ?
                    ORDER BY last_retry_date ASC
                """, (max_retries,))
                
                failed_files = []
                for row in cursor.fetchall():
                    failed_file = {
                        'file_path': row[0],
                        'filename': row[1],
                        'error_message': row[2],
                        'retry_count': row[3],
                        'failed_date': row[4],
                        'last_retry_date': row[5]
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
                tables = ['playlists', 'tracks', 'analysis_cache', 'failed_analysis', 'cache', 'tags', 'spectral_features', 'loudness_features', 'advanced_features', 'external_metadata', 'mfcc_features', 'musicnn_features', 'chroma_features', 'rhythm_features', 'playlist_tracks']
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
                    SELECT COUNT(*) FROM tracks 
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

    def _invalidate_related_caches(self, operation: str, data: Dict[str, Any] = None):
        """
        Invalidate related caches when data changes.
        
        Args:
            operation: Type of operation (insert, update, delete)
            data: Optional data that was changed
        """
        try:
            # Clear query result caches
            query_caches = [
                "query:all_artists",
                "query:all_albums", 
                "query:all_genres",
                "query:all_categories",
                "query:database_statistics"
            ]
            
            for cache_key in query_caches:
                self.delete_cache(cache_key)
            
            # Clear artist-specific caches if we have artist data
            if data and 'artist' in data:
                artist_cache_key = f"query:artist:{data['artist']}"
                self.delete_cache(artist_cache_key)
            
            # Clear album-specific caches if we have album data
            if data and 'album' in data:
                album_cache_key = f"query:album:{data['album']}"
                self.delete_cache(album_cache_key)
            
            # Clear genre-specific caches if we have genre data
            if data and 'genre' in data:
                genre_cache_key = f"query:genre:{data['genre']}"
                self.delete_cache(genre_cache_key)
            
            log_universal('DEBUG', 'Database', f'Invalidated related caches for {operation} operation')
            
        except Exception as e:
            log_universal('WARNING', 'Database', f'Error invalidating caches: {e}')
    
    @log_function_call
    def delete_cache(self, key: str) -> bool:
        """
        Delete a specific cache entry.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    log_universal('DEBUG', 'Database', f'Deleted cache entry: {key}')
                else:
                    log_universal('DEBUG', 'Database', f'Cache entry not found: {key}')
                
                return deleted
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Error deleting cache entry {key}: {e}')
            return False
    
    @log_function_call
    def clear_all_caches(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache")
                deleted_count = cursor.rowcount
                conn.commit()
                
                log_universal('INFO', 'Database', f'Cleared all cache entries: {deleted_count}')
                return deleted_count
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Error clearing all caches: {e}')
            return 0
    
    @log_function_call
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Total cache entries
                cursor.execute("SELECT COUNT(*) FROM cache")
                stats['total_entries'] = cursor.fetchone()[0]
                
                # Expired entries
                cursor.execute("""
                    SELECT COUNT(*) FROM cache 
                    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """)
                stats['expired_entries'] = cursor.fetchone()[0]
                
                # Valid entries
                cursor.execute("""
                    SELECT COUNT(*) FROM cache 
                    WHERE expires_at IS NULL OR expires_at >= CURRENT_TIMESTAMP
                """)
                stats['valid_entries'] = cursor.fetchone()[0]
                
                # Cache size in MB
                cursor.execute("""
                    SELECT SUM(LENGTH(value)) FROM cache
                """)
                size_bytes = cursor.fetchone()[0] or 0
                stats['size_mb'] = round(size_bytes / (1024 * 1024), 2)
                
                # Oldest and newest entries
                cursor.execute("""
                    SELECT MIN(created_at), MAX(created_at) FROM cache
                """)
                oldest, newest = cursor.fetchone()
                stats['oldest_entry'] = oldest
                stats['newest_entry'] = newest
                
                # Top cache keys by size
                cursor.execute("""
                    SELECT key, LENGTH(value) as size 
                    FROM cache 
                    ORDER BY size DESC 
                    LIMIT 10
                """)
                stats['largest_entries'] = dict(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Error getting cache statistics: {e}')
            return {}


# Global database manager instance - created lazily to avoid circular imports
_db_manager_instance = None

def get_db_manager() -> 'DatabaseManager':
    """Get the global database manager instance, creating it if necessary."""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
    return _db_manager_instance 
