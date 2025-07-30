"""
Database manager for Playlist Generator Simple.
Handles playlist storage, caching, analysis results, and metadata.
"""

import sqlite3
import json
import os
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path

# Import configuration and logging
from .config_loader import config_loader
from .logging_setup import get_logger, log_function_call, log_performance

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
            db_path = '/app/cache/playlista.db'  # Fixed Docker internal path
        
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
        
        logger.info(f"🗄️ Initializing DatabaseManager with path: {db_path}")
        logger.debug(f"📋 Database configuration: {config}")
        
        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        logger.debug(f"📁 Database directory ready: {db_dir}")
        
        # Initialize database tables
        start_time = time.time()
        self._init_database()
        init_time = time.time() - start_time
        
        # Log performance
        log_performance("DatabaseManager initialization", init_time)
        logger.info(f"✅ DatabaseManager initialized successfully in {init_time:.2f}s")

    @log_function_call
    def _init_database(self):
        """Initialize the database with all required tables."""
        logger.info("🗄️ Initializing database tables...")
        
        tables_created = 0
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create playlists table
                logger.debug("📋 Creating playlists table...")
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
                logger.debug("✅ Playlists table ready")
                
                # Create analysis_results table
                logger.debug("📊 Creating analysis_results table...")
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
                logger.debug("✅ Analysis results table ready")
                
                # Create cache table
                logger.debug("💾 Creating cache table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                tables_created += 1
                logger.debug("✅ Cache table ready")
                
                # Create tags table
                logger.debug("🏷️ Creating tags table...")
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
                logger.debug("✅ Tags table ready")
                
                # Create failed_analysis table
                logger.debug("❌ Creating failed_analysis table...")
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
                logger.debug("✅ Failed analysis table ready")
                
                # Create statistics table
                logger.debug("📈 Creating statistics table...")
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
                logger.debug("✅ Statistics table ready")
                
                conn.commit()
                init_time = time.time() - start_time
                logger.info(f"✅ Database initialization completed successfully")
                logger.info(f"📊 Created {tables_created} tables in {init_time:.2f}s")
                
                # Log performance
                log_performance("Database table creation", init_time, tables_created=tables_created)
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

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
        logger.info(f"📋 Saving playlist '{name}' with {len(tracks)} tracks")
        
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                logger.info(f"✅ Successfully saved playlist '{name}' to database in {save_time:.2f}s")
                
                # Log performance
                log_performance("Playlist save", save_time, playlist_name=name, track_count=len(tracks))
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving playlist '{name}': {e}")
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
        logger.debug(f"Retrieving playlist '{name}' from database")
        
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
                    logger.debug(f"Retrieved playlist '{name}' with {len(playlist['tracks'])} tracks")
                    return playlist
                else:
                    logger.debug(f"Playlist '{name}' not found in database")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving playlist '{name}': {e}")
            return None

    @log_function_call
    def get_all_playlists(self) -> List[Dict[str, Any]]:
        """
        Get all playlists from the database.
        
        Returns:
            List of playlist dictionaries
        """
        logger.debug("Retrieving all playlists from database")
        
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
                
                logger.info(f"Retrieved {len(playlists)} playlists from database")
                return playlists
                
        except Exception as e:
            logger.error(f"Error retrieving all playlists: {e}")
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
        logger.info(f"Deleting playlist '{name}' from database")
        
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
            logger.error(f"Error deleting playlist '{name}': {e}")
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
                
                logger.debug(f"📁 Retrieved {len(tracks)} analyzed tracks")
                return tracks
                
        except Exception as e:
            logger.error(f"❌ Error getting analyzed tracks: {e}")
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
                
                logger.debug(f"📁 Retrieved {len(cached_playlists)} cached playlists")
                return cached_playlists
                
        except Exception as e:
            logger.error(f"❌ Error getting cached playlists: {e}")
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
            logger.error(f"❌ Error getting track features for {file_path}: {e}")
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
        logger.debug(f"🎤 Getting tracks by artist: {artist}")
        
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
                
                logger.debug(f"🎤 Retrieved {len(tracks)} tracks by artist: {artist}")
                return tracks
                
        except Exception as e:
            logger.error(f"❌ Error getting tracks by artist: {e}")
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
        logger.debug(f"💿 Getting tracks by album: {album}")
        
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
                
                logger.debug(f"💿 Retrieved {len(tracks)} tracks from album: {album}")
                return tracks
                
        except Exception as e:
            logger.error(f"❌ Error getting tracks by album: {e}")
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
        logger.debug(f"🎵 Getting tracks by genre: {genre}")
        
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
                
                logger.debug(f"🎵 Retrieved {len(tracks)} tracks of genre: {genre}")
                return tracks
                
        except Exception as e:
            logger.error(f"❌ Error getting tracks by genre: {e}")
            return []

    @log_function_call
    def get_all_artists(self) -> List[str]:
        """
        Get all unique artists in the database.
        
        Returns:
            List of artist names
        """
        logger.debug("🎤 Getting all unique artists...")
        
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
                logger.debug(f"🎤 Found {len(artists)} unique artists")
                return artists
                
        except Exception as e:
            logger.error(f"❌ Error getting all artists: {e}")
            return []

    @log_function_call
    def get_all_albums(self) -> List[str]:
        """
        Get all unique albums in the database.
        
        Returns:
            List of album names
        """
        logger.debug("💿 Getting all unique albums...")
        
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
                logger.debug(f"💿 Found {len(albums)} unique albums")
                return albums
                
        except Exception as e:
            logger.error(f"❌ Error getting all albums: {e}")
            return []

    @log_function_call
    def get_all_genres(self) -> List[str]:
        """
        Get all unique genres in the database.
        
        Returns:
            List of genre names
        """
        logger.debug("🎵 Getting all unique genres...")
        
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
                logger.debug(f"🎵 Found {len(genres)} unique genres")
                return genres
                
        except Exception as e:
            logger.error(f"❌ Error getting all genres: {e}")
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
        logger.debug(f"📅 Getting tracks from year: {year}")
        
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
                
                logger.debug(f"📅 Retrieved {len(tracks)} tracks from year: {year}")
                return tracks
                
        except Exception as e:
            logger.error(f"❌ Error getting tracks by year: {e}")
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
                    logger.error(f"❌ Invalid feature name: {feature}")
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
                
                logger.debug(f"📁 Retrieved {len(tracks)} tracks with {feature} between {min_value} and {max_value}")
                return tracks
                
        except Exception as e:
            logger.error(f"❌ Error getting tracks by feature range: {e}")
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
        logger.debug(f"📊 Saving analysis results for: {filename}")
        
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                analysis_json = json.dumps(analysis_data)
                metadata_json = json.dumps(metadata) if metadata else None
                
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
                logger.debug(f"✅ Successfully saved analysis results for: {filename} in {save_time:.2f}s")
                
                # Log performance
                log_performance("Analysis result save", save_time, filename=filename, file_size=file_size_bytes)
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving analysis results for {filename}: {e}")
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
        logger.debug(f"Retrieving analysis results for: {file_path}")
        
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
                    logger.debug(f"Retrieved analysis results for: {row[0]}")
                    return result
                else:
                    logger.debug(f"No analysis results found for: {file_path}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving analysis results for {file_path}: {e}")
            return None

    @log_function_call
    def get_all_analysis_results(self) -> List[Dict[str, Any]]:
        """
        Get all analysis results from the database.
        
        Returns:
            List of analysis result dictionaries
        """
        logger.debug("Retrieving all analysis results from database")
        
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
                
                logger.info(f"Retrieved {len(results)} analysis results from database")
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving all analysis results: {e}")
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
        logger.debug(f"Deleting analysis results for: {file_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM analysis_results WHERE file_path = ?", (file_path,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.debug(f"Successfully deleted analysis results for: {file_path}")
                    return True
                else:
                    logger.debug(f"No analysis results found for: {file_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting analysis results for {file_path}: {e}")
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
        logger.debug(f"Deleting failed analysis for: {file_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM failed_analysis WHERE file_path = ?", (file_path,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.debug(f"Successfully deleted failed analysis for: {file_path}")
                    return True
                else:
                    logger.debug(f"No failed analysis found for: {file_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting failed analysis for {file_path}: {e}")
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
        
        logger.debug(f"💾 Saving cache entry: {key} (expires in {expires_hours}h)")
        
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
                logger.debug(f"✅ Successfully saved cache entry: {key} in {save_time:.2f}s")
                
                # Log performance
                log_performance("Cache save", save_time, cache_key=key, expires_hours=expires_hours)
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving cache entry {key}: {e}")
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
        logger.debug(f"Retrieving cache entry: {key}")
        
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
                    logger.debug(f"Retrieved cache entry: {key}")
                    return value
                else:
                    logger.debug(f"Cache entry not found or expired: {key}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving cache entry {key}: {e}")
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
        
        logger.info(f"🧹 Cleaning up cache entries older than {max_age_hours} hours")
        
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
                
                logger.info(f"✅ Cleaned up {deleted_count} expired cache entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up cache: {e}")
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
        logger.debug(f"Saving tags for: {file_path}")
        
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
                logger.debug(f"Successfully saved tags for: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving tags for {file_path}: {e}")
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
        logger.debug(f"Retrieving tags for: {file_path}")
        
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
                    logger.debug(f"Retrieved tags for: {file_path}")
                    return tags_data
                else:
                    logger.debug(f"No tags found for: {file_path}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving tags for {file_path}: {e}")
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
        logger.info(f"❌ Marking analysis as failed for: {filename}")
        logger.debug(f"   Error: {error_message}")
        
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
                logger.info(f"✅ Successfully marked analysis as failed for: {filename} in {save_time:.2f}s")
                
                # Log performance
                log_performance("Failed analysis mark", save_time, filename=filename, error_message=error_message)
                return True
                
        except Exception as e:
            logger.error(f"❌ Error marking analysis as failed for {filename}: {e}")
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
        logger.debug(f"Retrieving failed analysis files (max retries: {max_retries})")
        
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
                
                logger.info(f"Retrieved {len(failed_files)} failed analysis files")
                return failed_files
                
        except Exception as e:
            logger.error(f"Error retrieving failed analysis files: {e}")
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
        logger.debug(f"Saving statistic: {category}.{key}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                value_json = json.dumps(value)
                
                cursor.execute("""
                    INSERT INTO statistics (category, key, value, timestamp)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (category, key, value_json))
                
                conn.commit()
                logger.debug(f"Successfully saved statistic: {category}.{key}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving statistic {category}.{key}: {e}")
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
        
        logger.debug(f"📈 Retrieving statistics (category: {category}, hours: {hours})")
        
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
                
                logger.info(f"✅ Retrieved statistics: {len(stats)} categories")
                return stats
                
        except Exception as e:
            logger.error(f"❌ Error retrieving statistics: {e}")
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
        logger.debug("📈 Generating database statistics")
        
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
                    logger.debug(f"📊 {table}: {count} records")
                
                # Get database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                stats['database_size_bytes'] = db_size
                stats['database_size_mb'] = db_size / (1024 * 1024)
                logger.debug(f"📁 Database size: {stats['database_size_mb']:.2f} MB")
                
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
                logger.info(f"✅ Database statistics generated successfully in {gen_time:.2f}s")
                
                # Log performance
                log_performance("Database statistics generation", gen_time, 
                              total_records=sum(stats[f'{table}_count'] for table in tables))
                return stats
                
        except Exception as e:
            logger.error(f"❌ Error generating database statistics: {e}")
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
        
        logger.info(f"🧹 Cleaning up data older than {days} days")
        logger.debug(f"   Failed analysis retention: {failed_retention_days} days")
        logger.debug(f"   Statistics retention: {stats_retention_days} days")
        
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
                logger.debug(f"🗑️ Cleaned {results['cache_cleaned']} cache entries")
                
                # Clean up old statistics
                cursor.execute("""
                    DELETE FROM statistics 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(stats_retention_days))
                results['statistics_cleaned'] = cursor.rowcount
                logger.debug(f"🗑️ Cleaned {results['statistics_cleaned']} statistics entries")
                
                # Clean up old failed analysis (keep for shorter time)
                cursor.execute("""
                    DELETE FROM failed_analysis 
                    WHERE failed_date < datetime('now', '-{} days')
                """.format(failed_retention_days))
                results['failed_analysis_cleaned'] = cursor.rowcount
                logger.debug(f"🗑️ Cleaned {results['failed_analysis_cleaned']} failed analysis entries")
                
                conn.commit()
                
                cleanup_time = time.time() - start_time
                total_cleaned = sum(results.values())
                logger.info(f"✅ Cleanup completed in {cleanup_time:.2f}s: {results}")
                logger.info(f"📊 Total cleaned: {total_cleaned} entries")
                
                # Log performance
                log_performance("Database cleanup", cleanup_time, 
                              total_cleaned=total_cleaned, days=days)
                return results
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
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
        logger.info(f"📤 Exporting database to: {export_path}")
        
        start_time = time.time()
        
        try:
            # Get source database size
            source_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            logger.debug(f"📁 Source database size: {source_size / (1024 * 1024):.2f} MB")
            
            with sqlite3.connect(self.db_path) as source_conn:
                with sqlite3.connect(export_path) as dest_conn:
                    source_conn.backup(dest_conn)
            
            # Get exported database size
            export_size = os.path.getsize(export_path) if os.path.exists(export_path) else 0
            export_time = time.time() - start_time
            
            logger.info(f"✅ Successfully exported database to: {export_path}")
            logger.info(f"📊 Export size: {export_size / (1024 * 1024):.2f} MB in {export_time:.2f}s")
            
            # Log performance
            log_performance("Database export", export_time, 
                          source_size_mb=source_size/(1024*1024), 
                          export_size_mb=export_size/(1024*1024))
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting database: {e}")
            return False


    def get_database_config(self) -> Dict[str, Any]:
        """
        Get current database configuration.
        
        Returns:
            Dictionary with current database settings
        """
        return self.config.copy()
    
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
            logger.info(f"📋 Updated database configuration: {new_config}")
            return True
        except Exception as e:
            logger.error(f"❌ Error updating database configuration: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager() 