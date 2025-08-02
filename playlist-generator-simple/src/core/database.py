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
                
                # Read and execute schema - try multiple possible paths
                schema_paths = [
                    os.path.join(os.path.dirname(self.db_path), 'database_schema.sql'),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database_schema.sql'),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'database_schema.sql'),
                ]
                
                schema_sql = None
                for schema_path in schema_paths:
                    if os.path.exists(schema_path):
                        with open(schema_path, 'r') as f:
                            schema_sql = f.read()
                        log_universal('INFO', 'Database', f"Found schema file: {schema_path}")
                        break
                
                if schema_sql:
                    cursor.executescript(schema_sql)
                    log_universal('INFO', 'Database', "Database schema created successfully")
                else:
                    log_universal('ERROR', 'Database', f"Schema file not found in any of these paths: {schema_paths}")
                    raise FileNotFoundError(f"Schema file not found in any of these paths: {schema_paths}")
                
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
                    (directory_path, file_count, scan_duration, status, error_message, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (directory_path, file_count, scan_duration, status, error_message, datetime.now()))
                
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
                
                # Debug logging for metadata
                log_universal('DEBUG', 'Database', f'Metadata received: {metadata}')
                if metadata is None:
                    log_universal('WARNING', 'Database', f'Metadata is None for file: {file_path}')
                elif not metadata:
                    log_universal('WARNING', 'Database', f'Metadata is empty for file: {file_path}')
                
                # Extract core data with better fallbacks
                title = 'Unknown'
                artist = 'Unknown'
                
                if metadata:
                    title = metadata.get('title', 'Unknown')
                    artist = metadata.get('artist', 'Unknown')
                
                # If still unknown, try to extract from filename
                if title == 'Unknown' or artist == 'Unknown':
                    # Try to parse artist - title from filename
                    filename_without_ext = os.path.splitext(filename)[0]
                    if ' - ' in filename_without_ext:
                        parts = filename_without_ext.split(' - ', 1)
                        if len(parts) == 2:
                            if artist == 'Unknown':
                                artist = parts[0].strip()
                            if title == 'Unknown':
                                title = parts[1].strip()
                    else:
                        # No separator found, use filename as title
                        if title == 'Unknown':
                            title = filename_without_ext
                
                # Final fallback - use filename as title
                if title == 'Unknown':
                    title = os.path.splitext(filename)[0]
                
                log_universal('DEBUG', 'Database', f'Final title: {title}, artist: {artist} for file: {file_path}')
                
                album = metadata.get('album') if metadata else None
                track_number = metadata.get('track_number') if metadata else None
                genre = metadata.get('genre') if metadata else None
                year = metadata.get('year') if metadata else None
                duration = analysis_data.get('duration')
                
                # Extract audio features
                bpm = analysis_data.get('bpm')
                key = analysis_data.get('key')
                mode = analysis_data.get('mode')
                loudness = analysis_data.get('loudness')
                danceability = analysis_data.get('danceability')
                energy = analysis_data.get('energy')
                
                # Extract rhythm features
                rhythm_confidence = analysis_data.get('rhythm_confidence')
                bpm_estimates = json.dumps(analysis_data.get('bpm_estimates', []))
                bpm_intervals = json.dumps(analysis_data.get('bpm_intervals', []))
                external_bpm = analysis_data.get('external_bpm')
                
                # Extract spectral features
                spectral_centroid = analysis_data.get('spectral_centroid')
                spectral_flatness = analysis_data.get('spectral_flatness')
                spectral_rolloff = analysis_data.get('spectral_rolloff')
                spectral_bandwidth = analysis_data.get('spectral_bandwidth')
                spectral_contrast_mean = analysis_data.get('spectral_contrast_mean')
                spectral_contrast_std = analysis_data.get('spectral_contrast_std')
                
                # Extract loudness features
                dynamic_complexity = analysis_data.get('dynamic_complexity')
                loudness_range = analysis_data.get('loudness_range')
                dynamic_range = analysis_data.get('dynamic_range')
                
                # Extract key features
                scale = analysis_data.get('scale')
                key_strength = analysis_data.get('key_strength')
                key_confidence = analysis_data.get('key_confidence')
                
                # Extract MFCC features
                mfcc_coefficients = json.dumps(analysis_data.get('mfcc_coefficients', []))
                mfcc_bands = json.dumps(analysis_data.get('mfcc_bands', []))
                mfcc_std = json.dumps(analysis_data.get('mfcc_std', []))
                
                # Extract MusiCNN features
                embedding = json.dumps(analysis_data.get('embedding', []))
                tags = json.dumps(analysis_data.get('tags', {}))
                
                # Debug logging for MusiCNN data
                if analysis_data.get('embedding'):
                    log_universal('DEBUG', 'Database', f'MusiCNN embedding length: {len(analysis_data.get("embedding", []))}')
                if analysis_data.get('tags'):
                    log_universal('DEBUG', 'Database', f'MusiCNN tags count: {len(analysis_data.get("tags", {}))}')
                    top_tags = sorted(analysis_data.get('tags', {}).items(), key=lambda x: x[1], reverse=True)[:3]
                    log_universal('DEBUG', 'Database', f'Top MusiCNN tags: {top_tags}')
                
                # Extract chroma features
                chroma_mean = json.dumps(analysis_data.get('chroma_mean', []))
                chroma_std = json.dumps(analysis_data.get('chroma_std', []))
                
                # Extract additional metadata
                bitrate = metadata.get('bitrate') if metadata else None
                sample_rate = metadata.get('sample_rate') if metadata else None
                channels = metadata.get('channels') if metadata else None
                composer = metadata.get('composer') if metadata else None
                lyricist = metadata.get('lyricist') if metadata else None
                band = metadata.get('band') if metadata else None
                conductor = metadata.get('conductor') if metadata else None
                remixer = metadata.get('remixer') if metadata else None
                subtitle = metadata.get('subtitle') if metadata else None
                grouping = metadata.get('grouping') if metadata else None
                publisher = metadata.get('publisher') if metadata else None
                copyright = metadata.get('copyright') if metadata else None
                encoded_by = metadata.get('encoded_by') if metadata else None
                language = metadata.get('language') if metadata else None
                mood = metadata.get('mood') if metadata else None
                style = metadata.get('style') if metadata else None
                quality = metadata.get('quality') if metadata else None
                original_artist = metadata.get('original_artist') if metadata else None
                original_album = metadata.get('original_album') if metadata else None
                original_year = metadata.get('original_year') if metadata else None
                original_filename = metadata.get('original_filename') if metadata else None
                content_group = metadata.get('content_group') if metadata else None
                encoder = metadata.get('encoder') if metadata else None
                file_type = metadata.get('file_type') if metadata else None
                playlist_delay = metadata.get('playlist_delay') if metadata else None
                recording_time = metadata.get('recording_time') if metadata else None
                tempo = metadata.get('tempo') if metadata else None
                length = metadata.get('length') if metadata else None
                replaygain_track_gain = metadata.get('replaygain_track_gain') if metadata else None
                replaygain_album_gain = metadata.get('replaygain_album_gain') if metadata else None
                replaygain_track_peak = metadata.get('replaygain_track_peak') if metadata else None
                replaygain_album_peak = metadata.get('replaygain_album_peak') if metadata else None
                
                # Extract missing audio analysis fields
                # Rhythm & Tempo Analysis
                tempo_confidence = analysis_data.get('tempo_confidence')
                tempo_strength = analysis_data.get('tempo_strength')
                rhythm_pattern = analysis_data.get('rhythm_pattern')
                beat_positions = json.dumps(analysis_data.get('beat_positions', []))
                onset_times = json.dumps(analysis_data.get('onset_times', []))
                rhythm_complexity = analysis_data.get('rhythm_complexity')
                
                # Harmonic Analysis
                harmonic_complexity = analysis_data.get('harmonic_complexity')
                chord_progression = json.dumps(analysis_data.get('chord_progression', []))
                harmonic_centroid = analysis_data.get('harmonic_centroid')
                harmonic_contrast = analysis_data.get('harmonic_contrast')
                chord_changes = analysis_data.get('chord_changes')
                
                # Extended Spectral Analysis
                spectral_flux = analysis_data.get('spectral_flux')
                spectral_crest = analysis_data.get('spectral_crest')
                spectral_decrease = analysis_data.get('spectral_decrease')
                spectral_entropy = analysis_data.get('spectral_entropy')
                spectral_kurtosis = analysis_data.get('spectral_kurtosis')
                spectral_skewness = analysis_data.get('spectral_skewness')
                spectral_slope = analysis_data.get('spectral_slope')
                spectral_rolloff_85 = analysis_data.get('spectral_rolloff_85')
                spectral_rolloff_95 = analysis_data.get('spectral_rolloff_95')
                
                # Timbre Analysis
                timbre_brightness = analysis_data.get('timbre_brightness')
                timbre_warmth = analysis_data.get('timbre_warmth')
                timbre_hardness = analysis_data.get('timbre_hardness')
                timbre_depth = analysis_data.get('timbre_depth')
                mfcc_delta = json.dumps(analysis_data.get('mfcc_delta', []))
                mfcc_delta2 = json.dumps(analysis_data.get('mfcc_delta2', []))
                
                # Perceptual Features
                acousticness = analysis_data.get('acousticness')
                instrumentalness = analysis_data.get('instrumentalness')
                speechiness = analysis_data.get('speechiness')
                valence = analysis_data.get('valence')
                liveness = analysis_data.get('liveness')
                popularity = analysis_data.get('popularity')
                
                # Advanced Audio Features
                zero_crossing_rate = analysis_data.get('zero_crossing_rate')
                root_mean_square = analysis_data.get('root_mean_square')
                peak_amplitude = analysis_data.get('peak_amplitude')
                crest_factor = analysis_data.get('crest_factor')
                signal_to_noise_ratio = analysis_data.get('signal_to_noise_ratio')
                
                # Musical Structure Analysis
                intro_duration = analysis_data.get('intro_duration')
                verse_duration = analysis_data.get('verse_duration')
                chorus_duration = analysis_data.get('chorus_duration')
                bridge_duration = analysis_data.get('bridge_duration')
                outro_duration = analysis_data.get('outro_duration')
                section_boundaries = json.dumps(analysis_data.get('section_boundaries', []))
                repetition_rate = analysis_data.get('repetition_rate')
                
                # Advanced Key Analysis
                key_scale_notes = json.dumps(analysis_data.get('key_scale_notes', []))
                key_chord_progression = json.dumps(analysis_data.get('key_chord_progression', []))
                modulation_points = json.dumps(analysis_data.get('modulation_points', []))
                tonal_centroid = analysis_data.get('tonal_centroid')
                
                # Audio Quality Metrics
                bitrate_quality = analysis_data.get('bitrate_quality')
                sample_rate_quality = analysis_data.get('sample_rate_quality')
                encoding_quality = analysis_data.get('encoding_quality')
                compression_artifacts = analysis_data.get('compression_artifacts')
                clipping_detection = analysis_data.get('clipping_detection')
                
                # Genre-Specific Features
                electronic_elements = analysis_data.get('electronic_elements')
                classical_period = analysis_data.get('classical_period')
                jazz_style = analysis_data.get('jazz_style')
                rock_subgenre = analysis_data.get('rock_subgenre')
                folk_style = analysis_data.get('folk_style')
                
                # Determine analysis type and category
                analysis_type = analysis_data.get('analysis_type', 'full')
                long_audio_category = analysis_data.get('long_audio_category')
                
                # Insert or update track with all features including Spotify-style features
                cursor.execute("""
                    INSERT OR REPLACE INTO tracks (
                        file_path, file_hash, filename, file_size_bytes, analysis_date, discovery_date,
                        status, analysis_status, modified_time, retry_count, last_retry_date, error_message,
                        title, artist, album, track_number, genre, year, duration,
                        bitrate, sample_rate, channels,
                        bpm, rhythm_confidence, bpm_estimates, bpm_intervals, external_bpm,
                        key, scale, key_strength, key_confidence,
                        spectral_centroid, spectral_flatness, spectral_rolloff, spectral_bandwidth, spectral_contrast_mean, spectral_contrast_std,
                        loudness, dynamic_complexity, loudness_range, dynamic_range,
                        danceability, energy, mode,
                        acousticness, instrumentalness, speechiness, valence, liveness, popularity,
                        mfcc_coefficients, mfcc_bands, mfcc_std, mfcc_delta, mfcc_delta2,
                        embedding, tags,
                        chroma_mean, chroma_std,
                        composer, lyricist, band, conductor, remixer, subtitle, grouping, publisher, copyright, encoded_by, language, mood, style, quality, original_artist, original_album, original_year, original_filename, content_group, encoder, file_type, playlist_delay, recording_time, tempo, length, replaygain_track_gain, replaygain_album_gain, replaygain_track_peak, replaygain_album_peak,
                        analysis_type, analyzed, long_audio_category, discovery_source,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_path, file_hash, filename, file_size_bytes, datetime.now(), datetime.now(),
                    'analyzed', 'completed', metadata.get('modified_time') if metadata else None, 0, None, None,
                    title, artist, album, track_number, genre, year, duration,
                    bitrate, sample_rate, channels,
                    bpm, rhythm_confidence, bpm_estimates, bpm_intervals, external_bpm,
                    key, scale, key_strength, key_confidence,
                    spectral_centroid, spectral_flatness, spectral_rolloff, spectral_bandwidth, spectral_contrast_mean, spectral_contrast_std,
                    loudness, dynamic_complexity, loudness_range, dynamic_range,
                    danceability, energy, mode,
                    acousticness, instrumentalness, speechiness, valence, liveness, popularity,
                    mfcc_coefficients, mfcc_bands, mfcc_std, mfcc_delta, mfcc_delta2,
                    embedding, tags,
                    chroma_mean, chroma_std,
                    composer, lyricist, band, conductor, remixer, subtitle, grouping, publisher, copyright, encoded_by, language, mood, style, quality, original_artist, original_album, original_year, original_filename, content_group, encoder, file_type, playlist_delay, recording_time, tempo, length, replaygain_track_gain, replaygain_album_gain, replaygain_track_peak, replaygain_album_peak,
                    analysis_type, True, long_audio_category, discovery_source,
                    datetime.now(), datetime.now()
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

    @log_function_call
    def get_failed_analysis_files(self, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Get list of failed analysis files from analysis_cache table."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path, filename, error_message, retry_count, last_retry_date
                    FROM analysis_cache 
                    WHERE status = 'failed' AND retry_count < ?
                    ORDER BY last_retry_date DESC
                """, (max_retries,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'file_path': row[0],
                        'filename': row[1],
                        'error_message': row[2],
                        'retry_count': row[3],
                        'last_retry_date': row[4]
                    })
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(results)} failed analysis files")
                return results
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get failed analysis files: {e}")
            return []

    @log_function_call
    def mark_analysis_failed(self, file_path: str, filename: str, error_message: str) -> bool:
        """Mark an analysis as failed in the analysis_cache table."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_cache 
                    (file_path, filename, error_message, status, retry_count, last_retry_date)
                    VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                """, (file_path, filename, error_message))
                conn.commit()
                
                log_universal('INFO', 'Database', f"Marked analysis as failed: {filename}")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to mark analysis as failed: {e}")
            return False

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

    def _save_tags(self, cursor, track_id: int, tags):
        """Save tags for a track."""
        if isinstance(tags, dict):
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
        elif isinstance(tags, list):
            # Handle list of tags directly
            for tag_name in tags:
                cursor.execute("""
                    INSERT OR REPLACE INTO tags (track_id, source, tag_name)
                    VALUES (?, ?, ?)
                """, (track_id, 'user', tag_name))

    @log_function_call
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Get track counts
                cursor.execute("SELECT COUNT(*) FROM tracks")
                stats['total_tracks'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM tracks WHERE analysis_type = 'full'")
                stats['fully_analyzed_tracks'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM tracks WHERE analysis_type = 'discovery_only'")
                stats['discovered_tracks'] = cursor.fetchone()[0]
                
                # Get playlist counts
                cursor.execute("SELECT COUNT(*) FROM playlists")
                stats['total_playlists'] = cursor.fetchone()[0]
                
                # Get failed analysis counts
                cursor.execute("SELECT COUNT(*) FROM analysis_cache WHERE status = 'failed'")
                stats['failed_analyses'] = cursor.fetchone()[0]
                
                # Get cache statistics
                cursor.execute("SELECT COUNT(*) FROM cache")
                stats['cache_entries'] = cursor.fetchone()[0]
                
                # Get discovery statistics
                cursor.execute("SELECT COUNT(*) FROM discovery_cache")
                stats['discovery_entries'] = cursor.fetchone()[0]
                
                # Get tag statistics
                cursor.execute("SELECT COUNT(*) FROM tags")
                stats['total_tags'] = cursor.fetchone()[0]
                
                log_universal('DEBUG', 'Database', f"Retrieved database statistics: {stats}")
                return stats
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get database statistics: {e}")
            return {}

    @log_function_call
    def get_all_analysis_results(self) -> List[Dict[str, Any]]:
        """Get all analysis results from tracks table."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, file_path, filename, file_size_bytes, file_hash, 
                           analysis_date, analysis_type, long_audio_category,
                           title, artist, album, genre, year, duration,
                           bpm, key, mode, loudness, danceability, energy
                    FROM tracks
                    ORDER BY analysis_date DESC
                """)
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'file_path': row[1],
                        'filename': row[2],
                        'file_size_bytes': row[3],
                        'file_hash': row[4],
                        'analysis_date': row[5],
                        'analysis_type': row[6],
                        'long_audio_category': row[7],
                        'title': row[8],
                        'artist': row[9],
                        'album': row[10],
                        'genre': row[11],
                        'year': row[12],
                        'duration': row[13],
                        'bpm': row[14],
                        'key': row[15],
                        'mode': row[16],
                        'loudness': row[17],
                        'danceability': row[18],
                        'energy': row[19]
                    })
                
                log_universal('DEBUG', 'Database', f"Retrieved {len(results)} analysis results")
                return results
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get all analysis results: {e}")
            return []

    @log_function_call
    def delete_failed_analysis(self, file_path: str) -> bool:
        """Delete a failed analysis entry from analysis_cache table."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM analysis_cache WHERE file_path = ?
                """, (file_path,))
                conn.commit()
                
                log_universal('INFO', 'Database', f"Deleted failed analysis: {file_path}")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to delete failed analysis: {e}")
            return False

    @log_function_call
    def get_analysis_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get analysis result for a specific file from tracks table."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, file_path, filename, file_size_bytes, file_hash, 
                           analysis_date, analysis_type, analyzed, long_audio_category,
                           title, artist, album, genre, year, duration,
                           bpm, key, mode, loudness, danceability, energy,
                           discovery_date, discovery_source, created_at, updated_at
                    FROM tracks 
                    WHERE file_path = ?
                """, (file_path,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                # Convert to dictionary
                analysis_result = {
                    'id': result[0],
                    'file_path': result[1],
                    'filename': result[2],
                    'file_size_bytes': result[3],
                    'file_hash': result[4],
                    'analysis_date': result[5],
                    'analysis_type': result[6],
                    'analyzed': result[7],
                    'long_audio_category': result[8],
                    'title': result[9],
                    'artist': result[10],
                    'album': result[11],
                    'genre': result[12],
                    'year': result[13],
                    'duration': result[14],
                    'bpm': result[15],
                    'key': result[16],
                    'mode': result[17],
                    'loudness': result[18],
                    'danceability': result[19],
                    'energy': result[20],
                    'discovery_date': result[21],
                    'discovery_source': result[22],
                    'created_at': result[23],
                    'updated_at': result[24]
                }
                
                # Get tags for this track
                cursor.execute("""
                    SELECT source, tag_name, tag_value, confidence
                    FROM tags 
                    WHERE track_id = ?
                """, (analysis_result['id'],))
                
                tags = {}
                for tag_row in cursor.fetchall():
                    source = tag_row[0]
                    if source not in tags:
                        tags[source] = {}
                    tags[source][tag_row[1]] = tag_row[2]
                
                analysis_result['tags'] = tags
                
                log_universal('DEBUG', 'Database', f"Retrieved analysis result for: {file_path}")
                return analysis_result
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get analysis result: {e}")
            return None

    @log_function_call
    def delete_analysis_result(self, file_path: str) -> bool:
        """Delete analysis result for a specific file from tracks table."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get track_id first
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                result = cursor.fetchone()
                if not result:
                    log_universal('DEBUG', 'Database', f"No analysis result found for: {file_path}")
                    return True  # Nothing to delete
                
                track_id = result[0]
                
                # Delete tags first (due to foreign key constraint)
                cursor.execute("DELETE FROM tags WHERE track_id = ?", (track_id,))
                
                # Delete from playlist_tracks
                cursor.execute("DELETE FROM playlist_tracks WHERE track_id = ?", (track_id,))
                
                # Delete from tracks
                cursor.execute("DELETE FROM tracks WHERE file_path = ?", (file_path,))
                
                conn.commit()
                log_universal('INFO', 'Database', f"Deleted analysis result for: {file_path}")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to delete analysis result: {e}")
            return False

    # =============================================================================
    # DATABASE MANAGEMENT METHODS
    # =============================================================================

    @log_function_call
    def initialize_schema(self) -> bool:
        """Initialize database schema."""
        try:
            self._init_database()
            log_universal('INFO', 'Database', "Database schema initialized successfully")
            return True
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to initialize schema: {e}")
            return False

    @log_function_call
    def migrate_database(self) -> bool:
        """Migrate existing database to new schema."""
        try:
            # Import migration script functionality
            import os
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            
            from migrate_database import migrate_database as migrate_db
            
            success = migrate_db(self.db_path)
            if success:
                log_universal('INFO', 'Database', "Database migration completed successfully")
            else:
                log_universal('ERROR', 'Database', "Database migration failed")
            return success
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to migrate database: {e}")
            return False

    @log_function_call
    def create_backup(self) -> str:
        """Create database backup."""
        try:
            import shutil
            from datetime import datetime
            
            backup_path = f"{self.db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.db_path, backup_path)
            
            log_universal('INFO', 'Database', f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to create backup: {e}")
            raise

    @log_function_call
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore database from backup."""
        try:
            import shutil
            
            if not os.path.exists(backup_path):
                log_universal('ERROR', 'Database', f"Backup file not found: {backup_path}")
                return False
            
            # Create backup of current database
            current_backup = f"{self.db_path}.before_restore.{int(time.time())}"
            shutil.copy2(self.db_path, current_backup)
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            log_universal('INFO', 'Database', f"Database restored from: {backup_path}")
            log_universal('INFO', 'Database', f"Previous version backed up to: {current_backup}")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to restore from backup: {e}")
            return False

    @log_function_call
    def check_integrity(self) -> bool:
        """Check database integrity."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                
                if result and result[0] == 'ok':
                    log_universal('INFO', 'Database', "Database integrity check passed")
                    return True
                else:
                    log_universal('ERROR', 'Database', f"Database integrity check failed: {result}")
                    return False
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to check integrity: {e}")
            return False

    @log_function_call
    def vacuum_database(self) -> bool:
        """Vacuum database to reclaim space."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("VACUUM")
                
                log_universal('INFO', 'Database', "Database vacuumed successfully")
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to vacuum database: {e}")
            return False

    @log_function_call
    def get_database_size(self) -> Dict[str, Any]:
        """Get database size information."""
        try:
            if not os.path.exists(self.db_path):
                return {'size_bytes': 0, 'size_mb': 0, 'exists': False}
            
            size_bytes = os.path.getsize(self.db_path)
            size_mb = size_bytes / (1024 * 1024)
            
            return {
                'size_bytes': size_bytes,
                'size_mb': size_mb,
                'exists': True
            }
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to get database size: {e}")
            return {'size_bytes': 0, 'size_mb': 0, 'exists': False}

    @log_function_call
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data from database."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cleanup_stats = {}
                
                # Clean up old cache entries
                cursor.execute("""
                    DELETE FROM cache 
                    WHERE expires_at IS NOT NULL 
                    AND expires_at < datetime('now', '-{} days')
                """.format(days_to_keep))
                cleanup_stats['cache_entries'] = cursor.rowcount
                
                # Clean up old failed analysis entries
                cursor.execute("""
                    DELETE FROM analysis_cache 
                    WHERE last_retry_date IS NOT NULL 
                    AND last_retry_date < datetime('now', '-{} days')
                """.format(days_to_keep))
                cleanup_stats['failed_analysis'] = cursor.rowcount
                
                # Clean up old statistics
                cursor.execute("""
                    DELETE FROM statistics 
                    WHERE date_recorded < datetime('now', '-{} days')
                """.format(days_to_keep))
                cleanup_stats['statistics'] = cursor.rowcount
                
                conn.commit()
                
                log_universal('INFO', 'Database', f"Cleanup completed: {cleanup_stats}")
                return cleanup_stats
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Failed to cleanup old data: {e}")
            return {}


def get_db_manager() -> 'DatabaseManager':
    """Get database manager instance."""
    return DatabaseManager()
