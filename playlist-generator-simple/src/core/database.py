import sqlite3
import json
import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import time
import sys
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
import hashlib

# Import configuration and logging
from .config_loader import config_loader
from .logging_setup import get_logger, log_function_call, log_universal

logger = get_logger('playlista.database')


class DatabaseManager:
    """
    Fully schema-aligned database manager for Playlist Generator.
    Restores original environment detection, configuration wiring,
    schema initialization, and reconnects utility functions.
    """

    def __init__(self, db_path: str = None, config: Dict[str, Any] = None):
        if db_path is None:
            # Use local dev or PyInstaller path
            if os.path.exists('/app/cache'):
                db_path = '/app/cache/playlista.db'
            else:
                if hasattr(sys, '_MEIPASS'):
                    base_path = Path(sys._MEIPASS)
                else:
                    base_path = Path(__file__).resolve().parents[2]
                db_path = str(base_path / 'cache' / 'playlista.db')

        self.db_path = db_path
        self.config = config or config_loader.get_database_config()

        # Configuration settings
        self.cache_default_expiry_hours = self.config.get('DB_CACHE_DEFAULT_EXPIRY_HOURS', 24)
        self.connection_timeout_seconds = self.config.get('DB_CONNECTION_TIMEOUT_SECONDS', 30)
        self.max_retry_attempts = self.config.get('DB_MAX_RETRY_ATTEMPTS', 3)
        self.batch_size = self.config.get('DB_BATCH_SIZE', 100)

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        log_universal('INFO', 'Database', f"Initialized DatabaseManager with path: {self.db_path}")

        # Initialize schema if necessary
        self._init_database()

    @contextmanager
    def _get_db_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=self.connection_timeout_seconds)
            conn.row_factory = sqlite3.Row
            
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA foreign_keys=ON")
            
            yield conn
        except sqlite3.OperationalError as e:
            if conn: 
                try:
                    conn.rollback()
                except:
                    pass
            log_universal('ERROR', 'Database', f"Database operational error: {e}")
            raise
        except sqlite3.IntegrityError as e:
            if conn: 
                try:
                    conn.rollback()
                except:
                    pass
            log_universal('ERROR', 'Database', f"Database integrity error: {e}")
            raise
        except Exception as e:
            if conn: 
                try:
                    conn.rollback()
                except:
                    pass
            log_universal('ERROR', 'Database', f"Connection error: {e}")
            raise
        finally:
            if conn: 
                try:
                    conn.close()
                except Exception as e:
                    log_universal('WARNING', 'Database', f"Error closing connection: {e}")

    def _init_database(self):
        """Initialize schema from SQL file if needed."""
        log_universal('INFO', 'Database', "Checking database schema...")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'")
                if cursor.fetchone():
                    log_universal('INFO', 'Database', "Schema already initialized.")
                    return

                # Schema path (Docker internal only)
                schema_path = '/app/database/database_schema.sql'

                schema_sql = None
                schema_path_used = None
                
                if os.path.exists(schema_path):
                    try:
                        with open(schema_path, 'r', encoding='utf-8') as f:
                            schema_sql = f.read()
                        schema_path_used = schema_path
                        log_universal('INFO', 'Database', f"Loaded schema from: {schema_path}")
                    except Exception as e:
                        log_universal('ERROR', 'Database', f"Failed to read schema from {schema_path}: {e}")
                        raise
                else:
                    error_msg = f"Schema file not found at: {schema_path}"
                    log_universal('ERROR', 'Database', error_msg)
                    raise FileNotFoundError(error_msg)

                if schema_sql:
                    try:
                        cursor.executescript(schema_sql)
                        
                        # Enable WAL mode for better performance
                        cursor.execute("PRAGMA journal_mode=WAL")
                        cursor.execute("PRAGMA synchronous=NORMAL")
                        cursor.execute("PRAGMA cache_size=10000")
                        cursor.execute("PRAGMA temp_store=MEMORY")
                        cursor.execute("PRAGMA foreign_keys=ON")
                        
                        log_universal('INFO', 'Database', f"Schema initialized successfully from {schema_path_used}")
                    except Exception as e:
                        log_universal('ERROR', 'Database', f"Failed to execute schema: {e}")
                        raise
                else:
                    error_msg = f"No valid schema file found in paths: {schema_path}"
                    log_universal('ERROR', 'Database', error_msg)
                    raise FileNotFoundError(error_msg)

        except Exception as e:
            log_universal('ERROR', 'Database', f"Schema initialization failed: {e}")
            raise

    @log_function_call
    def initialize_schema(self) -> bool:
        """Public method to initialize schema."""
        try:
            self._init_database()
            return True
        except Exception as e:
            log_universal('ERROR', 'Database', f"Manual schema init failed: {e}")
            return False

    @log_function_call
    def migrate_database(self) -> bool:
        """Run external migration script."""
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from migrate_database import migrate_database as migrate_db
            success = migrate_db(self.db_path)
            if success:
                log_universal('INFO', 'Database', "Migration completed.")
            else:
                log_universal('ERROR', 'Database', "Migration failed.")
            return success
        except Exception as e:
            log_universal('ERROR', 'Database', f"Migration error: {e}")
            return False

    # Save analysis result with optimized schema
    @log_function_call
    def save_analysis_result(self, file_path: str, filename: str, file_size_bytes: int,
                             file_hash: str, analysis_data: Dict[str, Any],
                             metadata: Dict[str, Any] = None, discovery_source: str = 'file_system') -> bool:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now()

                get = lambda d, k, default=None: d.get(k, default) if d else default
                j = lambda k: json.dumps(get(analysis_data, k, {}))
                jd = lambda k: json.dumps(get(analysis_data, k, {}))

                # Extract core metadata
                title = get(metadata, 'title', filename)
                artist = get(metadata, 'artist', 'Unknown')
                album = get(metadata, 'album')
                track_number = get(metadata, 'track_number')
                genre = get(metadata, 'genre')
                year = get(metadata, 'year')
                duration = get(analysis_data, 'duration')

                # Extract audio properties
                bitrate = get(metadata, 'bitrate')
                sample_rate = get(metadata, 'sample_rate')
                channels = get(metadata, 'channels')

                # Extract additional metadata
                composer = get(metadata, 'composer')
                mood = get(metadata, 'mood')
                style = get(metadata, 'style')

                # Extract essential audio features
                bpm = get(analysis_data, 'bpm')
                key = get(analysis_data, 'key')
                mode = get(analysis_data, 'mode')
                loudness = get(analysis_data, 'loudness')
                energy = get(analysis_data, 'energy')
                danceability = get(analysis_data, 'danceability')
                valence = get(analysis_data, 'valence')
                acousticness = get(analysis_data, 'acousticness')
                instrumentalness = get(analysis_data, 'instrumentalness')

                # Extract rhythm features (Essentia)
                rhythm_confidence = get(analysis_data, 'rhythm_confidence')
                bpm_estimates = j('bpm_estimates')
                bpm_intervals = j('bpm_intervals')
                external_bpm = get(analysis_data, 'external_bpm')

                # Extract spectral features (Essentia)
                spectral_centroid = get(analysis_data, 'spectral_centroid')
                spectral_flatness = get(analysis_data, 'spectral_flatness')
                spectral_rolloff = get(analysis_data, 'spectral_rolloff')
                spectral_bandwidth = get(analysis_data, 'spectral_bandwidth')
                spectral_contrast_mean = get(analysis_data, 'spectral_contrast_mean')
                spectral_contrast_std = get(analysis_data, 'spectral_contrast_std')

                # Extract loudness features (Essentia)
                dynamic_complexity = get(analysis_data, 'dynamic_complexity')
                loudness_range = get(analysis_data, 'loudness_range')
                dynamic_range = get(analysis_data, 'dynamic_range')

                # Extract key features (Essentia)
                scale = get(analysis_data, 'scale')
                key_strength = get(analysis_data, 'key_strength')
                key_confidence = get(analysis_data, 'key_confidence')

                # Extract MFCC features (Essentia)
                mfcc_coefficients = j('mfcc_coefficients')
                mfcc_bands = j('mfcc_bands')
                mfcc_std = j('mfcc_std')
                mfcc_delta = j('mfcc_delta')
                mfcc_delta2 = j('mfcc_delta2')

                # Extract MusiCNN features
                embedding = jd('embedding')
                embedding_std = jd('embedding_std')
                embedding_min = jd('embedding_min')
                embedding_max = jd('embedding_max')
                tags = jd('tags')
                musicnn_skipped = get(analysis_data, 'musicnn_skipped', 0)

                # Extract chroma features (Essentia)
                chroma_mean = j('chroma_mean')
                chroma_std = j('chroma_std')

                # Extract extended features as JSON
                rhythm_features = j('rhythm_features')
                spectral_features = j('spectral_features')
                mfcc_features = j('mfcc_features')
                musicnn_features = jd('musicnn_features')
                spotify_features = jd('spotify_features')

                # Extract analysis metadata
                analysis_type = get(analysis_data, 'analysis_type', 'full')
                long_audio_category = get(analysis_data, 'long_audio_category')

                cursor.execute("""
                    INSERT OR REPLACE INTO tracks (
                        file_path, file_hash, filename, file_size_bytes, analysis_date, created_at, updated_at, status, analysis_status, retry_count, error_message, title, artist, album, track_number, genre, year, duration, bitrate, sample_rate, channels, bpm, key, mode, loudness, energy, danceability, valence, acousticness, instrumentalness, rhythm_confidence, bpm_estimates, bpm_intervals, external_bpm, spectral_centroid, spectral_flatness, spectral_rolloff, spectral_bandwidth, spectral_contrast_mean, spectral_contrast_std, dynamic_complexity, loudness_range, dynamic_range, scale, key_strength, key_confidence, composer, mood, style, analysis_type, long_audio_category, mfcc_coefficients, mfcc_bands, mfcc_std, mfcc_delta, mfcc_delta2, embedding, embedding_std, embedding_min, embedding_max, tags, musicnn_skipped, chroma_mean, chroma_std, rhythm_features, spectral_features, mfcc_features, musicnn_features, spotify_features
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_path, file_hash, filename, file_size_bytes, now, now, now, 'analyzed', 'completed', 0, None, title, artist, album, track_number, genre, year, duration, bitrate, sample_rate, channels, bpm, key, mode, loudness, energy, danceability, valence, acousticness, instrumentalness, rhythm_confidence, bpm_estimates, bpm_intervals, external_bpm, spectral_centroid, spectral_flatness, spectral_rolloff, spectral_bandwidth, spectral_contrast_mean, spectral_contrast_std, dynamic_complexity, loudness_range, dynamic_range, scale, key_strength, key_confidence, composer, mood, style, analysis_type, long_audio_category, mfcc_coefficients, mfcc_bands, mfcc_std, mfcc_delta, mfcc_delta2, embedding, embedding_std, embedding_min, embedding_max, tags, musicnn_skipped, chroma_mean, chroma_std, rhythm_features, spectral_features, mfcc_features, musicnn_features, spotify_features
                ))

                track_id = cursor.lastrowid

                # Save tags if provided
                if metadata and 'tags' in metadata:
                    self._save_tags(cursor, track_id, metadata['tags'])

                conn.commit()
                log_universal('DEBUG', 'Database', f'Saved analysis result for: {file_path}')
                return True

        except Exception as e:
            log_universal('ERROR', 'Database', f"Save analysis failed: {e}")
            return False

    # 🌍 Save discovery result
    @log_function_call
    def save_discovery_result(self, directory_path: str, file_count: int, scan_duration: float,
                              status: str = 'completed', error_message: str = None) -> bool:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO discovery_cache (directory_path, file_count, scan_duration, status, error_message, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (directory_path, file_count, scan_duration, status, error_message, datetime.now()))
                conn.commit()
                return True
        except Exception as e:
            log_universal('ERROR', 'Database', f"Discovery save failed: {e}")
            return False

    # 👀 Get discovery status
    @log_function_call
    def get_discovery_status(self, directory_path: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM discovery_cache WHERE directory_path = ?", (directory_path,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            log_universal('ERROR', 'Database', f"Discovery fetch failed: {e}")
            return None

    # 🚨 Mark failed analysis
    @log_function_call
    def mark_analysis_failed(self, file_path: str, filename: str, error_message: str, file_hash: str = None) -> bool:
        """
        Mark an analysis as failed in the cache table.
        
        Args:
            file_path: Path to the failed file
            filename: Name of the file
            error_message: Error message
            file_hash: File hash (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if this file already has a failed analysis entry
                cursor.execute("""
                    SELECT cache_value FROM cache 
                    WHERE cache_key = ? AND cache_type = 'failed_analysis'
                """, (f"failed_analysis:{file_path}",))
                
                existing_entry = cursor.fetchone()
                retry_count = 0
                
                if existing_entry:
                    try:
                        existing_data = json.loads(existing_entry[0])
                        retry_count = existing_data.get('retry_count', 0) + 1
                        log_universal('DEBUG', 'Database', f'Incrementing retry count for {filename}: {retry_count}')
                    except json.JSONDecodeError:
                        retry_count = 1
                        log_universal('WARNING', 'Database', f'Invalid JSON in existing failed analysis entry for {filename}')
                else:
                    retry_count = 0
                    log_universal('DEBUG', 'Database', f'First failure for {filename}')
                
                # Use cache table for failed analysis with cache_type='failed_analysis'
                cache_key = f"failed_analysis:{file_path}"
                cache_value = json.dumps({
                    'file_path': file_path,
                    'filename': filename,
                    'file_hash': file_hash,
                    'error_message': error_message,
                    'retry_count': retry_count,
                    'last_retry_date': datetime.now().isoformat(),
                    'status': 'failed'
                })
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache 
                    (cache_key, cache_value, cache_type, created_at)
                    VALUES (?, ?, 'failed_analysis', CURRENT_TIMESTAMP)
                """, (cache_key, cache_value))
                
                conn.commit()
                log_universal('INFO', 'Database', f'Marked analysis as failed: {filename} - Error: {error_message}')
                log_universal('DEBUG', 'Database', f'Stored failed analysis in cache table with key: failed_analysis:{file_path}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to mark analysis as failed: {e}')
            return False

    def _calculate_file_hash_for_failed(self, file_path: str) -> str:
        """
        Calculate file hash for failed files, handling cases where file doesn't exist.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File hash string or "unknown" if file doesn't exist
        """
        try:
            if not os.path.exists(file_path):
                # Use filename + timestamp for non-existent files
                filename = os.path.basename(file_path)
                import hashlib
                content = f"{filename}:{datetime.now().isoformat()}"
                return hashlib.md5(content.encode()).hexdigest()
            
            # Use the same hash calculation as the audio analyzer
            import hashlib
            stat = os.stat(file_path)
            filename = os.path.basename(file_path)
            content = f"{filename}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(content.encode()).hexdigest()
            
        except Exception as e:
            log_universal('WARNING', 'Database', f"Could not calculate hash for {file_path}: {e}")
            # Return a hash based on filename and timestamp
            filename = os.path.basename(file_path)
            import hashlib
            content = f"{filename}:{datetime.now().isoformat()}:error"
            return hashlib.md5(content.encode()).hexdigest()

    # 📁 Move failed file to failed directory
    @log_function_call
    def move_failed_file_to_directory(self, file_path: str, failed_dir: str = None) -> bool:
        """
        Move a failed file to the failed directory.
        
        Args:
            file_path: Path to the failed file
            failed_dir: Directory to move failed files to (defaults to /app/cache/failed_dir)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if failed_dir is None:
                failed_dir = '/app/cache/failed_dir'
            
            # Create failed directory if it doesn't exist
            os.makedirs(failed_dir, exist_ok=True)
            
            if not os.path.exists(file_path):
                log_universal('WARNING', 'Database', f"File not found for moving: {file_path}")
                return False
            
            filename = os.path.basename(file_path)
            failed_path = os.path.join(failed_dir, filename)
            
            # Handle duplicate filenames
            counter = 1
            original_failed_path = failed_path
            while os.path.exists(failed_path):
                name, ext = os.path.splitext(original_failed_path)
                failed_path = f"{name}_{counter}{ext}"
                counter += 1
            
            # Move the file
            import shutil
            shutil.move(file_path, failed_path)
            log_universal('INFO', 'Database', f"Moved failed file: {file_path} -> {failed_path}")
            
            # Update database to reflect the move
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get existing cache data
                cursor.execute("SELECT cache_value FROM cache WHERE cache_key = ?", (f"failed_analysis:{file_path}",))
                row = cursor.fetchone()
                
                if row:
                    try:
                        cache_data = json.loads(row[0])
                        cache_data['file_path'] = failed_path
                        cache_data['error_message'] = f"Moved to failed directory: {failed_path}"
                        
                        # Update cache with new path
                        cursor.execute("""
                            UPDATE cache 
                            SET cache_value = ? 
                            WHERE cache_key = ?
                        """, (json.dumps(cache_data), f"failed_analysis:{file_path}"))
                        conn.commit()
                        
                    except json.JSONDecodeError:
                        log_universal('WARNING', 'Database', f'Invalid JSON in cache for failed analysis')
                        # Create new cache entry
                        cache_data = {
                            'file_path': failed_path,
                            'filename': os.path.basename(file_path),
                            'error_message': f"Moved to failed directory: {failed_path}",
                            'retry_count': 0,
                            'last_retry_date': datetime.now().isoformat(),
                            'status': 'failed'
                        }
                        cursor.execute("""
                            INSERT OR REPLACE INTO cache 
                            (cache_key, cache_value, cache_type, created_at)
                            VALUES (?, ?, 'failed_analysis', CURRENT_TIMESTAMP)
                        """, (f"failed_analysis:{file_path}", json.dumps(cache_data)))
                        conn.commit()
            
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error moving failed file {file_path}: {e}")
            return False

    # 🧹 Clean up failed files by moving them to failed directory
    @log_function_call
    def cleanup_failed_files(self, failed_dir: str = None, max_retries: int = 3) -> Dict[str, int]:
        """
        Clean up failed files by moving them to a failed directory.
        
        Args:
            failed_dir: Directory to move failed files to
            max_retries: Maximum number of retries before moving to failed directory
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            failed_files = self.get_failed_analysis_files(max_retries=max_retries)
            
            moved_count = 0
            skipped_count = 0
            error_count = 0
            
            for failed_file in failed_files:
                file_path = failed_file['file_path']
                
                # Skip if file doesn't exist
                if not os.path.exists(file_path):
                    skipped_count += 1
                    continue
                
                # Move to failed directory
                if self.move_failed_file_to_directory(file_path, failed_dir):
                    moved_count += 1
                else:
                    error_count += 1
            
            log_universal('INFO', 'Database', f"Failed files cleanup: {moved_count} moved, {skipped_count} skipped, {error_count} errors")
            
            return {
                'moved_count': moved_count,
                'skipped_count': skipped_count,
                'error_count': error_count,
                'total_processed': len(failed_files)
            }
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Error during failed files cleanup: {e}")
            return {'error': str(e)}

    # 🧾 Get failed analysis files
    @log_function_call
    def get_failed_analysis_files(self, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Get list of failed analysis files from cache table.
        
        Args:
            max_retries: Maximum retry count to include
            
        Returns:
            List of failed analysis file dictionaries
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT cache_value
                    FROM cache
                    WHERE cache_type = 'failed_analysis'
                    ORDER BY created_at DESC
                """)
                
                failed_files = []
                cursor.execute("""
                    SELECT cache_value
                    FROM cache
                    WHERE cache_type = 'failed_analysis'
                    ORDER BY created_at DESC
                """)
                rows = cursor.fetchall()
                log_universal('DEBUG', 'Database', f'Found {len(rows)} failed analysis entries in cache table')
                for row in rows:
                    try:
                        cache_data = json.loads(row[0])
                        retry_count = cache_data.get('retry_count', 0)
                        
                        if retry_count < max_retries:
                            failed_files.append({
                                'file_path': cache_data.get('file_path'),
                                'filename': cache_data.get('filename'),
                                'file_hash': cache_data.get('file_hash'),
                                'error_message': cache_data.get('error_message'),
                                'retry_count': retry_count,
                                'last_retry_date': cache_data.get('last_retry_date'),
                                'status': cache_data.get('status', 'failed')
                            })
                            log_universal('DEBUG', 'Database', f'Retrieved failed file: {cache_data.get("filename")} (retry count: {retry_count})')
                    except json.JSONDecodeError:
                        log_universal('WARNING', 'Database', f'Invalid JSON in cache for failed analysis')
                        continue
                
                return failed_files
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Fetch failed analysis error: {e}')
            return []

    # 📀 Save playlist and track positions
    @log_function_call
    def save_playlist(self, name: str, tracks: List[str], description: str = None,
                      generation_method: str = 'manual', generation_params: Dict[str, Any] = None) -> bool:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now()
                params_json = json.dumps(generation_params) if generation_params else None

                cursor.execute("""
                    INSERT INTO playlists (name, description, generation_method, generation_params, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (name, description, generation_method, params_json, now, now))
                playlist_id = cursor.lastrowid

                for position, file_path in enumerate(tracks, 1):
                    cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                    row = cursor.fetchone()
                    if row:
                        cursor.execute("""
                            INSERT INTO playlist_tracks (playlist_id, track_id, position)
                            VALUES (?, ?, ?)
                        """, (playlist_id, row['id'], position))

                conn.commit()
                return True
        except Exception as e:
            log_universal('ERROR', 'Database', f"Save playlist failed: {e}")
            return False

    # 🎵 Get a playlist by name
    @log_function_call
    def get_playlist(self, name: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM playlists WHERE name = ?", (name,))
                playlist = cursor.fetchone()
                if not playlist:
                    return None

                cursor.execute("""
                    SELECT t.*, pt.position
                    FROM playlist_tracks pt
                    JOIN tracks t ON pt.track_id = t.id
                    WHERE pt.playlist_id = ?
                    ORDER BY pt.position
                """, (playlist['id'],))
                track_list = [dict(row) for row in cursor.fetchall()]

                return {
                    'id': playlist['id'],
                    'name': playlist['name'],
                    'description': playlist['description'],
                    'generation_method': playlist['generation_method'],
                    'generation_params': json.loads(playlist['generation_params']) if playlist['generation_params'] else None,
                    'created_at': playlist['created_at'],
                    'updated_at': playlist['updated_at'],
                    'tracks': track_list
                }
        except Exception as e:
            log_universal('ERROR', 'Database', f"Get playlist failed: {e}")
            return None

    # 📑 Save tags for a track
    def _save_tags(self, cursor, track_id: int, tags: Dict[str, Any]):
        if isinstance(tags, dict):
            for source, tag_data in tags.items():
                if isinstance(tag_data, dict):
                    for tag_name, tag_value in tag_data.items():
                        cursor.execute("""
                            INSERT INTO tags (track_id, source, tag_name, tag_value)
                            VALUES (?, ?, ?, ?)
                        """, (track_id, source, tag_name, str(tag_value)))
                elif isinstance(tag_data, list):
                    for tag_name in tag_data:
                        cursor.execute("""
                            INSERT INTO tags (track_id, source, tag_name)
                            VALUES (?, ?, ?)
                        """, (track_id, source, tag_name))

    # 🗃️ Save data to cache
    @log_function_call
    def save_cache(self, key: str, value: Any, cache_type: str = 'general',
                   expires_hours: int = None) -> bool:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now()
                expires_at = now + timedelta(hours=expires_hours) if expires_hours else None
                cursor.execute("""
                    INSERT OR REPLACE INTO cache (cache_key, cache_value, cache_type, expires_at, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (key, json.dumps(value), cache_type, expires_at, now))
                conn.commit()
                return True
        except Exception as e:
            log_universal('ERROR', 'Database', f"Save cache failed: {e}")
            return False

    # 🔍 Get cache entry by key
    @log_function_call
    def get_cache(self, key: str) -> Optional[Any]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cache_value, expires_at FROM cache WHERE cache_key = ?", (key,))
                row = cursor.fetchone()
                if not row:
                    return None

                if row['expires_at'] and datetime.fromisoformat(row['expires_at']) < datetime.now():
                    cursor.execute("DELETE FROM cache WHERE cache_key = ?", (key,))
                    conn.commit()
                    return None

                return json.loads(row['cache_value'])
        except Exception as e:
            log_universal('ERROR', 'Database', f"Get cache failed: {e}")
            return None

    # 🧩 Get cache entries by type
    @log_function_call
    def get_cache_by_type(self, cache_type: str) -> List[Dict[str, Any]]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM cache WHERE cache_type = ? ORDER BY created_at DESC", (cache_type,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            log_universal('ERROR', 'Database', f"Cache type fetch failed: {e}")
            return []

    # 📊 Save a statistic entry
    @log_function_call
    def save_statistic(self, category: str, metric_name: str, metric_value: float,
                       metric_data: Dict[str, Any] = None) -> bool:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                data_json = json.dumps(metric_data) if metric_data else None
                cursor.execute("""
                    INSERT INTO statistics (category, metric_name, metric_value, metric_data, date_recorded)
                    VALUES (?, ?, ?, ?, ?)
                """, (category, metric_name, metric_value, data_json, datetime.now()))
                conn.commit()
                return True
        except Exception as e:
            log_universal('ERROR', 'Database', f"Save statistic failed: {e}")
            return False

    # 📈 Get statistics summary view
    @log_function_call
    def get_statistics_summary(self) -> Dict[str, Any]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM statistics_summary")
                rows = cursor.fetchall()
                summary = {}
                for row in rows:
                    summary.setdefault(row['category'], []).append(dict(row))
                return summary
        except Exception:
            return {}

    # 🎛️ Get tracks for UI display
    @log_function_call
    def get_tracks_for_web_ui(self, limit: int = 50, offset: int = 0,
                               artist: str = None, genre: str = None,
                               year: int = None, sort_by: str = 'analysis_date') -> List[Dict[str, Any]]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
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

                sort_fields = ['analysis_date', 'title', 'artist', 'album', 'year', 'bpm']
                query += f" ORDER BY {sort_by if sort_by in sort_fields else 'analysis_date'} DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception:
            return []

    # 🧾 Get all analysis results
    @log_function_call
    def get_all_analysis_results(self) -> List[Dict[str, Any]]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, file_path, filename, file_size_bytes, file_hash, 
                           analysis_date, analysis_type, long_audio_category,
                           title, artist, album, genre, year, duration,
                           bpm, key, mode, loudness, danceability, energy
                    FROM tracks ORDER BY analysis_date DESC
                """)
                return [dict(row) for row in cursor.fetchall()]
        except Exception:
            return []

    # 📁 Get one track analysis result
    @log_function_call
    def get_analysis_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tracks WHERE file_path = ?
                """, (file_path,))
                row = cursor.fetchone()
                if not row:
                    return None

                result = dict(row)
                cursor.execute("""
                    SELECT source, tag_name, tag_value FROM tags WHERE track_id = ?
                """, (row['id'],))
                tags = {}
                for tag_row in cursor.fetchall():
                    source = tag_row['source']
                    tags.setdefault(source, {})[tag_row['tag_name']] = tag_row['tag_value']
                result['tags'] = tags
                return result
        except Exception:
            return None

    # 🧹 Delete analysis result (track + tags + playlist links)
    @log_function_call
    def delete_analysis_result(self, file_path: str) -> bool:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                row = cursor.fetchone()
                if not row:
                    return True

                track_id = row['id']
                cursor.execute("DELETE FROM tags WHERE track_id = ?", (track_id,))
                cursor.execute("DELETE FROM playlist_tracks WHERE track_id = ?", (track_id,))
                cursor.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
                conn.commit()
                return True
        except Exception:
            return False

    # 🗑️ Delete failed analysis entry
    @log_function_call
    def delete_failed_analysis(self, file_path: str) -> bool:
        """
        Delete a failed analysis entry from cache table.
        
        Args:
            file_path: Path to the file to remove from failed analysis
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE cache_key = ?", (f"failed_analysis:{file_path}",))
                conn.commit()
                return True
        except Exception as e:
            log_universal('ERROR', 'Database', f'Delete failed analysis error: {e}')
            return False

    # 🔄 Increment failed analysis retry count
    @log_function_call
    def increment_failed_analysis_retry(self, file_path: str) -> bool:
        """
        Increment the retry count for a failed analysis entry.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get current retry count
                cursor.execute("""
                    SELECT cache_value FROM cache 
                    WHERE cache_key = ? AND cache_type = 'failed_analysis'
                """, (f"failed_analysis:{file_path}",))
                
                existing_entry = cursor.fetchone()
                if not existing_entry:
                    log_universal('WARNING', 'Database', f'No failed analysis entry found for: {file_path}')
                    return False
                
                try:
                    existing_data = json.loads(existing_entry[0])
                    current_retry_count = existing_data.get('retry_count', 0)
                    new_retry_count = current_retry_count + 1
                    
                    # Update the retry count
                    existing_data['retry_count'] = new_retry_count
                    existing_data['last_retry_date'] = datetime.now().isoformat()
                    
                    cache_value = json.dumps(existing_data)
                    
                    cursor.execute("""
                        UPDATE cache 
                        SET cache_value = ?, created_at = CURRENT_TIMESTAMP
                        WHERE cache_key = ? AND cache_type = 'failed_analysis'
                    """, (cache_value, f"failed_analysis:{file_path}"))
                    
                    conn.commit()
                    log_universal('INFO', 'Database', f'Incremented retry count for {file_path}: {new_retry_count}')
                    return True
                    
                except json.JSONDecodeError:
                    log_universal('ERROR', 'Database', f'Invalid JSON in failed analysis entry for: {file_path}')
                    return False
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to increment retry count: {e}')
            return False

    # 🧮 Database statistics snapshot
    @log_function_call
    def get_database_statistics(self) -> Dict[str, Any]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                stats = {}
                queries = {
                    'total_tracks': "SELECT COUNT(*) FROM tracks",
                    'playlists': "SELECT COUNT(*) FROM playlists",
                    'failed_analyses': "SELECT COUNT(*) FROM cache WHERE cache_type = 'failed_analysis'",
                    'discovery_entries': "SELECT COUNT(*) FROM cache WHERE cache_type = 'discovery'",
                    'cache_entries': "SELECT COUNT(*) FROM cache",
                    'statistics_entries': "SELECT COUNT(*) FROM statistics"
                }
                for key, q in queries.items():
                    cursor.execute(q)
                    stats[key] = cursor.fetchone()[0]
                return stats
        except Exception:
            return {}

    # 🔍 Check DB integrity
    @log_function_call
    def check_integrity(self) -> bool:
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
            log_universal('ERROR', 'Database', f"Integrity check error: {e}")
            return False

    # 🔧 Repair database issues
    @log_function_call
    def repair_database(self) -> Dict[str, Any]:
        """
        Repair common database issues.
        
        Returns:
            Dictionary with repair results
        """
        repair_results = {
            'integrity_fixed': False,
            'indexes_rebuilt': False,
            'cache_cleaned': False,
            'statistics_updated': False,
            'errors': []
        }
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check and fix integrity
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()
                
                if integrity_result and integrity_result[0] != 'ok':
                    log_universal('WARNING', 'Database', f"Database integrity issues detected: {integrity_result}")
                    
                    # Try to fix integrity issues
                    cursor.execute("PRAGMA integrity_check")
                    cursor.execute("PRAGMA optimize")
                    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    
                    # Re-check integrity
                    cursor.execute("PRAGMA integrity_check")
                    new_integrity = cursor.fetchone()
                    if new_integrity and new_integrity[0] == 'ok':
                        repair_results['integrity_fixed'] = True
                        log_universal('INFO', 'Database', "Database integrity issues fixed")
                    else:
                        repair_results['errors'].append(f"Could not fix integrity issues: {new_integrity}")
                
                # Rebuild indexes
                try:
                    cursor.execute("REINDEX")
                    repair_results['indexes_rebuilt'] = True
                    log_universal('INFO', 'Database', "Database indexes rebuilt")
                except Exception as e:
                    repair_results['errors'].append(f"Failed to rebuild indexes: {e}")
                
                # Clean up expired cache entries
                try:
                    cursor.execute("""
                        DELETE FROM cache 
                        WHERE expires_at IS NOT NULL 
                        AND expires_at < datetime('now')
                    """)
                    cache_cleaned = cursor.rowcount
                    if cache_cleaned > 0:
                        repair_results['cache_cleaned'] = True
                        log_universal('INFO', 'Database', f"Cleaned {cache_cleaned} expired cache entries")
                except Exception as e:
                    repair_results['errors'].append(f"Failed to clean cache: {e}")
                
                # Update statistics
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO statistics (category, metric_name, metric_value, date_recorded)
                        VALUES 
                        ('system', 'database_size_mb', ?, datetime('now')),
                        ('system', 'total_tracks', (SELECT COUNT(*) FROM tracks), datetime('now')),
                        ('system', 'total_playlists', (SELECT COUNT(*) FROM playlists), datetime('now')),
                        ('system', 'cache_entries', (SELECT COUNT(*) FROM cache), datetime('now'))
                    """, (self.get_database_size().get('size_mb', 0),))
                    
                    repair_results['statistics_updated'] = True
                    log_universal('INFO', 'Database', "Database statistics updated")
                except Exception as e:
                    repair_results['errors'].append(f"Failed to update statistics: {e}")
                
                conn.commit()
                
                if repair_results['errors']:
                    log_universal('WARNING', 'Database', f"Database repair completed with {len(repair_results['errors'])} errors")
                else:
                    log_universal('INFO', 'Database', "Database repair completed successfully")
                
                return repair_results
                
        except Exception as e:
            repair_results['errors'].append(f"Database repair failed: {e}")
            log_universal('ERROR', 'Database', f"Database repair failed: {e}")
            return repair_results

    # 🧼 Vacuum
    @log_function_call
    def vacuum_database(self) -> bool:
        try:
            with self._get_db_connection() as conn:
                conn.execute("VACUUM")
                return True
        except Exception:
            return False

    # 🧽 Cleanup old entries
    @log_function_call
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cutoff = datetime.now() - timedelta(days=days_to_keep)

                cursor.execute("DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?", (cutoff,))
                cache_deleted = cursor.rowcount

                # Clean up old failed analysis entries
                cutoff = datetime.now() - timedelta(days=days_to_keep)
                cursor.execute("""
                    DELETE FROM cache 
                    WHERE cache_type = 'failed_analysis' 
                    AND created_at < ?
                """, (cutoff,))
                failed_deleted = cursor.rowcount

                cursor.execute("DELETE FROM statistics WHERE date_recorded < ?", (cutoff,))
                stats_deleted = cursor.rowcount

                conn.commit()
                return {
                    'cache_deleted': cache_deleted,
                    'failed_analysis_deleted': failed_deleted,
                    'statistics_deleted': stats_deleted
                }
        except Exception:
            return {}

    # 💾 Create backup
    @log_function_call
    def create_backup(self) -> str:
        try:
            backup_path = f"{self.db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.db_path, backup_path)
            return backup_path
        except Exception:
            return ""

    # 🧯 Restore backup
    @log_function_call
    def restore_from_backup(self, backup_path: str) -> bool:
        try:
            if not os.path.exists(backup_path):
                return False
            timestamp = int(time.time())
            current_backup = f"{self.db_path}.before_restore.{timestamp}"
            shutil.copy2(self.db_path, current_backup)
            shutil.copy2(backup_path, self.db_path)
            return True
        except Exception:
            return False

    # 📐 Get database size
    @log_function_call
    def get_database_size(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.db_path):
                return {'size_bytes': 0, 'size_mb': 0, 'exists': False}
            size_bytes = os.path.getsize(self.db_path)
            return {
                'size_bytes': size_bytes,
                'size_mb': round(size_bytes / (1024 * 1024), 2),
                'exists': True
            }
        except Exception:
            return {'size_bytes': 0, 'size_mb': 0, 'exists': False}

    @log_function_call
    def get_file_size_mb(self, file_path: str) -> float:
        """
        Get file size in MB from database.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in MB, or 0 if not found
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_size_bytes FROM tracks 
                    WHERE file_path = ?
                """, (file_path,))
                row = cursor.fetchone()
                
                if row and row[0]:
                    return row[0] / (1024 * 1024)
                else:
                    # Fallback to filesystem if not in database
                    if os.path.exists(file_path):
                        return os.path.getsize(file_path) / (1024 * 1024)
                    return 0.0
                    
        except Exception as e:
            log_universal('WARNING', 'Database', f"Error getting file size for {file_path}: {e}")
            # Fallback to filesystem
            try:
                if os.path.exists(file_path):
                    return os.path.getsize(file_path) / (1024 * 1024)
            except Exception:
                pass
            return 0.0

    @log_function_call
    def save_metadata(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Save metadata to database.
        
        Args:
            file_path: Path to the file
            metadata: Metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate required fields
                filename = os.path.basename(file_path)
                file_size_bytes = os.path.getsize(file_path)
                file_hash = self._calculate_file_hash_for_failed(file_path)
                
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
                    elif '_-_' in filename_without_ext:
                        # Try underscore separator
                        parts = filename_without_ext.split('_-_', 1)
                        if len(parts) == 2:
                            if artist == 'Unknown':
                                artist = parts[0].strip()
                            if title == 'Unknown':
                                title = parts[1].strip()
                    elif '__' in filename_without_ext:
                        # Try double underscore separator
                        parts = filename_without_ext.split('__', 1)
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
                year = metadata.get('year') if metadata else None
                genre = metadata.get('genre') if metadata else None
                duration = metadata.get('duration')
                
                # Update existing record or create new one with all required fields
                cursor.execute("""
                    INSERT OR REPLACE INTO tracks 
                    (file_path, file_hash, filename, file_size_bytes, artist, title, album, year, genre, duration, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (file_path, file_hash, filename, file_size_bytes, artist, title, album, year, genre, duration))
                
                conn.commit()
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save metadata: {e}')
            return False

    @log_function_call
    def save_essentia_features(self, file_path: str, features: Dict[str, Any]) -> bool:
        """
        Save Essentia features to database.
        
        Args:
            file_path: Path to the file
            features: Essentia features dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Extract key features and update tracks table
                cursor.execute("""
                    UPDATE tracks 
                    SET bpm = ?, rhythm_confidence = ?, spectral_centroid = ?, 
                        spectral_flatness = ?, spectral_rolloff = ?, loudness = ?,
                        key = ?, scale = ?, danceability = ?, energy = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                """, (
                    features.get('bpm'),
                    features.get('rhythm_confidence'),
                    features.get('spectral_centroid'),
                    features.get('spectral_flatness'),
                    features.get('spectral_rolloff'),
                    features.get('loudness'),
                    features.get('key'),
                    features.get('scale'),
                    features.get('danceability'),
                    features.get('energy'),
                    file_path
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save Essentia features: {e}')
            return False

    @log_function_call
    def save_musicnn_features(self, file_path: str, features: Dict[str, Any]) -> bool:
        """
        Save MusicNN features to database.
        
        Args:
            file_path: Path to the file
            features: MusicNN features dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Convert features to JSON for storage
                embedding_json = json.dumps(features.get('embedding', []))
                tags_json = json.dumps(features.get('tags', {}))
                
                # Update tracks table with MusicNN features
                cursor.execute("""
                    UPDATE tracks 
                    SET embedding = ?, tags = ?, musicnn_skipped = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                """, (
                    embedding_json,
                    tags_json,
                    features.get('musicnn_skipped', 0),
                    file_path
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save MusicNN features: {e}')
            return False

    @log_function_call
    def save_advanced_categorization_features(self, file_path: str, features: Dict[str, Any]) -> bool:
        """
        Save advanced categorization features to database.
        
        Args:
            file_path: Path to the file
            features: Advanced categorization features dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Update the tracks table with advanced features
                cursor.execute("""
                    UPDATE tracks SET
                        danceability = ?,
                        energy = ?,
                        acousticness = ?,
                        instrumentalness = ?,
                        speechiness = ?,
                        valence = ?,
                        liveness = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                """, (
                    features.get('danceability', 0.0),
                    features.get('energy', 0.0),
                    features.get('acousticness', 0.0),
                    features.get('instrumentalness', 0.0),
                    features.get('speechiness', 0.0),
                    features.get('valence', 0.0),
                    features.get('liveness', 0.0),
                    file_path
                ))
                
                if cursor.rowcount > 0:
                    log_universal('DEBUG', 'Database', f'Saved advanced categorization features for {os.path.basename(file_path)}')
                    return True
                else:
                    log_universal('WARNING', 'Database', f'No track found for {os.path.basename(file_path)}')
                    return False
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save advanced categorization features: {e}')
            return False

    @log_function_call
    def save_spotify_features(self, file_path: str, features: Dict[str, Any]) -> bool:
        """
        Save Spotify-style features to database.
        
        Args:
            file_path: Path to the file
            features: Spotify-style features dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Update the tracks table with Spotify-style features
                cursor.execute("""
                    UPDATE tracks SET
                        danceability = ?,
                        energy = ?,
                        mode = ?,
                        acousticness = ?,
                        instrumentalness = ?,
                        speechiness = ?,
                        valence = ?,
                        liveness = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                """, (
                    features.get('danceability', 0.0),
                    features.get('energy', 0.0),
                    features.get('mode', 0.0),
                    features.get('acousticness', 0.0),
                    features.get('instrumentalness', 0.0),
                    features.get('speechiness', 0.0),
                    features.get('valence', 0.0),
                    features.get('liveness', 0.0),
                    file_path
                ))
                
                if cursor.rowcount > 0:
                    log_universal('DEBUG', 'Database', f'Saved Spotify-style features for {os.path.basename(file_path)}')
                    return True
                else:
                    log_universal('WARNING', 'Database', f'No track found for {os.path.basename(file_path)}')
                    return False
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save Spotify-style features: {e}')
            return False

    @log_function_call
    def commit_analysis_results(self, file_path: str, all_features: Dict[str, Any]) -> bool:
        """
        Commit all analysis results to database.
        
        Args:
            file_path: Path to the file
            all_features: Dictionary containing all analysis results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Mark analysis as completed
                cursor.execute("""
                    UPDATE tracks 
                    SET status = 'analyzed', analysis_status = 'completed', 
                        analysis_date = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                """, (file_path,))
                
                conn.commit()
                log_universal('INFO', 'Database', f'Analysis results committed for {os.path.basename(file_path)}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to commit analysis results: {e}')
            return False

    # 🔍 Validate database schema
    @log_function_call
    def validate_schema(self) -> Dict[str, Any]:
        """
        Validate database schema and report any issues.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'schema_valid': True,
            'missing_tables': [],
            'missing_columns': [],
            'index_issues': [],
            'view_issues': [],
            'errors': []
        }
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check required tables
                required_tables = [
                    'tracks', 'tags', 'playlists', 'playlist_tracks', 
                    'cache', 'statistics', 'discovery_cache'
                ]
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                for table in required_tables:
                    if table not in existing_tables:
                        validation_results['missing_tables'].append(table)
                        validation_results['schema_valid'] = False
                
                # Check required columns in tracks table
                if 'tracks' in existing_tables:
                    cursor.execute("PRAGMA table_info(tracks)")
                    track_columns = [row[1] for row in cursor.fetchall()]
                    
                    required_track_columns = [
                        'id', 'file_path', 'file_hash', 'filename', 'file_size_bytes',
                        'title', 'artist', 'album', 'genre', 'year', 'duration',
                        'bpm', 'key', 'mode', 'energy', 'danceability', 'status'
                    ]
                    
                    for column in required_track_columns:
                        if column not in track_columns:
                            validation_results['missing_columns'].append(f'tracks.{column}')
                            validation_results['schema_valid'] = False
                
                # Check indexes
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                existing_indexes = [row[0] for row in cursor.fetchall()]
                
                required_indexes = [
                    'idx_tracks_file_path', 'idx_tracks_status', 'idx_tracks_artist',
                    'idx_tracks_title', 'idx_tracks_genre', 'idx_tracks_bpm'
                ]
                
                for index in required_indexes:
                    if index not in existing_indexes:
                        validation_results['index_issues'].append(f"Missing index: {index}")
                
                # Check views
                cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
                existing_views = [row[0] for row in cursor.fetchall()]
                
                required_views = [
                    'track_summary', 'audio_analysis_complete', 'playlist_summary'
                ]
                
                for view in required_views:
                    if view not in existing_views:
                        validation_results['view_issues'].append(f"Missing view: {view}")
                
                if validation_results['schema_valid']:
                    log_universal('INFO', 'Database', "Database schema validation passed")
                else:
                    log_universal('WARNING', 'Database', f"Database schema validation failed: {validation_results}")
                
                return validation_results
                
        except Exception as e:
            validation_results['errors'].append(f"Schema validation failed: {e}")
            log_universal('ERROR', 'Database', f"Schema validation error: {e}")
            return validation_results

    # 🔧 Fix schema issues
    @log_function_call
    def fix_schema_issues(self) -> Dict[str, Any]:
        """
        Fix common schema issues.
        
        Returns:
            Dictionary with fix results
        """
        fix_results = {
            'tables_created': [],
            'columns_added': [],
            'indexes_created': [],
            'views_created': [],
            'errors': []
        }
        
        try:
            validation = self.validate_schema()
            
            if validation['schema_valid']:
                log_universal('INFO', 'Database', "No schema issues to fix")
                return fix_results
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Create missing tables
                for table in validation['missing_tables']:
                    try:
                        if table == 'tracks':
                            # Re-initialize database to create all tables
                            self._init_database()
                            fix_results['tables_created'].append(table)
                            log_universal('INFO', 'Database', f"Recreated table: {table}")
                        else:
                            # Create individual missing tables
                            self._create_missing_table(cursor, table)
                            fix_results['tables_created'].append(table)
                            log_universal('INFO', 'Database', f"Created table: {table}")
                    except Exception as e:
                        fix_results['errors'].append(f"Failed to create table {table}: {e}")
                
                # Add missing columns
                for column_info in validation['missing_columns']:
                    try:
                        table, column = column_info.split('.', 1)
                        self._add_missing_column(cursor, table, column)
                        fix_results['columns_added'].append(column_info)
                        log_universal('INFO', 'Database', f"Added column: {column_info}")
                    except Exception as e:
                        fix_results['errors'].append(f"Failed to add column {column_info}: {e}")
                
                # Create missing indexes
                for index_issue in validation['index_issues']:
                    try:
                        index_name = index_issue.split(': ')[1]
                        self._create_missing_index(cursor, index_name)
                        fix_results['indexes_created'].append(index_name)
                        log_universal('INFO', 'Database', f"Created index: {index_name}")
                    except Exception as e:
                        fix_results['errors'].append(f"Failed to create index {index_name}: {e}")
                
                conn.commit()
                
                if fix_results['errors']:
                    log_universal('WARNING', 'Database', f"Schema fixes completed with {len(fix_results['errors'])} errors")
                else:
                    log_universal('INFO', 'Database', "Schema fixes completed successfully")
                
                return fix_results
                
        except Exception as e:
            fix_results['errors'].append(f"Schema fix failed: {e}")
            log_universal('ERROR', 'Database', f"Schema fix error: {e}")
            return fix_results

    def _create_missing_table(self, cursor, table_name: str):
        """Create a missing table."""
        if table_name == 'tags':
            cursor.execute("""
                CREATE TABLE tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    tag_name TEXT NOT NULL,
                    tag_value TEXT,
                    source TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
                    UNIQUE(track_id, tag_name, source)
                )
            """)
        elif table_name == 'playlists':
            cursor.execute("""
                CREATE TABLE playlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    generation_method TEXT DEFAULT 'manual',
                    generation_params TEXT,
                    track_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        elif table_name == 'playlist_tracks':
            cursor.execute("""
                CREATE TABLE playlist_tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    playlist_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE,
                    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
                    UNIQUE(playlist_id, track_id, position)
                )
            """)
        elif table_name == 'cache':
            cursor.execute("""
                CREATE TABLE cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    cache_value TEXT NOT NULL,
                    cache_type TEXT NOT NULL,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        elif table_name == 'statistics':
            cursor.execute("""
                CREATE TABLE statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_data TEXT,
                    date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _add_missing_column(self, cursor, table_name: str, column_name: str):
        """Add a missing column to a table."""
        # This is a simplified approach - in production you'd want more sophisticated column addition
        log_universal('WARNING', 'Database', f"Column addition not implemented for {table_name}.{column_name}")

    def _create_missing_index(self, cursor, index_name: str):
        """Create a missing index."""
        if index_name == 'idx_tracks_file_path':
            cursor.execute("CREATE INDEX idx_tracks_file_path ON tracks(file_path)")
        elif index_name == 'idx_tracks_status':
            cursor.execute("CREATE INDEX idx_tracks_status ON tracks(status)")
        elif index_name == 'idx_tracks_artist':
            cursor.execute("CREATE INDEX idx_tracks_artist ON tracks(artist)")
        elif index_name == 'idx_tracks_title':
            cursor.execute("CREATE INDEX idx_tracks_title ON tracks(title)")
        elif index_name == 'idx_tracks_genre':
            cursor.execute("CREATE INDEX idx_tracks_genre ON tracks(genre)")
        elif index_name == 'idx_tracks_bpm':
            cursor.execute("CREATE INDEX idx_tracks_bpm ON tracks(bpm)")

    # 🛠️ Comprehensive database maintenance
    @log_function_call
    def perform_maintenance(self) -> Dict[str, Any]:
        """
        Perform comprehensive database maintenance.
        
        Returns:
            Dictionary with maintenance results
        """
        maintenance_results = {
            'integrity_check': False,
            'schema_validation': False,
            'schema_fixes': {},
            'repair_results': {},
            'vacuum_performed': False,
            'backup_created': False,
            'errors': []
        }
        
        try:
            log_universal('INFO', 'Database', "Starting comprehensive database maintenance...")
            
            # Step 1: Check database integrity
            try:
                maintenance_results['integrity_check'] = self.check_integrity()
                if not maintenance_results['integrity_check']:
                    log_universal('WARNING', 'Database', "Database integrity issues detected")
            except Exception as e:
                maintenance_results['errors'].append(f"Integrity check failed: {e}")
            
            # Step 2: Validate schema
            try:
                schema_validation = self.validate_schema()
                maintenance_results['schema_validation'] = schema_validation['schema_valid']
                if not schema_validation['schema_valid']:
                    log_universal('WARNING', 'Database', "Schema validation issues detected")
            except Exception as e:
                maintenance_results['errors'].append(f"Schema validation failed: {e}")
            
            # Step 3: Fix schema issues if needed
            if not maintenance_results['schema_validation']:
                try:
                    maintenance_results['schema_fixes'] = self.fix_schema_issues()
                except Exception as e:
                    maintenance_results['errors'].append(f"Schema fixes failed: {e}")
            
            # Step 4: Repair database issues
            try:
                maintenance_results['repair_results'] = self.repair_database()
            except Exception as e:
                maintenance_results['errors'].append(f"Database repair failed: {e}")
            
            # Step 5: Perform vacuum
            try:
                maintenance_results['vacuum_performed'] = self.vacuum_database()
                if maintenance_results['vacuum_performed']:
                    log_universal('INFO', 'Database', "Database vacuum completed")
            except Exception as e:
                maintenance_results['errors'].append(f"Database vacuum failed: {e}")
            
            # Step 6: Create backup
            try:
                backup_path = self.create_backup()
                if backup_path:
                    maintenance_results['backup_created'] = True
                    log_universal('INFO', 'Database', f"Database backup created: {backup_path}")
            except Exception as e:
                maintenance_results['errors'].append(f"Backup creation failed: {e}")
            
            # Step 7: Final integrity check
            try:
                final_integrity = self.check_integrity()
                if final_integrity:
                    log_universal('INFO', 'Database', "Final integrity check passed")
                else:
                    maintenance_results['errors'].append("Final integrity check failed")
            except Exception as e:
                maintenance_results['errors'].append(f"Final integrity check failed: {e}")
            
            # Summary
            if maintenance_results['errors']:
                log_universal('WARNING', 'Database', f"Database maintenance completed with {len(maintenance_results['errors'])} errors")
            else:
                log_universal('INFO', 'Database', "Database maintenance completed successfully")
            
            return maintenance_results
            
        except Exception as e:
            maintenance_results['errors'].append(f"Maintenance failed: {e}")
            log_universal('ERROR', 'Database', f"Database maintenance failed: {e}")
            return maintenance_results

    # 📊 Get database health status
    @log_function_call
    def get_database_health(self) -> Dict[str, Any]:
        """
        Get comprehensive database health status.
        
        Returns:
            Dictionary with health information
        """
        health_status = {
            'database_exists': False,
            'database_size': {},
            'integrity_ok': False,
            'schema_valid': False,
            'connection_ok': False,
            'tables_count': 0,
            'tracks_count': 0,
            'playlists_count': 0,
            'cache_entries': 0,
            'failed_analyses': 0,
            'last_backup': None,
            'issues': []
        }
        
        try:
            # Check if database exists
            health_status['database_exists'] = os.path.exists(self.db_path)
            if not health_status['database_exists']:
                health_status['issues'].append("Database file does not exist")
                return health_status
            
            # Get database size
            health_status['database_size'] = self.get_database_size()
            
            # Test connection
            try:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    health_status['connection_ok'] = True
            except Exception as e:
                health_status['issues'].append(f"Connection test failed: {e}")
            
            # Check integrity
            try:
                health_status['integrity_ok'] = self.check_integrity()
                if not health_status['integrity_ok']:
                    health_status['issues'].append("Database integrity check failed")
            except Exception as e:
                health_status['issues'].append(f"Integrity check failed: {e}")
            
            # Validate schema
            try:
                schema_validation = self.validate_schema()
                health_status['schema_valid'] = schema_validation['schema_valid']
                if not schema_validation['schema_valid']:
                    health_status['issues'].append("Schema validation failed")
            except Exception as e:
                health_status['issues'].append(f"Schema validation failed: {e}")
            
            # Get statistics
            try:
                stats = self.get_database_statistics()
                health_status['tracks_count'] = stats.get('total_tracks', 0)
                health_status['playlists_count'] = stats.get('playlists', 0)
                health_status['cache_entries'] = stats.get('cache_entries', 0)
                health_status['failed_analyses'] = stats.get('failed_analyses', 0)
            except Exception as e:
                health_status['issues'].append(f"Statistics retrieval failed: {e}")
            
            # Count tables
            try:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    health_status['tables_count'] = cursor.fetchone()[0]
            except Exception as e:
                health_status['issues'].append(f"Table count failed: {e}")
            
            # Check for recent backup
            try:
                backup_dir = os.path.dirname(self.db_path)
                backup_files = [f for f in os.listdir(backup_dir) if f.startswith('playlista.db.backup')]
                if backup_files:
                    backup_files.sort(reverse=True)
                    health_status['last_backup'] = backup_files[0]
            except Exception as e:
                health_status['issues'].append(f"Backup check failed: {e}")
            
            if health_status['issues']:
                log_universal('WARNING', 'Database', f"Database health check found {len(health_status['issues'])} issues")
            else:
                log_universal('INFO', 'Database', "Database health check passed")
            
            return health_status
            
        except Exception as e:
            health_status['issues'].append(f"Health check failed: {e}")
            log_universal('ERROR', 'Database', f"Database health check failed: {e}")
            return health_status

    # 🔍 Hash-based file tracking methods
    @log_function_call
    def find_track_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Find a track by its file hash.
        
        Args:
            file_hash: Hash of the file to find
            
        Returns:
            Track data dictionary or None if not found
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tracks WHERE file_hash = ?
                """, (file_hash,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            log_universal('ERROR', 'Database', f"Find track by hash failed: {e}")
            return None

    @log_function_call
    def find_tracks_by_hash_pattern(self, hash_pattern: str) -> List[Dict[str, Any]]:
        """
        Find tracks by hash pattern (useful for partial matches).
        
        Args:
            hash_pattern: Pattern to match against file hashes
            
        Returns:
            List of matching track dictionaries
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tracks WHERE file_hash LIKE ?
                """, (f"%{hash_pattern}%",))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            log_universal('ERROR', 'Database', f"Find tracks by hash pattern failed: {e}")
            return []

    @log_function_call
    def update_file_path_by_hash(self, file_hash: str, new_file_path: str) -> bool:
        """
        Update file path for a track identified by hash.
        
        Args:
            file_hash: Hash of the file to update
            new_file_path: New file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tracks 
                    SET file_path = ?, filename = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE file_hash = ?
                """, (new_file_path, os.path.basename(new_file_path), file_hash))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    log_universal('INFO', 'Database', f"Updated file path for hash {file_hash[:8]}... to {new_file_path}")
                    return True
                else:
                    log_universal('WARNING', 'Database', f"No track found with hash {file_hash[:8]}...")
                    return False
        except Exception as e:
            log_universal('ERROR', 'Database', f"Update file path by hash failed: {e}")
            return False

    @log_function_call
    def get_duplicate_files_by_hash(self) -> List[Dict[str, Any]]:
        """
        Find files with duplicate content (same hash but different paths).
        
        Returns:
            List of duplicate file groups
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_hash, COUNT(*) as count, 
                           GROUP_CONCAT(file_path, '|') as file_paths,
                           GROUP_CONCAT(filename, '|') as filenames
                    FROM tracks 
                    WHERE file_hash IS NOT NULL
                    GROUP BY file_hash 
                    HAVING COUNT(*) > 1
                    ORDER BY count DESC
                """)
                
                duplicates = []
                for row in cursor.fetchall():
                    file_paths = row['file_paths'].split('|')
                    filenames = row['filenames'].split('|')
                    
                    duplicates.append({
                        'file_hash': row['file_hash'],
                        'count': row['count'],
                        'file_paths': file_paths,
                        'filenames': filenames
                    })
                
                return duplicates
        except Exception as e:
            log_universal('ERROR', 'Database', f"Get duplicate files by hash failed: {e}")
            return []

    @log_function_call
    def get_moved_files(self, current_files: List[str]) -> Dict[str, Any]:
        """
        Detect files that have been moved by comparing current files with database.
        
        Args:
            current_files: List of current file paths
            
        Returns:
            Dictionary with moved file information
        """
        try:
            # Get current file hashes
            from .file_discovery import FileDiscovery
            fd = FileDiscovery()
            current_hashes = {}
            
            for file_path in current_files:
                try:
                    file_hash = fd._get_file_hash(file_path)
                    current_hashes[file_hash] = file_path
                except Exception as e:
                    log_universal('WARNING', 'Database', f"Could not hash file {file_path}: {e}")
            
            # Get database hashes
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_hash, file_path FROM tracks 
                    WHERE file_hash IS NOT NULL
                """)
                
                db_hashes = {row['file_hash']: row['file_path'] for row in cursor.fetchall()}
            
            # Find moved files
            moved_files = []
            for file_hash, current_path in current_hashes.items():
                if file_hash in db_hashes:
                    db_path = db_hashes[file_hash]
                    if db_path != current_path:
                        moved_files.append({
                            'file_hash': file_hash,
                            'old_path': db_path,
                            'new_path': current_path,
                            'filename': os.path.basename(current_path)
                        })
            
            # Find missing files (in DB but not in current files)
            missing_files = []
            for file_hash, db_path in db_hashes.items():
                if file_hash not in current_hashes:
                    missing_files.append({
                        'file_hash': file_hash,
                        'file_path': db_path,
                        'filename': os.path.basename(db_path)
                    })
            
            return {
                'moved_files': moved_files,
                'missing_files': missing_files,
                'total_moved': len(moved_files),
                'total_missing': len(missing_files)
            }
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Get moved files failed: {e}")
            return {'moved_files': [], 'missing_files': [], 'total_moved': 0, 'total_missing': 0}

    @log_function_call
    def update_moved_files(self, moved_files: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Update database with moved file information.
        
        Args:
            moved_files: List of moved file dictionaries
            
        Returns:
            Dictionary with update statistics
        """
        stats = {
            'updated': 0,
            'errors': 0,
            'total': len(moved_files)
        }
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                for moved_file in moved_files:
                    try:
                        cursor.execute("""
                            UPDATE tracks 
                            SET file_path = ?, filename = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE file_hash = ?
                        """, (
                            moved_file['new_path'],
                            moved_file['filename'],
                            moved_file['file_hash']
                        ))
                        
                        if cursor.rowcount > 0:
                            stats['updated'] += 1
                            log_universal('INFO', 'Database', f"Updated moved file: {moved_file['old_path']} -> {moved_file['new_path']}")
                        else:
                            stats['errors'] += 1
                            log_universal('WARNING', 'Database', f"No track found for moved file hash: {moved_file['file_hash'][:8]}...")
                            
                    except Exception as e:
                        stats['errors'] += 1
                        log_universal('ERROR', 'Database', f"Error updating moved file {moved_file['new_path']}: {e}")
                
                conn.commit()
                
            log_universal('INFO', 'Database', f"Moved files update complete: {stats['updated']} updated, {stats['errors']} errors")
            return stats
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Update moved files failed: {e}")
            return {'updated': 0, 'errors': len(moved_files), 'total': len(moved_files)}

    @log_function_call
    def get_hash_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about file hashes in the database.
        
        Returns:
            Dictionary with hash statistics
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Total files with hashes
                cursor.execute("SELECT COUNT(*) FROM tracks WHERE file_hash IS NOT NULL")
                total_with_hash = cursor.fetchone()[0]
                
                # Total files without hashes
                cursor.execute("SELECT COUNT(*) FROM tracks WHERE file_hash IS NULL")
                total_without_hash = cursor.fetchone()[0]
                
                # Duplicate hashes
                cursor.execute("""
                    SELECT COUNT(*) FROM (
                        SELECT file_hash FROM tracks 
                        WHERE file_hash IS NOT NULL
                        GROUP BY file_hash 
                        HAVING COUNT(*) > 1
                    )
                """)
                duplicate_hashes = cursor.fetchone()[0]
                
                # Hash algorithm distribution (if stored)
                cursor.execute("""
                    SELECT LENGTH(file_hash) as hash_length, COUNT(*) as count
                    FROM tracks 
                    WHERE file_hash IS NOT NULL
                    GROUP BY LENGTH(file_hash)
                """)
                hash_lengths = {row['hash_length']: row['count'] for row in cursor.fetchall()}
                
                return {
                    'total_with_hash': total_with_hash,
                    'total_without_hash': total_without_hash,
                    'duplicate_hashes': duplicate_hashes,
                    'hash_lengths': hash_lengths,
                    'total_files': total_with_hash + total_without_hash
                }
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Get hash statistics failed: {e}")
            return {}

    @log_function_call
    def calculate_file_hash(self, file_path: str, hash_type: str = 'discovery') -> str:
        """
        Calculate file hash using filename and size only.
        Simple, fast, and reliable approach.
        
        Args:
            file_path: Path to the file
            hash_type: Type of hash to calculate (kept for compatibility)
            
        Returns:
            Hash string based on filename + size
        """
        try:
            if not os.path.exists(file_path):
                log_universal('WARNING', 'Database', f"File does not exist: {file_path}")
                return "file_not_found"
            
            # Simple approach: filename + size
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            content = f"{filename}:{file_size}"
            
            return hashlib.md5(content.encode()).hexdigest()
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Hash calculation failed for {file_path}: {e}")
            # Fallback to filename hash
            filename = os.path.basename(file_path)
            return hashlib.md5(filename.encode()).hexdigest()

    @log_function_call
    def find_similar_tracks_by_metadata_hash(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Find tracks with similar metadata (same artist, title, album, year).
        
        Args:
            file_path: Path to the file to find similar tracks for
            
        Returns:
            List of similar track dictionaries
        """
        try:
            # Calculate metadata hash for the input file
            metadata_hash = self.calculate_file_hash(file_path, 'metadata')
            
            # Find tracks with similar metadata
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tracks 
                    WHERE file_hash LIKE ? 
                    AND file_path != ?
                """, (f"{metadata_hash}%", file_path))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Find similar tracks failed: {e}")
            return []

    @log_function_call
    def get_file_tracking_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of file tracking capabilities.
        
        Returns:
            Dictionary with file tracking summary
        """
        try:
            # Get hash statistics
            hash_stats = self.get_hash_statistics()
            
            # Get duplicate files
            duplicates = self.get_duplicate_files_by_hash()
            
            # Get total files
            total_files = hash_stats.get('total_files', 0)
            files_with_hash = hash_stats.get('total_with_hash', 0)
            
            # Calculate tracking coverage
            tracking_coverage = (files_with_hash / total_files * 100) if total_files > 0 else 0
            
            return {
                'total_files': total_files,
                'files_with_hash': files_with_hash,
                'files_without_hash': hash_stats.get('total_without_hash', 0),
                'tracking_coverage_percent': round(tracking_coverage, 2),
                'duplicate_files_count': len(duplicates),
                'duplicate_groups': duplicates,
                'hash_lengths': hash_stats.get('hash_lengths', {}),
                'can_track_moves': files_with_hash > 0,
                'can_detect_duplicates': len(duplicates) > 0
            }
            
        except Exception as e:
            log_universal('ERROR', 'Database', f"Get file tracking summary failed: {e}")
            return {}


# Global database manager instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        DatabaseManager: The global database manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
