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
            yield conn
        except Exception as e:
            if conn: conn.rollback()
            log_universal('ERROR', 'Database', f"Connection error: {e}")
            raise
        finally:
            if conn: conn.close()

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

                # Fallback paths to locate schema
                candidate_paths = [
                    os.path.join(os.path.dirname(self.db_path), 'database_schema.sql'),
                    os.path.join(os.path.dirname(__file__), 'database_schema.sql'),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database_schema.sql'),
                ]

                schema_sql = None
                for path in candidate_paths:
                    if os.path.exists(path):
                        with open(path, 'r', encoding='utf-8') as f:
                            schema_sql = f.read()
                        log_universal('INFO', 'Database', f"Loaded schema from: {path}")
                        break

                if schema_sql:
                    cursor.executescript(schema_sql)
                    log_universal('INFO', 'Database', "Schema initialized successfully.")
                else:
                    raise FileNotFoundError(f"No valid schema file found in paths: {candidate_paths}")

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

    # 🧪 Save complete analysis result (94 fields)
    @log_function_call
    def save_analysis_result(self, file_path: str, filename: str, file_size_bytes: int,
                             file_hash: str, analysis_data: Dict[str, Any],
                             metadata: Dict[str, Any] = None, discovery_source: str = 'file_system') -> bool:
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now()

                get = lambda d, k, default=None: d.get(k, default) if d else default
                j = lambda k: json.dumps(get(analysis_data, k, []))
                jd = lambda k: json.dumps(get(analysis_data, k, {}))

                values = [
                    file_path, file_hash, filename, file_size_bytes, now, now,
                    'analyzed', 'completed', get(metadata, 'modified_time'), 0, None, None,
                    get(metadata, 'title', 'Unknown'), get(metadata, 'artist', 'Unknown'),
                    get(metadata, 'album'), get(metadata, 'track_number'),
                    get(metadata, 'genre'), get(metadata, 'year'), get(analysis_data, 'duration'),
                    get(metadata, 'bitrate'), get(metadata, 'sample_rate'), get(metadata, 'channels'),
                    get(analysis_data, 'bpm'), get(analysis_data, 'rhythm_confidence'), j('bpm_estimates'), j('bpm_intervals'), get(analysis_data, 'external_bpm'),
                    get(analysis_data, 'key'), get(analysis_data, 'scale'), get(analysis_data, 'key_strength'), get(analysis_data, 'key_confidence'),
                    get(analysis_data, 'spectral_centroid'), get(analysis_data, 'spectral_flatness'), get(analysis_data, 'spectral_rolloff'),
                    get(analysis_data, 'spectral_bandwidth'), get(analysis_data, 'spectral_contrast_mean'), get(analysis_data, 'spectral_contrast_std'),
                    get(analysis_data, 'loudness'), get(analysis_data, 'dynamic_complexity'),
                    get(analysis_data, 'loudness_range'), get(analysis_data, 'dynamic_range'),
                    get(analysis_data, 'danceability'), get(analysis_data, 'energy'), get(analysis_data, 'mode'),
                    get(analysis_data, 'acousticness'), get(analysis_data, 'instrumentalness'),
                    get(analysis_data, 'speechiness'), get(analysis_data, 'valence'), get(analysis_data, 'liveness'),
                    get(analysis_data, 'popularity'), j('mfcc_coefficients'), j('mfcc_bands'), j('mfcc_std'), j('mfcc_delta'), j('mfcc_delta2'),
                    jd('embedding'), jd('embedding_std'), jd('embedding_min'), jd('embedding_max'), jd('tags'), get(analysis_data, 'musicnn_skipped', 0), j('chroma_mean'), j('chroma_std'),
                    get(metadata, 'composer'), get(metadata, 'lyricist'), get(metadata, 'band'), get(metadata, 'conductor'),
                    get(metadata, 'remixer'), get(metadata, 'subtitle'), get(metadata, 'grouping'), get(metadata, 'publisher'),
                    get(metadata, 'copyright'), get(metadata, 'encoded_by'), get(metadata, 'language'), get(metadata, 'mood'),
                    get(metadata, 'style'), get(metadata, 'quality'), get(metadata, 'original_artist'), get(metadata, 'original_album'),
                    get(metadata, 'original_year'), get(metadata, 'original_filename'), get(metadata, 'content_group'),
                    get(metadata, 'encoder'), get(metadata, 'file_type'), get(metadata, 'playlist_delay'),
                    get(metadata, 'recording_time'), get(metadata, 'tempo'), get(metadata, 'length'),
                    get(metadata, 'replaygain_track_gain'), get(metadata, 'replaygain_album_gain'),
                    get(metadata, 'replaygain_track_peak'), get(metadata, 'replaygain_album_peak'),
                    get(analysis_data, 'analysis_type', 'full'), True, get(analysis_data, 'audio_type', 'normal'), get(analysis_data, 'audio_category', get(analysis_data, 'long_audio_category')),
                    discovery_source, now, now
                ]

                cursor.execute(f"""
                    INSERT OR REPLACE INTO tracks (
                        file_path, file_hash, filename, file_size_bytes, analysis_date, discovery_date,
                        status, analysis_status, modified_time, retry_count, last_retry_date, error_message,
                        title, artist, album, track_number, genre, year, duration,
                        bitrate, sample_rate, channels,
                        bpm, rhythm_confidence, bpm_estimates, bpm_intervals, external_bpm,
                        key, scale, key_strength, key_confidence,
                        spectral_centroid, spectral_flatness, spectral_rolloff,
                        spectral_bandwidth, spectral_contrast_mean, spectral_contrast_std,
                        loudness, dynamic_complexity, loudness_range, dynamic_range,
                        danceability, energy, mode,
                        acousticness, instrumentalness, speechiness, valence, liveness, popularity,
                        mfcc_coefficients, mfcc_bands, mfcc_std, mfcc_delta, mfcc_delta2,
                        embedding, embedding_std, embedding_min, embedding_max, tags, musicnn_skipped,
                        chroma_mean, chroma_std,
                        composer, lyricist, band, conductor, remixer, subtitle, grouping, publisher,
                        copyright, encoded_by, language, mood,
                        style, quality, original_artist, original_album, original_year, original_filename,
                        content_group, encoder, file_type, playlist_delay, recording_time, tempo, length,
                        replaygain_track_gain, replaygain_album_gain, replaygain_track_peak, replaygain_album_peak,
                        analysis_type, analyzed, audio_type, long_audio_category, discovery_source,
                        created_at, updated_at
                    ) VALUES ({', '.join(['?'] * 99)})
                """, values)

                # Save tags if present
                if metadata and 'tags' in metadata:
                    self._save_tags(cursor, cursor.lastrowid, metadata['tags'])

                conn.commit()
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
                
                # Use cache table for failed analysis with cache_type='failed_analysis'
                cache_key = f"failed_analysis:{file_path}"
                cache_value = json.dumps({
                    'file_path': file_path,
                    'filename': filename,
                    'file_hash': file_hash,
                    'error_message': error_message,
                    'retry_count': 0,
                    'last_retry_date': datetime.now().isoformat(),
                    'status': 'failed'
                })
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache 
                    (cache_key, cache_value, cache_type, created_at)
                    VALUES (?, ?, 'failed_analysis', CURRENT_TIMESTAMP)
                """, (cache_key, cache_value))
                
                conn.commit()
                log_universal('INFO', 'Database', f'Marked analysis as failed: {filename}')
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
                for row in cursor.fetchall():
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
                return cursor.fetchone()[0] == 'ok'
        except Exception:
            return False

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
                
                # Extract basic metadata fields
                artist = metadata.get('artist', 'Unknown')
                title = metadata.get('title', 'Unknown')
                album = metadata.get('album')
                year = metadata.get('year')
                genre = metadata.get('genre')
                duration = metadata.get('duration')
                
                # Update existing record or create new one
                cursor.execute("""
                    INSERT OR REPLACE INTO tracks 
                    (file_path, filename, artist, title, album, year, genre, duration, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (file_path, os.path.basename(file_path), artist, title, album, year, genre, duration))
                
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
