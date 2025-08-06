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
from typing import Dict, Any, List, Optional, Tuple
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
        """Initialize database with complete schema if it doesn't exist."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check if database is empty (no tables exist)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = cursor.fetchall()
                
                if existing_tables:
                    log_universal('INFO', 'Database', f'Database already exists with {len(existing_tables)} tables')
                    return True
                
                # Database is empty, initialize with component-based schema
                schema_file = Path(__file__).parent.parent.parent / 'database' / 'component_based_schema.sql'
                
                if schema_file.exists():
                    with open(schema_file, 'r') as f:
                        schema_sql = f.read()
                    
                    # Execute the complete schema
                    cursor.executescript(schema_sql)
                    conn.commit()
                    
                    log_universal('INFO', 'Database', 'Complete database schema initialized successfully')
                    return True
                else:
                    log_universal('ERROR', 'Database', f'Schema file not found: {schema_file}')
                    return False
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f'Database initialization failed: {e}')
            return False

    def _ensure_dynamic_columns(self, cursor, features: Dict[str, Any]) -> List[str]:
        """
        Dynamically ensure all feature columns exist in the tracks table.
        Creates new columns if they don't exist.
        
        Args:
            cursor: Database cursor
            features: Dictionary of features to ensure columns for
            
        Returns:
            List of column names that exist
        """
        try:
            # Get current table schema
            cursor.execute("PRAGMA table_info(tracks)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # Define feature type mappings
            type_mappings = {
                'int': 'INTEGER',
                'float': 'REAL', 
                'str': 'TEXT',
                'bool': 'INTEGER',
                'list': 'TEXT',  # JSON string
                'dict': 'TEXT'   # JSON string
            }
            
            new_columns = []
            
            def add_feature_columns(feature_dict: Dict[str, Any], prefix: str = ''):
                """Recursively add columns for nested features."""
                for key, value in feature_dict.items():
                    if value is None:
                        continue
                    
                    # Special handling for custom_tags - store as JSON in single column
                    if key == 'custom_tags' and isinstance(value, dict):
                        column_name = 'custom_tags'
                        if column_name not in existing_columns:
                            try:
                                cursor.execute(f"ALTER TABLE tracks ADD COLUMN {column_name} TEXT")
                                new_columns.append(column_name)
                                existing_columns.add(column_name)
                                log_universal('DEBUG', 'Database', f'Added custom_tags column: {column_name}')
                            except sqlite3.OperationalError as e:
                                if "duplicate column name" not in str(e):
                                    log_universal('WARNING', 'Database', f'Failed to add column {column_name}: {e}')
                        continue  # Skip recursive processing for custom_tags
                        
                    # Create column name
                    column_name = f"{prefix}_{key}" if prefix else key
                    column_name = column_name.replace('.', '_').replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('&', 'and').replace('+', 'plus').lower()
                    
                    # Skip reserved SQLite keywords
                    reserved_keywords = {'index', 'order', 'group', 'table', 'select', 'where', 'from'}
                    if column_name in reserved_keywords:
                        column_name = f"feature_{column_name}"
                    
                    # Determine SQLite type
                    if isinstance(value, (int, float, str, bool)):
                        sqlite_type = type_mappings[type(value).__name__]
                    elif isinstance(value, (list, dict)):
                        sqlite_type = 'TEXT'  # JSON string
                    else:
                        sqlite_type = 'TEXT'  # Default to TEXT
                    
                    # Add column if it doesn't exist
                    if column_name not in existing_columns:
                        try:
                            cursor.execute(f"ALTER TABLE tracks ADD COLUMN {column_name} {sqlite_type}")
                            new_columns.append(column_name)
                            log_universal('DEBUG', 'Database', f'Added dynamic column: {column_name} ({sqlite_type})')
                        except sqlite3.OperationalError as e:
                            if "duplicate column name" not in str(e):
                                log_universal('WARNING', 'Database', f'Failed to add column {column_name}: {e}')
                    
                    # Recursively process nested dictionaries
                    if isinstance(value, dict):
                        add_feature_columns(value, column_name)
            
            # Process all features
            add_feature_columns(features)
            
            if new_columns:
                log_universal('INFO', 'Database', f'Added {len(new_columns)} dynamic columns: {new_columns}')
            
            return list(existing_columns) + new_columns
            
        except Exception as e:
            log_universal('ERROR', 'Database', f'Dynamic column creation failed: {e}')
            return []

    def _prepare_dynamic_values(self, features: Dict[str, Any], columns: List[str]) -> Tuple[List[str], List[Any]]:
        """
        Prepare dynamic INSERT values based on available columns.
        
        Args:
            features: Dictionary of features
            columns: List of available columns
            
        Returns:
            Tuple of (column_names, values)
        """
        def flatten_features(feature_dict: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
            """Flatten nested feature dictionary."""
            flattened = {}
            for key, value in feature_dict.items():
                if value is None:
                    continue
                
                # Special handling for custom_tags - store as JSON
                if key == 'custom_tags' and isinstance(value, dict):
                    flattened['custom_tags'] = json.dumps(value) if value else None
                    continue  # Skip recursive processing for custom_tags
                    
                column_name = f"{prefix}_{key}" if prefix else key
                column_name = column_name.replace('.', '_').replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('&', 'and').replace('+', 'plus').lower()
                
                # Skip reserved SQLite keywords
                reserved_keywords = {'index', 'order', 'group', 'table', 'select', 'where', 'from'}
                if column_name in reserved_keywords:
                    column_name = f"feature_{column_name}"
                
                if isinstance(value, dict):
                    flattened.update(flatten_features(value, column_name))
                else:
                    # Convert complex types to JSON strings
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value) if value else None
                    flattened[column_name] = value
            
            return flattened
        
        # Flatten all features
        flattened_features = flatten_features(features)
        
        # Prepare values for available columns
        values = []
        column_names = []
        
        for column in columns:
            value = flattened_features.get(column)
            if value is not None:
                column_names.append(column)
                values.append(value)
        
        return column_names, values

    def consolidate_cache_to_tracks(self, file_path: str) -> bool:
        """
        Consolidate all cached analysis data into the tracks table.
        This ensures all features are stored as proper columns instead of JSON blobs.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get all cached data for this file
                cache_data = {}
                
                # Get metadata
                metadata_cache = self.get_cache(f"metadata_{file_path}")
                if metadata_cache:
                    cache_data.update(metadata_cache)
                
                # Get essentia features
                essentia_cache = self.get_cache(f"essentia_{file_path}")
                if essentia_cache:
                    cache_data.update(essentia_cache)
                
                # Get Spotify features
                spotify_cache = self.get_cache(f"spotify_{file_path}")
                if spotify_cache:
                    cache_data.update(spotify_cache)
                
                # Get MusicNN features
                musicnn_cache = self.get_cache(f"musicnn_{file_path}")
                if musicnn_cache:
                    cache_data.update(musicnn_cache)
                
                # Get advanced categorization features
                advanced_cache = self.get_cache(f"advanced_{file_path}")
                if advanced_cache:
                    cache_data.update(advanced_cache)
                
                # Get external API enrichment data
                external_cache = self.get_cache(f"external_{file_path}")
                if external_cache:
                    cache_data.update(external_cache)
                
                # Get comprehensive categorization data
                categorization_cache = self.get_cache(f"categorization_{file_path}")
                if categorization_cache:
                    cache_data.update(categorization_cache)
                
                # Also check for analysis result cache
                analysis_cache = self.get_cache(f"analysis_{file_path}")
                if analysis_cache and isinstance(analysis_cache, dict):
                    # Extract features from analysis result
                    if 'features' in analysis_cache:
                        features = analysis_cache['features']
                        if isinstance(features, dict):
                            # Flatten nested features
                            for feature_type, feature_data in features.items():
                                if isinstance(feature_data, dict):
                                    cache_data.update(feature_data)
                                else:
                                    cache_data[f"{feature_type}_data"] = feature_data
                    
                    # Extract metadata from analysis result
                    if 'metadata' in analysis_cache:
                        metadata = analysis_cache['metadata']
                        if isinstance(metadata, dict):
                            cache_data.update(metadata)
                    
                    # Extract external API data
                    if 'external_api_data' in analysis_cache:
                        external_data = analysis_cache['external_api_data']
                        if isinstance(external_data, dict):
                            cache_data.update(external_data)
                
                # If no cache data found, try to get data from the tracks table itself
                if not cache_data:
                    log_universal('DEBUG', 'Database', f'No cache data found, checking tracks table for {file_path}')
                    
                    # Get existing track data
                    cursor.execute("SELECT * FROM tracks WHERE file_path = ?", (file_path,))
                    existing_track = cursor.fetchone()
                    
                    if existing_track:
                        # Convert row to dict for processing
                        track_dict = dict(existing_track)
                        log_universal('INFO', 'Database', f'Found existing track data for {file_path}')
                        return True  # Data already exists in tracks table
                    else:
                        log_universal('WARNING', 'Database', f'No cached data found for {file_path}')
                        return False
                
                # Ensure all columns exist
                available_columns = self._ensure_dynamic_columns(cursor, cache_data)
                
                # Prepare values
                column_names, values = self._prepare_dynamic_values(cache_data, available_columns)
                
                if not column_names:
                    log_universal('WARNING', 'Database', f'No valid columns found for {file_path}')
                    return False
                
                # Build dynamic INSERT/UPDATE
                placeholders = ', '.join(['?' for _ in column_names])
                column_list = ', '.join(column_names)
                
                # Check if track already exists
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                existing_track = cursor.fetchone()
                
                if existing_track:
                    # UPDATE existing track
                    set_clause = ', '.join([f"{col} = ?" for col in column_names])
                    cursor.execute(f"""
                        UPDATE tracks 
                        SET {set_clause}, updated_at = ?
                        WHERE file_path = ?
                    """, values + [datetime.now(), file_path])
                    log_universal('INFO', 'Database', f'Updated track with {len(column_names)} dynamic columns: {file_path}')
                else:
                    # INSERT new track with basic required fields
                    basic_columns = ['file_path', 'filename', 'file_size_bytes', 'created_at', 'updated_at']
                    basic_values = [file_path, os.path.basename(file_path), 0, datetime.now(), datetime.now()]
                    
                    # Add dynamic columns
                    all_columns = basic_columns + column_names
                    all_values = basic_values + values
                    all_placeholders = ', '.join(['?' for _ in all_values])  # Use actual values count
                    all_column_list = ', '.join(all_columns)
                    
                    cursor.execute(f"""
                        INSERT INTO tracks ({all_column_list})
                        VALUES ({all_placeholders})
                    """, all_values)
                    log_universal('INFO', 'Database', f'Inserted track with {len(column_names)} dynamic columns: {file_path}')
                
                conn.commit()
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to consolidate cache to tracks: {e}')
            return False

    def migrate_json_to_columns(self) -> Dict[str, int]:
        """
        Migrate existing JSON data in tracks table to proper columns.
        This converts JSON blobs to individual columns for better performance.
        
        Returns:
            Dictionary with migration statistics
        """
        try:
            stats = {
                'tracks_processed': 0,
                'columns_created': 0,
                'values_migrated': 0,
                'errors': 0
            }
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get all tracks with JSON data
                cursor.execute("""
                    SELECT id, file_path, spotify_features, mfcc_features, musicnn_features, 
                           rhythm_features, spectral_features, chroma_mean, chroma_std
                    FROM tracks 
                    WHERE spotify_features IS NOT NULL 
                       OR mfcc_features IS NOT NULL 
                       OR musicnn_features IS NOT NULL
                       OR rhythm_features IS NOT NULL
                       OR spectral_features IS NOT NULL
                       OR chroma_mean IS NOT NULL
                       OR chroma_std IS NOT NULL
                """)
                
                tracks = cursor.fetchall()
                log_universal('INFO', 'Database', f'Found {len(tracks)} tracks with JSON data to migrate')
                
                for track in tracks:
                    try:
                        track_id, file_path = track[0], track[1]
                        
                        # Parse JSON data
                        json_data = {}
                        
                        for i, field_name in enumerate(['spotify_features', 'mfcc_features', 'musicnn_features', 
                                                      'rhythm_features', 'spectral_features', 'chroma_mean', 'chroma_std']):
                            if track[i+2]:  # Skip id and file_path
                                try:
                                    if isinstance(track[i+2], str):
                                        parsed = json.loads(track[i+2])
                                    else:
                                        parsed = track[i+2]
                                    json_data.update(parsed)
                                except (json.JSONDecodeError, TypeError):
                                    continue
                        
                        if json_data:
                            # Ensure columns exist
                            available_columns = self._ensure_dynamic_columns(cursor, json_data)
                            
                            # Prepare values
                            column_names, values = self._prepare_dynamic_values(json_data, available_columns)
                            
                            if column_names:
                                # Update track with new columns
                                set_clause = ', '.join([f"{col} = ?" for col in column_names])
                                cursor.execute(f"""
                                    UPDATE tracks 
                                    SET {set_clause}, updated_at = ?
                                    WHERE id = ?
                                """, values + [datetime.now(), track_id])
                                
                                stats['values_migrated'] += len(column_names)
                                stats['tracks_processed'] += 1
                                
                                # Clear old JSON columns
                                cursor.execute("""
                                    UPDATE tracks 
                                    SET spotify_features = NULL, mfcc_features = NULL, musicnn_features = NULL,
                                        rhythm_features = NULL, spectral_features = NULL, chroma_mean = NULL, chroma_std = NULL
                                    WHERE id = ?
                                """, (track_id,))
                    
                    except Exception as e:
                        log_universal('ERROR', 'Database', f'Failed to migrate track {file_path}: {e}')
                        stats['errors'] += 1
                
                conn.commit()
                log_universal('INFO', 'Database', f'Migration completed: {stats}')
                return stats
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Migration failed: {e}')
            return {'error': str(e)}

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get information about the current database schema.
        
        Returns:
            Dictionary with schema information
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get table information
                cursor.execute("PRAGMA table_info(tracks)")
                columns = cursor.fetchall()
                
                # Get table statistics
                cursor.execute("SELECT COUNT(*) FROM tracks")
                track_count = cursor.fetchone()[0]
                
                # Get column statistics
                column_stats = {}
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    
                    # Count non-null values
                    cursor.execute(f"SELECT COUNT(*) FROM tracks WHERE {col_name} IS NOT NULL")
                    non_null_count = cursor.fetchone()[0]
                    
                    column_stats[col_name] = {
                        'type': col_type,
                        'non_null_count': non_null_count,
                        'null_percentage': ((track_count - non_null_count) / track_count * 100) if track_count > 0 else 0
                    }
                
                return {
                    'total_tracks': track_count,
                    'total_columns': len(columns),
                    'columns': column_stats
                }
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to get schema info: {e}')
            return {'error': str(e)}

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
        """
        Save analysis result using component-based approach.
        
        Args:
            file_path: Path to the audio file
            filename: Just the filename
            file_size_bytes: File size in bytes
            file_hash: MD5 hash of the file
            analysis_data: Complete analysis data structure
            metadata: Additional metadata
            discovery_source: Source of discovery
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Save file discovery
            self.save_file_discovery(file_path, file_hash, file_size_bytes, filename, discovery_source)
            
            # 2. Save mutagen metadata if available
            if metadata:
                self.save_mutagen_metadata(file_path, metadata)
            
            # 3. Save analysis components
            if analysis_data:
                # Save MusicNN features
                if 'musicnn' in analysis_data:
                    self.save_musicnn_features(file_path, analysis_data['musicnn'])
                
                # Save Essentia features
                if 'essentia' in analysis_data:
                    self.save_essentia_features(file_path, analysis_data['essentia'])
                
                # Save external metadata
                if 'external' in analysis_data:
                    self.save_external_metadata(file_path, analysis_data['external'])
                
                # Save audio analysis
                if 'librosa' in analysis_data:
                    self.save_audio_analysis(file_path, 'librosa', analysis_data['librosa'])
                
                # Save Spotify features
                if 'spotify' in analysis_data:
                    self.save_spotify_features(file_path, analysis_data['spotify'])
            
            # 4. Update main table with essential features for fast queries
            self._update_main_track_features(file_path, analysis_data, metadata)
            
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save analysis result: {e}')
            return False

    def _update_main_track_features(self, file_path: str, analysis_data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        Update main tracks table with essential features for fast queries.
        
        Args:
            file_path: Path to the file
            analysis_data: Analysis data
            metadata: Metadata
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get track ID
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                track_row = cursor.fetchone()
                if not track_row:
                    return
                
                track_id = track_row[0]
                
                # Extract essential features for playlist queries
                update_fields = {}
                
                # From Essentia features
                if analysis_data and 'essentia' in analysis_data:
                    essentia = analysis_data['essentia']
                    if essentia.get('bpm'):
                        update_fields['bpm'] = essentia['bpm']
                    if essentia.get('key'):
                        update_fields['key'] = essentia['key']
                    if essentia.get('scale'):
                        update_fields['mode'] = essentia['scale']
                
                # From Spotify features
                if analysis_data and 'spotify' in analysis_data:
                    spotify = analysis_data['spotify']
                    if spotify.get('energy'):
                        update_fields['energy'] = spotify['energy']
                    if spotify.get('danceability'):
                        update_fields['danceability'] = spotify['danceability']
                    if spotify.get('valence'):
                        update_fields['valence'] = spotify['valence']
                    if spotify.get('acousticness'):
                        update_fields['acousticness'] = spotify['acousticness']
                    if spotify.get('instrumentalness'):
                        update_fields['instrumentalness'] = spotify['instrumentalness']
                    if spotify.get('speechiness'):
                        update_fields['speechiness'] = spotify['speechiness']
                    if spotify.get('liveness'):
                        update_fields['liveness'] = spotify['liveness']
                
                # Update main table if we have features
                if update_fields:
                    set_clause = ', '.join([f"{k} = ?" for k in update_fields.keys()])
                    values = list(update_fields.values()) + [track_id]
                    
                    cursor.execute(f"""
                        UPDATE tracks 
                        SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, values)
                
                conn.commit()
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to update main track features: {e}')
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now()

                # Extract core metadata with fallbacks
                get = lambda d, k, default=None: d.get(k, default) if d else default
                
                # Core file info
                title = get(metadata, 'title', filename)
                artist = get(metadata, 'artist', 'Unknown')
                album = get(metadata, 'album')
                track_number = get(metadata, 'track_number')
                
                # Ensure genre is always stored as JSON array
                raw_genre = get(metadata, 'genre')
                if raw_genre is not None:
                    if isinstance(raw_genre, list):
                        genre = json.dumps(raw_genre)
                    elif isinstance(raw_genre, str):
                        # If it's already a JSON string, keep it
                        if raw_genre.startswith('[') and raw_genre.endswith(']'):
                            try:
                                json.loads(raw_genre)  # Validate JSON
                                genre = raw_genre
                            except json.JSONDecodeError:
                                genre = json.dumps([raw_genre])
                        else:
                            genre = json.dumps([raw_genre])
                    else:
                        genre = json.dumps([str(raw_genre)])
                else:
                    genre = None
                
                year = get(metadata, 'year')
                duration = get(analysis_data, 'duration')

                # Audio properties
                bitrate = get(metadata, 'bitrate')
                sample_rate = get(metadata, 'sample_rate')
                channels = get(metadata, 'channels')

                # Extract all nested features dynamically
                all_features = {}
                
                # Extract from analysis_data structure
                if isinstance(analysis_data, dict):
                    # Handle nested structure: {'essentia': {...}, 'musicnn': {...}, 'metadata': {...}}
                    for category, category_data in analysis_data.items():
                        if isinstance(category_data, dict):
                            all_features.update(category_data)
                        else:
                            all_features[category] = category_data
                    
                    # Also extract top-level features
                    for key, value in analysis_data.items():
                        if key not in ['essentia', 'musicnn', 'metadata']:
                            all_features[key] = value
                
                # Extract from metadata if provided
                if metadata:
                    all_features.update(metadata)
                
                # Ensure dynamic columns exist for all features
                available_columns = self._ensure_dynamic_columns(cursor, all_features)
                
                # Prepare values using dynamic approach
                column_names, values = self._prepare_dynamic_values(all_features, available_columns)
                
                # Add core fields that are always required
                core_fields = {
                    'file_path': file_path,
                    'file_hash': file_hash,
                    'filename': filename,
                    'file_size_bytes': file_size_bytes,
                    'created_at': now,
                    'updated_at': now,
                    'status': 'analyzed',
                    'analysis_status': 'completed',
                    'retry_count': 0,
                    'error_message': None,
                    'title': title,
                    'artist': artist,
                    'album': album,
                    'genre': genre,
                    'year': year,
                    'duration': duration
                }
                
                # Merge core fields with dynamic features
                for key, value in core_fields.items():
                    if key not in column_names:
                        column_names.append(key)
                        values.append(value)
                
                # Build INSERT statement
                placeholders = ', '.join(['?' for _ in column_names])
                column_list = ', '.join(column_names)
                
                # Ensure all values are SQLite-compatible
                def sanitize_value(value):
                    if value is None:
                        return None
                    if isinstance(value, (int, float, str)):
                        return value
                    if isinstance(value, (list, dict)):
                        return json.dumps(value)
                    if hasattr(value, 'tolist'):  # numpy arrays
                        return json.dumps(value.tolist())
                    return str(value)
                
                sanitized_values = [sanitize_value(v) for v in values]
                
                cursor.execute(f"""
                    INSERT OR REPLACE INTO tracks ({column_list})
                    VALUES ({placeholders})
                """, sanitized_values)

                track_id = cursor.lastrowid

                # Save tags if provided
                if metadata and 'tags' in metadata:
                    self._save_tags(cursor, track_id, metadata['tags'])
                
                # Save external API tags if available
                external_tags = get(metadata, 'tags', [])
                external_genres = get(metadata, 'genres', [])
                
                if external_tags:
                    self._save_tags(cursor, track_id, {'external_apis': external_tags})
                
                if external_genres:
                    self._save_tags(cursor, track_id, {'external_genres': external_genres})

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
        """
        Get analysis result with parsed JSON fields.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Analysis result dictionary with parsed JSON fields or None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM tracks WHERE file_path = ?
                """, (file_path,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                # Convert to dictionary
                analysis_result = dict(result)
                
                # Parse JSON fields back to structured data
                json_fields = [
                    'bpm_estimates', 'bpm_intervals', 'mfcc_coefficients', 'mfcc_bands', 
                    'mfcc_std', 'mfcc_delta', 'mfcc_delta2', 'embedding', 'embedding_std',
                    'embedding_min', 'embedding_max', 'tags', 'chroma_mean', 'chroma_std',
                    'rhythm_features', 'spectral_features', 'mfcc_features', 'musicnn_features',
                    'spotify_features', 'enrichment_sources'
                ]
                
                for field in json_fields:
                    if field in analysis_result and analysis_result[field]:
                        try:
                            if isinstance(analysis_result[field], str):
                                analysis_result[field] = json.loads(analysis_result[field])
                        except (json.JSONDecodeError, TypeError):
                            log_universal('WARNING', 'Database', f'Failed to parse JSON field: {field}')
                
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
    def save_file_discovery(self, file_path: str, file_hash: str, file_size_bytes: int, filename: str, discovery_source: str = 'file_system') -> bool:
        """
        Save file discovery to component table.
        
        Args:
            file_path: Path to the file
            file_hash: File hash
            file_size_bytes: File size in bytes
            filename: Just the filename
            discovery_source: Source of discovery
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Insert or replace file discovery
                cursor.execute("""
                    INSERT OR REPLACE INTO file_discovery 
                    (file_path, file_hash, file_size_bytes, filename, discovery_source, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (file_path, file_hash, file_size_bytes, filename, discovery_source))
                
                # Also create basic track entry
                cursor.execute("""
                    INSERT OR REPLACE INTO tracks 
                    (file_path, file_hash, filename, file_size_bytes, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (file_path, file_hash, filename, file_size_bytes))
                
                conn.commit()
                log_universal('DEBUG', 'Database', f'Saved file discovery for: {file_path}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save file discovery: {e}')
            return False

    @log_function_call
    def save_mutagen_metadata(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Save mutagen metadata to component table.
        
        Args:
            file_path: Path to the file
            metadata: Mutagen metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get track ID
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                track_row = cursor.fetchone()
                if not track_row:
                    log_universal('WARNING', 'Database', f'Track not found for mutagen metadata: {file_path}')
                    return False
                
                track_id = track_row[0]
                
                # Extract metadata fields
                title = metadata.get('title', 'Unknown')
                artist = metadata.get('artist', 'Unknown')
                album = metadata.get('album')
                track_number = metadata.get('track_number')
                disc_number = metadata.get('disc_number')
                year = metadata.get('year')
                
                # Ensure genre is always stored as JSON array
                raw_genre = metadata.get('genre')
                if raw_genre is not None:
                    if isinstance(raw_genre, list):
                        genre = json.dumps(raw_genre)
                    elif isinstance(raw_genre, str):
                        if raw_genre.startswith('[') and raw_genre.endswith(']'):
                            try:
                                json.loads(raw_genre)
                                genre = raw_genre
                            except json.JSONDecodeError:
                                genre = json.dumps([raw_genre])
                        else:
                            genre = json.dumps([raw_genre])
                    else:
                        genre = json.dumps([str(raw_genre)])
                else:
                    genre = None
                
                duration = metadata.get('duration')
                bitrate = metadata.get('bitrate')
                sample_rate = metadata.get('sample_rate')
                channels = metadata.get('channels')
                
                # Extract all other mutagen fields
                encoded_by = metadata.get('encoded_by')
                language = metadata.get('language')
                copyright = metadata.get('copyright')
                publisher = metadata.get('publisher')
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
                replaygain_track_gain = metadata.get('replaygain_track_gain')
                replaygain_album_gain = metadata.get('replaygain_album_gain')
                replaygain_track_peak = metadata.get('replaygain_track_peak')
                replaygain_album_peak = metadata.get('replaygain_album_peak')
                lyricist = metadata.get('lyricist')
                band = metadata.get('band')
                conductor = metadata.get('conductor')
                remixer = metadata.get('remixer')
                
                # Insert or replace mutagen metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO mutagen_metadata 
                    (track_id, title, artist, album, track_number, disc_number, year, genre,
                     duration, bitrate, sample_rate, channels, encoded_by, language, copyright,
                     publisher, original_artist, original_album, original_year, original_filename,
                     content_group, encoder, file_type, playlist_delay, recording_time, tempo,
                     length, replaygain_track_gain, replaygain_album_gain, replaygain_track_peak,
                     replaygain_album_peak, lyricist, band, conductor, remixer, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    track_id, title, artist, album, track_number, disc_number, year, genre,
                    duration, bitrate, sample_rate, channels, encoded_by, language, copyright,
                    publisher, original_artist, original_album, original_year, original_filename,
                    content_group, encoder, file_type, playlist_delay, recording_time, tempo,
                    length, replaygain_track_gain, replaygain_album_gain, replaygain_track_peak,
                    replaygain_album_peak, lyricist, band, conductor, remixer
                ))
                
                conn.commit()
                log_universal('DEBUG', 'Database', f'Saved mutagen metadata to component table for: {file_path}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save mutagen metadata: {e}')
            return False

    @log_function_call
    def save_metadata(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Save metadata to database (legacy method - now uses component tables).
        
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
                
                # Ensure genre is always stored as JSON array
                raw_genre = metadata.get('genre') if metadata else None
                if raw_genre is not None:
                    if isinstance(raw_genre, list):
                        genre = json.dumps(raw_genre)
                    elif isinstance(raw_genre, str):
                        # If it's already a JSON string, keep it
                        if raw_genre.startswith('[') and raw_genre.endswith(']'):
                            try:
                                json.loads(raw_genre)  # Validate JSON
                                genre = raw_genre
                            except json.JSONDecodeError:
                                genre = json.dumps([raw_genre])
                        else:
                            genre = json.dumps([raw_genre])
                    else:
                        genre = json.dumps([str(raw_genre)])
                else:
                    genre = None
                
                duration = metadata.get('duration')
                
                # Extract technical metadata
                bitrate = metadata.get('bitrate') if metadata else None
                sample_rate = metadata.get('sample_rate') if metadata else None
                channels = metadata.get('channels') if metadata else None
                
                # Extract external API data
                musicbrainz_id = metadata.get('musicbrainz_id') if metadata else None
                musicbrainz_artist_id = metadata.get('musicbrainz_artist_id') if metadata else None
                musicbrainz_album_id = metadata.get('musicbrainz_album_id') if metadata else None
                discogs_id = metadata.get('discogs_id') if metadata else None
                spotify_id = metadata.get('spotify_id') if metadata else None
                release_date = metadata.get('release_date') if metadata else None
                disc_number = metadata.get('disc_number') if metadata else None
                duration_ms = metadata.get('duration_ms') if metadata else None
                play_count = metadata.get('play_count') if metadata else None
                listeners = metadata.get('listeners') if metadata else None
                rating = metadata.get('rating') if metadata else None
                popularity = metadata.get('popularity') if metadata else None
                url = metadata.get('url') if metadata else None
                image_url = metadata.get('image_url') if metadata else None
                enrichment_sources = metadata.get('enrichment_sources') if metadata else None
                
                # Convert enrichment_sources to JSON if it's a list
                if isinstance(enrichment_sources, list):
                    enrichment_sources = json.dumps(enrichment_sources)
                
                # Extract mutagen-specific metadata
                encoded_by = metadata.get('encoded_by') if metadata else None
                language = metadata.get('language') if metadata else None
                copyright = metadata.get('copyright') if metadata else None
                publisher = metadata.get('publisher') if metadata else None
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
                lyricist = metadata.get('lyricist') if metadata else None
                band = metadata.get('band') if metadata else None
                conductor = metadata.get('conductor') if metadata else None
                remixer = metadata.get('remixer') if metadata else None
                
                # Prepare metadata for dynamic insertion
                metadata_dict = {
                    'file_path': file_path,
                    'file_hash': file_hash,
                    'filename': filename,
                    'file_size_bytes': file_size_bytes,
                    'artist': artist,
                    'title': title,
                    'album': album,
                    'year': year,
                    'genre': genre,
                    'duration': duration,
                    'bitrate': bitrate,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'musicbrainz_id': musicbrainz_id,
                    'musicbrainz_artist_id': musicbrainz_artist_id,
                    'musicbrainz_album_id': musicbrainz_album_id,
                    'discogs_id': discogs_id,
                    'spotify_id': spotify_id,
                    'release_date': release_date,
                    'disc_number': disc_number,
                    'duration_ms': duration_ms,
                    'play_count': play_count,
                    'listeners': listeners,
                    'rating': rating,
                    'popularity': popularity,
                    'url': url,
                    'image_url': image_url,
                    'enrichment_sources': enrichment_sources,
                    'encoded_by': encoded_by,
                    'language': language,
                    'copyright': copyright,
                    'publisher': publisher,
                    'original_artist': original_artist,
                    'original_album': original_album,
                    'original_year': original_year,
                    'original_filename': original_filename,
                    'content_group': content_group,
                    'encoder': encoder,
                    'file_type': file_type,
                    'playlist_delay': playlist_delay,
                    'recording_time': recording_time,
                    'tempo': tempo,
                    'length': length,
                    'replaygain_track_gain': replaygain_track_gain,
                    'replaygain_album_gain': replaygain_album_gain,
                    'replaygain_track_peak': replaygain_track_peak,
                    'replaygain_album_peak': replaygain_album_peak,
                    'lyricist': lyricist,
                    'band': band,
                    'conductor': conductor,
                    'remixer': remixer
                }
                
                # Remove None values to avoid inserting NULLs
                metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None}
                
                # Ensure all columns exist
                available_columns = self._ensure_dynamic_columns(cursor, metadata_dict)
                
                # Prepare values dynamically
                column_names, values = self._prepare_dynamic_values(metadata_dict, available_columns)
                
                if column_names:
                    # Build dynamic INSERT statement
                    all_columns = ['file_path'] + column_names + ['updated_at']
                    all_values = [file_path] + values + [datetime.now()]
                    all_placeholders = ', '.join(['?' for _ in all_values])
                    all_column_list = ', '.join(all_columns)
                    
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO tracks ({all_column_list})
                        VALUES ({all_placeholders})
                    """, all_values)
                    
                    log_universal('DEBUG', 'Database', f'Saved metadata with {len(column_names)} dynamic columns')
                else:
                    # Fallback to basic insert if no dynamic columns
                    cursor.execute("""
                        INSERT OR REPLACE INTO tracks 
                        (file_path, filename, file_size_bytes, artist, title, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (file_path, filename, file_size_bytes, artist, title))
                
                conn.commit()
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save metadata: {e}')
            return False

    @log_function_call
    def save_essentia_features(self, file_path: str, features: Dict[str, Any]) -> bool:
        """
        Save Essentia features to component table.
        
        Args:
            file_path: Path to the file
            features: Essentia features dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get track ID
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                track_row = cursor.fetchone()
                if not track_row:
                    log_universal('WARNING', 'Database', f'Track not found for Essentia features: {file_path}')
                    return False
                
                track_id = track_row[0]
                
                # Convert features to JSON for storage
                rhythm_features_json = json.dumps(features.get('rhythm', {})) if features.get('rhythm') else None
                spectral_features_json = json.dumps(features.get('spectral', {})) if features.get('spectral') else None
                mfcc_features_json = json.dumps(features.get('mfcc', {})) if features.get('mfcc') else None
                harmonic_features_json = json.dumps(features.get('harmonic', {})) if features.get('harmonic') else None
                
                # Extract key values for main table
                bpm = features.get('bpm')
                key = features.get('key')
                scale = features.get('scale')
                rhythm_confidence = features.get('rhythm_confidence', 0.0)
                key_confidence = features.get('key_confidence', 0.0)
                processing_time = features.get('processing_time', 0.0)
                
                # Insert or replace Essentia features in component table
                cursor.execute("""
                    INSERT OR REPLACE INTO essentia_features 
                    (track_id, rhythm_features, spectral_features, mfcc_features, harmonic_features,
                     bpm, key, scale, rhythm_confidence, key_confidence, processing_time, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    track_id,
                    rhythm_features_json,
                    spectral_features_json,
                    mfcc_features_json,
                    harmonic_features_json,
                    bpm,
                    key,
                    scale,
                    rhythm_confidence,
                    key_confidence,
                    processing_time
                ))
                
                # Update main tracks table with essential features
                if bpm or key or scale:
                    cursor.execute("""
                        UPDATE tracks 
                        SET bpm = COALESCE(?, bpm), 
                            key = COALESCE(?, key), 
                            mode = COALESCE(?, mode),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (bpm, key, scale, track_id))
                
                conn.commit()
                log_universal('DEBUG', 'Database', f'Saved Essentia features to component table for: {file_path}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save Essentia features: {e}')
            return False

    @log_function_call
    def save_musicnn_features(self, file_path: str, features: Dict[str, Any]) -> bool:
        """
        Save MusicNN features to component table.
        
        Args:
            file_path: Path to the file
            features: MusicNN features dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get track ID
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                track_row = cursor.fetchone()
                if not track_row:
                    log_universal('WARNING', 'Database', f'Track not found for MusicNN features: {file_path}')
                    return False
                
                track_id = track_row[0]
                
                # Convert features to JSON for storage
                embedding_json = json.dumps(features.get('embedding', []))
                tags_json = json.dumps(features.get('tags', {}))
                confidence = features.get('confidence', 0.0)
                processing_time = features.get('processing_time', 0.0)
                
                # Insert or replace MusicNN features in component table
                cursor.execute("""
                    INSERT OR REPLACE INTO musicnn_features 
                    (track_id, embedding, tags, confidence, processing_time, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    track_id,
                    embedding_json,
                    tags_json,
                    confidence,
                    processing_time
                ))
                
                conn.commit()
                log_universal('DEBUG', 'Database', f'Saved MusicNN features to component table for: {file_path}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save MusicNN features: {e}')
            return False

    @log_function_call
    def save_external_metadata(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Save external metadata to component table.
        
        Args:
            file_path: Path to the file
            metadata: External metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get track ID
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                track_row = cursor.fetchone()
                if not track_row:
                    log_universal('WARNING', 'Database', f'Track not found for external metadata: {file_path}')
                    return False
                
                track_id = track_row[0]
                
                # Extract external API IDs
                musicbrainz_id = metadata.get('musicbrainz_id')
                musicbrainz_artist_id = metadata.get('musicbrainz_artist_id')
                musicbrainz_album_id = metadata.get('musicbrainz_album_id')
                spotify_id = metadata.get('spotify_id')
                discogs_id = metadata.get('discogs_id')
                
                # Convert metadata to JSON
                metadata_json = json.dumps(metadata) if metadata else None
                enrichment_sources = metadata.get('enrichment_sources', [])
                enrichment_sources_json = json.dumps(enrichment_sources) if enrichment_sources else None
                
                # Insert or replace external metadata in component table
                cursor.execute("""
                    INSERT OR REPLACE INTO external_metadata 
                    (track_id, musicbrainz_id, musicbrainz_artist_id, musicbrainz_album_id,
                     spotify_id, discogs_id, metadata, enrichment_sources, last_updated, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    track_id,
                    musicbrainz_id,
                    musicbrainz_artist_id,
                    musicbrainz_album_id,
                    spotify_id,
                    discogs_id,
                    metadata_json,
                    enrichment_sources_json
                ))
                
                conn.commit()
                log_universal('DEBUG', 'Database', f'Saved external metadata to component table for: {file_path}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save external metadata: {e}')
            return False

    @log_function_call
    def save_audio_analysis(self, file_path: str, analysis_type: str, features: Dict[str, Any]) -> bool:
        """
        Save audio analysis features to component table.
        
        Args:
            file_path: Path to the file
            analysis_type: Type of analysis ('librosa', 'custom', 'advanced')
            features: Analysis features dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get track ID
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                track_row = cursor.fetchone()
                if not track_row:
                    log_universal('WARNING', 'Database', f'Track not found for audio analysis: {file_path}')
                    return False
                
                track_id = track_row[0]
                
                # Convert features to JSON
                features_json = json.dumps(features) if features else None
                confidence = features.get('confidence', 0.0)
                processing_time = features.get('processing_time', 0.0)
                
                # Insert or replace audio analysis in component table
                cursor.execute("""
                    INSERT OR REPLACE INTO audio_analysis 
                    (track_id, analysis_type, features, confidence, processing_time, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    track_id,
                    analysis_type,
                    features_json,
                    confidence,
                    processing_time
                ))
                
                conn.commit()
                log_universal('DEBUG', 'Database', f'Saved {analysis_type} analysis to component table for: {file_path}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save audio analysis: {e}')
            return False

    @log_function_call
    def save_advanced_categorization_features(self, file_path: str, features: Dict[str, Any]) -> bool:
        """
        Save advanced categorization features to audio analysis component table.
        
        Args:
            file_path: Path to the file
            features: Advanced categorization features dictionary
            
        Returns:
            True if successful, False otherwise
        """
        return self.save_audio_analysis(file_path, 'advanced_categorization', features)

    @log_function_call
    def save_spotify_features(self, file_path: str, features: Dict[str, Any]) -> bool:
        """
        Save Spotify-style features to audio analysis component table and update main table.
        
        Args:
            file_path: Path to the file
            features: Spotify-style features dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get track ID
                cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                track_row = cursor.fetchone()
                if not track_row:
                    log_universal('WARNING', 'Database', f'Track not found for Spotify features: {file_path}')
                    return False
                
                track_id = track_row[0]
                
                # Save to audio analysis component table
                success = self.save_audio_analysis(file_path, 'spotify', features)
                if not success:
                    return False
                
                # Update main tracks table with essential features for playlist queries
                if features is not None:
                    cursor.execute("""
                        UPDATE tracks SET
                            danceability = COALESCE(?, danceability),
                            energy = COALESCE(?, energy),
                            mode = COALESCE(?, mode),
                            acousticness = COALESCE(?, acousticness),
                            instrumentalness = COALESCE(?, instrumentalness),
                            speechiness = COALESCE(?, speechiness),
                            valence = COALESCE(?, valence),
                            liveness = COALESCE(?, liveness),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (
                        features.get('danceability'),
                        features.get('energy'),
                        features.get('mode'),
                        features.get('acousticness'),
                        features.get('instrumentalness'),
                        features.get('speechiness'),
                        features.get('valence'),
                        features.get('liveness'),
                        track_id
                    ))
                
                conn.commit()
                log_universal('DEBUG', 'Database', f'Saved Spotify features to component table and updated main table for: {file_path}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save Spotify features: {e}')
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
                        'bpm', 'key', 'mode', 'energy', 'danceability', 'status',
                        'speechiness', 'liveness'
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

    @log_function_call
    def validate_data_completeness(self, file_path: str) -> Dict[str, Any]:
        """
        Validate that all analysis data is properly stored.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with validation results
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get track data
                cursor.execute("SELECT * FROM tracks WHERE file_path = ?", (file_path,))
                track_data = cursor.fetchone()
                
                if not track_data:
                    return {
                        'valid': False,
                        'error': 'Track not found in database',
                        'missing_fields': [],
                        'data_quality': 0.0
                    }
                
                track_dict = dict(track_data)
                
                # Define expected fields by category
                expected_fields = {
                    'core': ['title', 'artist', 'album', 'duration', 'file_size_bytes'],
                    'audio_features': ['bpm', 'key', 'mode', 'loudness', 'energy', 'danceability', 'valence'],
                    'rhythm': ['rhythm_confidence', 'bpm_estimates', 'bpm_intervals'],
                    'spectral': ['spectral_centroid', 'spectral_flatness', 'spectral_rolloff'],
                    'mfcc': ['mfcc_coefficients', 'mfcc_bands', 'mfcc_std'],
                    'musicnn': ['embedding', 'tags', 'musicnn_skipped'],
                    'chroma': ['chroma_mean', 'chroma_std'],
                    'metadata': ['bitrate', 'sample_rate', 'channels', 'genre', 'year']
                }
                
                missing_fields = {}
                data_quality_scores = {}
                
                for category, fields in expected_fields.items():
                    category_missing = []
                    category_present = 0
                    
                    for field in fields:
                        if field not in track_dict or track_dict[field] is None:
                            category_missing.append(field)
                        else:
                            category_present += 1
                    
                    if category_missing:
                        missing_fields[category] = category_missing
                    
                    data_quality_scores[category] = category_present / len(fields)
                
                # Calculate overall data quality
                overall_quality = sum(data_quality_scores.values()) / len(data_quality_scores)
                
                # Check for JSON parsing issues
                json_fields = [
                    'bpm_estimates', 'bpm_intervals', 'mfcc_coefficients', 'mfcc_bands',
                    'mfcc_std', 'mfcc_delta', 'mfcc_delta2', 'embedding', 'embedding_std',
                    'embedding_min', 'embedding_max', 'tags', 'chroma_mean', 'chroma_std'
                ]
                
                json_parsing_issues = []
                for field in json_fields:
                    if field in track_dict and track_dict[field]:
                        try:
                            if isinstance(track_dict[field], str):
                                json.loads(track_dict[field])
                        except (json.JSONDecodeError, TypeError):
                            json_parsing_issues.append(field)
                
                # Get tags data
                cursor.execute("""
                    SELECT source, COUNT(*) as count
                    FROM tags 
                    WHERE track_id = ?
                    GROUP BY source
                """, (track_dict['id'],))
                
                tag_sources = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    'valid': len(missing_fields) == 0 and len(json_parsing_issues) == 0,
                    'data_quality': overall_quality,
                    'missing_fields': missing_fields,
                    'data_quality_by_category': data_quality_scores,
                    'json_parsing_issues': json_parsing_issues,
                    'tag_sources': tag_sources,
                    'total_fields_checked': sum(len(fields) for fields in expected_fields.values()),
                    'fields_present': sum(len(fields) - len(missing) for fields, missing in zip(expected_fields.values(), missing_fields.values()) if fields)
                }
                
        except Exception as e:
            log_universal('ERROR', 'Database', f"Data completeness validation failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'missing_fields': [],
                'data_quality': 0.0
            }

    @log_function_call
    def migrate_to_complete_schema(self) -> Dict[str, Any]:
        """
        Migrate existing database to complete schema with all missing fields.
        
        Returns:
            Dictionary with migration results
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get current schema
                cursor.execute("PRAGMA table_info(tracks)")
                current_columns = {row[1] for row in cursor.fetchall()}
                
                # Define all missing columns from complete schema
                missing_columns = {
                    'tempo_confidence': 'REAL',
                    'tempo_strength': 'REAL',
                    'rhythm_pattern': 'TEXT',
                    'beat_positions': 'TEXT',
                    'onset_times': 'TEXT',
                    'rhythm_complexity': 'REAL',
                    'spectral_flux': 'REAL',
                    'spectral_entropy': 'REAL',
                    'spectral_crest': 'REAL',
                    'spectral_decrease': 'REAL',
                    'spectral_kurtosis': 'REAL',
                    'spectral_skewness': 'REAL',
                    'key_scale_notes': 'TEXT',
                    'key_chord_progression': 'TEXT',
                    'modulation_points': 'TEXT',
                    'tonal_centroid': 'REAL',
                    'harmonic_complexity': 'REAL',
                    'chord_progression': 'TEXT',
                    'harmonic_centroid': 'REAL',
                    'harmonic_contrast': 'REAL',
                    'chord_changes': 'INTEGER',
                    'zero_crossing_rate': 'REAL',
                    'root_mean_square': 'REAL',
                    'peak_amplitude': 'REAL',
                    'crest_factor': 'REAL',
                    'signal_to_noise_ratio': 'REAL',
                    'timbre_brightness': 'REAL',
                    'timbre_warmth': 'REAL',
                    'timbre_hardness': 'REAL',
                    'timbre_depth': 'REAL',
                    'intro_duration': 'REAL',
                    'verse_duration': 'REAL',
                    'chorus_duration': 'REAL',
                    'bridge_duration': 'REAL',
                    'outro_duration': 'REAL',
                    'section_boundaries': 'TEXT',
                    'repetition_rate': 'REAL',
                    'bitrate_quality': 'REAL',
                    'sample_rate_quality': 'REAL',
                    'encoding_quality': 'REAL',
                    'compression_artifacts': 'REAL',
                    'clipping_detection': 'REAL',
                    'electronic_elements': 'REAL',
                    'classical_period': 'TEXT',
                    'jazz_style': 'TEXT',
                    'rock_subgenre': 'TEXT',
                    'folk_style': 'TEXT',
                    'harmonic_features': 'TEXT',
                    'timbre_features': 'TEXT',
                    'structure_features': 'TEXT'
                }
                
                added_columns = []
                failed_columns = []
                
                # Add missing columns
                for column_name, column_type in missing_columns.items():
                    if column_name not in current_columns:
                        try:
                            cursor.execute(f"ALTER TABLE tracks ADD COLUMN {column_name} {column_type}")
                            added_columns.append(column_name)
                            log_universal('INFO', 'Database', f'Added column: {column_name}')
                        except Exception as e:
                            failed_columns.append(f'{column_name}: {str(e)}')
                            log_universal('ERROR', 'Database', f'Failed to add column {column_name}: {e}')
                
                # Create missing indexes
                missing_indexes = [
                    'idx_tracks_tempo_confidence',
                    'idx_tracks_rhythm_complexity',
                    'idx_tracks_harmonic_complexity',
                    'idx_tracks_timbre_brightness',
                    'idx_tracks_spectral_flux',
                    'idx_tracks_root_mean_square'
                ]
                
                created_indexes = []
                failed_indexes = []
                
                for index_name in missing_indexes:
                    try:
                        column_name = index_name.replace('idx_tracks_', '')
                        cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON tracks({column_name})")
                        created_indexes.append(index_name)
                        log_universal('INFO', 'Database', f'Created index: {index_name}')
                    except Exception as e:
                        failed_indexes.append(f'{index_name}: {str(e)}')
                        log_universal('ERROR', 'Database', f'Failed to create index {index_name}: {e}')
                
                conn.commit()
                
                return {
                    'success': len(failed_columns) == 0 and len(failed_indexes) == 0,
                    'added_columns': added_columns,
                    'failed_columns': failed_columns,
                    'created_indexes': created_indexes,
                    'failed_indexes': failed_indexes,
                    'total_columns_added': len(added_columns),
                    'total_indexes_created': len(created_indexes)
                }
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Schema migration failed: {e}')
            return {
                'success': False,
                'error': str(e),
                'added_columns': [],
                'failed_columns': [str(e)],
                'created_indexes': [],
                'failed_indexes': [str(e)]
            }

    @log_function_call
    def validate_all_data(self) -> Dict[str, Any]:
        """
        Validate data completeness for all tracks in database.
        
        Returns:
            Dictionary with validation results
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get all tracks
                cursor.execute("SELECT file_path FROM tracks WHERE status = 'analyzed'")
                tracks = cursor.fetchall()
                
                validation_results = []
                total_tracks = len(tracks)
                valid_tracks = 0
                invalid_tracks = 0
                
                for track in tracks:
                    file_path = track[0]
                    result = self.validate_data_completeness(file_path)
                    validation_results.append({
                        'file_path': file_path,
                        'valid': result['valid'],
                        'data_quality': result['data_quality'],
                        'missing_fields': result['missing_fields']
                    })
                    
                    if result['valid']:
                        valid_tracks += 1
                    else:
                        invalid_tracks += 1
                
                # Calculate overall statistics
                overall_quality = sum(r['data_quality'] for r in validation_results) / len(validation_results) if validation_results else 0.0
                
                return {
                    'total_tracks': total_tracks,
                    'valid_tracks': valid_tracks,
                    'invalid_tracks': invalid_tracks,
                    'overall_quality': overall_quality,
                    'validation_rate': valid_tracks / total_tracks if total_tracks > 0 else 0.0,
                    'results': validation_results
                }
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Data validation failed: {e}')
            return {
                'error': str(e),
                'total_tracks': 0,
                'valid_tracks': 0,
                'invalid_tracks': 0,
                'overall_quality': 0.0,
                'validation_rate': 0.0,
                'results': []
            }

    @log_function_call
    def repair_corrupted_data(self) -> Dict[str, Any]:
        """
        Repair corrupted data entries in the database.
        
        Returns:
            Dictionary with repair results
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Find corrupted JSON fields
                json_fields = [
                    'bpm_estimates', 'bpm_intervals', 'mfcc_coefficients', 'mfcc_bands',
                    'mfcc_std', 'mfcc_delta', 'mfcc_delta2', 'embedding', 'embedding_std',
                    'embedding_min', 'embedding_max', 'tags', 'chroma_mean', 'chroma_std'
                ]
                
                repaired_fields = []
                failed_repairs = []
                
                for field in json_fields:
                    try:
                        # Find records with invalid JSON
                        cursor.execute(f"""
                            SELECT id, {field} FROM tracks 
                            WHERE {field} IS NOT NULL 
                            AND {field} != ''
                            AND {field} != 'null'
                        """)
                        
                        records = cursor.fetchall()
                        
                        for record_id, field_value in records:
                            try:
                                # Try to parse JSON
                                if isinstance(field_value, str):
                                    json.loads(field_value)
                            except (json.JSONDecodeError, TypeError):
                                # Invalid JSON - set to NULL
                                cursor.execute(f"UPDATE tracks SET {field} = NULL WHERE id = ?", (record_id,))
                                repaired_fields.append(f'{field}:{record_id}')
                                log_universal('INFO', 'Database', f'Repaired invalid JSON in {field} for record {record_id}')
                        
                    except Exception as e:
                        failed_repairs.append(f'{field}: {str(e)}')
                        log_universal('ERROR', 'Database', f'Failed to repair {field}: {e}')
                
                # Fix NULL values in required fields
                required_fields = ['title', 'artist']
                for field in required_fields:
                    try:
                        cursor.execute(f"UPDATE tracks SET {field} = 'Unknown' WHERE {field} IS NULL")
                        affected = cursor.rowcount
                        if affected > 0:
                            repaired_fields.append(f'{field}:{affected}_records')
                            log_universal('INFO', 'Database', f'Fixed {affected} NULL values in {field}')
                    except Exception as e:
                        failed_repairs.append(f'{field}: {str(e)}')
                
                conn.commit()
                
                return {
                    'success': len(failed_repairs) == 0,
                    'repaired_fields': repaired_fields,
                    'failed_repairs': failed_repairs,
                    'total_repairs': len(repaired_fields)
                }
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Data repair failed: {e}')
            return {
                'success': False,
                'error': str(e),
                'repaired_fields': [],
                'failed_repairs': [str(e)],
                'total_repairs': 0
            }

    @log_function_call
    def show_schema_info(self) -> Dict[str, Any]:
        """
        Show current database schema information.
        
        Returns:
            Dictionary with schema information
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get table information
                cursor.execute("PRAGMA table_info(tracks)")
                columns = cursor.fetchall()
                
                # Get index information
                cursor.execute("PRAGMA index_list(tracks)")
                indexes = cursor.fetchall()
                
                # Get view information
                cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
                views = cursor.fetchall()
                
                return {
                    'table_name': 'tracks',
                    'total_columns': len(columns),
                    'columns': [{'name': col[1], 'type': col[2], 'not_null': col[3], 'default': col[4]} for col in columns],
                    'total_indexes': len(indexes),
                    'indexes': [idx[1] for idx in indexes],
                    'total_views': len(views),
                    'views': [view[0] for view in views]
                }
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Schema info failed: {e}')
            return {
                'error': str(e),
                'table_name': 'tracks',
                'total_columns': 0,
                'columns': [],
                'total_indexes': 0,
                'indexes': [],
                'total_views': 0,
                'views': []
            }


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
