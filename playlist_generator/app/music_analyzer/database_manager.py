#!/usr/bin/env python3
"""
Database manager for audio analysis features.
"""

import sqlite3
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class AudioDatabaseManager:
    """Manage database operations for audio analysis features."""
    
    def __init__(self, cache_file: str):
        """Initialize the database manager.
        
        Args:
            cache_file (str): Path to the cache database file.
        """
        self.cache_file = cache_file
        self._init_db()
        self._check_and_fix_schema()
    
    def _init_db(self):
        """Initialize the database with required tables."""
        logger.debug(f"Initializing audio database: {self.cache_file}")
        try:
            # Ensure the database directory exists
            db_dir = os.path.dirname(self.cache_file)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created database directory: {db_dir}")
            
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                # Check if tables exist and have correct schema
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = [row[0] for row in cursor.fetchall()]
                logger.debug(f"Existing tables: {existing_tables}")
                
                # Create audio_features table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audio_features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filepath TEXT UNIQUE NOT NULL,
                        file_hash TEXT,
                        file_size INTEGER,
                        features TEXT NOT NULL,
                        metadata TEXT,
                        failed INTEGER DEFAULT 0,
                        musicnn_skipped INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create file_discovery_state table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS file_discovery_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filepath TEXT UNIQUE NOT NULL,
                        file_size INTEGER,
                        last_modified REAL,
                        discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Verify table structure
                cursor.execute("PRAGMA table_info(audio_features)")
                audio_features_columns = [row[1] for row in cursor.fetchall()]
                logger.debug(f"Audio_features table columns: {audio_features_columns}")
                
                cursor.execute("PRAGMA table_info(file_discovery_state)")
                file_discovery_columns = [row[1] for row in cursor.fetchall()]
                logger.debug(f"File_discovery_state table columns: {file_discovery_columns}")
                
                conn.commit()
                logger.info("Audio database initialization completed successfully")
                
        except Exception as e:
            logger.error(f"Audio database initialization failed: {str(e)}")
            import traceback
            logger.error(f"Audio database init error traceback: {traceback.format_exc()}")
            raise
    
    def _check_and_fix_schema(self):
        """Check and fix database schema if needed."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                # Check audio_features table
                cursor.execute("PRAGMA table_info(audio_features)")
                audio_features_columns = [row[1] for row in cursor.fetchall()]
                
                if 'filepath' not in audio_features_columns:
                    logger.warning("Audio_features table missing 'filepath' column, recreating table")
                    cursor.execute("DROP TABLE IF EXISTS audio_features")
                    cursor.execute("""
                        CREATE TABLE audio_features (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filepath TEXT UNIQUE NOT NULL,
                            file_hash TEXT,
                            file_size INTEGER,
                            features TEXT NOT NULL,
                            metadata TEXT,
                            failed INTEGER DEFAULT 0,
                            musicnn_skipped INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                
                # Check file_discovery_state table
                cursor.execute("PRAGMA table_info(file_discovery_state)")
                file_discovery_columns = [row[1] for row in cursor.fetchall()]
                
                if 'filepath' not in file_discovery_columns:
                    logger.warning("File_discovery_state table missing 'filepath' column, recreating table")
                    cursor.execute("DROP TABLE IF EXISTS file_discovery_state")
                    cursor.execute("""
                        CREATE TABLE file_discovery_state (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filepath TEXT UNIQUE NOT NULL,
                            file_size INTEGER,
                            last_modified REAL,
                            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                
                conn.commit()
                logger.info("Database schema check completed")
                
        except Exception as e:
            logger.error(f"Schema check failed: {e}")
            import traceback
            logger.error(f"Schema check error traceback: {traceback.format_exc()}")
    
    def save_features(self, file_info: Dict[str, Any], features: Dict[str, Any], failed: int = 0) -> bool:
        """Save features to the database.
        
        Args:
            file_info (Dict[str, Any]): File information.
            features (Dict[str, Any]): Extracted features.
            failed (int): Failed flag (0=success, 1=failed).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                filepath = file_info['filepath']
                file_hash = file_info.get('file_hash', '')
                file_size = file_info.get('file_size', 0)
                
                # Convert features to JSON
                features_json = json.dumps(features)
                metadata_json = json.dumps(file_info.get('metadata', {}))
                
                cursor.execute("""
                    INSERT OR REPLACE INTO audio_features 
                    (filepath, file_hash, file_size, features, metadata, failed, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (filepath, file_hash, file_size, features_json, metadata_json, failed))
                
                conn.commit()
                logger.debug(f"Successfully saved features for {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save features for {file_info.get('filepath', 'unknown')}: {e}")
            return False
    
    def get_features(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Get features for a specific file.
        
        Args:
            filepath (str): File path.
            
        Returns:
            Optional[Dict[str, Any]]: Features dictionary or None.
        """
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT features, metadata, failed, musicnn_skipped
                    FROM audio_features 
                    WHERE filepath = ?
                """, (filepath,))
                
                result = cursor.fetchone()
                if result:
                    features = json.loads(result[0])
                    metadata = json.loads(result[1]) if result[1] else {}
                    failed = result[2]
                    musicnn_skipped = result[3]
                    
                    return {
                        'features': features,
                        'metadata': metadata,
                        'failed': failed,
                        'musicnn_skipped': musicnn_skipped
                    }
                
        except Exception as e:
            logger.error(f"Failed to get features for {filepath}: {e}")
        
        return None
    
    def get_all_features(self, include_failed: bool = False) -> List[Dict[str, Any]]:
        """Get all features from the database.
        
        Args:
            include_failed (bool): Whether to include failed files.
            
        Returns:
            List[Dict[str, Any]]: List of feature dictionaries.
        """
        try:
            logger.debug(f"Getting all features from database: {self.cache_file}")
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                # First, let's check if the table exists and what columns it has
                cursor.execute("PRAGMA table_info(audio_features)")
                columns = cursor.fetchall()
                logger.debug(f"Audio_features table columns: {[col[1] for col in columns]}")
                
                if include_failed:
                    cursor.execute("""
                        SELECT filepath, features, metadata, failed, musicnn_skipped
                        FROM audio_features
                    """)
                else:
                    cursor.execute("""
                        SELECT filepath, features, metadata, failed, musicnn_skipped
                        FROM audio_features
                        WHERE failed = 0
                    """)
                
                results = []
                for row in cursor.fetchall():
                    filepath, features_json, metadata_json, failed, musicnn_skipped = row
                    
                    features = json.loads(features_json)
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    # Add filepath to the features dict for consistency
                    features['filepath'] = filepath
                    features['metadata'] = metadata
                    features['failed'] = failed
                    features['musicnn_skipped'] = musicnn_skipped
                    
                    results.append(features)
                
                logger.debug(f"Retrieved {len(results)} features from database")
                return results
                
        except Exception as e:
            logger.error(f"Failed to get all features: {e}")
            import traceback
            logger.error(f"Get all features error traceback: {traceback.format_exc()}")
            return []
    
    def mark_as_failed(self, filepath: str) -> bool:
        """Mark a file as failed in the database.
        
        Args:
            filepath (str): File path.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE audio_features 
                    SET failed = 1, updated_at = CURRENT_TIMESTAMP
                    WHERE filepath = ?
                """, (filepath,))
                
                conn.commit()
                logger.debug(f"Marked {filepath} as failed")
                return True
                
        except Exception as e:
            logger.error(f"Failed to mark {filepath} as failed: {e}")
            return False
    
    def mark_musicnn_skipped(self, filepath: str) -> bool:
        """Mark a file as having skipped MusicNN processing.
        
        Args:
            filepath (str): File path.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE audio_features 
                    SET musicnn_skipped = 1, updated_at = CURRENT_TIMESTAMP
                    WHERE filepath = ?
                """, (filepath,))
                
                conn.commit()
                logger.debug(f"Marked {filepath} as MusicNN skipped")
                return True
                
        except Exception as e:
            logger.error(f"Failed to mark {filepath} as MusicNN skipped: {e}")
            return False
    
    def unmark_as_failed(self, filepath: str) -> bool:
        """Unmark a file as failed in the database.
        
        Args:
            filepath (str): File path.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE audio_features 
                    SET failed = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE filepath = ?
                """, (filepath,))
                
                conn.commit()
                logger.debug(f"Unmarked {filepath} as failed")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unmark {filepath} as failed: {e}")
            return False
    
    def unmark_musicnn_skipped(self, filepath: str) -> bool:
        """Unmark a file as having skipped MusicNN processing.
        
        Args:
            filepath (str): File path.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE audio_features 
                    SET musicnn_skipped = 0, updated_at = CURRENT_TIMESTAMP
                    WHERE filepath = ?
                """, (filepath,))
                
                conn.commit()
                logger.debug(f"Unmarked {filepath} as MusicNN skipped")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unmark {filepath} as MusicNN skipped: {e}")
            return False
    
    def get_failed_files(self) -> List[str]:
        """Get list of failed files.
        
        Returns:
            List[str]: List of failed file paths.
        """
        try:
            logger.debug(f"Getting failed files from database: {self.cache_file}")
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                # First, let's check if the table exists and what columns it has
                cursor.execute("PRAGMA table_info(audio_features)")
                columns = cursor.fetchall()
                logger.debug(f"Audio_features table columns: {[col[1] for col in columns]}")
                
                cursor.execute("""
                    SELECT filepath FROM audio_features WHERE failed = 1
                """)
                
                failed_files = [row[0] for row in cursor.fetchall()]
                logger.debug(f"Retrieved {len(failed_files)} failed files from database")
                return failed_files
                
        except Exception as e:
            logger.error(f"Failed to get failed files: {e}")
            import traceback
            logger.error(f"Get failed files error traceback: {traceback.format_exc()}")
            return []
    
    def get_files_with_skipped_musicnn(self) -> List[str]:
        """Get list of files that skipped MusicNN processing.
        
        Returns:
            List[str]: List of file paths that skipped MusicNN.
        """
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT filepath FROM audio_features WHERE musicnn_skipped = 1
                """)
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get files with skipped MusicNN: {e}")
            return []
    
    def cleanup_database(self) -> List[str]:
        """Remove entries for files that no longer exist.
        
        Returns:
            List[str]: List of removed file paths.
        """
        try:
            logger.debug(f"Cleaning up database: {self.cache_file}")
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                # First, let's check if the table exists and what columns it has
                cursor.execute("PRAGMA table_info(audio_features)")
                columns = cursor.fetchall()
                logger.debug(f"Audio_features table columns: {[col[1] for col in columns]}")
                
                # Get all filepaths from database
                cursor.execute("SELECT filepath FROM audio_features")
                db_files = [row[0] for row in cursor.fetchall()]
                
                # Check which files still exist
                missing_files = []
                for filepath in db_files:
                    if not os.path.exists(filepath):
                        missing_files.append(filepath)
                
                # Remove missing files from database
                if missing_files:
                    placeholders = ','.join(['?' for _ in missing_files])
                    cursor.execute(f"""
                        DELETE FROM audio_features 
                        WHERE filepath IN ({placeholders})
                    """, missing_files)
                    
                    conn.commit()
                    logger.info(f"Removed {len(missing_files)} missing files from database")
                
                return missing_files
                
        except Exception as e:
            logger.error(f"Failed to cleanup database: {e}")
            import traceback
            logger.error(f"Cleanup database error traceback: {traceback.format_exc()}")
            return []
    
    def update_file_discovery_state(self, file_paths: List[str]) -> None:
        """Update file discovery state in the database.
        
        Args:
            file_paths (List[str]): List of file paths to update.
        """
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                for filepath in file_paths:
                    try:
                        stat = os.stat(filepath)
                        file_size = stat.st_size
                        last_modified = stat.st_mtime
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO file_discovery_state 
                            (filepath, file_size, last_modified, discovered_at)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """, (filepath, file_size, last_modified))
                        
                    except OSError:
                        # File doesn't exist, skip it
                        continue
                
                conn.commit()
                logger.debug(f"Updated file discovery state for {len(file_paths)} files")
                
        except Exception as e:
            logger.error(f"Failed to update file discovery state: {e}")
    
    def get_file_discovery_changes(self) -> Tuple[List[str], List[str], List[str]]:
        """Get file discovery changes (new, modified, removed files).
        
        Returns:
            Tuple[List[str], List[str], List[str]]: (new_files, modified_files, removed_files)
        """
        try:
            logger.debug(f"Getting file discovery changes from database: {self.cache_file}")
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                # First, let's check if the table exists and what columns it has
                cursor.execute("PRAGMA table_info(file_discovery_state)")
                columns = cursor.fetchall()
                logger.debug(f"File_discovery_state table columns: {[col[1] for col in columns]}")
                
                # Get all files in discovery state
                cursor.execute("SELECT filepath, file_size, last_modified FROM file_discovery_state")
                db_files = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
                
                new_files = []
                modified_files = []
                removed_files = []
                
                # Check current files
                for filepath in db_files.keys():
                    if not os.path.exists(filepath):
                        removed_files.append(filepath)
                    else:
                        try:
                            stat = os.stat(filepath)
                            current_size = stat.st_size
                            current_mtime = stat.st_mtime
                            
                            stored_size, stored_mtime = db_files[filepath]
                            
                            if current_size != stored_size or current_mtime != stored_mtime:
                                modified_files.append(filepath)
                                
                        except OSError:
                            removed_files.append(filepath)
                
                return new_files, modified_files, removed_files
                
        except Exception as e:
            logger.error(f"Failed to get file discovery changes: {e}")
            return [], [], []
    
    def cleanup_file_discovery_state(self) -> None:
        """Clean up file discovery state for non-existent files."""
        try:
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT filepath FROM file_discovery_state")
                db_files = [row[0] for row in cursor.fetchall()]
                
                missing_files = [f for f in db_files if not os.path.exists(f)]
                
                if missing_files:
                    placeholders = ','.join(['?' for _ in missing_files])
                    cursor.execute(f"""
                        DELETE FROM file_discovery_state 
                        WHERE filepath IN ({placeholders})
                    """, missing_files)
                    
                    conn.commit()
                    logger.debug(f"Cleaned up file discovery state for {len(missing_files)} missing files")
                
        except Exception as e:
            logger.error(f"Failed to cleanup file discovery state: {e}")
    
    def get_file_sizes_from_db(self, file_paths: List[str]) -> Dict[str, int]:
        """Get file sizes from database for given file paths.
        
        Args:
            file_paths (List[str]): List of file paths.
            
        Returns:
            Dict[str, int]: Dictionary mapping file paths to sizes.
        """
        try:
            logger.debug(f"Getting file sizes from database: {self.cache_file}")
            with sqlite3.connect(self.cache_file) as conn:
                cursor = conn.cursor()
                
                # First, let's check if the table exists and what columns it has
                cursor.execute("PRAGMA table_info(audio_features)")
                columns = cursor.fetchall()
                logger.debug(f"Audio_features table columns: {[col[1] for col in columns]}")
                
                placeholders = ','.join(['?' for _ in file_paths])
                cursor.execute(f"""
                    SELECT filepath, file_size FROM audio_features 
                    WHERE filepath IN ({placeholders})
                """, file_paths)
                
                result = {row[0]: row[1] for row in cursor.fetchall()}
                logger.debug(f"Retrieved file sizes for {len(result)} files from database")
                return result
                
        except Exception as e:
            logger.error(f"Failed to get file sizes from database: {e}")
            import traceback
            logger.error(f"Get file sizes error traceback: {traceback.format_exc()}")
            return {} 