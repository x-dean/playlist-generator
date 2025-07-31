"""
Simplified file discovery for audio files.
Based on the original working implementation from old_working_setup.
"""

import os
import hashlib
import json
from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

# Import configuration loader and logging
from .config_loader import config_loader
from .logging_setup import get_logger, log_function_call, log_performance
from .database import db_manager

logger = get_logger('playlista.file_discovery')


class FileDiscovery:
    """
    Simplified file discovery for audio analysis.
    
    Handles:
    - Directory scanning for audio files
    - File validation and filtering
    - Exclusion of failed files directory
    - Change tracking and file info
    - Database integration for persistence
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize file discovery with configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        if config is None:
            config = config_loader.load_config()
        
        # Fixed paths for Docker environment
        self.music_dir = '/music'  # Fixed Docker path
        self.failed_dir = '/app/cache/failed_dir'  # Fixed Docker path
        self.db_path = '/app/cache/playlista.db'  # Fixed Docker path
        
        # Configuration settings
        self.min_file_size_bytes = config.get('MIN_FILE_SIZE_BYTES', 1024)
        self.valid_extensions = config.get('VALID_EXTENSIONS', 
                                         ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus'])
        self.hash_algorithm = config.get('HASH_ALGORITHM', 'md5')
        self.max_retry_count = config.get('MAX_RETRY_COUNT', 3)
        self.enable_recursive_scan = config.get('ENABLE_RECURSIVE_SCAN', True)
        self.log_level = config.get('LOG_LEVEL', 'INFO')
        self.enable_detailed_logging = config.get('ENABLE_DETAILED_LOGGING', True)
        
        self.current_files = set()
        
        # Database manager is already initialized globally
        logger.info(f"FileDiscovery initialized with config:")
        logger.info(f"  Music directory: {self.music_dir} (fixed)")
        logger.info(f"  Failed directory: {self.failed_dir} (fixed)")
        logger.info(f"  Database path: {self.db_path} (fixed)")
        logger.info(f"  Min file size: {self.min_file_size_bytes} bytes")
        logger.info(f"  Valid extensions: {', '.join(self.valid_extensions)}")
        logger.info(f"  Hash algorithm: {self.hash_algorithm}")
        logger.info(f"  Max retry count: {self.max_retry_count}")
        logger.info(f"  Recursive scan: {self.enable_recursive_scan}")

    @log_function_call
    def _get_file_hash(self, filepath: str) -> str:
        """
        Generate a hash for a file based on filename, modification time, and size.
        This allows files to be moved/reorganized without being treated as new files.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Hash string based on filename + modification time + size
        """
        logger.debug(f"Generating hash for: {filepath}")
        try:
            stat = os.stat(filepath)
            filename = os.path.basename(filepath)
            # Use filename + modification time + size for hash
            # This allows files to be moved without being treated as new
            content = f"{filename}:{stat.st_mtime}:{stat.st_size}"
            
            # Use configurable hash algorithm
            if self.hash_algorithm.lower() == 'md5':
                hash_result = hashlib.md5(content.encode()).hexdigest()
            elif self.hash_algorithm.lower() == 'sha1':
                hash_result = hashlib.sha1(content.encode()).hexdigest()
            elif self.hash_algorithm.lower() == 'sha256':
                hash_result = hashlib.sha256(content.encode()).hexdigest()
            else:
                # Default to MD5
                hash_result = hashlib.md5(content.encode()).hexdigest()
            
            logger.debug(f"Generated hash: {hash_result[:8]}... for {filename}")
            return hash_result
            
        except Exception as e:
            logger.error(f"Error generating hash for {filepath}: {e}")
            # Fallback to filename-based hash
            return hashlib.md5(os.path.basename(filepath).encode()).hexdigest()

    @log_function_call
    def _is_valid_audio_file(self, filepath: str) -> bool:
        """
        Check if a file is a valid audio file for analysis.
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if valid audio file, False otherwise
        """
        try:
            # Check file extension
            file_ext = os.path.splitext(filepath)[1].lower()
            if file_ext not in self.valid_extensions:
                logger.debug(f"Invalid extension: {file_ext}")
                return False
            
            # Check file size
            file_size = os.path.getsize(filepath)
            if file_size < self.min_file_size_bytes:
                logger.debug(f"File too small: {file_size} bytes")
                return False
            
            # Check if file is readable
            if not os.access(filepath, os.R_OK):
                logger.debug(f"File not readable: {filepath}")
                return False
            
            # Check if file is in failed directory
            if self._is_in_excluded_directory(filepath):
                logger.debug(f"File in excluded directory: {filepath}")
                return False
            
            logger.debug(f"Valid audio file: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {filepath}: {e}")
            return False

    @log_function_call
    def _is_in_excluded_directory(self, filepath: str) -> bool:
        """
        Check if file is in an excluded directory (like failed files).
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if in excluded directory, False otherwise
        """
        try:
            filepath_abs = os.path.abspath(filepath)
            failed_dir_abs = os.path.abspath(self.failed_dir)
            
            # Check if file is in failed directory
            if filepath_abs.startswith(failed_dir_abs):
                return True
            
            # Check for other excluded patterns
            excluded_patterns = [
                '/tmp/',
                '/temp/',
                '/cache/',
                '/.Trash/',
                '/.recycle/'
            ]
            
            for pattern in excluded_patterns:
                if pattern in filepath_abs:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking excluded directory for {filepath}: {e}")
            return False

    @log_function_call
    def discover_files(self) -> List[str]:
        """
        Discover audio files in the music directory.
        
        Returns:
            List of valid audio file paths
        """
        logger.info(f"Starting file discovery in: {self.music_dir}")
        
        if not os.path.exists(self.music_dir):
            logger.warning(f"Music directory does not exist: {self.music_dir}")
            return []
        
        discovered_files = []
        
        try:
            if self.enable_recursive_scan:
                logger.debug("Scanning recursively...")
                for root, dirs, files in os.walk(self.music_dir):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not self._is_in_excluded_directory(os.path.join(root, d))]
                    
                    for file in files:
                        filepath = os.path.join(root, file)
                        if self._is_valid_audio_file(filepath):
                            discovered_files.append(filepath)
            else:
                logger.debug("Scanning non-recursively...")
                for file in os.listdir(self.music_dir):
                    filepath = os.path.join(self.music_dir, file)
                    if os.path.isfile(filepath) and self._is_valid_audio_file(filepath):
                        discovered_files.append(filepath)
            
            # Sort files by size (largest first for better progress indication)
            discovered_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
            
            logger.info(f"File discovery complete: {len(discovered_files)} files found")
            return discovered_files
            
        except Exception as e:
            logger.error(f"File discovery failed: {e}")
            return []

    @log_function_call
    def save_discovered_files_to_db(self, filepaths: List[str]) -> Dict[str, int]:
        """
        Save discovered files to database with tracking information.
        
        Args:
            filepaths: List of file paths to save
            
        Returns:
            Dictionary with save statistics
        """
        if not filepaths:
            logger.info("No files to save to database")
            return {'new': 0, 'updated': 0, 'unchanged': 0, 'errors': 0}
        
        logger.info(f"Starting database save for {len(filepaths)} files...")
        
        stats = {'new': 0, 'updated': 0, 'unchanged': 0, 'errors': 0}
        
        try:
            logger.debug("Database connection established")
            
            for filepath in filepaths:
                try:
                    # Get file information
                    stat = os.stat(filepath)
                    filename = os.path.basename(filepath)
                    file_size = stat.st_size
                    file_hash = self._get_file_hash(filepath)
                    modified_time = stat.st_mtime
                    
                    # Check if file with same hash already exists in database
                    existing_result = db_manager.get_analysis_result(filepath)
                    
                    if existing_result:
                        existing_hash = existing_result.get('file_hash')
                        existing_size = existing_result.get('file_size_bytes')
                        
                        if existing_hash == file_hash and existing_size == file_size:
                            # File unchanged
                            stats['unchanged'] += 1
                            logger.debug(f"File unchanged: {filename}")
                        else:
                            # File modified
                            db_manager.save_analysis_result(
                                file_path=filepath,
                                filename=filename,
                                file_size_bytes=file_size,
                                file_hash=file_hash,
                                analysis_data={'status': 'discovered', 'modified_time': modified_time},
                                metadata={'discovered_date': datetime.now().isoformat()}
                            )
                            stats['updated'] += 1
                            logger.debug(f"Updated modified file in database: {filepath}")
                    else:
                        # New file
                        db_manager.save_analysis_result(
                            file_path=filepath,
                            filename=filename,
                            file_size_bytes=file_size,
                            file_hash=file_hash,
                            analysis_data={'status': 'discovered', 'modified_time': modified_time},
                            metadata={'discovered_date': datetime.now().isoformat()}
                        )
                        stats['new'] += 1
                        logger.debug(f"Saved new file to database: {filepath}")
                        
                except Exception as e:
                    stats['errors'] += 1
                    logger.error(f"Error saving file {filepath} to database: {e}")
            
            logger.debug("Committing database changes...")
            logger.debug("Database changes committed")
            
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            stats['errors'] = len(filepaths)
        
        logger.info(f"Database save complete:")
        logger.info(f"  New files: {stats['new']}")
        logger.info(f"  Updated files: {stats['updated']}")
        logger.info(f"  Unchanged files: {stats['unchanged']}")
        logger.info(f"  Errors: {stats['errors']}")
        
        return stats

    @log_function_call
    def get_db_files(self) -> Set[str]:
        """
        Get all files currently in the database.
        
        Returns:
            Set of file paths from database
        """
        logger.debug("Retrieving files from database...")
        
        try:
            results = db_manager.get_all_analysis_results()
            file_paths = {result['file_path'] for result in results 
                         if result.get('analysis_data', {}).get('status') == 'discovered'}
            
            logger.debug(f"Retrieved {len(file_paths)} valid files from database")
            return file_paths
            
        except Exception as e:
            logger.error(f"Error reading from database: {e}")
            return set()

    @log_function_call
    def get_failed_files(self) -> Set[str]:
        """
        Get all files marked as failed in the database.
        
        Returns:
            Set of failed file paths from database
        """
        logger.debug("Retrieving failed files from database...")
        
        try:
            failed_files = db_manager.get_failed_analysis_files()
            failed_paths = {failed_file['file_path'] for failed_file in failed_files}
            
            logger.debug(f"Retrieved {len(failed_paths)} failed files from database")
            return failed_paths
            
        except Exception as e:
            logger.error(f"Error reading failed files from database: {e}")
            return set()

    @log_function_call
    def mark_file_as_failed(self, filepath: str, error_message: str = "Unknown error"):
        """
        Mark a file as failed in the database.
        
        Args:
            filepath: Path to the failed file
            error_message: Error message describing the failure
        """
        logger.info(f"Marking file as failed: {filepath}")
        
        try:
            filename = os.path.basename(filepath)
            file_hash = self._get_file_hash(filepath)
            
            # Use DatabaseManager's failed analysis tracking
            db_manager.mark_analysis_failed(filepath, filename, error_message)
            
            logger.info(f"Successfully marked file as failed: {filename}")
            
        except Exception as e:
            logger.error(f"Error marking file as failed {filepath}: {e}")

    @log_function_call
    def cleanup_removed_files_from_db(self) -> int:
        """
        Remove files from database that no longer exist on disk.
        
        Returns:
            Number of files removed from database
        """
        logger.info("Starting cleanup of removed files from database...")
        
        try:
            logger.debug("Database connection established for cleanup")
            
            # Get all files from database
            db_files = self.get_db_files()
            logger.debug(f"Found {len(db_files)} files in database")
            
            # Find files that no longer exist on disk
            removed_files = []
            for filepath in db_files:
                if not os.path.exists(filepath):
                    removed_files.append(filepath)
            
            if removed_files:
                logger.info(f"Removing {len(removed_files)} non-existent files from database")
                
                # Remove from analysis results and failed analysis
                for filepath in removed_files:
                    try:
                        # Remove from analysis results
                        db_manager.delete_analysis_result(filepath)
                        # Remove from failed analysis
                        db_manager.delete_failed_analysis(filepath)
                        logger.debug(f"Removed from database: {filepath}")
                    except Exception as e:
                        logger.error(f"Error removing file {filepath}: {e}")
                
                logger.info(f"Removed {len(removed_files)} files from database that no longer exist")
                return len(removed_files)
            else:
                logger.info("No files to remove - all database files still exist on disk")
                return 0
                
        except Exception as e:
            logger.error(f"Error cleaning up database: {e}")
            return 0

    @log_function_call
    def get_files_for_analysis(self, force: bool = False, failed_mode: bool = False) -> List[str]:
        """
        Get files that need analysis based on current state and database.
        
        Args:
            force: Force re-analysis of all files
            failed_mode: Only return files that previously failed
            
        Returns:
            List of file paths that need analysis
        """
        logger.info(f"Getting files for analysis (force={force}, failed_mode={failed_mode})")
        
        try:
            if failed_mode:
                # Get only failed files
                failed_files = self.get_failed_files()
                logger.info(f"Found {len(failed_files)} failed files for retry")
                return list(failed_files)
            
            # Get files from database
            logger.debug("Retrieving files from database...")
            db_files = self.get_db_files()
            failed_db_files = self.get_failed_files()
            logger.debug(f"Database has {len(db_files)} valid files and {len(failed_db_files)} failed files")
            
            if force:
                # Force re-analysis of all files
                all_files = db_files.union(failed_db_files)
                logger.info(f"Force mode: {len(all_files)} files for re-analysis")
                return list(all_files)
            
            # Get current files on disk
            current_files = set(self.discover_files())
            logger.debug(f"Found {len(current_files)} files on disk")
            
            # Files that need analysis:
            # 1. New files on disk not in database
            new_files = current_files - db_files - failed_db_files
            
            # 2. Failed files (for retry)
            retry_files = failed_db_files.intersection(current_files)
            
            # 3. Files that have been modified (different hash)
            modified_files = set()
            for filepath in current_files.intersection(db_files):
                try:
                    current_hash = self._get_file_hash(filepath)
                    existing_result = db_manager.get_analysis_result(filepath)
                    if existing_result and existing_result.get('file_hash') != current_hash:
                        modified_files.add(filepath)
                except Exception as e:
                    logger.error(f"Error checking file hash for {filepath}: {e}")
            
            files_for_analysis = list(new_files.union(retry_files).union(modified_files))
            
            logger.info(f"Analysis queue prepared:")
            logger.info(f"  New files: {len(new_files)}")
            logger.info(f"  Retry files: {len(retry_files)}")
            logger.info(f"  Modified files: {len(modified_files)}")
            logger.info(f"  Total for analysis: {len(files_for_analysis)}")
            
            return files_for_analysis
            
        except Exception as e:
            logger.error(f"Error getting files for analysis: {e}")
            return []

    @log_function_call
    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get detailed information about a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dictionary with file information
        """
        try:
            stat = os.stat(filepath)
            filename = os.path.basename(filepath)
            file_hash = self._get_file_hash(filepath)
            
            # Get database info
            db_info = db_manager.get_analysis_result(filepath)
            
            file_info = {
                'filepath': filepath,
                'filename': filename,
                'file_size_bytes': stat.st_size,
                'file_size_mb': stat.st_size / (1024 * 1024),
                'modified_time': stat.st_mtime,
                'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'file_hash': file_hash,
                'is_valid': self._is_valid_audio_file(filepath),
                'database_info': db_info
            }
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting file info for {filepath}: {e}")
            return {}

    @log_function_call
    def validate_file_paths(self, filepaths: List[str]) -> List[str]:
        """
        Validate a list of file paths and return only valid ones.
        
        Args:
            filepaths: List of file paths to validate
            
        Returns:
            List of valid file paths
        """
        logger.info(f"Validating {len(filepaths)} file paths...")
        
        valid_files = []
        invalid_files = []
        
        for filepath in filepaths:
            if self._is_valid_audio_file(filepath):
                valid_files.append(filepath)
            else:
                invalid_files.append(filepath)
        
        logger.info(f"Validation complete:")
        logger.info(f"  Valid files: {len(valid_files)}")
        logger.info(f"  Invalid files: {len(invalid_files)}")
        
        if invalid_files:
            logger.warning(f"Invalid files: {invalid_files[:5]}{'...' if len(invalid_files) > 5 else ''}")
        
        return valid_files

    @log_function_call
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get discovery statistics including database information.
        
        Returns:
            Dictionary with statistics
        """
        logger.debug("Generating discovery statistics...")
        
        try:
            # Get current files on disk
            current_files = self.discover_files()
            
            # Get database statistics
            logger.debug("Retrieving database statistics...")
            db_files = self.get_db_files()
            failed_files = self.get_failed_files()
            db_files_count = len(db_files)
            failed_files_count = len(failed_files)
            
            logger.debug(f"Database files: {db_files_count} valid, {failed_files_count} failed")
            
            # Calculate statistics
            stats = {
                'current_files': len(current_files),
                'database_files': db_files_count,
                'failed_files': failed_files_count,
                'new_files': len(set(current_files) - db_files - failed_files),
                'removed_files': len(db_files - set(current_files)),
                'database_path': self.db_path,
                'music_directory': self.music_dir,
                'failed_directory': self.failed_dir,
                'min_file_size_bytes': self.min_file_size_bytes,
                'valid_extensions': self.valid_extensions,
                'hash_algorithm': self.hash_algorithm,
                'max_retry_count': self.max_retry_count,
                'enable_recursive_scan': self.enable_recursive_scan
            }
            
            logger.info(f"Statistics generated successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {
                'current_files': 0,
                'database_files': 0,
                'failed_files': 0,
                'new_files': 0,
                'removed_files': 0,
                'database_path': self.db_path,
                'music_directory': self.music_dir,
                'failed_directory': self.failed_dir,
                'error': str(e)
            } 