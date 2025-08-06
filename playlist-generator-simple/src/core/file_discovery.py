"""
Simplified file discovery for audio files.
Based on the original working implementation from old_working_setup.
"""

import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import hashlib
import json
from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

# Import configuration loader and logging
from .config_loader import config_loader
from .logging_setup import get_logger, log_function_call, log_universal
from .database import get_db_manager

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
            config = config_loader.get_file_discovery_config()
        
        # Fixed paths for Docker environment (not configurable)
        self.music_dir = '/music'  # Fixed Docker path - mapped from compose
        self.failed_dir = '/app/cache/failed_dir'  # Fixed Docker path
        self.db_path = '/app/cache/playlista.db'  # Fixed Docker path
        
        # Configuration settings
        self.min_file_size_bytes = config.get('MIN_FILE_SIZE_BYTES', 10240)
        self.max_file_size_bytes = config.get('MAX_FILE_SIZE_BYTES', None)  # No limit by default
        self.valid_extensions = config.get('VALID_EXTENSIONS', 
                                         ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus'])
        self.hash_algorithm = config.get('HASH_ALGORITHM', 'md5')
        self.max_retry_count = config.get('MAX_RETRY_COUNT', 3)
        self.enable_recursive_scan = config.get('ENABLE_RECURSIVE_SCAN', True)
        self.log_level = config.get('LOG_LEVEL', 'INFO')
        self.enable_detailed_logging = config.get('ENABLE_DETAILED_LOGGING', True)
        
        # Fixed exclude directories (not configurable)
        self.exclude_directories = [self.failed_dir]  # Always exclude failed directory
        
        self.current_files = set()
        
        # Database manager is already initialized globally
        log_universal('INFO', 'FileDiscovery', 'Initialized with config:')
        log_universal('INFO', 'FileDiscovery', f'  Music directory: {self.music_dir} (fixed Docker path)')
        log_universal('INFO', 'FileDiscovery', f'  Failed directory: {self.failed_dir} (fixed Docker path)')
        log_universal('INFO', 'FileDiscovery', f'  Database path: {self.db_path} (fixed Docker path)')
        log_universal('INFO', 'FileDiscovery', f'  Min file size: {self.min_file_size_bytes} bytes')
        if self.max_file_size_bytes:
            log_universal('INFO', 'FileDiscovery', f'  Max file size: {self.max_file_size_bytes} bytes')
        log_universal('INFO', 'FileDiscovery', f'  Valid extensions: {", ".join(self.valid_extensions)}')
        log_universal('INFO', 'FileDiscovery', f'  Hash algorithm: {self.hash_algorithm}')
        log_universal('INFO', 'FileDiscovery', f'  Max retry count: {self.max_retry_count}')
        log_universal('INFO', 'FileDiscovery', f'  Recursive scan: {self.enable_recursive_scan}')
        log_universal('INFO', 'FileDiscovery', f'  Exclude directories: {", ".join(self.exclude_directories)} (fixed)')

    def override_config(self, **kwargs):
        """
        Override configuration settings for CLI arguments.
        
        Args:
            **kwargs: Configuration overrides
        """
        # Note: music_dir and exclude_directories are fixed and cannot be overridden
        
        if 'min_file_size_bytes' in kwargs:
            self.min_file_size_bytes = kwargs['min_file_size_bytes']
            log_universal('INFO', 'FileDiscovery', f'Min file size overridden: {self.min_file_size_bytes} bytes')
        
        if 'max_file_size_bytes' in kwargs:
            self.max_file_size_bytes = kwargs['max_file_size_bytes']
            log_universal('INFO', 'FileDiscovery', f'Max file size overridden: {self.max_file_size_bytes} bytes')
        
        if 'valid_extensions' in kwargs:
            self.valid_extensions = kwargs['valid_extensions']
            log_universal('INFO', 'FileDiscovery', f'Valid extensions overridden: {", ".join(self.valid_extensions)}')
        
        if 'enable_recursive_scan' in kwargs:
            self.enable_recursive_scan = kwargs['enable_recursive_scan']
            log_universal('INFO', 'FileDiscovery', f'Recursive scan overridden: {self.enable_recursive_scan}')

    @log_function_call
    def _get_file_hash(self, filepath: str) -> str:
        """
        Generate a hash for a file based on filename and size only.
        Simple, fast, and reliable approach.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Hash string based on filename + size
        """
        try:
            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath)
            # Simple approach: filename + size
            content = f"{filename}:{file_size}"
            
            # Use MD5 for consistency
            hash_result = hashlib.md5(content.encode()).hexdigest()
            
            return hash_result
            
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"Error generating hash for {filepath}: {e}")
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
                log_universal('DEBUG', 'FileDiscovery', f"Invalid extension: {file_ext}")
                return False
            
            # Check file size
            file_size = os.path.getsize(filepath)
            if file_size < self.min_file_size_bytes:
                log_universal('DEBUG', 'FileDiscovery', f"File too small: {file_size} bytes")
                return False
            
            # Check max file size if set
            if self.max_file_size_bytes and file_size > self.max_file_size_bytes:
                log_universal('DEBUG', 'FileDiscovery', f"File too large: {file_size} bytes")
                return False
            
            # Check if file is readable
            if not os.access(filepath, os.R_OK):
                log_universal('DEBUG', 'FileDiscovery', f"File not readable: {filepath}")
                return False
            
            # Check if file is in excluded directory
            if self._is_in_excluded_directory(filepath):
                log_universal('DEBUG', 'FileDiscovery', f"File in excluded directory: {filepath}")
                return False
            
            log_universal('DEBUG', 'FileDiscovery', f"Valid audio file: {filepath}")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"Error validating file {filepath}: {e}")
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
            
            # Check configured exclude directories
            for exclude_dir in self.exclude_directories:
                exclude_dir_abs = os.path.abspath(exclude_dir)
                if filepath_abs.startswith(exclude_dir_abs):
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
            log_universal('ERROR', 'FileDiscovery', f"Error checking excluded directory for {filepath}: {e}")
            return False

    @log_function_call
    def discover_files(self) -> List[str]:
        """
        Discover audio files in the music directory with caching.
        
        Returns:
            List of valid audio file paths
        """
        log_universal('INFO', 'FileDiscovery', f"Starting file discovery in: {self.music_dir}")
        
        if not os.path.exists(self.music_dir):
            log_universal('WARNING', 'FileDiscovery', f"Music directory does not exist: {self.music_dir}")
            return []
        
        # Check if we have a recent discovery cache
        db_manager = get_db_manager()
        discovery_status = db_manager.get_discovery_status(self.music_dir)
        
        if discovery_status and discovery_status.get('status') == 'completed':
            # Use cached discovery if recent (within 1 hour)
            cache_age = datetime.now() - datetime.fromisoformat(discovery_status['created_at'])
            if cache_age.total_seconds() < 3600:  # 1 hour
                log_universal('INFO', 'FileDiscovery', f"Using cached discovery from {discovery_status['created_at']}")
                # Return files from database instead of re-scanning
                return list(self.get_db_files())
        
        discovered_files = []
        start_time = datetime.now()
        
        try:
            if self.enable_recursive_scan:
                log_universal('DEBUG', 'FileDiscovery', "Scanning recursively...")
                for root, dirs, files in os.walk(self.music_dir):
                    # Skip excluded directories
                    dirs[:] = [d for d in dirs if not self._is_in_excluded_directory(os.path.join(root, d))]
                    
                    for file in files:
                        filepath = os.path.join(root, file)
                        if self._is_valid_audio_file(filepath):
                            discovered_files.append(filepath)
            else:
                log_universal('DEBUG', 'FileDiscovery', "Scanning non-recursively...")
                for file in os.listdir(self.music_dir):
                    filepath = os.path.join(self.music_dir, file)
                    if os.path.isfile(filepath) and self._is_valid_audio_file(filepath):
                        discovered_files.append(filepath)
            
            # Sort files by size (largest first for better progress indication)
            discovered_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
            
            # Save discovery result to cache
            scan_duration = (datetime.now() - start_time).total_seconds()
            db_manager.save_discovery_result(
                directory_path=self.music_dir,
                file_count=len(discovered_files),
                scan_duration=scan_duration,
                status='completed'
            )
            
            log_universal('INFO', 'FileDiscovery', f"File discovery complete: {len(discovered_files)} files found in {scan_duration:.2f}s")
            return discovered_files
            
        except Exception as e:
            # Save failed discovery result
            scan_duration = (datetime.now() - start_time).total_seconds()
            db_manager.save_discovery_result(
                directory_path=self.music_dir,
                file_count=0,
                scan_duration=scan_duration,
                status='failed',
                error_message=str(e)
            )
            log_universal('ERROR', 'FileDiscovery', f"File discovery failed: {e}")
            return []

    @log_function_call
    def save_discovered_files_to_db(self, filepaths: List[str]) -> Dict[str, int]:
        """
        Save discovered files to database with proper tracking.
        
        Args:
            filepaths: List of file paths to save
            
        Returns:
            Dictionary with statistics about the save operation
        """
        log_universal('INFO', 'FileDiscovery', f"Saving {len(filepaths)} discovered files to database...")
        
        stats = {
            'new': 0,
            'updated': 0,
            'unchanged': 0,
            'errors': 0
        }
        
        try:
            db_manager = get_db_manager()
            
            # Batch process files for better performance
            batch_size = 100
            for i in range(0, len(filepaths), batch_size):
                batch = filepaths[i:i + batch_size]
                
                # Pre-collect file info for batch
                file_info_batch = []
                for filepath in batch:
                    try:
                        filename = os.path.basename(filepath)
                        file_size = os.path.getsize(filepath)
                        file_hash = self._get_file_hash(filepath)
                        modified_time = os.path.getmtime(filepath)
                        
                        file_info_batch.append({
                            'filepath': filepath,
                            'filename': filename,
                            'file_size': file_size,
                            'file_hash': file_hash,
                            'modified_time': modified_time
                        })
                    except Exception as e:
                        stats['errors'] += 1
                        log_universal('ERROR', 'FileDiscovery', f"Error collecting file info for {filepath}: {e}")
                
                # Process batch
                for file_info in file_info_batch:
                    try:
                        # Check if file already exists in database
                        existing_result = db_manager.get_analysis_result(file_info['filepath'])
                        
                        if existing_result:
                            existing_hash = existing_result.get('file_hash')
                            existing_size = existing_result.get('file_size_bytes')
                            
                            if existing_hash == file_info['file_hash'] and existing_size == file_info['file_size']:
                                # File unchanged
                                stats['unchanged'] += 1
                            else:
                                # File modified - update basic discovery info only
                                db_manager.save_file_discovery(
                                    file_path=file_info['filepath'],
                                    file_hash=file_info['file_hash'],
                                    file_size_bytes=file_info['file_size'],
                                    filename=file_info['filename'],
                                    discovery_source='file_system'
                                )
                                stats['updated'] += 1
                        else:
                            # New file - save basic discovery info only
                            db_manager.save_file_discovery(
                                file_path=file_info['filepath'],
                                file_hash=file_info['file_hash'],
                                file_size_bytes=file_info['file_size'],
                                filename=file_info['filename'],
                                discovery_source='file_system'
                            )
                            stats['new'] += 1
                            
                    except Exception as e:
                        stats['errors'] += 1
                        log_universal('ERROR', 'FileDiscovery', f"Error saving file {file_info['filepath']} to database: {e}")
                
                # Log progress for large batches
                if len(filepaths) > 1000:
                    progress = min(i + batch_size, len(filepaths))
                    log_universal('INFO', 'FileDiscovery', f"Processed {progress}/{len(filepaths)} files...")
            
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"Database operation failed: {e}")
            stats['errors'] = len(filepaths)
        
        log_universal('INFO', 'FileDiscovery', f"Database save complete:")
        log_universal('INFO', 'FileDiscovery', f"  New files: {stats['new']}")
        log_universal('INFO', 'FileDiscovery', f"  Updated files: {stats['updated']}")
        log_universal('INFO', 'FileDiscovery', f"  Unchanged files: {stats['unchanged']}")
        log_universal('INFO', 'FileDiscovery', f"  Errors: {stats['errors']}")
        
        return stats

    @log_function_call
    def get_db_files(self) -> Set[str]:
        """
        Get all files currently in the database.
        
        Returns:
            Set of file paths from database
        """
        log_universal('DEBUG', 'FileDiscovery', "Retrieving files from database...")
        
        try:
            results = get_db_manager().get_all_analysis_results()
            file_paths = {result['file_path'] for result in results}
            
            log_universal('DEBUG', 'FileDiscovery', f"Retrieved {len(file_paths)} files from database")
            return file_paths
            
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"Error reading from database: {e}")
            return set()

    @log_function_call
    def get_failed_files(self) -> Set[str]:
        """
        Get set of failed files from cache table.
        
        Returns:
            Set of failed file paths
        """
        try:
            db_manager = get_db_manager()
            failed_files = db_manager.get_failed_analysis_files()
            return {f['file_path'] for f in failed_files if f.get('file_path')}
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f'Error getting failed files: {e}')
            return set()

    @log_function_call
    def mark_file_as_failed(self, filepath: str, error_message: str = "Unknown error"):
        """
        Mark a file as failed in the database.
        
        Args:
            filepath: Path to the failed file
            error_message: Error message describing the failure
        """
        log_universal('INFO', 'FileDiscovery', f"Marking file as failed: {filepath}")
        
        try:
            db_manager = get_db_manager()
            db_manager.mark_analysis_failed(filepath, os.path.basename(filepath), error_message)
            log_universal('INFO', 'FileDiscovery', f"File marked as failed: {filepath}")
            
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"Error marking file as failed: {e}")

    @log_function_call
    def cleanup_removed_files_from_db(self) -> int:
        """
        Remove files from database that no longer exist on disk.
        Cleans up from all database locations: analysis_cache, failed_analysis, etc.
        
        Returns:
            Number of files removed from database
        """
        log_universal('INFO', 'FileDiscovery', "Starting cleanup of removed files from database...")
        
        try:
            log_universal('DEBUG', 'FileDiscovery', "Database connection established for cleanup")
            
            # Get all files from database
            db_files = self.get_db_files()
            failed_files = self.get_failed_files()
            all_db_files = db_files.union(failed_files)
            log_universal('DEBUG', 'FileDiscovery', f"Found {len(all_db_files)} total files in database")
            
            # Find files that no longer exist on disk
            removed_files = []
            for filepath in all_db_files:
                if not os.path.exists(filepath):
                    removed_files.append(filepath)
            
            if removed_files:
                log_universal('INFO', 'FileDiscovery', f"Removing {len(removed_files)} non-existent files from database")
                
                # Remove from all database locations
                for filepath in removed_files:
                    try:
                        # Remove from analysis results
                        get_db_manager().delete_analysis_result(filepath)
                        
                        # Remove from failed analysis
                        get_db_manager().delete_failed_analysis(filepath)
                        
                        # Remove from cache table directly
                        for filepath in removed_files:
                            try:
                                get_db_manager().delete_failed_analysis(filepath)
                            except Exception as e:
                                log_universal('WARNING', 'FileDiscovery', f'Error removing failed file from cache: {e}')
                            
                            # Clean up from cache table (unified cache)
                            cursor.execute("""
                                DELETE FROM cache WHERE cache_key LIKE ?
                            """, (f"%{filepath}%",))
                            
                            conn.commit()
                        
                        log_universal('DEBUG', 'FileDiscovery', f"Removed from all database locations: {filepath}")
                    except Exception as e:
                        log_universal('ERROR', 'FileDiscovery', f"Error removing file {filepath}: {e}")
                
                log_universal('INFO', 'FileDiscovery', f"Removed {len(removed_files)} files from all database locations")
                return len(removed_files)
            else:
                log_universal('INFO', 'FileDiscovery', "No files to remove - all database files still exist on disk")
                return 0
                
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"Error cleaning up database: {e}")
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
        log_universal('INFO', 'FileDiscovery', f"Getting files for analysis (force={force}, failed_mode={failed_mode})")
        
        try:
            if failed_mode:
                # Get only failed files
                failed_files = self.get_failed_files()
                log_universal('INFO', 'FileDiscovery', f"Found {len(failed_files)} failed files for retry")
                return list(failed_files)
            
            # Get files from database
            log_universal('DEBUG', 'FileDiscovery', "Retrieving files from database...")
            db_files = self.get_db_files()
            failed_db_files = self.get_failed_files()
            log_universal('DEBUG', 'FileDiscovery', f"Database has {len(db_files)} valid files and {len(failed_db_files)} failed files")
            
            if force:
                # Force re-analysis of all files
                all_files = db_files.union(failed_db_files)
                log_universal('INFO', 'FileDiscovery', f"Force mode: {len(all_files)} files for re-analysis")
                return list(all_files)
            
            # Get current files on disk
            current_files = set(self.discover_files())
            log_universal('DEBUG', 'FileDiscovery', f"Found {len(current_files)} files on disk")
            
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
                    existing_result = get_db_manager().get_analysis_result(filepath)
                    if existing_result and existing_result.get('file_hash') != current_hash:
                        modified_files.add(filepath)
                except Exception as e:
                    log_universal('ERROR', 'FileDiscovery', f"Error checking file hash for {filepath}: {e}")
            
            files_for_analysis = list(new_files.union(retry_files).union(modified_files))
            
            log_universal('INFO', 'FileDiscovery', f"Analysis queue prepared:")
            log_universal('INFO', 'FileDiscovery', f"  New files: {len(new_files)}")
            log_universal('INFO', 'FileDiscovery', f"  Retry files: {len(retry_files)}")
            log_universal('INFO', 'FileDiscovery', f"  Modified files: {len(modified_files)}")
            log_universal('INFO', 'FileDiscovery', f"  Total for analysis: {len(files_for_analysis)}")
            
            return files_for_analysis
            
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"Error getting files for analysis: {e}")
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
            db_info = get_db_manager().get_analysis_result(filepath)
            
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
            log_universal('ERROR', 'FileDiscovery', f"Error getting file info for {filepath}: {e}")
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
        log_universal('INFO', 'FileDiscovery', f"Validating {len(filepaths)} file paths...")
        
        valid_files = []
        invalid_files = []
        
        for filepath in filepaths:
            if self._is_valid_audio_file(filepath):
                valid_files.append(filepath)
            else:
                invalid_files.append(filepath)
        
        log_universal('INFO', 'FileDiscovery', f"Validation complete:")
        log_universal('INFO', 'FileDiscovery', f"  Valid files: {len(valid_files)}")
        log_universal('INFO', 'FileDiscovery', f"  Invalid files: {len(invalid_files)}")
        
        if invalid_files:
            log_universal('WARNING', 'FileDiscovery', f"Invalid files: {invalid_files[:5]}{'...' if len(invalid_files) > 5 else ''}")
        
        return valid_files

    @log_function_call
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get file discovery statistics.
        
        Returns:
            Dictionary with discovery statistics
        """
        try:
            db_manager = get_db_manager()
            
            # Get discovery cache statistics
            discovery_status = db_manager.get_discovery_status(self.music_dir)
            
            # Get database statistics
            db_stats = db_manager.get_database_statistics()
            
            # Get file counts
            total_files = len(self.get_db_files())
            failed_files = len(self.get_failed_files())
            
            stats = {
                'music_directory': self.music_dir,
                'total_files': total_files,
                'failed_files': failed_files,
                'successful_files': total_files - failed_files,
                'last_discovery': discovery_status.get('created_at') if discovery_status else None,
                'discovery_status': discovery_status.get('status') if discovery_status else 'unknown',
                'scan_duration': discovery_status.get('scan_duration') if discovery_status else None,
                'database_stats': db_stats
            }
            
            return stats
            
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"Failed to get statistics: {e}")
            return {'error': str(e)}

    @log_function_call
    def get_discovery_history(self) -> List[Dict[str, Any]]:
        """
        Get discovery history from database.
        
        Returns:
            List of discovery cache entries
        """
        try:
            db_manager = get_db_manager()
            
            # Get all discovery cache entries for this directory
            cache_entries = db_manager.get_cache_by_type('discovery')
            
            # Filter for this directory
            discovery_history = []
            for entry in cache_entries:
                try:
                    cache_data = json.loads(entry['cache_value'])
                    if cache_data.get('directory_path') == self.music_dir:
                        discovery_history.append({
                            'created_at': entry['created_at'],
                            'file_count': cache_data.get('file_count', 0),
                            'scan_duration': cache_data.get('scan_duration', 0),
                            'status': cache_data.get('status', 'unknown'),
                            'error_message': cache_data.get('error_message')
                        })
                except json.JSONDecodeError:
                    continue
            
            # Sort by creation date (newest first)
            discovery_history.sort(key=lambda x: x['created_at'], reverse=True)
            
            return discovery_history
            
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"Failed to get discovery history: {e}")
            return [] 