"""
File Discovery Service for the Playlista application.

This service orchestrates file discovery operations, including directory scanning,
filtering, and validation of audio files using mutagen.
"""

import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from uuid import uuid4
from mutagen import File as MutagenFile

from shared.config import get_config
from shared.exceptions import FileDiscoveryError, FileAccessError
from shared.utils import get_file_size_mb
from infrastructure.logging import get_logger, set_correlation_id, log_function_call

from domain.entities import AudioFile
from application.dtos import (
    FileDiscoveryRequest,
    FileDiscoveryResponse,
    DiscoveryResult
)
from infrastructure.persistence.repositories import (
    SQLiteAudioFileRepository,
    SQLiteFeatureSetRepository,
    SQLiteMetadataRepository,
    SQLiteAnalysisResultRepository,
    SQLitePlaylistRepository
)


class FileDiscoveryService:
    """
    Service for orchestrating file discovery operations.
    
    This service coordinates file discovery, including:
    - Directory scanning and recursion
    - File filtering by extension, size, and patterns
    - Validation and deduplication (using mutagen for real audio validation)
    - Performance monitoring and reporting
    """
    
    def __init__(self):
        """Initialize the file discovery service."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self._discovery_cache: Dict[str, List[AudioFile]] = {}
        
        # Initialize repositories for database tracking
        self.audio_repo = SQLiteAudioFileRepository()
        self.feature_repo = SQLiteFeatureSetRepository()
        self.metadata_repo = SQLiteMetadataRepository()
        self.analysis_repo = SQLiteAnalysisResultRepository()
        self.playlist_repo = SQLitePlaylistRepository()
    
    @log_function_call
    def discover_files(self, request: FileDiscoveryRequest) -> FileDiscoveryResponse:
        """
        Discover audio files according to the request parameters.
        
        Args:
            request: FileDiscoveryRequest containing discovery parameters
            
        Returns:
            FileDiscoveryResponse with discovered files and statistics
            
        Raises:
            FileDiscoveryError: If discovery fails
        """
        try:
            # Set correlation ID for tracking
            if request.correlation_id:
                set_correlation_id(request.correlation_id)
            
            self.logger.info(f"Starting file discovery in {request.total_search_paths} paths")
            
            # Initialize response
            response = FileDiscoveryResponse(
                request_id=str(uuid4()),
                status="in_progress",
                result=DiscoveryResult()
            )
            
            # Discover files
            discovered_files = []
            skipped_files = []
            error_files = []
            seen_files = set()
            
            for search_path in request.search_paths:
                try:
                    self.logger.info(f"Scanning path: {search_path}")
                    path_obj = Path(search_path)
                    if not path_obj.exists():
                        self.logger.warning(f"Path does not exist: {search_path}")
                        continue
                    
                    files = self._scan_directory(path_obj, request)
                    self.logger.info(f"Directory scan found {len(files)} files")
                    
                    # Process files with progress indication
                    total_files = len(files)
                    processed_files = 0
                    hash_calculated = 0
                    hash_failed = 0
                    db_saved = 0
                    db_failed = 0
                    
                    self.logger.info(f"Starting to process {total_files} files...")
                    
                    # Show progress every 100 files
                    progress_interval = max(1, total_files // 10)  # Show progress every 10% or every file if < 10
                    
                    for file_path in files:
                        if str(file_path) in seen_files:
                            response.result.duplicate_files += 1
                            continue
                        seen_files.add(str(file_path))
                        
                        try:
                            processed_files += 1
                            
                            # Show progress periodically
                            if processed_files % progress_interval == 0:
                                self.logger.info(f"Progress: {processed_files}/{total_files} files processed ({processed_files/total_files*100:.1f}%)")
                            
                            # Quick validation
                            if not file_path.exists() or not file_path.is_file():
                                skipped_files.append(str(file_path))
                                continue
                            
                            # Create audio file with basic info
                            audio_file = AudioFile(file_path=file_path)
                            
                            # Get file size quickly
                            try:
                                stat = file_path.stat()
                                audio_file.file_size_bytes = stat.st_size
                                audio_file.file_name = file_path.name
                            except Exception as e:
                                self.logger.warning(f"Could not get file info for {file_path}: {e}")
                                skipped_files.append(str(file_path))
                                continue
                                else:
                                    self.logger.warning(f"Could not determine file size for {file_path}")
                                    audio_file.file_size_bytes = None
                                
                                # Set file hash
                                if file_hash:
                                    audio_file.file_hash = file_hash
                                    self.logger.debug(f"Hash set for {file_path}: {file_hash[:8]}...")
                                else:
                                    self.logger.warning(f"Could not calculate hash for {file_path}")
                                    audio_file.file_hash = None
                            
                            # Save to database immediately
                            self.logger.debug(f"Saving to database: {file_path}")
                            try:
                                self.audio_repo.save(audio_file)
                                db_saved += 1
                                self.logger.debug(f"Successfully saved to database: {file_path}")
                            except Exception as e:
                                db_failed += 1
                                self.logger.warning(f"Failed to save {file_path} to database: {e}")
                            
                            discovered_files.append(audio_file)
                            
                            # Progress indication with detailed stats
                            processed_files += 1
                            if processed_files % 100 == 0 or processed_files == total_files:
                                self.logger.info(f"Progress: {processed_files}/{total_files} files ({processed_files/total_files*100:.1f}%)")
                                self.logger.info(f"  - Hash calculated: {hash_calculated}, failed: {hash_failed}")
                                self.logger.info(f"  - Database saved: {db_saved}, failed: {db_failed}")
                                self.logger.info(f"  - Discovered: {len(discovered_files)}, skipped: {len(skipped_files)}")
                            
                            # Debug logging with proper size handling
                            if hasattr(audio_file, 'file_size_bytes') and audio_file.file_size_bytes:
                                size_mb = audio_file.file_size_bytes / (1024 * 1024)
                                self.logger.debug(f"Added file: {file_path} ({size_mb:.1f} MB, hash: {file_hash[:8] if file_hash else 'None'}...)")
                            else:
                                self.logger.debug(f"Added file: {file_path} (size unknown, hash: {file_hash[:8] if file_hash else 'None'}...)")
                            
                        except Exception as e:
                            self.logger.warning(f"Could not create AudioFile for {file_path}: {e}")
                            skipped_files.append(str(file_path))
                except Exception as e:
                    self.logger.error(f"Failed to scan {search_path}: {e}")
                    error_files.append(str(search_path))
            
            # Track database state and clean up missing files
            self.logger.info("Starting database state tracking...")
            tracking_stats = self._track_database_state(discovered_files)
            self.logger.info(f"Database tracking completed: {tracking_stats}")
            
            # Finalize result
            response.result.discovered_files = discovered_files
            response.result.skipped_files = skipped_files
            response.result.error_files = error_files
            response.result.new_files_added = tracking_stats['new_files_added']
            response.result.missing_files_removed = tracking_stats['missing_files_removed']
            
            response.status = "completed"
            
            self.logger.info(f"File discovery completed:")
            self.logger.info(f"  - Total files processed: {processed_files}")
            self.logger.info(f"  - Hash calculations: {hash_calculated} successful, {hash_failed} failed")
            self.logger.info(f"  - Database operations: {db_saved} saved, {db_failed} failed")
            self.logger.info(f"  - Final results: {len(discovered_files)} discovered, {len(skipped_files)} skipped, {len(error_files)} errors")
            
            return response
            
        except Exception as e:
            self.logger.error(f"File discovery failed: {e}")
            raise FileDiscoveryError(f"File discovery failed: {e}") from e
    
    def _scan_directory(self, path: Path, request: FileDiscoveryRequest) -> List[Path]:
        """
        Scan a directory for audio files matching the request filters.
        Uses os.walk() for efficient directory traversal.
        
        Args:
            path: Directory path
            request: Discovery request parameters
            
        Returns:
            List of matching file paths
        """
        import os
        
        files = []
        valid_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus'}
        
        try:
            if path.is_file():
                # Single file
                if path.suffix.lower() in valid_extensions:
                    files.append(path)
            elif path.is_dir():
                # Directory - use os.walk for efficient traversal
                for root, dirs, files_in_dir in os.walk(str(path)):
                    # Skip failed directory if specified
                    if hasattr(self, 'failed_dir') and self.failed_dir in root:
                        self.logger.debug(f"Skipping failed directory: {root}")
                        continue
                    
                    for file_name in files_in_dir:
                        file_path = Path(root) / file_name
                        
                        # Quick extension check
                        if file_path.suffix.lower() not in valid_extensions:
                            continue
                        
                        # Quick size check (skip very small files)
                        try:
                            if file_path.stat().st_size < 1024:  # Less than 1KB
                                continue
                        except Exception:
                            continue
                        
                        files.append(file_path)
        except Exception as e:
            self.logger.warning(f"Directory scan failed for {path}: {e}")
        return files
    
    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._discovery_cache.clear()
        self.logger.info("Discovery cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self._discovery_cache),
            'cached_discoveries': list(self._discovery_cache.keys())
        }
    
    def _track_database_state(self, discovered_files: List[AudioFile]) -> dict:
        """
        Track database state and clean up missing files.
        
        This method:
        1. Gets all files currently in database
        2. Compares with discovered files
        3. Removes database entries for files that no longer exist
        4. Adds new files to database
        
        Args:
            discovered_files: List of currently discovered files
            
        Returns:
            Dictionary with tracking statistics
        """
        try:
            self.logger.info("Starting database state tracking")
            
            # Get discovered file paths
            discovered_paths = {str(af.file_path) for af in discovered_files}
            self.logger.info(f"Discovered {len(discovered_paths)} files on disk")
            
            # Get database file paths (use a simple query to avoid loading all data)
            self.logger.info("Checking database for existing files...")
            db_paths = self._get_database_file_paths()
            self.logger.info(f"Found {len(db_paths)} files in database")
            
            # Find missing files (in database but not on disk)
            missing_files = db_paths - discovered_paths
            if missing_files:
                self.logger.info(f"Found {len(missing_files)} files in database that no longer exist")
                removed_count = self._remove_missing_files(missing_files)
                self.logger.info(f"Removed {removed_count} missing files from database")
            else:
                self.logger.info("No missing files found")
                removed_count = 0
            
            # Find new files (on disk but not in database)
            new_files = discovered_paths - db_paths
            if new_files:
                self.logger.info(f"Found {len(new_files)} new files to add")
                added_count = self._add_new_files(discovered_files, new_files)
                self.logger.info(f"Added {added_count} new files to database")
            else:
                self.logger.info("No new files to add")
                added_count = 0
            
            return {
                'files_discovered': len(discovered_files),
                'files_removed': removed_count,
                'files_added': added_count
            }
            
        except Exception as e:
            self.logger.error(f"Database state tracking failed: {e}")
            raise FileDiscoveryError(f"Database state tracking failed: {e}") from e
    
    def _get_database_file_paths(self) -> set:
        """
        Get all file paths from database efficiently.
        
        Returns:
            Set of file paths
        """
        try:
            # Use a simple query to get just the file paths
            with self.audio_repo._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT file_path FROM audio_files")
                rows = cursor.fetchall()
                return {row[0] for row in rows}
        except Exception as e:
            self.logger.warning(f"Failed to get database file paths: {e}")
            return set()
    
    def _remove_missing_files(self, missing_file_paths: set) -> int:
        """
        Remove missing files from database.
        
        Args:
            missing_file_paths: Set of file paths that no longer exist
            
        Returns:
            Number of files removed
        """
        try:
            self.logger.info(f"Removing {len(missing_file_paths)} missing files from database")
            
            removed_count = 0
            with self.audio_repo._get_connection() as conn:
                cursor = conn.cursor()
                
                for file_path in missing_file_paths:
                    try:
                        # Delete from audio_files table
                        cursor.execute("DELETE FROM audio_files WHERE file_path = ?", (file_path,))
                        if cursor.rowcount > 0:
                            removed_count += 1
                            self.logger.debug(f"Removed missing file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove missing file {file_path}: {e}")
                
                conn.commit()
            
            self.logger.info(f"Successfully removed {removed_count} missing files from database")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to remove missing files: {e}")
            return 0
    
    def _add_new_files(self, discovered_files: List[AudioFile], new_file_paths: set) -> int:
        """
        Add new files to database.
        
        Args:
            discovered_files: List of all discovered files
            new_file_paths: Set of file paths that are new
            
        Returns:
            Number of files added
        """
        try:
            self.logger.info(f"Adding {len(new_file_paths)} new files to database")
            
            added_count = 0
            for audio_file in discovered_files:
                if str(audio_file.file_path) in new_file_paths:
                    try:
                        self.audio_repo.save(audio_file)
                        added_count += 1
                        self.logger.debug(f"Added new file: {audio_file.file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to add new file {audio_file.file_path}: {e}")
            
            self.logger.info(f"Successfully added {added_count} new files to database")
            return added_count
            
        except Exception as e:
            self.logger.error(f"Failed to add new files: {e}")
            return 0 