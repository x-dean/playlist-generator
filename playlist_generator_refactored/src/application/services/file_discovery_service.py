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
from shared.utils import get_file_size_mb, calculate_file_hash, create_audio_files_with_size
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
                    
                    for file_path in files:
                        if str(file_path) in seen_files:
                            response.result.duplicate_files += 1
                            continue
                        seen_files.add(str(file_path))
                        
                        try:
                            # Double-check that this is actually a file and exists
                            if not file_path.exists():
                                self.logger.warning(f"File no longer exists: {file_path}")
                                skipped_files.append(str(file_path))
                                continue
                            
                            if not file_path.is_file():
                                self.logger.warning(f"Path is not a file: {file_path}")
                                skipped_files.append(str(file_path))
                                continue
                            
                            # Calculate file hash first (with timeout protection)
                            try:
                                file_hash = calculate_file_hash(file_path)
                            except Exception as e:
                                self.logger.warning(f"Could not calculate hash for {file_path}: {e}")
                                file_hash = None
                            
                            # Check if file already exists in database by hash
                            existing_audio_file = None
                            if file_hash:
                                existing_audio_file = self._find_by_hash(file_hash)
                            
                            if existing_audio_file:
                                # File exists, update path if needed and reuse UUID
                                audio_file = existing_audio_file
                                if str(audio_file.file_path) != str(file_path):
                                    self.logger.info(f"File moved: {audio_file.file_path} -> {file_path}")
                                    audio_file.file_path = file_path
                                    audio_file.file_name = file_path.name
                            else:
                                # New file, create with new UUID
                                audio_file = AudioFile(file_path=file_path)
                                
                                # Get file size
                                size_mb = get_file_size_mb(file_path)
                                if size_mb is not None:
                                    audio_file.file_size_bytes = int(size_mb * 1024 * 1024)
                                else:
                                    self.logger.warning(f"Could not determine file size for {file_path}")
                                    audio_file.file_size_bytes = None
                                
                                # Set file hash
                                if file_hash:
                                    audio_file.file_hash = file_hash
                                    self.logger.debug(f"Calculated hash for {file_path}: {file_hash[:8]}...")
                                else:
                                    self.logger.warning(f"Could not calculate hash for {file_path}")
                                    audio_file.file_hash = None
                            
                            # Save to database immediately
                            try:
                                self.audio_repo.save(audio_file)
                                self.logger.debug(f"Saved to database: {file_path}")
                            except Exception as e:
                                self.logger.warning(f"Failed to save {file_path} to database: {e}")
                            
                            discovered_files.append(audio_file)
                            
                            # Progress indication
                            processed_files += 1
                            if processed_files % 100 == 0 or processed_files == total_files:
                                self.logger.info(f"Processed {processed_files}/{total_files} files ({processed_files/total_files*100:.1f}%)")
                            
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
            tracking_stats = self._track_database_state(discovered_files)
            
            # Finalize result
            response.result.discovered_files = discovered_files
            response.result.skipped_files = skipped_files
            response.result.error_files = error_files
            response.result.new_files_added = tracking_stats['new_files_added']
            response.result.missing_files_removed = tracking_stats['missing_files_removed']
            
            response.status = "completed"
            
            self.logger.info(f"File discovery completed: {len(discovered_files)} files found, "
                           f"{len(skipped_files)} skipped, {len(error_files)} errors")
            
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
        try:
            if path.is_file():
                # Single file
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
                        
                        # Filter by extension
                        if request.file_extensions:
                            file_ext = file_path.suffix.lower().lstrip('.')
                            if file_ext not in [ext.lower() for ext in request.file_extensions]:
                                continue
                        
                        # Filter by size
                        if request.min_file_size_mb is not None or request.max_file_size_mb is not None:
                            size_mb = get_file_size_mb(file_path)
                            if size_mb is None:
                                self.logger.debug(f"Skipping file with unknown size: {file_path}")
                                continue
                            if request.min_file_size_mb is not None and size_mb < request.min_file_size_mb:
                                self.logger.debug(f"Skipping file too small ({size_mb:.1f} MB): {file_path}")
                                continue
                            if request.max_file_size_mb is not None and size_mb > request.max_file_size_mb:
                                self.logger.debug(f"Skipping file too large ({size_mb:.1f} MB): {file_path}")
                                continue
                        
                        # Filter by patterns
                        if request.exclude_patterns and any(file_path.match(pat) for pat in request.exclude_patterns):
                            continue
                        if request.include_patterns and not any(file_path.match(pat) for pat in request.include_patterns):
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
            
            # Get all files currently in database
            db_files = self.audio_repo.find_all()
            db_file_paths = {str(af.file_path) for af in db_files}
            
            # Get discovered file paths
            discovered_file_paths = {str(af.file_path) for af in discovered_files}
            
            # Find files that exist in database but not in discovery (missing files)
            missing_files = db_file_paths - discovered_file_paths
            
            # Find files that exist in discovery but not in database (new files)
            new_files = discovered_file_paths - db_file_paths
            
            self.logger.info(f"Database tracking: {len(db_files)} in DB, {len(discovered_files)} discovered")
            self.logger.info(f"Missing files: {len(missing_files)}, New files: {len(new_files)}")
            
            # Remove missing files from all databases
            missing_removed = 0
            if missing_files:
                missing_removed = self._remove_missing_files(missing_files)
            
            # Add new files to database
            new_added = 0
            if new_files:
                new_added = self._add_new_files(discovered_files, new_files)
            
            self.logger.info("Database state tracking completed")
            
            return {
                'new_files_added': new_added,
                'missing_files_removed': missing_removed
            }
            
        except Exception as e:
            self.logger.error(f"Database state tracking failed: {e}")
            raise FileDiscoveryError(f"Database state tracking failed: {e}") from e
    
    def _remove_missing_files(self, missing_file_paths: set) -> int:
        """
        Remove missing files from all databases.
        
        Args:
            missing_file_paths: Set of file paths that no longer exist
            
        Returns:
            Number of files removed
        """
        try:
            self.logger.info(f"Removing {len(missing_file_paths)} missing files from database")
            
            removed_count = 0
            
            for file_path_str in missing_file_paths:
                try:
                    # Find the audio file in database
                    audio_file = self.audio_repo.find_by_path(Path(file_path_str))
                    if audio_file:
                        # Remove from all related tables
                        self._remove_audio_file_data(audio_file.id)
                        removed_count += 1
                        self.logger.debug(f"Removed missing file: {file_path_str}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove missing file {file_path_str}: {e}")
            
            self.logger.info(f"Successfully removed {removed_count} missing files from database")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to remove missing files: {e}")
            raise FileDiscoveryError(f"Failed to remove missing files: {e}") from e
    
    def _remove_audio_file_data(self, audio_file_id) -> None:
        """
        Remove all data related to an audio file from all databases.
        
        This method removes the file from:
        - audio_files table
        - analysis_results table  
        - feature_sets table (via cascade)
        - metadata table (via cascade)
        - playlists table (if referenced)
        
        Args:
            audio_file_id: ID of the audio file to remove
        """
        try:
            # Remove from playlists first
            playlist_updates = self.playlist_repo.remove_audio_file_references(audio_file_id)
            if playlist_updates > 0:
                self.logger.debug(f"Removed audio file from {playlist_updates} playlists")
            
            # Remove from analysis results (this will cascade to features and metadata)
            self.analysis_repo.delete_by_audio_file_id(audio_file_id)
            
            # Remove from audio files table
            self.audio_repo.delete(audio_file_id)
            
            self.logger.debug(f"Removed all data for audio file UUID: {audio_file_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to remove audio file data {audio_file_id}: {e}")
    
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
                        # Save to database
                        self.audio_repo.save(audio_file)
                        added_count += 1
                        self.logger.debug(f"Added new file: {audio_file.file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to add new file {audio_file.file_path}: {e}")
            
            self.logger.info(f"Successfully added {added_count} new files to database")
            return added_count
            
        except Exception as e:
            self.logger.error(f"Failed to add new files: {e}")
            raise FileDiscoveryError(f"Failed to add new files: {e}") from e
    
    def _find_by_hash(self, file_hash: str) -> Optional[AudioFile]:
        """
        Find audio file by hash.
        
        Args:
            file_hash: MD5 hash of the file
            
        Returns:
            AudioFile if found, None otherwise
        """
        try:
            return self.audio_repo.find_by_hash(file_hash)
        except Exception as e:
            self.logger.warning(f"Failed to find file by hash {file_hash[:8]}...: {e}")
            return None 