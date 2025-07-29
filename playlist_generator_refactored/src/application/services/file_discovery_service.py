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
from infrastructure.logging import get_logger, set_correlation_id, log_function_call

from domain.entities import AudioFile
from application.dtos import (
    FileDiscoveryRequest,
    FileDiscoveryResponse,
    DiscoveryResult
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
                    for file_path in files:
                        if str(file_path) in seen_files:
                            response.result.duplicate_files += 1
                            continue
                        seen_files.add(str(file_path))
                        try:
                            # Real audio validation using mutagen
                            mutagen_file = MutagenFile(str(file_path))
                            if mutagen_file is None:
                                raise ValueError("Not a valid audio file")
                            audio_file = AudioFile(file_path=file_path)
                            discovered_files.append(audio_file)
                        except Exception as e:
                            self.logger.warning(f"Invalid audio file: {file_path} ({e})")
                            skipped_files.append(str(file_path))
                except Exception as e:
                    self.logger.error(f"Failed to scan {search_path}: {e}")
                    error_files.append(str(search_path))
            
            # Finalize result
            response.result.discovered_files = discovered_files
            response.result.skipped_files = skipped_files
            response.result.error_files = error_files
            
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
        Only files that can be opened by mutagen are considered valid audio files.
        
        Args:
            path: Directory path
            request: Discovery request parameters
            
        Returns:
            List of matching file paths
        """
        files = []
        try:
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                for entry in path.rglob("*" if request.recursive else "*"):
                    if entry.is_file():
                        # Filter by extension
                        if request.file_extensions and entry.suffix.lower() not in [ext.lower() for ext in request.file_extensions]:
                            continue
                        # Filter by size
                        if request.min_file_size_mb is not None or request.max_file_size_mb is not None:
                            try:
                                size_mb = entry.stat().st_size / (1024 * 1024)
                                if request.min_file_size_mb is not None and size_mb < request.min_file_size_mb:
                                    continue
                                if request.max_file_size_mb is not None and size_mb > request.max_file_size_mb:
                                    continue
                            except Exception:
                                continue
                        # Filter by patterns
                        if request.exclude_patterns and any(entry.match(pat) for pat in request.exclude_patterns):
                            continue
                        if request.include_patterns and not any(entry.match(pat) for pat in request.include_patterns):
                            continue
                        files.append(entry)
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