"""
File Feeder Service for the Playlista application.

This service retrieves discovered files from the database and feeds them
to the processor, separating large and small files for optimal processing.
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path

from shared.config import get_config
from shared.exceptions.file_system import FileFeederError
from shared.utils import get_file_size_mb, split_files_by_size
from infrastructure.logging import get_logger, log_function_call

from domain.entities import AudioFile
from application.dtos import (
    FileFeederRequest,
    FileFeederResponse,
    FeederResult
)
from infrastructure.persistence.repositories import SQLiteAudioFileRepository


class FileFeederService:
    """
    Service for feeding files from database to processors.
    
    This service coordinates file retrieval and separation:
    - Retrieves files from database
    - Separates large and small files
    - Provides files to processors in optimal order
    """
    
    def __init__(self):
        """Initialize the file feeder service."""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.audio_repo = SQLiteAudioFileRepository()
    
    @log_function_call
    def feed_files(self, request: FileFeederRequest) -> FileFeederResponse:
        """
        Feed files from database to processors.
        
        Args:
            request: Feeder request parameters
            
        Returns:
            FileFeederResponse with separated files
        """
        response = FileFeederResponse(
            status="processing",
            result=FeederResult()
        )
        
        try:
            self.logger.info("Starting file feeding from database")
            
            # Get all files from database
            all_files = self.audio_repo.find_all()
            self.logger.info(f"Retrieved {len(all_files)} files from database")
            
            if not all_files:
                self.logger.warning("No files found in database")
                response.status = "completed"
                return response
            
            # Convert to file paths for processing
            file_paths = [audio_file.file_path for audio_file in all_files]
            
            # Separate large and small files
            small_files, large_files = split_files_by_size(
                file_paths, 
                request.large_file_threshold_mb
            )
            
            self.logger.info(f"Separated files: {len(small_files)} small, {len(large_files)} large")
            
            # Create AudioFile entities for each category
            small_audio_files = []
            large_audio_files = []
            
            # Process small files
            for file_path in small_files:
                audio_file = self.audio_repo.find_by_path(file_path)
                if audio_file:
                    small_audio_files.append(audio_file)
            
            # Process large files
            for file_path in large_files:
                audio_file = self.audio_repo.find_by_path(file_path)
                if audio_file:
                    large_audio_files.append(audio_file)
            
            # Set results
            response.result.small_files = small_audio_files
            response.result.large_files = large_audio_files
            response.result.total_files = len(small_audio_files) + len(large_audio_files)
            
            response.status = "completed"
            
            self.logger.info(f"File feeding completed: {len(small_audio_files)} small files, "
                           f"{len(large_audio_files)} large files ready for processing")
            
            return response
            
        except Exception as e:
            self.logger.error(f"File feeding failed: {e}")
            raise FileFeederError(f"File feeding failed: {e}") from e
    
    def get_processing_order(self, request: FileFeederRequest) -> List[AudioFile]:
        """
        Get files in optimal processing order.
        
        Args:
            request: Feeder request parameters
            
        Returns:
            List of AudioFile entities in processing order
        """
        response = self.feed_files(request)
        
        # Return large files first (for sequential processing), then small files (for parallel)
        processing_order = []
        processing_order.extend(response.result.large_files)
        processing_order.extend(response.result.small_files)
        
        self.logger.info(f"Processing order: {len(response.result.large_files)} large files first, "
                        f"then {len(response.result.small_files)} small files")
        
        return processing_order 