"""
Shared utilities for the Playlista application.

This module contains utility functions that are used across multiple
services and components to avoid code duplication.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
from domain.entities import AudioFile

logger = logging.getLogger(__name__)


def get_file_size_mb(file_path: Path) -> Optional[float]:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB, or None if size cannot be determined
    """
    try:
        file_size_bytes = file_path.stat().st_size
        return file_size_bytes / (1024 * 1024)
    except Exception as e:
        logger.debug(f"Could not get file size for {file_path}: {e}")
        return None


def sort_files_by_size(file_paths: List[Path], reverse: bool = True) -> List[Path]:
    """
    Sort files by size (largest first by default).
    
    Args:
        file_paths: List of file paths to sort
        reverse: If True, sort largest first; if False, sort smallest first
        
    Returns:
        List of file paths sorted by size
    """
    files_with_size = []
    
    for file_path in file_paths:
        size_mb = get_file_size_mb(file_path)
        files_with_size.append((file_path, size_mb or 0))
    
    # Sort by size
    files_with_size.sort(key=lambda x: x[1], reverse=reverse)
    
    return [file_path for file_path, size in files_with_size]


def split_files_by_size(
    file_paths: List[Path], 
    threshold_mb: float
) -> Tuple[List[Path], List[Path]]:
    """
    Split files into small and large based on size threshold.
    
    Args:
        file_paths: List of file paths to split
        threshold_mb: Size threshold in MB
        
    Returns:
        Tuple of (small_files, large_files)
    """
    small_files = []
    large_files = []
    
    for file_path in file_paths:
        size_mb = get_file_size_mb(file_path)
        if size_mb and size_mb > threshold_mb:
            large_files.append(file_path)
        else:
            small_files.append(file_path)
    
    return small_files, large_files


def log_largest_files(file_paths: List[Path], count: int = 5) -> None:
    """
    Log the largest files for debugging/monitoring.
    
    Args:
        file_paths: List of file paths
        count: Number of largest files to log
    """
    if not file_paths:
        return
    
    files_with_size = []
    for file_path in file_paths:
        size_mb = get_file_size_mb(file_path)
        files_with_size.append((file_path, size_mb or 0))
    
    # Sort by size (largest first)
    files_with_size.sort(key=lambda x: x[1], reverse=True)
    
    # Log the largest files
    largest_files = files_with_size[:count]
    if largest_files:
        logger.info(f"Largest files to be processed first:")
        for i, (file_path, size_mb) in enumerate(largest_files, 1):
            logger.info(f"  {i}. {file_path.name} ({size_mb:.1f} MB)")


def create_audio_files_with_size(file_paths: List[Path]) -> List[AudioFile]:
    """
    Create AudioFile entities with size information.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        List of AudioFile entities with size information populated
    """
    audio_files = []
    
    for file_path in file_paths:
        try:
            audio_file = AudioFile(file_path=file_path)
            
            # Set file size if available
            size_mb = get_file_size_mb(file_path)
            if size_mb:
                audio_file.file_size_bytes = int(size_mb * 1024 * 1024)
            
            audio_files.append(audio_file)
            
        except Exception as e:
            logger.warning(f"Could not create AudioFile for {file_path}: {e}")
    
    return audio_files 