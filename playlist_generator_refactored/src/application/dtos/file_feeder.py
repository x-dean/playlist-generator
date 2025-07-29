"""
File Feeder DTOs for the Playlista application.

This module defines the data transfer objects used by the file feeder service
for communication between the application and infrastructure layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from domain.entities import AudioFile


@dataclass
class FeederResult:
    """Result of file feeding operation."""
    
    # File categories
    small_files: List[AudioFile] = field(default_factory=list)
    large_files: List[AudioFile] = field(default_factory=list)
    
    # Statistics
    total_files: int = 0
    small_file_count: int = 0
    large_file_count: int = 0
    
    # Timing information
    feeding_time_ms: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.start_time and self.end_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        # Calculate file counts
        self.small_file_count = len(self.small_files)
        self.large_file_count = len(self.large_files)
        self.total_files = self.small_file_count + self.large_file_count
    
    def get_summary(self) -> str:
        """Get summary of feeding results."""
        return (f"Feeding completed: {self.total_files} total files "
                f"({self.small_file_count} small, {self.large_file_count} large)")
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'small_files': len(self.small_files),
            'large_files': len(self.large_files),
            'total_files': self.total_files,
            'feeding_time_ms': self.feeding_time_ms,
            'summary': self.get_summary()
        }


@dataclass
class FileFeederRequest:
    """Request for file feeding operation."""
    
    # Processing thresholds
    large_file_threshold_mb: float = 50.0
    
    # Processing options
    process_large_files_first: bool = True
    max_files_per_batch: Optional[int] = None
    
    # User preferences
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 0
    
    def __post_init__(self):
        """Validate request parameters."""
        if self.large_file_threshold_mb <= 0:
            raise ValueError("Large file threshold must be positive")
        
        if self.max_files_per_batch is not None and self.max_files_per_batch <= 0:
            raise ValueError("Max files per batch must be positive")


@dataclass
class FileFeederResponse:
    """Response from file feeding operation."""
    
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[FeederResult] = None
    error_message: Optional[str] = None
    
    def is_successful(self) -> bool:
        """Check if feeding was successful."""
        return self.status == "completed" and self.result is not None
    
    def get_error_message(self) -> str:
        """Get error message if any."""
        return self.error_message or "Unknown error" 