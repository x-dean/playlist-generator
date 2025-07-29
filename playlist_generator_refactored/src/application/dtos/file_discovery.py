"""
File discovery DTOs for the application layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

from domain.entities import AudioFile


class DiscoveryFilter(str, Enum):
    """Types of discovery filters."""
    EXTENSION = "extension"
    SIZE = "size"
    DATE = "date"
    PATH = "path"
    NAME = "name"


@dataclass
class DiscoveryResult:
    """Result of file discovery."""
    discovered_files: List[AudioFile] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    error_files: List[str] = field(default_factory=list)
    
    # Statistics (calculated dynamically)
    duplicate_files: int = 0
    
    # Database tracking statistics
    new_files_added: int = 0
    missing_files_removed: int = 0
    
    # File type breakdown
    file_extensions: Dict[str, int] = field(default_factory=dict)
    file_sizes: Dict[str, int] = field(default_factory=dict)  # small, medium, large
    
    # Timing information
    discovery_time_ms: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.start_time and self.end_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        # Calculate file type breakdown
        for audio_file in self.discovered_files:
            extension = audio_file.file_path.suffix.lower()
            self.file_extensions[extension] = self.file_extensions.get(extension, 0) + 1
            
            # Categorize by size
            if audio_file.file_size_mb is not None:
                if audio_file.file_size_mb < 5:
                    size_category = "small"
                elif audio_file.file_size_mb < 20:
                    size_category = "medium"
                else:
                    size_category = "large"
                self.file_sizes[size_category] = self.file_sizes.get(size_category, 0) + 1
    
    @property
    def total_files_found(self) -> int:
        """Get total number of files found."""
        return len(self.discovered_files) + len(self.skipped_files) + len(self.error_files)
    
    @property
    def valid_audio_files(self) -> int:
        """Get number of valid audio files."""
        return len(self.discovered_files)
    
    @property
    def invalid_files(self) -> int:
        """Get number of invalid files."""
        return len(self.skipped_files)
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_files_found == 0:
            return 0.0
        return (self.valid_audio_files / self.total_files_found) * 100
    
    @property
    def total_size_mb(self) -> float:
        """Get total size of discovered files in MB."""
        total_size = 0.0
        for audio_file in self.discovered_files:
            if audio_file.file_size_mb is not None:
                total_size += audio_file.file_size_mb
        return total_size
    
    def get_files_by_extension(self, extension: str) -> List[AudioFile]:
        """Get files by extension."""
        return [f for f in self.discovered_files if f.file_path.suffix.lower() == extension.lower()]
    
    def get_files_by_size_category(self, category: str) -> List[AudioFile]:
        """Get files by size category."""
        if category == "small":
            return [f for f in self.discovered_files if f.file_size_mb is not None and f.file_size_mb < 5]
        elif category == "medium":
            return [f for f in self.discovered_files if f.file_size_mb is not None and 5 <= f.file_size_mb < 20]
        elif category == "large":
            return [f for f in self.discovered_files if f.file_size_mb is not None and f.file_size_mb >= 20]
        else:
            return []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get discovery summary."""
        return {
            'total_files_found': self.total_files_found,
            'valid_audio_files': self.valid_audio_files,
            'invalid_files': self.invalid_files,
            'duplicate_files': self.duplicate_files,
            'success_rate': self.success_rate,
            'total_size_mb': self.total_size_mb,
            'file_extensions': self.file_extensions,
            'file_sizes': self.file_sizes,
            'discovery_time_ms': self.discovery_time_ms,
            'total_duration_seconds': self.total_duration_seconds,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'discovered_files_count': len(self.discovered_files),
            'skipped_files_count': len(self.skipped_files),
            'error_files_count': len(self.error_files),
            'total_files_found': self.total_files_found,
            'valid_audio_files': self.valid_audio_files,
            'invalid_files': self.invalid_files,
            'duplicate_files': self.duplicate_files,
            'success_rate': self.success_rate,
            'total_size_mb': self.total_size_mb,
            'file_extensions': self.file_extensions,
            'file_sizes': self.file_sizes,
            'discovery_time_ms': self.discovery_time_ms,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_seconds': self.total_duration_seconds,
            'summary': self.get_summary()
        }


@dataclass
class FileDiscoveryRequest:
    """Request for file discovery."""
    search_paths: List[str] = field(default_factory=list)
    recursive: bool = True
    follow_symlinks: bool = False
    
    # File filters
    file_extensions: List[str] = field(default_factory=lambda: ['.mp3', '.flac', '.wav', '.m4a', '.ogg', '.aac'])
    min_file_size_mb: Optional[float] = None
    max_file_size_mb: Optional[float] = None
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    
    # Processing options
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    batch_size: Optional[int] = None
    timeout_seconds: Optional[int] = None
    
    # User preferences
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 0
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.search_paths:
            raise ValueError("At least one search path must be provided")
        
        if self.min_file_size_mb is not None and self.min_file_size_mb < 0:
            raise ValueError("min_file_size_mb must be non-negative")
        
        if self.max_file_size_mb is not None and self.max_file_size_mb < 0:
            raise ValueError("max_file_size_mb must be non-negative")
        
        if (self.min_file_size_mb is not None and self.max_file_size_mb is not None and 
            self.min_file_size_mb > self.max_file_size_mb):
            raise ValueError("min_file_size_mb must be less than or equal to max_file_size_mb")
        
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if self.timeout_seconds is not None and self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
    
    @property
    def total_search_paths(self) -> int:
        """Get total number of search paths."""
        return len(self.search_paths)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'search_paths': self.search_paths,
            'recursive': self.recursive,
            'follow_symlinks': self.follow_symlinks,
            'file_extensions': self.file_extensions,
            'min_file_size_mb': self.min_file_size_mb,
            'max_file_size_mb': self.max_file_size_mb,
            'exclude_patterns': self.exclude_patterns,
            'include_patterns': self.include_patterns,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'timeout_seconds': self.timeout_seconds,
            'user_id': self.user_id,
            'correlation_id': self.correlation_id,
            'priority': self.priority,
            'total_search_paths': self.total_search_paths
        }


@dataclass
class FileDiscoveryResponse:
    """Response from file discovery."""
    request_id: str
    status: str = "completed"  # completed, failed, in_progress
    result: DiscoveryResult = field(default_factory=DiscoveryResult)
    
    # Error information
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate response."""
        if not self.result:
            self.result = DiscoveryResult()
    
    @property
    def is_successful(self) -> bool:
        """Check if discovery was successful."""
        return self.status == "completed" and self.result.valid_audio_files > 0
    
    @property
    def discovered_files(self) -> List[AudioFile]:
        """Get discovered audio files."""
        return self.result.discovered_files
    
    @property
    def total_files(self) -> int:
        """Get total number of files found."""
        return self.result.total_files_found
    
    @property
    def valid_files(self) -> int:
        """Get number of valid audio files."""
        return self.result.valid_audio_files
    
    def get_summary(self) -> Dict[str, Any]:
        """Get discovery summary."""
        return {
            'request_id': self.request_id,
            'status': self.status,
            'total_files': self.total_files,
            'valid_files': self.valid_files,
            'success_rate': self.result.success_rate,
            'total_size_mb': self.result.total_size_mb,
            'file_extensions': self.result.file_extensions,
            'file_sizes': self.result.file_sizes,
            'discovery_time_ms': self.result.discovery_time_ms,
            'total_duration_seconds': self.result.total_duration_seconds,
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'status': self.status,
            'result': self.result.to_dict(),
            'errors': self.errors,
            'warnings': self.warnings,
            'discovered_files_count': len(self.discovered_files),
            'total_files': self.total_files,
            'valid_files': self.valid_files,
            'is_successful': self.is_successful,
            'summary': self.get_summary()
        } 