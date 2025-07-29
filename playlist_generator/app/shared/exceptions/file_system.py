"""
File system and path conversion exceptions.
"""

from .base import PlaylistaException
from typing import Optional, Any, Dict


class FileDiscoveryError(PlaylistaException):
    """Raised when file discovery fails."""
    
    def __init__(
        self,
        message: str,
        directory_path: Optional[str] = None,
        file_pattern: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.directory_path = directory_path
        self.file_pattern = file_pattern
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'directory_path': self.directory_path,
            'file_pattern': self.file_pattern
        })
        return base_dict


class FileAccessError(PlaylistaException):
    """Raised when file access fails."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        access_mode: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.access_mode = access_mode
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'file_path': self.file_path,
            'access_mode': self.access_mode
        })
        return base_dict


class PathConversionError(PlaylistaException):
    """Raised when path conversion between host and container fails."""
    
    def __init__(
        self,
        message: str,
        source_path: Optional[str] = None,
        target_path: Optional[str] = None,
        conversion_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.source_path = source_path
        self.target_path = target_path
        self.conversion_type = conversion_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'source_path': self.source_path,
            'target_path': self.target_path,
            'conversion_type': self.conversion_type
        })
        return base_dict 