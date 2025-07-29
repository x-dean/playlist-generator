"""
Playlist generation and validation exceptions.
"""

from .base import PlaylistaException
from typing import Optional, Any, Dict, List


class PlaylistGenerationError(PlaylistaException):
    """Base exception for playlist generation errors."""
    
    def __init__(
        self,
        message: str,
        playlist_method: Optional[str] = None,
        num_tracks: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.playlist_method = playlist_method
        self.num_tracks = num_tracks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'playlist_method': self.playlist_method,
            'num_tracks': self.num_tracks
        })
        return base_dict


class PlaylistValidationError(PlaylistGenerationError):
    """Raised when playlist validation fails."""
    
    def __init__(
        self,
        message: str,
        playlist_name: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.playlist_name = playlist_name
        self.validation_errors = validation_errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'playlist_name': self.playlist_name,
            'validation_errors': self.validation_errors
        })
        return base_dict


class PlaylistMethodError(PlaylistGenerationError):
    """Raised when an invalid playlist generation method is used."""
    
    def __init__(
        self,
        message: str,
        method_name: Optional[str] = None,
        available_methods: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, playlist_method=method_name, **kwargs)
        self.method_name = method_name
        self.available_methods = available_methods or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'method_name': self.method_name,
            'available_methods': self.available_methods
        })
        return base_dict 