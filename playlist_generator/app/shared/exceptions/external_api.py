"""
External API integration exceptions.
"""

from .base import PlaylistaException
from typing import Optional, Any, Dict


class ExternalAPIError(PlaylistaException):
    """Base exception for external API errors."""
    
    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.api_name = api_name
        self.endpoint = endpoint
        self.status_code = status_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'api_name': self.api_name,
            'endpoint': self.endpoint,
            'status_code': self.status_code
        })
        return base_dict


class MusicBrainzError(ExternalAPIError):
    """Raised when MusicBrainz API calls fail."""
    
    def __init__(
        self,
        message: str,
        mb_id: Optional[str] = None,
        search_query: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, api_name="MusicBrainz", **kwargs)
        self.mb_id = mb_id
        self.search_query = search_query
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'mb_id': self.mb_id,
            'search_query': self.search_query
        })
        return base_dict


class LastFMError(ExternalAPIError):
    """Raised when Last.fm API calls fail."""
    
    def __init__(
        self,
        message: str,
        track_name: Optional[str] = None,
        artist_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, api_name="Last.fm", **kwargs)
        self.track_name = track_name
        self.artist_name = artist_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'track_name': self.track_name,
            'artist_name': self.artist_name
        })
        return base_dict


class RateLimitError(ExternalAPIError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        rate_limit: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.rate_limit = rate_limit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'retry_after': self.retry_after,
            'rate_limit': self.rate_limit
        })
        return base_dict 