"""
Cache related exceptions.
"""

from .base import PlaylistaException


class CacheError(PlaylistaException):
    """Base exception for cache operations."""
    pass


class CacheConnectionError(CacheError):
    """Exception raised when cache connection fails."""
    pass


class CacheKeyError(CacheError):
    """Exception raised when cache key operations fail."""
    pass 