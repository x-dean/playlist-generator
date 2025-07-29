"""
File system related exceptions.
"""

from .base import PlaylistaException


class FileSystemError(PlaylistaException):
    """Base exception for file system operations."""
    pass


class FileDiscoveryError(FileSystemError):
    """Exception raised when file discovery fails."""
    pass


class FileAccessError(FileSystemError):
    """Exception raised when file access is denied or fails."""
    pass


class PathConversionError(FileSystemError):
    """Exception raised when path conversion fails."""
    pass 