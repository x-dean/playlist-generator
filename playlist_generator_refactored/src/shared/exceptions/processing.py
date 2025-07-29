"""
Processing-related exceptions.
"""

from .base import PlaylistaException


class ProcessingError(PlaylistaException):
    """Base exception for processing errors."""
    pass


class TimeoutError(PlaylistaException):
    """Exception raised when processing times out."""
    pass


class WorkerError(PlaylistaException):
    """Exception raised when a worker process fails."""
    pass


class MemoryError(PlaylistaException):
    """Exception raised when memory limits are exceeded."""
    pass


class BatchProcessingError(PlaylistaException):
    """Exception raised when batch processing fails."""
    pass 