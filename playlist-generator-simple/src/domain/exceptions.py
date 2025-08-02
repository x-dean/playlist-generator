"""
Domain exceptions for Playlist Generator.
Specific exceptions for different error scenarios.
"""


class DomainException(Exception):
    """Base domain exception."""
    pass


class TrackNotFoundException(DomainException):
    """Raised when a track is not found."""
    pass


class AnalysisFailedException(DomainException):
    """Raised when audio analysis fails."""
    pass


class PlaylistGenerationException(DomainException):
    """Raised when playlist generation fails."""
    pass


class InvalidTrackException(DomainException):
    """Raised when track data is invalid."""
    pass


class UnsupportedFormatException(DomainException):
    """Raised when audio format is not supported."""
    pass


class MetadataEnrichmentException(DomainException):
    """Raised when metadata enrichment fails."""
    pass


class RepositoryException(DomainException):
    """Raised when repository operations fail."""
    pass 