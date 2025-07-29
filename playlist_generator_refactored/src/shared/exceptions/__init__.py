"""
Custom exceptions for the Playlista application.
"""

from .base import (
    PlaylistaException,
    ConfigurationError,
    ValidationError
)

from .audio import (
    AudioAnalysisError,
    AudioFileError,
    FeatureExtractionError,
    BPMExtractionError,
    MFCCExtractionError,
    ChromaExtractionError,
    MusicNNAnalysisError
)

from .playlist import (
    PlaylistGenerationError,
    PlaylistValidationError,
    PlaylistMethodError
)

from .file_system import (
    FileDiscoveryError,
    FileAccessError,
    PathConversionError,
    FileSystemError
)

from .database import (
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseMigrationError,
    EntityNotFoundError
)

from .external_api import (
    ExternalAPIError,
    MetadataEnrichmentError,
    MusicBrainzError,
    LastFMError,
    RateLimitError
)

from .cache import (
    CacheError,
    CacheConnectionError,
    CacheKeyError
)

__all__ = [
    # Base exceptions
    'PlaylistaException',
    'ConfigurationError', 
    'ValidationError',
    
    # Audio exceptions
    'AudioAnalysisError',
    'AudioFileError',
    'FeatureExtractionError',
    'BPMExtractionError',
    'MFCCExtractionError',
    'ChromaExtractionError',
    'MusicNNAnalysisError',
    
    # Playlist exceptions
    'PlaylistGenerationError',
    'PlaylistValidationError',
    'PlaylistMethodError',
    
    # File system exceptions
    'FileDiscoveryError',
    'FileAccessError',
    'PathConversionError',
    'FileSystemError',
    
    # Database exceptions
    'DatabaseError',
    'DatabaseConnectionError',
    'DatabaseQueryError',
    'DatabaseMigrationError',
    'EntityNotFoundError',
    
    # External API exceptions
    'ExternalAPIError',
    'MetadataEnrichmentError',
    'MusicBrainzError',
    'LastFMError',
    'RateLimitError',
    
    # Cache exceptions
    'CacheError',
    'CacheConnectionError',
    'CacheKeyError'
] 