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
    PathConversionError
)

from .database import (
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseMigrationError
)

from .external_api import (
    ExternalAPIError,
    MusicBrainzError,
    LastFMError,
    RateLimitError
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
    
    # Database exceptions
    'DatabaseError',
    'DatabaseConnectionError',
    'DatabaseQueryError',
    'DatabaseMigrationError',
    
    # External API exceptions
    'ExternalAPIError',
    'MusicBrainzError',
    'LastFMError',
    'RateLimitError'
] 