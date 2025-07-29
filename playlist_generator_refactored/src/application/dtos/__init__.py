"""
Data Transfer Objects (DTOs) for the application layer.

These objects define the contracts for data exchange between
the application layer and external components.
"""

from .audio_analysis import (
    AudioAnalysisRequest,
    AudioAnalysisResponse,
    AnalysisStatus,
    AnalysisProgress
)

from .playlist_generation import (
    PlaylistGenerationRequest,
    PlaylistGenerationResponse,
    PlaylistGenerationMethod,
    PlaylistQualityMetrics
)

from .metadata_enrichment import (
    MetadataEnrichmentRequest,
    MetadataEnrichmentResponse,
    EnrichmentSource,
    EnrichmentResult
)

from .file_discovery import (
    FileDiscoveryRequest,
    FileDiscoveryResponse,
    DiscoveryFilter,
    DiscoveryResult
)

__all__ = [
    # Audio Analysis DTOs
    'AudioAnalysisRequest',
    'AudioAnalysisResponse',
    'AnalysisStatus',
    'AnalysisProgress',
    
    # Playlist Generation DTOs
    'PlaylistGenerationRequest',
    'PlaylistGenerationResponse',
    'PlaylistGenerationMethod',
    'PlaylistQualityMetrics',
    
    # Metadata Enrichment DTOs
    'MetadataEnrichmentRequest',
    'MetadataEnrichmentResponse',
    'EnrichmentSource',
    'EnrichmentResult',
    
    # File Discovery DTOs
    'FileDiscoveryRequest',
    'FileDiscoveryResponse',
    'DiscoveryFilter',
    'DiscoveryResult'
] 