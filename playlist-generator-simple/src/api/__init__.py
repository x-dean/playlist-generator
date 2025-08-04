"""
API layer for Playlist Generator.
REST API implementation using FastAPI.
"""

from .routes import router
from .models import (
    # Enums
    AnalysisStatus,
    PlaylistMethod,
    DatabaseOperation,
    
    # Request Models
    AnalyzeTrackRequest,
    ImportTracksRequest,
    GeneratePlaylistRequest,
    SearchTracksRequest,
    DatabaseManagementRequest,
    
    # Response Models
    TrackMetadataResponse,
    AnalysisResultResponse,
    TrackResponse,
    PlaylistResponse,
    AnalysisStatsResponse,
    ImportResultResponse,
    SearchResultResponse,
    DatabaseManagementResponse,
    DatabaseInfoResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    StatusResponse
)

__all__ = [
    'router',
    # Enums
    'AnalysisStatus',
    'PlaylistMethod', 
    'DatabaseOperation',
    # Request Models
    'AnalyzeTrackRequest',
    'ImportTracksRequest',
    'GeneratePlaylistRequest',
    'SearchTracksRequest',
    'DatabaseManagementRequest',
    # Response Models
    'TrackMetadataResponse',
    'AnalysisResultResponse',
    'TrackResponse',
    'PlaylistResponse',
    'AnalysisStatsResponse',
    'ImportResultResponse',
    'SearchResultResponse',
    'DatabaseManagementResponse',
    'DatabaseInfoResponse',
    'ErrorResponse',
    'HealthResponse',
    'MetricsResponse',
    'StatusResponse'
] 