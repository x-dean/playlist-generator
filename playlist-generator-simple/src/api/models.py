"""
Pydantic models for Playlist Generator API.
Request and response models with validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AnalysisStatus(str, Enum):
    """Analysis status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PlaylistMethod(str, Enum):
    """Playlist generation method enumeration."""
    RANDOM = "random"
    KMEANS = "kmeans"
    SIMILARITY = "similarity"
    GENRE_BASED = "genre_based"


# Request Models
class AnalyzeTrackRequest(BaseModel):
    """Request model for track analysis."""
    file_path: str = Field(..., description="Path to audio file")
    force_reanalysis: bool = Field(False, description="Force reanalysis if already analyzed")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")
        return v


class ImportTracksRequest(BaseModel):
    """Request model for track import."""
    directory_path: str = Field(..., description="Directory path to scan")
    recursive: bool = Field(True, description="Scan subdirectories recursively")
    supported_formats: Optional[List[str]] = Field(
        default=[".mp3", ".flac", ".wav", ".m4a"],
        description="Supported audio formats"
    )


class GeneratePlaylistRequest(BaseModel):
    """Request model for playlist generation."""
    method: PlaylistMethod = Field(..., description="Playlist generation method")
    size: int = Field(..., ge=1, le=1000, description="Number of tracks in playlist")
    name: str = Field(..., min_length=1, max_length=255, description="Playlist name")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Method-specific parameters")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Playlist name cannot be empty")
        return v.strip()


class SearchTracksRequest(BaseModel):
    """Request model for track search."""
    query: str = Field(..., min_length=1, description="Search query")
    search_fields: Optional[List[str]] = Field(
        default=["title", "artist", "album"],
        description="Fields to search in"
    )
    limit: Optional[int] = Field(50, ge=1, le=1000, description="Maximum results")
    offset: Optional[int] = Field(0, ge=0, description="Results offset")


# Response Models
class TrackMetadataResponse(BaseModel):
    """Response model for track metadata."""
    title: str
    artist: str
    album: str
    duration: Optional[float] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class AnalysisResultResponse(BaseModel):
    """Response model for analysis results."""
    features: Dict[str, Any]
    confidence: float
    analysis_date: datetime
    processing_time: Optional[float] = None


class TrackResponse(BaseModel):
    """Response model for tracks."""
    class Config:
        from_attributes = True
    
    id: str
    path: str
    metadata: TrackMetadataResponse
    analysis_result: Optional[AnalysisResultResponse] = None
    created_at: datetime
    updated_at: datetime


class PlaylistResponse(BaseModel):
    """Response model for playlists."""
    class Config:
        from_attributes = True
    
    id: str
    name: str
    tracks: List[TrackResponse]
    created_at: datetime
    updated_at: datetime


class AnalysisStatsResponse(BaseModel):
    """Response model for analysis statistics."""
    total_analyses: int
    average_confidence: float
    total_tracks: int
    analyzed_tracks: int
    pending_tracks: int


class ImportResultResponse(BaseModel):
    """Response model for import results."""
    imported: int
    skipped: int
    errors: int
    total_files: int
    duration: float


class SearchResultResponse(BaseModel):
    """Response model for search results."""
    tracks: List[TrackResponse]
    total: int
    limit: int
    offset: int


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: datetime
    uptime: float
    system_stats: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    metrics: str
    content_type: str = "text/plain; version=0.0.4; charset=utf-8"


# Status Models
class StatusResponse(BaseModel):
    """Response model for operation status."""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now) 