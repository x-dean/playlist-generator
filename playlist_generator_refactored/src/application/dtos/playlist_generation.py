"""
Playlist generation DTOs for the application layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from uuid import UUID

from domain.entities import Playlist


class PlaylistGenerationMethod(str, Enum):
    """Methods for playlist generation."""
    KMEANS = "kmeans"
    SIMILARITY = "similarity"
    FEATURE_BASED = "feature_based"
    RANDOM = "random"
    TIME_BASED = "time_based"
    TAG_BASED = "tag_based"
    FEATURE_GROUP = "feature_group"
    CACHE = "cache"
    ADVANCED = "advanced"
    MIXED = "mixed"


@dataclass
class PlaylistQualityMetrics:
    """Quality metrics for playlist generation."""
    coherence_score: Optional[float] = None  # 0.0 to 1.0
    diversity_score: Optional[float] = None  # 0.0 to 1.0
    balance_score: Optional[float] = None  # 0.0 to 1.0
    overall_score: Optional[float] = None  # 0.0 to 1.0
    overall_quality: Optional[float] = None  # 0.0 to 1.0
    genre_diversity: Optional[float] = None  # 0.0 to 1.0
    artist_diversity: Optional[float] = None  # 0.0 to 1.0
    tempo_consistency: Optional[float] = None  # 0.0 to 1.0
    energy_flow: Optional[float] = None  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate quality metrics."""
        for field_name, value in [
            ('coherence_score', self.coherence_score),
            ('diversity_score', self.diversity_score),
            ('balance_score', self.balance_score),
            ('overall_score', self.overall_score),
            ('overall_quality', self.overall_quality),
            ('genre_diversity', self.genre_diversity),
            ('artist_diversity', self.artist_diversity),
            ('tempo_consistency', self.tempo_consistency),
            ('energy_flow', self.energy_flow)
        ]:
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")


@dataclass
class PlaylistGenerationRequest:
    """Request for playlist generation."""
    audio_files: List['AudioFile'] = field(default_factory=list)  # Audio files to generate playlists from
    method: PlaylistGenerationMethod = PlaylistGenerationMethod.KMEANS
    playlist_size: Optional[int] = None
    
    # Generation parameters
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Quality constraints
    min_quality_score: Optional[float] = None
    min_coherence_score: Optional[float] = None
    min_diversity_score: Optional[float] = None
    
    # Filtering options
    genre_filter: Optional[List[str]] = None
    artist_filter: Optional[List[str]] = None
    year_range: Optional[tuple] = None  # (start_year, end_year)
    bpm_range: Optional[tuple] = None  # (min_bpm, max_bpm)
    energy_range: Optional[tuple] = None  # (min_energy, max_energy)
    
    # User preferences
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 0
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.audio_files:
            raise ValueError("At least one audio file must be provided")
        
        if self.playlist_size is not None and self.playlist_size < 1:
            raise ValueError("playlist_size must be at least 1")
        
        # Validate quality thresholds
        for threshold_name, threshold_value in [
            ('min_quality_score', self.min_quality_score),
            ('min_coherence_score', self.min_coherence_score),
            ('min_diversity_score', self.min_diversity_score)
        ]:
            if threshold_value is not None and not (0.0 <= threshold_value <= 1.0):
                raise ValueError(f"{threshold_name} must be between 0.0 and 1.0, got {threshold_value}")
        
        # Validate ranges
        if self.year_range and len(self.year_range) != 2:
            raise ValueError("year_range must be a tuple of (start_year, end_year)")
        
        if self.bpm_range and len(self.bpm_range) != 2:
            raise ValueError("bpm_range must be a tuple of (min_bpm, max_bpm)")
        
        if self.energy_range and len(self.energy_range) != 2:
            raise ValueError("energy_range must be a tuple of (min_energy, max_energy)")
    
    @property
    def total_tracks(self) -> int:
        """Get total number of tracks available."""
        return len(self.audio_files)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'audio_files_count': len(self.audio_files),
            'method': self.method.value,
            'playlist_size': self.playlist_size,
            'generation_parameters': self.generation_parameters,
            'min_quality_score': self.min_quality_score,
            'min_coherence_score': self.min_coherence_score,
            'min_diversity_score': self.min_diversity_score,
            'genre_filter': self.genre_filter,
            'artist_filter': self.artist_filter,
            'year_range': self.year_range,
            'bpm_range': self.bpm_range,
            'energy_range': self.energy_range,
            'user_id': self.user_id,
            'correlation_id': self.correlation_id,
            'priority': self.priority,
            'total_tracks': self.total_tracks
        }


@dataclass
class PlaylistGenerationResponse:
    """Response from playlist generation."""
    request_id: str
    status: str = "completed"  # completed, failed, in_progress
    playlist: Optional['Playlist'] = None
    quality_metrics: Optional[PlaylistQualityMetrics] = None
    
    # Generation metadata
    processing_time_ms: Optional[float] = None
    method_used: Optional[str] = None
    
    # Error information
    error_message: Optional[str] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.start_time and self.end_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        if self.playlist:
            self.processing_time_ms = self.total_duration_seconds * 1000 if self.total_duration_seconds else None
            self.method_used = self.playlist.generation_method
    
    @property
    def is_successful(self) -> bool:
        """Check if generation was successful."""
        return self.status == "completed" and self.playlist is not None
    
    @property
    def playlist_count(self) -> int:
        """Get number of generated playlists."""
        return 1 if self.playlist else 0
    
    @property
    def total_playlist_tracks(self) -> int:
        """Get total number of tracks across all playlists."""
        return len(self.playlist.track_ids) if self.playlist else 0
    
    def get_playlist_by_name(self, name: str) -> Optional[Playlist]:
        """Get playlist by name."""
        return self.playlist if self.playlist and self.playlist.name == name else None
    
    def get_playlists_by_type(self, playlist_type: str) -> List[Playlist]:
        """Get playlists by type."""
        return [self.playlist] if self.playlist and self.playlist.generation_method == playlist_type else []
    
    def get_best_playlist(self) -> Optional[Playlist]:
        """Get playlist with highest quality score."""
        return self.playlist if self.playlist and self.playlist.overall_quality is not None else None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get generation summary."""
        return {
            'request_id': self.request_id,
            'status': self.status,
            'playlist_count': self.playlist_count,
            'total_playlist_tracks': self.total_playlist_tracks,
            'processing_time_ms': self.processing_time_ms,
            'method_used': self.method_used,
            'error_message': self.error_message,
            'errors_count': len(self.errors),
            'warnings_count': len(self.warnings)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'status': self.status,
            'playlist': self.playlist.to_dict() if self.playlist else None,
            'quality_metrics': {
                'coherence_score': self.quality_metrics.coherence_score if self.quality_metrics else None,
                'diversity_score': self.quality_metrics.diversity_score if self.quality_metrics else None,
                'overall_quality': self.quality_metrics.overall_quality if self.quality_metrics else None,
                'genre_diversity': self.quality_metrics.genre_diversity if self.quality_metrics else None,
                'artist_diversity': self.quality_metrics.artist_diversity if self.quality_metrics else None,
                'tempo_consistency': self.quality_metrics.tempo_consistency if self.quality_metrics else None,
                'energy_flow': self.quality_metrics.energy_flow if self.quality_metrics else None
            },
            'processing_time_ms': self.processing_time_ms,
            'method_used': self.method_used,
            'error_message': self.error_message,
            'errors': self.errors,
            'warnings': self.warnings,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_seconds': self.total_duration_seconds,
            'playlist_count': self.playlist_count,
            'total_playlist_tracks': self.total_playlist_tracks,
            'is_successful': self.is_successful,
            'summary': self.get_summary()
        } 