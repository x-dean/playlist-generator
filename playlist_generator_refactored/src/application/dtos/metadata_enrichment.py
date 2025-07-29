"""
Metadata enrichment DTOs for the application layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from uuid import UUID

from domain.entities import Metadata


class EnrichmentSource(str, Enum):
    """Sources for metadata enrichment."""
    MUSICBRAINZ = "musicbrainz"
    LASTFM = "lastfm"
    DISCOGS = "discogs"
    SPOTIFY = "spotify"
    ID3 = "id3"
    MANUAL = "manual"


@dataclass
class EnrichmentResult:
    """Result of metadata enrichment from a single source."""
    source: EnrichmentSource
    success: bool
    confidence: float  # 0.0 to 1.0
    fields_updated: List[str] = field(default_factory=list)
    fields_added: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time_ms: Optional[float] = None
    
    def __post_init__(self):
        """Validate enrichment result."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class MetadataEnrichmentRequest:
    """Request for metadata enrichment."""
    audio_file_ids: List[UUID] = field(default_factory=list)
    sources: List[EnrichmentSource] = field(default_factory=list)
    
    # Enrichment options
    force_reenrichment: bool = False
    skip_existing: bool = True
    confidence_threshold: Optional[float] = None
    
    # Source-specific options
    musicbrainz_options: Dict[str, Any] = field(default_factory=dict)
    lastfm_options: Dict[str, Any] = field(default_factory=dict)
    discogs_options: Dict[str, Any] = field(default_factory=dict)
    spotify_options: Dict[str, Any] = field(default_factory=dict)
    
    # Processing options
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    timeout_seconds: Optional[int] = None
    retry_failed: bool = True
    max_retries: int = 3
    
    # User preferences
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 0
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.audio_file_ids:
            raise ValueError("At least one audio file ID must be provided")
        
        if not self.sources:
            # Default to all sources
            self.sources = [
                EnrichmentSource.MUSICBRAINZ,
                EnrichmentSource.LASTFM,
                EnrichmentSource.DISCOGS
            ]
        
        if self.confidence_threshold is not None and not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        if self.timeout_seconds is not None and self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
    
    @property
    def total_files(self) -> int:
        """Get total number of files to enrich."""
        return len(self.audio_file_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'audio_file_ids_count': len(self.audio_file_ids),
            'sources': [s.value for s in self.sources],
            'force_reenrichment': self.force_reenrichment,
            'skip_existing': self.skip_existing,
            'confidence_threshold': self.confidence_threshold,
            'musicbrainz_options': self.musicbrainz_options,
            'lastfm_options': self.lastfm_options,
            'discogs_options': self.discogs_options,
            'spotify_options': self.spotify_options,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'timeout_seconds': self.timeout_seconds,
            'retry_failed': self.retry_failed,
            'max_retries': self.max_retries,
            'user_id': self.user_id,
            'correlation_id': self.correlation_id,
            'priority': self.priority,
            'total_files': self.total_files
        }


@dataclass
class MetadataEnrichmentResponse:
    """Response from metadata enrichment."""
    request_id: str
    status: str = "completed"  # completed, failed, in_progress
    results: List[EnrichmentResult] = field(default_factory=list)
    enriched_metadata: List[Metadata] = field(default_factory=list)
    
    # Statistics
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    
    # Quality metrics
    average_confidence: Optional[float] = None
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Processing information
    processing_time_ms: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Error information
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.start_time and self.end_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        if self.results:
            confidence_scores = [r.confidence for r in self.results if r.success]
            if confidence_scores:
                self.average_confidence = sum(confidence_scores) / len(confidence_scores)
                
                # Calculate confidence distribution
                for result in self.results:
                    if result.success:
                        if result.confidence >= 0.9:
                            category = "excellent"
                        elif result.confidence >= 0.7:
                            category = "good"
                        elif result.confidence >= 0.5:
                            category = "fair"
                        else:
                            category = "poor"
                        
                        self.confidence_distribution[category] = self.confidence_distribution.get(category, 0) + 1
    
    @property
    def is_successful(self) -> bool:
        """Check if enrichment was successful."""
        return self.status == "completed" and self.successful_files > 0
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100
    
    def get_results_by_source(self, source: EnrichmentSource) -> List[EnrichmentResult]:
        """Get results for a specific source."""
        return [r for r in self.results if r.source == source]
    
    def get_successful_results(self) -> List[EnrichmentResult]:
        """Get successful enrichment results."""
        return [r for r in self.results if r.success]
    
    def get_failed_results(self) -> List[EnrichmentResult]:
        """Get failed enrichment results."""
        return [r for r in self.results if not r.success]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get enrichment summary."""
        return {
            'request_id': self.request_id,
            'status': self.status,
            'total_files': self.total_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files,
            'success_rate': self.success_rate,
            'average_confidence': self.average_confidence,
            'confidence_distribution': self.confidence_distribution,
            'processing_time_ms': self.processing_time_ms,
            'total_duration_seconds': self.total_duration_seconds,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'status': self.status,
            'results': [
                {
                    'source': r.source.value,
                    'success': r.success,
                    'confidence': r.confidence,
                    'fields_updated': r.fields_updated,
                    'fields_added': r.fields_added,
                    'errors': r.errors,
                    'processing_time_ms': r.processing_time_ms
                }
                for r in self.results
            ],
            'enriched_metadata_count': len(self.enriched_metadata),
            'total_files': self.total_files,
            'successful_files': self.successful_files,
            'failed_files': self.failed_files,
            'skipped_files': self.skipped_files,
            'success_rate': self.success_rate,
            'average_confidence': self.average_confidence,
            'confidence_distribution': self.confidence_distribution,
            'processing_time_ms': self.processing_time_ms,
            'errors': self.errors,
            'warnings': self.warnings,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_seconds': self.total_duration_seconds,
            'is_successful': self.is_successful,
            'summary': self.get_summary()
        } 