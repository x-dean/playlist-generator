"""
Audio analysis DTOs for the application layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from uuid import UUID

from domain.entities import AudioFile, FeatureSet, Metadata, AnalysisResult


class AnalysisStatus(str, Enum):
    """Status of audio analysis."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AnalysisProgress:
    """Progress information for audio analysis."""
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    current_file: Optional[str] = None
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    processing_rate: Optional[float] = None  # files per second
    
    @property
    def percentage_complete(self) -> float:
        """Get percentage of completion."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.processed_files == 0:
            return 0.0
        return (self.successful_files / self.processed_files) * 100
    
    @property
    def remaining_files(self) -> int:
        """Get number of remaining files."""
        return self.total_files - self.processed_files


@dataclass
class AudioAnalysisRequest:
    """Request for audio analysis."""
    file_paths: List[str] = field(default_factory=list)
    analysis_method: str = "essentia"
    analysis_version: str = "2.1"
    force_reanalysis: bool = False
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    batch_size: Optional[int] = None
    timeout_seconds: Optional[int] = None
    priority: int = 0
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Analysis options
    extract_features: bool = True
    extract_metadata: bool = True
    extract_musicnn: bool = False
    quality_threshold: Optional[float] = None
    
    # Processing options
    skip_existing: bool = True
    retry_failed: bool = True
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.file_paths:
            raise ValueError("At least one file path must be provided")
        
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if self.timeout_seconds is not None and self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
        
        if self.quality_threshold is not None and not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError("quality_threshold must be between 0.0 and 1.0")
    
    @property
    def total_files(self) -> int:
        """Get total number of files to analyze."""
        return len(self.file_paths)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'file_paths': self.file_paths,
            'analysis_method': self.analysis_method,
            'analysis_version': self.analysis_version,
            'force_reanalysis': self.force_reanalysis,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'timeout_seconds': self.timeout_seconds,
            'priority': self.priority,
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'extract_features': self.extract_features,
            'extract_metadata': self.extract_metadata,
            'extract_musicnn': self.extract_musicnn,
            'quality_threshold': self.quality_threshold,
            'skip_existing': self.skip_existing,
            'retry_failed': self.retry_failed,
            'max_retries': self.max_retries,
            'total_files': self.total_files
        }


@dataclass
class AudioAnalysisResponse:
    """Response from audio analysis."""
    request_id: str
    status: AnalysisStatus
    progress: AnalysisProgress
    results: List[AnalysisResult] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Quality metrics
    average_quality_score: Optional[float] = None
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    processing_rate: Optional[float] = None  # files per second
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.start_time and self.end_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        if self.total_duration_seconds and self.progress.processed_files > 0:
            self.processing_rate = self.progress.processed_files / self.total_duration_seconds
        
        if self.results:
            quality_scores = [r.quality_score for r in self.results if r.quality_score is not None]
            if quality_scores:
                self.average_quality_score = sum(quality_scores) / len(quality_scores)
                
                # Calculate quality distribution
                for result in self.results:
                    if result.quality_score is not None:
                        if result.quality_score >= 0.8:
                            category = "excellent"
                        elif result.quality_score >= 0.6:
                            category = "good"
                        elif result.quality_score >= 0.4:
                            category = "fair"
                        else:
                            category = "poor"
                        
                        self.quality_distribution[category] = self.quality_distribution.get(category, 0) + 1
    
    @property
    def is_complete(self) -> bool:
        """Check if analysis is complete."""
        return self.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED]
    
    @property
    def is_successful(self) -> bool:
        """Check if analysis was successful."""
        return self.status == AnalysisStatus.COMPLETED
    
    @property
    def successful_results(self) -> List[AnalysisResult]:
        """Get successful analysis results."""
        return [r for r in self.results if r.is_successful]
    
    @property
    def failed_results(self) -> List[AnalysisResult]:
        """Get failed analysis results."""
        return [r for r in self.results if r.is_failed]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        return {
            'request_id': self.request_id,
            'status': self.status.value,
            'total_files': self.progress.total_files,
            'processed_files': self.progress.processed_files,
            'successful_files': self.progress.successful_files,
            'failed_files': self.progress.failed_files,
            'percentage_complete': self.progress.percentage_complete,
            'success_rate': self.progress.success_rate,
            'average_quality_score': self.average_quality_score,
            'quality_distribution': self.quality_distribution,
            'processing_rate': self.processing_rate,
            'total_duration_seconds': self.total_duration_seconds,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'status': self.status.value,
            'progress': {
                'total_files': self.progress.total_files,
                'processed_files': self.progress.processed_files,
                'successful_files': self.progress.successful_files,
                'failed_files': self.progress.failed_files,
                'percentage_complete': self.progress.percentage_complete,
                'success_rate': self.progress.success_rate,
                'current_file': self.progress.current_file,
                'remaining_files': self.progress.remaining_files
            },
            'results_count': len(self.results),
            'errors_count': len(self.errors),
            'summary': self.get_summary(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_seconds': self.total_duration_seconds,
            'average_quality_score': self.average_quality_score,
            'quality_distribution': self.quality_distribution,
            'processing_rate': self.processing_rate,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'is_complete': self.is_complete,
            'is_successful': self.is_successful
        } 