"""
AnalysisResult entity representing the result of analyzing an audio file.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from .audio_file import AudioFile
from .feature_set import FeatureSet
from .metadata import Metadata

from shared.exceptions import AudioAnalysisError


@dataclass
class AnalysisResult:
    """
    Represents the complete result of analyzing an audio file.
    
    This entity encapsulates the audio file, its extracted features,
    metadata, and analysis status in a single cohesive unit.
    """
    
    # Core identification
    audio_file: AudioFile = field()
    id: UUID = field(default_factory=uuid4)
    feature_set: Optional[FeatureSet] = None
    metadata: Optional[Metadata] = None
    
    # Analysis status
    is_complete: bool = False
    is_successful: bool = False
    analysis_date: datetime = field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    
    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Analysis metadata
    analysis_method: str = "essentia"
    analysis_version: str = "2.1"
    confidence_score: Optional[float] = None  # 0.0 to 1.0
    
    # Processing information
    worker_id: Optional[str] = None
    batch_id: Optional[str] = None
    processing_priority: int = 0  # Higher number = higher priority
    
    # Quality metrics
    quality_score: Optional[float] = None  # 0.0 to 1.0
    quality_issues: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate analysis result after initialization."""
        if not self.audio_file:
            raise AudioAnalysisError("Audio file is required")
        
        # Validate confidence score if provided
        if self.confidence_score is not None and not (0.0 <= self.confidence_score <= 1.0):
            raise AudioAnalysisError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")
        
        # Validate quality score if provided
        if self.quality_score is not None and not (0.0 <= self.quality_score <= 1.0):
            raise AudioAnalysisError(f"Quality score must be between 0.0 and 1.0, got {self.quality_score}")
    
    @property
    def audio_file_id(self) -> UUID:
        """Get the audio file ID."""
        return self.audio_file.id
    
    @property
    def file_path(self) -> str:
        """Get the file path."""
        return str(self.audio_file.file_path)
    
    @property
    def file_name(self) -> str:
        """Get the file name."""
        return self.audio_file.file_name
    
    @property
    def analysis_status(self) -> str:
        """Get the analysis status."""
        if self.is_successful:
            return "successful"
        elif self.error_message:
            return "failed"
        elif self.is_complete:
            return "completed"
        else:
            return "pending"
    
    @property
    def has_features(self) -> bool:
        """Check if features have been extracted."""
        return self.feature_set is not None
    
    @property
    def has_metadata(self) -> bool:
        """Check if metadata has been extracted."""
        return self.metadata is not None
    
    @property
    def is_ready_for_playlist(self) -> bool:
        """Check if the result is ready for playlist generation."""
        return (self.is_successful and 
                self.has_features and 
                self.has_metadata and
                self.quality_score is not None and
                self.quality_score > 0.5)
    
    @property
    def display_title(self) -> str:
        """Get display title from metadata or file name."""
        if self.metadata and self.metadata.display_title:
            return self.metadata.display_title
        return self.file_name
    
    @property
    def display_artist(self) -> str:
        """Get display artist from metadata."""
        if self.metadata and self.metadata.display_artist:
            return self.metadata.display_artist
        return "Unknown Artist"
    
    @property
    def bpm(self) -> Optional[float]:
        """Get BPM from features."""
        if self.feature_set:
            return self.feature_set.bpm
        return None
    
    @property
    def key(self) -> Optional[str]:
        """Get key from features."""
        if self.feature_set:
            return self.feature_set.key
        return None
    
    @property
    def energy(self) -> Optional[float]:
        """Get energy from features."""
        if self.feature_set:
            return self.feature_set.energy
        return None
    
    def mark_as_successful(self, processing_time_ms: Optional[float] = None) -> None:
        """Mark the analysis as successful."""
        self.is_successful = True
        self.is_complete = True
        self.analysis_date = datetime.now()
        self.error_message = None
        self.error_type = None
        if processing_time_ms is not None:
            self.processing_time_ms = processing_time_ms
    
    def mark_as_failed(self, error_message: str, error_type: Optional[str] = None) -> None:
        """Mark the analysis as failed."""
        self.is_successful = False
        self.is_complete = True
        self.analysis_date = datetime.now()
        self.error_message = error_message
        self.error_type = error_type
        self.retry_count += 1
    
    def can_retry(self) -> bool:
        """Check if the analysis can be retried."""
        return self.retry_count < self.max_retries
    
    def reset_for_retry(self) -> None:
        """Reset the result for retry."""
        self.is_complete = False
        self.is_successful = False
        self.error_message = None
        self.error_type = None
    
    def set_feature_set(self, feature_set: FeatureSet) -> None:
        """Set the feature set."""
        if feature_set.audio_file_id != self.audio_file.id:
            raise AudioAnalysisError("Feature set audio file ID must match analysis result audio file ID")
        self.feature_set = feature_set
    
    def set_metadata(self, metadata: Metadata) -> None:
        """Set the metadata."""
        if metadata.audio_file_id != self.audio_file.id:
            raise AudioAnalysisError("Metadata audio file ID must match analysis result audio file ID")
        self.metadata = metadata
    
    def calculate_quality_score(self) -> float:
        """Calculate quality score based on available data."""
        score = 0.0
        factors = 0
        
        # Feature completeness
        if self.feature_set:
            feature_score = 0.0
            feature_count = 0
            
            # Basic features
            if self.feature_set.bpm is not None:
                feature_score += 1.0
                feature_count += 1
            if self.feature_set.key is not None:
                feature_score += 1.0
                feature_count += 1
            if self.feature_set.energy is not None:
                feature_score += 1.0
                feature_count += 1
            
            # Advanced features
            if self.feature_set.mfcc_mean is not None:
                feature_score += 0.5
                feature_count += 1
            if self.feature_set.chroma_mean is not None:
                feature_score += 0.5
                feature_count += 1
            if self.feature_set.musicnn_mean is not None:
                feature_score += 0.5
                feature_count += 1
            
            if feature_count > 0:
                score += (feature_score / feature_count) * 0.6
                factors += 1
        
        # Metadata completeness
        if self.metadata:
            metadata_score = 0.0
            metadata_count = 0
            
            if self.metadata.title:
                metadata_score += 1.0
                metadata_count += 1
            if self.metadata.artist:
                metadata_score += 1.0
                metadata_count += 1
            if self.metadata.album:
                metadata_score += 0.5
                metadata_count += 1
            if self.metadata.genre:
                metadata_score += 0.5
                metadata_count += 1
            
            if metadata_count > 0:
                score += (metadata_score / metadata_count) * 0.4
                factors += 1
        
        # Processing success
        if self.is_successful:
            score += 0.2
            factors += 1
        
        # Calculate final score
        if factors > 0:
            final_score = score / factors
        else:
            final_score = 0.0
        
        self.quality_score = min(1.0, max(0.0, final_score))
        return self.quality_score
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis result."""
        return {
            'id': str(self.id),
            'file_name': self.file_name,
            'display_title': self.display_title,
            'display_artist': self.display_artist,
            'analysis_status': self.analysis_status,
            'bpm': self.bpm,
            'key': self.key,
            'energy': self.energy,
            'has_features': self.has_features,
            'has_metadata': self.has_metadata,
            'quality_score': self.quality_score,
            'processing_time_ms': self.processing_time_ms,
            'analysis_date': self.analysis_date.isoformat(),
            'error_message': self.error_message
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': str(self.id),
            'audio_file': self.audio_file.to_dict(),
            'feature_set': self.feature_set.to_dict() if self.feature_set else None,
            'metadata': self.metadata.to_dict() if self.metadata else None,
            'is_complete': self.is_complete,
            'is_successful': self.is_successful,
            'analysis_date': self.analysis_date.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'analysis_method': self.analysis_method,
            'analysis_version': self.analysis_version,
            'confidence_score': self.confidence_score,
            'worker_id': self.worker_id,
            'batch_id': self.batch_id,
            'processing_priority': self.processing_priority,
            'quality_score': self.quality_score,
            'quality_issues': self.quality_issues,
            'analysis_status': self.analysis_status,
            'has_features': self.has_features,
            'has_metadata': self.has_metadata,
            'is_ready_for_playlist': self.is_ready_for_playlist,
            'display_title': self.display_title,
            'display_artist': self.display_artist,
            'bpm': self.bpm,
            'key': self.key,
            'energy': self.energy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create AnalysisResult from dictionary."""
        # Convert string dates back to datetime objects
        if 'analysis_date' in data and isinstance(data['analysis_date'], str):
            data['analysis_date'] = datetime.fromisoformat(data['analysis_date'])
        
        # Convert string IDs back to UUID objects
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        
        # Convert nested objects
        if 'audio_file' in data and isinstance(data['audio_file'], dict):
            data['audio_file'] = AudioFile.from_dict(data['audio_file'])
        
        if 'feature_set' in data and isinstance(data['feature_set'], dict):
            data['feature_set'] = FeatureSet.from_dict(data['feature_set'])
        
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = Metadata.from_dict(data['metadata'])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"AnalysisResult(id={self.id}, file={self.file_name}, status={self.analysis_status})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"AnalysisResult(id={self.id}, audio_file_id={self.audio_file_id}, "
                f"status={self.analysis_status}, successful={self.is_successful})") 