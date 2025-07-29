"""
Audio analysis and feature extraction exceptions.
"""

from .base import PlaylistaException
from typing import Optional, Any, Dict


class AudioAnalysisError(PlaylistaException):
    """Base exception for audio analysis errors."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        analysis_step: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.analysis_step = analysis_step
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'file_path': self.file_path,
            'analysis_step': self.analysis_step
        })
        return base_dict


class AudioFileError(AudioAnalysisError):
    """Raised when there's an error with audio file handling."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        file_format: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, file_path=file_path, **kwargs)
        self.file_size = file_size
        self.file_format = file_format
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'file_size': self.file_size,
            'file_format': self.file_format
        })
        return base_dict


class FeatureExtractionError(AudioAnalysisError):
    """Base exception for feature extraction errors."""
    
    def __init__(
        self,
        message: str,
        feature_name: Optional[str] = None,
        extraction_method: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.feature_name = feature_name
        self.extraction_method = extraction_method
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'feature_name': self.feature_name,
            'extraction_method': self.extraction_method
        })
        return base_dict


class BPMExtractionError(FeatureExtractionError):
    """Raised when BPM extraction fails."""
    
    def __init__(
        self,
        message: str,
        bpm_value: Optional[float] = None,
        confidence: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, feature_name="bpm", **kwargs)
        self.bpm_value = bpm_value
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'bpm_value': self.bpm_value,
            'confidence': self.confidence
        })
        return base_dict


class MFCCExtractionError(FeatureExtractionError):
    """Raised when MFCC extraction fails."""
    
    def __init__(
        self,
        message: str,
        mfcc_coefficients: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, feature_name="mfcc", **kwargs)
        self.mfcc_coefficients = mfcc_coefficients
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'mfcc_coefficients': self.mfcc_coefficients
        })
        return base_dict


class ChromaExtractionError(FeatureExtractionError):
    """Raised when chroma feature extraction fails."""
    
    def __init__(
        self,
        message: str,
        chroma_bins: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, feature_name="chroma", **kwargs)
        self.chroma_bins = chroma_bins
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'chroma_bins': self.chroma_bins
        })
        return base_dict


class MusicNNAnalysisError(FeatureExtractionError):
    """Raised when MusicNN analysis fails."""
    
    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        embedding_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, feature_name="musicnn", **kwargs)
        self.model_path = model_path
        self.embedding_size = embedding_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'model_path': self.model_path,
            'embedding_size': self.embedding_size
        })
        return base_dict 