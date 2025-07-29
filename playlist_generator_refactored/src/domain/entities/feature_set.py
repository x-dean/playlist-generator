"""
FeatureSet entity representing extracted audio features.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from uuid import UUID, uuid4
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a dummy numpy for type hints
    class np:
        class ndarray:
            pass

from shared.exceptions import FeatureExtractionError


@dataclass
class FeatureSet:
    """
    Represents a set of extracted audio features for a music file.
    
    This entity encapsulates all the musical features extracted from
    an audio file, including tempo, key, energy, and other characteristics.
    """
    
    # Core identification
    audio_file_id: UUID = field()
    id: UUID = field(default_factory=uuid4)
    
    # Basic musical features
    bpm: Optional[float] = None
    key: Optional[str] = None
    mode: Optional[str] = None  # major/minor
    energy: Optional[float] = None  # 0.0 to 1.0
    danceability: Optional[float] = None  # 0.0 to 1.0
    valence: Optional[float] = None  # 0.0 to 1.0 (positivity)
    acousticness: Optional[float] = None  # 0.0 to 1.0
    instrumentalness: Optional[float] = None  # 0.0 to 1.0
    liveness: Optional[float] = None  # 0.0 to 1.0
    speechiness: Optional[float] = None  # 0.0 to 1.0
    
    # Spectral features
    spectral_centroid: Optional[float] = None
    spectral_rolloff: Optional[float] = None
    spectral_bandwidth: Optional[float] = None
    spectral_contrast: Optional[List[float]] = None
    
    # MFCC features (Mel-frequency cepstral coefficients)
    mfcc: Optional[np.ndarray] = None  # Shape: (n_frames, n_coefficients)
    mfcc_mean: Optional[List[float]] = None
    mfcc_std: Optional[List[float]] = None
    
    # Chroma features
    chroma: Optional[np.ndarray] = None  # Shape: (12, n_frames)
    chroma_mean: Optional[List[float]] = None
    chroma_std: Optional[List[float]] = None
    
    # Rhythm features
    rhythm_strength: Optional[float] = None
    rhythm_regularity: Optional[float] = None
    rhythm_clarity: Optional[float] = None
    
    # Timbre features
    zero_crossing_rate: Optional[float] = None
    root_mean_square: Optional[float] = None
    spectral_flux: Optional[float] = None
    
    # Advanced features
    loudness: Optional[float] = None  # dB
    tempo_confidence: Optional[float] = None  # 0.0 to 1.0
    key_confidence: Optional[float] = None  # 0.0 to 1.0
    
    # MusicNN features (if available)
    musicnn_features: Optional[np.ndarray] = None  # Shape: (n_frames, 200)
    musicnn_mean: Optional[List[float]] = None
    musicnn_std: Optional[List[float]] = None
    
    # Metadata
    extraction_date: datetime = field(default_factory=datetime.now)
    extraction_method: str = "essentia"
    extraction_version: str = "2.1"
    processing_time_ms: Optional[float] = None
    
    # Confidence scores
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate feature set after initialization."""
        if not self.audio_file_id:
            raise FeatureExtractionError("Audio file ID is required")
        
        # Validate feature ranges
        self._validate_feature_ranges()
    
    def _validate_feature_ranges(self) -> None:
        """Validate that features are within expected ranges."""
        # Energy should be between 0 and 1
        if self.energy is not None and not (0.0 <= self.energy <= 1.0):
            raise FeatureExtractionError(f"Energy must be between 0.0 and 1.0, got {self.energy}")
        
        # BPM should be positive and reasonable
        if self.bpm is not None and self.bpm <= 0:
            raise FeatureExtractionError(f"BPM must be positive, got {self.bpm}")
        
        # Key should be valid if provided
        if self.key is not None:
            valid_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            if self.key not in valid_keys:
                raise FeatureExtractionError(f"Invalid key: {self.key}")
    
    @property
    def tempo_category(self) -> Optional[str]:
        """Get tempo category based on BPM."""
        if self.bpm is None:
            return None
        
        if self.bpm < 60:
            return "largo"
        elif self.bpm < 76:
            return "adagio"
        elif self.bpm < 108:
            return "andante"
        elif self.bpm < 168:
            return "allegro"
        else:
            return "presto"
    
    @property
    def energy_category(self) -> Optional[str]:
        """Get energy category."""
        if self.energy is None:
            return None
        
        if self.energy < 0.3:
            return "low"
        elif self.energy < 0.7:
            return "medium"
        else:
            return "high"
    
    @property
    def mood_category(self) -> Optional[str]:
        """Get mood category based on valence and energy."""
        if self.valence is None or self.energy is None:
            return None
        
        if self.valence > 0.6 and self.energy > 0.6:
            return "happy"
        elif self.valence < 0.4 and self.energy < 0.4:
            return "sad"
        elif self.valence > 0.6 and self.energy < 0.4:
            return "calm"
        elif self.valence < 0.4 and self.energy > 0.6:
            return "angry"
        else:
            return "neutral"
    
    @property
    def is_instrumental(self) -> Optional[bool]:
        """Check if the track is instrumental."""
        if self.instrumentalness is None:
            return None
        return self.instrumentalness > 0.5
    
    @property
    def is_acoustic(self) -> Optional[bool]:
        """Check if the track is acoustic."""
        if self.acousticness is None:
            return None
        return self.acousticness > 0.5
    
    def get_feature_vector(self, feature_names: Optional[List[str]] = None) -> Union[np.ndarray, List[float]]:
        """Get feature vector for machine learning."""
        if feature_names is None:
            feature_names = [
                'bpm', 'energy', 'danceability', 'valence', 'acousticness',
                'instrumentalness', 'liveness', 'speechiness',
                'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
                'zero_crossing_rate', 'root_mean_square', 'loudness'
            ]
        
        features = []
        for name in feature_names:
            value = getattr(self, name, None)
            if value is None:
                value = 0.0  # Default value for missing features
            features.append(float(value))
        
        if NUMPY_AVAILABLE:
            return np.array(features)
        else:
            return features
    
    def get_mfcc_vector(self) -> Optional[Union[np.ndarray, List[float]]]:
        """Get MFCC feature vector."""
        if self.mfcc_mean is None:
            return None
        if NUMPY_AVAILABLE:
            return np.array(self.mfcc_mean)
        else:
            return self.mfcc_mean
    
    def get_chroma_vector(self) -> Optional[Union[np.ndarray, List[float]]]:
        """Get chroma feature vector."""
        if self.chroma_mean is None:
            return None
        if NUMPY_AVAILABLE:
            return np.array(self.chroma_mean)
        else:
            return self.chroma_mean
    
    def get_musicnn_vector(self) -> Optional[Union[np.ndarray, List[float]]]:
        """Get MusicNN feature vector."""
        if self.musicnn_mean is None:
            return None
        if NUMPY_AVAILABLE:
            return np.array(self.musicnn_mean)
        else:
            return self.musicnn_mean
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': str(self.id),
            'audio_file_id': str(self.audio_file_id),
            'bpm': self.bpm,
            'key': self.key,
            'mode': self.mode,
            'energy': self.energy,
            'danceability': self.danceability,
            'valence': self.valence,
            'acousticness': self.acousticness,
            'instrumentalness': self.instrumentalness,
            'liveness': self.liveness,
            'speechiness': self.speechiness,
            'spectral_centroid': self.spectral_centroid,
            'spectral_rolloff': self.spectral_rolloff,
            'spectral_bandwidth': self.spectral_bandwidth,
            'spectral_contrast': self.spectral_contrast,
            'mfcc_mean': self.mfcc_mean,
            'mfcc_std': self.mfcc_std,
            'chroma_mean': self.chroma_mean,
            'chroma_std': self.chroma_std,
            'rhythm_strength': self.rhythm_strength,
            'rhythm_regularity': self.rhythm_regularity,
            'rhythm_clarity': self.rhythm_clarity,
            'zero_crossing_rate': self.zero_crossing_rate,
            'root_mean_square': self.root_mean_square,
            'spectral_flux': self.spectral_flux,
            'loudness': self.loudness,
            'tempo_confidence': self.tempo_confidence,
            'key_confidence': self.key_confidence,
            'musicnn_mean': self.musicnn_mean,
            'musicnn_std': self.musicnn_std,
            'extraction_date': self.extraction_date.isoformat(),
            'extraction_method': self.extraction_method,
            'extraction_version': self.extraction_version,
            'processing_time_ms': self.processing_time_ms,
            'confidence_scores': self.confidence_scores,
            'tempo_category': self.tempo_category,
            'energy_category': self.energy_category,
            'mood_category': self.mood_category,
            'is_instrumental': self.is_instrumental,
            'is_acoustic': self.is_acoustic
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSet':
        """Create FeatureSet from dictionary."""
        # Convert string dates back to datetime objects
        if 'extraction_date' in data and isinstance(data['extraction_date'], str):
            data['extraction_date'] = datetime.fromisoformat(data['extraction_date'])
        
        # Convert string IDs back to UUID objects
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'audio_file_id' in data and isinstance(data['audio_file_id'], str):
            data['audio_file_id'] = UUID(data['audio_file_id'])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"FeatureSet(id={self.id}, audio_file_id={self.audio_file_id}, bpm={self.bpm}, key={self.key})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"FeatureSet(id={self.id}, audio_file_id={self.audio_file_id}, "
                f"bpm={self.bpm}, key={self.key}, energy={self.energy})") 