"""
Modular audio analyzer for parallel processing.

This module provides pre-initialized Essentia algorithms for efficient
parallel audio analysis without pickling issues.
"""

import time
import numpy as np
import essentia
import essentia.standard as es
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from domain.entities.audio_file import AudioFile
from domain.entities.analysis_result import AnalysisResult
from domain.entities.feature_set import FeatureSet
from domain.entities.metadata import Metadata


@dataclass
class AnalysisAlgorithms:
    """Pre-initialized Essentia algorithms for efficient processing."""
    
    # Core algorithms
    rhythm_extractor: es.RhythmExtractor2013
    key_extractor: es.KeyExtractor
    energy_extractor: es.Energy
    loudness_extractor: es.Loudness
    spectral_extractor: es.SpectralCentroid
    mfcc_extractor: es.MFCC
    metadata_extractor: es.MetadataReader
    mono_loader: es.MonoLoader
    
    # Additional algorithms
    spectral_rolloff: es.RollOff
    spectral_bandwidth: es.Bandwidth
    spectral_contrast: es.SpectralContrast
    spectral_peaks: es.SpectralPeaks
    hpcp_extractor: es.HPCP
    chord_detector: es.ChordsDetection
    
    def __post_init__(self):
        """Validate algorithm initialization."""
        if not all([
            self.rhythm_extractor, self.key_extractor, self.energy_extractor,
            self.loudness_extractor, self.spectral_extractor, self.mfcc_extractor,
            self.metadata_extractor, self.mono_loader
        ]):
            raise ValueError("All algorithms must be properly initialized")


class AudioAnalyzer:
    """Modular audio analyzer with pre-initialized algorithms."""
    
    def __init__(self):
        """Initialize the analyzer with pre-created algorithms."""
        self.algorithms = self._initialize_algorithms()
        self._setup_logging()
    
    def _initialize_algorithms(self) -> AnalysisAlgorithms:
        """Initialize all Essentia algorithms once."""
        return AnalysisAlgorithms(
            rhythm_extractor=es.RhythmExtractor2013(),
            key_extractor=es.KeyExtractor(),
            energy_extractor=es.Energy(),
            loudness_extractor=es.Loudness(),
            spectral_extractor=es.SpectralCentroid(),
            mfcc_extractor=es.MFCC(),
            metadata_extractor=es.MetadataReader(),
            mono_loader=es.MonoLoader(),
            spectral_rolloff=es.RollOff(),
            spectral_bandwidth=es.Bandwidth(),
            spectral_contrast=es.SpectralContrast(),
            spectral_peaks=es.SpectralPeaks(),
            hpcp_extractor=es.HPCP(),
            chord_detector=es.ChordsDetection()
        )
    
    def _setup_logging(self):
        """Setup logging for the analyzer."""
        import logging
        self.logger = logging.getLogger(__name__)
    
    def analyze_file(self, file_path: Path) -> AnalysisResult:
        """Analyze a single audio file with full feature extraction."""
        start_time = time.time()
        
        try:
            # Create AudioFile entity
            audio_file = AudioFile(file_path=file_path)
            
            # Load audio
            audio = self._load_audio(file_path)
            
            # Extract all features
            metadata = self._extract_metadata(file_path, audio_file.id)
            feature_set = self._extract_features(audio, audio_file.id)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(feature_set, metadata)
            
            # Create analysis result
            result = AnalysisResult(
                audio_file=audio_file,
                feature_set=feature_set,
                metadata=metadata,
                quality_score=quality_score,
                is_successful=True,
                is_complete=True,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            audio_file = AudioFile(file_path=file_path)
            result = AnalysisResult(
                audio_file=audio_file,
                is_successful=False,
                is_complete=True,
                error_message=str(e),
                processing_time_ms=processing_time_ms
            )
            
            return result
    
    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load audio file using pre-initialized loader."""
        self.algorithms.mono_loader.configure(filename=str(file_path))
        return self.algorithms.mono_loader()
    
    def _extract_metadata(self, file_path: Path, audio_file_id: str) -> Metadata:
        """Extract metadata from audio file."""
        try:
            metadata_dict = self.algorithms.metadata_extractor(str(file_path))
            
            return Metadata(
                audio_file_id=audio_file_id,
                title=metadata_dict.get('title', file_path.stem),
                artist=metadata_dict.get('artist', 'Unknown'),
                album=metadata_dict.get('album', 'Unknown'),
                duration=metadata_dict.get('length', 0.0)
            )
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed for {file_path}: {e}")
            return Metadata(
                audio_file_id=audio_file_id,
                title=file_path.stem,
                artist='Unknown',
                album='Unknown',
                duration=0.0
            )
    
    def _extract_features(self, audio: np.ndarray, audio_file_id: str) -> FeatureSet:
        """Extract comprehensive audio features."""
        # Rhythm features
        bpm, _, _, _ = self.algorithms.rhythm_extractor(audio)
        
        # Harmonic features
        key, scale, key_strength = self.algorithms.key_extractor(audio)
        
        # Energy and loudness
        energy = self.algorithms.energy_extractor(audio)
        loudness = self.algorithms.loudness_extractor(audio)
        
        # Spectral features
        spectral_centroid = self.algorithms.spectral_extractor(audio)
        spectral_rolloff = self.algorithms.spectral_rolloff(audio)
        spectral_bandwidth = self.algorithms.spectral_bandwidth(audio)
        
        # MFCC features
        mfcc_bands, mfcc_coeffs = self.algorithms.mfcc_extractor(audio)
        
        # HPCP features for chord analysis
        hpcp = self.algorithms.hpcp_extractor(audio)
        chords, chord_strength = self.algorithms.chord_detector(hpcp)
        
        # Calculate derived features
        danceability = self._calculate_danceability(audio, bpm)
        valence = self._calculate_valence(audio, spectral_centroid, energy)
        acousticness = self._calculate_acousticness(audio, spectral_centroid, spectral_rolloff)
        
        return FeatureSet(
            audio_file_id=audio_file_id,
            bpm=float(bpm),
            key=f"{key} {scale}",
            energy=float(energy),
            danceability=danceability,
            valence=valence,
            acousticness=acousticness,
            instrumentalness=0.5,  # Would need more complex analysis
            liveness=0.5,  # Would need more complex analysis
            speechiness=0.5,  # Would need more complex analysis
            loudness=float(loudness),
            tempo=float(bpm),
            mode=1 if scale == 'major' else 0,
            time_signature=4,  # Default
            duration=len(audio) / 44100.0,
            sample_rate=44100,
            bit_rate=0,
            channels=1
        )
    
    def _calculate_danceability(self, audio: np.ndarray, bpm: float) -> float:
        """Calculate danceability based on rhythm and energy."""
        try:
            # Simple danceability calculation based on BPM and energy
            if bpm < 90:
                return 0.3
            elif bpm < 120:
                return 0.6
            elif bpm < 140:
                return 0.8
            else:
                return 0.9
        except Exception:
            return 0.5
    
    def _calculate_valence(self, audio: np.ndarray, spectral_centroid: float, energy: float) -> float:
        """Calculate valence (positivity) based on spectral features."""
        try:
            # Simple valence calculation
            # Higher spectral centroid and energy = more positive
            normalized_centroid = min(1.0, spectral_centroid / 5000.0)
            normalized_energy = min(1.0, energy / 0.1)
            return (normalized_centroid + normalized_energy) / 2.0
        except Exception:
            return 0.5
    
    def _calculate_acousticness(self, audio: np.ndarray, spectral_centroid: float, spectral_rolloff: float) -> float:
        """Calculate acousticness based on spectral characteristics."""
        try:
            # Lower spectral centroid and rolloff = more acoustic
            normalized_centroid = max(0.0, 1.0 - (spectral_centroid / 5000.0))
            normalized_rolloff = max(0.0, 1.0 - (spectral_rolloff / 8000.0))
            return (normalized_centroid + normalized_rolloff) / 2.0
        except Exception:
            return 0.5
    
    def _calculate_quality_score(self, feature_set: FeatureSet, metadata: Metadata) -> float:
        """Calculate overall quality score based on extracted features."""
        try:
            # Quality factors
            energy_score = min(1.0, feature_set.energy / 0.1)
            loudness_score = min(1.0, (feature_set.loudness + 60) / 60)  # Normalize to 0-1
            duration_score = min(1.0, metadata.duration / 300.0)  # Prefer longer tracks
            
            # Weighted average
            quality_score = (energy_score * 0.3 + loudness_score * 0.3 + duration_score * 0.4)
            return max(0.0, min(1.0, quality_score))
        except Exception:
            return 0.5


# Global analyzer instance for reuse
_analyzer: Optional[AudioAnalyzer] = None


def get_analyzer() -> AudioAnalyzer:
    """Get the global analyzer instance (singleton pattern)."""
    global _analyzer
    if _analyzer is None:
        _analyzer = AudioAnalyzer()
    return _analyzer


def analyze_audio_file(file_path: Path) -> AnalysisResult:
    """Standalone function for parallel processing."""
    analyzer = get_analyzer()
    return analyzer.analyze_file(file_path) 