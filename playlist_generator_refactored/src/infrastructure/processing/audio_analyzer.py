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
    spectral_extractor: es.Centroid
    mfcc_extractor: es.MFCC
    metadata_extractor: es.MetadataReader
    mono_loader: es.MonoLoader
    
    # Additional algorithms
    spectral_rolloff: Optional[Any]  # Could be RollOff or Rolloff
    spectral_bandwidth: Optional[Any]  # Could be SpectralBandwidth or None
    spectral_contrast: Optional[Any]  # Could be SpectralContrast or None
    spectral_peaks: Optional[Any]  # Could be SpectralPeaks or None
    hpcp_extractor: Optional[Any]  # Could be HPCP or None
    chord_detector: Optional[Any]  # Could be ChordsDetection or None
    
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
        self._setup_logging()
        self.algorithms = self._initialize_algorithms()
    
    def _initialize_algorithms(self) -> AnalysisAlgorithms:
        """Initialize all Essentia algorithms once."""
        try:
            self.logger.info("Initializing Essentia algorithms...")
            
            # Initialize core algorithms (these should always be available)
            self.logger.debug("Initializing core algorithms...")
            core_algorithms = {
                'rhythm_extractor': es.RhythmExtractor2013(),
                'key_extractor': es.KeyExtractor(),
                'energy_extractor': es.Energy(),
                'loudness_extractor': es.Loudness(),
                'spectral_extractor': es.Centroid(),
                'mfcc_extractor': es.MFCC(),
                'metadata_extractor': es.MetadataReader(),
                'mono_loader': es.MonoLoader(),
            }
            self.logger.debug("Core algorithms initialized successfully")
            
            # Initialize optional algorithms with fallbacks
            self.logger.debug("Initializing optional algorithms...")
            optional_algorithms = {}
            
            # Try to initialize optional spectral algorithms
            try:
                optional_algorithms['spectral_rolloff'] = es.RollOff()
                self.logger.debug("Spectral rolloff algorithm initialized successfully")
            except AttributeError:
                try:
                    optional_algorithms['spectral_rolloff'] = es.Rolloff()
                    self.logger.debug("Spectral rolloff algorithm initialized successfully (alternative name)")
                except AttributeError:
                    optional_algorithms['spectral_rolloff'] = None
                    self.logger.warning("RollOff/Rolloff algorithm not available in Essentia")
            
            try:
                if hasattr(es, 'SpectralBandwidth'):
                    optional_algorithms['spectral_bandwidth'] = es.SpectralBandwidth()
                else:
                    optional_algorithms['spectral_bandwidth'] = None
                    self.logger.warning("SpectralBandwidth algorithm not available in Essentia")
            except Exception as e:
                optional_algorithms['spectral_bandwidth'] = None
                self.logger.warning(f"SpectralBandwidth algorithm not available in Essentia: {e}")
            
            try:
                if hasattr(es, 'SpectralContrast'):
                    optional_algorithms['spectral_contrast'] = es.SpectralContrast()
                else:
                    optional_algorithms['spectral_contrast'] = None
                    self.logger.warning("SpectralContrast algorithm not available in Essentia")
            except Exception as e:
                optional_algorithms['spectral_contrast'] = None
                self.logger.warning(f"SpectralContrast algorithm not available in Essentia: {e}")
            
            try:
                if hasattr(es, 'SpectralPeaks'):
                    optional_algorithms['spectral_peaks'] = es.SpectralPeaks()
                else:
                    optional_algorithms['spectral_peaks'] = None
                    self.logger.warning("SpectralPeaks algorithm not available in Essentia")
            except Exception as e:
                optional_algorithms['spectral_peaks'] = None
                self.logger.warning(f"SpectralPeaks algorithm not available in Essentia: {e}")
            
            try:
                if hasattr(es, 'HPCP'):
                    optional_algorithms['hpcp_extractor'] = es.HPCP()
                else:
                    optional_algorithms['hpcp_extractor'] = None
                    self.logger.warning("HPCP algorithm not available in Essentia")
            except Exception as e:
                optional_algorithms['hpcp_extractor'] = None
                self.logger.warning(f"HPCP algorithm not available in Essentia: {e}")
            
            try:
                if hasattr(es, 'ChordsDetection'):
                    optional_algorithms['chord_detector'] = es.ChordsDetection()
                else:
                    optional_algorithms['chord_detector'] = None
                    self.logger.warning("ChordsDetection algorithm not available in Essentia")
            except Exception as e:
                optional_algorithms['chord_detector'] = None
                self.logger.warning(f"ChordsDetection algorithm not available in Essentia: {e}")
            
            # Combine core and optional algorithms
            all_algorithms = {**core_algorithms, **optional_algorithms}
            
            self.logger.info("All algorithms initialized successfully")
            self.logger.debug(f"Available algorithms: {list(all_algorithms.keys())}")
            
            return AnalysisAlgorithms(**all_algorithms)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Essentia algorithms: {e}")
            # Try with minimal set of algorithms as fallback
            try:
                return AnalysisAlgorithms(
                    rhythm_extractor=es.RhythmExtractor2013(),
                    key_extractor=es.KeyExtractor(),
                    energy_extractor=es.Energy(),
                    loudness_extractor=es.Loudness(),
                    spectral_extractor=es.Centroid(),
                    mfcc_extractor=es.MFCC(),
                    metadata_extractor=es.MetadataReader(),
                    mono_loader=es.MonoLoader(),
                    spectral_rolloff=None,
                    spectral_bandwidth=None,
                    spectral_contrast=None,
                    spectral_peaks=None,
                    hpcp_extractor=None,
                    chord_detector=None
                )
            except AttributeError as e2:
                self.logger.error(f"Failed to initialize even basic algorithms: {e2}")
                raise
    
    def _setup_logging(self):
        """Setup logging for the analyzer."""
        import logging
        self.logger = logging.getLogger(__name__)
    
    def analyze_file(self, file_path: Path) -> AnalysisResult:
        """Analyze a single audio file with full feature extraction."""
        start_time = time.time()
        
        self.logger.info(f"Starting analysis of: {file_path.name}")
        
        try:
            # Create AudioFile entity
            audio_file = AudioFile(file_path=file_path)
            self.logger.debug(f"Created AudioFile entity with ID: {audio_file.id}")
            
            # Load audio
            self.logger.debug(f"Loading audio file: {file_path}")
            audio = self._load_audio(file_path)
            self.logger.debug(f"Audio loaded successfully. Shape: {audio.shape}, Duration: {len(audio)/44100:.2f}s")
            
            # Extract metadata
            self.logger.debug(f"Extracting metadata from: {file_path}")
            metadata = self._extract_metadata(file_path, audio_file.id)
            duration = metadata.additional_metadata.get('duration', 0.0)
            self.logger.debug(f"Metadata extracted: Title='{metadata.title}', Artist='{metadata.artist}', Duration={duration:.2f}s")
            
            # Extract features
            self.logger.debug(f"Extracting audio features from: {file_path}")
            feature_set = self._extract_features(audio, audio_file.id)
            self.logger.debug(f"Features extracted: BPM={feature_set.bpm:.1f}, Key='{feature_set.key}', Energy={feature_set.energy:.3f}")
            
            # Calculate quality score
            self.logger.debug(f"Calculating quality score for: {file_path}")
            quality_score = self._calculate_quality_score(feature_set, metadata)
            self.logger.debug(f"Quality score calculated: {quality_score:.3f}")
            
            # Create analysis result
            processing_time_ms = (time.time() - start_time) * 1000
            result = AnalysisResult(
                audio_file=audio_file,
                feature_set=feature_set,
                metadata=metadata,
                quality_score=quality_score,
                is_successful=True,
                is_complete=True,
                processing_time_ms=processing_time_ms
            )
            
            self.logger.info(f"Analysis completed successfully: {file_path.name} (BPM: {feature_set.bpm:.1f}, Key: {feature_set.key}, Quality: {quality_score:.3f}, Time: {processing_time_ms:.1f}ms)")
            
            # Final memory cleanup
            import gc
            gc.collect()
            
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
            
            self.logger.error(f"Analysis failed for {file_path.name}: {e} (Time: {processing_time_ms:.1f}ms)")
            return result
    
    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load audio file using pre-initialized loader."""
        try:
            self.logger.debug(f"Configuring MonoLoader for: {file_path}")
            self.algorithms.mono_loader.configure(filename=str(file_path))
            
            self.logger.debug("Loading audio data...")
            audio_data = self.algorithms.mono_loader()
            
            # Force garbage collection after loading large audio files
            import gc
            gc.collect()
            
            self.logger.debug(f"Audio loaded successfully: Shape={audio_data.shape}, Sample rate=44100Hz, Duration={len(audio_data)/44100:.2f}s")
            return audio_data
        except Exception as e:
            self.logger.error(f"Failed to load audio file {file_path}: {e}")
            raise
    
    def _extract_metadata(self, file_path: Path, audio_file_id: str) -> Metadata:
        """Extract metadata from audio file."""
        try:
            self.logger.debug(f"Starting metadata extraction for: {file_path}")
            
            # Configure the metadata extractor with the file path
            self.algorithms.metadata_extractor.configure(filename=str(file_path))
            metadata_tuple = self.algorithms.metadata_extractor()
            self.logger.debug(f"Raw metadata tuple: {metadata_tuple}")
            
            # MetadataReader returns a tuple with metadata fields
            # Handle the tuple format more robustly
            title = file_path.stem
            artist = 'Unknown'
            album = 'Unknown'
            duration = 0.0
            
            if isinstance(metadata_tuple, tuple):
                self.logger.debug(f"Processing metadata tuple with {len(metadata_tuple)} elements")
                # Try to extract meaningful metadata from the tuple
                # The format can vary, so we'll be defensive
                for i, item in enumerate(metadata_tuple):
                    if item and isinstance(item, str):
                        self.logger.debug(f"Processing metadata item {i}: '{item}'")
                        # Look for duration-like values (numbers)
                        try:
                            if i == 3 or (isinstance(item, str) and any(c.isdigit() for c in item)):
                                # Try to extract duration from this field
                                import re
                                numbers = re.findall(r'\d+\.?\d*', item)
                                if numbers:
                                    duration = float(numbers[0])
                                    self.logger.debug(f"Extracted duration: {duration}s from item: '{item}'")
                                    break
                        except (ValueError, TypeError):
                            pass
                        
                        # Use the first non-empty string as title if we don't have one
                        if not title or title == file_path.stem:
                            title = item
                            self.logger.debug(f"Set title to: '{title}'")
                        elif not artist or artist == 'Unknown':
                            artist = item
                            self.logger.debug(f"Set artist to: '{artist}'")
                        elif not album or album == 'Unknown':
                            album = item
                            self.logger.debug(f"Set album to: '{album}'")
            else:
                self.logger.debug(f"Metadata tuple is not a tuple: {type(metadata_tuple)}")
            
            self.logger.debug(f"Final metadata: Title='{title}', Artist='{artist}', Album='{album}', Duration={duration:.2f}s")
            
            return Metadata(
                audio_file_id=audio_file_id,
                title=title,
                artist=artist,
                album=album,
                additional_metadata={'duration': duration}
            )
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed for {file_path}: {e}")
            return Metadata(
                audio_file_id=audio_file_id,
                title=file_path.stem,
                artist='Unknown',
                album='Unknown',
                additional_metadata={'duration': 0.0}
            )
    
    def _extract_features(self, audio: np.ndarray, audio_file_id: str) -> FeatureSet:
        """Extract comprehensive audio features."""
        try:
            self.logger.debug(f"Starting feature extraction for audio with shape: {audio.shape}")
            
            # Rhythm features
            self.logger.debug("Extracting rhythm features (BPM, rhythm strength, etc.)")
            bpm, _, _, _ = self.algorithms.rhythm_extractor(audio)
            self.logger.debug(f"Rhythm extraction complete: BPM={bpm:.1f}")
            
            # Harmonic features
            self.logger.debug("Extracting harmonic features (key, scale, key strength)")
            key, scale, key_strength = self.algorithms.key_extractor(audio)
            self.logger.debug(f"Harmonic extraction complete: Key={key} {scale}, Strength={key_strength:.3f}")
            
            # Energy and loudness
            self.logger.debug("Extracting energy and loudness features")
            energy = self.algorithms.energy_extractor(audio)
            loudness = self.algorithms.loudness_extractor(audio)
            self.logger.debug(f"Energy/Loudness extraction complete: Energy={energy:.3f}, Loudness={loudness:.1f}dB")
            
            # Spectral features
            self.logger.debug("Extracting spectral centroid")
            spectral_centroid = self.algorithms.spectral_extractor(audio)
            self.logger.debug(f"Spectral centroid extraction complete: {spectral_centroid:.1f}Hz")
            
            # Optional spectral features
            spectral_rolloff = 0.0
            spectral_bandwidth = 0.0
            
            if self.algorithms.spectral_rolloff:
                self.logger.debug("Extracting spectral rolloff")
                spectral_rolloff = self.algorithms.spectral_rolloff(audio)
                self.logger.debug(f"Spectral rolloff extraction complete: {spectral_rolloff:.1f}Hz")
            else:
                self.logger.debug("Spectral rolloff algorithm not available, using default value")
            
            if self.algorithms.spectral_bandwidth:
                self.logger.debug("Extracting spectral bandwidth")
                spectral_bandwidth = self.algorithms.spectral_bandwidth(audio)
                self.logger.debug(f"Spectral bandwidth extraction complete: {spectral_bandwidth:.1f}Hz")
            else:
                # Manual spectral bandwidth calculation as fallback
                self.logger.debug("Spectral bandwidth algorithm not available, calculating manually")
                spectral_bandwidth = self._calculate_manual_spectral_bandwidth(audio, spectral_centroid)
                self.logger.debug(f"Manual spectral bandwidth calculation complete: {spectral_bandwidth:.1f}Hz")
            
            # MFCC features
            self.logger.debug("Extracting MFCC features")
            mfcc_bands, mfcc_coeffs = self.algorithms.mfcc_extractor(audio)
            self.logger.debug(f"MFCC extraction complete: {len(mfcc_coeffs)} coefficients")
            
            # Optional HPCP features for chord analysis
            chords = "C"
            chord_strength = 0.0
            if self.algorithms.hpcp_extractor and self.algorithms.chord_detector:
                self.logger.debug("Extracting HPCP and chord detection features")
                hpcp = self.algorithms.hpcp_extractor(audio)
                chords, chord_strength = self.algorithms.chord_detector(hpcp)
                self.logger.debug(f"Chord detection complete: Chords={chords}, Strength={chord_strength:.3f}")
            else:
                self.logger.debug("HPCP/Chord detection algorithms not available, using defaults")
            
            # Calculate derived features
            self.logger.debug("Calculating derived features (danceability, valence, acousticness)")
            danceability = self._calculate_danceability(audio, bpm)
            valence = self._calculate_valence(audio, spectral_centroid, energy)
            acousticness = self._calculate_acousticness(audio, spectral_centroid, spectral_rolloff)
            self.logger.debug(f"Derived features calculated: Danceability={danceability:.3f}, Valence={valence:.3f}, Acousticness={acousticness:.3f}")
            
            # Force garbage collection after feature extraction
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            # Return minimal feature set with defaults
            return FeatureSet(
                audio_file_id=audio_file_id,
                bpm=120.0,
                key="C major",
                energy=0.5,
                danceability=0.5,
                valence=0.5,
                acousticness=0.5,
                instrumentalness=0.5,
                liveness=0.5,
                speechiness=0.5,
                loudness=-20.0,
                spectral_centroid=2000.0,
                spectral_rolloff=4000.0,
                spectral_bandwidth=2000.0,
                rhythm_strength=0.5,
                rhythm_regularity=0.5,
                rhythm_clarity=0.5,
                zero_crossing_rate=0.0,
                root_mean_square=0.5,
                spectral_flux=0.0,
                tempo_confidence=0.5,
                key_confidence=0.5,
                extraction_method="essentia_fallback",
                extraction_version="2.1"
            )
        
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
            spectral_centroid=float(spectral_centroid),
            spectral_rolloff=float(spectral_rolloff),
            spectral_bandwidth=float(spectral_bandwidth),
            rhythm_strength=0.5,  # Placeholder
            rhythm_regularity=0.5,  # Placeholder
            rhythm_clarity=0.5,  # Placeholder
            zero_crossing_rate=0.0,  # Would need separate calculation
            root_mean_square=float(energy),  # Use energy as RMS approximation
            spectral_flux=0.0,  # Would need separate calculation
            tempo_confidence=0.8,  # Placeholder
            key_confidence=float(key_strength),
            extraction_method="essentia",
            extraction_version="2.1"
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
    
    def _calculate_manual_spectral_bandwidth(self, audio: np.ndarray, spectral_centroid: float) -> float:
        """Calculate spectral bandwidth manually using FFT."""
        try:
            # Use FFT to get spectrum
            fft = np.fft.fft(audio)
            magnitude_spectrum = np.abs(fft[:len(fft)//2])
            frequencies = np.fft.fftfreq(len(audio), 1.0/44100.0)[:len(fft)//2]
            
            # Calculate spectral bandwidth as weighted standard deviation
            if np.sum(magnitude_spectrum) > 0:
                # Normalize magnitude spectrum
                normalized_magnitude = magnitude_spectrum / np.sum(magnitude_spectrum)
                
                # Calculate weighted mean (spectral centroid)
                weighted_mean = np.sum(frequencies * normalized_magnitude)
                
                # Calculate weighted variance
                weighted_variance = np.sum(normalized_magnitude * (frequencies - weighted_mean)**2)
                
                # Spectral bandwidth is the square root of variance
                bandwidth = np.sqrt(weighted_variance)
                return float(bandwidth)
            else:
                return 0.0
        except Exception as e:
            self.logger.warning(f"Manual spectral bandwidth calculation failed: {e}")
            return 0.0
    
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
            duration = metadata.additional_metadata.get('duration', 0.0)
            duration_score = min(1.0, duration / 300.0)  # Prefer longer tracks
            
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