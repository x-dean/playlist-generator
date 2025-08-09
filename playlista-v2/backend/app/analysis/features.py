"""
Advanced feature extraction for audio analysis
Combines traditional signal processing with ML-based features
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.logging import get_logger, LogContext, log_operation_start, log_operation_success, log_operation_error
from ..core.config import get_settings

logger = get_logger("analysis.features")
settings = get_settings()


class FeatureExtractor:
    """Professional feature extraction with comprehensive logging"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_fft = 2048
        self.n_mels = 128
        self._librosa_available = False
        self._essentia_available = False
        
        self._check_dependencies()
        
        logger.info(
            "FeatureExtractor initialized",
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            librosa_available=self._librosa_available,
            essentia_available=self._essentia_available
        )
    
    def _check_dependencies(self) -> None:
        """Check availability of audio processing libraries"""
        try:
            import librosa
            self._librosa_available = True
            logger.info(
                "Librosa available for feature extraction",
                librosa_version=librosa.__version__
            )
        except ImportError:
            logger.warning("Librosa not available - some features will be unavailable")
        
        try:
            import essentia.standard as es
            import essentia
            self._essentia_available = True
            
            # Check if TensorFlow models are available
            try:
                from essentia.standard import TensorflowPredictEffnetDiscogs
                tensorflow_available = True
            except ImportError:
                tensorflow_available = False
            
            logger.info(
                "Essentia available for advanced features",
                essentia_version=essentia.__version__,
                tensorflow_support=tensorflow_available
            )
        except ImportError:
            self._essentia_available = False
            logger.warning("Essentia not available - using basic features only")
    
    async def extract_comprehensive_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive audio features from file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        start_time = time.time()
        
        with LogContext(
            operation="extract_comprehensive_features",
            file_path=Path(audio_path).name,
            file_size_mb=round(Path(audio_path).stat().st_size / 1024 / 1024, 2)
        ):
            log_operation_start(logger, "comprehensive feature extraction", file=audio_path)
            
            try:
                # Load audio
                audio_data, sr = await self._load_audio(audio_path)
                
                # Extract different types of features in parallel
                tasks = [
                    self._extract_basic_features(audio_data, sr),
                    self._extract_spectral_features(audio_data, sr),
                    self._extract_rhythm_features(audio_data, sr),
                    self._extract_harmonic_features(audio_data, sr),
                    self._extract_timbral_features(audio_data, sr),
                    self._extract_essentia_tensorflow_features(audio_data, sr)
                ]
                
                feature_results = await asyncio.gather(*tasks)
                
                # Combine all features
                all_features = {}
                for features in feature_results:
                    all_features.update(features)
                
                duration_ms = (time.time() - start_time) * 1000
                
                log_operation_success(
                    logger,
                    "comprehensive feature extraction",
                    duration_ms,
                    feature_count=len(all_features),
                    audio_duration=len(audio_data) / sr
                )
                
                return all_features
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_operation_error(logger, "comprehensive feature extraction", e, duration_ms)
                raise
    
    async def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with error handling"""
        if not self._librosa_available:
            raise RuntimeError("Librosa required for audio loading")
        
        try:
            import librosa
            
            # Load audio with librosa
            audio_data, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                duration=None  # Load full file
            )
            
            logger.debug(
                "Audio loaded successfully",
                duration_seconds=len(audio_data) / sr,
                sample_rate=sr,
                samples=len(audio_data)
            )
            
            return audio_data, sr
            
        except Exception as e:
            logger.error(
                "Audio loading failed",
                file_path=audio_path,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    async def _extract_basic_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract basic audio features"""
        if not self._librosa_available:
            return {}
        
        try:
            import librosa
            
            features = {}
            
            # Duration
            features["duration"] = len(audio_data) / sr
            
            # RMS Energy
            rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=self.hop_length)[0]
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))
            
            # Loudness (approximated from RMS)
            features["loudness"] = float(np.mean(20 * np.log10(rms + 1e-8)))
            
            logger.debug(
                "Basic features extracted",
                duration=features["duration"],
                loudness=round(features["loudness"], 2)
            )
            
            return features
            
        except Exception as e:
            logger.error(
                "Basic feature extraction failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return {}
    
    async def _extract_spectral_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract spectral features"""
        if not self._librosa_available:
            return {}
        
        try:
            import librosa
            
            features = {}
            
            # Spectral Centroid
            centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )[0]
            features["spectral_centroid"] = float(np.mean(centroid))
            
            # Spectral Bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )[0]
            features["spectral_bandwidth"] = float(np.mean(bandwidth))
            
            # Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )[0]
            features["spectral_rolloff"] = float(np.mean(rolloff))
            
            # Spectral Flatness
            flatness = librosa.feature.spectral_flatness(
                y=audio_data, hop_length=self.hop_length
            )[0]
            features["spectral_flatness"] = float(np.mean(flatness))
            
            # MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data, sr=sr, n_mfcc=13, hop_length=self.hop_length
            )
            features["mfcc_features"] = {
                f"mfcc_{i}": float(np.mean(mfccs[i])) 
                for i in range(13)
            }
            
            logger.debug(
                "Spectral features extracted",
                centroid=round(features["spectral_centroid"], 2),
                bandwidth=round(features["spectral_bandwidth"], 2)
            )
            
            return features
            
        except Exception as e:
            logger.error(
                "Spectral feature extraction failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return {}
    
    async def _extract_rhythm_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract rhythm and tempo features"""
        if not self._librosa_available:
            return {}
        
        try:
            import librosa
            
            features = {}
            
            # Tempo and Beat Tracking
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )
            features["bpm"] = float(tempo)
            features["beat_count"] = len(beats)
            
            # Onset Detection
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )
            features["onset_count"] = len(onset_frames)
            features["onset_rate"] = len(onset_frames) / (len(audio_data) / sr)
            
            # Rhythm Regularity (simplified)
            if len(beats) > 1:
                beat_intervals = np.diff(beats) * self.hop_length / sr
                features["rhythm_regularity"] = float(1.0 / (np.std(beat_intervals) + 1e-8))
            else:
                features["rhythm_regularity"] = 0.0
            
            logger.debug(
                "Rhythm features extracted",
                bpm=round(features["bpm"], 1),
                onset_rate=round(features["onset_rate"], 2)
            )
            
            return features
            
        except Exception as e:
            logger.error(
                "Rhythm feature extraction failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return {}
    
    async def _extract_harmonic_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract harmonic and tonal features"""
        if not self._librosa_available:
            return {}
        
        try:
            import librosa
            
            features = {}
            
            # Chromagram
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )
            features["chroma_features"] = {
                f"chroma_{i}": float(np.mean(chroma[i])) 
                for i in range(12)
            }
            
            # Key estimation (simplified)
            chroma_mean = np.mean(chroma, axis=1)
            key_idx = np.argmax(chroma_mean)
            key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            features["key"] = key_names[key_idx]
            features["key_strength"] = float(chroma_mean[key_idx])
            
            # Harmonic-Percussive Separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
            
            # Harmonic vs Percussive ratio
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy > 0:
                features["harmonicity"] = float(harmonic_energy / total_energy)
                features["percussiveness"] = float(percussive_energy / total_energy)
            else:
                features["harmonicity"] = 0.0
                features["percussiveness"] = 0.0
            
            logger.debug(
                "Harmonic features extracted",
                key=features["key"],
                key_strength=round(features["key_strength"], 3),
                harmonicity=round(features["harmonicity"], 3)
            )
            
            return features
            
        except Exception as e:
            logger.error(
                "Harmonic feature extraction failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return {}
    
    async def _extract_timbral_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract timbral and texture features"""
        if not self._librosa_available:
            return {}
        
        try:
            import librosa
            
            features = {}
            
            # Spectral Contrast
            contrast = librosa.feature.spectral_contrast(
                y=audio_data, sr=sr, hop_length=self.hop_length
            )
            features["spectral_contrast_mean"] = float(np.mean(contrast))
            features["spectral_contrast_std"] = float(np.std(contrast))
            
            # Tonnetz (Tonal Centroid Features)
            tonnetz = librosa.feature.tonnetz(
                y=librosa.effects.harmonic(audio_data), sr=sr
            )
            features["tonnetz_features"] = {
                f"tonnetz_{i}": float(np.mean(tonnetz[i])) 
                for i in range(6)
            }
            
            # Mel-frequency cepstral coefficients (additional)
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length
            )
            
            # Spectral features from mel spectrogram
            features["mel_spectral_mean"] = float(np.mean(mel_spec))
            features["mel_spectral_std"] = float(np.std(mel_spec))
            features["mel_spectral_skew"] = float(
                np.mean((mel_spec - np.mean(mel_spec)) ** 3) / (np.std(mel_spec) ** 3 + 1e-8)
            )
            
            logger.debug(
                "Timbral features extracted",
                contrast_mean=round(features["spectral_contrast_mean"], 3),
                mel_mean=round(features["mel_spectral_mean"], 3)
            )
            
            return features
            
        except Exception as e:
            logger.error(
                "Timbral feature extraction failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return {}


    async def _extract_essentia_tensorflow_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract advanced features using Essentia-TensorFlow models"""
        features = {}
        
        try:
            import essentia.standard as es
            
            # Convert to mono and resample if needed for Essentia
            if sr != 16000:
                import librosa
                audio_16k = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            else:
                audio_16k = audio_data
            
            # Try to use real Essentia TensorFlow models
            try:
                from essentia.standard import TensorflowPredictMusiCNN
                
                # Initialize MusiCNN predictor with model path
                model_path = "/models/msd-musicnn-1.pb"
                
                # Check if model exists and use it
                from pathlib import Path
                if Path(model_path).exists():
                    musicnn_predictor = TensorflowPredictMusiCNN(
                        graphFilename=model_path,
                        output="model/Sigmoid"
                    )
                    
                    # Convert audio to format expected by MusiCNN
                    # MusiCNN expects mono audio at 16kHz
                    predictions = musicnn_predictor(audio_16k.astype(np.float32))
                    
                    # Load MusiCNN tags from JSON config
                    config_path = "/models/msd-musicnn-1.json"
                    if Path(config_path).exists():
                        import json
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            tags = config.get('tags', [])
                    else:
                        # Default MusiCNN tags
                        tags = [
                            'genre---rock', 'genre---pop', 'genre---alternative',
                            'genre---indie', 'genre---electronic', 'genre---female vocals',
                            'genre---dance', 'genre---00s', 'genre---alternative rock'
                        ]
                    
                    # Create tag predictions
                    tag_predictions = {}
                    for i, tag in enumerate(tags[:len(predictions)]):
                        clean_tag = tag.replace('genre---', '').replace('mood---', '')
                        tag_predictions[clean_tag] = float(predictions[i])
                    
                    features["musicnn_real_tags"] = tag_predictions
                    features["musicnn_method"] = "essentia_tensorflow"
                    
                    logger.info(
                        "Real MusiCNN inference completed",
                        predictions_count=len(predictions),
                        top_tag=max(tag_predictions.items(), key=lambda x: x[1])[0] if tag_predictions else "none"
                    )
                    
                else:
                    logger.warning(f"MusiCNN model not found at {model_path}, using simulation")
                    from .musicnn_inference import musicnn_inference
                    musicnn_results = musicnn_inference._simulate_musicnn_output()
                    features["musicnn_tags"] = musicnn_results["tags"]
                    features["musicnn_method"] = "simulation"
                    
            except ImportError as e:
                logger.warning(f"Essentia TensorFlow models not available: {e}")
                from .musicnn_inference import musicnn_inference
                musicnn_results = musicnn_inference._simulate_musicnn_output()
                features["musicnn_tags"] = musicnn_results["tags"]
                features["musicnn_method"] = "simulation_fallback"
            
            # Basic Essentia-style features using librosa (fallback)
            if self._librosa_available:
                import librosa
                
                # Spectral peaks analysis using librosa
                stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft)
                magnitude = np.abs(stft)
                
                # Find spectral peaks for each frame
                peak_frequencies = []
                peak_magnitudes = []
                
                for frame in range(magnitude.shape[1]):
                    frame_magnitude = magnitude[:, frame]
                    peaks = np.where(
                        (frame_magnitude[1:-1] > frame_magnitude[:-2]) &
                        (frame_magnitude[1:-1] > frame_magnitude[2:])
                    )[0] + 1
                    
                    if len(peaks) > 0:
                        peak_freqs = peaks * sr / self.n_fft
                        peak_mags = frame_magnitude[peaks]
                        
                        # Take top 5 peaks
                        top_indices = np.argsort(peak_mags)[-5:]
                        peak_frequencies.extend(peak_freqs[top_indices])
                        peak_magnitudes.extend(peak_mags[top_indices])
                
                if peak_frequencies:
                    features["spectral_peaks"] = {
                        "mean_peak_freq": float(np.mean(peak_frequencies)),
                        "mean_peak_mag": float(np.mean(peak_magnitudes)),
                        "std_peak_freq": float(np.std(peak_frequencies)),
                        "peak_count": len(peak_frequencies)
                    }
            
            logger.debug(
                "Advanced TensorFlow features extracted",
                feature_count=len(features),
                musicnn_enabled=True
            )
            
            return features
            
        except Exception as e:
            logger.error(
                "Advanced feature extraction failed",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            return {}


# Global feature extractor instance
feature_extractor = FeatureExtractor()
