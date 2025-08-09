"""
Optimized Audio Analysis Pipeline with Essentia & MusiCNN.

This module implements a balanced, resource-efficient, and accurate method for analyzing audio
using Essentia and MusiCNN, designed for tracks ranging from 5 MB to 200+ MB.

Key features:
- Tiered, chunk-based processing pipeline
- Downsample and downmix audio for efficiency  
- Extract representative segments instead of analyzing entire track
- Essentia runs first to detect interesting audio segments
- MusiCNN runs only on selected segments
- Cache results to avoid reprocessing
"""

import os
import sys

import hashlib
import subprocess
import tempfile
from typing import Dict, Any, Optional, Tuple, List, Generator
from datetime import datetime
import numpy as np
from pathlib import Path

# Import local modules
from .logging_setup import get_logger, log_universal
from .config_loader import config_loader
from .lazy_imports import (
    get_essentia, is_essentia_available, 
    is_tensorflow_available, is_librosa_available
)
from .musicnn_integration import get_musicnn_integration

logger = get_logger('playlista.optimized_pipeline')

# Check for required libraries
ESSENTIA_AVAILABLE = is_essentia_available()
TENSORFLOW_AVAILABLE = is_tensorflow_available()
LIBROSA_AVAILABLE = is_librosa_available()

# Pipeline configuration constants
DEFAULT_OPTIMIZED_SAMPLE_RATE = 22050  # Good compromise for MusiCNN accuracy
DEFAULT_SEGMENT_LENGTH = 30  # seconds
DEFAULT_MIN_TRACK_LENGTH = 180  # 3 minutes - threshold for chunk processing
DEFAULT_MAX_SEGMENTS = 4
DEFAULT_MIN_SEGMENTS = 2


class OptimizedAudioPipeline:
    """
    Optimized audio analysis pipeline implementing the efficient chunk-based approach.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the optimized pipeline."""
        self.config = config or config_loader.get_audio_analysis_config()
        
        # Pipeline settings - configurable via config
        self.optimized_sample_rate = self.config.get('OPTIMIZED_SAMPLE_RATE', DEFAULT_OPTIMIZED_SAMPLE_RATE)
        self.segment_length = self.config.get('SEGMENT_LENGTH', DEFAULT_SEGMENT_LENGTH)
        self.min_track_length = self.config.get('MIN_TRACK_LENGTH', DEFAULT_MIN_TRACK_LENGTH)
        self.max_segments = self.config.get('MAX_SEGMENTS', DEFAULT_MAX_SEGMENTS)
        self.min_segments = self.config.get('MIN_SEGMENTS', DEFAULT_MIN_SEGMENTS)
        
        # Resource optimization settings
        self.resource_mode = self.config.get('PIPELINE_RESOURCE_MODE', 'balanced')  # low, balanced, high_accuracy
        self._configure_resource_settings()
        
        # Cache settings
        self.cache_enabled = self.config.get('CACHE_ENABLED', True)
        # Cache is now handled by PostgreSQL database - no file cache needed
        
        # Initialize MusiCNN integration
        self.musicnn = get_musicnn_integration(self.musicnn_model_size)
        
        log_universal('INFO', 'OptimizedPipeline', 
                     f'Initialized with sample_rate={self.optimized_sample_rate}, '
                     f'segment_length={self.segment_length}s, resource_mode={self.resource_mode}')
    
    def _configure_resource_settings(self):
        """Configure resource settings based on mode."""
        resource_configs = {
            'low': {
                'sample_rate': 16000,
                'channels': 1,
                'segment_length': 15,
                'max_segments': 2,
                'musicnn_model': 'compact'
            },
            'balanced': {
                'sample_rate': 22050,
                'channels': 1,
                'segment_length': 30,
                'max_segments': 4,
                'musicnn_model': 'standard'
            },
            'high_accuracy': {
                'sample_rate': 44100,
                'channels': 2,
                'segment_length': 60,
                'max_segments': 6,
                'musicnn_model': 'large'
            }
        }
        
        mode_config = resource_configs.get(self.resource_mode, resource_configs['balanced'])
        
        # Update settings based on resource mode
        self.optimized_sample_rate = mode_config['sample_rate']
        self.segment_length = mode_config['segment_length']
        self.max_segments = mode_config['max_segments']
        self.target_channels = mode_config['channels']
        self.musicnn_model_size = mode_config['musicnn_model']
        
        log_universal('INFO', 'OptimizedPipeline', 
                     f'Resource mode {self.resource_mode}: sr={self.optimized_sample_rate}, '
                     f'segments={self.max_segments}, model={self.musicnn_model_size}')
    
    def analyze_track(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a track using the optimized pipeline.
        
        Args:
            file_path: Path to audio file
            metadata: Optional metadata dictionary
            
        Returns:
            Analysis results dictionary
        """
        try:
            log_universal('INFO', 'OptimizedPipeline', f'Starting optimized analysis for {file_path}')
            
            # Check cache first
            if self.cache_enabled:
                cached_result = self._load_from_cache(file_path)
                if cached_result:
                    log_universal('INFO', 'OptimizedPipeline', f'Loaded cached result for {file_path}')
                    return cached_result
            
            # Get track duration to determine processing strategy
            duration = self._get_track_duration(file_path)
            if duration is None:
                raise Exception("Could not determine track duration")
            
            log_universal('INFO', 'OptimizedPipeline', f'Track duration: {duration:.1f}s')
            
            # Apply appropriate processing strategy
            if duration <= self.min_track_length:
                # Short track - analyze full track
                result = self._analyze_full_track(file_path, metadata)
            else:
                # Long track - use optimized chunk-based approach
                result = self._analyze_with_chunks(file_path, duration, metadata)
            
            # Add pipeline metadata
            result['pipeline_info'] = {
                'method': 'optimized_pipeline',
                'resource_mode': self.resource_mode,
                'duration': duration,
                'processing_strategy': 'full_track' if duration <= self.min_track_length else 'chunk_based',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            if self.cache_enabled:
                self._save_to_cache(file_path, result)
            
            log_universal('INFO', 'OptimizedPipeline', f'Optimized analysis completed for {file_path}')
            return result
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedPipeline', f'Analysis failed for {file_path}: {e}')
            raise
    
    def _get_track_duration(self, file_path: str) -> Optional[float]:
        """Get track duration using FFmpeg probe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration = float(data['format']['duration'])
                return duration
            else:
                log_universal('WARNING', 'OptimizedPipeline', 
                             f'FFprobe failed for {file_path}: {result.stderr}')
                return None
                
        except Exception as e:
            log_universal('WARNING', 'OptimizedPipeline', 
                         f'Duration detection failed for {file_path}: {e}')
            return None
    
    def _analyze_full_track(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze full track for short audio files."""
        log_universal('INFO', 'OptimizedPipeline', f'Analyzing full track: {file_path}')
        
        # Load and preprocess audio
        audio, sample_rate = self._load_and_preprocess_audio(file_path)
        if audio is None:
            raise Exception("Failed to load audio")
        
        # Extract features using both Essentia and MusiCNN
        features = {}
        
        # Step 1: Essentia low-level features
        essentia_features = self._extract_essentia_features(audio, sample_rate)
        features.update(essentia_features)
        
        # Step 2: MusiCNN high-level features
        musicnn_features = self._extract_musicnn_features(audio, sample_rate)
        features.update(musicnn_features)
        
        return features
    
    def _analyze_with_chunks(self, file_path: str, duration: float, 
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze track using optimized chunk-based approach."""
        log_universal('INFO', 'OptimizedPipeline', 
                     f'Analyzing with chunks: {file_path} (duration: {duration:.1f}s)')
        
        # Step 1: Extract representative segments using intelligent selection
        segments = self._extract_representative_segments(file_path, duration)
        if not segments:
            raise Exception("Failed to extract representative segments")
        
        log_universal('INFO', 'OptimizedPipeline', f'Extracted {len(segments)} representative segments')
        
        # Step 2: Analyze each segment
        segment_results = []
        for i, (audio_segment, start_time, end_time) in enumerate(segments):
            log_universal('DEBUG', 'OptimizedPipeline', 
                         f'Analyzing segment {i+1}/{len(segments)} ({start_time:.1f}-{end_time:.1f}s)')
            
            # Essentia features for this segment
            essentia_features = self._extract_essentia_features(audio_segment, self.optimized_sample_rate)
            
            # MusiCNN features for this segment  
            musicnn_features = self._extract_musicnn_features(audio_segment, self.optimized_sample_rate)
            
            segment_result = {
                'essentia': essentia_features,
                'musicnn': musicnn_features,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            }
            segment_results.append(segment_result)
        
        # Step 3: Aggregate results from all segments
        aggregated_results = self._aggregate_segment_results(segment_results)
        
        return aggregated_results
    
    def _extract_representative_segments(self, file_path: str, duration: float) -> List[Tuple[np.ndarray, float, float]]:
        """
        Extract representative segments using Essentia-based intelligent selection.
        
        Returns:
            List of (audio_segment, start_time, end_time) tuples
        """
        segments = []
        
        try:
            # Calculate number of segments based on track duration
            num_segments = min(self.max_segments, max(self.min_segments, int(duration / 60)))
            
            if num_segments <= 2:
                # For shorter tracks, use fixed positions
                segment_positions = self._get_fixed_segment_positions(duration, num_segments)
            else:
                # For longer tracks, use intelligent segment selection
                segment_positions = self._get_intelligent_segment_positions(file_path, duration, num_segments)
            
            # Extract audio segments
            for start_time, end_time in segment_positions:
                audio_segment = self._extract_audio_segment(file_path, start_time, end_time)
                if audio_segment is not None:
                    segments.append((audio_segment, start_time, end_time))
            
            log_universal('INFO', 'OptimizedPipeline', 
                         f'Extracted {len(segments)} segments from {duration:.1f}s track')
            
            return segments
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedPipeline', f'Segment extraction failed: {e}')
            return []
    
    def _get_fixed_segment_positions(self, duration: float, num_segments: int) -> List[Tuple[float, float]]:
        """Get fixed segment positions for shorter tracks."""
        positions = []
        
        if num_segments == 1:
            # Single segment from middle
            start = max(0, (duration - self.segment_length) / 2)
            end = min(duration, start + self.segment_length)
            positions.append((start, end))
        elif num_segments == 2:
            # Beginning and end segments
            positions.append((0, min(self.segment_length, duration / 2)))
            start_end = max(duration - self.segment_length, duration / 2)
            positions.append((start_end, duration))
        else:
            # Distribute evenly across track
            segment_interval = duration / num_segments
            for i in range(num_segments):
                start = i * segment_interval
                end = min(start + self.segment_length, duration)
                positions.append((start, end))
        
        return positions
    
    def _get_intelligent_segment_positions(self, file_path: str, duration: float, 
                                         num_segments: int) -> List[Tuple[float, float]]:
        """
        Get intelligent segment positions using Essentia spectral novelty detection.
        """
        try:
            # Load audio at low resolution for novelty detection
            audio_lowres = self._load_audio_for_novelty_detection(file_path)
            if audio_lowres is None:
                # Fallback to fixed positions
                return self._get_fixed_segment_positions(duration, num_segments)
            
            # Detect novelty using Essentia
            novelty_curve = self._calculate_spectral_novelty(audio_lowres, self.optimized_sample_rate)
            
            # Find peaks in novelty curve
            peak_times = self._find_novelty_peaks(novelty_curve, duration, num_segments)
            
            # Convert peak times to segment positions
            positions = []
            for peak_time in peak_times:
                start = max(0, peak_time - self.segment_length / 2)
                end = min(duration, start + self.segment_length)
                positions.append((start, end))
            
            return positions
            
        except Exception as e:
            log_universal('WARNING', 'OptimizedPipeline', 
                         f'Intelligent segmentation failed, using fixed positions: {e}')
            return self._get_fixed_segment_positions(duration, num_segments)
    
    def _load_and_preprocess_audio(self, file_path: str, start_time: float = None, 
                                 end_time: float = None) -> Tuple[Optional[np.ndarray], int]:
        """
        Load and preprocess audio using FFmpeg streaming.
        
        Args:
            file_path: Path to audio file
            start_time: Optional start time for segment extraction
            end_time: Optional end time for segment extraction
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Build FFmpeg command for preprocessing
            cmd = ['ffmpeg', '-i', file_path]
            
            # Add time range if specified
            if start_time is not None:
                cmd.extend(['-ss', str(start_time)])
            if end_time is not None:
                duration = end_time - (start_time or 0)
                cmd.extend(['-t', str(duration)])
            
            # Audio preprocessing: mono, specific sample rate, float32 PCM
            cmd.extend([
                '-ac', str(self.target_channels),  # Mono or stereo based on config
                '-ar', str(self.optimized_sample_rate),  # Target sample rate
                '-f', 'f32le',  # Float32 little-endian format
                '-'  # Output to stdout
            ])
            
            # Run FFmpeg and capture audio data
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            
            if result.returncode != 0:
                log_universal('ERROR', 'OptimizedPipeline', 
                             f'FFmpeg failed: {result.stderr.decode()}')
                return None, None
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(result.stdout, dtype=np.float32)
            
            # Reshape for stereo if needed
            if self.target_channels == 2 and len(audio_data) % 2 == 0:
                audio_data = audio_data.reshape(-1, 2)
                # Convert to mono by averaging channels
                audio_data = np.mean(audio_data, axis=1)
            
            log_universal('DEBUG', 'OptimizedPipeline', 
                         f'Loaded audio: shape={audio_data.shape}, sr={self.optimized_sample_rate}')
            
            return audio_data.astype(np.float32), self.optimized_sample_rate
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedPipeline', f'Audio loading failed: {e}')
            return None, None
    
    def _extract_audio_segment(self, file_path: str, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """Extract a specific audio segment."""
        audio, _ = self._load_and_preprocess_audio(file_path, start_time, end_time)
        return audio
    
    def _load_audio_for_novelty_detection(self, file_path: str) -> Optional[np.ndarray]:
        """Load audio at low resolution for novelty detection."""
        # Use lower sample rate for faster novelty detection
        novelty_sample_rate = 16000
        
        try:
            cmd = [
                'ffmpeg', '-i', file_path,
                '-ac', '1',  # Mono
                '-ar', str(novelty_sample_rate),  # Lower sample rate
                '-f', 'f32le',
                '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            
            if result.returncode == 0:
                audio_data = np.frombuffer(result.stdout, dtype=np.float32)
                return audio_data
            else:
                return None
                
        except Exception:
            return None
    
    def _calculate_spectral_novelty(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Calculate spectral novelty curve using Essentia."""
        if not ESSENTIA_AVAILABLE:
            # Fallback: simple energy-based novelty
            return self._calculate_energy_novelty(audio)
        
        try:
            import essentia.standard as es
            
            # Parameters
            frame_size = 2048
            hop_size = 512
            
            # Initialize algorithms
            windowing = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            
            # Calculate spectra for novelty detection
            spectra = []
            
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                windowed_frame = windowing(frame)
                spectrum_frame = spectrum(windowed_frame)
                spectra.append(spectrum_frame)
            
            # Calculate novelty from spectral differences
            novelty_values = []
            for i in range(1, len(spectra)):
                # Calculate spectral difference
                diff = np.sum(np.abs(spectra[i] - spectra[i-1]))
                novelty_values.append(diff)
            
            # Normalize novelty curve
            novelty_array = np.array(novelty_values)
            if len(novelty_array) > 0 and np.max(novelty_array) > 0:
                novelty_array = novelty_array / np.max(novelty_array)
            
            return novelty_array
            
        except Exception as e:
            log_universal('WARNING', 'OptimizedPipeline', f'Spectral novelty failed: {e}')
            return self._calculate_energy_novelty(audio)
    
    def _calculate_energy_novelty(self, audio: np.ndarray) -> np.ndarray:
        """Calculate simple energy-based novelty as fallback."""
        frame_size = 2048
        hop_size = 512
        
        energy_values = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            energy = np.sum(frame ** 2)
            energy_values.append(energy)
        
        # Calculate novelty as energy differences
        energy_values = np.array(energy_values)
        novelty = np.diff(energy_values)
        novelty = np.maximum(novelty, 0)  # Keep only increases
        
        return novelty
    
    def _find_novelty_peaks(self, novelty_curve: np.ndarray, duration: float, 
                          num_segments: int) -> List[float]:
        """Find peak times in novelty curve."""
        if len(novelty_curve) == 0:
            # No novelty data, use fixed positions
            return [duration * i / num_segments for i in range(num_segments)]
        
        try:
            # Try using scipy for peak detection
            from scipy.signal import find_peaks
            
            # Set minimum height threshold
            min_height = np.mean(novelty_curve) if len(novelty_curve) > 0 else 0
            peaks, _ = find_peaks(novelty_curve, height=min_height, distance=10)
            
        except ImportError:
            # Fallback: simple peak detection
            peaks = []
            if len(novelty_curve) > 2:
                for i in range(1, len(novelty_curve) - 1):
                    if (novelty_curve[i] > novelty_curve[i-1] and 
                        novelty_curve[i] > novelty_curve[i+1] and
                        novelty_curve[i] > np.mean(novelty_curve)):
                        peaks.append(i)
            peaks = np.array(peaks)
        except Exception as e:
            log_universal('WARNING', 'OptimizedPipeline', f'Peak detection failed: {e}')
            peaks = np.array([])
        
        if len(peaks) == 0:
            # No peaks found, use fixed positions
            log_universal('DEBUG', 'OptimizedPipeline', 'No peaks found, using fixed positions')
            return [duration * i / num_segments for i in range(num_segments)]
        
        # Convert peak indices to time
        hop_size = 512
        
        # Calculate time per frame
        time_per_frame = hop_size / self.optimized_sample_rate if self.optimized_sample_rate > 0 else hop_size / 22050
        peak_times = peaks * time_per_frame
        
        # Select most significant peaks if we have too many
        if len(peak_times) > num_segments:
            # Sort by novelty value and take top peaks
            try:
                peak_values = novelty_curve[peaks]
                sorted_indices = np.argsort(peak_values)[::-1]
                selected_peaks = sorted_indices[:num_segments]
                peak_times = peak_times[selected_peaks]
            except IndexError:
                # Fallback to taking first N peaks
                peak_times = peak_times[:num_segments]
        
        # Ensure we have enough peaks
        while len(peak_times) < num_segments:
            # Add evenly distributed peaks
            missing_peaks = num_segments - len(peak_times)
            for i in range(missing_peaks):
                position = duration * (i + 1) / (missing_peaks + 1)
                peak_times = np.append(peak_times, position)
        
        # Ensure peaks are within track duration and allow for segment length
        max_start_time = max(0, duration - self.segment_length)
        peak_times = np.clip(peak_times, 0, max_start_time)
        
        return sorted(peak_times.tolist())
    
    def _extract_essentia_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract Essentia low-level features."""
        features = {}
        
        if not ESSENTIA_AVAILABLE:
            log_universal('WARNING', 'OptimizedPipeline', 'Essentia not available, skipping features')
            return features
        
        try:
            import essentia.standard as es
            
            # Validate and convert audio for Essentia compatibility
            if audio is None or len(audio) == 0:
                log_universal('WARNING', 'OptimizedPipeline', 'Empty audio data, skipping Essentia features')
                return features
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Ensure audio is 1D and float32
            audio = audio.astype(np.float32)
            if len(audio.shape) != 1:
                log_universal('WARNING', 'OptimizedPipeline', 'Audio is not 1D after conversion')
                return features
            
            # Basic validation
            if sample_rate <= 0:
                log_universal('WARNING', 'OptimizedPipeline', f'Invalid sample rate: {sample_rate}')
                sample_rate = 22050  # Fallback sample rate
            
            # Extract key features
            features['duration'] = len(audio) / sample_rate
            
            # Tempo and rhythm
            rhythm_extractor = es.RhythmExtractor2013()
            bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
            features['tempo'] = float(bpm)
            features['tempo_confidence'] = float(beats_confidence)
            
            # Key detection
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio)
            features['key'] = key
            features['scale'] = scale
            features['key_strength'] = float(strength)
            
            # Loudness
            loudness_extractor = es.Loudness()
            loudness = loudness_extractor(audio)
            features['loudness'] = float(loudness)
            
            # Spectral features using correct Essentia algorithm
            windowing = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            spectral_centroid = es.Centroid(range=sample_rate/2)
            
            centroids = []
            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
                windowed_frame = windowing(frame)
                spectrum_frame = spectrum(windowed_frame)
                centroid = spectral_centroid(spectrum_frame)
                centroids.append(centroid)
            
            if centroids:
                features['spectral_centroid_mean'] = float(np.mean(centroids))
                features['spectral_centroid_std'] = float(np.std(centroids))
            
            log_universal('DEBUG', 'OptimizedPipeline', 'Essentia features extracted successfully')
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedPipeline', f'Essentia feature extraction failed: {e}')
        
        return features
    
    def _extract_musicnn_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract MusiCNN high-level features using the integration."""
        if not self.musicnn.is_available():
            log_universal('WARNING', 'OptimizedPipeline', 'MusiCNN not available, skipping features')
            return {'musicnn_available': False}
        
        try:
            # Extract features using MusiCNN integration
            musicnn_result = self.musicnn.extract_features(audio, sample_rate)
            
            if musicnn_result.get('error'):
                log_universal('ERROR', 'OptimizedPipeline', f'MusiCNN extraction error: {musicnn_result["error"]}')
                return {'musicnn_available': False, 'error': musicnn_result['error']}
            
            # Process results
            features = {
                'musicnn_tags': musicnn_result.get('tags', {}),
                'musicnn_embeddings': musicnn_result.get('embeddings', []),
                'musicnn_available': True,
                'musicnn_model_size': musicnn_result.get('model_size', self.musicnn_model_size)
            }
            
            # Add derived predictions
            tags = musicnn_result.get('tags', {})
            if tags:
                features['musicnn_genre'] = self.musicnn.get_genre_prediction(tags)
                features['musicnn_mood'] = self.musicnn.get_mood_prediction(tags)
                features['musicnn_top_tags'] = self.musicnn.get_top_tags(tags, top_k=5)
            
            log_universal('DEBUG', 'OptimizedPipeline', 
                         f'MusiCNN features extracted: {len(tags)} tags, '
                         f'{len(musicnn_result.get("embeddings", []))} embeddings')
            
            return features
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedPipeline', f'MusiCNN feature extraction failed: {e}')
            return {'musicnn_available': False, 'error': str(e)}
    
    def _aggregate_segment_results(self, segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple segments using appropriate methods.
        
        Args:
            segment_results: List of analysis results from each segment
            
        Returns:
            Aggregated results dictionary
        """
        if not segment_results:
            return {}
        
        aggregated = {}
        
        try:
            # Aggregate Essentia features
            essentia_features = [result['essentia'] for result in segment_results if result.get('essentia')]
            if essentia_features:
                aggregated.update(self._aggregate_essentia_features(essentia_features))
            
            # Aggregate MusiCNN features
            musicnn_features = [result['musicnn'] for result in segment_results if result.get('musicnn')]
            if musicnn_features:
                aggregated.update(self._aggregate_musicnn_features(musicnn_features))
            
            # Add segment metadata
            aggregated['segment_info'] = {
                'num_segments': len(segment_results),
                'segment_times': [(r['start_time'], r['end_time']) for r in segment_results],
                'total_analyzed_duration': sum(r['duration'] for r in segment_results)
            }
            
            log_universal('INFO', 'OptimizedPipeline', 
                         f'Aggregated results from {len(segment_results)} segments')
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedPipeline', f'Result aggregation failed: {e}')
        
        return aggregated
    
    def _aggregate_essentia_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate Essentia features using appropriate statistical methods."""
        aggregated = {}
        
        # Numerical features - use mean/std/median
        numerical_features = ['tempo', 'tempo_confidence', 'key_strength', 'loudness', 
                            'spectral_centroid_mean', 'spectral_centroid_std']
        
        for feature in numerical_features:
            values = [f.get(feature) for f in features_list if f.get(feature) is not None]
            if values:
                aggregated[f'{feature}_mean'] = float(np.mean(values))
                aggregated[f'{feature}_std'] = float(np.std(values))
                aggregated[f'{feature}_median'] = float(np.median(values))
        
        # Categorical features - use majority voting
        categorical_features = ['key', 'scale']
        
        for feature in categorical_features:
            values = [f.get(feature) for f in features_list if f.get(feature) is not None]
            if values:
                from collections import Counter
                most_common = Counter(values).most_common(1)
                if most_common:
                    aggregated[feature] = most_common[0][0]
                    aggregated[f'{feature}_confidence'] = most_common[0][1] / len(values)
        
        return aggregated
    
    def _aggregate_musicnn_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate MusiCNN features using probability averaging."""
        aggregated = {}
        
        # Aggregate embeddings by averaging
        embeddings_list = [f.get('musicnn_embeddings', []) for f in features_list]
        embeddings_list = [emb for emb in embeddings_list if emb]
        
        if embeddings_list:
            # Average embeddings across segments
            aggregated['musicnn_embeddings'] = np.mean(embeddings_list, axis=0).tolist()
        
        # Aggregate tags by probability averaging
        tags_list = [f.get('musicnn_tags', {}) for f in features_list]
        tags_list = [tags for tags in tags_list if tags]
        
        if tags_list:
            all_tag_names = set()
            for tags in tags_list:
                all_tag_names.update(tags.keys())
            
            aggregated_tags = {}
            for tag_name in all_tag_names:
                tag_values = [tags.get(tag_name, 0.0) for tags in tags_list]
                aggregated_tags[tag_name] = float(np.mean(tag_values))
            
            aggregated['musicnn_tags'] = aggregated_tags
        
        return aggregated
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for file."""
        # Use file path and modification time for cache key
        try:
            stat = os.stat(file_path)
            key_data = f"{file_path}_{stat.st_mtime}_{stat.st_size}_{self.resource_mode}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def _load_from_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load analysis result from database cache."""
        try:
            # Use database manager to check for existing analysis
            from .database import get_postgresql_manager
            db_manager = get_postgresql_manager()
            return db_manager.get_analysis_result(file_path)
            
        except Exception as e:
            log_universal('WARNING', 'OptimizedPipeline', f'Database cache load failed: {e}')
        
        return None
    
    def _save_to_cache(self, file_path: str, result: Dict[str, Any]):
        """Save analysis result to database cache."""
        try:
            # Note: Actual saving is handled by SingleAnalyzer._save_result()
            # This method is kept for compatibility but does nothing
            # since the database save happens at a higher level
            log_universal('DEBUG', 'OptimizedPipeline', f'Cache save skipped - handled by SingleAnalyzer')
            
        except Exception as e:
            log_universal('WARNING', 'OptimizedPipeline', f'Cache save failed: {e}')
    

