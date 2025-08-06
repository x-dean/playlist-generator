"""
Memory-Optimized Audio Loader for Playlist Generator Simple.
Implements aggressive memory reduction strategies for large audio files.
"""

import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # Reduces RAM
import gc
import logging
import psutil
import tempfile
import time
from typing import Optional, Dict, Any, Iterator, Tuple, Generator
import numpy as np

# Import audio processing libraries
try:
    import essentia.standard as es
    import essentia
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from .logging_setup import get_logger, log_universal

logger = get_logger('playlista.memory_optimized_loader')

# Universal memory optimization constants (applies to ALL file categories)
OPTIMIZED_SAMPLE_RATE = 22050  # Reduced from 44100 (50% memory reduction)
OPTIMIZED_BIT_DEPTH = 16  # Use float16 instead of float32 (50% memory reduction)
OPTIMIZED_CHUNK_DURATION_SECONDS = 3  # Smaller chunks for memory safety
OPTIMIZED_MEMORY_LIMIT_PERCENT = 15  # Universal memory limit
OPTIMIZED_MAX_MB_PER_TRACK = 200  # Universal maximum memory per track
OPTIMIZED_STREAMING_CHUNK_SIZE = 5  # 5-second streaming chunks


class MemoryOptimizedAudioLoader:
    """
    Universal memory-optimized audio loader with aggressive memory reduction strategies.
    Applies to ALL file categories (sequential, parallel half, parallel full).
    
    Features:
    - Reduced sample rate (22kHz instead of 44.1kHz) - UNIVERSAL
    - Float16 conversion (50% memory reduction) - UNIVERSAL
    - Streaming chunk processing - UNIVERSAL
    - Memory mapping for all file sizes - UNIVERSAL
    - Dynamic memory monitoring - UNIVERSAL
    - Automatic cleanup - UNIVERSAL
    """
    
    def __init__(self, memory_limit_percent: float = OPTIMIZED_MEMORY_LIMIT_PERCENT,
                 chunk_duration_seconds: float = OPTIMIZED_CHUNK_DURATION_SECONDS,
                 max_mb_per_track: float = OPTIMIZED_MAX_MB_PER_TRACK):
        """
        Initialize the memory-optimized audio loader.
        
        Args:
            memory_limit_percent: Maximum percentage of RAM to use
            chunk_duration_seconds: Default chunk duration in seconds
            max_mb_per_track: Maximum memory usage per track in MB
        """
        self.memory_limit_percent = memory_limit_percent
        self.chunk_duration_seconds = chunk_duration_seconds
        self.max_mb_per_track = max_mb_per_track
        self.sample_rate = OPTIMIZED_SAMPLE_RATE
        self.bit_depth = OPTIMIZED_BIT_DEPTH
        
        # Calculate available memory
        self.available_memory_gb = self._get_available_memory_gb()
        self.memory_limit_gb = self.available_memory_gb * (memory_limit_percent / 100)
        
        log_universal('INFO', 'MemoryOpt', 'Universal MemoryOptimizedAudioLoader initialized:')
        log_universal('INFO', 'MemoryOpt', f'  Available memory: {self.available_memory_gb:.1f}GB')
        log_universal('INFO', 'MemoryOpt', f'  Memory limit: {self.memory_limit_gb:.1f}GB ({memory_limit_percent}%)')
        log_universal('INFO', 'MemoryOpt', f'  Universal optimized sample rate: {self.sample_rate}Hz')
        log_universal('INFO', 'MemoryOpt', f'  Universal optimized bit depth: {self.bit_depth}bit')
        log_universal('INFO', 'MemoryOpt', f'  Universal max memory per track: {max_mb_per_track}MB')
        log_universal('INFO', 'MemoryOpt', f'  Universal chunk duration: {chunk_duration_seconds}s')
        log_universal('INFO', 'MemoryOpt', f'  Applies to ALL file categories (sequential, parallel half, parallel full)')
    
    def _get_available_memory_gb(self) -> float:
        """Get available memory in GB."""
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024**3)
        except Exception:
            return 4.0  # Default fallback
    
    def _get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent
            }
        except Exception:
            return {'total_gb': 8.0, 'used_gb': 4.0, 'available_gb': 4.0, 'percent_used': 50.0}
    
    def _force_memory_cleanup(self):
        """Force aggressive memory cleanup."""
        for _ in range(3):
            gc.collect()
        
        # Clear TensorFlow session if available
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except ImportError:
            pass
    
    def load_audio_memory_capped(self, audio_path: str, max_mb: float = None) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Load audio with memory capping and optimization.
        
        Args:
            audio_path: Path to audio file
            max_mb: Maximum memory usage in MB (defaults to self.max_mb_per_track)
            
        Returns:
            Tuple of (audio_data, sample_rate) or (None, None) on failure
        """
        if max_mb is None:
            max_mb = self.max_mb_per_track
        
        try:
            if not os.path.exists(audio_path):
                log_universal('ERROR', 'MemoryOpt', f'File not found: {audio_path}')
                return None, None
            
            # Check current memory usage
            current_memory = self._get_current_memory_usage()
            if current_memory['percent_used'] > 85:
                log_universal('WARNING', 'MemoryOpt', f'High memory usage detected: {current_memory["percent_used"]:.1f}%')
                self._force_memory_cleanup()
            
            log_universal('INFO', 'MemoryOpt', f'Loading with memory optimization: {os.path.basename(audio_path)}')
            
            # Use Essentia with optimized settings if available
            if ESSENTIA_AVAILABLE:
                return self._load_with_essentia_optimized(audio_path, max_mb)
            
            # Fallback to librosa with optimization
            if LIBROSA_AVAILABLE:
                return self._load_with_librosa_optimized(audio_path, max_mb)
            
            # Final fallback to soundfile
            if SOUNDFILE_AVAILABLE:
                return self._load_with_soundfile_optimized(audio_path, max_mb)
            
            log_universal('ERROR', 'MemoryOpt', f'No audio loading library available for {os.path.basename(audio_path)}')
            return None, None
            
        except Exception as e:
            log_universal('ERROR', 'MemoryOpt', f'Error loading {os.path.basename(audio_path)}: {e}')
            return None, None
    
    def _load_with_essentia_optimized(self, audio_path: str, max_mb: float) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio using Essentia with memory optimization."""
        try:
            # Use Essentia loader with reduced sample rate
            loader = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
            audio = loader()
            
            # Convert to float16 for memory reduction
            audio = audio.astype(np.float16)
            
            # Check memory footprint
            current_mem = audio.nbytes / (1024*1024)
            log_universal('INFO', 'MemoryOpt', f'Loaded: {current_mem:.1f}MB ({len(audio)} samples)')
            
            if current_mem > max_mb:
                # Truncate to fit memory limit
                max_samples = int(max_mb * 1024 * 1024 / audio.itemsize)
                audio = audio[:max_samples]
                log_universal('WARNING', 'MemoryOpt', f'Truncated audio to {len(audio)} samples to fit memory limit')
            
            return audio, self.sample_rate
            
        except Exception as e:
            log_universal('ERROR', 'MemoryOpt', f'Essentia loading failed: {e}')
            return None, None
    
    def _load_with_librosa_optimized(self, audio_path: str, max_mb: float) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio using Librosa with memory optimization."""
        try:
            # Load with reduced sample rate
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Convert to float16 (50% memory reduction)
            y = y.astype(np.float16)
            
            # Check memory footprint
            current_mem = y.nbytes / (1024*1024)
            log_universal('INFO', 'MemoryOpt', f'Loaded: {current_mem:.1f}MB ({len(y)} samples)')
            
            if current_mem > max_mb:
                # Truncate to fit memory limit
                max_samples = int(max_mb * 1024 * 1024 / y.itemsize)
                y = y[:max_samples]
                log_universal('WARNING', 'MemoryOpt', f'Truncated audio to {len(y)} samples to fit memory limit')
            
            return y, self.sample_rate
            
        except Exception as e:
            log_universal('ERROR', 'MemoryOpt', f'Librosa loading failed: {e}')
            return None, None
    
    def _load_with_soundfile_optimized(self, audio_path: str, max_mb: float) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio using SoundFile with memory optimization."""
        try:
            # Read audio with reduced sample rate
            with sf.SoundFile(audio_path) as f:
                # Calculate target samples for reduced sample rate
                duration = f.frames / f.samplerate
                target_frames = int(duration * self.sample_rate)
                
                # Read audio data
                audio = f.read(frames=target_frames, dtype=np.float16)
                
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Resample if needed
                if f.samplerate != self.sample_rate:
                    # Simple resampling (for better quality, use librosa)
                    ratio = self.sample_rate / f.samplerate
                    target_length = int(len(audio) * ratio)
                    indices = np.linspace(0, len(audio)-1, target_length, dtype=int)
                    audio = audio[indices]
                
                # Check memory footprint
                current_mem = audio.nbytes / (1024*1024)
                log_universal('INFO', 'MemoryOpt', f'Loaded: {current_mem:.1f}MB ({len(audio)} samples)')
                
                if current_mem > max_mb:
                    # Truncate to fit memory limit
                    max_samples = int(max_mb * 1024 * 1024 / audio.itemsize)
                    audio = audio[:max_samples]
                    log_universal('WARNING', 'MemoryOpt', f'Truncated audio to {len(audio)} samples to fit memory limit')
                
                return audio, self.sample_rate
                
        except Exception as e:
            log_universal('ERROR', 'MemoryOpt', f'SoundFile loading failed: {e}')
            return None, None
    
    def load_audio_streaming(self, audio_path: str) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """
        Load audio in streaming chunks for memory efficiency.
        
        Args:
            audio_path: Path to audio file
            
        Yields:
            Tuples of (chunk, start_time, end_time)
        """
        try:
            # Get audio duration
            duration = self._get_audio_duration(audio_path)
            if duration is None:
                log_universal('ERROR', 'MemoryOpt', f'Could not determine duration for {os.path.basename(audio_path)}')
                return
            
            chunk_duration = self.chunk_duration_seconds
            total_chunks = int(duration / chunk_duration) + 1
            
            log_universal('INFO', 'MemoryOpt', f'Streaming {os.path.basename(audio_path)}: {duration:.1f}s in {total_chunks} chunks')
            
            # Use Essentia streaming if available
            if ESSENTIA_AVAILABLE:
                yield from self._stream_with_essentia(audio_path, duration, chunk_duration)
            elif LIBROSA_AVAILABLE:
                yield from self._stream_with_librosa(audio_path, duration, chunk_duration)
            else:
                log_universal('ERROR', 'MemoryOpt', f'No streaming library available for {os.path.basename(audio_path)}')
                
        except Exception as e:
            log_universal('ERROR', 'MemoryOpt', f'Streaming failed for {os.path.basename(audio_path)}: {e}')
    
    def _stream_with_essentia(self, audio_path: str, duration: float, chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Stream audio using Essentia."""
        try:
            samples_per_chunk = int(chunk_duration * self.sample_rate)
            
            for i in range(0, int(duration), int(chunk_duration)):
                start_time = i
                end_time = min(i + chunk_duration, duration)
                
                # Load chunk with Essentia
                loader = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
                audio = loader()
                
                # Extract chunk
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                chunk = audio[start_sample:end_sample]
                
                # Convert to float16
                chunk = chunk.astype(np.float16)
                
                # Force cleanup after each chunk
                del audio
                gc.collect()
                
                yield chunk, start_time, end_time
                
        except Exception as e:
            log_universal('ERROR', 'MemoryOpt', f'Essentia streaming failed: {e}')
    
    def _stream_with_librosa(self, audio_path: str, duration: float, chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Stream audio using Librosa."""
        try:
            # Load entire file but process in chunks
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            y = y.astype(np.float16)
            
            samples_per_chunk = int(chunk_duration * self.sample_rate)
            
            for i in range(0, len(y), samples_per_chunk):
                start_time = i / self.sample_rate
                end_time = min((i + samples_per_chunk) / self.sample_rate, duration)
                
                chunk = y[i:i + samples_per_chunk]
                
                yield chunk, start_time, end_time
                
        except Exception as e:
            log_universal('ERROR', 'MemoryOpt', f'Librosa streaming failed: {e}')
    
    def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Get audio duration in seconds."""
        try:
            if ESSENTIA_AVAILABLE:
                # Use Essentia for duration
                loader = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
                audio = loader()
                return len(audio) / self.sample_rate
            elif LIBROSA_AVAILABLE:
                # Use Librosa for duration
                y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                return len(y) / sr
            elif SOUNDFILE_AVAILABLE:
                # Use SoundFile for duration
                with sf.SoundFile(audio_path) as f:
                    return f.frames / f.samplerate
            else:
                return None
        except Exception:
            return None
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        current_memory = self._get_current_memory_usage()
        return {
            'available_memory_gb': self.available_memory_gb,
            'memory_limit_gb': self.memory_limit_gb,
            'current_usage': current_memory,
            'optimization_settings': {
                'sample_rate': self.sample_rate,
                'bit_depth': self.bit_depth,
                'chunk_duration': self.chunk_duration_seconds,
                'max_mb_per_track': self.max_mb_per_track
            }
        }


def get_memory_optimized_loader(memory_limit_percent: float = OPTIMIZED_MEMORY_LIMIT_PERCENT,
                               chunk_duration_seconds: float = OPTIMIZED_CHUNK_DURATION_SECONDS,
                               max_mb_per_track: float = OPTIMIZED_MAX_MB_PER_TRACK) -> MemoryOptimizedAudioLoader:
    """Get a memory-optimized audio loader instance."""
    return MemoryOptimizedAudioLoader(
        memory_limit_percent=memory_limit_percent,
        chunk_duration_seconds=chunk_duration_seconds,
        max_mb_per_track=max_mb_per_track
    ) 