"""
Streaming Audio Loader for Playlist Generator Simple.
Processes audio files in chunks to reduce memory usage and handle large files.
"""

import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import gc
import logging
import psutil
from typing import Optional, Dict, Any, Iterator, Tuple, Generator
import numpy as np
import time # Added for timeout mechanism

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

# Try to import wave module for basic WAV support
try:
    import wave
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False

# Try to import soundfile for broader format support
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from .logging_setup import get_logger, log_universal

logger = get_logger('playlista.streaming_loader')

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHUNK_DURATION_SECONDS = 5  # Reduced to 5 seconds per chunk for very large files
DEFAULT_MEMORY_LIMIT_PERCENT = 15  # Reduced to 15% of available RAM for very large files
MIN_CHUNK_DURATION_SECONDS = 3  # Minimum chunk duration
MAX_CHUNK_DURATION_SECONDS = 10  # Reduced to 10 seconds for memory safety


class StreamingAudioLoader:
    """
    Streaming audio loader that processes audio files in chunks.
    
    Features:
    - Memory-aware chunk sizing
    - Streaming processing for large files
    - Automatic chunk duration calculation
    - Progress tracking
    - Error handling and recovery
    """
    
    def __init__(self, memory_limit_percent: float = DEFAULT_MEMORY_LIMIT_PERCENT,
                 chunk_duration_seconds: float = DEFAULT_CHUNK_DURATION_SECONDS,
                 use_slicer: bool = False, use_streaming: bool = False):
        """
        Initialize the streaming audio loader.
        
        Args:
            memory_limit_percent: Maximum percentage of RAM to use
            chunk_duration_seconds: Default chunk duration in seconds
            use_slicer: If True, use Slicer algorithm instead of FrameCutter
            use_streaming: If True, use proper streaming network (based on tutorials)
        """
        self.memory_limit_percent = memory_limit_percent
        self.chunk_duration_seconds = chunk_duration_seconds
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.use_slicer = use_slicer
        self.use_streaming = use_streaming
        
        # Calculate available memory
        self.available_memory_gb = self._get_available_memory_gb()
        self.memory_limit_gb = self.available_memory_gb * (memory_limit_percent / 100)
        
        log_universal('INFO', 'Streaming', 'StreamingAudioLoader initialized:')
        log_universal('INFO', 'Streaming', f'  Available memory: {self.available_memory_gb:.1f}GB')
        log_universal('INFO', 'Streaming', f'  Memory limit: {self.memory_limit_gb:.1f}GB ({memory_limit_percent}%)')
        log_universal('INFO', 'Streaming', f'  Default chunk duration: {chunk_duration_seconds}s')
        log_universal('INFO', 'Streaming', f'  Sample rate: {self.sample_rate}Hz')
        log_universal('INFO', 'Streaming', f'  Use Slicer: {use_slicer}')
        log_universal('INFO', 'Streaming', f'  Use Streaming: {use_streaming}')
        log_universal('INFO', 'Streaming', f'  Essentia available: {ESSENTIA_AVAILABLE}')
        log_universal('INFO', 'Streaming', f'  Librosa available: {LIBROSA_AVAILABLE}')
    
    def _handle_critical_memory(self):
        """Handle critical memory situations by pausing and forcing cleanup."""
        import time
        
        log_universal('ERROR', 'Streaming', 'CRITICAL MEMORY USAGE! Pausing processing for cleanup...')
        
        # Force aggressive cleanup
        self._force_memory_cleanup()
        
        # Wait a moment for cleanup to take effect
        time.sleep(1)
        
        # Check memory again
        current_memory = self._get_current_memory_usage()
        log_universal('INFO', 'Streaming', f"Memory after cleanup: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
        
        if current_memory['percent_used'] > 85:
            log_universal('ERROR', 'Streaming', f"Memory still critical after cleanup! Consider reducing chunk size or stopping processing.")
    
    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        current_memory = self._get_current_memory_usage()
        return current_memory['percent_used'] > 75  # Memory pressure threshold
    
    def _adjust_for_memory_pressure(self, chunk_duration: float) -> float:
        """Dynamically adjust chunk duration based on memory pressure."""
        if self._check_memory_pressure():
            # Reduce chunk duration by 50% under memory pressure
            adjusted_duration = chunk_duration * 0.5
            log_universal('WARNING', 'Streaming', f"Memory pressure detected! Reducing chunk duration from {chunk_duration:.1f}s to {adjusted_duration:.1f}s")
            return max(adjusted_duration, MIN_CHUNK_DURATION_SECONDS)
        return chunk_duration
    
    def _force_memory_cleanup(self):
        """Force aggressive memory cleanup to prevent saturation."""
        import gc
        import sys
        
        # Force multiple garbage collection cycles
        for _ in range(3):
            gc.collect()
        
        # Clear any cached references
        if hasattr(sys, 'exc_clear'):
            sys.exc_clear()
        
        # Force memory cleanup
        gc.collect()
        
        log_universal('DEBUG', 'Streaming', "Forced memory cleanup completed")
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024 ** 3)  # Convert to GB
        except Exception as e:
            log_universal('WARNING', 'Streaming', f"Could not get memory info: {e}")
            return 4.0  # Default to 4GB
    
    def _calculate_optimal_chunk_duration(self, file_size_mb: float, 
                                        duration_seconds: float) -> float:
        """
        Calculate optimal chunk duration based on file size and available memory.
        
        Args:
            file_size_mb: File size in MB
            duration_seconds: Total audio duration in seconds
            
        Returns:
            Optimal chunk duration in seconds
        """
        # Get current memory usage
        current_memory = self._get_current_memory_usage()
        available_memory_gb = current_memory.get('available_gb', 1.0)
        
        # Check if this is an extremely large file (>500MB)
        if file_size_mb > 500:
            log_universal('WARNING', 'Streaming', f"Extremely large file detected: {file_size_mb:.1f}MB")
            # Use ultra-conservative settings for extremely large files
            conservative_memory_gb = min(available_memory_gb * 0.02, self.memory_limit_gb * 0.02)
            safety_factor = 0.02
            max_chunk_duration = 3.0
        elif file_size_mb > 200:
            log_universal('WARNING', 'Streaming', f"Very large file detected: {file_size_mb:.1f}MB")
            # Use conservative settings for very large files
            conservative_memory_gb = min(available_memory_gb * 0.05, self.memory_limit_gb * 0.05)
            safety_factor = 0.05
            max_chunk_duration = 5.0
        else:
            # Use standard settings for large files
            conservative_memory_gb = min(available_memory_gb * 0.10, self.memory_limit_gb * 0.10)
            safety_factor = 0.10
            max_chunk_duration = 10.0
        
        # Estimate memory usage per second of audio (mono, 44.1kHz, float32)
        bytes_per_second = DEFAULT_SAMPLE_RATE * 4  # 4 bytes per float32 sample
        mb_per_second = bytes_per_second / (1024 ** 2)
        
        # Calculate how many seconds we can fit in memory
        max_seconds_in_memory = conservative_memory_gb * 1024 / mb_per_second
        
        # Use safety factor to leave room for processing
        safe_seconds = max_seconds_in_memory * safety_factor
        
        # Calculate optimal chunk duration
        optimal_duration = min(
            max(safe_seconds, MIN_CHUNK_DURATION_SECONDS),
            MAX_CHUNK_DURATION_SECONDS,
            duration_seconds,  # Don't exceed total duration
            max_chunk_duration  # File size specific maximum
        )
        
        log_universal('INFO', 'Streaming', f"Memory-aware chunk calculation:")
        log_universal('INFO', 'Streaming', f"  File size: {file_size_mb:.1f}MB")
        log_universal('INFO', 'Streaming', f"  Duration: {duration_seconds:.1f}s")
        log_universal('INFO', 'Streaming', f"  Available memory: {available_memory_gb:.1f}GB")
        log_universal('INFO', 'Streaming', f"  Conservative memory limit: {conservative_memory_gb:.1f}GB")
        log_universal('INFO', 'Streaming', f"  Safety factor: {safety_factor}")
        log_universal('INFO', 'Streaming', f"  Optimal chunk duration: {optimal_duration:.1f}s")
        
        return optimal_duration
    
    def _get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024 ** 3),
                'available_gb': memory.available / (1024 ** 3),
                'used_gb': memory.used / (1024 ** 3),
                'percent_used': memory.percent
            }
        except Exception as e:
            log_universal('WARNING', 'Streaming', f"Could not get memory info: {e}")
            return {'available_gb': 1.0, 'percent_used': 0.0}
    
    def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """
        Get audio file duration without loading the entire file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds, or None if failed
        """
        try:
            if ESSENTIA_AVAILABLE:
                # Use Essentia MonoLoader for duration - load audio and calculate duration
                log_universal('DEBUG', 'Streaming', f"Getting duration with Essentia MonoLoader: {os.path.basename(audio_path)}")
                loader = es.MonoLoader(
                    filename=audio_path,
                    sampleRate=self.sample_rate,
                    downmix='mix',  # Mix stereo to mono
                    resampleQuality=1  # Good quality resampling
                )
                audio = loader()
                duration = len(audio) / self.sample_rate
                log_universal('DEBUG', 'Streaming', f"Duration calculated: {duration:.1f}s ({len(audio)} samples)")
                return duration
            elif LIBROSA_AVAILABLE:
                # Use librosa for duration
                log_universal('DEBUG', 'Streaming', f"Getting duration with Librosa: {os.path.basename(audio_path)}")
                duration = librosa.get_duration(path=audio_path, sr=self.sample_rate)
                log_universal('DEBUG', 'Streaming', f"Duration calculated: {duration:.1f}s")
                return duration
            elif SOUNDFILE_AVAILABLE:
                # Use soundfile for duration
                log_universal('DEBUG', 'Streaming', f"Getting duration with SoundFile: {os.path.basename(audio_path)}")
                info = sf.info(audio_path)
                duration = info.duration
                log_universal('DEBUG', 'Streaming', f"Duration calculated: {duration:.1f}s")
                return duration
            elif WAVE_AVAILABLE and audio_path.lower().endswith('.wav'):
                # Use wave module for WAV files
                log_universal('DEBUG', 'Streaming', f"Getting duration with Wave: {os.path.basename(audio_path)}")
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    log_universal('DEBUG', 'Streaming', f"Duration calculated: {duration:.1f}s")
                    return duration
            else:
                log_universal('ERROR', 'Streaming', "No audio library available for duration detection")
                log_universal('ERROR', 'Streaming', f"  Available libraries: Essentia={ESSENTIA_AVAILABLE}, Librosa={LIBROSA_AVAILABLE}, SoundFile={SOUNDFILE_AVAILABLE}, Wave={WAVE_AVAILABLE}")
                return None
                
        except Exception as e:
            log_universal('ERROR', 'Streaming', f"Error getting duration for {audio_path}: {e}")
            return None
    
    def _get_file_size_mb(self, audio_path: str) -> float:
        """Get file size in MB."""
        try:
            size_bytes = os.path.getsize(audio_path)
            return size_bytes / (1024 ** 2)
        except Exception as e:
            log_universal('WARNING', 'Streaming', f"Could not get file size: {e}")
            return 0.0
    
    def load_audio_chunks(self, audio_path: str, 
                         chunk_duration: Optional[float] = None) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """
        Load audio file in chunks.
        
        Args:
            audio_path: Path to the audio file
            chunk_duration: Chunk duration in seconds (auto-calculated if None)
            
        Yields:
            Tuple of (audio_chunk, start_time, end_time)
        """
        if not os.path.exists(audio_path):
            log_universal('ERROR', 'Streaming', f"File not found: {audio_path}")
            return
        
        # Get file info
        file_size_mb = self._get_file_size_mb(audio_path)
        total_duration = self._get_audio_duration(audio_path)
        
        if total_duration is None:
            log_universal('ERROR', 'Streaming', f"Could not determine duration for {audio_path}")
            return
        
        # Check memory before starting
        initial_memory = self._get_current_memory_usage()
        log_universal('WARNING', 'Streaming', f"Memory usage before processing: {initial_memory['percent_used']:.1f}% ({initial_memory['used_gb']:.1f}GB / {initial_memory['total_gb']:.1f}GB)")
        
        # Calculate optimal chunk duration
        if chunk_duration is None:
            chunk_duration = self._calculate_optimal_chunk_duration(file_size_mb, total_duration)
        
        # Adjust chunk duration for memory pressure
        chunk_duration = self._adjust_for_memory_pressure(chunk_duration)
        
        log_universal('INFO', 'Streaming', f"Streaming audio: {os.path.basename(audio_path)}")
        log_universal('INFO', 'Streaming', f"  Total duration: {total_duration:.1f}s")
        log_universal('INFO', 'Streaming', f"  Chunk duration: {chunk_duration:.1f}s")
        log_universal('INFO', 'Streaming', f"  Estimated chunks: {int(total_duration / chunk_duration) + 1}")
        
        chunk_count = 0
        start_time = time.time()
        max_processing_time = 600  # 10 minutes max processing time
        
        try:
            # Prioritize true streaming with librosa if available
            if LIBROSA_AVAILABLE:
                log_universal('INFO', 'Streaming', f"Using Librosa true streaming for: {os.path.basename(audio_path)}")
                for chunk, start_time, end_time in self._load_chunks_librosa_streaming(audio_path, total_duration, chunk_duration):
                    chunk_count += 1
                    
                    # Check timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_processing_time:
                        log_universal('ERROR', 'Streaming', f"Processing timeout after {elapsed_time:.1f}s (max: {max_processing_time}s)")
                        log_universal('ERROR', 'Streaming', f"Processed {chunk_count} chunks before timeout")
                        return
                    
                    # Monitor memory every 5 chunks
                    if chunk_count % 5 == 0:
                        current_memory = self._get_current_memory_usage()
                        log_universal('WARNING', 'Streaming', f"Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                        
                        # More aggressive memory management - trigger GC at 70% instead of 90%
                        if current_memory['percent_used'] > 70:
                            log_universal('WARNING', 'Streaming', f"High memory usage detected! Forcing garbage collection...")
                            import gc
                            gc.collect()
                    
                    # Always force garbage collection after each chunk to prevent accumulation
                    import gc
                    gc.collect()
                    
                    yield chunk, start_time, end_time
            elif ESSENTIA_AVAILABLE:
                if self.use_slicer:
                    log_universal('INFO', 'Streaming', f"Using Essentia Slicer for streaming audio: {os.path.basename(audio_path)}")
                    for chunk, start_time, end_time in self._load_chunks_essentia_slicer(audio_path, total_duration, chunk_duration):
                        chunk_count += 1
                        
                        # Check timeout
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_processing_time:
                            log_universal('ERROR', 'Streaming', f"Processing timeout after {elapsed_time:.1f}s (max: {max_processing_time}s)")
                            log_universal('ERROR', 'Streaming', f"Processed {chunk_count} chunks before timeout")
                            return
                        
                        # Monitor memory every 5 chunks
                        if chunk_count % 5 == 0:
                            current_memory = self._get_current_memory_usage()
                            log_universal('WARNING', 'Streaming', f"Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                            
                            # Handle critical memory situations
                            if current_memory['percent_used'] > 85:
                                self._handle_critical_memory()
                            # More aggressive memory management - trigger GC at 70% instead of 90%
                            elif current_memory['percent_used'] > 70:
                                log_universal('WARNING', 'Streaming', f"High memory usage detected! Forcing aggressive memory cleanup...")
                                self._force_memory_cleanup()
                        
                        # Always force garbage collection after each chunk to prevent accumulation
                        import gc
                        gc.collect()
                        
                        yield chunk, start_time, end_time
                elif self.use_streaming:
                    log_universal('INFO', 'Streaming', f"Using Essentia Streaming for streaming audio: {os.path.basename(audio_path)}")
                    for chunk, start_time, end_time in self._load_chunks_essentia_streaming(audio_path, total_duration, chunk_duration):
                        chunk_count += 1
                        
                        # Check timeout
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_processing_time:
                            log_universal('ERROR', 'Streaming', f"Processing timeout after {elapsed_time:.1f}s (max: {max_processing_time}s)")
                            log_universal('ERROR', 'Streaming', f"Processed {chunk_count} chunks before timeout")
                            return
                        
                        # Monitor memory every 3 chunks for large files
                        if chunk_count % 3 == 0:
                            current_memory = self._get_current_memory_usage()
                            log_universal('WARNING', 'Streaming', f"Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                            
                            # More aggressive memory management - trigger GC at 60% for large files
                            if current_memory['percent_used'] > 60:
                                log_universal('WARNING', 'Streaming', f"High memory usage detected! Forcing aggressive cleanup...")
                                self._force_memory_cleanup()
                            elif current_memory['percent_used'] > 50:
                                log_universal('WARNING', 'Streaming', f"Moderate memory usage detected! Forcing garbage collection...")
                                import gc
                                gc.collect()
                        
                        yield chunk, start_time, end_time
                else:
                    log_universal('INFO', 'Streaming', f"Using Essentia with memory optimization for streaming audio: {os.path.basename(audio_path)}")
                    for chunk, start_time, end_time in self._load_chunks_essentia(audio_path, total_duration, chunk_duration):
                        chunk_count += 1
                        
                        # Check timeout
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_processing_time:
                            log_universal('ERROR', 'Streaming', f"Processing timeout after {elapsed_time:.1f}s (max: {max_processing_time}s)")
                            log_universal('ERROR', 'Streaming', f"Processed {chunk_count} chunks before timeout")
                            return
                        
                        # Monitor memory every 3 chunks for large files
                        if chunk_count % 3 == 0:
                            current_memory = self._get_current_memory_usage()
                            log_universal('WARNING', 'Streaming', f"Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                            
                            # More aggressive memory management - trigger GC at 60% for large files
                            if current_memory['percent_used'] > 60:
                                log_universal('WARNING', 'Streaming', f"High memory usage detected! Forcing aggressive cleanup...")
                                self._force_memory_cleanup()
                            elif current_memory['percent_used'] > 50:
                                log_universal('WARNING', 'Streaming', f"Moderate memory usage detected! Forcing garbage collection...")
                                import gc
                                gc.collect()
                                # Force a second collection to be more thorough
                                gc.collect()
                        
                        # Always force garbage collection after each chunk to prevent accumulation
                        import gc
                        gc.collect()
                        
                        yield chunk, start_time, end_time
                    
            elif LIBROSA_AVAILABLE:
                log_universal('INFO', 'Streaming', f"Using Librosa for streaming audio: {os.path.basename(audio_path)}")
                for chunk, start_time, end_time in self._load_chunks_librosa(audio_path, total_duration, chunk_duration):
                    chunk_count += 1
                    
                    # Check timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_processing_time:
                        log_universal('ERROR', 'Streaming', f"Processing timeout after {elapsed_time:.1f}s (max: {max_processing_time}s)")
                        log_universal('ERROR', 'Streaming', f"Processed {chunk_count} chunks before timeout")
                        return
                    
                    # Monitor memory every 3 chunks for large files
                    if chunk_count % 3 == 0:
                        current_memory = self._get_current_memory_usage()
                        log_universal('WARNING', 'Streaming', f"Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                        
                        # More aggressive memory management - trigger GC at 60% for large files
                        if current_memory['percent_used'] > 60:
                            log_universal('WARNING', 'Streaming', f"High memory usage detected! Forcing aggressive cleanup...")
                            self._force_memory_cleanup()
                        elif current_memory['percent_used'] > 50:
                            log_universal('WARNING', 'Streaming', f"Moderate memory usage detected! Forcing garbage collection...")
                            import gc
                            gc.collect()
                            # Force a second collection to be more thorough
                            gc.collect()
                    
                    # Always force garbage collection after each chunk to prevent accumulation
                    import gc
                    gc.collect()
                    
                    yield chunk, start_time, end_time
            else:
                # Use fallback method when no main audio library is available
                log_universal('INFO', 'Streaming', f"Using fallback method for streaming audio: {os.path.basename(audio_path)}")
                for chunk, start_time, end_time in self._load_chunks_fallback(audio_path, total_duration, chunk_duration):
                    chunk_count += 1
                    
                    # Check timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_processing_time:
                        log_universal('ERROR', 'Streaming', f"Processing timeout after {elapsed_time:.1f}s (max: {max_processing_time}s)")
                        log_universal('ERROR', 'Streaming', f"Processed {chunk_count} chunks before timeout")
                        return
                    
                    # Monitor memory every 3 chunks for large files
                    if chunk_count % 3 == 0:
                        current_memory = self._get_current_memory_usage()
                        log_universal('WARNING', 'Streaming', f"Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                        
                        # More aggressive memory management - trigger GC at 60% for large files
                        if current_memory['percent_used'] > 60:
                            log_universal('WARNING', 'Streaming', f"High memory usage detected! Forcing aggressive cleanup...")
                            self._force_memory_cleanup()
                        elif current_memory['percent_used'] > 50:
                            log_universal('WARNING', 'Streaming', f"Moderate memory usage detected! Forcing garbage collection...")
                            import gc
                            gc.collect()
                            # Force a second collection to be more thorough
                            gc.collect()
                    
                    # Always force garbage collection after each chunk to prevent accumulation
                    import gc
                    gc.collect()
                    
                    yield chunk, start_time, end_time
                
        except Exception as e:
            log_universal('ERROR', 'Streaming', f"Error streaming audio {audio_path}: {e}")
        finally:
            # Final memory check
            final_memory = self._get_current_memory_usage()
            total_time = time.time() - start_time
            log_universal('INFO', 'Streaming', f"Final memory usage: {final_memory['percent_used']:.1f}% ({final_memory['used_gb']:.1f}GB / {final_memory['total_gb']:.1f}GB)")
            log_universal('INFO', 'Streaming', f"Processed {chunk_count} chunks successfully in {total_time:.1f}s")
            
            if chunk_count == 0:
                log_universal('ERROR', 'Streaming', f"No chunks were processed for {os.path.basename(audio_path)}")
                log_universal('ERROR', 'Streaming', f"This may indicate a problem with the audio file or loading method")
                log_universal('ERROR', 'Streaming', f"File size: {file_size_mb:.1f}MB, Duration: {total_duration:.1f}s")
                log_universal('ERROR', 'Streaming', f"Chunk duration: {chunk_duration:.1f}s, Expected chunks: {int(total_duration / chunk_duration) + 1}")
                log_universal('ERROR', 'Streaming', f"Available libraries: Essentia={ESSENTIA_AVAILABLE}, Librosa={LIBROSA_AVAILABLE}, SoundFile={SOUNDFILE_AVAILABLE}, Wave={WAVE_AVAILABLE}")
    
    def _load_chunks_essentia(self, audio_path: str, total_duration: float, 
                             chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Load audio chunks using true streaming without loading entire file into memory."""
        try:
            # Calculate chunk parameters
            samples_per_chunk = int(chunk_duration * self.sample_rate)
            total_samples = int(total_duration * self.sample_rate)
            
            log_universal('INFO', 'Streaming', f"Starting true streaming for: {os.path.basename(audio_path)}")
            log_universal('INFO', 'Streaming', f"Total duration: {total_duration:.1f}s, Chunk duration: {chunk_duration:.1f}s")
            log_universal('INFO', 'Streaming', f"Expected chunks: {int(total_duration / chunk_duration) + 1}")
            log_universal('INFO', 'Streaming', f"Samples per chunk: {samples_per_chunk}")
            
            # Use librosa for true streaming if available
            if LIBROSA_AVAILABLE:
                log_universal('INFO', 'Streaming', f"Using Librosa for true streaming: {os.path.basename(audio_path)}")
                yield from self._load_chunks_librosa_streaming(audio_path, total_duration, chunk_duration)
                return
            
            # Fallback to Essentia with memory optimization
            log_universal('INFO', 'Streaming', f"Using Essentia with memory optimization: {os.path.basename(audio_path)}")
            
            # Use very small processing chunks to keep memory usage low
            processing_chunk_size = min(samples_per_chunk, int(5 * self.sample_rate))  # 5-second max processing chunk
            
            log_universal('INFO', 'Streaming', f"Starting memory-optimized chunking process...")
            log_universal('INFO', 'Streaming', f"Processing chunk size: {processing_chunk_size} samples ({processing_chunk_size/self.sample_rate:.1f}s)")
            
            # Process audio in small chunks using streaming approach
            chunk_index = 0
            current_sample = 0
            
            while current_sample < total_samples:
                # Calculate chunk boundaries
                start_sample = current_sample
                end_sample = min(start_sample + samples_per_chunk, total_samples)
                
                # Process this chunk in smaller sub-chunks to avoid memory issues
                chunk_parts = []
                sub_start = start_sample
                
                while sub_start < end_sample:
                    sub_end = min(sub_start + processing_chunk_size, end_sample)
                    
                    # Load only this small sub-chunk using streaming approach
                    try:
                        # Calculate time boundaries for this sub-chunk
                        sub_start_time = sub_start / self.sample_rate
                        sub_end_time = sub_end / self.sample_rate
                        sub_duration = sub_end_time - sub_start_time
                        
                                                # Use librosa's streaming capabilities if available
                        if LIBROSA_AVAILABLE:
                            # Load only this small chunk using librosa's offset and duration
                            # Use librosa API compatible with older versions
                            # Use multi-segment loading for consistency
                            from .audio_analyzer import extract_multiple_segments
                            sub_chunk, sr = extract_multiple_segments(
                                audio_path,
                                self.sample_rate,
                                {'OPTIMIZED_SEGMENT_DURATION_SECONDS': sub_duration},
                                'streaming'
                            )
                            if sub_chunk is not None:
                                # Extract only the needed portion from the multi-segment result
                                start_offset = int(sub_start_time * sr)
                                end_offset = start_offset + int(sub_duration * sr)
                                sub_chunk = sub_chunk[start_offset:end_offset]
                            
                            # Ensure correct sample rate
                            if sr != self.sample_rate:
                                sub_chunk = librosa.resample(sub_chunk, orig_sr=sr, target_sr=self.sample_rate)
                            
                            chunk_parts.append(sub_chunk)
                            
                        else:
                            # Fallback to Essentia with memory optimization
                            # Load the entire file but immediately extract and clear
                            loader = es.MonoLoader(
                                filename=audio_path,
                                sampleRate=self.sample_rate,
                                downmix='mix',
                                resampleQuality=1
                            )
                            full_audio = loader()
                            
                            # Extract only the sub-chunk we need
                            sub_chunk = full_audio[sub_start:sub_end]
                            
                            # Clear the full audio to free memory immediately
                            del full_audio
                            import gc
                            gc.collect()
                            
                            chunk_parts.append(sub_chunk)
                        
                    except Exception as e:
                        log_universal('ERROR', 'Streaming', f"Error loading sub-chunk {sub_start}-{sub_end}: {e}")
                        # Skip this sub-chunk and continue
                        pass
                    
                    sub_start = sub_end
                    
                    # Force garbage collection after each sub-chunk
                    import gc
                    gc.collect()
                
                # Combine sub-chunks into final chunk
                if chunk_parts:
                    chunk = np.concatenate(chunk_parts)
                    
                    # Calculate time boundaries
                    start_time = start_sample / self.sample_rate
                    end_time = end_sample / self.sample_rate
                    
                    log_universal('DEBUG', 'Streaming', f"Memory-Optimized Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk)} samples)")
                    
                    yield chunk, start_time, end_time
                    
                    # Clear chunk_parts to free memory
                    chunk_parts.clear()
                    
                    # Force garbage collection after each chunk
                    import gc
                    gc.collect()
                else:
                    log_universal('WARNING', 'Streaming', f"Empty chunk {chunk_index + 1}")
                
                current_sample = end_sample
                chunk_index += 1
            
            log_universal('INFO', 'Streaming', f"Memory-optimized processing completed: {chunk_index} chunks processed")
                
        except Exception as e:
            log_universal('ERROR', 'Streaming', f"Error in memory-optimized processing: {e}")
            # Fallback to alternative audio libraries if streaming fails
            log_universal('INFO', 'Streaming', "Falling back to alternative audio libraries...")
            yield from self._load_chunks_fallback(audio_path, total_duration, chunk_duration)
    
    def _load_chunks_essentia_slicer(self, audio_path: str, total_duration: float, 
                                    chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Load audio chunks using Essentia's MonoLoader with time-based slicing."""
        try:
            log_universal('INFO', 'Streaming', f"Starting Essentia Slicer loading for: {os.path.basename(audio_path)}")
            log_universal('INFO', 'Streaming', f"Total duration: {total_duration:.1f}s, Chunk duration: {chunk_duration:.1f}s")
            
            # Calculate number of chunks
            num_chunks = int(total_duration / chunk_duration) + 1
            
            # Load the full audio first (more reliable than streaming network)
            try:
                # Initialize MonoLoader with proper parameters
                loader = es.MonoLoader(
                    filename=audio_path,
                    sampleRate=self.sample_rate,
                    downmix='mix',  # Mix stereo to mono
                    resampleQuality=1  # Good quality resampling
                )
                audio = loader()
                sample_rate = self.sample_rate
                log_universal('INFO', 'Streaming', f"Audio loaded successfully with MonoLoader: {len(audio)} samples, {sample_rate}Hz")
                
            except Exception as e:
                log_universal('ERROR', 'Streaming', f"Failed to load audio with MonoLoader: {e}")
                log_universal('INFO', 'Streaming', "Falling back to AudioLoader...")
                
                try:
                    # Fallback to AudioLoader if MonoLoader fails
                    loader = es.AudioLoader(filename=audio_path)
                    audio, sample_rate, _, _ = loader()
                    log_universal('INFO', 'Streaming', f"Audio loaded with AudioLoader: {len(audio)} samples, {sample_rate}Hz")
                    
                except Exception as e2:
                    log_universal('ERROR', 'Streaming', f"Failed to load audio with AudioLoader: {e2}")
                    raise Exception(f"Could not load audio with any Essentia method: {e}, {e2}")
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                log_universal('INFO', 'Streaming', f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
                resampler = es.Resample(
                    inputSampleRate=sample_rate, 
                    outputSampleRate=self.sample_rate,
                    quality=1  # Good quality resampling
                )
                audio = resampler(audio)
                log_universal('INFO', 'Streaming', f"Resampling completed: {len(audio)} samples")
            
            # Process each chunk using time-based slicing
            for chunk_index in range(num_chunks):
                start_time = chunk_index * chunk_duration
                end_time = min(start_time + chunk_duration, total_duration)
                
                # Calculate sample boundaries
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                
                # Ensure we don't exceed audio length
                end_sample = min(end_sample, len(audio))
                start_sample = min(start_sample, end_sample)
                
                # Extract chunk
                chunk = audio[start_sample:end_sample]
                
                if len(chunk) > 0:  # Only yield non-empty chunks
                    log_universal('DEBUG', 'Streaming', f"Essentia Slicer Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk)} samples)")
                    yield chunk, start_time, end_time
                    
                    # Force garbage collection after each chunk
                    import gc
                    gc.collect()
                else:
                    log_universal('WARNING', 'Streaming', f"Empty chunk {chunk_index + 1}")
            
            log_universal('INFO', 'Streaming', f"Essentia Slicer processing completed: {num_chunks} chunks processed")
                
        except Exception as e:
            log_universal('ERROR', 'Streaming', f"Error in Essentia Slicer processing: {e}")
            # Fallback to alternative methods if Slicer fails
            log_universal('INFO', 'Streaming', "Falling back to alternative audio libraries...")
            yield from self._load_chunks_fallback(audio_path, total_duration, chunk_duration)
    
    def _load_chunks_essentia_streaming(self, audio_path: str, total_duration: float, 
                                       chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Load audio chunks using Essentia's proper streaming mode with VectorInput."""
        try:
            # Calculate chunk parameters
            samples_per_chunk = int(chunk_duration * self.sample_rate)
            total_samples = int(total_duration * self.sample_rate)
            
            log_universal('INFO', 'Streaming', f"Starting Essentia streaming for: {os.path.basename(audio_path)}")
            log_universal('INFO', 'Streaming', f"Total duration: {total_duration:.1f}s, Chunk duration: {chunk_duration:.1f}s")
            log_universal('INFO', 'Streaming', f"Expected chunks: {int(total_duration / chunk_duration) + 1}")
            log_universal('INFO', 'Streaming', f"Samples per chunk: {samples_per_chunk}")
            
            # Load the full audio first (for file processing)
            try:
                # Initialize MonoLoader with proper parameters
                loader = es.MonoLoader(
                    filename=audio_path,
                    sampleRate=self.sample_rate,
                    downmix='mix',  # Mix stereo to mono
                    resampleQuality=1  # Good quality resampling
                )
                audio = loader()
                sample_rate = self.sample_rate
                log_universal('INFO', 'Streaming', f"Audio loaded successfully with MonoLoader: {len(audio)} samples, {sample_rate}Hz")
                
            except Exception as e:
                log_universal('ERROR', 'Streaming', f"Failed to load audio with MonoLoader: {e}")
                log_universal('INFO', 'Streaming', "Falling back to AudioLoader...")
                
                try:
                    # Fallback to AudioLoader if MonoLoader fails
                    loader = es.AudioLoader(filename=audio_path)
                    audio, sample_rate, _, _ = loader()
                    log_universal('INFO', 'Streaming', f"Audio loaded with AudioLoader: {len(audio)} samples, {sample_rate}Hz")
                    
                except Exception as e2:
                    log_universal('ERROR', 'Streaming', f"Failed to load audio with AudioLoader: {e2}")
                    raise Exception(f"Could not load audio with any Essentia method: {e}, {e2}")
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                log_universal('INFO', 'Streaming', f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
                resampler = es.Resample(
                    inputSampleRate=sample_rate, 
                    outputSampleRate=self.sample_rate,
                    quality=1  # Good quality resampling
                )
                audio = resampler(audio)
                log_universal('INFO', 'Streaming', f"Resampling completed: {len(audio)} samples")
            
            # Process audio in chunks using streaming network
            chunk_index = 0
            current_sample = 0
            
            log_universal('INFO', 'Streaming', f"Starting streaming network processing...")
            
            while current_sample < len(audio):
                # Calculate chunk boundaries
                start_sample = current_sample
                end_sample = min(start_sample + samples_per_chunk, len(audio))
                
                # Extract chunk
                chunk = audio[start_sample:end_sample]
                
                # Create streaming network for this chunk (based on tutorials)
                buffer = chunk.astype('float32')
                
                # Initialize streaming algorithms
                vimp = es.VectorInput(buffer)
                fc = es.FrameCutter(frameSize=min(512, len(chunk)), hopSize=min(256, len(chunk)))
                
                # Create pool to collect results
                pool = essentia.Pool()
                
                # Connect the streaming network (following tutorial pattern)
                vimp.data >> fc.signal
                fc.frame >> (pool, 'frames')
                
                # Run the streaming network for this chunk
                essentia.run(vimp)
                
                # Process the collected frames
                if 'frames' in pool:
                    frames = pool['frames']
                    if len(frames) > 0:
                        # Use the first frame as the chunk (or combine frames if needed)
                        processed_chunk = frames[0] if len(frames) == 1 else np.concatenate(frames)
                        
                        # Calculate time boundaries
                        start_time = start_sample / self.sample_rate
                        end_time = end_sample / self.sample_rate
                        
                        log_universal('DEBUG', 'Streaming', f"Essentia Streaming Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s ({len(processed_chunk)} samples)")
                        
                        yield processed_chunk, start_time, end_time
                        
                        # Force garbage collection after each chunk
                        import gc
                        gc.collect()
                    else:
                        log_universal('WARNING', 'Streaming', f"No frames generated for chunk {chunk_index + 1}")
                else:
                    log_universal('WARNING', 'Streaming', f"No frames in pool for chunk {chunk_index + 1}")
                
                current_sample = end_sample
                chunk_index += 1
            
            log_universal('INFO', 'Streaming', f"Essentia streaming completed: {chunk_index} chunks processed")
                
        except Exception as e:
            log_universal('ERROR', 'Streaming', f"Error in Essentia streaming: {e}")
            # Fallback to alternative audio libraries if streaming fails
            log_universal('INFO', 'Streaming', "Falling back to alternative audio libraries...")
            yield from self._load_chunks_fallback(audio_path, total_duration, chunk_duration)
    
    def _load_chunks_fallback(self, audio_path: str, total_duration: float, 
                             chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Fallback method using alternative audio libraries."""
        try:
            # Calculate chunk parameters
            samples_per_chunk = int(chunk_duration * self.sample_rate)
            total_samples = int(total_duration * self.sample_rate)
            
            log_universal('INFO', 'Streaming', f"Using fallback audio loading for: {os.path.basename(audio_path)}")
            
            # Try different audio loading methods
            audio = None
            sample_rate = self.sample_rate
            
            if LIBROSA_AVAILABLE:
                try:
                    log_universal('INFO', 'Streaming', "Trying Librosa audio loading...")
                    # Use librosa API compatible with older versions
                    # Use multi-segment loading for consistency
                    from .audio_analyzer import extract_multiple_segments
                    audio, sample_rate = extract_multiple_segments(
                        audio_path,
                        self.sample_rate,
                        {'OPTIMIZED_SEGMENT_DURATION_SECONDS': 30},
                        'streaming'
                    )
                    log_universal('INFO', 'Streaming', f"Audio loaded with Librosa: {len(audio)} samples, {sample_rate}Hz")
                except Exception as e:
                    log_universal('WARNING', 'Streaming', f"Librosa loading failed: {e}")
            
            if audio is None and SOUNDFILE_AVAILABLE:
                try:
                    log_universal('INFO', 'Streaming', "Trying SoundFile audio loading...")
                    audio, sample_rate = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)  # Convert to mono
                    log_universal('INFO', 'Streaming', f"Audio loaded with SoundFile: {len(audio)} samples, {sample_rate}Hz")
                except Exception as e:
                    log_universal('WARNING', 'Streaming', f"SoundFile loading failed: {e}")
            
            if audio is None and WAVE_AVAILABLE and audio_path.lower().endswith('.wav'):
                try:
                    log_universal('INFO', 'Streaming', "Trying Wave audio loading...")
                    with wave.open(audio_path, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        sample_rate = wav_file.getframerate()
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        log_universal('INFO', 'Streaming', f"Audio loaded with Wave: {len(audio)} samples, {sample_rate}Hz")
                except Exception as e:
                    log_universal('WARNING', 'Streaming', f"Wave loading failed: {e}")
            
            if audio is None:
                log_universal('ERROR', 'Streaming', "No audio loading method available")
                return
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                log_universal('INFO', 'Streaming', f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
                if LIBROSA_AVAILABLE:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                else:
                    # Simple resampling (not ideal but works)
                    ratio = self.sample_rate / sample_rate
                    new_length = int(len(audio) * ratio)
                    indices = np.linspace(0, len(audio) - 1, new_length)
                    audio = np.interp(indices, np.arange(len(audio)), audio)
                log_universal('INFO', 'Streaming', f"Resampling completed: {len(audio)} samples")
            
            # Chunk the audio manually
            chunk_index = 0
            current_sample = 0
            
            log_universal('INFO', 'Streaming', f"Starting chunking process...")
            
            while current_sample < len(audio):
                # Calculate chunk boundaries
                start_sample = current_sample
                end_sample = min(start_sample + samples_per_chunk, len(audio))
                
                # Extract chunk
                chunk = audio[start_sample:end_sample]
                
                # Calculate time boundaries
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                log_universal('DEBUG', 'Streaming', f"Fallback Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk)} samples)")
                
                yield chunk, start_time, end_time
                
                current_sample = end_sample
                chunk_index += 1
                
                # Force garbage collection after each chunk to free memory
                import gc
                gc.collect()
            
            log_universal('INFO', 'Streaming', f"Fallback streaming completed: {chunk_index} chunks processed")
                
        except Exception as e:
            log_universal('ERROR', 'Streaming', f"Error in fallback streaming: {e}")
    
    def _load_chunks_librosa(self, audio_path: str, total_duration: float, 
                            chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Load audio chunks using Librosa."""
        try:
            # Calculate chunk parameters
            samples_per_chunk = int(chunk_duration * self.sample_rate)
            total_samples = int(total_duration * self.sample_rate)
            
            current_sample = 0
            chunk_index = 0
            
            while current_sample < total_samples:
                # Calculate chunk boundaries
                start_sample = current_sample
                end_sample = min(start_sample + samples_per_chunk, total_samples)
                
                # Load chunk using librosa with compatible API
                chunk, sr = librosa.load(
                    audio_path, 
                    sr=self.sample_rate, 
                    mono=True,
                    offset=start_sample / self.sample_rate,
                    duration=(end_sample - start_sample) / self.sample_rate,
                    res_type='kaiser_best'  # Use high-quality resampling
                )
                
                # Calculate time boundaries
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                log_universal('DEBUG', 'Streaming', f"Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s")
                
                yield chunk, start_time, end_time
                
                current_sample = end_sample
                chunk_index += 1
                
        except Exception as e:
            log_universal('ERROR', 'Streaming', f"Error in Librosa streaming: {e}")
    
    def _load_chunks_librosa_streaming(self, audio_path: str, total_duration: float, 
                                      chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Load audio chunks using Librosa's true streaming capabilities - only one chunk in memory at a time."""
        try:
            log_universal('INFO', 'Streaming', f"Using Librosa true streaming for: {os.path.basename(audio_path)}")
            log_universal('INFO', 'Streaming', f"Total duration: {total_duration:.1f}s, Chunk duration: {chunk_duration:.1f}s")
            log_universal('INFO', 'Streaming', f"Expected chunks: {int(total_duration / chunk_duration) + 1}")
            
            chunk_index = 0
            current_time = 0.0
            
            # Use a much smaller processing window to minimize memory usage
            # Process only 10 seconds at a time to keep memory usage very low
            processing_window = min(chunk_duration, 10.0)  # Max 10 seconds per processing window
            
            log_universal('INFO', 'Streaming', f"Using true streaming with {processing_window:.1f}s processing windows")
            
            while current_time < total_duration:
                # Calculate time boundaries for this chunk
                start_time = current_time
                end_time = min(start_time + chunk_duration, total_duration)
                duration = end_time - start_time
                
                # Process this chunk in smaller windows to minimize memory
                chunk_parts = []
                window_start = start_time
                
                while window_start < end_time:
                    window_end = min(window_start + processing_window, end_time)
                    window_duration = window_end - window_start
                    
                    try:
                        # Use soundfile for true streaming - load only the specific window
                        if SOUNDFILE_AVAILABLE:
                            # Calculate frame positions for this window
                            start_frame = int(window_start * self.sample_rate)
                            end_frame = int(window_end * self.sample_rate)
                            num_frames = end_frame - start_frame
                            
                            # Load only this specific window using soundfile
                            with sf.SoundFile(audio_path, 'r') as audio_file:
                                # Seek to the start position
                                audio_file.seek(start_frame)
                                
                                # Read only the frames we need
                                window_chunk = audio_file.read(num_frames)
                                
                                # Convert to mono if stereo
                                if len(window_chunk.shape) > 1:
                                    window_chunk = window_chunk.mean(axis=1)
                                
                                # Ensure correct sample rate
                                if audio_file.samplerate != self.sample_rate:
                                    if LIBROSA_AVAILABLE:
                                        window_chunk = librosa.resample(window_chunk, orig_sr=audio_file.samplerate, target_sr=self.sample_rate)
                                    else:
                                        # Simple resampling
                                        ratio = self.sample_rate / audio_file.samplerate
                                        new_length = int(len(window_chunk) * ratio)
                                        indices = np.linspace(0, len(window_chunk) - 1, new_length)
                                        window_chunk = np.interp(indices, np.arange(len(window_chunk)), window_chunk)
                        
                        elif WAVE_AVAILABLE and audio_path.lower().endswith('.wav'):
                            # Use wave module for WAV files
                            with wave.open(audio_path, 'rb') as wav_file:
                                # Calculate frame positions
                                start_frame = int(window_start * wav_file.getframerate())
                                end_frame = int(window_end * wav_file.getframerate())
                                num_frames = end_frame - start_frame
                                
                                # Seek to position and read frames
                                wav_file.setpos(start_frame)
                                frames = wav_file.readframes(num_frames)
                                window_chunk = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                                
                                # Resample if needed
                                if wav_file.getframerate() != self.sample_rate:
                                    if LIBROSA_AVAILABLE:
                                        window_chunk = librosa.resample(window_chunk, orig_sr=wav_file.getframerate(), target_sr=self.sample_rate)
                                    else:
                                        ratio = self.sample_rate / wav_file.getframerate()
                                        new_length = int(len(window_chunk) * ratio)
                                        indices = np.linspace(0, len(window_chunk) - 1, new_length)
                                        window_chunk = np.interp(indices, np.arange(len(window_chunk)), window_chunk)
                        
                        else:
                            # Fallback to librosa but with much smaller chunks
                            # Use only 5-second windows to minimize memory usage
                            small_window = min(processing_window, 5.0)
                            small_end = min(window_start + small_window, window_end)
                            small_duration = small_end - window_start
                            
                            # Load only this small window
                            window_chunk, sr = librosa.load(
                                audio_path,
                                sr=self.sample_rate,
                                mono=True,
                                offset=window_start,
                                duration=small_duration,
                                res_type='kaiser_best'
                            )
                            
                            # If we need more data for this window, load additional small chunks
                            while small_end < window_end:
                                window_start = small_end
                                small_end = min(window_start + small_window, window_end)
                                small_duration = small_end - window_start
                                
                                additional_chunk, sr = librosa.load(
                                    audio_path,
                                    sr=self.sample_rate,
                                    mono=True,
                                    offset=window_start,
                                    duration=small_duration,
                                    res_type='kaiser_best'
                                )
                                
                                window_chunk = np.concatenate([window_chunk, additional_chunk])
                                
                                # Force GC after each small chunk
                                import gc
                                gc.collect()
                        
                        chunk_parts.append(window_chunk)
                        
                        # Force garbage collection after each window to free memory immediately
                        import gc
                        gc.collect()
                        
                        log_universal('DEBUG', 'Streaming', f"Window {window_start:.1f}s - {window_end:.1f}s ({len(window_chunk)} samples)")
                        
                    except Exception as e:
                        log_universal('ERROR', 'Streaming', f"Error loading window {window_start:.1f}s - {window_end:.1f}s: {e}")
                        # Skip this window and continue
                        pass
                    
                    window_start = window_end
                
                # Combine window chunks into final chunk
                if chunk_parts:
                    chunk = np.concatenate(chunk_parts)
                    
                    log_universal('DEBUG', 'Streaming', f"Librosa Streaming Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk)} samples)")
                    
                    yield chunk, start_time, end_time
                    
                    # Clear chunk_parts to free memory immediately
                    chunk_parts.clear()
                    
                    # Force garbage collection after each chunk
                    import gc
                    gc.collect()
                    
                else:
                    log_universal('WARNING', 'Streaming', f"Empty chunk {chunk_index + 1}")
                
                current_time = end_time
                chunk_index += 1
                
                # Monitor memory every 3 chunks for large files
                if chunk_index % 3 == 0:
                    current_memory = self._get_current_memory_usage()
                    log_universal('WARNING', 'Streaming', f"Memory usage after chunk {chunk_index}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                    
                    # More aggressive memory management - trigger GC at 60% for large files
                    if current_memory['percent_used'] > 60:
                        log_universal('WARNING', 'Streaming', f"High memory usage detected! Forcing aggressive cleanup...")
                        self._force_memory_cleanup()
                    elif current_memory['percent_used'] > 50:
                        log_universal('WARNING', 'Streaming', f"Moderate memory usage detected! Forcing garbage collection...")
                        import gc
                        gc.collect()
            
            log_universal('INFO', 'Streaming', f"Librosa streaming completed: {chunk_index} chunks processed")
                
        except Exception as e:
            log_universal('ERROR', 'Streaming', f"Error in Librosa streaming: {e}")
            # Fallback to alternative methods if streaming fails
            log_universal('INFO', 'Streaming', "Falling back to alternative audio libraries...")
            yield from self._load_chunks_fallback(audio_path, total_duration, chunk_duration)
    
    def process_audio_streaming(self, audio_path: str, 
                              processor_func: callable,
                              chunk_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Process audio file in chunks using a custom processor function.
        
        Args:
            audio_path: Path to the audio file
            processor_func: Function to process each chunk (chunk, start_time, end_time) -> dict
            chunk_duration: Chunk duration in seconds (auto-calculated if None)
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'success': False,
            'chunks_processed': 0,
            'total_chunks': 0,
            'errors': [],
            'results': []
        }
        
        start_time = time.time()
        max_processing_time = 600  # 10 minutes max processing time
        
        try:
            # Get total duration for progress tracking
            total_duration = self._get_audio_duration(audio_path)
            if total_duration is None:
                results['errors'].append("Could not determine audio duration")
                return results
            
            # Calculate total chunks
            if chunk_duration is None:
                chunk_duration = self._calculate_optimal_chunk_duration(
                    self._get_file_size_mb(audio_path), total_duration
                )
            
            total_chunks = int(total_duration / chunk_duration) + 1
            results['total_chunks'] = total_chunks
            
            log_universal('INFO', 'Streaming', f"Processing {os.path.basename(audio_path)} in {total_chunks} chunks")
            
            # Process each chunk
            for chunk_index, (chunk, start_time, end_time) in enumerate(
                self.load_audio_chunks(audio_path, chunk_duration)
            ):
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > max_processing_time:
                    timeout_msg = f"Processing timeout after {elapsed_time:.1f}s (max: {max_processing_time}s)"
                    log_universal('ERROR', 'Streaming', timeout_msg)
                    results['errors'].append(timeout_msg)
                    results['errors'].append(f"Processed {chunk_index} chunks before timeout")
                    return results
                
                try:
                    # Process chunk
                    chunk_result = processor_func(chunk, start_time, end_time)
                    
                    # Add chunk info to result
                    chunk_result['chunk_index'] = chunk_index
                    chunk_result['start_time'] = start_time
                    chunk_result['end_time'] = end_time
                    chunk_result['duration'] = end_time - start_time
                    
                    results['results'].append(chunk_result)
                    results['chunks_processed'] += 1
                    
                    log_universal('DEBUG', 'Streaming', f"Processed chunk {chunk_index + 1}/{total_chunks}")
                    
                    # Force memory cleanup after each chunk for memory-intensive operations
                    if chunk_index % 5 == 0:  # Every 5 chunks
                        gc.collect()
                    
                except Exception as e:
                    error_msg = f"Error processing chunk {chunk_index + 1}: {e}"
                    log_universal('ERROR', 'Streaming', f"{error_msg}")
                    results['errors'].append(error_msg)
            
            # Check if processing was successful
            if results['chunks_processed'] > 0:
                results['success'] = True
                total_time = time.time() - start_time
                log_universal('INFO', 'Streaming', f"Successfully processed {results['chunks_processed']}/{total_chunks} chunks in {total_time:.1f}s")
            else:
                log_universal('ERROR', 'Streaming', "No chunks were processed successfully")
            
        except Exception as e:
            error_msg = f"Error in streaming processing: {e}"
            log_universal('ERROR', 'Streaming', f"{error_msg}")
            results['errors'].append(error_msg)
        
        return results
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024 ** 3),
                'available_gb': memory.available / (1024 ** 3),
                'used_gb': memory.used / (1024 ** 3),
                'percent_used': memory.percent,
                'memory_limit_gb': self.memory_limit_gb,
                'memory_limit_percent': self.memory_limit_percent
            }
        except Exception as e:
            log_universal('WARNING', 'Streaming', f"Could not get memory info: {e}")
            return {}


# Global streaming loader instance
_streaming_loader: Optional[StreamingAudioLoader] = None


def get_streaming_loader(memory_limit_percent: float = DEFAULT_MEMORY_LIMIT_PERCENT,
                        chunk_duration_seconds: float = DEFAULT_CHUNK_DURATION_SECONDS,
                        use_slicer: bool = False, use_streaming: bool = False) -> StreamingAudioLoader:
    """
    Get the global streaming audio loader instance.
    
    Args:
        memory_limit_percent: Maximum percentage of RAM to use
        chunk_duration_seconds: Default chunk duration in seconds
        use_slicer: If True, use Slicer algorithm instead of FrameCutter
        use_streaming: If True, use proper streaming network (based on tutorials)
        
    Returns:
        StreamingAudioLoader instance
    """
    global _streaming_loader
    
    # Check if we need to create a new instance with different parameters
    if _streaming_loader is not None:
        # If parameters are different from current instance, reset it
        if (memory_limit_percent != _streaming_loader.memory_limit_percent or
            chunk_duration_seconds != _streaming_loader.chunk_duration_seconds or
            use_slicer != _streaming_loader.use_slicer or
            use_streaming != _streaming_loader.use_streaming):
            
            log_universal('INFO', 'Streaming', f"Parameters changed, creating new StreamingAudioLoader instance:")
            log_universal('INFO', 'Streaming', f"  Memory limit: {memory_limit_percent}% (was {_streaming_loader.memory_limit_percent}%)")
            log_universal('INFO', 'Streaming', f"  Chunk duration: {chunk_duration_seconds}s (was {_streaming_loader.chunk_duration_seconds}s)")
            log_universal('INFO', 'Streaming', f"  Use Slicer: {use_slicer} (was {_streaming_loader.use_slicer})")
            log_universal('INFO', 'Streaming', f"  Use Streaming: {use_streaming} (was {_streaming_loader.use_streaming})")
            
            _streaming_loader = None  # Reset the global instance
    
    if _streaming_loader is None:
        log_universal('INFO', 'Streaming', f"Creating new StreamingAudioLoader instance:")
        log_universal('INFO', 'Streaming', f"  Memory limit: {memory_limit_percent}%")
        log_universal('INFO', 'Streaming', f"  Chunk duration: {chunk_duration_seconds}s")
        log_universal('INFO', 'Streaming', f"  Use Slicer: {use_slicer}")
        log_universal('INFO', 'Streaming', f"  Use Streaming: {use_streaming}")
        _streaming_loader = StreamingAudioLoader(
            memory_limit_percent=memory_limit_percent,
            chunk_duration_seconds=chunk_duration_seconds,
            use_slicer=use_slicer,
            use_streaming=use_streaming
        )
    else:
        log_universal('INFO', 'Streaming', f"Reusing existing StreamingAudioLoader instance")
    
    return _streaming_loader 