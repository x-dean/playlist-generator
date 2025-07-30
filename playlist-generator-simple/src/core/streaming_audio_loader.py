"""
Streaming Audio Loader for Playlist Generator Simple.
Processes audio files in chunks to reduce memory usage and handle large files.
"""

import os
import logging
import psutil
from typing import Optional, Dict, Any, Iterator, Tuple, Generator
import numpy as np

# Import audio processing libraries
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .logging_setup import get_logger

logger = get_logger('playlista.streaming_loader')

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHUNK_DURATION_SECONDS = 30  # 30 seconds per chunk
DEFAULT_MEMORY_LIMIT_PERCENT = 80  # Use up to 80% of available RAM
MIN_CHUNK_DURATION_SECONDS = 5  # Minimum chunk duration
MAX_CHUNK_DURATION_SECONDS = 120  # Maximum chunk duration


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
                 chunk_duration_seconds: float = DEFAULT_CHUNK_DURATION_SECONDS):
        """
        Initialize the streaming audio loader.
        
        Args:
            memory_limit_percent: Maximum percentage of RAM to use
            chunk_duration_seconds: Default chunk duration in seconds
        """
        self.memory_limit_percent = memory_limit_percent
        self.chunk_duration_seconds = chunk_duration_seconds
        self.sample_rate = DEFAULT_SAMPLE_RATE
        
        # Calculate available memory
        self.available_memory_gb = self._get_available_memory_gb()
        self.memory_limit_gb = self.available_memory_gb * (memory_limit_percent / 100)
        
        logger.info(f"🔧 StreamingAudioLoader initialized:")
        logger.info(f"   Available memory: {self.available_memory_gb:.1f}GB")
        logger.info(f"   Memory limit: {self.memory_limit_gb:.1f}GB ({memory_limit_percent}%)")
        logger.info(f"   Default chunk duration: {chunk_duration_seconds}s")
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024 ** 3)  # Convert to GB
        except Exception as e:
            logger.warning(f"⚠️ Could not get memory info: {e}")
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
        # Estimate memory usage per second of audio (mono, 44.1kHz, float32)
        bytes_per_second = DEFAULT_SAMPLE_RATE * 4  # 4 bytes per float32 sample
        mb_per_second = bytes_per_second / (1024 ** 2)
        
        # Calculate how many seconds we can fit in memory
        max_seconds_in_memory = self.memory_limit_gb * 1024 / mb_per_second
        
        # Use a safety factor of 0.5 to leave room for processing
        safe_seconds = max_seconds_in_memory * 0.5
        
        # Calculate optimal chunk duration
        optimal_duration = min(
            max(safe_seconds, MIN_CHUNK_DURATION_SECONDS),
            MAX_CHUNK_DURATION_SECONDS,
            duration_seconds  # Don't exceed total duration
        )
        
        logger.debug(f"📊 Chunk calculation:")
        logger.debug(f"   File size: {file_size_mb:.1f}MB")
        logger.debug(f"   Duration: {duration_seconds:.1f}s")
        logger.debug(f"   Memory limit: {self.memory_limit_gb:.1f}GB")
        logger.debug(f"   Optimal chunk duration: {optimal_duration:.1f}s")
        
        return optimal_duration
    
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
                # Use essentia for duration
                loader = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
                # Get duration without loading audio
                duration = loader.computeDuration()
                return duration
            elif LIBROSA_AVAILABLE:
                # Use librosa for duration
                duration = librosa.get_duration(path=audio_path, sr=self.sample_rate)
                return duration
            else:
                logger.error("❌ No audio library available for duration detection")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting duration for {audio_path}: {e}")
            return None
    
    def _get_file_size_mb(self, audio_path: str) -> float:
        """Get file size in MB."""
        try:
            size_bytes = os.path.getsize(audio_path)
            return size_bytes / (1024 ** 2)
        except Exception as e:
            logger.warning(f"⚠️ Could not get file size: {e}")
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
            logger.error(f"❌ File not found: {audio_path}")
            return
        
        # Get file info
        file_size_mb = self._get_file_size_mb(audio_path)
        total_duration = self._get_audio_duration(audio_path)
        
        if total_duration is None:
            logger.error(f"❌ Could not determine duration for {audio_path}")
            return
        
        # Calculate optimal chunk duration
        if chunk_duration is None:
            chunk_duration = self._calculate_optimal_chunk_duration(file_size_mb, total_duration)
        
        logger.info(f"🎵 Streaming audio: {os.path.basename(audio_path)}")
        logger.info(f"   Total duration: {total_duration:.1f}s")
        logger.info(f"   Chunk duration: {chunk_duration:.1f}s")
        logger.info(f"   Estimated chunks: {int(total_duration / chunk_duration) + 1}")
        
        try:
            if ESSENTIA_AVAILABLE:
                yield from self._load_chunks_essentia(audio_path, total_duration, chunk_duration)
            elif LIBROSA_AVAILABLE:
                yield from self._load_chunks_librosa(audio_path, total_duration, chunk_duration)
            else:
                logger.error("❌ No audio loading library available")
                return
                
        except Exception as e:
            logger.error(f"❌ Error streaming audio {audio_path}: {e}")
    
    def _load_chunks_essentia(self, audio_path: str, total_duration: float, 
                             chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Load audio chunks using Essentia."""
        try:
            # Create streaming loader
            loader = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
            
            # Calculate chunk parameters
            samples_per_chunk = int(chunk_duration * self.sample_rate)
            total_samples = int(total_duration * self.sample_rate)
            
            current_sample = 0
            chunk_index = 0
            
            while current_sample < total_samples:
                # Calculate chunk boundaries
                start_sample = current_sample
                end_sample = min(start_sample + samples_per_chunk, total_samples)
                
                # Load chunk
                chunk = loader.computeChunk(start_sample, end_sample - start_sample)
                
                # Calculate time boundaries
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                logger.debug(f"📊 Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s")
                
                yield chunk, start_time, end_time
                
                current_sample = end_sample
                chunk_index += 1
                
        except Exception as e:
            logger.error(f"❌ Error in Essentia streaming: {e}")
    
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
                
                # Load chunk using librosa
                chunk, sr = librosa.load(
                    audio_path, 
                    sr=self.sample_rate, 
                    mono=True,
                    offset=start_sample / self.sample_rate,
                    duration=(end_sample - start_sample) / self.sample_rate
                )
                
                # Calculate time boundaries
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                logger.debug(f"📊 Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s")
                
                yield chunk, start_time, end_time
                
                current_sample = end_sample
                chunk_index += 1
                
        except Exception as e:
            logger.error(f"❌ Error in Librosa streaming: {e}")
    
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
            
            logger.info(f"🔄 Processing {os.path.basename(audio_path)} in {total_chunks} chunks")
            
            # Process each chunk
            for chunk_index, (chunk, start_time, end_time) in enumerate(
                self.load_audio_chunks(audio_path, chunk_duration)
            ):
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
                    
                    logger.debug(f"✅ Processed chunk {chunk_index + 1}/{total_chunks}")
                    
                except Exception as e:
                    error_msg = f"Error processing chunk {chunk_index + 1}: {e}"
                    logger.error(f"❌ {error_msg}")
                    results['errors'].append(error_msg)
            
            # Check if processing was successful
            if results['chunks_processed'] > 0:
                results['success'] = True
                logger.info(f"✅ Successfully processed {results['chunks_processed']}/{total_chunks} chunks")
            else:
                logger.error("❌ No chunks were processed successfully")
            
        except Exception as e:
            error_msg = f"Error in streaming processing: {e}"
            logger.error(f"❌ {error_msg}")
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
            logger.warning(f"⚠️ Could not get memory info: {e}")
            return {}


# Global streaming loader instance
_streaming_loader: Optional[StreamingAudioLoader] = None


def get_streaming_loader(memory_limit_percent: float = DEFAULT_MEMORY_LIMIT_PERCENT,
                        chunk_duration_seconds: float = DEFAULT_CHUNK_DURATION_SECONDS) -> StreamingAudioLoader:
    """
    Get the global streaming audio loader instance.
    
    Args:
        memory_limit_percent: Maximum percentage of RAM to use
        chunk_duration_seconds: Default chunk duration in seconds
        
    Returns:
        Streaming audio loader instance
    """
    global _streaming_loader
    if _streaming_loader is None:
        _streaming_loader = StreamingAudioLoader(
            memory_limit_percent=memory_limit_percent,
            chunk_duration_seconds=chunk_duration_seconds
        )
    return _streaming_loader 