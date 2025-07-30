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

from .logging_setup import get_logger

logger = get_logger('playlista.streaming_loader')

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHUNK_DURATION_SECONDS = 15  # Reduced from 30 to 15 seconds per chunk
DEFAULT_MEMORY_LIMIT_PERCENT = 50  # Reduced from 80% to 50% of available RAM
MIN_CHUNK_DURATION_SECONDS = 5  # Minimum chunk duration
MAX_CHUNK_DURATION_SECONDS = 30  # Reduced from 120 to 30 seconds for memory safety


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
                 use_slicer: bool = False):
        """
        Initialize the streaming audio loader.
        
        Args:
            memory_limit_percent: Maximum percentage of RAM to use
            chunk_duration_seconds: Default chunk duration in seconds
            use_slicer: If True, use Slicer algorithm instead of FrameCutter
        """
        self.memory_limit_percent = memory_limit_percent
        self.chunk_duration_seconds = chunk_duration_seconds
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.use_slicer = use_slicer
        
        # Calculate available memory
        self.available_memory_gb = self._get_available_memory_gb()
        self.memory_limit_gb = self.available_memory_gb * (memory_limit_percent / 100)
        
        logger.info(f"🔧 StreamingAudioLoader initialized:")
        logger.info(f"   Available memory: {self.available_memory_gb:.1f}GB")
        logger.info(f"   Memory limit: {self.memory_limit_gb:.1f}GB ({memory_limit_percent}%)")
        logger.info(f"   Default chunk duration: {chunk_duration_seconds}s")
        logger.info(f"   Sample rate: {self.sample_rate}Hz")
        logger.info(f"   Use Slicer: {use_slicer}")
        logger.info(f"   Essentia available: {ESSENTIA_AVAILABLE}")
        logger.info(f"   Librosa available: {LIBROSA_AVAILABLE}")
    
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
        # Get current memory usage
        current_memory = self._get_current_memory_usage()
        available_memory_gb = current_memory.get('available_gb', 1.0)
        
        # Use a more conservative approach - use only 25% of available memory
        conservative_memory_gb = min(available_memory_gb * 0.25, self.memory_limit_gb * 0.25)
        
        # Estimate memory usage per second of audio (mono, 44.1kHz, float32)
        bytes_per_second = DEFAULT_SAMPLE_RATE * 4  # 4 bytes per float32 sample
        mb_per_second = bytes_per_second / (1024 ** 2)
        
        # Calculate how many seconds we can fit in memory
        max_seconds_in_memory = conservative_memory_gb * 1024 / mb_per_second
        
        # Use a very conservative safety factor of 0.25 to leave room for processing
        safe_seconds = max_seconds_in_memory * 0.25
        
        # Calculate optimal chunk duration
        optimal_duration = min(
            max(safe_seconds, MIN_CHUNK_DURATION_SECONDS),
            MAX_CHUNK_DURATION_SECONDS,
            duration_seconds,  # Don't exceed total duration
            30.0  # Maximum 30 seconds per chunk for memory safety
        )
        
        logger.info(f"📊 Memory-aware chunk calculation:")
        logger.info(f"   File size: {file_size_mb:.1f}MB")
        logger.info(f"   Duration: {duration_seconds:.1f}s")
        logger.info(f"   Available memory: {available_memory_gb:.1f}GB")
        logger.info(f"   Conservative memory limit: {conservative_memory_gb:.1f}GB")
        logger.info(f"   Optimal chunk duration: {optimal_duration:.1f}s")
        
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
            logger.warning(f"⚠️ Could not get memory info: {e}")
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
                logger.debug(f"🔍 Getting duration with Essentia MonoLoader: {os.path.basename(audio_path)}")
                loader = es.MonoLoader(
                    filename=audio_path,
                    sampleRate=self.sample_rate,
                    downmix='mix',  # Mix stereo to mono
                    resampleQuality=1  # Good quality resampling
                )
                audio = loader()
                duration = len(audio) / self.sample_rate
                logger.debug(f"✅ Duration calculated: {duration:.1f}s ({len(audio)} samples)")
                return duration
            elif LIBROSA_AVAILABLE:
                # Use librosa for duration
                logger.debug(f"🔍 Getting duration with Librosa: {os.path.basename(audio_path)}")
                duration = librosa.get_duration(path=audio_path, sr=self.sample_rate)
                logger.debug(f"✅ Duration calculated: {duration:.1f}s")
                return duration
            elif SOUNDFILE_AVAILABLE:
                # Use soundfile for duration
                logger.debug(f"🔍 Getting duration with SoundFile: {os.path.basename(audio_path)}")
                info = sf.info(audio_path)
                duration = info.duration
                logger.debug(f"✅ Duration calculated: {duration:.1f}s")
                return duration
            elif WAVE_AVAILABLE and audio_path.lower().endswith('.wav'):
                # Use wave module for WAV files
                logger.debug(f"🔍 Getting duration with Wave: {os.path.basename(audio_path)}")
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    duration = frames / sample_rate
                    logger.debug(f"✅ Duration calculated: {duration:.1f}s")
                    return duration
            else:
                logger.error("❌ No audio library available for duration detection")
                logger.error(f"   Available libraries: Essentia={ESSENTIA_AVAILABLE}, Librosa={LIBROSA_AVAILABLE}, SoundFile={SOUNDFILE_AVAILABLE}, Wave={WAVE_AVAILABLE}")
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
        
        # Check memory before starting
        initial_memory = self._get_current_memory_usage()
        logger.warning(f"⚠️ Memory usage before processing: {initial_memory['percent_used']:.1f}% ({initial_memory['used_gb']:.1f}GB / {initial_memory['total_gb']:.1f}GB)")
        
        # Calculate optimal chunk duration
        if chunk_duration is None:
            chunk_duration = self._calculate_optimal_chunk_duration(file_size_mb, total_duration)
        
        logger.info(f"🎵 Streaming audio: {os.path.basename(audio_path)}")
        logger.info(f"   Total duration: {total_duration:.1f}s")
        logger.info(f"   Chunk duration: {chunk_duration:.1f}s")
        logger.info(f"   Estimated chunks: {int(total_duration / chunk_duration) + 1}")
        
        chunk_count = 0
        try:
            if ESSENTIA_AVAILABLE:
                if self.use_slicer:
                    logger.info(f"🎵 Using Essentia Slicer for streaming audio: {os.path.basename(audio_path)}")
                    for chunk, start_time, end_time in self._load_chunks_essentia_slicer(audio_path, total_duration, chunk_duration):
                        chunk_count += 1
                        
                        # Monitor memory every 5 chunks
                        if chunk_count % 5 == 0:
                            current_memory = self._get_current_memory_usage()
                            logger.warning(f"⚠️ Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                            
                            # If memory usage is too high, force garbage collection
                            if current_memory['percent_used'] > 90:
                                logger.warning(f"⚠️ High memory usage detected! Forcing garbage collection...")
                                import gc
                                gc.collect()
                        
                        yield chunk, start_time, end_time
                else:
                    logger.info(f"🎵 Using Essentia FrameCutter for streaming audio: {os.path.basename(audio_path)}")
                    for chunk, start_time, end_time in self._load_chunks_essentia(audio_path, total_duration, chunk_duration):
                        chunk_count += 1
                        
                        # Monitor memory every 5 chunks
                        if chunk_count % 5 == 0:
                            current_memory = self._get_current_memory_usage()
                            logger.warning(f"⚠️ Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                            
                            # If memory usage is too high, force garbage collection
                            if current_memory['percent_used'] > 90:
                                logger.warning(f"⚠️ High memory usage detected! Forcing garbage collection...")
                                import gc
                                gc.collect()
                        
                        yield chunk, start_time, end_time
                    
            elif LIBROSA_AVAILABLE:
                logger.info(f"🎵 Using Librosa for streaming audio: {os.path.basename(audio_path)}")
                for chunk, start_time, end_time in self._load_chunks_librosa(audio_path, total_duration, chunk_duration):
                    chunk_count += 1
                    
                    # Monitor memory every 5 chunks
                    if chunk_count % 5 == 0:
                        current_memory = self._get_current_memory_usage()
                        logger.warning(f"⚠️ Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                        
                        # If memory usage is too high, force garbage collection
                        if current_memory['percent_used'] > 90:
                            logger.warning(f"⚠️ High memory usage detected! Forcing garbage collection...")
                            import gc
                            gc.collect()
                    
                    yield chunk, start_time, end_time
            else:
                # Use fallback method when no main audio library is available
                logger.info(f"🎵 Using fallback method for streaming audio: {os.path.basename(audio_path)}")
                for chunk, start_time, end_time in self._load_chunks_fallback(audio_path, total_duration, chunk_duration):
                    chunk_count += 1
                    
                    # Monitor memory every 5 chunks
                    if chunk_count % 5 == 0:
                        current_memory = self._get_current_memory_usage()
                        logger.warning(f"⚠️ Memory usage after chunk {chunk_count}: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB / {current_memory['total_gb']:.1f}GB)")
                        
                        # If memory usage is too high, force garbage collection
                        if current_memory['percent_used'] > 90:
                            logger.warning(f"⚠️ High memory usage detected! Forcing garbage collection...")
                            import gc
                            gc.collect()
                    
                    yield chunk, start_time, end_time
                
        except Exception as e:
            logger.error(f"❌ Error streaming audio {audio_path}: {e}")
        finally:
            # Final memory check
            final_memory = self._get_current_memory_usage()
            logger.info(f"📊 Final memory usage: {final_memory['percent_used']:.1f}% ({final_memory['used_gb']:.1f}GB / {final_memory['total_gb']:.1f}GB)")
            logger.info(f"📊 Processed {chunk_count} chunks successfully")
            
            if chunk_count == 0:
                logger.error(f"❌ No chunks were processed for {os.path.basename(audio_path)}")
                logger.error(f"❌ This may indicate a problem with the audio file or loading method")
                logger.error(f"❌ File size: {file_size_mb:.1f}MB, Duration: {total_duration:.1f}s")
                logger.error(f"❌ Chunk duration: {chunk_duration:.1f}s, Expected chunks: {int(total_duration / chunk_duration) + 1}")
                logger.error(f"❌ Available libraries: Essentia={ESSENTIA_AVAILABLE}, Librosa={LIBROSA_AVAILABLE}, SoundFile={SOUNDFILE_AVAILABLE}, Wave={WAVE_AVAILABLE}")
    
    def _load_chunks_essentia(self, audio_path: str, total_duration: float, 
                             chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Load audio chunks using Essentia's proper streaming mode with FrameCutter."""
        try:
            # Calculate chunk parameters
            samples_per_chunk = int(chunk_duration * self.sample_rate)
            total_samples = int(total_duration * self.sample_rate)
            
            logger.info(f"🎵 Starting Essentia streaming for: {os.path.basename(audio_path)}")
            logger.info(f"📊 Total duration: {total_duration:.1f}s, Chunk duration: {chunk_duration:.1f}s")
            logger.info(f"📊 Expected chunks: {int(total_duration / chunk_duration) + 1}")
            logger.info(f"📊 Samples per chunk: {samples_per_chunk}")
            
            # Initialize Essentia streaming network
            # Use AudioLoader for streaming input
            loader = es.AudioLoader(filename=audio_path)
            
            # Initialize FrameCutter for chunking
            # frameSize: size of each chunk in samples
            # hopSize: overlap between consecutive chunks (set to frameSize for no overlap)
            frame_cutter = es.FrameCutter(
                frameSize=samples_per_chunk,
                hopSize=samples_per_chunk,  # No overlap for clean chunks
                startFromZero=True
            )
            
            # Create a pool to collect results
            pool = essentia.Pool()
            
            # Connect the streaming network
            loader.audio >> frame_cutter.signal
            frame_cutter.frame >> pool.add('audio_frames')
            
            # Run the streaming network
            logger.info(f"🔄 Running Essentia streaming network...")
            essentia.run(loader)
            
            # Process the collected frames
            if 'audio_frames' in pool:
                frames = pool['audio_frames']
                logger.info(f"✅ Essentia streaming completed: Generated {len(frames)} frames")
                
                for chunk_index, frame in enumerate(frames):
                    # Calculate time boundaries
                    start_time = chunk_index * chunk_duration
                    end_time = start_time + chunk_duration
                    
                    logger.debug(f"📊 Essentia Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s ({len(frame)} samples)")
                    
                    yield frame, start_time, end_time
                    
                    # Force garbage collection after each chunk to free memory
                    import gc
                    gc.collect()
            else:
                logger.warning("⚠️ No frames generated by Essentia streaming")
                # Fallback to alternative methods if streaming fails
                logger.info("🔄 Falling back to alternative audio libraries...")
                yield from self._load_chunks_fallback(audio_path, total_duration, chunk_duration)
                
        except Exception as e:
            logger.error(f"❌ Error in Essentia streaming: {e}")
            # Fallback to alternative audio libraries if Essentia fails
            logger.info("🔄 Falling back to alternative audio libraries...")
            yield from self._load_chunks_fallback(audio_path, total_duration, chunk_duration)
    
    def _load_chunks_essentia_slicer(self, audio_path: str, total_duration: float, 
                                    chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Load audio chunks using Essentia's Slicer algorithm for flexible time-based chunking."""
        try:
            logger.info(f"🎵 Starting Essentia Slicer streaming for: {os.path.basename(audio_path)}")
            logger.info(f"📊 Total duration: {total_duration:.1f}s, Chunk duration: {chunk_duration:.1f}s")
            
            # Calculate number of chunks
            num_chunks = int(total_duration / chunk_duration) + 1
            
            # Initialize Essentia streaming network with Slicer
            loader = es.AudioLoader(filename=audio_path)
            
            # Create a pool to collect results
            pool = essentia.Pool()
            
            # Process each chunk using Slicer
            for chunk_index in range(num_chunks):
                start_time = chunk_index * chunk_duration
                end_time = min(start_time + chunk_duration, total_duration)
                
                # Initialize Slicer for this time segment
                slicer = es.Slicer(
                    startTime=start_time,
                    endTime=end_time
                )
                
                # Connect the streaming network for this chunk
                loader.audio >> slicer.signal
                slicer.frame >> pool.add(f'audio_chunk_{chunk_index}')
                
                # Run the streaming network for this chunk
                logger.debug(f"🔄 Processing chunk {chunk_index + 1}/{num_chunks}: {start_time:.1f}s - {end_time:.1f}s")
                essentia.run(loader)
                
                # Get the chunk from the pool
                chunk_key = f'audio_chunk_{chunk_index}'
                if chunk_key in pool:
                    frame = pool[chunk_key]
                    if len(frame) > 0:  # Only yield non-empty chunks
                        logger.debug(f"📊 Essentia Slicer Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s ({len(frame)} samples)")
                        yield frame, start_time, end_time
                        
                        # Force garbage collection after each chunk
                        import gc
                        gc.collect()
                    else:
                        logger.warning(f"⚠️ Empty chunk {chunk_index + 1}")
                else:
                    logger.warning(f"⚠️ Chunk {chunk_index + 1} not found in pool")
            
            logger.info(f"✅ Essentia Slicer streaming completed: {num_chunks} chunks processed")
                
        except Exception as e:
            logger.error(f"❌ Error in Essentia Slicer streaming: {e}")
            # Fallback to alternative methods if Slicer fails
            logger.info("🔄 Falling back to alternative audio libraries...")
            yield from self._load_chunks_fallback(audio_path, total_duration, chunk_duration)
    
    def _load_chunks_fallback(self, audio_path: str, total_duration: float, 
                             chunk_duration: float) -> Generator[Tuple[np.ndarray, float, float], None, None]:
        """Fallback method using alternative audio libraries."""
        try:
            # Calculate chunk parameters
            samples_per_chunk = int(chunk_duration * self.sample_rate)
            total_samples = int(total_duration * self.sample_rate)
            
            logger.info(f"🔄 Using fallback audio loading for: {os.path.basename(audio_path)}")
            
            # Try different audio loading methods
            audio = None
            sample_rate = self.sample_rate
            
            if LIBROSA_AVAILABLE:
                try:
                    logger.info("🔄 Trying Librosa audio loading...")
                    audio, sample_rate = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                    logger.info(f"✅ Audio loaded with Librosa: {len(audio)} samples, {sample_rate}Hz")
                except Exception as e:
                    logger.warning(f"⚠️ Librosa loading failed: {e}")
            
            if audio is None and SOUNDFILE_AVAILABLE:
                try:
                    logger.info("🔄 Trying SoundFile audio loading...")
                    audio, sample_rate = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)  # Convert to mono
                    logger.info(f"✅ Audio loaded with SoundFile: {len(audio)} samples, {sample_rate}Hz")
                except Exception as e:
                    logger.warning(f"⚠️ SoundFile loading failed: {e}")
            
            if audio is None and WAVE_AVAILABLE and audio_path.lower().endswith('.wav'):
                try:
                    logger.info("🔄 Trying Wave audio loading...")
                    with wave.open(audio_path, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        sample_rate = wav_file.getframerate()
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        logger.info(f"✅ Audio loaded with Wave: {len(audio)} samples, {sample_rate}Hz")
                except Exception as e:
                    logger.warning(f"⚠️ Wave loading failed: {e}")
            
            if audio is None:
                logger.error("❌ No audio loading method available")
                return
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                logger.info(f"🔄 Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
                if LIBROSA_AVAILABLE:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                else:
                    # Simple resampling (not ideal but works)
                    ratio = self.sample_rate / sample_rate
                    new_length = int(len(audio) * ratio)
                    indices = np.linspace(0, len(audio) - 1, new_length)
                    audio = np.interp(indices, np.arange(len(audio)), audio)
                logger.info(f"✅ Resampling completed: {len(audio)} samples")
            
            # Chunk the audio manually
            chunk_index = 0
            current_sample = 0
            
            logger.info(f"🔄 Starting chunking process...")
            
            while current_sample < len(audio):
                # Calculate chunk boundaries
                start_sample = current_sample
                end_sample = min(start_sample + samples_per_chunk, len(audio))
                
                # Extract chunk
                chunk = audio[start_sample:end_sample]
                
                # Calculate time boundaries
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                logger.debug(f"📊 Fallback Chunk {chunk_index + 1}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk)} samples)")
                
                yield chunk, start_time, end_time
                
                current_sample = end_sample
                chunk_index += 1
                
                # Force garbage collection after each chunk to free memory
                import gc
                gc.collect()
            
            logger.info(f"✅ Fallback streaming completed: {chunk_index} chunks processed")
                
        except Exception as e:
            logger.error(f"❌ Error in fallback streaming: {e}")
    
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
                        chunk_duration_seconds: float = DEFAULT_CHUNK_DURATION_SECONDS,
                        use_slicer: bool = False) -> StreamingAudioLoader:
    """
    Get the global streaming audio loader instance.
    
    Args:
        memory_limit_percent: Maximum percentage of RAM to use
        chunk_duration_seconds: Default chunk duration in seconds
        use_slicer: If True, use Slicer algorithm instead of FrameCutter
        
    Returns:
        Streaming audio loader instance
    """
    global _streaming_loader
    if _streaming_loader is None:
        logger.info(f"🔧 Creating new StreamingAudioLoader instance:")
        logger.info(f"   Memory limit: {memory_limit_percent}%")
        logger.info(f"   Chunk duration: {chunk_duration_seconds}s")
        logger.info(f"   Use Slicer: {use_slicer}")
        _streaming_loader = StreamingAudioLoader(
            memory_limit_percent=memory_limit_percent,
            chunk_duration_seconds=chunk_duration_seconds,
            use_slicer=use_slicer
        )
    else:
        logger.debug(f"🔄 Reusing existing StreamingAudioLoader instance")
    return _streaming_loader 