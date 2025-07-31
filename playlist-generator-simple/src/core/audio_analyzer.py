"""
Audio Analyzer for Playlist Generator Simple.
Extracts essential audio features for playlist generation with on/off feature control.
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from functools import wraps
import numpy as np

# Import audio processing libraries
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logging.warning("Essentia not available - limited feature extraction")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("Librosa not available - limited feature extraction")

try:
    import mutagen
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logging.warning("Mutagen not available - no metadata extraction")

# Import TensorFlow for MusiCNN
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available - MusiCNN features will be limited")

# Import SoundFile for audio loading
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.warning("SoundFile not available - limited audio loading options")

# Import wave module for WAV files
try:
    import wave
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    logging.warning("Wave module not available - WAV file support limited")



# Import local modules
from .logging_setup import get_logger, log_function_call, log_universal

logger = get_logger('playlista.audio_analyzer')

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_HOP_SIZE = 512
DEFAULT_FRAME_SIZE = 2048
DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes for large files
TIMEOUT_LARGE_FILES = 1200  # 20 minutes for very large files
TIMEOUT_EXTREMELY_LARGE = 1800  # 30 minutes for extremely large files

# File size thresholds (in samples)
LARGE_FILE_THRESHOLD = 100000000  # ~2.3 hours at 44kHz
EXTREMELY_LARGE_THRESHOLD = 200000000  # ~4.5 hours at 44kHz
EXTREMELY_LARGE_PROCESSING_THRESHOLD = 500000000  # ~11.3 hours at 44kHz


class TimeoutException(Exception):
    """Exception raised when analysis times out."""
    pass


def timeout(seconds=DEFAULT_TIMEOUT_SECONDS, error_message="Processing timed out"):
    """Decorator to add timeout to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            import platform
            import threading
            
            # Only use SIGALRM on Unix-like systems
            if platform.system() != 'Windows':
                def _handle_timeout(signum, frame):
                    raise TimeoutException(error_message)
                
                # Set signal handler
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel alarm
                    return result
                except TimeoutException:
                    signal.alarm(0)  # Cancel alarm
                    raise
                except Exception as e:
                    signal.alarm(0)  # Cancel alarm
                    raise
            else:
                # On Windows, use threading-based timeout
                result = [None]
                exception = [None]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(seconds)
                
                if thread.is_alive():
                    # Thread is still running, timeout occurred
                    raise TimeoutException(error_message)
                elif exception[0]:
                    # Exception occurred in thread
                    raise exception[0]
                else:
                    return result[0]
        
        return wrapper
    return decorator


def get_timeout_for_file_size(audio_length: int) -> int:
    """Get appropriate timeout based on file size."""
    if audio_length > EXTREMELY_LARGE_PROCESSING_THRESHOLD:
        return TIMEOUT_EXTREMELY_LARGE
    elif audio_length > EXTREMELY_LARGE_THRESHOLD:
        return TIMEOUT_LARGE_FILES
    else:
        return DEFAULT_TIMEOUT_SECONDS


def safe_json_dumps(obj):
    """Safely serialize object to JSON, handling NumPy types."""
    if obj is None:
        return None
    try:
        return json.dumps(obj)
    except TypeError as e:
        if "not JSON serializable"in str(e):
            # Convert NumPy types to Python native types
            import numpy as np
            if isinstance(obj, dict):
                converted = {}
                for k, v in obj.items():
                    if isinstance(v, np.floating):
                        converted[k] = float(v)
                    elif isinstance(v, np.integer):
                        converted[k] = int(v)
                    elif isinstance(v, np.ndarray):
                        converted[k] = v.tolist()
                    elif isinstance(v, dict):
                        converted[k] = safe_json_dumps(v)
                    else:
                        converted[k] = v
                return json.dumps(converted)
            elif isinstance(obj, list):
                converted = []
                for v in obj:
                    if isinstance(v, np.floating):
                        converted.append(float(v))
                    elif isinstance(v, np.integer):
                        converted.append(int(v))
                    elif isinstance(v, np.ndarray):
                        converted.append(v.tolist())
                    else:
                        converted.append(v)
                return json.dumps(converted)
            else:
                return json.dumps(str(obj))
        else:
            raise e


def convert_to_python_types(obj):
    """Convert NumPy types to Python native types for database storage."""
    if obj is None:
        return None
    import numpy as np
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    else:
        return obj


def ensure_float(value):
    """Ensure a value is a Python float, handling numpy types."""
    if value is None:
        return 0.0
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            else:
                return float(np.mean(value))
        elif isinstance(value, (np.floating, np.integer)):
            return float(value)
        else:
            return float(value)
    except (ValueError, TypeError):
        return 0.0





class AudioAnalyzer:
    """
    Analyzes audio files and extracts features for playlist generation.
    
    Handles:
    - Audio feature extraction with on/off control
    - Metadata extraction
    - Error handling and timeout management
    - Feature validation
    - Resource-aware feature selection
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, cache_file: str = None, library: str = None, music: str = None, config: Dict[str, Any] = None) -> None:
        """
        Initialize the audio analyzer.
        
        Args:
            cache_file: Cache file path (uses default if None)
            library: Library path (uses default if None)
            music: Music path (uses default if None)
            config: Configuration dictionary (uses global config if None)
        """
        # Load configuration
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_analysis_config()
        
        self.config = config
        
        # Set up paths
        self.cache_file = cache_file or '/app/cache/audio_analysis.db'
        self.library = library or '/root/music/library'
        self.music = music or '/music'
        
        # Streaming audio configuration
        self.streaming_enabled = config.get('STREAMING_AUDIO_ENABLED', True)
        self.streaming_memory_limit_percent = config.get('STREAMING_MEMORY_LIMIT_PERCENT', 80)
        self.streaming_chunk_duration_seconds = config.get('STREAMING_CHUNK_DURATION_SECONDS', 30)
        self.streaming_large_file_threshold_mb = config.get('STREAMING_LARGE_FILE_THRESHOLD_MB', 50)
        
        # MusiCNN configuration (simplified)
        self.musicnn_model_path = config.get('MUSICNN_MODEL_PATH', '/app/models/msd-musicnn-1.pb')
        self.musicnn_json_path = config.get('MUSICNN_JSON_PATH', '/app/models/msd-musicnn-1.json')
        
        # Initialize MusiCNN model
        self.musicnn_model = None
        self._init_musicnn()
        
        # Check library availability
        self._check_library_availability()
        
        log_universal('INFO', 'Audio', f"Initializing AudioAnalyzer v{self.VERSION}")
        # Configuration loaded successfully
        log_universal('INFO', 'Audio', f"AudioAnalyzer initialized successfully")

    def _init_musicnn(self):
        """Initialize MusiCNN model if available."""
        if not TENSORFLOW_AVAILABLE or not ESSENTIA_AVAILABLE:
            log_universal('WARNING', 'Audio', "TensorFlow or Essentia not available - MusiCNN disabled")
            return
        
        try:
            # Check if model files exist
            if os.path.exists(self.musicnn_model_path) and os.path.exists(self.musicnn_json_path):
                # Initialize MusiCNN model
                self.musicnn_model = es.TensorflowPredictMusiCNN(
                    graphFilename=self.musicnn_model_path,
                    input='model/Placeholder',
                    output='model/Sigmoid'
                )
                log_universal('INFO', 'Audio', "MusiCNN model loaded successfully")
            else:
                log_universal('WARNING', 'Audio', f"MusiCNN model files not found:")
                log_universal('WARNING', 'Audio', f"  Model: {self.musicnn_model_path}")
                log_universal('WARNING', 'Audio', f"  Config: {self.musicnn_json_path}")
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Could not initialize MusiCNN: {e}")
            self.musicnn_model = None

    def _check_library_availability(self):
        """Check availability of audio processing libraries."""
        log_universal('DEBUG', 'Audio', "Checking audio processing library availability")
        
        if ESSENTIA_AVAILABLE:
            log_universal('INFO', 'Audio', "Essentia available for feature extraction")
        else:
            log_universal('WARNING', 'Audio', "Essentia not available - limited features")
        
        if LIBROSA_AVAILABLE:
            log_universal('INFO', 'Audio', "Librosa available for feature extraction")
        else:
            log_universal('WARNING', 'Audio', "Librosa not available - limited features")
        
        if MUTAGEN_AVAILABLE:
            log_universal('INFO', 'Audio', "Mutagen available for metadata extraction")
        else:
            log_universal('WARNING', 'Audio', "Mutagen not available - no metadata")
        
        if TENSORFLOW_AVAILABLE:
            log_universal('INFO', 'Audio', "TensorFlow available for MusiCNN features")
        else:
            log_universal('WARNING', 'Audio', "TensorFlow not available - MusiCNN features disabled")
        
        if self.musicnn_model is not None:
            log_universal('INFO', 'Audio', "MusiCNN model loaded successfully")
        else:
            log_universal('WARNING', 'Audio', "MusiCNN model not available - advanced features disabled")

    @log_function_call
    def extract_features(self, audio_path: str, analysis_config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Extract features from an audio file with on/off control.
        
        Args:
            audio_path: Path to the audio file
            analysis_config: Analysis configuration with feature flags (uses defaults if None)
            
        Returns:
            Dictionary with features and metadata, or None if failed
        """
        filename = os.path.basename(audio_path)
        
        # Get analysis configuration
        if analysis_config is None:
            analysis_config = self._get_default_analysis_config()
        
        # Check resource manager for forced guidance
        forced_guidance = self._get_forced_guidance()
        if forced_guidance['force_basic_analysis']:
            log_universal('WARNING', 'Audio', f"Resource manager forcing basic analysis: {forced_guidance['reason']}")
            analysis_config = self._apply_forced_basic_config(analysis_config)
        
        analysis_type = analysis_config.get('analysis_type', 'basic')
        features_config = analysis_config.get('features_config', {})
        
        log_universal('INFO', 'Audio', f"Extracting {analysis_type} features from: {filename}")
        log_universal('DEBUG', 'Audio', f"Feature config: {features_config}")
        
        start_time = time.time()
        
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                log_universal('ERROR', 'Audio', f"File not found: {audio_path}")
                return None
            
            # Get file info
            file_size_bytes = os.path.getsize(audio_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # File size calculated
            
            # Load audio
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                log_universal('ERROR', 'Audio', f"Failed to load audio: {filename}")
                return None
            
            # Check file size and set appropriate timeout
            audio_length = len(audio)
            timeout_seconds = get_timeout_for_file_size(audio_length)
            
            # Determine file size categories for feature skipping
            is_large_file = audio_length > LARGE_FILE_THRESHOLD
            is_extremely_large = audio_length > EXTREMELY_LARGE_THRESHOLD
            is_extremely_large_for_processing = audio_length > EXTREMELY_LARGE_PROCESSING_THRESHOLD
            
            if is_extremely_large_for_processing:
                log_universal('WARNING', 'Audio', f"Extremely large file detected ({audio_length} samples), using minimal features only")
            elif is_extremely_large:
                log_universal('WARNING', 'Audio', f"Very large file detected ({audio_length} samples), skipping some features")
            elif is_large_file:
                log_universal('INFO', 'Audio', f"Large file detected ({audio_length} samples), using extended timeout")
            
            log_universal('INFO', 'Audio', f"Using timeout: {timeout_seconds} seconds")
            
            # Extract metadata (always enabled)
            metadata = self._extract_metadata(audio_path)
            
            # Extract features based on configuration with timeout
            try:
                @timeout(timeout_seconds, f"Analysis timed out for {filename}")
                def extract_with_timeout():
                    return self._extract_features_by_config(audio_path, audio, metadata, features_config)
                
                features = extract_with_timeout()
                
                if features is None:
                    log_universal('ERROR', 'Audio', f"Failed to extract features: {filename}")
                    return None
                    
            except TimeoutException as te:
                log_universal('ERROR', 'Audio', f"Analysis timed out for {filename}: {te}")
                return None
            except Exception as e:
                log_universal('ERROR', 'Audio', f"Analysis failed for {filename}: {e}")
                return None
            
            # Validate features
            if not self._validate_features(features):
                log_universal('WARNING', 'Audio', f"Features validation failed: {filename}")
                return None
            
            # Prepare result
            result = {
                'success': True,
                'features': features,
                'metadata': metadata,
                'file_info': {
                    'path': audio_path,
                    'filename': filename,
                    'size_bytes': file_size_bytes,
                    'size_mb': file_size_mb
                },
                'analysis_type': analysis_type,
                'analysis_config': analysis_config,
                'forced_guidance': forced_guidance
            }
            
            extract_time = time.time() - start_time
            log_universal('INFO', 'Audio', f"Successfully extracted {analysis_type} features from {filename} in {extract_time:.2f}s")
            
            # Log performance
            log_universal('INFO', 'Audio', f"Audio feature extraction completed in {extract_time:.2f}s")
            
            return result
            
        except TimeoutException:
            log_universal('ERROR', 'Audio', f"⏰ Analysis timed out for {filename}")
            return None
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Error extracting features from {filename}: {e}")
            return None

    def _get_default_analysis_config(self) -> Dict[str, Any]:
        """Get default analysis configuration."""
        return {
            'analysis_type': 'basic',
            'use_full_analysis': False,
            'features_config': {
                'extract_rhythm': True,
                'extract_spectral': True,
                'extract_loudness': True,
                'extract_key': True,
                'extract_mfcc': True,
                'extract_musicnn': False,
                'extract_metadata': True,
                'extract_danceability': True,
                'extract_onset_rate': True,
                'extract_zcr': True,
                'extract_spectral_contrast': True,
                'extract_chroma': True
            }
        }

    def _get_forced_guidance(self) -> Dict[str, Any]:
        """Get forced guidance from resource manager."""
        try:
            from .resource_manager import get_resource_manager
            return get_resource_manager().get_forced_analysis_guidance()
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Could not get resource manager guidance: {e}")
            return {
                'force_basic_analysis': False,
                'reason': 'Resource manager unavailable',
                'timestamp': None
            }

    def _apply_forced_basic_config(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply forced basic analysis configuration."""
        forced_config = analysis_config.copy()
        forced_config['analysis_type'] = 'basic'
        forced_config['use_full_analysis'] = False
        
        # Disable expensive features
        features_config = forced_config.get('features_config', {}).copy()
        features_config['extract_musicnn'] = False
        forced_config['features_config'] = features_config
        
        return forced_config

    def _get_audio_duration(self, audio_path: str) -> Optional[float]:
        """
        Get audio file duration in seconds.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds or None if failed
        """
        # Try Essentia first (most reliable)
        if ESSENTIA_AVAILABLE:
            try:
                loader = es.MonoLoader(filename=audio_path, sampleRate=DEFAULT_SAMPLE_RATE)
                audio = loader()
                duration = len(audio) / DEFAULT_SAMPLE_RATE
                # Duration extracted from Essentia
                return duration
            except Exception as e:
                # Essentia duration extraction failed
                pass
        
        # Try mutagen as fallback
        if MUTAGEN_AVAILABLE:
            try:
                audio = MutagenFile(audio_path)
                if audio is not None:
                    duration = audio.info.length
                    log_universal('DEBUG', 'Audio', f"Got duration from mutagen: {duration:.2f}s")
                    return duration
            except Exception as e:
                log_universal('DEBUG', 'Audio', f"Could not get duration from mutagen: {e}")
        
        # Try soundfile as last resort (avoid audioread warnings)
        if SOUNDFILE_AVAILABLE:
            try:
                info = sf.info(audio_path)
                duration = info.duration
                log_universal('DEBUG', 'Audio', f"Got duration from soundfile: {duration:.2f}s")
                return duration
            except Exception as e:
                log_universal('DEBUG', 'Audio', f"Could not get duration from soundfile: {e}")
        
        log_universal('WARNING', 'Audio', f"Could not get duration for {audio_path} using any method")
        return None

    def _safe_audio_load(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Safely load audio file using streaming loader for large files.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Audio data as numpy array, or None if failed
        """
        try:
            # Get file size to decide loading method
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            
            log_universal('DEBUG', 'Audio', f"Audio loading decision for {os.path.basename(audio_path)}:")
            log_universal('DEBUG', 'Audio', f"  File size: {file_size_mb:.1f}MB")
            log_universal('DEBUG', 'Audio', f"  Streaming enabled: {self.streaming_enabled}")
            log_universal('DEBUG', 'Audio', f"  Streaming threshold: {self.streaming_large_file_threshold_mb}MB")
            log_universal('DEBUG', 'Audio', f"  Streaming memory limit: {self.streaming_memory_limit_percent}%")
            log_universal('DEBUG', 'Audio', f"  Streaming chunk duration: {self.streaming_chunk_duration_seconds}s")
            
            # Use streaming for large files to save memory
            if file_size_mb > self.streaming_large_file_threshold_mb and self.streaming_enabled:
                log_universal('DEBUG', 'Audio', f"Using streaming loading for {file_size_mb:.1f}MB file")
                return self._load_audio_streaming(audio_path)
            else:
                log_universal('DEBUG', 'Audio', f"Using traditional loading for {file_size_mb:.1f}MB file")
                return self._load_audio_traditional(audio_path)
                
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Error loading audio {audio_path}: {e}")
            return None
    
    def _load_audio_traditional(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio using traditional method (entire file in memory)."""
        import gc
        
        try:
            # Try Essentia first (most reliable, no audioread warnings)
            if ESSENTIA_AVAILABLE:
                try:
                    log_universal('DEBUG', 'Audio', f"Trying Essentia loading for {os.path.basename(audio_path)}")
                    loader = es.MonoLoader(
                        filename=audio_path,
                        sampleRate=DEFAULT_SAMPLE_RATE,
                        downmix='mix',  # Mix stereo to mono
                        resampleQuality=1  # Good quality resampling
                    )
                    audio = loader()
                    # Limit to 60 seconds if file is too long (increased from 30s for better analysis)
                    max_samples = DEFAULT_SAMPLE_RATE * 60
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]
                        log_universal('DEBUG', 'Audio', f"Truncated audio to 60 seconds")
                    log_universal('DEBUG', 'Audio', f"Essentia loaded audio: {len(audio)} samples, {DEFAULT_SAMPLE_RATE}Hz")
                    
                    # Force garbage collection for large files
                    if len(audio) > 500000:  # ~11 seconds at 44kHz - more aggressive
                        gc.collect()
                        log_universal('DEBUG', 'Audio', "Forced garbage collection after loading audio file")
                    
                    return audio
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Essentia loading failed: {e}")
            
            # Try soundfile as fallback (no audioread warnings)
            if SOUNDFILE_AVAILABLE:
                try:
                    log_universal('DEBUG', 'Audio', f"Trying soundfile loading for {os.path.basename(audio_path)}")
                    # Use soundfile with better error handling
                    audio, sr = sf.read(audio_path, dtype='float32')
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)  # Convert to mono
                    if sr != DEFAULT_SAMPLE_RATE:
                        # Use librosa for resampling if available
                        if LIBROSA_AVAILABLE:
                            import librosa
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=DEFAULT_SAMPLE_RATE)
                        else:
                            # Simple resampling
                            ratio = DEFAULT_SAMPLE_RATE / sr
                            new_length = int(len(audio) * ratio)
                            indices = np.linspace(0, len(audio) - 1, new_length)
                            audio = np.interp(indices, np.arange(len(audio)), audio)
                    # Limit to 60 seconds
                    max_samples = DEFAULT_SAMPLE_RATE * 60
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]
                    log_universal('DEBUG', 'Audio', f"Soundfile loaded audio: {len(audio)} samples, {DEFAULT_SAMPLE_RATE}Hz")
                    return audio
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Soundfile loading failed: {e}")
                    # Try with different dtype if float32 fails
                    try:
                        audio, sr = sf.read(audio_path, dtype='int16')
                        audio = audio.astype(np.float32) / 32768.0
                        if len(audio.shape) > 1:
                            audio = audio.mean(axis=1)
                        if sr != DEFAULT_SAMPLE_RATE:
                            if LIBROSA_AVAILABLE:
                                import librosa
                                audio = librosa.resample(audio, orig_sr=sr, target_sr=DEFAULT_SAMPLE_RATE)
                            else:
                                ratio = DEFAULT_SAMPLE_RATE / sr
                                new_length = int(len(audio) * ratio)
                                indices = np.linspace(0, len(audio) - 1, new_length)
                                audio = np.interp(indices, np.arange(len(audio)), audio)
                        max_samples = DEFAULT_SAMPLE_RATE * 60
                        if len(audio) > max_samples:
                            audio = audio[:max_samples]
                        log_universal('DEBUG', 'Audio', f"Soundfile loaded audio (int16): {len(audio)} samples, {DEFAULT_SAMPLE_RATE}Hz")
                        return audio
                    except Exception as e2:
                        log_universal('WARNING', 'Audio', f"Soundfile loading with int16 also failed: {e2}")
            
            # Try wave module for WAV files
            if WAVE_AVAILABLE and audio_path.lower().endswith('.wav'):
                try:
                    log_universal('DEBUG', 'Audio', f"Trying wave loading for {os.path.basename(audio_path)}")
                    with wave.open(audio_path, 'rb') as wav_file:
                        frames = wav_file.readframes(wav_file.getnframes())
                        sr = wav_file.getframerate()
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        if sr != DEFAULT_SAMPLE_RATE:
                            # Simple resampling
                            ratio = DEFAULT_SAMPLE_RATE / sr
                            new_length = int(len(audio) * ratio)
                            indices = np.linspace(0, len(audio) - 1, new_length)
                            audio = np.interp(indices, np.arange(len(audio)), audio)
                        # Limit to 60 seconds
                        max_samples = DEFAULT_SAMPLE_RATE * 60
                        if len(audio) > max_samples:
                            audio = audio[:max_samples]
                        log_universal('DEBUG', 'Audio', f"Wave loaded audio: {len(audio)} samples, {DEFAULT_SAMPLE_RATE}Hz")
                        return audio
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Wave loading failed: {e}")
            
            # Try librosa as last resort with improved error handling
            if LIBROSA_AVAILABLE:
                try:
                    log_universal('DEBUG', 'Audio', f"Trying librosa loading for {os.path.basename(audio_path)}")
                    # Use librosa for loading with newer API to avoid deprecation warnings
                    import librosa
                    # Use librosa.load with explicit backend specification
                    audio, sr = librosa.load(
                        audio_path, 
                        sr=DEFAULT_SAMPLE_RATE, 
                        mono=True, 
                        duration=30.0,
                        res_type='kaiser_best'  # Use high-quality resampling
                    )
                    log_universal('DEBUG', 'Audio', f"Librosa loaded audio: {len(audio)} samples, {sr}Hz")
                    
                    # Force garbage collection for large files
                    if len(audio) > 500000:  # ~11 seconds at 44kHz - more aggressive
                        gc.collect()
                        log_universal('DEBUG', 'Audio', "Forced garbage collection after loading audio file")
                    
                    return audio
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Librosa loading failed: {e}")
                    # Try with different resampling type if kaiser_best fails
                    try:
                        audio, sr = librosa.load(
                            audio_path, 
                            sr=DEFAULT_SAMPLE_RATE, 
                            mono=True, 
                            duration=30.0,
                            res_type='linear'  # Fallback to linear resampling
                        )
                        log_universal('DEBUG', 'Audio', f"Librosa loaded audio (linear): {len(audio)} samples, {sr}Hz")
                        return audio
                    except Exception as e2:
                        log_universal('WARNING', 'Audio', f"Librosa loading with linear resampling also failed: {e2}")
            
            # If all methods failed, create a dummy audio
            log_universal('ERROR', 'Audio', "All audio loading methods failed")
            log_universal('ERROR', 'Audio', f"  Available libraries: Essentia={ESSENTIA_AVAILABLE}, Librosa={LIBROSA_AVAILABLE}, SoundFile={SOUNDFILE_AVAILABLE}, Wave={WAVE_AVAILABLE}")
            
            # Create a dummy audio signal (1 second of silence)
            log_universal('WARNING', 'Audio', "Creating dummy audio signal for testing")
            dummy_audio = np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32)
            return dummy_audio
                
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Error loading audio {audio_path}: {e}")
            # Create a dummy audio signal as fallback
            log_universal('WARNING', 'Audio', "Creating dummy audio signal as fallback")
            dummy_audio = np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32)
            return dummy_audio
    
    def _load_audio_streaming(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio using streaming method for memory efficiency."""
        try:
            from .streaming_audio_loader import get_streaming_loader
            
            # Get streaming loader with configuration
            streaming_loader = get_streaming_loader(
                memory_limit_percent=self.streaming_memory_limit_percent,
                chunk_duration_seconds=self.streaming_chunk_duration_seconds,
                use_slicer=False,  # Use FrameCutter for better memory management
                use_streaming=True  # Enable true streaming for large files
            )
            
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            
            # For very large files, use sampling approach instead of full loading
            if file_size_mb > 50:  # Large file threshold
                log_universal('INFO', 'Audio', f"Large file detected ({file_size_mb:.1f}MB) - using sampling approach")
                return self._load_audio_sampling(audio_path)
            
            # For smaller files, collect chunks and concatenate
            chunks = []
            total_samples = 0
            
            for chunk, start_time, end_time in streaming_loader.load_audio_chunks(audio_path):
                chunks.append(chunk)
                total_samples += len(chunk)
                log_universal('DEBUG', 'Audio', f"Loaded chunk: {start_time:.1f}s - {end_time:.1f}s ({len(chunk)} samples)")
                
                # Limit total samples to prevent memory issues
                if total_samples > DEFAULT_SAMPLE_RATE * 60:  # 60 seconds max
                    log_universal('WARNING', 'Audio', "Reached 60-second limit, truncating audio")
                    break
            
            if not chunks:
                log_universal('ERROR', 'Audio', "No chunks loaded from streaming loader")
                log_universal('WARNING', 'Audio', "Falling back to traditional loading...")
                return self._load_audio_traditional(audio_path)
            
            # Concatenate all chunks
            audio = np.concatenate(chunks)
            log_universal('DEBUG', 'Audio', f"Concatenated audio: {len(audio)} samples total")
            
            return audio
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Error in streaming audio load: {e}")
            log_universal('WARNING', 'Audio', "Falling back to traditional loading...")
            return self._load_audio_traditional(audio_path)
    
    def _load_audio_sampling(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio using sampling approach for very large files."""
        try:
            # Get file duration
            duration = self._get_audio_duration(audio_path)
            if duration is None:
                log_universal('ERROR', 'Audio', "Could not determine audio duration for sampling")
                return None
            
            # Sample multiple segments throughout the file
            segment_duration = 30.0  # 30 seconds per segment
            num_segments = min(5, int(duration / segment_duration))  # Max 5 segments
            segment_interval = duration / (num_segments + 1)  # Evenly spaced segments
            
            log_universal('INFO', 'Audio', f"Sampling {num_segments} segments of {segment_duration}s each from {duration:.1f}s file")
            
            segments = []
            
            # Sample segments throughout the file
            for i in range(num_segments):
                segment_start = (i + 1) * segment_interval
                
                # Ensure we don't exceed file duration
                if segment_start + segment_duration > duration:
                    segment_start = max(0, duration - segment_duration)
                
                try:
                    log_universal('DEBUG', 'Audio', f"Loading segment {i+1}/{num_segments}: {segment_start:.1f}s - {segment_start + segment_duration:.1f}s")
                    
                    # Use librosa for segment loading
                    if LIBROSA_AVAILABLE:
                        import librosa
                        chunk, sr = librosa.load(
                            audio_path,
                            sr=DEFAULT_SAMPLE_RATE,
                            mono=True,
                            offset=segment_start,
                            duration=segment_duration,
                            res_type='kaiser_best'  # Use high-quality resampling
                        )
                        
                        segments.append(chunk)
                        log_universal('DEBUG', 'Audio', f"Loaded segment {i+1}: {len(chunk)} samples")
                        
                        # Force garbage collection after each segment
                        import gc
                        gc.collect()
                        
                    else:
                        log_universal('WARNING', 'Audio', "Librosa not available for segment loading")
                        break
                        
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Error loading segment {i+1}: {e}")
                    continue
            
            if not segments:
                log_universal('ERROR', 'Audio', "No segments loaded successfully")
                return None
            
            # Concatenate segments
            audio = np.concatenate(segments)
            log_universal('INFO', 'Audio', f"Sampled audio: {len(audio)} samples total from {len(segments)} segments")
            
            return audio
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Error in audio sampling: {e}")
            return None
    
    def _analyze_large_file_streaming(self, audio_path: str, streaming_loader) -> Optional[np.ndarray]:
        """Analyze large files using true streaming - process multiple segments for mixed tracks."""
        try:
            log_universal('INFO', 'Audio', f"Starting true streaming analysis for large file: {os.path.basename(audio_path)}")
            
            # Get file duration
            duration = self._get_audio_duration(audio_path)
            if duration is None:
                log_universal('ERROR', 'Audio', "Could not determine audio duration")
                return None
            
            # For very large files (> 100MB), use multiple segments throughout the file
            # This is better for mixed tracks that have different characteristics throughout
            
            # Calculate segment parameters
            segment_duration = 120.0  # 2 minutes per segment
            num_segments = min(10, int(duration / segment_duration))  # Max 10 segments
            segment_interval = duration / (num_segments + 1)  # Evenly spaced segments
            
            log_universal('INFO', 'Audio', f"Large file analysis: {duration:.1f}s total, {num_segments} segments of {segment_duration}s each")
            
            segments = []
            
            # Sample segments throughout the file
            for i in range(num_segments):
                segment_start = (i + 1) * segment_interval
                
                # Ensure we don't exceed file duration
                if segment_start + segment_duration > duration:
                    segment_start = max(0, duration - segment_duration)
                
                try:
                    log_universal('DEBUG', 'Audio', f"Loading segment {i+1}/{num_segments}: {segment_start:.1f}s - {segment_start + segment_duration:.1f}s")
                    
                    # Use librosa for segment loading
                    chunk, sr = librosa.load(
                        audio_path,
                        sr=DEFAULT_SAMPLE_RATE,
                        mono=True,
                        offset=segment_start,
                        duration=segment_duration,
                        res_type='kaiser_best'  # Use high-quality resampling
                    )
                    
                    segments.append(chunk)
                    log_universal('DEBUG', 'Audio', f"Loaded segment {i+1}: {len(chunk)} samples")
                    
                    # Force garbage collection after each segment
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Error loading segment {i+1}: {e}")
                    continue
            
            if not segments:
                log_universal('ERROR', 'Audio', "No segments loaded from large file")
                return None
            
            # Concatenate segments to create representative audio
            representative_audio = np.concatenate(segments)
            total_duration = len(representative_audio) / DEFAULT_SAMPLE_RATE
            
            log_universal('INFO', 'Audio', f"Created representative audio: {len(representative_audio)} samples ({total_duration:.1f}s) from {len(segments)} segments")
            log_universal('INFO', 'Audio', f"Representative audio covers {total_duration/duration*100:.1f}% of original file")
            
            return representative_audio
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Error in true streaming analysis: {e}")
            return None

    @timeout(30, "Metadata extraction timed out")  # 30 seconds for metadata
    def _extract_metadata(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract metadata from audio file and enrich with external APIs.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        if not MUTAGEN_AVAILABLE:
            log_universal('WARNING', 'Audio', "Mutagen not available - skipping metadata extraction")
            return metadata
        
        try:
            log_universal('DEBUG', 'Audio', f"Starting metadata extraction for: {os.path.basename(audio_path)}")
            audio_file = MutagenFile(audio_path)
            
            if audio_file is not None:
                # Log all available tags for debugging
                all_tags = list(audio_file.keys())
                log_universal('DEBUG', 'Audio', f"Available tags in file: {all_tags}")
                
                # Extract common metadata fields
                extracted_tags = []
                
                # ID3 tag mapping - expanded with more useful tags
                tag_mapping = {
                    # Basic metadata
                    'title': ['TIT2', 'title'],
                    'artist': ['TPE1', 'artist'],
                    'album': ['TALB', 'album'],
                    'genre': ['TCON', 'genre'],
                    'year': ['TDRC', 'year'],
                    'tracknumber': ['TRCK', 'tracknumber'],
                    
                    # Additional metadata
                    'composer': ['TCOM', 'composer'],
                    'lyricist': ['TEXT', 'lyricist'],
                    'band': ['TPE2', 'band'],
                    'conductor': ['TPE3', 'conductor'],
                    'remixer': ['TPE4', 'remixer'],
                    'subtitle': ['TIT3', 'subtitle'],
                    'grouping': ['TIT1', 'grouping'],
                    'publisher': ['TPUB', 'publisher'],
                    'copyright': ['TCOP', 'copyright'],
                    'encoded_by': ['TENC', 'encoded_by'],
                    
                    # Music-specific metadata
                    'key': ['TKEY', 'key'],
                    'bpm': ['TBPM', 'bpm'],
                    'tempo': ['TBPM', 'tempo'],
                    'length': ['TLEN', 'length'],
                    'language': ['TLAN', 'language'],
                    'mood': ['TMOO', 'mood'],
                    'style': ['TSTYLE', 'style'],
                    'quality': ['TQUAL', 'quality'],
                    
                    # Extended metadata
                    'original_artist': ['TOPE', 'original_artist'],
                    'original_album': ['TOAL', 'original_album'],
                    'original_year': ['TORY', 'original_year'],
                    'original_filename': ['TOFN', 'original_filename'],
                    'content_group': ['TIT1', 'content_group'],
                    'encoder': ['TSSE', 'encoder'],
                    'file_type': ['TFLT', 'file_type'],
                    'playlist_delay': ['TDLY', 'playlist_delay'],
                    'recording_time': ['TDTG', 'recording_time'],
                    
                    # User-defined tags (TXXX frames)
                    'musicbrainz_artist_id': ['TXXX:MusicBrainz Artist Id', 'musicbrainz_artist_id'],
                    'musicbrainz_album_id': ['TXXX:MusicBrainz Album Id', 'musicbrainz_album_id'],
                    'musicbrainz_track_id': ['TXXX:MusicBrainz Track Id', 'musicbrainz_track_id'],
                    'replaygain_track_gain': ['TXXX:ReplayGain_Track_Gain', 'replaygain_track_gain'],
                    'replaygain_album_gain': ['TXXX:ReplayGain_Album_Gain', 'replaygain_album_gain'],
                    'replaygain_track_peak': ['TXXX:ReplayGain_Track_Peak', 'replaygain_track_peak'],
                    'replaygain_album_peak': ['TXXX:ReplayGain_Album_Peak', 'replaygain_album_peak']
                }
                
                log_universal('DEBUG', 'Audio', f"Looking for ID3 tags: {list(tag_mapping.keys())}")
                
                for human_tag, id3_tags in tag_mapping.items():
                    for id3_tag in id3_tags:
                        if id3_tag in audio_file:
                            value = audio_file[id3_tag]
                            if isinstance(value, list) and len(value) > 0:
                                metadata[human_tag] = str(value[0])
                            else:
                                metadata[human_tag] = str(value)
                            extracted_tags.append(f"{human_tag}: {metadata[human_tag]}")
                            break  # Found the tag, move to next human tag
                
                if extracted_tags:
                    log_universal('INFO', 'Audio', f"Mutagen extracted metadata: {extracted_tags}")
                else:
                    log_universal('INFO', 'Audio', "No metadata tags found in audio file")
                
                log_universal('DEBUG', 'Audio', f"Extracted basic metadata keys: {list(metadata.keys())}")
                
                # Enrich metadata with external APIs
                enriched_metadata = self._enrich_metadata_with_external_apis(metadata)
                
                # Log final metadata summary
                if enriched_metadata:
                    log_universal('INFO', 'Audio', f"Final metadata summary: {list(enriched_metadata.keys())}")
                    for key, value in enriched_metadata.items():
                        log_universal('DEBUG', 'Audio', f"  {key}: {value}")
                else:
                    log_universal('INFO', 'Audio', "No metadata available after extraction and enrichment")
                
                return enriched_metadata
            else:
                log_universal('WARNING', 'Audio', "MutagenFile returned None - no metadata available")
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error extracting metadata from {audio_path}: {e}")
        
        return metadata
    
    @timeout(60, "External API enrichment timed out")  # 1 minute for external APIs
    def _enrich_metadata_with_external_apis(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich metadata using external APIs (MusicBrainz and Last.fm).
        
        Args:
            metadata: Basic metadata from audio file
            
        Returns:
            Enriched metadata dictionary
        """
        try:
            log_universal('DEBUG', 'Audio', "Starting external API metadata enrichment")
            
            # Import external APIs module
            from .external_apis import get_metadata_enrichment_service
            
            # Get metadata enrichment service
            metadata_enrichment_service = get_metadata_enrichment_service()
            
            # Check if external APIs are available
            if not metadata_enrichment_service.is_available():
                log_universal('INFO', 'Audio', "External APIs not available - skipping enrichment")
                return metadata
            
            log_universal('INFO', 'Audio', "External APIs available - attempting enrichment")
            
            # Log input metadata for enrichment
            if metadata:
                log_universal('DEBUG', 'Audio', f"Input metadata for enrichment: {list(metadata.keys())}")
                for key, value in metadata.items():
                    log_universal('DEBUG', 'Audio', f"  {key}: {value}")
            else:
                log_universal('DEBUG', 'Audio', "No input metadata for enrichment")
            
            # Enrich metadata
            enriched_metadata = metadata_enrichment_service.enrich_metadata(metadata)
            
            # Log enrichment results
            if enriched_metadata != metadata:
                new_fields = set(enriched_metadata.keys()) - set(metadata.keys())
                if new_fields:
                    log_universal('INFO', 'Audio', f"External APIs enriched metadata with: {list(new_fields)}")
                    for field in new_fields:
                        if field in enriched_metadata:
                            log_universal('DEBUG', 'Audio', f"  {field}: {enriched_metadata[field]}")
                else:
                    log_universal('DEBUG', 'Audio', "No new fields added by external APIs")
            else:
                log_universal('INFO', 'Audio', "No enrichment from external APIs - metadata unchanged")
            
            return enriched_metadata
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error enriching metadata with external APIs: {e}")
            return metadata

    def _extract_features_by_config(self, audio_path: str, audio: np.ndarray, 
                                  metadata: Dict[str, Any], features_config: Dict[str, bool]) -> Optional[Dict[str, Any]]:
        """
        Extract features based on configuration flags.
        
        Args:
            audio_path: Path to the audio file
            audio: Audio data as numpy array
            metadata: Metadata dictionary
            features_config: Dictionary with feature extraction flags
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        extracted_features = []
        failed_features = []
        
        # Check file size for feature skipping
        audio_length = len(audio)
        
        # Get original file size for proper threshold checking
        original_file_size = None
        try:
            original_duration = self._get_audio_duration(audio_path)
            if original_duration:
                original_file_size = int(original_duration * DEFAULT_SAMPLE_RATE)
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Could not determine original file size: {e}")
        
        # Use original file size if available, otherwise use audio length
        size_for_threshold = original_file_size if original_file_size else audio_length
        
        is_large_file = size_for_threshold > LARGE_FILE_THRESHOLD
        is_extremely_large = size_for_threshold > EXTREMELY_LARGE_THRESHOLD
        is_extremely_large_for_processing = size_for_threshold > EXTREMELY_LARGE_PROCESSING_THRESHOLD
        
        # Log the threshold decision
        if original_file_size and original_file_size != audio_length:
            log_universal('INFO', 'Audio', f"File size thresholds: original={original_file_size:,} samples, representative={audio_length:,} samples")
            log_universal('INFO', 'Audio', f"Using original file size for threshold decisions")
        else:
            log_universal('DEBUG', 'Audio', f"File size: {audio_length:,} samples")
        
        # Log feature skipping decisions
        if is_extremely_large_for_processing:
            log_universal('WARNING', 'Audio', f"Extremely large file detected - analyzing representative audio only")
        elif is_extremely_large:
            log_universal('WARNING', 'Audio', f"Very large file detected - skipping MFCC and MusiCNN features")
        elif is_large_file:
            log_universal('INFO', 'Audio', f"Large file detected - using extended timeouts")
        
        try:
            # Extract rhythm features (if enabled)
            if features_config.get('extract_rhythm', True):
                start_time = time.time()
                try:
                    rhythm_features = self._extract_rhythm_features(audio, audio_path, metadata)
                    if rhythm_features:
                        features.update(rhythm_features)
                        extracted_features.append('rhythm')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: rhythm completed")
                    else:
                        failed_features.append('rhythm')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: rhythm failed")
                except Exception as e:
                    failed_features.append('rhythm')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: rhythm error: {e}")
            
            # Extract spectral features (if enabled)
            if features_config.get('extract_spectral', True):
                start_time = time.time()
                try:
                    spectral_features = self._extract_spectral_features(audio)
                    if spectral_features:
                        features.update(spectral_features)
                        extracted_features.append('spectral')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: spectral completed")
                    else:
                        failed_features.append('spectral')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: spectral failed")
                except Exception as e:
                    failed_features.append('spectral')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: spectral error: {e}")
            
            # Extract loudness (if enabled)
            if features_config.get('extract_loudness', True):
                start_time = time.time()
                try:
                    loudness_features = self._extract_loudness(audio)
                    if loudness_features:
                        features.update(loudness_features)
                        extracted_features.append('loudness')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: loudness completed")
                    else:
                        failed_features.append('loudness')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: loudness failed")
                except Exception as e:
                    failed_features.append('loudness')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: loudness error: {e}")
            
            # Extract key (if enabled) - disabled for extremely large files
            if features_config.get('extract_key', True):
                if is_extremely_large:
                    log_universal('WARNING', 'Audio', f"Skipping key extraction for extremely large file")
                    failed_features.append('key')
                else:
                    start_time = time.time()
                    try:
                        key_features = self._extract_key(audio)
                        if key_features:
                            features.update(key_features)
                            extracted_features.append('key')
                            duration = time.time() - start_time
                            log_universal('INFO', 'Audio', f"Feature extraction step: key completed")
                        else:
                            failed_features.append('key')
                            duration = time.time() - start_time
                            log_universal('ERROR', 'Audio', f"Feature extraction step: key failed")
                    except Exception as e:
                        failed_features.append('key')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: key error: {e}")
            
            # Extract MFCC (if enabled)
            if features_config.get('extract_mfcc', True) and not is_extremely_large:
                start_time = time.time()
                try:
                    mfcc_features = self._extract_mfcc(audio)
                    if mfcc_features:
                        features.update(mfcc_features)
                        extracted_features.append('mfcc')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: mfcc completed")
                    else:
                        failed_features.append('mfcc')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: mfcc failed")
                except Exception as e:
                    failed_features.append('mfcc')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: mfcc error: {e}")
            
            # Extract MusiCNN features (if enabled)
            if features_config.get('extract_musicnn', False) and not is_extremely_large:
                start_time = time.time()
                try:
                    musicnn_features = self._extract_musicnn_features(audio)
                    if musicnn_features:
                        features.update(musicnn_features)
                        extracted_features.append('musicnn')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: musicnn completed")
                    else:
                        failed_features.append('musicnn')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: musicnn failed")
                except Exception as e:
                    failed_features.append('musicnn')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: musicnn error: {e}")
            
            # Extract Danceability (if enabled)
            if features_config.get('extract_danceability', True):
                start_time = time.time()
                try:
                    danceability_features = self._extract_danceability(audio)
                    if danceability_features:
                        features.update(danceability_features)
                        extracted_features.append('danceability')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: danceability completed")
                    else:
                        failed_features.append('danceability')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: danceability failed")
                except Exception as e:
                    failed_features.append('danceability')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: danceability error: {e}")
            
            # Extract Onset Rate (if enabled)
            if features_config.get('extract_onset_rate', True):
                start_time = time.time()
                try:
                    onset_rate_features = self._extract_onset_rate(audio)
                    if onset_rate_features:
                        features.update(onset_rate_features)
                        extracted_features.append('onset_rate')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: onset_rate completed")
                    else:
                        failed_features.append('onset_rate')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: onset_rate failed")
                except Exception as e:
                    failed_features.append('onset_rate')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: onset_rate error: {e}")
            
            # Extract Zero Crossing Rate (if enabled)
            if features_config.get('extract_zcr', True):
                start_time = time.time()
                try:
                    zcr_features = self._extract_zcr(audio)
                    if zcr_features:
                        features.update(zcr_features)
                        extracted_features.append('zcr')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: zcr completed")
                    else:
                        failed_features.append('zcr')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: zcr failed")
                except Exception as e:
                    failed_features.append('zcr')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: zcr error: {e}")
            
            # Extract Spectral Contrast (if enabled)
            if features_config.get('extract_spectral_contrast', True):
                start_time = time.time()
                try:
                    spectral_contrast_features = self._extract_spectral_contrast(audio)
                    if spectral_contrast_features:
                        features.update(spectral_contrast_features)
                        extracted_features.append('spectral_contrast')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: spectral_contrast completed")
                    else:
                        failed_features.append('spectral_contrast')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: spectral_contrast failed")
                except Exception as e:
                    failed_features.append('spectral_contrast')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: spectral_contrast error: {e}")
            
            # Extract Chroma (if enabled)
            if features_config.get('extract_chroma', True):
                start_time = time.time()
                try:
                    chroma_features = self._extract_chroma(audio)
                    if chroma_features:
                        features.update(chroma_features)
                        extracted_features.append('chroma')
                        duration = time.time() - start_time
                        log_universal('INFO', 'Audio', f"Feature extraction step: chroma completed")
                    else:
                        failed_features.append('chroma')
                        duration = time.time() - start_time
                        log_universal('ERROR', 'Audio', f"Feature extraction step: chroma failed")
                except Exception as e:
                    failed_features.append('chroma')
                    duration = time.time() - start_time
                    log_universal('ERROR', 'Audio', f"Feature extraction step: chroma error: {e}")
            
            # Calculate duration (always included)
            try:
                duration = audio_length / DEFAULT_SAMPLE_RATE
                features['duration'] = ensure_float(duration)
                log_universal('INFO', 'Audio', f"Duration: {features['duration']:.2f}s")
            except Exception as e:
                log_universal('WARNING', 'Audio', f"Duration calculation failed: {str(e)}")
                features['duration'] = 0.0
            
            # Add fallback values for skipped features due to file size
            if is_extremely_large:
                log_universal('WARNING', 'Audio', "Skipping MFCC and MusiCNN for very large file")
                features.setdefault('mfcc', [-999.0] * 13)  # Invalid MFCC values
                features.setdefault('musicnn_features', [-999.0] * 50)  # Invalid MusiCNN features
            
            # Add metadata (always included)
            features['metadata'] = metadata
            
            # Log overall feature extraction summary
            log_universal('DEBUG', 'Audio', f"Feature extraction summary: {len(extracted_features)} successful, {len(failed_features)} failed")
            log_universal('DEBUG', 'Audio', f"Extracted features: {extracted_features}")
            if failed_features:
                log_universal('DEBUG', 'Audio', f"Failed features: {failed_features}")
            
            return features
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Error extracting features: {e}")
            return None

    @timeout(300, "Rhythm feature extraction timed out")  # 5 minutes for rhythm analysis
    def _extract_rhythm_features(self, audio: np.ndarray, audio_path: str = None,
                                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract rhythm-related features.
        
        Args:
            audio: Audio data
            audio_path: Path to audio file (for external BPM lookup)
            metadata: Metadata dictionary
            
        Returns:
            Dictionary with rhythm features
        """
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Use essentia for rhythm features
                rhythm_extractor = es.RhythmExtractor2013()
                rhythm_result = rhythm_extractor(audio)
                
                # Handle different return types from Essentia (like old working version)
                if isinstance(rhythm_result, tuple):
                    # Try to get BPM from the first element
                    if len(rhythm_result) > 0:
                        bpm = rhythm_result[0]
                        log_universal('DEBUG', 'Audio', f"Extracted BPM from tuple[0]: {bpm}")
                    else:
                        log_universal('WARNING', 'Audio', "Empty rhythm result tuple")
                        bpm = -1.0  # Special marker for failed BPM extraction
                else:
                    # Single value return
                    bpm = rhythm_result
                    log_universal('DEBUG', 'Audio', f"Extracted BPM from single value: {bpm}")

                # Ensure BPM is a valid number (like old working version)
                try:
                    bpm = ensure_float(bpm)
                    if not np.isfinite(bpm) or bpm <= 0:
                        log_universal('WARNING', 'Audio', f"Invalid BPM value: {bpm}, using failed marker")
                        bpm = -999.0  # Invalid BPM marker (normal range: 30-300)
                    else:
                        # Round BPM to reasonable precision
                        bpm = round(bpm, 1)
                except (ValueError, TypeError):
                    log_universal('WARNING', 'Audio', f"Could not convert BPM to float: {bpm}, using failed marker")
                    bpm = -999.0  # Invalid BPM marker (normal range: 30-300)

                features['bpm'] = bpm
                
                # Add additional rhythm features if available
                if isinstance(rhythm_result, tuple) and len(rhythm_result) >= 4:
                    # Handle confidence value - normalize to 0-1 range if needed
                    confidence_raw = rhythm_result[1]
                    if isinstance(confidence_raw, (int, float, np.number)):
                        confidence = float(confidence_raw)
                        # Normalize confidence to 0-1 range if it's outside
                        if confidence > 1.0:
                            confidence = confidence / 1000.0  # Assume it's in 0-1000 range
                        elif confidence < 0:
                            confidence = 0.0
                        features['confidence'] = ensure_float(confidence)
                    else:
                        features['confidence'] = 0.5  # Default confidence
                    
                    if hasattr(rhythm_result[2], 'tolist'):
                        features['estimates'] = rhythm_result[2].tolist()
                    else:
                        features['estimates'] = convert_to_python_types(rhythm_result[2])
                    if hasattr(rhythm_result[3], 'tolist'):
                        features['bpm_intervals'] = rhythm_result[3].tolist()
                    else:
                        features['bpm_intervals'] = convert_to_python_types(rhythm_result[3])
                
            elif LIBROSA_AVAILABLE:
                # Use librosa for rhythm features
                tempo, beats = librosa.beat.beat_track(y=audio, sr=DEFAULT_SAMPLE_RATE)
                features['bpm'] = ensure_float(tempo)
                features['confidence'] = 0.5  # Default confidence for librosa
                
            # Try to get external BPM from metadata
            if metadata and 'bpm' in metadata:
                try:
                    external_bpm = ensure_float(metadata['bpm'])
                    features['external_bpm'] = external_bpm
                except (ValueError, TypeError):
                    pass
            
            log_universal('INFO', 'Audio', f"BPM extracted: {features.get('bpm', 'N/A')}")
            if 'confidence' in features:
                log_universal('INFO', 'Audio', f"Rhythm confidence: {features['confidence']:.3f}")
            if 'external_bpm' in features:
                log_universal('INFO', 'Audio', f"External BPM: {features['external_bpm']:.1f}")
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error extracting rhythm features: {e}")
            # Return invalid marker like old working version
            features['bpm'] = -999.0
        
        return features

    @timeout(120, "Spectral feature extraction timed out")  # 2 minutes for spectral analysis
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract spectral features.
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary with spectral features
        """
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Use the same algorithms as the old working setup
                
                # Spectral centroid - use SpectralCentroidTime like old setup
                try:
                    centroid_algo = es.SpectralCentroidTime()
                    centroid_values = centroid_algo(audio)
                    centroid_mean = float(np.nanmean(centroid_values)) if isinstance(
                        centroid_values, (list, np.ndarray)) else float(centroid_values)
                    features['spectral_centroid'] = ensure_float(centroid_mean)
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Spectral centroid failed: {e}, using librosa fallback")
                    if LIBROSA_AVAILABLE:
                        centroid = librosa.feature.spectral_centroid(y=audio, sr=DEFAULT_SAMPLE_RATE)
                        features['spectral_centroid'] = ensure_float(np.mean(centroid))
                
                # Spectral rolloff - use custom frame-by-frame calculation like old setup
                try:
                    frame_size = 2048
                    hop_size = 1024
                    window = es.Windowing(type='hann')
                    spectrum = es.Spectrum()
                    
                    rolloff_list = []
                    frame_count = 0
                    
                    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                        frame_count += 1
                        if frame_count % 100 == 0:
                            log_universal('DEBUG', 'Audio', f"Processed {frame_count} frames for spectral rolloff")
                        
                        try:
                            spec = spectrum(window(frame))
                            if len(spec) > 0 and np.any(spec > 0):
                                energy = spec ** 2
                                total_energy = np.sum(energy)
                                if total_energy > 0:
                                    cumulative_energy = np.cumsum(energy)
                                    threshold = 0.85 * total_energy
                                    rolloff_idx = np.where(cumulative_energy >= threshold)[0]
                                    if len(rolloff_idx) > 0:
                                        rolloff_freq = (rolloff_idx[0] / len(spec)) * 22050
                                        rolloff_list.append(float(rolloff_freq))
                        except Exception as frame_error:
                            log_universal('DEBUG', 'Audio', f"Frame {frame_count} processing error: {frame_error}")
                            continue
                    
                    if rolloff_list:
                        rolloff_mean = float(np.mean(rolloff_list))
                        features['spectral_rolloff'] = ensure_float(rolloff_mean)
                    else:
                        features['spectral_rolloff'] = 0.0
                        
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Spectral rolloff failed: {e}, using librosa fallback")
                    if LIBROSA_AVAILABLE:
                        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=DEFAULT_SAMPLE_RATE)
                        features['spectral_rolloff'] = ensure_float(np.mean(rolloff))
                
                # Spectral flatness - use custom frame-by-frame calculation like old setup
                try:
                    frame_size = 2048
                    hop_size = 1024
                    window = es.Windowing(type='hann')
                    spectrum = es.Spectrum()
                    
                    flatness_list = []
                    frame_count = 0
                    
                    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                        frame_count += 1
                        if frame_count % 100 == 0:
                            log_universal('DEBUG', 'Audio', f"Processed {frame_count} frames for spectral flatness")
                        
                        try:
                            spec = spectrum(window(frame))
                            if len(spec) > 0 and np.any(spec > 0):
                                eps = 1e-10
                                spec_safe = spec + eps
                                geometric_mean = np.exp(np.mean(np.log(spec_safe)))
                                arithmetic_mean = np.mean(spec_safe)
                                flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
                                flatness_list.append(float(flatness))
                        except Exception as frame_error:
                            log_universal('DEBUG', 'Audio', f"Frame {frame_count} processing error: {frame_error}")
                            continue
                    
                    if flatness_list:
                        flatness_mean = float(np.mean(flatness_list))
                        features['spectral_flatness'] = ensure_float(flatness_mean)
                    else:
                        features['spectral_flatness'] = 0.0
                        
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Spectral flatness failed: {e}, using librosa fallback")
                    if LIBROSA_AVAILABLE:
                        flatness = librosa.feature.spectral_flatness(y=audio, sr=DEFAULT_SAMPLE_RATE)
                        features['spectral_flatness'] = ensure_float(np.mean(flatness))
                
            elif LIBROSA_AVAILABLE:
                # Use librosa for spectral features
                centroid = librosa.feature.spectral_centroid(y=audio, sr=DEFAULT_SAMPLE_RATE)
                features['spectral_centroid'] = ensure_float(np.mean(centroid))
                
                rolloff = librosa.feature.spectral_rolloff(y=audio, sr=DEFAULT_SAMPLE_RATE)
                features['spectral_rolloff'] = ensure_float(np.mean(rolloff))
                
                flatness = librosa.feature.spectral_flatness(y=audio, sr=DEFAULT_SAMPLE_RATE)
                features['spectral_flatness'] = ensure_float(np.mean(flatness))
            
            log_universal('INFO', 'Audio', f"Spectral centroid extracted: {features.get('spectral_centroid', 'N/A'):.3f}")
            log_universal('INFO', 'Audio', f"Spectral rolloff extracted: {features.get('spectral_rolloff', 'N/A'):.3f}")
            log_universal('INFO', 'Audio', f"Spectral flatness extracted: {features.get('spectral_flatness', 'N/A'):.3f}")
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error extracting spectral features: {e}")
        
        return features

    @timeout(60, "Loudness feature extraction timed out")  # 1 minute for loudness analysis
    def _extract_loudness(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract loudness features.
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary with loudness features
        """
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Use RMS like the old working setup
                try:
                    rms_algo = es.RMS()
                    rms_values = rms_algo(audio)
                    rms_mean = float(np.nanmean(rms_values)) if isinstance(
                        rms_values, (list, np.ndarray)) else float(rms_values)
                    features['loudness'] = ensure_float(rms_mean)
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"RMS loudness failed: {e}, using librosa fallback")
                    if LIBROSA_AVAILABLE:
                        rms = librosa.feature.rms(y=audio)
                        features['loudness'] = ensure_float(np.mean(rms))
                
                # Dynamic complexity
                try:
                    dynamic_complexity = es.DynamicComplexity()
                    complexity_value = dynamic_complexity(audio)
                    if isinstance(complexity_value, tuple):
                        features['dynamic_complexity'] = ensure_float(complexity_value[0])
                    else:
                        features['dynamic_complexity'] = ensure_float(complexity_value)
                except Exception as e:
                    log_universal('WARNING', 'Audio', f"Dynamic complexity failed: {e}")
                
            elif LIBROSA_AVAILABLE:
                # Use librosa for loudness
                rms = librosa.feature.rms(y=audio)
                features['loudness'] = ensure_float(np.mean(rms))
                
                # Calculate dynamic range
                dynamic_range = np.max(audio) - np.min(audio)
                features['dynamic_range'] = ensure_float(dynamic_range)
            
            log_universal('INFO', 'Audio', f"Loudness extracted: {features.get('loudness', 'N/A'):.3f}")
            if 'dynamic_complexity' in features:
                log_universal('INFO', 'Audio', f"Dynamic complexity extracted: {features['dynamic_complexity']:.3f}")
            if 'dynamic_range' in features:
                log_universal('INFO', 'Audio', f"Dynamic range extracted: {features['dynamic_range']:.3f}")
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error extracting loudness features: {e}")
        
        return features

    @timeout(180, "Key feature extraction timed out")  # 3 minutes for key analysis
    def _extract_key(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract key features.
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary with key features
        """
        features = {}
        
        try:
            log_universal('DEBUG', 'Audio', f"Starting key extraction for audio with {len(audio)} samples")
            
            if ESSENTIA_AVAILABLE:
                log_universal('DEBUG', 'Audio', "Using Essentia for key detection")
                # Key detection
                key = es.Key()
                log_universal('DEBUG', 'Audio', "Created Essentia Key object")
                log_universal('DEBUG', 'Audio', "Calling Essentia key detection...")
                key_result = key(audio)
                log_universal('DEBUG', 'Audio', f"Essentia key result: {key_result}")
                features['key'] = str(key_result[0])  # Ensure it's a string
                features['scale'] = str(key_result[1])  # Ensure it's a string
                features['key_strength'] = ensure_float(key_result[2])
                log_universal('DEBUG', 'Audio', "Essentia key detection completed successfully")
                
            elif LIBROSA_AVAILABLE:
                log_universal('DEBUG', 'Audio', "Using Librosa for key detection")
                # Use librosa for key detection
                chroma = librosa.feature.chroma_cqt(y=audio, sr=DEFAULT_SAMPLE_RATE)
                # Simple key detection (simplified)
                chroma_mean = np.mean(chroma, axis=1)
                key_idx = np.argmax(chroma_mean)
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                features['key'] = keys[key_idx]
                features['key_strength'] = ensure_float(chroma_mean[key_idx])
                log_universal('DEBUG', 'Audio', "Librosa key detection completed successfully")
            else:
                log_universal('WARNING', 'Audio', "No key detection library available")
            
            log_universal('INFO', 'Audio', f"Key extracted: {features.get('key', 'N/A')}")
            log_universal('INFO', 'Audio', f"Scale extracted: {features.get('scale', 'N/A')}")
            log_universal('INFO', 'Audio', f"Key strength extracted: {features.get('key_strength', 'N/A'):.3f}")
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error extracting key features: {e}")
            log_universal('DEBUG', 'Audio', f"Key extraction exception details: {type(e).__name__}: {e}")
        
        return features

    @timeout(240, "MFCC feature extraction timed out")  # 4 minutes for MFCC analysis
    def _extract_mfcc(self, audio: np.ndarray, num_coeffs: int = 13) -> Dict[str, Any]:
        """
        Extract MFCC features.
        
        Args:
            audio: Audio data
            num_coeffs: Number of MFCC coefficients
            
        Returns:
            Dictionary with MFCC features
        """
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # MFCC
                mfcc = es.MFCC(numberCoefficients=num_coeffs)
                mfcc_result = mfcc(audio)
                features['mfcc'] = convert_to_python_types(mfcc_result[0])
                features['mfcc_bands'] = convert_to_python_types(mfcc_result[1])
                
            elif LIBROSA_AVAILABLE:
                # Use librosa for MFCC
                mfcc = librosa.feature.mfcc(y=audio, sr=DEFAULT_SAMPLE_RATE, n_mfcc=num_coeffs)
                features['mfcc'] = convert_to_python_types(np.mean(mfcc, axis=1))
                features['mfcc_std'] = convert_to_python_types(np.std(mfcc, axis=1))
            
            log_universal('INFO', 'Audio', f"MFCC extracted: {num_coeffs} coefficients")
            if 'mfcc_bands' in features:
                log_universal('INFO', 'Audio', f"MFCC bands extracted: {len(features['mfcc_bands'])} bands")
            if 'mfcc_std' in features:
                log_universal('INFO', 'Audio', f"MFCC std extracted: {len(features['mfcc_std'])} values")
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error extracting MFCC features: {e}")
        
        return features

    @timeout(600, "MusiCNN feature extraction timed out")  # 10 minutes for MusiCNN analysis
    def _extract_musicnn_features(self, audio: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract MusiCNN features from audio.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary with MusiCNN features or None if failed
        """
        if not TENSORFLOW_AVAILABLE or not ESSENTIA_AVAILABLE:
            log_universal('WARNING', 'Audio', "MusiCNN extraction skipped - TensorFlow or Essentia not available")
            return None
        
        try:
            log_universal('INFO', 'Audio', "Extracting MusiCNN features...")
            
            # Check if MusiCNN is available
            if not hasattr(es, 'TensorflowPredictMusiCNN'):
                log_universal('WARNING', 'Audio', "MusiCNN extraction skipped - TensorflowPredictMusiCNN not available")
                return None
            
            # Initialize MusiCNN model
            musicnn = es.TensorflowPredictMusiCNN()
            
            # Extract embeddings
            embeddings = musicnn(audio)
            
            # Get tags (if available)
            tags = {}
            try:
                # Try to get tags from the model
                if hasattr(musicnn, 'getTags'):
                    tags = musicnn.getTags()
            except Exception as e:
                log_universal('DEBUG', 'Audio', f"Could not get MusiCNN tags: {e}")
            
            # Convert to list for JSON serialization
            embedding_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
            
            log_universal('INFO', 'Audio', f"MusiCNN extracted: {len(embedding_list)} dimensions")
            if tags:
                log_universal('INFO', 'Audio', f"MusiCNN tags extracted: {len(tags)} tags")
            return {
                'musicnn_embedding': embedding_list,
                'musicnn_tags': tags
            }
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"MusiCNN extraction failed: {str(e)}")
            return None

    @timeout(120, "Danceability extraction timed out")  # 2 minutes for danceability analysis
    def _extract_danceability(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract danceability features from audio.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary with danceability features
        """
        if not ESSENTIA_AVAILABLE:
            log_universal('WARNING', 'Audio', "Danceability extraction skipped - Essentia not available")
            return {'danceability': 0.0}
        
        try:
            log_universal('INFO', 'Audio', "Extracting danceability features...")
            
            # Use Essentia's Danceability algorithm
            dance_algo = es.Danceability()
            dance_result = dance_algo(audio)
            
            # Handle different return types from Essentia
            if isinstance(dance_result, tuple):
                if len(dance_result) >= 1:
                    dance_values = dance_result[0]
                else:
                    log_universal('WARNING', 'Audio', "Empty danceability result tuple")
                    dance_values = [0.0]
            else:
                dance_values = dance_result
            
            # Handle numpy arrays
            if isinstance(dance_values, np.ndarray):
                if dance_values.size == 1:
                    dance_mean = float(dance_values.item())
                else:
                    dance_mean = float(np.nanmean(dance_values))
            elif isinstance(dance_values, (list, tuple)):
                dance_mean = float(np.nanmean(dance_values))
            else:
                dance_mean = float(dance_values)
            
            # Ensure danceability is a valid number and normalize if needed
            if not np.isfinite(dance_mean):
                log_universal('WARNING', 'Audio', f"Invalid danceability value (non-finite): {dance_mean}, using default")
                dance_mean = 0.0
            elif dance_mean < 0:
                log_universal('WARNING', 'Audio', f"Negative danceability value: {dance_mean}, using default")
                dance_mean = 0.0
            elif dance_mean > 1:
                # The algorithm might return values in a different scale
                if dance_mean <= 10:  # If it's in 0-10 scale, normalize
                    dance_mean = dance_mean / 10.0
                    log_universal('DEBUG', 'Audio', f"Normalized danceability from 0-10 scale: {dance_mean:.3f}")
                elif dance_mean <= 100:  # If it's in 0-100 scale, normalize
                    dance_mean = dance_mean / 100.0
                    log_universal('DEBUG', 'Audio', f"Normalized danceability from 0-100 scale: {dance_mean:.3f}")
                else:
                    log_universal('WARNING', 'Audio', f"Danceability value out of expected range: {dance_mean}, using default")
                    dance_mean = 0.0
            
            log_universal('INFO', 'Audio', f"Danceability extracted: {dance_mean:.3f}")
            return {'danceability': ensure_float(dance_mean)}
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Danceability extraction failed: {str(e)}")
            return {'danceability': 0.0}

    @timeout(120, "Onset rate extraction timed out")  # 2 minutes for onset rate analysis
    def _extract_onset_rate(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract onset rate features from audio.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary with onset rate features
        """
        if not ESSENTIA_AVAILABLE:
            log_universal('WARNING', 'Audio', "Onset rate extraction skipped - Essentia not available")
            return {'onset_rate': 0.0}
        
        try:
            log_universal('INFO', 'Audio', "Extracting onset rate features...")
            
            # Use Essentia's OnsetRate algorithm
            onset_algo = es.OnsetRate()
            onset_result = onset_algo(audio)
            
            # Handle different return types from Essentia
            if isinstance(onset_result, tuple):
                if len(onset_result) >= 1:
                    onset_rate = onset_result[0]
                else:
                    log_universal('WARNING', 'Audio', "Empty onset rate result tuple")
                    onset_rate = 0.0
            else:
                onset_rate = onset_result
            
            # Handle numpy arrays
            if isinstance(onset_rate, np.ndarray):
                if onset_rate.size == 1:
                    onset_rate = float(onset_rate.item())
                else:
                    onset_rate = float(np.nanmean(onset_rate))
            else:
                # Convert to float
                onset_rate = float(onset_rate)
            
            # Ensure onset rate is a valid number
            if not np.isfinite(onset_rate) or onset_rate < 0:
                log_universal('WARNING', 'Audio', f"Invalid onset rate value: {onset_rate}, using default")
                onset_rate = 0.0
            
            log_universal('INFO', 'Audio', f"Onset rate extracted: {onset_rate:.2f} onsets/sec")
            return {'onset_rate': ensure_float(onset_rate)}
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Onset rate extraction failed: {str(e)}")
            return {'onset_rate': 0.0}

    @timeout(60, "Zero crossing rate extraction timed out")  # 1 minute for ZCR analysis
    def _extract_zcr(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract zero crossing rate features from audio.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary with zero crossing rate features
        """
        if not ESSENTIA_AVAILABLE:
            log_universal('WARNING', 'Audio', "Zero crossing rate extraction skipped - Essentia not available")
            return {'zcr': 0.0}
        
        try:
            log_universal('INFO', 'Audio', "Extracting zero crossing rate features...")
            
            # Use Essentia's ZeroCrossingRate algorithm
            zcr_algo = es.ZeroCrossingRate()
            zcr_values = zcr_algo(audio)
            
            zcr_mean = float(np.nanmean(zcr_values)) if isinstance(
                zcr_values, (list, np.ndarray)) else float(zcr_values)
            
            log_universal('INFO', 'Audio', f"Zero crossing rate extracted: {zcr_mean:.3f}")
            return {'zcr': ensure_float(zcr_mean)}
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Zero crossing rate extraction failed: {str(e)}")
            return {'zcr': 0.0}

    @timeout(180, "Spectral contrast extraction timed out")  # 3 minutes for spectral contrast analysis
    def _extract_spectral_contrast(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract spectral contrast features from audio.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary with spectral contrast features
        """
        if not ESSENTIA_AVAILABLE:
            log_universal('WARNING', 'Audio', "Spectral contrast extraction skipped - Essentia not available")
            return {'spectral_contrast': 0.0}
        
        try:
            log_universal('INFO', 'Audio', "Extracting spectral contrast features...")
            
            # Use frame-by-frame processing for spectral contrast
            frame_size = 2048
            hop_size = 1024
            
            # Initialize Essentia algorithms for spectral contrast
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            spectral_peaks = es.SpectralPeaks()
            
            contrast_list = []
            frame_count = 0
            
            # Process audio frame by frame
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                frame_count += 1
                if frame_count % 100 == 0:
                    log_universal('DEBUG', 'Audio', f"Processed {frame_count} frames for spectral contrast")
                
                try:
                    spec = spectrum(window(frame))
                    freqs, mags = spectral_peaks(spec)
                    
                    if len(freqs) > 0 and len(mags) > 0:
                        # Ensure mags is a numpy array and has valid values
                        mags_array = np.array(mags)
                        if len(mags_array) > 0 and np.any(mags_array > 0):
                            # Calculate spectral contrast manually
                            # Sort magnitudes and find valleys
                            sorted_mags = np.sort(mags_array)
                            # Ensure at least 1 element
                            third = max(1, len(sorted_mags) // 3)
                            valleys = sorted_mags[:third]  # Bottom third
                            peaks = sorted_mags[-third:]   # Top third
                            
                            if len(peaks) > 0 and len(valleys) > 0:
                                contrast = float(np.mean(peaks) - np.mean(valleys))
                                contrast_list.append(contrast)
                
                except Exception as frame_error:
                    log_universal('DEBUG', 'Audio', f"Frame {frame_count} processing error: {frame_error}")
                    continue
            
            # Return mean contrast across all frames
            if contrast_list:
                contrast_mean = float(np.mean(contrast_list))
                log_universal('INFO', 'Audio', f"Spectral contrast extracted: {contrast_mean:.3f}")
                return {'spectral_contrast': ensure_float(contrast_mean)}
            else:
                log_universal('WARNING', 'Audio', "No valid contrast values calculated, returning 0.0")
                return {'spectral_contrast': 0.0}
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Spectral contrast extraction failed: {str(e)}")
            return {'spectral_contrast': 0.0}

    @timeout(120, "Chroma extraction timed out")  # 2 minutes for chroma analysis
    def _extract_chroma(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Extract chroma features from audio.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary with chroma features
        """
        if not LIBROSA_AVAILABLE:
            log_universal('WARNING', 'Audio', "Chroma extraction skipped - Librosa not available")
            return {'chroma': [0.0] * 12}
        
        try:
            log_universal('INFO', 'Audio', "Extracting chroma features...")
            
            # Extract chroma features using librosa
            chroma = librosa.feature.chroma_cqt(y=audio, sr=DEFAULT_SAMPLE_RATE)
            
            # Calculate mean chroma values across time
            chroma_mean = np.mean(chroma, axis=1)
            
            # Convert to list for JSON serialization
            chroma_list = chroma_mean.tolist()
            
            log_universal('INFO', 'Audio', f"Chroma extracted: {len(chroma_list)} features, values: {[f'{x:.3f}' for x in chroma_list]}")
            return {'chroma': [ensure_float(x) for x in chroma_list]}
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f"Chroma extraction failed: {str(e)}")
            return {'chroma': [0.0] * 12}

    def _validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate extracted features.
        
        Args:
            features: Features dictionary
            
        Returns:
            True if features are valid, False otherwise
        """
        try:
            # Check for essential features
            essential_features = ['bpm', 'loudness']
            missing_features = [f for f in essential_features if f not in features]
            
            if missing_features:
                log_universal('WARNING', 'Audio', f"Missing essential features: {missing_features}")
                return False
            
            # Check for valid BPM (excluding invalid markers)
            bpm = features.get('bpm')
            if bpm is not None and bpm == -999.0:
                log_universal('WARNING', 'Audio', f"BPM extraction failed (invalid marker: {bpm})")
                return False
            elif bpm is not None and (bpm < 30 or bpm > 300):
                log_universal('WARNING', 'Audio', f"Invalid BPM value: {bpm}")
                return False
            
            # Check for valid loudness (excluding invalid markers)
            loudness = features.get('loudness')
            if loudness is not None and loudness == -999.0:
                log_universal('WARNING', 'Audio', f"Loudness extraction failed (invalid marker: {loudness})")
                return False
            elif loudness is not None and (loudness < -100 or loudness > 100):
                log_universal('WARNING', 'Audio', f"Invalid loudness value: {loudness}")
                return False
            
            # Check for valid spectral centroid (excluding invalid markers)
            spectral_centroid = features.get('spectral_centroid')
            if spectral_centroid is not None and spectral_centroid == -999.0:
                log_universal('WARNING', 'Audio', f"Spectral centroid extraction failed (invalid marker: {spectral_centroid})")
                return False
            elif spectral_centroid is not None and (spectral_centroid < 0 or spectral_centroid > 22050):
                log_universal('WARNING', 'Audio', f"Invalid spectral centroid value: {spectral_centroid}")
                return False
            
            # Check for valid key (excluding invalid markers)
            key = features.get('key')
            if key is not None and key == 'INVALID':
                log_universal('WARNING', 'Audio', f"Key extraction failed (invalid marker: {key})")
                return False
            
            # Check for valid key strength (excluding invalid markers)
            key_strength = features.get('key_strength')
            if key_strength is not None and key_strength == -999.0:
                log_universal('WARNING', 'Audio', f"Key strength extraction failed (invalid marker: {key_strength})")
                return False
            elif key_strength is not None and (key_strength < 0 or key_strength > 1):
                log_universal('WARNING', 'Audio', f"Invalid key strength value: {key_strength}")
                return False
            
            log_universal('DEBUG', 'Audio', "Features validation passed")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Error validating features: {e}")
            return False
        
    def _is_valid_for_playlist(self, features: Dict[str, Any]) -> bool:
        """
        Check if features are valid for playlist generation (no invalid markers).
        
        Args:
            features: Features dictionary
            
        Returns:
            True if features can be used for playlist generation, False otherwise
        """
        try:
            # Check for invalid markers that indicate failed extraction
            invalid_markers = {
                'bpm': -999.0,
                'loudness': -999.0,
                'spectral_centroid': -999.0,
                'key_strength': -999.0,
                'key': 'INVALID',
                'scale': 'INVALID'
            }
            
            for feature, invalid_value in invalid_markers.items():
                if feature in features and features[feature] == invalid_value:
                    log_universal('DEBUG', 'Audio', f"Feature '{feature}' has invalid marker: {invalid_value}")
                    return False
            
            # Check for invalid MFCC (all values -999.0)
            mfcc = features.get('mfcc')
            if mfcc and isinstance(mfcc, list) and all(x == -999.0 for x in mfcc):
                log_universal('DEBUG', 'Audio', "MFCC has invalid markers")
                return False
            
            # Check for invalid MusiCNN features (all values -999.0)
            musicnn = features.get('musicnn_features')
            if musicnn and isinstance(musicnn, list) and all(x == -999.0 for x in musicnn):
                log_universal('DEBUG', 'Audio', "MusiCNN features have invalid markers")
                return False
            
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Error checking playlist validity: {e}")
            return False

    def get_version(self) -> str:
        """Get analyzer version."""
        return self.VERSION
    



# Global audio analyzer instance - created lazily to avoid circular imports
_audio_analyzer_instance = None

def get_audio_analyzer() -> 'AudioAnalyzer':
    """Get the global audio analyzer instance, creating it if necessary."""
    global _audio_analyzer_instance
    if _audio_analyzer_instance is None:
        _audio_analyzer_instance = AudioAnalyzer()
    return _audio_analyzer_instance 
