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

# Import local modules
from .logging_setup import get_logger, log_function_call, log_performance, log_feature_extraction_step, log_resource_usage

logger = get_logger('playlista.audio_analyzer')

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_HOP_SIZE = 512
DEFAULT_FRAME_SIZE = 2048
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes


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
                # On Windows, just call the function without timeout
                # TODO: Implement proper timeout for Windows using threading
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def safe_json_dumps(obj):
    """Safely serialize object to JSON, handling NumPy types."""
    if obj is None:
        return None
    try:
        return json.dumps(obj)
    except TypeError as e:
        if "not JSON serializable" in str(e):
            # Convert NumPy types to Python native types
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
            else:
                # Handle non-dict objects
                if isinstance(obj, np.floating):
                    return json.dumps(float(obj))
                elif isinstance(obj, np.integer):
                    return json.dumps(int(obj))
                elif isinstance(obj, np.ndarray):
                    return json.dumps(obj.tolist())
                else:
                    return json.dumps(str(obj))
        else:
            raise e


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
        
        # MusiCNN configuration
        self.musicnn_model_path = config.get('MUSICNN_MODEL_PATH', '/app/models/msd-musicnn-1.pb')
        self.musicnn_json_path = config.get('MUSICNN_JSON_PATH', '/app/models/msd-musicnn-1.json')
        self.musicnn_timeout_seconds = config.get('MUSICNN_TIMEOUT_SECONDS', 60)
        self.musicnn_batch_size = config.get('MUSICNN_BATCH_SIZE', 1)
        self.musicnn_min_audio_length_seconds = config.get('MUSICNN_MIN_AUDIO_LENGTH_SECONDS', 1)
        self.musicnn_sample_rate = config.get('MUSICNN_SAMPLE_RATE', 44100)
        self.musicnn_memory_limit_gb = config.get('MUSICNN_MEMORY_LIMIT_GB', 2.0)
        self.musicnn_gpu_enabled = config.get('MUSICNN_GPU_ENABLED', False)
        self.musicnn_cache_enabled = config.get('MUSICNN_CACHE_ENABLED', True)
        self.musicnn_feature_dimensions = config.get('MUSICNN_FEATURE_DIMENSIONS', 200)
        self.musicnn_confidence_threshold = config.get('MUSICNN_CONFIDENCE_THRESHOLD', 0.5)
        self.musicnn_fallback_enabled = config.get('MUSICNN_FALLBACK_ENABLED', True)
        
        # Initialize MusiCNN model
        self.musicnn_model = None
        self._init_musicnn()
        
        # Check library availability
        self._check_library_availability()
        
        logger.info(f"🔧 Initializing AudioAnalyzer v{self.VERSION}")
        logger.debug(f"📋 Cache file: {self.cache_file}")
        logger.debug(f"📋 Library path: {self.library}")
        logger.debug(f"📋 Music path: {self.music}")
        logger.debug(f"📋 MusiCNN config: timeout={self.musicnn_timeout_seconds}s, batch_size={self.musicnn_batch_size}")
        logger.info(f"✅ AudioAnalyzer initialized successfully")

    def _init_musicnn(self):
        """Initialize MusiCNN model if available."""
        if not TENSORFLOW_AVAILABLE or not ESSENTIA_AVAILABLE:
            logger.warning("⚠️ TensorFlow or Essentia not available - MusiCNN disabled")
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
                logger.info("✅ MusiCNN model loaded successfully")
            else:
                logger.warning(f"⚠️ MusiCNN model files not found:")
                logger.warning(f"   Model: {self.musicnn_model_path}")
                logger.warning(f"   Config: {self.musicnn_json_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize MusiCNN: {e}")
            self.musicnn_model = None

    def _check_library_availability(self):
        """Check availability of audio processing libraries."""
        logger.debug("🔍 Checking audio processing library availability")
        
        if ESSENTIA_AVAILABLE:
            logger.info("✅ Essentia available for feature extraction")
        else:
            logger.warning("⚠️ Essentia not available - limited features")
        
        if LIBROSA_AVAILABLE:
            logger.info("✅ Librosa available for feature extraction")
        else:
            logger.warning("⚠️ Librosa not available - limited features")
        
        if MUTAGEN_AVAILABLE:
            logger.info("✅ Mutagen available for metadata extraction")
        else:
            logger.warning("⚠️ Mutagen not available - no metadata")
        
        if TENSORFLOW_AVAILABLE:
            logger.info("✅ TensorFlow available for MusiCNN features")
        else:
            logger.warning("⚠️ TensorFlow not available - MusiCNN features disabled")
        
        if self.musicnn_model is not None:
            logger.info("✅ MusiCNN model loaded successfully")
        else:
            logger.warning("⚠️ MusiCNN model not available - advanced features disabled")

    @log_function_call
    @timeout(DEFAULT_TIMEOUT_SECONDS)
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
            logger.warning(f"🔒 Resource manager forcing basic analysis: {forced_guidance['reason']}")
            analysis_config = self._apply_forced_basic_config(analysis_config)
        
        analysis_type = analysis_config.get('analysis_type', 'basic')
        features_config = analysis_config.get('features_config', {})
        
        logger.info(f"🎵 Extracting {analysis_type} features from: {filename}")
        logger.debug(f"📋 Feature config: {features_config}")
        
        start_time = time.time()
        
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"❌ File not found: {audio_path}")
                return None
            
            # Get file info
            file_size_bytes = os.path.getsize(audio_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            logger.debug(f"📊 File size: {file_size_mb:.1f}MB")
            
            # Load audio
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                logger.error(f"❌ Failed to load audio: {filename}")
                return None
            
            # Extract metadata (always enabled)
            metadata = self._extract_metadata(audio_path)
            
            # Extract features based on configuration
            features = self._extract_features_by_config(audio_path, audio, metadata, features_config)
            
            if features is None:
                logger.error(f"❌ Failed to extract features: {filename}")
                return None
            
            # Validate features
            if not self._validate_features(features):
                logger.warning(f"⚠️ Features validation failed: {filename}")
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
            logger.info(f"✅ Successfully extracted {analysis_type} features from {filename} in {extract_time:.2f}s")
            
            # Log performance
            log_performance("Audio feature extraction", extract_time,
                          filename=filename, file_size_mb=file_size_mb, analysis_type=analysis_type)
            
            return result
            
        except TimeoutException:
            logger.error(f"⏰ Analysis timed out for {filename}")
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting features from {filename}: {e}")
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
                'extract_metadata': True
            }
        }

    def _get_forced_guidance(self) -> Dict[str, Any]:
        """Get forced guidance from resource manager."""
        try:
            from .resource_manager import resource_manager
            return resource_manager.get_forced_analysis_guidance()
        except Exception as e:
            logger.warning(f"Could not get resource manager guidance: {e}")
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
            
            logger.debug(f"📊 Audio loading decision for {os.path.basename(audio_path)}:")
            logger.debug(f"   File size: {file_size_mb:.1f}MB")
            logger.debug(f"   Streaming enabled: {self.streaming_enabled}")
            logger.debug(f"   Streaming threshold: {self.streaming_large_file_threshold_mb}MB")
            logger.debug(f"   Streaming memory limit: {self.streaming_memory_limit_percent}%")
            logger.debug(f"   Streaming chunk duration: {self.streaming_chunk_duration_seconds}s")
            
            # Use streaming loader for large files if enabled
            if self.streaming_enabled and file_size_mb > self.streaming_large_file_threshold_mb:
                logger.info(f"📊 Large file detected ({file_size_mb:.1f}MB) - using streaming loader")
                return self._load_audio_streaming(audio_path)
            else:
                # Use traditional loading for small files
                logger.debug(f"📊 Small file detected ({file_size_mb:.1f}MB) - using traditional loading")
                return self._load_audio_traditional(audio_path)
                
        except Exception as e:
            logger.error(f"❌ Error loading audio {audio_path}: {e}")
            return None
    
    def _load_audio_traditional(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio using traditional method (entire file in memory)."""
        try:
            if LIBROSA_AVAILABLE:
                # Use librosa for loading
                audio, sr = librosa.load(audio_path, sr=DEFAULT_SAMPLE_RATE, mono=True)
                logger.debug(f"📊 Loaded audio: {len(audio)} samples, {sr}Hz")
                return audio
            elif ESSENTIA_AVAILABLE:
                # Use Essentia MonoLoader with proper parameters
                loader = es.MonoLoader(
                    filename=audio_path,
                    sampleRate=DEFAULT_SAMPLE_RATE,
                    downmix='mix',  # Mix stereo to mono
                    resampleQuality=1  # Good quality resampling
                )
                audio = loader()
                logger.debug(f"📊 Loaded audio: {len(audio)} samples, {DEFAULT_SAMPLE_RATE}Hz")
                return audio
            elif SOUNDFILE_AVAILABLE:
                # Use soundfile for loading
                audio, sr = sf.read(audio_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)  # Convert to mono
                if sr != DEFAULT_SAMPLE_RATE:
                    # Simple resampling
                    ratio = DEFAULT_SAMPLE_RATE / sr
                    new_length = int(len(audio) * ratio)
                    indices = np.linspace(0, len(audio) - 1, new_length)
                    audio = np.interp(indices, np.arange(len(audio)), audio)
                logger.debug(f"📊 Loaded audio: {len(audio)} samples, {DEFAULT_SAMPLE_RATE}Hz")
                return audio
            elif WAVE_AVAILABLE and audio_path.lower().endswith('.wav'):
                # Use wave module for WAV files
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
                    logger.debug(f"📊 Loaded audio: {len(audio)} samples, {DEFAULT_SAMPLE_RATE}Hz")
                    return audio
            else:
                logger.error("❌ No audio loading library available")
                logger.error(f"   Available libraries: Essentia={ESSENTIA_AVAILABLE}, Librosa={LIBROSA_AVAILABLE}, SoundFile={SOUNDFILE_AVAILABLE}, Wave={WAVE_AVAILABLE}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading audio {audio_path}: {e}")
            return None
    
    def _load_audio_streaming(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio using streaming method - for small files only."""
        try:
            from .streaming_audio_loader import get_streaming_loader
            
            # Get streaming loader with configuration
            streaming_loader = get_streaming_loader(
                memory_limit_percent=self.streaming_memory_limit_percent,
                chunk_duration_seconds=self.streaming_chunk_duration_seconds
            )
            
            # For small files, we can still concatenate chunks
            # For large files, we should use true streaming analysis
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            
            if file_size_mb > 100:  # Large file threshold
                logger.warning(f"⚠️ Large file detected ({file_size_mb:.1f}MB) - using true streaming analysis")
                return self._analyze_large_file_streaming(audio_path, streaming_loader)
            
            # For smaller files, collect chunks and concatenate
            chunks = []
            total_samples = 0
            
            for chunk, start_time, end_time in streaming_loader.load_audio_chunks(audio_path):
                chunks.append(chunk)
                total_samples += len(chunk)
                logger.debug(f"📊 Loaded chunk: {start_time:.1f}s - {end_time:.1f}s ({len(chunk)} samples)")
            
            if not chunks:
                logger.error("❌ No chunks loaded from streaming loader")
                logger.warning("🔄 Falling back to traditional loading...")
                return self._load_audio_traditional(audio_path)
            
            # Concatenate all chunks
            audio = np.concatenate(chunks)
            logger.debug(f"📊 Concatenated audio: {len(audio)} samples total")
            
            return audio
            
        except Exception as e:
            logger.error(f"❌ Error in streaming audio load: {e}")
            logger.warning("🔄 Falling back to traditional loading...")
            return self._load_audio_traditional(audio_path)
    
    def _analyze_large_file_streaming(self, audio_path: str, streaming_loader) -> Optional[np.ndarray]:
        """Analyze large files using true streaming - process one chunk at a time."""
        try:
            logger.info(f"🎵 Starting true streaming analysis for large file: {os.path.basename(audio_path)}")
            
            # For large files, we'll use a representative sample instead of loading the entire file
            # This prevents memory issues while still providing useful analysis
            
            # Get file duration
            duration = self._get_audio_duration(audio_path)
            if duration is None:
                logger.error("❌ Could not determine audio duration")
                return None
            
            # For very large files (> 100MB), use a representative sample
            # Take samples from beginning, middle, and end
            sample_duration = min(60.0, duration / 10)  # 60 seconds or 1/10th of file
            
            samples = []
            
            # Sample from beginning
            if duration > sample_duration:
                try:
                    chunk, sr = librosa.load(
                        audio_path,
                        sr=DEFAULT_SAMPLE_RATE,
                        mono=True,
                        offset=0,
                        duration=sample_duration
                    )
                    samples.append(chunk)
                    logger.debug(f"📊 Loaded beginning sample: {len(chunk)} samples")
                except Exception as e:
                    logger.warning(f"⚠️ Error loading beginning sample: {e}")
            
            # Sample from middle
            if duration > sample_duration * 2:
                try:
                    middle_start = duration / 2 - sample_duration / 2
                    chunk, sr = librosa.load(
                        audio_path,
                        sr=DEFAULT_SAMPLE_RATE,
                        mono=True,
                        offset=middle_start,
                        duration=sample_duration
                    )
                    samples.append(chunk)
                    logger.debug(f"📊 Loaded middle sample: {len(chunk)} samples")
                except Exception as e:
                    logger.warning(f"⚠️ Error loading middle sample: {e}")
            
            # Sample from end
            if duration > sample_duration:
                try:
                    end_start = max(0, duration - sample_duration)
                    chunk, sr = librosa.load(
                        audio_path,
                        sr=DEFAULT_SAMPLE_RATE,
                        mono=True,
                        offset=end_start,
                        duration=sample_duration
                    )
                    samples.append(chunk)
                    logger.debug(f"📊 Loaded end sample: {len(chunk)} samples")
                except Exception as e:
                    logger.warning(f"⚠️ Error loading end sample: {e}")
            
            if not samples:
                logger.error("❌ No samples loaded from large file")
                return None
            
            # Concatenate samples to create a representative audio
            representative_audio = np.concatenate(samples)
            logger.info(f"📊 Created representative audio: {len(representative_audio)} samples from {len(samples)} samples")
            
            return representative_audio
            
        except Exception as e:
            logger.error(f"❌ Error in true streaming analysis: {e}")
            return None

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
            return metadata
        
        try:
            audio_file = MutagenFile(audio_path)
            if audio_file is not None:
                # Extract common metadata fields
                for tag in ['title', 'artist', 'album', 'genre', 'year', 'tracknumber']:
                    if tag in audio_file:
                        value = audio_file[tag]
                        if isinstance(value, list) and len(value) > 0:
                            metadata[tag] = str(value[0])
                        else:
                            metadata[tag] = str(value)
                
                logger.debug(f"📋 Extracted basic metadata: {list(metadata.keys())}")
                
                # Enrich metadata with external APIs
                enriched_metadata = self._enrich_metadata_with_external_apis(metadata)
                
                return enriched_metadata
                
        except Exception as e:
            logger.warning(f"⚠️ Error extracting metadata from {audio_path}: {e}")
        
        return metadata
    
    def _enrich_metadata_with_external_apis(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich metadata using external APIs (MusicBrainz and Last.fm).
        
        Args:
            metadata: Basic metadata from audio file
            
        Returns:
            Enriched metadata dictionary
        """
        try:
            # Import external APIs module
            from .external_apis import metadata_enrichment_service
            
            # Check if external APIs are available
            if not metadata_enrichment_service.is_available():
                logger.debug("External APIs not available - skipping enrichment")
                return metadata
            
            # Enrich metadata
            enriched_metadata = metadata_enrichment_service.enrich_metadata(metadata)
            
            # Log enrichment results
            if enriched_metadata != metadata:
                new_fields = set(enriched_metadata.keys()) - set(metadata.keys())
                if new_fields:
                    logger.debug(f"🔗 Enriched metadata with: {list(new_fields)}")
            
            return enriched_metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Error enriching metadata with external APIs: {e}")
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
                        log_feature_extraction_step(audio_path, 'rhythm', duration, True, 
                                                  feature_value=len(rhythm_features))
                    else:
                        failed_features.append('rhythm')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'rhythm', duration, False, 
                                                  error="No rhythm features returned")
                except Exception as e:
                    failed_features.append('rhythm')
                    duration = time.time() - start_time
                    log_feature_extraction_step(audio_path, 'rhythm', duration, False, error=str(e))
            
            # Extract spectral features (if enabled)
            if features_config.get('extract_spectral', True):
                start_time = time.time()
                try:
                    spectral_features = self._extract_spectral_features(audio)
                    if spectral_features:
                        features.update(spectral_features)
                        extracted_features.append('spectral')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'spectral', duration, True, 
                                                  feature_value=len(spectral_features))
                    else:
                        failed_features.append('spectral')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'spectral', duration, False, 
                                                  error="No spectral features returned")
                except Exception as e:
                    failed_features.append('spectral')
                    duration = time.time() - start_time
                    log_feature_extraction_step(audio_path, 'spectral', duration, False, error=str(e))
            
            # Extract loudness (if enabled)
            if features_config.get('extract_loudness', True):
                start_time = time.time()
                try:
                    loudness_features = self._extract_loudness(audio)
                    if loudness_features:
                        features.update(loudness_features)
                        extracted_features.append('loudness')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'loudness', duration, True, 
                                                  feature_value=len(loudness_features))
                    else:
                        failed_features.append('loudness')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'loudness', duration, False, 
                                                  error="No loudness features returned")
                except Exception as e:
                    failed_features.append('loudness')
                    duration = time.time() - start_time
                    log_feature_extraction_step(audio_path, 'loudness', duration, False, error=str(e))
            
            # Extract key (if enabled)
            if features_config.get('extract_key', True):
                start_time = time.time()
                try:
                    key_features = self._extract_key(audio)
                    if key_features:
                        features.update(key_features)
                        extracted_features.append('key')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'key', duration, True, 
                                                  feature_value=len(key_features))
                    else:
                        failed_features.append('key')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'key', duration, False, 
                                                  error="No key features returned")
                except Exception as e:
                    failed_features.append('key')
                    duration = time.time() - start_time
                    log_feature_extraction_step(audio_path, 'key', duration, False, error=str(e))
            
            # Extract MFCC (if enabled)
            if features_config.get('extract_mfcc', True):
                start_time = time.time()
                try:
                    mfcc_features = self._extract_mfcc(audio)
                    if mfcc_features:
                        features.update(mfcc_features)
                        extracted_features.append('mfcc')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'mfcc', duration, True, 
                                                  feature_value=len(mfcc_features))
                    else:
                        failed_features.append('mfcc')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'mfcc', duration, False, 
                                                  error="No MFCC features returned")
                except Exception as e:
                    failed_features.append('mfcc')
                    duration = time.time() - start_time
                    log_feature_extraction_step(audio_path, 'mfcc', duration, False, error=str(e))
            
            # Extract MusiCNN features (if enabled)
            if features_config.get('extract_musicnn', False):
                start_time = time.time()
                try:
                    musicnn_features = self._extract_musicnn_features(audio)
                    if musicnn_features:
                        features.update(musicnn_features)
                        extracted_features.append('musicnn')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'musicnn', duration, True, 
                                                  feature_value=len(musicnn_features))
                    else:
                        failed_features.append('musicnn')
                        duration = time.time() - start_time
                        log_feature_extraction_step(audio_path, 'musicnn', duration, False, 
                                                  error="No MusiCNN features returned")
                except Exception as e:
                    failed_features.append('musicnn')
                    duration = time.time() - start_time
                    log_feature_extraction_step(audio_path, 'musicnn', duration, False, error=str(e))
            
            # Add metadata (always included)
            features['metadata'] = metadata
            
            # Log overall feature extraction summary
            logger.debug(f"📊 Feature extraction summary: {len(extracted_features)} successful, {len(failed_features)} failed")
            logger.debug(f"✅ Extracted features: {extracted_features}")
            if failed_features:
                logger.debug(f"❌ Failed features: {failed_features}")
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return None

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
                rhythm = rhythm_extractor(audio)
                
                features['bpm'] = float(rhythm[0])
                features['confidence'] = float(rhythm[1])
                features['estimates'] = rhythm[2].tolist()
                features['bpm_intervals'] = rhythm[3].tolist()
                
            elif LIBROSA_AVAILABLE:
                # Use librosa for rhythm features
                tempo, beats = librosa.beat.beat_track(y=audio, sr=DEFAULT_SAMPLE_RATE)
                features['bpm'] = float(tempo)
                features['confidence'] = 0.5  # Default confidence for librosa
                
            # Try to get external BPM from metadata
            if metadata and 'bpm' in metadata:
                try:
                    external_bpm = float(metadata['bpm'])
                    features['external_bpm'] = external_bpm
                except (ValueError, TypeError):
                    pass
            
            logger.debug(f"🎵 Rhythm features: BPM={features.get('bpm', 'N/A')}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting rhythm features: {e}")
        
        return features

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
                # Spectral centroid
                centroid = es.Centroid()
                features['spectral_centroid'] = float(centroid(audio))
                
                # Spectral rolloff
                rolloff = es.RollOff()
                features['spectral_rolloff'] = float(rolloff(audio))
                
                # Spectral flatness
                flatness = es.Flatness()
                features['spectral_flatness'] = float(flatness(audio))
                
            elif LIBROSA_AVAILABLE:
                # Use librosa for spectral features
                centroid = librosa.feature.spectral_centroid(y=audio, sr=DEFAULT_SAMPLE_RATE)
                features['spectral_centroid'] = float(np.mean(centroid))
                
                rolloff = librosa.feature.spectral_rolloff(y=audio, sr=DEFAULT_SAMPLE_RATE)
                features['spectral_rolloff'] = float(np.mean(rolloff))
                
                flatness = librosa.feature.spectral_flatness(y=audio, sr=DEFAULT_SAMPLE_RATE)
                features['spectral_flatness'] = float(np.mean(flatness))
            
            logger.debug(f"📊 Spectral features extracted")
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting spectral features: {e}")
        
        return features

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
                # Loudness
                loudness = es.Loudness()
                features['loudness'] = float(loudness(audio))
                
                # Dynamic complexity
                dynamic_complexity = es.DynamicComplexity()
                features['dynamic_complexity'] = float(dynamic_complexity(audio))
                
            elif LIBROSA_AVAILABLE:
                # Use librosa for loudness
                rms = librosa.feature.rms(y=audio)
                features['loudness'] = float(np.mean(rms))
                
                # Calculate dynamic range
                dynamic_range = np.max(audio) - np.min(audio)
                features['dynamic_range'] = float(dynamic_range)
            
            logger.debug(f"🔊 Loudness features extracted")
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting loudness features: {e}")
        
        return features

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
            if ESSENTIA_AVAILABLE:
                # Key detection
                key = es.Key()
                key_result = key(audio)
                features['key'] = key_result[0]
                features['scale'] = key_result[1]
                features['key_strength'] = float(key_result[2])
                
            elif LIBROSA_AVAILABLE:
                # Use librosa for key detection
                chroma = librosa.feature.chroma_cqt(y=audio, sr=DEFAULT_SAMPLE_RATE)
                # Simple key detection (simplified)
                chroma_mean = np.mean(chroma, axis=1)
                key_idx = np.argmax(chroma_mean)
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                features['key'] = keys[key_idx]
                features['key_strength'] = float(chroma_mean[key_idx])
            
            logger.debug(f"🎼 Key features: {features.get('key', 'N/A')}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting key features: {e}")
        
        return features

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
                features['mfcc'] = mfcc_result[0].tolist()
                features['mfcc_bands'] = mfcc_result[1].tolist()
                
            elif LIBROSA_AVAILABLE:
                # Use librosa for MFCC
                mfcc = librosa.feature.mfcc(y=audio, sr=DEFAULT_SAMPLE_RATE, n_mfcc=num_coeffs)
                features['mfcc'] = np.mean(mfcc, axis=1).tolist()
                features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
            
            logger.debug(f"📊 MFCC features extracted ({num_coeffs} coefficients)")
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting MFCC features: {e}")
        
        return features

    def _extract_musicnn_features(self, audio: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract MusiCNN features using TensorFlow.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dictionary with MusiCNN features, or None if failed
        """
        if not TENSORFLOW_AVAILABLE or self.musicnn_model is None:
            logger.warning("⚠️ TensorFlow or MusiCNN model not available - skipping MusiCNN features")
            return None
        
        try:
            # Reshape audio for MusiCNN input (batch_size, frames, channels)
            # MusiCNN expects 1D audio, so we need to convert to 2D (frames, 1)
            # and then add a batch dimension.
            # For simplicity, we'll assume a single channel for now.
            # If audio is mono, shape is (n_samples,)
            # If audio is stereo, shape is (n_samples, 2)
            # MusiCNN expects (batch_size, frames, channels)
            # We need to pad or truncate audio to a multiple of the frame size
            # and then add a batch dimension.
            
            # Ensure audio is mono and has a reasonable length for MusiCNN
            # MusiCNN typically expects 1-second segments or more.
            # For simplicity, let's assume a minimum length of 1 second.
            # If audio is shorter, pad it.
            min_audio_length_for_musicnn = DEFAULT_SAMPLE_RATE * 1 # 1 second
            if len(audio) < min_audio_length_for_musicnn:
                # Pad audio to the minimum length
                padding_length = min_audio_length_for_musicnn - len(audio)
                audio = np.pad(audio, (0, padding_length), 'constant')
                logger.debug(f"📊 Padded audio for MusiCNN: {len(audio)} samples")
            
            # Reshape for MusiCNN input (batch_size, frames, channels)
            # MusiCNN expects (batch_size, frames, channels)
            # We need to add a batch dimension.
            # For simplicity, we'll assume a batch size of 1.
            # If audio is mono, shape is (n_samples,) -> (1, n_samples, 1)
            # If audio is stereo, shape is (n_samples, 2) -> (1, n_samples, 2)
            
            # Add a batch dimension
            audio_for_musicnn = audio.reshape(1, -1, 1) # (1, n_samples, 1)
            
            # Predict using MusiCNN
            # MusiCNN output is (batch_size, num_classes)
            # We need to reshape it to (num_classes,)
            musicnn_output = self.musicnn_model(audio_for_musicnn)
            
            # The output is a tensor, we need to convert it to a numpy array
            # and then reshape it to (num_classes,)
            musicnn_output_np = musicnn_output.numpy()
            
            # The output is (batch_size, num_classes)
            # We need to take the mean across the batch dimension
            # and then reshape it to (num_classes,)
            musicnn_features = np.mean(musicnn_output_np, axis=0).tolist()
            
            # Map MusiCNN output to feature names
            # This is a placeholder. In a real scenario, you'd define
            # a mapping based on the MusiCNN model's output.
            # For example, if MusiCNN predicts 100 classes, you might
            # have features like 'genre_1', 'genre_2', etc.
            # For now, we'll just return the raw output.
            
            logger.debug(f"📊 MusiCNN features extracted: {len(musicnn_features)}")
            return {'musicnn_features': musicnn_features}
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting MusiCNN features: {e}")
            return None

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
                logger.warning(f"⚠️ Missing essential features: {missing_features}")
                return False
            
            # Check for valid BPM
            bpm = features.get('bpm')
            if bpm is not None and (bpm < 30 or bpm > 300):
                logger.warning(f"⚠️ Invalid BPM value: {bpm}")
                return False
            
            # Check for valid loudness
            loudness = features.get('loudness')
            if loudness is not None and (loudness < -100 or loudness > 100):
                logger.warning(f"⚠️ Invalid loudness value: {loudness}")
                return False
            
            logger.debug("✅ Features validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating features: {e}")
            return False

    def get_version(self) -> str:
        """Get analyzer version."""
        return self.VERSION
    
    def get_config(self) -> Dict[str, Any]:
        """Get current analyzer configuration."""
        return {
            'version': self.VERSION,
            'cache_file': self.cache_file,
            'library': self.library,
            'music': self.music,
            'essentia_available': ESSENTIA_AVAILABLE,
            'librosa_available': LIBROSA_AVAILABLE,
            'mutagen_available': MUTAGEN_AVAILABLE,
            'musicnn_config': {
                'model_path': self.musicnn_model_path,
                'json_path': self.musicnn_json_path,
                'timeout_seconds': self.musicnn_timeout_seconds,
                'batch_size': self.musicnn_batch_size,
                'min_audio_length_seconds': self.musicnn_min_audio_length_seconds,
                'sample_rate': self.musicnn_sample_rate,
                'memory_limit_gb': self.musicnn_memory_limit_gb,
                'gpu_enabled': self.musicnn_gpu_enabled,
                'cache_enabled': self.musicnn_cache_enabled,
                'feature_dimensions': self.musicnn_feature_dimensions,
                'confidence_threshold': self.musicnn_confidence_threshold,
                'fallback_enabled': self.musicnn_fallback_enabled
            }
        }


# Global audio analyzer instance
audio_analyzer = AudioAnalyzer() 