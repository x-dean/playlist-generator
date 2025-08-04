"""
Model Manager for Playlist Generator Simple.
Provides thread-safe, shared instances of machine learning models.
"""

import os
import time
import json
import threading
from typing import Dict, Any, Optional, Tuple
from threading import RLock
from datetime import datetime

# Import local modules
from .logging_setup import get_logger, log_function_call, log_universal
from .config_loader import config_loader

logger = get_logger('playlista.model_manager')

# Check for TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    log_universal('WARNING', 'Model', 'TensorFlow not available')

# Check for Essentia
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    log_universal('WARNING', 'Model', 'Essentia not available')


class ModelManager:
    """
    Thread-safe model manager for shared model instances.
    
    Handles:
    - MusicNN model loading and caching
    - Thread-safe model access
    - Model lifecycle management
    - Memory-efficient model sharing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        if config is None:
            try:
                config = config_loader.get_audio_analysis_config()
            except NameError:
                # Handle case where config_loader is not available (e.g., in multiprocessing)
                from .config_loader import ConfigLoader
                local_config_loader = ConfigLoader()
                config = local_config_loader.get_audio_analysis_config()
        
        self.config = config
        
        # Model paths from config file
        self.musicnn_model_path = config.get('MUSICNN_MODEL_PATH', '/app/models/msd-musicnn-1.pb')
        self.musicnn_json_path = config.get('MUSICNN_JSON_PATH', '/app/models/msd-musicnn-1.json')
        
        # MusicNN specific settings from config
        self.musicnn_enabled = config.get('MUSICNN_ENABLED', True)
        self.musicnn_timeout_seconds = config.get('MUSICNN_TIMEOUT_SECONDS', 60)
        self.musicnn_max_file_size_mb = config.get('MUSICNN_MAX_FILE_SIZE_MB', 500)
        self.musicnn_min_memory_gb = config.get('MUSICNN_MIN_MEMORY_GB', 3)
        self.musicnn_max_cpu_percent = config.get('MUSICNN_MAX_CPU_PERCENT', 70)
        
        # Model instances (thread-safe)
        self._musicnn_activations_model = None
        self._musicnn_embeddings_model = None
        self._musicnn_tag_names = None
        self._musicnn_metadata = None
        
        # Thread safety
        self._model_lock = threading.RLock()
        self._initialization_lock = threading.Lock()
        self._models_initialized = False
        
        # Model loading settings
        self.model_cache_enabled = config.get('MODEL_CACHE_ENABLED', True)
        self.model_loading_timeout = config.get('MODEL_LOADING_TIMEOUT_SECONDS', 60)
        self.model_memory_limit_gb = config.get('MODEL_MEMORY_LIMIT_GB', 4.0)
        
        log_universal('INFO', 'Model', 'Initializing ModelManager')
        log_universal('DEBUG', 'Model', f'MusicNN model path: {self.musicnn_model_path}')
        log_universal('DEBUG', 'Model', f'MusicNN JSON path: {self.musicnn_json_path}')
        log_universal('INFO', 'Model', 'ModelManager initialized successfully')

    @log_function_call
    def get_musicnn_models(self) -> Tuple[Optional[Any], Optional[Any], Optional[list], Optional[Dict[str, Any]]]:
        """
        Get shared MusicNN model instances (thread-safe).
        
        Returns:
            Tuple of (activations_model, embeddings_model, tag_names, metadata)
        """
        with self._model_lock:
            if not self._models_initialized:
                self._initialize_musicnn_models()
            
            return (
                self._musicnn_activations_model,
                self._musicnn_embeddings_model,
                self._musicnn_tag_names,
                self._musicnn_metadata
            )

    def _initialize_musicnn_models(self):
        """Initialize MusicNN models with enhanced error handling and performance optimizations."""
        if self._models_initialized:
            return
            
        with self._model_lock:
            if self._models_initialized:  # Double-check after acquiring lock
                return
                
            start_time = time.time()
            log_universal('INFO', 'Model', 'Initializing MusicNN models...')
            
            try:
                # Check dependencies with enhanced validation
                if not self._check_dependencies():
                    self._models_initialized = True
                    return
                
                # Validate model files with enhanced checks
                if not self._validate_model_files():
                    self._models_initialized = True
                    return
                
                # Load models with enhanced error handling
                if not self._load_musicnn_models():
                    self._models_initialized = True
                    return
                
                initialization_time = time.time() - start_time
                log_universal('INFO', 'Model', f'MusicNN models initialized successfully in {initialization_time:.2f}s')
                
            except Exception as e:
                log_universal('ERROR', 'Model', f'MusicNN model initialization failed: {e}')
                self._models_initialized = True
    
    def _check_dependencies(self) -> bool:
        """Check dependencies with enhanced validation."""
        try:
            # Check if MusicNN is enabled in config
            if not self.musicnn_enabled:
                log_universal('INFO', 'Model', 'MusicNN disabled in config')
                return False
            
            # Check TensorFlow
            if not TENSORFLOW_AVAILABLE:
                log_universal('WARNING', 'Model', 'TensorFlow not available - MusicNN models disabled')
                return False
            
            # Check Essentia
            if not ESSENTIA_AVAILABLE:
                log_universal('WARNING', 'Model', 'Essentia not available - MusicNN models disabled')
                return False
            
            # Test TensorFlow session creation
            try:
                import tensorflow as tf
                with tf.compat.v1.Session() as session:
                    log_universal('DEBUG', 'Model', 'TensorFlow session test successful')
            except Exception as e:
                log_universal('WARNING', 'Model', f'TensorFlow session test failed: {e}')
                return False
            
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Model', f'Dependency check failed: {e}')
            return False
    
    def _validate_model_files(self) -> bool:
        """Validate model files with enhanced checks."""
        try:
            # Check model file
            if not os.path.exists(self.musicnn_model_path):
                log_universal('WARNING', 'Model', f'MusicNN model file not found: {self.musicnn_model_path}')
                return False
            
            # Check file size
            model_size = os.path.getsize(self.musicnn_model_path)
            if model_size < 1000000:  # Less than 1MB
                log_universal('WARNING', 'Model', f'MusicNN model file too small: {model_size} bytes')
                return False
            
            log_universal('DEBUG', 'Model', f'MusicNN model file size: {model_size / (1024*1024):.1f} MB')
            
            # Check JSON config file
            if not os.path.exists(self.musicnn_json_path):
                log_universal('WARNING', 'Model', f'MusicNN JSON config not found: {self.musicnn_json_path}')
                return False
            
            # Validate JSON file
            try:
                with open(self.musicnn_json_path, 'r') as f:
                    test_data = json.load(f)
                if 'classes' not in test_data:
                    log_universal('WARNING', 'Model', 'MusicNN JSON config missing classes')
                    return False
                log_universal('DEBUG', 'Model', f'MusicNN JSON config validated: {len(test_data["classes"])} classes')
            except Exception as e:
                log_universal('WARNING', 'Model', f'MusicNN JSON config validation failed: {e}')
                return False
            
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Model', f'Model file validation failed: {e}')
            return False
    
    def _load_musicnn_models(self) -> bool:
        """Load MusicNN models with enhanced error handling."""
        try:
            import essentia.standard as es
            import tensorflow as tf
            
            # Configure TensorFlow to suppress warnings
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all warnings including GPU
            
            # Configure GPU memory growth to prevent warnings
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    log_universal('DEBUG', 'Model', 'GPU memory growth enabled')
                else:
                    # Disable GPU logging when no GPU is available
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
                    log_universal('DEBUG', 'Model', 'No GPU detected, using CPU only')
            except Exception as e:
                log_universal('DEBUG', 'Model', f'GPU configuration failed: {e}')
                # Ensure GPU warnings are suppressed
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            # Create TensorFlow session with optimized settings
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads = 1
            # Suppress GPU warnings in session config
            config.log_device_placement = False
            
            with tf.compat.v1.Session(config=config) as session:
                # Load MusicNN JSON configuration
                if not self._load_musicnn_config():
                    return False
                
                # Load MusicNN models
                if not self._load_musicnn_model_instances():
                    return False
            
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Model', f'Model loading failed: {e}')
            return False
    
    def _load_musicnn_config(self) -> bool:
        """Load MusicNN configuration with enhanced validation."""
        try:
            with open(self.musicnn_json_path, 'r') as f:
                self._musicnn_metadata = json.load(f)
            
            self._musicnn_tag_names = self._musicnn_metadata.get('classes', [])
            if not self._musicnn_tag_names:
                log_universal('WARNING', 'Model', 'No tag names found in MusicNN config')
                return False
            
            # Validate tag names
            valid_tags = [tag for tag in self._musicnn_tag_names if isinstance(tag, str) and len(tag) > 0]
            if len(valid_tags) != len(self._musicnn_tag_names):
                log_universal('WARNING', 'Model', f'Some invalid tag names found: {len(self._musicnn_tag_names) - len(valid_tags)} invalid')
                self._musicnn_tag_names = valid_tags
            
            log_universal('DEBUG', 'Model', f'Loaded {len(self._musicnn_tag_names)} valid tag names from MusicNN config')
            return True
            
        except Exception as e:
            log_universal('WARNING', 'Model', f'Failed to load MusicNN JSON config: {e}')
            return False
    
    def _load_musicnn_model_instances(self) -> bool:
        """Load MusicNN model instances with enhanced error handling."""
        try:
            import essentia.standard as es
            
            # Load activations model
            log_universal('DEBUG', 'Model', 'Loading MusicNN activations model...')
            try:
                self._musicnn_activations_model = es.TensorflowPredictMusiCNN(graphFilename=self.musicnn_model_path)
                log_universal('INFO', 'Model', 'Loaded MusicNN activations model')
            except Exception as e:
                log_universal('ERROR', 'Model', f'Failed to load activations model: {e}')
                return False
            
            # Load embeddings model
            output_layer = self._get_embeddings_output_layer()
            log_universal('DEBUG', 'Model', f'Loading MusicNN embeddings model with output layer: {output_layer}')
            
            try:
                self._musicnn_embeddings_model = es.TensorflowPredictMusiCNN(
                    graphFilename=self.musicnn_model_path,
                    output=output_layer
                )
                log_universal('INFO', 'Model', 'Loaded MusicNN embeddings model')
            except Exception as e:
                log_universal('ERROR', 'Model', f'Failed to load embeddings model: {e}')
                return False
            
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Model', f'Model instance loading failed: {e}')
            return False
    
    def _get_embeddings_output_layer(self) -> str:
        """Get embeddings output layer with fallback."""
        output_layer = 'model/dense_1/BiasAdd'  # Default fallback
        
        if 'schema' in self._musicnn_metadata and 'outputs' in self._musicnn_metadata['schema']:
            for output in self._musicnn_metadata['schema']['outputs']:
                if 'description' in output and output['description'] == 'embeddings':
                    output_layer = output['name']
                    log_universal('DEBUG', 'Model', f'Found embeddings output layer: {output_layer}')
                    break
        
        return output_layer

    def is_musicnn_available(self) -> bool:
        """Check if MusicNN models are available."""
        with self._model_lock:
            if not self._models_initialized:
                self._initialize_musicnn_models()
            
            return (self._musicnn_activations_model is not None and 
                   self._musicnn_embeddings_model is not None and
                   self._musicnn_tag_names is not None)

    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced model manager information."""
        with self._model_lock:
            info = {
                'models_initialized': self._models_initialized,
                'musicnn_available': self.is_musicnn_available(),
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'essentia_available': ESSENTIA_AVAILABLE,
                'model_cache_enabled': self.model_cache_enabled,
                'musicnn_model_path': self.musicnn_model_path,
                'musicnn_json_path': self.musicnn_json_path,
                'model_loading_timeout': self.model_loading_timeout,
                'model_memory_limit_gb': self.model_memory_limit_gb,
                # MusicNN config settings
                'musicnn_enabled': self.musicnn_enabled,
                'musicnn_timeout_seconds': self.musicnn_timeout_seconds,
                'musicnn_max_file_size_mb': self.musicnn_max_file_size_mb,
                'musicnn_min_memory_gb': self.musicnn_min_memory_gb,
                'musicnn_max_cpu_percent': self.musicnn_max_cpu_percent
            }
            
            # Add detailed model information if available
            if self._models_initialized:
                info.update({
                    'tag_names_count': len(self._musicnn_tag_names) if self._musicnn_tag_names else 0,
                    'activations_model_loaded': self._musicnn_activations_model is not None,
                    'embeddings_model_loaded': self._musicnn_embeddings_model is not None,
                    'metadata_loaded': self._musicnn_metadata is not None
                })
                
                # Add sample tag names if available
                if self._musicnn_tag_names and len(self._musicnn_tag_names) > 0:
                    info['sample_tag_names'] = self._musicnn_tag_names[:10]  # First 10 tags
                
                # Add model file information
                try:
                    if os.path.exists(self.musicnn_model_path):
                        model_size = os.path.getsize(self.musicnn_model_path)
                        info['model_file_size_mb'] = model_size / (1024 * 1024)
                        info['model_file_exists'] = True
                    else:
                        info['model_file_exists'] = False
                        
                    if os.path.exists(self.musicnn_json_path):
                        json_size = os.path.getsize(self.musicnn_json_path)
                        info['json_file_size_kb'] = json_size / 1024
                        info['json_file_exists'] = True
                    else:
                        info['json_file_exists'] = False
                except Exception as e:
                    log_universal('WARNING', 'Model', f'Failed to get file info: {e}')
            
            return info

    def cleanup(self):
        """Clean up model resources."""
        with self._model_lock:
            log_universal('INFO', 'Model', 'Cleaning up model resources')
            
            # Clear model references
            self._musicnn_activations_model = None
            self._musicnn_embeddings_model = None
            self._musicnn_tag_names = None
            self._musicnn_metadata = None
            self._models_initialized = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            log_universal('INFO', 'Model', 'Model cleanup completed')

    def is_file_suitable_for_musicnn(self, file_size_bytes: int, available_memory_gb: float = None) -> bool:
        """Check if a file is suitable for MusicNN processing based on config settings."""
        try:
            # Check if MusicNN is enabled
            if not self.musicnn_enabled:
                return False
            
            # Check file size limit
            file_size_mb = file_size_bytes / (1024 * 1024)
            if file_size_mb > self.musicnn_max_file_size_mb:
                log_universal('DEBUG', 'Model', f'File too large for MusicNN: {file_size_mb:.1f}MB > {self.musicnn_max_file_size_mb}MB')
                return False
            
            # Check memory requirement if available memory is provided
            if available_memory_gb is not None and available_memory_gb < self.musicnn_min_memory_gb:
                log_universal('DEBUG', 'Model', f'Insufficient memory for MusicNN: {available_memory_gb:.1f}GB < {self.musicnn_min_memory_gb}GB')
                return False
            
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Model', f'File suitability check failed: {e}')
            return False


# Global model manager instance
_model_manager_instance = None
_model_manager_lock = threading.Lock()

def get_model_manager(config: Dict[str, Any] = None) -> 'ModelManager':
    """Get the global model manager instance (thread-safe singleton)."""
    global _model_manager_instance
    
    if _model_manager_instance is None:
        with _model_manager_lock:
            if _model_manager_instance is None:
                _model_manager_instance = ModelManager(config)
    
    return _model_manager_instance 