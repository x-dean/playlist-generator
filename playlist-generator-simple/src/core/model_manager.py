"""
Model Manager for Playlist Generator Simple.
Provides thread-safe shared model instances for MusicNN and other ML models.
"""

import os
import threading
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Import local modules
from .logging_setup import get_logger, log_function_call, log_universal

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
            from .config_loader import config_loader
            config = config_loader.get_audio_analysis_config()
        
        self.config = config
        
        # Model paths
        self.musicnn_model_path = config.get('MUSICNN_MODEL_PATH', '/app/models/msd-musicnn-1.pb')
        self.musicnn_json_path = config.get('MUSICNN_JSON_PATH', '/app/models/msd-musicnn-1.json')
        
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
        """Initialize MusicNN models (called once, thread-safe)."""
        with self._initialization_lock:
            if self._models_initialized:
                return
            
            log_universal('INFO', 'Model', 'Initializing MusicNN models...')
            start_time = time.time()
            
            try:
                if not TENSORFLOW_AVAILABLE:
                    log_universal('WARNING', 'Model', 'TensorFlow not available - MusicNN models disabled')
                    self._models_initialized = True
                    return
                
                if not ESSENTIA_AVAILABLE:
                    log_universal('WARNING', 'Model', 'Essentia not available - MusicNN models disabled')
                    self._models_initialized = True
                    return
                
                # Check model files exist
                if not os.path.exists(self.musicnn_model_path):
                    log_universal('WARNING', 'Model', f'MusicNN model not found: {self.musicnn_model_path}')
                    self._models_initialized = True
                    return
                
                if not os.path.exists(self.musicnn_json_path):
                    log_universal('WARNING', 'Model', f'MusicNN JSON config not found: {self.musicnn_json_path}')
                    self._models_initialized = True
                    return
                
                # Load tag names from JSON
                try:
                    import json
                    with open(self.musicnn_json_path, 'r') as f:
                        metadata = json.load(f)
                    self._musicnn_tag_names = metadata.get('classes', [])
                    self._musicnn_metadata = metadata
                    log_universal('DEBUG', 'Model', f'Loaded {len(self._musicnn_tag_names)} tag names from MusicNN config')
                except Exception as e:
                    log_universal('WARNING', 'Model', f'Failed to load MusicNN JSON config: {e}')
                    self._models_initialized = True
                    return
                
                # Load MusicNN models
                try:
                    log_universal('DEBUG', 'Model', 'Loading MusicNN activations model...')
                    
                    # Initialize MusiCNN for activations (auto-tagging)
                    self._musicnn_activations_model = es.TensorflowPredictMusiCNN(graphFilename=self.musicnn_model_path)
                    log_universal('INFO', 'Model', 'Loaded MusicNN activations model')
                    
                    # Get embeddings using different output layer
                    output_layer = 'model/dense_1/BiasAdd'
                    if 'schema' in self._musicnn_metadata and 'outputs' in self._musicnn_metadata['schema']:
                        for output in self._musicnn_metadata['schema']['outputs']:
                            if 'description' in output and output['description'] == 'embeddings':
                                output_layer = output['name']
                                break
                    
                    log_universal('DEBUG', 'Model', f'Loading MusicNN embeddings model with output layer: {output_layer}')
                    
                    self._musicnn_embeddings_model = es.TensorflowPredictMusiCNN(
                        graphFilename=self.musicnn_model_path,
                        output=output_layer
                    )
                    log_universal('INFO', 'Model', 'Loaded MusicNN embeddings model')
                    
                    initialization_time = time.time() - start_time
                    log_universal('INFO', 'Model', f'MusicNN models initialized successfully in {initialization_time:.2f}s')
                    
                except Exception as e:
                    log_universal('ERROR', 'Model', f'Failed to load MusicNN models: {e}')
                    self._musicnn_activations_model = None
                    self._musicnn_embeddings_model = None
                
                self._models_initialized = True
                
            except Exception as e:
                log_universal('ERROR', 'Model', f'MusicNN model initialization failed: {e}')
                self._models_initialized = True

    def is_musicnn_available(self) -> bool:
        """Check if MusicNN models are available."""
        with self._model_lock:
            if not self._models_initialized:
                self._initialize_musicnn_models()
            
            return (self._musicnn_activations_model is not None and 
                   self._musicnn_embeddings_model is not None and
                   self._musicnn_tag_names is not None)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model manager information."""
        with self._model_lock:
            return {
                'models_initialized': self._models_initialized,
                'musicnn_available': self.is_musicnn_available(),
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'essentia_available': ESSENTIA_AVAILABLE,
                'model_cache_enabled': self.model_cache_enabled,
                'musicnn_model_path': self.musicnn_model_path,
                'musicnn_json_path': self.musicnn_json_path
            }

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