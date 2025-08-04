"""
Lazy import utilities for heavy libraries to improve startup time.
Defers loading of TensorFlow, Essentia, and other compute-heavy libraries.
"""

import functools
import importlib
from typing import Any, Optional
import threading

_import_lock = threading.Lock()
_imported_modules = {}


class LazyImport:
    """
    Lazy import wrapper that delays module loading until first access.
    """
    
    def __init__(self, module_name: str, fallback_available: bool = False):
        self.module_name = module_name
        self.fallback_available = fallback_available
        self._module = None
        self._import_error = None
        
    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._load_module()
        
        if self._module is None:
            raise self._import_error
            
        return getattr(self._module, name)
    
    def _load_module(self):
        """Load the module with thread safety."""
        with _import_lock:
            if self._module is not None:
                return
                
            try:
                self._module = importlib.import_module(self.module_name)
                _imported_modules[self.module_name] = self._module
            except ImportError as e:
                self._import_error = e
                
    @property
    def available(self) -> bool:
        """Check if module is available without importing it."""
        if self._module is not None:
            return True
        if self._import_error is not None:
            return False
            
        try:
            importlib.util.find_spec(self.module_name)
            return True
        except (ImportError, AttributeError, ValueError):
            return False


# Lazy imports for heavy libraries
tensorflow = LazyImport('tensorflow')
essentia_standard = LazyImport('essentia.standard')
essentia = LazyImport('essentia')
librosa = LazyImport('librosa')
mutagen = LazyImport('mutagen')

# TensorFlow specific lazy loading
def get_tensorflow():
    """Get TensorFlow with proper configuration."""
    import os
    # Set TensorFlow logging before import
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    
    tf = tensorflow
    if tf.available:
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
    return tf

def get_essentia():
    """Get Essentia with proper configuration."""
    import os
    os.environ.setdefault('ESSENTIA_LOG_LEVEL', 'error')
    return essentia_standard

def is_tensorflow_available() -> bool:
    """Check if TensorFlow is available without importing."""
    return tensorflow.available

def is_essentia_available() -> bool:
    """Check if Essentia is available without importing."""
    return essentia_standard.available

def is_librosa_available() -> bool:
    """Check if Librosa is available without importing."""
    return librosa.available

def is_mutagen_available() -> bool:
    """Check if Mutagen is available without importing."""
    return mutagen.available

@functools.lru_cache(maxsize=1)
def get_audio_libraries_status():
    """Get status of all audio processing libraries."""
    return {
        'tensorflow': {
            'available': is_tensorflow_available(),
            'description': 'Deep learning framework for MusiCNN features'
        },
        'essentia': {
            'available': is_essentia_available(),
            'description': 'Audio analysis library for feature extraction'
        },
        'librosa': {
            'available': is_librosa_available(),
            'description': 'Audio analysis library (fallback for Essentia)'
        },
        'mutagen': {
            'available': is_mutagen_available(),
            'description': 'Audio metadata extraction library'
        }
    }