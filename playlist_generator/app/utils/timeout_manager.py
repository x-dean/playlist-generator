"""
Adaptive timeout management for audio analysis.
"""
import os
import time
import logging
import threading
import signal
from typing import Optional, Callable, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class AdaptiveTimeoutManager:
    """Manages adaptive timeouts based on file characteristics."""
    
    def __init__(self):
        # Base timeouts in seconds
        self.base_timeout = 180  # 3 minutes for normal files
        self.large_file_threshold_mb = 50
        self.very_large_file_threshold_mb = 100
        self.extremely_large_file_threshold_mb = 200
        
        # Timeout multipliers
        self.large_file_multiplier = 1.5
        self.very_large_file_multiplier = 2.0
        self.extremely_large_file_multiplier = 3.0
        
        # Memory-based adjustments
        self.memory_threshold_percent = 85
        self.memory_timeout_reduction = 0.7  # Reduce timeout by 30% when memory is high
        
        # Feature-specific timeouts
        self.feature_timeouts = {
            'rhythm': 120,      # 2 minutes for BPM extraction
            'spectral': 90,     # 1.5 minutes for spectral features
            'mfcc': 60,         # 1 minute for MFCC
            'chroma': 45,       # 45 seconds for chroma
            'musicnn': 180,     # 3 minutes for MusiCNN
            'metadata': 30      # 30 seconds for metadata enrichment
        }
    
    def calculate_timeout(self, file_path: str, feature: str = None) -> int:
        """Calculate adaptive timeout based on file characteristics."""
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except (OSError, FileNotFoundError):
            file_size_mb = 0
        
        # Base timeout for the feature
        if feature and feature in self.feature_timeouts:
            base_timeout = self.feature_timeouts[feature]
        else:
            base_timeout = self.base_timeout
        
        # Adjust based on file size
        if file_size_mb > self.extremely_large_file_threshold_mb:
            timeout = base_timeout * self.extremely_large_file_multiplier
        elif file_size_mb > self.very_large_file_threshold_mb:
            timeout = base_timeout * self.very_large_file_multiplier
        elif file_size_mb > self.large_file_threshold_mb:
            timeout = base_timeout * self.large_file_multiplier
        else:
            timeout = base_timeout
        
        # Adjust based on memory usage
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.memory_threshold_percent:
                timeout = int(timeout * self.memory_timeout_reduction)
                logger.debug(f"Reduced timeout to {timeout}s due to high memory usage ({memory_percent:.1f}%)")
        except ImportError:
            pass
        
        # Ensure minimum timeout
        timeout = max(timeout, 30)  # Minimum 30 seconds
        
        logger.debug(f"Calculated timeout for {os.path.basename(file_path)} ({file_size_mb:.1f}MB, {feature}): {timeout}s")
        return timeout
    
    @contextmanager
    def timeout_context(self, file_path: str, feature: str = None, error_message: str = None):
        """Context manager for adaptive timeouts."""
        timeout_seconds = self.calculate_timeout(file_path, feature)
        
        if error_message is None:
            error_message = f"Processing timed out after {timeout_seconds}s"
        
        with timeout_context(timeout_seconds, error_message):
            yield


class TimeoutException(Exception):
    """Custom exception for timeout errors."""
    pass


class TimeoutContext:
    """Thread-safe timeout context manager."""
    
    def __init__(self, timeout_seconds: int, error_message: str = "Processing timed out"):
        self.timeout_seconds = timeout_seconds
        self.error_message = error_message
        self._timer = None
        self._timed_out = False
        self._lock = threading.Lock()
    
    def _timeout_handler(self):
        """Handle timeout by raising exception in main thread."""
        with self._lock:
            if not self._timed_out:
                self._timed_out = True
                # Use signal to interrupt the main thread
                try:
                    import signal
                    os.kill(os.getpid(), signal.SIGALRM)
                except (ImportError, OSError):
                    # Fallback: just log the timeout
                    logger.error(f"TIMEOUT: {self.error_message}")
    
    def __enter__(self):
        """Start the timeout timer."""
        self._timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
        self._timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cancel the timeout timer."""
        if self._timer:
            self._timer.cancel()
        
        if self._timed_out:
            raise TimeoutException(self.error_message)
        
        return False


@contextmanager
def timeout_context(timeout_seconds: int, error_message: str = "Processing timed out"):
    """Context manager for timeouts with better error handling."""
    timeout_mgr = TimeoutContext(timeout_seconds, error_message)
    with timeout_mgr:
        yield


def timeout_decorator(seconds: int = 60, error_message: str = "Processing timed out"):
    """Decorator for timeout handling."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with timeout_context(seconds, error_message):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_timeout_call(func: Callable, *args, timeout_seconds: int = 60, 
                     error_message: str = "Function call timed out", **kwargs) -> Optional[Any]:
    """Safely call a function with timeout, returning None on timeout."""
    try:
        with timeout_context(timeout_seconds, error_message):
            return func(*args, **kwargs)
    except TimeoutException as e:
        logger.warning(f"Function {func.__name__} timed out: {e}")
        return None
    except Exception as e:
        logger.error(f"Function {func.__name__} failed: {e}")
        return None


# Global timeout manager instance
timeout_manager = AdaptiveTimeoutManager() 