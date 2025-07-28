"""
Improved error recovery and retry logic for audio analysis.
"""
import time
import logging
import random
from typing import Callable, Any, Optional, Tuple, List
from functools import wraps

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt using exponential backoff."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            jitter = random.uniform(0, 0.1 * delay)
            delay += jitter
        
        return delay


class ErrorRecoveryManager:
    """Manages error recovery and retry logic."""
    
    def __init__(self, retry_config: RetryConfig = None):
        self.retry_config = retry_config or RetryConfig()
        self.failure_counts = {}  # Track failures per file
        self.success_counts = {}   # Track successes per file
    
    def should_retry(self, file_path: str, error: Exception) -> bool:
        """Determine if we should retry based on error type and previous failures."""
        # Don't retry on certain error types
        non_retryable_errors = (
            FileNotFoundError,
            PermissionError,
            OSError,  # File system errors
        )
        
        if isinstance(error, non_retryable_errors):
            logger.debug(f"Not retrying {file_path} due to non-retryable error: {type(error).__name__}")
            return False
        
        # Check failure count
        failure_count = self.failure_counts.get(file_path, 0)
        if failure_count >= self.retry_config.max_attempts:
            logger.warning(f"Max retry attempts reached for {file_path} ({failure_count} failures)")
            return False
        
        return True
    
    def record_failure(self, file_path: str, error: Exception):
        """Record a failure for the given file."""
        self.failure_counts[file_path] = self.failure_counts.get(file_path, 0) + 1
        logger.debug(f"Recorded failure for {file_path} (total: {self.failure_counts[file_path]})")
    
    def record_success(self, file_path: str):
        """Record a success for the given file."""
        self.success_counts[file_path] = self.success_counts.get(file_path, 0) + 1
        # Reset failure count on success
        if file_path in self.failure_counts:
            del self.failure_counts[file_path]
        logger.debug(f"Recorded success for {file_path}")
    
    def get_failure_stats(self) -> Tuple[int, int]:
        """Get failure statistics."""
        total_failures = sum(self.failure_counts.values())
        unique_failed_files = len(self.failure_counts)
        return total_failures, unique_failed_files


def retry_with_backoff(retry_config: RetryConfig = None):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = retry_config or RetryConfig()
            recovery_mgr = ErrorRecoveryManager(config)
            
            # Extract file path from args if possible
            file_path = None
            if args and isinstance(args[0], str):
                file_path = args[0]
            
            last_error = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    if file_path:
                        recovery_mgr.record_success(file_path)
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    if file_path:
                        recovery_mgr.record_failure(file_path, e)
                    
                    if not recovery_mgr.should_retry(file_path, e):
                        logger.error(f"Not retrying {file_path or 'operation'} after {attempt} attempts")
                        raise e
                    
                    if attempt < config.max_attempts:
                        delay = config.calculate_delay(attempt)
                        logger.warning(f"Attempt {attempt} failed for {file_path or 'operation'}, "
                                     f"retrying in {delay:.1f}s (error: {type(e).__name__})")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed for {file_path or 'operation'}")
                        raise e
            
            # This should never be reached, but just in case
            raise last_error
        
        return wrapper
    return decorator


def safe_operation(operation_name: str = "operation", default_value: Any = None):
    """Decorator for safe operation execution with fallback value."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{operation_name} failed: {e}")
                return default_value
        return wrapper
    return decorator


class AnalysisErrorHandler:
    """Handles analysis-specific errors and recovery."""
    
    def __init__(self):
        self.recovery_mgr = ErrorRecoveryManager()
        self.error_patterns = {
            'memory_error': ['MemoryError', 'OutOfMemoryError'],
            'timeout_error': ['TimeoutException', 'TimeoutError'],
            'audio_error': ['AudioException', 'EssentiaException'],
            'database_error': ['sqlite3.Error', 'DatabaseError'],
            'network_error': ['requests.RequestException', 'ConnectionError']
        }
    
    def classify_error(self, error: Exception) -> str:
        """Classify the error type for appropriate handling."""
        error_type = type(error).__name__
        
        for category, patterns in self.error_patterns.items():
            if any(pattern in error_type for pattern in patterns):
                return category
        
        return 'unknown'
    
    def handle_analysis_error(self, file_path: str, error: Exception, 
                            context: str = "analysis") -> bool:
        """Handle analysis error and determine if retry is appropriate."""
        error_category = self.classify_error(error)
        
        logger.warning(f"Analysis error for {file_path} ({context}): {error_category} - {error}")
        
        # Different handling based on error category
        if error_category == 'memory_error':
            logger.error(f"Memory error for {file_path} - marking as failed")
            return False  # Don't retry memory errors
        
        elif error_category == 'timeout_error':
            logger.warning(f"Timeout error for {file_path} - will retry")
            return self.recovery_mgr.should_retry(file_path, error)
        
        elif error_category == 'audio_error':
            logger.warning(f"Audio processing error for {file_path} - will retry")
            return self.recovery_mgr.should_retry(file_path, error)
        
        elif error_category == 'database_error':
            logger.error(f"Database error for {file_path} - marking as failed")
            return False  # Don't retry database errors
        
        elif error_category == 'network_error':
            logger.warning(f"Network error for {file_path} - will retry")
            return self.recovery_mgr.should_retry(file_path, error)
        
        else:
            logger.warning(f"Unknown error for {file_path} - will retry")
            return self.recovery_mgr.should_retry(file_path, error)
    
    def get_error_stats(self) -> dict:
        """Get error statistics."""
        total_failures, unique_failed_files = self.recovery_mgr.get_failure_stats()
        return {
            'total_failures': total_failures,
            'unique_failed_files': unique_failed_files,
            'success_counts': dict(self.recovery_mgr.success_counts)
        }


# Global error handler instance
error_handler = AnalysisErrorHandler() 