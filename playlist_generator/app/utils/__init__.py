"""Utility modules for the playlist generator."""

from .cli import *
from .enrichment import *
from .logging_setup import *
from .memory_monitor import *
from .path_converter import *
from .path_utils import *
from .timeout_manager import *
from .error_recovery import *
from .progress_tracker import *
from .circuit_breaker import *
from .adaptive_timeout import *
from .error_classifier import *
from .smart_retry import *

__all__ = [
    # Existing utilities
    'setup_logging',
    'get_logger',
    'MemoryMonitor',
    'get_optimal_worker_count',
    'get_memory_aware_batch_size',
    'check_total_python_rss_limit',
    'convert_path',
    'normalize_path',
    'enrich_metadata',
    
    # New robustness utilities
    'CircuitBreaker',
    'CircuitBreakerOpenError',
    'circuit_breaker',
    'get_circuit_breaker',
    'register_circuit_breaker',
    'get_all_circuit_breakers',
    'reset_all_circuit_breakers',
    
    'AdaptiveTimeoutManager',
    'TimeoutStrategy',
    'TimeoutConfig',
    'get_timeout_manager',
    'calculate_timeout',
    
    'ErrorClassifier',
    'ErrorType',
    'ErrorSeverity',
    'ErrorInfo',
    'get_error_classifier',
    'classify_error',
    
    'SmartRetryManager',
    'RetryStrategy',
    'RetryResult',
    'smart_retry',
    'get_retry_manager',
    'retry_with_strategy'
]
