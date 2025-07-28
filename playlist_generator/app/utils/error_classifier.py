import logging
import time
import os
import sqlite3
import psutil
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Classification of different error types."""
    TIMEOUT = "timeout"
    MEMORY = "memory"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    NETWORK = "network"
    AUDIO_PROCESSING = "audio_processing"
    PERMISSION = "permission"
    CORRUPT_FILE = "corrupt_file"
    UNSUPPORTED_FORMAT = "unsupported_format"
    SYSTEM_RESOURCE = "system_resource"
    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    """Information about a classified error."""
    error_type: ErrorType
    severity: ErrorSeverity
    error_message: str
    original_exception: Exception
    context: Dict[str, Any]
    recovery_strategy: str
    retryable: bool
    timestamp: float

class ErrorClassifier:
    """
    Advanced error classification system that categorizes errors and suggests recovery strategies.
    
    Features:
    - Automatic error classification
    - Severity assessment
    - Recovery strategy recommendation
    - Error pattern recognition
    - Context-aware classification
    """
    
    def __init__(self):
        # Error patterns for classification
        self.error_patterns = {
            ErrorType.TIMEOUT: [
                "timeout", "timed out", "time out", "deadline exceeded",
                "operation timed out", "request timed out"
            ],
            ErrorType.MEMORY: [
                "memory", "out of memory", "memory error", "memory allocation",
                "insufficient memory", "memory limit", "memory leak"
            ],
            ErrorType.FILE_SYSTEM: [
                "file not found", "no such file", "file system", "disk full",
                "permission denied", "access denied", "file exists",
                "directory not found", "path not found"
            ],
            ErrorType.DATABASE: [
                "database", "sqlite", "database is locked", "database error",
                "connection", "transaction", "constraint", "foreign key"
            ],
            ErrorType.NETWORK: [
                "network", "connection", "http", "request", "url",
                "dns", "timeout", "connection refused", "host unreachable"
            ],
            ErrorType.AUDIO_PROCESSING: [
                "audio", "essentia", "librosa", "soundfile", "audio format",
                "codec", "decoder", "encoder", "sample rate", "bit depth"
            ],
            ErrorType.PERMISSION: [
                "permission", "access denied", "forbidden", "unauthorized",
                "insufficient privileges", "read only", "write protected"
            ],
            ErrorType.CORRUPT_FILE: [
                "corrupt", "invalid", "malformed", "truncated", "damaged",
                "checksum", "crc", "integrity", "parse error"
            ],
            ErrorType.UNSUPPORTED_FORMAT: [
                "unsupported", "format not supported", "codec not found",
                "unknown format", "invalid format", "unsupported codec"
            ],
            ErrorType.SYSTEM_RESOURCE: [
                "resource", "system resource", "too many open files",
                "file descriptor", "process limit", "ulimit"
            ]
        }
        
        # Recovery strategies for each error type
        self.recovery_strategies = {
            ErrorType.TIMEOUT: {
                'strategy': 'increase_timeout',
                'retryable': True,
                'severity': ErrorSeverity.MEDIUM,
                'description': 'Increase timeout and retry'
            },
            ErrorType.MEMORY: {
                'strategy': 'reduce_memory_usage',
                'retryable': True,
                'severity': ErrorSeverity.HIGH,
                'description': 'Free memory and retry with reduced features'
            },
            ErrorType.FILE_SYSTEM: {
                'strategy': 'file_system_check',
                'retryable': False,
                'severity': ErrorSeverity.HIGH,
                'description': 'Check file system and permissions'
            },
            ErrorType.DATABASE: {
                'strategy': 'database_recovery',
                'retryable': True,
                'severity': ErrorSeverity.MEDIUM,
                'description': 'Reset database connection and retry'
            },
            ErrorType.NETWORK: {
                'strategy': 'network_retry',
                'retryable': True,
                'severity': ErrorSeverity.MEDIUM,
                'description': 'Retry with exponential backoff'
            },
            ErrorType.AUDIO_PROCESSING: {
                'strategy': 'audio_fallback',
                'retryable': True,
                'severity': ErrorSeverity.MEDIUM,
                'description': 'Try alternative audio processing method'
            },
            ErrorType.PERMISSION: {
                'strategy': 'permission_check',
                'retryable': False,
                'severity': ErrorSeverity.HIGH,
                'description': 'Check and fix file permissions'
            },
            ErrorType.CORRUPT_FILE: {
                'strategy': 'skip_file',
                'retryable': False,
                'severity': ErrorSeverity.MEDIUM,
                'description': 'Skip corrupted file and continue'
            },
            ErrorType.UNSUPPORTED_FORMAT: {
                'strategy': 'format_conversion',
                'retryable': True,
                'severity': ErrorSeverity.LOW,
                'description': 'Convert to supported format and retry'
            },
            ErrorType.SYSTEM_RESOURCE: {
                'strategy': 'resource_cleanup',
                'retryable': True,
                'severity': ErrorSeverity.HIGH,
                'description': 'Clean up system resources and retry'
            },
            ErrorType.UNKNOWN: {
                'strategy': 'generic_retry',
                'retryable': True,
                'severity': ErrorSeverity.MEDIUM,
                'description': 'Generic retry with backoff'
            }
        }
        
        # Error history for pattern recognition
        self.error_history = []
        self.max_history_size = 1000
    
    def classify_error(self, 
                      exception: Exception, 
                      context: Dict[str, Any] = None) -> ErrorInfo:
        """
        Classify an error and provide recovery information.
        
        Args:
            exception: The exception that occurred
            context: Additional context about the error
            
        Returns:
            ErrorInfo object with classification and recovery strategy
        """
        error_message = str(exception).lower()
        exception_type = type(exception).__name__
        
        # Determine error type based on patterns
        error_type = self._determine_error_type(error_message, exception_type, context)
        
        # Get recovery strategy
        strategy_info = self.recovery_strategies[error_type]
        
        # Assess severity
        severity = self._assess_severity(error_type, context)
        
        # Create error info
        error_info = ErrorInfo(
            error_type=error_type,
            severity=severity,
            error_message=error_message,
            original_exception=exception,
            context=context or {},
            recovery_strategy=strategy_info['strategy'],
            retryable=strategy_info['retryable'],
            timestamp=time.time()
        )
        
        # Record error for pattern analysis
        self._record_error(error_info)
        
        logger.debug(f"Classified error: {error_type.value} (severity: {severity.value}, "
                    f"retryable: {error_info.retryable}, strategy: {error_info.recovery_strategy})")
        
        return error_info
    
    def _determine_error_type(self, 
                             error_message: str, 
                             exception_type: str, 
                             context: Dict[str, Any] = None) -> ErrorType:
        """Determine the type of error based on message and context."""
        
        # Check for specific exception types first
        if exception_type in ['TimeoutException', 'TimeoutError']:
            return ErrorType.TIMEOUT
        elif exception_type in ['MemoryError', 'OSError'] and 'memory' in error_message:
            return ErrorType.MEMORY
        elif exception_type in ['FileNotFoundError', 'PermissionError']:
            return ErrorType.FILE_SYSTEM
        elif exception_type in ['sqlite3.Error', 'sqlite3.OperationalError']:
            return ErrorType.DATABASE
        elif exception_type in ['requests.RequestException', 'urllib.error.URLError']:
            return ErrorType.NETWORK
        elif exception_type in ['ValueError', 'RuntimeError'] and any(
            audio_term in error_message for audio_term in ['audio', 'essentia', 'librosa']
        ):
            return ErrorType.AUDIO_PROCESSING
        
        # Check error patterns
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type
        
        # Context-based classification
        if context:
            if context.get('operation') == 'database':
                return ErrorType.DATABASE
            elif context.get('operation') == 'network':
                return ErrorType.NETWORK
            elif context.get('operation') == 'audio_processing':
                return ErrorType.AUDIO_PROCESSING
            elif context.get('file_size_mb', 0) > 100 and 'memory' in error_message:
                return ErrorType.MEMORY
        
        return ErrorType.UNKNOWN
    
    def _assess_severity(self, error_type: ErrorType, context: Dict[str, Any] = None) -> ErrorSeverity:
        """Assess the severity of an error based on type and context."""
        
        # Base severity from error type
        base_severity = self.recovery_strategies[error_type]['severity']
        
        # Adjust based on context
        if context:
            # High severity if it's a critical file
            if context.get('is_critical_file', False):
                return ErrorSeverity.CRITICAL
            
            # High severity if system resources are low
            if self._is_system_stressed():
                if error_type in [ErrorType.MEMORY, ErrorType.SYSTEM_RESOURCE]:
                    return ErrorSeverity.CRITICAL
            
            # Medium severity for large files
            if context.get('file_size_mb', 0) > 50:
                if error_type in [ErrorType.TIMEOUT, ErrorType.MEMORY]:
                    return ErrorSeverity.HIGH
        
        return base_severity
    
    def _is_system_stressed(self) -> bool:
        """Check if the system is under stress."""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            return memory.percent > 90 or cpu > 90
        except Exception:
            return False
    
    def _record_error(self, error_info: ErrorInfo):
        """Record error for pattern analysis."""
        self.error_history.append(error_info)
        
        # Keep history size manageable
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def get_error_patterns(self, error_type: ErrorType = None) -> Dict[str, Any]:
        """Get error pattern statistics."""
        if not self.error_history:
            return {}
        
        # Filter by error type if specified
        history = self.error_history
        if error_type:
            history = [error for error in history if error.error_type == error_type]
        
        if not history:
            return {}
        
        # Calculate patterns
        total_errors = len(history)
        recent_errors = [error for error in history 
                        if time.time() - error.timestamp < 3600]  # Last hour
        
        patterns = {
            'total_errors': total_errors,
            'recent_errors': len(recent_errors),
            'error_types': {},
            'severity_distribution': {},
            'retryable_rate': sum(1 for e in history if e.retryable) / total_errors,
            'avg_time_between_errors': self._calculate_avg_time_between_errors(history)
        }
        
        # Error type distribution
        for error in history:
            error_type = error.error_type.value
            patterns['error_types'][error_type] = patterns['error_types'].get(error_type, 0) + 1
        
        # Severity distribution
        for error in history:
            severity = error.severity.value
            patterns['severity_distribution'][severity] = patterns['severity_distribution'].get(severity, 0) + 1
        
        return patterns
    
    def _calculate_avg_time_between_errors(self, history: List[ErrorInfo]) -> float:
        """Calculate average time between errors."""
        if len(history) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x.timestamp)
        
        # Calculate time differences
        time_diffs = []
        for i in range(1, len(sorted_history)):
            diff = sorted_history[i].timestamp - sorted_history[i-1].timestamp
            time_diffs.append(diff)
        
        return sum(time_diffs) / len(time_diffs) if time_diffs else 0.0
    
    def get_recovery_recommendation(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Get detailed recovery recommendation for an error."""
        strategy = error_info.recovery_strategy
        
        recommendations = {
            'increase_timeout': {
                'action': 'Increase timeout by 50% and retry',
                'parameters': {'timeout_multiplier': 1.5},
                'max_retries': 3
            },
            'reduce_memory_usage': {
                'action': 'Free memory and retry with minimal features',
                'parameters': {'skip_memory_intensive': True, 'gc_collect': True},
                'max_retries': 2
            },
            'file_system_check': {
                'action': 'Check file permissions and disk space',
                'parameters': {'check_permissions': True, 'check_disk_space': True},
                'max_retries': 0
            },
            'database_recovery': {
                'action': 'Reset database connection and retry',
                'parameters': {'reset_connection': True, 'clear_cache': True},
                'max_retries': 3
            },
            'network_retry': {
                'action': 'Retry with exponential backoff',
                'parameters': {'exponential_backoff': True, 'max_delay': 60},
                'max_retries': 5
            },
            'audio_fallback': {
                'action': 'Try alternative audio processing method',
                'parameters': {'use_fallback_method': True},
                'max_retries': 2
            },
            'permission_check': {
                'action': 'Check and fix file permissions',
                'parameters': {'check_permissions': True},
                'max_retries': 0
            },
            'skip_file': {
                'action': 'Skip file and mark as failed',
                'parameters': {'skip_file': True, 'mark_failed': True},
                'max_retries': 0
            },
            'format_conversion': {
                'action': 'Convert to supported format and retry',
                'parameters': {'convert_format': True},
                'max_retries': 1
            },
            'resource_cleanup': {
                'action': 'Clean up system resources and retry',
                'parameters': {'gc_collect': True, 'close_unused_connections': True},
                'max_retries': 2
            },
            'generic_retry': {
                'action': 'Generic retry with exponential backoff',
                'parameters': {'exponential_backoff': True},
                'max_retries': 3
            }
        }
        
        return recommendations.get(strategy, recommendations['generic_retry'])
    
    def should_retry(self, error_info: ErrorInfo, retry_count: int = 0) -> bool:
        """Determine if an error should be retried."""
        if not error_info.retryable:
            return False
        
        recommendation = self.get_recovery_recommendation(error_info)
        max_retries = recommendation.get('max_retries', 3)
        
        return retry_count < max_retries
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors."""
        if not self.error_history:
            return {'total_errors': 0}
        
        total_errors = len(self.error_history)
        error_types = {}
        severity_counts = {}
        
        for error in self.error_history:
            # Count error types
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count severity levels
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'severity_distribution': severity_counts,
            'retryable_errors': sum(1 for e in self.error_history if e.retryable),
            'non_retryable_errors': sum(1 for e in self.error_history if not e.retryable),
            'most_common_error': max(error_types.items(), key=lambda x: x[1]) if error_types else None,
            'most_severe_errors': severity_counts.get('critical', 0) + severity_counts.get('high', 0)
        }

# Global error classifier instance
_error_classifier = None

def get_error_classifier() -> ErrorClassifier:
    """Get the global error classifier instance."""
    global _error_classifier
    if _error_classifier is None:
        _error_classifier = ErrorClassifier()
    return _error_classifier

def classify_error(exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
    """Convenience function to classify an error."""
    return get_error_classifier().classify_error(exception, context) 