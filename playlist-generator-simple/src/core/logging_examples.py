"""
Logging examples and templates for Playlist Generator.

This module provides practical examples of how to use the unified logging
system effectively throughout the application.
"""

import time
import traceback
from typing import Any, Dict, Optional
from functools import wraps

from .unified_logging import (
    get_logger, log_structured, log_api_call, log_resource_usage,
    log_performance, log_exceptions
)
from .logging_config import LoggingPatterns, StandardLogMessages, create_logger_name


class LoggingExamples:
    """
    Examples of proper logging usage patterns.
    
    These examples demonstrate best practices for different scenarios
    throughout the application.
    """
    
    def __init__(self, module_name: str):
        """Initialize with module-specific logger."""
        self.logger_name = create_logger_name(module_name)
        self.logger = get_logger(self.logger_name)
    
    def example_basic_logging(self):
        """Basic logging usage."""
        # Standard log levels
        self.logger.debug("Detailed debugging information")
        self.logger.info("General information")
        self.logger.warning("Warning about potential issues")
        self.logger.error("Error that needs attention")
        self.logger.critical("Critical error that may cause shutdown")
    
    def example_structured_logging(self):
        """Structured logging with additional context."""
        # Simple structured logging
        log_structured(
            'INFO',
            LoggingPatterns.COMPONENTS['AUDIO_ANALYZER'],
            'Processing audio file',
            file_path='/path/to/audio.mp3',
            file_size_mb=4.5,
            duration_seconds=180
        )
        
        # More complex structured data
        log_structured(
            'DEBUG',
            LoggingPatterns.COMPONENTS['DATABASE'],
            'Query executed',
            query_type='SELECT',
            table='tracks',
            execution_time_ms=45.2,
            rows_returned=127,
            cache_hit=False
        )
    
    def example_api_call_logging(self):
        """API call logging patterns."""
        # Successful API call
        log_api_call(
            api_name='MusicBrainz',
            operation='search',
            target='artist: The Beatles',
            success=True,
            duration=0.845,
            details='Found 15 recordings'
        )
        
        # Failed API call
        log_api_call(
            api_name='LastFM',
            operation='lookup',
            target='track: Unknown Song',
            success=False,
            duration=2.1,
            failure_type='no_data'
        )
        
        # Network error
        log_api_call(
            api_name='MusicBrainz',
            operation='lookup',
            target='recording: 12345',
            success=False,
            duration=30.0,
            failure_type='timeout'
        )
    
    def example_performance_logging(self):
        """Performance monitoring examples."""
        start_time = time.time()
        
        # Simulate some work
        time.sleep(0.1)
        
        duration = time.time() - start_time
        
        # Log performance metrics
        log_resource_usage(
            LoggingPatterns.COMPONENTS['AUDIO_ANALYZER'],
            cpu_percent=45.2,
            memory_mb=234.5,
            processing_time_seconds=duration,
            files_processed=10
        )
    
    def example_session_logging(self):
        """Session and operation lifecycle logging."""
        # Start operation
        operation = "audio_analysis"
        target = "/path/to/audio/files"
        
        start_msg = StandardLogMessages.operation_start(operation, target)
        self.logger.info(start_msg)
        
        start_time = time.time()
        
        try:
            # Simulate work
            time.sleep(0.05)
            
            # Success
            duration = time.time() - start_time
            success_msg = StandardLogMessages.operation_complete(operation, target, duration)
            self.logger.info(success_msg)
            
        except Exception as e:
            error_msg = StandardLogMessages.operation_failed(operation, target, str(e))
            self.logger.error(error_msg)
    
    def example_error_handling(self):
        """Error handling and exception logging patterns."""
        try:
            # Simulate an operation that might fail
            result = self._risky_operation()
            self.logger.info("Operation completed successfully")
            return result
            
        except ValueError as e:
            # Handle expected errors with appropriate level
            self.logger.warning(f"Invalid input provided: {e}")
            return None
            
        except ConnectionError as e:
            # Handle network errors
            self.logger.error(f"Network connection failed: {e}")
            return None
            
        except Exception as e:
            # Handle unexpected errors with full traceback
            self.logger.exception(f"Unexpected error in operation: {e}")
            return None
    
    def _risky_operation(self):
        """Simulate an operation that might fail."""
        import random
        if random.random() < 0.3:
            raise ValueError("Invalid parameter")
        elif random.random() < 0.6:
            raise ConnectionError("Network timeout")
        return "success"
    
    def example_cache_logging(self):
        """Cache operation logging."""
        cache_key = "audio_features:track123"
        
        # Cache hit
        log_structured(
            'DEBUG',
            LoggingPatterns.COMPONENTS['CACHE'],
            StandardLogMessages.cache_operation('get', cache_key, hit=True),
            cache_key=cache_key,
            cache_size_mb=45.2
        )
        
        # Cache miss
        log_structured(
            'DEBUG',
            LoggingPatterns.COMPONENTS['CACHE'],
            StandardLogMessages.cache_operation('get', cache_key, hit=False),
            cache_key=cache_key,
            cache_size_mb=45.2
        )
        
        # Cache update
        log_structured(
            'DEBUG',
            LoggingPatterns.COMPONENTS['CACHE'],
            StandardLogMessages.cache_operation('set', cache_key),
            cache_key=cache_key,
            data_size_bytes=1024,
            ttl_seconds=3600
        )
    
    def example_database_logging(self):
        """Database operation logging."""
        # Successful query
        log_structured(
            'DEBUG',
            LoggingPatterns.COMPONENTS['DATABASE'],
            StandardLogMessages.database_operation('SELECT', 'tracks', True, 150),
            table='tracks',
            query_time_ms=23.4,
            connection_pool_active=5
        )
        
        # Failed query
        log_structured(
            'ERROR',
            LoggingPatterns.COMPONENTS['DATABASE'],
            StandardLogMessages.database_operation('INSERT', 'playlists', False),
            table='playlists',
            error_code='UNIQUE_CONSTRAINT_VIOLATION',
            query_time_ms=12.1
        )


class DatabaseOperationLogger:
    """Example class showing integrated logging for database operations."""
    
    def __init__(self, module_name: str):
        self.logger = get_logger(create_logger_name(module_name, 'database'))
    
    @log_performance()
    @log_exceptions()
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> Optional[list]:
        """
        Execute database query with comprehensive logging.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results or None if failed
        """
        log_structured(
            'DEBUG',
            LoggingPatterns.COMPONENTS['DATABASE'],
            'Executing query',
            query_type=query.split()[0].upper(),
            has_params=params is not None,
            param_count=len(params) if params else 0
        )
        
        start_time = time.time()
        
        try:
            # Simulate database operation
            time.sleep(0.01)  # Simulate query time
            
            # Simulate results
            results = [{'id': 1, 'name': 'test'}]
            
            execution_time = time.time() - start_time
            
            log_structured(
                'DEBUG',
                LoggingPatterns.COMPONENTS['DATABASE'],
                'Query completed',
                execution_time_ms=execution_time * 1000,
                rows_returned=len(results),
                success=True
            )
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            log_structured(
                'ERROR',
                LoggingPatterns.COMPONENTS['DATABASE'],
                'Query failed',
                execution_time_ms=execution_time * 1000,
                error=str(e),
                success=False
            )
            
            raise


class AudioAnalyzerLogger:
    """Example class for audio processing logging."""
    
    def __init__(self, module_name: str):
        self.logger = get_logger(create_logger_name(module_name, 'analyzer'))
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process audio file with comprehensive logging.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Analysis results
        """
        # Log operation start
        self.logger.info(
            StandardLogMessages.operation_start('audio_analysis', file_path)
        )
        
        start_time = time.time()
        
        try:
            # Simulate file size check
            file_size_mb = 4.2
            
            log_structured(
                'DEBUG',
                LoggingPatterns.COMPONENTS['AUDIO_ANALYZER'],
                'File validated',
                file_path=file_path,
                file_size_mb=file_size_mb,
                format='mp3'
            )
            
            # Simulate processing stages
            self._log_processing_stage('loading', file_path)
            time.sleep(0.01)
            
            self._log_processing_stage('feature_extraction', file_path)
            time.sleep(0.02)
            
            self._log_processing_stage('analysis_complete', file_path)
            
            duration = time.time() - start_time
            
            results = {
                'tempo': 120.5,
                'key': 'C major',
                'energy': 0.75
            }
            
            # Log successful completion
            log_structured(
                'INFO',
                LoggingPatterns.COMPONENTS['AUDIO_ANALYZER'],
                StandardLogMessages.operation_complete('audio_analysis', file_path, duration),
                file_path=file_path,
                processing_time_seconds=duration,
                features_extracted=len(results),
                success=True
            )
            
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log failure with context
            log_structured(
                'ERROR',
                LoggingPatterns.COMPONENTS['AUDIO_ANALYZER'],
                StandardLogMessages.operation_failed('audio_analysis', file_path, str(e)),
                file_path=file_path,
                processing_time_seconds=duration,
                error_type=type(e).__name__,
                success=False
            )
            
            raise
    
    def _log_processing_stage(self, stage: str, file_path: str):
        """Log processing stage."""
        log_structured(
            'DEBUG',
            LoggingPatterns.COMPONENTS['AUDIO_ANALYZER'],
            f'Processing stage: {stage}',
            stage=stage,
            file_path=file_path
        )


# Decorator examples for common patterns
def log_method_calls(component: str = None):
    """Decorator to log method entry and exit."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logger = get_logger(create_logger_name(self.__class__.__module__))
            
            # Determine component name
            comp_name = component or self.__class__.__name__
            
            # Log method entry
            log_structured(
                'DEBUG',
                comp_name,
                f'Entering {func.__name__}',
                method=func.__name__,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
            
            start_time = time.time()
            
            try:
                result = func(self, *args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful completion
                log_structured(
                    'DEBUG',
                    comp_name,
                    f'Completed {func.__name__}',
                    method=func.__name__,
                    duration_seconds=duration,
                    success=True
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log failure
                log_structured(
                    'ERROR',
                    comp_name,
                    f'Failed {func.__name__}',
                    method=func.__name__,
                    duration_seconds=duration,
                    error=str(e),
                    success=False
                )
                
                raise
        
        return wrapper
    return decorator


def log_api_method(api_name: str):
    """Decorator for API method calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation = func.__name__
            
            # Extract target from args/kwargs if possible
            target = kwargs.get('target') or (args[0] if args else 'unknown')
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                log_api_call(
                    api_name=api_name,
                    operation=operation,
                    target=str(target),
                    success=True,
                    duration=duration
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Determine failure type
                failure_type = 'network' if 'connection' in str(e).lower() else 'error'
                if 'not found' in str(e).lower() or 'no data' in str(e).lower():
                    failure_type = 'no_data'
                
                log_api_call(
                    api_name=api_name,
                    operation=operation,
                    target=str(target),
                    success=False,
                    duration=duration,
                    failure_type=failure_type
                )
                
                raise
        
        return wrapper
    return decorator


# Context manager for operation logging
class LoggedOperation:
    """Context manager for logging operations with automatic timing."""
    
    def __init__(self, operation_name: str, target: str = None, component: str = None, logger_name: str = None):
        self.operation_name = operation_name
        self.target = target
        self.component = component or 'System'
        self.logger = get_logger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        
        message = StandardLogMessages.operation_start(self.operation_name, self.target)
        log_structured(
            'INFO',
            self.component,
            message,
            operation=self.operation_name,
            target=self.target
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            # Success
            message = StandardLogMessages.operation_complete(self.operation_name, self.target, duration)
            log_structured(
                'INFO',
                self.component,
                message,
                operation=self.operation_name,
                target=self.target,
                duration_seconds=duration,
                success=True
            )
        else:
            # Failure
            message = StandardLogMessages.operation_failed(self.operation_name, self.target, str(exc_val))
            log_structured(
                'ERROR',
                self.component,
                message,
                operation=self.operation_name,
                target=self.target,
                duration_seconds=duration,
                error=str(exc_val),
                error_type=exc_type.__name__,
                success=False
            )
        
        # Don't suppress exceptions
        return False


# Example usage functions
def demonstrate_logging_patterns():
    """Demonstrate various logging patterns."""
    examples = LoggingExamples(__name__)
    
    print("=== Basic Logging ===")
    examples.example_basic_logging()
    
    print("\n=== Structured Logging ===")
    examples.example_structured_logging()
    
    print("\n=== API Call Logging ===")
    examples.example_api_call_logging()
    
    print("\n=== Performance Logging ===")
    examples.example_performance_logging()
    
    print("\n=== Session Logging ===")
    examples.example_session_logging()
    
    print("\n=== Error Handling ===")
    examples.example_error_handling()
    
    print("\n=== Cache Logging ===")
    examples.example_cache_logging()
    
    print("\n=== Database Logging ===")
    examples.example_database_logging()


def demonstrate_context_manager():
    """Demonstrate the LoggedOperation context manager."""
    # Simple operation
    with LoggedOperation("file_processing", "/path/to/file.mp3"):
        time.sleep(0.1)  # Simulate work
    
    # Operation with error
    try:
        with LoggedOperation("risky_operation", "dangerous_target", "TestComponent"):
            time.sleep(0.05)
            raise ValueError("Something went wrong")
    except ValueError:
        pass  # Expected for demo


if __name__ == "__main__":
    # Setup logging for demonstration
    from .unified_logging import setup_logging
    from .logging_config import get_logging_config
    
    setup_logging(get_logging_config('development'))
    
    demonstrate_logging_patterns()
    demonstrate_context_manager()