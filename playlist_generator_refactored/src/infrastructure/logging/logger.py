"""
Main logging setup and management for the Playlista application.
"""

import logging
import logging.handlers
import sys
import threading
import time
import os
import signal
import uuid
from typing import Optional, Dict, Any
from pathlib import Path

from shared.config import get_config
from shared.exceptions import ConfigurationError

# Global state
_log_level_monitor_thread: Optional[threading.Thread] = None
_log_setup_complete = False
_correlation_id_context = threading.local()


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def filter(self, record):
        record.correlation_id = getattr(_correlation_id_context, 'correlation_id', None)
        return True


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return getattr(_correlation_id_context, 'correlation_id', None)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set the correlation ID for the current thread."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _correlation_id_context.correlation_id = correlation_id
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current thread."""
    if hasattr(_correlation_id_context, 'correlation_id'):
        delattr(_correlation_id_context, 'correlation_id')


def setup_logging(
    config: Optional[Any] = None,
    log_file: Optional[Path] = None,
    correlation_id: Optional[str] = None
) -> logging.Logger:
    """
    Setup application logging with structured output.
    
    Args:
        config: Application configuration
        log_file: Optional log file path
        correlation_id: Optional correlation ID for request tracking
        
    Returns:
        Configured logger instance
    """
    global _log_setup_complete
    
    if _log_setup_complete:
        return logging.getLogger('playlista')
    
    # Load configuration if not provided
    if config is None:
        config = get_config()
    
    # Set correlation ID if provided
    if correlation_id:
        set_correlation_id(correlation_id)
    
    # Create main logger
    logger = logging.getLogger('playlista')
    
    # Get log level from config (handle both direct config and nested logging config)
    if hasattr(config, 'logging'):
        log_level = config.logging.level
        file_logging = config.logging.file_logging
        console_logging = config.logging.console_logging
        log_dir = config.logging.log_dir
        log_file_prefix = config.logging.log_file_prefix
        colored_output = config.logging.colored_output
        verbose_output = config.logging.verbose_output
        show_progress = config.logging.show_progress
        log_memory_usage = config.logging.log_memory_usage
        log_performance = config.logging.log_performance
        max_log_files = config.logging.max_log_files
        log_file_size_mb = config.logging.log_file_size_mb
    else:
        # Fallback to direct config attributes
        log_level = getattr(config, 'level', 'DEBUG')
        file_logging = getattr(config, 'file_logging', True)
        console_logging = getattr(config, 'console_logging', True)
        log_dir = getattr(config, 'log_dir', Path('/app/logs'))
        log_file_prefix = getattr(config, 'log_file_prefix', 'playlista')
        colored_output = getattr(config, 'colored_output', True)
        verbose_output = getattr(config, 'verbose_output', True)
        show_progress = getattr(config, 'show_progress', True)
        log_memory_usage = getattr(config, 'log_memory_usage', True)
        log_performance = getattr(config, 'log_performance', True)
        max_log_files = getattr(config, 'max_log_files', 10)
        log_file_size_mb = getattr(config, 'log_file_size_mb', 50)
    
    logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    logger.addFilter(correlation_filter)
    
    # Add structured log filters (temporarily disabled for Python 3.7 compatibility)
    # from infrastructure.logging.processors import (
    #     get_structured_processor,
    #     get_database_processor,
    #     get_analysis_processor,
    #     get_performance_processor
    # )
    
    # Add filters to logger (temporarily disabled)
    # logger.addFilter(get_structured_processor())
    # logger.addFilter(get_database_processor())
    # logger.addFilter(get_analysis_processor())
    # logger.addFilter(get_performance_processor())
    
    # Setup formatters
    from infrastructure.logging.formatters import StructuredFormatter, ColoredFormatter
    
    # Console handler (if enabled) - minimal output for user
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        # Set encoding to handle Unicode characters
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
        # Use simple formatter for console to avoid JSON spam
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        # Set console to only show WARNING and above by default
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)
    
    # File handler (if enabled) - full JSON logging
    if file_logging:
        if log_file is None:
            log_file = log_dir / f"{log_file_prefix}.log"
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler with configurable size
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_file_size_mb * 1024 * 1024,  # Configurable size
            backupCount=max_log_files
        )
        # Use JSON formatter for file logging
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        # File handler captures all levels
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    
    # Setup TensorFlow and Essentia logging
    _setup_external_logging(config)
    
    # Setup signal handlers for runtime log level changes
    setup_signal_handlers()
    
    # Start log level monitor
    setup_log_level_monitor()
    
    _log_setup_complete = True
    
    # Log memory usage if enabled
    if log_memory_usage:
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            logger.info(f"Memory usage: {memory_mb:.1f}MB / {memory_info.total / (1024 * 1024 * 1024):.1f}GB")
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
    
    # Log initialization only once
    logger.info("Logging system initialized", extra={
        'log_level': log_level,
        'file_logging': file_logging,
        'console_logging': console_logging,
        'log_file': str(log_file) if log_file else None,
        'verbose_output': verbose_output,
        'show_progress': show_progress,
        'log_memory_usage': log_memory_usage,
        'log_performance': log_performance
    })
    
    return logger


def _setup_external_logging(config: Any) -> None:
    """Setup logging for external libraries (TensorFlow, Essentia)."""
    try:
        import tensorflow as tf
        
        # Configure TensorFlow logging - suppress all warnings
        tf_logger = tf.get_logger()
        tf_logger.handlers.clear()
        tf_logger.setLevel(logging.ERROR)  # Only show errors
        
        # Suppress TensorFlow warnings more aggressively
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF warnings
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage
        
        # Get TensorFlow log level from config
        if hasattr(config, 'logging'):
            tf_log_level = getattr(logging, config.logging.tensorflow_log_level.upper(), logging.ERROR)
            file_logging = config.logging.file_logging
            log_dir = config.logging.log_dir
        else:
            tf_log_level = getattr(logging, getattr(config, 'tensorflow_log_level', '2').upper(), logging.ERROR)
            file_logging = getattr(config, 'file_logging', True)
            log_dir = getattr(config, 'log_dir', Path('/app/logs'))
        
        tf_logger.setLevel(tf_log_level)
        
        # Add file handler for TensorFlow logs
        if file_logging:
            tf_log_file = log_dir / "tensorflow.log"
            tf_handler = logging.handlers.RotatingFileHandler(
                tf_log_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3
            )
            tf_formatter = logging.Formatter(
                '%(asctime)s [TF] %(levelname)s - %(message)s'
            )
            tf_handler.setFormatter(tf_formatter)
            tf_logger.addHandler(tf_handler)
        
    except ImportError:
        pass  # TensorFlow not available
    
    try:
        import essentia
        
        # Configure Essentia logging
        essentia.log.infoActive = True
        essentia.log.warningActive = True
        essentia.log.errorActive = True
        
        # Get Essentia log level from config
        if hasattr(config, 'logging'):
            essentia_log_level = config.logging.essentia_logging_level.upper()
        else:
            essentia_log_level = getattr(config, 'essentia_logging_level', 'error').upper()
        
        if essentia_log_level == 'ERROR':
            essentia.log.infoActive = False
            essentia.log.warningActive = False
        elif essentia_log_level == 'WARNING':
            essentia.log.infoActive = False
        
    except ImportError:
        pass  # Essentia not available


def get_logger(name: str = 'playlista') -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def change_log_level(new_level: str) -> bool:
    """
    Change log level at runtime.
    
    Args:
        new_level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        level = getattr(logging, new_level.upper(), logging.INFO)
        
        # Update root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Update all other loggers
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            logger.setLevel(level)
        
        # Update environment variable
        os.environ['LOG_LEVEL'] = new_level.upper()
        
        # Log the change
        root_logger.info(f"Log level changed to: {new_level.upper()}")
        return True
        
    except Exception as e:
        print(f"Failed to change log level: {e}")
        return False


def setup_log_level_monitor() -> None:
    """Start background thread that monitors LOG_LEVEL environment variable changes."""
    global _log_level_monitor_thread
    
    if _log_level_monitor_thread is not None and _log_level_monitor_thread.is_alive():
        return
    
    def monitor_log_level_changes():
        """Background thread that monitors LOG_LEVEL environment variable changes."""
        last_level = os.getenv('LOG_LEVEL', 'INFO')
        
        while True:
            try:
                # Get current level from environment
                current_level = os.getenv('LOG_LEVEL', 'INFO')
                
                if current_level != last_level:
                    logger = get_logger('playlista.logging')
                    logger.info(f"Environment LOG_LEVEL changed from {last_level} to {current_level}")
                    
                    if change_log_level(current_level):
                        logger.info(f"Log level updated to: {current_level}")
                    else:
                        logger.error(f"Failed to update log level to: {current_level}")
                    
                    last_level = current_level
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"Error in log level monitor: {e}")
                time.sleep(5)  # Wait longer on error
    
    _log_level_monitor_thread = threading.Thread(
        target=monitor_log_level_changes,
        daemon=True,
        name="LogLevelMonitor"
    )
    _log_level_monitor_thread.start()


def setup_signal_handlers() -> None:
    """Setup signal handlers for runtime log level control."""
    
    def cycle_log_level_handler(signum, frame):
        """Cycle through log levels on signal."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        current_level = os.getenv('LOG_LEVEL', 'INFO')
        
        try:
            current_index = levels.index(current_level)
            next_index = (current_index + 1) % len(levels)
            new_level = levels[next_index]
            
            if change_log_level(new_level):
                print(f"\n🔄 Log level cycled to: {new_level}")
            else:
                print(f"\n❌ Failed to change log level")
        except Exception as e:
            print(f"\n❌ Error changing log level: {e}")
    
    def set_log_level_handler(level: str):
        """Set specific log level on signal."""
        def handler(signum, frame):
            if change_log_level(level):
                print(f"\n🔄 Log level set to: {level}")
            else:
                print(f"\n❌ Failed to set log level to: {level}")
        return handler
    
    try:
        # Use SIGUSR1 to cycle through log levels (Unix/Linux)
        signal.signal(signal.SIGUSR1, cycle_log_level_handler)
        print("📝 Log level control: Send SIGUSR1 signal to cycle through log levels")
        print("   Example: docker compose exec playlista kill -SIGUSR1 1")
    except (AttributeError, OSError):
        # SIGUSR1 not available on Windows
        pass
    
    try:
        # Use SIGUSR2 for specific log levels (Unix/Linux)
        signal.signal(signal.SIGUSR2, set_log_level_handler('DEBUG'))
        print("📝 Debug mode: Send SIGUSR2 signal to set DEBUG level")
    except (AttributeError, OSError):
        # SIGUSR2 not available on Windows
        pass


def log_function_call(func):
    """Decorator to log function calls with correlation ID."""
    def wrapper(*args, **kwargs):
        logger = get_logger('playlista.function')
        correlation_id = get_correlation_id()
        
        logger.debug(f"Calling {func.__name__}", extra={
            'function_name': func.__name__,
            'module_name': func.__module__,
            'correlation_id': correlation_id
        })
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully", extra={
                'function_name': func.__name__,
                'correlation_id': correlation_id
            })
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}", extra={
                'function_name': func.__name__,
                'correlation_id': correlation_id,
                'error': str(e)
            })
            raise
    
    return wrapper 