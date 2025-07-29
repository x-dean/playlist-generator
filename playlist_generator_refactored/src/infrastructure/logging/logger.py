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
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    logger.addFilter(correlation_filter)
    
    # Setup formatters
    from infrastructure.logging.formatters import StructuredFormatter, ColoredFormatter
    
    # Console handler (if enabled)
    if config.logging.console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        # Set encoding to handle Unicode characters
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
        console_formatter = ColoredFormatter(
            '%(log_color)s%(asctime)s [%(correlation_id)s] %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if config.logging.file_logging:
        if log_file is None:
            log_file = config.logging.log_dir / f"{config.logging.log_file_prefix}.log"
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Setup TensorFlow and Essentia logging
    _setup_external_logging(config)
    
    # Setup signal handlers for runtime log level changes
    setup_signal_handlers()
    
    # Start log level monitor
    setup_log_level_monitor()
    
    _log_setup_complete = True
    
    logger.info("Logging system initialized", extra={
        'log_level': config.logging.level,
        'file_logging': config.logging.file_logging,
        'console_logging': config.logging.console_logging,
        'log_file': str(log_file) if log_file else None
    })
    
    return logger


def _setup_external_logging(config: Any) -> None:
    """Setup logging for external libraries (TensorFlow, Essentia)."""
    try:
        import tensorflow as tf
        
        # Configure TensorFlow logging
        tf_logger = tf.get_logger()
        tf_logger.handlers.clear()
        
        # Set TensorFlow log level
        tf_log_level = getattr(logging, config.logging.tensorflow_log_level.upper(), logging.ERROR)
        tf_logger.setLevel(tf_log_level)
        
        # Add file handler for TensorFlow logs
        if config.logging.file_logging:
            tf_log_file = config.logging.log_dir / "tensorflow.log"
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
        
        # Set Essentia log level
        essentia_log_level = config.logging.essentia_logging_level.upper()
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
            'function': func.__name__,
            'module': func.__module__,
            'correlation_id': correlation_id
        })
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully", extra={
                'function': func.__name__,
                'correlation_id': correlation_id
            })
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}", extra={
                'function': func.__name__,
                'correlation_id': correlation_id,
                'error': str(e)
            })
            raise
    
    return wrapper 