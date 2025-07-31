"""
Production-grade logging setup for Playlist Generator Simple.
Uses standard logging library with configurable output formats and handlers.
"""

import logging
import logging.handlers
import os
import sys
import threading
import time
import signal
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

# Global state
_log_setup_complete = False
_log_level_monitor_thread = None
_log_config = {}


class ColoredFormatter(logging.Formatter):
    """
    Universal colored formatter for console output.
    Provides consistent formatting with color coding for different log levels.
    """
    
    # Enhanced color codes with better contrast
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    # Component colors for structured logging
    COMPONENT_COLORS = {
        'MB API': '\033[94m',     # Blue
        'LF API': '\033[95m',     # Magenta
        'Enrichment': '\033[96m', # Bright Cyan
        'Analysis': '\033[93m',   # Bright Yellow
        'Database': '\033[92m',   # Bright Green
        'Cache': '\033[90m',      # Gray
        'Worker': '\033[97m',     # White
        'Sequential': '\033[94m', # Blue
        'Parallel': '\033[95m',   # Magenta
        'Resource': '\033[96m',   # Bright Cyan
        'Progress': '\033[93m',   # Bright Yellow
        'Playlist': '\033[92m',   # Bright Green
        'Streaming': '\033[90m',  # Gray
        'CLI': '\033[97m',        # White
        'Config': '\033[94m',     # Blue
        'Export': '\033[95m',     # Magenta
        'Pipeline': '\033[96m',   # Bright Cyan
        'System': '\033[93m',     # Bright Yellow
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.supports_color = self._supports_color()
    
    def _supports_color(self):
        """Check if the terminal supports color output."""
        if os.name == 'nt':
            return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        else:
            return True
    
    def format(self, record):
        # Color the level name
        levelname = record.levelname
        if self.supports_color and levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Color common component prefixes in messages
        if self.supports_color and hasattr(record, 'msg'):
            message = str(record.msg)
            for component, color in self.COMPONENT_COLORS.items():
                if component in message:
                    message = message.replace(
                        f"{component}:", 
                        f"{color}{component}{self.COMPONENT_COLORS['RESET']}:"
                    )
            record.msg = message
        
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging to files.
    """
    
    def __init__(self, include_extra_fields=True, include_exception_details=True):
        super().__init__()
        self.include_extra_fields = include_extra_fields
        self.include_exception_details = include_exception_details
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if self.include_exception_details and record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        if self.include_extra_fields and hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """
    Simple text formatter for file logging.
    """
    
    def __init__(self, fmt=None, datefmt=None):
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        super().__init__(fmt, datefmt)


def setup_logging(
    log_level: str = None,
    log_dir: str = None,  # Will be auto-detected if None
    log_file_prefix: str = 'playlista',
    console_logging: bool = True,
    file_logging: bool = True,
    colored_output: bool = True,
    max_log_files: int = 10,
    log_file_size_mb: int = 50,
    log_file_format: str = 'json',
    log_file_encoding: str = 'utf-8',
    console_format: str = None,
    console_date_format: str = None,
    include_extra_fields: bool = True,
    include_exception_details: bool = True,
    environment_monitoring: bool = True,
    signal_handling: bool = True,
    performance_enabled: bool = True,
    function_calls_enabled: bool = True,
    signal_cycle_levels: bool = True
) -> logging.Logger:
    """
    Setup production-grade logging with configurable output formats.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (auto-detected if None)
        log_file_prefix: Prefix for log files
        console_logging: Enable console output
        file_logging: Enable file output
        colored_output: Enable colored console output
        max_log_files: Maximum number of log files to keep
        log_file_size_mb: Maximum size of each log file in MB
        log_file_format: Format for log files ('json' or 'text')
        log_file_encoding: Encoding for log files
        console_format: Custom format for console output
        console_date_format: Custom date format for console output
        include_extra_fields: Include extra fields in JSON logs
        include_exception_details: Include exception details in logs
        environment_monitoring: Monitor environment variables for log level changes
        signal_handling: Setup signal handlers for log level control
        performance_enabled: Enable performance logging
        function_calls_enabled: Enable function call logging
        signal_cycle_levels: Enable signal-based log level cycling
    
    Returns:
        Configured logger instance
    """
    global _log_setup_complete, _log_config
    
    if _log_setup_complete:
        return logging.getLogger('playlista')
    
    # Auto-detect log directory if not provided
    if log_dir is None:
        # Try to use /app/logs in Docker, fallback to current directory
        if os.path.exists('/app/logs'):
            log_dir = '/app/logs'
        else:
            log_dir = 'logs'
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Get log level from environment or use default
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level not in valid_levels:
        log_level = 'INFO'
    
    # Get main logger
    logger = logging.getLogger('playlista')
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))
        
        if console_format is None:
            console_format = '%(asctime)s - %(levelname)s - %(message)s'
        if console_date_format is None:
            console_date_format = '%H:%M:%S'
        
        if colored_output:
            formatter = ColoredFormatter(console_format, console_date_format)
        else:
            formatter = TextFormatter(console_format, console_date_format)
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_logging:
        log_file = os.path.join(log_dir, f"{log_file_prefix}.log")
        
        if log_file_format == 'json':
            formatter = JsonFormatter(include_extra_fields, include_exception_details)
        elif colored_output:
            formatter = ColoredFormatter()
        else:
            formatter = TextFormatter()
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_file_size_mb * 1024 * 1024,
            backupCount=max_log_files,
            encoding=log_file_encoding
        )
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    # Store configuration
    _log_config = {
        'log_level': log_level,
        'log_dir': log_dir,
        'console_logging': console_logging,
        'file_logging': file_logging,
        'colored_output': colored_output,
        'log_file_format': log_file_format
    }
    
    # Setup external library logging
    _setup_external_logging(logger, log_dir, file_logging)
    
    # Setup signal handlers if enabled
    if signal_handling:
        setup_signal_handlers()
    
    # Start log level monitoring if enabled
    if environment_monitoring:
        start_log_level_monitor()
    
    # Log initialization
    log_universal('INFO', 'System', "Logging system initialized")
    log_universal('INFO', 'System', f"Log level: {log_level}")
    log_universal('INFO', 'System', f"Log directory: {log_dir}")
    log_universal('INFO', 'System', f"Console logging: {console_logging}")
    log_universal('INFO', 'System', f"File logging: {file_logging}")
    log_universal('INFO', 'System', f"Colored output: {colored_output}")
    log_universal('INFO', 'System', f"File format: {log_file_format}")
    
    _log_setup_complete = True
    return logger


def change_log_level(new_level: str) -> bool:
    """
    Change the log level dynamically.
    
    Args:
        new_level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        level = getattr(logging, new_level.upper(), None)
        if level is None:
            return False
        
        logger = logging.getLogger('playlista')
        logger.setLevel(level)
        
        # Update all handlers
        for handler in logger.handlers:
            handler.setLevel(level)
        
        log_universal('INFO', 'System', f"Log level changed to: {new_level.upper()}")
        return True
        
    except Exception as e:
        log_universal('ERROR', 'System', f"Failed to change log level: {e}")
        return False


def monitor_log_level_changes():
    """Monitor environment variable for log level changes."""
    global _log_config
    
    last_level = _log_config.get('log_level', 'INFO')
    
    while True:
        try:
            current_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
            
            if current_level != last_level:
                if change_log_level(current_level):
                    log_universal('INFO', 'System', f"Environment LOG_LEVEL changed from {last_level} to {current_level}")
                    log_universal('INFO', 'System', f"Log level updated to: {current_level}")
                    last_level = current_level
                else:
                    log_universal('ERROR', 'System', f"Failed to update log level to: {current_level}")
            
            time.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            log_universal('ERROR', 'System', f"Error in log level monitor: {e}")
            time.sleep(10)  # Wait longer on error


def start_log_level_monitor():
    """Start the log level monitoring thread."""
    global _log_level_monitor_thread
    
    if _log_level_monitor_thread is None or not _log_level_monitor_thread.is_alive():
        _log_level_monitor_thread = threading.Thread(
            target=monitor_log_level_changes,
            daemon=True,
            name="LogLevelMonitor"
        )
        _log_level_monitor_thread.start()


def setup_signal_handlers():
    """Setup signal handlers for log level control."""
    
    def cycle_log_level(signum, frame):
        """Cycle through log levels on signal."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        logger = logging.getLogger('playlista')
        current_level = logging.getLevelName(logger.level)
        
        try:
            current_index = levels.index(current_level)
            next_index = (current_index + 1) % len(levels)
            new_level = levels[next_index]
            
            if change_log_level(new_level):
                log_universal('INFO', 'System', f"Log level cycled to: {new_level}")
            else:
                log_universal('ERROR', 'System', f"Failed to change log level")
        except Exception as e:
            log_universal('ERROR', 'System', f"Error changing log level: {e}")
    
    # Register signal handlers
    try:
        signal.signal(signal.SIGUSR1, cycle_log_level)
        log_universal('INFO', 'System', "Log level control: Send SIGUSR1 signal to cycle through log levels")
        log_universal('INFO', 'System', "  Example: docker compose exec playlista kill -SIGUSR1 1")
    except (AttributeError, OSError):
        # Windows doesn't support SIGUSR1
        pass


def _setup_external_logging(logger: logging.Logger, log_dir: str, file_logging: bool) -> None:
    """
    Setup logging for external libraries to reduce noise.
    
    Args:
        logger: Main logger instance
        log_dir: Log directory
        file_logging: Whether file logging is enabled
    """
    
    # TensorFlow logging
    try:
        import tensorflow as tf
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.handlers.clear()
        tf_logger.setLevel(logging.ERROR)  # Only show errors
        
        if file_logging:
            tf_log_file = os.path.join(log_dir, 'tensorflow.log')
            tf_handler = logging.handlers.RotatingFileHandler(
                tf_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3,
                encoding='utf-8'
            )
            tf_handler.setFormatter(TextFormatter())
            tf_logger.addHandler(tf_handler)
        
        log_universal('DEBUG', 'System', "TensorFlow logging configured")
        
    except ImportError:
        log_universal('DEBUG', 'System', "TensorFlow not available - skipping TF logging setup")
    
    # Essentia logging
    try:
        import essentia
        essentia_logger = logging.getLogger('essentia')
        essentia_logger.handlers.clear()
        essentia_logger.setLevel(logging.ERROR)
        
        if file_logging:
            essentia_log_file = os.path.join(log_dir, 'essentia.log')
            essentia_handler = logging.handlers.RotatingFileHandler(
                essentia_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3,
                encoding='utf-8'
            )
            essentia_handler.setFormatter(TextFormatter())
            essentia_logger.addHandler(essentia_handler)
        
        log_universal('DEBUG', 'System', "Essentia logging configured")
        
    except ImportError:
        log_universal('DEBUG', 'System', "Essentia not available - skipping Essentia logging setup")


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with proper configuration.
    
    Args:
        name: Logger name (optional)
    
    Returns:
        Configured logger instance
    """
    logger_instance = logging.getLogger(name)
    
    if name != 'playlista' and not logger_instance.handlers:
        # Inherit handlers from main logger
        main_logger = logging.getLogger('playlista')
        logger_instance.handlers = main_logger.handlers
        logger_instance.setLevel(main_logger.level)
        logger_instance.propagate = False
    
    return logger_instance


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        log_universal('DEBUG', 'System', f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            log_universal('DEBUG', 'System', f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            log_universal('ERROR', 'System', f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


def log_universal(level: str, component: str, message: str, **kwargs):
    """
    Universal logging function with component-based formatting.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        component: Component name for color coding
        message: Log message
        **kwargs: Additional fields for structured logging
    """
    logger = logging.getLogger('playlista')
    
    # Create structured message with component prefix
    structured_message = f"{component}: {message}"
    
    # Add extra fields if provided
    if kwargs:
        extra_fields = kwargs
    else:
        extra_fields = {}
    
    # Create log record with extra fields
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 0, structured_message, (), None
    )
    record.extra_fields = extra_fields
    logger.handle(record)
    
    # Use appropriate log method
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(structured_message, extra=extra_fields)


def log_api_call(api_name: str, operation: str, target: str, success: bool = True, 
                details: str = None, duration: float = None, **kwargs):
    """
    Log API call with structured information.
    
    Args:
        api_name: Name of the API (e.g., 'MusicBrainz', 'Last.fm')
        operation: Operation performed (e.g., 'search', 'get_metadata')
        target: Target of the operation (e.g., 'artist', 'album')
        success: Whether the operation was successful
        details: Additional details about the operation
        duration: Duration of the operation in seconds
        **kwargs: Additional fields
    """
    logger = logging.getLogger('playlista')
    
    # Create structured message
    status = "SUCCESS" if success else "FAILED"
    message = f"{api_name} API {operation} {target}: {status}"
    
    if details:
        message += f" - {details}"
    
    if duration:
        message += f" ({duration:.2f}s)"
    
    # Use log_universal for consistency
    log_level = 'INFO' if success else 'ERROR'
    log_universal(log_level, api_name, message, **kwargs)


def cleanup_logging():
    """Cleanup logging resources."""
    global _log_setup_complete, _log_level_monitor_thread
    
    if _log_level_monitor_thread and _log_level_monitor_thread.is_alive():
        _log_level_monitor_thread.join(timeout=5)
    
    _log_setup_complete = False


def get_log_config() -> Dict[str, Any]:
    """Get current logging configuration."""
    return _log_config.copy()


def reload_logging_from_config(config: Dict[str, Any]) -> logging.Logger:
    """
    Reload logging configuration from a config dictionary.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Updated logger instance
    """
    global _log_setup_complete
    
    # Reset setup flag to allow reconfiguration
    _log_setup_complete = False
    
    # Extract parameters from config
    log_level = config.get('log_level', 'INFO')
    log_dir = config.get('log_dir', None)
    console_logging = config.get('console_logging', True)
    file_logging = config.get('file_logging', True)
    colored_output = config.get('colored_output', True)
    log_file_format = config.get('log_file_format', 'json')
    
    # Setup logging with new configuration
    return setup_logging(
        log_level=log_level,
        log_dir=log_dir,
        console_logging=console_logging,
        file_logging=file_logging,
        colored_output=colored_output,
        log_file_format=log_file_format
    ) 