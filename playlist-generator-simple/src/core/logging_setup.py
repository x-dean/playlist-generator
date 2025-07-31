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
    Colored formatter for console output.
    Uses ANSI color codes for better readability.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        # Check if terminal supports colors
        self.supports_color = self._supports_color()
    
    def _supports_color(self):
        """Check if the terminal supports color output."""
        # Check if we're on Windows
        if os.name == 'nt':
            # On Windows, check if we're in a terminal that supports colors
            return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        else:
            # On Unix-like systems, assume colors are supported
            return True
    
    def format(self, record):
        # Add color to levelname only if colors are supported
        levelname = record.levelname
        if self.supports_color and levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
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
        
        # Add exception info if present and enabled
        if record.exc_info and self.include_exception_details:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present and enabled
        if self.include_extra_fields and hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """
    Text formatter for file logging.
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
    Setup comprehensive logging system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (auto-detected if None)
        log_file_prefix: Prefix for log file names
        console_logging: Enable console output
        file_logging: Enable file logging
        colored_output: Enable colored console output
        max_log_files: Maximum number of log files to keep
        log_file_size_mb: Maximum size of each log file in MB
        log_file_format: File format ('json' or 'text')
        log_file_encoding: File encoding for log files
        console_format: Custom console format string
        console_date_format: Custom console date format
        include_extra_fields: Include extra fields in JSON logs
        include_exception_details: Include full exception details
        environment_monitoring: Enable environment variable monitoring
        signal_handling: Enable signal-based log level control
    
    Returns:
        Configured logger instance
    """
    global _log_setup_complete, _log_config
    
    if _log_setup_complete:
        return logging.getLogger('playlista')
    
    # Auto-detect log directory if not specified
    if log_dir is None:
        # Check if we're in Docker (container has /app directory)
        if os.path.exists('/app'):
            log_dir = '/app/logs'  # Docker mount point
        else:
            log_dir = 'logs'  # Local development
    
    # Store configuration
    _log_config = {
        'log_level': log_level,
        'log_dir': log_dir,
        'log_file_prefix': log_file_prefix,
        'console_logging': console_logging,
        'file_logging': file_logging,
        'colored_output': colored_output,
        'max_log_files': max_log_files,
        'log_file_size_mb': log_file_size_mb,
        'log_file_format': log_file_format,
        'log_file_encoding': log_file_encoding,
        'console_format': console_format,
        'console_date_format': console_date_format,
        'include_extra_fields': include_extra_fields,
        'include_exception_details': include_exception_details,
        'environment_monitoring': environment_monitoring,
        'signal_handling': signal_handling,
        'performance_enabled': performance_enabled,
        'function_calls_enabled': function_calls_enabled,
        'signal_cycle_levels': signal_cycle_levels
    }
    
    # Get log level from environment or parameter
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create main logger
    logger = logging.getLogger('playlista')
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler (if enabled)
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Set console format
        if console_format is None:
            console_format = '%(asctime)s - %(levelname)s - %(message)s'
        if console_date_format is None:
            console_date_format = '%H:%M:%S'
        
        if colored_output:
            console_formatter = ColoredFormatter(console_format, console_date_format)
        else:
            console_formatter = logging.Formatter(console_format, console_date_format)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if file_logging:
        # Ensure log directory exists
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_path / f"{log_file_prefix}_{timestamp}.log"
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=log_file_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=max_log_files,
            encoding=log_file_encoding
        )
        file_handler.setLevel(logging.DEBUG)  # File captures all levels
        
        # Set file formatter based on format
        if log_file_format.lower() == 'json':
            file_formatter = JsonFormatter(include_extra_fields, include_exception_details)
        else:
            file_formatter = TextFormatter()
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    # Start environment monitoring if enabled
    if environment_monitoring:
        start_log_level_monitor()
    
    # Setup signal handlers if enabled
    if signal_handling:
        setup_signal_handlers()
    
    # Setup external library logging (TensorFlow, Essentia)
    _setup_external_logging(logger, log_dir, file_logging)
    
    _log_setup_complete = True
    
    # Log initialization
    logger.info("Logging system initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Console logging: {console_logging}")
    logger.info(f"File logging: {file_logging}")
    logger.info(f"Colored output: {colored_output}")
    logger.info(f"File format: {log_file_format}")
    
    return logger


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
        logger = logging.getLogger('playlista')
        logger.setLevel(level)
        
        # Update all handlers
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)
        
        logger.info(f"Log level changed to: {new_level.upper()}")
        return True
    except Exception as e:
        logger.error(f"Failed to change log level: {e}")
        return False


def monitor_log_level_changes():
    """Background thread that monitors LOG_LEVEL environment variable changes."""
    last_level = os.getenv('LOG_LEVEL', 'INFO')
    
    while True:
        try:
            # Get current level from environment
            current_level = os.getenv('LOG_LEVEL', 'INFO')
            
            if current_level != last_level:
                logger.info(f"Environment LOG_LEVEL changed from {last_level} to {current_level}")
                if change_log_level(current_level):
                    logger.info(f"Log level updated to: {current_level}")
                else:
                    logger.error(f"Failed to update log level to: {current_level}")
                last_level = current_level
            
            time.sleep(2)  # Check every 2 seconds
            
        except Exception as e:
            logger.error(f"Error in log level monitor: {e}")
            time.sleep(5)  # Wait longer on error


def start_log_level_monitor():
    """Start the background thread that monitors LOG_LEVEL changes."""
    global _log_level_monitor_thread
    
    if _log_level_monitor_thread is None or not _log_level_monitor_thread.is_alive():
        _log_level_monitor_thread = threading.Thread(
            target=monitor_log_level_changes,
            daemon=True,
            name="LogLevelMonitor"
        )
        _log_level_monitor_thread.start()


def setup_signal_handlers():
    """Setup signal handlers for runtime log level control."""
    logger = get_logger()
    
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
                logger.info(f"Log level cycled to: {new_level}")
            else:
                logger.error(f"Failed to change log level")
        except Exception as e:
            logger.error(f"Error changing log level: {e}")
    
    # Use SIGUSR1 to cycle through log levels (if available)
    try:
        signal.signal(signal.SIGUSR1, cycle_log_level)
        logger.info("Log level control: Send SIGUSR1 signal to cycle through log levels")
        logger.info("  Example: docker compose exec playlista kill -SIGUSR1 1")
    except (AttributeError, OSError):
        # SIGUSR1 not available on Windows
        pass


def _setup_external_logging(logger: logging.Logger, log_dir: str, file_logging: bool) -> None:
    """
    Setup logging for external libraries (TensorFlow, Essentia).
    
    Args:
        logger: Main application logger
        log_dir: Log directory path
        file_logging: Whether file logging is enabled
    """
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
        
        # Add file handler for TensorFlow logs if file logging is enabled
        if file_logging:
            tf_log_file = os.path.join(log_dir, "tensorflow.log")
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
        
        logger.debug("TensorFlow logging configured")
        
    except ImportError:
        logger.debug("TensorFlow not available - skipping TF logging setup")
    
    try:
        import essentia
        
        # Configure Essentia logging - suppress info and warnings
        essentia.log.infoActive = False
        essentia.log.warningActive = False
        essentia.log.errorActive = True  # Keep errors
        
        # Create Essentia logger and redirect to file if enabled
        if file_logging:
            essentia_log_file = os.path.join(log_dir, "essentia.log")
            essentia_logger = logging.getLogger('essentia')
            essentia_logger.handlers.clear()
            essentia_logger.setLevel(logging.ERROR)
            
            essentia_handler = logging.handlers.RotatingFileHandler(
                essentia_log_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3
            )
            essentia_formatter = logging.Formatter(
                '%(asctime)s [ESSENTIA] %(levelname)s - %(message)s'
            )
            essentia_handler.setFormatter(essentia_formatter)
            essentia_logger.addHandler(essentia_handler)
        
        logger.debug("Essentia logging configured")
        
    except ImportError:
        logger.debug("Essentia not available - skipping Essentia logging setup")


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (uses 'playlista' if None)
    
    Returns:
        Logger instance
    """
    if name is None:
        name = 'playlista'
    
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_info(message: str, **kwargs):
    """Log info message with optional extra fields."""
    logger = get_logger()
    if kwargs:
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0, message, (), None
        )
        record.extra_fields = kwargs
        logger.handle(record)
    else:
        logger.info(message)


def log_error(message: str, error: Exception = None, **kwargs):
    """Log error message with optional exception and extra fields."""
    logger = get_logger()
    if error:
        logger.error(f"{message}: {error}", exc_info=True)
    else:
        logger.error(message)
    
    if kwargs:
        record = logger.makeRecord(
            logger.name, logging.ERROR, __file__, 0, message, (), None
        )
        record.extra_fields = kwargs
        logger.handle(record)


def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    logger = get_logger()
    message = f"{operation} took {duration:.3f}s"
    if kwargs:
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0, message, (), None
        )
        record.extra_fields = {'operation': operation, 'duration': duration, **kwargs}
        logger.handle(record)
    else:
        logger.info(message)


def log_analysis_operation(operation: str, file_path: str = None, file_size_mb: float = None, 
                          analysis_type: str = None, duration: float = None, 
                          features_extracted: int = None, success: bool = True, 
                          error: str = None, **kwargs):
    """Log analysis operation with structured data."""
    logger = get_logger()
    
    # Build structured message
    message_parts = [f"Analysis operation: {operation}"]
    if analysis_type:
        message_parts.append(f"type={analysis_type}")
    if file_path:
        message_parts.append(f"file={file_path}")
    if duration is not None:
        message_parts.append(f"duration={duration:.3f}s")
    if features_extracted:
        message_parts.append(f"features={features_extracted}")
    if not success:
        message_parts.append("FAILED")
    
    message = "| ".join(message_parts)
    
    # Create structured record
    record = logger.makeRecord(
        logger.name, 
        logging.INFO if success else logging.ERROR, 
        __file__, 0, message, (), None
    )
    
    # Add structured fields
    record.extra_fields = {
        'operation': operation,
        'analysis_type': analysis_type,
        'file_path': file_path,
        'file_size_mb': file_size_mb,
        'duration': duration,
        'features_extracted': features_extracted,
        'success': success,
        'error': error,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def log_resource_decision(file_path: str, file_size_mb: float, analysis_type: str, 
                         reason: str, memory_available_gb: float = None, 
                         cpu_percent: float = None, forced: bool = False, **kwargs):
    """Log resource-based analysis decisions."""
    logger = get_logger()
    
    decision_type = "FORCED"if forced else "DETERMINISTIC"
    message = f"Resource decision: {decision_type} | {analysis_type} | {reason}"
    
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'decision_type': decision_type,
        'analysis_type': analysis_type,
        'file_path': file_path,
        'file_size_mb': file_size_mb,
        'reason': reason,
        'memory_available_gb': memory_available_gb,
        'cpu_percent': cpu_percent,
        'forced': forced,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def log_feature_extraction(file_path: str, features_enabled: list, features_extracted: list,
                          duration: float, success: bool = True, error: str = None, **kwargs):
    """Log feature extraction details."""
    logger = get_logger()
    
    status = "SUCCESS"if success else "FAILED"
    message = f"Feature extraction: {status} | {len(features_extracted)}/{len(features_enabled)} features | {duration:.3f}s"
    
    record = logger.makeRecord(
        logger.name, 
        logging.INFO if success else logging.ERROR, 
        __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'file_path': file_path,
        'features_enabled': features_enabled,
        'features_extracted': features_extracted,
        'features_failed': list(set(features_enabled) - set(features_extracted)),
        'duration': duration,
        'success': success,
        'error': error,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def log_worker_operation(worker_id: str, operation: str, file_path: str, 
                        duration: float, success: bool = True, error: str = None, **kwargs):
    """Log worker operation details."""
    logger = get_logger()
    
    status = "SUCCESS"if success else "FAILED"
    message = f"Worker {worker_id}: {operation} | {status} | {duration:.3f}s"
    
    record = logger.makeRecord(
        logger.name, 
        logging.INFO if success else logging.ERROR, 
        __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'worker_id': worker_id,
        'operation': operation,
        'file_path': file_path,
        'duration': duration,
        'success': success,
        'error': error,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def log_batch_processing(batch_id: str, total_files: int, successful_files: int, 
                        failed_files: int, total_duration: float, 
                        processing_mode: str = None, **kwargs):
    """Log batch processing summary."""
    logger = get_logger()
    
    success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
    avg_duration = total_duration / total_files if total_files > 0 else 0
    
    message = f"Batch {batch_id}: {successful_files}/{total_files} files | {success_rate:.1f}% success | {avg_duration:.3f}s avg"
    
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'batch_id': batch_id,
        'total_files': total_files,
        'successful_files': successful_files,
        'failed_files': failed_files,
        'success_rate': success_rate,
        'total_duration': total_duration,
        'avg_duration': avg_duration,
        'processing_mode': processing_mode,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def cleanup_logging():
    """Clean up logging handlers to avoid file access issues."""
    logger = logging.getLogger('playlista')
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # Also clean up any other loggers that might have been created
    for name in logging.root.manager.loggerDict:
        if name.startswith('playlista'):
            other_logger = logging.getLogger(name)
            for handler in other_logger.handlers[:]:
                handler.close()
                other_logger.removeHandler(handler)


def get_log_config() -> Dict[str, Any]:
    """Get the current logging configuration."""
    return _log_config.copy()


def reload_logging_from_config(config: Dict[str, Any]) -> logging.Logger:
    """
    Reload logging configuration from a config dictionary.
    
    Args:
        config: Configuration dictionary with logging settings
    
    Returns:
        Configured logger instance
    """
    # Extract logging settings from config
    log_settings = {
        'log_level': config.get('LOG_LEVEL', 'INFO'),
        'log_dir': '/app/logs',  # Fixed Docker internal path
        'log_file_prefix': config.get('LOG_FILE_PREFIX', 'playlista'),
        'console_logging': config.get('LOG_CONSOLE_ENABLED', True),
        'file_logging': config.get('LOG_FILE_ENABLED', True),
        'colored_output': config.get('LOG_COLORED_OUTPUT', True),
        'max_log_files': config.get('LOG_MAX_FILES', 10),
        'log_file_size_mb': config.get('LOG_FILE_SIZE_MB', 50),
        'log_file_format': config.get('LOG_FILE_FORMAT', 'json'),
        'log_file_encoding': config.get('LOG_FILE_ENCODING', 'utf-8'),
        'console_format': config.get('LOG_CONSOLE_FORMAT'),
        'console_date_format': config.get('LOG_CONSOLE_DATE_FORMAT'),
        'include_extra_fields': config.get('LOG_FILE_INCLUDE_EXTRA_FIELDS', True),
        'include_exception_details': config.get('LOG_FILE_INCLUDE_EXCEPTION_DETAILS', True),
        'environment_monitoring': config.get('LOG_ENVIRONMENT_MONITORING', True),
        'signal_handling': config.get('LOG_SIGNAL_HANDLING_ENABLED', True),
        'performance_enabled': config.get('LOG_PERFORMANCE_ENABLED', True),
        'function_calls_enabled': config.get('LOG_FUNCTION_CALLS_ENABLED', True),
        'signal_cycle_levels': config.get('LOG_SIGNAL_CYCLE_LEVELS', True)
    }
    
    # Clean up existing logging
    cleanup_logging()
    global _log_setup_complete
    _log_setup_complete = False
    
    # Setup new logging configuration
    return setup_logging(**log_settings) 


def log_feature_extraction_step(file_path: str, feature_name: str, duration: float, 
                               success: bool = True, error: str = None, 
                               feature_value: Any = None, **kwargs):
    """Log individual feature extraction step with detailed metrics."""
    logger = get_logger()
    
    status = "SUCCESS"if success else "FAILED"
    message = f"Feature extraction step: {feature_name} | {status} | {duration:.3f}s"
    
    record = logger.makeRecord(
        logger.name, 
        logging.INFO if success else logging.ERROR, 
        __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'file_path': file_path,
        'feature_name': feature_name,
        'duration': duration,
        'success': success,
        'error': error,
        'feature_value': feature_value,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def log_worker_performance(worker_id: str, operation: str, file_path: str, 
                          duration: float, memory_usage_mb: float = None,
                          cpu_usage_percent: float = None, success: bool = True,
                          error: str = None, **kwargs):
    """Log detailed worker performance metrics."""
    logger = get_logger()
    
    status = "SUCCESS"if success else "FAILED"
    message = f"Worker {worker_id}: {operation} | {status} | {duration:.3f}s"
    if memory_usage_mb:
        message += f"| Memory: {memory_usage_mb:.1f}MB"
    if cpu_usage_percent:
        message += f"| CPU: {cpu_usage_percent:.1f}%"
    
    record = logger.makeRecord(
        logger.name, 
        logging.INFO if success else logging.ERROR, 
        __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'worker_id': worker_id,
        'operation': operation,
        'file_path': file_path,
        'duration': duration,
        'memory_usage_mb': memory_usage_mb,
        'cpu_usage_percent': cpu_usage_percent,
        'success': success,
        'error': error,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def log_batch_processing_detailed(batch_id: str, total_files: int, successful_files: int, 
                                 failed_files: int, total_duration: float,
                                 processing_mode: str = None, avg_memory_usage_mb: float = None,
                                 avg_cpu_usage_percent: float = None, peak_memory_mb: float = None,
                                 peak_cpu_percent: float = None, **kwargs):
    """Log detailed batch processing statistics with resource usage."""
    logger = get_logger()
    
    success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
    avg_duration = total_duration / total_files if total_files > 0 else 0
    throughput = total_files / total_duration if total_duration > 0 else 0
    
    message = f"Batch {batch_id}: {successful_files}/{total_files} files | {success_rate:.1f}% success | {avg_duration:.3f}s avg | {throughput:.2f} files/s"
    
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'batch_id': batch_id,
        'total_files': total_files,
        'successful_files': successful_files,
        'failed_files': failed_files,
        'success_rate': success_rate,
        'total_duration': total_duration,
        'avg_duration': avg_duration,
        'throughput_files_per_second': throughput,
        'processing_mode': processing_mode,
        'avg_memory_usage_mb': avg_memory_usage_mb,
        'avg_cpu_usage_percent': avg_cpu_usage_percent,
        'peak_memory_mb': peak_memory_mb,
        'peak_cpu_percent': peak_cpu_percent,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def log_resource_usage(file_path: str = None, operation: str = None, 
                      memory_usage_mb: float = None, cpu_usage_percent: float = None,
                      disk_usage_mb: float = None, network_usage_mb: float = None,
                      **kwargs):
    """Log detailed resource usage metrics."""
    logger = get_logger()
    
    message_parts = []
    if operation:
        message_parts.append(f"Operation: {operation}")
    if memory_usage_mb:
        message_parts.append(f"Memory: {memory_usage_mb:.1f}MB")
    if cpu_usage_percent:
        message_parts.append(f"CPU: {cpu_usage_percent:.1f}%")
    if disk_usage_mb:
        message_parts.append(f"Disk: {disk_usage_mb:.1f}MB")
    if network_usage_mb:
        message_parts.append(f"Network: {network_usage_mb:.1f}MB")
    
    message = "| ".join(message_parts) if message_parts else "Resource usage logged"
    
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'file_path': file_path,
        'operation': operation,
        'memory_usage_mb': memory_usage_mb,
        'cpu_usage_percent': cpu_usage_percent,
        'disk_usage_mb': disk_usage_mb,
        'network_usage_mb': network_usage_mb,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def log_processing_pipeline(file_path: str, pipeline_stages: list, total_duration: float,
                           stage_durations: dict = None, stage_success: dict = None,
                           **kwargs):
    """Log processing pipeline with stage-by-stage breakdown."""
    logger = get_logger()
    
    success_count = sum(1 for success in stage_success.values() if success) if stage_success else 0
    total_stages = len(pipeline_stages)
    success_rate = (success_count / total_stages * 100) if total_stages > 0 else 0
    
    message = f"Processing pipeline: {success_count}/{total_stages} stages successful | {success_rate:.1f}% success | {total_duration:.3f}s total"
    
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'file_path': file_path,
        'pipeline_stages': pipeline_stages,
        'total_duration': total_duration,
        'stage_durations': stage_durations,
        'stage_success': stage_success,
        'success_count': success_count,
        'total_stages': total_stages,
        'success_rate': success_rate,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record)


def log_performance_metrics(operation: str, duration: float, file_count: int = None,
                           file_size_mb: float = None, memory_peak_mb: float = None,
                           cpu_peak_percent: float = None, throughput_files_per_second: float = None,
                           **kwargs):
    """Log comprehensive performance metrics."""
    logger = get_logger()
    
    message_parts = [f"{operation}: {duration:.3f}s"]
    if file_count:
        message_parts.append(f"{file_count} files")
    if file_size_mb:
        message_parts.append(f"{file_size_mb:.1f}MB")
    if throughput_files_per_second:
        message_parts.append(f"{throughput_files_per_second:.2f} files/s")
    if memory_peak_mb:
        message_parts.append(f"Peak memory: {memory_peak_mb:.1f}MB")
    if cpu_peak_percent:
        message_parts.append(f"Peak CPU: {cpu_peak_percent:.1f}%")
    
    message = "| ".join(message_parts)
    
    record = logger.makeRecord(
        logger.name, logging.INFO, __file__, 0, message, (), None
    )
    
    record.extra_fields = {
        'operation': operation,
        'duration': duration,
        'file_count': file_count,
        'file_size_mb': file_size_mb,
        'memory_peak_mb': memory_peak_mb,
        'cpu_peak_percent': cpu_peak_percent,
        'throughput_files_per_second': throughput_files_per_second,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    logger.handle(record) 