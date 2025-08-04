"""
Simple logging setup using standard Python logging.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored log levels and modules."""
    
    # ANSI color codes for log levels
    LEVEL_COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    # ANSI color codes for modules
    MODULE_COLORS = {
        # Core components
        'CLI': '\033[97m',        # White
        'Audio': '\033[94m',      # Blue
        'Database': '\033[95m',   # Magenta
        'Config': '\033[96m',     # Cyan
        'Playlist': '\033[92m',   # Green
        'Analysis': '\033[93m',   # Yellow (changed from red)
        'Resource': '\033[93m',   # Yellow
        'System': '\033[90m',     # Gray
        'FileDiscovery': '\033[96m', # Cyan (changed from white)
        'Enrichment': '\033[96m', # Cyan
        'Export': '\033[95m',     # Magenta
        'Pipeline': '\033[92m',   # Green
        
        # Processing modes
        'Sequential': '\033[94m', # Blue
        'Parallel': '\033[95m',   # Magenta
        'Streaming': '\033[96m',  # Cyan
        
        # APIs
        'API': '\033[93m',        # Yellow
        'MB API': '\033[93m',     # Yellow (MusicBrainz)
        'LF API': '\033[93m',     # Yellow (LastFM)
        
        # Audio processing libraries
        'Essentia': '\033[94m',   # Blue
        'Librosa': '\033[95m',    # Magenta
        'TensorFlow': '\033[96m', # Cyan
        'MusicBrainz': '\033[93m', # Yellow
        
        # Other components
        'Cache': '\033[90m',      # Gray
        'File': '\033[97m',       # White
    }
    
    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)
        
        # Add color to the level name
        level_name = record.levelname
        if level_name in self.LEVEL_COLORS:
            colored_level = f"{self.LEVEL_COLORS[level_name]}{level_name}{self.LEVEL_COLORS['RESET']}"
            formatted = formatted.replace(level_name, colored_level)
        
        # Add color to module names in the message
        message = record.getMessage()
        for module, color in self.MODULE_COLORS.items():
            if f"{module}:" in message:
                colored_module = f"{color}{module}{self.LEVEL_COLORS['RESET']}"
                message = message.replace(f"{module}:", f"{colored_module}:")
        
        # Replace the message in the formatted string
        if record.getMessage() != message:
            formatted = formatted.replace(record.getMessage(), message)
        
        return formatted


def setup_logging(
    log_level: str = None,
    log_dir: str = None,
    log_file_prefix: str = 'playlista',
    console_logging: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Setup simple logging with standard Python logging.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_file_prefix: Prefix for log file names
        console_logging: Enable console output
        file_logging: Enable file output
    
    Returns:
        Configured logger instance
    """
    # Auto-detect log directory
    if log_dir is None:
        if os.path.exists('/app/logs'):
            log_dir = '/app/logs'  # Docker container
        elif os.path.exists('./logs'):
            log_dir = './logs'  # Local development
        else:
            log_dir = os.path.join(os.getcwd(), 'logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Set log level
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level not in valid_levels:
        log_level = 'INFO'
    
    # Get logger
    logger = logging.getLogger('playlista')
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False
    
    # Console handler with colored output
    if console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))
        console_formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s', '%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler (no colors in file)
    if file_logging:
        log_file = os.path.join(log_dir, f"{log_file_prefix}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Setup external library logging
    _setup_external_logging(log_dir, file_logging)
    
    # Log initialization
    logger.info("Logging system initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Console logging: {console_logging}")
    logger.info(f"File logging: {file_logging}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name or 'playlista')


def log_universal(level: str, component: str, message: str, **kwargs):
    """
    Universal logging function.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        component: Component name
        message: Log message
        **kwargs: Additional fields (ignored in simple setup)
    """
    logger = get_logger()
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(f"{component}: {message}")


def log_api_call(api_name: str, operation: str, target: str, success: bool = True, 
                details: str = None, duration: float = None, failure_type: str = None, **kwargs):
    """
    Log API call with detailed information.
    
    Args:
        api_name: Name of the API (e.g., 'MusicBrainz', 'LastFM')
        operation: Operation being performed (e.g., 'search', 'lookup')
        target: Target of the operation (e.g., 'artist', 'album')
        success: Whether the API call was successful
        details: Additional details about the call
        duration: Duration of the API call in seconds
        failure_type: Type of failure ('no_data', 'network', 'timeout', etc.)
        **kwargs: Additional keyword arguments
    """
    if success:
        level = 'INFO'
    else:
        # Determine log level based on failure type
        if failure_type in ['no_data', 'not_found', 'no_recordings']:
            level = 'INFO'  # API responded but no data found - not a warning
        else:
            level = 'ERROR'    # Network issues, timeouts, etc.
    
    # Build the message with more detailed information
    message_parts = [f"{api_name} API: {operation} {target}"]
    
    if details:
        message_parts.append(f"({details})")
    
    if duration is not None:
        message_parts.append(f"took {duration:.2f}s")
    
    status = "SUCCESS" if success else "FAILED"
    message_parts.append(f"- {status}")
    
    message = " ".join(message_parts)
    
    log_universal(level, f"{api_name} API", message, **kwargs)


def log_extracted_fields(api_name: str, track_title: str, artist_name: str, 
                        extracted_fields: Dict[str, Any], **kwargs):
    """
    Log detailed information about fields extracted from external APIs.
    
    Args:
        api_name: Name of the API (e.g., 'MusicBrainz', 'LastFM')
        track_title: Title of the track
        artist_name: Name of the artist
        extracted_fields: Dictionary of extracted field names and their values
        **kwargs: Additional keyword arguments
    """
    if not extracted_fields:
        log_universal('DEBUG', f"{api_name} API", f"No fields extracted for '{track_title}' by '{artist_name}'")
        return
    
    # Filter out None/empty values for cleaner logging
    non_empty_fields = {k: v for k, v in extracted_fields.items() if v is not None and v != ''}
    
    if not non_empty_fields:
        log_universal('DEBUG', f"{api_name} API", f"No non-empty fields extracted for '{track_title}' by '{artist_name}'")
        return
    
    # Build field summary
    field_summary = []
    for field_name, value in non_empty_fields.items():
        if isinstance(value, list):
            if value:
                field_summary.append(f"{field_name}: {len(value)} items")
            else:
                continue  # Skip empty lists
        elif isinstance(value, (int, float)):
            field_summary.append(f"{field_name}: {value}")
        else:
            # Truncate long string values
            str_value = str(value)
            if len(str_value) > 50:
                str_value = str_value[:47] + "..."
            field_summary.append(f"{field_name}: {str_value}")
    
    if field_summary:
        message = f"Extracted fields for '{track_title}' by '{artist_name}': {', '.join(field_summary)}"
        log_universal('DEBUG', f"{api_name} API", message, **kwargs)


def log_session_header(session_name: str = None, **kwargs):
    """
    Log a session header with separator lines.
    
    Args:
        session_name: Name of the session (optional)
        **kwargs: Additional keyword arguments
    """
    separator = "-" * 50
    log_universal('INFO', 'CLI', separator, **kwargs)
    
    if session_name:
        log_universal('INFO', 'CLI', f"Session: {session_name}", **kwargs)
        log_universal('INFO', 'CLI', separator, **kwargs)
    else:
        log_universal('INFO', 'CLI', separator, **kwargs)


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug(f"Calling {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper


def change_log_level(new_level: str) -> bool:
    """
    Change the log level dynamically.
    
    Args:
        new_level: New log level
    
    Returns:
        True if successful, False otherwise
    """
    try:
        level = getattr(logging, new_level.upper(), None)
        if level is None:
            return False
        
        logger = get_logger()
        logger.setLevel(level)
        
        for handler in logger.handlers:
            handler.setLevel(level)
        
        logger.info(f"Log level changed to: {new_level.upper()}")
        return True
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"Failed to change log level: {e}")
        return False


def _setup_external_logging(log_dir: str, file_logging: bool) -> None:
    """Setup logging for external libraries."""
    
    # TensorFlow logging
    try:
        import tensorflow as tf
        # Aggressive TensorFlow warning suppression
        tf.get_logger().setLevel(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.autograph.set_verbosity(0)
        
        # Suppress all TensorFlow warnings
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
        
        # Disable TensorFlow GPU warnings
        os.environ['TF_GPU_ALLOCATOR'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.handlers.clear()
        tf_logger.setLevel(logging.ERROR)
        
        if file_logging:
            tf_handler = logging.FileHandler(os.path.join(log_dir, 'tensorflow.log'))
            tf_handler.setLevel(logging.ERROR)
            tf_logger.addHandler(tf_handler)
    except ImportError:
        pass
    
    # Essentia logging
    try:
        essentia_logger = logging.getLogger('essentia')
        essentia_logger.handlers.clear()
        essentia_logger.setLevel(logging.ERROR)
        
        if file_logging:
            essentia_handler = logging.FileHandler(os.path.join(log_dir, 'essentia.log'))
            essentia_handler.setLevel(logging.ERROR)
            essentia_logger.addHandler(essentia_handler)
    except ImportError:
        pass
    
    # Librosa logging
    try:
        librosa_logger = logging.getLogger('librosa')
        librosa_logger.handlers.clear()
        librosa_logger.setLevel(logging.ERROR)
        
        if file_logging:
            librosa_handler = logging.FileHandler(os.path.join(log_dir, 'librosa.log'))
            librosa_handler.setLevel(logging.ERROR)
            librosa_logger.addHandler(librosa_handler)
    except ImportError:
        pass


def cleanup_logging():
    """Cleanup logging resources."""
    logger = get_logger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def get_log_config() -> dict:
    """Get current logging configuration."""
    logger = get_logger()
    return {
        'level': logger.level,
        'handlers': len(logger.handlers)
    } 