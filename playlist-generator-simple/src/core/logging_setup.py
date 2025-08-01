"""
Simple logging setup using standard Python logging.
"""

import os
import logging
from logging.handlers import RotatingFileHandler


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
    
    # Console handler
    if console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_logging:
        log_file = os.path.join(log_dir, f"{log_file_prefix}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
                details: str = None, duration: float = None, **kwargs):
    """
    Log API call.
    
    Args:
        api_name: Name of the API
        operation: Operation performed
        target: Target of the operation
        success: Whether the operation was successful
        details: Additional details
        duration: Duration in seconds
        **kwargs: Additional fields (ignored)
    """
    status = "SUCCESS" if success else "FAILED"
    message = f"{api_name} API {operation} {target}: {status}"
    
    if details:
        message += f" - {details}"
    
    if duration:
        message += f" ({duration:.2f}s)"
    
    log_level = 'INFO' if success else 'ERROR'
    log_universal(log_level, api_name, message)


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