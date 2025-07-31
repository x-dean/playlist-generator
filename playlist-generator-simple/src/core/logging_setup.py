"""
Production-grade logging setup for Playlist Generator Simple.
Uses Loguru for better features, performance, and structured logging.
"""

import os
import sys
import time
import json
import threading
import signal
from typing import Dict, Any, Optional
from datetime import datetime

# Import Loguru for better logging
import logging  # Always import for fallback and external logging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    logger = logging.getLogger('playlista')
    # Import handlers for fallback logging
    from logging import handlers

# Import colorama for cross-platform color support
try:
    from colorama import init, Fore
    init(autoreset=True)  # Initialize colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Fallback colors if colorama not available
    class Fore:
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        RED = '\033[31m'
        CYAN = '\033[36m'
        MAGENTA = '\033[35m'
        BLUE = '\033[34m'
        WHITE = '\033[37m'
        RESET = '\033[0m'
    
    class Style:
        BRIGHT = '\033[1m'
        RESET_ALL = '\033[0m'

# Global state
_log_setup_complete = False
_log_level_monitor_thread = None
_log_config = {}

# Component colors for structured logging
COMPONENT_COLORS = {
    'MB API': Fore.BLUE,
    'LF API': Fore.MAGENTA,
    'Enrichment': Fore.CYAN,
    'Analysis': Fore.YELLOW,
    'Database': Fore.GREEN,
    'Cache': Fore.WHITE,
    'Worker': Fore.WHITE,
    'Sequential': Fore.BLUE,
    'Parallel': Fore.MAGENTA,
    'Resource': Fore.CYAN,
    'Progress': Fore.YELLOW,
    'Playlist': Fore.GREEN,
    'Streaming': Fore.WHITE,
    'CLI': Fore.WHITE,
    'Config': Fore.BLUE,
    'Export': Fore.MAGENTA,
    'Pipeline': Fore.CYAN,
    'System': Fore.YELLOW,
    'Audio': Fore.GREEN,
    'CPU Optimizer': Fore.CYAN,
    'FileDiscovery': Fore.BLUE,
    'RESET': Fore.RESET
}

def setup_logging(
    log_level: str = None,
    log_dir: str = None,  # Will be auto-detected if None
    log_file_prefix: str = 'playlista',
    console_logging: bool = True,
    file_logging: bool = True,
    colored_output: bool = True,
    file_colored_output: bool = None,  # Will use colored_output if None
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
    function_calls_enabled: bool = True,
    signal_cycle_levels: bool = True
) -> 'logger':
    """
    Setup production-grade logging with Loguru.
    
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
        include_extra_fields: Include extra fields in structured logging
        include_exception_details: Include exception details in logs
        environment_monitoring: Enable environment variable monitoring
        signal_handling: Enable signal handlers for log level cycling
        performance_enabled: Enable performance logging
        function_calls_enabled: Enable function call logging
        signal_cycle_levels: Enable signal-based log level cycling
    
    Returns:
        Configured logger instance
    """
    global _log_setup_complete, _log_config
    
    if _log_setup_complete:
        return logger
    
    # Auto-detect log directory if not provided
    if log_dir is None:
        if os.path.exists('/app/logs'):
            log_dir = '/app/logs'  # Docker container
        elif os.path.exists('./logs'):
            log_dir = './logs'  # Local development
        else:
            log_dir = os.path.join(os.getcwd(), 'logs')
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Set default log level
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level not in valid_levels:
        log_level = 'INFO'
    
    # Remove default handlers if using Loguru
    if LOGURU_AVAILABLE:
        logger.remove()  # Remove default handler
        
        # Console handler with color support
        if console_logging:
            if console_format is None:
                console_format = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            
            logger.add(
                sys.stdout,
                format=console_format,
                level=log_level,
                colorize=colored_output,
                backtrace=True,
                diagnose=True
            )
        
        # File handler
        if file_logging:
            log_file = os.path.join(log_dir, f"{log_file_prefix}.log")
            
            if log_file_format == 'json':
                # JSON format for structured logging
                logger.add(
                    log_file,
                    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
                    level=log_level,
                    rotation=f"{log_file_size_mb} MB",
                    retention=max_log_files,
                    compression="zip",
                    serialize=True,  # JSON serialization
                    backtrace=True,
                    diagnose=True
                )
            else:
                # Text format with optional colors
                # Use file_colored_output if specified, otherwise use colored_output
                file_colors = file_colored_output if file_colored_output is not None else colored_output
                
                if file_colors:
                    # Colored format for file logs
                    logger.add(
                        log_file,
                        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | <level>{message}</level>",
                        level=log_level,
                        rotation=f"{log_file_size_mb} MB",
                        retention=max_log_files,
                        compression="zip",
                        backtrace=True,
                        diagnose=True
                    )
                else:
                    # Plain text format
                    logger.add(
                        log_file,
                        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
                        level=log_level,
                        rotation=f"{log_file_size_mb} MB",
                        retention=max_log_files,
                        compression="zip",
                        backtrace=True,
                        diagnose=True
                    )
    else:
        # Fallback to standard logging
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level, logging.INFO))
            
            if console_format is None:
                console_format = '%(asctime)s - %(levelname)s - %(message)s'
            if console_date_format is None:
                console_date_format = '%H:%M:%S'
            
            formatter = logging.Formatter(console_format, console_date_format)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if file_logging:
            log_file = os.path.join(log_dir, f"{log_file_prefix}.log")
            file_handler = handlers.RotatingFileHandler(
                log_file,
                maxBytes=log_file_size_mb * 1024 * 1024,
                backupCount=max_log_files,
                encoding=log_file_encoding
            )
            file_handler.setLevel(getattr(logging, log_level, logging.INFO))
            
            if log_file_format == 'json':
                formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}')
            else:
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    # Store configuration
    _log_config = {
        'log_level': log_level,
        'log_dir': log_dir,
        'console_logging': console_logging,
        'file_logging': file_logging,
        'colored_output': colored_output,
        'file_colored_output': file_colored_output,
        'log_file_format': log_file_format
    }
    
    # Setup external library logging
    _setup_external_logging(log_dir, file_logging)
    
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
    log_universal('INFO', 'System', f"Using Loguru: {LOGURU_AVAILABLE}")
    
    _log_setup_complete = True
    return logger


def get_logger(name: str = None) -> 'logger':
    """
    Get a logger instance with proper configuration.
    
    Args:
        name: Logger name (optional)
    
    Returns:
        Configured logger instance
    """
    if LOGURU_AVAILABLE:
        return logger.bind(name=name) if name else logger
    else:
        return logging.getLogger(name or 'playlista')


def log_universal(level: str, component: str, message: str, **kwargs):
    """
    Universal logging function with component-based formatting.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
        component: Component name for color coding
        message: Log message
        **kwargs: Additional fields for structured logging
    """
    # Create structured message with component prefix
    structured_message = f"{component}: {message}"
    
    # Add component color if available
    if COLORAMA_AVAILABLE and component in COMPONENT_COLORS:
        structured_message = f"{COMPONENT_COLORS[component]}{component}{Fore.RESET}: {message}"
    
    # Handle TRACE level mapping
    if level.upper() == 'TRACE':
        if LOGURU_AVAILABLE:
            # Loguru supports TRACE level
            log_level = 'trace'
        else:
            # Standard logging doesn't support TRACE, map to DEBUG
            log_level = 'debug'
    else:
        log_level = level.lower()
    
    # Use appropriate log method
    if LOGURU_AVAILABLE:
        log_method = getattr(logger, log_level, logger.info)
        log_method(structured_message, extra=kwargs)
    else:
        log_method = getattr(logger, log_level, logger.info)
        log_method(structured_message, extra=kwargs)


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


def change_log_level(new_level: str) -> bool:
    """
    Change the log level dynamically.
    
    Args:
        new_level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Handle TRACE level mapping
        if new_level.upper() == 'TRACE':
            if LOGURU_AVAILABLE:
                # Loguru supports TRACE level
                level_name = 'TRACE'
            else:
                # Standard logging doesn't support TRACE, map to DEBUG
                level_name = 'DEBUG'
        else:
            level_name = new_level.upper()
        
        level = getattr(logging, level_name, None)
        if level is None:
            return False
        
        if LOGURU_AVAILABLE:
            logger.remove()
            logger.add(sys.stdout, level=level_name)
            logger.add(
                os.path.join(_log_config.get('log_dir', './logs'), f"{_log_config.get('log_file_prefix', 'playlista')}.log"),
                level=level_name
            )
        else:
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
        
        log_universal('INFO', 'System', f"Log level changed to: {level_name}")
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
                    last_level = current_level
                else:
                    log_universal('ERROR', 'System', f"Failed to update log level to: {current_level}")
            
            time.sleep(5)  # Check every 5 seconds
        except Exception as e:
            time.sleep(5)  # Continue monitoring even if there's an error


def start_log_level_monitor():
    """Start the log level monitoring thread."""
    global _log_level_monitor_thread
    
    if _log_level_monitor_thread is None or not _log_level_monitor_thread.is_alive():
        _log_level_monitor_thread = threading.Thread(
            target=monitor_log_level_changes,
            daemon=True
        )
        _log_level_monitor_thread.start()


def setup_signal_handlers():
    """Setup signal handlers for log level cycling."""
    
    def cycle_log_level(signum, frame):
        """Cycle through log levels on signal."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        current_level = _log_config.get('log_level', 'INFO')
        
        try:
            current_index = levels.index(current_level)
            next_index = (current_index + 1) % len(levels)
            new_level = levels[next_index]
            
            if change_log_level(new_level):
                log_universal('INFO', 'System', f"Log level cycled to: {new_level}")
            else:
                log_universal('ERROR', 'System', f"Failed to cycle log level to: {new_level}")
        except Exception as e:
            log_universal('ERROR', 'System', f"Error cycling log level: {e}")
    
    # Register signal handlers
    try:
        signal.signal(signal.SIGUSR1, cycle_log_level)
        log_universal('INFO', 'System', "Signal handler registered for SIGUSR1 (log level cycling)")
    except Exception as e:
        log_universal('WARNING', 'System', f"Could not register signal handler: {e}")


def _setup_external_logging(log_dir: str, file_logging: bool) -> None:
    """Setup logging for external libraries."""
    
    # TensorFlow logging
    try:
        import tensorflow as tf
        tf_logger = logging.getLogger('tensorflow')
        tf_logger.handlers.clear()
        tf_logger.setLevel(logging.ERROR)  # Only show errors
        
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


def cleanup_logging():
    """Cleanup logging resources."""
    global _log_setup_complete, _log_level_monitor_thread
    
    if _log_level_monitor_thread and _log_level_monitor_thread.is_alive():
        _log_level_monitor_thread.join(timeout=5)
    
    _log_setup_complete = False


def get_log_config() -> Dict[str, Any]:
    """Get current logging configuration."""
    return _log_config.copy()


def reload_logging_from_config(config: Dict[str, Any]) -> 'logger':
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