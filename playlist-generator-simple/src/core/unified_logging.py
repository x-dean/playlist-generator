"""
Professional unified logging system for Playlist Generator.

This module provides a consolidated, professional logging implementation that:
- Uses Python's standard logging library as the foundation
- Provides structured logging capabilities
- Supports colored console output and file logging
- Manages external library logging properly
- Follows professional logging best practices
- Provides performance optimization features
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from functools import wraps
import threading
import time

# Optional imports for enhanced features
try:
    from colorama import Fore, Back, Style, init as colorama_init
    colorama_init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Enhanced colored formatter for console output."""
    
    # Color mappings for different log levels
    LEVEL_COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    
    # Component color mappings for better visual organization
    COMPONENT_COLORS = {
        'CLI': '\033[97m',        # White
        'Audio': '\033[94m',      # Blue
        'Database': '\033[95m',   # Magenta
        'Config': '\033[96m',     # Cyan
        'Playlist': '\033[92m',   # Green
        'Analysis': '\033[93m',   # Yellow
        'Resource': '\033[93m',   # Yellow
        'System': '\033[90m',     # Gray
        'FileDiscovery': '\033[96m',  # Cyan
        'API': '\033[93m',        # Yellow
        'Cache': '\033[90m',      # Gray
    }
    
    RESET = '\033[0m'
    
    def __init__(self, format_string=None, datefmt=None, style='%', validate=True):
        """Initialize colored formatter."""
        if format_string is None:
            format_string = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        if datefmt is None:
            datefmt = '%H:%M:%S'
        
        super().__init__(format_string, datefmt, style, validate)
    
    def format(self, record):
        """Format the log record with colors."""
        # Create a copy to avoid modifying the original record
        record_copy = logging.makeLogRecord(record.__dict__)
        
        # Apply level colors
        level_name = record_copy.levelname
        if level_name in self.LEVEL_COLORS:
            colored_level = f"{self.LEVEL_COLORS[level_name]}{level_name}{self.RESET}"
            record_copy.levelname = colored_level
        
        # Apply component colors to the message
        message = record_copy.getMessage()
        for component, color in self.COMPONENT_COLORS.items():
            if f"{component}:" in message:
                colored_component = f"{color}{component}{self.RESET}"
                message = message.replace(f"{component}:", f"{colored_component}:")
        
        # Update the record with the colored message
        record_copy.msg = message
        record_copy.args = ()
        
        return super().format(record_copy)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format the log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogFilter(logging.Filter):
    """Filter to reduce logging overhead for high-frequency operations."""
    
    def __init__(self, sample_rate=0.1):
        """Initialize with sampling rate (0.1 = 10% of messages)."""
        super().__init__()
        self.sample_rate = sample_rate
        self.counter = 0
        self.lock = threading.Lock()
    
    def filter(self, record):
        """Filter based on sampling rate for DEBUG level messages."""
        if record.levelno > logging.DEBUG:
            return True  # Always pass non-DEBUG messages
        
        with self.lock:
            self.counter += 1
            return (self.counter % int(1/self.sample_rate)) == 0


class UnifiedLogger:
    """
    Professional unified logging system.
    
    Features:
    - Structured and colored logging
    - Performance optimization
    - External library management
    - Multiple output formats
    - Proper error handling
    """
    
    def __init__(self):
        self._loggers_cache = {}
        self._configured = False
        self._config = {}
        self._handlers = []
    
    def configure(self, config: Dict[str, Any] = None):
        """
        Configure the logging system.
        
        Args:
            config: Configuration dictionary with logging settings
        """
        self._config = config or {}
        
        # Set default configuration
        defaults = {
            'level': 'INFO',
            'console_enabled': True,
            'file_enabled': True,
            'structured_logging': False,
            'colored_output': True,
            'log_dir': 'logs',
            'log_file_prefix': 'playlista',
            'max_file_size_mb': 50,
            'backup_count': 10,
            'performance_sampling': False,
            'sample_rate': 0.1,
            'external_library_level': 'WARNING',
        }
        
        for key, value in defaults.items():
            self._config.setdefault(key, value)
        
        # Override with environment variables
        self._apply_environment_overrides()
        
        # Setup root logger
        self._setup_root_logger()
        
        # Setup external library logging
        self._setup_external_libraries()
        
        self._configured = True
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'LOG_LEVEL': 'level',
            'LOG_CONSOLE_ENABLED': 'console_enabled',
            'LOG_FILE_ENABLED': 'file_enabled',
            'LOG_STRUCTURED': 'structured_logging',
            'LOG_COLORED': 'colored_output',
            'LOG_DIR': 'log_dir',
            'LOG_FILE_PREFIX': 'log_file_prefix',
            'LOG_MAX_SIZE_MB': 'max_file_size_mb',
            'LOG_BACKUP_COUNT': 'backup_count',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Type conversion based on config key
                if config_key in ['console_enabled', 'file_enabled', 'structured_logging', 'colored_output', 'performance_sampling']:
                    self._config[config_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif config_key in ['max_file_size_mb', 'backup_count']:
                    try:
                        self._config[config_key] = int(env_value)
                    except ValueError:
                        pass  # Keep default if conversion fails
                else:
                    self._config[config_key] = env_value
    
    def _setup_root_logger(self):
        """Setup the root logger with appropriate handlers."""
        root_logger = logging.getLogger()
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root level
        log_level = getattr(logging, self._config['level'].upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # Add console handler
        if self._config['console_enabled']:
            self._add_console_handler(root_logger)
        
        # Add file handler
        if self._config['file_enabled']:
            self._add_file_handler(root_logger)
        
        # Add performance filter if enabled
        if self._config['performance_sampling']:
            perf_filter = PerformanceLogFilter(self._config['sample_rate'])
            for handler in root_logger.handlers:
                handler.addFilter(perf_filter)
    
    def _add_console_handler(self, logger):
        """Add console handler with appropriate formatting."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self._config['level'].upper(), logging.INFO))
        
        if self._config['colored_output'] and COLORAMA_AVAILABLE:
            formatter = ColoredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        self._handlers.append(console_handler)
    
    def _add_file_handler(self, logger):
        """Add file handler with rotation."""
        log_dir = Path(self._config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self._config['log_file_prefix']}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self._config['max_file_size_mb'] * 1024 * 1024,
            backupCount=self._config['backup_count'],
            encoding='utf-8'
        )
        
        file_handler.setLevel(getattr(logging, self._config['level'].upper(), logging.INFO))
        
        if self._config['structured_logging']:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self._handlers.append(file_handler)
    
    def _setup_external_libraries(self):
        """Configure logging for external libraries."""
        external_level = getattr(logging, self._config['external_library_level'].upper(), logging.WARNING)
        
        # TensorFlow
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            import tensorflow as tf
            tf.get_logger().setLevel(external_level)
            tf.autograph.set_verbosity(0)
        except ImportError:
            pass
        
        # Essentia
        try:
            os.environ['ESSENTIA_LOG_LEVEL'] = 'error'
            essentia_logger = logging.getLogger('essentia')
            essentia_logger.setLevel(external_level)
        except Exception:
            pass
        
        # Librosa
        try:
            librosa_logger = logging.getLogger('librosa')
            librosa_logger.setLevel(external_level)
        except Exception:
            pass
        
        # Musicbrainzngs
        try:
            mb_logger = logging.getLogger('musicbrainzngs')
            mb_logger.setLevel(external_level)
        except Exception:
            pass
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name, defaults to 'playlista'
        
        Returns:
            Configured logger instance
        """
        if not self._configured:
            self.configure()
        
        logger_name = name or 'playlista'
        
        if logger_name not in self._loggers_cache:
            logger = logging.getLogger(logger_name)
            self._loggers_cache[logger_name] = logger
        
        return self._loggers_cache[logger_name]
    
    def log_structured(self, level: str, component: str, message: str, **extra_fields):
        """
        Log with structured data.
        
        Args:
            level: Log level
            component: Component name
            message: Log message
            **extra_fields: Additional structured fields
        """
        logger = self.get_logger()
        log_method = getattr(logger, level.lower(), logger.info)
        
        # Create a custom log record with extra fields
        record = logger.makeRecord(
            logger.name, getattr(logging, level.upper(), logging.INFO),
            '', 0, f"{component}: {message}", (), None
        )
        record.extra_fields = extra_fields
        
        # Handle the record through all handlers
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)
    
    def change_level(self, new_level: str) -> bool:
        """
        Change the logging level dynamically.
        
        Args:
            new_level: New log level
        
        Returns:
            True if successful, False otherwise
        """
        try:
            level = getattr(logging, new_level.upper(), None)
            if level is None:
                return False
            
            # Update root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(level)
            
            # Update all handlers
            for handler in self._handlers:
                handler.setLevel(level)
            
            # Update config
            self._config['level'] = new_level.upper()
            
            logger = self.get_logger()
            logger.info(f"Log level changed to: {new_level.upper()}")
            return True
            
        except Exception as e:
            logger = self.get_logger()
            logger.error(f"Failed to change log level: {e}")
            return False
    
    def cleanup(self):
        """Cleanup logging resources."""
        for handler in self._handlers:
            try:
                handler.close()
            except Exception:
                pass
        
        self._handlers.clear()
        self._loggers_cache.clear()
        self._configured = False


# Global unified logger instance
_unified_logger = UnifiedLogger()


def setup_logging(config: Dict[str, Any] = None):
    """
    Setup the unified logging system.
    
    Args:
        config: Configuration dictionary
    """
    _unified_logger.configure(config)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Configured logger instance
    """
    return _unified_logger.get_logger(name)


def log_structured(level: str, component: str, message: str, **extra_fields):
    """
    Log with structured data.
    
    Args:
        level: Log level
        component: Component name  
        message: Log message
        **extra_fields: Additional structured fields
    """
    _unified_logger.log_structured(level, component, message, **extra_fields)


def change_log_level(new_level: str) -> bool:
    """
    Change the logging level dynamically.
    
    Args:
        new_level: New log level
    
    Returns:
        True if successful, False otherwise
    """
    return _unified_logger.change_level(new_level)


def cleanup_logging():
    """Cleanup logging resources."""
    _unified_logger.cleanup()


# Performance and convenience decorators
def log_performance(logger_name: str = None):
    """Decorator to log function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = time.time()
            
            try:
                logger.debug(f"Starting {func.__name__}")
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"Completed {func.__name__} in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


def log_exceptions(logger_name: str = None, reraise: bool = True):
    """Decorator to log exceptions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(logger_name)
                logger.exception(f"Exception in {func.__name__}: {e}")
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


# Specialized logging functions for common use cases
def log_api_call(api_name: str, operation: str, target: str, success: bool = True,
                details: str = None, duration: float = None, failure_type: str = None):
    """Log API call with structured information."""
    level = 'INFO' if success else ('INFO' if failure_type in ['no_data', 'not_found'] else 'ERROR')
    
    extra_fields = {
        'api_name': api_name,
        'operation': operation,
        'target': target,
        'success': success,
        'failure_type': failure_type,
    }
    
    if duration is not None:
        extra_fields['duration'] = duration
    
    message_parts = [f"{operation} {target}"]
    if details:
        message_parts.append(f"({details})")
    if duration is not None:
        message_parts.append(f"took {duration:.2f}s")
    
    status = "SUCCESS" if success else "FAILED"
    message_parts.append(f"- {status}")
    
    message = " ".join(message_parts)
    log_structured(level, f"{api_name} API", message, **extra_fields)


def log_resource_usage(component: str, **metrics):
    """Log resource usage metrics."""
    log_structured('DEBUG', component, f"Resource usage", **metrics)


def log_session_start(session_name: str, **session_info):
    """Log session start with context."""
    separator = "-" * 50
    logger = get_logger()
    logger.info(separator)
    log_structured('INFO', 'CLI', f"Session started: {session_name}", **session_info)
    logger.info(separator)


def log_session_end(session_name: str, duration: float = None, **session_info):
    """Log session end with summary."""
    extra_fields = session_info.copy()
    if duration is not None:
        extra_fields['duration'] = duration
    
    separator = "-" * 50
    logger = get_logger()
    logger.info(separator)
    log_structured('INFO', 'CLI', f"Session completed: {session_name}", **extra_fields)
    logger.info(separator)


# Legacy compatibility functions
def log_universal(level: str, component: str, message: str, **kwargs):
    """Legacy compatibility function."""
    log_structured(level, component, message, **kwargs)