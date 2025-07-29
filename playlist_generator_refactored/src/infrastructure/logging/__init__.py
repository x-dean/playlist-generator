"""
Logging infrastructure for the Playlista application.

This module provides structured logging with correlation IDs,
log level management, and monitoring capabilities.
"""

from .logger import (
    setup_logging,
    get_logger,
    change_log_level,
    setup_log_level_monitor,
    setup_signal_handlers,
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id,
    log_function_call
)

from .formatters import (
    StructuredFormatter,
    ColoredFormatter,
    JsonFormatter
)

from .handlers import (
    FileHandler,
    ConsoleHandler,
    RotatingFileHandler
)

__all__ = [
    # Main logging functions
    'setup_logging',
    'get_logger',
    'change_log_level',
    'setup_log_level_monitor',
    'setup_signal_handlers',
    'set_correlation_id',
    'get_correlation_id',
    'clear_correlation_id',
    'log_function_call',
    
    # Formatters
    'StructuredFormatter',
    'ColoredFormatter',
    'JsonFormatter',
    
    # Handlers
    'FileHandler',
    'ConsoleHandler',
    'RotatingFileHandler'
] 