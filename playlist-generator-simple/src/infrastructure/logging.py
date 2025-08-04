"""
Structured logging implementation for Playlist Generator.
Provides colored console output and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from colorama import Fore, Back, Style, init
from loguru import logger

# Initialize colorama for Windows compatibility
init()


class StructuredLogger:
    """Structured logger with colored console and file output."""
    
    def __init__(self, 
                 console_enabled: bool = True,
                 file_enabled: bool = True,
                 file_path: str = "logs/playlista.log",
                 level: str = "INFO"):
        self.console_enabled = console_enabled
        self.file_enabled = file_enabled
        self.file_path = file_path
        self.level = level
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Remove default handler
        logger.remove()
        
        # Console handler with colors
        if self.console_enabled:
            logger.add(
                sys.stdout,
                format=self._get_console_format(),
                level=self.level,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # File handler without colors
        if self.file_enabled:
            # Ensure log directory exists
            log_dir = Path(self.file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                self.file_path,
                format=self._get_file_format(),
                level=self.level,
                rotation="50 MB",
                retention="10 days",
                compression="zip",
                backtrace=True,
                diagnose=True
            )
    
    def _get_console_format(self) -> str:
        """Get colored console format."""
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    def _get_file_format(self) -> str:
        """Get file format without colors."""
        return (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        logger.exception(message, **kwargs)
    
    def log_domain_event(self, event_type: str, entity_id: str, details: Dict[str, Any]):
        """Log domain events."""
        self.info(
            f"Domain Event: {event_type}",
            entity_id=entity_id,
            event_type=event_type,
            details=details
        )
    
    def log_use_case_execution(self, use_case: str, command: str, duration: float):
        """Log use case execution."""
        self.info(
            f"Use Case Executed: {use_case}",
            use_case=use_case,
            command=command,
            duration=duration
        )
    
    def log_repository_operation(self, operation: str, entity_type: str, entity_id: str, success: bool):
        """Log repository operations."""
        level = "info" if success else "error"
        getattr(self, level)(
            f"Repository Operation: {operation}",
            operation=operation,
            entity_type=entity_type,
            entity_id=entity_id,
            success=success
        )
    
    def log_service_call(self, service: str, operation: str, duration: float, success: bool):
        """Log service calls."""
        level = "info" if success else "error"
        getattr(self, level)(
            f"Service Call: {service}.{operation}",
            service=service,
            operation=operation,
            duration=duration,
            success=success
        )


# Global logger instance
_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Get global logger instance."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger


def setup_logging(config: Dict[str, Any] = None):
    """Set up global logging configuration."""
    global _logger
    
    config = config or {}
    _logger = StructuredLogger(
        console_enabled=config.get('console_enabled', True),
        file_enabled=config.get('file_enabled', True),
        file_path=config.get('file_path', 'logs/playlista.log'),
        level=config.get('level', 'INFO')
    )
    
    # Configure TensorFlow logging to suppress warnings
    try:
        import os
        # Configure TensorFlow logging BEFORE any imports
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
        import tensorflow as tf
        
        # Aggressive TensorFlow warning suppression
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF warnings
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
        os.environ['TF_GPU_ALLOCATOR'] = 'cpu'  # Disable GPU warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage
        
        # Configure TensorFlow logger
        tf_logger = tf.get_logger()
        tf_logger.setLevel(logging.ERROR)  # Only show ERROR level
        
        # Suppress TensorFlow warnings about network creation
        tf.autograph.set_verbosity(0)
        
        # Suppress all TensorFlow warnings
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
        
    except ImportError:
        # TensorFlow not available, skip configuration
        pass


def log_universal(message: str, level: str = "info", **kwargs):
    """Universal logging function."""
    logger_instance = get_logger()
    getattr(logger_instance, level.lower())(message, **kwargs) 