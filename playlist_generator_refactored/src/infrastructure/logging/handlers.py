"""
Custom logging handlers for different output destinations.
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from infrastructure.logging.formatters import StructuredFormatter, ColoredFormatter, JsonFormatter


class FileHandler(logging.handlers.RotatingFileHandler):
    """Enhanced file handler with structured logging."""
    
    def __init__(
        self,
        filename: Path,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: str = 'utf-8',
        formatter: Optional[logging.Formatter] = None
    ):
        super().__init__(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding
        )
        
        if formatter is None:
            formatter = StructuredFormatter()
        
        self.setFormatter(formatter)


class ConsoleHandler(logging.StreamHandler):
    """Enhanced console handler with colored output."""
    
    def __init__(
        self,
        stream=None,
        formatter: Optional[logging.Formatter] = None
    ):
        if stream is None:
            stream = sys.stdout
        
        super().__init__(stream)
        
        if formatter is None:
            formatter = ColoredFormatter()
        
        self.setFormatter(formatter)


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Enhanced rotating file handler with structured logging."""
    
    def __init__(
        self,
        filename: Path,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: str = 'utf-8',
        formatter: Optional[logging.Formatter] = None
    ):
        super().__init__(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding
        )
        
        if formatter is None:
            formatter = StructuredFormatter()
        
        self.setFormatter(formatter)


class TimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Enhanced timed rotating file handler."""
    
    def __init__(
        self,
        filename: Path,
        when: str = 'midnight',
        interval: int = 1,
        backup_count: int = 30,
        encoding: str = 'utf-8',
        formatter: Optional[logging.Formatter] = None
    ):
        super().__init__(
            filename=filename,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding=encoding
        )
        
        if formatter is None:
            formatter = StructuredFormatter()
        
        self.setFormatter(formatter)


class MemoryHandler(logging.handlers.MemoryHandler):
    """Memory handler for buffering log records."""
    
    def __init__(
        self,
        capacity: int = 1000,
        flush_level: int = logging.ERROR,
        target: Optional[logging.Handler] = None,
        formatter: Optional[logging.Formatter] = None
    ):
        super().__init__(
            capacity=capacity,
            flushLevel=flush_level,
            target=target
        )
        
        if formatter is None:
            formatter = StructuredFormatter()
        
        self.setFormatter(formatter)


class QueueHandler(logging.handlers.QueueHandler):
    """Queue handler for asynchronous logging."""
    
    def __init__(
        self,
        queue,
        formatter: Optional[logging.Formatter] = None
    ):
        super().__init__(queue)
        
        if formatter is None:
            formatter = StructuredFormatter()
        
        self.setFormatter(formatter)


class NullHandler(logging.NullHandler):
    """Null handler for suppressing log output."""
    
    def __init__(self):
        super().__init__()


class PerformanceHandler(logging.Handler):
    """Handler for performance metrics logging."""
    
    def __init__(self, filename: Optional[Path] = None):
        super().__init__()
        
        if filename:
            self.filename = filename
            self.file_handler = logging.handlers.RotatingFileHandler(
                filename,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3
            )
            self.file_handler.setFormatter(JsonFormatter())
        else:
            self.filename = None
            self.file_handler = None
    
    def emit(self, record: logging.LogRecord):
        """Emit performance log record."""
        # Add performance metadata
        record.performance_metrics = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        
        # Add timing if available
        if hasattr(record, 'duration_ms'):
            record.performance_metrics['duration_ms'] = record.duration_ms
        
        # Add memory usage if available
        if hasattr(record, 'memory_mb'):
            record.performance_metrics['memory_mb'] = record.memory_mb
        
        # Write to file if configured
        if self.file_handler:
            self.file_handler.emit(record)


class ErrorHandler(logging.Handler):
    """Handler for error logging with additional context."""
    
    def __init__(self, filename: Optional[Path] = None):
        super().__init__()
        
        if filename:
            self.filename = filename
            self.file_handler = logging.handlers.RotatingFileHandler(
                filename,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=5
            )
            self.file_handler.setFormatter(JsonFormatter())
        else:
            self.filename = None
            self.file_handler = None
    
    def emit(self, record: logging.LogRecord):
        """Emit error log record with context."""
        # Add error context
        record.error_context = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            record.error_context['exception'] = self.formatException(record.exc_info)
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id') and record.correlation_id:
            record.error_context['correlation_id'] = record.correlation_id
        
        # Write to file if configured
        if self.file_handler:
            self.file_handler.emit(record)


class AuditHandler(logging.Handler):
    """Handler for audit logging."""
    
    def __init__(self, filename: Optional[Path] = None):
        super().__init__()
        
        if filename:
            self.filename = filename
            self.file_handler = logging.handlers.TimedRotatingFileHandler(
                filename,
                when='midnight',
                interval=1,
                backupCount=90  # Keep 90 days of audit logs
            )
            self.file_handler.setFormatter(JsonFormatter())
        else:
            self.filename = None
            self.file_handler = None
    
    def emit(self, record: logging.LogRecord):
        """Emit audit log record."""
        # Add audit metadata
        record.audit_info = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'user_id': getattr(record, 'user_id', 'unknown'),
            'action': getattr(record, 'action', 'unknown'),
            'resource': getattr(record, 'resource', 'unknown'),
            'ip_address': getattr(record, 'ip_address', 'unknown')
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id') and record.correlation_id:
            record.audit_info['correlation_id'] = record.correlation_id
        
        # Write to file if configured
        if self.file_handler:
            self.file_handler.emit(record) 