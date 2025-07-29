"""
Logging formatters for structured and colored output.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from colorlog import ColoredFormatter as BaseColoredFormatter
except ImportError:
    # Fallback if colorlog is not available
    class BaseColoredFormatter(logging.Formatter):
        def format(self, record):
            return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Structured formatter for JSON log output."""
    
    def __init__(self, include_timestamp: bool = True, include_level: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'message': record.getMessage(),
            'logger': record.name,
            'level': record.levelname,
            'timestamp': getattr(record, 'timestamp', datetime.fromtimestamp(record.created).isoformat()),
            'module': getattr(record, 'module_name', record.module),
            'function': getattr(record, 'function_name', record.funcName),
            'line': getattr(record, 'line_number', record.lineno)
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_entry['correlation_id'] = record.correlation_id
        
        # Add extra fields from processors
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add process and thread info
        log_entry['process_id'] = getattr(record, 'process_id', record.process)
        log_entry['thread_id'] = getattr(record, 'thread_id', record.thread)
        log_entry['thread_name'] = getattr(record, 'thread_name', record.threadName)
        
        # Add database-specific fields
        if hasattr(record, 'db_operation'):
            log_entry['db_operation'] = record.db_operation
        if hasattr(record, 'db_entity'):
            log_entry['db_entity'] = record.db_entity
        if hasattr(record, 'db_entity_id'):
            log_entry['db_entity_id'] = record.db_entity_id
        if hasattr(record, 'db_query_type'):
            log_entry['db_query_type'] = record.db_query_type
        if hasattr(record, 'db_table'):
            log_entry['db_table'] = record.db_table
        if hasattr(record, 'db_duration_ms'):
            log_entry['db_duration_ms'] = record.db_duration_ms
        if hasattr(record, 'db_success'):
            log_entry['db_success'] = record.db_success
        if hasattr(record, 'db_rows_affected'):
            log_entry['db_rows_affected'] = record.db_rows_affected
        if hasattr(record, 'db_error_type'):
            log_entry['db_error_type'] = record.db_error_type
        
        # Add analysis-specific fields
        if hasattr(record, 'analysis_phase'):
            log_entry['analysis_phase'] = record.analysis_phase
        if hasattr(record, 'analysis_file_path'):
            log_entry['analysis_file_path'] = record.analysis_file_path
        if hasattr(record, 'analysis_file_size_mb'):
            log_entry['analysis_file_size_mb'] = record.analysis_file_size_mb
        if hasattr(record, 'analysis_mode'):
            log_entry['analysis_mode'] = record.analysis_mode
        if hasattr(record, 'analysis_duration_ms'):
            log_entry['analysis_duration_ms'] = record.analysis_duration_ms
        if hasattr(record, 'analysis_success'):
            log_entry['analysis_success'] = record.analysis_success
        
        # Add performance-specific fields
        if hasattr(record, 'performance_duration_ms'):
            log_entry['performance_duration_ms'] = record.performance_duration_ms
        if hasattr(record, 'operation_id'):
            log_entry['operation_id'] = record.operation_id
        if hasattr(record, 'operation_name'):
            log_entry['operation_name'] = record.operation_name
        
        # Add duration if available
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class ColoredFormatter(BaseColoredFormatter):
    """Colored formatter for console output."""
    
    def __init__(self, fmt: Optional[str] = None):
        if fmt is None:
            fmt = '%(log_color)s%(asctime)s [%(correlation_id)s] %(name)s - %(levelname)s - %(message)s'
        
        # Define colors for different log levels
        colors = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        }
        
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S', log_colors=colors)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and correlation ID."""
        # Ensure correlation_id is available
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON formatter with additional metadata."""
    
    def __init__(self, include_metadata: bool = True):
        super().__init__()
        self.include_metadata = include_metadata
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with metadata."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'thread_name': record.threadName
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_entry['correlation_id'] = record.correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'correlation_id']:
                log_entry[f'extra_{key}'] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class CompactFormatter(logging.Formatter):
    """Compact formatter for high-volume logging."""
    
    def __init__(self):
        super().__init__('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record in compact format."""
        # Add correlation ID if available
        if hasattr(record, 'correlation_id') and record.correlation_id:
            record.correlation_id = record.correlation_id[:8]  # Truncate for compactness
        else:
            record.correlation_id = 'N/A'
        
        return super().format(record)


class PerformanceFormatter(logging.Formatter):
    """Formatter optimized for performance logging."""
    
    def __init__(self):
        super().__init__('%(asctime)s [PERF] %(name)s - %(message)s')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format performance log record."""
        # Add timing information if available
        if hasattr(record, 'duration_ms'):
            record.duration_ms = f"{record.duration_ms:.2f}ms"
        else:
            record.duration_ms = "N/A"
        
        # Add memory usage if available
        if hasattr(record, 'memory_mb'):
            record.memory_mb = f"{record.memory_mb:.1f}MB"
        else:
            record.memory_mb = "N/A"
        
        return super().format(record) 