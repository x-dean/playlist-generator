"""
Logging helpers for clean structured logging.
"""

import logging
import time
import uuid
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredLogger:
    """Helper for structured logging with automatic metadata."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._operation_stack = []
    
    def _add_metadata(self, **kwargs) -> Dict[str, Any]:
        """Add metadata to log record."""
        metadata = {}
        
        # Add operation context if available
        if self._operation_stack:
            metadata['operation_context'] = self._operation_stack[-1]
        
        # Add provided metadata
        metadata.update(kwargs)
        
        return metadata
    
    def _log_with_metadata(self, level: int, message: str, **metadata):
        """Log message with structured metadata."""
        extra_metadata = self._add_metadata(**metadata)
        self.logger.log(level, message, extra={'extra_fields': extra_metadata})
    
    def debug(self, message: str, **metadata):
        """Log debug message with metadata."""
        self._log_with_metadata(logging.DEBUG, message, **metadata)
    
    def info(self, message: str, **metadata):
        """Log info message with metadata."""
        self._log_with_metadata(logging.INFO, message, **metadata)
    
    def warning(self, message: str, **metadata):
        """Log warning message with metadata."""
        self._log_with_metadata(logging.WARNING, message, **metadata)
    
    def error(self, message: str, **metadata):
        """Log error message with metadata."""
        self._log_with_metadata(logging.ERROR, message, **metadata)
    
    def critical(self, message: str, **metadata):
        """Log critical message with metadata."""
        self._log_with_metadata(logging.CRITICAL, message, **metadata)


class DatabaseLogger(StructuredLogger):
    """Specialized logger for database operations."""
    
    def save_operation(self, entity_type: str, entity_id: str, **metadata):
        """Log database save operation."""
        self.info(f"Database save operation", 
                 operation_type='save',
                 entity_type=entity_type,
                 entity_id=entity_id,
                 **metadata)
    
    def find_operation(self, entity_type: str, entity_id: str, **metadata):
        """Log database find operation."""
        self.info(f"Database find operation",
                 operation_type='find',
                 entity_type=entity_type,
                 entity_id=entity_id,
                 **metadata)
    
    def delete_operation(self, entity_type: str, entity_id: str, **metadata):
        """Log database delete operation."""
        self.info(f"Database delete operation",
                 operation_type='delete',
                 entity_type=entity_type,
                 entity_id=entity_id,
                 **metadata)
    
    def query_operation(self, query_type: str, table: str, **metadata):
        """Log database query operation."""
        self.debug(f"Database query operation",
                  query_type=query_type,
                  table=table,
                  **metadata)
    
    def operation_success(self, duration_ms: int, **metadata):
        """Log successful operation."""
        self.info(f"Operation completed successfully",
                 success=True,
                 duration_ms=duration_ms,
                 **metadata)
    
    def operation_failed(self, error_type: str, error_message: str, duration_ms: int, **metadata):
        """Log failed operation."""
        self.error(f"Operation failed: {error_message}",
                  success=False,
                  error_type=error_type,
                  error_message=error_message,
                  duration_ms=duration_ms,
                  **metadata)


class AnalysisLogger(StructuredLogger):
    """Specialized logger for analysis operations."""
    
    def start_analysis(self, file_path: Path, **metadata):
        """Log analysis start."""
        self.info(f"Starting analysis",
                 analysis_phase='start',
                 file_path=str(file_path),
                 **metadata)
    
    def analysis_phase(self, phase: str, file_path: Path, **metadata):
        """Log analysis phase."""
        self.debug(f"Analysis phase: {phase}",
                  analysis_phase=phase,
                  file_path=str(file_path),
                  **metadata)
    
    def analysis_complete(self, file_path: Path, duration_ms: int, **metadata):
        """Log analysis completion."""
        self.info(f"Analysis completed",
                 analysis_phase='complete',
                 file_path=str(file_path),
                 duration_ms=duration_ms,
                 success=True,
                 **metadata)
    
    def analysis_failed(self, file_path: Path, error_message: str, duration_ms: int, **metadata):
        """Log analysis failure."""
        self.error(f"Analysis failed: {error_message}",
                  analysis_phase='failed',
                  file_path=str(file_path),
                  duration_ms=duration_ms,
                  success=False,
                  **metadata)


class PerformanceLogger(StructuredLogger):
    """Specialized logger for performance monitoring."""
    
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
        self._timers = {}
    
    def start_timer(self, operation_id: str, operation_name: str, **metadata):
        """Start timing an operation."""
        self._timers[operation_id] = {
            'start_time': time.time(),
            'operation_name': operation_name,
            'metadata': metadata
        }
        self.debug(f"Started timing: {operation_name}",
                  operation_id=operation_id,
                  operation_name=operation_name,
                  **metadata)
    
    def end_timer(self, operation_id: str, **metadata):
        """End timing an operation."""
        if operation_id in self._timers:
            timer_info = self._timers[operation_id]
            duration = time.time() - timer_info['start_time']
            duration_ms = int(duration * 1000)
            
            # Merge metadata
            all_metadata = {**timer_info['metadata'], **metadata}
            
            self.info(f"Completed: {timer_info['operation_name']}",
                     operation_id=operation_id,
                     operation_name=timer_info['operation_name'],
                     duration_ms=duration_ms,
                     **all_metadata)
            
            del self._timers[operation_id]
            return duration_ms
        return None


def get_structured_logger(name: str = 'playlista') -> StructuredLogger:
    """Get a structured logger instance."""
    logger = logging.getLogger(name)
    return StructuredLogger(logger)


def get_database_logger(name: str = 'playlista.database') -> DatabaseLogger:
    """Get a database logger instance."""
    logger = logging.getLogger(name)
    return DatabaseLogger(logger)


def get_analysis_logger(name: str = 'playlista.analysis') -> AnalysisLogger:
    """Get an analysis logger instance."""
    logger = logging.getLogger(name)
    return AnalysisLogger(logger)


def get_performance_logger(name: str = 'playlista.performance') -> PerformanceLogger:
    """Get a performance logger instance."""
    logger = logging.getLogger(name)
    return PerformanceLogger(logger) 