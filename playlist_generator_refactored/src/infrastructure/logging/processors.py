"""
Logging processors for automatic field addition and structured logging.
"""

import logging
import time
import threading
from typing import Any, Dict, Optional
from datetime import datetime


class StructuredLogProcessor(logging.Processor):
    """Processor that automatically adds structured fields to log records."""
    
    def __init__(self):
        super().__init__()
        self._start_times = {}
    
    def process(self, record: logging.LogRecord) -> logging.LogRecord:
        """Process log record to add structured fields."""
        # Add timestamp
        record.timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Add duration if this is a continuation of an operation
        if hasattr(record, 'operation_id') and record.operation_id in self._start_times:
            duration = time.time() - self._start_times[record.operation_id]
            record.duration_ms = int(duration * 1000)
        
        # Add thread info
        record.thread_name = record.threadName
        record.thread_id = record.thread
        
        # Add process info
        record.process_id = record.process
        
        # Add module and function info
        record.module_name = record.module
        record.function_name = record.funcName
        record.line_number = record.lineno
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            record.correlation_id = record.correlation_id
        
        # Add structured fields from extra
        if hasattr(record, 'extra_fields'):
            for key, value in record.extra_fields.items():
                setattr(record, key, value)
        
        return record


class DatabaseLogProcessor(StructuredLogProcessor):
    """Processor specifically for database operations."""
    
    def process(self, record: logging.LogRecord) -> logging.LogRecord:
        """Process database log record."""
        # Call parent processor first
        record = super().process(record)
        
        # Add database-specific fields
        if hasattr(record, 'operation_type'):
            record.db_operation = record.operation_type
        
        if hasattr(record, 'entity_type'):
            record.db_entity = record.entity_type
        
        if hasattr(record, 'entity_id'):
            record.db_entity_id = record.entity_id
        
        if hasattr(record, 'query_type'):
            record.db_query_type = record.query_type
        
        if hasattr(record, 'table'):
            record.db_table = record.table
        
        if hasattr(record, 'duration_ms'):
            record.db_duration_ms = record.duration_ms
        
        if hasattr(record, 'success'):
            record.db_success = record.success
        
        if hasattr(record, 'rows_affected'):
            record.db_rows_affected = record.rows_affected
        
        if hasattr(record, 'error_type'):
            record.db_error_type = record.error_type
        
        return record


class AnalysisLogProcessor(StructuredLogProcessor):
    """Processor specifically for analysis operations."""
    
    def process(self, record: logging.LogRecord) -> logging.LogRecord:
        """Process analysis log record."""
        # Call parent processor first
        record = super().process(record)
        
        # Add analysis-specific fields
        if hasattr(record, 'analysis_phase'):
            record.analysis_phase = record.analysis_phase
        
        if hasattr(record, 'file_path'):
            record.analysis_file_path = record.file_path
        
        if hasattr(record, 'file_size_mb'):
            record.analysis_file_size_mb = record.file_size_mb
        
        if hasattr(record, 'processing_mode'):
            record.analysis_mode = record.processing_mode
        
        if hasattr(record, 'duration_ms'):
            record.analysis_duration_ms = record.duration_ms
        
        if hasattr(record, 'success'):
            record.analysis_success = record.success
        
        return record


class PerformanceLogProcessor(StructuredLogProcessor):
    """Processor for performance monitoring."""
    
    def __init__(self):
        super().__init__()
        self._operation_timers = {}
    
    def start_timer(self, operation_id: str) -> None:
        """Start timing an operation."""
        self._operation_timers[operation_id] = time.time()
    
    def end_timer(self, operation_id: str) -> Optional[float]:
        """End timing an operation and return duration."""
        if operation_id in self._operation_timers:
            duration = time.time() - self._operation_timers[operation_id]
            del self._operation_timers[operation_id]
            return duration
        return None
    
    def process(self, record: logging.LogRecord) -> logging.LogRecord:
        """Process performance log record."""
        # Call parent processor first
        record = super().process(record)
        
        # Add performance-specific fields
        if hasattr(record, 'operation_id') and record.operation_id in self._operation_timers:
            duration = self.end_timer(record.operation_id)
            if duration:
                record.performance_duration_ms = int(duration * 1000)
        
        return record


# Global processors
_structured_processor = StructuredLogProcessor()
_database_processor = DatabaseLogProcessor()
_analysis_processor = AnalysisLogProcessor()
_performance_processor = PerformanceLogProcessor()


def get_structured_processor() -> StructuredLogProcessor:
    """Get the global structured log processor."""
    return _structured_processor


def get_database_processor() -> DatabaseLogProcessor:
    """Get the global database log processor."""
    return _database_processor


def get_analysis_processor() -> AnalysisLogProcessor:
    """Get the global analysis log processor."""
    return _analysis_processor


def get_performance_processor() -> PerformanceLogProcessor:
    """Get the global performance log processor."""
    return _performance_processor 