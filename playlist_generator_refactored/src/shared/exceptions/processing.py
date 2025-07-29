"""
Unified error handling for processing operations.

This module provides comprehensive error handling and aggregation
for the discovery-analysis pipeline.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    FILE_ACCESS = "file_access"
    PROCESSING = "processing"
    DATABASE = "database"
    NETWORK = "network"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ProcessingError:
    """Represents a processing error with context."""
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    file_path: Optional[str] = None
    service: Optional[str] = None
    operation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate error data."""
        if not self.error_type:
            raise ValueError("Error type is required")
        if not self.message:
            raise ValueError("Error message is required")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'file_path': self.file_path,
            'service': self.service,
            'operation': self.operation,
            'timestamp': self.timestamp.isoformat(),
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'context': self.context
        }
    
    def can_retry(self) -> bool:
        """Check if this error can be retried."""
        return self.retry_count < self.max_retries and self.severity != ErrorSeverity.CRITICAL
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1


@dataclass
class ErrorAggregator:
    """Aggregates and manages processing errors."""
    errors: List[ProcessingError] = field(default_factory=list)
    warnings: List[ProcessingError] = field(default_factory=list)
    
    def add_error(self, error: ProcessingError) -> None:
        """Add an error to the aggregator."""
        self.errors.append(error)
    
    def add_warning(self, warning: ProcessingError) -> None:
        """Add a warning to the aggregator."""
        self.warnings.append(warning)
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ProcessingError]:
        """Get all errors of a specific category."""
        return [error for error in self.errors if error.category == category]
    
    def get_errors_by_service(self, service: str) -> List[ProcessingError]:
        """Get all errors from a specific service."""
        return [error for error in self.errors if error.service == service]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ProcessingError]:
        """Get all errors of a specific severity."""
        return [error for error in self.errors if error.severity == severity]
    
    def get_retryable_errors(self) -> List[ProcessingError]:
        """Get all errors that can be retried."""
        return [error for error in self.errors if error.can_retry()]
    
    def get_critical_errors(self) -> List[ProcessingError]:
        """Get all critical errors."""
        return [error for error in self.errors if error.severity == ErrorSeverity.CRITICAL]
    
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return len(self.get_critical_errors()) > 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors."""
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        
        # Count by category
        category_counts = {}
        for category in ErrorCategory:
            category_counts[category.value] = len(self.get_errors_by_category(category))
        
        # Count by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = len(self.get_errors_by_severity(severity))
        
        # Count by service
        service_counts = {}
        services = set(error.service for error in self.errors if error.service)
        for service in services:
            service_counts[service] = len(self.get_errors_by_service(service))
        
        return {
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'critical_errors': len(self.get_critical_errors()),
            'retryable_errors': len(self.get_retryable_errors()),
            'category_counts': category_counts,
            'severity_counts': severity_counts,
            'service_counts': service_counts,
            'has_critical_errors': self.has_critical_errors()
        }
    
    def clear_errors(self) -> None:
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'errors': [error.to_dict() for error in self.errors],
            'warnings': [warning.to_dict() for warning in self.warnings],
            'summary': self.get_error_summary()
        }


class UnifiedErrorHandler:
    """Unified error handler for processing operations."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.aggregator = ErrorAggregator()
        self.logger = None  # Will be set by the service
    
    def set_logger(self, logger):
        """Set the logger for error reporting."""
        self.logger = logger
    
    def handle_error(self, 
                    error_type: str,
                    message: str,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    file_path: Optional[str] = None,
                    service: Optional[str] = None,
                    operation: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> ProcessingError:
        """Handle an error and add it to the aggregator."""
        error = ProcessingError(
            error_type=error_type,
            message=message,
            severity=severity,
            category=category,
            file_path=file_path,
            service=service,
            operation=operation,
            context=context or {}
        )
        
        self.aggregator.add_error(error)
        
        # Log the error
        if self.logger:
            log_message = f"[{service}:{operation}] {message}"
            if file_path:
                log_message += f" (File: {file_path})"
            
            if severity == ErrorSeverity.CRITICAL:
                self.logger.error(log_message)
            elif severity == ErrorSeverity.HIGH:
                self.logger.error(log_message)
            elif severity == ErrorSeverity.MEDIUM:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
        
        return error
    
    def handle_warning(self,
                      warning_type: str,
                      message: str,
                      category: ErrorCategory = ErrorCategory.UNKNOWN,
                      file_path: Optional[str] = None,
                      service: Optional[str] = None,
                      operation: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> ProcessingError:
        """Handle a warning and add it to the aggregator."""
        warning = ProcessingError(
            error_type=warning_type,
            message=message,
            severity=ErrorSeverity.LOW,
            category=category,
            file_path=file_path,
            service=service,
            operation=operation,
            context=context or {}
        )
        
        self.aggregator.add_warning(warning)
        
        # Log the warning
        if self.logger:
            log_message = f"[{service}:{operation}] WARNING: {message}"
            if file_path:
                log_message += f" (File: {file_path})"
            self.logger.warning(log_message)
        
        return warning
    
    def get_aggregator(self) -> ErrorAggregator:
        """Get the error aggregator."""
        return self.aggregator
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.aggregator.errors) > 0
    
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return self.aggregator.has_critical_errors()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return self.aggregator.get_error_summary()
    
    def clear_errors(self) -> None:
        """Clear all errors."""
        self.aggregator.clear_errors()


# Global error handler instance
_error_handler: Optional[UnifiedErrorHandler] = None


def get_error_handler() -> UnifiedErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = UnifiedErrorHandler()
    return _error_handler


def set_error_handler_logger(logger):
    """Set the logger for the global error handler."""
    handler = get_error_handler()
    handler.set_logger(logger) 