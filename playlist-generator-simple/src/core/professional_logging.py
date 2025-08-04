"""
Professional logging system for Playlist Generator Simple.

This module provides:
- Structured logging with consistent formatting
- Performance tracking and metrics
- Security and audit logging
- Configurable log levels and outputs
- Integration with monitoring systems
- Thread-safe operations
- Error context and tracing
"""

import logging
import logging.handlers
import json
import time
import threading
import traceback
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from enum import Enum
import uuid


class LogLevel(Enum):
    """Standard log levels with numeric values."""
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    TRACE = 5


class LogCategory(Enum):
    """Log categories for better organization."""
    SYSTEM = "system"
    AUDIO = "audio"
    DATABASE = "database"
    API = "api"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    EXTERNAL = "external"


@dataclass
class LogContext:
    """Structured context for log entries."""
    timestamp: str
    level: str
    category: str
    component: str
    message: str
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self, logger: 'ProfessionalLogger'):
        self.logger = logger
    
    def authentication_attempt(self, user_id: str, success: bool, ip_address: str = None):
        """Log authentication attempts."""
        self.logger.log(
            LogLevel.INFO if success else LogLevel.WARNING,
            LogCategory.SECURITY,
            "authentication",
            f"Authentication {'successful' if success else 'failed'} for user {user_id}",
            metadata={"ip_address": ip_address, "success": success}
        )
    
    def access_denied(self, user_id: str, resource: str, reason: str = None):
        """Log access denied events."""
        self.logger.log(
            LogLevel.WARNING,
            LogCategory.SECURITY,
            "authorization",
            f"Access denied for user {user_id} to resource {resource}",
            metadata={"resource": resource, "reason": reason}
        )
    
    def suspicious_activity(self, description: str, severity: str = "medium", **metadata):
        """Log suspicious activities."""
        self.logger.log(
            LogLevel.ERROR if severity == "high" else LogLevel.WARNING,
            LogCategory.SECURITY,
            "threat_detection",
            f"Suspicious activity detected: {description}",
            metadata={"severity": severity, **metadata}
        )


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, logger: 'ProfessionalLogger'):
        self.logger = logger
    
    @contextmanager
    def time_operation(self, operation: str, component: str, **metadata):
        """Context manager for timing operations."""
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        self.logger.log(
            LogLevel.DEBUG,
            LogCategory.PERFORMANCE,
            component,
            f"Starting operation: {operation}",
            correlation_id=correlation_id,
            operation=operation,
            metadata=metadata
        )
        
        try:
            yield correlation_id
            duration_ms = (time.time() - start_time) * 1000
            
            self.logger.log(
                LogLevel.INFO,
                LogCategory.PERFORMANCE,
                component,
                f"Operation completed: {operation}",
                correlation_id=correlation_id,
                operation=operation,
                duration_ms=duration_ms,
                metadata={**metadata, "status": "success"}
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.logger.log(
                LogLevel.ERROR,
                LogCategory.PERFORMANCE,
                component,
                f"Operation failed: {operation}",
                correlation_id=correlation_id,
                operation=operation,
                duration_ms=duration_ms,
                error_code=type(e).__name__,
                stack_trace=traceback.format_exc(),
                metadata={**metadata, "status": "error", "error": str(e)}
            )
            raise
    
    def log_metric(self, metric_name: str, value: Union[int, float], 
                  component: str, unit: str = None, **metadata):
        """Log performance metrics."""
        self.logger.log(
            LogLevel.INFO,
            LogCategory.PERFORMANCE,
            component,
            f"Metric: {metric_name} = {value}" + (f" {unit}" if unit else ""),
            metadata={
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                **metadata
            }
        )


class CustomJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract LogContext if present
        if hasattr(record, 'log_context'):
            context = record.log_context
            return json.dumps(context.to_dict(), default=str, ensure_ascii=False)
        
        # Fallback for standard log records
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "category": "system",
            "component": record.name,
            "message": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }
        
        if record.exc_info:
            log_entry["stack_trace"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        'CRITICAL': '\033[95m',  # Magenta
        'ERROR': '\033[91m',     # Red
        'WARNING': '\033[93m',   # Yellow
        'INFO': '\033[92m',      # Green
        'DEBUG': '\033[94m',     # Blue
        'TRACE': '\033[96m',     # Cyan
    }
    
    CATEGORY_COLORS = {
        'system': '\033[90m',      # Gray
        'audio': '\033[94m',       # Blue
        'database': '\033[95m',    # Magenta
        'api': '\033[93m',         # Yellow
        'security': '\033[91m',    # Red
        'performance': '\033[96m', # Cyan
        'business': '\033[92m',    # Green
        'external': '\033[97m',    # White
    }
    
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if hasattr(record, 'log_context'):
            context = record.log_context
            
            # Colorize level
            level_color = self.COLORS.get(context.level, '')
            colored_level = f"{level_color}{context.level:<8}{self.RESET}"
            
            # Colorize category
            category_color = self.CATEGORY_COLORS.get(context.category, '')
            colored_category = f"{category_color}{context.category}{self.RESET}"
            
            # Build formatted message
            parts = [
                f"{context.timestamp}",
                colored_level,
                f"{colored_category}:{context.component}",
                context.message
            ]
            
            if context.duration_ms is not None:
                parts.append(f"({context.duration_ms:.2f}ms)")
            
            return " | ".join(parts)
        
        # Fallback formatting
        return super().format(record)


class ProfessionalLogger:
    """
    Professional logging system with structured output, performance tracking,
    and comprehensive error handling.
    """
    
    def __init__(self, 
                 name: str = "playlista",
                 level: LogLevel = LogLevel.INFO,
                 console_enabled: bool = True,
                 file_enabled: bool = True,
                 json_enabled: bool = False,
                 log_dir: str = "logs",
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 backup_count: int = 10,
                 include_performance: bool = True,
                 include_security: bool = True):
        """
        Initialize professional logger.
        
        Args:
            name: Logger name
            level: Default log level
            console_enabled: Enable console output
            file_enabled: Enable file output
            json_enabled: Enable JSON formatting
            log_dir: Directory for log files
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            include_performance: Include performance logging
            include_security: Include security logging
        """
        self.name = name
        self.level = level
        self.console_enabled = console_enabled
        self.file_enabled = file_enabled
        self.json_enabled = json_enabled
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Thread-local storage for context
        self._context = threading.local()
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Initialize specialized loggers
        self.security = SecurityLogger(self) if include_security else None
        self.performance = PerformanceLogger(self) if include_performance else None
        
        # Log system initialization
        self.log(LogLevel.INFO, LogCategory.SYSTEM, "logging", 
                f"Professional logging system initialized: {name}")
    
    def _setup_handlers(self):
        """Setup log handlers."""
        # Console handler
        if self.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            if self.json_enabled:
                console_handler.setFormatter(CustomJSONFormatter())
            else:
                console_handler.setFormatter(ColoredConsoleFormatter())
            self.logger.addHandler(console_handler)
        
        # File handlers
        if self.file_enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Main log file
            main_log_file = self.log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            
            if self.json_enabled:
                file_handler.setFormatter(CustomJSONFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
                ))
            
            self.logger.addHandler(file_handler)
            
            # Error log file (ERROR and above only)
            error_log_file = self.log_dir / f"{self.name}-errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            error_handler.setLevel(LogLevel.ERROR.value)
            error_handler.setFormatter(CustomJSONFormatter())
            self.logger.addHandler(error_handler)
    
    def set_context(self, **context):
        """Set logging context for current thread."""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        self._context.data.update(context)
    
    def clear_context(self):
        """Clear logging context for current thread."""
        if hasattr(self._context, 'data'):
            self._context.data.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        if hasattr(self._context, 'data'):
            return self._context.data.copy()
        return {}
    
    def log(self, 
            level: LogLevel,
            category: LogCategory,
            component: str,
            message: str,
            correlation_id: str = None,
            user_id: str = None,
            session_id: str = None,
            operation: str = None,
            duration_ms: float = None,
            error_code: str = None,
            stack_trace: str = None,
            metadata: Dict[str, Any] = None):
        """
        Log a structured message.
        
        Args:
            level: Log level
            category: Log category
            component: Component name
            message: Log message
            correlation_id: Correlation ID for tracking
            user_id: User ID if applicable
            session_id: Session ID if applicable
            operation: Operation name if applicable
            duration_ms: Duration in milliseconds
            error_code: Error code if applicable
            stack_trace: Stack trace if applicable
            metadata: Additional metadata
        """
        # Get current context
        context_data = self.get_context()
        
        # Create log context
        log_context = LogContext(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.name,
            category=category.value,
            component=component,
            message=message,
            correlation_id=correlation_id or context_data.get('correlation_id'),
            user_id=user_id or context_data.get('user_id'),
            session_id=session_id or context_data.get('session_id'),
            operation=operation or context_data.get('operation'),
            duration_ms=duration_ms,
            error_code=error_code,
            stack_trace=stack_trace,
            metadata={**(metadata or {}), **context_data.get('metadata', {})}
        )
        
        # Create log record
        record = logging.LogRecord(
            name=self.logger.name,
            level=level.value,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Attach context
        record.log_context = log_context
        
        # Log the record
        self.logger.handle(record)
    
    def trace(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log trace message."""
        self.log(LogLevel.TRACE, category, component, message, **kwargs)
    
    def debug(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, category, component, message, **kwargs)
    
    def info(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, category, component, message, **kwargs)
    
    def warning(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, category, component, message, **kwargs)
    
    def error(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, category, component, message, **kwargs)
    
    def critical(self, category: LogCategory, component: str, message: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, category, component, message, **kwargs)
    
    def log_exception(self, category: LogCategory, component: str, 
                     message: str, exception: Exception = None, **kwargs):
        """Log exception with full context."""
        if exception is None:
            # Get current exception
            exception = sys.exc_info()[1]
        
        self.log(
            LogLevel.ERROR,
            category,
            component,
            message,
            error_code=type(exception).__name__ if exception else "UnknownError",
            stack_trace=traceback.format_exc(),
            metadata={**(kwargs.get('metadata', {})), 
                     "exception_type": type(exception).__name__ if exception else None,
                     "exception_message": str(exception) if exception else None},
            **{k: v for k, v in kwargs.items() if k != 'metadata'}
        )
    
    @contextmanager
    def operation_context(self, operation: str, component: str, **context):
        """Context manager for operation tracking."""
        correlation_id = str(uuid.uuid4())
        
        # Set context
        self.set_context(
            correlation_id=correlation_id,
            operation=operation,
            **context
        )
        
        start_time = time.time()
        
        self.debug(
            LogCategory.SYSTEM,
            component,
            f"Starting operation: {operation}",
            correlation_id=correlation_id,
            operation=operation
        )
        
        try:
            yield correlation_id
            
            duration_ms = (time.time() - start_time) * 1000
            self.info(
                LogCategory.SYSTEM,
                component,
                f"Operation completed: {operation}",
                correlation_id=correlation_id,
                operation=operation,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log_exception(
                LogCategory.SYSTEM,
                component,
                f"Operation failed: {operation}",
                exception=e,
                correlation_id=correlation_id,
                operation=operation,
                duration_ms=duration_ms
            )
            raise
        finally:
            self.clear_context()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "name": self.name,
            "level": self.level.name,
            "handlers": len(self.logger.handlers),
            "console_enabled": self.console_enabled,
            "file_enabled": self.file_enabled,
            "json_enabled": self.json_enabled,
            "log_dir": str(self.log_dir),
            "performance_logging": self.performance is not None,
            "security_logging": self.security is not None
        }


# Global logger instance
_global_logger: Optional[ProfessionalLogger] = None
_logger_lock = threading.Lock()


def get_logger(name: str = None) -> ProfessionalLogger:
    """Get or create global logger instance."""
    global _global_logger
    
    with _logger_lock:
        if _global_logger is None:
            # Determine configuration from environment
            level = LogLevel[os.getenv('LOG_LEVEL', 'INFO').upper()]
            json_enabled = os.getenv('LOG_FORMAT', 'text').lower() == 'json'
            log_dir = os.getenv('LOG_DIR', 'logs')
            
            _global_logger = ProfessionalLogger(
                name=name or "playlista",
                level=level,
                json_enabled=json_enabled,
                log_dir=log_dir
            )
        
        return _global_logger


def configure_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    console_enabled: bool = True,
    file_enabled: bool = True,
    json_enabled: bool = False,
    log_dir: str = "logs"
) -> ProfessionalLogger:
    """Configure global logging system."""
    global _global_logger
    
    with _logger_lock:
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        
        _global_logger = ProfessionalLogger(
            level=level,
            console_enabled=console_enabled,
            file_enabled=file_enabled,
            json_enabled=json_enabled,
            log_dir=log_dir
        )
        
        return _global_logger


# Compatibility functions for existing code
def log_universal(level: str, component: str, message: str, **kwargs):
    """Legacy compatibility function."""
    logger = get_logger()
    log_level = LogLevel[level.upper()] if isinstance(level, str) else level
    
    # Map component to category
    category_map = {
        'Audio': LogCategory.AUDIO,
        'Database': LogCategory.DATABASE,
        'API': LogCategory.API,
        'Config': LogCategory.SYSTEM,
        'Sequential': LogCategory.AUDIO,
        'Parallel': LogCategory.AUDIO,
        'Streaming': LogCategory.AUDIO,
        'CLI': LogCategory.SYSTEM,
        'Analysis': LogCategory.AUDIO,
        'Resource': LogCategory.SYSTEM,
        'System': LogCategory.SYSTEM,
    }
    
    category = category_map.get(component, LogCategory.SYSTEM)
    
    logger.log(log_level, category, component.lower(), message, 
              metadata=kwargs if kwargs else None)


def log_function_call(func):
    """Decorator for logging function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = f"{func.__module__}.{func.__name__}"
        
        with logger.operation_context(func_name, "function"):
            return func(*args, **kwargs)
    
    return wrapper