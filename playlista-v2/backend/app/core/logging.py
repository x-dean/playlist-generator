"""
Professional logging configuration for Playlista v2
Clean, consistent, and structured logging across all components
"""

import logging
import logging.config
import sys
import time
from functools import wraps
from typing import Any, Dict, Optional

import structlog


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Setup structured logging with professional formatting
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log format (json, text)
    """
    
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Professional structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        filter_sensitive_data,
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer(sort_keys=True))
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                pad_event=25,
                exception_formatter=structlog.dev.plain_traceback
            )
        )
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def filter_sensitive_data(logger, method_name, event_dict):
    """Filter sensitive information from logs"""
    sensitive_keys = {'password', 'token', 'key', 'secret', 'authorization'}
    
    for key in list(event_dict.keys()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            event_dict[key] = "[REDACTED]"
    
    return event_dict


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance with component context
    
    Args:
        name: Logger name (e.g., 'api.library', 'analysis.engine')
    """
    return structlog.get_logger(name)


def log_performance(operation: str = None):
    """
    Professional performance logging decorator
    
    Args:
        operation: Custom operation name for clearer logs
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger("performance")
            op_name = operation or func.__name__
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = round((time.time() - start_time) * 1000, 2)
                
                if duration_ms > 1000:  # Log slow operations prominently
                    logger.warning(
                        "Slow operation completed",
                        operation=op_name,
                        duration_ms=duration_ms,
                        module=func.__module__
                    )
                else:
                    logger.debug(
                        "Operation completed",
                        operation=op_name,
                        duration_ms=duration_ms
                    )
                
                return result
                
            except Exception as e:
                duration_ms = round((time.time() - start_time) * 1000, 2)
                logger.error(
                    "Operation failed",
                    operation=op_name,
                    duration_ms=duration_ms,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    module=func.__module__
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger("performance")
            op_name = operation or func.__name__
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.time() - start_time) * 1000, 2)
                
                if duration_ms > 1000:
                    logger.warning(
                        "Slow operation completed",
                        operation=op_name,
                        duration_ms=duration_ms,
                        module=func.__module__
                    )
                else:
                    logger.debug(
                        "Operation completed",
                        operation=op_name,
                        duration_ms=duration_ms
                    )
                
                return result
                
            except Exception as e:
                duration_ms = round((time.time() - start_time) * 1000, 2)
                logger.error(
                    "Operation failed",
                    operation=op_name,
                    duration_ms=duration_ms,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    module=func.__module__
                )
                raise
        
        # Return appropriate wrapper
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Context managers for structured logging
class LogContext:
    """Context manager for adding structured context to logs"""
    
    def __init__(self, **context):
        self.context = context
    
    def __enter__(self):
        structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_operation_start(logger: structlog.BoundLogger, operation: str, **context):
    """Log the start of an operation with context"""
    logger.info(
        f"Starting {operation}",
        operation=operation,
        **context
    )


def log_operation_success(logger: structlog.BoundLogger, operation: str, duration_ms: float, **context):
    """Log successful operation completion"""
    logger.info(
        f"Completed {operation}",
        operation=operation,
        duration_ms=round(duration_ms, 2),
        status="success",
        **context
    )


def log_operation_error(logger: structlog.BoundLogger, operation: str, error: Exception, duration_ms: float, **context):
    """Log operation failure with structured error information"""
    logger.error(
        f"Failed {operation}",
        operation=operation,
        duration_ms=round(duration_ms, 2),
        status="error",
        error_type=type(error).__name__,
        error_message=str(error),
        **context
    )
