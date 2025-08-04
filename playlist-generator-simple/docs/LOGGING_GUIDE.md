# Professional Logging System Guide

This guide covers the professional logging system implemented for Playlist Generator Simple, providing structured, standardized, and production-ready logging capabilities.

## Overview

The professional logging system addresses critical issues found in the original logging implementation:

### ❌ Issues with Original Logging
- **Inconsistent formatting**: Mixed use of `log_universal`, standard logging, and print statements
- **No structured data**: Plain text messages without context or metadata
- **Poor performance tracking**: No correlation IDs or operation timing
- **No security logging**: Missing audit trails and security events
- **No JSON support**: Difficult to integrate with log aggregation systems
- **Thread-unsafe operations**: Race conditions in multi-threaded environments
- **No error context**: Missing stack traces and error correlation

### ✅ Professional Logging Features
- **Structured logging**: JSON-formatted logs with consistent schema
- **Categorized logging**: System, Audio, Database, API, Security, Performance categories
- **Performance tracking**: Built-in timing and correlation IDs
- **Security auditing**: Specialized security event logging
- **Thread-safe operations**: Safe for concurrent access
- **Multiple output formats**: Console (colored), file (structured), JSON
- **Error context**: Full stack traces and error correlation
- **Production-ready**: Log rotation, compression, and retention policies

## Architecture

### Core Components

```python
# Main logger class
ProfessionalLogger
├── SecurityLogger      # Security events and audit trails
├── PerformanceLogger   # Performance metrics and timing
├── LogContext         # Structured log entry context
├── CustomJSONFormatter # JSON output formatting
└── ColoredConsoleFormatter # Development console output
```

### Log Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| `SYSTEM` | System operations | Startup, shutdown, configuration |
| `AUDIO` | Audio processing | File loading, analysis, feature extraction |
| `DATABASE` | Database operations | Queries, connections, migrations |
| `API` | API requests/responses | HTTP endpoints, external API calls |
| `SECURITY` | Security events | Authentication, authorization, threats |
| `PERFORMANCE` | Performance metrics | Timing, resource usage, bottlenecks |
| `BUSINESS` | Business logic | Playlist generation, user actions |
| `EXTERNAL` | External services | Third-party API calls, file system |

### Log Levels

| Level | Numeric | Purpose | When to Use |
|-------|---------|---------|-------------|
| `TRACE` | 5 | Very detailed debugging | Development debugging only |
| `DEBUG` | 10 | Debugging information | Development and staging |
| `INFO` | 20 | General information | Normal operations |
| `WARNING` | 30 | Warnings | Recoverable errors, deprecations |
| `ERROR` | 40 | Errors | Error conditions |
| `CRITICAL` | 50 | Critical errors | System failures |

## Configuration

### Environment Variables

```bash
# Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Log format (text, json)
LOG_FORMAT=json

# Log directory
LOG_DIR=logs

# Console output (true, false)
LOG_CONSOLE=true

# File output (true, false)
LOG_FILE=true
```

### Programmatic Configuration

```python
from src.core.professional_logging import configure_logging, LogLevel

# Configure logging system
logger = configure_logging(
    level=LogLevel.INFO,
    console_enabled=True,
    file_enabled=True,
    json_enabled=True,
    log_dir="logs"
)
```

### Docker Configuration

```dockerfile
# Production JSON logging
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO
ENV LOG_DIR=/app/logs

# Development colored console
ENV LOG_FORMAT=text
ENV LOG_LEVEL=DEBUG
ENV LOG_CONSOLE=true
```

## Usage Examples

### Basic Logging

```python
from src.core.professional_logging import get_logger, LogCategory

logger = get_logger()

# Simple info message
logger.info(LogCategory.SYSTEM, "startup", "Application starting")

# Warning with metadata
logger.warning(
    LogCategory.AUDIO, 
    "file_processor", 
    "Large file detected",
    metadata={"file_size_mb": 150, "file_path": "/path/to/file.wav"}
)

# Error with error code
logger.error(
    LogCategory.DATABASE, 
    "connection", 
    "Database connection failed",
    error_code="ConnectionTimeout",
    metadata={"host": "localhost", "port": 5432}
)
```

### Performance Logging

```python
# Using context manager for timing
with logger.performance.time_operation("audio_analysis", "audio_processor"):
    result = analyze_audio_file(file_path)

# Manual performance logging
logger.performance.log_metric(
    "cache_hit_rate", 
    0.85, 
    "cache_manager", 
    unit="percent"
)
```

### Security Logging

```python
# Authentication events
logger.security.authentication_attempt(
    user_id="user123", 
    success=True, 
    ip_address="192.168.1.100"
)

# Access denied
logger.security.access_denied(
    user_id="user123", 
    resource="/admin/settings", 
    reason="insufficient_privileges"
)

# Suspicious activity
logger.security.suspicious_activity(
    "Multiple failed login attempts",
    severity="high",
    ip_address="192.168.1.100",
    attempts=5
)
```

### Exception Logging

```python
try:
    process_audio_file(file_path)
except FileNotFoundError as e:
    logger.log_exception(
        LogCategory.AUDIO,
        "file_processor",
        f"Audio file not found: {file_path}",
        exception=e,
        metadata={"file_path": file_path}
    )
except Exception as e:
    logger.log_exception(
        LogCategory.SYSTEM,
        "file_processor",
        "Unexpected error in audio processing",
        exception=e
    )
```

### Operation Context

```python
# Track related operations with correlation IDs
with logger.operation_context("playlist_generation", "playlist_service", user_id="user123"):
    # All logs within this context will have the same correlation_id
    logger.info(LogCategory.BUSINESS, "playlist_service", "Starting playlist generation")
    
    tracks = find_similar_tracks(seed_track)
    logger.info(LogCategory.BUSINESS, "playlist_service", f"Found {len(tracks)} similar tracks")
    
    playlist = create_playlist(tracks)
    logger.info(LogCategory.BUSINESS, "playlist_service", "Playlist generation completed")
```

## Log Output Formats

### Console Output (Development)

```
2024-01-15T10:30:45.123Z | INFO     | system:startup | Application starting
2024-01-15T10:30:45.124Z | WARNING  | audio:file_processor | Large file detected (150.5MB)
2024-01-15T10:30:45.125Z | ERROR    | database:connection | Database connection failed
```

### JSON Output (Production)

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "category": "system",
  "component": "startup",
  "message": "Application starting",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000"
}

{
  "timestamp": "2024-01-15T10:30:45.124Z",
  "level": "WARNING",
  "category": "audio",
  "component": "file_processor",
  "message": "Large file detected",
  "metadata": {
    "file_size_mb": 150.5,
    "file_path": "/path/to/file.wav"
  }
}

{
  "timestamp": "2024-01-15T10:30:45.125Z",
  "level": "ERROR",
  "category": "database",
  "component": "connection",
  "message": "Database connection failed",
  "error_code": "ConnectionTimeout",
  "metadata": {
    "host": "localhost",
    "port": 5432
  }
}
```

## File Organization

### Log Files

- `playlista.log` - Main application log (all levels)
- `playlista-errors.log` - Error log (ERROR and above only)
- Log rotation: 50MB max size, 10 backup files
- Compression: Gzip for rotated files
- Retention: 10 days for backup files

### Directory Structure

```
logs/
├── playlista.log              # Current main log
├── playlista.log.1            # Previous main log
├── playlista.log.2.gz         # Compressed backup
├── playlista-errors.log       # Current error log
├── playlista-errors.log.1     # Previous error log
└── playlista-errors.log.2.gz  # Compressed error backup
```

## Migration from Legacy Logging

### Replacing `log_universal`

```python
# Old way
from src.core.logging_setup import log_universal
log_universal('INFO', 'Audio', 'Processing file')

# New way
from src.core.professional_logging import get_logger, LogCategory
logger = get_logger()
logger.info(LogCategory.AUDIO, "processor", "Processing file")
```

### Replacing Standard Logging

```python
# Old way
import logging
logger = logging.getLogger(__name__)
logger.info("Processing started")

# New way
from src.core.professional_logging import get_logger, LogCategory
logger = get_logger()
logger.info(LogCategory.SYSTEM, "processor", "Processing started")
```

### Function Decorators

```python
# Old way
from src.core.logging_setup import log_function_call

@log_function_call
def process_audio(file_path):
    # Function implementation
    pass

# New way
from src.core.professional_logging import get_logger, LogCategory

def process_audio(file_path):
    logger = get_logger()
    with logger.operation_context("process_audio", "audio_processor"):
        # Function implementation
        pass
```

## Best Practices

### 1. Use Appropriate Log Levels

```python
# ✅ Good
logger.debug(LogCategory.AUDIO, "analyzer", "Starting feature extraction")
logger.info(LogCategory.AUDIO, "analyzer", "Audio analysis completed")
logger.warning(LogCategory.AUDIO, "analyzer", "Low quality audio detected")
logger.error(LogCategory.AUDIO, "analyzer", "Analysis failed due to corruption")

# ❌ Bad - Wrong levels
logger.error(LogCategory.AUDIO, "analyzer", "Starting feature extraction")  # Not an error
logger.info(LogCategory.AUDIO, "analyzer", "Analysis failed")  # Should be error
```

### 2. Include Relevant Metadata

```python
# ✅ Good - Rich context
logger.info(
    LogCategory.AUDIO, 
    "analyzer", 
    "Audio analysis completed",
    duration_ms=1500,
    metadata={
        "file_path": "/music/song.mp3",
        "file_size_mb": 4.2,
        "sample_rate": 44100,
        "duration_seconds": 180,
        "features_extracted": 128
    }
)

# ❌ Bad - No context
logger.info(LogCategory.AUDIO, "analyzer", "Analysis done")
```

### 3. Use Correlation IDs

```python
# ✅ Good - Trackable operations
with logger.operation_context("playlist_generation", "service", user_id="user123"):
    logger.info(LogCategory.BUSINESS, "service", "Starting playlist generation")
    tracks = find_tracks()
    logger.info(LogCategory.BUSINESS, "service", f"Found {len(tracks)} tracks")

# ❌ Bad - Unrelated log entries
logger.info(LogCategory.BUSINESS, "service", "Starting playlist generation")
logger.info(LogCategory.BUSINESS, "service", "Found tracks")  # No correlation
```

### 4. Handle Exceptions Properly

```python
# ✅ Good - Full exception context
try:
    result = risky_operation()
except ValueError as e:
    logger.log_exception(
        LogCategory.SYSTEM,
        "processor",
        "Invalid input provided",
        exception=e,
        metadata={"input_value": input_data}
    )
    raise

# ❌ Bad - Lost exception context
try:
    result = risky_operation()
except ValueError as e:
    logger.error(LogCategory.SYSTEM, "processor", "Error occurred")  # No details
    raise
```

### 5. Security-Sensitive Information

```python
# ✅ Good - Secure logging
logger.info(
    LogCategory.SECURITY, 
    "auth", 
    "User authentication successful",
    metadata={
        "user_id": "user123",
        "method": "oauth",
        "ip_address": "192.168.1.100"
        # Don't log passwords, tokens, or sensitive data
    }
)

# ❌ Bad - Exposing sensitive data
logger.info(
    LogCategory.SECURITY, 
    "auth", 
    f"User {username} logged in with password {password}"  # Never log passwords!
)
```

## Production Deployment

### 1. Log Aggregation

For production deployments, configure log aggregation:

```yaml
# docker-compose.yml
services:
  playlista:
    environment:
      - LOG_FORMAT=json
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
    
  # Log shipping (example with Fluentd)
  fluentd:
    image: fluent/fluentd:v1.16-1
    volumes:
      - ./logs:/var/log/playlista:ro
      - ./fluentd.conf:/fluentd/etc/fluent.conf
```

### 2. Monitoring and Alerting

Set up monitoring for critical log events:

```python
# Alert on error rates
SELECT COUNT(*) 
FROM logs 
WHERE level='ERROR' 
  AND timestamp > NOW() - INTERVAL '5 minutes'
HAVING COUNT(*) > 10

# Alert on security events
SELECT COUNT(*) 
FROM logs 
WHERE category='security' 
  AND level IN ('WARNING', 'ERROR')
  AND timestamp > NOW() - INTERVAL '1 minute'
```

### 3. Performance Monitoring

Track performance metrics:

```python
# Query response times
SELECT 
  component,
  AVG(duration_ms) as avg_duration,
  MAX(duration_ms) as max_duration
FROM logs 
WHERE category='performance' 
  AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY component
```

## Troubleshooting

### Common Issues

1. **Logs not appearing**
   - Check `LOG_LEVEL` environment variable
   - Verify file permissions in log directory
   - Check disk space

2. **Poor performance**
   - Reduce log level in production
   - Ensure log directory is on fast storage
   - Monitor log file sizes

3. **JSON parsing errors**
   - Validate JSON formatter configuration
   - Check for special characters in messages
   - Verify log shipping configuration

### Debug Configuration

```python
# Enable detailed logging for debugging
logger = configure_logging(
    level=LogLevel.TRACE,
    console_enabled=True,
    json_enabled=False  # Use colored output for readability
)

# Check logger configuration
stats = logger.get_stats()
print(f"Logger stats: {stats}")
```

## Migration Checklist

- [ ] Replace all `log_universal` calls with structured logging
- [ ] Replace standard logging with professional logging
- [ ] Add appropriate log categories to all components
- [ ] Include relevant metadata in log entries
- [ ] Add correlation IDs for operation tracking
- [ ] Update exception handling to use `log_exception`
- [ ] Configure environment variables for production
- [ ] Set up log rotation and retention policies
- [ ] Configure log aggregation and monitoring
- [ ] Test logging in development and staging environments

The professional logging system provides a solid foundation for production-ready applications with comprehensive logging, monitoring, and debugging capabilities.