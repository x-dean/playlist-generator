# Unified Logging System Documentation

## Overview

The Playlist Generator now uses a professional, unified logging system that provides:

- **Structured logging** with JSON output support
- **Colored console output** for better readability
- **Performance optimization** with sampling for high-frequency operations
- **External library management** for TensorFlow, Essentia, and other dependencies
- **Multiple environment configurations** (development, production, testing, API)
- **Comprehensive error handling** patterns
- **Migration tools** for updating existing code

## Quick Start

### Basic Setup

```python
from playlist-generator-simple.src.core.unified_logging import setup_logging, get_logger
from playlist-generator-simple.src.core.logging_config import get_logging_config

# Setup logging with default configuration
setup_logging(get_logging_config('development'))

# Get a logger for your module
logger = get_logger('your.module.name')

# Use the logger
logger.info("Application started")
logger.debug("Debug information")
logger.warning("Something might be wrong")
logger.error("An error occurred")
```

### Environment-Specific Configuration

```python
# Development environment (verbose, colored output)
setup_logging(get_logging_config('development'))

# Production environment (structured JSON logging)
setup_logging(get_logging_config('production'))

# API server environment
setup_logging(get_logging_config('api'))

# CLI with verbosity levels
setup_logging(get_logging_config('cli', verbose_level=2))
```

## Core Components

### 1. Unified Logger (`unified_logging.py`)

The main logging system providing:

- **ColoredFormatter**: Console output with ANSI colors
- **StructuredFormatter**: JSON output for structured logging
- **PerformanceLogFilter**: Sampling filter for high-frequency debug messages
- **UnifiedLogger**: Main class managing the logging system

### 2. Configuration System (`logging_config.py`)

Provides standardized configurations:

- **LoggingPatterns**: Standard component names and level mappings
- **LoggingConfigurations**: Pre-defined configs for different environments
- **StandardLogMessages**: Template functions for consistent messaging

### 3. Examples and Templates (`logging_examples.py`)

Comprehensive examples showing:

- Basic logging patterns
- Structured logging usage
- API call logging
- Performance monitoring
- Error handling patterns
- Context managers and decorators

### 4. Migration Tools (`logging_migration_guide.py`)

Tools for updating existing code:

- **LoggingMigrationHelper**: Automated migration assistance
- Code analysis and recommendations
- Migration examples and patterns

## Usage Patterns

### Basic Logging

```python
from playlist-generator-simple.src.core.unified_logging import get_logger
from playlist-generator-simple.src.core.logging_config import create_logger_name

logger = get_logger(create_logger_name(__name__))

# Standard log levels
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning about potential issues")
logger.error("Error that needs attention")
logger.critical("Critical error that may cause shutdown")
```

### Structured Logging

```python
from playlist-generator-simple.src.core.unified_logging import log_structured
from playlist-generator-simple.src.core.logging_config import LoggingPatterns

# Simple structured logging
log_structured(
    'INFO',
    LoggingPatterns.COMPONENTS['AUDIO_ANALYZER'],
    'Processing audio file',
    file_path='/path/to/audio.mp3',
    file_size_mb=4.5,
    duration_seconds=180
)

# Complex structured data
log_structured(
    'DEBUG',
    LoggingPatterns.COMPONENTS['DATABASE'],
    'Query executed',
    query_type='SELECT',
    table='tracks',
    execution_time_ms=45.2,
    rows_returned=127,
    cache_hit=False
)
```

### API Call Logging

```python
from playlist-generator-simple.src.core.unified_logging import log_api_call

# Successful API call
log_api_call(
    api_name='MusicBrainz',
    operation='search',
    target='artist: The Beatles',
    success=True,
    duration=0.845,
    details='Found 15 recordings'
)

# Failed API call with categorized failure type
log_api_call(
    api_name='LastFM',
    operation='lookup',
    target='track: Unknown Song',
    success=False,
    duration=2.1,
    failure_type='no_data'  # or 'network', 'timeout', etc.
)
```

### Performance Logging

```python
from playlist-generator-simple.src.core.unified_logging import log_resource_usage
from playlist-generator-simple.src.core.logging_examples import LoggedOperation

# Resource usage logging
log_resource_usage(
    LoggingPatterns.COMPONENTS['AUDIO_ANALYZER'],
    cpu_percent=45.2,
    memory_mb=234.5,
    processing_time_seconds=3.4,
    files_processed=10
)

# Context manager for operation timing
with LoggedOperation("file_processing", "/path/to/files", "Audio"):
    process_audio_files(files)
```

### Error Handling

```python
from playlist-generator-simple.src.core.logging_config import StandardLogMessages

def process_audio_file(file_path: str):
    operation = "audio_analysis"
    
    logger.info(StandardLogMessages.operation_start(operation, file_path))
    
    start_time = time.time()
    
    try:
        # Processing logic here
        result = analyze_file(file_path)
        
        duration = time.time() - start_time
        logger.info(StandardLogMessages.operation_complete(operation, file_path, duration))
        
        return result
        
    except ValueError as e:
        # Expected errors - warning level
        logger.warning(f"Invalid file format: {e}")
        return None
        
    except ConnectionError as e:
        # Network errors - error level
        logger.error(f"Network connection failed: {e}")
        return None
        
    except Exception as e:
        # Unexpected errors - exception level with traceback
        duration = time.time() - start_time
        logger.exception(StandardLogMessages.operation_failed(operation, file_path, str(e)))
        raise
```

### Decorators

```python
from playlist-generator-simple.src.core.unified_logging import log_performance, log_exceptions

class AudioProcessor:
    
    @log_performance('audio.processor')
    @log_exceptions('audio.processor')
    def process_file(self, file_path: str):
        """Process audio file with automatic logging."""
        # Implementation here
        pass
```

## Configuration

### Environment Variables

The logging system respects these environment variables:

```bash
# Basic configuration
LOG_LEVEL=DEBUG                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_CONSOLE_ENABLED=true          # Enable/disable console output
LOG_FILE_ENABLED=true             # Enable/disable file output
LOG_STRUCTURED=false              # Enable JSON structured logging
LOG_COLORED=true                  # Enable colored console output

# File configuration
LOG_DIR=logs                      # Log directory path
LOG_FILE_PREFIX=playlista         # Log file prefix
LOG_MAX_SIZE_MB=50               # Max file size before rotation
LOG_BACKUP_COUNT=10              # Number of backup files to keep

# Performance tuning
ENVIRONMENT=development           # development, production, testing, api
```

### Configuration Examples

```python
# Custom configuration
custom_config = {
    'level': 'DEBUG',
    'console_enabled': True,
    'file_enabled': True,
    'structured_logging': True,
    'colored_output': False,
    'log_dir': '/custom/log/path',
    'log_file_prefix': 'my_app',
    'max_file_size_mb': 100,
    'backup_count': 20,
    'performance_sampling': True,
    'sample_rate': 0.1,  # 10% sampling for debug messages
    'external_library_level': 'ERROR',
}

setup_logging(custom_config)
```

## External Library Management

The unified logging system automatically manages logging for external libraries:

### TensorFlow
- Sets `TF_CPP_MIN_LOG_LEVEL=2` to suppress INFO and WARNING
- Configures TensorFlow logger to ERROR level
- Disables autograph verbosity

### Essentia
- Sets `ESSENTIA_LOG_LEVEL=error`
- Configures Essentia logger to WARNING level

### Librosa
- Configures Librosa logger to WARNING level

### Custom External Library Configuration

```python
# In your unified_logging configuration
config = {
    'external_library_level': 'ERROR',  # or 'WARNING', 'INFO', 'DEBUG'
    # ... other settings
}
```

## Migration Guide

### Step 1: Update Imports

**Before:**
```python
import logging
from core.logging_setup import get_logger, setup_logging
```

**After:**
```python
from playlist-generator-simple.src.core.unified_logging import get_logger, setup_logging, log_structured
from playlist-generator-simple.src.core.logging_config import LoggingPatterns, StandardLogMessages, create_logger_name
```

### Step 2: Update Logger Instantiation

**Before:**
```python
logger = logging.getLogger(__name__)
# or
logger = get_logger(__name__)
```

**After:**
```python
logger = get_logger(create_logger_name(__name__))
```

### Step 3: Replace Print Statements

**Before:**
```python
print(f"Processing file: {file_path}")
```

**After:**
```python
logger.debug(f"Processing file: {file_path}")
# or better yet:
log_structured('DEBUG', LoggingPatterns.COMPONENTS['AUDIO'], 'Processing file', file_path=file_path)
```

### Step 4: Use Structured Logging

**Before:**
```python
logger.info(f"API call took {duration:.2f}s and returned {count} results")
```

**After:**
```python
log_api_call('MusicBrainz', 'search', 'artist', True, duration=duration)
# or
log_structured('INFO', 'MB API', 'Search completed', duration=duration, results_count=count)
```

### Automated Migration

Use the migration helper:

```python
from playlist-generator-simple.src.core.logging_migration_guide import LoggingMigrationHelper

helper = LoggingMigrationHelper()

# Analyze a single file
analysis = helper.analyze_file(Path('path/to/file.py'))
print(analysis['recommendations'])

# Migrate a file (dry run first)
result = helper.migrate_file(Path('path/to/file.py'), dry_run=True)
print(result['changes_made'])

# Scan entire directory
analyses = helper.scan_directory(Path('src/'), recursive=True)
report = helper.generate_migration_report(analyses)
print(report)
```

## Best Practices

### 1. Use Appropriate Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General operational information
- **WARNING**: Unexpected behavior that doesn't stop operation
- **ERROR**: Serious problems that need attention
- **CRITICAL**: Very serious errors that may cause shutdown

### 2. Use Structured Logging for Important Events

```python
# Instead of:
logger.info(f"User {user_id} created playlist {playlist_name} with {track_count} tracks")

# Use:
log_structured(
    'INFO',
    LoggingPatterns.COMPONENTS['PLAYLIST'],
    'Playlist created',
    user_id=user_id,
    playlist_name=playlist_name,
    track_count=track_count,
    creation_time=datetime.now().isoformat()
)
```

### 3. Use Standard Message Templates

```python
from playlist-generator-simple.src.core.logging_config import StandardLogMessages

# Consistent operation logging
logger.info(StandardLogMessages.operation_start('audio_analysis', file_path))
# ... do work ...
logger.info(StandardLogMessages.operation_complete('audio_analysis', file_path, duration))
```

### 4. Handle Exceptions Properly

```python
try:
    risky_operation()
except ValueError as e:
    # Expected errors - use warning
    logger.warning(f"Invalid input: {e}")
except Exception as e:
    # Unexpected errors - use exception for traceback
    logger.exception(f"Unexpected error: {e}")
    raise
```

### 5. Use Context Managers for Operations

```python
with LoggedOperation("database_migration", "user_table", "Database"):
    migrate_user_table()
```

### 6. Log Performance Metrics

```python
@log_performance()
def expensive_operation():
    # Implementation
    pass

# Or manually:
log_resource_usage(
    'AudioAnalyzer',
    cpu_percent=psutil.cpu_percent(),
    memory_mb=psutil.virtual_memory().used / 1024 / 1024,
    processing_time=duration
)
```

## Configuration for Different Environments

### Development Environment

```python
# Full logging with colors and detailed output
config = get_logging_config('development')
# Features:
# - DEBUG level logging
# - Colored console output
# - File logging enabled
# - External libraries at INFO level (for debugging)
```

### Production Environment

```python
# Optimized for production with structured logging
config = get_logging_config('production')
# Features:
# - INFO level logging
# - Structured JSON output
# - Performance sampling (5% of debug messages)
# - External libraries at ERROR level
# - Larger log files with more backups
```

### Testing Environment

```python
# Minimal logging for tests
config = get_logging_config('testing')
# Features:
# - INFO level logging
# - File-only output (no console)
# - Smaller log files
# - External libraries at ERROR level
```

### API Server Environment

```python
# Optimized for API servers
config = get_logging_config('api')
# Features:
# - INFO level logging
# - Structured JSON output
# - Performance sampling
# - Suitable for log aggregation systems
```

## Troubleshooting

### Common Issues

1. **Colors not showing in console**
   - Ensure `colorama` is installed: `pip install colorama`
   - Check that `colored_output` is enabled in configuration
   - Verify terminal supports ANSI colors

2. **Log files not being created**
   - Check directory permissions for `log_dir`
   - Verify `file_enabled` is True in configuration
   - Check disk space availability

3. **Too much output from external libraries**
   - Adjust `external_library_level` in configuration
   - Set environment variables before importing libraries

4. **Performance issues with high-frequency logging**
   - Enable `performance_sampling` in configuration
   - Adjust `sample_rate` (lower = less frequent logging)
   - Use appropriate log levels (avoid DEBUG in production)

### Debugging Logging Issues

```python
from playlist-generator-simple.src.core.unified_logging import get_logger

# Check current logging configuration
logger = get_logger()
print(f"Logger level: {logger.level}")
print(f"Handlers: {len(logger.handlers)}")

for handler in logger.handlers:
    print(f"Handler: {type(handler).__name__}, Level: {handler.level}")
```

## Advanced Features

### Custom Formatters

```python
from playlist-generator-simple.src.core.unified_logging import UnifiedLogger

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Custom formatting logic
        return super().format(record)

# Use custom formatter
logger_instance = UnifiedLogger()
logger_instance.configure({
    'level': 'INFO',
    'console_enabled': True,
    # Add custom formatter configuration
})
```

### Dynamic Log Level Changes

```python
from playlist-generator-simple.src.core.unified_logging import change_log_level

# Change log level at runtime
success = change_log_level('DEBUG')
if success:
    logger.info("Log level changed to DEBUG")
```

### Performance Monitoring

```python
from playlist-generator-simple.src.core.logging_examples import LoggedOperation

# Automatically time and log operations
with LoggedOperation("complex_analysis", "large_dataset", "Analytics") as op:
    # Your complex operation here
    pass
    # Timing and success/failure automatically logged
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from playlist-generator-simple.src.core.unified_logging import setup_logging, get_logger
from playlist-generator-simple.src.core.logging_config import get_logging_config

app = FastAPI()

# Setup logging for API
setup_logging(get_logging_config('api'))
logger = get_logger('api.main')

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    log_structured(
        'INFO',
        'API',
        'Request processed',
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration_ms=duration * 1000
    )
    
    return response
```

### CLI Integration

```python
import argparse
from playlist-generator-simple.src.core.unified_logging import setup_logging, get_logger
from playlist-generator-simple.src.core.logging_config import get_logging_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()
    
    # Setup logging based on verbosity
    config = get_logging_config('cli', verbose_level=args.verbose)
    setup_logging(config)
    
    logger = get_logger('cli.main')
    logger.info("CLI application started")
```

## Migration Checklist

- [ ] Update all logging imports
- [ ] Replace logger instantiation patterns
- [ ] Convert print statements to appropriate log levels
- [ ] Implement structured logging for key operations
- [ ] Add proper error handling with appropriate log levels
- [ ] Use standard message templates where appropriate
- [ ] Configure environment-specific logging
- [ ] Test logging in different environments
- [ ] Update documentation and examples
- [ ] Remove old logging configuration files

## Support and Maintenance

### Regular Maintenance Tasks

1. **Monitor log file sizes** and rotation settings
2. **Review external library logging** configurations
3. **Update performance sampling** rates based on volume
4. **Audit structured logging** usage for consistency
5. **Check for deprecated** logging patterns in new code

### Performance Monitoring

```python
# Monitor logging performance
import time
from playlist-generator-simple.src.core.unified_logging import log_resource_usage

def monitor_logging_performance():
    start = time.time()
    
    # Log many messages
    for i in range(1000):
        logger.debug(f"Test message {i}")
    
    duration = time.time() - start
    log_resource_usage(
        'LoggingSystem',
        messages_per_second=1000/duration,
        total_duration=duration
    )
```

This unified logging system provides a professional, scalable foundation for all logging needs in the Playlist Generator application. It supports both simple and complex logging scenarios while maintaining consistency and performance.