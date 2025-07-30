# Logging Improvements

This document outlines the improvements made to ensure all logs are properly written to files instead of just being printed to the console.

## Problem Identified

The original implementation was using `get_logger()` without properly initializing the logging system with `setup_logging()`. This meant that:
- Logs were only printed to console
- No log files were created
- No persistent logging for debugging and monitoring
- **Docker Issue**: Logs were not being written to the Docker mount point

## Root Cause Analysis

1. **Missing Logging Initialization**: Files were using `get_logger()` without calling `setup_logging()`
2. **No File Handlers**: The logging system wasn't configured to write to files
3. **Inconsistent Logging**: Different files had different logging approaches
4. **Docker Mount Point**: Logs were being written to relative `logs/` instead of Docker mount point `/app/logs`

## Solutions Implemented

### 1. Proper Logging Initialization

**Before**: Using logger without initialization
```python
from core.logging_setup import get_logger
logger = get_logger('playlista.enhanced_cli')
```

**After**: Proper logging initialization with Docker mount point
```python
from core.logging_setup import get_logger, setup_logging

# Initialize logging system
setup_logging(
    log_level='INFO',
    log_dir='/app/logs',  # Docker mount point
    log_file_prefix='playlista',
    console_logging=True,
    file_logging=True,
    colored_output=True,
    max_log_files=10,
    log_file_size_mb=50,
    log_file_format='text',
    log_file_encoding='utf-8'
)

logger = get_logger('playlista.enhanced_cli')
```

### 2. Updated Files

The following files were updated to include proper logging initialization with Docker mount point:

#### Enhanced CLI (`src/enhanced_cli.py`)
- Added `setup_logging()` call
- Configured for main application logging
- Log file prefix: `playlista`
- **Docker mount point**: `/app/logs`

#### Analysis CLI (`src/analysis_cli.py`)
- Added `setup_logging()` call
- Configured for analysis-specific logging
- Log file prefix: `playlista_analysis`
- **Docker mount point**: `/app/logs`

#### Test Files (`test_streaming_simple.py`)
- Added `setup_logging()` call
- Configured for test logging
- Log file prefix: `test_streaming`
- **Docker mount point**: `/app/logs`

### 3. Docker Configuration

The Docker setup includes proper volume mounting for logs:

```yaml
# docker-compose.yml
volumes:
  # Logs directory
  - ./logs:/app/logs
```

This ensures that:
- Logs written to `/app/logs` inside the container
- Appear in `./logs` directory on the host
- Persist across container restarts
- Accessible from the host system

### 4. Logging Configuration

Each application component now has its own logging configuration:

| Component | Log File Prefix | Docker Path | Host Path |
|-----------|----------------|-------------|-----------|
| Main CLI | `playlista` | `/app/logs` | `./logs` |
| Analysis CLI | `playlista_analysis` | `/app/logs` | `./logs` |
| Tests | `test_streaming` | `/app/logs` | `./logs` |

### 5. Log File Structure

Log files are created in the Docker mount point with the following naming convention:
```
Host: ./logs/
├── playlista_YYYYMMDD.log          # Main application logs
├── playlista_analysis_YYYYMMDD.log # Analysis logs
└── test_streaming_YYYYMMDD.log     # Test logs

Container: /app/logs/
├── playlista_YYYYMMDD.log          # Main application logs
├── playlista_analysis_YYYYMMDD.log # Analysis logs
└── test_streaming_YYYYMMDD.log     # Test logs
```

## Logging Features

### 1. File Logging
- **Location**: `/app/logs` (Docker container) → `./logs` (host)
- **Format**: Text format with timestamps
- **Rotation**: Up to 10 log files per component
- **Size Limit**: 50MB per log file
- **Persistence**: Logs persist across container restarts

### 2. Console Logging
- **Colored Output**: Different colors for different log levels
- **Real-time**: Immediate console output
- **Formatted**: Structured log messages

### 3. Log Levels
- **INFO**: General information and progress
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors and failures
- **DEBUG**: Detailed debugging information

### 4. Memory Management Logging

The memory management improvements now include detailed logging:
- Memory usage logged every 5 chunks
- Aggressive cleanup triggered at 70%
- Critical memory handling at 85%
- Dynamic chunk size adjustment under pressure

## Example Log Output

```
2025-07-30 18:51:53 - playlista - INFO - Logging system initialized
2025-07-30 18:51:53 - playlista.streaming_loader - INFO - 🔧 Creating new StreamingAudioLoader instance:
2025-07-30 18:51:53 - playlista.streaming_loader - INFO -    Memory limit: 50%
2025-07-30 18:51:53 - playlista.streaming_loader - INFO -    Chunk duration: 15s
2025-07-30 18:51:53 - playlista.streaming_loader - WARNING - ⚠️ Memory usage after chunk 425: 69.7% (10.4GB / 15.5GB)
2025-07-30 18:51:53 - playlista.streaming_loader - WARNING - ⚠️ High memory usage detected! Forcing aggressive memory cleanup...
2025-07-30 18:51:53 - playlista.streaming_loader - INFO - 🧹 Forced memory cleanup completed
```

## Benefits

1. **Persistent Logging**: All logs are saved to files for later analysis
2. **Docker Integration**: Logs properly written to Docker mount point
3. **Debugging Support**: Detailed logs help identify issues
4. **Monitoring**: Log files can be monitored for system health
5. **Audit Trail**: Complete record of all operations
6. **Memory Tracking**: Detailed memory management logs
7. **Error Tracking**: All errors are logged with context
8. **Host Access**: Logs accessible from host system

## Usage

### Running the Application

#### Local Development
```bash
# Main application (creates playlista_YYYYMMDD.log in ./logs)
python playlista analyze /path/to/music

# Analysis CLI (creates playlista_analysis_YYYYMMDD.log in ./logs)
python src/analysis_cli.py analyze /path/to/music

# Tests (creates test_streaming_YYYYMMDD.log in ./logs)
python test_streaming_simple.py
```

#### Docker Environment
```bash
# Start container with volume mounting
docker-compose up

# Logs will appear in ./logs directory on host
ls -la logs/

# View logs from host
tail -f logs/playlista_$(date +%Y%m%d).log
```

### Checking Log Files

#### From Host System
```bash
# View latest log file
tail -f logs/playlista_$(date +%Y%m%d).log

# Search for memory issues
grep "memory" logs/playlista_*.log

# Search for errors
grep "ERROR" logs/playlista_*.log

# List all log files
ls -la logs/
```

#### From Docker Container
```bash
# Access container
docker exec -it playlista-simple bash

# View logs inside container
tail -f /app/logs/playlista_$(date +%Y%m%d).log

# List log files
ls -la /app/logs/
```

### Docker Volume Mounting

The Docker configuration ensures logs are properly mounted:

```yaml
volumes:
  - ./logs:/app/logs
```

This means:
- **Container path**: `/app/logs/`
- **Host path**: `./logs/`
- **Persistence**: Logs survive container restarts
- **Accessibility**: Logs accessible from both container and host

This ensures that all logs are properly written to files in the Docker mount point and can be used for debugging, monitoring, and auditing purposes. 