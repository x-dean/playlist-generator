# File Discovery Feature Documentation

## Overview

The file discovery feature provides comprehensive audio file discovery and management capabilities for the playlist generator. It handles directory scanning, file validation, database integration, and change tracking.

## Features

### Core Functionality

1. **File Discovery & Scanning**
   - Recursive and non-recursive directory scanning
   - Configurable file validation criteria
   - Support for multiple audio formats
   - Fixed Docker paths for containerized environment
   - Automatic exclusion of failed files directory

2. **File Validation**
   - Extension-based filtering (mp3, wav, flac, ogg, m4a, aac, opus)
   - File size limits (minimum and maximum)
   - File readability checks
   - Fixed directory exclusion patterns

3. **Database Integration**
   - Automatic persistence of discovered files
   - Change tracking (new, modified, unchanged files)
   - Failed file management and retry logic
   - Comprehensive cleanup of non-existent files from all database locations

4. **Smart Hashing**
   - Filename + modification time + file size based hashing
   - Support for MD5, SHA1, SHA256 algorithms
   - File movement tolerance (files can be moved without being treated as new)

## CLI Usage

### Basic Discovery
```bash
# Discover files in music directory (fixed at /music)
playlista discover

# Discover with custom extensions
playlista discover --extensions mp3,wav,flac

# Non-recursive discovery
playlista discover --recursive false

# Set file size limits
playlista discover --min-size 10240 --max-size 524288000
```

### Advanced Options
```bash
# Discover with all custom settings
playlista discover \
  --extensions mp3,wav,flac \
  --recursive \
  --min-size 10240 \
  --max-size 524288000
```

**Note**: The music directory is fixed at `/music` (Docker path mapped from compose) and cannot be changed.

## Configuration

### File Discovery Settings

Configuration is loaded from `playlista.conf`:

```ini
# File Discovery Configuration
# Note: Music directory is fixed at /music (Docker path mapped from compose)
# Note: Failed directory is fixed at /app/cache/failed_dir (Docker path)
# Note: Database path is fixed at /app/cache/playlista.db (Docker path)
# Note: Exclude directories are fixed and not configurable

MIN_FILE_SIZE_BYTES=10240
MAX_FILE_SIZE_BYTES=524288000
VALID_EXTENSIONS=.mp3,.wav,.flac,.ogg,.m4a,.aac,.opus
HASH_ALGORITHM=md5
MAX_RETRY_COUNT=3
ENABLE_RECURSIVE_SCAN=true
ENABLE_DETAILED_LOGGING=true
```

### CLI Argument Overrides

CLI arguments override configuration file settings:

- `--extensions`: Override valid file extensions
- `--recursive`: Override recursive scan setting
- `--min-size`: Override minimum file size (bytes)
- `--max-size`: Override maximum file size (bytes)

**Note**: Music directory and exclude directories are fixed and cannot be overridden.

## Fixed Docker Paths

The following paths are fixed for the Docker environment and cannot be configured:

- **Music Directory**: `/music` (mapped from Docker Compose)
- **Failed Directory**: `/app/cache/failed_dir` (internal container path)
- **Database Path**: `/app/cache/playlista.db` (internal container path)
- **Exclude Directories**: Fixed to exclude failed directory only

The failed directory is automatically excluded from discovery to prevent processing of failed files.

## API Methods

### Core Methods

1. **`discover_files()`**
   - Scans directory for valid audio files
   - Returns list of file paths

2. **`save_discovered_files_to_db(filepaths)`**
   - Persists files to database
   - Returns statistics (new, updated, unchanged, errors)

3. **`get_files_for_analysis(force=False, failed_mode=False)`**
   - Returns files needing analysis
   - Supports force re-analysis and failed-only modes

4. **`cleanup_removed_files_from_db()`**
   - Removes non-existent files from database
   - Returns number of files removed

### Utility Methods

1. **`get_statistics()`**
   - Returns comprehensive discovery statistics

2. **`validate_file_paths(filepaths)`**
   - Validates list of file paths
   - Returns only valid files

3. **`get_file_info(filepath)`**
   - Returns detailed file information

4. **`override_config(**kwargs)`**
   - Overrides configuration settings for CLI arguments

## Database Integration

### Tables Used

1. **`analysis_cache`**
   - Stores file analysis results
   - Tracks file hashes and metadata

2. **`failed_analysis`**
   - Tracks failed analysis attempts
   - Stores error messages and retry counts

### Database Operations

- **Save**: New files and modified files
- **Update**: Changed files with new hashes
- **Delete**: Non-existent files during cleanup
- **Query**: Statistics and file lists

## Error Handling

### File Validation Errors
- Invalid file extensions
- File size outside limits
- Unreadable files
- Files in excluded directories

### Database Errors
- Connection failures
- Transaction rollbacks
- Constraint violations

### CLI Argument Errors
- Invalid file size values
- Malformed extension lists
- Non-existent directories

## Logging

### Log Levels
- **INFO**: Discovery progress and statistics
- **DEBUG**: Detailed file validation
- **WARNING**: Non-critical issues
- **ERROR**: Failures and exceptions

### Log Categories
- **FileDiscovery**: Main discovery operations
- **Database**: Database operations
- **CLI**: Command-line interface

## Examples

### Basic Discovery
```python
from core.file_discovery import FileDiscovery

fd = FileDiscovery()
files = fd.discover_files()
stats = fd.save_discovered_files_to_db(files)
```

### Custom Configuration
```python
fd = FileDiscovery()
fd.override_config(
    min_file_size_bytes=2048,
    valid_extensions=['.mp3', '.wav'],
    enable_recursive_scan=False
)
files = fd.discover_files()
```

### Analysis Integration
```python
# Get files for analysis
files = fd.get_files_for_analysis(force=False, failed_mode=False)

# Get only failed files
failed_files = fd.get_files_for_analysis(failed_mode=True)

# Force re-analysis of all files
all_files = fd.get_files_for_analysis(force=True)
```

## Troubleshooting

### Common Issues

1. **No files discovered**
   - Check music directory path
   - Verify file extensions
   - Check file size limits

2. **Database errors**
   - Verify database path
   - Check file permissions
   - Review error logs

3. **CLI argument errors**
   - Validate argument formats
   - Check file size values
   - Verify directory paths

### Debug Mode
```bash
# Enable debug logging
playlista discover --log-level DEBUG

# Verbose output
playlista discover --verbose
```

## Recent Fixes

### CLI Argument Implementation
- ✅ CLI arguments now properly override configuration
- ✅ Argument validation and error handling
- ✅ Support for all discover command options

### Configuration Management
- ✅ Dedicated file discovery configuration method
- ✅ String to list conversion for extensions and directories
- ✅ Proper configuration override system

### Error Handling
- ✅ Comprehensive error messages
- ✅ Graceful failure handling
- ✅ Input validation for CLI arguments

### Database Integration
- ✅ Proper transaction handling
- ✅ Error recovery mechanisms
- ✅ Statistics tracking

## Testing

Run the test script to verify functionality:

```bash
python test_file_discovery.py
```

This will test:
- FileDiscovery initialization
- Configuration override
- File validation
- Statistics generation 