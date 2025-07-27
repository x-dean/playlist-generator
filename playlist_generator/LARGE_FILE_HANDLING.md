# Large File Handling

## Overview

The playlist generator now includes improved handling for large audio files (>50MB) to prevent the application from getting stuck during processing.

## Problem

Large audio files (like the 274.9MB file mentioned) can cause the sequential processing to hang indefinitely, blocking the entire analysis pipeline.

## Solution

### Separate Process Handling

Files larger than the threshold (default: 50MB) are now processed in a separate process with:

- **Timeout Protection**: Automatic timeout based on file size
  - Files >200MB: 15 minutes timeout
  - Files >100MB: 10 minutes timeout  
  - Files >50MB: 5 minutes timeout

- **Process Isolation**: Large files are processed in a separate Python process to prevent blocking the main application

- **Automatic Cleanup**: Failed or timed-out processes are automatically terminated

### Progress Monitoring

- **Individual File Timing**: Each large file's processing time is tracked and logged
- **Periodic Updates**: Progress updates every 30 seconds for long-running files
- **Timeout Monitoring**: Background thread monitors for stuck processes (5+ minutes without progress)
- **Memory Monitoring**: Memory usage is logged when timeouts occur

## Configuration

### Command Line Options

```bash
# Set custom threshold for large file detection (default: 50MB)
playlista --analyze --large_file_threshold 100

# Enable memory-aware processing
playlista --analyze --memory_aware

# Use default 50MB threshold
playlista --analyze
```

### Environment Variables

```bash
# Set threshold via environment variable
export LARGE_FILE_THRESHOLD=75
export MEMORY_AWARE=true
playlista --analyze
```

## How It Works

1. **File Size Detection**: Files are checked against the threshold during processing
2. **Process Selection**: Large files use `LargeFileProcessor`, normal files use standard processing
3. **Timeout Management**: Each large file gets a timeout based on its size
4. **Progress Tracking**: Individual file timing and periodic status updates
5. **Error Handling**: Failed processes are logged and the file is marked as failed

## Benefits

- **No More Hanging**: Large files can't block the entire pipeline
- **Memory Management**: Automatic memory monitoring and cleanup
- **Better Visibility**: Progress updates and timing information
- **Configurable**: Adjustable threshold for different use cases
- **Automatic Recovery**: Failed processes are cleaned up automatically
- **Memory-Aware Processing**: Skip memory-intensive features when memory is low

## Testing

Use the test script to verify large file processing:

```bash
python test_large_file_processing.py
```

This will:
- Find large files in your music library
- Test the separate process handling
- Verify timeout and cleanup functionality

## Troubleshooting

### If a large file still gets stuck:

1. **Increase timeout**: Use a higher `--large_file_threshold` value
2. **Check memory**: Monitor system memory usage during processing
3. **Reduce workers**: Use `--workers=1` for sequential-only processing
4. **Skip large files**: Temporarily increase the threshold to skip problematic files
5. **Use memory monitoring**: Run `python memory_monitor.py --continuous`

### Memory Issues:

- Use `--memory_aware` flag to skip memory-intensive features
- Use `--low_memory` flag to reduce memory usage
- Set `--workers=1` for sequential processing only
- Monitor memory usage with `python memory_monitor.py`
- Check for memory leaks in logs

## Log Messages

Look for these log messages to track large file processing:

```
INFO: Large file detected (274.9MB), using separate process: filename.mp3
INFO: Processing large file (274.9MB) in separate process: filename.mp3
INFO: Large file processing completed: filename.mp3
WARNING: Large file processing timed out after 600s: filename.mp3
WARNING: Memory usage critical (95.2%), will skip memory-intensive features
ERROR: Memory usage critical during processing, cancelling future
INFO: Memory after cleanup: 78.5%
``` 