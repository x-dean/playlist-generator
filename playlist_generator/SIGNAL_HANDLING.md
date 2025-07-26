# Signal Handling and Graceful Shutdown

This document explains the improved signal handling implementation that allows for graceful shutdown when Ctrl+C is pressed.

## Overview

The application now properly handles Ctrl+C (SIGINT) signals to achieve the following goals:

1. **Stop feeding new files to workers** - Prevents new files from being queued for processing
2. **Let workers finish their current tasks** - Workers complete their current file processing
3. **Prevent new file discovery** - Stops scanning for new files to process
4. **Provide clear feedback** - Shows progress during graceful shutdown

## Implementation Details

### Main Signal Handler (`playlista`)

The main signal handler is implemented in `playlista` and provides:

- **60-second grace period** for workers to finish current tasks
- **Real-time status updates** showing active worker count
- **Automatic cleanup** of child processes if timeout is reached
- **Clear user feedback** with emoji indicators

```python
def handle_sigint(signum, frame):
    logger.info("Received SIGINT (Ctrl+C), initiating graceful shutdown...")
    print("\n🛑 Process termination requested - stopping gracefully...")
    stop_event.set()
    
    # Give workers time to finish current tasks
    timeout = 60  # 60 seconds grace period
    # ... monitoring and cleanup logic
```

### Parallel Processor Improvements

The `ParallelProcessor` class now:

- **Checks stop events frequently** - Before starting new batches, after each file completion
- **Stops feeding new files** - Prevents new files from being queued when stop event is set
- **Lets current workers finish** - Doesn't terminate pools immediately, allows current tasks to complete
- **Provides detailed logging** - Shows exactly when and where stop events are detected

Key improvements:
```python
# Check stop_event before starting new batch
if stop_event and stop_event.is_set():
    logger.info("Stop event detected - stopping new file processing")
    break

# Check stop_event after each file completion
for features, filepath, db_write_success in pool.imap_unordered(worker_func, batch):
    if stop_event and stop_event.is_set():
        logger.info("Stop event detected during processing - stopping gracefully")
        break
```

### Sequential Processor Improvements

The `SequentialProcessor` class now:

- **Checks stop events before each file** - Prevents starting new file processing
- **Checks stop events after each file** - Stops gracefully after completing current file
- **Provides clear feedback** - Shows when stop events are detected in sequential mode

### Worker Function Improvements

The `process_file_worker` function now:

- **Checks for interruption before processing** - Detects if parent process is still running
- **Avoids database writes during shutdown** - Prevents errors during graceful shutdown
- **Provides clean exit** - Returns gracefully when interrupted

```python
def is_interrupted():
    """Check if the parent process is still running (indicates interruption)"""
    try:
        import psutil
        parent_pid = os.getppid()
        return not psutil.pid_exists(parent_pid)
    except:
        return False

# Check for interruption before starting processing
if is_interrupted():
    logger.info(f"Worker interrupted - stopping processing of {filepath}")
    return None, filepath, False
```

### Analysis Manager Improvements

The analysis manager functions now:

- **Check stop events in main loops** - Detect interruptions during analysis phases
- **Provide summary statistics** - Show progress when interrupted
- **Give clear feedback** - Display which mode was interrupted

## User Experience

### When Ctrl+C is Pressed

1. **Immediate feedback**: "🛑 Process termination requested - stopping gracefully..."
2. **Stop feeding files**: New files stop being queued for processing
3. **Monitor workers**: "⏳ Waiting for workers to finish current tasks..."
4. **Real-time updates**: "⏳ X workers still active, Ys remaining..."
5. **Graceful completion**: "✅ All workers finished - exiting gracefully."
6. **Summary**: Shows how many files were processed before interruption

### Grace Period

- **60 seconds** to allow workers to finish current tasks
- **Automatic cleanup** if timeout is reached
- **Process monitoring** using `psutil` to track active workers

### Error Handling

- **Database errors ignored** during shutdown to prevent corruption
- **Worker processes terminated** if they don't finish within grace period
- **Clean exit codes** to indicate graceful vs forced shutdown

## Testing

A test script is provided (`test_signal_handling.py`) that demonstrates the signal handling functionality:

```bash
python test_signal_handling.py
```

This script simulates the processing pipeline and allows you to test Ctrl+C handling.

## Configuration

The signal handling behavior can be customized by modifying:

- **Grace period timeout**: Change `timeout = 60` in the signal handler
- **Worker monitoring frequency**: Adjust sleep intervals in monitoring loops
- **Cleanup behavior**: Modify the `cleanup_child_processes()` function

## Best Practices

1. **Always check stop events** before starting new work
2. **Let workers finish current tasks** rather than terminating immediately
3. **Provide clear feedback** to users about shutdown progress
4. **Handle database operations carefully** during shutdown
5. **Use appropriate timeouts** to balance responsiveness with data safety

## Troubleshooting

### Workers Not Finishing

If workers don't finish within the grace period:
- Check if workers are stuck in infinite loops
- Verify that workers check for stop events properly
- Consider increasing the grace period timeout

### Database Errors During Shutdown

If you see database errors during shutdown:
- These are expected and handled gracefully
- Workers avoid database writes when interrupted
- No data corruption should occur

### Force Termination

If graceful shutdown fails:
- The system will force terminate after the grace period
- All child processes will be killed
- A summary of processed files will be shown

## Future Improvements

Potential enhancements:
- **Configurable grace periods** via command line arguments
- **More detailed progress reporting** during shutdown
- **Automatic recovery** of interrupted operations
- **Persistent state tracking** to resume interrupted processing 