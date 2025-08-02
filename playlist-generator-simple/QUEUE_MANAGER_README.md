# Queue Manager for Parallel Processing

## Overview

The Queue Manager is a sophisticated task management system designed to handle parallel audio file processing with advanced features like priority queuing, automatic retries, progress monitoring, and memory-aware resource management.

## Key Features

### 1. Priority-Based Task Queuing
- **High Priority Tasks**: Processed first regardless of queue order
- **Dynamic Priority Adjustment**: Failed tasks get lower priority on retry
- **Creation Time Ordering**: Earlier tasks processed first when priorities are equal

### 2. Automatic Retry Mechanism
- **Configurable Retry Count**: Default 3 retries per task
- **Exponential Backoff**: Retry delay increases with each attempt
- **Failure Tracking**: Comprehensive error logging and database persistence

### 3. Progress Monitoring and Statistics
- **Real-time Progress**: Live updates on processing status
- **Detailed Statistics**: Throughput, average processing time, success rates
- **Custom Callbacks**: User-defined progress update functions

### 4. Memory-Aware Worker Management
- **Dynamic Worker Adjustment**: Reduces workers under high memory pressure
- **Memory Monitoring**: Continuous system memory usage tracking
- **Garbage Collection**: Forced cleanup between batches

### 5. Database Integration
- **Persistent State**: Task status saved to database
- **Error Tracking**: Failed analysis recorded with error messages
- **Cache Integration**: Works with existing analysis cache system

## Architecture

### Core Components

#### QueueManager Class
```python
class QueueManager:
    def __init__(self, max_workers=None, queue_size=1000, 
                 worker_timeout=300, max_retries=3, ...)
```

**Key Methods:**
- `add_task()` / `add_tasks()`: Add files to processing queue
- `start_processing()`: Begin parallel processing
- `stop_processing()`: Gracefully stop all workers
- `get_statistics()`: Get current processing statistics
- `get_task_status()`: Check individual task status

#### ProcessingTask Class
```python
@dataclass
class ProcessingTask:
    file_path: str
    task_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    retry_count: int = 0
    # ... additional fields
```

**Features:**
- Automatic task ID generation
- Priority-based comparison for queue ordering
- Comprehensive status tracking

#### TaskStatus Enum
```python
class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"
```

## Usage Examples

### Basic Usage

```python
from core.queue_manager import get_queue_manager

# Get global queue manager instance
queue_manager = get_queue_manager()

# Add files to queue
task_ids = queue_manager.add_tasks([
    "/path/to/audio1.mp3",
    "/path/to/audio2.mp3",
    "/path/to/audio3.mp3"
])

# Start processing
success = queue_manager.start_processing()

# Monitor progress
while True:
    stats = queue_manager.get_statistics()
    if stats['pending_tasks'] == 0 and stats['processing_tasks'] == 0:
        break
    time.sleep(1)

# Stop processing
queue_manager.stop_processing()
```

### Advanced Usage with Progress Callback

```python
def progress_callback(stats):
    progress = (stats['completed_tasks'] + stats['failed_tasks']) / stats['total_tasks'] * 100
    print(f"Progress: {progress:.1f}% ({stats['completed_tasks']}/{stats['total_tasks']})")

# Start with progress monitoring
queue_manager.start_processing(progress_callback)
```

### Priority-Based Processing

```python
# High priority task (processed first)
high_priority_id = queue_manager.add_task("/path/to/important.mp3", priority=10)

# Normal priority task
normal_priority_id = queue_manager.add_task("/path/to/normal.mp3", priority=5)

# Low priority task (processed last)
low_priority_id = queue_manager.add_task("/path/to/low.mp3", priority=1)
```

## Integration with Analysis Manager

The Queue Manager is seamlessly integrated with the Analysis Manager:

```python
# Analysis Manager automatically uses Queue Manager for small files
analysis_manager = AnalysisManager()
results = analysis_manager.analyze_files(file_list)
```

**Integration Features:**
- Automatic file categorization (small vs large files)
- Queue-based processing for small files
- Fallback to direct parallel processing if queue fails
- Progress reporting through analysis manager

## Configuration Options

### Queue Manager Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_workers` | Auto-determined | Maximum number of worker processes |
| `queue_size` | 1000 | Maximum number of tasks in queue |
| `worker_timeout` | 300s | Timeout for individual worker processes |
| `max_retries` | 3 | Maximum retry attempts per task |
| `retry_delay` | 5s | Delay between retry attempts |
| `progress_update_interval` | 10s | Progress update frequency |

### Memory Management Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `memory_threshold_percent` | 85% | Memory usage threshold for warnings |
| `memory_check_interval` | 30s | Memory monitoring frequency |
| `worker_memory_limit` | 256MB | Per-worker memory limit |

## Performance Characteristics

### Memory Usage
- **Per Worker**: 256MB limit (configurable)
- **Available Memory Check**: 300MB minimum required
- **Dynamic Adjustment**: Workers reduced under memory pressure

### Throughput
- **Optimal Workers**: Auto-determined based on system resources
- **Batch Processing**: Configurable batch sizes
- **Garbage Collection**: Forced cleanup between batches

### Error Handling
- **Retry Logic**: Automatic retry with exponential backoff
- **Failure Tracking**: Comprehensive error logging
- **Database Persistence**: Failed tasks recorded in analysis_cache

## Monitoring and Debugging

### Statistics Available

```python
stats = queue_manager.get_statistics()
print(f"Total tasks: {stats['total_tasks']}")
print(f"Completed: {stats['completed_tasks']}")
print(f"Failed: {stats['failed_tasks']}")
print(f"Retries: {stats['retry_tasks']}")
print(f"Pending: {stats['pending_tasks']}")
print(f"Processing: {stats['processing_tasks']}")
print(f"Throughput: {stats['throughput']:.2f} tasks/s")
print(f"Average time: {stats['average_processing_time']:.2f}s")
```

### Task Status Monitoring

```python
status = queue_manager.get_task_status(task_id)
if status:
    print(f"Task {task_id}: {status['status']}")
    print(f"Worker: {status['worker_id']}")
    print(f"Started: {status['started_at']}")
    print(f"Completed: {status['completed_at']}")
    if status['error_message']:
        print(f"Error: {status['error_message']}")
```

### Logging

The Queue Manager provides comprehensive logging:

```
Queue: Added task task_1234 for audio.mp3 (priority: 5)
Queue: Started processing with 4 workers
Queue: Progress: 45.2% (23/51)
Queue: Task task_1234 retry 1/3
Queue: Task task_1234 failed after 3 retries
Queue: High memory usage: 87.3%
Queue: Reduced workers from 4 to 2
```

## Error Handling

### Common Error Scenarios

1. **Memory Exhaustion**
   - Workers automatically reduced
   - Garbage collection forced
   - Processing continues with fewer workers

2. **Worker Timeout**
   - Task marked as failed
   - Retry mechanism activated
   - Error logged to database

3. **File Not Found**
   - Task marked as failed immediately
   - No retry attempted
   - Error logged with file path

4. **Analysis Failure**
   - Task retried up to max_retries
   - Error message captured
   - Database updated with failure status

### Recovery Mechanisms

- **Automatic Retries**: Failed tasks retried with backoff
- **Worker Recovery**: New workers spawned if needed
- **Queue Recovery**: Queue state preserved across restarts
- **Database Recovery**: Failed tasks can be retried later

## Best Practices

### 1. Resource Management
- Monitor memory usage during processing
- Adjust worker count based on system capabilities
- Use appropriate timeout values for your files

### 2. Error Handling
- Implement proper error callbacks
- Monitor retry statistics
- Handle queue full conditions gracefully

### 3. Performance Optimization
- Use appropriate batch sizes
- Monitor throughput statistics
- Adjust priority levels based on file importance

### 4. Monitoring
- Implement progress callbacks for user feedback
- Log important statistics
- Monitor worker health and memory usage

## Testing

The Queue Manager includes comprehensive testing:

```bash
python test_queue_manager.py
```

**Test Coverage:**
- Basic functionality testing
- Integration with analysis manager
- Advanced features (priority, retries)
- Memory management
- Global instance management

## Future Enhancements

### Planned Features
- **Distributed Processing**: Multi-machine queue processing
- **Persistent Queues**: Queue state saved to disk
- **Advanced Scheduling**: Time-based and dependency-based scheduling
- **Web Interface**: Real-time monitoring dashboard
- **Metrics Export**: Prometheus/Grafana integration

### Performance Improvements
- **Worker Pool Optimization**: Dynamic worker allocation
- **Memory Optimization**: Streaming audio processing
- **Cache Optimization**: Intelligent caching strategies
- **Network Optimization**: Distributed processing support

## Troubleshooting

### Common Issues

1. **Queue Full Errors**
   - Increase queue_size parameter
   - Process files in smaller batches
   - Monitor queue statistics

2. **Memory Issues**
   - Reduce max_workers
   - Increase memory thresholds
   - Monitor system memory usage

3. **Timeout Errors**
   - Increase worker_timeout
   - Check file sizes and complexity
   - Monitor system resources

4. **Worker Failures**
   - Check system resources
   - Review error logs
   - Adjust retry settings

### Debug Mode

Enable detailed logging for debugging:

```python
import logging
logging.getLogger('playlista.queue_manager').setLevel(logging.DEBUG)
```

## Conclusion

The Queue Manager provides a robust, scalable solution for parallel audio file processing with advanced features for reliability, monitoring, and resource management. Its integration with the Analysis Manager makes it seamless to use while providing powerful capabilities for handling large-scale audio processing tasks. 