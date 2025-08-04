# Analysis Step Documentation

This document provides comprehensive documentation for the analysis step in Playlist Generator Simple, focusing on workers and sequential processing.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Analysis Manager](#analysis-manager)
4. [Sequential Analyzer](#sequential-analyzer)
5. [Parallel Analyzer](#parallel-analyzer)
6. [Resource Management](#resource-management)
7. [Configuration](#configuration)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

## Overview

The analysis step is responsible for processing audio files to extract musical features and metadata. It uses a dual-approach system:

- **Sequential Processing**: For large files (>50MB) to manage memory usage
- **Parallel Processing**: For smaller files (<50MB) to maximize throughput

### Key Components

- **AnalysisManager**: Coordinates file selection and routing
- **SequentialAnalyzer**: Processes large files one at a time
- **ParallelAnalyzer**: Processes small files in parallel
- **ResourceManager**: Manages worker count and resource limits

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Input    │───▶│ Analysis Manager │───▶│ File Categorize │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Large Files     │◀───│ Sequential       │    │ Parallel        │◀───│ Small Files   │
│ (>50MB)         │    │ Analyzer         │    │ Analyzer        │    │ (<50MB)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Database Storage │    │ Process Pool    │
                       └──────────────────┘    └─────────────────┘
```

## Analysis Manager

The `AnalysisManager` is the central coordinator that handles file selection, categorization, and routing.

### Key Methods

#### `analyze_files(files, force_reextract=False, max_workers=None)`

Main entry point for file analysis.

**Parameters:**
- `files`: List of file paths to analyze
- `force_reextract`: Bypass cache for all files
- `max_workers`: Maximum parallel workers (auto-determined if None)

**Returns:**
```python
{
    'success_count': int,
    'failed_count': int,
    'total_time': float,
    'big_files_processed': int,
    'small_files_processed': int
}
```

#### `select_files_for_analysis(music_path=None, force_reextract=False, include_failed=False)`

Selects files for analysis based on various criteria.

**Parameters:**
- `music_path`: Path to music directory
- `force_reextract`: Include already analyzed files
- `include_failed`: Include previously failed files

**Returns:** List of file paths to analyze

#### `_categorize_files_by_size(files)`

Categorizes files by size for appropriate processing.

**Parameters:**
- `files`: List of file paths

**Returns:** Tuple of (big_files, small_files)

### File Categorization Logic

```python
def _categorize_files_by_size(self, files):
    big_files = []
    small_files = []
    
    for file_path in files:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb >= self.big_file_size_mb:  # Default: 50MB
            big_files.append(file_path)
        else:
            small_files.append(file_path)
    
    return big_files, small_files
```

## Sequential Analyzer

The `SequentialAnalyzer` processes large files one at a time to manage memory usage.

### Key Features

- **Memory Management**: Built-in cleanup between files
- **Timeout Protection**: 600-second timeout per file
- **Error Recovery**: Comprehensive error handling
- **Resource Monitoring**: Memory threshold checks

### Configuration

```python
DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes
DEFAULT_MEMORY_THRESHOLD_PERCENT = 85
DEFAULT_RSS_LIMIT_GB = 6.0
```

### Processing Flow

```
1. Check file exists
2. Get file size
3. Check memory before processing
4. Process file in same process
5. Save results to database
6. Cleanup memory
7. Repeat for next file
```

### Key Methods

#### `process_files(files, force_reextract=False)`

Processes files sequentially.

**Parameters:**
- `files`: List of file paths
- `force_reextract`: Bypass cache

**Returns:**
```python
{
    'success_count': int,
    'failed_count': int,
    'total_time': float,
    'processed_files': List[Dict]
}
```

#### `_process_single_file(file_path, force_reextract=False)`

Processes a single file with timeout and memory monitoring.

**Parameters:**
- `file_path`: Path to file
- `force_reextract`: Bypass cache

**Returns:** True if successful, False otherwise

### Memory Management

```python
def _cleanup_memory(self):
    """Force memory cleanup between files."""
    gc.collect()  # Force garbage collection
    # Log memory usage after cleanup
```

## Parallel Analyzer

The `ParallelAnalyzer` processes smaller files in parallel for maximum throughput.

### Key Features

- **Auto Worker Count**: Calculates optimal workers based on resources
- **Process Pool**: Uses `ProcessPoolExecutor`
- **Standalone Workers**: Avoids pickling issues
- **Fallback Mechanism**: Falls back to sequential if needed

### Configuration

```python
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_MEMORY_THRESHOLD_PERCENT = 85
DEFAULT_MAX_WORKERS = None  # Auto-determined
```

### Processing Flow

```
1. Calculate optimal worker count
2. Create ProcessPoolExecutor
3. Submit all files to worker pool
4. Collect results as they complete
5. Handle timeouts and errors
6. Fallback to sequential if needed
```

### Key Methods

#### `process_files(files, force_reextract=False, max_workers=None)`

Processes files in parallel.

**Parameters:**
- `files`: List of file paths
- `force_reextract`: Bypass cache
- `max_workers`: Maximum workers (auto-determined if None)

**Returns:**
```python
{
    'success_count': int,
    'failed_count': int,
    'total_time': float,
    'processed_files': List[Dict],
    'worker_count': int
}
```

#### `get_optimal_worker_count(max_workers=None)`

Calculates optimal worker count for parallel processing.

**Parameters:**
- `max_workers`: Maximum workers (auto-determined if None)

**Returns:** Optimal number of workers

### Standalone Worker Process

```python
def _standalone_worker_process(file_path, force_reextract=False, 
                             timeout_seconds=300, db_path=None,
                             analysis_config=None) -> bool:
    """
    Standalone worker function for multiprocessing.
    
    This function can be pickled and runs in separate processes.
    """
    # Set up timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        # Import audio analyzer
        from .audio_analyzer import AudioAnalyzer
        
        # Create analyzer with configuration
        analyzer = AudioAnalyzer(config=analysis_config)
        
        # Extract features
        analysis_result = analyzer.analyze_audio_file(file_path, force_reextract)
        
        # Save to database
        if analysis_result:
            # Save results...
            return True
        else:
            return False
            
    except TimeoutException:
        return False
    except Exception as e:
        return False
```

## Resource Management

The `ResourceManager` handles worker count calculation and resource monitoring.

### Worker Count Calculation

```python
def get_optimal_worker_count(self, max_workers=None, memory_limit_str=None) -> int:
    """
    Calculate optimal worker count based on available memory.
    """
    # Parse memory limit
    memory_limit_gb = self.memory_limit_gb
    
    # Get current memory usage
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    # Estimate memory per worker (conservative estimate)
    memory_per_worker_gb = 0.5  # 500MB per worker
    
    # Calculate optimal workers based on available memory
    memory_based_workers = max(1, int(available_gb / memory_per_worker_gb))
    
    # Get CPU count
    cpu_count = mp.cpu_count()
    
    # Use the minimum of memory-based and CPU-based workers
    optimal_workers = min(memory_based_workers, cpu_count)
    
    # Apply max_workers limit
    if max_workers:
        optimal_workers = min(optimal_workers, max_workers)
    
    return optimal_workers
```

### Memory Monitoring

```python
def is_memory_critical(self) -> bool:
    """Check if memory usage is critical."""
    memory = psutil.virtual_memory()
    return memory.percent > 90 or memory.used / (1024**3) > self.memory_limit_gb
```

## Configuration

### Analysis Configuration

```conf
# Analysis Mode
ANALYSIS_MODE=smart
ANALYSIS_TIMEOUT_SECONDS=300

# File Size Thresholds
PARALLEL_MAX_FILE_SIZE_MB=20
SEQUENTIAL_MAX_FILE_SIZE_MB=50

# Worker Configuration
MAX_WORKERS=8
WORKER_TIMEOUT_SECONDS=300
MEMORY_PER_WORKER_GB=0.5

# Resource Limits
MEMORY_LIMIT_GB=6.0
CPU_THRESHOLD_PERCENT=90
```

### Environment Variables

```bash
# Override configuration via environment variables
export MAX_WORKERS=12
export MEMORY_PER_WORKER_GB=0.3
export PARALLEL_MAX_FILE_SIZE_MB=30
```

### Configuration Priority

1. Environment variables (highest priority)
2. Configuration file (`playlista.conf`)
3. Default values (lowest priority)

## Performance Optimization

### Worker Count Optimization

**For High-Performance Systems:**
```conf
MAX_WORKERS=12
MEMORY_PER_WORKER_GB=0.3
PARALLEL_MAX_FILE_SIZE_MB=30
```

**For Memory-Constrained Systems:**
```conf
MAX_WORKERS=4
MEMORY_PER_WORKER_GB=0.8
PARALLEL_MAX_FILE_SIZE_MB=15
```

### File Size Thresholds

**Optimize for your file distribution:**

```python
# If most files are < 30MB
PARALLEL_MAX_FILE_SIZE_MB=30

# If most files are > 100MB
SEQUENTIAL_MAX_FILE_SIZE_MB=100
```

### Memory Management

**Monitor memory usage:**
```python
# Check current memory usage
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent}%")
print(f"Available: {memory.available / (1024**3):.2f}GB")
```

## Troubleshooting

### Common Issues

#### 1. Memory Errors

**Symptoms:**
- `MemoryError` exceptions
- High memory usage
- Process killed by OOM killer

**Solutions:**
```conf
# Reduce worker count
MAX_WORKERS=4
MEMORY_PER_WORKER_GB=0.8

# Increase file size threshold for sequential
SEQUENTIAL_MAX_FILE_SIZE_MB=30
```

#### 2. Timeout Errors

**Symptoms:**
- Analysis times out
- Files marked as failed

**Solutions:**
```conf
# Increase timeout for large files
ANALYSIS_TIMEOUT_SECONDS=600

# Reduce timeout for parallel processing
WORKER_TIMEOUT_SECONDS=180
```

#### 3. Worker Pool Errors

**Symptoms:**
- Multiprocessing errors
- Pickling errors

**Solutions:**
- The system automatically falls back to sequential processing
- Check system resources
- Reduce `MAX_WORKERS`

### Debug Mode

Enable debug logging to see detailed analysis information:

```bash
playlista analyze --log-level DEBUG
```

### Performance Monitoring

Monitor analysis performance:

```bash
# Check analysis statistics
playlista stats

# Monitor resource usage
playlista status
```

## API Reference

### AnalysisManager

```python
class AnalysisManager:
    def __init__(self, db_manager=None, config=None)
    def analyze_files(self, files, force_reextract=False, max_workers=None)
    def select_files_for_analysis(self, music_path=None, force_reextract=False, include_failed=False)
    def get_analysis_statistics(self)
    def final_retry_failed_files(self)
```

### SequentialAnalyzer

```python
class SequentialAnalyzer:
    def __init__(self, db_manager=None, resource_manager=None, timeout_seconds=None, memory_threshold_percent=None, rss_limit_gb=None)
    def process_files(self, files, force_reextract=False)
    def _process_single_file(self, file_path, force_reextract=False)
    def get_config(self)
    def update_config(self, new_config)
```

### ParallelAnalyzer

```python
class ParallelAnalyzer:
    def __init__(self, db_manager=None, resource_manager=None, timeout_seconds=None, memory_threshold_percent=None, max_workers=None)
    def process_files(self, files, force_reextract=False, max_workers=None)
    def get_optimal_worker_count(self, max_workers=None)
    def get_config(self)
    def update_config(self, new_config)
```

### ResourceManager

```python
class ResourceManager:
    def __init__(self, config=None)
    def get_optimal_worker_count(self, max_workers=None, memory_limit_str=None)
    def is_memory_critical(self)
    def get_resource_statistics(self, minutes=60)
    def get_config(self)
    def update_config(self, new_config)
```

## Best Practices

### 1. Monitor Resource Usage

```python
# Check resource usage before analysis
resource_manager = ResourceManager()
if resource_manager.is_memory_critical():
    print("High memory usage detected")
```

### 2. Optimize Worker Count

```python
# Calculate optimal workers for your system
optimal_workers = resource_manager.get_optimal_worker_count()
print(f"Optimal workers: {optimal_workers}")
```

### 3. Handle Large File Sets

```python
# Process files in batches for large collections
batch_size = 100
for i in range(0, len(files), batch_size):
    batch = files[i:i + batch_size]
    results = analysis_manager.analyze_files(batch)
```

### 4. Monitor Progress

```python
# Check analysis progress
stats = analysis_manager.get_analysis_statistics()
print(f"Success rate: {stats['success_rate']:.1f}%")
```

## Performance Metrics

### Key Metrics

- **Success Rate**: Percentage of successfully analyzed files
- **Throughput**: Files processed per second
- **Memory Usage**: Peak memory usage during analysis
- **CPU Usage**: Average CPU utilization
- **Processing Time**: Total time for analysis

### Benchmarking

```python
# Benchmark analysis performance
import time

start_time = time.time()
results = analysis_manager.analyze_files(files)
end_time = time.time()

throughput = len(files) / (end_time - start_time)
success_rate = (results['success_count'] / len(files)) * 100

print(f"Throughput: {throughput:.2f} files/s")
print(f"Success rate: {success_rate:.1f}%")
```

This documentation provides a comprehensive guide to the analysis step, covering all aspects of workers and sequential processing in the Playlist Generator Simple system. 