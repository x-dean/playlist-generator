# Memory Fixes for Parallel Processing

## Problem
The playlist generator was experiencing memory issues during parallel processing, specifically:
- "Out of memory interning an attribute name" errors during metadata extraction
- "std::bad_alloc" errors during audio loading
- Memory exhaustion when processing large files or many files simultaneously

## Root Causes
1. **Metadata Extraction Memory Issues**: Mutagen library was consuming excessive memory when processing files with many tags
2. **Worker Process Memory Limits**: No memory limits on individual worker processes
3. **Aggressive Parallel Processing**: Too many workers and large batch sizes
4. **Lack of Memory Monitoring**: No proactive memory management during processing

## Solutions Implemented

### 1. Enhanced Metadata Extraction Memory Protection
**File**: `src/core/audio_analyzer.py`

- **Tag Limits**: Limited to 100 tags per file to prevent memory explosion
- **Value Size Limits**: Tag values limited to 1000 characters
- **List Size Limits**: Tag lists limited to 10 items
- **Memory Checks**: Skip metadata extraction if less than 100MB available
- **Error Handling**: Graceful fallback to basic metadata on memory errors
- **Garbage Collection**: Force GC before and after metadata extraction

### 2. Conservative Worker Process Memory Limits
**File**: `src/core/parallel_analyzer.py`

- **Memory Limit**: Reduced from 512MB to 256MB per worker process
- **Available Memory Threshold**: Increased from 200MB to 300MB minimum
- **Large File Detection**: Warn about files larger than 100MB
- **Memory Error Handling**: Specific handling for MemoryError exceptions
- **File Size Checks**: Pre-check file size before processing

### 3. Conservative Resource Management
**File**: `src/core/resource_manager.py`

- **Memory Per Worker**: Reduced from 1GB to 0.5GB per worker
- **Memory Threshold**: Lowered from 80% to 75% system memory usage
- **Available Memory Threshold**: Reduced from 1GB to 0.5GB minimum
- **Worker Count Reduction**: More aggressive worker reduction for memory stability

### 4. Optimized Batch Processing
**File**: `src/core/parallel_analyzer.py`

- **Batch Size**: Reduced from 2 to 1 file per worker
- **Max Batch Size**: Reduced from 10 to 5 files per batch
- **Low Worker Count**: Reduced from 5 to 3 files for low worker counts
- **Memory Monitoring**: Check memory usage before each batch
- **Garbage Collection**: Force GC between batches
- **Worker Count Reduction**: Reduce optimal workers by half for stability

### 5. Enhanced Error Handling
- **MemoryError Catching**: Specific handling for memory-related errors
- **Graceful Degradation**: Fall back to basic processing when memory is low
- **Database Logging**: Log memory errors to database for tracking
- **Process Isolation**: Better isolation between worker processes

## Configuration Changes

### Memory Limits
```python
# Per worker process memory limit
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, -1))

# Available memory threshold
if available_memory_mb < 300:  # Increased from 200MB
    return False

# System memory threshold
if system_memory_percent > 75:  # Reduced from 80%
    optimal_workers = max(1, optimal_workers // 2)
```

### Batch Processing
```python
# Conservative batch sizes
max_batch_size = min(len(files), max_workers)  # 1 file per worker
batch_size = max(1, min(max_batch_size, 5))    # Cap at 5 files

# Low worker count safety
if max_workers <= 2:
    batch_size = max(1, min(batch_size, 3))    # Cap at 3 files
```

### Metadata Extraction
```python
# Tag limits
max_tags = 100  # Limit number of tags
if tag_count >= max_tags:
    break

# Value size limits
if isinstance(value, str) and len(value) > 1000:
    value = value[:1000] + "..."

# List size limits
value = [str(item)[:500] for item in value[:10]]
```

## Testing

Run the memory test script to verify improvements:
```bash
python test_memory_fixes.py
```

## Expected Results

1. **Reduced Memory Usage**: Lower memory consumption per worker process
2. **Better Stability**: Fewer "Out of memory" errors during processing
3. **Graceful Degradation**: System continues working even under memory pressure
4. **Improved Monitoring**: Better visibility into memory usage during processing
5. **Conservative Processing**: More reliable but slightly slower processing

## Monitoring

The system now provides detailed logging for memory-related issues:
- Memory usage warnings when above 75%
- Large file detection and warnings
- Memory error logging with specific error messages
- Batch processing statistics with memory information

## Usage Recommendations

1. **For Large Files**: The system will automatically detect and handle large files more conservatively
2. **For Memory-Constrained Systems**: The reduced worker counts will provide better stability
3. **For High-Volume Processing**: Smaller batch sizes will prevent memory exhaustion
4. **For Monitoring**: Check logs for memory warnings and adjust processing accordingly

## Performance Impact

- **Slightly Slower Processing**: More conservative settings may reduce throughput by 10-20%
- **Much Better Stability**: Significantly reduced chance of memory-related crashes
- **Better Resource Utilization**: More predictable memory usage patterns
- **Improved Reliability**: Graceful handling of memory pressure situations 