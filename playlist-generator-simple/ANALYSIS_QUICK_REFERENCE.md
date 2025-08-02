# Analysis Step Quick Reference

Quick reference guide for the analysis step in Playlist Generator Simple.

## Quick Start

### Basic Analysis
```bash
# Analyze all files
playlista analyze

# Analyze with force re-extract
playlista analyze --force

# Analyze with custom workers
playlista analyze --workers 8
```

### Check Status
```bash
# Check analysis statistics
playlista stats

# Monitor resource usage
playlista status

# Check failed files
playlista failed
```

## Configuration Quick Reference

### Essential Settings
```conf
# Analysis Mode
ANALYSIS_MODE=smart

# File Size Thresholds
PARALLEL_MAX_FILE_SIZE_MB=20
SEQUENTIAL_MAX_FILE_SIZE_MB=50

# Worker Configuration
MAX_WORKERS=8
MEMORY_PER_WORKER_GB=0.5

# Timeouts
ANALYSIS_TIMEOUT_SECONDS=300
WORKER_TIMEOUT_SECONDS=300
```

### Performance Tuning
```conf
# High Performance
MAX_WORKERS=12
MEMORY_PER_WORKER_GB=0.3
PARALLEL_MAX_FILE_SIZE_MB=30

# Memory Constrained
MAX_WORKERS=4
MEMORY_PER_WORKER_GB=0.8
PARALLEL_MAX_FILE_SIZE_MB=15
```

## Architecture Overview

```
Files → Analysis Manager → File Categorization
                              │
                              ├── Large Files (>50MB) → Sequential Analyzer
                              └── Small Files (<50MB) → Parallel Analyzer
```

## Key Components

### AnalysisManager
- **Purpose**: Central coordinator
- **Key Method**: `analyze_files(files, force_reextract=False, max_workers=None)`
- **Responsibility**: File selection, categorization, routing

### SequentialAnalyzer
- **Purpose**: Process large files one at a time
- **Timeout**: 600 seconds per file
- **Memory**: Built-in cleanup between files
- **Use Case**: Files > 50MB

### ParallelAnalyzer
- **Purpose**: Process small files in parallel
- **Timeout**: 300 seconds per file
- **Workers**: Auto-determined based on memory
- **Use Case**: Files < 50MB

### ResourceManager
- **Purpose**: Calculate optimal worker count
- **Formula**: `min(available_memory_gb / 0.5, cpu_count)`
- **Memory**: 500MB per worker estimate

## Common Commands

### Analysis Commands
```bash
# Basic analysis
playlista analyze

# Force re-analysis
playlista analyze --force

# Custom worker count
playlista analyze --workers 12

# Debug mode
playlista analyze --log-level DEBUG
```

### Monitoring Commands
```bash
# Check statistics
playlista stats

# Check status
playlista status

# Check failed files
playlista failed

# Retry failed files
playlista retry
```

## Troubleshooting Quick Reference

### Memory Issues
```conf
# Reduce workers
MAX_WORKERS=4
MEMORY_PER_WORKER_GB=0.8

# Increase sequential threshold
SEQUENTIAL_MAX_FILE_SIZE_MB=30
```

### Timeout Issues
```conf
# Increase timeout
ANALYSIS_TIMEOUT_SECONDS=600
WORKER_TIMEOUT_SECONDS=300
```

### Performance Issues
```conf
# Increase workers
MAX_WORKERS=12
MEMORY_PER_WORKER_GB=0.3

# Adjust file size thresholds
PARALLEL_MAX_FILE_SIZE_MB=30
```

## Performance Metrics

### Key Metrics
- **Success Rate**: `(successful / total) * 100`
- **Throughput**: `files / seconds`
- **Memory Usage**: Peak memory during analysis
- **Processing Time**: Total analysis time

### Benchmarking
```python
import time

start = time.time()
results = analysis_manager.analyze_files(files)
end = time.time()

throughput = len(files) / (end - start)
success_rate = (results['success_count'] / len(files)) * 100

print(f"Throughput: {throughput:.2f} files/s")
print(f"Success rate: {success_rate:.1f}%")
```

## Environment Variables

### Override Configuration
```bash
export MAX_WORKERS=12
export MEMORY_PER_WORKER_GB=0.3
export PARALLEL_MAX_FILE_SIZE_MB=30
export ANALYSIS_TIMEOUT_SECONDS=600
```

### Priority Order
1. Environment variables (highest)
2. Configuration file (`playlista.conf`)
3. Default values (lowest)

## File Processing Flow

### Sequential Processing
```
1. Check file exists
2. Get file size
3. Check memory before processing
4. Process file in same process
5. Save results to database
6. Cleanup memory
7. Repeat for next file
```

### Parallel Processing
```
1. Calculate optimal worker count
2. Create ProcessPoolExecutor
3. Submit all files to worker pool
4. Collect results as they complete
5. Handle timeouts and errors
6. Fallback to sequential if needed
```

## Error Handling

### Automatic Fallbacks
- **Memory Critical**: Sequential processing
- **Worker Pool Error**: Sequential processing
- **Timeout**: Mark as failed, retry later
- **File Not Found**: Mark as failed

### Manual Recovery
```bash
# Retry failed files
playlista retry

# Check failed files
playlista failed

# Force re-analysis
playlista analyze --force
```

## Best Practices

### 1. Monitor Resources
```python
resource_manager = ResourceManager()
if resource_manager.is_memory_critical():
    print("High memory usage detected")
```

### 2. Optimize Worker Count
```python
optimal_workers = resource_manager.get_optimal_worker_count()
print(f"Optimal workers: {optimal_workers}")
```

### 3. Batch Processing
```python
batch_size = 100
for i in range(0, len(files), batch_size):
    batch = files[i:i + batch_size]
    results = analysis_manager.analyze_files(batch)
```

### 4. Monitor Progress
```python
stats = analysis_manager.get_analysis_statistics()
print(f"Success rate: {stats['success_rate']:.1f}%")
```

## Configuration Examples

### High-Performance System
```conf
MAX_WORKERS=12
MEMORY_PER_WORKER_GB=0.3
PARALLEL_MAX_FILE_SIZE_MB=30
ANALYSIS_TIMEOUT_SECONDS=600
```

### Memory-Constrained System
```conf
MAX_WORKERS=4
MEMORY_PER_WORKER_GB=0.8
PARALLEL_MAX_FILE_SIZE_MB=15
ANALYSIS_TIMEOUT_SECONDS=300
```

### Balanced System
```conf
MAX_WORKERS=8
MEMORY_PER_WORKER_GB=0.5
PARALLEL_MAX_FILE_SIZE_MB=20
ANALYSIS_TIMEOUT_SECONDS=300
```

## Debug Commands

### Enable Debug Logging
```bash
playlista analyze --log-level DEBUG
```

### Check Resource Usage
```bash
playlista status
```

### Monitor Analysis Progress
```bash
playlista stats
```

This quick reference provides essential information for working with the analysis step in Playlist Generator Simple. 