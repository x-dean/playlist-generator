# Memory Optimization Guide

This document explains how to limit and optimize RAM usage in the playlist generator.

## Overview

The playlist generator can consume significant memory during audio analysis, especially when processing large files or using many parallel workers. This guide shows you how to control memory usage through various configuration options.

## Memory Usage Patterns

### High Memory Usage Scenarios
1. **Large Audio Files**: Files longer than 2-3 hours can consume 2-4GB each
2. **Parallel Processing**: Each worker loads audio files into memory
3. **Feature Extraction**: MFCC, chroma, and spectral analysis require significant RAM
4. **Batch Processing**: Multiple files processed simultaneously

### Memory-Efficient Scenarios
1. **Sequential Processing**: One file at a time (use `--workers=1`)
2. **Small Files**: Files under 30 minutes typically use <1GB
3. **Cached Results**: Already analyzed files skip heavy computation

## Configuration Options

### 1. Worker Count Control

**Default Behavior**: Uses all CPU cores
```bash
# Use all available CPUs (default)
docker compose run --rm playlista --analyze

# Limit to specific number of workers
docker compose run --rm playlista --analyze --workers 4

# Use sequential processing (lowest memory usage)
docker compose run --rm playlista --analyze --workers 1
```

**Memory-Aware Mode**: Automatically calculates optimal workers based on available RAM
```bash
# Let the system decide based on available memory
docker compose run --rm playlista --analyze  # No --workers flag = automatic memory-aware calculation
```

### 2. Batch Size Control

**Default**: Batch size equals number of workers
```bash
# Set custom batch size
docker compose run --rm playlista --analyze --batch_size 2

# Use environment variable
BATCH_SIZE=2 docker compose run --rm playlista --analyze
```

### 3. Low Memory Mode

**Automatic memory reduction**:
```bash
# Reduces workers and batch size automatically
docker compose run --rm playlista --analyze --low_memory
```

This mode:
- Reduces worker count by half
- Sets batch size to half the worker count
- Enables memory monitoring

### 4. Memory Limit Per Worker

**Set memory constraints**:
```bash
# Limit memory per worker
docker compose run --rm playlista --analyze --memory_limit "2GB"
docker compose run --rm playlista --analyze --memory_limit "512MB"
```

### 5. Environment Variables

**System-wide configuration**:
```bash
# Set maximum workers
export MAX_WORKERS=4

# Set batch size
export BATCH_SIZE=2

# Set memory limit per worker
export MEMORY_LIMIT_PER_WORKER="2GB"

# Run with environment variables
docker compose run --rm playlista --analyze
```

## Memory Monitoring

### Built-in Monitoring
The system automatically monitors memory usage and logs:
- Available memory at start
- Memory usage during processing
- Warnings when usage exceeds 85%
- Critical alerts when usage exceeds 95%

### Memory Status Logs
```
Memory-aware worker calculation:
  Available memory: 8.2GB
  Memory per worker: 2.0GB
  Workers by memory: 4
  Max workers: 8
  Optimal workers: 4
```

## Recommended Configurations

### For Low Memory Systems (<8GB RAM)
```bash
# Conservative approach
docker compose run --rm playlista --analyze --workers 2 --batch_size 1

# Or use low memory mode
docker compose run --rm playlista --analyze --low_memory
```

### For Medium Memory Systems (8-16GB RAM)
```bash
# Balanced approach
docker compose run --rm playlista --analyze --workers 4 --batch_size 2

# Or let system decide
docker compose run --rm playlista --analyze
```

### For High Memory Systems (>16GB RAM)
```bash
# Maximum performance
docker compose run --rm playlista --analyze --workers 8 --batch_size 4

# Or use all available resources
docker compose run --rm playlista --analyze
```

## Troubleshooting

### High Memory Usage
**Symptoms**:
- Process killed by system
- "Out of memory" errors
- System becomes unresponsive

**Solutions**:
1. Use sequential processing: `docker compose run --rm playlista --analyze --workers 1`
2. Enable low memory mode: `docker compose run --rm playlista --analyze --low_memory`
3. Reduce batch size: `docker compose run --rm playlista --analyze --batch_size 1`
4. Process files in smaller batches

### Memory Leaks
**Symptoms**:
- Memory usage increases over time
- Performance degrades during long runs

**Solutions**:
1. Restart the process periodically
2. Use smaller batch sizes
3. Monitor with system tools (htop, top)

## Advanced Configuration

### Custom Memory Limits
```bash
# Set specific memory constraints
export MAX_WORKERS=4
export BATCH_SIZE=2
export MEMORY_LIMIT_PER_WORKER="1.5GB"
docker compose run --rm playlista --analyze
```

### Docker Memory Limits
```yaml
# In docker-compose.yaml
services:
  playlista:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### System-Level Memory Management
```bash
# Monitor memory usage
watch -n 1 'free -h'

# Check process memory usage
ps aux --sort=-%mem | head -10

# Set system memory limits
ulimit -v 8589934592  # 8GB virtual memory limit
```

## Performance vs Memory Trade-offs

| Configuration | Memory Usage | Speed | Use Case |
|---------------|--------------|-------|----------|
| `--workers 1` | Lowest | Slowest | Debugging, low RAM |
| `--workers 2` | Low | Slow | Small systems |
| `--workers 4` | Medium | Medium | Balanced |
| `--workers 8` | High | Fast | High-end systems |
| Auto (default) | Adaptive | Adaptive | Most systems |

## Best Practices

1. **Start Conservative**: Begin with fewer workers and increase if needed
2. **Monitor Usage**: Watch memory usage during first few files
3. **Use Caching**: Let the system cache results to avoid re-processing
4. **Batch Processing**: Process large libraries in smaller batches
5. **System Resources**: Ensure adequate swap space is available

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_WORKERS` | CPU count | Maximum number of parallel workers |
| `BATCH_SIZE` | Worker count | Number of files processed per batch |
| `MEMORY_LIMIT_PER_WORKER` | None | Memory limit per worker (e.g., "2GB") |
| `LOG_LEVEL` | INFO | Logging level for memory monitoring |

## Monitoring Commands

```bash
# Real-time memory monitoring
watch -n 1 'free -h && echo "---" && ps aux --sort=-%mem | head -5'

# Check Docker container memory
docker stats playlista

# Monitor specific process
top -p $(pgrep -f playlista)
``` 