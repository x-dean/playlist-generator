# Container Memory Limit Configuration Guide

## Overview

When running the playlist generator in Docker/LXC containers, the system reports the host's memory instead of the container's allocated memory limits. This leads to incorrect worker calculations and suboptimal performance. The new `CONTAINER_MEMORY_LIMIT_GB` setting allows you to specify the actual memory limit for your container environment.

## Problem

In containerized environments:
- **Host memory detection**: Reports the full host memory (e.g., 32GB)
- **Container reality**: Container may only have 2-4GB allocated
- **Result**: System calculates workers based on host memory, leading to:
  - Too many workers allocated
  - Memory exhaustion
  - Container crashes
  - Poor performance

## Solution

The `CONTAINER_MEMORY_LIMIT_GB` setting overrides host memory detection and uses your specified container memory limit for worker calculations.

## Configuration

### 1. Edit Configuration File

Add the setting to `config/playlista.conf`:

```bash
# Container/Environment Memory Limit (overrides host memory detection)
# Set this to your container's memory limit (e.g., "4GB", "2.5GB", "512MB")
# Leave empty to use host memory detection
CONTAINER_MEMORY_LIMIT_GB=4.0
```

### 2. Configuration Options

| Value | Behavior |
|-------|----------|
| `CONTAINER_MEMORY_LIMIT_GB=` | Use host memory detection (default) |
| `CONTAINER_MEMORY_LIMIT_GB=2.0` | Use 2GB as container memory limit |
| `CONTAINER_MEMORY_LIMIT_GB=4.5` | Use 4.5GB as container memory limit |
| `CONTAINER_MEMORY_LIMIT_GB=1.0` | Use 1GB as container memory limit |

### 3. Docker Example

For a Docker container with 4GB memory limit:

```bash
# docker-compose.yml
services:
  playlist-generator:
    image: playlist-generator:latest
    deploy:
      resources:
        limits:
          memory: 4G
    environment:
      - CONTAINER_MEMORY_LIMIT_GB=4.0
```

Or in `config/playlista.conf`:
```bash
CONTAINER_MEMORY_LIMIT_GB=4.0
```

### 4. LXC Example

For an LXC container with 2GB memory limit:

```bash
# In config/playlista.conf
CONTAINER_MEMORY_LIMIT_GB=2.0
```

## How It Works

### Memory Calculation

1. **Host Mode** (default):
   ```
   Total Memory = Host Memory (e.g., 32GB)
   Available for Workers = Host Memory - Reserved Memory - Current Usage
   ```

2. **Container Mode** (when `CONTAINER_MEMORY_LIMIT_GB` is set):
   ```
   Total Memory = Container Memory Limit (e.g., 4GB)
   Available for Workers = Container Limit - Reserved Memory - Current Usage
   ```

### Worker Calculation

The system calculates optimal workers using:
- **Memory-based**: `available_memory_gb / memory_per_worker_gb` (1.2GB per worker)
- **CPU-based**: `cpu_count // 2` (half of CPU cores)
- **Conservative approach**: Uses minimum of memory and CPU-based workers

### Example Calculations

| Container Limit | Reserved Memory | Available for Workers | Memory-based Workers | Final Workers |
|----------------|-----------------|---------------------|---------------------|---------------|
| 2GB | 0.3GB | 1.7GB | 1 | 1 |
| 4GB | 0.6GB | 3.4GB | 2 | 2 |
| 6GB | 0.9GB | 5.1GB | 4 | 4 |
| 8GB | 1.2GB | 6.8GB | 5 | 5 |

## Benefits

1. **Accurate Resource Usage**: Workers calculated based on actual available memory
2. **Prevents OOM**: Avoids memory exhaustion in containers
3. **Better Performance**: Optimal worker count for container environment
4. **Flexible Configuration**: Easy to adjust for different container sizes
5. **Backward Compatible**: Defaults to host detection if not specified

## Monitoring

The system provides detailed logging when container mode is active:

```
INFO Resource: Container mode: Using 4.00GB as total memory (host reports 32.00GB)
INFO Resource: Container memory usage: 45.2%
INFO Resource: Container mode: Using 4.00GB limit
```

## Troubleshooting

### Issue: Still using host memory
**Solution**: Check that `CONTAINER_MEMORY_LIMIT_GB` is set and not empty

### Issue: Too few workers
**Solution**: Increase `CONTAINER_MEMORY_LIMIT_GB` or decrease `MEMORY_PER_WORKER_GB`

### Issue: Too many workers
**Solution**: Decrease `CONTAINER_MEMORY_LIMIT_GB` or increase `MEMORY_PER_WORKER_GB`

### Issue: Container still running out of memory
**Solution**: 
1. Verify the container memory limit matches `CONTAINER_MEMORY_LIMIT_GB`
2. Reduce `CONTAINER_MEMORY_LIMIT_GB` to account for other processes
3. Increase container memory allocation

## Testing

Use the test script to verify your configuration:

```bash
python test_container_memory.py
```

This will show worker calculations for different container memory limits.

## Migration

### From Host Detection to Container Mode

1. **Determine your container memory limit**
2. **Set `CONTAINER_MEMORY_LIMIT_GB`** in config
3. **Monitor logs** for container mode activation
4. **Adjust if needed** based on performance

### Example Migration

```bash
# Before (host detection)
# CONTAINER_MEMORY_LIMIT_GB=  # Empty, uses host memory

# After (container mode)
CONTAINER_MEMORY_LIMIT_GB=4.0  # 4GB container limit
```

## Best Practices

1. **Set realistic limits**: Use actual container memory allocation
2. **Monitor usage**: Check logs for memory usage patterns
3. **Test thoroughly**: Verify worker counts in your environment
4. **Adjust gradually**: Start conservative and increase if needed
5. **Document limits**: Keep track of container memory allocations 