# Queue Manager Load Balancing Features

## Overview

The Queue Manager now includes advanced load balancing capabilities that dynamically spawn and reduce workers based on specific criteria. This addresses your question: **"queue can also be load and resource balancer. spawn a new worker only if criteria is meet?"**

## Key Features

### 1. Dynamic Worker Spawning
- **Criteria-based spawning**: Workers are spawned only when specific conditions are met
- **Resource-aware**: Considers CPU and memory usage before spawning
- **Queue utilization monitoring**: Spawns workers when queue utilization exceeds threshold
- **Cooldown periods**: Prevents rapid worker spawning/reduction cycles

### 2. Load Balancing Criteria

The queue manager spawns new workers only when **ALL** of the following criteria are met:

```python
should_spawn = (
    queue_utilization > worker_spawn_threshold and      # Queue is busy
    current_workers < max_workers_limit and            # Haven't reached max
    cpu_percent < cpu_threshold_for_spawn and          # CPU has capacity
    memory_percent < memory_threshold_for_spawn and    # Memory has capacity
    time_since_last_spawn > worker_spawn_cooldown      # Cooldown period passed
)
```

### 3. Worker Reduction
Workers are reduced when:
```python
should_reduce = (
    queue_utilization < worker_reduce_threshold and    # Queue is quiet
    current_workers > min_workers and                  # Above minimum
    time_since_last_reduce > worker_reduce_cooldown    # Cooldown period passed
)
```

## Configuration Parameters

### Load Balancing Settings
```python
QueueManager(
    enable_load_balancing=True,           # Enable/disable load balancing
    min_workers=1,                       # Minimum workers to maintain
    max_workers_limit=8,                 # Maximum workers allowed
    worker_spawn_threshold=0.7,          # Queue utilization to spawn (70%)
    worker_reduce_threshold=0.3,         # Queue utilization to reduce (30%)
    cpu_threshold_for_spawn=80,          # CPU % threshold for spawning
    memory_threshold_for_spawn=75,       # Memory % threshold for spawning
    worker_spawn_cooldown=60,            # Seconds between spawns
    worker_reduce_cooldown=120           # Seconds between reductions
)
```

### Default Values
- **Min Workers**: 1
- **Max Workers**: 8
- **Spawn Threshold**: 70% queue utilization
- **Reduce Threshold**: 30% queue utilization
- **CPU Threshold**: 80% (spawn only if CPU < 80%)
- **Memory Threshold**: 75% (spawn only if memory < 75%)
- **Spawn Cooldown**: 60 seconds
- **Reduce Cooldown**: 120 seconds

## Load Balancing Logic

### 1. Queue Utilization Monitoring
```python
queue_utilization = queue_size / total_queue_size
```

### 2. Resource Monitoring
- **CPU Usage**: Real-time CPU percentage monitoring
- **Memory Usage**: Real-time memory percentage monitoring
- **System Resources**: Uses `psutil` for accurate resource measurement

### 3. Worker Lifecycle Management
- **Spawn Process**: Creates new worker threads dynamically
- **Reduction Process**: Gracefully reduces worker count
- **Statistics Tracking**: Monitors spawn/reduce attempts and success rates

## Usage Examples

### Basic Load Balancing
```python
from src.core.queue_manager import QueueManager

# Create queue manager with load balancing
queue_manager = QueueManager(
    enable_load_balancing=True,
    min_workers=1,
    max_workers_limit=4,
    worker_spawn_threshold=0.5,  # Spawn at 50% queue utilization
    cpu_threshold_for_spawn=70,  # Spawn only if CPU < 70%
    memory_threshold_for_spawn=70 # Spawn only if memory < 70%
)

# Start processing
queue_manager.start_processing()

# Add tasks (workers will spawn automatically if criteria met)
task_ids = queue_manager.add_tasks(file_paths)

# Monitor load balancing
stats = queue_manager.get_statistics()
lb_stats = stats.get('load_balancing', {})
print(f"Current workers: {stats['current_workers']}")
print(f"Worker spawns: {lb_stats.get('spawns', 0)}")
```

### Conservative Load Balancing
```python
# More conservative settings for resource-constrained environments
queue_manager = QueueManager(
    enable_load_balancing=True,
    min_workers=1,
    max_workers_limit=3,
    worker_spawn_threshold=0.8,  # Higher threshold
    worker_reduce_threshold=0.2,  # Lower threshold
    cpu_threshold_for_spawn=60,   # Lower CPU threshold
    memory_threshold_for_spawn=65, # Lower memory threshold
    worker_spawn_cooldown=120,    # Longer cooldown
    worker_reduce_cooldown=180    # Longer cooldown
)
```

### Aggressive Load Balancing
```python
# More aggressive settings for high-performance environments
queue_manager = QueueManager(
    enable_load_balancing=True,
    min_workers=2,
    max_workers_limit=12,
    worker_spawn_threshold=0.3,  # Lower threshold
    worker_reduce_threshold=0.1,  # Lower threshold
    cpu_threshold_for_spawn=90,   # Higher CPU threshold
    memory_threshold_for_spawn=85, # Higher memory threshold
    worker_spawn_cooldown=30,     # Shorter cooldown
    worker_reduce_cooldown=60     # Shorter cooldown
)
```

## Monitoring and Statistics

### Load Balancing Statistics
```python
lb_stats = queue_manager.get_load_balancing_stats()
print(f"Current workers: {lb_stats['current_workers']}")
print(f"Worker spawns: {lb_stats['spawns']}")
print(f"Worker reductions: {lb_stats['reductions']}")
print(f"Spawn attempts: {lb_stats['spawn_attempts']}")
print(f"Reduce attempts: {lb_stats['reduce_attempts']}")
```

### Queue Statistics with Load Balancing
```python
stats = queue_manager.get_statistics()
print(f"Queue size: {stats['queue_size']}")
print(f"Current workers: {stats['current_workers']}")
print(f"Load balancing: {stats.get('load_balancing', {})}")
```

## Integration with Analysis Manager

The queue manager is automatically integrated with the Analysis Manager:

```python
# In analysis_manager.py
if small_files:
    queue_manager = get_queue_manager()
    task_ids = queue_manager.add_tasks(small_files)
    queue_manager.start_processing(progress_callback)
    # Workers will spawn automatically based on criteria
```

## Resource Management Integration

The queue manager works with the existing Resource Manager:

```python
# Resource-aware worker spawning
if memory.percent > 85:
    # Force garbage collection
    gc.collect()
    
    # Reduce worker count if necessary
    if memory.percent > 90 and self.max_workers > 1:
        self.max_workers = max(1, self.max_workers // 2)
```

## Testing Load Balancing

Run the comprehensive test suite:

```bash
python test_load_balancing.py
```

This tests:
1. **Basic Load Balancing**: Simple worker spawning/reduction
2. **Criteria-based Load Balancing**: Strict criteria enforcement
3. **Resource-aware Spawning**: CPU/memory threshold testing
4. **Global Queue Manager**: Integration testing

## Benefits

### 1. Resource Efficiency
- **No wasted workers**: Workers only spawn when needed
- **Resource protection**: Prevents spawning under high resource usage
- **Automatic scaling**: Scales up and down based on demand

### 2. Performance Optimization
- **Reduced overhead**: Fewer idle workers
- **Better throughput**: Optimal worker count for current load
- **Responsive scaling**: Quick response to changing load

### 3. Stability
- **Cooldown periods**: Prevents rapid scaling cycles
- **Resource thresholds**: Protects against resource exhaustion
- **Graceful reduction**: Maintains minimum worker count

## Configuration Recommendations

### For Low-Resource Systems
```python
enable_load_balancing=True,
min_workers=1,
max_workers_limit=2,
worker_spawn_threshold=0.8,
cpu_threshold_for_spawn=60,
memory_threshold_for_spawn=65,
worker_spawn_cooldown=120
```

### For High-Performance Systems
```python
enable_load_balancing=True,
min_workers=2,
max_workers_limit=12,
worker_spawn_threshold=0.3,
cpu_threshold_for_spawn=85,
memory_threshold_for_spawn=80,
worker_spawn_cooldown=30
```

### For Production Systems
```python
enable_load_balancing=True,
min_workers=1,
max_workers_limit=8,
worker_spawn_threshold=0.7,
worker_reduce_threshold=0.3,
cpu_threshold_for_spawn=75,
memory_threshold_for_spawn=70,
worker_spawn_cooldown=60,
worker_reduce_cooldown=120
```

## Conclusion

The enhanced Queue Manager now provides sophisticated load balancing capabilities that:

1. **Spawns workers only when criteria are met** (queue utilization + resource availability)
2. **Acts as a load balancer** by distributing work across optimal number of workers
3. **Manages resources intelligently** by monitoring CPU and memory usage
4. **Scales dynamically** based on actual demand and system capacity
5. **Prevents resource exhaustion** through conservative thresholds and cooldowns

This addresses your question about the queue being a "load and resource balancer" that "spawns a new worker only if criteria is met" - the implementation now provides exactly this functionality with configurable criteria and comprehensive monitoring. 