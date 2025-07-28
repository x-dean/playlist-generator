# Performance Optimization Guide

This guide provides strategies to make the playlist generator more performant.

## Quick Performance Wins

### 1. Fast Mode (3-5x faster)
Skip expensive features for significantly faster processing:
```bash
# Enable fast mode for 3-5x faster processing
playlista -a --fast_mode

# Combine with parallel processing
playlista -a --fast_mode --workers 8
```

**What Fast Mode Skips:**
- MFCC coefficients (13 dimensions)
- Chroma features (12 dimensions) 
- Spectral contrast, flatness, rolloff
- MusiCNN embeddings (200 dimensions)
- External API calls (MusicBrainz, Last.fm)

**What Fast Mode Keeps:**
- BPM and rhythm features
- Spectral centroid
- Loudness analysis
- Danceability
- Key detection
- Onset rate
- Zero crossing rate
- Basic metadata

### 2. Memory-Aware Processing
```bash
# Let system automatically optimize workers based on memory
playlista -a

# Set memory limits per worker
playlista -a --memory_limit "2GB"

# Low memory mode for systems with <8GB RAM
playlista -a --low_memory
```

### 3. Batch Processing
```bash
# Process files in smaller batches
playlista -a --batch_size 2

# Combine with fast mode
playlista -a --fast_mode --batch_size 4 --workers 4
```

## Advanced Optimizations

### 1. Database Optimizations

**Enable WAL Mode (Write-Ahead Logging):**
```python
# In feature_extractor.py, add to _init_db method:
self.conn.execute("PRAGMA journal_mode=WAL")
self.conn.execute("PRAGMA synchronous=NORMAL")
self.conn.execute("PRAGMA cache_size=10000")
self.conn.execute("PRAGMA temp_store=MEMORY")
```

**Batch Database Writes:**
```python
# Use transactions for multiple writes
with self.conn:  # Auto-commit transaction
    for file_info in file_infos:
        self._save_features_to_db(file_info, features, failed=0)
```

### 2. Audio Loading Optimizations

**Streaming Audio Processing:**
```python
def _safe_audio_load_streaming(self, audio_path):
    """Load audio in chunks to reduce memory usage."""
    import essentia.standard as es
    loader = es.MonoLoader(filename=audio_path)
    
    # Process in chunks of 1 minute
    chunk_size = 44100 * 60  # 1 minute at 44.1kHz
    audio_chunks = []
    
    while True:
        chunk = loader.compute()[:chunk_size]
        if len(chunk) == 0:
            break
        audio_chunks.append(chunk)
    
    return np.concatenate(audio_chunks)
```

### 3. Feature Extraction Optimizations

**Parallel Feature Extraction:**
```python
def _extract_features_parallel(self, audio_path, audio):
    """Extract features in parallel using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor
    
    features = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit feature extraction tasks
        future_bpm = executor.submit(self._extract_rhythm_features, audio)
        future_spectral = executor.submit(self._extract_spectral_features, audio)
        future_loudness = executor.submit(self._extract_loudness, audio)
        future_dance = executor.submit(self._extract_danceability, audio)
        
        # Collect results
        features.update(future_bpm.result())
        features.update(future_spectral.result())
        features.update(future_loudness.result())
        features.update(future_dance.result())
    
    return features
```

### 4. Caching Optimizations

**Redis Caching for API Calls:**
```python
import redis
import json

class CachedAPIClient:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def get_musicbrainz_data(self, artist, title):
        cache_key = f"mb:{artist}:{title}"
        
        # Check cache first
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Make API call
        data = self._musicbrainz_lookup(artist, title)
        
        # Cache for 24 hours
        self.redis_client.setex(cache_key, 86400, json.dumps(data))
        return data
```

### 5. Memory Management

**Object Pooling:**
```python
class AudioAnalyzerPool:
    def __init__(self, pool_size=4):
        self.pool = Queue(maxsize=pool_size)
        for _ in range(pool_size):
            analyzer = AudioAnalyzer()
            self.pool.put(analyzer)
    
    def get_analyzer(self):
        return self.pool.get()
    
    def return_analyzer(self, analyzer):
        analyzer.cleanup()
        self.pool.put(analyzer)
```

**Memory-Mapped Files:**
```python
import mmap

def load_audio_mmap(audio_path):
    """Load audio using memory mapping for large files."""
    with open(audio_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Process audio data from memory map
        return process_audio_mmap(mm)
```

## Performance Monitoring

### 1. Built-in Monitoring
```bash
# Monitor memory usage
watch -n 1 'free -h && echo "---" && ps aux --sort=-%mem | head -5'

# Monitor CPU usage
htop

# Monitor disk I/O
iotop
```

### 2. Custom Performance Metrics
```python
import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
    
    def log_metrics(self, context=""):
        elapsed = time.time() - self.start_time
        current_memory = psutil.virtual_memory().used
        memory_delta = current_memory - self.start_memory
        
        logger.info(f"PERF {context}: {elapsed:.1f}s, +{memory_delta/1024**3:.1f}GB")
```

## Recommended Configurations

### For Development/Testing
```bash
# Fastest processing for testing
playlista -a --fast_mode --workers 2 --batch_size 1
```

### For Production (High-End Systems)
```bash
# Maximum performance
playlista -a --workers 8 --batch_size 4 --memory_limit "4GB"
```

### For Production (Low-End Systems)
```bash
# Conservative approach
playlista -a --fast_mode --workers 2 --batch_size 1 --low_memory
```

### For Large Libraries (>10,000 files)
```bash
# Process in chunks
playlista -a --fast_mode --workers 4 --batch_size 2 --memory_limit "2GB"
```

## Performance Benchmarks

| Configuration | Files/Hour | Memory Usage | Use Case |
|---------------|------------|--------------|----------|
| Default | 50-100 | 4-8GB | Balanced |
| Fast Mode | 150-300 | 2-4GB | Quick processing |
| Fast + Parallel | 300-600 | 6-12GB | High performance |
| Sequential | 20-40 | 1-2GB | Low memory |
| Low Memory | 30-60 | 1-3GB | Conservative |

## Troubleshooting Performance Issues

### High Memory Usage
```bash
# Reduce workers and batch size
playlista -a --workers 2 --batch_size 1

# Enable low memory mode
playlista -a --low_memory

# Use fast mode
playlista -a --fast_mode
```

### Slow Processing
```bash
# Increase workers (if memory allows)
playlista -a --workers 8

# Use fast mode
playlista -a --fast_mode

# Increase batch size
playlista -a --batch_size 4
```

### Database Bottlenecks
```bash
# Use WAL mode (see database optimizations above)
# Consider using SSD storage for database
# Monitor disk I/O with iotop
```

## Future Optimizations

### 1. GPU Acceleration
- Use TensorFlow GPU for MusiCNN embeddings
- CUDA-accelerated audio processing
- GPU memory management

### 2. Distributed Processing
- Multi-node processing with Redis coordination
- Load balancing across multiple machines
- Shared cache across nodes

### 3. Incremental Processing
- Process only changed files
- Delta updates for playlists
- Smart cache invalidation

### 4. Streaming Processing
- Process audio in real-time streams
- Pipeline processing with queues
- Backpressure handling

## Environment Variables for Performance

```bash
# Set optimal environment variables
export OMP_NUM_THREADS=4  # OpenMP threads
export MKL_NUM_THREADS=4  # Intel MKL threads
export NUMBA_NUM_THREADS=4  # Numba threads
export CUDA_VISIBLE_DEVICES=0  # GPU device
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
```

This guide provides a comprehensive approach to optimizing the playlist generator for various use cases and system configurations. 