# Analysis Manager Documentation

## Overview

The Analysis Manager is the core component responsible for coordinating audio file analysis in the Playlist Generator Simple system. It handles file discovery, selection, analysis type determination, and orchestrates the complete analysis pipeline.

## Architecture

### Core Components

```
AnalysisManager
├── FileDiscovery (file_discovery.py)
├── DatabaseManager (database.py)
├── SequentialAnalyzer (sequential_analyzer.py)
├── ParallelAnalyzer (parallel_analyzer.py)
└── AudioAnalyzer (audio_analyzer.py)
```

### Pipeline Flow

```
1. File Discovery
   ↓
2. File Selection & Filtering
   ↓
3. Analysis Type Determination
   ↓
4. File Categorization (Size-based)
   ↓
5. Feature Extraction
   ↓
6. Database Storage
   ↓
7. Results & Statistics
```

## File Location

**Primary Component:**
- `playlist-generator-simple/src/core/analysis_manager.py`

**Supporting Components:**
- `playlist-generator-simple/src/core/file_discovery.py`
- `playlist-generator-simple/src/core/audio_analyzer.py`
- `playlist-generator-simple/src/core/sequential_analyzer.py`
- `playlist-generator-simple/src/core/parallel_analyzer.py`
- `playlist-generator-simple/src/core/database.py`

## Features

### 1. File Discovery & Selection

**File Discovery:**
- Scans `/music` directory recursively
- Supports multiple audio formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.opus`
- Filters by file size (configurable min/max)
- Excludes failed files directory (`/app/cache/failed_dir`)
- Generates file hashes for change detection

**File Selection Logic:**
```python
def select_files_for_analysis(self, music_path=None, force_reextract=False, include_failed=False):
    # 1. Discover all audio files
    audio_files = self.file_discovery.discover_files()
    
    # 2. Filter based on analysis state
    for file_path in audio_files:
        should_analyze = self._should_analyze_file(file_path, force_reextract, include_failed)
        if should_analyze:
            files_to_analyze.append(file_path)
    
    # 3. Return filtered list
    return files_to_analyze
```

### 2. Analysis Type Determination

**Smart Analysis (Default):**
- Monitors system resources (CPU, memory)
- Makes dynamic decisions based on current system state
- Considers file size, available memory, and CPU usage

**Deterministic Analysis:**
- Based solely on file size thresholds
- Simpler but less resource-aware

**Analysis Types:**

| Type | Features | Use Case |
|------|----------|----------|
| **Full Analysis** | All features + MusiCNN | Small files with good resources |
| **Basic Analysis** | Core features only | Large files or resource-constrained |

**Feature Matrix:**

| Feature | Full Analysis | Basic Analysis |
|---------|---------------|----------------|
| Rhythm Analysis | ✅ | ✅ |
| Spectral Features | ✅ | ✅ |
| Loudness Analysis | ✅ | ✅ |
| Key Detection | ✅ | ✅ |
| MFCC Features | ✅ | ✅ |
| MusiCNN Features | ✅ | ❌ |
| Chroma Features | ✅ | ✅ |
| Metadata Extraction | ✅ | ✅ |

### 3. File Categorization

**Size-based Processing:**
- **Large files** (≥50MB): Sequential processing
- **Small files** (<50MB): Parallel processing

**Resource Thresholds:**
```python
# Large file threshold
BIG_FILE_SIZE_MB = 50

# Memory thresholds
MIN_MEMORY_FOR_FULL_ANALYSIS_GB = 4.0
MEMORY_BUFFER_GB = 1.0

# CPU thresholds
MAX_CPU_FOR_FULL_ANALYSIS_PERCENT = 80
```

### 4. Feature Extraction

**Audio Features Extracted:**

**Rhythm Features:**
- BPM (Beats Per Minute)
- Beat positions
- Rhythm strength
- Tempo confidence

**Spectral Features:**
- Spectral centroid
- Spectral rolloff
- Spectral bandwidth
- Spectral contrast

**Loudness Features:**
- RMS energy
- Dynamic range
- Loudness war detection
- Peak amplitude

**Key Features:**
- Musical key
- Mode (major/minor)
- Key strength
- Key confidence

**MFCC Features:**
- Mel-frequency cepstral coefficients
- Audio fingerprinting
- Similarity matching

**MusiCNN Features:**
- Deep learning audio features
- Genre classification
- Mood detection
- Style analysis

### 5. Caching & Performance

**Analysis Caching:**
- File hash-based cache invalidation
- Configurable cache expiry (default: 24 hours)
- Automatic cache cleanup

**Performance Optimizations:**
- Parallel processing for small files
- Sequential processing for large files
- Memory-aware processing
- Timeout protection
- Resource monitoring

## Configuration

### Configuration Sources

1. **Default Values** (hardcoded in `config_loader.py`)
2. **Configuration File** (`playlista.conf`)
3. **Environment Variables** (highest priority)
4. **CLI Arguments** (override all others)

### Key Configuration Parameters

**Analysis Settings:**
```ini
# File size thresholds
BIG_FILE_SIZE_MB=50
MAX_FULL_ANALYSIS_SIZE_MB=100
MIN_FULL_ANALYSIS_SIZE_MB=1

# Resource thresholds
MIN_MEMORY_FOR_FULL_ANALYSIS_GB=4.0
MEMORY_BUFFER_GB=1.0
MAX_CPU_FOR_FULL_ANALYSIS_PERCENT=80

# Timeout settings
ANALYSIS_TIMEOUT_SECONDS=300
SEQUENTIAL_TIMEOUT_SECONDS=300
PARALLEL_TIMEOUT_SECONDS=300

# Feature extraction
EXTRACT_RHYTHM=true
EXTRACT_SPECTRAL=true
EXTRACT_LOUDNESS=true
EXTRACT_KEY=true
EXTRACT_MFCC=true
EXTRACT_MUSICNN=true
EXTRACT_METADATA=true
```

**Processing Settings:**
```ini
# Worker configuration
MAX_WORKERS=4
WORKER_TIMEOUT_SECONDS=300

# Caching
ANALYSIS_CACHE_ENABLED=true
ANALYSIS_CACHE_EXPIRY_HOURS=24

# Retry settings
ANALYSIS_RETRY_ATTEMPTS=3
ANALYSIS_RETRY_DELAY_SECONDS=5
FAILED_FILES_MAX_RETRIES=3
```

## Logging System

### Logging Features

- **Universal logging** via `log_universal()` function
- **Function call logging** via `@log_function_call` decorator
- **Colored console output** for development
- **File logging** for production
- **Performance metrics** and timing
- **Decision tracking** for analysis types

### Log Levels

- **ERROR**: System errors and failures
- **WARNING**: Resource constraints and fallbacks
- **INFO**: Analysis decisions and progress
- **DEBUG**: Detailed processing information

### Key Log Messages

**File Selection:**
```
INFO - Analysis - Discovered 150 audio files
INFO - Analysis - Selected 45 files for analysis
INFO - Analysis - Skipped 105 files (already analyzed)
INFO - Analysis - Previously failed: 3 files
```

**Analysis Type Determination:**
```
INFO - Analysis - Determined analysis type for /music/song.mp3: full
INFO - Analysis - Smart analysis: File 25.3MB, Memory 8.2GB, CPU 45.2%
INFO - Analysis - Reason: Smart analysis: File 25.3MB, Memory 8.2GB, CPU 45.2%
```

**Processing Progress:**
```
INFO - Analysis - Processing 30 large files sequentially
INFO - Analysis - Processing 15 small files in parallel
INFO - Analysis - File analysis completed in 245.67s
INFO - Analysis - Results: 42 successful, 3 failed
```

## Usage Examples

### Basic Analysis

```bash
# Analyze all files
playlista analyze

# Force re-analysis (bypass cache)
playlista analyze --force

# Include previously failed files
playlista analyze --include-failed

# Specify music directory
playlista analyze --music-path /custom/music/path
```

### Advanced Analysis

```bash
# Analyze with custom workers
playlista analyze --workers 8

# Analyze with verbose logging
playlista analyze -v

# Analyze with debug logging
playlista analyze -vv

# Analyze specific file types
playlista analyze --extensions mp3,flac
```

### Monitoring & Statistics

```bash
# View analysis statistics
playlista stats

# View detailed statistics
playlista stats --detailed

# View failed files
playlista stats --failed-files

# View memory usage
playlista stats --memory-usage
```

### Retry & Cleanup

```bash
# Retry failed files
playlista retry-failed

# Clean up failed analysis entries
playlista cleanup

# Monitor system resources
playlista monitor
```

## API Reference

### AnalysisManager Class

**Constructor:**
```python
def __init__(self, db_manager: DatabaseManager = None, config: Dict[str, Any] = None)
```

**Key Methods:**

**File Selection:**
```python
def select_files_for_analysis(self, music_path: str = None, force_reextract: bool = False, include_failed: bool = False) -> List[str]
```

**Analysis Type Determination:**
```python
def determine_analysis_type(self, file_path: str) -> Dict[str, Any]
```

**File Analysis:**
```python
def analyze_files(self, files: List[str], force_reextract: bool = False, max_workers: int = None) -> Dict[str, Any]
```

**Statistics:**
```python
def get_analysis_statistics(self) -> Dict[str, Any]
```

**Failed File Management:**
```python
def retry_current_failed_files(self) -> Dict[str, Any]
def final_retry_failed_files(self) -> Dict[str, Any]
def cleanup_failed_analysis(self, max_retries: int = 3) -> int
```

### Return Values

**Analysis Results:**
```python
{
    'success_count': 42,
    'failed_count': 3,
    'total_time': 245.67,
    'big_files_processed': 30,
    'small_files_processed': 15
}
```

**Analysis Type Configuration:**
```python
{
    'analysis_type': 'full',
    'use_full_analysis': True,
    'file_size_mb': 25.3,
    'features_config': {
        'extract_rhythm': True,
        'extract_spectral': True,
        'extract_loudness': True,
        'extract_key': True,
        'extract_mfcc': True,
        'extract_musicnn': True,
        'extract_metadata': True
    },
    'reason': 'Smart analysis: File 25.3MB, Memory 8.2GB, CPU 45.2%'
}
```

## Error Handling

### Common Error Scenarios

1. **File Not Found:**
   - Logs: `ERROR - Analysis - Music directory not found: /music`
   - Action: Check music directory path

2. **Permission Denied:**
   - Logs: `ERROR - Analysis - Permission denied accessing music directory`
   - Action: Check file permissions

3. **Resource Constraints:**
   - Logs: `WARNING - Analysis - Insufficient memory: 2.1GB < 4.0GB`
   - Action: Falls back to basic analysis

4. **Analysis Timeout:**
   - Logs: `ERROR - Analysis - Analysis timeout for file: song.mp3`
   - Action: Moves to failed files, retries later

### Error Recovery

**Automatic Recovery:**
- Failed files are retried automatically
- Resource constraints trigger fallback analysis
- Timeout protection prevents hanging

**Manual Recovery:**
```bash
# Retry failed files
playlista retry-failed

# Force re-analysis
playlista analyze --force

# Clean up failed entries
playlista cleanup
```

## Performance Considerations

### Memory Management

- **Large files** (>50MB) use sequential processing
- **Memory monitoring** prevents system overload
- **Resource thresholds** trigger fallback analysis
- **Automatic cleanup** of temporary data

### CPU Optimization

- **Parallel processing** for small files
- **CPU monitoring** during analysis
- **Worker limits** prevent system overload
- **Timeout protection** prevents hanging

### Disk I/O

- **Caching** reduces repeated analysis
- **Batch processing** optimizes database operations
- **File hash tracking** detects changes
- **Incremental updates** minimize I/O

## Best Practices

### Configuration

1. **Set appropriate thresholds** based on system resources
2. **Enable caching** for production environments
3. **Configure logging** for your use case
4. **Monitor resource usage** during analysis

### File Organization

1. **Use consistent file naming** for better tracking
2. **Organize files** in subdirectories
3. **Avoid extremely large files** (>500MB)
4. **Use supported formats** for best results

### Monitoring

1. **Check logs regularly** for issues
2. **Monitor resource usage** during analysis
3. **Review failed files** periodically
4. **Clean up old cache** entries

## Troubleshooting

### Common Issues

**No files found for analysis:**
- Check music directory path
- Verify file extensions are supported
- Check file permissions

**Analysis taking too long:**
- Reduce worker count
- Increase timeout settings
- Check system resources

**Memory errors:**
- Reduce file size thresholds
- Enable basic analysis only
- Increase memory buffer

**Failed analysis:**
- Check file integrity
- Verify audio format support
- Review error logs

### Debug Commands

```bash
# Test audio file analysis
playlista test-audio /path/to/file.mp3

# Monitor system resources
playlista monitor

# View detailed logs
playlista analyze -vv

# Check configuration
playlista config
```

## Integration

### CLI Integration

The Analysis Manager integrates with the main CLI through:
- `playlist-generator-simple/src/enhanced_cli.py`
- Command: `playlista analyze`
- Arguments: `--force`, `--include-failed`, `--workers`

### API Integration

The system provides REST API endpoints through:
- `playlist-generator-simple/src/main.py`
- FastAPI application
- OpenAPI documentation at `/docs`

### Database Integration

Analysis results are stored in:
- SQLite database: `/app/cache/playlista.db`
- Tables: `analysis_results`, `failed_analysis`, `metadata`
- Automatic cleanup and maintenance

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - Genre classification
   - Mood detection
   - Similarity matching

2. **Advanced Caching**
   - Distributed caching
   - Cache invalidation strategies
   - Performance optimization

3. **Real-time Processing**
   - Streaming analysis
   - Live monitoring
   - WebSocket updates

4. **Cloud Integration**
   - Cloud storage support
   - Distributed processing
   - Auto-scaling

### Extension Points

The Analysis Manager is designed for extensibility:
- **Custom analyzers** can be added
- **New features** can be integrated
- **External APIs** can be connected
- **Custom configurations** are supported

---

*This documentation covers the complete Analysis Manager system. For specific implementation details, refer to the source code in `playlist-generator-simple/src/core/analysis_manager.py`.* 