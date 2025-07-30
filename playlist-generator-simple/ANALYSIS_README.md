# Analysis Components for Playlist Generator Simple

This document describes the analysis components that have been ported to the simple playlist generator. The analysis system provides comprehensive audio file analysis with resource management and database integration.

## Overview

The analysis system consists of several key components:

1. **Analysis Manager** - Coordinates analysis operations and selects appropriate analyzers
2. **Resource Manager** - Monitors system resources in real-time
3. **Sequential Analyzer** - Processes large files sequentially
4. **Parallel Analyzer** - Processes smaller files in parallel
5. **Audio Analyzer** - Extracts audio features from files
6. **Database Integration** - Stores and retrieves analysis results

## Components

### Analysis Manager (`analysis_manager.py`)

The central coordinator for analysis operations.

**Key Features:**
- File selection for analysis
- Analyzer type selection (sequential vs parallel)
- Database integration for results
- Progress tracking and statistics

**Usage:**
```python
from core.analysis_manager import AnalysisManager

# Create analysis manager
analysis_manager = AnalysisManager()

# Select files for analysis
files = analysis_manager.select_files_for_analysis(
    music_path='/music',
    force_reextract=False,
    include_failed=False
)

# Analyze files
results = analysis_manager.analyze_files(files, force_reextract=False)
```

### Resource Manager (`resource_manager.py`)

Monitors system resources and manages analysis operations.

**Key Features:**
- Real-time memory, CPU, and disk monitoring
- Automatic resource cleanup
- Optimal worker count calculation
- Resource history and statistics

**Usage:**
```python
from core.resource_manager import ResourceManager

# Create resource manager
resource_manager = ResourceManager()

# Start monitoring
resource_manager.start_monitoring()

# Get current resources
resources = resource_manager.get_current_resources()

# Get optimal worker count
worker_count = resource_manager.get_optimal_worker_count()
```

### Sequential Analyzer (`sequential_analyzer.py`)

Processes large files (>50MB) sequentially to manage memory usage.

**Key Features:**
- Large file processing with timeout
- Memory monitoring during analysis
- Process isolation for stability
- Error handling and recovery

**Usage:**
```python
from core.sequential_analyzer import SequentialAnalyzer

# Create sequential analyzer
sequential_analyzer = SequentialAnalyzer()

# Process large files
results = sequential_analyzer.process_files(large_files, force_reextract=False)
```

### Parallel Analyzer (`parallel_analyzer.py`)

Processes smaller files (<50MB) in parallel for efficiency.

**Key Features:**
- Parallel file processing with multiple workers
- Memory-aware worker count calculation
- Process pool management
- Error handling and recovery

**Usage:**
```python
from core.parallel_analyzer import ParallelAnalyzer

# Create parallel analyzer
parallel_analyzer = ParallelAnalyzer()

# Process small files in parallel
results = parallel_analyzer.process_files(small_files, force_reextract=False, max_workers=4)
```

### Audio Analyzer (`audio_analyzer.py`)

Extracts audio features from files.

**Key Features:**
- Audio feature extraction (rhythm, spectral, loudness, etc.)
- Metadata extraction
- Error handling and timeout management
- Feature validation

**Supported Libraries:**
- Essentia (preferred)
- Librosa (fallback)
- Mutagen (metadata)

**Usage:**
```python
from core.audio_analyzer import AudioAnalyzer

# Create audio analyzer
audio_analyzer = AudioAnalyzer()

# Extract features
result = audio_analyzer.extract_features('/path/to/audio.mp3', force_reextract=False)
```

## Configuration

The analysis components use configuration from the main config loader. Key settings include:

### Analysis Configuration
```ini
# File size threshold for sequential vs parallel processing
BIG_FILE_SIZE_MB=50

# Analysis timeouts
ANALYSIS_TIMEOUT_SECONDS=300
SEQUENTIAL_TIMEOUT_SECONDS=600
PARALLEL_TIMEOUT_SECONDS=300

# Memory thresholds
MEMORY_THRESHOLD_PERCENT=85

# Audio processing settings
AUDIO_SAMPLE_RATE=44100
AUDIO_HOP_SIZE=512
AUDIO_FRAME_SIZE=2048
```

### Resource Configuration
```ini
# Memory limits
MEMORY_LIMIT_GB=6.0
MEMORY_PER_WORKER_GB=0.5

# Resource thresholds
CPU_THRESHOLD_PERCENT=90
DISK_THRESHOLD_PERCENT=85

# Monitoring settings
MONITORING_INTERVAL_SECONDS=5
```

## CLI Interface

A simple CLI interface is provided for testing and using the analysis components.

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run integration tests
python test_analysis_integration.py
```

### Usage

```bash
# Analyze files
python src/analysis_cli.py analyze --music-path /path/to/music

# Show statistics
python src/analysis_cli.py stats

# Test audio analyzer
python src/analysis_cli.py test-audio --file /path/to/audio.mp3

# Monitor resources
python src/analysis_cli.py monitor --duration 60

# Clean up failed analysis
python src/analysis_cli.py cleanup --max-retries 3
```

## Integration Testing

Run the integration tests to verify all components work together:

```bash
python test_analysis_integration.py
```

The tests verify:
- Analysis manager functionality
- Resource manager functionality
- Audio analyzer functionality
- Sequential and parallel analyzers
- Database integration
- Full component integration

## Database Schema

The analysis results are stored in the database with the following schema:

### Analysis Results Table
```sql
CREATE TABLE analysis_results (
    file_path TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_hash TEXT NOT NULL,
    analysis_data TEXT NOT NULL,
    metadata TEXT,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Failed Analysis Table
```sql
CREATE TABLE failed_analysis (
    file_path TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    error_message TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0,
    failed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_retry TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Performance Considerations

### Memory Management
- Large files (>50MB) are processed sequentially to avoid memory issues
- Memory monitoring triggers cleanup when usage exceeds thresholds
- Process isolation prevents memory leaks

### Resource Optimization
- Worker count is automatically calculated based on available memory
- CPU and disk usage are monitored to prevent system overload
- Timeout mechanisms prevent hanging processes

### Caching
- Analysis results are cached in the database
- File hashes detect changes and trigger re-analysis
- Failed analysis entries are tracked for retry

## Error Handling

The analysis system includes comprehensive error handling:

- **Timeout Protection**: All analysis operations have configurable timeouts
- **Memory Monitoring**: Automatic cleanup when memory usage is high
- **Process Isolation**: Failed processes don't affect others
- **Retry Logic**: Failed files can be retried with exponential backoff
- **Error Logging**: Detailed error messages for debugging

## Dependencies

### Required Libraries
- `psutil` - System resource monitoring
- `numpy` - Numerical computations
- `sqlite3` - Database operations (built-in)

### Optional Libraries
- `essentia` - Audio feature extraction (recommended)
- `librosa` - Audio feature extraction (fallback)
- `mutagen` - Audio metadata extraction
- `scipy` - Audio file creation for testing

## Troubleshooting

### Common Issues

1. **No audio libraries available**
   - Install essentia: `pip install essentia`
   - Or install librosa: `pip install librosa`

2. **Memory issues with large files**
   - Increase `MEMORY_LIMIT_GB` in configuration
   - Reduce `BIG_FILE_SIZE_MB` to process more files sequentially

3. **Timeout errors**
   - Increase timeout settings in configuration
   - Check system resources and reduce worker count

4. **Database errors**
   - Ensure database directory is writable
   - Check disk space availability

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
logging.getLogger('playlista').setLevel(logging.DEBUG)
```

## Future Enhancements

Planned improvements include:

- **GPU Acceleration**: Support for GPU-based audio processing
- **Distributed Processing**: Multi-machine analysis capabilities
- **Advanced Features**: More sophisticated audio feature extraction
- **Real-time Analysis**: Streaming audio analysis capabilities
- **Plugin System**: Extensible analysis pipeline

## Contributing

When contributing to the analysis components:

1. Follow the existing code style and patterns
2. Add comprehensive error handling
3. Include performance monitoring
4. Write integration tests for new features
5. Update documentation for any changes

## License

This analysis system is part of the Playlist Generator Simple project and follows the same license terms. 