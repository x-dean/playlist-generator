# 🎵 Enhanced CLI Guide - Playlist Generator Simple

## Overview

The Enhanced CLI provides a comprehensive interface that integrates all features from both the simple and refactored versions of the playlist generator. This guide covers all available commands, options, and usage examples.

---

## 📋 Available Commands

### **Core Analysis Commands**

| Command | Description | Key Options |
|---------|-------------|-------------|
| `analyze` | Analyze audio files for features | `--fast-mode`, `--parallel`, `--memory-aware` |
| `stats` | Show analysis and resource statistics | `--detailed`, `--failed-files`, `--memory-usage` |
| `test-audio` | Test audio analyzer with a single file | `--file`, `--force` |
| `monitor` | Monitor system resources | `--duration` |
| `cleanup` | Clean up failed analysis | `--max-retries` |

### **Playlist Generation Commands**

| Command | Description | Key Options |
|---------|-------------|-------------|
| `playlist` | Generate playlists using various methods | `--method`, `--num-playlists`, `--playlist-size` |
| `playlist-methods` | List available playlist generation methods | None |

### **Enhanced Commands**

| Command | Description | Key Options |
|---------|-------------|-------------|
| `discover` | Discover audio files in a directory | `--recursive`, `--extensions`, `--exclude-dirs` |
| `enrich` | Enrich metadata from external APIs | `--musicbrainz`, `--lastfm`, `--force` |
| `export` | Export playlists in different formats | `--format`, `--output-dir` |
| `status` | Show database and system status | `--detailed`, `--failed-files` |
| `pipeline` | Run full analysis and generation pipeline | `--force`, `--failed`, `--generate` |
| `config` | Show configuration information | `--json`, `--validate`, `--reload` |

---

## 🎼 Playlist Generation Methods

### **Available Methods**

| Method | Description | Best For |
|--------|-------------|----------|
| `kmeans` | K-means clustering based on audio features | General clustering |
| `similarity` | Similarity-based selection using cosine similarity | Finding similar tracks |
| `random` | Random selection for variety | Quick variety playlists |
| `time_based` | Time-based scheduling for different times of day | Time-appropriate music |
| `tag_based` | Tag-based selection using metadata tags | Genre-based playlists |
| `cache_based` | Cache-based selection using previously generated playlists | Consistent playlists |
| `feature_group` | Feature group selection based on audio characteristics | Audio feature grouping |
| `mixed` | Mixed approach combining multiple methods | Balanced playlists |
| `all` | All methods combined for comprehensive coverage | Complete analysis |
| `ensemble` | Ensemble methods combining multiple algorithms | Advanced clustering |
| `hierarchical` | Hierarchical clustering for nested groupings | Nested playlists |
| `recommendation` | Recommendation-based using collaborative filtering | Personalized playlists |
| `mood_based` | Mood-based generation using emotional features | Mood-based playlists |

---

## ⚡ Processing Modes

### **Memory Management**

```bash
# Memory-aware processing
python enhanced_cli.py analyze --music-path /music --memory-aware --memory-limit 2GB

# Low memory mode
python enhanced_cli.py analyze --music-path /music --low-memory --rss-limit-gb 4.0

# Custom memory limits
python enhanced_cli.py analyze --music-path /music --memory-limit 512MB --rss-limit-gb 6.0
```

### **Processing Options**

```bash
# Fast mode (3-5x faster)
python enhanced_cli.py analyze --music-path /music --fast-mode

# Parallel processing
python enhanced_cli.py analyze --music-path /music --parallel --workers 4

# Sequential processing (for debugging)
python enhanced_cli.py analyze --music-path /music --sequential

# Force re-analysis
python enhanced_cli.py analyze --music-path /music --force

# Bypass cache
python enhanced_cli.py analyze --music-path /music --no-cache

# Re-analyze only failed files
python enhanced_cli.py analyze --music-path /music --failed
```

---

## 📊 Export Formats

### **Available Formats**

| Format | Description | Extension |
|--------|-------------|-----------|
| `m3u` | M3U playlist format | `.m3u` |
| `pls` | PLS playlist format | `.pls` |
| `xspf` | XSPF playlist format | `.xspf` |
| `json` | JSON format | `.json` |
| `all` | All formats | Multiple files |

---

## 🚀 Usage Examples

### **Basic Analysis**

```bash
# Simple analysis
python enhanced_cli.py analyze --music-path /path/to/music

# Fast analysis with parallel processing
python enhanced_cli.py analyze --music-path /path/to/music --fast-mode --parallel --workers 4

# Memory-aware analysis
python enhanced_cli.py analyze --music-path /path/to/music --memory-aware --memory-limit 2GB
```

### **Playlist Generation**

```bash
# Generate playlists using K-means
python enhanced_cli.py playlist --method kmeans --num-playlists 5 --playlist-size 20

# Generate time-based playlists
python enhanced_cli.py playlist --method time_based --num-playlists 4 --playlist-size 15

# Generate and save playlists
python enhanced_cli.py playlist --method similarity --num-playlists 3 --playlist-size 25 --save --output-dir playlists
```

### **Full Pipeline**

```bash
# Complete pipeline with analysis and playlist generation
python enhanced_cli.py pipeline --music-path /path/to/music --force --generate

# Pipeline with export
python enhanced_cli.py pipeline --music-path /path/to/music --force --generate --export
```

### **File Discovery**

```bash
# Discover audio files
python enhanced_cli.py discover --path /path/to/music --recursive

# Discover with specific extensions
python enhanced_cli.py discover --path /path/to/music --extensions mp3,flac,wav --recursive

# Discover with size limits
python enhanced_cli.py discover --path /path/to/music --min-size 1000000 --max-size 100000000
```

### **Metadata Enrichment**

```bash
# Enrich metadata using MusicBrainz
python enhanced_cli.py enrich --path /path/to/music --musicbrainz

# Enrich specific files
python enhanced_cli.py enrich --audio-ids 1,2,3,4,5 --musicbrainz --lastfm

# Force re-enrichment
python enhanced_cli.py enrich --path /path/to/music --force --musicbrainz
```

### **Export Playlists**

```bash
# Export in M3U format
python enhanced_cli.py export --playlist-file playlists.json --format m3u

# Export in all formats
python enhanced_cli.py export --playlist-file playlists.json --format all --output-dir exports
```

### **System Monitoring**

```bash
# Monitor resources for 60 seconds
python enhanced_cli.py monitor --duration 60

# Monitor indefinitely (press Ctrl+C to stop)
python enhanced_cli.py monitor

# Show detailed statistics
python enhanced_cli.py stats --detailed --memory-usage

# Show system status
python enhanced_cli.py status --detailed --failed-files
```

### **Configuration Management**

```bash
# Show configuration
python enhanced_cli.py config

# Show configuration as JSON
python enhanced_cli.py config --json

# Validate configuration
python enhanced_cli.py config --validate

# Reload configuration
python enhanced_cli.py config --reload
```

---

## 🔧 Advanced Features

### **Resource Management**

The enhanced CLI includes sophisticated resource management:

- **Memory-aware processing**: Automatically adjusts processing based on available memory
- **Fast mode**: 3-5x faster processing for large libraries
- **Parallel processing**: Multi-core processing for faster analysis
- **Resource monitoring**: Real-time monitoring of system resources
- **Automatic cleanup**: Cleanup of failed files and cache entries

### **Error Handling**

- **Failed file tracking**: Tracks and reports failed analysis attempts
- **Retry mechanisms**: Automatic retry of failed operations
- **Graceful degradation**: Continues processing even when some files fail
- **Detailed logging**: Comprehensive logging for debugging

### **Performance Optimization**

- **Caching**: Intelligent caching of analysis results
- **Batch processing**: Efficient batch processing of files
- **Memory optimization**: Memory-efficient processing for large libraries
- **Worker optimization**: Automatic worker count optimization

---

## 📈 Performance Tips

### **For Large Libraries**

```bash
# Use fast mode with memory awareness
python enhanced_cli.py analyze --music-path /large/music/library --fast-mode --memory-aware --memory-limit 4GB

# Use parallel processing with optimal workers
python enhanced_cli.py analyze --music-path /large/music/library --parallel --workers 8

# Process in batches
python enhanced_cli.py analyze --music-path /large/music/library --batch-size 100
```

### **For Limited Resources**

```bash
# Use low memory mode
python enhanced_cli.py analyze --music-path /music --low-memory --rss-limit-gb 2.0

# Use sequential processing
python enhanced_cli.py analyze --music-path /music --sequential

# Limit memory usage
python enhanced_cli.py analyze --music-path /music --memory-limit 1GB
```

### **For Debugging**

```bash
# Use sequential processing for debugging
python enhanced_cli.py analyze --music-path /music --sequential

# Test with a single file
python enhanced_cli.py test-audio --file /path/to/test.mp3

# Monitor resources during processing
python enhanced_cli.py monitor --duration 300
```

---

## 🎯 Best Practices

### **Workflow Recommendations**

1. **Start with discovery**: Use `discover` to understand your music library
2. **Analyze with fast mode**: Use `--fast-mode` for initial analysis
3. **Generate diverse playlists**: Try different methods with `playlist-methods`
4. **Monitor resources**: Use `monitor` during long operations
5. **Export in multiple formats**: Use `export` with `--format all`

### **Configuration Tips**

- Set appropriate memory limits based on your system
- Use parallel processing for multi-core systems
- Enable memory-aware processing for large libraries
- Use fast mode for initial analysis, full mode for detailed analysis

### **Troubleshooting**

- Use `stats --detailed` to diagnose issues
- Use `cleanup` to remove failed files
- Use `status --failed-files` to see problematic files
- Use `config --validate` to check configuration

---

## 📚 Command Reference

### **Global Options**

All commands support these global options:

```bash
--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL  # Set logging level
--verbose, -v                                  # Verbose output
--quiet, -q                                    # Quiet output
```

### **Help and Documentation**

```bash
# Show main help
python enhanced_cli.py

# Show command-specific help
python enhanced_cli.py analyze --help
python enhanced_cli.py playlist --help
python enhanced_cli.py pipeline --help
```

---

## 🔄 Migration from Old CLI

If you're migrating from the old CLI:

### **Old Command → New Command**

```bash
# Old: python analysis_cli.py analyze --music-path /music
# New: python enhanced_cli.py analyze --music-path /music

# Old: python analysis_cli.py playlist --method kmeans
# New: python enhanced_cli.py playlist --method kmeans

# Old: python analysis_cli.py stats
# New: python enhanced_cli.py stats
```

### **New Features**

- **Pipeline command**: Complete analysis and generation in one command
- **Export functionality**: Export playlists in multiple formats
- **Enhanced monitoring**: Better resource monitoring and management
- **Configuration management**: Validate and manage configuration
- **Metadata enrichment**: Enrich metadata from external APIs

---

This enhanced CLI provides a comprehensive interface for all playlist generator functionality, combining the best features from both the simple and refactored versions while adding new capabilities for better performance and usability. 