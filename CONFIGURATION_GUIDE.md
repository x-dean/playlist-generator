# Playlista Configuration Guide

This guide explains how to configure the Playlista music analysis and playlist generation system using the comprehensive JSON configuration file.

## Overview

The Playlista system supports multiple configuration methods with the following priority order:

1. **CLI arguments** (highest priority)
2. **Environment variables**
3. **JSON configuration file**
4. **Hardcoded defaults** (lowest priority)

## Quick Start

### 1. Copy the Configuration File

Copy the `playlistarrays_config.json` file to your host system:

```bash
cp playlistarrays_config.json /path/to/your/config/
```

### 2. Modify Settings

Edit the configuration file to customize settings for your environment:

```json
{
  "playlista_config": {
    "memory_management": {
      "settings": {
        "MEMORY_LIMIT_GB": {
          "value": 8.0,  // Change this to your system's memory
          "description": "Memory limit per worker in GB"
        }
      }
    }
  }
}
```

### 3. Mount in Docker

Mount the configuration file when running the container:

```bash
docker run -v /path/to/playlistarrays_config.json:/app/config/playlista_config.json playlista
```

### 4. Verify Configuration

Check that your configuration is loaded correctly:

```bash
docker run -v /path/to/playlistarrays_config.json:/app/config/playlista_config.json playlista config --json
```

## Configuration Categories

### Memory Management

Control how the system uses memory:

```json
"memory_management": {
  "settings": {
    "MEMORY_LIMIT_GB": {
      "value": 6.0,
      "description": "Memory limit per worker in GB"
    },
    "MEMORY_AWARE": {
      "value": true,
      "description": "Enable memory-aware processing"
    },
    "MEMORY_PRESSURE_THRESHOLD": {
      "value": 0.8,
      "description": "Memory pressure threshold (80% default)"
    }
  }
}
```

### File Processing

Configure how files are processed:

```json
"file_processing": {
  "settings": {
    "LARGE_FILE_THRESHOLD_MB": {
      "value": 50,
      "description": "File size threshold for large file processing"
    },
    "FILE_TIMEOUT_MINUTES": {
      "value": 10,
      "description": "Timeout for individual file processing"
    },
    "MAX_RETRIES": {
      "value": 3,
      "description": "Maximum number of retries for failed files"
    }
  }
}
```

### Audio Processing

Configure audio analysis limits:

```json
"audio_processing": {
  "settings": {
    "MAX_AUDIO_SAMPLES": {
      "value": 150000000,
      "description": "Maximum audio samples to process (~5.7 hours at 44kHz)"
    },
    "BPM_MIN": {
      "value": 60,
      "description": "Minimum BPM for detection"
    },
    "BPM_MAX": {
      "value": 200,
      "description": "Maximum BPM for detection"
    }
  }
}
```

### Feature Extraction

Control which audio features are extracted:

```json
"feature_extraction": {
  "settings": {
    "EXTRACT_BPM": {
      "value": true,
      "description": "Extract BPM (tempo) from audio files"
    },
    "EXTRACT_MFCC": {
      "value": true,
      "description": "Extract MFCC features from audio files"
    },
    "EXTRACT_MUSICNN": {
      "value": true,
      "description": "Extract MusiCNN features (requires model)"
    }
  }
}
```

### Playlist Generation

Configure playlist generation parameters:

```json
"playlist_generation": {
  "settings": {
    "MIN_TRACKS_PER_PLAYLIST": {
      "value": 10,
      "description": "Minimum tracks per playlist"
    },
    "MAX_TRACKS_PER_PLAYLIST": {
      "value": 500,
      "description": "Maximum tracks per playlist"
    },
    "KMEANS_N_CLUSTERS": {
      "value": 8,
      "description": "Number of clusters for K-means playlist generation"
    }
  }
}
```

### External APIs

Configure external API integrations:

```json
"external_apis": {
  "settings": {
    "LASTFM_API_KEY": {
      "value": "your_api_key_here",
      "description": "Last.fm API key for metadata enrichment"
    },
    "MUSICBRAINZ_RATE_LIMIT": {
      "value": 1.0,
      "description": "MusicBrainz API rate limit (requests per second)"
    }
  }
}
```

### Paths

Configure file system paths:

```json
"paths": {
  "settings": {
    "HOST_LIBRARY_PATH": {
      "value": "/path/to/your/music",
      "description": "Host library path for music files"
    },
    "OUTPUT_DIR": {
      "value": "/app/playlists",
      "description": "Output directory for generated playlists"
    },
    "CACHE_DIR": {
      "value": "/app/cache",
      "description": "Cache directory for analysis results"
    }
  }
}
```

## Docker Compose Examples

### Basic Configuration

```yaml
version: '3.8'
services:
  playlista:
    build: .
    volumes:
      - ./playlistarrays_config.json:/app/config/playlista_config.json:ro
      - /path/to/your/music:/music:ro
      - ./playlists:/app/playlists
      - ./cache:/app/cache
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
    command: ["playlista", "config", "--json"]
```

### Development Configuration

```yaml
version: '3.8'
services:
  playlista-dev:
    build: .
    volumes:
      - ./playlistarrays_config.json:/app/config/playlista_config.json:ro
      - /path/to/your/music:/music:ro
      - ./playlists:/app/playlists
      - ./cache:/app/cache
      - ./logs:/app/logs
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
      - MEMORY_LIMIT_GB=4.0
      - FAST_MODE_ENABLED=true
    command: ["playlista", "config", "--validate"]
```

### High-Performance Configuration

```yaml
version: '3.8'
services:
  playlista-high-perf:
    build: .
    volumes:
      - ./playlistarrays_config.json:/app/config/playlista_config.json:ro
      - /path/to/your/music:/music:ro
      - ./playlists:/app/playlists
      - ./cache:/app/cache
      - ./logs:/app/logs
    environment:
      - MEMORY_LIMIT_GB=16.0
      - MEMORY_AWARE=true
      - LARGE_FILE_THRESHOLD=200
      - FAST_MODE_ENABLED=true
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8.0'
    command: ["playlista", "pipeline", "/music", "--force", "--generate"]
```

## CLI Configuration Commands

### View Current Configuration

```bash
# Show all configuration
playlista config

# Show JSON configuration
playlista config --json

# Validate configuration
playlista config --validate

# Reload configuration
playlista config --reload
```

### Override Configuration

```bash
# Override memory limit
MEMORY_LIMIT_GB=8.0 playlista analyze /music

# Override multiple settings
MEMORY_LIMIT_GB=8.0 LARGE_FILE_THRESHOLD=100 playlista analyze /music

# Use CLI arguments (highest priority)
playlista analyze /music --memory-limit 4GB --large-file-threshold 50
```

## Environment Variables

You can override any setting using environment variables:

```bash
# Memory settings
export MEMORY_LIMIT_GB=8.0
export MEMORY_AWARE=true
export MEMORY_PRESSURE_THRESHOLD=0.8

# Processing settings
export LARGE_FILE_THRESHOLD=100
export FILE_TIMEOUT_MINUTES=15
export MAX_RETRIES=5

# Audio settings
export BPM_MIN=50
export BPM_MAX=200
export MAX_AUDIO_SAMPLES=200000000

# Playlist settings
export MIN_TRACKS_PER_PLAYLIST=15
export MAX_TRACKS_PER_PLAYLIST=300
export KMEANS_N_CLUSTERS=10

# API settings
export LASTFM_API_KEY=your_api_key
export MUSICBRAINZ_RATE_LIMIT=2.0
```

## Configuration Validation

The system validates configuration values:

- **Type validation**: Ensures values match expected types
- **Range validation**: Checks values are within specified ranges
- **Options validation**: Verifies values are in allowed options lists

### Validation Errors

If validation fails, you'll see errors like:

```
Configuration validation failed: 2 errors
  MEMORY_LIMIT_GB: Value 0.1 is below minimum 0.5
  BPM_MAX: Value 300 is above maximum 200
```

## Best Practices

### 1. Start with Defaults

Begin with the default configuration and adjust based on your needs:

```bash
# Use default configuration
docker run -v ./playlistarrays_config.json:/app/config/playlista_config.json playlista config --json
```

### 2. Test with Small Dataset

Test your configuration with a small music library first:

```bash
# Test with a few files
playlista analyze /test/music --parallel --workers 2
```

### 3. Monitor Resource Usage

Watch memory and CPU usage during processing:

```bash
# Monitor during processing
docker stats playlista-container
```

### 4. Use Appropriate Settings for Your System

**Low-memory systems (4GB RAM):**
```json
{
  "MEMORY_LIMIT_GB": 2.0,
  "MEMORY_AWARE": true,
  "LARGE_FILE_THRESHOLD": 25,
  "FAST_MODE_ENABLED": true
}
```

**High-performance systems (16GB+ RAM):**
```json
{
  "MEMORY_LIMIT_GB": 12.0,
  "MEMORY_AWARE": false,
  "LARGE_FILE_THRESHOLD": 200,
  "FAST_MODE_ENABLED": false
}
```

### 5. Configure for Your Use Case

**Development/Debugging:**
```json
{
  "DEBUG": true,
  "LOG_LEVEL": "DEBUG",
  "MEMORY_LIMIT_GB": 4.0,
  "FAST_MODE_ENABLED": true
}
```

**Production Processing:**
```json
{
  "DEBUG": false,
  "LOG_LEVEL": "INFO",
  "MEMORY_LIMIT_GB": 8.0,
  "FAST_MODE_ENABLED": false
}
```

## Troubleshooting

### Configuration Not Loading

1. Check file path:
```bash
docker run -v /absolute/path/to/config.json:/app/config/playlista_config.json playlista config --json
```

2. Check file permissions:
```bash
chmod 644 playlistarrays_config.json
```

3. Validate JSON syntax:
```bash
python -m json.tool playlistarrays_config.json
```

### Memory Issues

1. Reduce memory limits:
```json
{
  "MEMORY_LIMIT_GB": 2.0,
  "MEMORY_AWARE": true,
  "LARGE_FILE_THRESHOLD": 25
}
```

2. Enable fast mode:
```json
{
  "FAST_MODE_ENABLED": true
}
```

### Performance Issues

1. Increase memory limits:
```json
{
  "MEMORY_LIMIT_GB": 12.0,
  "MEMORY_AWARE": false
}
```

2. Adjust batch settings:
```json
{
  "BATCH_SIZE_MULTIPLIER": 20,
  "MAX_BATCH_SIZE": 200
}
```

## Advanced Configuration

### Custom Feature Ranges

Modify the cache-based playlist generation ranges:

```json
"cache_based_ranges": {
  "settings": {
    "BPM_RANGES": {
      "value": {
        "Very_Slow": [0, 50],
        "Slow": [50, 80],
        "Medium": [80, 110],
        "Upbeat": [110, 140],
        "Fast": [140, 170],
        "Very_Fast": [170, 999]
      }
    }
  }
}
```

### Custom Time Slots

Define your own time-based playlist categories:

```json
"time_based_playlists": {
  "settings": {
    "TIME_SLOTS": {
      "value": {
        "Workout": {"min_bpm": 120, "max_bpm": 160, "min_centroid": 2000},
        "Relaxation": {"min_bpm": 60, "max_bpm": 90, "max_centroid": 1500},
        "Party": {"min_bpm": 130, "max_bpm": 180, "min_centroid": 2500}
      }
    }
  }
}
```

### Quality Scoring Weights

Adjust how playlist quality is calculated:

```json
"quality_scoring": {
  "settings": {
    "DIVERSITY_WEIGHT": {
      "value": 0.5,
      "description": "Weight for diversity in overall quality score"
    },
    "COHERENCE_WEIGHT": {
      "value": 0.3,
      "description": "Weight for coherence in overall quality score"
    },
    "BALANCE_WEIGHT": {
      "value": 0.2,
      "description": "Weight for balance in overall quality score"
    }
  }
}
```

This configuration system provides comprehensive control over all aspects of the Playlista system while maintaining sensible defaults for most use cases. 