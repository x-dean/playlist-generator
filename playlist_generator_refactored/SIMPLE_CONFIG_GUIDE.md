# Simple Configuration Guide

This guide shows how to use the JSON configuration file with your existing Docker Compose setup.

## Current Setup

Your Docker Compose file is now simplified to only include essential environment variables:

```yaml
services:
  playlista:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: playlista
    volumes:
      # Mount the JSON configuration file
      - ./playlista_config.json:/app/config/playlista_config.json:ro
      # Mount music library
      - /root/music/library:/music:ro
      # Mount output directories
      - /root/music/playlista/cache:/app/cache
      - /root/music/playlista/logs:/app/logs
      - /root/music/playlista/playlists:/app/playlists
      - /root/music/playlista/models:/app/feature_extraction/models
    environment:
      # Only essential environment variables
      - PYTHONPATH=/app/src
      - LASTFM_API_KEY=9fd1f789ebdf1297e6aa1590a13d85e0
    stdin_open: true
    tty: true
```

## Configuration File

All settings are now controlled through `playlista_config.json`:

```json
{
  "playlista_config": {
    "paths": {
      "settings": {
        "MUSIC_PATH": { "value": "/music" },
        "CACHE_DIR": { "value": "/app/cache" },
        "LOG_DIR": { "value": "/app/logs" },
        "OUTPUT_DIR": { "value": "/app/playlists" }
      }
    },
    "memory_management": {
      "settings": {
        "MEMORY_LIMIT_GB": { "value": 6.0 },
        "MEMORY_AWARE": { "value": true }
      }
    },
    "logging": {
      "settings": {
        "LOG_LEVEL": { "value": "INFO" }
      }
    },
    "file_processing": {
      "settings": {
        "LARGE_FILE_THRESHOLD_MB": { "value": 50 }
      }
    },
    "playlist_generation": {
      "settings": {
        "MIN_TRACKS_PER_GENRE": { "value": 10 }
      }
    }
  }
}
```

## How to Use

### 1. Edit Configuration

Simply modify the `value` fields in `playlista_config.json`:

```json
{
  "playlista_config": {
    "memory_management": {
      "settings": {
        "MEMORY_LIMIT_GB": {
          "value": 8.0  // Change from 6.0 to 8.0
        }
      }
    },
    "logging": {
      "settings": {
        "LOG_LEVEL": {
          "value": "DEBUG"  // Change from INFO to DEBUG
        }
      }
    }
  }
}
```

### 2. Run Container

```bash
# Start the container
docker-compose up -d

# Or run interactively
docker-compose run --rm playlista bash
```

### 3. Verify Configuration

```bash
# Check that configuration is loaded
docker-compose run --rm playlista playlista config --json
```

## Common Configuration Changes

### Memory Settings

```json
"memory_management": {
  "settings": {
    "MEMORY_LIMIT_GB": { "value": 8.0 },  // Increase for more memory
    "MEMORY_AWARE": { "value": true }       // Enable memory monitoring
  }
}
```

### Logging

```json
"logging": {
  "settings": {
    "LOG_LEVEL": { "value": "DEBUG" }  // More detailed logs
  }
}
```

### File Processing

```json
"file_processing": {
  "settings": {
    "LARGE_FILE_THRESHOLD_MB": { "value": 100 }  // Process larger files
  }
}
```

### Playlist Generation

```json
"playlist_generation": {
  "settings": {
    "MIN_TRACKS_PER_GENRE": { "value": 15 }  // More tracks per genre
  }
}
```

### Processing Modes

```json
"processing_modes": {
  "settings": {
    "DEBUG": { "value": true },           // Enable debug mode
    "FAST_MODE_ENABLED": { "value": true } // Enable fast processing
  }
}
```

## Benefits

1. **Simplified Compose**: Only essential environment variables remain
2. **Centralized Configuration**: All settings in one JSON file
3. **Easy to Modify**: Just edit the JSON file and restart
4. **Version Control**: Configuration file can be tracked in git
5. **No CLI Commands**: Everything configured through JSON

## Quick Commands

```bash
# Start with current configuration
docker-compose up -d

# Check configuration
docker-compose run --rm playlista playlista config --json

# Run analysis with current settings
docker-compose run --rm playlista playlista analyze /music

# Run full pipeline
docker-compose run --rm playlista playlista pipeline /music --generate
```

That's it! Just edit the JSON file and restart the container to apply changes. 