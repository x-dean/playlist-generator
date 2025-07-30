# Docker Usage Guide for Playlist Generator Simple

This guide explains how to use the Docker setup for real-life testing of the playlist generator.

## Prerequisites

- Docker and Docker Compose installed
- Music library directory with audio files
- Optional: LastFM API key for metadata enrichment

## Quick Start

### 1. Prepare Your Environment

Create the required directories:

```bash
mkdir -p music playlists cache logs failed_files models
```

### 2. Add Your Music

Place your music files in the `music/` directory:

```bash
# Example structure
music/
├── Artist1/
│   ├── Album1/
│   │   ├── song1.mp3
│   │   └── song2.flac
│   └── Album2/
└── Artist2/
    └── Album3/
```

### 3. Configure Settings (Optional)

Edit `playlista.conf` to customize settings:

```bash
# Edit playlista.conf to customize:
# - Analysis settings
# - Progress bar options
# - Resource limits
# - External API keys
```

### 4. Build and Run

```bash
# Build the container
docker-compose build

# Run analysis
docker-compose run --rm playlista-simple analyze

# Run with specific music path
docker-compose run --rm playlista-simple analyze /music/Artist1

# Run playlist generation
docker-compose run --rm playlista-simple playlist

# Interactive shell
docker-compose run --rm playlista-simple bash
```

## Volume Mappings

The Docker setup includes the following volume mappings:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./music` | `/music` | Your music library (read-only) |
| `./playlists` | `/app/playlists` | Generated playlists output |
| `./cache` | `/app/cache` | Analysis cache and database |
| `./logs` | `/app/logs` | Application logs |
| `./failed_files` | `/app/failed_files` | Files that failed analysis |
| `./models` | `/app/models` | Custom models (optional) |
| `./playlista.conf` | `/app/playlista.conf` | Configuration file |

## Configuration

### Configuration

All settings are managed through the `playlista.conf` file:

| Setting Category | Description |
|-----------------|-------------|
| **Logging** | Log levels, file paths, formatting |
| **Database** | Cache settings, cleanup, retention |
| **File Discovery** | Supported formats, size limits, patterns |
| **Analysis** | Processing modes, timeouts, resource limits |
| **Progress Bars** | Visual progress tracking options |
| **External APIs** | LastFM, MusicBrainz integration |

### Resource Limits

The container is configured with:

- **Memory**: 8GB limit, 2GB reservation
- **CPU**: 4 cores limit, 1 core reservation
- **Health Check**: 30s interval

## Usage Examples

### Basic Analysis

```bash
# Analyze all music files
docker-compose run --rm playlista-simple analyze

# Analyze specific directory
docker-compose run --rm playlista-simple analyze /music/Rock

# Force re-analysis (ignore cache)
docker-compose run --rm playlista-simple analyze --force
```

### Playlist Generation

```bash
# Generate all playlist types
docker-compose run --rm playlista-simple playlist

# Generate specific playlist type
docker-compose run --rm playlista-simple playlist --method tag_based

# Generate with custom settings
docker-compose run --rm playlista-simple playlist --min_tracks 20
```

### Interactive Usage

```bash
# Start interactive shell
docker-compose run --rm playlista-simple bash

# Inside container
playlista analyze
playlista playlist
playlista --help
```

### Monitoring and Debugging

```bash
# View logs
docker-compose logs playlista-simple

# Follow logs in real-time
docker-compose logs -f playlista-simple

# Check container status
docker-compose ps

# Access container shell
docker-compose exec playlista-simple bash
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Fix directory permissions
   sudo chown -R $USER:$USER music playlists cache logs failed_files
   ```

2. **Out of Memory**
   ```bash
   # Increase memory limit in docker-compose.yml
   memory: 12G
   ```

3. **Slow Analysis**
   ```bash
   # Reduce CPU threshold
   CPU_THRESHOLD_PERCENT=70
   ```

4. **Progress Bar Not Showing**
   ```bash
   # Enable in environment
   PROGRESS_BAR_ENABLED=true
   ```

### Debug Mode

```bash
# Run with debug logging
docker-compose run --rm -e LOG_LEVEL=DEBUG playlista-simple analyze
```

### Clean Start

```bash
# Remove all containers and volumes
docker-compose down -v

# Rebuild from scratch
docker-compose build --no-cache

# Start fresh
docker-compose up
```

## Performance Tips

1. **Use SSD storage** for cache and database
2. **Allocate sufficient memory** (8GB+ recommended)
3. **Use multiple CPU cores** for parallel processing
4. **Mount music from fast storage** (SSD/NVMe)
5. **Enable progress bars** for monitoring

## Security Notes

- Music directory is mounted read-only
- Configuration file is mounted read-only
- Container runs with limited privileges
- Failed files are isolated in separate volume

## Next Steps

After successful analysis:

1. Check generated playlists in `./playlists/`
2. Review logs in `./logs/`
3. Examine failed files in `./failed_files/`
4. Customize configuration in `playlista.conf`
5. Add custom models to `./models/` 