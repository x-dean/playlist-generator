# Playlist Generator Simple

A streamlined music analysis and playlist generation system with advanced audio feature extraction.

## Features

- **Audio Analysis**: Comprehensive music analysis with 100+ features
- **Memory Optimized**: Efficient processing for large audio files
- **Playlist Generation**: Multiple algorithms for different music styles
- **Docker Ready**: Complete containerized environment
- **Simplified CLI**: Clean, modular command interface

## Quick Start

### Using Docker

```bash
# Build and run the container
docker-compose up --build

# Or use the run script
./run.sh
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python scripts/init_database.py cache/playlista.db

# Analyze music library
playlista analyze --music-path /path/to/music

# Generate playlists
playlista playlist --method all --num-playlists 5
```

## Usage

### Analysis

```bash
# Analyze music files
playlista analyze --music-path /music

# Force re-analysis
playlista analyze --force --music-path /music

# Show statistics
playlista stats
```

### Playlist Generation

```bash
# Generate playlists using all methods
playlista playlist --method all --num-playlists 5

# Generate specific playlist types
playlista playlist --method kmeans --num-playlists 3
playlista playlist --method tag_based --num-playlists 2

# List available methods
playlista playlist-methods
```

### Database Management

```bash
# Initialize database
playlista db --init

# Check integrity
playlista db --integrity-check

# Create backup
playlista db --backup

# Validate structure
playlista validate-database
```

### System Management

```bash
# Show system status
playlista status

# Clean up old data
playlista cleanup

# Retry failed analysis
playlista retry-failed
```

## Configuration

Edit `playlista.conf` to customize:

- Logging settings
- Database configuration
- Analysis parameters
- External API settings
- Playlist generation options

## Project Structure

```
playlist-generator-simple/
├── src/                    # Source code
│   ├── cli/               # CLI modules
│   │   ├── commands.py    # Command handlers
│   │   └── main.py        # CLI entry point
│   ├── core/              # Core functionality
│   ├── api/               # Web API (optional)
│   └── main.py            # Main entry point
├── scripts/               # Utility scripts
├── database/              # Database schemas
├── documentation/         # Essential documentation
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
└── README.md             # This file
```

## Development

The project follows a clean architecture pattern with separated concerns:

- **CLI Layer**: Modular command handling
- **Core Layer**: Business logic and analysis
- **Infrastructure Layer**: Database and external services

## Troubleshooting

### Common Issues

1. **Database not found**: Run `python scripts/init_database.py cache/playlista.db`
2. **Analysis fails**: Check file permissions and audio format support
3. **Memory issues**: Adjust `ANALYSIS_MEMORY_LIMIT_MB` in config
4. **Performance issues**: Check database indexes and WAL mode

### Verification Commands

```bash
# Check database integrity
sqlite3 cache/playlista.db "PRAGMA integrity_check;"

# Count analyzed tracks
sqlite3 cache/playlista.db "SELECT COUNT(*) FROM tracks WHERE analyzed = 1;"

# Check system status
playlista status
```

## License

This project is licensed under the MIT License. 