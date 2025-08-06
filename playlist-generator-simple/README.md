# Playlist Generator Simple

A comprehensive music analysis and playlist generation system with advanced audio feature extraction and sophisticated playlist algorithms.

## Features

- **Advanced Audio Analysis**: 100+ audio features including Spotify-style perceptual features
- **Memory Optimization**: 75-95% memory reduction for large audio files
- **Sophisticated Playlist Generation**: Multiple algorithms for different music styles
- **Web UI**: Interactive dashboard for music library management
- **Performance Optimized**: Fast analysis and playlist generation
- **Docker Ready**: Complete containerized environment

## Quick Start

### Using Docker (Recommended)

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

# Initialize database with complete schema
python scripts/init_database.py cache/playlista.db

# Analyze your music library
playlista analyze --music-path /path/to/music

# Generate playlists
playlista playlist --method all --num-playlists 5
```

## Database Schema

The system uses a **complete database schema** (`database_schema_complete.sql`) that includes:

- **100+ Audio Analysis Fields**: Comprehensive music analysis data
- **Spotify-Style Features**: `valence`, `acousticness`, `instrumentalness`, etc.
- **Advanced Analysis**: Rhythm, harmonic, timbre, and spectral analysis
- **Performance Optimized**: Comprehensive indexing and views
- **Web UI Ready**: Optimized views for dashboard performance

### Key Features

#### Perceptual Features (Spotify-style)
- `valence` - Positivity/negativity
- `acousticness` - Acoustic vs electronic nature
- `instrumentalness` - Instrumental vs vocal content
- `speechiness` - Presence of speech
- `liveness` - Presence of live audience
- `popularity` - Popularity score

#### Advanced Audio Analysis
- **Rhythm Analysis**: `tempo_confidence`, `rhythm_complexity`, `beat_positions`
- **Harmonic Analysis**: `harmonic_complexity`, `chord_progression`, `chord_changes`
- **Timbre Analysis**: `timbre_brightness`, `timbre_warmth`, `mfcc_delta`
- **Spectral Analysis**: `spectral_flux`, `spectral_entropy`, `spectral_crest`
- **Musical Structure**: `section_boundaries`, `repetition_rate`

## Usage

### Database Management

```bash
# Initialize database
playlista db --init

# Check integrity
playlista db --integrity-check

# Create backup
playlista db --backup

# Vacuum database
playlista db --vacuum
```

### Analysis

```bash
# Analyze music files
playlista analyze --music-path /music

# Force re-analysis
playlista analyze --force --music-path /music

# Analyze with specific options
playlista analyze --no-cache --music-path /music
```

### Playlist Generation

```bash
# Generate playlists using all methods
playlista playlist --method all --num-playlists 5

# Generate specific playlist types
playlista playlist --method kmeans --num-playlists 3
playlista playlist --method tag_based --num-playlists 2
playlista playlist --method time_based --num-playlists 1

# Generate with advanced features
playlista playlist --method all --num-playlists 5 --use-advanced-features
```

### Web UI

```bash
# Start web interface
playlista web --port 8500

# Access at http://localhost:8500
```

## Configuration

Edit `config/playlista.conf` to customize:

- Database settings
- Analysis parameters
- Playlist generation options
- Web UI settings
- Logging configuration

## Project Structure

```
playlist-generator-simple/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   ├── api/               # Web API
│   ├── application/       # Application layer
│   ├── domain/            # Domain models
│   ├── infrastructure/    # Infrastructure layer
│   └── main.py            # Main entry point
├── documentation/         # All documentation files
├── scripts/               # Utility scripts
├── config/                # Configuration files
├── database/              # Database schemas
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
└── README.md             # This file
```

## Advanced Usage

### Query Examples

#### Mood-Based Playlists
```sql
SELECT title, artist, valence, acousticness, energy
FROM tracks 
WHERE valence > 0.7 AND acousticness > 0.5
ORDER BY valence DESC;
```

#### Rhythm-Based Playlists
```sql
SELECT title, artist, bpm, rhythm_complexity, tempo_confidence
FROM tracks 
WHERE rhythm_complexity > 0.6 AND tempo_confidence > 0.8
ORDER BY bpm;
```

#### Advanced Audio Features
```sql
SELECT title, artist, valence, acousticness, instrumentalness, 
       rhythm_complexity, harmonic_complexity, timbre_brightness
FROM tracks 
WHERE analyzed = TRUE 
LIMIT 10;
```

## Development

### Development

The project follows a clean architecture pattern with separated layers for domain, application, infrastructure, and API concerns.

### Building

```bash
# Build Docker image
docker build -t playlist-generator .

# Run with custom configuration
docker run -v /music:/app/music -p 8500:8500 playlist-generator
```

## Troubleshooting

### Common Issues

1. **Database not found**: Run `python scripts/init_database.py cache/playlista.db`
2. **Analysis fails**: Check file permissions and audio format support
3. **Web UI not accessible**: Verify port 8500 is available
4. **Performance issues**: Check database indexes and WAL mode

### Verification Commands

```bash
# Check database integrity
sqlite3 cache/playlista.db "PRAGMA integrity_check;"

# Count audio analysis fields
sqlite3 cache/playlista.db "SELECT COUNT(*) FROM pragma_table_info('tracks');"

# Check if advanced features exist
sqlite3 cache/playlista.db "SELECT name FROM pragma_table_info('tracks') WHERE name IN ('valence', 'acousticness', 'rhythm_complexity');"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 