# Database Setup Guide

## Overview
This project now uses a single, complete database schema that includes all audio analysis fields from the start.

## Quick Setup

### 1. Initialize Database
```bash
# Option 1: Using the CLI
playlista db --init-schema

# Option 2: Using the initialization script
python init_database.py
```

### 2. Verify Setup
```bash
# Check database status
playlista status

# Show schema information
playlista db --show-schema
```

### 3. Start Analysis
```bash
# Analyze your music collection
playlista analyze --music-path /path/to/your/music

# Generate playlists
playlista playlist --method kmeans --num-playlists 5
```

## Database Schema

The complete schema includes:

### Core Audio Features
- BPM, key, mode, energy, danceability
- Valence, acousticness, instrumentalness
- Speechiness, liveness

### Advanced Analysis
- Harmonic analysis (chord detection, complexity)
- Beat tracking (onset detection, rhythm patterns)
- Spectral analysis (flux, entropy, crest)
- Timbre analysis (brightness, warmth, hardness)
- Audio quality metrics

### Metadata
- Complete file metadata
- External API data (MusicBrainz, Spotify)
- MusiCNN embeddings and tags
- Essentia features

## Database Management Commands

```bash
# Initialize schema
playlista db --init-schema

# Show schema information
playlista db --show-schema

# Validate all data
playlista db --validate-all-data

# Repair corrupted data
playlista db --repair-corrupted

# Check database health
playlista db --health

# Perform maintenance
playlista db --maintenance

# Create backup
playlista db --backup

# Check integrity
playlista db --integrity-check
```

## File Validation

```bash
# Validate individual file
playlista validate-database /path/to/audio/file --validate

# Fix individual file
playlista validate-database /path/to/audio/file --fix
```

## Database Location

The database is stored at:
- Default: `./database/playlist_generator.db`
- Configurable via environment variable: `PLAYLIST_DB_PATH`

## Schema Features

### 100+ Columns Including:
- All essential audio features for playlist generation
- Advanced harmonic and rhythmic analysis
- Spectral and timbre analysis
- Complete metadata storage
- External API integration fields
- Performance-optimized indexes
- Web UI optimization views

### Performance Optimizations:
- Indexed columns for fast queries
- Optimized views for web UI
- Efficient data storage
- Automatic maintenance triggers

## Troubleshooting

### Database Not Found
```bash
playlista db --init-schema
```

### Corrupted Data
```bash
playlista db --repair-corrupted
```

### Schema Issues
```bash
playlista db --fix-schema
```

### Performance Issues
```bash
playlista db --vacuum
playlista db --maintenance
```

## Next Steps

1. Initialize the database
2. Run analysis on your music collection
3. Generate playlists using various methods
4. Monitor database health regularly
5. Create backups before major operations

The database is now ready for comprehensive audio analysis and playlist generation! 