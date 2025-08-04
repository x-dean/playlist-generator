# Complete Database Schema Usage Guide

## Overview

The `database_schema_complete.sql` file contains a **complete, unified database schema** that includes all original fields plus all the missing audio analysis fields. This is the recommended schema for new installations.

## How to Use

### 1. For New Installations

```bash
# Initialize database with complete schema
python init_database.py cache/playlista.db

# Or manually create database
sqlite3 cache/playlista.db < database_schema_complete.sql
```

### 2. For Existing Installations

If you have an existing database, you can migrate to the complete schema:

```bash
# Backup your existing database first
cp cache/playlista.db cache/playlista.db.backup

# Apply the complete schema (this will add missing columns)
sqlite3 cache/playlista.db < database_schema_complete.sql
```

## What's Included

### ✅ All Original Fields
- Basic music metadata (title, artist, album, genre, year)
- Core audio features (BPM, key, loudness, danceability, energy)
- Analysis metadata and discovery tracking
- All original extended audio features

### ✅ All Missing Audio Analysis Fields

#### High Priority Fields
- **Perceptual Features**: `valence`, `acousticness`, `instrumentalness`, `speechiness`, `liveness`, `popularity`
- **Rhythm Analysis**: `tempo_confidence`, `rhythm_complexity`, `beat_positions`, `onset_times`
- **Harmonic Analysis**: `harmonic_complexity`, `chord_progression`, `chord_changes`
- **Timbre Analysis**: `timbre_brightness`, `timbre_warmth`, `mfcc_delta`, `mfcc_delta2`

#### Medium Priority Fields
- **Extended Spectral**: `spectral_flux`, `spectral_entropy`, `spectral_crest`, `spectral_decrease`
- **Advanced Audio**: `root_mean_square`, `peak_amplitude`, `crest_factor`
- **Musical Structure**: `section_boundaries`, `repetition_rate`
- **Advanced Key**: `key_scale_notes`, `key_chord_progression`, `modulation_points`

#### Low Priority Fields
- **Audio Quality**: `bitrate_quality`, `encoding_quality`, `compression_artifacts`
- **Genre-Specific**: `electronic_elements`, `classical_period`, `jazz_style`, `rock_subgenre`
- **Advanced Statistical**: `spectral_kurtosis`, `spectral_skewness`

## Database Structure

### Main Tables
1. **`tracks`** - Complete audio analysis data (100+ fields)
2. **`tags`** - External API enrichment data
3. **`playlists`** - Playlist definitions
4. **`playlist_tracks`** - Playlist-track relationships
5. **`analysis_cache`** - Failed analysis tracking
6. **`discovery_cache`** - File discovery results
7. **`cache`** - General caching system
8. **`statistics`** - Dashboard metrics

### Performance Views
1. **`track_complete`** - Complete track data with tags
2. **`track_summary`** - Basic track data for lists
3. **`audio_analysis_complete`** - All audio analysis features
4. **`playlist_features`** - Features for playlist generation
5. **`playlist_summary`** - Playlist data with counts
6. **`genre_analysis`** - Genre statistics

## Usage Examples

### 1. Initialize Database
```bash
# Use the complete schema for new installations
python init_database.py cache/playlista.db
```

### 2. Check Database Structure
```sql
-- Check all tables
.tables

-- Check tracks table structure
.schema tracks

-- Check if new fields exist
SELECT name FROM pragma_table_info('tracks') WHERE name LIKE '%valence%';
```

### 3. Query Examples

#### Basic Audio Features
```sql
SELECT title, artist, bpm, key, energy, danceability 
FROM tracks 
WHERE analyzed = TRUE 
LIMIT 10;
```

#### Advanced Audio Features
```sql
SELECT title, artist, valence, acousticness, instrumentalness, 
       rhythm_complexity, harmonic_complexity, timbre_brightness
FROM tracks 
WHERE analyzed = TRUE 
LIMIT 10;
```

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

## CLI Commands

### Database Management
```bash
# Initialize with complete schema
playlista db --init

# Check database integrity
playlista db --integrity-check

# Create backup
playlista db --backup

# Vacuum database
playlista db --vacuum
```

### Analysis and Playlists
```bash
# Analyze files (will populate all fields)
playlista analyze --music-path /music

# Generate playlists using advanced features
playlista playlist --method all --num-playlists 5

# Show statistics
playlista stats --detailed
```

## Benefits

### 1. Complete Audio Analysis
- **100+ audio analysis fields** in a single table
- **Spotify-style perceptual features** for mood-based playlists
- **Advanced rhythm and harmonic analysis** for sophisticated matching
- **Timbre analysis** for detailed music characterization

### 2. Performance Optimized
- **Comprehensive indexing** for fast queries
- **Optimized views** for web UI performance
- **Batch operations** support for efficient data processing

### 3. Future-Proof
- **Extensible schema** for additional features
- **Backward compatible** with existing data
- **Migration support** for schema updates

## Migration from Original Schema

If you're upgrading from the original schema:

1. **Backup your database**
   ```bash
   cp cache/playlista.db cache/playlista.db.backup
   ```

2. **Apply the complete schema**
   ```bash
   sqlite3 cache/playlista.db < database_schema_complete.sql
   ```

3. **Verify the upgrade**
   ```sql
   -- Check if new fields were added
   SELECT COUNT(*) FROM pragma_table_info('tracks');
   -- Should show 100+ columns
   ```

4. **Test functionality**
   ```bash
   python test_database.py
   ```

## Troubleshooting

### Common Issues

1. **Schema File Not Found**
   ```bash
   # Ensure you're in the correct directory
   ls database_schema_complete.sql
   ```

2. **Permission Errors**
   ```bash
   # Check write permissions
   ls -la cache/
   ```

3. **Migration Errors**
   ```bash
   # Restore from backup
   cp cache/playlista.db.backup cache/playlista.db
   ```

### Verification Commands

```bash
# Check database integrity
sqlite3 cache/playlista.db "PRAGMA integrity_check;"

# Count columns in tracks table
sqlite3 cache/playlista.db "SELECT COUNT(*) FROM pragma_table_info('tracks');"

# Check if new fields exist
sqlite3 cache/playlista.db "SELECT name FROM pragma_table_info('tracks') WHERE name IN ('valence', 'acousticness', 'rhythm_complexity');"
```

## Conclusion

The complete schema provides:

- ✅ **All original functionality** plus comprehensive audio analysis
- ✅ **Spotify-style features** for modern playlist generation
- ✅ **Advanced audio analysis** for sophisticated music matching
- ✅ **Performance optimization** for web UI and batch processing
- ✅ **Future-proof design** for additional features

This is the recommended schema for all new installations and provides the foundation for advanced playlist generation capabilities. 