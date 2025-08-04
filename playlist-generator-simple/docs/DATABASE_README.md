# Database Implementation Documentation

## Overview

The Playlist Generator Simple project uses a SQLite database optimized for web UI performance and comprehensive music analysis data storage.

## Schema Design

### Core Tables

#### `tracks` - Main Music Data Table
Primary table storing all music metadata and analysis results:

**Essential Fields:**
- `file_path` (TEXT, UNIQUE) - Full file path
- `file_hash` (TEXT) - MD5 hash for change detection
- `filename` (TEXT) - Just the filename
- `file_size_bytes` (INTEGER) - File size in bytes
- `analysis_date` (TIMESTAMP) - When analysis was performed

**Music Metadata:**
- `title` (TEXT) - Song title
- `artist` (TEXT) - Artist name
- `album` (TEXT) - Album name
- `track_number` (INTEGER) - Track number
- `genre` (TEXT) - Music genre
- `year` (INTEGER) - Release year
- `duration` (REAL) - Track duration in seconds

**Audio Features (for playlist generation):**
- `bpm` (REAL) - Beats per minute
- `key` (TEXT) - Musical key (C, D, E, etc.)
- `mode` (TEXT) - Major/minor mode
- `loudness` (REAL) - Loudness in dB
- `danceability` (REAL) - Danceability score (0-1)
- `energy` (REAL) - Energy score (0-1)

**Analysis Metadata:**
- `analysis_type` (TEXT) - 'full', 'basic', 'discovery_only'
- `analyzed` (BOOLEAN) - Whether analysis is complete
- `long_audio_category` (TEXT) - For long audio files

**Extended Audio Features:**
- Rhythm features: `rhythm_confidence`, `bpm_estimates`, `bpm_intervals`
- Spectral features: `spectral_centroid`, `spectral_flatness`, etc.
- Loudness features: `dynamic_complexity`, `loudness_range`
- Key features: `scale`, `key_strength`, `key_confidence`
- MFCC features: `mfcc_coefficients`, `mfcc_bands`, `mfcc_std`
- MusiCNN features: `embedding`, `tags` (JSON)
- Chroma features: `chroma_mean`, `chroma_std` (JSON)

#### `tags` - External API Data
Stores enrichment data from external APIs:

- `track_id` (INTEGER) - Foreign key to tracks
- `source` (TEXT) - API source ('musicbrainz', 'lastfm', 'spotify')
- `tag_name` (TEXT) - Tag name
- `tag_value` (TEXT) - Tag value
- `confidence` (REAL) - Confidence score (0-1)

#### `playlists` - Playlist Definitions
Stores playlist metadata:

- `name` (TEXT, UNIQUE) - Playlist name
- `description` (TEXT) - Playlist description
- `generation_method` (TEXT) - How playlist was generated
- `generation_params` (TEXT) - JSON parameters used
- `track_count` (INTEGER) - Number of tracks
- `total_duration` (REAL) - Total duration in seconds

#### `playlist_tracks` - Playlist-Track Junction
Links playlists to tracks with ordering:

- `playlist_id` (INTEGER) - Foreign key to playlists
- `track_id` (INTEGER) - Foreign key to tracks
- `position` (INTEGER) - Track position in playlist

#### `analysis_cache` - Failed Analysis Tracking
Tracks files that failed analysis:

- `file_path` (TEXT, UNIQUE) - File path
- `filename` (TEXT) - Filename
- `status` (TEXT) - 'failed', 'partial', 'success'
- `error_message` (TEXT) - Error details
- `retry_count` (INTEGER) - Number of retry attempts
- `last_retry_date` (TIMESTAMP) - Last retry attempt

#### `discovery_cache` - File Discovery Tracking
Tracks directory scanning results:

- `directory_path` (TEXT) - Scanned directory
- `file_count` (INTEGER) - Number of files found
- `scan_duration` (REAL) - Time taken to scan
- `status` (TEXT) - 'completed', 'failed', 'in_progress'

#### `cache` - General Caching System
Stores API responses and computed data:

- `cache_key` (TEXT, UNIQUE) - Cache key
- `cache_value` (TEXT) - JSON serialized data
- `cache_type` (TEXT) - 'api_response', 'computed', 'statistics'
- `expires_at` (TIMESTAMP) - Expiration time

#### `statistics` - Web UI Dashboard Data
Stores metrics for dashboards:

- `category` (TEXT) - Metric category
- `metric_name` (TEXT) - Metric name
- `metric_value` (REAL) - Metric value
- `metric_data` (TEXT) - JSON for complex metrics

## Performance Optimizations

### Indexes
The database includes comprehensive indexing for fast queries:

**Music Lookup Indexes:**
- `idx_tracks_artist` - Artist lookups
- `idx_tracks_title` - Title lookups
- `idx_tracks_artist_title` - Artist+title combinations
- `idx_tracks_album` - Album lookups
- `idx_tracks_genre` - Genre lookups
- `idx_tracks_year` - Year lookups

**Audio Feature Indexes:**
- `idx_tracks_bpm` - BPM-based queries
- `idx_tracks_key` - Key-based queries
- `idx_tracks_loudness` - Loudness queries
- `idx_tracks_danceability` - Danceability queries
- `idx_tracks_energy` - Energy queries

**Composite Indexes:**
- `idx_tracks_artist_album` - Artist+album combinations
- `idx_tracks_genre_year` - Genre+year combinations
- `idx_tracks_bpm_energy` - BPM+energy combinations

### Views for Web UI
Optimized views for web interface performance:

- `track_complete` - Complete track data with tags
- `track_summary` - Simplified track data for lists
- `playlist_summary` - Playlist data with track counts
- `statistics_summary` - Aggregated statistics

### Database Settings
Performance-optimized SQLite settings:

- WAL mode enabled for concurrent access
- Synchronous mode set to NORMAL
- Cache size: 10,000 pages
- Temp store in memory
- Connection timeout: 30 seconds

## Database Operations

### Initialization
```bash
# For new installations
python init_database.py cache/playlista.db

# For existing installations
python migrate_database.py cache/playlista.db
```

### CLI Commands
```bash
# Show database status
playlista status --detailed

# Show statistics
playlista stats --detailed

# Show failed files
playlista status --failed-files

# Clean up failed analysis
playlista cleanup --max-retries 3
```

### Database Manager API

**Core Methods:**
- `save_analysis_result()` - Save analysis data
- `get_analysis_result()` - Retrieve analysis data
- `save_playlist()` - Save playlist
- `get_playlist()` - Retrieve playlist
- `get_all_playlists()` - List all playlists

**Cache Methods:**
- `save_cache()` - Save to cache
- `get_cache()` - Retrieve from cache
- `get_cache_by_type()` - Get cache by type

**Statistics Methods:**
- `save_statistic()` - Save metric
- `get_statistics_summary()` - Get dashboard data
- `get_database_statistics()` - Get DB stats

**Web UI Methods:**
- `get_tracks_for_web_ui()` - Optimized track queries
- `get_web_ui_dashboard_data()` - Dashboard data

## Data Flow

### Analysis Pipeline
1. **File Discovery** → `discovery_cache` table
2. **Analysis Processing** → `tracks` table
3. **Failed Analysis** → `analysis_cache` table
4. **Metadata Enrichment** → `tags` table
5. **Statistics Collection** → `statistics` table

### Playlist Generation
1. **Query Tracks** → Read from `tracks` table
2. **Generate Playlist** → Algorithm processing
3. **Save Playlist** → `playlists` + `playlist_tracks` tables

### Caching Strategy
1. **API Responses** → `cache` table (type: 'api_response')
2. **Computed Data** → `cache` table (type: 'computed')
3. **Statistics** → `cache` table (type: 'statistics')

## Configuration

### Database Settings (playlista.conf)
```ini
# Database path (Docker internal path)
DB_PATH=/app/cache/playlista.db

# Cache settings
DB_CACHE_DEFAULT_EXPIRY_HOURS=24
DB_CACHE_MAX_SIZE_MB=100

# Performance settings
DB_CONNECTION_TIMEOUT_SECONDS=30
DB_MAX_RETRY_ATTEMPTS=3
DB_BATCH_SIZE=100

# Cleanup settings
DB_CLEANUP_RETENTION_DAYS=30
DB_FAILED_ANALYSIS_RETENTION_DAYS=7
```

## Migration and Maintenance

### Migration Process
1. **Backup Creation** - Automatic backup before migration
2. **Schema Creation** - New schema with optimized structure
3. **Data Migration** - Preserve existing data
4. **Column Addition** - Add missing columns if needed
5. **Index Creation** - Performance optimization

### Maintenance Tasks
- **Cache Cleanup** - Remove expired cache entries
- **Failed Analysis Cleanup** - Remove old failed entries
- **Statistics Cleanup** - Remove old statistics
- **Database Backup** - Regular backups

### Troubleshooting

**Common Issues:**
1. **Schema File Not Found** - Check `database_schema.sql` location
2. **Permission Errors** - Ensure write access to cache directory
3. **Locked Database** - Check for concurrent access
4. **Corrupted Database** - Restore from backup

**Debug Commands:**
```bash
# Check database integrity
sqlite3 cache/playlista.db "PRAGMA integrity_check;"

# Show table structure
sqlite3 cache/playlista.db ".schema tracks"

# Check database size
ls -lh cache/playlista.db
```

## Performance Monitoring

### Key Metrics
- **Query Performance** - Track slow queries
- **Cache Hit Rate** - Monitor cache effectiveness
- **Database Size** - Monitor growth
- **Connection Count** - Track concurrent access

### Optimization Tips
1. **Use Indexed Queries** - Leverage created indexes
2. **Batch Operations** - Use batch inserts/updates
3. **Cache Frequently** - Cache expensive operations
4. **Monitor Growth** - Regular size monitoring
5. **Backup Regularly** - Prevent data loss

## Security Considerations

### Data Protection
- **File Paths** - Store relative paths when possible
- **API Keys** - Never store in database
- **User Data** - No personal data stored
- **Backup Security** - Secure backup storage

### Access Control
- **Read-Only Access** - Web UI uses read-only queries
- **Write Access** - Limited to analysis processes
- **Connection Limits** - Max 10 concurrent connections
- **Timeout Protection** - 30-second connection timeout

## Future Enhancements

### Planned Improvements
1. **Partitioning** - Large table partitioning
2. **Compression** - Data compression for large datasets
3. **Replication** - Read replicas for web UI
4. **Advanced Caching** - Redis integration
5. **Analytics** - Advanced analytics queries

### Schema Evolution
- **Backward Compatibility** - Maintain existing APIs
- **Migration Scripts** - Automatic schema updates
- **Version Tracking** - Database version management
- **Rollback Support** - Safe rollback procedures 