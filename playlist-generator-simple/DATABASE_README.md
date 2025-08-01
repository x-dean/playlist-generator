# Database Redesign

The database has been simplified to focus on core functionality while maintaining performance.

## Schema Overview

### Core Tables

- **tracks**: Main table with essential music metadata and audio features
- **tags**: External API tags and enrichment data
- **playlists**: Playlist definitions
- **playlist_tracks**: Junction table linking playlists to tracks
- **analysis_cache**: Failed/partial analysis tracking
- **cache**: General caching system

### Key Changes

1. **Simplified Structure**: Removed complex normalized tables (spectral_features, loudness_features, etc.)
2. **Essential Data Only**: Focus on core music metadata and key audio features
3. **Performance Optimized**: Streamlined indexes and queries
4. **Backward Compatible**: Migration script preserves existing data

## Migration

### For New Installations

```bash
python init_database.py cache/playlista.db
```

### For Existing Installations

```bash
python migrate_database.py cache/playlista.db
```

## Schema Details

### Tracks Table

Essential fields for music identification and playlist generation:

- File metadata (path, hash, size)
- Music metadata (title, artist, album, genre, year)
- Audio features (BPM, key, loudness, danceability, energy)
- Analysis metadata (type, category)

### Tags Table

Flexible tag storage for external API data:

- Track association
- Source identification (musicbrainz, lastfm, etc.)
- Tag name/value pairs
- Confidence scoring

### Playlists

Simple playlist management:

- Playlist metadata (name, description)
- Track ordering via junction table
- Timestamp tracking

## Performance

- Optimized indexes for common queries
- Simplified joins for faster playlist generation
- Reduced storage overhead
- Streamlined data access patterns

## Migration Notes

- Existing data is preserved during migration
- Complex features are simplified but functional
- Backward compatibility maintained
- Performance improvements expected 