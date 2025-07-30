# Playlist Generation

This document describes the playlist generation functionality that has been ported from the original project to the simple project structure.

## Overview

The playlist generation system provides multiple algorithms for creating playlists from analyzed audio files. It integrates with the existing analysis and database infrastructure to generate playlists based on various criteria.

## Features

### Playlist Generation Methods

1. **K-means Clustering** (`kmeans`)
   - Groups tracks based on audio features (BPM, danceability, loudness, etc.)
   - Creates playlists with similar musical characteristics
   - Uses scikit-learn's KMeans algorithm

2. **Similarity-based Selection** (`similarity`)
   - Starts with a random track and adds similar tracks
   - Calculates similarity based on BPM, key, danceability
   - Creates cohesive playlists with gradual transitions

3. **Random Selection** (`random`)
   - Simple random track selection
   - Good for testing and baseline comparison
   - Fastest generation method

4. **Time-based Scheduling** (`time_based`)
   - Creates playlists for different times of day
   - Morning: moderate BPM (80-120)
   - Afternoon: higher BPM (100-140)
   - Evening: moderate to high BPM (90-130)
   - Night: lower BPM (60-100)

5. **Tag-based Selection** (`tag_based`)
   - Groups tracks by metadata tags (genre, artist, album)
   - Creates genre-specific or artist-specific playlists
   - Uses available metadata from analysis

6. **Cache-based Selection** (`cache_based`)
   - Uses previously generated playlists as templates
   - Selects tracks with similar characteristics to cached playlists
   - Good for maintaining consistency across sessions

7. **Feature Group Selection** (`feature_group`)
   - Groups tracks by audio feature characteristics
   - Creates playlists based on tempo (slow/medium/fast) or energy levels
   - Uses BPM and danceability for classification

8. **Mixed Approach** (`mixed`)
   - Combines multiple generation methods
   - Creates diverse playlists with different characteristics
   - Balances variety and coherence

9. **All Methods** (`all`)
   - Generates playlists using all available methods
   - Creates maximum variety of playlist types
   - Good for comprehensive playlist generation

## Usage

### Command Line Interface

```bash
# Generate playlists using K-means clustering
python src/analysis_cli.py playlist --method kmeans --num-playlists 5 --playlist-size 20 --save

# Generate playlists using all methods
python src/analysis_cli.py playlist --method all --num-playlists 8 --playlist-size 15

# List available playlist generation methods
python src/analysis_cli.py playlist-methods

# Generate and save playlists to custom directory
python src/analysis_cli.py playlist --method similarity --num-playlists 3 --playlist-size 25 --save --output-dir my_playlists
```

### Programmatic Usage

```python
from core.playlist_generator import PlaylistGenerator, PlaylistGenerationMethod

# Create playlist generator
playlist_generator = PlaylistGenerator()

# Generate playlists
playlists = playlist_generator.generate_playlists(
    method=PlaylistGenerationMethod.KMEANS,
    num_playlists=5,
    playlist_size=20
)

# Save playlists to files
playlist_generator.save_playlists(playlists, 'output_directory')

# Get playlist statistics
stats = playlist_generator.get_playlist_statistics(playlists)
print(f"Generated {stats['total_playlists']} playlists with {stats['total_tracks']} total tracks")
```

## Configuration

Playlist generation settings can be configured in `playlista.conf`:

```ini
# Playlist Generation Settings
DEFAULT_PLAYLIST_SIZE=20
MIN_TRACKS_PER_GENRE=10
MAX_PLAYLISTS=8
SIMILARITY_THRESHOLD=0.7
DIVERSITY_THRESHOLD=0.3
PLAYLIST_OUTPUT_DIR=playlists
PLAYLIST_GENERATION_ENABLED=true
PLAYLIST_SAVE_METADATA=true
PLAYLIST_OPTIMIZATION_ENABLED=true
PLAYLIST_DEFAULT_METHOD=all
PLAYLIST_KMEANS_CLUSTERS=5
PLAYLIST_SIMILARITY_ALGORITHM=cosine
PLAYLIST_TIME_SLOTS=morning,afternoon,evening,night
PLAYLIST_TAG_PRIORITY=genre,artist,album
PLAYLIST_CACHE_ENABLED=true
PLAYLIST_FEATURE_GROUPS=tempo,energy,mood
PLAYLIST_MIXED_WEIGHTS=kmeans:0.4,similarity:0.3,random:0.3
```

## Database Integration

The playlist generation system integrates with the existing database:

### Database Methods

- `get_analyzed_tracks()`: Retrieves all analyzed tracks with features
- `get_cached_playlists()`: Retrieves previously generated playlists
- `get_track_features()`: Gets features for a specific track
- `get_tracks_by_feature_range()`: Gets tracks within a feature range

### Playlist Storage

Playlists are stored in the database with:
- Name and description
- Track list
- Audio features (for cache-based generation)
- Metadata and creation timestamp

## Output Formats

### M3U Playlist Files

Playlists are saved as `.m3u` files with:
- Playlist header with metadata
- Track paths (one per line)
- Compatible with most media players

### Metadata JSON

A `playlists_metadata.json` file contains:
- Generation timestamp
- Total playlist count
- Individual playlist metadata
- Statistics and quality metrics

## Architecture

### Core Components

1. **PlaylistGenerator**: Main orchestrator class
2. **BasePlaylistGenerator**: Abstract base class for generators
3. **Specific Generators**: Implementation classes for each method
4. **Playlist**: Data class representing a playlist
5. **PlaylistGenerationMethod**: Enumeration of available methods

### Class Hierarchy

```
PlaylistGenerator
├── KMeansPlaylistGenerator
├── SimilarityPlaylistGenerator
├── RandomPlaylistGenerator
├── TimeBasedPlaylistGenerator
├── TagBasedPlaylistGenerator
├── CacheBasedPlaylistGenerator
└── FeatureGroupPlaylistGenerator
```

## Testing

Run the playlist generation tests:

```bash
python test_playlist_generation.py
```

The test script verifies:
- Database integration
- Playlist generation with different methods
- Playlist saving functionality
- Error handling and edge cases

## Dependencies

The playlist generation system requires:
- `numpy`: For numerical operations
- `scikit-learn`: For K-means clustering
- `pandas`: For data manipulation
- `sqlite3`: For database operations (built-in)

## Performance Considerations

- **K-means clustering**: Most computationally intensive, best for large datasets
- **Random selection**: Fastest method, good for testing
- **Similarity-based**: Moderate performance, good balance of speed and quality
- **Cache-based**: Fast if cached playlists exist, falls back to random
- **Time-based/Tag-based**: Fast, limited by available metadata

## Future Enhancements

Potential improvements for the playlist generation system:

1. **Advanced Algorithms**
   - Collaborative filtering
   - Neural network-based recommendations
   - Mood-based classification

2. **Enhanced Features**
   - Playlist quality scoring
   - User preference learning
   - Cross-fade optimization

3. **Integration**
   - Music streaming service APIs
   - Social playlist sharing
   - Real-time playlist generation

4. **Performance**
   - GPU acceleration for clustering
   - Parallel playlist generation
   - Incremental playlist updates

## Troubleshooting

### Common Issues

1. **No playlists generated**
   - Check if analyzed tracks exist in database
   - Verify playlist generation is enabled in config
   - Check log files for error messages

2. **Empty playlists**
   - Ensure sufficient analyzed tracks are available
   - Check feature extraction completed successfully
   - Verify database connectivity

3. **Poor playlist quality**
   - Adjust similarity thresholds in configuration
   - Try different generation methods
   - Ensure audio analysis completed with good quality

### Debug Mode

Enable debug logging to troubleshoot issues:

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Run with verbose output
python src/analysis_cli.py playlist --method kmeans --num-playlists 2 --playlist-size 5
```

## Contributing

To add new playlist generation methods:

1. Create a new generator class inheriting from `BasePlaylistGenerator`
2. Implement the `generate()` method
3. Add the method to `PlaylistGenerationMethod` enum
4. Update the main `PlaylistGenerator` class
5. Add CLI support and documentation
6. Include tests for the new method 