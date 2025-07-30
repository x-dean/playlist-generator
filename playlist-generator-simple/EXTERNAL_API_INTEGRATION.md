# External API Integration for Metadata Extraction

This document describes the external API integration that has been added to the playlist generator simple version to enable metadata enrichment for tracks.

## 🎯 Overview

The external API integration provides automatic metadata enrichment using two popular music APIs:
- **MusicBrainz API**: For track, artist, and album information
- **Last.fm API**: For tags, play counts, and popularity data

## 🔧 Implementation

### Files Added/Modified

1. **`src/core/external_apis.py`** (NEW)
   - MusicBrainz client for track search and metadata retrieval
   - Last.fm client for track information and tags
   - MetadataEnrichmentService that combines both APIs
   - Rate limiting and error handling

2. **`src/core/audio_analyzer.py`** (MODIFIED)
   - Enhanced `_extract_metadata()` method to include external API enrichment
   - Added `_enrich_metadata_with_external_apis()` method
   - Automatic enrichment during analysis phase

3. **`src/core/config_loader.py`** (MODIFIED)
   - Added `get_external_api_config()` method
   - Support for external API configuration settings

4. **`requirements.txt`** (MODIFIED)
   - Added `requests>=2.25.0` for HTTP API calls

5. **`playlista.conf.example`** (NEW)
   - Example configuration file with external API settings

## 🚀 Features

### MusicBrainz Integration
- **Track Search**: Search for tracks by title and artist
- **Metadata Retrieval**: Album, release date, artist ID, album ID
- **Tags**: Genre and style tags from MusicBrainz community
- **Rate Limiting**: Respects MusicBrainz API rate limits (1 request/second)

### Last.fm Integration
- **Track Information**: Play count, listener count, popularity
- **Tags**: User-generated tags and genres
- **URLs**: Direct links to Last.fm track pages
- **Rate Limiting**: Respects Last.fm API rate limits (2 requests/second)

### Combined Enrichment
- **Automatic Merging**: Combines data from both APIs
- **Tag Deduplication**: Removes duplicate tags while preserving order
- **Fallback Strategy**: If one API fails, continues with the other
- **Error Handling**: Graceful degradation when APIs are unavailable

## ⚙️ Configuration

### External API Settings

```ini
# Enable/disable external API integration
EXTERNAL_API_ENABLED=true

# MusicBrainz API settings
MUSICBRAINZ_ENABLED=true
MUSICBRAINZ_USER_AGENT=playlista-simple/1.0
MUSICBRAINZ_RATE_LIMIT=1

# Last.fm API settings
LASTFM_ENABLED=true
LASTFM_API_KEY=9fd1f789ebdf1297e6aa1590a13d85e0
LASTFM_RATE_LIMIT=2

# Metadata enrichment settings
METADATA_ENRICHMENT_ENABLED=true
METADATA_ENRICHMENT_TIMEOUT=30
METADATA_ENRICHMENT_MAX_TAGS=15
METADATA_ENRICHMENT_RETRY_COUNT=3
```

### Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `EXTERNAL_API_ENABLED` | `true` | Enable/disable all external API functionality |
| `MUSICBRAINZ_ENABLED` | `true` | Enable MusicBrainz API integration |
| `LASTFM_ENABLED` | `true` | Enable Last.fm API integration |
| `MUSICBRAINZ_USER_AGENT` | `playlista-simple/1.0` | User agent for MusicBrainz requests |
| `MUSICBRAINZ_RATE_LIMIT` | `1` | Requests per second for MusicBrainz |
| `LASTFM_API_KEY` | `9fd1f789ebdf1297e6aa1590a13d85e0` | Last.fm API key |
| `LASTFM_RATE_LIMIT` | `2` | Requests per second for Last.fm |
| `METADATA_ENRICHMENT_TIMEOUT` | `30` | Timeout for API requests in seconds |
| `METADATA_ENRICHMENT_MAX_TAGS` | `15` | Maximum number of tags to include |
| `METADATA_ENRICHMENT_RETRY_COUNT` | `3` | Number of retry attempts for failed requests |

## 📊 Enriched Metadata Fields

When external APIs are available, the following additional fields are added to track metadata:

### MusicBrainz Data
- `musicbrainz_id`: MusicBrainz recording ID
- `artist_id`: MusicBrainz artist ID
- `album_id`: MusicBrainz album ID
- `musicbrainz_tags`: Tags from MusicBrainz community
- `year`: Release year (extracted from release date)
- `album`: Album name (if different from local metadata)

### Last.fm Data
- `lastfm_tags`: Tags from Last.fm community
- `play_count`: Number of plays on Last.fm
- `listeners`: Number of unique listeners
- `lastfm_url`: Direct link to Last.fm track page

### Combined Data
- `enriched_tags`: Merged and deduplicated tags from both sources

## 🔍 Usage

### Automatic Enrichment
Metadata enrichment happens automatically during the analysis phase:

```python
# During audio analysis, metadata is automatically enriched
analyzer = AudioAnalyzer()
result = analyzer.extract_features("track.mp3")
# result['metadata'] now contains enriched data
```

### Manual Enrichment
You can also enrich metadata manually:

```python
from core.external_apis import MetadataEnrichmentService

service = MetadataEnrichmentService()
metadata = {'title': 'Bohemian Rhapsody', 'artist': 'Queen'}
enriched = service.enrich_metadata(metadata)
```

## 🧪 Testing

A comprehensive test suite is included:

```bash
# Run the external API integration tests
python test_external_apis_simple.py
```

The test suite verifies:
- API availability and connectivity
- MusicBrainz client functionality
- Last.fm client functionality
- Metadata enrichment service
- Error handling and fallback behavior

## 🔒 Rate Limiting

Both APIs implement proper rate limiting to respect API limits:

- **MusicBrainz**: 1 request per second (as per their guidelines)
- **Last.fm**: 2 requests per second (conservative limit)

Rate limiting is handled automatically by the clients.

## 🛡️ Error Handling

The integration includes robust error handling:

- **Network Failures**: Graceful degradation when APIs are unavailable
- **Rate Limit Exceeded**: Automatic retry with exponential backoff
- **Invalid Responses**: Fallback to basic metadata when API data is invalid
- **Configuration Errors**: Default to disabled state when configuration is invalid

## 📈 Performance Impact

- **Minimal Overhead**: API calls are made only when basic metadata is available
- **Caching**: Results are cached in the database for subsequent analysis
- **Parallel Processing**: API calls don't block the main analysis pipeline
- **Timeout Protection**: Requests timeout after 30 seconds to prevent hanging

## 🔄 Integration with Existing Workflow

The external API integration seamlessly integrates with the existing analysis workflow:

1. **File Discovery**: Files are discovered as usual
2. **Basic Metadata Extraction**: Local metadata is extracted first
3. **External API Enrichment**: If basic metadata is available, external APIs are queried
4. **Feature Extraction**: Audio features are extracted as usual
5. **Database Storage**: All metadata (basic + enriched) is stored together

## 🎯 Benefits

1. **Enhanced Playlist Generation**: More metadata leads to better playlist quality
2. **Improved Tagging**: Community tags provide genre and style information
3. **Popularity Data**: Play counts help identify popular tracks
4. **Release Information**: Accurate release dates and album information
5. **Fallback Strategy**: System works even when external APIs are unavailable

## 🔮 Future Enhancements

Potential future improvements:
- Spotify API integration for additional metadata
- Discogs API for vinyl and physical release information
- Local caching of API responses to reduce API calls
- Batch processing for multiple tracks
- User-configurable API keys and endpoints 