# External APIs Integration

This document describes the enhanced external APIs integration for Playlist Generator Simple.

## Overview

The system now supports multiple external APIs for comprehensive metadata enrichment:

- **MusicBrainz** (free, no API key required)
- **Last.fm** (requires API key)
- **Discogs** (requires API key and user token)
- **Spotify** (requires client ID and secret)

## Configuration

### Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# Last.fm API (required for Last.fm metadata)
# Get your API key from: https://www.last.fm/api/account/create
LASTFM_API_KEY=your_lastfm_api_key_here

# Discogs API (optional for Discogs metadata)
# Get your API key from: https://www.discogs.com/settings/developers
DISCOGS_API_KEY=your_discogs_api_key_here
DISCOGS_USER_TOKEN=your_discogs_user_token_here

# Spotify API (optional for Spotify metadata)
# Get your credentials from: https://developer.spotify.com/dashboard
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
```

### Configuration File

Update `playlista.conf` to enable/disable APIs:

```conf
# External API Configuration
EXTERNAL_API_ENABLED=true

# MusicBrainz API (free, no key required)
MUSICBRAINZ_ENABLED=true
MUSICBRAINZ_USER_AGENT=playlista-simple/1.0
MUSICBRAINZ_RATE_LIMIT=1

# Last.fm API (requires API key)
LASTFM_ENABLED=true
LASTFM_API_KEY=${LASTFM_API_KEY}
LASTFM_RATE_LIMIT=2

# Discogs API (requires API key and user token)
DISCOGS_ENABLED=false
DISCOGS_API_KEY=${DISCOGS_API_KEY}
DISCOGS_USER_TOKEN=${DISCOGS_USER_TOKEN}
DISCOGS_RATE_LIMIT=1

# Spotify API (requires client ID and secret)
SPOTIFY_ENABLED=false
SPOTIFY_CLIENT_ID=${SPOTIFY_CLIENT_ID}
SPOTIFY_CLIENT_SECRET=${SPOTIFY_CLIENT_SECRET}
SPOTIFY_RATE_LIMIT=1
```

## API Features

### MusicBrainz
- **Free**: No API key required
- **Data**: Artist, album, release date, track number, MusicBrainz IDs, tags
- **Rate Limit**: 1 request/second
- **Best For**: Comprehensive metadata, MusicBrainz IDs, release information

### Last.fm
- **Cost**: Free with API key
- **Data**: Play count, listeners, tags, rating, popularity
- **Rate Limit**: 2 requests/second
- **Best For**: Popularity metrics, user-generated tags, play counts

### Discogs
- **Cost**: Free with API key
- **Data**: Genre, style, release information, images
- **Rate Limit**: 1 request/second
- **Best For**: Genre classification, release details, cover art

### Spotify
- **Cost**: Free with client credentials
- **Data**: Popularity, genres, release dates, cover art
- **Rate Limit**: 1 request/second
- **Best For**: Popularity metrics, modern genre classification

## Enhanced Tag Mapping

The system now uses an enhanced tag mapper based on [Navidrome's mappings](https://github.com/navidrome/navidrome/blob/master/resources/mappings.yaml) for comprehensive metadata extraction.

### Supported Tag Formats
- **ID3v2**: TIT2, TPE1, TALB, etc.
- **Alternative Keys**: TITLE, ARTIST, ALBUM, etc.
- **Custom TXXX Tags**: MusicBrainz IDs, ReplayGain, etc.
- **Multiple Values**: Genres, composers, etc.

### Priority System
- **Priority 1**: Essential metadata (title, artist, album)
- **Priority 2**: Important metadata (track number, year, genre)
- **Priority 3**: Additional metadata (lyrics, comments, etc.)

## Logging

All API calls use unified logging with consistent format:

```
[API] [Provider] [Method] [Query] [Success/Failure] [Details] [Duration]
```

Examples:
```
[API] MusicBrainz search "Artist - Title" success found 5 tags 0.5s
[API] LastFM get_track_info "Artist - Title" failure no_data 0.3s
```

## Caching

- **Successful Results**: Cached for 24 hours
- **Failed Results**: Cached for 1 hour
- **Enrichment Results**: Cached for 24 hours
- **Cache Keys**: MD5 hash of search parameters

## Usage

The enhanced external APIs are automatically used during analysis:

```bash
# Analyze files with external API enrichment
playlista analyze

# Check available APIs
playlista status

# Test specific API
playlista test-audio --file /path/to/file.mp3
```

## Error Handling

- **Network Errors**: Graceful fallback, cached for short time
- **API Limits**: Rate limiting with exponential backoff
- **Missing Keys**: APIs disabled, analysis continues
- **Invalid Data**: Filtered out, logged as warnings

## Performance

- **Parallel Processing**: APIs called in sequence to avoid conflicts
- **Smart Caching**: Reduces API calls by 90%+ for repeated searches
- **Rate Limiting**: Prevents API abuse and ensures compliance
- **Timeout Handling**: 10-second timeout per API call

## Troubleshooting

### Common Issues

1. **No API data returned**
   - Check API keys in `.env` file
   - Verify API is enabled in `playlista.conf`
   - Check rate limits and try again later

2. **Rate limit exceeded**
   - Reduce rate limits in configuration
   - Wait before retrying
   - Check API provider status

3. **Authentication errors**
   - Verify API keys are correct
   - Check API key permissions
   - Ensure environment variables are loaded

### Debug Mode

Enable debug logging to see detailed API interactions:

```bash
playlista analyze --log-level DEBUG
```

## Future Enhancements

- **Additional APIs**: Apple Music, YouTube Music, etc.
- **Batch Processing**: Process multiple files in single API calls
- **Advanced Caching**: Redis-based distributed caching
- **API Analytics**: Usage statistics and performance metrics 