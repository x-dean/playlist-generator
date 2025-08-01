"""
External APIs for Playlist Generator Simple.
This module provides clients for MusicBrainz and Last.fm APIs
"""

import os
import time
import logging
import requests
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import musicbrainzngs for official MusicBrainz API
try:
    import musicbrainzngs
    MUSICBRAINZ_AVAILABLE = True
except ImportError:
    MUSICBRAINZ_AVAILABLE = False
    logging.warning("musicbrainzngs not available - MusicBrainz API disabled")

# Import universal logging
from .logging_setup import get_logger, log_universal, log_api_call
from .database import get_db_manager

logger = get_logger(__name__)


@dataclass
class MusicBrainzTrack:
    """MusicBrainz track information."""
    id: str
    title: str
    artist: str
    artist_id: str
    album: str
    album_id: str
    release_date: Optional[str] = None
    track_number: Optional[int] = None
    disc_number: Optional[int] = None
    duration_ms: Optional[int] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class LastFMTrack:
    """Last.fm track information."""
    name: str
    artist: str
    play_count: Optional[int] = None
    listeners: Optional[int] = None
    tags: List[str] = None
    rating: Optional[float] = None
    url: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MusicBrainzClient:
    """Client for interacting with the MusicBrainz API."""
    
    def __init__(self, user_agent: str = None):
        """Initialize MusicBrainz client."""
        if not MUSICBRAINZ_AVAILABLE:
            log_universal('WARNING', 'MB API', 'MusicBrainz library not available')
            return
            
        # Configure MusicBrainz
        musicbrainzngs.set_useragent(
            user_agent or "Playlista/1.0",
            "1.0",
            "https://github.com/playlista"
        )
        
        # Set rate limiting
        musicbrainzngs.set_rate_limit(limit_or_interval=1.0, new_requests=1)
        
        log_universal('INFO', 'MB API', 'Client initialized')
    
    def _get_cache_key(self, title: str, artist: str) -> str:
        """Generate cache key for API responses."""
        # Normalize and hash the search parameters
        normalized_title = title.lower().strip()
        normalized_artist = artist.lower().strip() if artist else ""
        search_string = f"mb:{normalized_title}:{normalized_artist}"
        return hashlib.md5(search_string.encode()).hexdigest()
    
    def search_track(self, title: str, artist: str = None) -> Optional[MusicBrainzTrack]:
        """
        Search for a track by title and artist using MusicBrainz API.
        
        Args:
            title: Track title
            artist: Artist name (optional)
            
        Returns:
            MusicBrainzTrack object or None if not found
        """
        if not MUSICBRAINZ_AVAILABLE:
            return None
            
        # Check cache first
        db_manager = get_db_manager()
        cache_key = self._get_cache_key(title, artist)
        cached_result = db_manager.get_cache(cache_key)
        
        if cached_result:
            log_universal('DEBUG', 'MB API', f'Using cached result for: {title} by {artist}')
            return cached_result
        
        try:
            start_time = time.time()
            duration = None  # Initialize duration
            
            # Build search query
            query_parts = [f'title:"{title}"']
            if artist:
                query_parts.append(f'artist:"{artist}"')
            
            query = ' AND '.join(query_parts)
            
            # Search for recordings
            result = musicbrainzngs.search_recordings(
                query=query,
                limit=1
            )
            
            duration = time.time() - start_time
            
            if not result or 'recording-list' not in result:
                log_api_call('MusicBrainz', 'search', f"'{title}' by '{artist or 'Unknown'}'", 
                           success=False, details='No data returned', duration=duration, failure_type='no_data')
                # Cache negative results for shorter time
                db_manager.save_cache(cache_key, None, expires_hours=1)
                return None
            
            recordings = result['recording-list']
            if not recordings:
                log_api_call('MusicBrainz', 'search', f"'{title}' by '{artist or 'Unknown'}'", 
                           success=False, details='No recordings found', duration=duration, failure_type='no_data')
                # Cache negative results for shorter time
                db_manager.save_cache(cache_key, None, expires_hours=1)
                return None
            
            recording = recordings[0]
            
            # Extract track information
            track_id = recording.get('id', '')
            track_title = recording.get('title', title)
            
            # Extract artist information
            artist_name = artist or 'Unknown Artist'
            artist_id = ''
            if 'artist-credit' in recording and recording['artist-credit']:
                artist_credit = recording['artist-credit'][0]
                artist_name = artist_credit.get('name', artist_name)
                artist_id = artist_credit.get('id', '')
            
            # Extract release information
            album_name = ''
            album_id = ''
            release_date = ''
            track_number = None
            disc_number = None
            duration_ms = None
            
            if 'release-list' in recording and recording['release-list']:
                release = recording['release-list'][0]
                album_name = release.get('title', '')
                album_id = release.get('id', '')
                release_date = release.get('date', '')
                
                # Get track number from medium
                if 'medium-list' in release and release['medium-list']:
                    medium = release['medium-list'][0]
                    if 'track-list' in medium and medium['track-list']:
                        track = medium['track-list'][0]
                        track_number = track.get('number')
                        disc_number = medium.get('position')
                        duration_ms = track.get('length')
            
            # Extract tags
            tags = []
            if 'tag-list' in recording:
                tags = [tag['name'] for tag in recording['tag-list']]
            
            mb_track = MusicBrainzTrack(
                id=track_id,
                title=track_title,
                artist=artist_name,
                artist_id=artist_id,
                album=album_name,
                album_id=album_id,
                release_date=release_date,
                track_number=track_number,
                disc_number=disc_number,
                duration_ms=duration_ms,
                tags=tags
            )
            
            log_api_call('MusicBrainz', 'search', f"'{mb_track.artist}' - '{mb_track.title}'", 
                        success=True, details=f"found {len(mb_track.tags)} tags", duration=duration)
            
            # Cache successful results for 24 hours
            db_manager.save_cache(cache_key, mb_track, expires_hours=24)
            
            return mb_track
            
        except Exception as e:
            # Calculate duration if not already calculated
            if duration is None:
                duration = time.time() - start_time
            log_api_call('MusicBrainz', 'search', f"'{title}' by '{artist}'", success=False, details=f"Error: {e}", duration=duration, failure_type='network')
            # Cache errors for shorter time
            db_manager.save_cache(cache_key, None, expires_hours=1)
            return None


class LastFMClient:
    """Client for interacting with the Last.fm API."""
    
    BASE_URL = "https://ws.audioscrobbler.com/2.0"
    
    def __init__(self, api_key: str = None, rate_limit: int = None):
        """Initialize Last.fm client."""
        self.api_key = api_key or os.getenv('LASTFM_API_KEY')
        if not self.api_key:
            log_universal('WARNING', 'LF API', 'No Last.fm API key provided')
            return
        
        self.rate_limit = rate_limit or 2.0  # requests per second
        self.last_request_time = 0
        
        log_universal('INFO', 'LF API', 'Client initialized')
    
    def _get_cache_key(self, track: str, artist: str) -> str:
        """Generate cache key for API responses."""
        # Normalize and hash the search parameters
        normalized_track = track.lower().strip()
        normalized_artist = artist.lower().strip()
        search_string = f"lf:{normalized_track}:{normalized_artist}"
        return hashlib.md5(search_string.encode()).hexdigest()
    
    def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a rate-limited request to Last.fm API."""
        if not self.api_key:
            return None
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.rate_limit):
            sleep_time = (1.0 / self.rate_limit) - time_since_last
            time.sleep(sleep_time)
        
        # Prepare request
        if params is None:
            params = {}
        
        params.update({
            'method': method,
            'api_key': self.api_key,
            'format': 'json'
        })
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                log_universal('ERROR', 'LF API', f'HTTP {response.status_code}: {response.text}')
                return None
                
        except Exception as e:
            log_universal('ERROR', 'LF API', f'Request failed: {e}')
            return None
    
    def get_track_info(self, track: str, artist: str) -> Optional[LastFMTrack]:
        """
        Get track information from Last.fm.
        
        Args:
            track: Track name
            artist: Artist name
            
        Returns:
            LastFMTrack object or None if not found
        """
        if not self.api_key:
            return None
        
        # Check cache first
        db_manager = get_db_manager()
        cache_key = self._get_cache_key(track, artist)
        cached_result = db_manager.get_cache(cache_key)
        
        if cached_result:
            log_universal('DEBUG', 'LF API', f'Using cached result for: {track} by {artist}')
            return cached_result
        
        try:
            start_time = time.time()
            duration = None
            
            result = self._make_request('track.getInfo', {
                'track': track,
                'artist': artist
            })
            
            duration = time.time() - start_time
            
            if not result or 'track' not in result:
                log_api_call('LastFM', 'get_track_info', f"'{track}' by '{artist}'", 
                           success=False, details='No data returned', duration=duration, failure_type='no_data')
                # Cache negative results for shorter time
                db_manager.save_cache(cache_key, None, expires_hours=1)
                return None
            
            track_data = result['track']
            
            # Extract basic info
            name = track_data.get('name', track)
            artist_name = track_data.get('artist', {}).get('name', artist)
            
            # Extract statistics
            play_count = None
            listeners = None
            if 'stats' in track_data:
                stats = track_data['stats']
                play_count = int(stats.get('playcount', 0)) if stats.get('playcount') else None
                listeners = int(stats.get('listeners', 0)) if stats.get('listeners') else None
            
            # Extract tags
            tags = []
            if 'toptags' in track_data and 'tag' in track_data['toptags']:
                tags = [tag['name'] for tag in track_data['toptags']['tag']]
            
            # Extract rating
            rating = None
            if 'userplaycount' in track_data and track_data['userplaycount']:
                rating = float(track_data['userplaycount'])
            
            # Extract URL
            url = track_data.get('url')
            
            lastfm_track = LastFMTrack(
                name=name,
                artist=artist_name,
                play_count=play_count,
                listeners=listeners,
                tags=tags,
                rating=rating,
                url=url
            )
            
            log_api_call('LastFM', 'get_track_info', f"'{lastfm_track.artist}' - '{lastfm_track.name}'", 
                        success=True, details=f"found {len(lastfm_track.tags)} tags", duration=duration)
            
            # Cache successful results for 24 hours
            db_manager.save_cache(cache_key, lastfm_track, expires_hours=24)
            
            return lastfm_track
            
        except Exception as e:
            # Calculate duration if not already calculated
            if duration is None:
                duration = time.time() - start_time
            log_api_call('LastFM', 'get_track_info', f"'{track}' by '{artist}'", success=False, details=f"Error: {e}", duration=duration, failure_type='network')
            # Cache errors for shorter time
            db_manager.save_cache(cache_key, None, expires_hours=1)
            return None
    
    def get_track_tags(self, track: str, artist: str) -> List[str]:
        """
        Get track tags from Last.fm.
        
        Args:
            track: Track name
            artist: Artist name
            
        Returns:
            List of tag names
        """
        track_info = self.get_track_info(track, artist)
        return track_info.tags if track_info else []


class MetadataEnrichmentService:
    """
    Service for enriching metadata using external APIs.
    
    Combines MusicBrainz and Last.fm data to provide
    comprehensive metadata enrichment.
    """
    
    def __init__(self, musicbrainz_enabled: bool = None, lastfm_enabled: bool = None):
        """
        Initialize the metadata enrichment service.
        
        Args:
            musicbrainz_enabled: Enable MusicBrainz API (uses config if None)
            lastfm_enabled: Enable Last.fm API (uses config if None)
        """
        # Initialize clients based on configuration
        self.musicbrainz_client = None
        if musicbrainz_enabled:
            try:
                self.musicbrainz_client = MusicBrainzClient()
                log_universal('INFO', 'MB API', 'Client initialized')
            except Exception as e:
                log_universal('WARNING', 'MB API', f'Failed to initialize - {e}')
        
        self.lastfm_client = None
        if lastfm_enabled:
            try:
                self.lastfm_client = LastFMClient()
                log_universal('INFO', 'LF API', 'Client initialized')
            except Exception as e:
                log_universal('WARNING', 'LF API', f'Failed to initialize - {e}')
    
    def _get_enrichment_cache_key(self, title: str, artist: str) -> str:
        """Generate cache key for enrichment results."""
        # Normalize and hash the enrichment parameters
        normalized_title = title.lower().strip()
        normalized_artist = artist.lower().strip()
        enrichment_string = f"enrich:{normalized_title}:{normalized_artist}"
        return hashlib.md5(enrichment_string.encode()).hexdigest()
    
    def enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich metadata using external APIs (MusicBrainz first, then Last.fm).
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Enriched metadata dictionary
        """
        enriched_metadata = metadata.copy()
        enrichment_results = []
        
        # Extract basic info for API calls
        title = metadata.get('title', '')
        artist = metadata.get('artist', '')
        
        if not title or not artist:
            log_universal('WARNING', 'Enrichment', 'Missing title or artist')
            return enriched_metadata
        
        # Check enrichment cache first
        db_manager = get_db_manager()
        enrichment_cache_key = self._get_enrichment_cache_key(title, artist)
        cached_enrichment = db_manager.get_cache(enrichment_cache_key)
        
        if cached_enrichment:
            log_universal('DEBUG', 'Enrichment', f'Using cached enrichment for: {title} by {artist}')
            return cached_enrichment
        
        # Try MusicBrainz enrichment FIRST
        if self.musicbrainz_client:
            try:
                log_universal('INFO', 'Enrichment', 'Calling MusicBrainz API...')
                mb_track = self.musicbrainz_client.search_track(title, artist)
                
                if mb_track:
                    # Add MusicBrainz data
                    enriched_metadata['musicbrainz_id'] = mb_track.id
                    enriched_metadata['musicbrainz_artist_id'] = mb_track.artist_id
                    enriched_metadata['musicbrainz_album_id'] = mb_track.album_id
                    enriched_metadata['musicbrainz_tags'] = mb_track.tags
                    enrichment_results.append(f"MB tags: {len(mb_track.tags)}")
                    
                    # Add additional fields if not already present
                    if not enriched_metadata.get('album') and mb_track.album:
                        enriched_metadata['album'] = mb_track.album
                        enrichment_results.append(f"album: {mb_track.album}")
                    
                    if not enriched_metadata.get('release_date') and mb_track.release_date:
                        enriched_metadata['release_date'] = mb_track.release_date
                        enrichment_results.append(f"release_date: {mb_track.release_date}")
                    
                    if not enriched_metadata.get('track_number') and mb_track.track_number:
                        enriched_metadata['track_number'] = mb_track.track_number
                        enrichment_results.append(f"track_number: {mb_track.track_number}")
                    
                    log_universal('INFO', 'Enrichment', f'MusicBrainz success - {", ".join(enrichment_results)}')
                else:
                    log_universal('INFO', 'Enrichment', 'MusicBrainz no data')
                    
            except Exception as e:
                # Check if this is a programming error (like variable reference issues)
                if 'referenced before assignment' in str(e) or 'UnboundLocalError' in str(type(e).__name__):
                    log_universal('ERROR', 'Enrichment', f'MusicBrainz failed - Programming error: {e}')
                else:
                    log_universal('WARNING', 'Enrichment', f'MusicBrainz failed - {e}')
        else:
            log_universal('INFO', 'Enrichment', 'MusicBrainz not available')
        
        # Try Last.fm enrichment SECOND
        if self.lastfm_client:
            try:
                log_universal('INFO', 'Enrichment', 'Calling Last.fm API...')
                lfm_track = self.lastfm_client.get_track_info(title, artist)
                
                if lfm_track:
                    # Add Last.fm data (only if not already present from MusicBrainz)
                    if not enriched_metadata.get('lastfm_tags'):
                        enriched_metadata['lastfm_tags'] = lfm_track.tags
                        enrichment_results.append(f"LF tags: {len(lfm_track.tags)}")
                    
                    if not enriched_metadata.get('play_count') and lfm_track.play_count:
                        enriched_metadata['play_count'] = lfm_track.play_count
                        enrichment_results.append(f"play_count: {lfm_track.play_count}")
                    
                    if not enriched_metadata.get('listeners') and lfm_track.listeners:
                        enriched_metadata['listeners'] = lfm_track.listeners
                        enrichment_results.append(f"listeners: {lfm_track.listeners}")
                    
                    if not enriched_metadata.get('rating') and lfm_track.rating:
                        enriched_metadata['rating'] = lfm_track.rating
                        enrichment_results.append(f"rating: {lfm_track.rating}")
                    
                    log_universal('INFO', 'Enrichment', f'Last.fm success - {", ".join(enrichment_results)}')
                else:
                    log_universal('INFO', 'Enrichment', 'Last.fm no data')
                    
            except Exception as e:
                # Check if this is a programming error (like variable reference issues)
                if 'referenced before assignment' in str(e) or 'UnboundLocalError' in str(type(e).__name__):
                    log_universal('ERROR', 'Enrichment', f'Last.fm failed - Programming error: {e}')
                else:
                    log_universal('WARNING', 'Enrichment', f'Last.fm failed - {e}')
        else:
            log_universal('INFO', 'Enrichment', 'Last.fm not available')
        
        # Combine all tags
        all_tags = []
        if enriched_metadata.get('musicbrainz_tags'):
            all_tags.extend(enriched_metadata['musicbrainz_tags'])
        if enriched_metadata.get('lastfm_tags'):
            all_tags.extend(enriched_metadata['lastfm_tags'])
        
        if all_tags:
            enriched_metadata['all_tags'] = list(set(all_tags))  # Remove duplicates
            enrichment_results.append(f"total_tags: {len(enriched_metadata['all_tags'])}")
        
        if enrichment_results:
            log_universal('INFO', 'Enrichment', f'Complete - {", ".join(enrichment_results)}')
        else:
            log_universal('INFO', 'Enrichment', 'No data added')
        
        # Cache enrichment results for 24 hours
        db_manager.save_cache(enrichment_cache_key, enriched_metadata, expires_hours=24)
        
        return enriched_metadata
    
    def is_available(self) -> bool:
        """Check if any external API is available."""
        return self.musicbrainz_client is not None or self.lastfm_client is not None


def get_metadata_enrichment_service() -> 'MetadataEnrichmentService':
    """Get a configured metadata enrichment service."""
    from .config_loader import config_loader
    
    config = config_loader.get_external_api_config()
    
    musicbrainz_enabled = config.get('MUSICBRAINZ_ENABLED', True)
    lastfm_enabled = config.get('LASTFM_ENABLED', True)
    
    return MetadataEnrichmentService(
        musicbrainz_enabled=musicbrainz_enabled,
        lastfm_enabled=lastfm_enabled
    ) 