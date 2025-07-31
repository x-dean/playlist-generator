"""
External APIs for Playlist Generator Simple.
This module provides clients for MusicBrainz and Last.fm APIs
"""

import os
import time
import logging
import requests
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


class MusicBrainzClient:
    """Client for interacting with the MusicBrainz API using official library."""
    
    def __init__(self, user_agent: str = None):
        """
        Initialize the MusicBrainz client.
        
        Args:
            user_agent: User agent string for API requests (uses config if None)
        """
        # Load configuration
        try:
            from .config_loader import config_loader
            config = config_loader.get_external_api_config()
        except ImportError:
            config = {}
        
        self.logger = get_logger(__name__)
        
        if not MUSICBRAINZ_AVAILABLE:
            log_universal('WARNING', 'MB API', 'Client not available - musicbrainzngs not installed')
            return
        
        # Configure MusicBrainz
        user_agent = user_agent or config.get('MUSICBRAINZ_USER_AGENT', 'playlista-simple/1.0')
        musicbrainzngs.set_useragent(
            'playlista-simple',
            '1.0',
            user_agent
        )
        
        log_universal('INFO', 'MB API', f'Initialized with user agent: {user_agent}')
    
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
            
        try:
            log_api_call('MB API', 'search', f"'{title}' by '{artist or 'Unknown'}'")
            
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
            
            if not result or 'recording-list' not in result:
                log_api_call('MB API', 'search', f"'{title}'", success=False, details='No data returned')
                return None
            
            recordings = result['recording-list']
            if not recordings:
                log_api_call('MB API', 'search', f"'{title}'", success=False, details='No recordings found')
                return None
            
            recording = recordings[0]
            
            # Extract basic info
            track_id = recording.get('id', '')
            track_title = recording.get('title', title)
            
            # Get artist info
            artist_name = artist or 'Unknown'
            artist_id = ''
            if 'artist-credit' in recording:
                artist_credit = recording['artist-credit'][0]
                artist_name = artist_credit.get('name', artist_name)
                artist_id = artist_credit.get('artist', {}).get('id', '')
            
            # Get release info
            album_name = 'Unknown'
            album_id = ''
            release_date = None
            track_number = None
            disc_number = None
            
            if 'release-list' in recording:
                releases = recording['release-list']
                if releases:
                    release = releases[0]
                    album_name = release.get('title', album_name)
                    album_id = release.get('id', '')
                    
                    # Get release date
                    if 'date' in release:
                        release_date = release['date']
                    
                    # Get track number
                    if 'medium-list' in release:
                        for medium in release['medium-list']:
                            if 'track-list' in medium:
                                for track in medium['track-list']:
                                    if track.get('title') == track_title:
                                        track_number = track.get('position')
                                        disc_number = medium.get('position')
                                        break
            
            # Get duration
            duration_ms = None
            if 'length' in recording:
                duration_ms = int(recording['length'])
            
            # Get tags
            tags = []
            if 'tag-list' in recording:
                tags = [tag['name'] for tag in recording['tag-list']]
            
            track = MusicBrainzTrack(
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
            
            log_api_call('MB API', 'search', f"'{track.artist}' - '{track.title}'", 
                        success=True, details=f"{len(tags)} tags")
            return track
            
        except Exception as e:
            log_api_call('MB API', 'search', f"'{title}'", success=False, details=f"Error: {e}")
            return None


class LastFMClient:
    """Client for interacting with the Last.fm API."""
    
    BASE_URL = "https://ws.audioscrobbler.com/2.0"
    
    def __init__(self, api_key: str = None, rate_limit: int = None):
        """
        Initialize the Last.fm client.
        
        Args:
            api_key: Last.fm API key (uses config if None)
            rate_limit: Requests per second (uses config if None)
        """
        # Load configuration
        try:
            from .config_loader import config_loader
            config = config_loader.get_external_api_config()
        except ImportError:
            config = {}
        
        self.logger = get_logger(__name__)
        self.session = requests.Session()
        
        # Use config values or defaults
        self.api_key = api_key or config.get('LASTFM_API_KEY') or os.getenv('LASTFM_API_KEY')
        rate_limit = rate_limit or config.get('LASTFM_RATE_LIMIT', 1)
        
        if not self.api_key:
            log_universal('WARNING', 'LF API', 'API key not configured - Last.fm API disabled')
            return
        
        self._rate_limit_delay = 1.0 / rate_limit
        
        log_universal('INFO', 'LF API', f'Initialized with rate limit: {rate_limit}/s')
    
    def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a rate-limited request to the Last.fm API.
        
        Args:
            method: API method
            params: Query parameters
            
        Returns:
            API response data or empty dict on error
        """
        if not self.api_key:
            return {}
        
        # Rate limiting
        time.sleep(self._rate_limit_delay)
        
        try:
            params = params or {}
            params.update({
                'method': method,
                'api_key': self.api_key,
                'format': 'json'
            })
            
            log_api_call('LF API', method, 'API request')
            
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            
            # Rate limiting
            time.sleep(self._rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    log_api_call('LF API', method, 'API request', success=False, details=data['message'])
                    return {}
                log_api_call('LF API', method, 'API request', success=True)
                return data
            else:
                log_api_call('LF API', method, 'API request', success=False, 
                           details=f"HTTP {response.status_code}: {response.text}")
                return {}
                
        except requests.RequestException as e:
            log_api_call('LF API', method, 'API request', success=False, details=f"Request failed: {e}")
            return {}
    
    def get_track_info(self, track: str, artist: str) -> Optional[LastFMTrack]:
        """
        Get track information from Last.fm.
        
        Args:
            track: Track name
            artist: Artist name
            
        Returns:
            LastFMTrack object or None if not found
        """
        try:
            log_api_call('LF API', 'get_track_info', f"'{track}' by '{artist}'")
            
            result = self._make_request('track.getInfo', {
                'track': track,
                'artist': artist
            })
            
            if not result or 'track' not in result:
                log_api_call('LF API', 'get_track_info', f"'{track}'", success=False, details='No data returned')
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
            if 'userplaycount' in track_data:
                rating = float(track_data.get('userplaycount', 0))
            
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
            
            log_api_call('LF API', 'get_track_info', f"'{lastfm_track.artist}' - '{lastfm_track.name}'", 
                        success=True, details=f"{len(lastfm_track.tags)} tags")
            return lastfm_track
            
        except Exception as e:
            log_api_call('LF API', 'get_track_info', f"'{track}'", success=False, details=f"Error: {e}")
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
        # Load configuration
        try:
            from .config_loader import config_loader
            config = config_loader.get_external_api_config()
        except ImportError:
            config = {}
        
        self.logger = get_logger(__name__)
        
        # Initialize clients based on configuration
        musicbrainz_enabled = musicbrainz_enabled if musicbrainz_enabled is not None else config.get('MUSICBRAINZ_ENABLED', True)
        lastfm_enabled = lastfm_enabled if lastfm_enabled is not None else config.get('LASTFM_ENABLED', True)
        
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
        
        return enriched_metadata
    
    def is_available(self) -> bool:
        """Check if any external API is available."""
        return self.musicbrainz_client is not None or self.lastfm_client is not None


def get_metadata_enrichment_service() -> 'MetadataEnrichmentService':
    """Get a configured metadata enrichment service."""
    return MetadataEnrichmentService() 