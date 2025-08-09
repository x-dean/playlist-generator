"""
Enhanced External APIs for Playlist Generator Simple.
This module provides clients for multiple music metadata APIs with unified logging.
"""

import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import time
import logging
import requests
import hashlib
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import musicbrainzngs for official MusicBrainz API
try:
    import musicbrainzngs
    MUSICBRAINZ_AVAILABLE = True
except ImportError:
    MUSICBRAINZ_AVAILABLE = False
    logging.warning("musicbrainzngs not available - MusicBrainz API disabled")

# Import universal logging
from .logging_setup import get_logger, log_universal, log_api_call, log_extracted_fields
from .database import get_db_manager

logger = get_logger(__name__)


@dataclass
class TrackMetadata:
    """Unified track metadata structure for all APIs."""
    # Basic info
    title: str
    artist: str
    album: Optional[str] = None
    release_date: Optional[str] = None
    track_number: Optional[int] = None
    disc_number: Optional[int] = None
    duration_ms: Optional[int] = None
    
    # IDs
    musicbrainz_id: Optional[str] = None
    musicbrainz_artist_id: Optional[str] = None
    musicbrainz_album_id: Optional[str] = None
    discogs_id: Optional[str] = None
    spotify_id: Optional[str] = None
    
    # Tags and genres
    tags: List[str] = None
    genres: List[str] = None
    
    # Statistics
    play_count: Optional[int] = None
    listeners: Optional[int] = None
    rating: Optional[float] = None
    popularity: Optional[float] = None
    
    # URLs
    url: Optional[str] = None
    image_url: Optional[str] = None
    
    # Source tracking
    sources: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.genres is None:
            self.genres = []
        if self.sources is None:
            self.sources = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'title': self.title,
            'artist': self.artist,
            'album': self.album,
            'release_date': self.release_date,
            'track_number': self.track_number,
            'disc_number': self.disc_number,
            'duration_ms': self.duration_ms,
            'musicbrainz_id': self.musicbrainz_id,
            'musicbrainz_artist_id': self.musicbrainz_artist_id,
            'musicbrainz_album_id': self.musicbrainz_album_id,
            'discogs_id': self.discogs_id,
            'spotify_id': self.spotify_id,
            'tags': self.tags,
            'genres': self.genres,
            'play_count': self.play_count,
            'listeners': self.listeners,
            'rating': self.rating,
            'popularity': self.popularity,
            'url': self.url,
            'image_url': self.image_url,
            'sources': self.sources
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackMetadata':
        """Create from dictionary."""
        return cls(**data)
    
    def merge(self, other: 'TrackMetadata') -> 'TrackMetadata':
        """Merge with another TrackMetadata object."""
        # Merge tags and genres (remove duplicates)
        all_tags = list(set(self.tags + other.tags))
        all_genres = list(set(self.genres + other.genres))
        all_sources = list(set(self.sources + other.sources))
        
        # Use the most complete data
        return TrackMetadata(
            title=other.title if other.title else self.title,
            artist=other.artist if other.artist else self.artist,
            album=other.album if other.album else self.album,
            release_date=other.release_date if other.release_date else self.release_date,
            track_number=other.track_number if other.track_number else self.track_number,
            disc_number=other.disc_number if other.disc_number else self.disc_number,
            duration_ms=other.duration_ms if other.duration_ms else self.duration_ms,
            musicbrainz_id=other.musicbrainz_id if other.musicbrainz_id else self.musicbrainz_id,
            musicbrainz_artist_id=other.musicbrainz_artist_id if other.musicbrainz_artist_id else self.musicbrainz_artist_id,
            musicbrainz_album_id=other.musicbrainz_album_id if other.musicbrainz_album_id else self.musicbrainz_album_id,
            discogs_id=other.discogs_id if other.discogs_id else self.discogs_id,
            spotify_id=other.spotify_id if other.spotify_id else self.spotify_id,
            tags=all_tags,
            genres=all_genres,
            play_count=other.play_count if other.play_count else self.play_count,
            listeners=other.listeners if other.listeners else self.listeners,
            rating=other.rating if other.rating else self.rating,
            popularity=other.popularity if other.popularity else self.popularity,
            url=other.url if other.url else self.url,
            image_url=other.image_url if other.image_url else self.image_url,
            sources=all_sources
        )


class BaseAPIClient(ABC):
    """Base class for all API clients with unified logging."""
    
    def __init__(self, api_name: str, rate_limit: float = 1.0):
        """
        Initialize base API client.
        
        Args:
            api_name: Name of the API for logging
            rate_limit: Requests per second
        """
        self.api_name = api_name
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.db_manager = get_db_manager()
        
        log_universal('INFO', f'{api_name} API', f'Client initialized (rate limit: {rate_limit}/s)')
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.rate_limit):
            sleep_time = (1.0 / self.rate_limit) - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, title: str, artist: str) -> str:
        """Generate cache key for API responses."""
        normalized_title = title.lower().strip()
        normalized_artist = artist.lower().strip() if artist else ""
        search_string = f"{self.api_name.lower()}:{normalized_title}:{normalized_artist}"
        return hashlib.md5(search_string.encode()).hexdigest()
    
    def _log_api_call(self, method: str, query: str, success: bool, details: str = "", 
                      duration: float = 0, failure_type: str = None):
        """Unified API call logging."""
        log_api_call(
            self.api_name, method, query, 
            success=success, details=details, 
            duration=duration, failure_type=failure_type
        )
    
    @abstractmethod
    def search_track(self, title: str, artist: str = None) -> Optional[TrackMetadata]:
        """Search for track metadata."""
        pass


class MusicBrainzClient(BaseAPIClient):
    """Client for interacting with the MusicBrainz API."""
    
    def __init__(self, user_agent: str = None, rate_limit: float = 1.0):
        """Initialize MusicBrainz client."""
        super().__init__("MusicBrainz", rate_limit)
        
        if not MUSICBRAINZ_AVAILABLE:
            log_universal('WARNING', 'MusicBrainz API', 'MusicBrainz library not available')
            return
            
        # Configure MusicBrainz
        musicbrainzngs.set_useragent(
            user_agent or "Playlista/1.0",
            "1.0",
            "https://github.com/playlista"
        )
        
        # Set rate limiting
        musicbrainzngs.set_rate_limit(limit_or_interval=1.0, new_requests=1)
        
        log_universal('INFO', 'MusicBrainz API', 'Client configured successfully')
    
    def search_track(self, title: str, artist: str = None) -> Optional[TrackMetadata]:
        """
        Search for a track by title and artist using MusicBrainz API.
        
        Args:
            title: Track title
            artist: Artist name (optional)
            
        Returns:
            TrackMetadata object or None if not found
        """
        if not MUSICBRAINZ_AVAILABLE:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(title, artist)
        cached_result = self.db_manager.get_cache(cache_key)
        
        if cached_result:
            log_universal('DEBUG', 'MusicBrainz API', f'Using cached result for: {title} by {artist}')
            if cached_result is not None:
                return TrackMetadata.from_dict(cached_result)
            return None
        
        try:
            start_time = time.time()
            duration = None
            
            # Build search query
            query_parts = [f'title:"{title}"']
            if artist:
                query_parts.append(f'artist:"{artist}"')
            
            query = ' AND '.join(query_parts)
            
            log_universal('DEBUG', 'MusicBrainz API', f'Searching for track: {title} by {artist or "Unknown"} - query: {query}')
            
            # Search for recordings
            result = musicbrainzngs.search_recordings(
                query=query,
                limit=1
            )
            
            duration = time.time() - start_time
            log_api_call('MusicBrainz', 'search_recordings', {'query': query}, duration, 'success')
            
            if not result or 'recording-list' not in result:
                self._log_api_call('search', f"'{title}' by '{artist or 'Unknown'}'", 
                                 success=False, details='No data returned', 
                                 duration=duration, failure_type='no_data')
                # Cache negative results for shorter time
                self.db_manager.save_cache(cache_key, None, expires_hours=1)
                return None
            
            recordings = result['recording-list']
            if not recordings:
                self._log_api_call('search', f"'{title}' by '{artist or 'Unknown'}'", 
                                 success=False, details='No recordings found', 
                                 duration=duration, failure_type='no_recordings')
                # Cache negative results for shorter time
                self.db_manager.save_cache(cache_key, None, expires_hours=1)
                return None
            
            recording = recordings[0]
            
            log_universal('DEBUG', 'MusicBrainz API', f'Found recording: {recording.get("title", title)} (ID: {recording.get("id", "Unknown")})')
            
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
            
            track_metadata = TrackMetadata(
                title=track_title,
                artist=artist_name,
                album=album_name,
                release_date=release_date,
                track_number=track_number,
                disc_number=disc_number,
                duration_ms=duration_ms,
                musicbrainz_id=track_id,
                musicbrainz_artist_id=artist_id,
                musicbrainz_album_id=album_id,
                tags=tags,
                sources=['musicbrainz']
            )
            
            # Log extracted fields
            extracted_fields = {
                'title': track_metadata.title,
                'artist': track_metadata.artist,
                'album': track_metadata.album,
                'release_date': track_metadata.release_date,
                'track_number': track_metadata.track_number,
                'disc_number': track_metadata.disc_number,
                'duration_ms': track_metadata.duration_ms,
                'musicbrainz_id': track_metadata.musicbrainz_id,
                'musicbrainz_artist_id': track_metadata.musicbrainz_artist_id,
                'musicbrainz_album_id': track_metadata.musicbrainz_album_id,
                'tags': track_metadata.tags
            }
            log_extracted_fields('MusicBrainz', f"{title} by {artist or 'Unknown'}", extracted_fields)
            
            self._log_api_call('search', f"'{track_metadata.artist}' - '{track_metadata.title}'", 
                             success=True, details=f"found {len(track_metadata.tags)} tags", 
                             duration=duration)
            
            # Cache successful results for 24 hours
            self.db_manager.save_cache(cache_key, track_metadata.to_dict(), expires_hours=24)
            
            return track_metadata
            
        except Exception as e:
            # Calculate duration if not already calculated
            if duration is None:
                duration = time.time() - start_time
            self._log_api_call('search', f"'{title}' by '{artist}'", 
                             success=False, details=f"Error: {e}", 
                             duration=duration, failure_type='network')
            # Cache errors for shorter time
            self.db_manager.save_cache(cache_key, None, expires_hours=1)
            return None


class LastFMClient(BaseAPIClient):
    """Client for interacting with the Last.fm API."""
    
    BASE_URL = "https://ws.audioscrobbler.com/2.0"
    
    def __init__(self, api_key: str = None, rate_limit: float = 2.0):
        """Initialize Last.fm client."""
        super().__init__("LastFM", rate_limit)
        
        self.api_key = api_key or os.getenv('LASTFM_API_KEY')
        if not self.api_key:
            log_universal('WARNING', 'LastFM API', 'No Last.fm API key provided')
            self.api_key = None
            return
        
        log_universal('INFO', 'LastFM API', 'Client configured successfully')
    
    def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a rate-limited request to Last.fm API."""
        if not self.api_key:
            return None
        
        self._rate_limit()
        
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
            
            if response.status_code == 200:
                return response.json()
            else:
                log_universal('ERROR', 'LastFM API', f'HTTP {response.status_code}: {response.text}')
                return None
                
        except Exception as e:
            log_universal('ERROR', 'LastFM API', f'Request failed: {e}')
            return None
    
    def search_track(self, title: str, artist: str = None) -> Optional[TrackMetadata]:
        """
        Get track information from Last.fm.
        
        Args:
            title: Track name
            artist: Artist name
            
        Returns:
            TrackMetadata object or None if not found
        """
        if not self.api_key:
            log_universal('DEBUG', 'LastFM API', f'No API key available for: {title} by {artist}')
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(title, artist)
        cached_result = self.db_manager.get_cache(cache_key)
        
        if cached_result:
            log_universal('DEBUG', 'LastFM API', f'Using cached result for: {title} by {artist}')
            if cached_result is not None:
                return TrackMetadata.from_dict(cached_result)
            return None
        
        try:
            start_time = time.time()
            duration = None
            
            result = self._make_request('track.getInfo', {
                'track': title,
                'artist': artist
            })
            
            duration = time.time() - start_time
            
            if not result or 'track' not in result:
                self._log_api_call('get_track_info', f"'{title}' by '{artist}'", 
                                 success=False, details='No data returned', 
                                 duration=duration, failure_type='no_data')
                # Cache negative results for shorter time
                self.db_manager.save_cache(cache_key, None, expires_hours=1)
                return None
            
            track_data = result['track']
            
            # Extract basic info
            name = track_data.get('name', title)
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
            
            track_metadata = TrackMetadata(
                title=name,
                artist=artist_name,
                play_count=play_count,
                listeners=listeners,
                tags=tags,
                rating=rating,
                url=url,
                sources=['lastfm']
            )
            
            # Log extracted fields
            extracted_fields = {
                'title': track_metadata.title,
                'artist': track_metadata.artist,
                'play_count': track_metadata.play_count,
                'listeners': track_metadata.listeners,
                'tags': track_metadata.tags,
                'rating': track_metadata.rating,
                'url': track_metadata.url
            }
            log_extracted_fields('LastFM', track_metadata.title, track_metadata.artist, extracted_fields)
            
            # Check if we actually found useful data
            if len(track_metadata.tags) == 0 and not track_metadata.play_count and not track_metadata.listeners:
                # No useful data found
                self._log_api_call('get_track_info', f"'{track_metadata.artist}' - '{track_metadata.title}'", 
                                 success=False, details='No useful data found', 
                                 duration=duration, failure_type='no_data')
                # Cache negative results for shorter time
                self.db_manager.save_cache(cache_key, None, expires_hours=1)
                return None
            else:
                # Found useful data
                self._log_api_call('get_track_info', f"'{track_metadata.artist}' - '{track_metadata.title}'", 
                                 success=True, details=f"found {len(track_metadata.tags)} tags", 
                                 duration=duration)
                
                # Cache successful results for 24 hours
                self.db_manager.save_cache(cache_key, track_metadata.to_dict(), expires_hours=24)
                
                return track_metadata
            
        except Exception as e:
            # Calculate duration if not already calculated
            if duration is None:
                duration = time.time() - start_time
            self._log_api_call('get_track_info', f"'{title}' by '{artist}'", 
                             success=False, details=f"Error: {e}", 
                             duration=duration, failure_type='network')
            # Cache errors for shorter time
            self.db_manager.save_cache(cache_key, None, expires_hours=1)
            return None


class DiscogsClient(BaseAPIClient):
    """Client for interacting with the Discogs API."""
    
    BASE_URL = "https://api.discogs.com"
    
    def __init__(self, api_key: str = None, rate_limit: float = 1.0):
        """Initialize Discogs client."""
        super().__init__("Discogs", rate_limit)
        
        self.api_key = api_key or os.getenv('DISCOGS_API_KEY')
        self.user_token = os.getenv('DISCOGS_USER_TOKEN')
        
        if not self.api_key:
            log_universal('WARNING', 'Discogs API', 'No Discogs API key provided')
            return
        
        log_universal('INFO', 'Discogs API', 'Client configured successfully')
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a rate-limited request to Discogs API."""
        if not self.api_key:
            return None
        
        self._rate_limit()
        
        # Prepare headers
        headers = {
            'User-Agent': 'Playlista/1.0',
            'Authorization': f'Discogs token={self.user_token}' if self.user_token else None
        }
        
        # Prepare request
        if params is None:
            params = {}
        
        params['key'] = self.api_key
        params['secret'] = self.api_key  # Discogs uses the same key for both
        
        try:
            response = requests.get(f"{self.BASE_URL}{endpoint}", 
                                 params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                log_universal('ERROR', 'Discogs API', f'HTTP {response.status_code}: {response.text}')
                return None
                
        except Exception as e:
            log_universal('ERROR', 'Discogs API', f'Request failed: {e}')
            return None
    
    def search_track(self, title: str, artist: str = None) -> Optional[TrackMetadata]:
        """
        Search for track information from Discogs.
        
        Args:
            title: Track name
            artist: Artist name
            
        Returns:
            TrackMetadata object or None if not found
        """
        if not self.api_key:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(title, artist)
        cached_result = self.db_manager.get_cache(cache_key)
        
        if cached_result:
            log_universal('DEBUG', 'Discogs API', f'Using cached result for: {title} by {artist}')
            if cached_result is not None:
                return TrackMetadata.from_dict(cached_result)
            return None
        
        try:
            start_time = time.time()
            duration = None
            
            # Build search query
            query = f"{title}"
            if artist:
                query = f"{artist} {title}"
            
            result = self._make_request('/database/search', {
                'q': query,
                'type': 'release',
                'per_page': 1
            })
            
            duration = time.time() - start_time
            
            if not result or 'results' not in result or not result['results']:
                self._log_api_call('search', f"'{title}' by '{artist}'", 
                                 success=False, details='No data returned', 
                                 duration=duration, failure_type='no_data')
                # Cache negative results for shorter time
                self.db_manager.save_cache(cache_key, None, expires_hours=1)
                return None
            
            release = result['results'][0]
            
            # Extract basic info
            track_title = title
            artist_name = artist or 'Unknown Artist'
            album_name = release.get('title', '')
            release_date = release.get('year')
            discogs_id = str(release.get('id', ''))
            
            # Extract genres and styles
            genres = release.get('genre', [])
            styles = release.get('style', [])
            tags = genres + styles
            
            # Extract image URL
            image_url = None
            if 'cover_image' in release:
                image_url = release['cover_image']
            
            track_metadata = TrackMetadata(
                title=track_title,
                artist=artist_name,
                album=album_name,
                release_date=str(release_date) if release_date else None,
                discogs_id=discogs_id,
                tags=tags,
                genres=genres,
                image_url=image_url,
                sources=['discogs']
            )
            
            # Log extracted fields
            extracted_fields = {
                'title': track_metadata.title,
                'artist': track_metadata.artist,
                'album': track_metadata.album,
                'release_date': track_metadata.release_date,
                'discogs_id': track_metadata.discogs_id,
                'tags': track_metadata.tags,
                'genres': track_metadata.genres,
                'image_url': track_metadata.image_url
            }
            log_extracted_fields('Discogs', track_metadata.title, track_metadata.artist, extracted_fields)
            
            self._log_api_call('search', f"'{track_metadata.artist}' - '{track_metadata.title}'", 
                             success=True, details=f"found {len(track_metadata.tags)} tags", 
                             duration=duration)
            
            # Cache successful results for 24 hours
            self.db_manager.save_cache(cache_key, track_metadata.to_dict(), expires_hours=24)
            
            return track_metadata
            
        except Exception as e:
            # Calculate duration if not already calculated
            if duration is None:
                duration = time.time() - start_time
            self._log_api_call('search', f"'{title}' by '{artist}'", 
                             success=False, details=f"Error: {e}", 
                             duration=duration, failure_type='network')
            # Cache errors for shorter time
            self.db_manager.save_cache(cache_key, None, expires_hours=1)
            return None


class SpotifyClient(BaseAPIClient):
    """Client for interacting with the Spotify API."""
    
    BASE_URL = "https://api.spotify.com/v1"
    
    def __init__(self, client_id: str = None, client_secret: str = None, rate_limit: float = 1.0):
        """Initialize Spotify client."""
        super().__init__("Spotify", rate_limit)
        
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.access_token = None
        self.token_expires = 0
        
        if not self.client_id or not self.client_secret:
            log_universal('WARNING', 'Spotify API', 'No Spotify API credentials provided')
            return
        
        log_universal('INFO', 'Spotify API', 'Client configured successfully')
    
    def _get_access_token(self) -> bool:
        """Get Spotify access token."""
        if self.access_token and time.time() < self.token_expires:
            return True
        
        try:
            auth_url = "https://accounts.spotify.com/api/token"
            auth_response = requests.post(auth_url, {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
            })
            
            if auth_response.status_code == 200:
                token_data = auth_response.json()
                self.access_token = token_data['access_token']
                self.token_expires = time.time() + token_data['expires_in'] - 60  # 1 minute buffer
                return True
            else:
                log_universal('ERROR', 'Spotify API', f'Authentication failed: {auth_response.status_code}')
                return False
                
        except Exception as e:
            log_universal('ERROR', 'Spotify API', f'Authentication error: {e}')
            return False
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a rate-limited request to Spotify API."""
        if not self._get_access_token():
            return None
        
        self._rate_limit()
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        
        try:
            response = requests.get(f"{self.BASE_URL}{endpoint}", 
                                 params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                log_universal('ERROR', 'Spotify API', f'HTTP {response.status_code}: {response.text}')
                return None
                
        except Exception as e:
            log_universal('ERROR', 'Spotify API', f'Request failed: {e}')
            return None
    
    def search_track(self, title: str, artist: str = None) -> Optional[TrackMetadata]:
        """
        Search for track information from Spotify.
        
        Args:
            title: Track name
            artist: Artist name
            
        Returns:
            TrackMetadata object or None if not found
        """
        if not self.client_id or not self.client_secret:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(title, artist)
        cached_result = self.db_manager.get_cache(cache_key)
        
        if cached_result:
            log_universal('DEBUG', 'Spotify API', f'Using cached result for: {title} by {artist}')
            if cached_result is not None:
                return TrackMetadata.from_dict(cached_result)
            return None
        
        try:
            start_time = time.time()
            duration = None
            
            # Build search query
            query = f"{title}"
            if artist:
                query = f"{artist} {title}"
            
            result = self._make_request('/search', {
                'q': query,
                'type': 'track',
                'limit': 1
            })
            
            duration = time.time() - start_time
            
            if not result or 'tracks' not in result or not result['tracks']['items']:
                self._log_api_call('search', f"'{title}' by '{artist}'", 
                                 success=False, details='No data returned', 
                                 duration=duration, failure_type='no_data')
                # Cache negative results for shorter time
                self.db_manager.save_cache(cache_key, None, expires_hours=1)
                return None
            
            track_data = result['tracks']['items'][0]
            
            # Extract basic info
            track_title = track_data.get('name', title)
            artist_name = track_data.get('artists', [{}])[0].get('name', artist or 'Unknown Artist')
            album_name = track_data.get('album', {}).get('name', '')
            spotify_id = track_data.get('id', '')
            duration_ms = track_data.get('duration_ms')
            popularity = track_data.get('popularity')
            
            # Extract genres (from album)
            genres = []
            if 'album' in track_data and 'genres' in track_data['album']:
                genres = track_data['album']['genres']
            
            # Extract image URL
            image_url = None
            if 'album' in track_data and 'images' in track_data['album']:
                images = track_data['album']['images']
                if images:
                    image_url = images[0].get('url')
            
            # Extract release date
            release_date = None
            if 'album' in track_data:
                release_date = track_data['album'].get('release_date')
            
            track_metadata = TrackMetadata(
                title=track_title,
                artist=artist_name,
                album=album_name,
                release_date=release_date,
                duration_ms=duration_ms,
                spotify_id=spotify_id,
                genres=genres,
                popularity=popularity,
                image_url=image_url,
                sources=['spotify']
            )
            
            # Log extracted fields
            extracted_fields = {
                'title': track_metadata.title,
                'artist': track_metadata.artist,
                'album': track_metadata.album,
                'release_date': track_metadata.release_date,
                'duration_ms': track_metadata.duration_ms,
                'spotify_id': track_metadata.spotify_id,
                'genres': track_metadata.genres,
                'popularity': track_metadata.popularity,
                'image_url': track_metadata.image_url
            }
            log_extracted_fields('Spotify', track_metadata.title, track_metadata.artist, extracted_fields)
            
            self._log_api_call('search', f"'{track_metadata.artist}' - '{track_metadata.title}'", 
                             success=True, details=f"found {len(track_metadata.genres)} genres", 
                             duration=duration)
            
            # Cache successful results for 24 hours
            self.db_manager.save_cache(cache_key, track_metadata.to_dict(), expires_hours=24)
            
            return track_metadata
            
        except Exception as e:
            # Calculate duration if not already calculated
            if duration is None:
                duration = time.time() - start_time
            self._log_api_call('search', f"'{title}' by '{artist}'", 
                             success=False, details=f"Error: {e}", 
                             duration=duration, failure_type='network')
            # Cache errors for shorter time
            self.db_manager.save_cache(cache_key, None, expires_hours=1)
            return None


class EnhancedMetadataEnrichmentService:
    """
    Enhanced service for enriching metadata using multiple external APIs.
    
    Combines MusicBrainz, Last.fm, Discogs, and Spotify data to provide
    comprehensive metadata enrichment with unified logging.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the enhanced metadata enrichment service.
        
        Args:
            config: Configuration dictionary (uses global config if None)
        """
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_external_api_config()
        
        self.config = config
        
        # Initialize clients based on configuration
        self.clients = {}
        
        # MusicBrainz
        if config.get('MUSICBRAINZ_ENABLED', True):
            try:
                self.clients['musicbrainz'] = MusicBrainzClient(
                    user_agent=config.get('MUSICBRAINZ_USER_AGENT', 'Playlista/1.0'),
                    rate_limit=config.get('MUSICBRAINZ_RATE_LIMIT', 1.0)
                )
                log_universal('INFO', 'Enrichment', 'MusicBrainz client initialized')
            except Exception as e:
                log_universal('WARNING', 'Enrichment', f'MusicBrainz client failed: {e}')
        
        # Last.fm
        if config.get('LASTFM_ENABLED', True):
            try:
                lastfm_client = LastFMClient(
                    api_key=config.get('LASTFM_API_KEY'),
                    rate_limit=config.get('LASTFM_RATE_LIMIT', 2.0)
                )
                if lastfm_client.api_key:
                    self.clients['lastfm'] = lastfm_client
                    log_universal('INFO', 'Enrichment', 'Last.fm client initialized')
                else:
                    log_universal('WARNING', 'Enrichment', 'Last.fm client not initialized - no API key')
            except Exception as e:
                log_universal('WARNING', 'Enrichment', f'Last.fm client failed: {e}')
        
        # Discogs
        if config.get('DISCOGS_ENABLED', False):
            try:
                self.clients['discogs'] = DiscogsClient(
                    api_key=config.get('DISCOGS_API_KEY'),
                    rate_limit=config.get('DISCOGS_RATE_LIMIT', 1.0)
                )
                log_universal('INFO', 'Enrichment', 'Discogs client initialized')
            except Exception as e:
                log_universal('WARNING', 'Enrichment', f'Discogs client failed: {e}')
        
        # Spotify
        if config.get('SPOTIFY_ENABLED', False):
            try:
                self.clients['spotify'] = SpotifyClient(
                    client_id=config.get('SPOTIFY_CLIENT_ID'),
                    client_secret=config.get('SPOTIFY_CLIENT_SECRET'),
                    rate_limit=config.get('SPOTIFY_RATE_LIMIT', 1.0)
                )
                log_universal('INFO', 'Enrichment', 'Spotify client initialized')
            except Exception as e:
                log_universal('WARNING', 'Enrichment', f'Spotify client failed: {e}')
        
        log_universal('INFO', 'Enrichment', f'Initialized {len(self.clients)} API clients')
    
    def _is_radio_show_episode(self, title: str, artist: str) -> bool:
        """
        Detect if this is a radio show episode that shouldn't be enriched via external APIs.
        
        Args:
            title: Track title
            artist: Artist name
            
        Returns:
            True if this appears to be a radio show episode
        """
        if not title or not artist:
            return False
        
        title_lower = title.lower()
        artist_lower = artist.lower()
        
        # Common radio show patterns
        radio_show_indicators = [
            'episode',
            'show',
            'radio',
            'mix',
            'podcast',
            'broadcast',
            'session',
            'live',
            'state of trance',
            'asot',
            'essential mix',
            'global djmix',
            'trance around the world',
            'tatw',
            'armada',
            'di.fm',
            'siriusxm',
            'bbc radio',
            'kiss fm',
            'radio 1',
            'radio 2'
        ]
        
        # Check if title contains radio show indicators
        for indicator in radio_show_indicators:
            if indicator in title_lower:
                return True
        
        # Check if artist is a known radio show host
        radio_hosts = [
            'armin van buuren',
            'tiesto',
            'paul van dyk',
            'ferry corsten',
            'markus schulz',
            'gareth emery',
            'above & beyond',
            'deadmau5',
            'skrillex',
            'david guetta',
            'calvin harris',
            'avicii',
            'martin garrix',
            'hardwell',
            'afrojack'
        ]
        
        # Check if artist is a radio host AND title looks like an episode
        if artist_lower in radio_hosts:
            episode_indicators = ['episode', 'show', 'mix', 'live', 'broadcast']
            for indicator in episode_indicators:
                if indicator in title_lower:
                    return True
        
        return False
    
    def _get_enrichment_cache_key(self, title: str, artist: str) -> str:
        """Generate cache key for enrichment results."""
        normalized_title = title.lower().strip()
        normalized_artist = artist.lower().strip()
        enrichment_string = f"enrich:{normalized_title}:{normalized_artist}"
        return hashlib.md5(enrichment_string.encode()).hexdigest()
    
    def enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich metadata using multiple external APIs.
        
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
        
        # Debug: Log what we found
        log_universal('DEBUG', 'Enrichment', f'Title: "{title}", Artist: "{artist}"')
        log_universal('DEBUG', 'Enrichment', f'Available metadata keys: {list(metadata.keys())}')
        
        # Handle "None" string values (from failed tag extraction)
        if title == "None":
            title = ""
        if artist == "None":
            artist = ""
        
        # Detect radio show episodes and handle them appropriately
        is_radio_show = self._is_radio_show_episode(title, artist)
        if is_radio_show:
            log_universal('INFO', 'Enrichment', f'Detected radio show episode: "{title}" by "{artist}" - skipping external API enrichment')
            return enriched_metadata
        
        # Try alternative field names if primary ones are empty
        if not title:
            title = metadata.get('TIT2', metadata.get('TITLE', ''))
            log_universal('DEBUG', 'Enrichment', f'Trying alternative title: "{title}"')
        
        if not artist:
            artist = metadata.get('TPE1', metadata.get('ARTIST', ''))
            log_universal('DEBUG', 'Enrichment', f'Trying alternative artist: "{artist}"')
        
        # Try to extract from filename if still missing
        if not title or not artist:
            filename = metadata.get('filename', '')
            if filename:
                log_universal('DEBUG', 'Enrichment', f'Attempting to extract from filename: {filename}')
                # Try to parse "Artist - Title" format
                if ' - ' in filename:
                    parts = filename.split(' - ', 1)
                    if len(parts) == 2:
                        extracted_artist = parts[0].strip()
                        extracted_title = parts[1].strip()
                        # Remove file extension
                        if '.' in extracted_title:
                            extracted_title = extracted_title.rsplit('.', 1)[0]
                        
                        if not artist and extracted_artist:
                            artist = extracted_artist
                            log_universal('DEBUG', 'Enrichment', f'Extracted artist from filename: "{artist}"')
                        
                        if not title and extracted_title:
                            title = extracted_title
                            log_universal('DEBUG', 'Enrichment', f'Extracted title from filename: "{title}"')
        
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
        
        # Collect results from all available APIs
        api_results = []
        
        for api_name, client in self.clients.items():
            try:
                log_universal('INFO', 'Enrichment', f'Calling {api_name} API...')
                result = client.search_track(title, artist)
                
                if result:
                    api_results.append(result)
                    enrichment_results.append(f"{api_name}: {len(result.tags)} tags")
                    log_universal('INFO', 'Enrichment', f'{api_name} success')
                else:
                    log_universal('INFO', 'Enrichment', f'{api_name} no data')
                    
            except Exception as e:
                log_universal('WARNING', 'Enrichment', f'{api_name} failed: {e}')
        
        # Merge all results
        if api_results:
            # Start with the first result
            merged_result = api_results[0]
            
            # Merge with remaining results
            for result in api_results[1:]:
                merged_result = merged_result.merge(result)
            
            # Add merged data to enriched metadata
            enriched_metadata.update(merged_result.to_dict())
            
            # Add source tracking
            enriched_metadata['enrichment_sources'] = merged_result.sources
            
            # Log merged extracted fields
            merged_fields = {
                'title': merged_result.title,
                'artist': merged_result.artist,
                'album': merged_result.album,
                'release_date': merged_result.release_date,
                'track_number': merged_result.track_number,
                'disc_number': merged_result.disc_number,
                'duration_ms': merged_result.duration_ms,
                'musicbrainz_id': merged_result.musicbrainz_id,
                'musicbrainz_artist_id': merged_result.musicbrainz_artist_id,
                'musicbrainz_album_id': merged_result.musicbrainz_album_id,
                'discogs_id': merged_result.discogs_id,
                'spotify_id': merged_result.spotify_id,
                'tags': merged_result.tags,
                'genres': merged_result.genres,
                'play_count': merged_result.play_count,
                'listeners': merged_result.listeners,
                'rating': merged_result.rating,
                'popularity': merged_result.popularity,
                'url': merged_result.url,
                'image_url': merged_result.image_url,
                'sources': merged_result.sources
            }
            log_extracted_fields('Enrichment', merged_result.title, merged_result.artist, merged_fields)
            
            log_universal('INFO', 'Enrichment', f'Complete - {", ".join(enrichment_results)}')
        else:
            log_universal('INFO', 'Enrichment', 'No data added')
        
        # Cache enrichment results for 24 hours
        db_manager.save_cache(enrichment_cache_key, enriched_metadata, expires_hours=24)
        
        return enriched_metadata
    
    def is_available(self) -> bool:
        """Check if any external API is available."""
        return len(self.clients) > 0
    
    def get_available_apis(self) -> List[str]:
        """Get list of available API names."""
        return list(self.clients.keys())


def get_enhanced_metadata_enrichment_service() -> 'EnhancedMetadataEnrichmentService':
    """Get a configured enhanced metadata enrichment service."""
    from .config_loader import config_loader
    
    config = config_loader.get_external_api_config()
    
    return EnhancedMetadataEnrichmentService(config=config)


# Backward compatibility
def get_metadata_enrichment_service() -> 'EnhancedMetadataEnrichmentService':
    """Get a configured metadata enrichment service (backward compatibility)."""
    return get_enhanced_metadata_enrichment_service() 