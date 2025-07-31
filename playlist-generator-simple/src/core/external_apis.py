"""
External API clients for metadata extraction.

This module provides clients for MusicBrainz and Last.fm APIs
to enrich track metadata with additional information.
"""

import logging
import time
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from urllib.parse import quote

# Import local modules
try:
    from .logging_setup import get_logger
    logger = get_logger('playlista.external_apis')
except ImportError:
    # Fallback for standalone testing
    import logging
    logger = logging.getLogger('playlista.external_apis')


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
    """Client for interacting with the MusicBrainz API."""
    
    BASE_URL = "https://musicbrainz.org/ws/2"
    
    def __init__(self, user_agent: str = None, rate_limit: int = None):
        """
        Initialize the MusicBrainz client.
        
        Args:
            user_agent: User agent string for API requests (uses config if None)
            rate_limit: Requests per second (uses config if None)
        """
        # Load configuration
        try:
            from .config_loader import config_loader
            config = config_loader.get_external_api_config()
        except ImportError:
            # Fallback for standalone testing
            config = {}
        
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        
        # Use config values or defaults
        user_agent = user_agent or config.get('MUSICBRAINZ_USER_AGENT', 'playlista-simple/1.0')
        rate_limit = rate_limit or config.get('MUSICBRAINZ_RATE_LIMIT', 1)
        
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'application/json'
        })
        self._rate_limit_delay = 1.0 / rate_limit
        
        logger.info(f"Initialized MusicBrainz client (rate limit: {rate_limit}/s)")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a rate-limited request to the MusicBrainz API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response data or empty dict on error
        """
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            params = params or {}
            
            self.logger.info(f"MusicBrainz API call: {endpoint}")
            self.logger.debug(f"URL: {url}")
            self.logger.debug(f"Params: {params}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            # Rate limiting
            time.sleep(self._rate_limit_delay)
            
            if response.status_code == 200:
                self.logger.info(f"MusicBrainz API call successful: {endpoint}")
                return response.json()
            elif response.status_code == 404:
                self.logger.info(f"MusicBrainz API call: Not found - {endpoint}")
                return {}
            elif response.status_code == 429:
                self.logger.warning("MusicBrainz API rate limit exceeded, waiting...")
                time.sleep(2)
                return self._make_request(endpoint, params)
            else:
                self.logger.error(f"MusicBrainz API call failed: {response.status_code} - {response.text}")
                return {}
                
        except requests.RequestException as e:
            self.logger.error(f"MusicBrainz API request failed: {e}")
            return {}
    
    def search_track(self, title: str, artist: str = None) -> Optional[MusicBrainzTrack]:
        """
        Search for a track by title and artist.
        
        Args:
            title: Track title
            artist: Artist name (optional)
            
        Returns:
            MusicBrainzTrack object or None if not found
        """
        try:
            self.logger.info(f"MusicBrainz search: '{title}' by '{artist or 'Unknown'}'")
            
            # Build search query
            query_parts = [f'title:"{title}"']
            if artist:
                query_parts.append(f'artist:"{artist}"')
            
            query = ' AND '.join(query_parts)
            
            params = {
                'query': query,
                'fmt': 'json',
                'limit': 5
            }
            
            data = self._make_request('recording', params)
            
            if not data or 'recordings' not in data:
                self.logger.info(f"MusicBrainz search: No data returned for '{title}'")
                return None
            
            recordings = data['recordings']
            if not recordings:
                self.logger.info(f"MusicBrainz search: No recordings found for '{title}'")
                return None
            
            # Get the best match (first result)
            recording = recordings[0]
            
            # Extract artist info
            artist_name = artist or "Unknown"
            artist_id = "Unknown"
            if 'artist-credit' in recording and recording['artist-credit']:
                artist_credit = recording['artist-credit'][0]
                artist_name = artist_credit.get('name', artist_name)
                artist_id = artist_credit.get('artist', {}).get('id', artist_id)
            
            # Extract release info
            album_name = "Unknown"
            album_id = "Unknown"
            release_date = None
            if 'releases' in recording and recording['releases']:
                release = recording['releases'][0]
                album_name = release.get('title', album_name)
                album_id = release.get('id', album_id)
                release_date = release.get('date', release_date)
            
            # Extract tags
            tags = []
            if 'tags' in recording:
                tags = [tag['name'] for tag in recording['tags'][:10]]
            
            track = MusicBrainzTrack(
                id=recording.get('id', ''),
                title=recording.get('title', title),
                artist=artist_name,
                artist_id=artist_id,
                album=album_name,
                album_id=album_id,
                release_date=release_date,
                tags=tags
            )
            
            self.logger.info(f"MusicBrainz search successful: '{track.artist}' - '{track.title}' (ID: {track.id})")
            if tags:
                self.logger.debug(f"Tags: {tags}")
            if release_date:
                self.logger.debug(f"Release date: {release_date}")
            
            return track
            
        except Exception as e:
            self.logger.error(f"MusicBrainz search error: {e}")
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
            # Fallback for standalone testing
            config = {}
        
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or config.get('LASTFM_API_KEY', '9fd1f789ebdf1297e6aa1590a13d85e0')
        rate_limit = rate_limit or config.get('LASTFM_RATE_LIMIT', 2)
        
        if not self.api_key:
            self.logger.warning("Last.fm API key not provided - functionality will be limited")
        
        self.session = requests.Session()
        self._rate_limit_delay = 1.0 / rate_limit
        
        logger.info(f"Initialized Last.fm client (rate limit: {rate_limit}/s)")
    
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
            self.logger.warning("Cannot make Last.fm request - no API key provided")
            return {}
        
        try:
            params = params or {}
            params.update({
                'method': method,
                'api_key': self.api_key,
                'format': 'json'
            })
            
            self.logger.info(f"Last.fm API call: {method}")
            self.logger.debug(f"URL: {self.BASE_URL}")
            self.logger.debug(f"Params: {dict((k, v) for k, v in params.items() if k != 'api_key')}")
            
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            
            # Rate limiting
            time.sleep(self._rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    self.logger.warning(f"Last.fm API error: {data.get('message', 'Unknown error')}")
                    return {}
                self.logger.info(f"Last.fm API call successful: {method}")
                return data
            else:
                self.logger.error(f"Last.fm API call failed: {response.status_code} - {response.text}")
                return {}
                
        except requests.RequestException as e:
            self.logger.error(f"Last.fm API request failed: {e}")
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
            params = {
                'track': track,
                'artist': artist
            }
            
            data = self._make_request('track.getInfo', params)
            
            if not data or 'track' not in data:
                return None
            
            track_data = data['track']
            
            # Extract tags
            tags = []
            if 'toptags' in track_data and 'tag' in track_data['toptags']:
                tags = [tag['name'] for tag in track_data['toptags']['tag'][:10]]
            
            # Extract play count
            play_count = None
            if 'playcount' in track_data:
                try:
                    play_count = int(track_data['playcount'])
                except (ValueError, TypeError):
                    pass
            
            # Extract listeners
            listeners = None
            if 'listeners' in track_data:
                try:
                    listeners = int(track_data['listeners'])
                except (ValueError, TypeError):
                    pass
            
            lastfm_track = LastFMTrack(
                name=track_data.get('name', track),
                artist=track_data.get('artist', {}).get('name', artist),
                play_count=play_count,
                listeners=listeners,
                tags=tags,
                url=track_data.get('url')
            )
            
            self.logger.debug(f"Found Last.fm track: {lastfm_track.artist} - {lastfm_track.name}")
            return lastfm_track
            
        except Exception as e:
            self.logger.error(f"Error getting track info: {e}")
            return None
    
    def get_track_tags(self, track: str, artist: str) -> List[str]:
        """
        Get tags for a track.
        
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
            # Fallback for standalone testing
            config = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Use config values or defaults
        musicbrainz_enabled = musicbrainz_enabled if musicbrainz_enabled is not None else config.get('MUSICBRAINZ_ENABLED', True)
        lastfm_enabled = lastfm_enabled if lastfm_enabled is not None else config.get('LASTFM_ENABLED', True)
        
        # Initialize API clients
        self.musicbrainz_client = None
        if musicbrainz_enabled:
            try:
                self.musicbrainz_client = MusicBrainzClient()
                logger.info("MusicBrainz client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MusicBrainz client: {e}")
        
        self.lastfm_client = None
        if lastfm_enabled:
            try:
                self.lastfm_client = LastFMClient()
                logger.info("Last.fm client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Last.fm client: {e}")
        
        logger.info(f"Metadata enrichment service initialized")
    
    def enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich metadata using external APIs.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Enriched metadata dictionary
        """
        if not metadata:
            self.logger.info("No metadata to enrich")
            return metadata
        
        try:
            # Extract basic info
            title = metadata.get('title', '')
            artist = metadata.get('artist', '')
            
            if not title or not artist:
                self.logger.info(f"Skipping enrichment - missing title or artist: title='{title}', artist='{artist}'")
                return metadata
            
            self.logger.info(f"Starting metadata enrichment for: '{artist}' - '{title}'")
            
            enriched_metadata = metadata.copy()
            enrichment_results = []
            
            # Try MusicBrainz enrichment
            if self.musicbrainz_client:
                try:
                    self.logger.info(f"Calling MusicBrainz API for enrichment...")
                    mb_track = self.musicbrainz_client.search_track(title, artist)
                    if mb_track:
                        # Add MusicBrainz data
                        if mb_track.release_date:
                            enriched_metadata['year'] = mb_track.release_date[:4]
                            enrichment_results.append(f"year: {mb_track.release_date[:4]}")
                        
                        if mb_track.tags:
                            enriched_metadata['musicbrainz_tags'] = mb_track.tags
                            enrichment_results.append(f"musicbrainz_tags: {len(mb_track.tags)} tags")
                        
                        if mb_track.album and mb_track.album != "Unknown":
                            enriched_metadata['album'] = mb_track.album
                            enrichment_results.append(f"album: {mb_track.album}")
                        
                        enriched_metadata['musicbrainz_id'] = mb_track.id
                        enriched_metadata['artist_id'] = mb_track.artist_id
                        enriched_metadata['album_id'] = mb_track.album_id
                        enrichment_results.append(f"musicbrainz_id: {mb_track.id}")
                        
                        self.logger.info(f"MusicBrainz enrichment successful: {', '.join(enrichment_results)}")
                    else:
                        self.logger.info(f"MusicBrainz enrichment: No data found")
                except Exception as e:
                    self.logger.warning(f"MusicBrainz enrichment failed: {e}")
            else:
                self.logger.info(f"MusicBrainz client not available")
            
            # Try Last.fm enrichment
            if self.lastfm_client:
                try:
                    self.logger.info(f"Calling Last.fm API for enrichment...")
                    lf_track = self.lastfm_client.get_track_info(title, artist)
                    if lf_track:
                        # Add Last.fm data
                        if lf_track.tags:
                            enriched_metadata['lastfm_tags'] = lf_track.tags
                            enrichment_results.append(f"lastfm_tags: {len(lf_track.tags)} tags")
                        
                        if lf_track.play_count:
                            enriched_metadata['play_count'] = lf_track.play_count
                            enrichment_results.append(f"play_count: {lf_track.play_count}")
                        
                        if lf_track.listeners:
                            enriched_metadata['listeners'] = lf_track.listeners
                            enrichment_results.append(f"listeners: {lf_track.listeners}")
                        
                        enriched_metadata['lastfm_url'] = lf_track.url
                        enrichment_results.append(f"lastfm_url: {lf_track.url}")
                        
                        self.logger.info(f"Last.fm enrichment successful: {', '.join(enrichment_results)}")
                    else:
                        self.logger.info(f"Last.fm enrichment: No data found")
                except Exception as e:
                    self.logger.warning(f"Last.fm enrichment failed: {e}")
            else:
                self.logger.info(f"Last.fm client not available")
            
            # Combine tags from both sources
            all_tags = []
            if enriched_metadata.get('musicbrainz_tags'):
                all_tags.extend(enriched_metadata['musicbrainz_tags'])
            if enriched_metadata.get('lastfm_tags'):
                all_tags.extend(enriched_metadata['lastfm_tags'])
            
            if all_tags:
                # Remove duplicates while preserving order
                unique_tags = []
                for tag in all_tags:
                    if tag.lower() not in [t.lower() for t in unique_tags]:
                        unique_tags.append(tag)
                enriched_metadata['enriched_tags'] = unique_tags[:15]  # Limit to 15 tags
                self.logger.info(f"Combined {len(unique_tags)} unique tags from both APIs")
            
            self.logger.info(f"Metadata enrichment completed for: '{title}'")
            return enriched_metadata
            
        except Exception as e:
            self.logger.error(f"Error enriching metadata: {e}")
            return metadata
    
    def is_available(self) -> bool:
        """
        Check if any external APIs are available.
        
        Returns:
            True if at least one API is available
        """
        return self.musicbrainz_client is not None or self.lastfm_client is not None


# Global instances
musicbrainz_client = MusicBrainzClient()
lastfm_client = LastFMClient()
metadata_enrichment_service = MetadataEnrichmentService() 