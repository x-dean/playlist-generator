"""
MusicBrainz API client for metadata enrichment.

This module provides a clean interface to the MusicBrainz API
for retrieving track, artist, and album information.
"""

import logging
import time
import requests
from typing import Optional, Dict, Any, List
from urllib.parse import quote
from dataclasses import dataclass

from shared.config import get_config
from shared.exceptions import ExternalAPIError


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
class MusicBrainzArtist:
    """MusicBrainz artist information."""
    id: str
    name: str
    country: Optional[str] = None
    type: Optional[str] = None
    gender: Optional[str] = None
    tags: List[str] = None


@dataclass
class MusicBrainzAlbum:
    """MusicBrainz album information."""
    id: str
    title: str
    artist: str
    artist_id: str
    release_date: Optional[str] = None
    country: Optional[str] = None
    type: Optional[str] = None
    tags: List[str] = None


class MusicBrainzClient:
    """Client for interacting with the MusicBrainz API."""
    
    BASE_URL = "https://musicbrainz.org/ws/2"
    
    def __init__(self):
        """Initialize the MusicBrainz client."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config.external_api.musicbrainz_user_agent,
            'Accept': 'application/json'
        })
        self._rate_limit_delay = 1.0 / self.config.external_api.musicbrainz_rate_limit
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a rate-limited request to the MusicBrainz API."""
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            params = params or {}
            
            self.logger.debug(f"Making request to: {url}")
            response = self.session.get(url, params=params, timeout=30)
            
            # Rate limiting
            time.sleep(self._rate_limit_delay)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                self.logger.warning(f"Not found: {url}")
                return {}
            elif response.status_code == 429:
                self.logger.warning("Rate limit exceeded, waiting...")
                time.sleep(2)
                return self._make_request(endpoint, params)
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                raise ExternalAPIError(f"MusicBrainz API request failed: {response.status_code}")
                
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise ExternalAPIError(f"MusicBrainz API request failed: {e}")
    
    def search_track(self, title: str, artist: str = None) -> Optional[MusicBrainzTrack]:
        """Search for a track by title and optionally artist."""
        try:
            query_parts = [f'title:"{title}"']
            if artist:
                query_parts.append(f'artist:"{artist}"')
            
            query = " AND ".join(query_parts)
            
            response = self._make_request("recording", {
                'query': query,
                'fmt': 'json',
                'limit': 5
            })
            
            if not response or 'recordings' not in response:
                return None
            
            recordings = response['recordings']
            if not recordings:
                return None
            
            # Get the best match (first result)
            recording = recordings[0]
            
            # Extract artist information
            artist_name = "Unknown"
            artist_id = None
            if recording.get('artist-credit'):
                artist_credit = recording['artist-credit'][0]
                artist_name = artist_credit.get('name', 'Unknown')
                if 'artist' in artist_credit:
                    artist_id = artist_credit['artist']['id']
            
            # Extract release information
            album_name = "Unknown"
            album_id = None
            release_date = None
            track_number = None
            disc_number = None
            
            if recording.get('releases'):
                release = recording['releases'][0]
                album_name = release.get('title', 'Unknown')
                album_id = release.get('id')
                release_date = release.get('date')
                
                # Try to get track number from media
                if release.get('media'):
                    media = release['media'][0]
                    if media.get('tracks'):
                        track = media['tracks'][0]
                        track_number = track.get('number')
                        disc_number = media.get('position')
            
            # Extract tags
            tags = []
            if recording.get('tags'):
                tags = [tag['name'] for tag in recording['tags']]
            
            return MusicBrainzTrack(
                id=recording['id'],
                title=recording.get('title', title),
                artist=artist_name,
                artist_id=artist_id,
                album=album_name,
                album_id=album_id,
                release_date=release_date,
                track_number=track_number,
                disc_number=disc_number,
                duration_ms=recording.get('length'),
                tags=tags
            )
            
        except Exception as e:
            self.logger.error(f"Failed to search track '{title}': {e}")
            return None
    
    def get_artist(self, artist_id: str) -> Optional[MusicBrainzArtist]:
        """Get artist information by ID."""
        try:
            response = self._make_request(f"artist/{artist_id}", {
                'fmt': 'json',
                'inc': 'tags'
            })
            
            if not response:
                return None
            
            # Extract tags
            tags = []
            if response.get('tags'):
                tags = [tag['name'] for tag in response['tags']]
            
            return MusicBrainzArtist(
                id=response['id'],
                name=response.get('name', 'Unknown'),
                country=response.get('country'),
                type=response.get('type'),
                gender=response.get('gender'),
                tags=tags
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get artist {artist_id}: {e}")
            return None
    
    def get_album(self, album_id: str) -> Optional[MusicBrainzAlbum]:
        """Get album information by ID."""
        try:
            response = self._make_request(f"release/{album_id}", {
                'fmt': 'json',
                'inc': 'tags'
            })
            
            if not response:
                return None
            
            # Extract artist information
            artist_name = "Unknown"
            artist_id = None
            if response.get('artist-credit'):
                artist_credit = response['artist-credit'][0]
                artist_name = artist_credit.get('name', 'Unknown')
                if 'artist' in artist_credit:
                    artist_id = artist_credit['artist']['id']
            
            # Extract tags
            tags = []
            if response.get('tags'):
                tags = [tag['name'] for tag in response['tags']]
            
            return MusicBrainzAlbum(
                id=response['id'],
                title=response.get('title', 'Unknown'),
                artist=artist_name,
                artist_id=artist_id,
                release_date=response.get('date'),
                country=response.get('country'),
                type=response.get('release-group', {}).get('type'),
                tags=tags
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get album {album_id}: {e}")
            return None
    
    def search_artist(self, name: str) -> Optional[MusicBrainzArtist]:
        """Search for an artist by name."""
        try:
            response = self._make_request("artist", {
                'query': f'name:"{name}"',
                'fmt': 'json',
                'limit': 5
            })
            
            if not response or 'artists' not in response:
                return None
            
            artists = response['artists']
            if not artists:
                return None
            
            # Get the best match (first result)
            artist = artists[0]
            
            # Extract tags
            tags = []
            if artist.get('tags'):
                tags = [tag['name'] for tag in artist['tags']]
            
            return MusicBrainzArtist(
                id=artist['id'],
                name=artist.get('name', 'Unknown'),
                country=artist.get('country'),
                type=artist.get('type'),
                gender=artist.get('gender'),
                tags=tags
            )
            
        except Exception as e:
            self.logger.error(f"Failed to search artist '{name}': {e}")
            return None
    
    def search_album(self, title: str, artist: str = None) -> Optional[MusicBrainzAlbum]:
        """Search for an album by title and optionally artist."""
        try:
            query_parts = [f'title:"{title}"']
            if artist:
                query_parts.append(f'artist:"{artist}"')
            
            query = " AND ".join(query_parts)
            
            response = self._make_request("release", {
                'query': query,
                'fmt': 'json',
                'limit': 5
            })
            
            if not response or 'releases' not in response:
                return None
            
            releases = response['releases']
            if not releases:
                return None
            
            # Get the best match (first result)
            release = releases[0]
            
            # Extract artist information
            artist_name = "Unknown"
            artist_id = None
            if release.get('artist-credit'):
                artist_credit = release['artist-credit'][0]
                artist_name = artist_credit.get('name', 'Unknown')
                if 'artist' in artist_credit:
                    artist_id = artist_credit['artist']['id']
            
            # Extract tags
            tags = []
            if release.get('tags'):
                tags = [tag['name'] for tag in release['tags']]
            
            return MusicBrainzAlbum(
                id=release['id'],
                title=release.get('title', 'Unknown'),
                artist=artist_name,
                artist_id=artist_id,
                release_date=release.get('date'),
                country=release.get('country'),
                type=release.get('release-group', {}).get('type'),
                tags=tags
            )
            
        except Exception as e:
            self.logger.error(f"Failed to search album '{title}': {e}")
            return None 