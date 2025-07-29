"""
Last.fm API client for metadata enrichment.

This module provides a clean interface to the Last.fm API
for retrieving track tags, play counts, and ratings.
"""

import logging
import time
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from shared.config import get_config
from shared.exceptions import ExternalAPIError


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


@dataclass
class LastFMArtist:
    """Last.fm artist information."""
    name: str
    play_count: Optional[int] = None
    listeners: Optional[int] = None
    tags: List[str] = None
    bio: Optional[str] = None
    url: Optional[str] = None


@dataclass
class LastFMAlbum:
    """Last.fm album information."""
    name: str
    artist: str
    play_count: Optional[int] = None
    listeners: Optional[int] = None
    tags: List[str] = None
    url: Optional[str] = None


class LastFMClient:
    """Client for interacting with the Last.fm API."""
    
    BASE_URL = "https://ws.audioscrobbler.com/2.0"
    
    def __init__(self):
        """Initialize the Last.fm client."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.api_key = self.config.external_api.lastfm_api_key
        
        if not self.api_key:
            self.logger.warning("Last.fm API key not provided - functionality will be limited")
        
        self.session = requests.Session()
        self._rate_limit_delay = 1.0 / self.config.external_api.lastfm_rate_limit
    
    def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a rate-limited request to the Last.fm API."""
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
            
            self.logger.debug(f"Making Last.fm request: {method}")
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            
            # Rate limiting
            time.sleep(self._rate_limit_delay)
            
            if response.status_code == 200:
                data = response.json()
                if 'error' in data:
                    self.logger.warning(f"Last.fm API error: {data.get('message', 'Unknown error')}")
                    return {}
                return data
            else:
                self.logger.error(f"Last.fm API request failed: {response.status_code} - {response.text}")
                return {}
                
        except requests.RequestException as e:
            self.logger.error(f"Last.fm request failed: {e}")
            return {}
    
    def get_track_info(self, track: str, artist: str) -> Optional[LastFMTrack]:
        """Get track information from Last.fm."""
        try:
            response = self._make_request('track.getInfo', {
                'track': track,
                'artist': artist
            })
            
            if not response or 'track' not in response:
                return None
            
            track_data = response['track']
            
            # Extract tags
            tags = []
            if 'toptags' in track_data and 'tag' in track_data['toptags']:
                tags = [tag['name'] for tag in track_data['toptags']['tag']]
            
            # Extract play count and listeners
            play_count = None
            listeners = None
            if 'stats' in track_data:
                stats = track_data['stats']
                play_count = int(stats.get('playcount', 0)) if stats.get('playcount') else None
                listeners = int(stats.get('listeners', 0)) if stats.get('listeners') else None
            
            return LastFMTrack(
                name=track_data.get('name', track),
                artist=track_data.get('artist', {}).get('name', artist),
                play_count=play_count,
                listeners=listeners,
                tags=tags,
                url=track_data.get('url')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get track info for '{track}' by '{artist}': {e}")
            return None
    
    def get_artist_info(self, artist: str) -> Optional[LastFMArtist]:
        """Get artist information from Last.fm."""
        try:
            response = self._make_request('artist.getInfo', {
                'artist': artist
            })
            
            if not response or 'artist' not in response:
                return None
            
            artist_data = response['artist']
            
            # Extract tags
            tags = []
            if 'tags' in artist_data and 'tag' in artist_data['tags']:
                tags = [tag['name'] for tag in artist_data['tags']['tag']]
            
            # Extract bio
            bio = None
            if 'bio' in artist_data and 'content' in artist_data['bio']:
                bio = artist_data['bio']['content']
            
            # Extract stats
            play_count = None
            listeners = None
            if 'stats' in artist_data:
                stats = artist_data['stats']
                play_count = int(stats.get('playcount', 0)) if stats.get('playcount') else None
                listeners = int(stats.get('listeners', 0)) if stats.get('listeners') else None
            
            return LastFMArtist(
                name=artist_data.get('name', artist),
                play_count=play_count,
                listeners=listeners,
                tags=tags,
                bio=bio,
                url=artist_data.get('url')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get artist info for '{artist}': {e}")
            return None
    
    def get_album_info(self, album: str, artist: str) -> Optional[LastFMAlbum]:
        """Get album information from Last.fm."""
        try:
            response = self._make_request('album.getInfo', {
                'album': album,
                'artist': artist
            })
            
            if not response or 'album' not in response:
                return None
            
            album_data = response['album']
            
            # Extract tags
            tags = []
            if 'tags' in album_data and 'tag' in album_data['tags']:
                tags = [tag['name'] for tag in album_data['tags']['tag']]
            
            # Extract stats
            play_count = None
            listeners = None
            if 'stats' in album_data:
                stats = album_data['stats']
                play_count = int(stats.get('playcount', 0)) if stats.get('playcount') else None
                listeners = int(stats.get('listeners', 0)) if stats.get('listeners') else None
            
            return LastFMAlbum(
                name=album_data.get('name', album),
                artist=album_data.get('artist', artist),
                play_count=play_count,
                listeners=listeners,
                tags=tags,
                url=album_data.get('url')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get album info for '{album}' by '{artist}': {e}")
            return None
    
    def get_track_tags(self, track: str, artist: str) -> List[str]:
        """Get tags for a track."""
        try:
            response = self._make_request('track.getTopTags', {
                'track': track,
                'artist': artist
            })
            
            if not response or 'toptags' not in response:
                return []
            
            toptags = response['toptags']
            if 'tag' not in toptags:
                return []
            
            # Return tags with count > 1 to filter out noise
            tags = []
            for tag in toptags['tag']:
                if int(tag.get('count', 0)) > 1:
                    tags.append(tag['name'])
            
            return tags[:10]  # Limit to top 10 tags
            
        except Exception as e:
            self.logger.error(f"Failed to get tags for '{track}' by '{artist}': {e}")
            return []
    
    def get_artist_tags(self, artist: str) -> List[str]:
        """Get tags for an artist."""
        try:
            response = self._make_request('artist.getTopTags', {
                'artist': artist
            })
            
            if not response or 'toptags' not in response:
                return []
            
            toptags = response['toptags']
            if 'tag' not in toptags:
                return []
            
            # Return tags with count > 1 to filter out noise
            tags = []
            for tag in toptags['tag']:
                if int(tag.get('count', 0)) > 1:
                    tags.append(tag['name'])
            
            return tags[:10]  # Limit to top 10 tags
            
        except Exception as e:
            self.logger.error(f"Failed to get tags for artist '{artist}': {e}")
            return []
    
    def get_similar_tracks(self, track: str, artist: str, limit: int = 10) -> List[LastFMTrack]:
        """Get similar tracks."""
        try:
            response = self._make_request('track.getSimilar', {
                'track': track,
                'artist': artist,
                'limit': limit
            })
            
            if not response or 'similartracks' not in response:
                return []
            
            similartracks = response['similartracks']
            if 'track' not in similartracks:
                return []
            
            tracks = []
            for track_data in similartracks['track']:
                tracks.append(LastFMTrack(
                    name=track_data.get('name', 'Unknown'),
                    artist=track_data.get('artist', {}).get('name', 'Unknown'),
                    url=track_data.get('url')
                ))
            
            return tracks
            
        except Exception as e:
            self.logger.error(f"Failed to get similar tracks for '{track}' by '{artist}': {e}")
            return []
    
    def get_similar_artists(self, artist: str, limit: int = 10) -> List[LastFMArtist]:
        """Get similar artists."""
        try:
            response = self._make_request('artist.getSimilar', {
                'artist': artist,
                'limit': limit
            })
            
            if not response or 'similarartists' not in response:
                return []
            
            similarartists = response['similarartists']
            if 'artist' not in similarartists:
                return []
            
            artists = []
            for artist_data in similarartists['artist']:
                artists.append(LastFMArtist(
                    name=artist_data.get('name', 'Unknown'),
                    url=artist_data.get('url')
                ))
            
            return artists
            
        except Exception as e:
            self.logger.error(f"Failed to get similar artists for '{artist}': {e}")
            return [] 