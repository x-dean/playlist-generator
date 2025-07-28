#!/usr/bin/env python3
"""
Metadata enrichment for audio files using MusicBrainz and LastFM APIs.
"""

import logging
import requests
import musicbrainzngs
import time
from typing import Dict, Any, Optional, Tuple
from functools import wraps
import os

logger = logging.getLogger(__name__)


def safe_api_call(func):
    """Decorator to safely handle API calls with retries and error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"API call failed after {max_retries} attempts: {e}")
                    return None
                logger.debug(f"API call attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        return None
    return wrapper


class MetadataEnricher:
    """Handle metadata enrichment using external APIs."""
    
    def __init__(self, lastfm_api_key: Optional[str] = None):
        """Initialize the metadata enricher.
        
        Args:
            lastfm_api_key (str, optional): LastFM API key. Defaults to None.
        """
        self.lastfm_api_key = lastfm_api_key or os.getenv('LASTFM_API_KEY')
        self._setup_musicbrainz()
    
    def _setup_musicbrainz(self):
        """Setup MusicBrainz API configuration."""
        musicbrainzngs.set_useragent(
            "Playlista Music Analyzer",
            "1.0",
            "https://github.com/your-repo/playlista"
        )
        musicbrainzngs.set_rate_limit(limit_or_interval=1.0, new_requests=1)
    
    @safe_api_call
    def _musicbrainz_lookup(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Look up track information from MusicBrainz.
        
        Args:
            artist (str): Artist name.
            title (str): Track title.
            
        Returns:
            Optional[Dict[str, Any]]: MusicBrainz track information or None.
        """
        try:
            # Search for recordings
            result = musicbrainzngs.search_recordings(
                query=f"{artist} AND {title}",
                limit=1
            )
            
            if result and 'recording-list' in result and result['recording-list']:
                recording = result['recording-list'][0]
                
                # Extract basic info
                mb_data = {
                    'musicbrainz_id': recording.get('id'),
                    'title': recording.get('title'),
                    'artist': recording.get('artist-credit-phrase', artist),
                    'length': recording.get('length'),
                }
                
                # Extract release info if available
                if 'release-list' in recording and recording['release-list']:
                    release = recording['release-list'][0]
                    mb_data.update({
                        'album': release.get('title'),
                        'mb_album_id': release.get('id'),
                        'release_date': release.get('date'),
                        'country': release.get('country'),
                    })
                
                # Extract artist info
                if 'artist-credit' in recording and recording['artist-credit']:
                    artist_credit = recording['artist-credit'][0]
                    mb_data['mb_artist_id'] = artist_credit.get('artist', {}).get('id')
                
                return mb_data
            
        except Exception as e:
            logger.debug(f"MusicBrainz lookup failed for {artist} - {title}: {e}")
        
        return None
    
    @safe_api_call
    def _lastfm_lookup(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Look up track information from LastFM.
        
        Args:
            artist (str): Artist name.
            title (str): Track title.
            
        Returns:
            Optional[Dict[str, Any]]: LastFM track information or None.
        """
        if not self.lastfm_api_key:
            logger.debug("No LastFM API key provided, skipping LastFM lookup")
            return None
        
        try:
            # Search for track info
            url = "http://ws.audioscrobbler.com/2.0/"
            params = {
                'method': 'track.getInfo',
                'artist': artist,
                'track': title,
                'api_key': self.lastfm_api_key,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'track' in data and data['track']:
                track = data['track']
                
                lastfm_data = {
                    'genre_lastfm': track.get('toptags', {}).get('tag', [{}])[0].get('name'),
                    'album_lastfm': track.get('album', {}).get('title'),
                    'listeners': track.get('listeners'),
                    'playcount': track.get('playcount'),
                    'wiki': track.get('wiki', {}).get('summary'),
                }
                
                return lastfm_data
            
        except Exception as e:
            logger.debug(f"LastFM lookup failed for {artist} - {title}: {e}")
        
        return None
    
    def enrich_metadata(self, artist: str, title: str) -> Dict[str, Any]:
        """Enrich metadata using both MusicBrainz and LastFM.
        
        Args:
            artist (str): Artist name.
            title (str): Track title.
            
        Returns:
            Dict[str, Any]: Enriched metadata.
        """
        enriched_metadata = {}
        
        # MusicBrainz lookup
        mb_data = self._musicbrainz_lookup(artist, title)
        if mb_data:
            enriched_metadata.update(mb_data)
        
        # LastFM lookup
        lfm_data = self._lastfm_lookup(artist, title)
        if lfm_data:
            enriched_metadata.update(lfm_data)
        
        return enriched_metadata
    
    def enrich_metadata_for_failed_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata for a failed file.
        
        Args:
            file_info (Dict[str, Any]): File information dictionary.
            
        Returns:
            Dict[str, Any]: Enriched file information.
        """
        try:
            # Extract artist and title from file info
            metadata = file_info.get('metadata', {})
            artist = metadata.get('artist', 'Unknown Artist')
            title = metadata.get('title', 'Unknown Title')
            
            # Clean up artist and title for better API matching
            artist = self._clean_artist_name(artist)
            title = self._clean_title(title)
            
            # Enrich metadata
            enriched = self.enrich_metadata(artist, title)
            
            if enriched:
                logger.info(f"Enriched metadata for {file_info.get('filepath', 'unknown')}: {len(enriched)} fields")
                metadata.update(enriched)
                file_info['metadata'] = metadata
            
        except Exception as e:
            logger.warning(f"Failed to enrich metadata for {file_info.get('filepath', 'unknown')}: {e}")
        
        return file_info
    
    def _clean_artist_name(self, artist: str) -> str:
        """Clean artist name for better API matching.
        
        Args:
            artist (str): Raw artist name.
            
        Returns:
            str: Cleaned artist name.
        """
        # Remove common suffixes and prefixes
        suffixes = [' feat.', ' featuring', ' ft.', ' & ', ' vs ', ' vs. ']
        for suffix in suffixes:
            if suffix in artist:
                artist = artist.split(suffix)[0].strip()
        
        # Remove extra whitespace
        artist = ' '.join(artist.split())
        
        return artist
    
    def _clean_title(self, title: str) -> str:
        """Clean title for better API matching.
        
        Args:
            title (str): Raw title.
            
        Returns:
            str: Cleaned title.
        """
        # Remove common prefixes/suffixes
        prefixes = ['[', '(']
        suffixes = [']', ')']
        
        for prefix in prefixes:
            if title.startswith(prefix):
                title = title[1:]
        
        for suffix in suffixes:
            if title.endswith(suffix):
                title = title[:-1]
        
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        return title 