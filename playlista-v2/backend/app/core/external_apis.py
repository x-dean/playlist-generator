"""
External API integrations for music metadata and enrichment
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
import requests

from .logging import get_logger, LogContext, log_operation_start, log_operation_success, log_operation_error
from .config import get_settings

logger = get_logger("external_apis")
settings = get_settings()


class ExternalAPIManager:
    """Manages external API integrations for music data enrichment"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Playlista-v2/2.0.0 (High-Performance Music Analysis)'
        })
        logger.info("ExternalAPIManager initialized")
    
    async def enrich_track_metadata(self, artist: str, title: str, album: Optional[str] = None) -> Dict[str, Any]:
        """
        Enrich track metadata using multiple external APIs
        
        Args:
            artist: Artist name
            title: Track title
            album: Album name (optional)
            
        Returns:
            Enriched metadata dictionary
        """
        enriched_data = {
            "artist": artist,
            "title": title,
            "album": album,
            "external_data": {}
        }
        
        with LogContext(operation="enrich_metadata", artist=artist, title=title):
            log_operation_start(logger, "metadata enrichment")
            start_time = time.time()
            
            try:
                # Get data from multiple sources
                lastfm_data = await self._get_lastfm_data(artist, title)
                musicbrainz_data = await self._get_musicbrainz_data(artist, title)
                spotify_data = await self._get_spotify_data(artist, title)
                
                # Combine all external data
                enriched_data["external_data"] = {
                    "lastfm": lastfm_data,
                    "musicbrainz": musicbrainz_data,
                    "spotify": spotify_data
                }
                
                # Extract key metadata
                enriched_data.update(self._extract_key_metadata(enriched_data["external_data"]))
                
                duration_ms = (time.time() - start_time) * 1000
                log_operation_success(
                    logger,
                    "metadata enrichment",
                    duration_ms,
                    sources_used=len([d for d in enriched_data["external_data"].values() if d]),
                    artist=artist,
                    title=title
                )
                
                return enriched_data
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_operation_error(logger, "metadata enrichment", e, duration_ms)
                return enriched_data
    
    async def _get_lastfm_data(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Get track data from Last.fm API (simulated)"""
        try:
            # Simulate API call
            await asyncio.sleep(0.1)
            
            # Simulated Last.fm response
            lastfm_data = {
                "artist": {"name": artist, "mbid": "artist-mbid-123"},
                "name": title,
                "listeners": 150000,
                "playcount": 750000,
                "toptags": {
                    "tag": [
                        {"name": "rock", "count": 100},
                        {"name": "alternative", "count": 85},
                        {"name": "indie", "count": 70}
                    ]
                },
                "wiki": {
                    "summary": f"Track information for {title} by {artist}"
                }
            }
            
            logger.debug(
                "Last.fm data retrieved",
                artist=artist,
                title=title,
                listeners=lastfm_data["listeners"],
                tags=len(lastfm_data["toptags"]["tag"])
            )
            
            return lastfm_data
            
        except Exception as e:
            logger.warning(f"Last.fm API failed: {e}")
            return None
    
    async def _get_musicbrainz_data(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Get track data from MusicBrainz API (simulated)"""
        try:
            # Simulate API call
            await asyncio.sleep(0.1)
            
            # Simulated MusicBrainz response
            musicbrainz_data = {
                "id": "recording-id-456",
                "title": title,
                "artist-credit": [{"artist": {"name": artist, "id": "artist-id-789"}}],
                "length": 240000,  # milliseconds
                "releases": [
                    {
                        "id": "release-id-101",
                        "title": "Album Title",
                        "date": "2023-01-15",
                        "country": "US"
                    }
                ],
                "tags": [
                    {"name": "rock", "count": 10},
                    {"name": "alternative rock", "count": 8}
                ]
            }
            
            logger.debug(
                "MusicBrainz data retrieved",
                artist=artist,
                title=title,
                mbid=musicbrainz_data["id"],
                releases=len(musicbrainz_data["releases"])
            )
            
            return musicbrainz_data
            
        except Exception as e:
            logger.warning(f"MusicBrainz API failed: {e}")
            return None
    
    async def _get_spotify_data(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Get track data from Spotify API (simulated)"""
        try:
            # Simulate API call
            await asyncio.sleep(0.1)
            
            # Simulated Spotify response
            spotify_data = {
                "id": "spotify-track-id-xyz",
                "name": title,
                "artists": [{"name": artist, "id": "spotify-artist-id-abc"}],
                "album": {"name": "Album Name", "release_date": "2023-01-15"},
                "duration_ms": 240000,
                "popularity": 75,
                "audio_features": {
                    "danceability": 0.65,
                    "energy": 0.78,
                    "key": 7,
                    "loudness": -6.5,
                    "mode": 1,
                    "speechiness": 0.04,
                    "acousticness": 0.15,
                    "instrumentalness": 0.0001,
                    "liveness": 0.12,
                    "valence": 0.82,
                    "tempo": 120.5,
                    "time_signature": 4
                }
            }
            
            logger.debug(
                "Spotify data retrieved",
                artist=artist,
                title=title,
                popularity=spotify_data["popularity"],
                tempo=spotify_data["audio_features"]["tempo"]
            )
            
            return spotify_data
            
        except Exception as e:
            logger.warning(f"Spotify API failed: {e}")
            return None
    
    def _extract_key_metadata(self, external_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metadata from all external sources"""
        metadata = {}
        
        # Extract from Last.fm
        if external_data.get("lastfm"):
            lastfm = external_data["lastfm"]
            metadata["listeners"] = lastfm.get("listeners")
            metadata["playcount"] = lastfm.get("playcount")
            
            # Extract tags
            if "toptags" in lastfm and "tag" in lastfm["toptags"]:
                metadata["tags"] = [tag["name"] for tag in lastfm["toptags"]["tag"]]
        
        # Extract from MusicBrainz
        if external_data.get("musicbrainz"):
            mb = external_data["musicbrainz"]
            metadata["mbid"] = mb.get("id")
            metadata["duration_mb"] = mb.get("length")
            
            # Extract release info
            if "releases" in mb and mb["releases"]:
                release = mb["releases"][0]
                metadata["release_date"] = release.get("date")
                metadata["country"] = release.get("country")
        
        # Extract from Spotify
        if external_data.get("spotify"):
            spotify = external_data["spotify"]
            metadata["spotify_id"] = spotify.get("id")
            metadata["popularity"] = spotify.get("popularity")
            
            # Extract audio features
            if "audio_features" in spotify:
                af = spotify["audio_features"]
                metadata.update({
                    "danceability": af.get("danceability"),
                    "energy": af.get("energy"),
                    "valence": af.get("valence"),
                    "tempo_spotify": af.get("tempo"),
                    "key_spotify": af.get("key"),
                    "mode_spotify": af.get("mode"),
                    "loudness_spotify": af.get("loudness")
                })
        
        return metadata
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of external API connections"""
        health_status = {
            "status": "healthy",
            "apis": {},
            "timestamp": time.time()
        }
        
        # Test each API (simulated)
        apis = ["lastfm", "musicbrainz", "spotify"]
        
        for api in apis:
            try:
                # Simulate health check
                await asyncio.sleep(0.05)
                health_status["apis"][api] = {
                    "status": "healthy",
                    "response_time_ms": 50
                }
            except Exception as e:
                health_status["apis"][api] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
        
        return health_status
