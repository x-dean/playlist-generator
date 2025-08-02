"""
Query objects for Playlist Generator.
Implements Query pattern for read operations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class GetAnalysisStatsQuery:
    """Query to get analysis statistics."""
    filters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


@dataclass
class GetPlaylistQuery:
    """Query to get a playlist."""
    playlist_id: str
    
    def __post_init__(self):
        if not self.playlist_id:
            raise ValueError("Playlist ID is required")


@dataclass
class GetTrackQuery:
    """Query to get a track."""
    track_id: str
    
    def __post_init__(self):
        if not self.track_id:
            raise ValueError("Track ID is required")


@dataclass
class GetTracksQuery:
    """Query to get tracks with filters."""
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    sort_by: Optional[str] = None
    sort_order: str = "asc"
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.sort_order not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")


@dataclass
class GetPlaylistsQuery:
    """Query to get playlists with filters."""
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


@dataclass
class SearchTracksQuery:
    """Query to search tracks."""
    query: str
    search_fields: Optional[List[str]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    
    def __post_init__(self):
        if not self.query:
            raise ValueError("Search query is required")
        if self.search_fields is None:
            self.search_fields = ["title", "artist", "album"]


@dataclass
class GetAnalysisResultQuery:
    """Query to get analysis result for a track."""
    track_id: str
    
    def __post_init__(self):
        if not self.track_id:
            raise ValueError("Track ID is required") 