"""
Command objects for Playlist Generator.
Implements Command pattern for write operations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class AnalyzeTrackCommand:
    """Command to analyze a track."""
    file_path: str
    force_reanalysis: bool = False
    
    def __post_init__(self):
        if not self.file_path:
            raise ValueError("File path is required")


@dataclass
class GeneratePlaylistCommand:
    """Command to generate a playlist."""
    method: str
    size: int
    name: str
    parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.method:
            raise ValueError("Method is required")
        if self.size <= 0:
            raise ValueError("Size must be positive")
        if not self.name:
            raise ValueError("Name is required")


@dataclass
class EnrichMetadataCommand:
    """Command to enrich track metadata."""
    track_id: str
    force_enrichment: bool = False
    
    def __post_init__(self):
        if not self.track_id:
            raise ValueError("Track ID is required")


@dataclass
class ImportTracksCommand:
    """Command to import tracks from directory."""
    directory_path: str
    recursive: bool = True
    supported_formats: Optional[list] = None
    
    def __post_init__(self):
        if not self.directory_path:
            raise ValueError("Directory path is required")
        if self.supported_formats is None:
            self.supported_formats = ['.mp3', '.flac', '.wav', '.m4a']


@dataclass
class DeleteTrackCommand:
    """Command to delete a track."""
    track_id: str
    
    def __post_init__(self):
        if not self.track_id:
            raise ValueError("Track ID is required")


@dataclass
class DeletePlaylistCommand:
    """Command to delete a playlist."""
    playlist_id: str
    
    def __post_init__(self):
        if not self.playlist_id:
            raise ValueError("Playlist ID is required") 