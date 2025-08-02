"""
Domain entities for Playlist Generator.
Core business objects with encapsulated behavior.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class TrackMetadata:
    """Track metadata value object."""
    title: str
    artist: str
    album: str
    duration: Optional[float] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.title or not self.artist:
            raise ValueError("Title and artist are required")
    
    def add_tag(self, tag: str):
        """Add a tag to the track."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if track has a specific tag."""
        return tag in self.tags


@dataclass
class AnalysisResult:
    """Audio analysis result value object."""
    features: Dict[str, Any]
    confidence: float
    analysis_date: datetime = field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def get_feature(self, name: str) -> Optional[Any]:
        """Get a specific feature value."""
        return self.features.get(name)
    
    def has_feature(self, name: str) -> bool:
        """Check if analysis has a specific feature."""
        return name in self.features


class Track:
    """Track domain entity."""
    
    def __init__(self, id: str, path: str, metadata: TrackMetadata):
        self.id = id
        self.path = path
        self.metadata = metadata
        self.analysis_result: Optional[AnalysisResult] = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def analyze(self, result: AnalysisResult):
        """Set analysis result for this track."""
        self.analysis_result = result
        self.updated_at = datetime.now()
    
    def is_analyzed(self) -> bool:
        """Check if track has been analyzed."""
        return self.analysis_result is not None
    
    def get_file_size(self) -> Optional[int]:
        """Get file size in bytes."""
        try:
            return Path(self.path).stat().st_size
        except OSError:
            return None
    
    def get_file_extension(self) -> str:
        """Get file extension."""
        return Path(self.path).suffix.lower()
    
    def is_audio_file(self) -> bool:
        """Check if file is a supported audio format."""
        return self.get_file_extension() in ['.mp3', '.flac', '.wav', '.m4a']


class Playlist:
    """Playlist domain entity."""
    
    def __init__(self, id: str, name: str, tracks: List[Track] = None):
        self.id = id
        self.name = name
        self.tracks = tracks or []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_track(self, track: Track):
        """Add a track to the playlist."""
        if track not in self.tracks:
            self.tracks.append(track)
            self.updated_at = datetime.now()
    
    def remove_track(self, track: Track):
        """Remove a track from the playlist."""
        if track in self.tracks:
            self.tracks.remove(track)
            self.updated_at = datetime.now()
    
    def get_size(self) -> int:
        """Get playlist size."""
        return len(self.tracks)
    
    def is_empty(self) -> bool:
        """Check if playlist is empty."""
        return len(self.tracks) == 0
    
    def get_duration(self) -> float:
        """Get total playlist duration."""
        total_duration = 0.0
        for track in self.tracks:
            if track.metadata.duration:
                total_duration += track.metadata.duration
        return total_duration
    
    def get_artists(self) -> List[str]:
        """Get unique artists in playlist."""
        return list(set(track.metadata.artist for track in self.tracks))
    
    def get_genres(self) -> List[str]:
        """Get unique genres in playlist."""
        genres = []
        for track in self.tracks:
            if track.metadata.genre:
                genres.append(track.metadata.genre)
        return list(set(genres)) 