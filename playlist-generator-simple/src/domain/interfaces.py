"""
Domain interfaces for Playlist Generator.
Defines contracts for infrastructure implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .entities import Track, AnalysisResult, Playlist


class ITrackRepository(ABC):
    """Interface for track data access."""
    
    @abstractmethod
    def save(self, track: Track) -> bool:
        """Save a track to the repository."""
        pass
    
    @abstractmethod
    def find_by_id(self, track_id: str) -> Optional[Track]:
        """Find a track by its ID."""
        pass
    
    @abstractmethod
    def find_by_path(self, path: str) -> Optional[Track]:
        """Find a track by its file path."""
        pass
    
    @abstractmethod
    def find_all(self) -> List[Track]:
        """Find all tracks."""
        pass
    
    @abstractmethod
    def find_unanalyzed(self) -> List[Track]:
        """Find tracks that haven't been analyzed."""
        pass
    
    @abstractmethod
    def delete(self, track_id: str) -> bool:
        """Delete a track by ID."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of tracks."""
        pass


class IAnalysisRepository(ABC):
    """Interface for analysis result data access."""
    
    @abstractmethod
    def save_analysis(self, track_id: str, result: AnalysisResult) -> bool:
        """Save analysis result for a track."""
        pass
    
    @abstractmethod
    def get_analysis(self, track_id: str) -> Optional[AnalysisResult]:
        """Get analysis result for a track."""
        pass
    
    @abstractmethod
    def delete_analysis(self, track_id: str) -> bool:
        """Delete analysis result for a track."""
        pass
    
    @abstractmethod
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        pass


class IPlaylistRepository(ABC):
    """Interface for playlist data access."""
    
    @abstractmethod
    def save_playlist(self, playlist: Playlist) -> bool:
        """Save a playlist."""
        pass
    
    @abstractmethod
    def get_playlist(self, playlist_id: str) -> Optional[Playlist]:
        """Get a playlist by ID."""
        pass
    
    @abstractmethod
    def get_all_playlists(self) -> List[Playlist]:
        """Get all playlists."""
        pass
    
    @abstractmethod
    def delete_playlist(self, playlist_id: str) -> bool:
        """Delete a playlist."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of playlists."""
        pass


class IAudioAnalyzer(ABC):
    """Interface for audio analysis services."""
    
    @abstractmethod
    def analyze_track(self, track: Track) -> AnalysisResult:
        """Analyze a track and return analysis result."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if analyzer is available."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        pass


class IMetadataEnrichmentService(ABC):
    """Interface for metadata enrichment services."""
    
    @abstractmethod
    def enrich_metadata(self, track: Track) -> Track:
        """Enrich track metadata with external data."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if enrichment service is available."""
        pass


class IPlaylistAlgorithm(ABC):
    """Interface for playlist generation algorithms."""
    
    @abstractmethod
    def generate_playlist(self, tracks: List[Track], size: int) -> Playlist:
        """Generate a playlist from available tracks."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get algorithm name."""
        pass 