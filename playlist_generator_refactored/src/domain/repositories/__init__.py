"""
Repository interfaces for the domain layer.

These define the contracts that infrastructure implementations
must fulfill for data persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from pathlib import Path

from domain.entities import AudioFile, FeatureSet, Metadata, AnalysisResult, Playlist


class AudioFileRepository(ABC):
    """Repository interface for AudioFile entities."""
    
    @abstractmethod
    def save(self, audio_file: AudioFile) -> AudioFile:
        """Save audio file to repository."""
        pass
    
    @abstractmethod
    def find_by_id(self, audio_file_id: UUID) -> Optional[AudioFile]:
        """Find audio file by ID."""
        pass
    
    @abstractmethod
    def find_by_path(self, file_path: Path) -> Optional[AudioFile]:
        """Find audio file by file path."""
        pass
    
    @abstractmethod
    def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[AudioFile]:
        """Find all audio files with optional pagination."""
        pass
    
    @abstractmethod
    def delete(self, audio_file_id: UUID) -> bool:
        """Delete audio file by ID."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of audio files."""
        pass


class FeatureSetRepository(ABC):
    """Repository interface for FeatureSet entities."""
    
    @abstractmethod
    def save(self, feature_set: FeatureSet) -> FeatureSet:
        """Save feature set to repository."""
        pass
    
    @abstractmethod
    def find_by_id(self, feature_set_id: UUID) -> Optional[FeatureSet]:
        """Find feature set by ID."""
        pass
    
    @abstractmethod
    def find_by_audio_file_id(self, audio_file_id: UUID) -> Optional[FeatureSet]:
        """Find feature set by audio file ID."""
        pass
    
    @abstractmethod
    def delete(self, feature_set_id: UUID) -> bool:
        """Delete feature set by ID."""
        pass


class MetadataRepository(ABC):
    """Repository interface for Metadata entities."""
    
    @abstractmethod
    def save(self, metadata: Metadata) -> Metadata:
        """Save metadata to repository."""
        pass
    
    @abstractmethod
    def find_by_id(self, metadata_id: UUID) -> Optional[Metadata]:
        """Find metadata by ID."""
        pass
    
    @abstractmethod
    def find_by_audio_file_id(self, audio_file_id: UUID) -> Optional[Metadata]:
        """Find metadata by audio file ID."""
        pass
    
    @abstractmethod
    def delete(self, metadata_id: UUID) -> bool:
        """Delete metadata by ID."""
        pass


class AnalysisResultRepository(ABC):
    """Repository interface for AnalysisResult entities."""
    
    @abstractmethod
    def save(self, analysis_result: AnalysisResult) -> AnalysisResult:
        """Save analysis result to repository."""
        pass
    
    @abstractmethod
    def find_by_id(self, analysis_result_id: UUID) -> Optional[AnalysisResult]:
        """Find analysis result by ID."""
        pass
    
    @abstractmethod
    def find_by_audio_file_id(self, audio_file_id: UUID) -> Optional[AnalysisResult]:
        """Find analysis result by audio file ID."""
        pass
    
    @abstractmethod
    def delete(self, analysis_result_id: UUID) -> bool:
        """Delete analysis result by ID."""
        pass


class PlaylistRepository(ABC):
    """Repository interface for Playlist entities."""
    
    @abstractmethod
    def save(self, playlist: Playlist) -> Playlist:
        """Save playlist to repository."""
        pass
    
    @abstractmethod
    def find_by_id(self, playlist_id: UUID) -> Optional[Playlist]:
        """Find playlist by ID."""
        pass
    
    @abstractmethod
    def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Playlist]:
        """Find all playlists with optional pagination."""
        pass
    
    @abstractmethod
    def delete(self, playlist_id: UUID) -> bool:
        """Delete playlist by ID."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of playlists."""
        pass 