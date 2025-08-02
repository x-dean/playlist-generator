"""
Test the new clean architecture implementation.
"""

import pytest
from pathlib import Path
import tempfile
import os

from src.domain.entities import Track, TrackMetadata, AnalysisResult, Playlist
from src.domain.interfaces import ITrackRepository, IAnalysisRepository, IPlaylistRepository
from src.application.commands import AnalyzeTrackCommand, GeneratePlaylistCommand
from src.application.queries import GetAnalysisStatsQuery, GetPlaylistQuery
from src.application.use_cases import AnalyzeTrackUseCase, GetAnalysisStatsUseCase


class MockTrackRepository(ITrackRepository):
    """Mock implementation for testing."""
    
    def __init__(self):
        self.tracks = {}
    
    def save(self, track: Track) -> bool:
        self.tracks[track.id] = track
        return True
    
    def find_by_id(self, track_id: str):
        return self.tracks.get(track_id)
    
    def find_by_path(self, path: str):
        for track in self.tracks.values():
            if track.path == path:
                return track
        return None
    
    def find_all(self):
        return list(self.tracks.values())
    
    def find_unanalyzed(self):
        return [track for track in self.tracks.values() if not track.is_analyzed()]
    
    def delete(self, track_id: str) -> bool:
        if track_id in self.tracks:
            del self.tracks[track_id]
            return True
        return False
    
    def count(self) -> int:
        return len(self.tracks)


class MockAnalysisRepository(IAnalysisRepository):
    """Mock implementation for testing."""
    
    def __init__(self):
        self.analyses = {}
    
    def save_analysis(self, track_id: str, result: AnalysisResult) -> bool:
        self.analyses[track_id] = result
        return True
    
    def get_analysis(self, track_id: str):
        return self.analyses.get(track_id)
    
    def delete_analysis(self, track_id: str) -> bool:
        if track_id in self.analyses:
            del self.analyses[track_id]
            return True
        return False
    
    def get_analysis_statistics(self):
        return {
            "total_analyses": len(self.analyses),
            "average_confidence": sum(r.confidence for r in self.analyses.values()) / len(self.analyses) if self.analyses else 0
        }


class MockAudioAnalyzer:
    """Mock audio analyzer for testing."""
    
    def analyze_track(self, track: Track) -> AnalysisResult:
        return AnalysisResult(
            features={"bpm": 120.0, "key": "C", "energy": 0.7},
            confidence=0.9
        )
    
    def is_available(self) -> bool:
        return True
    
    def get_supported_formats(self):
        return [".mp3", ".flac", ".wav"]


class TestDomainEntities:
    """Test domain entities."""
    
    def test_track_creation(self):
        """Test track entity creation."""
        metadata = TrackMetadata("Test Song", "Test Artist", "Test Album")
        track = Track("1", "/music/test.mp3", metadata)
        
        assert track.id == "1"
        assert track.path == "/music/test.mp3"
        assert track.metadata.title == "Test Song"
        assert not track.is_analyzed()
    
    def test_track_analysis(self):
        """Test track analysis."""
        metadata = TrackMetadata("Test Song", "Test Artist", "Test Album")
        track = Track("1", "/music/test.mp3", metadata)
        
        analysis = AnalysisResult(
            features={"bpm": 120.0},
            confidence=0.9
        )
        
        track.analyze(analysis)
        assert track.is_analyzed()
        assert track.analysis_result == analysis
    
    def test_playlist_creation(self):
        """Test playlist entity creation."""
        metadata = TrackMetadata("Test Song", "Test Artist", "Test Album")
        track = Track("1", "/music/test.mp3", metadata)
        
        playlist = Playlist("1", "Test Playlist", [track])
        
        assert playlist.id == "1"
        assert playlist.name == "Test Playlist"
        assert playlist.get_size() == 1
        assert not playlist.is_empty()


class TestUseCases:
    """Test application use cases."""
    
    def setup_method(self):
        """Set up test dependencies."""
        self.track_repo = MockTrackRepository()
        self.analysis_repo = MockAnalysisRepository()
        self.audio_analyzer = MockAudioAnalyzer()
        
        self.analyze_use_case = AnalyzeTrackUseCase(
            self.track_repo,
            self.analysis_repo,
            self.audio_analyzer
        )
        
        self.stats_use_case = GetAnalysisStatsUseCase(self.analysis_repo)
    
    def test_analyze_track_use_case(self, tmp_path):
        """Test analyze track use case."""
        # Create a temporary audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.write_text("fake audio data")
        
        command = AnalyzeTrackCommand(str(audio_file))
        result = self.analyze_use_case.execute(command)
        
        assert result is not None
        assert result.confidence == 0.9
        assert "bpm" in result.features
        
        # Check track was saved
        assert self.track_repo.count() == 1
    
    def test_get_analysis_stats_use_case(self):
        """Test get analysis stats use case."""
        # Add some mock analyses
        analysis1 = AnalysisResult({"bpm": 120}, 0.9)
        analysis2 = AnalysisResult({"bpm": 140}, 0.8)
        
        self.analysis_repo.save_analysis("1", analysis1)
        self.analysis_repo.save_analysis("2", analysis2)
        
        query = GetAnalysisStatsQuery()
        stats = self.stats_use_case.execute(query)
        
        assert stats["total_analyses"] == 2
        # Use approximate comparison for floating point
        assert abs(stats["average_confidence"] - 0.85) < 0.001


class TestCommands:
    """Test command objects."""
    
    def test_analyze_track_command(self):
        """Test analyze track command."""
        command = AnalyzeTrackCommand("/music/test.mp3", force_reanalysis=True)
        
        assert command.file_path == "/music/test.mp3"
        assert command.force_reanalysis is True
    
    def test_generate_playlist_command(self):
        """Test generate playlist command."""
        command = GeneratePlaylistCommand("kmeans", 20, "Test Playlist")
        
        assert command.method == "kmeans"
        assert command.size == 20
        assert command.name == "Test Playlist"
    
    def test_invalid_command_validation(self):
        """Test command validation."""
        with pytest.raises(ValueError):
            AnalyzeTrackCommand("")  # Empty file path
        
        with pytest.raises(ValueError):
            GeneratePlaylistCommand("", 20, "Test")  # Empty method
        
        with pytest.raises(ValueError):
            GeneratePlaylistCommand("kmeans", 0, "Test")  # Invalid size


class TestQueries:
    """Test query objects."""
    
    def test_get_analysis_stats_query(self):
        """Test analysis stats query."""
        query = GetAnalysisStatsQuery(filters={"confidence_min": 0.8})
        
        assert query.filters["confidence_min"] == 0.8
    
    def test_get_playlist_query(self):
        """Test get playlist query."""
        query = GetPlaylistQuery("playlist-1")
        
        assert query.playlist_id == "playlist-1"
    
    def test_invalid_query_validation(self):
        """Test query validation."""
        with pytest.raises(ValueError):
            GetPlaylistQuery("")  # Empty playlist ID 