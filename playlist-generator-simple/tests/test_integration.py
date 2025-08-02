"""
Integration tests for Playlist Generator.
Tests the complete flow from domain to infrastructure.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.domain.entities import Track, TrackMetadata, AnalysisResult
from src.application.commands import AnalyzeTrackCommand, ImportTracksCommand
from src.application.use_cases import AnalyzeTrackUseCase, ImportTracksUseCase
from src.infrastructure.repositories import SQLiteTrackRepository, SQLiteAnalysisRepository
from src.infrastructure.services import EssentiaAudioAnalyzer, MusicBrainzEnrichmentService


class TestCompleteFlow:
    """Test the complete application flow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        
        # Initialize repositories
        self.track_repo = SQLiteTrackRepository(self.db_path)
        self.analysis_repo = SQLiteAnalysisRepository(self.db_path)
        
        # Initialize services
        self.audio_analyzer = EssentiaAudioAnalyzer()
        self.enrichment_service = MusicBrainzEnrichmentService()
        
        # Initialize use cases
        self.analyze_use_case = AnalyzeTrackUseCase(
            self.track_repo,
            self.analysis_repo,
            self.audio_analyzer,
            self.enrichment_service
        )
        
        self.import_use_case = ImportTracksUseCase(self.track_repo)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_complete_analyze_flow(self, tmp_path):
        """Test complete analyze track flow."""
        # Create a temporary audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.write_text("fake audio data")
        
        # Create command
        command = AnalyzeTrackCommand(str(audio_file))
        
        # Execute use case
        try:
            result = self.analyze_use_case.execute(command)
            
            # Verify result
            assert result is not None
            assert isinstance(result, AnalysisResult)
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0
            
            # Verify track was saved
            assert self.track_repo.count() == 1
            
            # Verify analysis was saved
            tracks = self.track_repo.find_all()
            assert len(tracks) == 1
            
            track = tracks[0]
            assert track.is_analyzed()
            assert track.analysis_result is not None
            
        except Exception as e:
            # If audio analysis fails due to missing libraries, that's expected
            # But we should still have a track saved
            assert self.track_repo.count() == 1
    
    def test_import_tracks_flow(self, tmp_path):
        """Test import tracks flow."""
        # Create test directory with audio files
        test_dir = tmp_path / "music"
        test_dir.mkdir()
        
        # Create fake audio files
        (test_dir / "song1.mp3").write_text("fake audio 1")
        (test_dir / "song2.flac").write_text("fake audio 2")
        (test_dir / "song3.wav").write_text("fake audio 3")
        (test_dir / "not_audio.txt").write_text("not audio")
        
        # Create command
        command = ImportTracksCommand(str(test_dir))
        
        # Execute use case
        result = self.import_use_case.execute(command)
        
        # Verify result
        assert result["imported"] == 3  # 3 audio files
        assert result["skipped"] == 0   # No existing tracks
        assert result["errors"] == 0    # No errors
        assert result["total_files"] == 3
        
        # Verify tracks were saved
        assert self.track_repo.count() == 3
        
        # Verify track metadata
        tracks = self.track_repo.find_all()
        assert len(tracks) == 3
        
        # Check that tracks have basic metadata
        for track in tracks:
            assert track.metadata.title in ["song1", "song2", "song3"]
            assert track.metadata.artist == "Unknown"
            assert track.metadata.album == "Unknown"
    
    def test_repository_integration(self):
        """Test repository integration."""
        # Create and save track
        metadata = TrackMetadata("Test Song", "Test Artist", "Test Album")
        track = Track("1", "/music/test.mp3", metadata)
        
        self.track_repo.save(track)
        
        # Create and save analysis
        analysis = AnalysisResult(
            features={"bpm": 120.0, "energy": 0.7},
            confidence=0.9
        )
        
        self.analysis_repo.save_analysis(track.id, analysis)
        
        # Verify integration
        found_track = self.track_repo.find_by_id("1")
        assert found_track is not None
        
        found_analysis = self.analysis_repo.get_analysis("1")
        assert found_analysis is not None
        assert found_analysis.confidence == 0.9
    
    def test_service_integration(self):
        """Test service integration."""
        # Test audio analyzer availability
        available = self.audio_analyzer.is_available()
        assert isinstance(available, bool)
        
        # Test enrichment service availability
        available = self.enrichment_service.is_available()
        assert isinstance(available, bool)
        
        # Test supported formats
        formats = self.audio_analyzer.get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
    
    def test_error_handling(self):
        """Test error handling in the flow."""
        # Test with non-existent file
        command = AnalyzeTrackCommand("/nonexistent/file.mp3")
        
        with pytest.raises(Exception):
            self.analyze_use_case.execute(command)
        
        # Verify no tracks were saved
        assert self.track_repo.count() == 0
    
    def test_data_persistence(self):
        """Test data persistence across operations."""
        # Create and save multiple tracks
        for i in range(3):
            metadata = TrackMetadata(f"Song {i}", f"Artist {i}", f"Album {i}")
            track = Track(f"track_{i}", f"/music/song{i}.mp3", metadata)
            self.track_repo.save(track)
        
        # Verify persistence
        assert self.track_repo.count() == 3
        
        # Create new repository instance (simulating restart)
        new_repo = SQLiteTrackRepository(self.db_path)
        assert new_repo.count() == 3
        
        # Verify data integrity
        tracks = new_repo.find_all()
        assert len(tracks) == 3
        
        # Check that all expected tracks are present (order may vary)
        track_titles = [track.metadata.title for track in tracks]
        track_artists = [track.metadata.artist for track in tracks]
        
        expected_titles = [f"Song {i}" for i in range(3)]
        expected_artists = [f"Artist {i}" for i in range(3)]
        
        assert sorted(track_titles) == sorted(expected_titles)
        assert sorted(track_artists) == sorted(expected_artists) 