"""
Test infrastructure layer implementations.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.domain.entities import Track, TrackMetadata, AnalysisResult, Playlist
from src.infrastructure.repositories import SQLiteTrackRepository, SQLiteAnalysisRepository, SQLitePlaylistRepository
from src.infrastructure.services import EssentiaAudioAnalyzer, MusicBrainzEnrichmentService


class TestSQLiteTrackRepository:
    """Test SQLite track repository."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.repository = SQLiteTrackRepository(self.db_path)
    
    def teardown_method(self):
        """Clean up test database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_save_and_find_track(self):
        """Test saving and finding a track."""
        metadata = TrackMetadata("Test Song", "Test Artist", "Test Album")
        track = Track("1", "/music/test.mp3", metadata)
        
        # Save track
        result = self.repository.save(track)
        assert result is True
        
        # Find by ID
        found_track = self.repository.find_by_id("1")
        assert found_track is not None
        assert found_track.id == "1"
        assert found_track.metadata.title == "Test Song"
        
        # Find by path
        found_track = self.repository.find_by_path("/music/test.mp3")
        assert found_track is not None
        assert found_track.path == "/music/test.mp3"
    
    def test_find_all_tracks(self):
        """Test finding all tracks."""
        # Create multiple tracks
        track1 = Track("1", "/music/test1.mp3", TrackMetadata("Song 1", "Artist 1", "Album 1"))
        track2 = Track("2", "/music/test2.mp3", TrackMetadata("Song 2", "Artist 2", "Album 2"))
        
        self.repository.save(track1)
        self.repository.save(track2)
        
        tracks = self.repository.find_all()
        assert len(tracks) == 2
    
    def test_count_tracks(self):
        """Test counting tracks."""
        assert self.repository.count() == 0
        
        track = Track("1", "/music/test.mp3", TrackMetadata("Test", "Artist", "Album"))
        self.repository.save(track)
        
        assert self.repository.count() == 1
    
    def test_delete_track(self):
        """Test deleting a track."""
        track = Track("1", "/music/test.mp3", TrackMetadata("Test", "Artist", "Album"))
        self.repository.save(track)
        
        assert self.repository.count() == 1
        
        result = self.repository.delete("1")
        assert result is True
        assert self.repository.count() == 0


class TestSQLiteAnalysisRepository:
    """Test SQLite analysis repository."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.repository = SQLiteAnalysisRepository(self.db_path)
    
    def teardown_method(self):
        """Clean up test database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_save_and_get_analysis(self):
        """Test saving and getting analysis results."""
        analysis = AnalysisResult(
            features={"bpm": 120.0, "energy": 0.7},
            confidence=0.9
        )
        
        # Save analysis
        result = self.repository.save_analysis("track1", analysis)
        assert result is True
        
        # Get analysis
        found_analysis = self.repository.get_analysis("track1")
        assert found_analysis is not None
        assert found_analysis.confidence == 0.9
        assert found_analysis.features["bpm"] == 120.0
    
    def test_get_analysis_statistics(self):
        """Test getting analysis statistics."""
        # Add some analyses
        analysis1 = AnalysisResult({"bpm": 120}, 0.9)
        analysis2 = AnalysisResult({"bpm": 140}, 0.8)
        
        self.repository.save_analysis("track1", analysis1)
        self.repository.save_analysis("track2", analysis2)
        
        stats = self.repository.get_analysis_statistics()
        assert stats["total_analyses"] == 2
        assert abs(stats["average_confidence"] - 0.85) < 0.001
    
    def test_delete_analysis(self):
        """Test deleting analysis."""
        analysis = AnalysisResult({"bpm": 120}, 0.9)
        self.repository.save_analysis("track1", analysis)
        
        # Delete analysis
        result = self.repository.delete_analysis("track1")
        assert result is True
        
        # Verify deletion
        found_analysis = self.repository.get_analysis("track1")
        assert found_analysis is None


class TestSQLitePlaylistRepository:
    """Test SQLite playlist repository."""
    
    def setup_method(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.repository = SQLitePlaylistRepository(self.db_path)
        
        # Create track repository for testing
        self.track_repo = SQLiteTrackRepository(self.db_path)
    
    def teardown_method(self):
        """Clean up test database."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_save_and_get_playlist(self):
        """Test saving and getting a playlist."""
        # Create tracks
        track1 = Track("1", "/music/test1.mp3", TrackMetadata("Song 1", "Artist 1", "Album 1"))
        track2 = Track("2", "/music/test2.mp3", TrackMetadata("Song 2", "Artist 2", "Album 2"))
        
        self.track_repo.save(track1)
        self.track_repo.save(track2)
        
        # Create playlist
        playlist = Playlist("playlist1", "Test Playlist", [track1, track2])
        
        # Save playlist
        result = self.repository.save_playlist(playlist)
        assert result is True
        
        # Get playlist
        found_playlist = self.repository.get_playlist("playlist1")
        assert found_playlist is not None
        assert found_playlist.name == "Test Playlist"
        assert found_playlist.get_size() == 2
    
    def test_get_all_playlists(self):
        """Test getting all playlists."""
        # Create multiple playlists
        playlist1 = Playlist("playlist1", "Playlist 1")
        playlist2 = Playlist("playlist2", "Playlist 2")
        
        self.repository.save_playlist(playlist1)
        self.repository.save_playlist(playlist2)
        
        playlists = self.repository.get_all_playlists()
        assert len(playlists) == 2
    
    def test_count_playlists(self):
        """Test counting playlists."""
        assert self.repository.count() == 0
        
        playlist = Playlist("playlist1", "Test Playlist")
        self.repository.save_playlist(playlist)
        
        assert self.repository.count() == 1


class TestEssentiaAudioAnalyzer:
    """Test Essentia audio analyzer."""
    
    def setup_method(self):
        """Set up analyzer."""
        self.analyzer = EssentiaAudioAnalyzer()
    
    def test_is_available(self):
        """Test availability check."""
        # This will depend on whether essentia/librosa is installed
        available = self.analyzer.is_available()
        assert isinstance(available, bool)
    
    def test_get_supported_formats(self):
        """Test supported formats."""
        formats = self.analyzer.get_supported_formats()
        assert isinstance(formats, list)
        assert '.mp3' in formats
        assert '.flac' in formats
    
    def test_analyze_track_without_audio_file(self):
        """Test analyzing a track without an actual audio file."""
        metadata = TrackMetadata("Test", "Artist", "Album")
        track = Track("1", "/nonexistent/file.mp3", metadata)
        
        # This should raise an exception since the file doesn't exist
        with pytest.raises(Exception):
            self.analyzer.analyze_track(track)


class TestMusicBrainzEnrichmentService:
    """Test MusicBrainz enrichment service."""
    
    def setup_method(self):
        """Set up service."""
        self.service = MusicBrainzEnrichmentService()
    
    def test_is_available(self):
        """Test availability check."""
        available = self.service.is_available()
        assert isinstance(available, bool)
    
    def test_enrich_metadata(self):
        """Test metadata enrichment."""
        metadata = TrackMetadata("Test Song", "Test Artist", "Test Album")
        track = Track("1", "/music/test.mp3", metadata)
        
        # Enrich metadata (may not work without internet/API)
        enriched_track = self.service.enrich_metadata(track)
        
        # Should return a track (either enriched or original)
        assert enriched_track is not None
        assert isinstance(enriched_track, Track) 