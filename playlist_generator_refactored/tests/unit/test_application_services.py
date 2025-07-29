"""
Unit tests for application services.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from application.services.audio_analysis_service import AudioAnalysisService
from application.services.file_discovery_service import FileDiscoveryService
from application.services.playlist_generation_service import PlaylistGenerationService
from application.services.metadata_enrichment_service import MetadataEnrichmentService
from application.dtos.audio_analysis import AudioAnalysisRequest, AnalysisStatus
from application.dtos.file_discovery import FileDiscoveryRequest
from shared.exceptions import AudioFileError


class TestAudioAnalysisService:
    """Test AudioAnalysisService real implementation."""
    
    def test_audio_analysis_service_initialization(self):
        """Test AudioAnalysisService can be initialized."""
        service = AudioAnalysisService()
        assert service is not None
        assert hasattr(service, 'logger')
    
    def test_analyze_audio_file_with_nonexistent_file(self):
        """Test audio analysis with non-existent file."""
        service = AudioAnalysisService()
        request = AudioAnalysisRequest(file_paths=["/nonexistent/file.mp3"])
        
        response = service.analyze_audio_file(request)
        
        # The service handles errors gracefully and returns COMPLETED with errors
        assert response.status == AnalysisStatus.COMPLETED
        assert len(response.errors) > 0
    
    def test_analyze_audio_file_with_mock_audio(self, tmp_path):
        """Test audio analysis with a mock audio file."""
        service = AudioAnalysisService()
        
        # Create a mock audio file (this won't be real audio, but tests the flow)
        mock_audio_file = tmp_path / "test_audio.wav"
        mock_audio_file.write_bytes(b"RIFF    WAVEfmt ")  # Minimal WAV header
        
        request = AudioAnalysisRequest(
            file_paths=[str(mock_audio_file)],
            extract_features=True,
            extract_metadata=True
        )
        
        # This might fail due to invalid audio, but should handle gracefully
        response = service.analyze_audio_file(request)
        
        # Should either complete or fail gracefully
        assert response.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]
    
    def test_analyze_multiple_files(self, tmp_path):
        """Test batch analysis of multiple files."""
        service = AudioAnalysisService()
        
        # Create mock files
        files = []
        for i in range(3):
            mock_file = tmp_path / f"test_audio_{i}.wav"
            mock_file.write_bytes(b"RIFF    WAVEfmt ")
            files.append(mock_file)
        
        requests = [
            AudioAnalysisRequest(file_paths=[str(f)], extract_features=True, extract_metadata=True)
            for f in files
        ]
        
        responses = service.analyze_multiple_files(requests)
        
        assert len(responses) == 3
        for response in responses:
            assert response.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]
    
    def test_get_analysis_status(self):
        """Test getting analysis status."""
        service = AudioAnalysisService()
        status = service.get_analysis_status("test_id")
        assert status == AnalysisStatus.COMPLETED
    
    def test_cancel_analysis(self):
        """Test canceling analysis."""
        service = AudioAnalysisService()
        result = service.cancel_analysis("test_id")
        assert result is True


class TestFileDiscoveryService:
    """Test FileDiscoveryService real implementation."""
    
    def test_file_discovery_service_initialization(self):
        """Test FileDiscoveryService can be initialized."""
        service = FileDiscoveryService()
        assert service is not None
    
    def test_discover_files_with_nonexistent_directory(self):
        """Test file discovery with non-existent directory."""
        service = FileDiscoveryService()
        request = FileDiscoveryRequest(search_paths=["/nonexistent/directory"])
        
        response = service.discover_files(request)
        
        # The service handles errors gracefully and returns "completed" with no files found
        assert response.status == "completed"
        # No files should be discovered from a non-existent directory
        assert len(response.discovered_files) == 0
        assert response.result.total_files_found == 0
    
    def test_discover_files_with_empty_directory(self, tmp_path):
        """Test file discovery with empty directory."""
        service = FileDiscoveryService()
        request = FileDiscoveryRequest(search_paths=[str(tmp_path)])
        
        response = service.discover_files(request)
        
        assert response.status == "completed"
        assert len(response.discovered_files) == 0
    
    def test_discover_files_with_mock_audio_files(self, tmp_path):
        """Test file discovery with mock audio files."""
        service = FileDiscoveryService()
        
        # Create mock audio files
        audio_files = []
        for i in range(3):
            mock_file = tmp_path / f"test_audio_{i}.mp3"
            mock_file.write_bytes(b"ID3")  # Minimal MP3 header
            audio_files.append(mock_file)
        
        request = FileDiscoveryRequest(search_paths=[str(tmp_path)])
        
        response = service.discover_files(request)
        
        assert response.status == "completed"
        # Mock files are invalid, so they should be in skipped_files, not discovered_files
        assert len(response.result.skipped_files) == 3
        assert len(response.discovered_files) == 0
    
    def test_discover_files_with_filters(self, tmp_path):
        """Test file discovery with filters."""
        service = FileDiscoveryService()
        
        # Create files with different extensions
        mp3_file = tmp_path / "test.mp3"
        mp3_file.write_bytes(b"ID3")
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Not an audio file")
        
        request = FileDiscoveryRequest(
            search_paths=[str(tmp_path)],
            file_extensions=[".mp3"]
        )
        
        response = service.discover_files(request)
        
        assert response.status == "completed"
        # The MP3 file is invalid, so it should be in skipped_files
        assert len(response.result.skipped_files) == 1
        assert len(response.discovered_files) == 0


class TestPlaylistGenerationService:
    """Test PlaylistGenerationService."""
    
    def test_playlist_generation_service_initialization(self):
        """Test PlaylistGenerationService can be initialized."""
        service = PlaylistGenerationService()
        assert service is not None


class TestMetadataEnrichmentService:
    """Test MetadataEnrichmentService."""
    
    def test_metadata_enrichment_service_initialization(self):
        """Test MetadataEnrichmentService can be initialized."""
        service = MetadataEnrichmentService()
        assert service is not None 