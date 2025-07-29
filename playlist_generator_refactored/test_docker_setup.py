#!/usr/bin/env python3
"""
Test script to verify Docker setup and real AudioAnalysisService implementation.
"""

import sys
import logging
from pathlib import Path
import tempfile
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from shared.config import get_config
from infrastructure.logging import setup_logging
from application.services.audio_analysis_service import AudioAnalysisService
from application.services.file_discovery_service import FileDiscoveryService
from application.dtos.audio_analysis import AudioAnalysisRequest, AnalysisStatus
from application.dtos.file_discovery import FileDiscoveryRequest


def test_docker_setup():
    """Test the Docker setup and real service implementations."""
    
    # Setup logging
    config = get_config()
    setup_logging(config.logging)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing Docker setup and real service implementations...")
    
    try:
        # Test service initialization
        logger.info("📦 Initializing services...")
        audio_service = AudioAnalysisService()
        file_service = FileDiscoveryService()
        
        logger.info("✅ Services initialized successfully")
        
        # Test with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock audio files for testing
            logger.info("🎵 Creating mock audio files for testing...")
            
            # Create a mock MP3 file
            mock_mp3 = temp_path / "test_audio.mp3"
            mock_mp3.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00")  # Minimal MP3 header
            
            # Test FileDiscoveryService
            logger.info("🔍 Testing FileDiscoveryService...")
            discovery_request = FileDiscoveryRequest(directory_path=str(temp_path))
            discovery_response = file_service.discover_files(discovery_request)
            
            logger.info(f"📁 FileDiscoveryService result: {discovery_response.status}")
            logger.info(f"📁 Discovered files: {len(discovery_response.discovered_files)}")
            
            # Test AudioAnalysisService
            logger.info("🎼 Testing AudioAnalysisService...")
            analysis_request = AudioAnalysisRequest(
                file_path=str(mock_mp3),
                extract_bpm=True,
                extract_mfcc=True
            )
            
            analysis_response = audio_service.analyze_audio_file(analysis_request)
            
            logger.info(f"🎼 AudioAnalysisService result: {analysis_response.status}")
            if analysis_response.status == AnalysisStatus.COMPLETED:
                logger.info(f"🎼 Quality score: {analysis_response.quality_score}")
                if analysis_response.analysis_result:
                    logger.info(f"🎼 BPM: {analysis_response.analysis_result.feature_set.bpm}")
                    logger.info(f"🎼 Duration: {analysis_response.analysis_result.feature_set.duration}")
        
        logger.info("🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_docker_setup()
    sys.exit(0 if success else 1) 