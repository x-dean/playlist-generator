#!/usr/bin/env python3
"""
Simple test script for AudioAnalysisService without heavy dependencies.
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


def test_services_without_heavy_deps():
    """Test services without requiring heavy audio libraries."""
    
    # Setup logging
    config = get_config()
    setup_logging(config)  # Pass the entire config object, not config.logging
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing services without heavy dependencies...")
    
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
            discovery_request = FileDiscoveryRequest(search_paths=[str(temp_path)])  # Use search_paths instead of directory_path
            discovery_response = file_service.discover_files(discovery_request)
            
            logger.info(f"📁 FileDiscoveryService result: {discovery_response.status}")
            logger.info(f"📁 Discovered files: {len(discovery_response.discovered_files)}")
            
            # Test AudioAnalysisService (will likely fail due to missing librosa, but tests error handling)
            logger.info("🎼 Testing AudioAnalysisService...")
            analysis_request = AudioAnalysisRequest(
                file_paths=[str(mock_mp3)],  # Use file_paths list instead of file_path
                extract_features=True,
                extract_metadata=True
            )
            
            analysis_response = audio_service.analyze_audio_file(analysis_request)
            
            logger.info(f"🎼 AudioAnalysisService result: {analysis_response.status}")
            if analysis_response.status == AnalysisStatus.COMPLETED:
                logger.info(f"🎼 Successful files: {analysis_response.progress.successful_files}")
                logger.info(f"🎼 Failed files: {analysis_response.progress.failed_files}")
                if analysis_response.results:
                    first_result = analysis_response.results[0]
                    logger.info(f"🎼 Quality score: {first_result.quality_score}")
                    logger.info(f"🎼 BPM: {first_result.feature_set.bpm}")
                    logger.info(f"🎼 Duration: {first_result.feature_set.duration}")
            else:
                logger.info(f"🎼 Errors: {len(analysis_response.errors)}")
                for error in analysis_response.errors:
                    logger.info(f"🎼 Error: {error}")
        
        logger.info("🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_services_without_heavy_deps()
    sys.exit(0 if success else 1) 