#!/usr/bin/env python3
"""
Comprehensive test script for the refactored playlista application.
This script runs inside Docker and tests all major components.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, '/app/src')

def setup_logging():
    """Setup basic logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_module_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Test shared modules
        from shared.config import get_config
        from shared.exceptions import PlaylistaException
        print("   ✅ Shared modules imported")
        
        # Test infrastructure modules
        from infrastructure.logging import get_logger
        print("   ✅ Infrastructure modules imported")
        
        # Test application modules
        from application.services.audio_analysis_service import AudioAnalysisService
        from application.services.file_discovery_service import FileDiscoveryService
        from application.services.metadata_enrichment_service import MetadataEnrichmentService
        from application.services.playlist_generation_service import PlaylistGenerationService
        print("   ✅ Application modules imported")
        
        # Test DTOs
        from application.dtos.audio_analysis import AudioAnalysisRequest, AudioAnalysisResponse
        from application.dtos.file_discovery import FileDiscoveryRequest, FileDiscoveryResponse
        from application.dtos.metadata_enrichment import MetadataEnrichmentRequest, MetadataEnrichmentResponse
        from application.dtos.playlist_generation import PlaylistGenerationRequest, PlaylistGenerationResponse
        print("   ✅ DTOs imported")
        
        return True
    except Exception as e:
        print(f"   ❌ Module import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("🧪 Testing configuration...")
    
    try:
        from shared.config import get_config
        config = get_config()
        
        # Test config sections
        assert hasattr(config, 'logging'), "Config missing logging section"
        assert hasattr(config, 'processing'), "Config missing processing section"
        assert hasattr(config, 'memory'), "Config missing memory section"
        assert hasattr(config, 'database'), "Config missing database section"
        
        print("   ✅ Config has logging section")
        print("   ✅ Config has processing section")
        print("   ✅ Config has memory section")
        print("   ✅ Config has database section")
        
        return True
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_service_initialization():
    """Test service initialization."""
    print("🧪 Testing service initialization...")
    
    try:
        from shared.config import get_config
        from application.services.audio_analysis_service import AudioAnalysisService
        from application.services.file_discovery_service import FileDiscoveryService
        from application.services.metadata_enrichment_service import MetadataEnrichmentService
        from application.services.playlist_generation_service import PlaylistGenerationService
        
        config = get_config()
        
        # Initialize services with proper configs
        audio_service = AudioAnalysisService(
            processing_config=config.processing,
            memory_config=config.memory,
            audio_analysis_config=config.audio_analysis
        )
        discovery_service = FileDiscoveryService()
        enrichment_service = MetadataEnrichmentService()
        playlist_service = PlaylistGenerationService()
        
        print("   ✅ Service initialization successful")
        return True
    except Exception as e:
        print(f"   ❌ Service initialization failed: {e}")
        return False

def test_dto_creation():
    """Test DTO creation."""
    print("🧪 Testing DTO creation...")
    
    try:
        from application.dtos.audio_analysis import AudioAnalysisRequest
        from application.dtos.file_discovery import FileDiscoveryRequest
        from application.dtos.metadata_enrichment import MetadataEnrichmentRequest
        from application.dtos.playlist_generation import PlaylistGenerationRequest, PlaylistGenerationMethod
        
        # Test AudioAnalysisRequest creation (without fast_mode)
        audio_request = AudioAnalysisRequest(
            file_paths=["/music/test.mp3"],
            analysis_method="essentia",
            parallel_processing=True
        )
        
        # Test FileDiscoveryRequest creation
        discovery_request = FileDiscoveryRequest(
            search_paths=["/music"],
            file_extensions=[".mp3", ".flac", ".wav"]
        )
        
        # Test MetadataEnrichmentRequest creation
        from uuid import uuid4
        from application.dtos.metadata_enrichment import EnrichmentSource
        
        enrichment_request = MetadataEnrichmentRequest(
            audio_file_ids=[uuid4()],
            sources=[EnrichmentSource.MUSICBRAINZ, EnrichmentSource.LASTFM]
        )
        
        # Test PlaylistGenerationRequest creation
        from domain.entities.audio_file import AudioFile
        
        # Mock audio files for testing
        mock_audio_files = [
            AudioFile(file_path=Path('/music/test1.mp3')),
            AudioFile(file_path=Path('/music/test2.mp3')),
            AudioFile(file_path=Path('/music/test3.mp3'))
        ]
        
        playlist_request = PlaylistGenerationRequest(
            audio_files=mock_audio_files,
            method=PlaylistGenerationMethod.KMEANS,
            playlist_size=20
        )
        
        print("   ✅ DTO creation successful")
        return True
    except Exception as e:
        print(f"   ❌ DTO creation test failed: {e}")
        return False

def test_docker_environment():
    """Test Docker environment setup."""
    print("🧪 Testing Docker environment...")
    
    # Check environment variables
    pythonpath = os.getenv('PYTHONPATH', '')
    music_path = os.getenv('MUSIC_PATH', '')
    cache_dir = os.getenv('CACHE_DIR', '')
    log_dir = os.getenv('LOG_DIR', '')
    output_dir = os.getenv('OUTPUT_DIR', '')
    
    print(f"   ✅ PYTHONPATH: {pythonpath}")
    print(f"   ✅ MUSIC_PATH: {music_path}")
    print(f"   ✅ CACHE_DIR: {cache_dir}")
    print(f"   ✅ LOG_DIR: {log_dir}")
    print(f"   ✅ OUTPUT_DIR: {output_dir}")
    
    # Check directories
    directories = ['/app/src', '/app/cache', '/app/logs', '/app/playlists', '/music']
    for directory in directories:
        if Path(directory).exists():
            print(f"   ✅ Directory exists: {directory}")
        else:
            print(f"   ❌ Directory missing: {directory}")
    
    return True

def test_cli_interface():
    """Test CLI interface."""
    print("🧪 Testing CLI interface...")
    
    try:
        from presentation.cli.cli_interface import CLIInterface
        from shared.config import get_config
        
        config = get_config()
        cli = CLIInterface()
        
        # Test basic CLI functionality
        parser = cli._create_argument_parser()
        assert parser is not None, "CLI parser creation failed"
        
        print("   ✅ CLI interface works correctly")
        return True
    except Exception as e:
        print(f"   ❌ CLI interface test failed: {e}")
        return False

def test_playlista_entry_point():
    """Test the playlista entry point."""
    print("🧪 Testing playlista entry point...")
    
    try:
        # Test help command with proper CLI entry point
        result = subprocess.run(
            ['python', '/app/src/presentation/cli/main.py', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("   ✅ playlista help command works")
        else:
            print(f"   ❌ playlista help failed: {result.stderr}")
            return False
        
        # Test basic argument parsing
        result = subprocess.run(
            ['python', '/app/src/presentation/cli/main.py', 'analyze', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("   ✅ playlista argument parsing works")
        else:
            print(f"   ❌ playlista argument parsing failed: {result.stderr}")
            return False
        
        print("   ✅ playlista entry point works correctly")
        return True
    except Exception as e:
        print(f"   ❌ playlista entry point failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Docker Tests for Refactored Playlista")
    print("=" * 50)
    print()
    
    setup_logging()
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Configuration", test_configuration),
        ("Service Initialization", test_service_initialization),
        ("DTO Creation", test_dto_creation),
        ("Docker Environment", test_docker_environment),
        ("CLI Interface", test_cli_interface),
        ("Playlista Entry Point", test_playlista_entry_point),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        print("📋 Review the failed tests above and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 