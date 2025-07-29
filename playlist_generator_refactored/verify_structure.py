#!/usr/bin/env python3
"""
Comprehensive verification script for the refactored project structure.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")
    
    try:
        # Test shared components
        from shared.config import get_config
        from shared.exceptions import PlaylistaException
        print("✅ Shared components imported successfully")
        
        # Test domain entities
        from domain.entities import AudioFile, FeatureSet, Metadata, AnalysisResult, Playlist
        print("✅ Domain entities imported successfully")
        
        # Test application DTOs
        from application.dtos.audio_analysis import AudioAnalysisRequest, AudioAnalysisResponse
        from application.dtos.file_discovery import FileDiscoveryRequest, FileDiscoveryResponse
        from application.dtos.metadata_enrichment import MetadataEnrichmentRequest, MetadataEnrichmentResponse
        from application.dtos.playlist_generation import PlaylistGenerationRequest, PlaylistGenerationResponse
        print("✅ Application DTOs imported successfully")
        
        # Test application services
        from application.services.audio_analysis_service import AudioAnalysisService
        from application.services.file_discovery_service import FileDiscoveryService
        from application.services.metadata_enrichment_service import MetadataEnrichmentService
        from application.services.playlist_generation_service import PlaylistGenerationService
        print("✅ Application services imported successfully")
        
        # Test infrastructure
        from infrastructure.logging import setup_logging
        print("✅ Infrastructure components imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")
    
    try:
        from shared.config import get_config
        config = get_config()
        print("✅ Configuration loaded successfully")
        print(f"   - Log level: {config.logging.level}")
        print(f"   - Cache directory: {config.database.cache_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_services_initialization():
    """Test service initialization."""
    print("\n🔍 Testing service initialization...")
    
    try:
        from application.services.audio_analysis_service import AudioAnalysisService
        from application.services.file_discovery_service import FileDiscoveryService
        from application.services.metadata_enrichment_service import MetadataEnrichmentService
        from application.services.playlist_generation_service import PlaylistGenerationService
        
        # Initialize services
        audio_service = AudioAnalysisService()
        discovery_service = FileDiscoveryService()
        enrichment_service = MetadataEnrichmentService()
        playlist_service = PlaylistGenerationService()
        
        print("✅ All services initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False

def test_domain_entities():
    """Test domain entity creation."""
    print("\n🔍 Testing domain entities...")
    
    try:
        from domain.entities import AudioFile, FeatureSet, Metadata, AnalysisResult, Playlist
        from uuid import uuid4
        from pathlib import Path
        
        # Test AudioFile creation
        audio_file = AudioFile(
            file_path=Path("/test/song.mp3"),
            id=uuid4()
        )
        print("✅ AudioFile created successfully")
        
        # Test FeatureSet creation
        feature_set = FeatureSet(
            audio_file_id=uuid4(),
            bpm=120.0,
            energy=0.8
        )
        print("✅ FeatureSet created successfully")
        
        # Test Metadata creation
        metadata = Metadata(
            audio_file_id=uuid4(),
            title="Test Song",
            artist="Test Artist"
        )
        print("✅ Metadata created successfully")
        
        # Test Playlist creation
        playlist = Playlist(
            name="Test Playlist",
            track_ids=[uuid4()],
            track_paths=["/test/song.mp3"]
        )
        print("✅ Playlist created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Domain entity test failed: {e}")
        return False

def test_dto_creation():
    """Test DTO creation."""
    print("\n🔍 Testing DTO creation...")
    
    try:
        from application.dtos.audio_analysis import AudioAnalysisRequest, AudioAnalysisResponse
        from application.dtos.file_discovery import FileDiscoveryRequest, FileDiscoveryResponse
        from application.dtos.metadata_enrichment import MetadataEnrichmentRequest, MetadataEnrichmentResponse
        from application.dtos.playlist_generation import PlaylistGenerationRequest, PlaylistGenerationResponse
        from domain.entities import AudioFile
        from uuid import uuid4
        from pathlib import Path
        
        # Create a mock audio file for testing
        mock_audio_file = AudioFile(
            file_path=Path("/test/song.mp3"),
            id=uuid4()
        )
        
        # Test request DTOs
        audio_request = AudioAnalysisRequest(file_paths=["/test/song.mp3"])
        discovery_request = FileDiscoveryRequest(search_paths=["/test"])
        enrichment_request = MetadataEnrichmentRequest(audio_file_ids=[uuid4()])
        playlist_request = PlaylistGenerationRequest(audio_files=[mock_audio_file])
        
        print("✅ Request DTOs created successfully")
        
        # Test response DTOs
        audio_response = AudioAnalysisResponse(
            request_id=str(uuid4()),
            status="completed",
            progress=type('obj', (object,), {'total_files': 1, 'processed_files': 1, 'successful_files': 1, 'failed_files': 0})()
        )
        discovery_response = FileDiscoveryResponse(
            request_id=str(uuid4()),
            status="completed"
        )
        enrichment_response = MetadataEnrichmentResponse(
            request_id=str(uuid4()),
            status="completed"
        )
        playlist_response = PlaylistGenerationResponse(
            request_id=str(uuid4()),
            status="completed"
        )
        
        print("✅ Response DTOs created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ DTO test failed: {e}")
        return False

def check_directory_structure():
    """Check that all required directories exist."""
    print("\n🔍 Checking directory structure...")
    
    required_dirs = [
        'src/domain/entities',
        'src/domain/repositories',
        'src/domain/services',
        'src/domain/value_objects',
        'src/application/services',
        'src/application/dtos',
        'src/application/commands',
        'src/application/queries',
        'src/infrastructure/logging',
        'src/infrastructure/file_system',
        'src/infrastructure/external_apis',
        'src/infrastructure/persistence',
        'src/shared/config',
        'src/shared/exceptions',
        'src/shared/constants',
        'src/shared/utils',
        'src/presentation/cli',
        'tests/unit',
        'tests/integration'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ All required directories exist")
    return True

def main():
    """Run all verification tests."""
    print("🧪 Starting comprehensive verification...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Service Initialization Tests", test_services_initialization),
        ("Domain Entity Tests", test_domain_entities),
        ("DTO Creation Tests", test_dto_creation),
        ("Directory Structure Tests", check_directory_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"VERIFICATION SUMMARY")
    print('='*50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All verification tests passed! The refactored structure is complete.")
        return True
    else:
        print("⚠️  Some verification tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 