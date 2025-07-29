#!/usr/bin/env python3
"""
Test script to verify the new refactored structure works correctly.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing module imports...")
    
    try:
        # Test shared modules
        from shared.config import get_config, get_config_dict
        from shared.exceptions import PlaylistaException, ConfigurationError
        print("✅ Shared modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_configuration():
    """Test that configuration can be loaded successfully."""
    print("\n🧪 Testing configuration loading...")
    
    try:
        from shared.config import get_config, get_config_dict
        
        # Load configuration
        config = get_config()
        print("✅ Configuration loaded successfully")
        
        # Print some key values
        print(f"📁 Host library path: {config.host_library_path}")
        print(f"📁 Music path: {config.music_path}")
        print(f"📁 Output directory: {config.output_dir}")
        print(f"📁 Cache directory: {config.database.cache_dir}")
        print(f"📊 Log level: {config.logging.level}")
        print(f"⚙️  Large file threshold: {config.processing.large_file_threshold_mb}MB")
        print(f"🎵 Min tracks per playlist: {config.playlist.min_tracks_per_playlist}")
        print(f"🎵 Max tracks per playlist: {config.playlist.max_tracks_per_playlist}")
        print(f"🔧 Default workers: {config.processing.default_workers}")
        print(f"🔧 Max workers: {config.processing.max_workers}")
        
        # Test config dictionary
        config_dict = get_config_dict()
        print("✅ Configuration dictionary created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_exceptions():
    """Test that custom exceptions work correctly."""
    print("\n🧪 Testing exception handling...")
    
    try:
        from shared.exceptions import (
            PlaylistaException,
            ConfigurationError,
            AudioAnalysisError,
            PlaylistGenerationError
        )
        
        # Test base exception
        base_ex = PlaylistaException("Test base exception")
        print(f"✅ Base exception: {base_ex}")
        
        # Test configuration error
        config_ex = ConfigurationError("Test config error", config_key="test_key")
        print(f"✅ Configuration error: {config_ex}")
        
        # Test audio analysis error
        audio_ex = AudioAnalysisError("Test audio error", file_path="/test/file.mp3")
        print(f"✅ Audio analysis error: {audio_ex}")
        
        # Test playlist generation error
        playlist_ex = PlaylistGenerationError("Test playlist error", playlist_method="test_method")
        print(f"✅ Playlist generation error: {playlist_ex}")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception test failed: {e}")
        return False


def test_structure():
    """Test that the directory structure is correct."""
    print("\n🧪 Testing directory structure...")
    
    try:
        # Check that all required directories exist
        required_dirs = [
            "src/domain/entities",
            "src/domain/value_objects", 
            "src/domain/services",
            "src/domain/repositories",
            "src/application/services",
            "src/application/commands",
            "src/application/queries",
            "src/application/dtos",
            "src/infrastructure/persistence",
            "src/infrastructure/external_apis",
            "src/infrastructure/file_system",
            "src/infrastructure/logging",
            "src/presentation/cli",
            "src/presentation/api",
            "src/shared/config",
            "src/shared/exceptions",
            "src/shared/utils",
            "src/shared/constants",
            "tests/unit",
            "tests/integration",
            "tests/e2e"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"❌ Missing directories: {missing_dirs}")
            return False
        else:
            print("✅ All required directories exist")
            return True
            
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False


def main():
    """Run all tests for the new structure."""
    print("🚀 Testing New Playlista Refactored Structure\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration System", test_configuration),
        ("Exception Handling", test_exceptions),
        ("Directory Structure", test_structure)
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
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! New structure is working correctly.")
        print("\n📋 Next Steps:")
        print("1. Start implementing domain entities")
        print("2. Create application services")
        print("3. Implement infrastructure layer")
        print("4. Build presentation layer")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the structure.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 