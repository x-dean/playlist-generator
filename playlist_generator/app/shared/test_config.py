#!/usr/bin/env python3
"""
Simple test script to verify the configuration system works correctly.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import get_config, get_config_dict
from shared.exceptions import ConfigurationError


def test_config_loading():
    """Test that configuration can be loaded successfully."""
    print("🧪 Testing configuration loading...")
    
    try:
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
        
        return True
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_config_dict():
    """Test that configuration can be converted to dictionary."""
    print("\n🧪 Testing configuration dictionary conversion...")
    
    try:
        config_dict = get_config_dict()
        print("✅ Configuration dictionary created successfully")
        
        # Print some key values from dictionary
        print(f"📁 Host library path: {config_dict.get('host_library_path')}")
        print(f"📊 Log level: {config_dict.get('logging', {}).get('level')}")
        print(f"⚙️  Large file threshold: {config_dict.get('processing', {}).get('large_file_threshold_mb')}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating config dictionary: {e}")
        return False


def test_exception_handling():
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


def main():
    """Run all configuration tests."""
    print("🚀 Starting Playlista Configuration Tests\n")
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Configuration Dictionary", test_config_dict),
        ("Exception Handling", test_exception_handling)
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
        print("🎉 All tests passed! Configuration system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the configuration system.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 