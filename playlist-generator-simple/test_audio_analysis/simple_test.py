#!/usr/bin/env python3
"""
Simple test script to debug audio analyzer issues.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.audio_analyzer import AudioAnalyzer
        print("✅ AudioAnalyzer imported successfully")
    except Exception as e:
        print(f"❌ AudioAnalyzer import failed: {e}")
        return False
    
    try:
        from core.config_loader import config_loader
        print("✅ Config loader imported successfully")
    except Exception as e:
        print(f"❌ Config loader import failed: {e}")
        return False
    
    try:
        from core.resource_manager import resource_manager
        print("✅ Resource manager imported successfully")
    except Exception as e:
        print(f"❌ Resource manager import failed: {e}")
        return False
    
    try:
        from core.streaming_audio_loader import get_streaming_loader
        print("✅ Streaming audio loader imported successfully")
    except Exception as e:
        print(f"❌ Streaming audio loader import failed: {e}")
        return False
    
    try:
        from core.external_apis import metadata_enrichment_service
        print("✅ External APIs imported successfully")
    except Exception as e:
        print(f"❌ External APIs import failed: {e}")
        return False
    
    return True

def test_analyzer_initialization():
    """Test if the analyzer can be initialized."""
    print("\nTesting analyzer initialization...")
    
    try:
        from core.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        print("✅ AudioAnalyzer initialized successfully")
        return analyzer
    except Exception as e:
        print(f"❌ AudioAnalyzer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_file_exists():
    """Test if the test file exists."""
    print("\nTesting file existence...")
    
    test_file = "/music/22Gz - Crime Rate.mp3"
    
    if os.path.exists(test_file):
        print(f"✅ Test file exists: {test_file}")
        file_size = os.path.getsize(test_file)
        print(f"   File size: {file_size / (1024*1024):.1f} MB")
        return test_file
    else:
        print(f"❌ Test file not found: {test_file}")
        
        # List files in music directory
        music_dir = "/music"
        if os.path.exists(music_dir):
            print(f"Files in {music_dir}:")
            for file in os.listdir(music_dir)[:5]:  # Show first 5 files
                file_path = os.path.join(music_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file} ({size / (1024*1024):.1f} MB)")
        else:
            print(f"❌ Music directory not found: {music_dir}")
        
        return None

def test_simple_analysis(analyzer, test_file):
    """Test simple analysis without full feature extraction."""
    print("\nTesting simple analysis...")
    
    try:
        # Test just loading the audio file
        print("Testing audio loading...")
        
        # Get file info
        file_size_bytes = os.path.getsize(test_file)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f}MB")
        
        # Try to load audio
        audio = analyzer._safe_audio_load(test_file)
        if audio is not None:
            print(f"✅ Audio loaded successfully: {len(audio)} samples")
            print(f"   Duration: {len(audio) / 44100:.2f} seconds")
            return True
        else:
            print("❌ Audio loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Simple analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE AUDIO ANALYZER TEST")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("❌ Import tests failed")
        sys.exit(1)
    
    # Test analyzer initialization
    analyzer = test_analyzer_initialization()
    if analyzer is None:
        print("❌ Analyzer initialization failed")
        sys.exit(1)
    
    # Test file existence
    test_file = test_file_exists()
    if test_file is None:
        print("❌ Test file not found")
        sys.exit(1)
    
    # Test simple analysis
    if test_simple_analysis(analyzer, test_file):
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Simple analysis failed")
        sys.exit(1) 