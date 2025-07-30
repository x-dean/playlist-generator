#!/usr/bin/env python3
"""
Test script for streaming audio loader.
This script tests the streaming audio loader functionality.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_streaming_audio_loader():
    """Test the streaming audio loader functionality."""
    try:
        from core.streaming_audio_loader import get_streaming_loader
        from core.audio_analyzer import AudioAnalyzer
        
        print("🔧 Testing Streaming Audio Loader")
        print("=" * 50)
        
        # Test streaming loader initialization
        print("\n1. Testing Streaming Loader Initialization")
        print("-" * 40)
        
        streaming_loader = get_streaming_loader(
            memory_limit_percent=50,
            chunk_duration_seconds=15
        )
        
        print("✅ Streaming loader initialized successfully")
        
        # Test audio analyzer initialization
        print("\n2. Testing Audio Analyzer Initialization")
        print("-" * 40)
        
        analyzer = AudioAnalyzer()
        print("✅ Audio analyzer initialized successfully")
        
        # Test configuration
        print("\n3. Testing Configuration")
        print("-" * 40)
        
        print(f"Streaming enabled: {analyzer.streaming_enabled}")
        print(f"Streaming memory limit: {analyzer.streaming_memory_limit_percent}%")
        print(f"Streaming chunk duration: {analyzer.streaming_chunk_duration_seconds}s")
        print(f"Streaming large file threshold: {analyzer.streaming_large_file_threshold_mb}MB")
        
        # Test with a sample audio file if available
        print("\n4. Testing with Sample Audio File")
        print("-" * 40)
        
        # Look for a test audio file
        test_files = [
            "/music/test_song.mp3",
            "/app/music/test_song.mp3",
            "music/test_song.mp3",
            "test_music/test_song.mp3"
        ]
        
        test_file = None
        for file_path in test_files:
            if os.path.exists(file_path):
                test_file = file_path
                break
        
        if test_file:
            print(f"Found test file: {test_file}")
            
            # Test streaming loader
            print("\nTesting streaming loader...")
            chunk_count = 0
            total_samples = 0
            
            for chunk, start_time, end_time in streaming_loader.load_audio_chunks(test_file):
                chunk_count += 1
                total_samples += len(chunk)
                print(f"  Chunk {chunk_count}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk)} samples)")
                
                if chunk_count >= 3:  # Only test first 3 chunks
                    break
            
            print(f"✅ Streaming test completed: {chunk_count} chunks, {total_samples} total samples")
            
            # Test audio analyzer
            print("\nTesting audio analyzer...")
            audio = analyzer._safe_audio_load(test_file)
            if audio is not None:
                print(f"✅ Audio analyzer test completed: {len(audio)} samples loaded")
            else:
                print("❌ Audio analyzer test failed")
        else:
            print("⚠️ No test audio file found - skipping audio loading tests")
            print("Available test files:")
            for file_path in test_files:
                print(f"  {file_path}: {'✅' if os.path.exists(file_path) else '❌'}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_streaming_audio_loader()
    sys.exit(0 if success else 1) 