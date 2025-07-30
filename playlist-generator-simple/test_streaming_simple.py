#!/usr/bin/env python3
"""
Simple test script for streaming audio loader.
This script tests the streaming audio loader functionality without requiring specific audio files.
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
        
        # Test library availability
        print("\n4. Testing Library Availability")
        print("-" * 40)
        
        from core.streaming_audio_loader import (
            ESSENTIA_AVAILABLE, LIBROSA_AVAILABLE, 
            SOUNDFILE_AVAILABLE, WAVE_AVAILABLE
        )
        
        print(f"Essentia available: {ESSENTIA_AVAILABLE}")
        print(f"Librosa available: {LIBROSA_AVAILABLE}")
        print(f"SoundFile available: {SOUNDFILE_AVAILABLE}")
        print(f"Wave available: {WAVE_AVAILABLE}")
        
        # Test memory calculation
        print("\n5. Testing Memory Calculation")
        print("-" * 40)
        
        memory_info = streaming_loader.get_memory_info()
        print(f"Total memory: {memory_info.get('total_gb', 'N/A'):.1f}GB")
        print(f"Available memory: {memory_info.get('available_gb', 'N/A'):.1f}GB")
        print(f"Used memory: {memory_info.get('used_gb', 'N/A'):.1f}GB ({memory_info.get('percent_used', 'N/A')}%)")
        print(f"Memory limit: {memory_info.get('memory_limit_gb', 'N/A'):.1f}GB")
        
        # Test chunk duration calculation
        print("\n6. Testing Chunk Duration Calculation")
        print("-" * 40)
        
        test_cases = [
            (100, 300),   # 100MB, 5 minutes
            (500, 600),   # 500MB, 10 minutes
            (50, 180),    # 50MB, 3 minutes
        ]
        
        for file_size_mb, duration_seconds in test_cases:
            optimal_duration = streaming_loader._calculate_optimal_chunk_duration(file_size_mb, duration_seconds)
            print(f"File: {file_size_mb}MB, {duration_seconds}s -> {optimal_duration:.1f}s chunks")
        
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