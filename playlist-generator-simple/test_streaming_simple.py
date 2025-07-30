#!/usr/bin/env python3
"""
Simple test script for streaming audio loader.
This script tests the streaming audio loader functionality without requiring specific audio files.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import and set up proper logging
from core.logging_setup import setup_logging, get_logger

# Initialize logging system
setup_logging(
    log_level='INFO',
    log_dir='logs',
    log_file_prefix='test_streaming',
    console_logging=True,
    file_logging=True,
    colored_output=True,
    max_log_files=5,
    log_file_size_mb=10,
    log_file_format='text',
    log_file_encoding='utf-8'
)

logger = get_logger('test_streaming')

def test_streaming_audio_loader():
    """Test the streaming audio loader functionality."""
    try:
        from core.streaming_audio_loader import get_streaming_loader
        from core.audio_analyzer import AudioAnalyzer
        
        logger.info("🔧 Testing Streaming Audio Loader")
        logger.info("=" * 50)
        
        # Test streaming loader initialization
        logger.info("\n1. Testing Streaming Loader Initialization")
        logger.info("-" * 40)
        
        streaming_loader = get_streaming_loader(
            memory_limit_percent=50,
            chunk_duration_seconds=15
        )
        
        logger.info("✅ Streaming loader initialized successfully")
        
        # Test audio analyzer initialization
        logger.info("\n2. Testing Audio Analyzer Initialization")
        logger.info("-" * 40)
        
        analyzer = AudioAnalyzer()
        logger.info("✅ Audio analyzer initialized successfully")
        
        # Test configuration
        logger.info("\n3. Testing Configuration")
        logger.info("-" * 40)
        
        logger.info(f"Streaming enabled: {analyzer.streaming_enabled}")
        logger.info(f"Streaming memory limit: {analyzer.streaming_memory_limit_percent}%")
        logger.info(f"Streaming chunk duration: {analyzer.streaming_chunk_duration_seconds}s")
        logger.info(f"Streaming large file threshold: {analyzer.streaming_large_file_threshold_mb}MB")
        
        # Test library availability
        logger.info("\n4. Testing Library Availability")
        logger.info("-" * 40)
        
        from core.streaming_audio_loader import (
            ESSENTIA_AVAILABLE, LIBROSA_AVAILABLE, 
            SOUNDFILE_AVAILABLE, WAVE_AVAILABLE
        )
        
        logger.info(f"Essentia available: {ESSENTIA_AVAILABLE}")
        logger.info(f"Librosa available: {LIBROSA_AVAILABLE}")
        logger.info(f"SoundFile available: {SOUNDFILE_AVAILABLE}")
        logger.info(f"Wave available: {WAVE_AVAILABLE}")
        
        # Test memory calculation
        logger.info("\n5. Testing Memory Calculation")
        logger.info("-" * 40)
        
        memory_info = streaming_loader.get_memory_info()
        logger.info(f"Total memory: {memory_info.get('total_gb', 'N/A'):.1f}GB")
        logger.info(f"Available memory: {memory_info.get('available_gb', 'N/A'):.1f}GB")
        logger.info(f"Used memory: {memory_info.get('used_gb', 'N/A'):.1f}GB ({memory_info.get('percent_used', 'N/A')}%)")
        logger.info(f"Memory limit: {memory_info.get('memory_limit_gb', 'N/A'):.1f}GB")
        
        # Test chunk duration calculation
        logger.info("\n6. Testing Chunk Duration Calculation")
        logger.info("-" * 40)
        
        test_cases = [
            (100, 300),   # 100MB, 5 minutes
            (500, 600),   # 500MB, 10 minutes
            (50, 180),    # 50MB, 3 minutes
        ]
        
        for file_size_mb, duration_seconds in test_cases:
            optimal_duration = streaming_loader._calculate_optimal_chunk_duration(file_size_mb, duration_seconds)
            logger.info(f"File: {file_size_mb}MB, {duration_seconds}s -> {optimal_duration:.1f}s chunks")
        
        logger.info("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    try:
        test_streaming_audio_loader()
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 