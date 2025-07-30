#!/usr/bin/env python3
"""
Simple test script to verify that the streaming audio functionality works correctly.
This script tests the streaming audio loader without requiring audio libraries.
"""

import os
import sys
import logging
import tempfile
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.streaming_audio_loader import StreamingAudioLoader, get_streaming_loader
from core.config_loader import config_loader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_streaming_loader_initialization():
    """Test that the streaming loader initializes correctly."""
    logger.info("\n=== Testing Streaming Loader Initialization ===")
    
    try:
        # Test with default settings
        loader = StreamingAudioLoader()
        
        # Check memory info
        memory_info = loader.get_memory_info()
        logger.info(f"✅ Streaming loader initialized successfully")
        logger.info(f"   Available memory: {memory_info.get('available_gb', 'N/A'):.1f}GB")
        logger.info(f"   Memory limit: {memory_info.get('memory_limit_gb', 'N/A'):.1f}GB")
        logger.info(f"   Chunk duration: {loader.chunk_duration_seconds}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming loader initialization failed: {e}")
        return False

def test_streaming_loader_configuration():
    """Test that the streaming loader respects configuration."""
    logger.info("\n=== Testing Streaming Loader Configuration ===")
    
    try:
        # Test with custom settings
        loader = StreamingAudioLoader(
            memory_limit_percent=70,
            chunk_duration_seconds=15
        )
        
        # Verify settings
        assert loader.memory_limit_percent == 70
        assert loader.chunk_duration_seconds == 15
        
        logger.info(f"✅ Configuration test passed")
        logger.info(f"   Memory limit: {loader.memory_limit_percent}%")
        logger.info(f"   Chunk duration: {loader.chunk_duration_seconds}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_chunk_duration_calculation():
    """Test that chunk duration calculation works correctly."""
    logger.info("\n=== Testing Chunk Duration Calculation ===")
    
    try:
        loader = StreamingAudioLoader()
        
        # Test with different file sizes and durations
        test_cases = [
            (100, 300),   # 100MB, 5 minutes
            (500, 600),   # 500MB, 10 minutes
            (50, 180),    # 50MB, 3 minutes
        ]
        
        for file_size_mb, duration_seconds in test_cases:
            optimal_duration = loader._calculate_optimal_chunk_duration(file_size_mb, duration_seconds)
            
            logger.info(f"✅ File: {file_size_mb}MB, {duration_seconds}s -> {optimal_duration:.1f}s chunks")
            
            # Verify constraints
            assert optimal_duration >= 5, f"Chunk duration too small: {optimal_duration}"
            assert optimal_duration <= 120, f"Chunk duration too large: {optimal_duration}"
            assert optimal_duration <= duration_seconds, f"Chunk duration exceeds total duration: {optimal_duration}"
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Chunk duration calculation test failed: {e}")
        return False

def test_memory_awareness():
    """Test that the system is memory-aware."""
    logger.info("\n=== Testing Memory Awareness ===")
    
    try:
        # Get streaming loader
        loader = get_streaming_loader()
        
        # Get memory info
        memory_info = loader.get_memory_info()
        
        logger.info(f"✅ Memory awareness test:")
        logger.info(f"   Total RAM: {memory_info.get('total_gb', 'N/A'):.1f}GB")
        logger.info(f"   Available RAM: {memory_info.get('available_gb', 'N/A'):.1f}GB")
        logger.info(f"   Used RAM: {memory_info.get('used_gb', 'N/A'):.1f}GB ({memory_info.get('percent_used', 'N/A')}%)")
        logger.info(f"   Memory limit: {memory_info.get('memory_limit_gb', 'N/A'):.1f}GB")
        
        # Verify memory limit is reasonable
        if memory_info.get('memory_limit_gb', 0) > 0:
            logger.info(f"✅ Memory limit is set")
        else:
            logger.warning("⚠️ Memory limit not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory awareness test failed: {e}")
        return False

def test_configuration_integration():
    """Test that streaming configuration is properly integrated."""
    logger.info("\n=== Testing Configuration Integration ===")
    
    try:
        # Load configuration
        config = config_loader.get_analysis_config()
        
        # Check streaming settings
        streaming_enabled = config.get('STREAMING_AUDIO_ENABLED', True)
        memory_limit = config.get('STREAMING_MEMORY_LIMIT_PERCENT', 80)
        chunk_duration = config.get('STREAMING_CHUNK_DURATION_SECONDS', 30)
        large_file_threshold = config.get('STREAMING_LARGE_FILE_THRESHOLD_MB', 50)
        
        logger.info(f"✅ Configuration integration test:")
        logger.info(f"   Streaming enabled: {streaming_enabled}")
        logger.info(f"   Memory limit: {memory_limit}%")
        logger.info(f"   Chunk duration: {chunk_duration}s")
        logger.info(f"   Large file threshold: {large_file_threshold}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration integration test failed: {e}")
        return False

def test_file_size_detection():
    """Test that file size detection works correctly."""
    logger.info("\n=== Testing File Size Detection ===")
    
    try:
        loader = StreamingAudioLoader()
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write some data to make it a realistic size
            temp_file.write(b"test data" * 1000)  # ~9KB
            temp_path = temp_file.name
        
        try:
            # Test file size detection
            file_size_mb = loader._get_file_size_mb(temp_path)
            
            logger.info(f"✅ File size detection test:")
            logger.info(f"   File path: {temp_path}")
            logger.info(f"   File size: {file_size_mb:.3f}MB")
            
            # Verify the size is reasonable
            assert file_size_mb > 0, f"File size should be positive: {file_size_mb}"
            assert file_size_mb < 1, f"File size should be small: {file_size_mb}"
            
            return True
            
        finally:
            # Clean up
            os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"❌ File size detection test failed: {e}")
        return False

def run_all_tests():
    """Run all streaming audio tests."""
    logger.info("🚀 Starting Streaming Audio Tests")
    
    tests = [
        ("Streaming Loader Initialization", test_streaming_loader_initialization),
        ("Streaming Loader Configuration", test_streaming_loader_configuration),
        ("Chunk Duration Calculation", test_chunk_duration_calculation),
        ("Memory Awareness", test_memory_awareness),
        ("Configuration Integration", test_configuration_integration),
        ("File Size Detection", test_file_size_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All streaming audio tests passed!")
        return True
    else:
        logger.error(f"⚠️ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 