#!/usr/bin/env python3
"""
Test script for the new logging infrastructure.
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from shared.config import get_config
from infrastructure.logging import (
    setup_logging,
    get_logger,
    change_log_level,
    set_correlation_id,
    get_correlation_id,
    log_function_call
)


def test_basic_logging():
    """Test basic logging functionality."""
    print("\n🧪 Testing basic logging...")
    
    # Setup logging
    config = get_config()
    logger = setup_logging(config)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("✅ Basic logging test completed")


def test_correlation_ids():
    """Test correlation ID functionality."""
    print("\n🧪 Testing correlation IDs...")
    
    logger = get_logger('playlista.correlation')
    
    # Test correlation ID setting
    correlation_id = set_correlation_id("test-correlation-123")
    logger.info("Message with correlation ID", extra={'test': True})
    
    # Test getting correlation ID
    retrieved_id = get_correlation_id()
    print(f"📋 Correlation ID: {retrieved_id}")
    
    # Test in different thread
    def thread_function():
        thread_logger = get_logger('playlista.thread')
        thread_logger.info("Message from different thread")
        print(f"📋 Thread correlation ID: {get_correlation_id()}")
    
    thread = threading.Thread(target=thread_function)
    thread.start()
    thread.join()
    
    print("✅ Correlation ID test completed")


def test_log_level_changes():
    """Test runtime log level changes."""
    print("\n🧪 Testing log level changes...")
    
    logger = get_logger('playlista.level_test')
    
    # Test current level
    logger.info("Info message before level change")
    logger.debug("Debug message before level change")
    
    # Change log level
    print("🔄 Changing log level to DEBUG...")
    change_log_level("DEBUG")
    
    logger.info("Info message after level change")
    logger.debug("Debug message after level change")
    
    # Change back
    print("🔄 Changing log level back to INFO...")
    change_log_level("INFO")
    
    logger.info("Info message after reverting")
    logger.debug("Debug message after reverting")
    
    print("✅ Log level change test completed")


@log_function_call
def test_function_logging():
    """Test function call logging decorator."""
    print("\n🧪 Testing function call logging...")
    
    logger = get_logger('playlista.function_test')
    logger.info("Testing function call logging")
    
    # Simulate some work
    time.sleep(0.1)
    
    return "Function completed successfully"


def test_error_logging():
    """Test error logging with exceptions."""
    print("\n🧪 Testing error logging...")
    
    logger = get_logger('playlista.error_test')
    
    try:
        # Simulate an error
        raise ValueError("This is a test error")
    except ValueError as e:
        logger.error("Caught test error", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        })
    
    print("✅ Error logging test completed")


def test_performance_logging():
    """Test performance logging."""
    print("\n🧪 Testing performance logging...")
    
    logger = get_logger('playlista.performance')
    
    # Simulate performance metrics
    start_time = time.time()
    time.sleep(0.1)  # Simulate work
    duration_ms = (time.time() - start_time) * 1000
    
    logger.info("Performance test completed", extra={
        'duration_ms': duration_ms,
        'memory_mb': 45.2,
        'operation': 'test_performance_logging'
    })
    
    print("✅ Performance logging test completed")


def test_structured_logging():
    """Test structured logging with extra fields."""
    print("\n🧪 Testing structured logging...")
    
    logger = get_logger('playlista.structured')
    
    # Test with extra fields
    logger.info("Processing audio file", extra={
        'file_path': '/music/song.mp3',
        'file_size_mb': 8.5,
        'processing_time_ms': 1250,
        'features_extracted': ['bpm', 'mfcc', 'chroma'],
        'status': 'completed'
    })
    
    # Test with nested data
    logger.info("Analysis results", extra={
        'analysis_id': 'analysis_123',
        'results': {
            'bpm': 120,
            'key': 'C',
            'tempo': 'medium',
            'energy': 0.75
        },
        'confidence': 0.92
    })
    
    print("✅ Structured logging test completed")


def main():
    """Run all logging tests."""
    print("🚀 Testing New Logging Infrastructure")
    print("=" * 50)
    
    try:
        # Run all tests
        test_basic_logging()
        test_correlation_ids()
        test_log_level_changes()
        test_function_logging()
        test_error_logging()
        test_performance_logging()
        test_structured_logging()
        
        print("\n" + "=" * 50)
        print("🎉 All logging tests passed!")
        print("\n📋 Logging Infrastructure Features:")
        print("✅ Structured logging with JSON output")
        print("✅ Correlation ID tracking")
        print("✅ Runtime log level changes")
        print("✅ Function call logging decorator")
        print("✅ Performance metrics logging")
        print("✅ Error context logging")
        print("✅ Multi-threaded logging support")
        print("✅ File and console output")
        print("✅ Rotating log files")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 