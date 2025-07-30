#!/usr/bin/env python3
"""
Test script for the logging system.
Verifies colored output, file logging, and runtime configuration.
"""

import os
import sys
import time
import tempfile
import shutil
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.logging_setup import (
    setup_logging,
    get_logger,
    change_log_level,
    log_function_call,
    log_info,
    log_error,
    log_performance
)


def cleanup_logging():
    """Clean up logging handlers to avoid file access issues."""
    logger = logging.getLogger('playlista')
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def test_basic_logging():
    """Test basic logging functionality."""
    print("Testing Basic Logging...")
    
    # Setup logging with temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        logger = setup_logging(
            log_level='DEBUG',
            log_dir=temp_dir,
            log_file_prefix='test',
            console_logging=True,
            file_logging=True,
            colored_output=True
        )
        
        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        print("Basic logging test completed")
        
    finally:
        cleanup_logging()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_log_level_changes():
    """Test runtime log level changes."""
    print("\nTesting Log Level Changes...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        logger = setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        # Test initial level
        logger.debug("This should NOT appear (INFO level)")
        logger.info("This should appear")
        
        # Change log level
        print("Changing log level to DEBUG...")
        change_log_level('DEBUG')
        
        # Test new level
        logger.debug("This should NOW appear (DEBUG level)")
        
        print("Log level change test completed")
        
    finally:
        cleanup_logging()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_log_function_call():
    """Test the log_function_call decorator."""
    print("\nTesting Function Call Logging...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        setup_logging(
            log_level='DEBUG',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        @log_function_call
        def test_function(x, y=10):
            """Test function for logging."""
            time.sleep(0.1)  # Simulate work
            return x + y
        
        # Call the decorated function
        result = test_function(5, y=15)
        print(f"Function result: {result}")
        
        # Test function with exception
        @log_function_call
        def failing_function():
            """Function that raises an exception."""
            raise ValueError("Test error")
        
        try:
            failing_function()
        except ValueError:
            print("Exception logging test completed")
            
    finally:
        cleanup_logging()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_convenience_functions():
    """Test convenience logging functions."""
    print("\nTesting Convenience Functions...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        # Test log_info with extra fields
        log_info("User action", user_id=123, action="login", timestamp=time.time())
        
        # Test log_error with exception
        try:
            raise RuntimeError("Test exception")
        except Exception as e:
            log_error("Operation failed", error=e, operation="test", retry_count=3)
        
        # Test log_performance
        start_time = time.time()
        time.sleep(0.1)  # Simulate work
        duration = time.time() - start_time
        log_performance("File processing", duration, files_processed=10, size_mb=5.2)
        
        print("Convenience functions test completed")
        
    finally:
        cleanup_logging()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_file_logging():
    """Test file logging with JSON format."""
    print("\nTesting File Logging...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        setup_logging(
            log_level='DEBUG',
            log_dir=temp_dir,
            log_file_prefix='test',
            console_logging=False,  # Disable console for this test
            file_logging=True
        )
        
        logger = get_logger()
        
        # Log various messages
        logger.info("File logging test message")
        logger.warning("Warning with context", extra={'context': 'test'})
        
        # Check if log file was created
        log_files = list(Path(temp_dir).glob('*.log'))
        if log_files:
            print(f"Log file created: {log_files[0]}")
            
            # Read and display log file content
            with open(log_files[0], 'r') as f:
                content = f.read()
                print("Log file content:")
                print(content[:500] + "..." if len(content) > 500 else content)
        else:
            print("No log file created")
            
    finally:
        cleanup_logging()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_environment_variable_monitoring():
    """Test environment variable monitoring."""
    print("\nTesting Environment Variable Monitoring...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        logger = get_logger()
        
        # Test initial level
        logger.debug("This should NOT appear initially")
        logger.info("This should appear initially")
        
        # Simulate environment variable change
        print("Simulating LOG_LEVEL environment variable change...")
        os.environ['LOG_LEVEL'] = 'DEBUG'
        
        # Wait a moment for the monitor to detect the change
        time.sleep(3)
        
        # Test new level
        logger.debug("This should appear after environment change")
        
        # Clean up
        if 'LOG_LEVEL' in os.environ:
            del os.environ['LOG_LEVEL']
        
        print("Environment variable monitoring test completed")
        
    finally:
        cleanup_logging()
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all logging tests."""
    print("Testing Logging System")
    print("=" * 50)
    
    # Run all tests
    test_basic_logging()
    test_log_level_changes()
    test_log_function_call()
    test_convenience_functions()
    test_file_logging()
    test_environment_variable_monitoring()
    
    print("\nAll logging tests completed!")
    print("\nLogging System Features:")
    print("   Colored console output")
    print("   JSON file logging")
    print("   Runtime log level changes")
    print("   Performance logging")
    print("   Function call logging")
    print("   Environment variable monitoring")


if __name__ == "__main__":
    main() 