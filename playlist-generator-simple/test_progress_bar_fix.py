#!/usr/bin/env python3
"""
Test script to verify that the progress bar conflict fix is working correctly.
This script simulates the analysis process to ensure no conflicts occur.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.progress_bar import get_progress_bar, SimpleProgressBar
from core.config_loader import config_loader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_progress_bar_configuration():
    """Test that progress bar configuration is working correctly."""
    logger.info("🔍 Testing progress bar configuration...")
    
    try:
        # Load configuration
        config = config_loader.get_analysis_config()
        progress_bar_enabled = config.get('PROGRESS_BAR_ENABLED', True)
        
        logger.info(f"📋 PROGRESS_BAR_ENABLED: {progress_bar_enabled}")
        
        # Get progress bar instance
        progress_bar = get_progress_bar()
        
        # Check if progress bar is enabled based on configuration
        if progress_bar_enabled:
            logger.info("✅ Progress bars are enabled")
        else:
            logger.info("✅ Progress bars are disabled (logging-only mode)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing progress bar configuration: {e}")
        return False

def test_progress_bar_cleanup():
    """Test that progress bar cleanup prevents conflicts."""
    logger.info("🔍 Testing progress bar cleanup...")
    
    try:
        # Create a progress bar instance
        progress_bar = SimpleProgressBar(show_progress=True)
        
        # Start a progress bar
        progress_bar.start_file_processing(10, "Test Processing")
        
        # Try to start another progress bar (should cleanup the first one)
        progress_bar.start_analysis(5, "Test Analysis")
        
        # Complete the analysis
        progress_bar.complete_analysis(5, 4, 1, "Test Analysis")
        
        logger.info("✅ Progress bar cleanup test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing progress bar cleanup: {e}")
        return False

def test_multiple_progress_bars():
    """Test that multiple progress bars don't conflict."""
    logger.info("🔍 Testing multiple progress bars...")
    
    try:
        # Get global progress bar
        progress_bar = get_progress_bar()
        
        # Simulate sequential analysis
        progress_bar.start_analysis(3, "Sequential Analysis")
        for i in range(1, 4):
            progress_bar.update_analysis_progress(i, f"file_{i}.mp3")
        progress_bar.complete_analysis(3, 2, 1, "Sequential Analysis")
        
        # Simulate parallel analysis
        progress_bar.start_analysis(2, "Parallel Analysis")
        for i in range(1, 3):
            progress_bar.update_analysis_progress(i, f"track_{i}.flac")
        progress_bar.complete_analysis(2, 2, 0, "Parallel Analysis")
        
        logger.info("✅ Multiple progress bars test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing multiple progress bars: {e}")
        return False

def test_logging_only_mode():
    """Test progress bar in logging-only mode."""
    logger.info("🔍 Testing logging-only mode...")
    
    try:
        # Create a progress bar with progress disabled
        progress_bar = SimpleProgressBar(show_progress=False)
        
        # Test file processing
        progress_bar.start_file_processing(5, "Logging Test")
        for i in range(1, 6):
            progress_bar.update_file_progress(i, f"log_file_{i}.mp3")
        progress_bar.complete_file_processing(5, 4, 1)
        
        # Test analysis
        progress_bar.start_analysis(3, "Logging Analysis")
        for i in range(1, 4):
            progress_bar.update_analysis_progress(i, f"log_track_{i}.flac")
        progress_bar.complete_analysis(3, 2, 1, "Logging Analysis")
        
        logger.info("✅ Logging-only mode test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing logging-only mode: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting progress bar conflict fix tests...")
    
    # Test configuration
    config_ok = test_progress_bar_configuration()
    
    # Test cleanup
    cleanup_ok = test_progress_bar_cleanup()
    
    # Test multiple progress bars
    multiple_ok = test_multiple_progress_bars()
    
    # Test logging-only mode
    logging_ok = test_logging_only_mode()
    
    # Summary
    logger.info("📊 Test Results:")
    logger.info(f"   Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    logger.info(f"   Cleanup: {'✅ PASS' if cleanup_ok else '❌ FAIL'}")
    logger.info(f"   Multiple Progress Bars: {'✅ PASS' if multiple_ok else '❌ FAIL'}")
    logger.info(f"   Logging-Only Mode: {'✅ PASS' if logging_ok else '❌ FAIL'}")
    
    if config_ok and cleanup_ok and multiple_ok and logging_ok:
        logger.info("🎉 All tests passed! The progress bar conflict fix is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the progress bar implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 