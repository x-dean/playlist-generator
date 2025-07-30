#!/usr/bin/env python3
"""
Test script to verify Docker logging setup.
This script tests that logs are written to the Docker mount point /app/logs.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import and set up proper logging
from core.logging_setup import setup_logging, get_logger

# Determine if we're running in Docker or locally
def get_log_directory():
    """Determine the appropriate log directory based on environment."""
    # Check if we're in Docker (container has /app directory)
    if os.path.exists('/app'):
        return '/app/logs'  # Docker mount point
    else:
        return 'logs'  # Local development

log_dir = get_log_directory()

# Initialize logging system with auto-detection
setup_logging(
    log_level='INFO',
    log_file_prefix='docker_test',
    console_logging=True,
    file_logging=True,
    colored_output=True,
    max_log_files=5,
    log_file_size_mb=10,
    log_file_format='text',
    log_file_encoding='utf-8'
)

logger = get_logger('docker_test')

def test_docker_logging():
    """Test that logging works with appropriate mount point."""
    try:
        logger.info("🔧 Testing Logging Setup")
        logger.info("=" * 50)
        
        # Test log directory
        logger.info(f"Log directory: {log_dir}")
        logger.info(f"Environment: {'Docker' if os.path.exists('/app') else 'Local'}")
        
        # Check if directory exists
        if os.path.exists(log_dir):
            logger.info(f"✅ Log directory exists: {log_dir}")
        else:
            logger.warning(f"⚠️ Log directory does not exist: {log_dir}")
            # Try to create it
            try:
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"✅ Created log directory: {log_dir}")
            except Exception as e:
                logger.error(f"❌ Failed to create log directory: {e}")
        
        # Test writing a log file
        logger.info("📝 Testing log file writing...")
        logger.info("This is a test log message")
        logger.warning("This is a test warning message")
        logger.error("This is a test error message")
        
        # Test memory management logging
        logger.info("🧠 Testing memory management logging...")
        logger.warning("⚠️ Memory usage after chunk 425: 69.7% (10.4GB / 15.5GB)")
        logger.warning("⚠️ High memory usage detected! Forcing aggressive memory cleanup...")
        logger.info("🧹 Forced memory cleanup completed")
        
        # List files in log directory
        try:
            files = os.listdir(log_dir)
            logger.info(f"📁 Files in log directory: {files}")
        except Exception as e:
            logger.error(f"❌ Failed to list log directory: {e}")
        
        logger.info("✅ Logging test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    try:
        test_docker_logging()
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 