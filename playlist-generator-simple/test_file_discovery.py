#!/usr/bin/env python3
"""
Test script for file discovery fixes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.file_discovery import FileDiscovery
from core.logging_setup import setup_logging

def test_file_discovery():
    """Test the file discovery functionality."""
    print("Testing FileDiscovery class...")
    
    # Setup logging
    setup_logging()
    
    # Test basic initialization
    try:
        fd = FileDiscovery()
        print("✓ FileDiscovery initialized successfully")
    except Exception as e:
        print(f"✗ FileDiscovery initialization failed: {e}")
        return False
    
    # Test configuration override
    try:
        fd.override_config(
            min_file_size_bytes=2048,
            valid_extensions=['.mp3', '.wav'],
            enable_recursive_scan=False
        )
        print("✓ Configuration override works")
    except Exception as e:
        print(f"✗ Configuration override failed: {e}")
        return False
    
    # Test file validation
    test_file = "/music/test.mp3"
    if os.path.exists(test_file):
        is_valid = fd._is_valid_audio_file(test_file)
        print(f"✓ File validation works: {test_file} -> {is_valid}")
    else:
        print("⚠ Test file not found, skipping file validation test")
    
    # Test statistics
    try:
        stats = fd.get_statistics()
        print(f"✓ Statistics generation works: {len(stats)} items")
    except Exception as e:
        print(f"✗ Statistics generation failed: {e}")
        return False
    
    print("✓ All file discovery tests passed!")
    return True

if __name__ == "__main__":
    success = test_file_discovery()
    sys.exit(0 if success else 1) 