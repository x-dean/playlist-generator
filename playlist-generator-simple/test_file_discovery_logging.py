#!/usr/bin/env python3
"""
Test script to verify File Discovery uses our new logging system.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.logging_setup import setup_logging, cleanup_logging
from core.file_discovery import FileDiscovery


def test_file_discovery_logging():
    """Test that File Discovery uses our logging system."""
    print("Testing File Discovery with New Logging System...")
    
    # Setup logging
    temp_dir = tempfile.mkdtemp()
    try:
        setup_logging(
            log_level='DEBUG',
            log_dir=temp_dir,
            log_file_prefix='test_discovery',
            console_logging=True,
            file_logging=True,
            colored_output=True
        )
        
        # Create a test music directory with some files
        test_music_dir = os.path.join(temp_dir, 'music')
        os.makedirs(test_music_dir, exist_ok=True)
        
        # Create some test files
        test_files = [
            'test1.mp3',
            'test2.wav', 
            'test3.flac',
            'invalid.txt',
            'empty.mp3'  # Will be too small
        ]
        
        for filename in test_files:
            filepath = os.path.join(test_music_dir, filename)
            if filename == 'empty.mp3':
                # Create a small file that should be rejected
                with open(filepath, 'w') as f:
                    f.write('x')  # 1 byte file
            else:
                # Create a normal file
                with open(filepath, 'w') as f:
                    f.write('test content ' * 100)  # ~1300 bytes
        
        # Create FileDiscovery instance with test directory
        discovery = FileDiscovery({
            'MUSIC_DIR': test_music_dir,
            'MIN_FILE_SIZE_BYTES': 100,  # Small threshold for testing
            'VALID_EXTENSIONS': ['.mp3', '.wav', '.flac'],
            'HASH_ALGORITHM': 'md5',
            'MAX_RETRY_COUNT': 3,
            'ENABLE_RECURSIVE_SCAN': True,
            'LOG_LEVEL': 'DEBUG',
            'ENABLE_DETAILED_LOGGING': True
        })
        
        # Test file discovery
        print("\nRunning file discovery...")
        discovered_files = discovery.discover_files()
        
        print(f"\nFile discovery completed:")
        print(f"   Files discovered: {len(discovered_files)}")
        for file in discovered_files:
            print(f"   {os.path.basename(file)}")
        
        # Check log files
        log_files = list(Path(temp_dir).glob('*.log'))
        if log_files:
            print(f"\nLog files created:")
            for log_file in log_files:
                print(f"   {log_file.name}")
                
                # Show some log content
                with open(log_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    print(f"   Log entries: {len(lines)}")
                    
                    # Show a few sample log entries
                    print("   Sample log entries:")
                    for i, line in enumerate(lines[:5]):
                        if line.strip():
                            print(f"      {i+1}. {line[:100]}...")
        
        print("\nFile Discovery logging test completed!")
        
    finally:
        cleanup_logging()
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run the file discovery logging test."""
    print("Testing File Discovery with Production Logging")
    print("=" * 60)
    
    test_file_discovery_logging()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main() 