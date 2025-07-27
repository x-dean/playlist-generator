#!/usr/bin/env python3
"""
Test script for large file processing functionality.
"""

import os
import sys
import time
import logging
from music_analyzer.sequential import SequentialProcessor, LargeFileProcessor
from music_analyzer.feature_extractor import AudioAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_large_file_processor():
    """Test the large file processor with a sample file."""
    print("Testing Large File Processor...")
    
    # Create a test processor
    processor = LargeFileProcessor()
    
    # Find a large file in the music directory
    music_dir = '/music'
    large_files = []
    
    if os.path.exists(music_dir):
        for root, dirs, files in os.walk(music_dir):
            for file in files:
                if file.endswith(('.mp3', '.flac', '.m4a', '.wav')):
                    file_path = os.path.join(root, file)
                    try:
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        if size_mb > 50:  # Files larger than 50MB
                            large_files.append((file_path, size_mb))
                    except:
                        pass
    
    if not large_files:
        print("No large files found for testing")
        return
    
    # Sort by size (largest first)
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(large_files)} large files:")
    for file_path, size_mb in large_files[:5]:  # Show top 5
        print(f"  {os.path.basename(file_path)} ({size_mb:.1f}MB)")
    
    # Test with the largest file
    test_file, size_mb = large_files[0]
    print(f"\nTesting with: {os.path.basename(test_file)} ({size_mb:.1f}MB)")
    
    start_time = time.time()
    try:
        features, db_write_success, file_hash = processor.process_large_file(test_file, force_reextract=False)
        processing_time = time.time() - start_time
        
        if features and db_write_success:
            print(f"✅ SUCCESS: Processed in {processing_time:.1f}s")
            print(f"   Features extracted: {len(features)}")
            print(f"   Database write: {db_write_success}")
        else:
            print(f"❌ FAILED: Processing failed after {processing_time:.1f}s")
            
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ ERROR: {str(e)} after {processing_time:.1f}s")

def test_sequential_processor():
    """Test the sequential processor with large file detection."""
    print("\nTesting Sequential Processor...")
    
    # Create a test processor
    processor = SequentialProcessor()
    
    # Find large files
    music_dir = '/music'
    large_files = []
    
    if os.path.exists(music_dir):
        for root, dirs, files in os.walk(music_dir):
            for file in files:
                if file.endswith(('.mp3', '.flac', '.m4a', '.wav')):
                    file_path = os.path.join(root, file)
                    try:
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        if size_mb > 50:  # Files larger than 50MB
                            large_files.append(file_path)
                    except:
                        pass
    
    if not large_files:
        print("No large files found for testing")
        return
    
    # Test with first large file
    test_file = large_files[0]
    size_mb = os.path.getsize(test_file) / (1024 * 1024)
    print(f"Testing with: {os.path.basename(test_file)} ({size_mb:.1f}MB)")
    
    start_time = time.time()
    try:
        # Process just this one file
        for features, filepath, db_write_success in processor.process([test_file], workers=1, force_reextract=False):
            processing_time = time.time() - start_time
            
            if features and db_write_success:
                print(f"✅ SUCCESS: Processed in {processing_time:.1f}s")
                print(f"   Features extracted: {len(features)}")
                print(f"   Database write: {db_write_success}")
            else:
                print(f"❌ FAILED: Processing failed after {processing_time:.1f}s")
            break
            
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ ERROR: {str(e)} after {processing_time:.1f}s")

if __name__ == "__main__":
    print("Large File Processing Test")
    print("=" * 50)
    
    test_large_file_processor()
    test_sequential_processor()
    
    print("\nTest completed!") 