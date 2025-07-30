#!/usr/bin/env python3
"""
Test script for the simple progress bar functionality.
Demonstrates how the progress bar works with file processing.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.progress_bar import SimpleProgressBar, get_progress_bar


def test_progress_bar():
    """Test the progress bar functionality."""
    print("🧪 Testing Simple Progress Bar")
    print("=" * 50)
    
    # Create a progress bar instance
    progress_bar = SimpleProgressBar(show_progress=True)
    
    # Test 1: File processing simulation
    print("\n📁 Test 1: File Processing Simulation")
    total_files = 10
    progress_bar.start_file_processing(total_files, "Processing Music Files")
    
    for i in range(1, total_files + 1):
        # Simulate file processing
        time.sleep(0.2)
        filename = f"song_{i:02d}.mp3"
        progress_bar.update_file_progress(i, filename)
    
    progress_bar.complete_file_processing(total_files, 8, 2)
    
    # Test 2: Analysis simulation
    print("\n🎵 Test 2: Analysis Simulation")
    total_files = 15
    progress_bar.start_analysis(total_files, "Sequential Analysis")
    
    for i in range(1, total_files + 1):
        # Simulate analysis
        time.sleep(0.15)
        filename = f"track_{i:02d}.flac"
        progress_bar.update_analysis_progress(i, filename)
    
    progress_bar.complete_analysis(total_files, 13, 2, "Sequential Analysis")
    
    # Test 3: Status messages
    print("\n💬 Test 3: Status Messages")
    progress_bar.show_status("Starting analysis process...", "blue")
    time.sleep(0.5)
    progress_bar.show_success("Analysis completed successfully!")
    time.sleep(0.5)
    progress_bar.show_warning("Some files were skipped due to size")
    time.sleep(0.5)
    progress_bar.show_error("Failed to process corrupt file")
    
    print("\n✅ Progress bar tests completed!")


def test_global_progress_bar():
    """Test the global progress bar functionality."""
    print("\n🌐 Test 4: Global Progress Bar")
    print("=" * 50)
    
    # Get global progress bar
    progress_bar = get_progress_bar(show_progress=True)
    
    # Test parallel processing simulation
    total_files = 8
    progress_bar.start_analysis(total_files, "Parallel Analysis")
    
    for i in range(1, total_files + 1):
        # Simulate parallel processing
        time.sleep(0.1)
        filename = f"audio_{i:02d}.wav"
        progress_bar.update_analysis_progress(i, filename)
    
    progress_bar.complete_analysis(total_files, 7, 1, "Parallel Analysis")
    
    print("\n✅ Global progress bar test completed!")


def test_logging_only_mode():
    """Test progress bar in logging-only mode."""
    print("\n📝 Test 5: Logging-Only Mode")
    print("=" * 50)
    
    # Create progress bar with progress disabled
    progress_bar = SimpleProgressBar(show_progress=False)
    
    # Test file processing (should only log)
    total_files = 5
    progress_bar.start_file_processing(total_files, "Logging-Only Processing")
    
    for i in range(1, total_files + 1):
        time.sleep(0.1)
        filename = f"file_{i:02d}.mp3"
        progress_bar.update_file_progress(i, filename)
    
    progress_bar.complete_file_processing(total_files, 4, 1)
    
    print("\n✅ Logging-only mode test completed!")


if __name__ == "__main__":
    try:
        test_progress_bar()
        test_global_progress_bar()
        test_logging_only_mode()
        
        print("\n🎉 All progress bar tests completed successfully!")
        print("\n📋 Features demonstrated:")
        print("   ✅ File processing progress")
        print("   ✅ Analysis progress (sequential and parallel)")
        print("   ✅ Status messages (success, warning, error)")
        print("   ✅ Global progress bar instance")
        print("   ✅ Logging-only mode")
        print("   ✅ Rich console output with colors and tables")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        sys.exit(1) 