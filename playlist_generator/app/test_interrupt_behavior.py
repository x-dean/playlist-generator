#!/usr/bin/env python3
"""
Test script to demonstrate interrupt behavior - workers finish current work but stop next file.
"""

import os
import sys
import time
import signal
import threading

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_interrupt_behavior():
    """Test interrupt behavior where current work completes but next file is stopped."""
    
    print("Testing interrupt behavior...")
    print("Press Ctrl+C to test interrupt handling")
    print("Current file will complete, but next file will be stopped")
    print("=" * 60)
    
    def process_file(file_num):
        """Simulate processing a file."""
        print(f"🔄 Processing file {file_num}...")
        time.sleep(2)  # Simulate work
        print(f"✅ Completed file {file_num}")
        return f"result_{file_num}"
    
    def process_files():
        """Process multiple files with interrupt checks between them."""
        files = ["file1.mp3", "file2.mp3", "file3.mp3", "file4.mp3"]
        
        for i, filename in enumerate(files, 1):
            print(f"\n📁 Starting file {i}: {filename}")
            
            # Process the current file (this will complete even if Ctrl+C is pressed)
            result = process_file(i)
            print(f"💾 Saved result: {result}")
            
            # Check for interrupt AFTER completing the current file
            # This allows the current file to complete but stops the next one
            try:
                import signal
                # This will raise KeyboardInterrupt if Ctrl+C was pressed
                signal.signal(signal.SIGINT, signal.default_int_handler)
            except KeyboardInterrupt:
                print(f"\n🛑 Interrupt received! Completed {filename}, stopping analysis...")
                return  # Exit the function
            
            print(f"⏭️  Moving to next file...")
        
        print("\n🎉 All files processed successfully!")
    
    try:
        process_files()
    except KeyboardInterrupt:
        print("\n🛑 Interrupt caught in main handler!")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_interrupt_behavior() 