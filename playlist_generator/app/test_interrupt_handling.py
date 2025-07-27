#!/usr/bin/env python3
"""
Test script to verify interrupt handling functionality.
"""

import os
import sys
import time
import signal
import threading

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_interrupt_handling():
    """Test interrupt handling in a simulated long-running process."""
    
    print("Testing interrupt handling...")
    print("Press Ctrl+C to test interrupt handling")
    print("=" * 50)
    
    def long_running_task():
        """Simulate a long-running task."""
        for i in range(100):
            # Check for interrupt every iteration
            try:
                import signal
                # This will raise KeyboardInterrupt if Ctrl+C was pressed
                signal.signal(signal.SIGINT, signal.default_int_handler)
            except KeyboardInterrupt:
                print(f"\n🛑 Interrupt received at iteration {i}!")
                return
            
            print(f"Processing iteration {i}/100...")
            time.sleep(0.5)  # Simulate work
        
        print("Task completed successfully!")
    
    try:
        long_running_task()
    except KeyboardInterrupt:
        print("\n🛑 Interrupt caught in main handler!")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_interrupt_handling() 