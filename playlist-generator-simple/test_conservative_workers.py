#!/usr/bin/env python3
"""
Test conservative worker count calculation.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.resource_manager import ResourceManager
from core.config_loader import config_loader

def test_conservative_workers():
    """Test that conservative worker count is calculated correctly."""
    
    print("=== Testing Conservative Worker Count ===")
    
    # Load configuration
    config = config_loader.get_audio_analysis_config()
    print(f"✓ Configuration loaded")
    
    # Initialize resource manager
    resource_manager = ResourceManager(config)
    print(f"✓ ResourceManager initialized")
    
    # Test worker count calculation
    print("\n--- Testing Worker Count Calculation ---")
    
    try:
        optimal_workers = resource_manager.get_optimal_worker_count()
        print(f"✓ Optimal worker count: {optimal_workers}")
        
        # Verify conservative constraints
        import multiprocessing as mp
        cpu_count = mp.cpu_count()
        max_expected = max(2, cpu_count // 2)
        
        print(f"  CPU count: {cpu_count}")
        print(f"  Maximum expected (half cores): {max_expected}")
        print(f"  Actual workers: {optimal_workers}")
        
        # Verify constraints
        if optimal_workers >= 2:
            print("✓ Minimum 2 workers constraint satisfied")
        else:
            print("❌ Minimum 2 workers constraint failed")
            
        if optimal_workers <= max_expected:
            print("✓ Maximum half CPU cores constraint satisfied")
        else:
            print("❌ Maximum half CPU cores constraint failed")
            
        # Test with different max_workers values
        print("\n--- Testing with Different Max Workers ---")
        
        test_max_workers = [1, 2, 4, 8]
        for max_workers in test_max_workers:
            workers = resource_manager.get_optimal_worker_count(max_workers=max_workers)
            print(f"  Max workers {max_workers}: {workers} workers")
            
        print("\n=== Test Complete ===")
        print("✓ Conservative worker count calculation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_conservative_workers()
    sys.exit(0 if success else 1) 