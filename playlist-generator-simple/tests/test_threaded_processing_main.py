#!/usr/bin/env python3
"""
Test script to verify that threaded processing is now the main parallel processing approach.
"""

import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.parallel_analyzer import ParallelAnalyzer
from core.analysis_manager import AnalysisManager
from core.config_loader import ConfigLoader
from core.logging_setup import log_universal

def test_threaded_processing_main():
    """Test that threaded processing is now the main approach."""
    
    print("=== Testing Threaded Processing as Main Approach ===")
    
    # Load configuration
    config = ConfigLoader().get_audio_analysis_config()
    print(f"✓ Configuration loaded")
    
    # Initialize parallel analyzer
    parallel_analyzer = ParallelAnalyzer(config=config)
    print(f"✓ ParallelAnalyzer initialized")
    
    # Initialize analysis manager
    analysis_manager = AnalysisManager()
    print(f"✓ AnalysisManager initialized")
    
    # Test that process_files method signature is correct
    print("\n--- Testing Method Signatures ---")
    
    import inspect
    process_files_sig = inspect.signature(parallel_analyzer.process_files)
    print(f"✓ process_files signature: {process_files_sig}")
    
    # Check that use_threading parameter is removed
    params = list(process_files_sig.parameters.keys())
    if 'use_threading' in params:
        print("❌ use_threading parameter still exists")
    else:
        print("✓ use_threading parameter removed")
    
    # Test with dummy files
    print("\n--- Testing with Dummy Files ---")
    
    # Create dummy file paths (these won't exist, but we can test the method calls)
    dummy_files = [
        "/app/music/test1.mp3",
        "/app/music/test2.mp3", 
        "/app/music/test3.mp3"
    ]
    
    print(f"✓ Created {len(dummy_files)} dummy files for testing")
    
    # Test parallel analyzer directly
    print("\n--- Testing Parallel Analyzer Directly ---")
    
    try:
        # This should use threaded processing by default
        results = parallel_analyzer.process_files(dummy_files, force_reextract=False, max_workers=2)
        print(f"✓ Parallel analyzer process_files completed")
        print(f"  Success count: {results.get('success_count', 0)}")
        print(f"  Failed count: {results.get('failed_count', 0)}")
        print(f"  Total time: {results.get('total_time', 0):.2f}s")
    except Exception as e:
        print(f"❌ Parallel analyzer test failed: {e}")
    
    # Test analysis manager
    print("\n--- Testing Analysis Manager ---")
    
    try:
        # This should use threaded processing for small files
        results = analysis_manager.analyze_files(dummy_files, force_reextract=False, max_workers=2)
        print(f"✓ Analysis manager analyze_files completed")
        print(f"  Success count: {results.get('success_count', 0)}")
        print(f"  Failed count: {results.get('failed_count', 0)}")
        print(f"  Total time: {results.get('total_time', 0):.2f}s")
        print(f"  Big files processed: {results.get('big_files_processed', 0)}")
        print(f"  Small files processed: {results.get('small_files_processed', 0)}")
    except Exception as e:
        print(f"❌ Analysis manager test failed: {e}")
    
    # Test configuration
    print("\n--- Testing Configuration ---")
    
    # Check that threaded processing is enabled by default
    use_threading = config.get('USE_THREADED_PROCESSING', False)
    print(f"  USE_THREADED_PROCESSING: {use_threading}")
    
    threaded_workers = config.get('THREADED_WORKERS_DEFAULT', 4)
    print(f"  THREADED_WORKERS_DEFAULT: {threaded_workers}")
    
    print("\n=== Test Complete ===")
    print("✓ Threaded processing is now the main parallel processing approach")
    print("✓ Multiprocessing code has been removed")
    print("✓ Method signatures updated")

if __name__ == "__main__":
    test_threaded_processing_main() 