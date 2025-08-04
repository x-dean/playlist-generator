#!/usr/bin/env python3
"""
Test script to verify that sequential analyzer properly handles failures
without exiting the app using multiprocessing isolation.
"""

import os
import sys
import tempfile
import signal
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.sequential_analyzer import SequentialAnalyzer
from core.database import DatabaseManager
from core.logging_setup import setup_logging, log_universal

def create_test_files():
    """Create test files for testing."""
    test_files = []
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid audio file (empty file for testing)
        valid_file = os.path.join(temp_dir, "test_valid.mp3")
        with open(valid_file, 'w') as f:
            f.write("fake audio content")
        test_files.append(valid_file)
        
        # Create a non-existent file
        non_existent_file = os.path.join(temp_dir, "non_existent.mp3")
        test_files.append(non_existent_file)
        
        # Create a file that will cause analysis to fail
        invalid_file = os.path.join(temp_dir, "test_invalid.txt")
        with open(invalid_file, 'w') as f:
            f.write("this is not an audio file")
        test_files.append(invalid_file)
        
        return test_files

def test_sequential_failure_handling():
    """Test that sequential analyzer handles failures properly with multiprocessing isolation."""
    print("Testing sequential analyzer failure handling with multiprocessing isolation...")
    
    # Setup logging
    setup_logging(log_level='INFO', console_logging=True, file_logging=False)
    
    # Create test files
    test_files = create_test_files()
    
    print(f"Created {len(test_files)} test files:")
    for file in test_files:
        print(f"  - {file} (exists: {os.path.exists(file)})")
    
    # Initialize database and analyzer
    db_manager = DatabaseManager()
    analyzer = SequentialAnalyzer(db_manager=db_manager)
    
    print("\nStarting sequential analysis with multiprocessing isolation...")
    
    try:
        # Process files - this should not exit the app even if files fail
        results = analyzer.process_files(test_files, force_reextract=False)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results:")
        print(f"  Success count: {results['success_count']}")
        print(f"  Failed count: {results['failed_count']}")
        print(f"  Total time: {results['total_time']:.2f}s")
        
        # Check that the app didn't exit
        print(f"\nApp is still running - failure handling works correctly!")
        
        # Check failed files in database
        failed_files = db_manager.get_failed_analysis_files()
        print(f"\nFailed files in database: {len(failed_files)}")
        for failed in failed_files:
            print(f"  - {failed['filename']}: {failed['error_message']}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Sequential analyzer caused an unhandled exception: {e}")
        return False

def test_process_isolation():
    """Test that process isolation prevents main app crashes."""
    print("\nTesting process isolation...")
    
    # Setup logging
    setup_logging(log_level='INFO', console_logging=True, file_logging=False)
    
    # Initialize analyzer
    db_manager = DatabaseManager()
    analyzer = SequentialAnalyzer(db_manager=db_manager)
    
    # Create a file that will definitely cause an error
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is not an audio file and will cause analysis to fail")
        problematic_file = f.name
    
    print(f"Created problematic file: {problematic_file}")
    
    try:
        # This should not crash the main process
        result = analyzer._process_single_file_multiprocessing(problematic_file, force_reextract=False)
        
        print(f"Process isolation test completed - main process survived!")
        print(f"Result: {result}")
        
        # Clean up
        os.unlink(problematic_file)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Process isolation failed - main process crashed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sequential Analyzer Failure Handling")
    print("=" * 60)
    
    # Test 1: Basic failure handling
    success1 = test_sequential_failure_handling()
    
    # Test 2: Process isolation
    success2 = test_process_isolation()
    
    if success1 and success2:
        print("\n✅ All tests PASSED - Sequential analyzer handles failures correctly with process isolation")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED - Sequential analyzer has issues with failure handling")
        sys.exit(1) 