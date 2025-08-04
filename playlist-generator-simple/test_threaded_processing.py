#!/usr/bin/env python3
"""
Test script for threaded processing implementation.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_threaded_processing():
    """Test the threaded processing implementation."""
    try:
        print("Starting threaded processing test...")
        
        # Test imports
        try:
            from core.parallel_analyzer import ParallelAnalyzer
            print("✓ ParallelAnalyzer imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import ParallelAnalyzer: {e}")
            return False
            
        try:
            from core.database import DatabaseManager
            print("✓ DatabaseManager imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import DatabaseManager: {e}")
            return False
            
        try:
            from core.resource_manager import ResourceManager
            print("✓ ResourceManager imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import ResourceManager: {e}")
            return False
            
        try:
            from core.logging_setup import log_universal
            print("✓ Logging setup imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import logging setup: {e}")
            return False
        
        print("All imports successful!")
        
        # Initialize components
        print("Initializing components...")
        db_manager = DatabaseManager()
        resource_manager = ResourceManager()
        
        # Load config for testing
        from core.config_loader import config_loader
        config = config_loader.get_audio_analysis_config()
        
        parallel_analyzer = ParallelAnalyzer(
            db_manager=db_manager,
            resource_manager=resource_manager,
            config=config
        )
        print("✓ Components initialized")
        
        # Create test files list (use existing music files if available)
        test_files = []
        music_dir = '/app/music'
        if os.path.exists(music_dir):
            print(f"Found music directory: {music_dir}")
            for root, dirs, files in os.walk(music_dir):
                for file in files:
                    if file.lower().endswith(('.mp3', '.wav', '.flac')):
                        test_files.append(os.path.join(root, file))
                        if len(test_files) >= 5:  # Limit to 5 files for testing
                            break
                if len(test_files) >= 5:
                    break
        else:
            print(f"Music directory not found: {music_dir}")
        
        if not test_files:
            print("No test files found, creating dummy test")
            # Create dummy test with non-existent files for testing the framework
            test_files = [
                '/app/music/test1.mp3',
                '/app/music/test2.wav',
                '/app/music/test3.flac'
            ]
        
        print(f'Testing with {len(test_files)} files')
        
        # Test threaded processing
        print("Testing threaded processing...")
        start_time = time.time()
        results = parallel_analyzer.process_files(
            files=test_files,
            force_reextract=False,
            max_workers=2,
            use_threading=True
        )
        end_time = time.time()
        
        print(f'Threaded processing completed in {end_time - start_time:.2f}s')
        print(f'Threaded results: {results}')
        
        # Test regular processing for comparison
        print("Testing regular processing...")
        start_time = time.time()
        results_regular = parallel_analyzer.process_files(
            files=test_files,
            force_reextract=False,
            max_workers=2,
            use_threading=False
        )
        end_time = time.time()
        
        print(f'Regular processing completed in {end_time - start_time:.2f}s')
        print(f'Regular results: {results_regular}')
        
        print("✓ All tests completed successfully")
        return True
        
    except Exception as e:
        print(f'✗ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("THREADED PROCESSING TEST")
    print("=" * 50)
    success = test_threaded_processing()
    print("=" * 50)
    print(f"TEST RESULT: {'PASSED' if success else 'FAILED'}")
    print("=" * 50)
    sys.exit(0 if success else 1) 