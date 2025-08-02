#!/usr/bin/env python3
"""
Test script to verify Queue Manager environment setup.
Tests if spawned worker processes can access audio processing libraries.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.queue_manager import QueueManager, get_queue_manager
from src.core.database import DatabaseManager
from src.core.resource_manager import ResourceManager


def create_test_audio_file(size_mb: float = 0.1) -> str:
    """Create a test audio file for processing."""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "test_audio.mp3")
    
    # Create a file with specified size
    with open(file_path, 'wb') as f:
        # Write random data to simulate audio file
        f.write(os.urandom(int(size_mb * 1024 * 1024)))
    
    print(f"Created test audio file: {file_path} ({size_mb}MB)")
    return file_path


def test_environment_imports():
    """Test if worker processes can import required libraries."""
    print("\n=== Testing Environment Imports ===")
    
    # Create queue manager
    queue_manager = QueueManager(
        enable_load_balancing=False,  # Disable for simple test
        max_workers=1
    )
    
    # Start processing
    success = queue_manager.start_processing()
    if not success:
        print("Failed to start queue processing")
        return False
    
    print(f"Queue manager started with {queue_manager.max_workers} workers")
    
    # Create test file
    test_file = create_test_audio_file(0.05)  # Small file
    
    # Add task
    task_id = queue_manager.add_task(test_file)
    print(f"Added task: {task_id}")
    
    # Monitor for completion
    start_time = time.time()
    while time.time() - start_time < 60:  # Wait up to 60 seconds
        stats = queue_manager.get_statistics()
        
        print(f"Queue status: {stats['queue_size']} pending, "
              f"{stats['completed_tasks']} completed, "
              f"{stats['failed_tasks']} failed")
        
        if stats['completed_tasks'] > 0:
            print("✓ Task completed successfully!")
            queue_manager.stop_processing()
            return True
        elif stats['failed_tasks'] > 0:
            print("✗ Task failed!")
            # Get failed task details
            failed_tasks = list(queue_manager.failed_tasks.values())
            if failed_tasks:
                print(f"Error: {failed_tasks[0].error_message}")
            queue_manager.stop_processing()
            return False
        
        time.sleep(2)
    
    print("✗ Test timed out!")
    queue_manager.stop_processing()
    return False


def test_library_availability():
    """Test if required audio processing libraries are available."""
    print("\n=== Testing Library Availability ===")
    
    try:
        # Test Essentia
        try:
            import essentia
            print("✓ Essentia available")
        except ImportError:
            print("✗ Essentia not available")
        
        # Test librosa
        try:
            import librosa
            print("✓ Librosa available")
        except ImportError:
            print("✗ Librosa not available")
        
        # Test mutagen
        try:
            import mutagen
            print("✓ Mutagen available")
        except ImportError:
            print("✗ Mutagen not available")
        
        # Test tensorflow
        try:
            import tensorflow
            print("✓ TensorFlow available")
        except ImportError:
            print("✗ TensorFlow not available")
        
        # Test numpy
        try:
            import numpy
            print("✓ NumPy available")
        except ImportError:
            print("✗ NumPy not available")
        
        # Test scipy
        try:
            import scipy
            print("✓ SciPy available")
        except ImportError:
            print("✗ SciPy not available")
        
        return True
        
    except Exception as e:
        print(f"Error testing libraries: {e}")
        return False


def test_audio_analyzer_import():
    """Test if AudioAnalyzer can be imported and instantiated."""
    print("\n=== Testing AudioAnalyzer Import ===")
    
    try:
        from src.core.audio_analyzer import AudioAnalyzer
        print("✓ AudioAnalyzer imported successfully")
        
        # Try to create an instance
        analyzer = AudioAnalyzer()
        print("✓ AudioAnalyzer instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ AudioAnalyzer import/instantiation failed: {e}")
        return False


# Create a simple test function that checks environment
def check_worker_environment():
    import os
    import sys
    
    print(f"Worker process PID: {os.getpid()}")
    print(f"Worker Python path: {sys.path[:3]}...")
    print(f"Worker PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Try to import required modules
    try:
        from src.core.audio_analyzer import AudioAnalyzer
        print("✓ AudioAnalyzer import successful in worker")
        return True
    except Exception as e:
        print(f"✗ AudioAnalyzer import failed in worker: {e}")
        return False


def test_worker_process_environment():
    """Test worker process environment setup."""
    print("\n=== Testing Worker Process Environment ===")
    
    # Test with ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    
    try:
        # Create executor without env parameter
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context('spawn')
        ) as executor:
            future = executor.submit(check_worker_environment)
            result = future.result(timeout=30)
            
            if result:
                print("✓ Worker process environment test passed")
                return True
            else:
                print("✗ Worker process environment test failed")
                return False
                
    except Exception as e:
        print(f"✗ Worker process environment test error: {e}")
        return False


def main():
    """Run all environment tests."""
    print("Queue Manager Environment Test Suite")
    print("=" * 50)
    
    tests = [
        ("Library Availability", test_library_availability),
        ("AudioAnalyzer Import", test_audio_analyzer_import),
        ("Worker Process Environment", test_worker_process_environment),
        ("Environment Imports", test_environment_imports),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Queue manager should work correctly.")
    else:
        print("⚠️  Some tests failed. Queue manager may have issues.")
    
    return passed == len(results)


if __name__ == "__main__":
    main() 