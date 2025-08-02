#!/usr/bin/env python3
"""
Test script for the improved queue manager implementing producer-consumer pattern.
"""

import os
import time
import tempfile
import shutil
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.improved_queue_manager import ImprovedQueueManager, get_improved_queue_manager
from src.core.logging_setup import get_logger, log_universal

logger = get_logger('playlista.test_improved_queue')


def create_test_audio_files(num_files: int = 5) -> list:
    """Create test audio files for testing."""
    test_files = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        for i in range(num_files):
            # Create a dummy audio file (just a text file for testing)
            file_path = os.path.join(temp_dir, f'test_audio_{i}.mp3')
            with open(file_path, 'w') as f:
                f.write(f'Test audio file {i}\n' * 1000)  # Create a file with some content
            
            test_files.append(file_path)
            log_universal('INFO', 'Test', f'Created test file: {file_path}')
        
        return test_files, temp_dir
        
    except Exception as e:
        log_universal('ERROR', 'Test', f'Failed to create test files: {e}')
        return [], temp_dir


def progress_callback(stats):
    """Progress callback function."""
    total = stats.get('total_tasks', 0)
    completed = stats.get('completed_tasks', 0)
    failed = stats.get('failed_tasks', 0)
    pending = stats.get('pending_tasks', 0)
    processing = stats.get('processing_tasks', 0)
    
    if total > 0:
        progress = (completed + failed) / total * 100
        log_universal('INFO', 'Test', f'Progress: {progress:.1f}% ({completed}/{total} completed, {failed} failed, {pending} pending, {processing} processing)')


def test_improved_queue_manager():
    """Test the improved queue manager."""
    log_universal('INFO', 'Test', 'Testing Improved Queue Manager')
    log_universal('INFO', 'Test', '=' * 50)
    
    # Create test files
    test_files, temp_dir = create_test_audio_files(10)
    if not test_files:
        log_universal('ERROR', 'Test', 'Failed to create test files')
        return False
    
    try:
        # Get queue manager instance
        queue_manager = get_improved_queue_manager()
        
        # Configure queue manager for testing
        queue_manager.max_workers = 2  # Use 2 workers for testing
        queue_manager.queue_size = 3   # Small queue size to test backpressure
        
        log_universal('INFO', 'Test', f'Queue manager configured with {queue_manager.max_workers} workers, queue size {queue_manager.queue_size}')
        
        # Add tasks with different priorities
        task_ids = []
        for i, file_path in enumerate(test_files):
            priority = 10 - i  # Higher priority for earlier files
            task_id = queue_manager.add_task(file_path, priority=priority)
            task_ids.append(task_id)
            log_universal('DEBUG', 'Test', f'Added task {task_id} with priority {priority}')
        
        log_universal('INFO', 'Test', f'Added {len(task_ids)} tasks to queue')
        
        # Start processing
        log_universal('INFO', 'Test', 'Starting queue processing...')
        start_time = time.time()
        
        success = queue_manager.start_processing(progress_callback=progress_callback)
        if not success:
            log_universal('ERROR', 'Test', 'Failed to start queue processing')
            return False
        
        # Wait for processing to complete
        while True:
            stats = queue_manager.get_statistics()
            total = stats.get('total_tasks', 0)
            completed = stats.get('completed_tasks', 0)
            failed = stats.get('failed_tasks', 0)
            pending = stats.get('pending_tasks', 0)
            processing = stats.get('processing_tasks', 0)
            
            # Check if all tasks are done
            if pending == 0 and processing == 0:
                break
            
            time.sleep(1)
        
        # Stop processing
        queue_manager.stop_processing(wait=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get final statistics
        final_stats = queue_manager.get_statistics()
        
        log_universal('INFO', 'Test', 'Processing completed!')
        log_universal('INFO', 'Test', f'Duration: {duration:.2f} seconds')
        log_universal('INFO', 'Test', f'Total tasks: {final_stats.get("total_tasks", 0)}')
        log_universal('INFO', 'Test', f'Completed: {final_stats.get("completed_tasks", 0)}')
        log_universal('INFO', 'Test', f'Failed: {final_stats.get("failed_tasks", 0)}')
        log_universal('INFO', 'Test', f'Retries: {final_stats.get("retry_tasks", 0)}')
        log_universal('INFO', 'Test', f'Average processing time: {final_stats.get("average_processing_time", 0):.2f}s')
        log_universal('INFO', 'Test', f'Throughput: {final_stats.get("throughput", 0):.2f} tasks/second')
        
        # Test individual task status
        if task_ids:
            task_status = queue_manager.get_task_status(task_ids[0])
            if task_status:
                log_universal('INFO', 'Test', f'First task status: {task_status.get("status", "unknown")}')
        
        return True
        
    except Exception as e:
        log_universal('ERROR', 'Test', f'Test failed: {e}')
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            log_universal('INFO', 'Test', f'Cleaned up temporary directory: {temp_dir}')
        except Exception as e:
            log_universal('WARNING', 'Test', f'Failed to cleanup: {e}')


def test_producer_consumer_pattern():
    """Test the producer-consumer pattern specifically."""
    log_universal('INFO', 'Test', 'Testing Producer-Consumer Pattern')
    log_universal('INFO', 'Test', '=' * 50)
    
    # Create test files
    test_files, temp_dir = create_test_audio_files(5)
    if not test_files:
        return False
    
    try:
        # Create queue manager with small queue to test backpressure
        queue_manager = ImprovedQueueManager(
            max_workers=2,
            queue_size=2,  # Small queue to test producer-consumer backpressure
            worker_timeout=60
        )
        
        log_universal('INFO', 'Test', f'Created queue manager with queue size {queue_manager.queue_size}')
        
        # Add tasks
        task_ids = queue_manager.add_tasks(test_files)
        log_universal('INFO', 'Test', f'Added {len(task_ids)} tasks')
        
        # Start processing
        queue_manager.start_processing()
        
        # Monitor queue size
        for i in range(30):  # Monitor for 30 seconds
            stats = queue_manager.get_statistics()
            queue_size = stats.get('queue_size', 0)
            pending = stats.get('pending_tasks', 0)
            processing = stats.get('processing_tasks', 0)
            
            log_universal('INFO', 'Test', f'Queue size: {queue_size}, Pending: {pending}, Processing: {processing}')
            
            if pending == 0 and processing == 0:
                break
                
            time.sleep(1)
        
        # Stop processing
        queue_manager.stop_processing()
        
        final_stats = queue_manager.get_statistics()
        log_universal('INFO', 'Test', f'Producer-Consumer test completed: {final_stats.get("completed_tasks", 0)} completed, {final_stats.get("failed_tasks", 0)} failed')
        
        return True
        
    except Exception as e:
        log_universal('ERROR', 'Test', f'Producer-Consumer test failed: {e}')
        return False
        
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def main():
    """Main test function."""
    log_universal('INFO', 'Test', 'Improved Queue Manager Test Suite')
    log_universal('INFO', 'Test', '=' * 60)
    
    # Test 1: Basic queue manager functionality
    log_universal('INFO', 'Test', 'Test 1: Basic Queue Manager Functionality')
    success1 = test_improved_queue_manager()
    
    # Test 2: Producer-consumer pattern
    log_universal('INFO', 'Test', 'Test 2: Producer-Consumer Pattern')
    success2 = test_producer_consumer_pattern()
    
    # Summary
    log_universal('INFO', 'Test', '=' * 60)
    log_universal('INFO', 'Test', 'Test Results Summary:')
    log_universal('INFO', 'Test', f'Test 1 (Basic Queue Manager): {"✓ PASS" if success1 else "✗ FAIL"}')
    log_universal('INFO', 'Test', f'Test 2 (Producer-Consumer): {"✓ PASS" if success2 else "✗ FAIL"}')
    
    overall_success = success1 and success2
    log_universal('INFO', 'Test', f'Overall: {"✓ ALL TESTS PASSED" if overall_success else "✗ SOME TESTS FAILED"}')
    
    return overall_success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 