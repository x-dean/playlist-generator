#!/usr/bin/env python3
"""
Test script for Queue Manager Load Balancing Features.
Demonstrates dynamic worker spawning based on criteria.
"""

import os
import sys
import time
import tempfile
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.queue_manager import QueueManager, get_queue_manager
from src.core.database import DatabaseManager
from src.core.resource_manager import ResourceManager


def create_test_files(count: int, size_mb: float = 1.0) -> list:
    """Create test files for processing."""
    test_files = []
    temp_dir = tempfile.mkdtemp()
    
    for i in range(count):
        file_path = os.path.join(temp_dir, f"test_audio_{i:03d}.mp3")
        
        # Create a file with specified size
        with open(file_path, 'wb') as f:
            # Write random data to simulate audio file
            f.write(os.urandom(int(size_mb * 1024 * 1024)))
        
        test_files.append(file_path)
    
    print(f"Created {count} test files in {temp_dir}")
    return test_files


def test_load_balancing_basic():
    """Test basic load balancing functionality."""
    print("\n=== Testing Basic Load Balancing ===")
    
    # Create queue manager with load balancing enabled
    queue_manager = QueueManager(
        enable_load_balancing=True,
        min_workers=1,
        max_workers_limit=4,
        worker_spawn_threshold=0.3,  # Lower threshold for testing
        worker_reduce_threshold=0.1,
        cpu_threshold_for_spawn=90,  # High threshold for testing
        memory_threshold_for_spawn=85,
        worker_spawn_cooldown=10,  # Short cooldown for testing
        worker_reduce_cooldown=20
    )
    
    # Start processing
    success = queue_manager.start_processing()
    if not success:
        print("Failed to start queue processing")
        return
    
    print(f"Initial workers: {queue_manager._current_workers}")
    
    # Add many tasks to trigger worker spawning
    test_files = create_test_files(20, 0.1)  # Small files for quick processing
    
    task_ids = queue_manager.add_tasks(test_files)
    print(f"Added {len(task_ids)} tasks to queue")
    
    # Monitor load balancing
    start_time = time.time()
    while time.time() - start_time < 30:  # Monitor for 30 seconds
        stats = queue_manager.get_statistics()
        lb_stats = stats.get('load_balancing', {})
        
        print(f"Queue size: {stats['queue_size']}, "
              f"Workers: {stats['current_workers']}, "
              f"Spawns: {lb_stats.get('spawns', 0)}, "
              f"Reductions: {lb_stats.get('reductions', 0)}")
        
        if stats['queue_size'] == 0:
            break
        
        time.sleep(2)
    
    # Get final statistics
    final_stats = queue_manager.get_statistics()
    lb_stats = final_stats.get('load_balancing', {})
    
    print(f"\nFinal Load Balancing Stats:")
    print(f"  Initial workers: 1")
    print(f"  Final workers: {final_stats['current_workers']}")
    print(f"  Worker spawns: {lb_stats.get('spawns', 0)}")
    print(f"  Worker reductions: {lb_stats.get('reductions', 0)}")
    print(f"  Completed tasks: {final_stats['completed_tasks']}")
    print(f"  Failed tasks: {final_stats['failed_tasks']}")
    
    queue_manager.stop_processing()
    print("Basic load balancing test completed")


def test_load_balancing_criteria():
    """Test load balancing with specific criteria."""
    print("\n=== Testing Load Balancing Criteria ===")
    
    # Create queue manager with strict criteria
    queue_manager = QueueManager(
        enable_load_balancing=True,
        min_workers=1,
        max_workers_limit=6,
        worker_spawn_threshold=0.5,  # 50% queue utilization required
        worker_reduce_threshold=0.2,  # 20% queue utilization for reduction
        cpu_threshold_for_spawn=70,   # CPU must be below 70%
        memory_threshold_for_spawn=70, # Memory must be below 70%
        worker_spawn_cooldown=15,     # 15 second cooldown
        worker_reduce_cooldown=30     # 30 second cooldown
    )
    
    # Start processing
    success = queue_manager.start_processing()
    if not success:
        print("Failed to start queue processing")
        return
    
    print(f"Initial workers: {queue_manager._current_workers}")
    print(f"Spawn criteria: Queue >50%, CPU <70%, Memory <70%")
    print(f"Reduce criteria: Queue <20%")
    
    # Add tasks in batches to test different scenarios
    test_files = create_test_files(30, 0.05)  # Very small files
    
    # Add first batch
    batch1 = test_files[:10]
    task_ids1 = queue_manager.add_tasks(batch1)
    print(f"Added batch 1: {len(task_ids1)} tasks")
    
    time.sleep(5)  # Let some processing happen
    
    # Add second batch
    batch2 = test_files[10:20]
    task_ids2 = queue_manager.add_tasks(batch2)
    print(f"Added batch 2: {len(task_ids2)} tasks")
    
    time.sleep(5)  # Let more processing happen
    
    # Add third batch
    batch3 = test_files[20:]
    task_ids3 = queue_manager.add_tasks(batch3)
    print(f"Added batch 3: {len(task_ids3)} tasks")
    
    # Monitor for criteria-based spawning
    start_time = time.time()
    while time.time() - start_time < 45:  # Monitor for 45 seconds
        stats = queue_manager.get_statistics()
        lb_stats = stats.get('load_balancing', {})
        
        # Get current resource usage
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        print(f"Queue: {stats['queue_size']}, "
              f"Workers: {stats['current_workers']}, "
              f"CPU: {cpu_percent:.1f}%, "
              f"Memory: {memory_percent:.1f}%, "
              f"Spawns: {lb_stats.get('spawns', 0)}")
        
        if stats['queue_size'] == 0:
            break
        
        time.sleep(3)
    
    # Get final statistics
    final_stats = queue_manager.get_statistics()
    lb_stats = final_stats.get('load_balancing', {})
    
    print(f"\nCriteria-Based Load Balancing Results:")
    print(f"  Final workers: {final_stats['current_workers']}")
    print(f"  Worker spawns: {lb_stats.get('spawns', 0)}")
    print(f"  Worker reductions: {lb_stats.get('reductions', 0)}")
    print(f"  Spawn attempts: {lb_stats.get('spawn_attempts', 0)}")
    print(f"  Reduce attempts: {lb_stats.get('reduce_attempts', 0)}")
    
    queue_manager.stop_processing()
    print("Criteria-based load balancing test completed")


def test_resource_aware_spawning():
    """Test resource-aware worker spawning."""
    print("\n=== Testing Resource-Aware Spawning ===")
    
    # Create queue manager with resource monitoring
    queue_manager = QueueManager(
        enable_load_balancing=True,
        min_workers=1,
        max_workers_limit=8,
        worker_spawn_threshold=0.4,
        worker_reduce_threshold=0.1,
        cpu_threshold_for_spawn=60,   # Conservative CPU threshold
        memory_threshold_for_spawn=65, # Conservative memory threshold
        worker_spawn_cooldown=20,
        worker_reduce_cooldown=40
    )
    
    # Start processing
    success = queue_manager.start_processing()
    if not success:
        print("Failed to start queue processing")
        return
    
    print(f"Initial workers: {queue_manager._current_workers}")
    print(f"Resource thresholds: CPU <60%, Memory <65%")
    
    # Add tasks to create load
    test_files = create_test_files(25, 0.1)
    task_ids = queue_manager.add_tasks(test_files)
    print(f"Added {len(task_ids)} tasks")
    
    # Monitor resource-aware spawning
    start_time = time.time()
    while time.time() - start_time < 40:  # Monitor for 40 seconds
        stats = queue_manager.get_statistics()
        lb_stats = stats.get('load_balancing', {})
        
        # Get current resource usage
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        print(f"Queue: {stats['queue_size']}, "
              f"Workers: {stats['current_workers']}, "
              f"CPU: {cpu_percent:.1f}%, "
              f"Memory: {memory_percent:.1f}%, "
              f"Spawns: {lb_stats.get('spawns', 0)}")
        
        if stats['queue_size'] == 0:
            break
        
        time.sleep(2)
    
    # Get final statistics
    final_stats = queue_manager.get_statistics()
    lb_stats = final_stats.get('load_balancing', {})
    
    print(f"\nResource-Aware Spawning Results:")
    print(f"  Final workers: {final_stats['current_workers']}")
    print(f"  Worker spawns: {lb_stats.get('spawns', 0)}")
    print(f"  Spawn attempts: {lb_stats.get('spawn_attempts', 0)}")
    print(f"  Completed tasks: {final_stats['completed_tasks']}")
    
    queue_manager.stop_processing()
    print("Resource-aware spawning test completed")


def test_global_queue_manager():
    """Test the global queue manager instance with load balancing."""
    print("\n=== Testing Global Queue Manager ===")
    
    # Get global instance
    queue_manager = get_queue_manager()
    
    # Configure load balancing
    queue_manager.enable_load_balancing = True
    queue_manager.min_workers = 1
    queue_manager.max_workers_limit = 4
    queue_manager.worker_spawn_threshold = 0.3
    queue_manager.worker_reduce_threshold = 0.1
    
    # Start processing
    success = queue_manager.start_processing()
    if not success:
        print("Failed to start global queue processing")
        return
    
    print(f"Global queue manager workers: {queue_manager._current_workers}")
    
    # Add tasks
    test_files = create_test_files(15, 0.05)
    task_ids = queue_manager.add_tasks(test_files)
    print(f"Added {len(task_ids)} tasks to global queue")
    
    # Monitor
    start_time = time.time()
    while time.time() - start_time < 25:  # Monitor for 25 seconds
        stats = queue_manager.get_statistics()
        lb_stats = stats.get('load_balancing', {})
        
        print(f"Global Queue: {stats['queue_size']}, "
              f"Workers: {stats['current_workers']}, "
              f"Spawns: {lb_stats.get('spawns', 0)}")
        
        if stats['queue_size'] == 0:
            break
        
        time.sleep(2)
    
    # Get final statistics
    final_stats = queue_manager.get_statistics()
    lb_stats = final_stats.get('load_balancing', {})
    
    print(f"\nGlobal Queue Manager Results:")
    print(f"  Final workers: {final_stats['current_workers']}")
    print(f"  Worker spawns: {lb_stats.get('spawns', 0)}")
    print(f"  Completed tasks: {final_stats['completed_tasks']}")
    
    queue_manager.stop_processing()
    print("Global queue manager test completed")


def main():
    """Run all load balancing tests."""
    print("Queue Manager Load Balancing Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Basic load balancing
        test_load_balancing_basic()
        
        time.sleep(2)  # Brief pause between tests
        
        # Test 2: Criteria-based load balancing
        test_load_balancing_criteria()
        
        time.sleep(2)  # Brief pause between tests
        
        # Test 3: Resource-aware spawning
        test_resource_aware_spawning()
        
        time.sleep(2)  # Brief pause between tests
        
        # Test 4: Global queue manager
        test_global_queue_manager()
        
        print("\n" + "=" * 50)
        print("All load balancing tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 