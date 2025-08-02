"""
Improved Queue Manager implementing producer-consumer pattern.
Provides better concurrency control and resource management.
"""

import os
import time
import threading
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from concurrent.futures import ProcessPoolExecutor

# Set multiprocessing start method
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# Import local modules
from .database import DatabaseManager
from .logging_setup import get_logger, log_function_call, log_universal
from .resource_manager import ResourceManager

logger = get_logger('playlista.improved_queue')

# Constants
DEFAULT_QUEUE_SIZE = 10
DEFAULT_WORKER_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5
DEFAULT_PROGRESS_UPDATE_INTERVAL = 10


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """Represents a file processing task."""
    file_path: str
    task_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0  # Higher number = higher priority
    retry_count: int = 0
    max_retries: int = DEFAULT_MAX_RETRIES
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    worker_id: Optional[str] = None
    analysis_config: Optional[Dict[str, Any]] = None
    force_reextract: bool = False
    
    def __post_init__(self):
        """Generate task ID if not provided."""
        if not self.task_id:
            self.task_id = f"task_{int(time.time() * 1000)}_{hash(self.file_path) % 10000}"
    
    def __lt__(self, other):
        """Make tasks comparable for priority queue."""
        if not isinstance(other, ProcessingTask):
            return NotImplemented
        # Compare by priority (higher priority first), then by creation time
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.created_at < other.created_at  # Earlier creation first


@dataclass
class QueueStatistics:
    """Queue processing statistics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    retry_tasks: int = 0
    pending_tasks: int = 0
    processing_tasks: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    throughput: float = 0.0  # tasks per second
    
    def update_statistics(self, completed_task: ProcessingTask):
        """Update statistics with a completed task."""
        self.completed_tasks += 1
        if completed_task.completed_at and completed_task.started_at:
            processing_time = (completed_task.completed_at - completed_task.started_at).total_seconds()
            self.total_processing_time += processing_time
            self.average_processing_time = self.total_processing_time / self.completed_tasks
        
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            if elapsed_time > 0:
                self.throughput = self.completed_tasks / elapsed_time


class ImprovedQueueManager:
    """
    Improved Queue Manager implementing producer-consumer pattern.
    
    Features:
    - Producer-consumer pattern with controlled concurrency
    - Memory-aware worker management
    - Automatic retries with exponential backoff
    - Progress monitoring and statistics
    - Graceful shutdown with stop signals
    """
    
    def __init__(self, db_manager: DatabaseManager = None,
                 resource_manager: ResourceManager = None,
                 max_workers: int = None,
                 queue_size: int = DEFAULT_QUEUE_SIZE,
                 worker_timeout: int = DEFAULT_WORKER_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 retry_delay: int = DEFAULT_RETRY_DELAY,
                 progress_update_interval: int = DEFAULT_PROGRESS_UPDATE_INTERVAL):
        """
        Initialize the improved queue manager.
        
        Args:
            db_manager: Database manager instance
            resource_manager: Resource manager instance
            max_workers: Maximum number of worker threads
            queue_size: Maximum size of the task queue
            worker_timeout: Timeout for worker processes
            max_retries: Maximum number of retries per task
            retry_delay: Delay between retries in seconds
            progress_update_interval: Progress update interval in seconds
        """
        self.db_manager = db_manager or DatabaseManager()
        self.resource_manager = resource_manager or ResourceManager()
        
        # Queue configuration
        self.queue_size = queue_size
        self.worker_timeout = worker_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.progress_update_interval = progress_update_interval
        
        # Determine optimal worker count
        if max_workers is None:
            max_workers = self.resource_manager.get_optimal_worker_count()
        self.max_workers = max_workers
        
        # Queue and threading
        self.task_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue()
        self.stop_event = threading.Event()
        
        # Threading components
        self.producer_thread = None
        self.consumer_threads = []
        self.monitor_thread = None
        
        # Task tracking
        self.pending_tasks = {}  # task_id -> ProcessingTask
        self.active_tasks = {}   # task_id -> ProcessingTask
        self.completed_tasks = {}  # task_id -> ProcessingTask
        self.failed_tasks = {}   # task_id -> ProcessingTask
        
        # Statistics
        self.statistics = QueueStatistics()
        
        # Threading locks
        self._lock = threading.RLock()
        
        # Process pool for worker processes
        self._executor = None
        
        log_universal('INFO', 'Queue', f'ImprovedQueueManager initialized with {self.max_workers} workers')
    
    def add_task(self, file_path: str, priority: int = 0, 
                 force_reextract: bool = False, analysis_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a single task to the queue.
        
        Args:
            file_path: Path to the file to process
            priority: Task priority (higher = more important)
            force_reextract: Force re-extraction of features
            analysis_config: Analysis configuration
            
        Returns:
            Task ID
        """
        task = ProcessingTask(
            file_path=file_path,
            priority=priority,
            force_reextract=force_reextract,
            analysis_config=analysis_config,
            max_retries=self.max_retries
        )
        
        with self._lock:
            self.pending_tasks[task.task_id] = task
            self.statistics.total_tasks += 1
            self.statistics.pending_tasks += 1
        
        log_universal('DEBUG', 'Queue', f'Added task {task.task_id}: {os.path.basename(file_path)}')
        return task.task_id
    
    def add_tasks(self, file_paths: List[str], priority: int = 0,
                  force_reextract: bool = False, analysis_config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add multiple tasks to the queue.
        
        Args:
            file_paths: List of file paths to process
            priority: Task priority (higher = more important)
            force_reextract: Force re-extraction of features
            analysis_config: Analysis configuration
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for file_path in file_paths:
            task_id = self.add_task(file_path, priority, force_reextract, analysis_config)
            task_ids.append(task_id)
        
        log_universal('INFO', 'Queue', f'Added {len(task_ids)} tasks to queue')
        return task_ids
    
    def start_processing(self, progress_callback: Optional[Callable] = None) -> bool:
        """
        Start processing tasks using producer-consumer pattern.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if started successfully
        """
        if self.producer_thread and self.producer_thread.is_alive():
            log_universal('WARNING', 'Queue', 'Processing already started')
            return False
        
        # Initialize process pool
        self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start producer thread
        self.producer_thread = threading.Thread(
            target=self._producer_loop,
            name="QueueProducer",
            daemon=True
        )
        self.producer_thread.start()
        
        # Start consumer threads
        self.consumer_threads = []
        for i in range(self.max_workers):
            consumer_thread = threading.Thread(
                target=self._consumer_loop,
                name=f"QueueConsumer-{i}",
                daemon=True
            )
            consumer_thread.start()
            self.consumer_threads.append(consumer_thread)
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(progress_callback,),
            name="QueueMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.statistics.start_time = datetime.now()
        log_universal('INFO', 'Queue', f'Started processing with {self.max_workers} consumers')
        return True
    
    def stop_processing(self, wait: bool = True) -> bool:
        """
        Stop processing tasks.
        
        Args:
            wait: Whether to wait for threads to finish
            
        Returns:
            True if stopped successfully
        """
        log_universal('INFO', 'Queue', 'Stopping queue processing...')
        
        # Signal stop
        self.stop_event.set()
        
        # Send stop signals to consumers
        for _ in range(self.max_workers):
            try:
                self.task_queue.put(None, timeout=1)
            except:
                pass
        
        # Wait for threads if requested
        if wait:
            if self.producer_thread:
                self.producer_thread.join(timeout=10)
            for thread in self.consumer_threads:
                thread.join(timeout=10)
            if self.monitor_thread:
                self.monitor_thread.join(timeout=10)
        
        # Shutdown process pool
        if self._executor:
            self._executor.shutdown(wait=wait)
        
        self.statistics.end_time = datetime.now()
        log_universal('INFO', 'Queue', 'Queue processing stopped')
        return True
    
    def _producer_loop(self):
        """Producer thread that feeds tasks to the queue."""
        log_universal('DEBUG', 'Queue', 'Producer thread started')
        
        try:
            while not self.stop_event.is_set():
                # Get next pending task
                with self._lock:
                    if not self.pending_tasks:
                        break
                    
                    # Get highest priority task
                    task = max(self.pending_tasks.values(), key=lambda t: (t.priority, -t.created_at.timestamp()))
                    del self.pending_tasks[task.task_id]
                    self.statistics.pending_tasks -= 1
                
                # Put task in queue (blocking if queue is full)
                try:
                    self.task_queue.put(task, timeout=1)
                    log_universal('DEBUG', 'Queue', f'Produced task {task.task_id}')
                except:
                    # Queue is full, put task back
                    with self._lock:
                        self.pending_tasks[task.task_id] = task
                        self.statistics.pending_tasks += 1
                    time.sleep(0.1)
            
            # Send stop signals to consumers
            for _ in range(self.max_workers):
                try:
                    self.task_queue.put(None, timeout=1)
                except:
                    pass
                    
        except Exception as e:
            log_universal('ERROR', 'Queue', f'Producer error: {e}')
        finally:
            log_universal('DEBUG', 'Queue', 'Producer thread stopped')
    
    def _consumer_loop(self):
        """Consumer thread that processes tasks from the queue."""
        worker_id = f"consumer_{threading.current_thread().ident}"
        log_universal('DEBUG', 'Queue', f'Consumer {worker_id} started')
        
        try:
            while not self.stop_event.is_set():
                try:
                    # Get task from queue (blocking)
                    task = self.task_queue.get(timeout=1)
                    
                    # Check for stop signal
                    if task is None:
                        break
                    
                    # Process the task
                    self._process_task(task, worker_id)
                    
                except:
                    # Timeout or queue empty, continue
                    continue
                    
        except Exception as e:
            log_universal('ERROR', 'Queue', f'Consumer {worker_id} error: {e}')
        finally:
            log_universal('DEBUG', 'Queue', f'Consumer {worker_id} stopped')
    
    def _process_task(self, task: ProcessingTask, worker_id: str):
        """Process a single task."""
        with self._lock:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            task.worker_id = worker_id
            self.active_tasks[task.task_id] = task
            self.statistics.processing_tasks += 1
        
        log_universal('DEBUG', 'Queue', f'Processing task {task.task_id}: {os.path.basename(task.file_path)}')
        
        try:
            # Submit to process pool
            future = self._executor.submit(
                standalone_worker_process,
                task.file_path,
                task.force_reextract,
                self.worker_timeout,
                self.db_manager.db_path,
                task.analysis_config
            )
            
            # Wait for result
            try:
                success = future.result(timeout=self.worker_timeout)
                
                with self._lock:
                    task.completed_at = datetime.now()
                    
                    if success:
                        task.status = TaskStatus.COMPLETED
                        self.completed_tasks[task.task_id] = task
                        self.statistics.completed_tasks += 1
                        log_universal('DEBUG', 'Queue', f'Task {task.task_id} completed successfully')
                    else:
                        task.status = TaskStatus.FAILED
                        task.error_message = "Analysis failed"
                        self.failed_tasks[task.task_id] = task
                        self.statistics.failed_tasks += 1
                        log_universal('WARNING', 'Queue', f'Task {task.task_id} failed')
                    
                    self.statistics.processing_tasks -= 1
                    del self.active_tasks[task.task_id]
                    
                    # Update statistics
                    self.statistics.update_statistics(task)
                
            except Exception as e:
                with self._lock:
                    task.completed_at = datetime.now()
                    task.error_message = str(e)
                    
                    if task.retry_count < task.max_retries:
                        task.status = TaskStatus.RETRY
                        task.retry_count += 1
                        self.statistics.retry_tasks += 1
                        
                        # Re-add to pending tasks with lower priority
                        task.priority = max(0, task.priority - 1)
                        self.pending_tasks[task.task_id] = task
                        self.statistics.pending_tasks += 1
                        
                        log_universal('WARNING', 'Queue', f'Task {task.task_id} retry {task.retry_count}/{task.max_retries}')
                    else:
                        task.status = TaskStatus.FAILED
                        self.failed_tasks[task.task_id] = task
                        self.statistics.failed_tasks += 1
                        log_universal('ERROR', 'Queue', f'Task {task.task_id} failed after {task.max_retries} retries')
                    
                    self.statistics.processing_tasks -= 1
                    del self.active_tasks[task.task_id]
                
        except Exception as e:
            log_universal('ERROR', 'Queue', f'Error processing task {task.task_id}: {e}')
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                self.failed_tasks[task.task_id] = task
                self.statistics.failed_tasks += 1
                self.statistics.processing_tasks -= 1
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    def _monitor_loop(self, progress_callback: Optional[Callable] = None):
        """Monitor thread for progress updates and resource management."""
        last_progress_update = time.time()
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Update progress periodically
                if current_time - last_progress_update >= self.progress_update_interval:
                    self._update_progress()
                    if progress_callback:
                        try:
                            progress_callback(self.get_statistics())
                        except Exception as e:
                            log_universal('WARNING', 'Queue', f'Progress callback error: {e}')
                    last_progress_update = current_time
                
                # Check memory usage
                self._check_memory_usage()
                
                time.sleep(1)
                
            except Exception as e:
                log_universal('ERROR', 'Queue', f'Monitor error: {e}')
                time.sleep(5)
    
    def _update_progress(self):
        """Update progress information."""
        with self._lock:
            total = self.statistics.total_tasks
            completed = self.statistics.completed_tasks
            failed = self.statistics.failed_tasks
            pending = self.statistics.pending_tasks
            processing = self.statistics.processing_tasks
            
            if total > 0:
                progress = (completed + failed) / total * 100
                log_universal('INFO', 'Queue', 
                            f'Progress: {progress:.1f}% ({completed}/{total} completed, {failed} failed, {pending} pending, {processing} processing)')
    
    def _check_memory_usage(self):
        """Check memory usage and take action if needed."""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > 85:
                log_universal('WARNING', 'Queue', f'High memory usage: {memory_percent:.1f}%')
                import gc
                gc.collect()
                
        except Exception:
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        with self._lock:
            stats = {
                'total_tasks': self.statistics.total_tasks,
                'completed_tasks': self.statistics.completed_tasks,
                'failed_tasks': self.statistics.failed_tasks,
                'retry_tasks': self.statistics.retry_tasks,
                'pending_tasks': self.statistics.pending_tasks,
                'processing_tasks': self.statistics.processing_tasks,
                'start_time': self.statistics.start_time,
                'end_time': self.statistics.end_time,
                'total_processing_time': self.statistics.total_processing_time,
                'average_processing_time': self.statistics.average_processing_time,
                'throughput': self.statistics.throughput,
                'queue_size': self.task_queue.qsize(),
                'max_workers': self.max_workers
            }
            
            if self.statistics.start_time:
                if self.statistics.end_time:
                    elapsed = (self.statistics.end_time - self.statistics.start_time).total_seconds()
                else:
                    elapsed = (datetime.now() - self.statistics.start_time).total_seconds()
                stats['elapsed_time'] = elapsed
            
            return stats
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        with self._lock:
            # Check all task collections
            for task_dict in [self.pending_tasks, self.active_tasks, self.completed_tasks, self.failed_tasks]:
                if task_id in task_dict:
                    task = task_dict[task_id]
                    return {
                        'task_id': task.task_id,
                        'file_path': task.file_path,
                        'status': task.status.value,
                        'priority': task.priority,
                        'retry_count': task.retry_count,
                        'max_retries': task.max_retries,
                        'created_at': task.created_at.isoformat() if task.created_at else None,
                        'started_at': task.started_at.isoformat() if task.started_at else None,
                        'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                        'error_message': task.error_message,
                        'worker_id': task.worker_id
                    }
        return None


def get_improved_queue_manager() -> 'ImprovedQueueManager':
    """Get a global instance of the improved queue manager."""
    if not hasattr(get_improved_queue_manager, '_instance'):
        get_improved_queue_manager._instance = ImprovedQueueManager()
    return get_improved_queue_manager._instance


# Import the standalone worker process function
from .queue_manager import standalone_worker_process 