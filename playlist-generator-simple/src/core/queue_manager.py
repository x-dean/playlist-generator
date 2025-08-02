"""
Queue Manager for Parallel Processing.
Manages file processing queues, worker distribution, and progress monitoring.
"""

import os
import time
import threading
import queue
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import local modules
from .database import DatabaseManager
from .logging_setup import get_logger, log_function_call, log_universal
from .resource_manager import ResourceManager

logger = get_logger('playlista.queue_manager')

# Constants
DEFAULT_QUEUE_SIZE = 1000
DEFAULT_WORKER_TIMEOUT = 300  # 5 minutes
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5  # seconds
DEFAULT_PROGRESS_UPDATE_INTERVAL = 10  # seconds
DEFAULT_MEMORY_CHECK_INTERVAL = 30  # seconds


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


class QueueManager:
    """
    Manages file processing queues for parallel processing.
    
    Features:
    - Priority-based task queuing
    - Automatic retry mechanism
    - Progress monitoring and statistics
    - Memory-aware worker management
    - Database integration for persistence
    - Real-time status updates
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
        Initialize the queue manager.
        
        Args:
            db_manager: Database manager instance
            resource_manager: Resource manager instance
            max_workers: Maximum number of workers
            queue_size: Maximum queue size
            worker_timeout: Worker timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            progress_update_interval: Progress update interval in seconds
        """
        self.db_manager = db_manager or DatabaseManager()
        self.resource_manager = resource_manager or ResourceManager()
        
        # Queue configuration
        self.max_workers = max_workers or self.resource_manager.get_optimal_worker_count()
        self.queue_size = queue_size
        self.worker_timeout = worker_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.progress_update_interval = progress_update_interval
        
        # Queue and task management
        self.task_queue = queue.PriorityQueue(maxsize=queue_size)
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, ProcessingTask] = {}
        self.failed_tasks: Dict[str, ProcessingTask] = {}
        
        # Statistics and monitoring
        self.statistics = QueueStatistics()
        self.progress_callback: Optional[Callable] = None
        
        # Threading and synchronization
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_threads: List[threading.Thread] = []
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Worker pool
        self._executor: Optional[ProcessPoolExecutor] = None
        self._worker_futures: Dict[str, Any] = {}
        
        log_universal('INFO', 'Queue', f"QueueManager initialized with {self.max_workers} workers")
        log_universal('DEBUG', 'Queue', f"Queue size: {queue_size}, Timeout: {worker_timeout}s")
        log_universal('DEBUG', 'Queue', f"Max retries: {max_retries}, Retry delay: {retry_delay}s")
    
    def add_task(self, file_path: str, priority: int = 0, 
                 force_reextract: bool = False, analysis_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a file processing task to the queue.
        
        Args:
            file_path: Path to the file to process
            priority: Task priority (higher = more important)
            force_reextract: If True, bypass cache
            analysis_config: Analysis configuration
            
        Returns:
            Task ID
        """
        with self._lock:
            if self.task_queue.full():
                log_universal('WARNING', 'Queue', f"Queue is full, cannot add task for {file_path}")
                raise queue.Full("Queue is full")
            
            task = ProcessingTask(
                file_path=file_path,
                priority=priority,
                force_reextract=force_reextract,
                analysis_config=analysis_config,
                max_retries=self.max_retries
            )
            
            # Add to queue (priority is handled by ProcessingTask.__lt__)
            self.task_queue.put(task)
            self.statistics.total_tasks += 1
            self.statistics.pending_tasks += 1
            
            log_universal('DEBUG', 'Queue', f"Added task {task.task_id} for {os.path.basename(file_path)} (priority: {priority})")
            return task.task_id
    
    def add_tasks(self, file_paths: List[str], priority: int = 0,
                  force_reextract: bool = False, analysis_config: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add multiple file processing tasks to the queue.
        
        Args:
            file_paths: List of file paths to process
            priority: Task priority (higher = more important)
            force_reextract: If True, bypass cache
            analysis_config: Analysis configuration
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for file_path in file_paths:
            try:
                task_id = self.add_task(file_path, priority, force_reextract, analysis_config)
                task_ids.append(task_id)
            except queue.Full:
                log_universal('WARNING', 'Queue', f"Queue full, stopped adding tasks at {file_path}")
                break
        
        log_universal('INFO', 'Queue', f"Added {len(task_ids)} tasks to queue")
        return task_ids
    
    def start_processing(self, progress_callback: Optional[Callable] = None) -> bool:
        """
        Start processing tasks in the queue.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._executor is not None:
                log_universal('WARNING', 'Queue', "Processing already started")
                return False
            
            self.progress_callback = progress_callback
            self.statistics.start_time = datetime.now()
            self._stop_event.clear()
            
            # Create process pool executor
            try:
                self._executor = ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    mp_context=mp.get_context('spawn')
                )
                
                # Start worker threads
                for i in range(self.max_workers):
                    worker_thread = threading.Thread(
                        target=self._worker_loop,
                        args=(f"worker_{i}",),
                        daemon=True
                    )
                    worker_thread.start()
                    self._worker_threads.append(worker_thread)
                
                # Start monitor thread
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    daemon=True
                )
                self._monitor_thread.start()
                
                log_universal('INFO', 'Queue', f"Started processing with {self.max_workers} workers")
                return True
                
            except Exception as e:
                log_universal('ERROR', 'Queue', f"Failed to start processing: {e}")
                return False
    
    def stop_processing(self, wait: bool = True) -> bool:
        """
        Stop processing tasks.
        
        Args:
            wait: If True, wait for current tasks to complete
            
        Returns:
            True if stopped successfully, False otherwise
        """
        with self._lock:
            if self._executor is None:
                log_universal('WARNING', 'Queue', "Processing not started")
                return False
            
            log_universal('INFO', 'Queue', "Stopping queue processing...")
            self._stop_event.set()
            
            if wait:
                # Wait for current tasks to complete
                log_universal('INFO', 'Queue', "Waiting for current tasks to complete...")
                for future in self._worker_futures.values():
                    if not future.done():
                        try:
                            future.result(timeout=30)  # 30 second timeout
                        except Exception:
                            pass
            
            # Shutdown executor
            if self._executor:
                self._executor.shutdown(wait=wait)
                self._executor = None
            
            # Wait for threads to finish
            for thread in self._worker_threads:
                if thread.is_alive():
                    thread.join(timeout=5)
            
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5)
            
            self.statistics.end_time = datetime.now()
            log_universal('INFO', 'Queue', "Queue processing stopped")
            return True
    
    def _worker_loop(self, worker_id: str):
        """Worker thread loop for processing tasks."""
        log_universal('DEBUG', 'Queue', f"Worker {worker_id} started")
        
        while not self._stop_event.is_set():
            try:
                # Get task from queue
                try:
                    task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Check if we should stop
                if self._stop_event.is_set():
                    break
                
                # Process the task
                self._process_task(task, worker_id)
                
            except Exception as e:
                log_universal('ERROR', 'Queue', f"Worker {worker_id} error: {e}")
        
        log_universal('DEBUG', 'Queue', f"Worker {worker_id} stopped")
    
    def _process_task(self, task: ProcessingTask, worker_id: str):
        """Process a single task."""
        with self._lock:
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            task.worker_id = worker_id
            self.active_tasks[task.task_id] = task
            self.statistics.pending_tasks -= 1
            self.statistics.processing_tasks += 1
        
        log_universal('DEBUG', 'Queue', f"Processing task {task.task_id}: {os.path.basename(task.file_path)}")
        
        try:
            # Submit to process pool
            future = self._executor.submit(
                self._standalone_worker_process,
                task.file_path,
                task.force_reextract,
                self.worker_timeout,
                self.db_manager.db_path,
                task.analysis_config
            )
            
            self._worker_futures[task.task_id] = future
            
            # Wait for result
            try:
                success = future.result(timeout=self.worker_timeout)
                
                with self._lock:
                    task.completed_at = datetime.now()
                    
                    if success:
                        task.status = TaskStatus.COMPLETED
                        self.completed_tasks[task.task_id] = task
                        self.statistics.completed_tasks += 1
                        log_universal('DEBUG', 'Queue', f"Task {task.task_id} completed successfully")
                    else:
                        task.status = TaskStatus.FAILED
                        task.error_message = "Analysis failed"
                        self.failed_tasks[task.task_id] = task
                        self.statistics.failed_tasks += 1
                        log_universal('WARNING', 'Queue', f"Task {task.task_id} failed")
                    
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
                        
                        # Re-add to queue with lower priority
                        task.priority = max(0, task.priority - 1)
                        self.task_queue.put(task)
                        self.statistics.pending_tasks += 1
                        
                        log_universal('WARNING', 'Queue', f"Task {task.task_id} retry {task.retry_count}/{task.max_retries}")
                    else:
                        task.status = TaskStatus.FAILED
                        self.failed_tasks[task.task_id] = task
                        self.statistics.failed_tasks += 1
                        log_universal('ERROR', 'Queue', f"Task {task.task_id} failed after {task.max_retries} retries")
                    
                    self.statistics.processing_tasks -= 1
                    del self.active_tasks[task.task_id]
            
            # Clean up future
            if task.task_id in self._worker_futures:
                del self._worker_futures[task.task_id]
                
        except Exception as e:
            log_universal('ERROR', 'Queue', f"Error processing task {task.task_id}: {e}")
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                self.failed_tasks[task.task_id] = task
                self.statistics.failed_tasks += 1
                self.statistics.processing_tasks -= 1
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    def _monitor_loop(self):
        """Monitor thread for progress updates and resource management."""
        last_progress_update = time.time()
        last_memory_check = time.time()
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Progress updates
                if current_time - last_progress_update >= self.progress_update_interval:
                    self._update_progress()
                    last_progress_update = current_time
                
                # Memory checks
                if current_time - last_memory_check >= DEFAULT_MEMORY_CHECK_INTERVAL:
                    self._check_memory_usage()
                    last_memory_check = current_time
                
                time.sleep(1)
                
            except Exception as e:
                log_universal('ERROR', 'Queue', f"Monitor thread error: {e}")
                time.sleep(5)
    
    def _update_progress(self):
        """Update progress and call progress callback."""
        with self._lock:
            stats = self.get_statistics()
            
            if self.progress_callback:
                try:
                    self.progress_callback(stats)
                except Exception as e:
                    log_universal('ERROR', 'Queue', f"Progress callback error: {e}")
            
            # Log progress
            if stats['total_tasks'] > 0:
                progress = (stats['completed_tasks'] + stats['failed_tasks']) / stats['total_tasks'] * 100
                log_universal('INFO', 'Queue', f"Progress: {progress:.1f}% ({stats['completed_tasks']}/{stats['total_tasks']})")
    
    def _check_memory_usage(self):
        """Check memory usage and adjust if necessary."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 85:
                log_universal('WARNING', 'Queue', f"High memory usage: {memory.percent:.1f}%")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Reduce worker count if necessary
                if memory.percent > 90 and self.max_workers > 1:
                    old_workers = self.max_workers
                    self.max_workers = max(1, self.max_workers // 2)
                    log_universal('WARNING', 'Queue', f"Reduced workers from {old_workers} to {self.max_workers}")
                    
        except Exception as e:
            log_universal('DEBUG', 'Queue', f"Memory check error: {e}")
    
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
                'queue_size': self.task_queue.qsize(),
                'active_workers': len(self._worker_threads),
                'average_processing_time': self.statistics.average_processing_time,
                'throughput': self.statistics.throughput
            }
            
            if self.statistics.start_time:
                stats['start_time'] = self.statistics.start_time.isoformat()
                if self.statistics.end_time:
                    stats['end_time'] = self.statistics.end_time.isoformat()
                    stats['total_time'] = (self.statistics.end_time - self.statistics.start_time).total_seconds()
                else:
                    stats['elapsed_time'] = (datetime.now() - self.statistics.start_time).total_seconds()
            
            return stats
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        with self._lock:
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'file_path': task.file_path,
                    'status': task.status.value,
                    'worker_id': task.worker_id,
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'retry_count': task.retry_count
                }
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'file_path': task.file_path,
                    'status': task.status.value,
                    'worker_id': task.worker_id,
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'retry_count': task.retry_count
                }
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                return {
                    'task_id': task.task_id,
                    'file_path': task.file_path,
                    'status': task.status.value,
                    'worker_id': task.worker_id,
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message,
                    'retry_count': task.retry_count
                }
            
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            # Check if task is in queue
            # Note: This is a simplified implementation
            # In a real implementation, you'd need to iterate through the queue
            
            # Check if task is active
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                log_universal('INFO', 'Queue', f"Cancelled active task {task_id}")
                return True
            
            return False
    
    def clear_queue(self) -> int:
        """Clear all pending tasks from the queue."""
        with self._lock:
            count = 0
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                    count += 1
                except queue.Empty:
                    break
            
            self.statistics.pending_tasks = 0
            log_universal('INFO', 'Queue', f"Cleared {count} pending tasks from queue")
            return count
    
    def _standalone_worker_process(self, file_path: str, force_reextract: bool = False,
                                 timeout_seconds: int = 300, db_path: str = None,
                                 analysis_config: Dict[str, Any] = None) -> bool:
        """
        Standalone worker function for process pool execution.
        This is a copy of the worker function from parallel_analyzer.py
        """
        import os
        import time
        import signal
        import psutil
        import threading
        import gc
        
        # Set up logging for worker process
        try:
            from .logging_setup import get_logger, log_universal
            logger = get_logger('playlista.queue_worker')
        except ImportError:
            import logging
            logger = logging.getLogger('playlista.queue_worker')
        
        worker_id = f"queue_worker_{threading.current_thread().ident}"
        start_time = time.time()
        
        try:
            # Force garbage collection before starting
            gc.collect()
            
            # Set up signal handler for timeout (only on Unix systems)
            try:
                from .parallel_analyzer import timeout_handler
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            except (AttributeError, OSError):
                # Windows doesn't support SIGALRM, skip timeout handling
                pass
            
            # Check if file exists
            if not os.path.exists(file_path):
                log_universal('WARNING', 'Queue', f'File not found: {file_path}')
                return False
            
            # Get initial resource usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            initial_cpu = process.cpu_percent()
            
            # Import audio analyzer here to avoid memory issues
            try:
                from .audio_analyzer import AudioAnalyzer
            except ImportError as e:
                log_universal('ERROR', 'Queue', f'Failed to import AudioAnalyzer: {e}')
                return False
            
            # Create analyzer instance with configuration
            try:
                if analysis_config is None:
                    # Use default configuration
                    analyzer = AudioAnalyzer()
                else:
                    # Apply the analysis configuration to the analyzer
                    analyzer = AudioAnalyzer(config=analysis_config)
            except Exception as e:
                log_universal('ERROR', 'Queue', f'Failed to create AudioAnalyzer: {e}')
                return False
            
            # Set memory limit for this process
            try:
                import resource
                # Set memory limit to 256MB per process
                resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, -1))
            except (ImportError, OSError):
                # Windows doesn't support resource module, skip memory limit
                pass
            
            # Check available memory before analysis
            try:
                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
                if available_memory_mb < 300:  # Less than 300MB available
                    log_universal('WARNING', 'Queue', f'Low memory available ({available_memory_mb:.1f}MB) - skipping {os.path.basename(file_path)}')
                    return False
            except Exception:
                pass  # Continue if memory check fails
            
            # Extract features using the correct method
            try:
                analysis_result = analyzer.analyze_audio_file(file_path, force_reextract)
            except Exception as e:
                log_universal('ERROR', 'Queue', f'Analysis failed for {file_path}: {e}')
                return False
            
            # Get final resource usage
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            final_cpu = process.cpu_percent()
            memory_usage = final_memory - initial_memory
            cpu_usage = (initial_cpu + final_cpu) / 2  # Average
            
            duration = time.time() - start_time
            
            # Force garbage collection after analysis
            gc.collect()
            
            if analysis_result:
                # Save to database using standalone database manager
                if db_path:
                    from .database import DatabaseManager
                    db_manager = DatabaseManager(db_path=db_path)
                    
                    filename = os.path.basename(file_path)
                    file_size_bytes = os.path.getsize(file_path)
                    
                    # Calculate hash (consistent with file discovery)
                    import hashlib
                    stat = os.stat(file_path)
                    filename = os.path.basename(file_path)
                    content = f"{filename}:{stat.st_mtime}:{stat.st_size}"
                    file_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    # Prepare analysis data with status
                    analysis_data = analysis_result.get('features', {})
                    analysis_data['status'] = 'analyzed'
                    analysis_data['analysis_type'] = 'full'
                    
                    # Extract metadata
                    metadata = analysis_result.get('metadata', {})
                    
                    # Determine long audio category
                    long_audio_category = None
                    if 'long_audio_category' in analysis_data:
                        long_audio_category = analysis_data['long_audio_category']
                    
                    success = db_manager.save_analysis_result(
                        file_path=file_path,
                        filename=filename,
                        file_size_bytes=file_size_bytes,
                        file_hash=file_hash,
                        analysis_data=analysis_data,
                        metadata=metadata,
                        discovery_source='file_system'
                    )
                    
                    log_universal('INFO', 'Queue', f'{worker_id} completed: {filename} in {duration:.2f}s')
                    return success
                else:
                    log_universal('INFO', 'Queue', f'{worker_id} completed: {os.path.basename(file_path)} in {duration:.2f}s')
                    return True
            else:
                # Mark as failed
                if db_path:
                    from .database import DatabaseManager
                    db_manager = DatabaseManager(db_path=db_path)
                    filename = os.path.basename(file_path)
                    
                    # Use analysis_cache table for failed analysis
                    with db_manager._get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO analysis_cache 
                            (file_path, filename, error_message, status, retry_count, last_retry_date)
                            VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                        """, (file_path, filename, "Analysis failed"))
                        conn.commit()
                
                log_universal('DEBUG', 'Queue', f"Worker {worker_id} failed: {os.path.basename(file_path)}")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            log_universal('ERROR', 'Queue', f"Worker {worker_id} error processing {os.path.basename(file_path)}: {e}")
            
            if db_path:
                from .database import DatabaseManager
                db_manager = DatabaseManager(db_path=db_path)
                filename = os.path.basename(file_path)
                
                # Use analysis_cache table for failed analysis
                with db_manager._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO analysis_cache 
                        (file_path, filename, error_message, status, retry_count, last_retry_date)
                        VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                    """, (file_path, filename, f"Worker error: {str(e)}"))
                    conn.commit()
            
            return False


# Global queue manager instance
_queue_manager_instance = None

def get_queue_manager() -> 'QueueManager':
    """Get the global queue manager instance, creating it if necessary."""
    global _queue_manager_instance
    if _queue_manager_instance is None:
        _queue_manager_instance = QueueManager()
    return _queue_manager_instance 