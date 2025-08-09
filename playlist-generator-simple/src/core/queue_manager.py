"""
Queue Manager for Playlist Generator Simple.
Provides intelligent task scheduling, prioritization, and queue management for analysis operations.
"""

import os
import time
import threading
import queue
import heapq
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import Future

# Import local modules
from .logging_setup import get_logger, log_function_call, log_universal
from .resource_manager import get_resource_manager

logger = get_logger('playlista.queue_manager')


class TaskPriority(Enum):
    """Task priority levels for queue scheduling."""
    CRITICAL = 1    # Failed files with high retry count
    HIGH = 2        # Small files (fast processing)
    NORMAL = 3      # Medium files
    LOW = 4         # Large files (slow processing)
    BACKGROUND = 5  # Maintenance tasks


class TaskStatus(Enum):
    """Task status tracking."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


@dataclass
class AnalysisTask:
    """Analysis task definition with metadata and tracking."""
    task_id: str
    file_path: str
    file_size_mb: float
    priority: TaskPriority
    created_at: datetime = field(default_factory=datetime.now)
    
    # Task configuration
    force_reanalysis: bool = False
    max_retries: int = 3
    timeout_seconds: int = 300
    
    # Status tracking
    status: TaskStatus = TaskStatus.QUEUED
    assigned_worker: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress and metrics
    progress_percent: float = 0.0
    retry_count: int = 0
    last_error: Optional[str] = None
    processing_time: Optional[float] = None
    
    # Resource requirements (calculated by PLAYLISTA rules)
    analysis_strategy: Optional[Dict[str, Any]] = None
    required_ram_gb: float = 0.0
    max_threads: int = 1
    enforce_sequential: bool = False
    
    def __lt__(self, other):
        """Enable priority queue sorting."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # Secondary sort by file size (smaller files first for same priority)
        return self.file_size_mb < other.file_size_mb
    
    def __hash__(self):
        """Enable use in sets."""
        return hash(self.task_id)
    
    def get_age_seconds(self) -> float:
        """Get task age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def is_expired(self, max_age_seconds: int = 3600) -> bool:
        """Check if task has expired."""
        return self.get_age_seconds() > max_age_seconds
    
    def should_retry(self) -> bool:
        """Determine if task should be retried."""
        return (self.status == TaskStatus.FAILED and 
                self.retry_count < self.max_retries)


class QueueManager:
    """
    Intelligent queue manager for analysis tasks.
    
    Features:
    - Priority-based task scheduling
    - Resource-aware task assignment
    - Dynamic load balancing
    - Retry management with exponential backoff
    - Progress tracking and statistics
    - Worker coordination
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the queue manager.
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_audio_analysis_config()
        
        self.config = config
        self.resource_manager = get_resource_manager()
        
        # Queue management
        self._task_queue = queue.PriorityQueue()
        self._active_tasks: Dict[str, AnalysisTask] = {}
        self._completed_tasks: Dict[str, AnalysisTask] = {}
        self._failed_tasks: Dict[str, AnalysisTask] = {}
        
        # Worker management
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._worker_assignments: Dict[str, str] = {}  # worker_id -> task_id
        
        # Statistics and monitoring
        self._stats = {
            'total_tasks_queued': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'queue_high_water_mark': 0,
            'current_queue_size': 0,
            'active_workers': 0
        }
        
        # Configuration
        self.max_queue_size = config.get('QUEUE_MAX_SIZE', 10000)
        self.task_timeout_seconds = config.get('TASK_TIMEOUT_SECONDS', 600)
        self.retry_backoff_multiplier = config.get('RETRY_BACKOFF_MULTIPLIER', 2.0)
        self.max_retry_delay_seconds = config.get('MAX_RETRY_DELAY_SECONDS', 300)
        self.queue_cleanup_interval = config.get('QUEUE_CLEANUP_INTERVAL', 60)
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Start background services
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        log_universal('INFO', 'QueueManager', 'Queue manager initialized')
        log_universal('INFO', 'QueueManager', 
                     f'Configuration: max_queue_size={self.max_queue_size}, '
                     f'task_timeout={self.task_timeout_seconds}s')
    
    def add_task(self, file_path: str, file_size_mb: float, 
                 priority: TaskPriority = None, **kwargs) -> str:
        """
        Add a new analysis task to the queue.
        
        Args:
            file_path: Path to the file to analyze
            file_size_mb: File size in megabytes
            priority: Task priority (auto-determined if None)
            **kwargs: Additional task parameters
            
        Returns:
            Task ID for tracking
        """
        with self._lock:
            # Check queue capacity
            if self._task_queue.qsize() >= self.max_queue_size:
                raise RuntimeError(f"Queue is full (max: {self.max_queue_size})")
            
            # Generate unique task ID
            task_id = f"task_{int(time.time() * 1000000)}_{abs(hash(file_path)) % 10000}"
            
            # Determine priority if not specified
            if priority is None:
                priority = self._determine_task_priority(file_path, file_size_mb)
            
            # Get PLAYLISTA analysis strategy
            strategy = self.resource_manager.get_file_analysis_strategy(file_size_mb)
            
            # Create task
            task = AnalysisTask(
                task_id=task_id,
                file_path=file_path,
                file_size_mb=file_size_mb,
                priority=priority,
                analysis_strategy=strategy,
                required_ram_gb=strategy['required_ram_gb'],
                max_threads=strategy['max_threads'],
                enforce_sequential=strategy['enforce_sequential'],
                **kwargs
            )
            
            # Add to queue
            self._task_queue.put(task)
            self._stats['total_tasks_queued'] += 1
            self._stats['current_queue_size'] = self._task_queue.qsize()
            self._stats['queue_high_water_mark'] = max(
                self._stats['queue_high_water_mark'],
                self._stats['current_queue_size']
            )
            
            log_universal('INFO', 'QueueManager', 
                         f'Task queued: {task_id} ({file_path}) - '
                         f'Priority: {priority.name}, Strategy: {strategy["analysis_type"]}')
            
            return task_id
    
    def get_next_task(self, worker_id: str) -> Optional[AnalysisTask]:
        """
        Get the next available task for a worker.
        
        Args:
            worker_id: Unique worker identifier
            
        Returns:
            Next task or None if no suitable tasks available
        """
        with self._lock:
            # Register/update worker
            self._workers[worker_id] = {
                'last_seen': datetime.now(),
                'current_task': None,
                'tasks_completed': self._workers.get(worker_id, {}).get('tasks_completed', 0)
            }
            
            # Try to get a task from queue
            try:
                while not self._task_queue.empty():
                    task = self._task_queue.get_nowait()
                    
                    # Check if task is still valid
                    if task.is_expired():
                        log_universal('WARNING', 'QueueManager', f'Task expired: {task.task_id}')
                        self._failed_tasks[task.task_id] = task
                        continue
                    
                    # Check resource availability for this task
                    if self._can_assign_task(task, worker_id):
                        # Assign task to worker
                        task.status = TaskStatus.ASSIGNED
                        task.assigned_worker = worker_id
                        task.started_at = datetime.now()
                        
                        self._active_tasks[task.task_id] = task
                        self._worker_assignments[worker_id] = task.task_id
                        self._workers[worker_id]['current_task'] = task.task_id
                        
                        self._stats['current_queue_size'] = self._task_queue.qsize()
                        self._stats['active_workers'] = len([w for w in self._workers.values() 
                                                           if w['current_task'] is not None])
                        
                        log_universal('INFO', 'QueueManager', 
                                     f'Task assigned: {task.task_id} to worker {worker_id}')
                        
                        return task
                    else:
                        # Can't assign this task now, put it back
                        self._task_queue.put(task)
                        log_universal('DEBUG', 'QueueManager', 
                                     f'Task {task.task_id} cannot be assigned due to resource constraints')
                        break
                
                return None
                
            except queue.Empty:
                return None
    
    def complete_task(self, task_id: str, success: bool, 
                     processing_time: float = None, error: str = None) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: Task identifier
            success: Whether task completed successfully
            processing_time: Time taken for processing
            error: Error message if failed
            
        Returns:
            True if task was found and updated
        """
        with self._lock:
            if task_id not in self._active_tasks:
                log_universal('WARNING', 'QueueManager', f'Task not found: {task_id}')
                return False
            
            task = self._active_tasks[task_id]
            task.completed_at = datetime.now()
            task.processing_time = processing_time
            
            # Remove from active tasks
            del self._active_tasks[task_id]
            
            # Update worker status
            if task.assigned_worker:
                worker_id = task.assigned_worker
                if worker_id in self._workers:
                    self._workers[worker_id]['current_task'] = None
                    self._workers[worker_id]['tasks_completed'] += 1
                
                if worker_id in self._worker_assignments:
                    del self._worker_assignments[worker_id]
            
            if success:
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.progress_percent = 100.0
                self._completed_tasks[task_id] = task
                
                self._stats['total_tasks_completed'] += 1
                if processing_time:
                    self._stats['total_processing_time'] += processing_time
                    self._stats['average_processing_time'] = (
                        self._stats['total_processing_time'] / 
                        self._stats['total_tasks_completed']
                    )
                
                log_universal('INFO', 'QueueManager', 
                             f'Task completed: {task_id} in {processing_time:.2f}s')
            else:
                # Task failed
                task.status = TaskStatus.FAILED
                task.last_error = error
                task.retry_count += 1
                
                if task.should_retry():
                    # Schedule retry with exponential backoff
                    retry_delay = min(
                        (self.retry_backoff_multiplier ** task.retry_count),
                        self.max_retry_delay_seconds
                    )
                    
                    task.status = TaskStatus.RETRY
                    # Re-queue task for retry (implementation would need delay mechanism)
                    self._task_queue.put(task)
                    
                    log_universal('WARNING', 'QueueManager', 
                                 f'Task failed, scheduling retry: {task_id} '
                                 f'(attempt {task.retry_count}/{task.max_retries}) '
                                 f'delay: {retry_delay}s')
                else:
                    # Max retries exceeded
                    self._failed_tasks[task_id] = task
                    self._stats['total_tasks_failed'] += 1
                    
                    log_universal('ERROR', 'QueueManager', 
                                 f'Task permanently failed: {task_id} - {error}')
            
            # Update statistics
            self._stats['active_workers'] = len([w for w in self._workers.values() 
                                               if w['current_task'] is not None])
            
            return True
    
    def _determine_task_priority(self, file_path: str, file_size_mb: float) -> TaskPriority:
        """Determine task priority based on file characteristics."""
        # High priority for small files (fast processing)
        if file_size_mb < 25:
            return TaskPriority.HIGH
        
        # Normal priority for medium files
        elif file_size_mb < 50:
            return TaskPriority.NORMAL
        
        # Low priority for large files (slow processing)
        else:
            return TaskPriority.LOW
    
    def _can_assign_task(self, task: AnalysisTask, worker_id: str) -> bool:
        """Check if a task can be assigned to a worker based on resource constraints."""
        # Check if worker is already busy
        if worker_id in self._worker_assignments:
            return False
        
        # Check resource availability
        try:
            memory_info = self.resource_manager.get_current_resources()
            available_memory_gb = memory_info.get('available_memory_gb', 0)
            
            # Check if enough memory is available
            if available_memory_gb < task.required_ram_gb:
                return False
            
            # Check if sequential task conflicts with other sequential tasks
            if task.enforce_sequential:
                # Only allow one sequential task at a time
                for active_task in self._active_tasks.values():
                    if active_task.enforce_sequential:
                        return False
            
            return True
            
        except Exception as e:
            log_universal('WARNING', 'QueueManager', f'Resource check failed: {e}')
            return False
    
    def _cleanup_worker(self):
        """Background worker for queue cleanup and maintenance."""
        while not self._shutdown_event.wait(self.queue_cleanup_interval):
            try:
                with self._lock:
                    current_time = datetime.now()
                    
                    # Clean up expired workers
                    expired_workers = []
                    for worker_id, worker_info in self._workers.items():
                        if (current_time - worker_info['last_seen']).total_seconds() > 300:
                            expired_workers.append(worker_id)
                    
                    for worker_id in expired_workers:
                        # Handle orphaned tasks
                        if worker_id in self._worker_assignments:
                            task_id = self._worker_assignments[worker_id]
                            if task_id in self._active_tasks:
                                task = self._active_tasks[task_id]
                                task.status = TaskStatus.QUEUED
                                task.assigned_worker = None
                                self._task_queue.put(task)
                                del self._active_tasks[task_id]
                            del self._worker_assignments[worker_id]
                        
                        del self._workers[worker_id]
                        log_universal('WARNING', 'QueueManager', f'Cleaned up expired worker: {worker_id}')
                    
                    # Clean up old completed/failed tasks
                    cutoff_time = current_time - timedelta(hours=1)
                    
                    old_completed = [tid for tid, task in self._completed_tasks.items() 
                                   if task.completed_at and task.completed_at < cutoff_time]
                    for tid in old_completed:
                        del self._completed_tasks[tid]
                    
                    old_failed = [tid for tid, task in self._failed_tasks.items() 
                                if task.completed_at and task.completed_at < cutoff_time]
                    for tid in old_failed:
                        del self._failed_tasks[tid]
                    
                    if old_completed or old_failed:
                        log_universal('DEBUG', 'QueueManager', 
                                     f'Cleaned up {len(old_completed)} completed and {len(old_failed)} failed tasks')
                        
            except Exception as e:
                log_universal('ERROR', 'QueueManager', f'Cleanup worker error: {e}')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        with self._lock:
            return {
                **self._stats.copy(),
                'queue_size': self._task_queue.qsize(),
                'active_tasks': len(self._active_tasks),
                'completed_tasks': len(self._completed_tasks),
                'failed_tasks': len(self._failed_tasks),
                'active_workers': len([w for w in self._workers.values() 
                                     if w['current_task'] is not None]),
                'total_workers': len(self._workers)
            }
    
    def get_active_tasks(self) -> List[AnalysisTask]:
        """Get list of currently active tasks."""
        with self._lock:
            return list(self._active_tasks.values())
    
    def shutdown(self):
        """Shutdown the queue manager."""
        log_universal('INFO', 'QueueManager', 'Shutting down queue manager')
        self._shutdown_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)


# Global queue manager instance
_queue_manager_instance = None
_queue_manager_lock = threading.Lock()

def get_queue_manager(config: Dict[str, Any] = None) -> QueueManager:
    """Get the global queue manager instance."""
    global _queue_manager_instance
    
    if _queue_manager_instance is None:
        with _queue_manager_lock:
            if _queue_manager_instance is None:
                _queue_manager_instance = QueueManager(config)
    
    return _queue_manager_instance
