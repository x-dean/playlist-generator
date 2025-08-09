"""
Progress Monitor for Playlist Generator Simple.
Provides real-time progress tracking, statistics collection, and performance monitoring.
"""

import os
import time
import threading
import psutil
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Import local modules
from .logging_setup import get_logger, log_function_call, log_universal

logger = get_logger('playlista.progress_monitor')


@dataclass
class ProgressSnapshot:
    """Progress snapshot at a point in time."""
    timestamp: datetime
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_tasks: int
    queue_size: int
    processing_rate: float  # tasks per second
    average_task_time: float  # seconds per task
    memory_usage_mb: float
    cpu_usage_percent: float
    active_workers: int


@dataclass
class TaskMetrics:
    """Metrics for individual task tracking."""
    task_id: str
    file_path: str
    file_size_mb: float
    analysis_type: str
    started_at: datetime
    progress_percent: float = 0.0
    estimated_completion: Optional[datetime] = None
    current_phase: str = "initializing"
    memory_usage_mb: float = 0.0
    
    def update_progress(self, percent: float, phase: str = None):
        """Update task progress."""
        self.progress_percent = min(100.0, max(0.0, percent))
        if phase:
            self.current_phase = phase
        
        # Estimate completion time
        if self.progress_percent > 0:
            elapsed = (datetime.now() - self.started_at).total_seconds()
            estimated_total_time = elapsed / (self.progress_percent / 100.0)
            self.estimated_completion = self.started_at + timedelta(seconds=estimated_total_time)


class PerformanceTracker:
    """Tracks system and analysis performance metrics."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.task_timing_history = deque(maxlen=history_size)
        self.error_history = deque(maxlen=100)
        
        # Performance calculations
        self._last_calculation = datetime.now()
        self._last_completed_count = 0
        
    def record_snapshot(self, snapshot: ProgressSnapshot):
        """Record a progress snapshot."""
        self.metrics_history.append(snapshot)
    
    def record_task_completion(self, task_id: str, processing_time: float, 
                             file_size_mb: float, analysis_type: str):
        """Record task completion metrics."""
        self.task_timing_history.append({
            'task_id': task_id,
            'processing_time': processing_time,
            'file_size_mb': file_size_mb,
            'analysis_type': analysis_type,
            'timestamp': datetime.now(),
            'throughput': file_size_mb / processing_time if processing_time > 0 else 0
        })
    
    def record_error(self, task_id: str, error: str, context: Dict[str, Any] = None):
        """Record error for analysis."""
        self.error_history.append({
            'task_id': task_id,
            'error': error,
            'context': context or {},
            'timestamp': datetime.now()
        })
    
    def get_processing_rate(self, window_minutes: int = 5) -> float:
        """Calculate processing rate (tasks per second) over time window."""
        if not self.metrics_history:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_snapshots = [s for s in self.metrics_history if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return 0.0
        
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]
        
        time_diff = (last_snapshot.timestamp - first_snapshot.timestamp).total_seconds()
        task_diff = last_snapshot.completed_tasks - first_snapshot.completed_tasks
        
        return task_diff / time_diff if time_diff > 0 else 0.0
    
    def get_average_task_time(self, analysis_type: str = None, 
                            window_minutes: int = 30) -> float:
        """Calculate average task processing time."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        relevant_tasks = [
            t for t in self.task_timing_history 
            if t['timestamp'] >= cutoff_time and 
            (analysis_type is None or t['analysis_type'] == analysis_type)
        ]
        
        if not relevant_tasks:
            return 0.0
        
        return sum(t['processing_time'] for t in relevant_tasks) / len(relevant_tasks)
    
    def get_throughput_by_type(self) -> Dict[str, float]:
        """Get throughput (MB/s) by analysis type."""
        throughput_by_type = defaultdict(list)
        
        for task in self.task_timing_history:
            if task['throughput'] > 0:
                throughput_by_type[task['analysis_type']].append(task['throughput'])
        
        return {
            analysis_type: sum(throughputs) / len(throughputs)
            for analysis_type, throughputs in throughput_by_type.items()
            if throughputs
        }
    
    def get_error_patterns(self) -> Dict[str, int]:
        """Analyze error patterns."""
        error_counts = defaultdict(int)
        
        for error_record in self.error_history:
            # Simplify error message for pattern detection
            error_key = error_record['error'][:100]  # First 100 chars
            error_counts[error_key] += 1
        
        return dict(error_counts)
    
    def predict_completion_time(self, remaining_tasks: int) -> Optional[datetime]:
        """Predict when all remaining tasks will complete."""
        processing_rate = self.get_processing_rate()
        
        if processing_rate <= 0:
            return None
        
        estimated_seconds = remaining_tasks / processing_rate
        return datetime.now() + timedelta(seconds=estimated_seconds)


class ProgressMonitor:
    """
    Comprehensive progress monitoring system.
    
    Features:
    - Real-time progress tracking
    - Performance metrics collection
    - Resource utilization monitoring
    - Task-level progress tracking
    - Predictive analytics
    - Alert system for anomalies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the progress monitor.
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_audio_analysis_config()
        
        self.config = config
        
        # Core components
        self.performance_tracker = PerformanceTracker()
        
        # Task tracking
        self._active_tasks: Dict[str, TaskMetrics] = {}
        self._completed_tasks_count = 0
        self._failed_tasks_count = 0
        self._total_tasks_count = 0
        
        # System monitoring
        self._system_metrics = {
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'disk_usage_percent': 0.0,
            'network_io_mb': 0.0
        }
        
        # Callbacks for external notifications
        self._progress_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        
        # Configuration
        self.snapshot_interval_seconds = config.get('PROGRESS_SNAPSHOT_INTERVAL', 30)
        self.alert_threshold_memory_percent = config.get('ALERT_MEMORY_THRESHOLD', 90)
        self.alert_threshold_cpu_percent = config.get('ALERT_CPU_THRESHOLD', 95)
        self.alert_threshold_failure_rate = config.get('ALERT_FAILURE_RATE_THRESHOLD', 0.3)
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self._monitor_thread.start()
        
        log_universal('INFO', 'ProgressMonitor', 'Progress monitor initialized')
        log_universal('INFO', 'ProgressMonitor', 
                     f'Snapshot interval: {self.snapshot_interval_seconds}s, '
                     f'Memory alert: {self.alert_threshold_memory_percent}%, '
                     f'CPU alert: {self.alert_threshold_cpu_percent}%')
    
    def start_task(self, task_id: str, file_path: str, file_size_mb: float, 
                   analysis_type: str) -> TaskMetrics:
        """
        Start tracking a new task.
        
        Args:
            task_id: Unique task identifier
            file_path: Path to file being analyzed
            file_size_mb: File size in megabytes
            analysis_type: Type of analysis being performed
            
        Returns:
            TaskMetrics object for progress updates
        """
        with self._lock:
            task_metrics = TaskMetrics(
                task_id=task_id,
                file_path=file_path,
                file_size_mb=file_size_mb,
                analysis_type=analysis_type,
                started_at=datetime.now()
            )
            
            self._active_tasks[task_id] = task_metrics
            self._total_tasks_count += 1
            
            log_universal('DEBUG', 'ProgressMonitor', f'Started tracking task: {task_id}')
            
            return task_metrics
    
    def update_task_progress(self, task_id: str, progress_percent: float, 
                           phase: str = None, memory_usage_mb: float = None):
        """
        Update task progress.
        
        Args:
            task_id: Task identifier
            progress_percent: Progress percentage (0-100)
            phase: Current processing phase
            memory_usage_mb: Current memory usage
        """
        with self._lock:
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id]
                task.update_progress(progress_percent, phase)
                
                if memory_usage_mb is not None:
                    task.memory_usage_mb = memory_usage_mb
                
                # Notify callbacks
                self._notify_progress_callbacks(task_id, progress_percent, phase)
    
    def complete_task(self, task_id: str, success: bool, 
                     processing_time: float = None, error: str = None):
        """
        Mark a task as completed.
        
        Args:
            task_id: Task identifier
            success: Whether task completed successfully
            processing_time: Total processing time in seconds
            error: Error message if failed
        """
        with self._lock:
            if task_id not in self._active_tasks:
                log_universal('WARNING', 'ProgressMonitor', f'Task not found: {task_id}')
                return
            
            task = self._active_tasks[task_id]
            
            if success:
                self._completed_tasks_count += 1
                task.progress_percent = 100.0
                task.current_phase = "completed"
                
                # Record performance metrics
                if processing_time:
                    self.performance_tracker.record_task_completion(
                        task_id, processing_time, task.file_size_mb, task.analysis_type
                    )
                
                log_universal('DEBUG', 'ProgressMonitor', 
                             f'Task completed: {task_id} in {processing_time:.2f}s')
            else:
                self._failed_tasks_count += 1
                task.current_phase = "failed"
                
                # Record error
                if error:
                    self.performance_tracker.record_error(task_id, error, {
                        'file_path': task.file_path,
                        'file_size_mb': task.file_size_mb,
                        'analysis_type': task.analysis_type
                    })
                
                log_universal('WARNING', 'ProgressMonitor', f'Task failed: {task_id} - {error}')
            
            # Remove from active tasks
            del self._active_tasks[task_id]
            
            # Check for alerts
            self._check_alerts()
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress statistics."""
        with self._lock:
            active_count = len(self._active_tasks)
            total_progress = sum(task.progress_percent for task in self._active_tasks.values())
            average_progress = total_progress / active_count if active_count > 0 else 0.0
            
            processing_rate = self.performance_tracker.get_processing_rate()
            remaining_tasks = active_count + max(0, self._total_tasks_count - 
                                               self._completed_tasks_count - 
                                               self._failed_tasks_count - active_count)
            
            predicted_completion = self.performance_tracker.predict_completion_time(remaining_tasks)
            
            return {
                'total_tasks': self._total_tasks_count,
                'completed_tasks': self._completed_tasks_count,
                'failed_tasks': self._failed_tasks_count,
                'active_tasks': active_count,
                'average_active_progress': average_progress,
                'overall_completion_rate': (
                    (self._completed_tasks_count / self._total_tasks_count * 100) 
                    if self._total_tasks_count > 0 else 0.0
                ),
                'processing_rate_per_second': processing_rate,
                'predicted_completion': predicted_completion.isoformat() if predicted_completion else None,
                'system_metrics': self._system_metrics.copy()
            }
    
    def get_task_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about active tasks."""
        with self._lock:
            return [
                {
                    'task_id': task.task_id,
                    'file_path': task.file_path,
                    'file_size_mb': task.file_size_mb,
                    'analysis_type': task.analysis_type,
                    'progress_percent': task.progress_percent,
                    'current_phase': task.current_phase,
                    'started_at': task.started_at.isoformat(),
                    'estimated_completion': (task.estimated_completion.isoformat() 
                                           if task.estimated_completion else None),
                    'memory_usage_mb': task.memory_usage_mb
                }
                for task in self._active_tasks.values()
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance analysis summary."""
        return {
            'average_processing_times': {
                'sequential_only': self.performance_tracker.get_average_task_time('sequential_only'),
                'parallel_large': self.performance_tracker.get_average_task_time('parallel_large'),
                'parallel_small': self.performance_tracker.get_average_task_time('parallel_small')
            },
            'throughput_by_type': self.performance_tracker.get_throughput_by_type(),
            'processing_rate_5min': self.performance_tracker.get_processing_rate(5),
            'processing_rate_15min': self.performance_tracker.get_processing_rate(15),
            'error_patterns': self.performance_tracker.get_error_patterns(),
            'system_utilization': self._system_metrics.copy()
        }
    
    def add_progress_callback(self, callback: Callable):
        """Add callback for progress updates."""
        self._progress_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for system alerts."""
        self._alert_callbacks.append(callback)
    
    def _monitoring_worker(self):
        """Background worker for system monitoring and snapshots."""
        while not self._shutdown_event.wait(self.snapshot_interval_seconds):
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Create progress snapshot
                snapshot = self._create_progress_snapshot()
                self.performance_tracker.record_snapshot(snapshot)
                
                # Check for alerts
                self._check_alerts()
                
            except Exception as e:
                log_universal('ERROR', 'ProgressMonitor', f'Monitoring worker error: {e}')
    
    def _update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self._system_metrics['memory_usage_mb'] = (memory.total - memory.available) / (1024 * 1024)
            self._system_metrics['memory_usage_percent'] = memory.percent
            
            # CPU usage
            self._system_metrics['cpu_usage_percent'] = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self._system_metrics['disk_usage_percent'] = (disk.used / disk.total) * 100
            
        except Exception as e:
            log_universal('WARNING', 'ProgressMonitor', f'System metrics update failed: {e}')
    
    def _create_progress_snapshot(self) -> ProgressSnapshot:
        """Create a progress snapshot."""
        with self._lock:
            return ProgressSnapshot(
                timestamp=datetime.now(),
                total_tasks=self._total_tasks_count,
                completed_tasks=self._completed_tasks_count,
                failed_tasks=self._failed_tasks_count,
                active_tasks=len(self._active_tasks),
                queue_size=0,  # Will be updated by queue manager
                processing_rate=self.performance_tracker.get_processing_rate(),
                average_task_time=self.performance_tracker.get_average_task_time(),
                memory_usage_mb=self._system_metrics['memory_usage_mb'],
                cpu_usage_percent=self._system_metrics['cpu_usage_percent'],
                active_workers=0  # Will be updated by queue manager
            )
    
    def _check_alerts(self):
        """Check for alert conditions."""
        try:
            alerts = []
            
            # Memory usage alert
            memory_percent = self._system_metrics.get('memory_usage_percent', 0)
            if memory_percent > self.alert_threshold_memory_percent:
                alerts.append({
                    'type': 'memory_high',
                    'message': f'Memory usage high: {memory_percent:.1f}%',
                    'severity': 'warning' if memory_percent < 95 else 'critical'
                })
            
            # CPU usage alert
            cpu_percent = self._system_metrics.get('cpu_usage_percent', 0)
            if cpu_percent > self.alert_threshold_cpu_percent:
                alerts.append({
                    'type': 'cpu_high',
                    'message': f'CPU usage high: {cpu_percent:.1f}%',
                    'severity': 'warning'
                })
            
            # High failure rate alert
            if self._total_tasks_count > 10:  # Only check if we have enough samples
                failure_rate = self._failed_tasks_count / self._total_tasks_count
                if failure_rate > self.alert_threshold_failure_rate:
                    alerts.append({
                        'type': 'failure_rate_high',
                        'message': f'High failure rate: {failure_rate:.1%}',
                        'severity': 'warning'
                    })
            
            # Notify alert callbacks
            for alert in alerts:
                self._notify_alert_callbacks(alert)
                
        except Exception as e:
            log_universal('ERROR', 'ProgressMonitor', f'Alert check failed: {e}')
    
    def _notify_progress_callbacks(self, task_id: str, progress: float, phase: str):
        """Notify progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(task_id, progress, phase)
            except Exception as e:
                log_universal('WARNING', 'ProgressMonitor', f'Progress callback failed: {e}')
    
    def _notify_alert_callbacks(self, alert: Dict[str, Any]):
        """Notify alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                log_universal('WARNING', 'ProgressMonitor', f'Alert callback failed: {e}')
    
    def shutdown(self):
        """Shutdown the progress monitor."""
        log_universal('INFO', 'ProgressMonitor', 'Shutting down progress monitor')
        self._shutdown_event.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)


# Global progress monitor instance
_progress_monitor_instance = None
_progress_monitor_lock = threading.Lock()

def get_progress_monitor(config: Dict[str, Any] = None) -> ProgressMonitor:
    """Get the global progress monitor instance."""
    global _progress_monitor_instance
    
    if _progress_monitor_instance is None:
        with _progress_monitor_lock:
            if _progress_monitor_instance is None:
                _progress_monitor_instance = ProgressMonitor(config)
    
    return _progress_monitor_instance
