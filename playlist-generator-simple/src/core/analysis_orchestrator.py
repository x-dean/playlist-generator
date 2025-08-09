"""
Analysis Orchestrator for Playlist Generator Simple.
Provides centralized coordination and control of the entire analysis pipeline.
"""

import os
import time
import threading
import asyncio
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

# Import local modules
from .logging_setup import get_logger, log_function_call, log_universal
from .queue_manager import get_queue_manager, TaskPriority, TaskStatus
from .progress_monitor import get_progress_monitor
from .resource_coordinator import get_resource_coordinator
from .resource_manager import get_resource_manager
from .database import get_db_manager

logger = get_logger('playlista.analysis_orchestrator')


class OrchestratorState(Enum):
    """Orchestrator operational states."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class WorkerType(Enum):
    """Types of analysis workers."""
    SEQUENTIAL = "sequential"
    PARALLEL_LARGE = "parallel_large"
    PARALLEL_SMALL = "parallel_small"
    OPTIMIZED = "optimized"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_concurrent_tasks: int = 10
    worker_spawn_delay: float = 1.0
    health_check_interval: int = 30
    statistics_update_interval: int = 10
    auto_scaling_enabled: bool = True
    retry_failed_tasks: bool = True
    max_retry_attempts: int = 3
    task_timeout_seconds: int = 600
    graceful_shutdown_timeout: int = 30


class AnalysisWorker:
    """Individual analysis worker managed by the orchestrator."""
    
    def __init__(self, worker_id: str, worker_type: WorkerType, 
                 orchestrator: 'AnalysisOrchestrator'):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.orchestrator = orchestrator
        self.is_active = False
        self.current_task_id: Optional[str] = None
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        self.last_activity = datetime.now()
        
        # Threading
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
    def start(self):
        """Start the worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        
        self.is_active = True
        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        log_universal('INFO', 'AnalysisWorker', f'Worker started: {self.worker_id} ({self.worker_type.value})')
    
    def stop(self, timeout: float = 10.0):
        """Stop the worker thread."""
        self.is_active = False
        self._shutdown_event.set()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            
        log_universal('INFO', 'AnalysisWorker', f'Worker stopped: {self.worker_id}')
    
    def _worker_loop(self):
        """Main worker loop."""
        while self.is_active and not self._shutdown_event.is_set():
            try:
                # Get next task from queue
                task = self.orchestrator.queue_manager.get_next_task(self.worker_id)
                
                if task is None:
                    # No tasks available, wait a bit
                    if self._shutdown_event.wait(1.0):
                        break
                    continue
                
                # Process the task
                self._process_task(task)
                
            except Exception as e:
                log_universal('ERROR', 'AnalysisWorker', 
                             f'Worker {self.worker_id} error: {e}')
                time.sleep(5)  # Brief pause on error
    
    def _process_task(self, task):
        """Process an individual task."""
        self.current_task_id = task.task_id
        self.last_activity = datetime.now()
        
        start_time = time.time()
        success = False
        error_message = None
        
        try:
            # Start progress tracking
            task_metrics = self.orchestrator.progress_monitor.start_task(
                task.task_id, task.file_path, task.file_size_mb, 
                task.analysis_strategy['analysis_type']
            )
            
            # Allocate resources
            allocated = self.orchestrator.resource_coordinator.allocate_task_resources(
                self.worker_id, task.required_ram_gb
            )
            
            if not allocated:
                raise RuntimeError("Failed to allocate required resources")
            
            # Perform the actual analysis
            log_universal('INFO', 'AnalysisWorker', 
                         f'Starting analysis: {task.file_path} (Strategy: {task.analysis_strategy["analysis_type"]})')
            
            # Update progress
            self.orchestrator.progress_monitor.update_task_progress(
                task.task_id, 10.0, "loading_file"
            )
            
            # Select appropriate analyzer based on task strategy
            result = self._run_analysis(task, task_metrics)
            
            if result:
                success = True
                self.tasks_completed += 1
                
                # Final progress update
                self.orchestrator.progress_monitor.update_task_progress(
                    task.task_id, 100.0, "completed"
                )
                
                log_universal('INFO', 'AnalysisWorker', 
                             f'Analysis completed: {task.file_path} in {time.time() - start_time:.2f}s')
            else:
                error_message = "Analysis returned no result"
                self.tasks_failed += 1
                
        except Exception as e:
            error_message = str(e)
            self.tasks_failed += 1
            log_universal('ERROR', 'AnalysisWorker', 
                         f'Analysis failed: {task.file_path} - {e}')
            
        finally:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Release resources
            self.orchestrator.resource_coordinator.release_task_resources(
                self.worker_id, task.required_ram_gb, processing_time, success
            )
            
            # Complete task tracking
            self.orchestrator.progress_monitor.complete_task(
                task.task_id, success, processing_time, error_message
            )
            
            # Complete task in queue
            self.orchestrator.queue_manager.complete_task(
                task.task_id, success, processing_time, error_message
            )
            
            self.current_task_id = None
            self.last_activity = datetime.now()
    
    def _run_analysis(self, task, task_metrics) -> bool:
        """Run the appropriate analysis based on task strategy."""
        try:
            # Import analyzers
            from .audio_analyzer import AudioAnalyzer
            from .config_loader import config_loader
            
            # Update progress
            self.orchestrator.progress_monitor.update_task_progress(
                task.task_id, 20.0, "initializing_analyzer"
            )
            
            # Get analysis configuration
            config = config_loader.get_audio_analysis_config()
            
            # Create analyzer
            analyzer = AudioAnalyzer(config=config)
            
            # Update progress
            self.orchestrator.progress_monitor.update_task_progress(
                task.task_id, 30.0, "analyzing_audio"
            )
            
            # Perform analysis
            result = analyzer.analyze_audio_file(task.file_path, task.force_reanalysis)
            
            # Update progress
            self.orchestrator.progress_monitor.update_task_progress(
                task.task_id, 90.0, "saving_results"
            )
            
            return result is not None
            
        except Exception as e:
            log_universal('ERROR', 'AnalysisWorker', f'Analysis execution failed: {e}')
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        return {
            'worker_id': self.worker_id,
            'worker_type': self.worker_type.value,
            'is_active': self.is_active,
            'current_task_id': self.current_task_id,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'total_processing_time': self.total_processing_time,
            'average_task_time': (self.total_processing_time / max(1, self.tasks_completed + self.tasks_failed)),
            'success_rate': (self.tasks_completed / max(1, self.tasks_completed + self.tasks_failed)),
            'last_activity': self.last_activity.isoformat()
        }


class AnalysisOrchestrator:
    """
    Central orchestrator for the entire analysis pipeline.
    
    Features:
    - Centralized coordination of all analysis components
    - Intelligent worker management and scaling
    - Resource optimization and load balancing
    - Health monitoring and recovery
    - Performance analytics and reporting
    - Graceful scaling and shutdown
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the analysis orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_audio_analysis_config()
        
        self.config = config
        
        # Initialize orchestrator configuration
        self.orchestrator_config = OrchestratorConfig(
            max_concurrent_tasks=config.get('MAX_CONCURRENT_TASKS', 10),
            worker_spawn_delay=config.get('WORKER_SPAWN_DELAY', 1.0),
            health_check_interval=config.get('HEALTH_CHECK_INTERVAL', 30),
            statistics_update_interval=config.get('STATISTICS_UPDATE_INTERVAL', 10),
            auto_scaling_enabled=config.get('AUTO_SCALING_ENABLED', True),
            retry_failed_tasks=config.get('RETRY_FAILED_TASKS', True),
            max_retry_attempts=config.get('MAX_RETRY_ATTEMPTS', 3),
            task_timeout_seconds=config.get('TASK_TIMEOUT_SECONDS', 600),
            graceful_shutdown_timeout=config.get('GRACEFUL_SHUTDOWN_TIMEOUT', 30)
        )
        
        # Initialize core components
        self.queue_manager = get_queue_manager(config)
        self.progress_monitor = get_progress_monitor(config)
        self.resource_coordinator = get_resource_coordinator(config)
        self.resource_manager = get_resource_manager()
        self.db_manager = get_db_manager()
        
        # State management
        self.state = OrchestratorState.IDLE
        self._workers: Dict[str, AnalysisWorker] = {}
        self._worker_counter = 0
        
        # Statistics
        self._start_time: Optional[datetime] = None
        self._total_files_processed = 0
        self._total_files_failed = 0
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Background services
        self._health_monitor_thread: Optional[threading.Thread] = None
        self._statistics_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._state_change_callbacks: List[Callable] = []
        self._completion_callbacks: List[Callable] = []
        
        log_universal('INFO', 'AnalysisOrchestrator', 'Analysis orchestrator initialized')
        log_universal('INFO', 'AnalysisOrchestrator', 
                     f'Configuration: max_tasks={self.orchestrator_config.max_concurrent_tasks}, '
                     f'auto_scaling={self.orchestrator_config.auto_scaling_enabled}')
    
    def start_analysis_pipeline(self, files: List[str], force_reanalysis: bool = False) -> str:
        """
        Start the analysis pipeline for a list of files.
        
        Args:
            files: List of file paths to analyze
            force_reanalysis: Force re-analysis even if cached
            
        Returns:
            Pipeline run ID for tracking
        """
        with self._lock:
            if self.state != OrchestratorState.IDLE:
                raise RuntimeError(f"Cannot start pipeline in state: {self.state}")
            
            self._change_state(OrchestratorState.STARTING)
            
            try:
                # Generate pipeline run ID
                run_id = f"pipeline_{int(time.time())}"
                
                # Queue all files for analysis
                task_ids = []
                for file_path in files:
                    try:
                        # Get file size for PLAYLISTA strategy
                        file_size_mb = self.db_manager.get_file_size_mb(file_path)
                        
                        # Determine task priority based on file size
                        if file_size_mb < 25:
                            priority = TaskPriority.HIGH  # Small files are fast
                        elif file_size_mb < 50:
                            priority = TaskPriority.NORMAL
                        else:
                            priority = TaskPriority.LOW  # Large files are slow
                        
                        # Add task to queue
                        task_id = self.queue_manager.add_task(
                            file_path=file_path,
                            file_size_mb=file_size_mb,
                            priority=priority,
                            force_reanalysis=force_reanalysis,
                            timeout_seconds=self.orchestrator_config.task_timeout_seconds
                        )
                        
                        task_ids.append(task_id)
                        
                    except Exception as e:
                        log_universal('WARNING', 'AnalysisOrchestrator', 
                                     f'Failed to queue file {file_path}: {e}')
                
                log_universal('INFO', 'AnalysisOrchestrator', 
                             f'Pipeline started: {len(task_ids)} tasks queued (run_id: {run_id})')
                
                # Start workers and background services
                self._start_workers()
                self._start_background_services()
                
                self._change_state(OrchestratorState.RUNNING)
                self._start_time = datetime.now()
                
                return run_id
                
            except Exception as e:
                log_universal('ERROR', 'AnalysisOrchestrator', f'Failed to start pipeline: {e}')
                self._change_state(OrchestratorState.ERROR)
                raise
    
    def pause_pipeline(self):
        """Pause the analysis pipeline."""
        with self._lock:
            if self.state == OrchestratorState.RUNNING:
                self._change_state(OrchestratorState.PAUSING)
                
                # Stop workers gracefully
                self._stop_workers(graceful=True)
                
                self._change_state(OrchestratorState.PAUSED)
                log_universal('INFO', 'AnalysisOrchestrator', 'Pipeline paused')
    
    def resume_pipeline(self):
        """Resume the paused analysis pipeline."""
        with self._lock:
            if self.state == OrchestratorState.PAUSED:
                self._change_state(OrchestratorState.STARTING)
                
                # Restart workers
                self._start_workers()
                
                self._change_state(OrchestratorState.RUNNING)
                log_universal('INFO', 'AnalysisOrchestrator', 'Pipeline resumed')
    
    def stop_pipeline(self, graceful: bool = True):
        """Stop the analysis pipeline."""
        with self._lock:
            if self.state in [OrchestratorState.RUNNING, OrchestratorState.PAUSED]:
                self._change_state(OrchestratorState.STOPPING)
                
                # Stop background services
                self._stop_background_services()
                
                # Stop workers
                self._stop_workers(graceful=graceful)
                
                self._change_state(OrchestratorState.STOPPED)
                log_universal('INFO', 'AnalysisOrchestrator', 'Pipeline stopped')
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        with self._lock:
            # Get component statistics
            queue_stats = self.queue_manager.get_statistics()
            progress_stats = self.progress_monitor.get_overall_progress()
            resource_status = self.resource_coordinator.get_status()
            performance_summary = self.progress_monitor.get_performance_summary()
            
            # Calculate pipeline metrics
            total_tasks = queue_stats['total_tasks_queued']
            completed_tasks = queue_stats['total_tasks_completed']
            failed_tasks = queue_stats['total_tasks_failed']
            active_tasks = queue_stats['active_tasks']
            
            completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            # Calculate estimated completion time
            processing_rate = progress_stats.get('processing_rate_per_second', 0)
            remaining_tasks = total_tasks - completed_tasks - failed_tasks
            estimated_completion = None
            
            if processing_rate > 0 and remaining_tasks > 0:
                estimated_seconds = remaining_tasks / processing_rate
                estimated_completion = (datetime.now() + timedelta(seconds=estimated_seconds)).isoformat()
            
            return {
                'orchestrator_state': self.state.value,
                'pipeline_metrics': {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'failed_tasks': failed_tasks,
                    'active_tasks': active_tasks,
                    'completion_percentage': completion_percentage,
                    'estimated_completion': estimated_completion
                },
                'workers': {
                    worker_id: worker.get_status()
                    for worker_id, worker in self._workers.items()
                },
                'queue_statistics': queue_stats,
                'progress_statistics': progress_stats,
                'resource_status': resource_status,
                'performance_summary': performance_summary,
                'uptime_seconds': (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            }
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable):
        """Add callback for pipeline completion."""
        self._completion_callbacks.append(callback)
    
    def _start_workers(self):
        """Start analysis workers based on system capacity."""
        # Get system capacity
        system_resources = self.resource_manager.get_current_resources()
        available_memory_gb = system_resources.get('available_memory_gb', 8)
        cpu_cores = system_resources.get('cpu_cores', 4)
        
        # Calculate optimal worker configuration based on PLAYLISTA rules
        max_sequential_workers = max(1, int(available_memory_gb // 4))  # 4GB per sequential worker
        max_parallel_large_workers = max(1, int(available_memory_gb // 2))  # 2GB per large worker
        max_parallel_small_workers = max(1, int(available_memory_gb // 1.5))  # 1.5GB per small worker
        
        # Limit by CPU cores
        total_max_workers = min(cpu_cores, max_sequential_workers + max_parallel_large_workers + max_parallel_small_workers)
        
        # Create workers
        workers_to_create = [
            (WorkerType.SEQUENTIAL, min(2, max_sequential_workers)),
            (WorkerType.PARALLEL_LARGE, min(3, max_parallel_large_workers)),
            (WorkerType.PARALLEL_SMALL, min(4, max_parallel_small_workers))
        ]
        
        for worker_type, count in workers_to_create:
            for i in range(count):
                self._create_worker(worker_type)
        
        log_universal('INFO', 'AnalysisOrchestrator', 
                     f'Started {len(self._workers)} workers: '
                     f'{sum(1 for w in self._workers.values() if w.worker_type == WorkerType.SEQUENTIAL)} sequential, '
                     f'{sum(1 for w in self._workers.values() if w.worker_type == WorkerType.PARALLEL_LARGE)} parallel_large, '
                     f'{sum(1 for w in self._workers.values() if w.worker_type == WorkerType.PARALLEL_SMALL)} parallel_small')
    
    def _create_worker(self, worker_type: WorkerType) -> AnalysisWorker:
        """Create and start a new worker."""
        with self._lock:
            self._worker_counter += 1
            worker_id = f"{worker_type.value}_worker_{self._worker_counter}"
            
            worker = AnalysisWorker(worker_id, worker_type, self)
            self._workers[worker_id] = worker
            
            # Register with resource coordinator
            if worker_type == WorkerType.SEQUENTIAL:
                max_threads = 1
                memory_limit_gb = 4.0
            elif worker_type == WorkerType.PARALLEL_LARGE:
                max_threads = 2
                memory_limit_gb = 4.0  # 2GB per thread
            else:  # PARALLEL_SMALL
                max_threads = 3
                memory_limit_gb = 4.5  # 1.5GB per thread
            
            self.resource_coordinator.register_worker(worker_id, max_threads, memory_limit_gb)
            
            # Start the worker
            worker.start()
            
            return worker
    
    def _stop_workers(self, graceful: bool = True):
        """Stop all workers."""
        timeout = self.orchestrator_config.graceful_shutdown_timeout if graceful else 5.0
        
        for worker in self._workers.values():
            worker.stop(timeout=timeout)
        
        self._workers.clear()
    
    def _start_background_services(self):
        """Start background monitoring services."""
        # Health monitor
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_worker, daemon=True
        )
        self._health_monitor_thread.start()
        
        # Statistics updater
        self._statistics_thread = threading.Thread(
            target=self._statistics_worker, daemon=True
        )
        self._statistics_thread.start()
    
    def _stop_background_services(self):
        """Stop background services."""
        self._shutdown_event.set()
        
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=5)
        
        if self._statistics_thread and self._statistics_thread.is_alive():
            self._statistics_thread.join(timeout=5)
    
    def _health_monitor_worker(self):
        """Background worker for health monitoring."""
        while not self._shutdown_event.wait(self.orchestrator_config.health_check_interval):
            try:
                self._perform_health_checks()
                self._auto_scale_workers()
            except Exception as e:
                log_universal('ERROR', 'AnalysisOrchestrator', f'Health monitor error: {e}')
    
    def _statistics_worker(self):
        """Background worker for statistics updates."""
        while not self._shutdown_event.wait(self.orchestrator_config.statistics_update_interval):
            try:
                self._update_statistics()
            except Exception as e:
                log_universal('ERROR', 'AnalysisOrchestrator', f'Statistics worker error: {e}')
    
    def _perform_health_checks(self):
        """Perform health checks on workers and components."""
        current_time = datetime.now()
        unhealthy_workers = []
        
        for worker_id, worker in self._workers.items():
            # Check if worker is responsive
            if worker.is_active and worker.current_task_id is None:
                time_since_activity = (current_time - worker.last_activity).total_seconds()
                if time_since_activity > 300:  # 5 minutes without activity
                    unhealthy_workers.append(worker_id)
        
        # Restart unhealthy workers
        for worker_id in unhealthy_workers:
            try:
                worker = self._workers[worker_id]
                worker_type = worker.worker_type
                
                log_universal('WARNING', 'AnalysisOrchestrator', 
                             f'Restarting unresponsive worker: {worker_id}')
                
                worker.stop(timeout=5.0)
                del self._workers[worker_id]
                
                # Create replacement worker
                self._create_worker(worker_type)
                
            except Exception as e:
                log_universal('ERROR', 'AnalysisOrchestrator', 
                             f'Failed to restart worker {worker_id}: {e}')
    
    def _auto_scale_workers(self):
        """Automatically scale workers based on load and resources."""
        if not self.orchestrator_config.auto_scaling_enabled:
            return
        
        try:
            # Get current metrics
            queue_stats = self.queue_manager.get_statistics()
            resource_recommendations = self.resource_coordinator.get_resource_recommendations()
            
            queue_size = queue_stats['queue_size']
            active_workers = len([w for w in self._workers.values() if w.is_active])
            
            # Scale up if queue is backing up and resources allow
            if queue_size > active_workers * 2 and active_workers < self.orchestrator_config.max_concurrent_tasks:
                recommendations = resource_recommendations.get('recommendations', [])
                can_scale_up = any(r['type'] == 'underutilization' for r in recommendations)
                
                if can_scale_up:
                    # Create additional parallel_small worker (fastest to process small files)
                    self._create_worker(WorkerType.PARALLEL_SMALL)
                    log_universal('INFO', 'AnalysisOrchestrator', 'Auto-scaled up: added parallel_small worker')
            
            # Scale down if too many idle workers
            elif queue_size == 0 and active_workers > 2:
                # Find least active worker to remove
                idle_workers = [w for w in self._workers.values() 
                              if w.is_active and w.current_task_id is None]
                
                if idle_workers:
                    worker_to_remove = min(idle_workers, key=lambda w: w.tasks_completed)
                    worker_to_remove.stop()
                    del self._workers[worker_to_remove.worker_id]
                    log_universal('INFO', 'AnalysisOrchestrator', 
                                 f'Auto-scaled down: removed worker {worker_to_remove.worker_id}')
                    
        except Exception as e:
            log_universal('ERROR', 'AnalysisOrchestrator', f'Auto-scaling error: {e}')
    
    def _update_statistics(self):
        """Update pipeline statistics."""
        try:
            # Update total files processed
            queue_stats = self.queue_manager.get_statistics()
            self._total_files_processed = queue_stats['total_tasks_completed']
            self._total_files_failed = queue_stats['total_tasks_failed']
            
            # Check for completion
            if (self.state == OrchestratorState.RUNNING and 
                queue_stats['queue_size'] == 0 and 
                queue_stats['active_tasks'] == 0 and
                self._total_files_processed > 0):
                
                log_universal('INFO', 'AnalysisOrchestrator', 
                             f'Pipeline completed: {self._total_files_processed} files processed, '
                             f'{self._total_files_failed} failed')
                
                # Notify completion callbacks
                for callback in self._completion_callbacks:
                    try:
                        callback({
                            'total_processed': self._total_files_processed,
                            'total_failed': self._total_files_failed,
                            'completion_time': datetime.now().isoformat()
                        })
                    except Exception as e:
                        log_universal('WARNING', 'AnalysisOrchestrator', f'Completion callback failed: {e}')
                
                # Transition to idle state
                self.stop_pipeline(graceful=True)
                self._change_state(OrchestratorState.IDLE)
                
        except Exception as e:
            log_universal('ERROR', 'AnalysisOrchestrator', f'Statistics update error: {e}')
    
    def _change_state(self, new_state: OrchestratorState):
        """Change orchestrator state and notify callbacks."""
        old_state = self.state
        self.state = new_state
        
        log_universal('INFO', 'AnalysisOrchestrator', f'State changed: {old_state.value} -> {new_state.value}')
        
        # Notify state change callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                log_universal('WARNING', 'AnalysisOrchestrator', f'State change callback failed: {e}')
    
    def shutdown(self):
        """Shutdown the orchestrator completely."""
        log_universal('INFO', 'AnalysisOrchestrator', 'Shutting down analysis orchestrator')
        
        self.stop_pipeline(graceful=True)
        
        # Shutdown components
        self.queue_manager.shutdown()
        self.progress_monitor.shutdown()
        self.resource_coordinator.shutdown()


# Global analysis orchestrator instance
_analysis_orchestrator_instance = None
_analysis_orchestrator_lock = threading.Lock()

def get_analysis_orchestrator(config: Dict[str, Any] = None) -> AnalysisOrchestrator:
    """Get the global analysis orchestrator instance."""
    global _analysis_orchestrator_instance
    
    if _analysis_orchestrator_instance is None:
        with _analysis_orchestrator_lock:
            if _analysis_orchestrator_instance is None:
                _analysis_orchestrator_instance = AnalysisOrchestrator(config)
    
    return _analysis_orchestrator_instance
