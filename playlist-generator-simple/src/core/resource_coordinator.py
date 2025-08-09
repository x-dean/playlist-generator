"""
Resource Coordinator for Playlist Generator Simple.
Provides dynamic resource rebalancing, optimization, and intelligent workload distribution.
"""

import os
import time
import threading
import psutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import local modules
from .logging_setup import get_logger, log_function_call, log_universal
from .resource_manager import get_resource_manager

logger = get_logger('playlista.resource_coordinator')


class ResourcePressure(Enum):
    """System resource pressure levels."""
    LOW = "low"           # <50% utilization
    MODERATE = "moderate" # 50-75% utilization
    HIGH = "high"         # 75-90% utilization
    CRITICAL = "critical" # >90% utilization


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RESOURCE_AWARE = "resource_aware"
    PRIORITY_BASED = "priority_based"


@dataclass
class ResourceProfile:
    """System resource profile at a point in time."""
    timestamp: datetime
    memory_total_gb: float
    memory_available_gb: float
    memory_used_percent: float
    cpu_cores: int
    cpu_usage_percent: float
    disk_io_read_mb_s: float
    disk_io_write_mb_s: float
    network_io_mb_s: float
    
    def get_memory_pressure(self) -> ResourcePressure:
        """Determine memory pressure level."""
        if self.memory_used_percent < 50:
            return ResourcePressure.LOW
        elif self.memory_used_percent < 75:
            return ResourcePressure.MODERATE
        elif self.memory_used_percent < 90:
            return ResourcePressure.HIGH
        else:
            return ResourcePressure.CRITICAL
    
    def get_cpu_pressure(self) -> ResourcePressure:
        """Determine CPU pressure level."""
        if self.cpu_usage_percent < 50:
            return ResourcePressure.LOW
        elif self.cpu_usage_percent < 75:
            return ResourcePressure.MODERATE
        elif self.cpu_usage_percent < 90:
            return ResourcePressure.HIGH
        else:
            return ResourcePressure.CRITICAL
    
    def get_overall_pressure(self) -> ResourcePressure:
        """Get overall system pressure (worst of memory and CPU)."""
        memory_pressure = self.get_memory_pressure()
        cpu_pressure = self.get_cpu_pressure()
        
        # Return the higher pressure level
        pressures = [ResourcePressure.LOW, ResourcePressure.MODERATE, 
                    ResourcePressure.HIGH, ResourcePressure.CRITICAL]
        memory_idx = pressures.index(memory_pressure)
        cpu_idx = pressures.index(cpu_pressure)
        
        return pressures[max(memory_idx, cpu_idx)]


@dataclass
class WorkerCapacity:
    """Worker capacity and current load."""
    worker_id: str
    max_threads: int
    current_tasks: int
    memory_allocated_gb: float
    memory_limit_gb: float
    last_task_completion: Optional[datetime] = None
    average_task_time: float = 0.0
    success_rate: float = 1.0
    
    def get_load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0)."""
        return self.current_tasks / self.max_threads if self.max_threads > 0 else 1.0
    
    def get_memory_utilization(self) -> float:
        """Calculate memory utilization (0.0 to 1.0)."""
        return self.memory_allocated_gb / self.memory_limit_gb if self.memory_limit_gb > 0 else 1.0
    
    def can_accept_task(self, required_memory_gb: float) -> bool:
        """Check if worker can accept a new task."""
        return (self.current_tasks < self.max_threads and 
                self.memory_allocated_gb + required_memory_gb <= self.memory_limit_gb)


class ResourceCoordinator:
    """
    Dynamic resource coordinator for intelligent workload management.
    
    Features:
    - Real-time resource monitoring
    - Dynamic thread rebalancing
    - Intelligent workload distribution
    - Resource pressure response
    - Performance optimization
    - Adaptive scaling
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the resource coordinator.
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_audio_analysis_config()
        
        self.config = config
        self.resource_manager = get_resource_manager()
        
        # Resource monitoring
        self._current_profile: Optional[ResourceProfile] = None
        self._resource_history: List[ResourceProfile] = []
        self._max_history_size = 1000
        
        # Worker management
        self._workers: Dict[str, WorkerCapacity] = {}
        self._load_balancing_strategy = LoadBalancingStrategy.RESOURCE_AWARE
        
        # Resource thresholds and limits
        self.memory_high_threshold = config.get('MEMORY_HIGH_THRESHOLD', 0.80)
        self.memory_critical_threshold = config.get('MEMORY_CRITICAL_THRESHOLD', 0.90)
        self.cpu_high_threshold = config.get('CPU_HIGH_THRESHOLD', 0.80)
        self.cpu_critical_threshold = config.get('CPU_CRITICAL_THRESHOLD', 0.95)
        
        # Adaptive scaling parameters
        self.min_threads_per_worker = config.get('MIN_THREADS_PER_WORKER', 1)
        self.max_threads_per_worker = config.get('MAX_THREADS_PER_WORKER', 8)
        self.scale_up_threshold = config.get('SCALE_UP_THRESHOLD', 0.70)
        self.scale_down_threshold = config.get('SCALE_DOWN_THRESHOLD', 0.30)
        
        # Monitoring intervals
        self.monitoring_interval = config.get('RESOURCE_MONITORING_INTERVAL', 15)
        self.rebalancing_interval = config.get('REBALANCING_INTERVAL', 60)
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Start background services
        self._monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self._rebalancer_thread = threading.Thread(target=self._rebalancing_worker, daemon=True)
        
        self._monitor_thread.start()
        self._rebalancer_thread.start()
        
        log_universal('INFO', 'ResourceCoordinator', 'Resource coordinator initialized')
        log_universal('INFO', 'ResourceCoordinator', 
                     f'Thresholds - Memory: {self.memory_high_threshold:.0%}/{self.memory_critical_threshold:.0%}, '
                     f'CPU: {self.cpu_high_threshold:.0%}/{self.cpu_critical_threshold:.0%}')
    
    def register_worker(self, worker_id: str, max_threads: int, 
                       memory_limit_gb: float) -> WorkerCapacity:
        """
        Register a new worker with the coordinator.
        
        Args:
            worker_id: Unique worker identifier
            max_threads: Maximum threads for this worker
            memory_limit_gb: Memory limit in GB
            
        Returns:
            WorkerCapacity object for tracking
        """
        with self._lock:
            capacity = WorkerCapacity(
                worker_id=worker_id,
                max_threads=max_threads,
                current_tasks=0,
                memory_allocated_gb=0.0,
                memory_limit_gb=memory_limit_gb
            )
            
            self._workers[worker_id] = capacity
            
            log_universal('INFO', 'ResourceCoordinator', 
                         f'Worker registered: {worker_id} - {max_threads} threads, {memory_limit_gb}GB memory')
            
            return capacity
    
    def select_worker_for_task(self, required_memory_gb: float, 
                              task_priority: str = "normal") -> Optional[str]:
        """
        Select the best worker for a task based on current strategy.
        
        Args:
            required_memory_gb: Memory required for the task
            task_priority: Task priority level
            
        Returns:
            Worker ID or None if no suitable worker available
        """
        with self._lock:
            available_workers = [
                (worker_id, capacity) 
                for worker_id, capacity in self._workers.items()
                if capacity.can_accept_task(required_memory_gb)
            ]
            
            if not available_workers:
                return None
            
            # Apply load balancing strategy
            if self._load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
                return self._select_least_loaded_worker(available_workers)
            elif self._load_balancing_strategy == LoadBalancingStrategy.RESOURCE_AWARE:
                return self._select_resource_aware_worker(available_workers, required_memory_gb)
            elif self._load_balancing_strategy == LoadBalancingStrategy.PRIORITY_BASED:
                return self._select_priority_based_worker(available_workers, task_priority)
            else:  # ROUND_ROBIN
                return self._select_round_robin_worker(available_workers)
    
    def allocate_task_resources(self, worker_id: str, required_memory_gb: float) -> bool:
        """
        Allocate resources for a task to a worker.
        
        Args:
            worker_id: Worker identifier
            required_memory_gb: Memory to allocate
            
        Returns:
            True if allocation successful
        """
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            capacity = self._workers[worker_id]
            
            if capacity.can_accept_task(required_memory_gb):
                capacity.current_tasks += 1
                capacity.memory_allocated_gb += required_memory_gb
                
                log_universal('DEBUG', 'ResourceCoordinator', 
                             f'Resources allocated to {worker_id}: {required_memory_gb}GB '
                             f'({capacity.current_tasks}/{capacity.max_threads} tasks)')
                
                return True
            
            return False
    
    def release_task_resources(self, worker_id: str, used_memory_gb: float, 
                              task_duration: float, success: bool):
        """
        Release resources after task completion.
        
        Args:
            worker_id: Worker identifier
            used_memory_gb: Memory to release
            task_duration: Task processing time
            success: Whether task completed successfully
        """
        with self._lock:
            if worker_id not in self._workers:
                return
            
            capacity = self._workers[worker_id]
            
            # Release resources
            capacity.current_tasks = max(0, capacity.current_tasks - 1)
            capacity.memory_allocated_gb = max(0, capacity.memory_allocated_gb - used_memory_gb)
            capacity.last_task_completion = datetime.now()
            
            # Update performance metrics
            if task_duration > 0:
                # Exponential moving average for task time
                alpha = 0.1
                capacity.average_task_time = (
                    alpha * task_duration + 
                    (1 - alpha) * capacity.average_task_time
                )
            
            # Update success rate
            if success:
                capacity.success_rate = min(1.0, capacity.success_rate + 0.01)
            else:
                capacity.success_rate = max(0.0, capacity.success_rate - 0.05)
            
            log_universal('DEBUG', 'ResourceCoordinator', 
                         f'Resources released from {worker_id}: {used_memory_gb}GB '
                         f'({capacity.current_tasks}/{capacity.max_threads} tasks)')
    
    def get_resource_recommendations(self) -> Dict[str, Any]:
        """
        Get resource optimization recommendations.
        
        Returns:
            Dictionary with recommendations
        """
        with self._lock:
            if not self._current_profile:
                return {'recommendations': []}
            
            recommendations = []
            
            # Memory recommendations
            memory_pressure = self._current_profile.get_memory_pressure()
            if memory_pressure == ResourcePressure.CRITICAL:
                recommendations.append({
                    'type': 'memory_critical',
                    'action': 'reduce_workers',
                    'message': 'Critical memory usage - consider reducing worker count',
                    'priority': 'high'
                })
            elif memory_pressure == ResourcePressure.HIGH:
                recommendations.append({
                    'type': 'memory_high',
                    'action': 'limit_parallel_tasks',
                    'message': 'High memory usage - limit parallel task execution',
                    'priority': 'medium'
                })
            
            # CPU recommendations
            cpu_pressure = self._current_profile.get_cpu_pressure()
            if cpu_pressure == ResourcePressure.CRITICAL:
                recommendations.append({
                    'type': 'cpu_critical',
                    'action': 'reduce_threads',
                    'message': 'Critical CPU usage - reduce thread count per worker',
                    'priority': 'high'
                })
            
            # Load balancing recommendations
            worker_loads = [capacity.get_load_factor() for capacity in self._workers.values()]
            if worker_loads:
                load_variance = max(worker_loads) - min(worker_loads)
                if load_variance > 0.3:
                    recommendations.append({
                        'type': 'load_imbalance',
                        'action': 'rebalance_workers',
                        'message': 'Significant load imbalance detected between workers',
                        'priority': 'medium'
                    })
            
            # Resource utilization recommendations
            if self._current_profile.memory_used_percent < 30 and self._current_profile.cpu_usage_percent < 30:
                recommendations.append({
                    'type': 'underutilization',
                    'action': 'increase_parallelism',
                    'message': 'System resources underutilized - consider increasing parallelism',
                    'priority': 'low'
                })
            
            return {
                'recommendations': recommendations,
                'current_pressure': {
                    'memory': memory_pressure.value,
                    'cpu': cpu_pressure.value,
                    'overall': self._current_profile.get_overall_pressure().value
                },
                'resource_utilization': {
                    'memory_percent': self._current_profile.memory_used_percent,
                    'cpu_percent': self._current_profile.cpu_usage_percent,
                    'worker_load_factors': {
                        worker_id: capacity.get_load_factor()
                        for worker_id, capacity in self._workers.items()
                    }
                }
            }
    
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """
        Perform automatic resource optimization.
        
        Returns:
            Dictionary with optimization actions taken
        """
        with self._lock:
            actions_taken = []
            
            if not self._current_profile:
                return {'actions': actions_taken}
            
            # Get current pressure levels
            memory_pressure = self._current_profile.get_memory_pressure()
            cpu_pressure = self._current_profile.get_cpu_pressure()
            
            # Optimize based on pressure levels
            if memory_pressure == ResourcePressure.CRITICAL or cpu_pressure == ResourcePressure.CRITICAL:
                # Emergency optimization: reduce resource usage
                actions_taken.extend(self._emergency_resource_reduction())
            
            elif memory_pressure == ResourcePressure.HIGH or cpu_pressure == ResourcePressure.HIGH:
                # Conservative optimization: gradual reduction
                actions_taken.extend(self._conservative_optimization())
            
            elif memory_pressure == ResourcePressure.LOW and cpu_pressure == ResourcePressure.LOW:
                # Aggressive optimization: increase utilization
                actions_taken.extend(self._aggressive_optimization())
            
            # Rebalance worker loads
            actions_taken.extend(self._rebalance_workers())
            
            return {'actions': actions_taken}
    
    def _select_least_loaded_worker(self, available_workers: List[Tuple[str, WorkerCapacity]]) -> str:
        """Select worker with lowest load factor."""
        return min(available_workers, key=lambda x: x[1].get_load_factor())[0]
    
    def _select_resource_aware_worker(self, available_workers: List[Tuple[str, WorkerCapacity]], 
                                    required_memory_gb: float) -> str:
        """Select worker based on resource availability and efficiency."""
        def score_worker(worker_id: str, capacity: WorkerCapacity) -> float:
            # Score based on multiple factors
            load_factor = capacity.get_load_factor()
            memory_util = capacity.get_memory_utilization()
            success_rate = capacity.success_rate
            
            # Lower is better for load and memory utilization
            # Higher is better for success rate
            score = (1 - load_factor) * 0.4 + (1 - memory_util) * 0.3 + success_rate * 0.3
            
            return score
        
        best_worker = max(available_workers, key=lambda x: score_worker(x[0], x[1]))
        return best_worker[0]
    
    def _select_priority_based_worker(self, available_workers: List[Tuple[str, WorkerCapacity]], 
                                    task_priority: str) -> str:
        """Select worker based on task priority."""
        # For high priority tasks, prefer least loaded workers
        if task_priority in ['high', 'critical']:
            return self._select_least_loaded_worker(available_workers)
        else:
            return self._select_resource_aware_worker(available_workers, 0)
    
    def _select_round_robin_worker(self, available_workers: List[Tuple[str, WorkerCapacity]]) -> str:
        """Select worker using round-robin strategy."""
        # Simple round-robin based on last task completion time
        return min(available_workers, 
                  key=lambda x: x[1].last_task_completion or datetime.min)[0]
    
    def _emergency_resource_reduction(self) -> List[Dict[str, Any]]:
        """Perform emergency resource reduction."""
        actions = []
        
        # Reduce threads on overloaded workers
        for worker_id, capacity in self._workers.items():
            if capacity.max_threads > self.min_threads_per_worker:
                old_threads = capacity.max_threads
                capacity.max_threads = max(self.min_threads_per_worker, 
                                         capacity.max_threads - 1)
                
                actions.append({
                    'type': 'reduce_worker_threads',
                    'worker_id': worker_id,
                    'old_threads': old_threads,
                    'new_threads': capacity.max_threads
                })
        
        return actions
    
    def _conservative_optimization(self) -> List[Dict[str, Any]]:
        """Perform conservative optimization."""
        actions = []
        
        # Slightly reduce threads on heavily loaded workers
        for worker_id, capacity in self._workers.items():
            if capacity.get_load_factor() > 0.8 and capacity.max_threads > self.min_threads_per_worker:
                old_threads = capacity.max_threads
                capacity.max_threads = max(self.min_threads_per_worker, 
                                         capacity.max_threads - 1)
                
                actions.append({
                    'type': 'conservative_thread_reduction',
                    'worker_id': worker_id,
                    'old_threads': old_threads,
                    'new_threads': capacity.max_threads
                })
        
        return actions
    
    def _aggressive_optimization(self) -> List[Dict[str, Any]]:
        """Perform aggressive optimization to increase utilization."""
        actions = []
        
        # Increase threads on underutilized workers
        for worker_id, capacity in self._workers.items():
            if (capacity.get_load_factor() < self.scale_up_threshold and 
                capacity.max_threads < self.max_threads_per_worker):
                
                old_threads = capacity.max_threads
                capacity.max_threads = min(self.max_threads_per_worker, 
                                         capacity.max_threads + 1)
                
                actions.append({
                    'type': 'aggressive_thread_increase',
                    'worker_id': worker_id,
                    'old_threads': old_threads,
                    'new_threads': capacity.max_threads
                })
        
        return actions
    
    def _rebalance_workers(self) -> List[Dict[str, Any]]:
        """Rebalance worker loads."""
        actions = []
        
        if len(self._workers) < 2:
            return actions
        
        # Calculate load variance
        loads = [capacity.get_load_factor() for capacity in self._workers.values()]
        load_variance = max(loads) - min(loads)
        
        if load_variance > 0.3:  # Significant imbalance
            # Find overloaded and underloaded workers
            overloaded = [(wid, cap) for wid, cap in self._workers.items() 
                         if cap.get_load_factor() > 0.7]
            underloaded = [(wid, cap) for wid, cap in self._workers.items() 
                          if cap.get_load_factor() < 0.3]
            
            # Suggest task redistribution (would need queue manager integration)
            if overloaded and underloaded:
                actions.append({
                    'type': 'suggest_task_redistribution',
                    'overloaded_workers': [wid for wid, _ in overloaded],
                    'underloaded_workers': [wid for wid, _ in underloaded]
                })
        
        return actions
    
    def _monitoring_worker(self):
        """Background worker for resource monitoring."""
        while not self._shutdown_event.wait(self.monitoring_interval):
            try:
                self._update_resource_profile()
            except Exception as e:
                log_universal('ERROR', 'ResourceCoordinator', f'Monitoring worker error: {e}')
    
    def _rebalancing_worker(self):
        """Background worker for resource rebalancing."""
        while not self._shutdown_event.wait(self.rebalancing_interval):
            try:
                optimization_result = self.optimize_resource_allocation()
                if optimization_result['actions']:
                    log_universal('INFO', 'ResourceCoordinator', 
                                 f'Performed {len(optimization_result["actions"])} optimization actions')
            except Exception as e:
                log_universal('ERROR', 'ResourceCoordinator', f'Rebalancing worker error: {e}')
    
    def _update_resource_profile(self):
        """Update current resource profile."""
        try:
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get I/O stats (if available)
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            profile = ResourceProfile(
                timestamp=datetime.now(),
                memory_total_gb=memory.total / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                memory_used_percent=memory.percent,
                cpu_cores=cpu_count,
                cpu_usage_percent=cpu_percent,
                disk_io_read_mb_s=0,  # Would need rate calculation
                disk_io_write_mb_s=0,  # Would need rate calculation
                network_io_mb_s=0  # Would need rate calculation
            )
            
            with self._lock:
                self._current_profile = profile
                self._resource_history.append(profile)
                
                # Limit history size
                if len(self._resource_history) > self._max_history_size:
                    self._resource_history = self._resource_history[-self._max_history_size:]
                
        except Exception as e:
            log_universal('WARNING', 'ResourceCoordinator', f'Resource profile update failed: {e}')
    
    def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status."""
        with self._lock:
            return {
                'current_profile': {
                    'memory_used_percent': self._current_profile.memory_used_percent if self._current_profile else 0,
                    'cpu_usage_percent': self._current_profile.cpu_usage_percent if self._current_profile else 0,
                    'memory_pressure': self._current_profile.get_memory_pressure().value if self._current_profile else 'unknown',
                    'cpu_pressure': self._current_profile.get_cpu_pressure().value if self._current_profile else 'unknown'
                },
                'workers': {
                    worker_id: {
                        'max_threads': capacity.max_threads,
                        'current_tasks': capacity.current_tasks,
                        'load_factor': capacity.get_load_factor(),
                        'memory_utilization': capacity.get_memory_utilization(),
                        'success_rate': capacity.success_rate,
                        'average_task_time': capacity.average_task_time
                    }
                    for worker_id, capacity in self._workers.items()
                },
                'load_balancing_strategy': self._load_balancing_strategy.value,
                'total_workers': len(self._workers),
                'active_tasks': sum(capacity.current_tasks for capacity in self._workers.values())
            }
    
    def shutdown(self):
        """Shutdown the resource coordinator."""
        log_universal('INFO', 'ResourceCoordinator', 'Shutting down resource coordinator')
        self._shutdown_event.set()
        
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        if self._rebalancer_thread.is_alive():
            self._rebalancer_thread.join(timeout=5)


# Global resource coordinator instance
_resource_coordinator_instance = None
_resource_coordinator_lock = threading.Lock()

def get_resource_coordinator(config: Dict[str, Any] = None) -> ResourceCoordinator:
    """Get the global resource coordinator instance."""
    global _resource_coordinator_instance
    
    if _resource_coordinator_instance is None:
        with _resource_coordinator_lock:
            if _resource_coordinator_instance is None:
                _resource_coordinator_instance = ResourceCoordinator(config)
    
    return _resource_coordinator_instance
