"""
Monitoring and metrics implementation for Playlist Generator.
Provides Prometheus metrics and performance monitoring.
"""

import time
import psutil
from typing import Dict, Any, Optional
from functools import wraps
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    """Metrics collector for application monitoring."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Counters
        self.track_analysis_total = Counter(
            'playlista_track_analysis_total',
            'Total number of track analyses',
            ['status', 'format'],
            registry=self.registry
        )
        
        self.repository_operations_total = Counter(
            'playlista_repository_operations_total',
            'Total number of repository operations',
            ['operation', 'entity_type', 'status'],
            registry=self.registry
        )
        
        self.use_case_executions_total = Counter(
            'playlista_use_case_executions_total',
            'Total number of use case executions',
            ['use_case', 'status'],
            registry=self.registry
        )
        
        # Histograms
        self.analysis_duration = Histogram(
            'playlista_analysis_duration_seconds',
            'Time spent on track analysis',
            ['format'],
            registry=self.registry
        )
        
        self.repository_operation_duration = Histogram(
            'playlista_repository_operation_duration_seconds',
            'Time spent on repository operations',
            ['operation', 'entity_type'],
            registry=self.registry
        )
        
        self.use_case_duration = Histogram(
            'playlista_use_case_duration_seconds',
            'Time spent on use case execution',
            ['use_case'],
            registry=self.registry
        )
        
        # Gauges
        self.tracks_total = Gauge(
            'playlista_tracks_total',
            'Total number of tracks in database',
            registry=self.registry
        )
        
        self.playlists_total = Gauge(
            'playlista_playlists_total',
            'Total number of playlists in database',
            registry=self.registry
        )
        
        self.analysis_results_total = Gauge(
            'playlista_analysis_results_total',
            'Total number of analysis results in database',
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'playlista_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'playlista_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Summaries
        self.analysis_confidence = Summary(
            'playlista_analysis_confidence',
            'Analysis confidence scores',
            ['format'],
            registry=self.registry
        )
    
    def record_track_analysis(self, status: str, format: str, duration: float, confidence: float = None):
        """Record track analysis metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.track_analysis_total.labels(status=status, format=format).inc()
        self.analysis_duration.labels(format=format).observe(duration)
        
        if confidence is not None:
            self.analysis_confidence.labels(format=format).observe(confidence)
    
    def record_repository_operation(self, operation: str, entity_type: str, status: str, duration: float):
        """Record repository operation metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.repository_operations_total.labels(
            operation=operation,
            entity_type=entity_type,
            status=status
        ).inc()
        
        self.repository_operation_duration.labels(
            operation=operation,
            entity_type=entity_type
        ).observe(duration)
    
    def record_use_case_execution(self, use_case: str, status: str, duration: float):
        """Record use case execution metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.use_case_executions_total.labels(
            use_case=use_case,
            status=status
        ).inc()
        
        self.use_case_duration.labels(use_case=use_case).observe(duration)
    
    def update_database_metrics(self, tracks_count: int, playlists_count: int, analysis_count: int):
        """Update database metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.tracks_total.set(tracks_count)
        self.playlists_total.set(playlists_count)
        self.analysis_results_total.set(analysis_count)
    
    def update_system_metrics(self):
        """Update system metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics as string."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"
        
        if self.registry:
            return generate_latest(self.registry)
        return generate_latest()


class PerformanceMonitor:
    """Performance monitoring with timing and resource tracking."""
    
    def __init__(self, metrics_collector: MetricsCollector = None):
        self.metrics = metrics_collector or MetricsCollector()
        self._timings: Dict[str, float] = {}
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._timings[operation_name] = duration
    
    def time_function(self, operation_name: str = None):
        """Decorator for timing functions."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                with self.time_operation(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_timing(self, operation_name: str) -> Optional[float]:
        """Get timing for an operation."""
        return self._timings.get(operation_name)
    
    def get_all_timings(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return self._timings.copy()
    
    def clear_timings(self):
        """Clear all recorded timings."""
        self._timings.clear()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "cpu": {
                "percent": cpu_percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free
            }
        }


# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_performance_monitor: Optional[PerformanceMonitor] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def monitor_operation(operation_name: str = None):
    """Decorator for monitoring operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()
            metrics = get_metrics_collector()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics
                metrics.record_use_case_execution(name, "success", duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_use_case_execution(name, "error", duration)
                raise
        return wrapper
    return decorator 