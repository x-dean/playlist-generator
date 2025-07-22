import psutil
import gc
import logging
import json
import os
from functools import wraps
from time import time

logger = logging.getLogger(__name__)

# Default thresholds
MEMORY_THRESHOLD = 0.85  # 85% of system memory
CPU_THRESHOLD = 0.90    # 90% of CPU usage
MEMORY_MIN_FREE = 500 * 1024 * 1024  # 500MB minimum free memory

class SystemMonitor:
    def __init__(self, memory_threshold=MEMORY_THRESHOLD, cpu_threshold=CPU_THRESHOLD):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.process = psutil.Process()
        self.start_time = time()
        self.metrics = {
            'peak_memory': 0,
            'peak_cpu': 0,
            'gc_collections': 0,
            'memory_warnings': 0
        }

    def check_memory(self):
        """Check memory usage and trigger GC if needed"""
        try:
            memory = psutil.virtual_memory()
            process_memory = self.process.memory_info()
            
            # Update peak memory
            self.metrics['peak_memory'] = max(self.metrics['peak_memory'], process_memory.rss)
            
            if memory.percent > (self.memory_threshold * 100):
                logger.warning(f"High memory usage: {memory.percent}%")
                self.metrics['memory_warnings'] += 1
                self._cleanup_memory()
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking memory: {str(e)}")
            return False

    def check_cpu(self):
        """Monitor CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['peak_cpu'] = max(self.metrics['peak_cpu'], cpu_percent)
            
            if cpu_percent > (self.cpu_threshold * 100):
                logger.warning(f"High CPU usage: {cpu_percent}%")
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking CPU: {str(e)}")
            return False

    def _cleanup_memory(self):
        """Perform memory cleanup"""
        before = self.process.memory_info().rss
        gc.collect()
        self.metrics['gc_collections'] += 1
        after = self.process.memory_info().rss
        freed = (before - after) / 1024 / 1024
        logger.info(f"Memory cleanup freed {freed:.2f}MB")

    def get_metrics(self):
        """Get current system metrics"""
        try:
            return {
                'memory_used_mb': self.process.memory_info().rss / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'runtime_seconds': time() - self.start_time,
                **self.metrics
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {}

    def save_metrics(self, filepath='system_metrics.json'):
        """Save metrics to file"""
        try:
            metrics = self.get_metrics()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved system metrics to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

def monitor_performance(func):
    """Decorator to monitor performance of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = SystemMonitor()
        start_time = time()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time() - start_time
            metrics = monitor.get_metrics()
            logger.info(
                f"Performance metrics for {func.__name__}:\n"
                f"  Runtime: {elapsed:.2f}s\n"
                f"  Peak Memory: {metrics['peak_memory']/1024/1024:.2f}MB\n"
                f"  Peak CPU: {metrics['peak_cpu']:.1f}%\n"
                f"  GC Collections: {metrics['gc_collections']}"
            )
            monitor.save_metrics(f"{func.__name__}_metrics.json")
    
    return wrapper 