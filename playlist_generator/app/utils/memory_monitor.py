"""
Memory monitoring utilities for playlist generator.
"""
import psutil
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor and control memory usage during processing."""
    
    def __init__(self, memory_limit_gb: Optional[float] = None):
        self.memory_limit_gb = memory_limit_gb
        self.initial_memory = self._get_memory_info()
        
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent': 0}
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and return status."""
        current = self._get_memory_info()
        
        # Calculate memory increase since initialization
        memory_increase_gb = current['used_gb'] - self.initial_memory['used_gb']
        
        status = {
            'current': current,
            'initial': self.initial_memory,
            'increase_gb': memory_increase_gb,
            'is_high': current['percent'] > 85,
            'is_critical': current['percent'] > 95
        }
        
        return status
    
    def log_memory_status(self, context: str = ""):
        """Log current memory status."""
        status = self.check_memory_usage()
        current = status['current']
        
        logger.info(f"Memory status {context}:")
        logger.info(f"  Used: {current['used_gb']:.1f}GB ({current['percent']:.1f}%)")
        logger.info(f"  Available: {current['available_gb']:.1f}GB")
        logger.info(f"  Increase since start: {status['increase_gb']:.1f}GB")
        
        if status['is_critical']:
            logger.warning("CRITICAL: Memory usage is very high (>95%)")
        elif status['is_high']:
            logger.warning("WARNING: Memory usage is high (>85%)")
    
    def should_reduce_workers(self) -> bool:
        """Determine if worker count should be reduced based on memory usage."""
        status = self.check_memory_usage()
        return status['is_high']
    
    def get_optimal_worker_count(self, max_workers: int, min_workers: int = 1) -> int:
        """Calculate optimal worker count based on available memory."""
        try:
            current = self._get_memory_info()
            
            # Conservative estimate: 2GB per worker
            memory_per_worker_gb = 2.0
            
            # Calculate how many workers we can support
            workers_by_memory = int(current['available_gb'] / memory_per_worker_gb)
            
            # Use the minimum of CPU-based and memory-based limits
            optimal_workers = min(max_workers, workers_by_memory)
            
            # Ensure we don't go below minimum
            optimal_workers = max(min_workers, optimal_workers)
            
            logger.info(f"Memory-aware worker calculation:")
            logger.info(f"  Available memory: {current['available_gb']:.1f}GB")
            logger.info(f"  Memory per worker: {memory_per_worker_gb}GB")
            logger.info(f"  Workers by memory: {workers_by_memory}")
            logger.info(f"  Max workers: {max_workers}")
            logger.info(f"  Optimal workers: {optimal_workers}")
            
            return optimal_workers
            
        except Exception as e:
            logger.warning(f"Could not calculate memory-aware worker count: {e}")
            return max(min_workers, max_workers // 2)  # Conservative fallback


def parse_memory_limit(memory_str: str) -> Optional[float]:
    """Parse memory limit string (e.g., '2GB', '512MB') into GB."""
    if not memory_str:
        return None
        
    memory_str = memory_str.upper().strip()
    
    try:
        if memory_str.endswith('GB'):
            return float(memory_str[:-2])
        elif memory_str.endswith('MB'):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith('KB'):
            return float(memory_str[:-2]) / (1024 * 1024)
        else:
            # Assume GB if no unit specified
            return float(memory_str)
    except ValueError:
        logger.warning(f"Invalid memory limit format: {memory_str}")
        return None


def get_memory_aware_batch_size(worker_count: int, available_memory_gb: float) -> int:
    """Calculate optimal batch size based on available memory."""
    # Conservative estimate: each batch item needs ~1GB
    memory_per_item_gb = 1.0
    
    # Calculate how many items we can process in a batch
    items_by_memory = int(available_memory_gb / memory_per_item_gb)
    
    # Use the minimum of worker count and memory-based limit
    optimal_batch_size = min(worker_count, items_by_memory)
    
    # Ensure at least 1 item per batch
    optimal_batch_size = max(1, optimal_batch_size)
    
    logger.info(f"Memory-aware batch size calculation:")
    logger.info(f"  Available memory: {available_memory_gb:.1f}GB")
    logger.info(f"  Memory per item: {memory_per_item_gb}GB")
    logger.info(f"  Items by memory: {items_by_memory}")
    logger.info(f"  Worker count: {worker_count}")
    logger.info(f"  Optimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size 