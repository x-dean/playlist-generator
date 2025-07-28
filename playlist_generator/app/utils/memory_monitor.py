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


def get_container_memory_info():
    """Get container memory usage and limits."""
    try:
        # Try to get container memory from cgroup
        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
            usage_bytes = int(f.read().strip())
        
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            limit_bytes = int(f.read().strip())
        
        # Check if limit is set (not unlimited)
        if limit_bytes != 9223372036854771712:  # Not unlimited
            usage_gb = usage_bytes / (1024**3)
            limit_gb = limit_bytes / (1024**3)
            usage_percent = (usage_bytes / limit_bytes) * 100
            
            return {
                'usage_bytes': usage_bytes,
                'limit_bytes': limit_bytes,
                'usage_gb': usage_gb,
                'limit_gb': limit_gb,
                'usage_percent': usage_percent,
                'available_gb': limit_gb - usage_gb,
                'is_limited': True
            }
        else:
            # No limit set, fall back to system memory
            return None
    except Exception as e:
        logger.debug(f"Could not get container memory info: {e}")
        return None

def check_memory_against_limit(user_limit_gb: float = None, user_limit_percent: float = None):
    """Check if memory usage exceeds user-defined limits."""
    try:
        # First try container memory
        container_info = get_container_memory_info()
        
        if container_info and container_info['is_limited']:
            # Use container memory limits
            usage_gb = container_info['usage_gb']
            limit_gb = container_info['limit_gb']
            usage_percent = container_info['usage_percent']
            
            # Check against user limits
            if user_limit_gb and usage_gb > user_limit_gb:
                return True, f"Container memory usage ({usage_gb:.1f}GB) exceeds user limit ({user_limit_gb:.1f}GB)"
            
            if user_limit_percent and usage_percent > user_limit_percent:
                return True, f"Container memory usage ({usage_percent:.1f}%) exceeds user limit ({user_limit_percent:.1f}%)"
            
            return False, f"Container memory: {usage_gb:.1f}GB/{limit_gb:.1f}GB ({usage_percent:.1f}%)"
        
        else:
            # Fall back to system memory
            memory = psutil.virtual_memory()
            usage_gb = memory.used / (1024**3)
            total_gb = memory.total / (1024**3)
            usage_percent = memory.percent
            
            # Check against user limits
            if user_limit_gb and usage_gb > user_limit_gb:
                return True, f"System memory usage ({usage_gb:.1f}GB) exceeds user limit ({user_limit_gb:.1f}GB)"
            
            if user_limit_percent and usage_percent > user_limit_percent:
                return True, f"System memory usage ({usage_percent:.1f}%) exceeds user limit ({user_limit_percent:.1f}%)"
            
            return False, f"System memory: {usage_gb:.1f}GB/{total_gb:.1f}GB ({usage_percent:.1f}%)"
    
    except Exception as e:
        logger.error(f"Error checking memory against limits: {e}")
        return False, "Could not check memory limits"

def get_detailed_memory_info():
    """Get detailed memory information for current process and container."""
    try:
        import psutil
        import os
        
        # Get current process info
        current_process = psutil.Process(os.getpid())
        process_memory = current_process.memory_info()
        
        # Get container memory (primary)
        container_info = get_container_memory_info()
        
        # Get system memory (fallback)
        system_memory = psutil.virtual_memory()
        
        return {
            'process': {
                'rss_mb': process_memory.rss / (1024**2),
                'vms_mb': process_memory.vms / (1024**2),
                'percent': current_process.memory_percent(),
                'pid': os.getpid()
            },
            'container': container_info,
            'system': {
                'total_gb': system_memory.total / (1024**3),
                'available_gb': system_memory.available / (1024**3),
                'used_gb': system_memory.used / (1024**3),
                'percent': system_memory.percent
            }
        }
    except Exception as e:
        logger.error(f"Error getting detailed memory info: {e}")
        return None

def log_detailed_memory_info(context: str = ""):
    """Log detailed memory information."""
    info = get_detailed_memory_info()
    if not info:
        return
    
    logger.info(f"=== Detailed Memory Info {context} ===")
    logger.info(f"Process Memory:")
    logger.info(f"  RSS: {info['process']['rss_mb']:.1f}MB")
    logger.info(f"  VMS: {info['process']['vms_mb']:.1f}MB")
    logger.info(f"  Percent: {info['process']['percent']:.1f}%")
    logger.info(f"  PID: {info['process']['pid']}")
    
    if info['container'] and info['container']['is_limited']:
        logger.info(f"Container Memory:")
        logger.info(f"  Used: {info['container']['usage_gb']:.1f}GB")
        logger.info(f"  Limit: {info['container']['limit_gb']:.1f}GB")
        logger.info(f"  Percent: {info['container']['usage_percent']:.1f}%")
        logger.info(f"  Available: {info['container']['available_gb']:.1f}GB")
    else:
        logger.info(f"System Memory (no container limits):")
        logger.info(f"  Total: {info['system']['total_gb']:.1f}GB")
        logger.info(f"  Used: {info['system']['used_gb']:.1f}GB ({info['system']['percent']:.1f}%)")
        logger.info(f"  Available: {info['system']['available_gb']:.1f}GB")
    
    logger.info("=" * 40) 

def get_total_python_rss_gb():
    """Sum the RSS of all Python processes and return total in GB."""
    import psutil
    total_rss = 0
    for proc in psutil.process_iter(['name', 'memory_info']):
        try:
            if 'python' in proc.info['name']:
                total_rss += proc.info['memory_info'].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total_rss / (1024 ** 3)

def check_total_python_rss_limit(rss_limit_gb=6.0):
    """Check if total RSS of all Python processes exceeds the given limit (in GB)."""
    total_rss_gb = get_total_python_rss_gb()
    if total_rss_gb > rss_limit_gb:
        return True, f"Total Python RSS {total_rss_gb:.2f}GB exceeds limit {rss_limit_gb}GB"
    return False, f"Total Python RSS {total_rss_gb:.2f}GB within limit {rss_limit_gb}GB" 