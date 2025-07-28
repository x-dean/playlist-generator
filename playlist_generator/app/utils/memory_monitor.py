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
        """Get current memory information (prefer container limits over host memory)."""
        try:
            # First try to get container memory info
            container_info = get_container_memory_info()
            
            if container_info and container_info['is_limited']:
                # Use container memory limits
                logger.debug(f"Using container memory limits: {container_info['usage_gb']:.1f}GB/{container_info['limit_gb']:.1f}GB")
                return {
                    'total_gb': container_info['limit_gb'],
                    'available_gb': container_info['available_gb'],
                    'used_gb': container_info['usage_gb'],
                    'percent': container_info['usage_percent'],
                    'is_container': True
                }
            else:
                # Fall back to host memory if no container limits
                logger.debug("No container memory limits found, using host memory")
                memory = psutil.virtual_memory()
                host_memory = {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent,
                    'is_container': False
                }
                logger.debug(f"Host memory info: {host_memory}")
                return host_memory
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent': 0, 'is_container': False}
    
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
        
        # Also check RSS limits
        rss_over_limit, rss_msg = check_total_python_rss_limit()
        if rss_over_limit:
            logger.warning(f"RSS limit exceeded: {rss_msg}")
            return True
            
        return status['is_high']
    
    def get_optimal_worker_count(self, max_workers: int, min_workers: int = 1, memory_limit_str: str = None) -> int:
        """Calculate optimal worker count based on available memory."""
        try:
            # Use RSS-based calculation for better container memory management
            total_rss_gb = get_total_python_rss_gb()
            current = self._get_memory_info()
            
            logger.info(f"🔄 [MemoryMonitor] get_optimal_worker_count called: max_workers={max_workers}, available_gb={current['available_gb']:.2f}, current_rss_gb={total_rss_gb:.2f}")
            
            # Use CLI memory limit if provided, otherwise default to 2GB
            if memory_limit_str:
                memory_per_worker_gb = parse_memory_limit(memory_limit_str)
                if memory_per_worker_gb is None:
                    memory_per_worker_gb = 2.0
                    logger.warning(f"Invalid memory limit '{memory_limit_str}', using default 2GB")
            else:
                memory_per_worker_gb = 2.0
            
            # Calculate available memory considering current RSS usage
            # Reserve some memory for the system and other processes
            reserved_memory_gb = 1.0  # Reserve 1GB for system
            available_for_workers = current['available_gb'] - total_rss_gb - reserved_memory_gb
            
            # Calculate how many workers we can support
            workers_by_memory = int(available_for_workers / memory_per_worker_gb)
            
            # Use the minimum of CPU-based and memory-based limits
            optimal_workers = min(max_workers, workers_by_memory)
            
            # Ensure we don't go below minimum
            optimal_workers = max(min_workers, optimal_workers)
            
            memory_source = "container" if current.get('is_container', False) else "host"
            logger.info(f"RSS-aware worker calculation ({memory_source}):")
            logger.info(f"  Total available: {current['available_gb']:.1f}GB")
            logger.info(f"  Current RSS: {total_rss_gb:.1f}GB")
            logger.info(f"  Reserved: {reserved_memory_gb:.1f}GB")
            logger.info(f"  Available for workers: {available_for_workers:.1f}GB")
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
        # Try multiple cgroup paths for different container runtimes
        cgroup_paths = [
            '/sys/fs/cgroup/memory/memory.usage_in_bytes',
            '/sys/fs/cgroup/memory.current',
            '/sys/fs/cgroup/memory.max',
            '/sys/fs/cgroup/memory.usage_in_bytes',
            '/sys/fs/cgroup/memory.limit_in_bytes'
        ]
        
        usage_bytes = None
        limit_bytes = None
        
        # Try to find the correct cgroup path
        for path in cgroup_paths:
            try:
                if 'usage' in path or 'current' in path:
                    with open(path, 'r') as f:
                        usage_bytes = int(f.read().strip())
                elif 'limit' in path or 'max' in path:
                    with open(path, 'r') as f:
                        limit_bytes = int(f.read().strip())
            except (FileNotFoundError, ValueError):
                continue
        
        # If we found usage but no limit, try the standard paths
        if usage_bytes is not None and limit_bytes is None:
            try:
                with open('/sys/fs/cgroup/memory.limit_in_bytes', 'r') as f:
                    limit_bytes = int(f.read().strip())
            except (FileNotFoundError, ValueError):
                pass
        
        # Check if we found both usage and limit
        if usage_bytes is not None and limit_bytes is not None:
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
        
        # Try to get memory limit from environment variables (Docker/LXC)
        try:
            import os
            memory_limit_str = os.getenv('MEMORY_LIMIT') or os.getenv('CONTAINER_MEMORY_LIMIT')
            if memory_limit_str:
                # Parse memory limit (e.g., "8G", "8192M")
                import re
                match = re.match(r'(\d+)([GMK])?', memory_limit_str.upper())
                if match:
                    value = int(match.group(1))
                    unit = match.group(2) or 'B'
                    
                    if unit == 'G':
                        limit_gb = value
                    elif unit == 'M':
                        limit_gb = value / 1024
                    elif unit == 'K':
                        limit_gb = value / (1024 * 1024)
                    else:
                        limit_gb = value / (1024**3)
                    
                    # Get current usage from psutil
                    import psutil
                    usage_gb = psutil.virtual_memory().used / (1024**3)
                    usage_percent = (usage_gb / limit_gb) * 100
                    
                    return {
                        'usage_bytes': int(usage_gb * (1024**3)),
                        'limit_bytes': int(limit_gb * (1024**3)),
                        'usage_gb': usage_gb,
                        'limit_gb': limit_gb,
                        'usage_percent': usage_percent,
                        'available_gb': limit_gb - usage_gb,
                        'is_limited': True
                    }
        except Exception as e:
            logger.debug(f"Could not get memory limit from environment: {e}")
        
        # No container limits found, fall back to system memory
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

def should_pause_between_batches() -> tuple[bool, str]:
    """Check if system should pause between batches to prevent halting."""
    try:
        # Get current memory status
        memory_monitor = MemoryMonitor()
        memory_status = memory_monitor.check_memory_usage()
        
        # Get RSS information
        rss_over_limit, rss_msg = check_total_python_rss_limit()
        
        # Check various conditions that might cause halting
        should_pause = False
        reason = ""
        
        if memory_status['is_critical']:
            should_pause = True
            reason = f"Memory critical ({memory_status['current']['percent']:.1f}%)"
        elif memory_status['is_high']:
            should_pause = True
            reason = f"Memory high ({memory_status['current']['percent']:.1f}%)"
        elif rss_over_limit:
            should_pause = True
            reason = f"RSS limit exceeded: {rss_msg}"
        
        # Check if memory has increased significantly since start
        memory_increase_gb = memory_status['increase_gb']
        if memory_increase_gb > 2.0:  # More than 2GB increase
            should_pause = True
            reason = f"Memory increased by {memory_increase_gb:.1f}GB since start"
        
        return should_pause, reason
        
    except Exception as e:
        logger.warning(f"Could not check pause conditions: {e}")
        return False, "Could not check memory status"

def get_pause_duration_seconds() -> int:
    """Get recommended pause duration between batches based on memory pressure."""
    try:
        memory_monitor = MemoryMonitor()
        memory_status = memory_monitor.check_memory_usage()
        
        # Base pause duration
        base_pause = 5
        
        # Increase pause time based on memory pressure
        if memory_status['is_critical']:
            return 15  # 15 seconds for critical memory
        elif memory_status['is_high']:
            return 10  # 10 seconds for high memory
        elif memory_status['current']['percent'] > 70:
            return 8   # 8 seconds for moderate memory pressure
        else:
            return base_pause  # 5 seconds for normal conditions
            
    except Exception as e:
        logger.debug(f"Could not calculate pause duration: {e}")
        return 5  # Default 5 seconds 