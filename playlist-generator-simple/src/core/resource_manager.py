"""
Resource Manager for Playlist Generator Simple.
Monitors system resources in real-time and provides forced guidance for feature extraction.
"""

import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import time
import threading
import psutil
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta

# Import local modules
from .logging_setup import get_logger, log_function_call, log_universal

logger = get_logger('playlista.resource_manager')

# Constants
DEFAULT_MEMORY_LIMIT_GB = 6.0
DEFAULT_CPU_THRESHOLD_PERCENT = 90
DEFAULT_DISK_THRESHOLD_PERCENT = 85
DEFAULT_MONITORING_INTERVAL_SECONDS = 5
DEFAULT_RESOURCE_HISTORY_SIZE = 1000
DEFAULT_RESOURCE_ALERT_THRESHOLD_PERCENT = 90
DEFAULT_RESOURCE_LOG_LEVEL = 'INFO'

# PLAYLISTA Pattern 4: Duration-Based Analysis Requirements
PLAYLISTA_SHORT_DURATION_MINUTES = 10      # Files under 10min get full analysis
PLAYLISTA_LONG_DURATION_MINUTES = 30       # Files over 30min get content classification
PLAYLISTA_STANDARD_RAM_GB = 1.2             # RAM per worker (consistent for all files)
PLAYLISTA_MAX_CONCURRENT_WORKERS = 4        # Maximum workers for safety


class ResourceManager:
    """
    Monitors system resources in real-time and provides forced guidance for feature extraction.
    
    Handles:
    - Memory monitoring and cleanup
    - CPU usage monitoring
    - Disk space monitoring
    - Automatic resource management
    - Forced feature extraction guidance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the resource manager.
        
        Args:
            config: Configuration dictionary (uses global config if None)
        """
        # Load configuration
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_resource_config()
        
        self.config = config
        
        # Resource thresholds
        self.memory_limit_gb = config.get('MEMORY_LIMIT_GB', DEFAULT_MEMORY_LIMIT_GB)
        self.cpu_threshold_percent = config.get('CPU_THRESHOLD_PERCENT', DEFAULT_CPU_THRESHOLD_PERCENT)
        self.disk_threshold_percent = config.get('DISK_THRESHOLD_PERCENT', DEFAULT_DISK_THRESHOLD_PERCENT)
        self.monitoring_interval = config.get('MONITORING_INTERVAL_SECONDS', DEFAULT_MONITORING_INTERVAL_SECONDS)
        
        # Container/Environment memory limit (overrides host memory detection)
        container_memory_limit_raw = config.get('CONTAINER_MEMORY_LIMIT_GB', None)
        if container_memory_limit_raw == '' or container_memory_limit_raw is None:
            self.container_memory_limit_gb = None  # Empty string means use host detection
        else:
            # Parse memory limit string (e.g., "4GB", "6.5GB", "2MB")
            import re
            match = re.match(r'(\d+(?:\.\d+)?)\s*([GMK])?B?', str(container_memory_limit_raw).upper())
            if match:
                value = float(match.group(1))
                unit = match.group(2) or 'G'
                if unit == 'M':
                    self.container_memory_limit_gb = value / 1024
                elif unit == 'K':
                    self.container_memory_limit_gb = value / (1024 * 1024)
                else:
                    self.container_memory_limit_gb = value
            else:
                # Try to convert to float directly (assume GB)
                try:
                    self.container_memory_limit_gb = float(container_memory_limit_raw)
                except (ValueError, TypeError):
                    log_universal('WARNING', 'Resource', f'Invalid CONTAINER_MEMORY_LIMIT_GB value: {container_memory_limit_raw}, using host memory detection')
                    self.container_memory_limit_gb = None
        
        # Advanced resource settings
        self.resource_history_size = config.get('RESOURCE_HISTORY_SIZE', DEFAULT_RESOURCE_HISTORY_SIZE)
        self.resource_alert_threshold_percent = config.get('RESOURCE_ALERT_THRESHOLD_PERCENT', DEFAULT_RESOURCE_ALERT_THRESHOLD_PERCENT)
        self.resource_auto_cleanup_enabled = config.get('RESOURCE_AUTO_CLEANUP_ENABLED', True)
        self.resource_callback_enabled = config.get('RESOURCE_CALLBACK_ENABLED', True)
        self.resource_performance_monitoring = config.get('RESOURCE_PERFORMANCE_MONITORING', True)
        self.resource_memory_limit_gb = config.get('RESOURCE_MEMORY_LIMIT_GB', DEFAULT_MEMORY_LIMIT_GB)
        self.resource_cpu_limit_percent = config.get('RESOURCE_CPU_LIMIT_PERCENT', DEFAULT_CPU_THRESHOLD_PERCENT)
        self.resource_log_level = config.get('RESOURCE_LOG_LEVEL', DEFAULT_RESOURCE_LOG_LEVEL)
        self.resource_monitoring_enabled = config.get('RESOURCE_MONITORING_ENABLED', True)
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Resource history
        self.resource_history = []
        self.max_history_size = self.resource_history_size
        
        # Callbacks
        self._resource_callbacks = []
        
        # Forced feature extraction state
        self._forced_basic_analysis = False
        self._forced_reason = None
        
        log_universal('INFO', 'Resource', 'Initializing ResourceManager')
        log_universal('INFO', 'Resource', 'ResourceManager initialized successfully')

    @log_function_call
    def start_monitoring(self):
        """Start resource monitoring in a background thread."""
        if not self.resource_monitoring_enabled:
            log_universal('INFO', 'Resource', 'Resource monitoring disabled in configuration')
            return
            
        if self._monitoring:
            log_universal('WARNING', 'Resource', 'Resource monitoring already active')
            return
        
        log_universal('INFO', 'Resource', f"Starting resource monitoring (interval: {self.monitoring_interval}s)")
        
        self._monitoring = True
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        
        log_universal('INFO', 'Resource', f"Resource monitoring started")

    @log_function_call
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self._monitoring:
            log_universal('WARNING', 'Resource', "Resource monitoring not active")
            return
        
        log_universal('INFO', 'Resource', f"Stopping resource monitoring")
        
        self._monitoring = False
        self._stop_monitoring.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
        
        log_universal('INFO', 'Resource', f"Resource monitoring stopped")

    def _monitor_resources(self):
        """Background thread for monitoring system resources."""
        log_universal('DEBUG', 'Resource', "Resource monitoring thread started")
        
        while not self._stop_monitoring.is_set():
            try:
                # Get current resource usage
                resource_data = self._get_current_resources()
                
                # Store in history
                self._add_to_history(resource_data)
                
                # Check for critical conditions and update forced state
                self._check_resource_conditions(resource_data)
                
                # Wait for next monitoring cycle
                self._stop_monitoring.wait(self.monitoring_interval)
                
            except Exception as e:
                log_universal('ERROR', 'Resource', f"Error in resource monitoring: {e}")
                time.sleep(1)  # Brief pause on error
        
        log_universal('DEBUG', 'Resource', "Resource monitoring thread stopped")

    def _get_current_resources(self) -> Dict[str, Any]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary with current resource data
        """
        try:
            # Memory information
            memory = psutil.virtual_memory()
            memory_data = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            }
            
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_data = {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': (disk.used / disk.total) * 100
            }
            
            # Process information
            process = psutil.Process()
            process_data = {
                'memory_rss_gb': process.memory_info().rss / (1024**3),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads()
            }
            
            resource_data = {
                'timestamp': datetime.now(),
                'memory': memory_data,
                'cpu_percent': cpu_percent,
                'disk': disk_data,
                'process': process_data
            }
            
            return resource_data
            
        except Exception as e:
            log_universal('ERROR', 'Resource', f"Error getting resource data: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }

    def _add_to_history(self, resource_data: Dict[str, Any]):
        """Add resource data to history, maintaining size limit."""
        self.resource_history.append(resource_data)
        
        # Trim history if too large
        if len(self.resource_history) > self.max_history_size:
            self.resource_history = self.resource_history[-self.max_history_size:]

    def _check_resource_conditions(self, resource_data: Dict[str, Any]):
        """Check for critical resource conditions and update forced state."""
        if 'error' in resource_data:
            return
        
        # Check memory usage
        memory_used_gb = resource_data['memory']['used_gb']
        memory_percent = resource_data['memory']['percent']
        
        # Check CPU usage
        cpu_percent = resource_data['cpu_percent']
        
        # Check disk usage
        disk_percent = resource_data['disk']['percent']
        
        # Update forced analysis state based on resources
        self._update_forced_analysis_state(memory_used_gb, memory_percent, cpu_percent, disk_percent)
        
        # Handle critical conditions
        if memory_used_gb > self.memory_limit_gb or memory_percent > 90:
            log_universal('WARNING', 'Resource', f"High memory usage: {memory_used_gb:.2f}GB ({memory_percent:.1f}%)")
            self._handle_high_memory()
        
        if cpu_percent > self.cpu_threshold_percent:
            log_universal('WARNING', 'Resource', f"High CPU usage: {cpu_percent:.1f}%")
            self._handle_high_cpu()
        
        if disk_percent > self.disk_threshold_percent:
            log_universal('WARNING', 'Resource', f"High disk usage: {disk_percent:.1f}%")
            self._handle_high_disk()
        
        # Notify callbacks
        self._notify_callbacks(resource_data)

    def _update_forced_analysis_state(self, memory_used_gb: float, memory_percent: float, 
                                    cpu_percent: float, disk_percent: float):
        """
        Update forced analysis state based on current resources.
        
        Args:
            memory_used_gb: Current memory usage in GB
            memory_percent: Current memory usage percentage
            cpu_percent: Current CPU usage percentage
            disk_percent: Current disk usage percentage
        """
        # Determine if we need to force basic analysis
        force_basic = False
        reason = None
        
        # Memory-based forcing
        if memory_percent > 85 or memory_used_gb > self.memory_limit_gb * 0.9:
            force_basic = True
            reason = f"High memory usage: {memory_percent:.1f}% ({memory_used_gb:.2f}GB)"
        
        # CPU-based forcing
        elif cpu_percent > 80:
            force_basic = True
            reason = f"High CPU usage: {cpu_percent:.1f}%"
        
        # Disk-based forcing
        elif disk_percent > 90:
            force_basic = True
            reason = f"High disk usage: {disk_percent:.1f}%"
        
        # Update forced state
        if force_basic != self._forced_basic_analysis:
            self._forced_basic_analysis = force_basic
            self._forced_reason = reason
            
            if force_basic:
                log_universal('WARNING', 'Resource', f"Forcing basic analysis: {reason}")
            else:
                log_universal('INFO', 'Resource', f"Resuming normal analysis (resources recovered)")
        
        log_universal('DEBUG', 'Resource', f"Resource state: Memory {memory_percent:.1f}%, CPU {cpu_percent:.1f}%, Disk {disk_percent:.1f}%")

    def _handle_high_memory(self):
        """Handle high memory usage."""
        log_universal('INFO', 'Resource', f"Initiating memory cleanup")
        
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Log memory after cleanup
            memory = psutil.virtual_memory()
            log_universal('INFO', 'Resource', f"Memory cleanup completed: {memory.used / (1024**3):.2f}GB used")
            
        except Exception as e:
            log_universal('ERROR', 'Resource', f"Error during memory cleanup: {e}")

    def _handle_high_cpu(self):
        """Handle high CPU usage."""
        log_universal('INFO', 'Resource', f"High CPU usage detected - consider throttling analysis")
        
        # Could implement analysis throttling here
        # For now, just log the condition

    def _handle_high_disk(self):
        """Handle high disk usage."""
        log_universal('INFO', 'Resource', f"High disk usage detected - consider cleanup")
        
        try:
            # Clean up old cache entries
            from .database import DatabaseManager, get_db_manager
            db_manager = get_db_manager()
            cleaned_count = db_manager.cleanup_cache()
            log_universal('INFO', 'Resource', f"Cleaned up {cleaned_count} cache entries")
            
        except Exception as e:
            log_universal('ERROR', 'Resource', f"Error during disk cleanup: {e}")

    def _notify_callbacks(self, resource_data: Dict[str, Any]):
        """Notify registered callbacks of resource changes."""
        for callback in self._resource_callbacks:
            try:
                callback(resource_data)
            except Exception as e:
                log_universal('ERROR', 'Resource', f"Error in resource callback: {e}")

    def get_forced_analysis_guidance(self) -> Dict[str, Any]:
        """
        Get forced analysis guidance based on current resources.
        
        Returns:
            Dictionary with forced analysis configuration
        """
        return {
            'force_basic_analysis': self._forced_basic_analysis,
            'reason': self._forced_reason,
            'timestamp': datetime.now().isoformat()
        }

    def should_force_basic_analysis(self) -> bool:
        """
        Check if basic analysis should be forced due to resource constraints.
        
        Returns:
            True if basic analysis should be forced
        """
        return self._forced_basic_analysis

    @log_function_call
    def get_current_resources(self) -> Dict[str, Any]:
        """
        Get current resource usage (synchronous).
        
        Returns:
            Dictionary with current resource data
        """
        return self._get_current_resources()

    @log_function_call
    def get_resource_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """
        Get resource history for the specified time period.
        
        Args:
            minutes: Number of minutes of history to retrieve
            
        Returns:
            List of resource data dictionaries
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        history = [
            data for data in self.resource_history
            if data.get('timestamp', datetime.min) >= cutoff_time
        ]
        
        log_universal('DEBUG', 'Resource', f"Retrieved {len(history)} resource history entries")
        return history

    @log_function_call
    def get_resource_statistics(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Get resource usage statistics for the specified time period.
        
        Args:
            minutes: Number of minutes to analyze
            
        Returns:
            Dictionary with resource statistics
        """
        history = self.get_resource_history(minutes)
        
        if not history:
            return {}
        
        # Calculate statistics
        memory_percents = [data['memory']['percent'] for data in history if 'memory' in data]
        cpu_percents = [data['cpu_percent'] for data in history if 'cpu_percent' in data]
        disk_percents = [data['disk']['percent'] for data in history if 'disk' in data]
        
        stats = {
            'period_minutes': minutes,
            'data_points': len(history),
            'memory': {
                'average_percent': sum(memory_percents) / len(memory_percents) if memory_percents else 0,
                'max_percent': max(memory_percents) if memory_percents else 0,
                'min_percent': min(memory_percents) if memory_percents else 0
            },
            'cpu': {
                'average_percent': sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0,
                'max_percent': max(cpu_percents) if cpu_percents else 0,
                'min_percent': min(cpu_percents) if cpu_percents else 0
            },
            'disk': {
                'average_percent': sum(disk_percents) / len(disk_percents) if disk_percents else 0,
                'max_percent': max(disk_percents) if disk_percents else 0,
                'min_percent': min(disk_percents) if disk_percents else 0
            }
        }
        
        log_universal('INFO', 'Resource', f"Resource statistics generated for {minutes} minutes")
        return stats

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback function to be called when resource data changes.
        
        Args:
            callback: Function to call with resource data
        """
        self._resource_callbacks.append(callback)
        log_universal('DEBUG', 'Resource', f"Registered resource callback")

    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Unregister a callback function.
        
        Args:
            callback: Function to unregister
        """
        if callback in self._resource_callbacks:
            self._resource_callbacks.remove(callback)
            log_universal('DEBUG', 'Resource', f"Unregistered resource callback")

    @log_function_call
    def get_optimal_worker_count(self, max_workers: int = None, memory_limit_str: str = None) -> int:
        """
        Calculate optimal worker count based on available memory - CONSERVATIVE APPROACH.
        
        Args:
            max_workers: Maximum number of workers (uses CPU count if None)
            memory_limit_str: Memory limit string (e.g., "6GB")
            
        Returns:
            Optimal number of workers (conservative approach)
        """
        try:
            # Parse memory limit
            memory_limit_gb = self.memory_limit_gb
            if memory_limit_str:
                # Parse memory limit string (e.g., "6GB", "4.5GB")
                import re
                match = re.match(r'(\d+(?:\.\d+)?)\s*([GMK])?B?', memory_limit_str.upper())
                if match:
                    value = float(match.group(1))
                    unit = match.group(2) or 'G'
                    if unit == 'M':
                        memory_limit_gb = value / 1024
                    elif unit == 'K':
                        memory_limit_gb = value / (1024 * 1024)
                    else:
                        memory_limit_gb = value
            
            # Use container memory limit if specified (overrides host memory detection)
            if self.container_memory_limit_gb is not None:
                memory_limit_gb = self.container_memory_limit_gb
                log_universal('INFO', 'Resource', f"Using container memory limit: {memory_limit_gb:.2f}GB")
            
            # Get current process memory usage (RSS)
            process = psutil.Process()
            current_rss_gb = process.memory_info().rss / (1024**3)
            
            # Get total system memory from psutil
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            system_memory_percent = memory.percent
            
            # Use container memory limit if specified, otherwise use host memory
            if self.container_memory_limit_gb is not None:
                total_memory_gb = self.container_memory_limit_gb
                log_universal('INFO', 'Resource', f"Container mode: Using {total_memory_gb:.2f}GB as total memory (host reports {memory.total / (1024**3):.2f}GB)")
            
            # CONSERVATIVE: Calculate available memory for workers
            # Reserve 2GB for system and other processes (increased from 1.5GB)
            reserved_memory_gb = min(2.0, total_memory_gb * 0.15)  # 15% of total memory or 2GB, whichever is smaller
            available_for_workers_gb = max(1.0, total_memory_gb - reserved_memory_gb - current_rss_gb)  # Minimum 1GB available
            
            # CONSERVATIVE: Estimate memory per worker (increased from 0.8GB)
            memory_per_worker_gb = 1.2  # Increased from 0.8GB to 1.2GB per worker for safety
            
            # Calculate optimal workers based on available memory for workers
            memory_based_workers = max(1, int(available_for_workers_gb / memory_per_worker_gb))
            
            # Get CPU count
            import multiprocessing as mp
            cpu_count = mp.cpu_count()
            
            # CONSERVATIVE: Use maximum of half CPU cores
            cpu_based_workers = max(2, cpu_count // 2)  # Minimum 2, maximum half CPU cores
            
            # Use the minimum of memory-based and CPU-based workers for conservative approach
            optimal_workers = min(memory_based_workers, cpu_based_workers)
            
            # CONSERVATIVE: If system memory usage is >75%, reduce workers (lowered from 85%)
            if system_memory_percent > 75:
                optimal_workers = max(2, optimal_workers // 2)  # Reduce by half, minimum 2
                log_universal('WARNING', 'Resource', f"System memory usage {system_memory_percent:.1f}% > 75%, reducing workers to {optimal_workers}")
            
            # In container mode, also check against container memory limit
            if self.container_memory_limit_gb is not None:
                container_memory_used_gb = current_rss_gb
                container_memory_percent = (container_memory_used_gb / total_memory_gb) * 100
                if container_memory_percent > 75:
                    optimal_workers = max(2, optimal_workers // 2)  # Reduce by half, minimum 2
                    log_universal('WARNING', 'Resource', f"Container memory usage {container_memory_percent:.1f}% > 75%, reducing workers to {optimal_workers}")
            
            # CONSERVATIVE: If available memory is very low, use only 2 workers (increased from 1)
            if available_for_workers_gb < 2.0:  # Increased from 0.5GB to 2GB minimum
                optimal_workers = 2
                log_universal('WARNING', 'Resource', f"Available memory {available_for_workers_gb:.2f}GB < 2GB, using only 2 workers")
            
            # Apply max_workers limit
            if max_workers:
                optimal_workers = min(optimal_workers, max_workers)
            
            # CONSERVATIVE: Ensure minimum of 2 workers
            optimal_workers = max(2, optimal_workers)
            
            log_universal('INFO', 'Resource', f"CONSERVATIVE optimal worker count: {optimal_workers}")
            log_universal('INFO', 'Resource', f"  Total memory: {total_memory_gb:.2f}GB")
            log_universal('INFO', 'Resource', f"  Current RSS: {current_rss_gb:.2f}GB")
            log_universal('INFO', 'Resource', f"  Available for workers: {available_for_workers_gb:.2f}GB")
            log_universal('INFO', 'Resource', f"  CPU count: {cpu_count}")
            log_universal('INFO', 'Resource', f"  Memory-based workers: {memory_based_workers}")
            log_universal('INFO', 'Resource', f"  CPU-based workers (half cores): {cpu_based_workers}")
            log_universal('INFO', 'Resource', f"  Memory per worker: {memory_per_worker_gb:.1f}GB")
            log_universal('INFO', 'Resource', f"  System memory usage: {system_memory_percent:.1f}%")
            
            if self.container_memory_limit_gb is not None:
                container_memory_percent = (current_rss_gb / total_memory_gb) * 100
                log_universal('INFO', 'Resource', f"  Container memory usage: {container_memory_percent:.1f}%")
                log_universal('INFO', 'Resource', f"  Container mode: Using {self.container_memory_limit_gb:.2f}GB limit")
            
            return optimal_workers
            
        except Exception as e:
            log_universal('WARNING', 'Resource', f"Could not determine optimal worker count: {e}")
            # Fallback to safe defaults - CONSERVATIVE
            try:
                import multiprocessing as mp
                cpu_count = mp.cpu_count()
                # Use minimum of 2 workers, maximum of 2 workers for safety (reduced from 4)
                safe_workers = min(2, cpu_count // 2)
                safe_workers = max(2, safe_workers)  # Ensure minimum 2
                
                # In container mode, also consider memory limits
                if self.container_memory_limit_gb is not None:
                    memory_based_workers = max(1, int(self.container_memory_limit_gb / 1.2))  # 1.2GB per worker
                    safe_workers = min(safe_workers, memory_based_workers)
                    log_universal('INFO', 'Resource', f"Container fallback: memory-based workers: {memory_based_workers}")
                
                if max_workers:
                    safe_workers = min(safe_workers, max_workers)
                log_universal('INFO', 'Resource', f"Using conservative fallback worker count: {safe_workers}")
                return safe_workers
            except Exception as fallback_error:
                log_universal('ERROR', 'Resource', f"Fallback worker count also failed: {fallback_error}")
                return 2  # Conservative fallback: 2 workers

    def get_available_memory_gb(self) -> float:
        """Get available memory for analysis in GB."""
        try:
            # Get current process memory usage
            process = psutil.Process()
            current_rss_gb = process.memory_info().rss / (1024**3)
            
            # Get total system memory
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            
            # Use container memory limit if specified
            if self.container_memory_limit_gb is not None:
                total_memory_gb = self.container_memory_limit_gb
            
            # Reserve memory for system and current process
            reserved_memory_gb = min(2.0, total_memory_gb * 0.15)  
            available_for_workers_gb = max(1.0, total_memory_gb - reserved_memory_gb - current_rss_gb)
            
            return available_for_workers_gb
            
        except Exception as e:
            log_universal('WARNING', 'Resource', f'Failed to calculate available memory: {e}')
            return 4.0  # Safe fallback

    def is_memory_critical(self) -> bool:
        """
        Check if memory usage is critical.
        
        Returns:
            True if memory usage is above threshold
        """
        try:
            memory = psutil.virtual_memory()
            return memory.percent > 90 or memory.used / (1024**3) > self.memory_limit_gb
        except Exception as e:
            log_universal('ERROR', 'Resource', f"Error checking memory status: {e}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """Get current resource configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update resource configuration.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config.update(new_config)
            
            # Update thresholds
            self.memory_limit_gb = self.config.get('MEMORY_LIMIT_GB', DEFAULT_MEMORY_LIMIT_GB)
            self.cpu_threshold_percent = self.config.get('CPU_THRESHOLD_PERCENT', DEFAULT_CPU_THRESHOLD_PERCENT)
            self.disk_threshold_percent = self.config.get('DISK_THRESHOLD_PERCENT', DEFAULT_DISK_THRESHOLD_PERCENT)
            self.monitoring_interval = self.config.get('MONITORING_INTERVAL_SECONDS', DEFAULT_MONITORING_INTERVAL_SECONDS)
            
            # Update advanced resource settings
            self.resource_history_size = self.config.get('RESOURCE_HISTORY_SIZE', DEFAULT_RESOURCE_HISTORY_SIZE)
            self.resource_alert_threshold_percent = self.config.get('RESOURCE_ALERT_THRESHOLD_PERCENT', DEFAULT_RESOURCE_ALERT_THRESHOLD_PERCENT)
            self.resource_auto_cleanup_enabled = self.config.get('RESOURCE_AUTO_CLEANUP_ENABLED', True)
            self.resource_callback_enabled = self.config.get('RESOURCE_CALLBACK_ENABLED', True)
            self.resource_performance_monitoring = self.config.get('RESOURCE_PERFORMANCE_MONITORING', True)
            self.resource_memory_limit_gb = self.config.get('RESOURCE_MEMORY_LIMIT_GB', DEFAULT_MEMORY_LIMIT_GB)
            self.resource_cpu_limit_percent = self.config.get('RESOURCE_CPU_LIMIT_PERCENT', DEFAULT_CPU_THRESHOLD_PERCENT)
            
            # Update history size
            self.max_history_size = self.resource_history_size
            
            log_universal('INFO', 'Resource', f"Updated resource configuration: {new_config}")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Resource', f"Error updating resource configuration: {e}")
            return False


    def check_resource_alerts(self) -> Dict[str, Any]:
        """
        Check for resource alerts based on configured thresholds.
        
        Returns:
            Dictionary with alert status for each resource type
        """
        try:
            current_resources = self.get_current_resources()
            
            alerts = {
                'memory_alert': False,
                'cpu_alert': False,
                'disk_alert': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check memory alerts
            memory_percent = current_resources['memory']['percent']
            if memory_percent > self.resource_alert_threshold_percent:
                alerts['memory_alert'] = True
                log_universal('WARNING', 'Resource', f"Memory alert: {memory_percent:.1f}% > {self.resource_alert_threshold_percent}%")
            
            # Check CPU alerts
            cpu_percent = current_resources['cpu_percent']
            if cpu_percent > self.resource_alert_threshold_percent:
                alerts['cpu_alert'] = True
                log_universal('WARNING', 'Resource', f"CPU alert: {cpu_percent:.1f}% > {self.resource_alert_threshold_percent}%")
            
            # Check disk alerts
            disk_percent = current_resources['disk']['percent']
            if disk_percent > self.resource_alert_threshold_percent:
                alerts['disk_alert'] = True
                log_universal('WARNING', 'Resource', f"Disk alert: {disk_percent:.1f}% > {self.resource_alert_threshold_percent}%")
            
            return alerts
            
        except Exception as e:
            log_universal('ERROR', 'Resource', f"Error checking resource alerts: {e}")
            return {'error': str(e)}

    def perform_auto_cleanup(self) -> Dict[str, Any]:
        """
        Perform automatic resource cleanup if enabled.
        
        Returns:
            Dictionary with cleanup results
        """
        if not self.resource_auto_cleanup_enabled:
            return {'enabled': False, 'message': 'Auto cleanup disabled'}
        
        try:
            cleanup_results = {
                'memory_cleanup': False,
                'disk_cleanup': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Memory cleanup
            if self.is_memory_critical():
                self._handle_high_memory()
                log_universal('INFO', 'Resource', "Performed automatic memory cleanup")
            
            # Disk cleanup
            current_resources = self.get_current_resources()
            if current_resources['disk']['percent'] > self.disk_threshold_percent:
                self._handle_high_disk()
                log_universal('INFO', 'Resource', "Performed automatic disk cleanup")
            
            return cleanup_results
            
        except Exception as e:
            log_universal('ERROR', 'Resource', f"Error during auto cleanup: {e}")
            return {'error': str(e)}

    def get_resource_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed resource performance metrics if enabled.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.resource_performance_monitoring:
            return {'enabled': False, 'message': 'Performance monitoring disabled'}
        
        try:
            current_resources = self.get_current_resources()
            history = self.get_resource_history(minutes=60)
            
            # Calculate performance metrics
            memory_trend = self._calculate_trend([h['memory']['percent'] for h in history])
            cpu_trend = self._calculate_trend([h['cpu_percent'] for h in history])
            disk_trend = self._calculate_trend([h['disk']['percent'] for h in history])
            
            metrics = {
                'current': current_resources,
                'trends': {
                    'memory_trend': memory_trend,
                    'cpu_trend': cpu_trend,
                    'disk_trend': disk_trend
                },
                'limits': {
                    'memory_limit_gb': self.resource_memory_limit_gb,
                    'cpu_limit_percent': self.resource_cpu_limit_percent,
                    'disk_threshold_percent': self.disk_threshold_percent
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            log_universal('ERROR', 'Resource', f"Error getting performance metrics: {e}")
            return {'error': str(e)}

    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend from a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Trend description: 'increasing', 'decreasing', 'stable'
        """
        if len(values) < 2:
            return 'stable'
        
        # Calculate simple trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if not first_half or not second_half:
            return 'stable'
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg + 5:  # 5% threshold
            return 'increasing'
        elif second_avg < first_avg - 5:
            return 'decreasing'
        else:
            return 'stable'
    
    # PLAYLISTA Pattern 4: Resource Management Methods
    
    def get_file_analysis_strategy(self, duration_minutes: float) -> Dict[str, Any]:
        """
        Determine analysis strategy based on PLAYLISTA duration thresholds.
        
        Args:
            duration_minutes: File duration in minutes
            
        Returns:
            Dictionary with strategy information
        """
        # All files now use consistent chunk analysis with same memory requirements
        max_workers = min(
            self.calculate_max_threads_for_memory(PLAYLISTA_STANDARD_RAM_GB),
            PLAYLISTA_MAX_CONCURRENT_WORKERS
        )
        
        if duration_minutes < PLAYLISTA_SHORT_DURATION_MINUTES:
            # < 10 minutes: Full file analysis
            strategy = {
                'duration_minutes': duration_minutes,
                'analysis_type': 'full_file',
                'max_threads': max_workers,
                'required_ram_gb': PLAYLISTA_STANDARD_RAM_GB,
                'enforce_sequential': False,
                'chunks': 1,
                'chunk_duration_seconds': min(duration_minutes * 60, 30),
                'reasoning': f'Short file (<{PLAYLISTA_SHORT_DURATION_MINUTES}min) gets full analysis'
            }
        elif duration_minutes < PLAYLISTA_LONG_DURATION_MINUTES:
            # 10-30 minutes: Multi-chunk analysis
            strategy = {
                'duration_minutes': duration_minutes,
                'analysis_type': 'multi_chunk',
                'max_threads': max_workers,
                'required_ram_gb': PLAYLISTA_STANDARD_RAM_GB,
                'enforce_sequential': False,
                'chunks': 3,
                'chunk_duration_seconds': 30,
                'reasoning': f'Medium file ({PLAYLISTA_SHORT_DURATION_MINUTES}-{PLAYLISTA_LONG_DURATION_MINUTES}min) gets 3-chunk analysis'
            }
        else:
            # > 30 minutes: Long content with classification
            strategy = {
                'duration_minutes': duration_minutes,
                'analysis_type': 'long_content',
                'max_threads': max_workers,
                'required_ram_gb': PLAYLISTA_STANDARD_RAM_GB,
                'enforce_sequential': False,
                'chunks': 3,
                'chunk_duration_seconds': 30,
                'content_classification': True,
                'reasoning': f'Long file (>{PLAYLISTA_LONG_DURATION_MINUTES}min) gets classification + 3-chunk analysis'
            }
        
        log_universal('DEBUG', 'ResourceManager', 
                     f'File {duration_minutes:.1f}min strategy: {strategy["analysis_type"]}, '
                     f'{strategy["max_threads"]} threads, {strategy["chunks"]} chunks')
        
        return strategy
    
    def calculate_max_threads_for_memory(self, ram_per_thread_gb: float) -> int:
        """
        Calculate maximum threads based on available memory and PLAYLISTA requirements.
        
        Args:
            ram_per_thread_gb: Required RAM per thread in GB
            
        Returns:
            Maximum number of threads that can be safely spawned
        """
        try:
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            
            # Reserve 25% of available memory for system operations
            usable_gb = available_gb * 0.75
            
            # Calculate max threads based on memory requirement
            max_threads = int(usable_gb / ram_per_thread_gb)
            
            # Apply practical limits
            max_threads = max(1, min(max_threads, psutil.cpu_count()))
            
            log_universal('DEBUG', 'ResourceManager',
                         f'Memory calculation: {available_gb:.1f}GB available, '
                         f'{usable_gb:.1f}GB usable, {ram_per_thread_gb}GB per thread = {max_threads} max threads')
            
            return max_threads
            
        except Exception as e:
            log_universal('WARNING', 'ResourceManager', f'Memory calculation failed: {e}')
            return 1  # Safe fallback
    
    def get_batch_processing_plan(self, file_sizes_mb: List[float]) -> Dict[str, Any]:
        """
        Create an optimal processing plan for a batch of files based on PLAYLISTA rules.
        
        Args:
            file_sizes_mb: List of file sizes in MB
            
        Returns:
            Processing plan with categorized files and resource allocation
        """
        plan = {
            'sequential_only_files': [],
            'parallel_large_files': [],
            'parallel_small_files': [],
            'total_files': len(file_sizes_mb),
            'estimated_ram_usage_gb': 0,
            'recommended_processing_order': []
        }
        
        # Categorize files by size
        for i, size_mb in enumerate(file_sizes_mb):
            strategy = self.get_file_analysis_strategy(size_mb)
            
            file_info = {
                'index': i,
                'size_mb': size_mb,
                'strategy': strategy
            }
            
            if strategy['analysis_type'] == 'sequential_only':
                plan['sequential_only_files'].append(file_info)
            elif strategy['analysis_type'] == 'parallel_large':
                plan['parallel_large_files'].append(file_info)
            else:
                plan['parallel_small_files'].append(file_info)
        
        # Calculate total estimated RAM usage (now consistent for all files)
        max_workers = min(
            self.calculate_max_threads_for_memory(PLAYLISTA_STANDARD_RAM_GB),
            PLAYLISTA_MAX_CONCURRENT_WORKERS
        )
        plan['estimated_ram_usage_gb'] = max_workers * PLAYLISTA_STANDARD_RAM_GB
        
        # Recommend processing order: sequential first (highest memory), then large, then small
        plan['recommended_processing_order'] = (
            plan['sequential_only_files'] +
            plan['parallel_large_files'] + 
            plan['parallel_small_files']
        )
        
        log_universal('INFO', 'ResourceManager',
                     f'Batch plan: {len(plan["sequential_only_files"])} sequential, '
                     f'{len(plan["parallel_large_files"])} large parallel, '
                     f'{len(plan["parallel_small_files"])} small parallel, '
                     f'estimated {plan["estimated_ram_usage_gb"]:.1f}GB RAM')
        
        return plan


# Global resource manager instance - created lazily to avoid circular imports
_resource_manager_instance = None

def get_resource_manager() -> 'ResourceManager':
    """Get the global resource manager instance, creating it if necessary."""
    global _resource_manager_instance
    if _resource_manager_instance is None:
        _resource_manager_instance = ResourceManager()
    return _resource_manager_instance 