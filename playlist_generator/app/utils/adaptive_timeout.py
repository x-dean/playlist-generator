import time
import logging
import psutil
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TimeoutStrategy(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    ADAPTIVE = "adaptive"

@dataclass
class TimeoutConfig:
    """Configuration for timeout calculations."""
    base_timeout: float = 60.0
    max_timeout: float = 600.0
    min_timeout: float = 10.0
    size_multiplier: float = 0.1  # seconds per MB
    load_multiplier: float = 1.5   # multiplier for high system load
    memory_threshold: float = 0.8  # 80% memory usage threshold

class AdaptiveTimeoutManager:
    """
    Dynamically adjusts timeouts based on file characteristics and system performance.
    
    Features:
    - File size-based timeout adjustment
    - System load consideration
    - Memory usage monitoring
    - Historical performance tracking
    - Adaptive timeout strategies
    """
    
    def __init__(self, config: Optional[TimeoutConfig] = None):
        self.config = config or TimeoutConfig()
        self.performance_history = []
        self.max_history_size = 1000
        
        # Feature-specific base timeouts (in seconds)
        self.feature_timeouts = {
            'rhythm': 120.0,
            'spectral': 90.0,
            'loudness': 30.0,
            'danceability': 60.0,
            'key': 45.0,
            'onset_rate': 30.0,
            'zcr': 20.0,
            'mfcc': 180.0,
            'chroma': 120.0,
            'spectral_contrast': 90.0,
            'spectral_flatness': 60.0,
            'spectral_rolloff': 75.0,
            'musicnn': 300.0,
            'metadata_enrichment': 30.0
        }
        
        logger.debug(f"AdaptiveTimeoutManager initialized with config: {self.config}")
    
    def calculate_timeout(self, 
                         file_size_mb: float, 
                         feature_type: str = None,
                         strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE) -> float:
        """
        Calculate optimal timeout based on file characteristics and system state.
        
        Args:
            file_size_mb: File size in megabytes
            feature_type: Type of feature being extracted
            strategy: Timeout calculation strategy
            
        Returns:
            Calculated timeout in seconds
        """
        # Get base timeout for feature type
        base_timeout = self.feature_timeouts.get(feature_type, self.config.base_timeout)
        
        if strategy == TimeoutStrategy.LINEAR:
            timeout = self._linear_timeout(base_timeout, file_size_mb)
        elif strategy == TimeoutStrategy.EXPONENTIAL:
            timeout = self._exponential_timeout(base_timeout, file_size_mb)
        else:  # ADAPTIVE
            timeout = self._adaptive_timeout(base_timeout, file_size_mb, feature_type)
        
        # Apply system load adjustment
        timeout = self._apply_system_load_adjustment(timeout)
        
        # Ensure timeout is within bounds
        timeout = max(self.config.min_timeout, min(timeout, self.config.max_timeout))
        
        logger.debug(f"Calculated timeout: {timeout:.1f}s for {feature_type} ({file_size_mb:.1f}MB)")
        return timeout
    
    def _linear_timeout(self, base_timeout: float, file_size_mb: float) -> float:
        """Linear timeout calculation based on file size."""
        return base_timeout + (file_size_mb * self.config.size_multiplier)
    
    def _exponential_timeout(self, base_timeout: float, file_size_mb: float) -> float:
        """Exponential timeout calculation for very large files."""
        if file_size_mb < 50:
            return base_timeout + (file_size_mb * self.config.size_multiplier)
        else:
            # Exponential growth for large files
            return base_timeout * (1.1 ** (file_size_mb / 50))
    
    def _adaptive_timeout(self, base_timeout: float, file_size_mb: float, feature_type: str) -> float:
        """Adaptive timeout based on historical performance and current conditions."""
        # Start with linear calculation
        timeout = self._linear_timeout(base_timeout, file_size_mb)
        
        # Adjust based on historical performance for this feature type
        historical_adjustment = self._get_historical_adjustment(feature_type, file_size_mb)
        timeout *= historical_adjustment
        
        # Adjust based on current system conditions
        system_adjustment = self._get_system_adjustment()
        timeout *= system_adjustment
        
        return timeout
    
    def _get_historical_adjustment(self, feature_type: str, file_size_mb: float) -> float:
        """Get timeout adjustment based on historical performance."""
        if not self.performance_history:
            return 1.0
        
        # Filter recent history for this feature type and similar file sizes
        recent_history = [
            entry for entry in self.performance_history[-100:]
            if entry.get('feature_type') == feature_type
            and abs(entry.get('file_size_mb', 0) - file_size_mb) < 10  # Within 10MB
        ]
        
        if not recent_history:
            return 1.0
        
        # Calculate average success rate and processing time
        successful = [entry for entry in recent_history if entry.get('success', False)]
        success_rate = len(successful) / len(recent_history)
        
        if success_rate < 0.5:  # Less than 50% success rate
            return 1.5  # Increase timeout
        elif success_rate > 0.9:  # More than 90% success rate
            return 0.8  # Decrease timeout
        else:
            return 1.0
    
    def _get_system_adjustment(self) -> float:
        """Get timeout adjustment based on current system conditions."""
        try:
            # Memory usage adjustment
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            if memory_usage > self.config.memory_threshold:
                # High memory usage - increase timeout to allow for slower processing
                memory_adjustment = 1.0 + (memory_usage - self.config.memory_threshold) * 2.0
            else:
                memory_adjustment = 1.0
            
            # CPU usage adjustment
            cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
            if cpu_usage > 0.8:  # High CPU usage
                cpu_adjustment = 1.0 + (cpu_usage - 0.8) * 1.5
            else:
                cpu_adjustment = 1.0
            
            return memory_adjustment * cpu_adjustment
            
        except Exception as e:
            logger.debug(f"Could not get system adjustment: {e}")
            return 1.0
    
    def _apply_system_load_adjustment(self, timeout: float) -> float:
        """Apply system load adjustment to timeout."""
        try:
            # Get system load average (Linux) or CPU usage (Windows)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()[0]  # 1-minute load average
                cpu_count = os.cpu_count() or 1
                load_ratio = load_avg / cpu_count
                
                if load_ratio > 1.0:  # System is overloaded
                    timeout *= self.config.load_multiplier
            else:
                # Windows - use CPU usage
                cpu_usage = psutil.cpu_percent(interval=0.1) / 100.0
                if cpu_usage > 0.8:  # High CPU usage
                    timeout *= self.config.load_multiplier
                    
        except Exception as e:
            logger.debug(f"Could not apply system load adjustment: {e}")
        
        return timeout
    
    def record_performance(self, 
                          feature_type: str,
                          file_size_mb: float,
                          processing_time: float,
                          success: bool,
                          timeout_used: float):
        """
        Record performance data for adaptive timeout calculation.
        
        Args:
            feature_type: Type of feature extracted
            file_size_mb: File size in megabytes
            processing_time: Actual processing time in seconds
            success: Whether the operation was successful
            timeout_used: Timeout that was used for this operation
        """
        entry = {
            'feature_type': feature_type,
            'file_size_mb': file_size_mb,
            'processing_time': processing_time,
            'success': success,
            'timeout_used': timeout_used,
            'timestamp': time.time()
        }
        
        self.performance_history.append(entry)
        
        # Keep history size manageable
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
        
        logger.debug(f"Recorded performance: {feature_type} {file_size_mb:.1f}MB "
                    f"took {processing_time:.1f}s (success={success})")
    
    def get_performance_stats(self, feature_type: str = None) -> Dict[str, Any]:
        """Get performance statistics for timeout optimization."""
        if not self.performance_history:
            return {}
        
        # Filter by feature type if specified
        history = self.performance_history
        if feature_type:
            history = [entry for entry in history if entry.get('feature_type') == feature_type]
        
        if not history:
            return {}
        
        successful = [entry for entry in history if entry.get('success', False)]
        failed = [entry for entry in history if not entry.get('success', False)]
        
        stats = {
            'total_operations': len(history),
            'successful_operations': len(successful),
            'failed_operations': len(failed),
            'success_rate': len(successful) / len(history) if history else 0,
            'avg_processing_time': sum(entry['processing_time'] for entry in history) / len(history),
            'avg_timeout_used': sum(entry['timeout_used'] for entry in history) / len(history),
            'timeout_efficiency': sum(entry['processing_time'] for entry in history) / sum(entry['timeout_used'] for entry in history) if history else 0
        }
        
        if successful:
            stats['avg_successful_time'] = sum(entry['processing_time'] for entry in successful) / len(successful)
        
        if failed:
            stats['avg_failed_time'] = sum(entry['processing_time'] for entry in failed) / len(failed)
        
        return stats
    
    def optimize_timeouts(self):
        """Optimize timeout values based on performance history."""
        for feature_type in self.feature_timeouts:
            stats = self.get_performance_stats(feature_type)
            if not stats:
                continue
            
            current_timeout = self.feature_timeouts[feature_type]
            avg_time = stats.get('avg_successful_time', current_timeout)
            success_rate = stats.get('success_rate', 1.0)
            
            # Adjust timeout based on performance
            if success_rate < 0.8:  # Low success rate
                new_timeout = current_timeout * 1.2  # Increase by 20%
            elif success_rate > 0.95 and avg_time < current_timeout * 0.7:
                new_timeout = current_timeout * 0.9  # Decrease by 10%
            else:
                new_timeout = current_timeout
            
            # Ensure timeout is reasonable
            new_timeout = max(10.0, min(new_timeout, 600.0))
            
            if abs(new_timeout - current_timeout) > 5.0:  # Only log significant changes
                logger.info(f"Optimized timeout for {feature_type}: {current_timeout:.1f}s -> {new_timeout:.1f}s")
                self.feature_timeouts[feature_type] = new_timeout
    
    def get_timeout_for_file(self, file_path: str, feature_type: str) -> float:
        """
        Calculate timeout for a specific file and feature type.
        
        Args:
            file_path: Path to the audio file
            feature_type: Type of feature to extract
            
        Returns:
            Calculated timeout in seconds
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except (OSError, FileNotFoundError):
            file_size_mb = 50.0  # Default size if file not accessible
        
        return self.calculate_timeout(file_size_mb, feature_type)

# Global timeout manager instance
_timeout_manager = None

def get_timeout_manager() -> AdaptiveTimeoutManager:
    """Get the global timeout manager instance."""
    global _timeout_manager
    if _timeout_manager is None:
        _timeout_manager = AdaptiveTimeoutManager()
    return _timeout_manager

def calculate_timeout(file_size_mb: float, feature_type: str = None) -> float:
    """Convenience function to calculate timeout."""
    return get_timeout_manager().calculate_timeout(file_size_mb, feature_type) 