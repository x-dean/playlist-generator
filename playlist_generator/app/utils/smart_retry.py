import time
import logging
import random
import math
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    """Different retry strategies."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    ADAPTIVE = "adaptive"
    FIBONACCI = "fibonacci"
    CONSTANT = "constant"

class RetryResult:
    """Result of a retry operation."""
    
    def __init__(self, success: bool, result: Any = None, exception: Exception = None, 
                 attempts: int = 0, total_time: float = 0.0):
        self.success = success
        self.result = result
        self.exception = exception
        self.attempts = attempts
        self.total_time = total_time

class SmartRetryManager:
    """
    Intelligent retry strategies with different backoff policies.
    
    Features:
    - Multiple retry strategies (exponential, linear, adaptive, fibonacci)
    - Jitter to prevent thundering herd
    - Context-aware retry decisions
    - Performance tracking
    - Circuit breaker integration
    """
    
    def __init__(self):
        # Strategy configurations
        self.strategy_configs = {
            RetryStrategy.EXPONENTIAL: {
                'base_delay': 1.0,
                'max_delay': 60.0,
                'multiplier': 2.0,
                'jitter': 0.1
            },
            RetryStrategy.LINEAR: {
                'base_delay': 1.0,
                'max_delay': 30.0,
                'increment': 2.0,
                'jitter': 0.2
            },
            RetryStrategy.ADAPTIVE: {
                'base_delay': 1.0,
                'max_delay': 120.0,
                'success_multiplier': 0.8,
                'failure_multiplier': 1.5,
                'jitter': 0.15
            },
            RetryStrategy.FIBONACCI: {
                'base_delay': 1.0,
                'max_delay': 60.0,
                'jitter': 0.1
            },
            RetryStrategy.CONSTANT: {
                'delay': 5.0,
                'jitter': 0.3
            }
        }
        
        # Performance tracking
        self.retry_history = []
        self.max_history_size = 1000
    
    def retry_with_strategy(self, 
                           func: Callable,
                           strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                           max_attempts: int = 3,
                           timeout: Optional[float] = None,
                           context: Dict[str, Any] = None) -> RetryResult:
        """
        Retry a function with the specified strategy.
        
        Args:
            func: Function to retry
            strategy: Retry strategy to use
            max_attempts: Maximum number of retry attempts
            timeout: Overall timeout for all attempts
            context: Additional context for the operation
            
        Returns:
            RetryResult with success status and details
        """
        start_time = time.time()
        attempts = 0
        last_exception = None
        config = self.strategy_configs[strategy]
        
        logger.debug(f"Starting retry with strategy: {strategy.value}, max_attempts: {max_attempts}")
        
        while attempts < max_attempts:
            attempts += 1
            
            try:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"Retry timeout after {attempts} attempts")
                    break
                
                # Execute function
                result = func()
                
                # Success - record and return
                total_time = time.time() - start_time
                retry_result = RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_time=total_time
                )
                
                self._record_retry(retry_result, strategy, context)
                logger.debug(f"Retry succeeded on attempt {attempts}")
                return retry_result
                
            except Exception as e:
                last_exception = e
                logger.debug(f"Retry attempt {attempts} failed: {type(e).__name__}: {str(e)}")
                
                # Don't sleep after the last attempt
                if attempts < max_attempts:
                    delay = self._calculate_delay(attempts, strategy, config)
                    logger.debug(f"Waiting {delay:.2f}s before retry attempt {attempts + 1}")
                    time.sleep(delay)
        
        # All attempts failed
        total_time = time.time() - start_time
        retry_result = RetryResult(
            success=False,
            exception=last_exception,
            attempts=attempts,
            total_time=total_time
        )
        
        self._record_retry(retry_result, strategy, context)
        logger.warning(f"Retry failed after {attempts} attempts")
        return retry_result
    
    def _calculate_delay(self, attempt: int, strategy: RetryStrategy, config: Dict[str, Any]) -> float:
        """Calculate delay for the next retry attempt."""
        
        if strategy == RetryStrategy.EXPONENTIAL:
            delay = config['base_delay'] * (config['multiplier'] ** (attempt - 1))
            
        elif strategy == RetryStrategy.LINEAR:
            delay = config['base_delay'] + (config['increment'] * (attempt - 1))
            
        elif strategy == RetryStrategy.ADAPTIVE:
            # Use historical performance to adjust delay
            historical_delay = self._get_adaptive_delay(strategy)
            delay = historical_delay * config['failure_multiplier']
            
        elif strategy == RetryStrategy.FIBONACCI:
            delay = config['base_delay'] * self._fibonacci(attempt)
            
        elif strategy == RetryStrategy.CONSTANT:
            delay = config['delay']
            
        else:
            delay = config['base_delay']
        
        # Apply maximum delay limit
        max_delay = config.get('max_delay', 60.0)
        delay = min(delay, max_delay)
        
        # Apply jitter to prevent thundering herd
        jitter = config.get('jitter', 0.1)
        if jitter > 0:
            jitter_amount = delay * jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure minimum delay
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number."""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)
    
    def _get_adaptive_delay(self, strategy: RetryStrategy) -> float:
        """Get adaptive delay based on historical performance."""
        if not self.retry_history:
            return self.strategy_configs[strategy]['base_delay']
        
        # Get recent history for this strategy
        recent_history = [
            entry for entry in self.retry_history[-50:]
            if entry.get('strategy') == strategy.value
        ]
        
        if not recent_history:
            return self.strategy_configs[strategy]['base_delay']
        
        # Calculate average delay from successful retries
        successful = [entry for entry in recent_history if entry.get('success', False)]
        if successful:
            avg_delay = sum(entry.get('avg_delay', 1.0) for entry in successful) / len(successful)
            return avg_delay
        
        return self.strategy_configs[strategy]['base_delay']
    
    def _record_retry(self, retry_result: RetryResult, strategy: RetryStrategy, context: Dict[str, Any] = None):
        """Record retry attempt for performance tracking."""
        entry = {
            'strategy': strategy.value,
            'success': retry_result.success,
            'attempts': retry_result.attempts,
            'total_time': retry_result.total_time,
            'avg_delay': retry_result.total_time / retry_result.attempts if retry_result.attempts > 0 else 0,
            'context': context or {},
            'timestamp': time.time()
        }
        
        self.retry_history.append(entry)
        
        # Keep history size manageable
        if len(self.retry_history) > self.max_history_size:
            self.retry_history = self.retry_history[-self.max_history_size:]
    
    def exponential_backoff(self, func: Callable, max_attempts: int = 3, timeout: float = None) -> RetryResult:
        """Exponential backoff retry strategy."""
        return self.retry_with_strategy(func, RetryStrategy.EXPONENTIAL, max_attempts, timeout)
    
    def linear_backoff(self, func: Callable, max_attempts: int = 3, timeout: float = None) -> RetryResult:
        """Linear backoff retry strategy."""
        return self.retry_with_strategy(func, RetryStrategy.LINEAR, max_attempts, timeout)
    
    def adaptive_backoff(self, func: Callable, max_attempts: int = 3, timeout: float = None) -> RetryResult:
        """Adaptive backoff retry strategy."""
        return self.retry_with_strategy(func, RetryStrategy.ADAPTIVE, max_attempts, timeout)
    
    def fibonacci_backoff(self, func: Callable, max_attempts: int = 3, timeout: float = None) -> RetryResult:
        """Fibonacci backoff retry strategy."""
        return self.retry_with_strategy(func, RetryStrategy.FIBONACCI, max_attempts, timeout)
    
    def constant_backoff(self, func: Callable, max_attempts: int = 3, timeout: float = None) -> RetryResult:
        """Constant backoff retry strategy."""
        return self.retry_with_strategy(func, RetryStrategy.CONSTANT, max_attempts, timeout)
    
    def get_retry_stats(self, strategy: RetryStrategy = None) -> Dict[str, Any]:
        """Get retry performance statistics."""
        if not self.retry_history:
            return {}
        
        # Filter by strategy if specified
        history = self.retry_history
        if strategy:
            history = [entry for entry in history if entry.get('strategy') == strategy.value]
        
        if not history:
            return {}
        
        successful = [entry for entry in history if entry.get('success', False)]
        failed = [entry for entry in history if not entry.get('success', False)]
        
        stats = {
            'total_retries': len(history),
            'successful_retries': len(successful),
            'failed_retries': len(failed),
            'success_rate': len(successful) / len(history) if history else 0,
            'avg_attempts': sum(entry['attempts'] for entry in history) / len(history),
            'avg_total_time': sum(entry['total_time'] for entry in history) / len(history),
            'avg_delay': sum(entry['avg_delay'] for entry in history) / len(history)
        }
        
        if successful:
            stats['avg_successful_time'] = sum(entry['total_time'] for entry in successful) / len(successful)
        
        if failed:
            stats['avg_failed_time'] = sum(entry['total_time'] for entry in failed) / len(failed)
        
        return stats
    
    def optimize_strategy(self, strategy: RetryStrategy) -> Dict[str, Any]:
        """Optimize retry strategy based on performance history."""
        stats = self.get_retry_stats(strategy)
        if not stats:
            return {}
        
        config = self.strategy_configs[strategy]
        success_rate = stats.get('success_rate', 0)
        avg_attempts = stats.get('avg_attempts', 1)
        
        optimizations = {}
        
        if success_rate < 0.7:  # Low success rate
            if strategy == RetryStrategy.EXPONENTIAL:
                optimizations['suggestion'] = 'Increase base_delay or max_delay'
                optimizations['new_base_delay'] = config['base_delay'] * 1.5
            elif strategy == RetryStrategy.LINEAR:
                optimizations['suggestion'] = 'Increase increment or max_delay'
                optimizations['new_increment'] = config['increment'] * 1.2
        
        elif success_rate > 0.9 and avg_attempts < 1.5:  # High success rate, few attempts
            if strategy == RetryStrategy.EXPONENTIAL:
                optimizations['suggestion'] = 'Decrease base_delay'
                optimizations['new_base_delay'] = config['base_delay'] * 0.8
            elif strategy == RetryStrategy.LINEAR:
                optimizations['suggestion'] = 'Decrease increment'
                optimizations['new_increment'] = config['increment'] * 0.8
        
        return optimizations

def smart_retry(strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                max_attempts: int = 3,
                timeout: Optional[float] = None):
    """
    Decorator to apply smart retry to a function.
    
    Args:
        strategy: Retry strategy to use
        max_attempts: Maximum number of retry attempts
        timeout: Overall timeout for all attempts
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_manager = SmartRetryManager()
            
            def retry_func():
                return func(*args, **kwargs)
            
            result = retry_manager.retry_with_strategy(
                retry_func, strategy, max_attempts, timeout
            )
            
            if not result.success:
                raise result.exception
            
            return result.result
        
        return wrapper
    
    return decorator

# Global retry manager instance
_retry_manager = None

def get_retry_manager() -> SmartRetryManager:
    """Get the global retry manager instance."""
    global _retry_manager
    if _retry_manager is None:
        _retry_manager = SmartRetryManager()
    return _retry_manager

def retry_with_strategy(func: Callable, strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                       max_attempts: int = 3, timeout: Optional[float] = None) -> RetryResult:
    """Convenience function to retry with a specific strategy."""
    return get_retry_manager().retry_with_strategy(func, strategy, max_attempts, timeout) 