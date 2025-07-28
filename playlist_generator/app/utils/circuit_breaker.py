import time
import logging
from typing import Callable, Any, Optional
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit Breaker pattern implementation to prevent cascading failures.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests are rejected immediately
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception,
                 name: str = "default"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        # State tracking
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
        # Success tracking for half-open state
        self.success_count = 0
        self.required_success_count = 3
        
        logger.debug(f"Circuit Breaker '{name}' initialized with threshold={failure_threshold}, timeout={recovery_timeout}s")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit Breaker '{self.name}' attempting reset to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerOpenError(f"Circuit Breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(f"Circuit Breaker '{self.name}' HALF_OPEN success {self.success_count}/{self.required_success_count}")
            
            if self.success_count >= self.required_success_count:
                logger.info(f"Circuit Breaker '{self.name}' reset to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.last_failure_time = None
        else:
            # Reset failure count on success in CLOSED state
            if self.failure_count > 0:
                logger.debug(f"Circuit Breaker '{self.name}' resetting failure count on success")
                self.failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"Circuit Breaker '{self.name}' failure {self.failure_count}/{self.failure_threshold}: {type(exception).__name__}")
        
        if self.failure_count >= self.failure_threshold:
            if self.state == CircuitState.CLOSED:
                logger.error(f"Circuit Breaker '{self.name}' opening circuit")
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                logger.error(f"Circuit Breaker '{self.name}' re-opening circuit")
                self.state = CircuitState.OPEN
                self.success_count = 0
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'failure_threshold': self.failure_threshold,
            'recovery_timeout': self.recovery_timeout
        }
    
    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        logger.info(f"Circuit Breaker '{self.name}' manually reset to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

def circuit_breaker(failure_threshold: int = 5, 
                   recovery_timeout: int = 60,
                   expected_exception: type = Exception,
                   name: str = None):
    """
    Decorator to apply circuit breaker pattern to a function.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting reset
        expected_exception: Exception type to monitor
        name: Circuit breaker name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        cb_name = name or f"{func.__module__}.{func.__name__}"
        breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception, cb_name)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker to function for external access
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator

# Global circuit breaker registry
_circuit_breakers = {}

def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get a circuit breaker by name."""
    return _circuit_breakers.get(name)

def register_circuit_breaker(name: str, breaker: CircuitBreaker):
    """Register a circuit breaker for global access."""
    _circuit_breakers[name] = breaker
    logger.debug(f"Registered circuit breaker: {name}")

def get_all_circuit_breakers() -> dict:
    """Get all registered circuit breakers."""
    return _circuit_breakers.copy()

def reset_all_circuit_breakers():
    """Reset all circuit breakers to CLOSED state."""
    for name, breaker in _circuit_breakers.items():
        breaker.reset()
        logger.info(f"Reset circuit breaker: {name}") 