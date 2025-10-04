# utils/circuit_breaker.py
"""
Production Circuit Breaker Pattern
==================================
Prevents cascading failures when external services (APIs) fail.
Implements automatic recovery and fallback mechanisms.
"""

import time
import logging
from enum import Enum
from typing import Callable, Optional, Any, Dict
from functools import wraps
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered
    
    Usage:
        breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60,
            expected_exception=requests.RequestException
        )
        
        @breaker
        def call_external_api():
            return requests.get("https://api.example.com")
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers the breaker
            name: Name for logging and monitoring
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self._failure_count
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return False
        
        time_since_failure = datetime.utcnow() - self._last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self._failure_count = 0
            self._last_failure_time = None
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info(f"Circuit breaker '{self.name}' recovered: HALF_OPEN -> CLOSED")
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery attempt
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' recovery failed: HALF_OPEN -> OPEN"
                )
            
            elif self._failure_count >= self.failure_threshold:
                # Too many failures, open circuit
                self._state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker '{self.name}' opened after {self._failure_count} failures"
                )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if we should attempt recovery
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}': OPEN -> HALF_OPEN (attempting recovery)")
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable. Retry after {self.recovery_timeout}s."
                    )
        
        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to wrap function with circuit breaker.
        
        Example:
            @circuit_breaker
            def risky_api_call():
                return requests.get("https://api.example.com")
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get circuit breaker status.
        
        Returns:
            Dictionary with current status
        """
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'failure_threshold': self.failure_threshold,
                'last_failure': self._last_failure_time.isoformat() if self._last_failure_time else None,
                'recovery_timeout': self.recovery_timeout
            }
    
    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self._failure_count = 0
            self._last_failure_time = None
            self._state = CircuitState.CLOSED
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different services.
    
    Usage:
        manager = CircuitBreakerManager()
        
        # Get or create circuit breaker for a service
        anthropic_breaker = manager.get_breaker(
            'anthropic',
            failure_threshold=3,
            recovery_timeout=30
        )
        
        @anthropic_breaker
        def call_anthropic_api():
            ...
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.
        
        Args:
            name: Service name
            failure_threshold: Failures before opening
            recovery_timeout: Recovery timeout in seconds
            expected_exception: Exception type to catch
            
        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    expected_exception=expected_exception,
                    name=name
                )
            return self._breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all circuit breakers.
        
        Returns:
            Dictionary mapping service names to status
        """
        with self._lock:
            return {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")


# Global circuit breaker manager
_manager = CircuitBreakerManager()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception
) -> CircuitBreaker:
    """
    Get or create a circuit breaker from global manager.
    
    Args:
        name: Service name
        failure_threshold: Failures before opening
        recovery_timeout: Recovery timeout in seconds
        expected_exception: Exception type to catch
        
    Returns:
        CircuitBreaker instance
    """
    return _manager.get_breaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )


def get_all_circuit_breakers_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status of all circuit breakers.
    
    Returns:
        Dictionary with all circuit breaker statuses
    """
    return _manager.get_all_status()


# Decorator for common external services
def with_anthropic_circuit_breaker(func: Callable) -> Callable:
    """Decorator for Anthropic API calls with circuit breaker."""
    breaker = get_circuit_breaker(
        name='anthropic',
        failure_threshold=3,
        recovery_timeout=30,
        expected_exception=Exception
    )
    return breaker(func)


def with_youtube_circuit_breaker(func: Callable) -> Callable:
    """Decorator for YouTube API calls with circuit breaker."""
    breaker = get_circuit_breaker(
        name='youtube',
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=Exception
    )
    return breaker(func)


def with_serpapi_circuit_breaker(func: Callable) -> Callable:
    """Decorator for SerpAPI calls with circuit breaker."""
    breaker = get_circuit_breaker(
        name='serpapi',
        failure_threshold=3,
        recovery_timeout=30,
        expected_exception=Exception
    )
    return breaker(func)
