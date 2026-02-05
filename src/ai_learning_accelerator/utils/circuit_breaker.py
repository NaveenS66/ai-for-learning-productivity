"""Circuit breaker implementation for handling external service failures."""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from dataclasses import dataclass, field
from functools import wraps

from ..logging_config import get_logger
from .exceptions import ExternalServiceException, TimeoutException

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds
    expected_exceptions: tuple = (Exception,)  # Exceptions that count as failures


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    state_changes: Dict[str, int] = field(default_factory=lambda: {
        CircuitState.CLOSED: 0,
        CircuitState.OPEN: 0,
        CircuitState.HALF_OPEN: 0
    })


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        
        logger.info(
            "Circuit breaker initialized",
            name=name,
            config=self.config.__dict__
        )
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            self.stats.total_requests += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    self._log_rejected_request()
                    raise ExternalServiceException(
                        service_name=self.name,
                        details={"reason": "Circuit breaker is open", "stats": self.stats.__dict__}
                    )
            
            # Execute the function
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.timeout
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(func, *args, **kwargs),
                        timeout=self.config.timeout
                    )
                
                await self._on_success()
                return result
                
            except asyncio.TimeoutError:
                await self._on_failure(TimeoutException(
                    operation=f"{self.name}.{func.__name__}",
                    timeout_seconds=self.config.timeout
                ))
                raise
            except self.config.expected_exceptions as e:
                await self._on_failure(e)
                raise
            except Exception as e:
                # Unexpected exceptions don't count as failures
                logger.warning(
                    "Unexpected exception in circuit breaker",
                    name=self.name,
                    exception=str(e),
                    exception_type=type(e).__name__
                )
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.stats.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.stats.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
    
    async def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.stats.success_count = 0  # Reset success count for half-open test
        self.stats.state_changes[self.state] += 1
        
        logger.info(
            "Circuit breaker state transition",
            name=self.name,
            from_state=old_state,
            to_state=self.state
        )
    
    async def _on_success(self):
        """Handle successful request."""
        self.stats.success_count += 1
        self.stats.total_successes += 1
        self.stats.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
        elif self.state == CircuitState.OPEN:
            # This shouldn't happen, but handle gracefully
            await self._transition_to_closed()
    
    async def _on_failure(self, exception: Exception):
        """Handle failed request."""
        self.stats.failure_count += 1
        self.stats.total_failures += 1
        self.stats.last_failure_time = time.time()
        
        logger.warning(
            "Circuit breaker recorded failure",
            name=self.name,
            exception=str(exception),
            failure_count=self.stats.failure_count,
            state=self.state
        )
        
        if self.state == CircuitState.CLOSED:
            if self.stats.failure_count >= self.config.failure_threshold:
                await self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            await self._transition_to_open()
    
    async def _transition_to_closed(self):
        """Transition circuit to closed state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.stats.failure_count = 0  # Reset failure count
        self.stats.state_changes[self.state] += 1
        
        logger.info(
            "Circuit breaker state transition",
            name=self.name,
            from_state=old_state,
            to_state=self.state,
            message="Service recovered"
        )
    
    async def _transition_to_open(self):
        """Transition circuit to open state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.stats.state_changes[self.state] += 1
        
        logger.error(
            "Circuit breaker state transition",
            name=self.name,
            from_state=old_state,
            to_state=self.state,
            failure_count=self.stats.failure_count,
            message="Service failing, circuit opened"
        )
    
    def _log_rejected_request(self):
        """Log rejected request due to open circuit."""
        logger.warning(
            "Request rejected by circuit breaker",
            name=self.name,
            state=self.state,
            time_since_failure=time.time() - (self.stats.last_failure_time or 0)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state,
            "config": self.config.__dict__,
            "stats": self.stats.__dict__,
            "health_ratio": (
                self.stats.total_successes / max(self.stats.total_requests, 1)
            )
        }
    
    async def reset(self):
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.stats.failure_count = 0
            self.stats.success_count = 0
            
            logger.info(
                "Circuit breaker manually reset",
                name=self.name,
                from_state=old_state,
                to_state=self.state
            )


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
    
    async def reset_breaker(self, name: str):
        """Reset a specific circuit breaker."""
        if name in self._breakers:
            await self._breakers[name].reset()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
):
    """Decorator for applying circuit breaker to functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            breaker = await circuit_breaker_registry.get_breaker(name, config)
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


# Convenience function for common configurations
def get_ai_model_circuit_breaker_config() -> CircuitBreakerConfig:
    """Get circuit breaker configuration optimized for AI models."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        success_threshold=2,
        timeout=60.0,  # AI models can be slow
        expected_exceptions=(Exception,)
    )


def get_database_circuit_breaker_config() -> CircuitBreakerConfig:
    """Get circuit breaker configuration optimized for database operations."""
    return CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=10.0,
        success_threshold=3,
        timeout=10.0,
        expected_exceptions=(Exception,)
    )


def get_external_api_circuit_breaker_config() -> CircuitBreakerConfig:
    """Get circuit breaker configuration optimized for external APIs."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        success_threshold=2,
        timeout=30.0,
        expected_exceptions=(Exception,)
    )