"""Resilience utilities for combining error handling, circuit breakers, and fallbacks."""

import asyncio
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps

from ..logging_config import get_logger
from .circuit_breaker import CircuitBreakerConfig, circuit_breaker_registry
from .fallback import FallbackConfig, FallbackStrategy, fallback_registry
from .exceptions import AILearningAcceleratorException, ErrorSeverity
from .monitoring import monitoring_system

logger = get_logger(__name__)

T = TypeVar('T')


def resilient_service(
    service_name: str,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    fallback_configs: Optional[List[FallbackConfig]] = None,
    cache_key_func: Optional[Callable[..., str]] = None,
    monitor_metrics: bool = True
):
    """
    Decorator that adds comprehensive resilience to service methods.
    
    Combines:
    - Circuit breaker protection
    - Fallback mechanisms
    - Error handling
    - Monitoring and metrics
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            operation_name = f"{service_name}.{func.__name__}"
            start_time = asyncio.get_event_loop().time()
            success = False
            
            try:
                # Get circuit breaker
                breaker = await circuit_breaker_registry.get_breaker(
                    f"{service_name}_circuit_breaker",
                    circuit_breaker_config
                )
                
                # Get fallback handler
                fallback_handler = fallback_registry.get_handler(f"{service_name}_fallback")
                
                # Add fallback configurations if provided
                if fallback_configs:
                    for config in fallback_configs:
                        fallback_handler.add_fallback(config)
                
                # Generate cache key if function provided
                cache_key = None
                if cache_key_func:
                    try:
                        cache_key = cache_key_func(*args, **kwargs)
                    except Exception as e:
                        logger.warning(f"Failed to generate cache key: {e}")
                
                # Execute with circuit breaker and fallback protection
                result = await fallback_handler.execute_with_fallback(
                    lambda: breaker.call(func, *args, **kwargs),
                    cache_key=cache_key
                )
                
                success = True
                
                # Record success metrics
                if monitor_metrics:
                    duration = asyncio.get_event_loop().time() - start_time
                    await monitoring_system.record_ai_model_metrics(
                        model_name=service_name,
                        operation=func.__name__,
                        duration=duration,
                        success=True
                    )
                
                # Return the actual result value
                return result.value
                
            except Exception as e:
                # Record failure metrics
                if monitor_metrics:
                    duration = asyncio.get_event_loop().time() - start_time
                    await monitoring_system.record_ai_model_metrics(
                        model_name=service_name,
                        operation=func.__name__,
                        duration=duration,
                        success=False
                    )
                
                # Log the error
                logger.error(
                    f"Service operation failed: {operation_name}",
                    exception=str(e),
                    exception_type=type(e).__name__,
                    duration=asyncio.get_event_loop().time() - start_time
                )
                
                # Re-raise the exception
                raise
        
        return wrapper
    return decorator


def ai_model_resilience(model_name: str):
    """Preconfigured resilience for AI model services."""
    from .circuit_breaker import get_ai_model_circuit_breaker_config
    from .fallback import create_ai_model_fallbacks
    
    return resilient_service(
        service_name=f"ai_model_{model_name}",
        circuit_breaker_config=get_ai_model_circuit_breaker_config(),
        fallback_configs=create_ai_model_fallbacks(),
        cache_key_func=lambda *args, **kwargs: f"{model_name}_{hash(str(args) + str(kwargs))}",
        monitor_metrics=True
    )


def database_resilience(operation_type: str):
    """Preconfigured resilience for database operations."""
    from .circuit_breaker import get_database_circuit_breaker_config
    from .fallback import create_database_fallbacks
    
    return resilient_service(
        service_name=f"database_{operation_type}",
        circuit_breaker_config=get_database_circuit_breaker_config(),
        fallback_configs=create_database_fallbacks(),
        monitor_metrics=True
    )


def external_service_resilience(service_name: str):
    """Preconfigured resilience for external service calls."""
    from .circuit_breaker import get_external_api_circuit_breaker_config
    from .fallback import create_external_service_fallbacks
    
    return resilient_service(
        service_name=f"external_{service_name}",
        circuit_breaker_config=get_external_api_circuit_breaker_config(),
        fallback_configs=create_external_service_fallbacks(),
        cache_key_func=lambda *args, **kwargs: f"{service_name}_{hash(str(args) + str(kwargs))}",
        monitor_metrics=True
    )


class ResilientServiceMixin:
    """Mixin class that provides resilience methods to service classes."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._circuit_breaker = None
        self._fallback_handler = None
    
    async def _get_circuit_breaker(self, config: Optional[CircuitBreakerConfig] = None):
        """Get or create circuit breaker for this service."""
        if self._circuit_breaker is None:
            self._circuit_breaker = await circuit_breaker_registry.get_breaker(
                f"{self.service_name}_circuit_breaker",
                config
            )
        return self._circuit_breaker
    
    def _get_fallback_handler(self):
        """Get or create fallback handler for this service."""
        if self._fallback_handler is None:
            self._fallback_handler = fallback_registry.get_handler(f"{self.service_name}_fallback")
        return self._fallback_handler
    
    async def execute_with_resilience(
        self,
        func: Callable[..., T],
        *args,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        fallback_configs: Optional[List[FallbackConfig]] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> T:
        """Execute a function with full resilience protection."""
        
        # Get circuit breaker and fallback handler
        breaker = await self._get_circuit_breaker(circuit_breaker_config)
        fallback_handler = self._get_fallback_handler()
        
        # Add fallback configurations
        if fallback_configs:
            for config in fallback_configs:
                fallback_handler.add_fallback(config)
        
        # Execute with protection
        result = await fallback_handler.execute_with_fallback(
            lambda: breaker.call(func, *args, **kwargs),
            cache_key=cache_key
        )
        
        return result.value


# Convenience functions for common error handling patterns
async def with_retry(
    func: Callable[..., T],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> T:
    """Execute function with retry logic."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = delay * (backoff_factor ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {wait_time}s",
                    exception=str(e),
                    max_retries=max_retries
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed",
                    exception=str(e)
                )
    
    # Re-raise the last exception
    raise last_exception


async def with_timeout(
    func: Callable[..., T],
    timeout_seconds: float,
    timeout_message: Optional[str] = None
) -> T:
    """Execute function with timeout protection."""
    try:
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(), timeout=timeout_seconds)
        else:
            return await asyncio.wait_for(
                asyncio.to_thread(func),
                timeout=timeout_seconds
            )
    except asyncio.TimeoutError:
        message = timeout_message or f"Operation timed out after {timeout_seconds} seconds"
        logger.error(message)
        from .exceptions import TimeoutException
        raise TimeoutException(
            operation=getattr(func, '__name__', 'unknown'),
            timeout_seconds=timeout_seconds
        )


def handle_exceptions(*exception_types, default_return=None, log_level="error"):
    """Decorator that handles specific exceptions and returns a default value."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except exception_types as e:
                log_func = getattr(logger, log_level, logger.error)
                log_func(
                    f"Handled exception in {func.__name__}",
                    exception=str(e),
                    exception_type=type(e).__name__
                )
                return default_return
        return wrapper
    return decorator