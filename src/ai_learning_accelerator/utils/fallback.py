"""Fallback mechanism system for graceful degradation."""

import asyncio
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass
from enum import Enum

from ..logging_config import get_logger
from .exceptions import AILearningAcceleratorException, ErrorSeverity

logger = get_logger(__name__)

T = TypeVar('T')


class FallbackStrategy(str, Enum):
    """Fallback strategies."""
    CACHED_RESPONSE = "cached_response"
    SIMPLIFIED_RESPONSE = "simplified_response"
    DEFAULT_RESPONSE = "default_response"
    ALTERNATIVE_SERVICE = "alternative_service"
    MANUAL_INTERVENTION = "manual_intervention"
    GRACEFUL_FAILURE = "graceful_failure"


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanism."""
    strategy: FallbackStrategy
    priority: int = 1  # Lower numbers = higher priority
    enabled: bool = True
    max_retries: int = 0
    timeout: Optional[float] = None
    conditions: Optional[Dict[str, Any]] = None  # Conditions when this fallback applies


class FallbackResult:
    """Result of a fallback operation."""
    
    def __init__(
        self,
        value: Any,
        strategy_used: FallbackStrategy,
        is_fallback: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.value = value
        self.strategy_used = strategy_used
        self.is_fallback = is_fallback
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"FallbackResult(strategy={self.strategy_used}, is_fallback={self.is_fallback})"


class FallbackHandler:
    """Handler for implementing fallback mechanisms."""
    
    def __init__(self, name: str):
        self.name = name
        self.fallbacks: List[FallbackConfig] = []
        self.cache: Dict[str, Any] = {}
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "fallback_calls": 0,
            "strategy_usage": {}
        }
    
    def add_fallback(self, config: FallbackConfig):
        """Add a fallback configuration."""
        self.fallbacks.append(config)
        # Sort by priority (lower number = higher priority)
        self.fallbacks.sort(key=lambda x: x.priority)
        
        logger.info(
            "Fallback added",
            handler=self.name,
            strategy=config.strategy,
            priority=config.priority
        )
    
    async def execute_with_fallback(
        self,
        primary_func: Callable[..., T],
        *args,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> FallbackResult:
        """Execute function with fallback protection."""
        self._stats["total_calls"] += 1
        
        # Try primary function first
        try:
            result = await self._execute_function(primary_func, *args, **kwargs)
            
            # Cache successful result if cache key provided
            if cache_key:
                self.cache[cache_key] = result
            
            self._stats["successful_calls"] += 1
            return FallbackResult(result, FallbackStrategy.CACHED_RESPONSE, is_fallback=False)
            
        except Exception as e:
            logger.warning(
                "Primary function failed, attempting fallbacks",
                handler=self.name,
                exception=str(e),
                exception_type=type(e).__name__
            )
            
            # Try fallback strategies
            for fallback_config in self.fallbacks:
                if not fallback_config.enabled:
                    continue
                
                if not self._should_use_fallback(fallback_config, e):
                    continue
                
                try:
                    result = await self._execute_fallback(
                        fallback_config,
                        primary_func,
                        e,
                        cache_key,
                        *args,
                        **kwargs
                    )
                    
                    self._stats["fallback_calls"] += 1
                    self._stats["strategy_usage"][fallback_config.strategy] = (
                        self._stats["strategy_usage"].get(fallback_config.strategy, 0) + 1
                    )
                    
                    logger.info(
                        "Fallback successful",
                        handler=self.name,
                        strategy=fallback_config.strategy
                    )
                    
                    return FallbackResult(
                        result,
                        fallback_config.strategy,
                        metadata={"original_exception": str(e)}
                    )
                    
                except Exception as fallback_error:
                    logger.warning(
                        "Fallback failed",
                        handler=self.name,
                        strategy=fallback_config.strategy,
                        exception=str(fallback_error)
                    )
                    continue
            
            # All fallbacks failed
            logger.error(
                "All fallbacks exhausted",
                handler=self.name,
                original_exception=str(e)
            )
            
            # Return graceful failure result
            return FallbackResult(
                self._get_graceful_failure_response(e),
                FallbackStrategy.GRACEFUL_FAILURE,
                metadata={"original_exception": str(e)}
            )
    
    def _should_use_fallback(self, config: FallbackConfig, exception: Exception) -> bool:
        """Check if fallback should be used based on conditions."""
        if not config.conditions:
            return True
        
        # Check exception type conditions
        if "exception_types" in config.conditions:
            allowed_types = config.conditions["exception_types"]
            if not any(isinstance(exception, exc_type) for exc_type in allowed_types):
                return False
        
        # Check severity conditions
        if "min_severity" in config.conditions:
            if hasattr(exception, 'severity'):
                min_severity = ErrorSeverity(config.conditions["min_severity"])
                severity_order = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
                if severity_order.index(exception.severity) < severity_order.index(min_severity):
                    return False
        
        return True
    
    async def _execute_function(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with optional timeout."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await asyncio.to_thread(func, *args, **kwargs)
    
    async def _execute_fallback(
        self,
        config: FallbackConfig,
        primary_func: Callable,
        exception: Exception,
        cache_key: Optional[str],
        *args,
        **kwargs
    ) -> Any:
        """Execute specific fallback strategy."""
        if config.strategy == FallbackStrategy.CACHED_RESPONSE:
            return self._get_cached_response(cache_key)
        
        elif config.strategy == FallbackStrategy.SIMPLIFIED_RESPONSE:
            return await self._get_simplified_response(primary_func, exception, *args, **kwargs)
        
        elif config.strategy == FallbackStrategy.DEFAULT_RESPONSE:
            return self._get_default_response(primary_func, exception)
        
        elif config.strategy == FallbackStrategy.ALTERNATIVE_SERVICE:
            return await self._get_alternative_service_response(primary_func, *args, **kwargs)
        
        elif config.strategy == FallbackStrategy.MANUAL_INTERVENTION:
            return self._get_manual_intervention_response(exception)
        
        else:
            raise ValueError(f"Unknown fallback strategy: {config.strategy}")
    
    def _get_cached_response(self, cache_key: Optional[str]) -> Any:
        """Get cached response if available."""
        if not cache_key or cache_key not in self.cache:
            raise ValueError("No cached response available")
        
        return self.cache[cache_key]
    
    async def _get_simplified_response(
        self,
        primary_func: Callable,
        exception: Exception,
        *args,
        **kwargs
    ) -> Any:
        """Get simplified response based on function type."""
        func_name = getattr(primary_func, '__name__', 'unknown')
        
        # AI model fallbacks
        if 'generate' in func_name or 'predict' in func_name:
            return {
                "content": "I'm experiencing technical difficulties. Please try again later.",
                "confidence": 0.0,
                "is_fallback": True
            }
        
        # Learning path fallbacks
        elif 'learning_path' in func_name:
            return {
                "path": [],
                "milestones": [],
                "message": "Unable to generate personalized path. Please check back later.",
                "is_fallback": True
            }
        
        # Debug assistant fallbacks
        elif 'debug' in func_name or 'analyze' in func_name:
            return {
                "suggestions": [
                    "Check the error message carefully",
                    "Review recent code changes",
                    "Consult documentation for the relevant technology"
                ],
                "confidence": 0.0,
                "is_fallback": True
            }
        
        # Generic fallback
        else:
            return {
                "message": "Service temporarily unavailable. Please try again later.",
                "is_fallback": True
            }
    
    def _get_default_response(self, primary_func: Callable, exception: Exception) -> Any:
        """Get default response for the function type."""
        return {
            "error": "Service unavailable",
            "message": "We're experiencing technical difficulties. Please try again later.",
            "is_fallback": True,
            "support_contact": "support@ailearningaccelerator.com"
        }
    
    async def _get_alternative_service_response(
        self,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Get response from alternative service (placeholder for future implementation)."""
        # This would integrate with alternative AI services or simpler algorithms
        # For now, return a basic response
        return {
            "message": "Using alternative processing method",
            "is_fallback": True,
            "quality": "reduced"
        }
    
    def _get_manual_intervention_response(self, exception: Exception) -> Any:
        """Get response indicating manual intervention is needed."""
        return {
            "message": "This request requires manual review. Our team has been notified.",
            "ticket_id": f"MANUAL_{int(asyncio.get_event_loop().time())}",
            "is_fallback": True,
            "estimated_resolution": "24 hours"
        }
    
    def _get_graceful_failure_response(self, exception: Exception) -> Any:
        """Get graceful failure response."""
        user_message = "We're sorry, but we're unable to process your request right now."
        
        if hasattr(exception, 'user_message'):
            user_message = exception.user_message
        
        return {
            "error": "Service unavailable",
            "message": user_message,
            "is_fallback": True,
            "retry_after": 300,  # 5 minutes
            "support_contact": "support@ailearningaccelerator.com"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback handler statistics."""
        total_calls = max(self._stats["total_calls"], 1)
        return {
            "name": self.name,
            "total_calls": self._stats["total_calls"],
            "success_rate": self._stats["successful_calls"] / total_calls,
            "fallback_rate": self._stats["fallback_calls"] / total_calls,
            "strategy_usage": self._stats["strategy_usage"],
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
        logger.info("Cache cleared", handler=self.name)


class FallbackRegistry:
    """Registry for managing fallback handlers."""
    
    def __init__(self):
        self._handlers: Dict[str, FallbackHandler] = {}
    
    def get_handler(self, name: str) -> FallbackHandler:
        """Get or create a fallback handler."""
        if name not in self._handlers:
            self._handlers[name] = FallbackHandler(name)
        return self._handlers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all handlers."""
        return {name: handler.get_stats() for name, handler in self._handlers.items()}
    
    def clear_all_caches(self):
        """Clear all handler caches."""
        for handler in self._handlers.values():
            handler.clear_cache()


# Global registry instance
fallback_registry = FallbackRegistry()


# Convenience functions for common fallback configurations
def create_ai_model_fallbacks() -> List[FallbackConfig]:
    """Create fallback configurations for AI models."""
    return [
        FallbackConfig(
            strategy=FallbackStrategy.CACHED_RESPONSE,
            priority=1,
            conditions={"exception_types": [Exception]}
        ),
        FallbackConfig(
            strategy=FallbackStrategy.SIMPLIFIED_RESPONSE,
            priority=2,
            conditions={"exception_types": [Exception]}
        ),
        FallbackConfig(
            strategy=FallbackStrategy.DEFAULT_RESPONSE,
            priority=3
        )
    ]


def create_database_fallbacks() -> List[FallbackConfig]:
    """Create fallback configurations for database operations."""
    return [
        FallbackConfig(
            strategy=FallbackStrategy.CACHED_RESPONSE,
            priority=1,
            conditions={"exception_types": [Exception]}
        ),
        FallbackConfig(
            strategy=FallbackStrategy.GRACEFUL_FAILURE,
            priority=2
        )
    ]


def create_external_service_fallbacks() -> List[FallbackConfig]:
    """Create fallback configurations for external services."""
    return [
        FallbackConfig(
            strategy=FallbackStrategy.CACHED_RESPONSE,
            priority=1
        ),
        FallbackConfig(
            strategy=FallbackStrategy.ALTERNATIVE_SERVICE,
            priority=2
        ),
        FallbackConfig(
            strategy=FallbackStrategy.DEFAULT_RESPONSE,
            priority=3
        )
    ]