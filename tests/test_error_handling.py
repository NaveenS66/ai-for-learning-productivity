"""Tests for comprehensive error handling system."""

import asyncio
import pytest
from unittest.mock import Mock, patch
from fastapi import Request
from fastapi.testclient import TestClient

from src.ai_learning_accelerator.utils.exceptions import (
    AILearningAcceleratorException,
    ErrorSeverity,
    ErrorCategory,
    ModelInferenceException,
    DatabaseConnectionException,
    ValidationException,
    PrivacyViolationException
)
from src.ai_learning_accelerator.utils.error_handler import ErrorHandler
from src.ai_learning_accelerator.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from src.ai_learning_accelerator.utils.fallback import FallbackHandler, FallbackConfig, FallbackStrategy
from src.ai_learning_accelerator.utils.monitoring import MonitoringSystem, AlertSeverity
from src.ai_learning_accelerator.utils.resilience import resilient_service, with_retry, with_timeout


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_ai_learning_accelerator_exception_creation(self):
        """Test creating custom exceptions."""
        exc = AILearningAcceleratorException(
            message="Test error",
            error_code="TEST_ERROR",
            category=ErrorCategory.AI_MODEL,
            severity=ErrorSeverity.HIGH,
            details={"test": "data"}
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == "TEST_ERROR"
        assert exc.category == ErrorCategory.AI_MODEL
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.details == {"test": "data"}
        assert exc.recoverable is True
        assert "AI services" in exc.user_message
    
    def test_model_inference_exception(self):
        """Test AI model specific exception."""
        exc = ModelInferenceException(
            model_name="test_model",
            input_data={"input": "test"}
        )
        
        assert exc.error_code == "MODEL_INFERENCE_FAILED"
        assert exc.category == ErrorCategory.AI_MODEL
        assert exc.details["model_name"] == "test_model"
        assert exc.details["input_data"] == {"input": "test"}
    
    def test_exception_to_dict(self):
        """Test exception serialization."""
        exc = ValidationException(
            message="Invalid input",
            field="test_field"
        )
        
        result = exc.to_dict()
        
        assert result["error_code"] == "VALIDATION_ERROR"
        assert result["message"] == "Invalid input"
        assert result["category"] == ErrorCategory.VALIDATION.value
        assert result["details"]["field"] == "test_field"


class TestErrorHandler:
    """Test error handler functionality."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler instance."""
        return ErrorHandler()
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.query_params = {}
        request.state.user_id = "test_user"
        request.headers = {"user-agent": "test-agent"}
        request.client.host = "127.0.0.1"
        return request
    
    @pytest.mark.asyncio
    async def test_handle_custom_exception(self, error_handler, mock_request):
        """Test handling custom exceptions."""
        exc = ModelInferenceException(model_name="test_model")
        
        response = await error_handler.handle_exception(mock_request, exc)
        
        assert response.status_code == 503
        response_data = response.body.decode()
        assert "MODEL_INFERENCE_FAILED" in response_data
        assert "test_model" in response_data
    
    @pytest.mark.asyncio
    async def test_handle_unknown_exception(self, error_handler, mock_request):
        """Test handling unknown exceptions."""
        exc = ValueError("Test error")
        
        response = await error_handler.handle_exception(mock_request, exc)
        
        assert response.status_code == 500
        response_data = response.body.decode()
        assert "UNKNOWN_ERROR" in response_data
    
    def test_error_statistics(self, error_handler):
        """Test error statistics tracking."""
        # Simulate some errors
        error_handler._track_error({
            "error_id": "1",
            "error_code": "TEST_ERROR",
            "category": "test",
            "severity": "medium",
            "timestamp": "2023-01-01T00:00:00"
        })
        
        stats = error_handler.get_error_stats()
        
        assert stats["total_errors"] == 1
        assert stats["error_counts_by_code"]["TEST_ERROR"] == 1
        assert len(stats["recent_errors_count"]) == 1


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker instance."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            success_threshold=1,
            timeout=1.0
        )
        return CircuitBreaker("test_breaker", config)
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call."""
        async def success_func():
            return "success"
        
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self, circuit_breaker):
        """Test circuit opens after threshold failures."""
        async def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)
        assert circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self, circuit_breaker):
        """Test circuit rejects calls when open."""
        # Force circuit to open state
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.stats.last_failure_time = asyncio.get_event_loop().time()
        
        async def test_func():
            return "should not execute"
        
        with pytest.raises(Exception) as exc_info:
            await circuit_breaker.call(test_func)
        
        assert "Circuit breaker is open" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_recovery(self, circuit_breaker):
        """Test circuit recovery after timeout."""
        # Force circuit to open state with old failure time
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.stats.last_failure_time = asyncio.get_event_loop().time() - 2.0
        
        async def success_func():
            return "recovered"
        
        result = await circuit_breaker.call(success_func)
        assert result == "recovered"
        assert circuit_breaker.state == CircuitState.CLOSED


class TestFallbackHandler:
    """Test fallback handler functionality."""
    
    @pytest.fixture
    def fallback_handler(self):
        """Create fallback handler instance."""
        handler = FallbackHandler("test_handler")
        
        # Add fallback configurations
        handler.add_fallback(FallbackConfig(
            strategy=FallbackStrategy.CACHED_RESPONSE,
            priority=1
        ))
        handler.add_fallback(FallbackConfig(
            strategy=FallbackStrategy.SIMPLIFIED_RESPONSE,
            priority=2
        ))
        handler.add_fallback(FallbackConfig(
            strategy=FallbackStrategy.DEFAULT_RESPONSE,
            priority=3
        ))
        
        return handler
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, fallback_handler):
        """Test successful primary function execution."""
        async def success_func():
            return "primary_result"
        
        result = await fallback_handler.execute_with_fallback(success_func)
        
        assert result.value == "primary_result"
        assert not result.is_fallback
    
    @pytest.mark.asyncio
    async def test_cached_fallback(self, fallback_handler):
        """Test cached response fallback."""
        # Add cached response
        fallback_handler.cache["test_key"] = "cached_result"
        
        async def failing_func():
            raise Exception("Primary failed")
        
        result = await fallback_handler.execute_with_fallback(
            failing_func,
            cache_key="test_key"
        )
        
        assert result.value == "cached_result"
        assert result.is_fallback
        assert result.strategy_used == FallbackStrategy.CACHED_RESPONSE
    
    @pytest.mark.asyncio
    async def test_simplified_response_fallback(self, fallback_handler):
        """Test simplified response fallback."""
        async def failing_func():
            raise Exception("Primary failed")
        
        # Mock function name to trigger AI model fallback
        failing_func.__name__ = "generate_response"
        
        result = await fallback_handler.execute_with_fallback(failing_func)
        
        assert result.is_fallback
        assert result.strategy_used == FallbackStrategy.SIMPLIFIED_RESPONSE
        assert "technical difficulties" in result.value["content"]
    
    @pytest.mark.asyncio
    async def test_graceful_failure_fallback(self, fallback_handler):
        """Test graceful failure when all fallbacks fail."""
        # Remove all fallback configs to force graceful failure
        fallback_handler.fallbacks.clear()
        
        async def failing_func():
            raise Exception("Primary failed")
        
        result = await fallback_handler.execute_with_fallback(failing_func)
        
        assert result.is_fallback
        assert result.strategy_used == FallbackStrategy.GRACEFUL_FAILURE
        assert "unable to process" in result.value["message"]


class TestMonitoringSystem:
    """Test monitoring system functionality."""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create monitoring system instance."""
        return MonitoringSystem()
    
    @pytest.mark.asyncio
    async def test_record_metrics(self, monitoring_system):
        """Test recording metrics."""
        await monitoring_system.metrics.increment_counter("test_counter", 5.0, {"tag": "value"})
        await monitoring_system.metrics.set_gauge("test_gauge", 10.0)
        await monitoring_system.metrics.record_timer("test_timer", 1.5)
        
        # Check metrics were recorded
        counter_metrics = monitoring_system.metrics.get_metrics("test_counter")
        assert len(counter_metrics) == 1
        assert counter_metrics[0].value == 5.0
        assert counter_metrics[0].tags == {"tag": "value"}
        
        gauge_metrics = monitoring_system.metrics.get_metrics("test_gauge")
        assert len(gauge_metrics) == 1
        assert gauge_metrics[0].value == 10.0
    
    @pytest.mark.asyncio
    async def test_create_alerts(self, monitoring_system):
        """Test creating alerts."""
        await monitoring_system.alerts.create_alert({
            "name": "test_alert",
            "message": "Test alert message",
            "severity": AlertSeverity.WARNING,
            "tags": {"component": "test"}
        })
        
        active_alerts = monitoring_system.alerts.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].name == "test_alert"
        assert active_alerts[0].severity == AlertSeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_health_checks(self, monitoring_system):
        """Test health check registration and execution."""
        def test_health_check():
            return {"status": "healthy", "test": True}
        
        monitoring_system.health.register_health_check("test_check", test_health_check)
        
        result = await monitoring_system.health.run_health_check("test_check")
        
        assert result["status"] == "healthy"
        assert result["test"] is True
        assert "timestamp" in result


class TestResilientService:
    """Test resilient service decorator."""
    
    @pytest.mark.asyncio
    async def test_resilient_service_success(self):
        """Test resilient service with successful execution."""
        @resilient_service("test_service")
        async def test_function(value):
            return f"processed_{value}"
        
        result = await test_function("input")
        assert result == "processed_input"
    
    @pytest.mark.asyncio
    async def test_resilient_service_with_fallback(self):
        """Test resilient service with fallback on failure."""
        call_count = 0
        
        @resilient_service(
            "test_service",
            fallback_configs=[
                FallbackConfig(
                    strategy=FallbackStrategy.DEFAULT_RESPONSE,
                    priority=1
                )
            ]
        )
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Function failed")
        
        result = await failing_function()
        
        # Should have attempted the function and then used fallback
        assert call_count >= 1
        assert "Service unavailable" in result["error"]
    
    @pytest.mark.asyncio
    async def test_with_retry_success_after_failure(self):
        """Test retry mechanism with eventual success."""
        call_count = 0
        
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await with_retry(flaky_function, max_retries=3, delay=0.1)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_with_timeout_success(self):
        """Test timeout mechanism with successful execution."""
        async def quick_function():
            await asyncio.sleep(0.1)
            return "completed"
        
        result = await with_timeout(quick_function, timeout_seconds=1.0)
        assert result == "completed"
    
    @pytest.mark.asyncio
    async def test_with_timeout_failure(self):
        """Test timeout mechanism with timeout exceeded."""
        async def slow_function():
            await asyncio.sleep(2.0)
            return "should not complete"
        
        with pytest.raises(Exception) as exc_info:
            await with_timeout(slow_function, timeout_seconds=0.5)
        
        assert "timed out" in str(exc_info.value)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple error handling components."""
    
    @pytest.mark.asyncio
    async def test_ai_model_failure_scenario(self):
        """Test complete AI model failure handling scenario."""
        from src.ai_learning_accelerator.utils.resilience import ai_model_resilience
        
        call_count = 0
        
        @ai_model_resilience("test_model")
        async def ai_model_inference(prompt):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ModelInferenceException(model_name="test_model")
            return {"response": f"Generated response for: {prompt}"}
        
        # Should use fallback after failures
        result = await ai_model_inference("test prompt")
        
        # Should get fallback response
        assert "technical difficulties" in result.get("content", "")
    
    @pytest.mark.asyncio
    async def test_database_failure_scenario(self):
        """Test database failure handling scenario."""
        from src.ai_learning_accelerator.utils.resilience import database_resilience
        
        @database_resilience("query")
        async def database_query():
            raise DatabaseConnectionException()
        
        # Should handle database failure gracefully
        result = await database_query()
        
        # Should get fallback response indicating service unavailable
        assert "unavailable" in result.get("message", "").lower()
    
    @pytest.mark.asyncio
    async def test_privacy_violation_handling(self):
        """Test privacy violation handling."""
        exc = PrivacyViolationException(
            violation_type="unauthorized_data_access",
            user_id="test_user"
        )
        
        error_handler = ErrorHandler()
        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/user/data"
        mock_request.query_params = {}
        mock_request.state.user_id = "test_user"
        mock_request.headers = {"user-agent": "test-agent"}
        mock_request.client.host = "127.0.0.1"
        
        response = await error_handler.handle_exception(mock_request, exc)
        
        assert response.status_code == 403
        response_data = response.body.decode()
        assert "PRIVACY_VIOLATION" in response_data
        assert not exc.recoverable  # Privacy violations should not be recoverable


if __name__ == "__main__":
    pytest.main([__file__])