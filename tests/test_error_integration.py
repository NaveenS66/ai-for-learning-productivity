"""Integration tests for error scenarios and recovery mechanisms."""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError, IntegrityError
import time

from src.ai_learning_accelerator.main import app
from src.ai_learning_accelerator.utils.exceptions import (
    ModelInferenceException,
    DatabaseConnectionException,
    ExternalServiceException,
    PrivacyViolationException,
    ErrorSeverity
)
from src.ai_learning_accelerator.utils.circuit_breaker import CircuitState
from src.ai_learning_accelerator.utils.monitoring import monitoring_system
from src.ai_learning_accelerator.utils.error_handler import error_handler
from src.ai_learning_accelerator.utils.circuit_breaker import circuit_breaker_registry
from src.ai_learning_accelerator.utils.fallback import fallback_registry


class TestDatabaseFailureScenarios:
    """Test database failure scenarios and recovery."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_database_connection_failure_recovery(self, client):
        """Test database connection failure and recovery."""
        with patch('src.ai_learning_accelerator.database.get_async_db') as mock_db:
            # Simulate database connection failure
            mock_db.side_effect = OperationalError("Connection failed", None, None)
            
            response = client.get("/api/v1/health/detailed")
            
            # Should handle gracefully and return error status
            assert response.status_code in [500, 503]
            data = response.json()
            assert "error" in data or data.get("status") == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_database_integrity_error_handling(self, client):
        """Test database integrity error handling."""
        with patch('src.ai_learning_accelerator.services.user.UserService.create_user') as mock_create:
            # Simulate integrity constraint violation
            mock_create.side_effect = IntegrityError("Duplicate key", None, None)
            
            response = client.post("/api/v1/users/", json={
                "email": "test@example.com",
                "password": "testpass123",
                "full_name": "Test User"
            })
            
            # Should return appropriate error response
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
            assert "integrity" in data["error"]["message"].lower() or "duplicate" in data["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_database_timeout_handling(self, client):
        """Test database timeout handling."""
        with patch('src.ai_learning_accelerator.database.get_async_db') as mock_db:
            # Simulate database timeout
            async def slow_db():
                await asyncio.sleep(2.0)  # Simulate slow query
                return Mock()
            
            mock_db.return_value = slow_db()
            
            # This should timeout and be handled gracefully
            response = client.get("/api/v1/health/ready", timeout=1.0)
            
            # Should handle timeout gracefully
            assert response.status_code in [500, 503, 408]


class TestAIModelFailureScenarios:
    """Test AI model failure scenarios and fallbacks."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_learning_engine_model_failure(self, client):
        """Test learning engine model failure with fallback."""
        with patch('src.ai_learning_accelerator.services.learning_engine.LearningEngine.generate_learning_path') as mock_generate:
            # Simulate model inference failure
            mock_generate.side_effect = ModelInferenceException(
                model_name="learning_path_generator",
                severity=ErrorSeverity.HIGH
            )
            
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                response = client.post("/api/v1/learning/paths", json={
                    "goals": ["Learn Python"],
                    "difficulty": "beginner"
                })
                
                # Should use fallback mechanism
                assert response.status_code in [200, 503]
                if response.status_code == 200:
                    data = response.json()
                    # Should indicate fallback was used
                    assert data.get("is_fallback") is True or "unavailable" in str(data).lower()
    
    @pytest.mark.asyncio
    async def test_debug_assistant_model_failure(self, client):
        """Test debug assistant model failure with fallback."""
        with patch('src.ai_learning_accelerator.services.debug_assistant.DebugAssistant.analyze_error') as mock_analyze:
            # Simulate model failure
            mock_analyze.side_effect = ModelInferenceException(
                model_name="debug_analyzer",
                severity=ErrorSeverity.MEDIUM
            )
            
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                response = client.post("/api/v1/debug/analyze", json={
                    "error_message": "TypeError: 'NoneType' object is not subscriptable",
                    "code_context": "def test(): return data[0]",
                    "stack_trace": "Traceback..."
                })
                
                # Should provide fallback debugging suggestions
                assert response.status_code in [200, 503]
                if response.status_code == 200:
                    data = response.json()
                    # Should have basic debugging suggestions
                    assert "suggestions" in data or "fallback" in str(data).lower()
    
    @pytest.mark.asyncio
    async def test_content_adaptation_failure(self, client):
        """Test content adaptation failure with simplified response."""
        with patch('src.ai_learning_accelerator.services.content_adaptation.ContentAdaptationService.adapt_content') as mock_adapt:
            # Simulate adaptation failure
            mock_adapt.side_effect = ModelInferenceException(
                model_name="content_adapter",
                severity=ErrorSeverity.MEDIUM
            )
            
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                response = client.post("/api/v1/multimodal/adapt", json={
                    "content": "Complex technical explanation",
                    "target_format": "visual",
                    "user_preferences": {"difficulty": "beginner"}
                })
                
                # Should return simplified content or fallback
                assert response.status_code in [200, 503]
                if response.status_code == 200:
                    data = response.json()
                    # Should indicate adaptation failed but provide fallback
                    assert data.get("is_fallback") is True or "original" in str(data).lower()


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration in real scenarios."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_repeated_failures(self, client):
        """Test circuit breaker opens after repeated failures."""
        # Reset circuit breakers
        await circuit_breaker_registry.reset_all()
        
        with patch('src.ai_learning_accelerator.services.learning_engine.LearningEngine.assess_skill_level') as mock_assess:
            # Simulate repeated failures
            mock_assess.side_effect = ModelInferenceException(
                model_name="skill_assessor",
                severity=ErrorSeverity.HIGH
            )
            
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                # Make multiple requests to trigger circuit breaker
                responses = []
                for i in range(5):
                    response = client.post("/api/v1/learning/assess", json={
                        "domain": "python",
                        "questions": ["What is a list?"]
                    })
                    responses.append(response)
                    await asyncio.sleep(0.1)  # Small delay between requests
                
                # Later requests should be rejected by circuit breaker
                last_response = responses[-1]
                assert last_response.status_code in [503, 502]
                
                # Check circuit breaker status
                stats = circuit_breaker_registry.get_all_stats()
                # Should have at least one open circuit breaker
                open_breakers = [name for name, stat in stats.items() if stat.get("state") == "open"]
                assert len(open_breakers) > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, client):
        """Test circuit breaker recovery after service restoration."""
        # Reset circuit breakers
        await circuit_breaker_registry.reset_all()
        
        with patch('src.ai_learning_accelerator.services.context_analyzer.ContextAnalyzer.analyze_workspace') as mock_analyze:
            # First, cause failures to open circuit
            mock_analyze.side_effect = ExternalServiceException(
                service_name="workspace_analyzer",
                status_code=500
            )
            
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                # Trigger circuit breaker opening
                for i in range(3):
                    client.post("/api/v1/context/analyze", json={
                        "workspace_path": "/test/path",
                        "files": ["test.py"]
                    })
                
                # Now simulate service recovery
                mock_analyze.side_effect = None
                mock_analyze.return_value = {
                    "insights": {"technology": "python"},
                    "recommendations": []
                }
                
                # Wait for recovery timeout (simulate)
                await asyncio.sleep(0.1)
                
                # Reset the specific circuit breaker to simulate recovery timeout
                breaker_name = "external_workspace_analyzer_circuit_breaker"
                try:
                    await circuit_breaker_registry.reset_breaker(breaker_name)
                except:
                    pass  # Breaker might not exist yet
                
                # Should work again after recovery
                response = client.post("/api/v1/context/analyze", json={
                    "workspace_path": "/test/path",
                    "files": ["test.py"]
                })
                
                # Should succeed or at least not be circuit breaker rejection
                assert response.status_code != 503 or "circuit breaker" not in response.json().get("error", {}).get("message", "").lower()


class TestFallbackMechanisms:
    """Test fallback mechanisms in various scenarios."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_cached_response_fallback(self, client):
        """Test cached response fallback mechanism."""
        # Clear fallback caches first
        fallback_registry.clear_all_caches()
        
        with patch('src.ai_learning_accelerator.services.analytics_engine.AnalyticsEngine.generate_progress_report') as mock_generate:
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                # First request succeeds and gets cached
                mock_generate.return_value = {
                    "progress": {"completed": 5, "total": 10},
                    "achievements": ["First Steps"]
                }
                
                response1 = client.get("/api/v1/analytics/progress")
                assert response1.status_code == 200
                
                # Second request fails, should use cached response
                mock_generate.side_effect = ModelInferenceException(
                    model_name="progress_analyzer",
                    severity=ErrorSeverity.MEDIUM
                )
                
                response2 = client.get("/api/v1/analytics/progress")
                
                # Should either succeed with cached data or provide fallback
                assert response2.status_code in [200, 503]
                if response2.status_code == 200:
                    data = response2.json()
                    # Should have progress data (either cached or fallback)
                    assert "progress" in data or "message" in data
    
    @pytest.mark.asyncio
    async def test_simplified_response_fallback(self, client):
        """Test simplified response fallback for complex operations."""
        with patch('src.ai_learning_accelerator.services.automation.PatternDetector.detect_patterns') as mock_detect:
            # Simulate complex operation failure
            mock_detect.side_effect = ModelInferenceException(
                model_name="pattern_detector",
                severity=ErrorSeverity.HIGH
            )
            
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                response = client.post("/api/v1/automation/detect-patterns", json={
                    "actions": [
                        {"type": "file_edit", "file": "test.py", "timestamp": "2023-01-01T10:00:00"},
                        {"type": "file_edit", "file": "test.py", "timestamp": "2023-01-01T10:05:00"}
                    ]
                })
                
                # Should provide simplified response or fallback
                assert response.status_code in [200, 503]
                if response.status_code == 200:
                    data = response.json()
                    # Should indicate fallback or provide basic response
                    assert data.get("is_fallback") is True or "unavailable" in str(data).lower()
    
    @pytest.mark.asyncio
    async def test_graceful_failure_fallback(self, client):
        """Test graceful failure when all fallbacks are exhausted."""
        with patch('src.ai_learning_accelerator.services.interaction_service.InteractionService.process_voice_input') as mock_process:
            # Simulate complete service failure
            mock_process.side_effect = ExternalServiceException(
                service_name="voice_processor",
                status_code=503
            )
            
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                response = client.post("/api/v1/interaction/voice", json={
                    "audio_data": "base64_encoded_audio",
                    "format": "wav"
                })
                
                # Should provide graceful failure response
                assert response.status_code in [503, 502]
                data = response.json()
                assert "error" in data
                # Should have user-friendly message
                assert "message" in data["error"]
                assert len(data["error"]["message"]) > 0


class TestPrivacyAndSecurityFailures:
    """Test privacy and security failure scenarios."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_privacy_violation_handling(self, client):
        """Test privacy violation detection and handling."""
        with patch('src.ai_learning_accelerator.services.privacy_service.PrivacyService.check_data_access') as mock_check:
            # Simulate privacy violation
            mock_check.side_effect = PrivacyViolationException(
                violation_type="unauthorized_data_access",
                user_id="test_user"
            )
            
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                response = client.get("/api/v1/privacy/data")
                
                # Should immediately reject with privacy error
                assert response.status_code == 403
                data = response.json()
                assert "error" in data
                assert "privacy" in data["error"]["message"].lower() or "violation" in data["error"]["message"].lower()
    
    @pytest.mark.asyncio
    async def test_encryption_failure_handling(self, client):
        """Test encryption failure handling."""
        with patch('src.ai_learning_accelerator.services.encryption_service.EncryptionService.encrypt_data') as mock_encrypt:
            # Simulate encryption failure
            from src.ai_learning_accelerator.utils.exceptions import EncryptionException
            mock_encrypt.side_effect = EncryptionException(
                operation="data_encryption"
            )
            
            # Mock authentication
            with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
                mock_auth.return_value = Mock(id="test_user")
                
                response = client.post("/api/v1/encryption/encrypt", json={
                    "data": "sensitive information",
                    "key_id": "test_key"
                })
                
                # Should handle encryption failure securely
                assert response.status_code in [500, 503]
                data = response.json()
                assert "error" in data
                # Should not expose sensitive details
                assert "sensitive information" not in str(data)


class TestMonitoringAndAlerting:
    """Test monitoring and alerting during error scenarios."""
    
    @pytest.mark.asyncio
    async def test_error_metrics_recording(self):
        """Test that errors are properly recorded in metrics."""
        # Clear existing metrics
        monitoring_system.metrics.metrics.clear()
        
        # Simulate an error scenario
        with patch('src.ai_learning_accelerator.services.learning_engine.LearningEngine.generate_explanation') as mock_generate:
            mock_generate.side_effect = ModelInferenceException(
                model_name="explanation_generator",
                severity=ErrorSeverity.HIGH
            )
            
            # Record the error through monitoring system
            await monitoring_system.record_ai_model_metrics(
                model_name="explanation_generator",
                operation="generate_explanation",
                duration=1.5,
                success=False
            )
            
            # Check that metrics were recorded
            metrics = monitoring_system.metrics.get_metrics("ai_model_requests_total")
            assert len(metrics) > 0
            
            # Check for failure metrics
            failure_metrics = [m for m in metrics if m.tags.get("success") == "False"]
            assert len(failure_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_alert_generation_on_critical_errors(self):
        """Test that critical errors generate alerts."""
        # Clear existing alerts
        monitoring_system.alerts.alerts.clear()
        
        # Simulate critical error
        await monitoring_system.alerts.create_alert({
            "name": "critical_ai_model_failure",
            "message": "AI model completely unavailable",
            "severity": "critical",
            "tags": {"model": "learning_engine", "component": "core"}
        })
        
        # Check that alert was created
        active_alerts = monitoring_system.alerts.get_active_alerts()
        assert len(active_alerts) > 0
        
        critical_alerts = [a for a in active_alerts if a.severity.value == "critical"]
        assert len(critical_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure_detection(self):
        """Test health check failure detection."""
        # Register a failing health check
        def failing_health_check():
            raise Exception("Health check failed")
        
        monitoring_system.health.register_health_check("test_failing_check", failing_health_check)
        
        # Run health check
        result = await monitoring_system.health.run_health_check("test_failing_check")
        
        # Should detect failure
        assert result["status"] == "error"
        assert "error" in result
        assert len(result["error"]) > 0


class TestEndToEndErrorScenarios:
    """Test end-to-end error scenarios across multiple components."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_complete_learning_journey_with_failures(self, client):
        """Test complete learning journey with various failures."""
        # Mock authentication
        with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id="test_user")
            
            # Step 1: User assessment fails
            with patch('src.ai_learning_accelerator.services.learning_engine.LearningEngine.assess_skill_level') as mock_assess:
                mock_assess.side_effect = ModelInferenceException(
                    model_name="skill_assessor",
                    severity=ErrorSeverity.MEDIUM
                )
                
                response1 = client.post("/api/v1/learning/assess", json={
                    "domain": "python",
                    "questions": ["What is a variable?"]
                })
                
                # Should handle gracefully with fallback
                assert response1.status_code in [200, 503]
            
            # Step 2: Learning path generation fails
            with patch('src.ai_learning_accelerator.services.learning_engine.LearningEngine.generate_learning_path') as mock_path:
                mock_path.side_effect = ModelInferenceException(
                    model_name="path_generator",
                    severity=ErrorSeverity.HIGH
                )
                
                response2 = client.post("/api/v1/learning/paths", json={
                    "goals": ["Learn Python basics"],
                    "difficulty": "beginner"
                })
                
                # Should provide fallback path or graceful failure
                assert response2.status_code in [200, 503]
            
            # Step 3: Content adaptation fails
            with patch('src.ai_learning_accelerator.services.content_adaptation.ContentAdaptationService.adapt_content') as mock_adapt:
                mock_adapt.side_effect = ModelInferenceException(
                    model_name="content_adapter",
                    severity=ErrorSeverity.MEDIUM
                )
                
                response3 = client.post("/api/v1/multimodal/adapt", json={
                    "content": "Variables store data",
                    "target_format": "interactive",
                    "user_preferences": {"learning_style": "visual"}
                })
                
                # Should provide original content or simplified version
                assert response3.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_debugging_session_with_cascading_failures(self, client):
        """Test debugging session with cascading failures."""
        # Mock authentication
        with patch('src.ai_learning_accelerator.auth.dependencies.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id="test_user")
            
            # Step 1: Code analysis fails
            with patch('src.ai_learning_accelerator.services.debug_assistant.DebugAssistant.analyze_error') as mock_analyze:
                mock_analyze.side_effect = ModelInferenceException(
                    model_name="code_analyzer",
                    severity=ErrorSeverity.HIGH
                )
                
                response1 = client.post("/api/v1/debug/analyze", json={
                    "error_message": "NameError: name 'x' is not defined",
                    "code_context": "print(x)",
                    "stack_trace": "Traceback (most recent call last)..."
                })
                
                # Should provide basic debugging suggestions
                assert response1.status_code in [200, 503]
            
            # Step 2: Solution ranking fails
            with patch('src.ai_learning_accelerator.services.debug_assistant.DebugAssistant.rank_solutions') as mock_rank:
                mock_rank.side_effect = ModelInferenceException(
                    model_name="solution_ranker",
                    severity=ErrorSeverity.MEDIUM
                )
                
                response2 = client.post("/api/v1/debug/solutions", json={
                    "analysis_id": "test_analysis",
                    "solutions": [
                        {"description": "Define variable x", "complexity": "low"},
                        {"description": "Import x from module", "complexity": "medium"}
                    ]
                })
                
                # Should provide unranked solutions or basic ranking
                assert response2.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])