"""
Integration tests for end-to-end user workflows.

These tests validate complete user journeys from initiation to completion,
ensuring all components work together correctly.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.ai_learning_accelerator.services.workflow_orchestrator import (
    WorkflowOrchestrator, WorkflowType, WorkflowStatus, WorkflowContext
)
from src.ai_learning_accelerator.models.learning import LearningGoal
from src.ai_learning_accelerator.models.debugging import ErrorContext
from src.ai_learning_accelerator.models.user import User
from src.ai_learning_accelerator.utils.exceptions import WorkflowError, ComponentError


@pytest.fixture
def mock_components():
    """Create mock components for workflow orchestrator."""
    return {
        'learning_engine': AsyncMock(),
        'debug_assistant': AsyncMock(),
        'context_analyzer': AsyncMock(),
        'pattern_detector': AsyncMock(),
        'workflow_generator': AsyncMock(),
        'analytics_engine': AsyncMock(),
        'interaction_service': AsyncMock()
    }


@pytest.fixture
def workflow_orchestrator(mock_components):
    """Create workflow orchestrator with mocked components."""
    return WorkflowOrchestrator(**mock_components)


@pytest.fixture
def sample_learning_goals():
    """Create sample learning goals for testing."""
    return [
        LearningGoal(
            title="Learn Python Basics",
            description="Master fundamental Python programming concepts",
            domain="python",
            difficulty_level="beginner",
            estimated_duration=20,
            prerequisites=[],
            success_criteria=["Complete basic syntax exercises", "Build simple programs"]
        ),
        LearningGoal(
            title="Understand Data Structures",
            description="Learn about lists, dictionaries, and sets",
            domain="python",
            difficulty_level="intermediate",
            estimated_duration=15,
            prerequisites=["python-basics"],
            success_criteria=["Implement common data structures", "Solve algorithmic problems"]
        )
    ]


@pytest.fixture
def sample_error_context():
    """Create sample error context for testing."""
    return ErrorContext(
        error_type="AttributeError",
        error_message="'NoneType' object has no attribute 'split'",
        stack_trace="Traceback (most recent call last):\n  File \"test.py\", line 5, in <module>\n    result = data.split(',')\nAttributeError: 'NoneType' object has no attribute 'split'",
        file_path="test.py",
        line_number=5,
        code_snippet="data = get_data()\nresult = data.split(',')",
        environment_info={"python_version": "3.9.0", "os": "linux"}
    )


@pytest.fixture
def sample_user_actions():
    """Create sample user actions for testing."""
    return [
        {
            "action_type": "file_save",
            "timestamp": datetime.utcnow(),
            "context": {"file_path": "main.py"},
            "parameters": {"content": "print('Hello, World!')"}
        },
        {
            "action_type": "command_run",
            "timestamp": datetime.utcnow(),
            "context": {"command": "python main.py"},
            "parameters": {"working_directory": "/project"}
        },
        {
            "action_type": "file_save",
            "timestamp": datetime.utcnow(),
            "context": {"file_path": "main.py"},
            "parameters": {"content": "print('Hello, World!')\nprint('Goodbye!')"}
        }
    ]


class TestLearningJourneyWorkflow:
    """Test complete learning journey workflows."""

    @pytest.mark.asyncio
    async def test_successful_learning_journey(self, workflow_orchestrator, sample_learning_goals, mock_components):
        """Test a complete successful learning journey workflow."""
        user_id = "test_user_123"
        
        # Configure mock responses
        mock_components['learning_engine'].assess_skill_level.return_value = {
            "skill_level": "beginner",
            "competencies": {"python": 0.2, "programming": 0.3}
        }
        mock_components['learning_engine'].analyze_learning_goals.return_value = {
            "validated_goals": sample_learning_goals,
            "difficulty_progression": "appropriate"
        }
        mock_components['learning_engine'].generate_learning_path.return_value = {
            "path_id": "path_123",
            "milestones": ["basics", "data_structures", "projects"],
            "estimated_duration": 35
        }
        mock_components['learning_engine'].adapt_content_for_user.return_value = {
            "adapted_content": {"format": "interactive", "complexity": "beginner"}
        }
        mock_components['analytics_engine'].setup_learning_analytics.return_value = {
            "analytics_id": "analytics_123"
        }
        mock_components['interaction_service'].deliver_learning_content.return_value = {
            "delivery_status": "success",
            "channels": ["web", "mobile"]
        }
        
        # Start learning journey
        workflow_id = await workflow_orchestrator.start_learning_journey(
            user_id=user_id,
            learning_goals=sample_learning_goals,
            preferences={"format": "interactive", "pace": "self_paced"}
        )
        
        # Wait for workflow completion
        await asyncio.sleep(0.1)  # Allow async execution
        
        # Verify workflow was created and completed
        assert workflow_id is not None
        context = await workflow_orchestrator.get_workflow_status(workflow_id)
        assert context is not None
        assert context.workflow_type == WorkflowType.LEARNING_JOURNEY
        assert context.status == WorkflowStatus.COMPLETED
        assert len(context.steps_completed) == 6  # All steps completed
        
        # Verify all components were called
        mock_components['learning_engine'].assess_skill_level.assert_called_once()
        mock_components['learning_engine'].analyze_learning_goals.assert_called_once()
        mock_components['learning_engine'].generate_learning_path.assert_called_once()
        mock_components['learning_engine'].adapt_content_for_user.assert_called_once()
        mock_components['analytics_engine'].setup_learning_analytics.assert_called_once()
        mock_components['interaction_service'].deliver_learning_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_learning_journey_with_component_failure(self, workflow_orchestrator, sample_learning_goals, mock_components):
        """Test learning journey workflow with component failure and recovery."""
        user_id = "test_user_456"
        
        # Configure mock responses with one failure
        mock_components['learning_engine'].assess_skill_level.return_value = {
            "skill_level": "beginner",
            "competencies": {"python": 0.2}
        }
        mock_components['learning_engine'].analyze_learning_goals.side_effect = [
            ComponentError("Temporary service unavailable"),
            ComponentError("Still unavailable"),
            {"validated_goals": sample_learning_goals}  # Success on third try
        ]
        
        # Start learning journey
        workflow_id = await workflow_orchestrator.start_learning_journey(
            user_id=user_id,
            learning_goals=sample_learning_goals
        )
        
        # Wait for workflow completion
        await asyncio.sleep(0.1)
        
        # Verify workflow eventually succeeded
        context = await workflow_orchestrator.get_workflow_status(workflow_id)
        assert context.status == WorkflowStatus.COMPLETED
        
        # Verify retry logic was used
        assert mock_components['learning_engine'].analyze_learning_goals.call_count == 3

    @pytest.mark.asyncio
    async def test_learning_journey_permanent_failure(self, workflow_orchestrator, sample_learning_goals, mock_components):
        """Test learning journey workflow with permanent component failure."""
        user_id = "test_user_789"
        
        # Configure mock to always fail
        mock_components['learning_engine'].assess_skill_level.side_effect = ComponentError("Permanent failure")
        
        # Start learning journey - should fail
        with pytest.raises(WorkflowError):
            await workflow_orchestrator.start_learning_journey(
                user_id=user_id,
                learning_goals=sample_learning_goals
            )


class TestDebuggingSessionWorkflow:
    """Test complete debugging session workflows."""

    @pytest.mark.asyncio
    async def test_successful_debugging_session(self, workflow_orchestrator, sample_error_context, mock_components):
        """Test a complete successful debugging session workflow."""
        user_id = "debug_user_123"
        
        # Configure mock responses
        mock_components['context_analyzer'].analyze_workspace.return_value = {
            "current_focus": "python_script",
            "technology_stack": ["python", "pandas"],
            "complexity_level": 2
        }
        mock_components['debug_assistant'].analyze_error.return_value = {
            "error_type": "AttributeError",
            "root_cause": "None value assignment",
            "complexity": "medium"
        }
        mock_components['debug_assistant'].suggest_solutions.return_value = [
            {"solution": "Add null check", "confidence": 0.9},
            {"solution": "Use try-except", "confidence": 0.7}
        ]
        mock_components['debug_assistant'].adapt_debugging_guidance.return_value = {
            "guidance_level": "intermediate",
            "step_by_step": True
        }
        mock_components['debug_assistant'].guide_through_debugging.return_value = {
            "session_id": "debug_session_123",
            "steps_completed": 5
        }
        mock_components['debug_assistant'].learn_from_success.return_value = {
            "pattern_stored": True,
            "pattern_id": "pattern_456"
        }
        
        # Start debugging session
        workflow_id = await workflow_orchestrator.start_debugging_session(
            user_id=user_id,
            error_context=sample_error_context,
            code_context={"project_type": "data_analysis"}
        )
        
        # Wait for workflow completion
        await asyncio.sleep(0.1)
        
        # Verify workflow was completed successfully
        context = await workflow_orchestrator.get_workflow_status(workflow_id)
        assert context.workflow_type == WorkflowType.DEBUGGING_SESSION
        assert context.status == WorkflowStatus.COMPLETED
        assert len(context.steps_completed) == 6  # All debugging steps completed
        
        # Verify all debugging components were called
        mock_components['context_analyzer'].analyze_workspace.assert_called_once()
        mock_components['debug_assistant'].analyze_error.assert_called_once()
        mock_components['debug_assistant'].suggest_solutions.assert_called_once()
        mock_components['debug_assistant'].adapt_debugging_guidance.assert_called_once()
        mock_components['debug_assistant'].guide_through_debugging.assert_called_once()
        mock_components['debug_assistant'].learn_from_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_debugging_session_with_skill_adaptation(self, workflow_orchestrator, sample_error_context, mock_components):
        """Test debugging session with skill level adaptation."""
        user_id = "beginner_user_123"
        
        # Configure mocks for beginner user
        mock_components['context_analyzer'].analyze_workspace.return_value = {
            "current_focus": "python_script",
            "complexity_level": 1
        }
        mock_components['debug_assistant'].analyze_error.return_value = {
            "error_type": "AttributeError",
            "root_cause": "None value assignment",
            "complexity": "low"
        }
        mock_components['debug_assistant'].suggest_solutions.return_value = [
            {"solution": "Check if variable is None", "confidence": 0.95, "difficulty": "easy"}
        ]
        mock_components['debug_assistant'].adapt_debugging_guidance.return_value = {
            "guidance_level": "beginner",
            "detailed_explanations": True,
            "code_examples": True
        }
        
        # Start debugging session
        workflow_id = await workflow_orchestrator.start_debugging_session(
            user_id=user_id,
            error_context=sample_error_context
        )
        
        await asyncio.sleep(0.1)
        
        # Verify adaptation occurred
        context = await workflow_orchestrator.get_workflow_status(workflow_id)
        assert context.status == WorkflowStatus.COMPLETED
        
        # Check that guidance was adapted for beginner
        adapt_call_args = mock_components['debug_assistant'].adapt_debugging_guidance.call_args
        assert adapt_call_args is not None


class TestAutomationExecutionWorkflow:
    """Test complete automation execution workflows."""

    @pytest.mark.asyncio
    async def test_successful_automation_execution(self, workflow_orchestrator, sample_user_actions, mock_components):
        """Test a complete successful automation execution workflow."""
        user_id = "automation_user_123"
        
        # Configure mock responses
        mock_components['pattern_detector'].detect_repetitive_patterns.return_value = [
            {
                "pattern_id": "pattern_123",
                "pattern_type": "file_save_run",
                "frequency": 5,
                "confidence": 0.9
            }
        ]
        mock_components['pattern_detector'].evaluate_automation_opportunities.return_value = [
            {
                "opportunity_id": "opp_123",
                "automation_potential": 0.8,
                "time_saving": 300,  # 5 minutes
                "complexity": "low"
            }
        ]
        mock_components['workflow_generator'].generate_automation.return_value = {
            "script_id": "script_123",
            "script_content": "#!/bin/bash\npython main.py",
            "validation_status": "pending"
        }
        mock_components['workflow_generator'].validate_automation.return_value = {
            "script_id": "script_123",
            "validation_status": "passed",
            "safety_score": 0.95
        }
        mock_components['workflow_generator'].execute_automation.return_value = {
            "execution_id": "exec_123",
            "status": "completed",
            "duration": 2.5
        }
        mock_components['analytics_engine'].monitor_automation.return_value = {
            "monitoring_id": "monitor_123",
            "metrics": {"success_rate": 1.0, "avg_duration": 2.5}
        }
        
        # Start automation execution
        workflow_id = await workflow_orchestrator.start_automation_execution(
            user_id=user_id,
            user_actions=sample_user_actions,
            automation_preferences={"risk_tolerance": "medium"}
        )
        
        # Wait for workflow completion
        await asyncio.sleep(0.1)
        
        # Verify workflow was completed successfully
        context = await workflow_orchestrator.get_workflow_status(workflow_id)
        assert context.workflow_type == WorkflowType.AUTOMATION_EXECUTION
        assert context.status == WorkflowStatus.COMPLETED
        assert len(context.steps_completed) == 6  # All automation steps completed
        
        # Verify all automation components were called
        mock_components['pattern_detector'].detect_repetitive_patterns.assert_called_once()
        mock_components['pattern_detector'].evaluate_automation_opportunities.assert_called_once()
        mock_components['workflow_generator'].generate_automation.assert_called_once()
        mock_components['workflow_generator'].validate_automation.assert_called_once()
        mock_components['workflow_generator'].execute_automation.assert_called_once()
        mock_components['analytics_engine'].monitor_automation.assert_called_once()

    @pytest.mark.asyncio
    async def test_automation_execution_with_validation_failure(self, workflow_orchestrator, sample_user_actions, mock_components):
        """Test automation execution with validation failure."""
        user_id = "automation_user_456"
        
        # Configure mocks with validation failure
        mock_components['pattern_detector'].detect_repetitive_patterns.return_value = [
            {"pattern_id": "pattern_456", "frequency": 3}
        ]
        mock_components['pattern_detector'].evaluate_automation_opportunities.return_value = [
            {"opportunity_id": "opp_456", "automation_potential": 0.6}
        ]
        mock_components['workflow_generator'].generate_automation.return_value = {
            "script_id": "script_456",
            "script_content": "rm -rf /"  # Dangerous script
        }
        mock_components['workflow_generator'].validate_automation.return_value = {
            "script_id": "script_456",
            "validation_status": "failed",
            "safety_score": 0.1,
            "issues": ["Potentially destructive command"]
        }
        
        # Start automation execution - should fail at validation
        with pytest.raises(WorkflowError):
            await workflow_orchestrator.start_automation_execution(
                user_id=user_id,
                user_actions=sample_user_actions
            )


class TestWorkflowManagement:
    """Test workflow management operations."""

    @pytest.mark.asyncio
    async def test_workflow_pause_and_resume(self, workflow_orchestrator, sample_learning_goals, mock_components):
        """Test pausing and resuming a workflow."""
        user_id = "pause_test_user"
        
        # Configure slow mock to allow pausing
        async def slow_analyze_goals(*args, **kwargs):
            await asyncio.sleep(0.2)  # Simulate slow operation
            return {"validated_goals": sample_learning_goals}
        
        mock_components['learning_engine'].assess_skill_level.return_value = {"skill_level": "beginner"}
        mock_components['learning_engine'].analyze_learning_goals.side_effect = slow_analyze_goals
        
        # Start workflow
        workflow_id = await workflow_orchestrator.start_learning_journey(
            user_id=user_id,
            learning_goals=sample_learning_goals
        )
        
        # Pause workflow quickly
        await asyncio.sleep(0.05)  # Let it start
        pause_success = await workflow_orchestrator.pause_workflow(workflow_id)
        assert pause_success
        
        # Verify paused status
        context = await workflow_orchestrator.get_workflow_status(workflow_id)
        assert context.status == WorkflowStatus.PAUSED
        
        # Resume workflow
        resume_success = await workflow_orchestrator.resume_workflow(workflow_id)
        assert resume_success
        
        # Wait for completion
        await asyncio.sleep(0.3)
        
        # Verify resumed and completed
        context = await workflow_orchestrator.get_workflow_status(workflow_id)
        assert context.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_workflow_cancellation(self, workflow_orchestrator, sample_learning_goals, mock_components):
        """Test cancelling a workflow."""
        user_id = "cancel_test_user"
        
        # Configure slow mock to allow cancellation
        async def slow_assess_skills(*args, **kwargs):
            await asyncio.sleep(0.2)
            return {"skill_level": "beginner"}
        
        mock_components['learning_engine'].assess_skill_level.side_effect = slow_assess_skills
        
        # Start workflow
        workflow_id = await workflow_orchestrator.start_learning_journey(
            user_id=user_id,
            learning_goals=sample_learning_goals
        )
        
        # Cancel workflow quickly
        await asyncio.sleep(0.05)
        cancel_success = await workflow_orchestrator.cancel_workflow(workflow_id)
        assert cancel_success
        
        # Verify cancelled status
        context = await workflow_orchestrator.get_workflow_status(workflow_id)
        assert context.status == WorkflowStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_get_active_workflows(self, workflow_orchestrator, sample_learning_goals, mock_components):
        """Test getting list of active workflows."""
        user_id = "list_test_user"
        
        # Configure mocks
        mock_components['learning_engine'].assess_skill_level.return_value = {"skill_level": "beginner"}
        mock_components['learning_engine'].analyze_learning_goals.return_value = {"validated_goals": sample_learning_goals}
        
        # Start multiple workflows
        workflow_id1 = await workflow_orchestrator.start_learning_journey(
            user_id=user_id,
            learning_goals=sample_learning_goals[:1]
        )
        workflow_id2 = await workflow_orchestrator.start_learning_journey(
            user_id=user_id,
            learning_goals=sample_learning_goals[1:]
        )
        
        # Get active workflows
        active_workflows = await workflow_orchestrator.get_active_workflows(user_id)
        
        # Verify both workflows are listed
        assert len(active_workflows) >= 2
        workflow_ids = [w.workflow_id for w in active_workflows]
        assert workflow_id1 in workflow_ids
        assert workflow_id2 in workflow_ids

    @pytest.mark.asyncio
    async def test_workflow_cleanup(self, workflow_orchestrator, sample_learning_goals, mock_components):
        """Test cleaning up completed workflows."""
        user_id = "cleanup_test_user"
        
        # Configure mocks for quick completion
        mock_components['learning_engine'].assess_skill_level.return_value = {"skill_level": "beginner"}
        mock_components['learning_engine'].analyze_learning_goals.return_value = {"validated_goals": sample_learning_goals}
        mock_components['learning_engine'].generate_learning_path.return_value = {"path_id": "path_123"}
        mock_components['learning_engine'].adapt_content_for_user.return_value = {"adapted_content": {}}
        mock_components['analytics_engine'].setup_learning_analytics.return_value = {"analytics_id": "analytics_123"}
        mock_components['interaction_service'].deliver_learning_content.return_value = {"delivery_status": "success"}
        
        # Start and complete workflow
        workflow_id = await workflow_orchestrator.start_learning_journey(
            user_id=user_id,
            learning_goals=sample_learning_goals
        )
        
        await asyncio.sleep(0.1)  # Wait for completion
        
        # Manually set old timestamp for testing
        context = workflow_orchestrator.active_workflows[workflow_id]
        context.updated_at = datetime.utcnow() - timedelta(hours=25)  # 25 hours ago
        
        # Run cleanup
        cleaned_count = await workflow_orchestrator.cleanup_completed_workflows(older_than_hours=24)
        
        # Verify workflow was cleaned up
        assert cleaned_count == 1
        assert workflow_id not in workflow_orchestrator.active_workflows


class TestWorkflowErrorHandling:
    """Test error handling in workflows."""

    @pytest.mark.asyncio
    async def test_component_timeout_handling(self, workflow_orchestrator, sample_learning_goals, mock_components):
        """Test handling of component timeouts."""
        user_id = "timeout_test_user"
        
        # Configure mock to timeout
        async def timeout_assess_skills(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate timeout
            return {"skill_level": "beginner"}
        
        mock_components['learning_engine'].assess_skill_level.side_effect = timeout_assess_skills
        
        # Start workflow - should handle timeout gracefully
        with pytest.raises(WorkflowError):
            await workflow_orchestrator.start_learning_journey(
                user_id=user_id,
                learning_goals=sample_learning_goals
            )

    @pytest.mark.asyncio
    async def test_invalid_workflow_operations(self, workflow_orchestrator):
        """Test invalid workflow operations."""
        # Test operations on non-existent workflow
        assert not await workflow_orchestrator.pause_workflow("non_existent_id")
        assert not await workflow_orchestrator.resume_workflow("non_existent_id")
        assert not await workflow_orchestrator.cancel_workflow("non_existent_id")
        
        # Test getting status of non-existent workflow
        status = await workflow_orchestrator.get_workflow_status("non_existent_id")
        assert status is None

    @pytest.mark.asyncio
    async def test_workflow_step_dependency_validation(self, workflow_orchestrator):
        """Test that workflow steps respect dependencies."""
        # This test would verify that steps with dependencies
        # are not executed until their dependencies are complete
        
        # Create a custom workflow with complex dependencies
        from src.ai_learning_accelerator.services.workflow_orchestrator import WorkflowStep
        
        steps = [
            WorkflowStep("step1", "Step 1", "First step", "component1", "action1", {}),
            WorkflowStep("step2", "Step 2", "Second step", "component2", "action2", {}, dependencies=["step1"]),
            WorkflowStep("step3", "Step 3", "Third step", "component3", "action3", {}, dependencies=["step1", "step2"])
        ]
        
        # Test dependency checking
        assert workflow_orchestrator._can_execute_step(steps[0], [])  # No dependencies
        assert not workflow_orchestrator._can_execute_step(steps[1], [])  # Missing step1
        assert workflow_orchestrator._can_execute_step(steps[1], ["step1"])  # Has step1
        assert not workflow_orchestrator._can_execute_step(steps[2], ["step1"])  # Missing step2
        assert workflow_orchestrator._can_execute_step(steps[2], ["step1", "step2"])  # Has both


@pytest.mark.integration
class TestEndToEndIntegration:
    """Integration tests for complete end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_complete_learning_and_debugging_flow(self, workflow_orchestrator, sample_learning_goals, sample_error_context, mock_components):
        """Test a complete flow involving both learning and debugging."""
        user_id = "integration_test_user"
        
        # Configure all mocks for both workflows
        # Learning journey mocks
        mock_components['learning_engine'].assess_skill_level.return_value = {"skill_level": "intermediate"}
        mock_components['learning_engine'].analyze_learning_goals.return_value = {"validated_goals": sample_learning_goals}
        mock_components['learning_engine'].generate_learning_path.return_value = {"path_id": "path_integration"}
        mock_components['learning_engine'].adapt_content_for_user.return_value = {"adapted_content": {}}
        mock_components['analytics_engine'].setup_learning_analytics.return_value = {"analytics_id": "analytics_integration"}
        mock_components['interaction_service'].deliver_learning_content.return_value = {"delivery_status": "success"}
        
        # Debugging session mocks
        mock_components['context_analyzer'].analyze_workspace.return_value = {"current_focus": "python_debugging"}
        mock_components['debug_assistant'].analyze_error.return_value = {"error_type": "AttributeError"}
        mock_components['debug_assistant'].suggest_solutions.return_value = [{"solution": "Add null check"}]
        mock_components['debug_assistant'].adapt_debugging_guidance.return_value = {"guidance_level": "intermediate"}
        mock_components['debug_assistant'].guide_through_debugging.return_value = {"session_id": "debug_integration"}
        mock_components['debug_assistant'].learn_from_success.return_value = {"pattern_stored": True}
        
        # Start learning journey
        learning_workflow_id = await workflow_orchestrator.start_learning_journey(
            user_id=user_id,
            learning_goals=sample_learning_goals
        )
        
        # Start debugging session
        debug_workflow_id = await workflow_orchestrator.start_debugging_session(
            user_id=user_id,
            error_context=sample_error_context
        )
        
        # Wait for both to complete
        await asyncio.sleep(0.2)
        
        # Verify both workflows completed successfully
        learning_context = await workflow_orchestrator.get_workflow_status(learning_workflow_id)
        debug_context = await workflow_orchestrator.get_workflow_status(debug_workflow_id)
        
        assert learning_context.status == WorkflowStatus.COMPLETED
        assert debug_context.status == WorkflowStatus.COMPLETED
        
        # Verify user has both workflows
        active_workflows = await workflow_orchestrator.get_active_workflows(user_id)
        completed_workflows = [w for w in active_workflows if w.status == WorkflowStatus.COMPLETED]
        assert len(completed_workflows) >= 2

    @pytest.mark.asyncio
    async def test_workflow_performance_under_load(self, workflow_orchestrator, sample_learning_goals, mock_components):
        """Test workflow performance with multiple concurrent workflows."""
        # Configure fast mocks
        mock_components['learning_engine'].assess_skill_level.return_value = {"skill_level": "beginner"}
        mock_components['learning_engine'].analyze_learning_goals.return_value = {"validated_goals": sample_learning_goals}
        mock_components['learning_engine'].generate_learning_path.return_value = {"path_id": "path_load_test"}
        mock_components['learning_engine'].adapt_content_for_user.return_value = {"adapted_content": {}}
        mock_components['analytics_engine'].setup_learning_analytics.return_value = {"analytics_id": "analytics_load_test"}
        mock_components['interaction_service'].deliver_learning_content.return_value = {"delivery_status": "success"}
        
        # Start multiple concurrent workflows
        workflow_ids = []
        start_time = datetime.utcnow()
        
        for i in range(10):  # Start 10 concurrent workflows
            workflow_id = await workflow_orchestrator.start_learning_journey(
                user_id=f"load_test_user_{i}",
                learning_goals=sample_learning_goals
            )
            workflow_ids.append(workflow_id)
        
        # Wait for all to complete
        await asyncio.sleep(0.5)
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        # Verify all workflows completed
        completed_count = 0
        for workflow_id in workflow_ids:
            context = await workflow_orchestrator.get_workflow_status(workflow_id)
            if context and context.status == WorkflowStatus.COMPLETED:
                completed_count += 1
        
        assert completed_count == 10
        assert total_time < 2.0  # Should complete within 2 seconds
        
        # Verify performance metrics
        print(f"Completed {completed_count} workflows in {total_time:.2f} seconds")
        print(f"Average time per workflow: {total_time / completed_count:.3f} seconds")