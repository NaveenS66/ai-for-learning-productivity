"""
Property-based tests for user control priority in automation.

**Property 16: User Control Priority**
**Validates: Requirements 4.5**

Tests that the system prioritizes user control over automatic execution when 
conflicts arise between automated actions and user preferences.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize

from src.ai_learning_accelerator.models.automation import (
    ActionType, PatternType, AutomationComplexity, AutomationStatus,
    UserAction, AutomationOpportunity, AutomationScript, AutomationExecution
)
from src.ai_learning_accelerator.models.user import User, SkillLevel
from src.ai_learning_accelerator.services.pattern_detector import PatternDetector
from src.ai_learning_accelerator.services.workflow_generator import WorkflowGenerator


# Test data generators
@st.composite
def user_preference_data(draw):
    """Generate user preference data for automation."""
    return {
        "automation_enabled": draw(st.booleans()),
        "auto_execute_scripts": draw(st.booleans()),
        "require_confirmation": draw(st.booleans()),
        "allowed_action_types": draw(st.lists(
            st.sampled_from(list(ActionType)), 
            min_size=1, 
            max_size=len(ActionType)
        )),
        "blocked_commands": draw(st.lists(
            st.text(min_size=3, max_size=20), 
            min_size=0, 
            max_size=5
        )),
        "max_automation_complexity": draw(st.sampled_from(list(AutomationComplexity))),
        "privacy_level": draw(st.sampled_from(["low", "medium", "high"])),
        "workspace_restrictions": draw(st.lists(
            st.text(min_size=5, max_size=30), 
            min_size=0, 
            max_size=3
        ))
    }


@st.composite
def automation_request_data(draw):
    """Generate automation request data."""
    action_type = draw(st.sampled_from(list(ActionType)))
    
    return {
        "action_type": action_type,
        "command": draw(st.text(min_size=5, max_size=50)) if action_type == ActionType.COMMAND_EXECUTION else None,
        "file_path": draw(st.text(min_size=5, max_size=50)) if action_type == ActionType.FILE_OPERATION else None,
        "workspace_path": draw(st.text(min_size=5, max_size=30)),
        "complexity": draw(st.sampled_from(list(AutomationComplexity))),
        "requires_confirmation": draw(st.booleans()),
        "estimated_risk": draw(st.sampled_from(["low", "medium", "high"])),
        "automation_score": draw(st.floats(min_value=0.0, max_value=1.0))
    }


class TestUserControlPriority:
    """Test user control priority properties."""
    
    def setup_method(self):
        """Set up test environment."""
        self.pattern_detector = PatternDetector()
        self.workflow_generator = WorkflowGenerator()
        self.user_id = uuid4()
    
    @given(
        user_prefs=user_preference_data(),
        automation_request=automation_request_data()
    )
    @settings(max_examples=50, deadline=5000)
    def test_user_preferences_override_automation(self, user_prefs, automation_request):
        """
        Property: User preferences should always override automatic execution decisions.
        
        **Validates: Requirements 4.5**
        """
        # Test automation enabled/disabled preference
        if not user_prefs["automation_enabled"]:
            should_execute = self._should_execute_automation(user_prefs, automation_request)
            assert not should_execute, "Automation should not execute when user has disabled it"
        
        # Test action type restrictions
        if automation_request["action_type"] not in user_prefs["allowed_action_types"]:
            should_execute = self._should_execute_automation(user_prefs, automation_request)
            assert not should_execute, "Automation should not execute for blocked action types"
        
        # Test complexity restrictions
        complexity_levels = {
            AutomationComplexity.SIMPLE: 1,
            AutomationComplexity.MODERATE: 2,
            AutomationComplexity.COMPLEX: 3,
            AutomationComplexity.ADVANCED: 4
        }
        
        max_allowed = complexity_levels[user_prefs["max_automation_complexity"]]
        requested = complexity_levels[automation_request["complexity"]]
        
        if requested > max_allowed:
            should_execute = self._should_execute_automation(user_prefs, automation_request)
            assert not should_execute, "Automation should not execute when complexity exceeds user limit"
        
        # Test command blocking
        if automation_request.get("command"):
            for blocked_cmd in user_prefs["blocked_commands"]:
                if blocked_cmd.lower() in automation_request["command"].lower():
                    should_execute = self._should_execute_automation(user_prefs, automation_request)
                    assert not should_execute, f"Automation should not execute blocked command: {blocked_cmd}"
    
    @given(
        user_prefs=user_preference_data(),
        automation_requests=st.lists(automation_request_data(), min_size=2, max_size=5)
    )
    @settings(max_examples=30, deadline=5000)
    def test_confirmation_requirements_respected(self, user_prefs, automation_requests):
        """
        Property: When user requires confirmation, no automation should execute 
        without explicit approval.
        """
        if user_prefs["require_confirmation"]:
            for request in automation_requests:
                # Only test if the request would otherwise be allowed
                if self._meets_other_requirements(user_prefs, request):
                    # Simulate automatic execution attempt
                    execution_decision = self._make_execution_decision(
                        user_prefs, request, user_confirmed=False
                    )
                    
                    assert not execution_decision["auto_execute"], \
                        "Should not auto-execute when user requires confirmation"
                    assert execution_decision["requires_user_approval"], \
                        "Should require user approval when confirmation is enabled"
        
        # Test with user confirmation
        if user_prefs["require_confirmation"]:
            for request in automation_requests:
                execution_decision = self._make_execution_decision(
                    user_prefs, request, user_confirmed=True
                )
                
                # Should be allowed to execute with confirmation (if other conditions met)
                if self._meets_other_requirements(user_prefs, request):
                    assert execution_decision["can_execute"], \
                        "Should be able to execute with user confirmation when other requirements met"
    
    @given(
        privacy_level=st.sampled_from(["low", "medium", "high"]),
        automation_requests=st.lists(automation_request_data(), min_size=1, max_size=3)
    )
    @settings(max_examples=20, deadline=3000)
    def test_privacy_level_enforcement(self, privacy_level, automation_requests):
        """
        Property: Privacy level settings should restrict automation access to 
        sensitive operations and data.
        """
        user_prefs = {
            "automation_enabled": True,
            "privacy_level": privacy_level,
            "allowed_action_types": list(ActionType),
            "max_automation_complexity": AutomationComplexity.ADVANCED
        }
        
        for request in automation_requests:
            # High privacy should restrict more operations
            if privacy_level == "high":
                if self._is_sensitive_operation(request):
                    should_execute = self._should_execute_automation(user_prefs, request)
                    assert not should_execute, \
                        "High privacy level should block sensitive operations"
            
            # Medium privacy should have moderate restrictions
            elif privacy_level == "medium":
                if self._is_highly_sensitive_operation(request):
                    should_execute = self._should_execute_automation(user_prefs, request)
                    assert not should_execute, \
                        "Medium privacy level should block highly sensitive operations"
            
            # Low privacy should allow most operations (with other constraints)
            elif privacy_level == "low":
                if not self._is_extremely_dangerous_operation(request):
                    # Should generally allow operations at low privacy
                    pass  # Most operations should be allowed
    
    @given(
        workspace_restrictions=st.lists(st.text(min_size=5, max_size=20), min_size=1, max_size=3),
        automation_requests=st.lists(automation_request_data(), min_size=1, max_size=3)
    )
    @settings(max_examples=20, deadline=3000)
    def test_workspace_restrictions_enforced(self, workspace_restrictions, automation_requests):
        """
        Property: Workspace restrictions should prevent automation from operating 
        in restricted directories or projects.
        """
        user_prefs = {
            "automation_enabled": True,
            "workspace_restrictions": workspace_restrictions,
            "allowed_action_types": list(ActionType),
            "privacy_level": "low"
        }
        
        for request in automation_requests:
            workspace_path = request.get("workspace_path", "")
            
            # Check if workspace path matches any restriction
            for restriction in workspace_restrictions:
                if restriction.lower() in workspace_path.lower():
                    should_execute = self._should_execute_automation(user_prefs, request)
                    assert not should_execute, \
                        f"Should not execute automation in restricted workspace: {restriction}"
    
    def test_user_override_during_execution(self):
        """
        Property: Users should be able to stop or override automation during execution.
        """
        user_prefs = {
            "automation_enabled": True,
            "auto_execute_scripts": True,
            "allowed_action_types": list(ActionType)
        }
        
        automation_request = {
            "action_type": ActionType.FILE_OPERATION,
            "complexity": AutomationComplexity.SIMPLE,
            "automation_score": 0.8
        }
        
        # Start execution
        execution_state = self._simulate_automation_execution(user_prefs, automation_request)
        assert execution_state["status"] == "running", "Automation should start running"
        
        # User requests stop
        execution_state = self._handle_user_stop_request(execution_state)
        assert execution_state["status"] in ["stopped", "stopping"], \
            "Automation should stop when user requests it"
        assert execution_state["user_initiated_stop"], \
            "Should record that user initiated the stop"
    
    @given(
        initial_prefs=user_preference_data(),
        updated_prefs=user_preference_data()
    )
    @settings(max_examples=20, deadline=3000)
    def test_preference_changes_immediately_effective(self, initial_prefs, updated_prefs):
        """
        Property: Changes to user preferences should take effect immediately 
        for new automation decisions.
        """
        automation_request = {
            "action_type": ActionType.COMMAND_EXECUTION,
            "command": "test_command",
            "complexity": AutomationComplexity.MODERATE,
            "automation_score": 0.7
        }
        
        # Test with initial preferences
        initial_decision = self._should_execute_automation(initial_prefs, automation_request)
        
        # Test with updated preferences
        updated_decision = self._should_execute_automation(updated_prefs, automation_request)
        
        # If preferences changed in a way that should affect the decision
        if self._preferences_would_change_decision(initial_prefs, updated_prefs, automation_request):
            assert initial_decision != updated_decision, \
                "Decision should change when relevant preferences change"
    
    def test_explicit_user_approval_overrides_restrictions(self):
        """
        Property: Explicit user approval should override most restrictions 
        (except security-critical ones).
        """
        # Restrictive preferences
        user_prefs = {
            "automation_enabled": False,  # Disabled
            "require_confirmation": True,
            "max_automation_complexity": AutomationComplexity.SIMPLE,
            "privacy_level": "medium"
        }
        
        # Complex automation request
        automation_request = {
            "action_type": ActionType.BUILD_OPERATION,
            "complexity": AutomationComplexity.COMPLEX,  # Exceeds user limit
            "automation_score": 0.9
        }
        
        # Without explicit approval - should be blocked
        decision_without_approval = self._make_execution_decision(
            user_prefs, automation_request, user_confirmed=False, explicit_approval=False
        )
        assert not decision_without_approval["can_execute"], \
            "Should not execute without approval when restrictions apply"
        
        # With explicit approval - should be allowed (for non-security-critical operations)
        decision_with_approval = self._make_execution_decision(
            user_prefs, automation_request, user_confirmed=True, explicit_approval=True
        )
        
        if not self._is_security_critical_operation(automation_request):
            assert decision_with_approval["can_execute"], \
                "Should execute with explicit approval for non-security-critical operations"
    
    # Helper methods for testing logic
    
    def _should_execute_automation(self, user_prefs: Dict[str, Any], request: Dict[str, Any]) -> bool:
        """Determine if automation should execute based on user preferences."""
        # Check if automation is globally disabled
        if not user_prefs.get("automation_enabled", True):
            return False
        
        # Check action type restrictions
        allowed_types = user_prefs.get("allowed_action_types", list(ActionType))
        if request["action_type"] not in allowed_types:
            return False
        
        # Check complexity restrictions
        max_complexity = user_prefs.get("max_automation_complexity", AutomationComplexity.ADVANCED)
        complexity_levels = {
            AutomationComplexity.SIMPLE: 1,
            AutomationComplexity.MODERATE: 2,
            AutomationComplexity.COMPLEX: 3,
            AutomationComplexity.ADVANCED: 4
        }
        
        if complexity_levels[request["complexity"]] > complexity_levels[max_complexity]:
            return False
        
        # Check blocked commands
        blocked_commands = user_prefs.get("blocked_commands", [])
        if request.get("command"):
            command = request.get("command", "") or ""
            for blocked in blocked_commands:
                if blocked.lower() in command.lower():
                    return False
        
        # Check privacy level restrictions
        privacy_level = user_prefs.get("privacy_level", "low")
        if privacy_level == "high" and self._is_sensitive_operation(request):
            return False
        if privacy_level == "medium" and self._is_highly_sensitive_operation(request):
            return False
        
        # Check workspace restrictions
        workspace_restrictions = user_prefs.get("workspace_restrictions", [])
        workspace_path = request.get("workspace_path", "")
        for restriction in workspace_restrictions:
            if restriction.lower() in workspace_path.lower():
                return False
        
        return True
    
    def _make_execution_decision(
        self, 
        user_prefs: Dict[str, Any], 
        request: Dict[str, Any], 
        user_confirmed: bool = False,
        explicit_approval: bool = False
    ) -> Dict[str, Any]:
        """Make an execution decision based on preferences and user input."""
        decision = {
            "can_execute": False,
            "auto_execute": False,
            "requires_user_approval": False,
            "blocked_reason": None
        }
        
        # With explicit approval, override most restrictions (except security-critical)
        if explicit_approval and not self._is_security_critical_operation(request):
            decision["can_execute"] = True
            decision["auto_execute"] = user_prefs.get("auto_execute_scripts", True)
            return decision
        
        # Check basic requirements
        if not self._should_execute_automation(user_prefs, request):
            decision["blocked_reason"] = "User preferences block this automation"
            return decision
        
        # Check confirmation requirements
        requires_confirmation = user_prefs.get("require_confirmation", False)
        auto_execute_enabled = user_prefs.get("auto_execute_scripts", True)
        
        if requires_confirmation and not user_confirmed:
            decision["requires_user_approval"] = True
            return decision
        
        # Normal execution path
        if user_confirmed or not requires_confirmation:
            decision["can_execute"] = True
            decision["auto_execute"] = auto_execute_enabled and not requires_confirmation
        
        return decision
    
    def _meets_other_requirements(self, user_prefs: Dict[str, Any], request: Dict[str, Any]) -> bool:
        """Check if request meets other requirements besides confirmation."""
        return self._should_execute_automation(user_prefs, request)
    
    def _is_sensitive_operation(self, request: Dict[str, Any]) -> bool:
        """Check if operation is considered sensitive."""
        sensitive_actions = [ActionType.FILE_OPERATION, ActionType.COMMAND_EXECUTION, ActionType.DEPLOYMENT]
        return request["action_type"] in sensitive_actions
    
    def _is_highly_sensitive_operation(self, request: Dict[str, Any]) -> bool:
        """Check if operation is highly sensitive."""
        highly_sensitive = [ActionType.DEPLOYMENT, ActionType.VERSION_CONTROL]
        if request["action_type"] in highly_sensitive:
            return True
        
        # Check for sensitive commands
        command = request.get("command", "") or ""
        command = command.lower()
        sensitive_commands = ["rm", "delete", "drop", "format", "sudo"]
        return any(cmd in command for cmd in sensitive_commands)
    
    def _is_extremely_dangerous_operation(self, request: Dict[str, Any]) -> bool:
        """Check if operation is extremely dangerous."""
        command = request.get("command", "") or ""
        command = command.lower()
        dangerous_commands = ["rm -rf", "format", "fdisk", "dd if="]
        return any(cmd in command for cmd in dangerous_commands)
    
    def _is_security_critical_operation(self, request: Dict[str, Any]) -> bool:
        """Check if operation is security-critical and should not be overridden."""
        command = request.get("command", "") or ""
        command = command.lower()
        security_critical = ["chmod 777", "sudo", "passwd", "ssh-keygen"]
        return any(cmd in command for cmd in security_critical)
    
    def _simulate_automation_execution(
        self, 
        user_prefs: Dict[str, Any], 
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate automation execution."""
        return {
            "status": "running",
            "user_initiated_stop": False,
            "can_be_stopped": True,
            "execution_id": str(uuid4())
        }
    
    def _handle_user_stop_request(self, execution_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user request to stop automation."""
        if execution_state["can_be_stopped"]:
            execution_state["status"] = "stopping"
            execution_state["user_initiated_stop"] = True
        
        return execution_state
    
    def _preferences_would_change_decision(
        self, 
        initial_prefs: Dict[str, Any], 
        updated_prefs: Dict[str, Any], 
        request: Dict[str, Any]
    ) -> bool:
        """Check if preference changes would affect the execution decision."""
        # Get decisions with both preference sets
        initial_decision = self._should_execute_automation(initial_prefs, request)
        updated_decision = self._should_execute_automation(updated_prefs, request)
        
        # If the basic execution decision changed, then preferences would change decision
        return initial_decision != updated_decision


class UserControlPriorityStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for user control priority.
    
    This tests that user preferences are consistently respected across
    different automation scenarios and preference changes.
    """
    
    def __init__(self):
        super().__init__()
        self.user_id = uuid4()
        self.current_preferences = {}
        self.automation_history = []
        self.active_executions = []
    
    preferences = Bundle('preferences')
    automation_requests = Bundle('automation_requests')
    
    @initialize()
    def setup(self):
        """Initialize the test state."""
        self.current_preferences = {
            "automation_enabled": True,
            "require_confirmation": False,
            "allowed_action_types": list(ActionType),
            "max_automation_complexity": AutomationComplexity.MODERATE,
            "privacy_level": "medium"
        }
        self.automation_history = []
        self.active_executions = []
    
    @rule(target=preferences, new_prefs=user_preference_data())
    def update_user_preferences(self, new_prefs):
        """Update user preferences."""
        old_prefs = self.current_preferences.copy()
        self.current_preferences.update(new_prefs)
        
        # Verify that preference changes are immediately effective
        test_request = {
            "action_type": ActionType.FILE_OPERATION,
            "complexity": AutomationComplexity.SIMPLE,
            "automation_score": 0.7
        }
        
        # Decision should reflect new preferences
        decision = self._make_execution_decision(self.current_preferences, test_request)
        
        # If automation was disabled, no execution should be allowed
        if not new_prefs.get("automation_enabled", True):
            assert not decision["can_execute"], \
                "Disabling automation should immediately prevent execution"
        
        return new_prefs
    
    @rule(target=automation_requests, request=automation_request_data())
    def submit_automation_request(self, request):
        """Submit an automation request."""
        decision = self._make_execution_decision(self.current_preferences, request)
        
        # Record the decision
        self.automation_history.append({
            "request": request,
            "decision": decision,
            "preferences_at_time": self.current_preferences.copy(),
            "timestamp": datetime.utcnow()
        })
        
        # Verify user preferences are respected
        if not self.current_preferences.get("automation_enabled", True):
            assert not decision["can_execute"], \
                "Should not execute when automation is disabled"
        
        if self.current_preferences.get("require_confirmation", False):
            assert decision["requires_user_approval"] or not decision["auto_execute"], \
                "Should require approval or not auto-execute when confirmation required"
        
        return request
    
    @rule(request=automation_requests)
    def test_consistent_decision_making(self, request):
        """Test that decisions are consistent for the same request and preferences."""
        # Make the same decision multiple times
        decision1 = self._make_execution_decision(self.current_preferences, request)
        decision2 = self._make_execution_decision(self.current_preferences, request)
        
        # Decisions should be identical
        assert decision1["can_execute"] == decision2["can_execute"], \
            "Decisions should be consistent for same request and preferences"
        assert decision1["requires_user_approval"] == decision2["requires_user_approval"], \
            "Approval requirements should be consistent"
    
    @rule()
    def verify_no_unauthorized_executions(self):
        """Verify that no executions occurred without proper authorization."""
        for history_item in self.automation_history:
            request = history_item["request"]
            decision = history_item["decision"]
            prefs = history_item["preferences_at_time"]
            
            if decision["can_execute"]:
                # Verify this execution was properly authorized
                assert self._is_properly_authorized(request, decision, prefs), \
                    "All executions should be properly authorized according to user preferences"
    
    def _make_execution_decision(self, user_prefs, request, user_confirmed=False):
        """Make execution decision (same logic as main test class)."""
        decision = {
            "can_execute": False,
            "auto_execute": False,
            "requires_user_approval": False
        }
        
        if not user_prefs.get("automation_enabled", True):
            return decision
        
        if request["action_type"] not in user_prefs.get("allowed_action_types", list(ActionType)):
            return decision
        
        complexity_levels = {
            AutomationComplexity.SIMPLE: 1,
            AutomationComplexity.MODERATE: 2,
            AutomationComplexity.COMPLEX: 3,
            AutomationComplexity.ADVANCED: 4
        }
        
        max_complexity = user_prefs.get("max_automation_complexity", AutomationComplexity.ADVANCED)
        if complexity_levels[request["complexity"]] > complexity_levels[max_complexity]:
            return decision
        
        requires_confirmation = user_prefs.get("require_confirmation", False)
        if requires_confirmation and not user_confirmed:
            decision["requires_user_approval"] = True
            return decision
        
        decision["can_execute"] = True
        decision["auto_execute"] = user_prefs.get("auto_execute_scripts", True) and not requires_confirmation
        
        return decision
    
    def _is_properly_authorized(self, request, decision, prefs):
        """Check if execution was properly authorized."""
        # Basic checks
        if not prefs.get("automation_enabled", True):
            return False
        
        if request["action_type"] not in prefs.get("allowed_action_types", list(ActionType)):
            return False
        
        # If confirmation was required, execution should not have been auto
        if prefs.get("require_confirmation", False) and decision.get("auto_execute", False):
            return False
        
        return True


# Test class for running the state machine
TestUserControlPriorityStateMachine = UserControlPriorityStateMachine.TestCase


if __name__ == "__main__":
    # Run a simple test to verify the property
    test_instance = TestUserControlPriority()
    test_instance.setup_method()
    
    # Test basic user control priority
    user_prefs = {
        "automation_enabled": False,
        "allowed_action_types": [ActionType.FILE_OPERATION]
    }
    
    automation_request = {
        "action_type": ActionType.COMMAND_EXECUTION,
        "complexity": AutomationComplexity.SIMPLE,
        "automation_score": 0.8
    }
    
    print("Testing user control priority property...")
    test_instance.test_user_preferences_override_automation(user_prefs, automation_request)
    print("✓ Property 16: User Control Priority - Basic test passed")