"""
Property-based tests for automation pattern detection.

**Property 13: Automation Pattern Detection**
**Validates: Requirements 4.1**

Tests that the system can detect repetitive patterns in user actions and suggest 
appropriate automation opportunities.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import uuid4

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize

from src.ai_learning_accelerator.models.automation import (
    ActionType, PatternType, AutomationComplexity, AutomationStatus
)
from src.ai_learning_accelerator.services.pattern_detector import PatternDetector


# Test data generators
@st.composite
def user_action_data(draw):
    """Generate realistic user action data."""
    action_type = draw(st.sampled_from(list(ActionType)))
    
    # Generate action names based on type
    action_names = {
        ActionType.FILE_OPERATION: ["create_file", "delete_file", "copy_file", "move_file"],
        ActionType.CODE_EDIT: ["add_function", "refactor_method", "fix_bug", "add_comment"],
        ActionType.COMMAND_EXECUTION: ["run_tests", "build_project", "deploy_app", "start_server"],
        ActionType.BUILD_OPERATION: ["compile", "package", "install_deps", "clean"],
        ActionType.TEST_EXECUTION: ["unit_test", "integration_test", "e2e_test", "performance_test"],
        ActionType.VERSION_CONTROL: ["git_commit", "git_push", "git_pull", "git_merge"],
    }
    
    action_name = draw(st.sampled_from(action_names.get(action_type, ["generic_action"])))
    
    return {
        "type": action_type.value,
        "name": action_name,
        "data": {
            "file_path": draw(st.text(min_size=5, max_size=50)) if action_type == ActionType.FILE_OPERATION else None,
            "command": draw(st.text(min_size=3, max_size=30)) if action_type == ActionType.COMMAND_EXECUTION else None,
            "workspace": draw(st.text(min_size=5, max_size=20)),
            "timestamp": datetime.utcnow().isoformat()
        }
    }


@st.composite
def repetitive_action_sequence(draw, min_repetitions=3, max_repetitions=10):
    """Generate a sequence of repetitive actions."""
    base_action = draw(user_action_data())
    repetitions = draw(st.integers(min_value=min_repetitions, max_value=max_repetitions))
    
    sequence = []
    for i in range(repetitions):
        action = base_action.copy()
        action["data"] = base_action["data"].copy()
        # Add slight variations to make it realistic
        action["data"]["sequence_number"] = i
        action["data"]["timestamp"] = (datetime.utcnow() + timedelta(minutes=i)).isoformat()
        sequence.append(action)
    
    return sequence


@st.composite
def mixed_action_sequence(draw):
    """Generate a mixed sequence with some repetitive patterns."""
    sequences = []
    
    # Add 1-3 repetitive patterns
    num_patterns = draw(st.integers(min_value=1, max_value=3))
    for _ in range(num_patterns):
        repetitive_seq = draw(repetitive_action_sequence())
        sequences.extend(repetitive_seq)
    
    # Add some random actions
    num_random = draw(st.integers(min_value=2, max_value=8))
    for _ in range(num_random):
        random_action = draw(user_action_data())
        sequences.append(random_action)
    
    # Shuffle to make it more realistic
    import random
    random.shuffle(sequences)
    return sequences


class TestAutomationPatternDetection:
    """Test automation pattern detection properties."""
    
    def setup_method(self):
        """Set up test environment."""
        self.pattern_detector = PatternDetector()
        self.user_id = uuid4()
    
    @given(repetitive_sequence=repetitive_action_sequence())
    @settings(max_examples=50, deadline=5000)
    def test_detects_repetitive_patterns(self, repetitive_sequence):
        """
        Property: For any sequence of repetitive user actions, the system should 
        detect the pattern and suggest appropriate automation opportunities.
        
        **Validates: Requirements 4.1**
        """
        assume(len(repetitive_sequence) >= 3)  # Need at least 3 repetitions
        
        # Convert to the format expected by pattern detector
        user_actions = []
        for i, action in enumerate(repetitive_sequence):
            user_actions.append({
                "id": str(uuid4()),
                "user_id": str(self.user_id),
                "action_type": action["type"],
                "action_name": action["name"],
                "action_data": action["data"],
                "timestamp": datetime.utcnow() + timedelta(minutes=i),
                "sequence_number": i
            })
        
        # Detect patterns
        patterns = self.pattern_detector._detect_patterns_in_actions(user_actions)
        
        # Assertions
        assert len(patterns) > 0, "Should detect at least one pattern in repetitive sequence"
        
        # Check that detected patterns have reasonable properties
        for pattern in patterns:
            assert pattern["frequency"] >= 3, "Pattern frequency should be at least 3"
            assert 0.0 <= pattern["confidence_score"] <= 1.0, "Confidence score should be between 0 and 1"
            assert pattern["pattern_type"] in [pt.value for pt in PatternType], "Should have valid pattern type"
            assert len(pattern["action_sequence"]) > 0, "Pattern should have action sequence"
    
    @given(mixed_sequence=mixed_action_sequence())
    @settings(max_examples=30, deadline=5000)
    def test_pattern_confidence_correlates_with_repetition(self, mixed_sequence):
        """
        Property: Pattern confidence should correlate with the frequency and 
        consistency of repetitive actions.
        """
        assume(len(mixed_sequence) >= 5)
        
        # Convert to user actions format
        user_actions = []
        for i, action in enumerate(mixed_sequence):
            user_actions.append({
                "id": str(uuid4()),
                "user_id": str(self.user_id),
                "action_type": action["type"],
                "action_name": action["name"],
                "action_data": action["data"],
                "timestamp": datetime.utcnow() + timedelta(minutes=i),
                "sequence_number": i
            })
        
        patterns = self.pattern_detector._detect_patterns_in_actions(user_actions)
        
        if len(patterns) > 1:
            # Sort patterns by frequency
            sorted_patterns = sorted(patterns, key=lambda p: p["frequency"], reverse=True)
            
            # Higher frequency patterns should generally have higher confidence
            for i in range(len(sorted_patterns) - 1):
                current_pattern = sorted_patterns[i]
                next_pattern = sorted_patterns[i + 1]
                
                if current_pattern["frequency"] > next_pattern["frequency"]:
                    # Allow some tolerance for confidence scores
                    assert (current_pattern["confidence_score"] >= next_pattern["confidence_score"] - 0.1), \
                        "Higher frequency patterns should have higher or similar confidence"
    
    @given(actions=st.lists(user_action_data(), min_size=3, max_size=15))
    @settings(max_examples=30, deadline=5000)
    def test_automation_opportunities_have_valid_properties(self, actions):
        """
        Property: Generated automation opportunities should have valid and 
        consistent properties.
        """
        # Convert to user actions format
        user_actions = []
        for i, action in enumerate(actions):
            user_actions.append({
                "id": str(uuid4()),
                "user_id": str(self.user_id),
                "action_type": action["type"],
                "action_name": action["name"],
                "action_data": action["data"],
                "timestamp": datetime.utcnow() + timedelta(minutes=i),
                "sequence_number": i
            })
        
        patterns = self.pattern_detector._detect_patterns_in_actions(user_actions)
        
        for pattern in patterns:
            opportunities = self.pattern_detector._generate_automation_opportunities_from_pattern(
                pattern, self.user_id
            )
            
            for opportunity in opportunities:
                # Validate opportunity properties
                assert 0.0 <= opportunity["automation_score"] <= 1.0, \
                    "Automation score should be between 0 and 1"
                assert opportunity["complexity"] in [c.value for c in AutomationComplexity], \
                    "Should have valid complexity level"
                assert opportunity["time_saving_potential"] >= 0, \
                    "Time saving potential should be non-negative"
                assert opportunity["frequency_per_week"] >= 0, \
                    "Frequency per week should be non-negative"
                assert opportunity["priority_score"] >= 0, \
                    "Priority score should be non-negative"
                assert len(opportunity["title"]) > 0, \
                    "Opportunity should have a title"
                assert len(opportunity["description"]) > 0, \
                    "Opportunity should have a description"
    
    @given(
        action_type=st.sampled_from(list(ActionType)),
        frequency=st.integers(min_value=3, max_value=20)
    )
    @settings(max_examples=20, deadline=3000)
    def test_pattern_type_classification(self, action_type, frequency):
        """
        Property: Pattern type classification should be consistent and appropriate 
        for the type of actions.
        """
        # Create a sequence of similar actions
        actions = []
        base_action = {
            "type": action_type.value,
            "name": f"test_{action_type.value}",
            "data": {"workspace": "test_workspace"}
        }
        
        for i in range(frequency):
            action = base_action.copy()
            action["data"] = base_action["data"].copy()
            action["data"]["sequence_number"] = i
            actions.append(action)
        
        # Convert to user actions format
        user_actions = []
        for i, action in enumerate(actions):
            user_actions.append({
                "id": str(uuid4()),
                "user_id": str(self.user_id),
                "action_type": action["type"],
                "action_name": action["name"],
                "action_data": action["data"],
                "timestamp": datetime.utcnow() + timedelta(minutes=i),
                "sequence_number": i
            })
        
        patterns = self.pattern_detector._detect_patterns_in_actions(user_actions)
        
        if patterns:
            pattern = patterns[0]  # Take the first detected pattern
            
            # Pattern type should be appropriate for the action sequence
            if frequency >= 5 and all(a["type"] == action_type.value for a in actions):
                assert pattern["pattern_type"] in [
                    PatternType.REPETITIVE.value,
                    PatternType.SEQUENTIAL.value
                ], "Highly repetitive actions should be classified as repetitive or sequential"
    
    def test_no_false_positives_for_random_actions(self):
        """
        Property: The system should not detect patterns in truly random, 
        non-repetitive action sequences.
        """
        # Create a sequence of completely different actions
        random_actions = [
            {
                "id": str(uuid4()),
                "user_id": str(self.user_id),
                "action_type": ActionType.FILE_OPERATION.value,
                "action_name": "create_unique_file_1",
                "action_data": {"file_path": "/unique/path/1.txt"},
                "timestamp": datetime.utcnow(),
                "sequence_number": 0
            },
            {
                "id": str(uuid4()),
                "user_id": str(self.user_id),
                "action_type": ActionType.COMMAND_EXECUTION.value,
                "action_name": "run_different_command",
                "action_data": {"command": "unique_command_xyz"},
                "timestamp": datetime.utcnow() + timedelta(minutes=1),
                "sequence_number": 1
            },
            {
                "id": str(uuid4()),
                "user_id": str(self.user_id),
                "action_type": ActionType.VERSION_CONTROL.value,
                "action_name": "git_special_operation",
                "action_data": {"operation": "unique_git_op"},
                "timestamp": datetime.utcnow() + timedelta(minutes=2),
                "sequence_number": 2
            }
        ]
        
        patterns = self.pattern_detector._detect_patterns_in_actions(random_actions)
        
        # Should not detect strong patterns in random actions
        high_confidence_patterns = [p for p in patterns if p["confidence_score"] > 0.7]
        assert len(high_confidence_patterns) == 0, \
            "Should not detect high-confidence patterns in random actions"
    
    @given(
        base_actions=st.lists(user_action_data(), min_size=2, max_size=5),
        repetitions=st.integers(min_value=3, max_value=8)
    )
    @settings(max_examples=20, deadline=5000)
    def test_pattern_detection_scalability(self, base_actions, repetitions):
        """
        Property: Pattern detection should work efficiently regardless of 
        the number of actions or complexity of patterns.
        """
        # Create a large sequence with multiple patterns
        all_actions = []
        action_id = 0
        
        for base_action in base_actions:
            for rep in range(repetitions):
                action = {
                    "id": str(uuid4()),
                    "user_id": str(self.user_id),
                    "action_type": base_action["type"],
                    "action_name": base_action["name"],
                    "action_data": base_action["data"].copy(),
                    "timestamp": datetime.utcnow() + timedelta(minutes=action_id),
                    "sequence_number": action_id
                }
                action["action_data"]["repetition"] = rep
                all_actions.append(action)
                action_id += 1
        
        # Pattern detection should complete without errors
        patterns = self.pattern_detector._detect_patterns_in_actions(all_actions)
        
        # Should detect patterns proportional to the input
        expected_patterns = len(base_actions)  # At least one pattern per base action type
        assert len(patterns) >= min(expected_patterns, 1), \
            "Should detect at least some patterns in structured repetitive data"
        
        # All detected patterns should be valid
        for pattern in patterns:
            assert isinstance(pattern["frequency"], int) and pattern["frequency"] > 0
            assert isinstance(pattern["confidence_score"], (int, float))
            assert 0.0 <= pattern["confidence_score"] <= 1.0


class AutomationPatternDetectionStateMachine(RuleBasedStateMachine):
    """
    Stateful property-based testing for automation pattern detection.
    
    This tests the pattern detection system through a series of user actions
    and verifies that patterns are detected consistently over time.
    """
    
    def __init__(self):
        super().__init__()
        self.pattern_detector = PatternDetector()
        self.user_id = uuid4()
        self.actions_history = []
        self.detected_patterns = []
    
    actions = Bundle('actions')
    
    @initialize()
    def setup(self):
        """Initialize the test state."""
        self.actions_history = []
        self.detected_patterns = []
    
    @rule(target=actions, action_data=user_action_data())
    def add_user_action(self, action_data):
        """Add a user action to the history."""
        action = {
            "id": str(uuid4()),
            "user_id": str(self.user_id),
            "action_type": action_data["type"],
            "action_name": action_data["name"],
            "action_data": action_data["data"],
            "timestamp": datetime.utcnow() + timedelta(minutes=len(self.actions_history)),
            "sequence_number": len(self.actions_history)
        }
        
        self.actions_history.append(action)
        return action
    
    @rule(actions=st.lists(st.just(actions), min_size=1, max_size=5))
    def detect_patterns(self, actions):
        """Detect patterns in the current action history."""
        if len(self.actions_history) >= 3:  # Need minimum actions for pattern detection
            patterns = self.pattern_detector._detect_patterns_in_actions(self.actions_history)
            
            # Patterns should be consistent - if we detect a pattern, 
            # it should still be detectable with more data
            if self.detected_patterns and patterns:
                # Check if previously detected patterns are still present or evolved
                previous_signatures = {p["pattern_signature"] for p in self.detected_patterns}
                current_signatures = {p["pattern_signature"] for p in patterns}
                
                # At least some patterns should persist or evolve
                # (allowing for pattern evolution as more data is added)
                if len(previous_signatures) > 0:
                    overlap = len(previous_signatures.intersection(current_signatures))
                    evolution_ratio = overlap / len(previous_signatures)
                    
                    # Allow for some pattern evolution, but not complete disappearance
                    assert evolution_ratio >= 0.3 or len(patterns) >= len(self.detected_patterns), \
                        "Patterns should persist or evolve, not completely disappear"
            
            self.detected_patterns = patterns
    
    @rule()
    def verify_pattern_properties(self):
        """Verify that all detected patterns have valid properties."""
        for pattern in self.detected_patterns:
            # Basic property validation
            assert pattern["frequency"] >= 1, "Pattern frequency should be at least 1"
            assert 0.0 <= pattern["confidence_score"] <= 1.0, "Confidence should be between 0 and 1"
            assert len(pattern["action_sequence"]) > 0, "Pattern should have actions"
            
            # Pattern should make sense given the action history
            pattern_actions = pattern["action_sequence"]
            if len(self.actions_history) >= pattern["frequency"]:
                # The pattern should be findable in the action history
                action_types_in_history = [a["action_type"] for a in self.actions_history]
                pattern_should_exist = any(
                    action_type in str(pattern_actions) for action_type in action_types_in_history
                )
                assert pattern_should_exist, "Detected pattern should relate to actual actions"


# Test class for running the state machine
TestAutomationPatternDetectionStateMachine = AutomationPatternDetectionStateMachine.TestCase


if __name__ == "__main__":
    # Run a simple test to verify the property
    test_instance = TestAutomationPatternDetection()
    test_instance.setup_method()
    
    # Test with a simple repetitive sequence
    repetitive_actions = [
        {"type": ActionType.FILE_OPERATION.value, "name": "create_file", "data": {"file_path": f"test_{i}.txt"}},
        {"type": ActionType.FILE_OPERATION.value, "name": "create_file", "data": {"file_path": f"test_{i}.txt"}},
        {"type": ActionType.FILE_OPERATION.value, "name": "create_file", "data": {"file_path": f"test_{i}.txt"}},
        {"type": ActionType.FILE_OPERATION.value, "name": "create_file", "data": {"file_path": f"test_{i}.txt"}},
    ]
    
    print("Testing automation pattern detection property...")
    test_instance.test_detects_repetitive_patterns(repetitive_actions)
    print("✓ Property 13: Automation Pattern Detection - Basic test passed")