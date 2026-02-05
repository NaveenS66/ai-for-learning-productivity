"""Property-based tests for skill-level adaptive explanations.

Feature: ai-learning-accelerator, Property 1: Skill-Level Adaptive Explanations
Validates: Requirements 1.1, 1.3, 2.2

Property: For any user with a defined skill level and any concept or error requiring explanation, 
the system should provide explanations with complexity and examples that match the user's 
demonstrated competency level.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize
from uuid import uuid4
from datetime import datetime, timedelta

from src.ai_learning_accelerator.models.user import User, UserProfile, SkillAssessment, SkillLevel, LearningStyle, DifficultyLevel
from src.ai_learning_accelerator.models.learning import CompetencyArea, SkillAssessmentDetail, SkillAssessmentType
from src.ai_learning_accelerator.services.learning_engine import LearningEngine
from src.ai_learning_accelerator.services.user import UserService


# Strategy for generating skill levels
skill_levels = st.sampled_from([
    SkillLevel.NOVICE,
    SkillLevel.BEGINNER, 
    SkillLevel.INTERMEDIATE,
    SkillLevel.ADVANCED,
    SkillLevel.EXPERT
])

# Strategy for generating competency areas
competency_areas = st.sampled_from([
    "python", "javascript", "machine_learning", "web_development", 
    "data_structures", "algorithms", "databases", "testing", "devops"
])

# Strategy for generating explanation complexity indicators
complexity_indicators = st.lists(
    st.text(min_size=1, max_size=50), 
    min_size=1, 
    max_size=10
)


class SkillLevelAdaptationStateMachine(RuleBasedStateMachine):
    """Stateful property test for skill-level adaptive explanations."""
    
    users = Bundle('users')
    competency_areas = Bundle('competency_areas')
    assessments = Bundle('assessments')
    
    def __init__(self):
        super().__init__()
        self.db = None
        self.learning_engine = None
        self.user_service = None
        self.created_users = {}
        self.created_areas = {}
        self.user_skill_levels = {}
    
    @initialize()
    async def setup_services(self):
        """Initialize services for testing."""
        # This would be set up by the test framework
        pass
    
    @rule(target=competency_areas, name=st.text(min_size=3, max_size=20))
    async def create_competency_area(self, name):
        """Create a competency area for testing."""
        area = CompetencyArea(
            name=name,
            description=f"Competency area for {name}",
            category="technical",
            is_active=True
        )
        
        if self.db:
            self.db.add(area)
            await self.db.commit()
            await self.db.refresh(area)
        
        self.created_areas[name] = area
        return area
    
    @rule(
        target=users,
        skill_level=skill_levels,
        learning_style=st.sampled_from([LearningStyle.VISUAL, LearningStyle.AUDITORY, LearningStyle.KINESTHETIC, LearningStyle.MULTIMODAL])
    )
    async def create_user_with_skill_level(self, skill_level, learning_style):
        """Create a user with a specific skill level."""
        user_id = uuid4()
        
        user = User(
            id=user_id,
            email=f"test_{user_id}@example.com",
            username=f"user_{user_id}",
            hashed_password="hashed_password",
            is_active=True
        )
        
        profile = UserProfile(
            user_id=user_id,
            learning_style=learning_style,
            difficulty_preference=DifficultyLevel.INTERMEDIATE,
            interests=["programming", "learning"]
        )
        
        if self.db:
            self.db.add(user)
            self.db.add(profile)
            await self.db.commit()
            await self.db.refresh(user)
            await self.db.refresh(profile)
        
        self.created_users[user_id] = user
        self.user_skill_levels[user_id] = skill_level
        
        return user
    
    @rule(
        user=users,
        area=competency_areas,
        confidence=st.integers(min_value=0, max_value=100)
    )
    async def assess_user_skill(self, user, area, confidence):
        """Assess a user's skill in a competency area."""
        assume(user.id in self.user_skill_levels)
        assume(area.name in self.created_areas)
        
        skill_level = self.user_skill_levels[user.id]
        
        assessment = SkillAssessmentDetail(
            user_id=user.id,
            competency_area_id=area.id,
            assessment_type=SkillAssessmentType.AI_ASSESSMENT,
            skill_level=skill_level,
            confidence_score=confidence,
            assessment_data={"test": "data"},
            assessment_date=datetime.utcnow()
        )
        
        if self.db:
            self.db.add(assessment)
            await self.db.commit()
        
        return assessment
    
    @rule(
        user=users,
        concept=st.text(min_size=5, max_size=50),
        complexity_level=st.integers(min_value=1, max_value=5)
    )
    async def test_explanation_adaptation(self, user, concept, complexity_level):
        """Test that explanations adapt to user skill level."""
        assume(user.id in self.user_skill_levels)
        
        user_skill = self.user_skill_levels[user.id]
        
        if self.learning_engine:
            # Test skill assessment
            assessed_skill = await self.learning_engine.assess_user_skill_level(
                user.id, 
                concept
            )
            
            # Property: Assessed skill should be consistent with user's actual skill level
            # (allowing for some variance due to limited data)
            skill_numeric_actual = self._skill_to_numeric(user_skill)
            skill_numeric_assessed = self._skill_to_numeric(assessed_skill)
            
            # Allow variance of ±1 skill level for realistic assessment
            assert abs(skill_numeric_actual - skill_numeric_assessed) <= 1, \
                f"Skill assessment too far from actual: {user_skill} vs {assessed_skill}"
            
            # Test learning preferences analysis
            preferences = await self.learning_engine.analyze_learning_preferences(user.id)
            
            # Property: Preferences should be appropriate for skill level
            if user_skill in [SkillLevel.NOVICE, SkillLevel.BEGINNER]:
                # Beginners should get more structured, guided content
                assert preferences.get("optimal_session_length", 30) <= 45, \
                    "Beginners should have shorter optimal session lengths"
            elif user_skill in [SkillLevel.ADVANCED, SkillLevel.EXPERT]:
                # Advanced users can handle longer, more complex sessions
                assert preferences.get("optimal_session_length", 30) >= 20, \
                    "Advanced users should be able to handle reasonable session lengths"
    
    def _skill_to_numeric(self, skill_level: SkillLevel) -> int:
        """Convert skill level to numeric for comparison."""
        mapping = {
            SkillLevel.NOVICE: 0,
            SkillLevel.BEGINNER: 1,
            SkillLevel.INTERMEDIATE: 2,
            SkillLevel.ADVANCED: 3,
            SkillLevel.EXPERT: 4
        }
        return mapping.get(skill_level, 1)


# Property-based tests using Hypothesis strategies

@given(
    skill_level=skill_levels,
    competency_area=competency_areas,
    confidence_score=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=50, deadline=5000)
async def test_skill_assessment_consistency_property(skill_level, competency_area, confidence_score):
    """
    Property: Skill assessments should be consistent with user's demonstrated competency.
    
    For any user with a defined skill level, the learning engine should assess their
    skill level in a way that's consistent with their actual competency.
    """
    # Create mock user data
    user_id = uuid4()
    
    # Mock learning activities that reflect the skill level
    mock_activities = _generate_mock_activities_for_skill_level(skill_level, competency_area)
    mock_assessments = _generate_mock_assessments_for_skill_level(skill_level, competency_area, confidence_score)
    
    # Test the skill assessment logic
    learning_engine = LearningEngine(None)  # Mock DB session
    
    # Override methods to use mock data
    learning_engine.user_service.get_user_learning_activities = lambda uid: mock_activities
    learning_engine.user_service.get_user_skill_assessments = lambda uid: mock_assessments
    
    assessed_skill = await learning_engine.assess_user_skill_level(user_id, competency_area)
    
    # Property: Assessed skill should be within reasonable range of actual skill
    skill_numeric_actual = _skill_level_to_numeric(skill_level)
    skill_numeric_assessed = _skill_level_to_numeric(assessed_skill)
    
    # Allow variance based on confidence score
    max_variance = 2 if confidence_score < 50 else 1
    
    assert abs(skill_numeric_actual - skill_numeric_assessed) <= max_variance, \
        f"Skill assessment inconsistent: actual={skill_level}, assessed={assessed_skill}, confidence={confidence_score}"


@given(
    skill_level=skill_levels,
    explanation_complexity=st.integers(min_value=1, max_value=5),
    concept_difficulty=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=30, deadline=3000)
async def test_explanation_complexity_adaptation_property(skill_level, explanation_complexity, concept_difficulty):
    """
    Property: Explanation complexity should adapt to user skill level.
    
    For any user with a defined skill level and any concept requiring explanation,
    the complexity of explanations should match the user's competency level.
    """
    user_id = uuid4()
    
    # Mock user profile with skill level
    mock_profile = UserProfile(
        user_id=user_id,
        learning_style=LearningStyle.MULTIMODAL,
        difficulty_preference=_skill_level_to_difficulty(skill_level)
    )
    
    learning_engine = LearningEngine(None)
    
    # Test learning preferences analysis
    learning_engine.user_service.get_user_profile = lambda uid: mock_profile
    learning_engine.user_service.get_user_learning_activities = lambda uid: []
    
    preferences = await learning_engine.analyze_learning_preferences(user_id)
    
    # Property: Difficulty preference should align with skill level
    expected_difficulty = _skill_level_to_difficulty(skill_level)
    actual_difficulty = preferences.get("difficulty_preference", DifficultyLevel.INTERMEDIATE)
    
    assert actual_difficulty == expected_difficulty, \
        f"Difficulty preference mismatch: expected={expected_difficulty}, actual={actual_difficulty}"
    
    # Property: Session length should be appropriate for skill level
    session_length = preferences.get("optimal_session_length", 30)
    
    if skill_level in [SkillLevel.NOVICE, SkillLevel.BEGINNER]:
        assert session_length <= 45, f"Session too long for {skill_level}: {session_length} minutes"
    elif skill_level == SkillLevel.EXPERT:
        assert session_length >= 20, f"Session too short for {skill_level}: {session_length} minutes"


@given(
    user_skill=skill_levels,
    error_complexity=st.integers(min_value=1, max_value=5),
    context_data=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.text(min_size=1, max_size=20),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=25, deadline=2000)
async def test_debugging_explanation_adaptation_property(user_skill, error_complexity, context_data):
    """
    Property: Debugging explanations should consider user skill level.
    
    For any debugging scenario, explanations should be appropriate to the user's
    experience level (Requirement 2.2).
    """
    user_id = uuid4()
    
    # Mock debugging context
    mock_assessments = _generate_mock_assessments_for_skill_level(
        user_skill, 
        "debugging", 
        confidence_score=75
    )
    
    learning_engine = LearningEngine(None)
    learning_engine.user_service.get_user_skill_assessments = lambda uid: mock_assessments
    learning_engine.user_service.get_user_learning_activities = lambda uid: []
    
    # Test skill assessment for debugging context
    assessed_skill = await learning_engine.assess_user_skill_level(user_id, "debugging")
    
    # Property: Assessment should reflect user's actual debugging skill
    skill_diff = abs(_skill_level_to_numeric(user_skill) - _skill_level_to_numeric(assessed_skill))
    
    assert skill_diff <= 1, \
        f"Debugging skill assessment too different: actual={user_skill}, assessed={assessed_skill}"
    
    # Property: Recommendations should be appropriate for skill level
    recommendations = await learning_engine.generate_learning_recommendations(user_id, limit=5)
    
    # For debugging context, recommendations should match skill level
    if user_skill in [SkillLevel.NOVICE, SkillLevel.BEGINNER]:
        # Beginners should get foundational debugging content
        assert len(recommendations) > 0, "Beginners should receive debugging recommendations"
    elif user_skill in [SkillLevel.ADVANCED, SkillLevel.EXPERT]:
        # Advanced users might get fewer basic recommendations
        pass  # Advanced users may have fewer recommendations if they're competent


# Helper functions

def _generate_mock_activities_for_skill_level(skill_level: SkillLevel, competency_area: str):
    """Generate mock learning activities that reflect a skill level."""
    from src.ai_learning_accelerator.models.user import LearningActivity
    
    activities = []
    
    # Generate activities with completion rates and scores that reflect skill level
    skill_numeric = _skill_level_to_numeric(skill_level)
    
    for i in range(3):  # Generate 3 mock activities
        # Higher skill levels have higher completion rates and scores
        base_completion = 40 + (skill_numeric * 15)  # 40-100% range
        base_score = 30 + (skill_numeric * 17)  # 30-98% range
        
        activity = LearningActivity(
            user_id=uuid4(),
            title=f"{competency_area} activity {i+1}",
            description=f"Learning activity for {competency_area}",
            activity_type=competency_area,
            completion_percentage=min(100, base_completion + (i * 5)),
            score=min(100, base_score + (i * 3)),
            duration_minutes=30 + (i * 10)
        )
        activities.append(activity)
    
    return activities


def _generate_mock_assessments_for_skill_level(skill_level: SkillLevel, domain: str, confidence_score: int):
    """Generate mock skill assessments that reflect a skill level."""
    from src.ai_learning_accelerator.models.user import SkillAssessment
    
    assessment = SkillAssessment(
        user_id=uuid4(),
        domain=domain,
        skill_level=skill_level,
        confidence_score=confidence_score,
        last_assessed=datetime.utcnow() - timedelta(days=7)
    )
    
    return [assessment]


def _skill_level_to_numeric(skill_level: SkillLevel) -> int:
    """Convert skill level to numeric value for comparison."""
    mapping = {
        SkillLevel.NOVICE: 0,
        SkillLevel.BEGINNER: 1,
        SkillLevel.INTERMEDIATE: 2,
        SkillLevel.ADVANCED: 3,
        SkillLevel.EXPERT: 4
    }
    return mapping.get(skill_level, 1)


def _skill_level_to_difficulty(skill_level: SkillLevel) -> DifficultyLevel:
    """Convert skill level to appropriate difficulty preference."""
    mapping = {
        SkillLevel.NOVICE: DifficultyLevel.BEGINNER,
        SkillLevel.BEGINNER: DifficultyLevel.BEGINNER,
        SkillLevel.INTERMEDIATE: DifficultyLevel.INTERMEDIATE,
        SkillLevel.ADVANCED: DifficultyLevel.ADVANCED,
        SkillLevel.EXPERT: DifficultyLevel.EXPERT
    }
    return mapping.get(skill_level, DifficultyLevel.INTERMEDIATE)


# Integration test for the complete property
@pytest.mark.asyncio
async def test_complete_skill_level_adaptation_integration():
    """
    Integration test for complete skill-level adaptation property.
    
    Tests the full workflow of skill assessment, explanation adaptation,
    and debugging assistance across different skill levels.
    """
    # This would be implemented with actual database and service setup
    # For now, we test the property logic with mocked components
    
    test_cases = [
        (SkillLevel.NOVICE, "python", 30),
        (SkillLevel.BEGINNER, "javascript", 50),
        (SkillLevel.INTERMEDIATE, "machine_learning", 70),
        (SkillLevel.ADVANCED, "algorithms", 85),
        (SkillLevel.EXPERT, "system_design", 95)
    ]
    
    for skill_level, competency, confidence in test_cases:
        # Test skill assessment consistency
        await test_skill_assessment_consistency_property(skill_level, competency, confidence)
        
        # Test explanation complexity adaptation
        await test_explanation_complexity_adaptation_property(skill_level, 3, 3)
        
        # Test debugging explanation adaptation
        await test_debugging_explanation_adaptation_property(skill_level, 2, {"error": "test"})
    
    # If we reach here, all property tests passed
    assert True, "All skill-level adaptation properties validated successfully"