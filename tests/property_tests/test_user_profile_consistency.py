"""Property-based tests for user profile consistency.

Feature: ai-learning-accelerator, Property 2: User Profile Consistency
Validates: Requirements 1.2, 7.1
"""

import pytest
from datetime import datetime
from hypothesis import given, strategies as st, settings
from sqlalchemy.ext.asyncio import AsyncSession

from ai_learning_accelerator.models.user import User, UserProfile, SkillAssessment, LearningActivity
from ai_learning_accelerator.models.user import LearningStyle, DifficultyLevel, DataSharingLevel, SkillLevel
from ai_learning_accelerator.services.user import UserService
from ai_learning_accelerator.schemas.user import (
    UserCreate, LearningActivityCreate, SkillAssessmentCreate
)


class TestUserProfileConsistency:
    """Property-based tests for user profile consistency."""

    @pytest.mark.asyncio
    @given(
        email=st.emails(),
        username=st.text(min_size=3, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        full_name=st.text(min_size=1, max_size=50),
        password=st.text(min_size=8, max_size=50)
    )
    @settings(max_examples=10, deadline=None)
    async def test_user_creation_profile_consistency_property(
        self, 
        email: str,
        username: str,
        full_name: str,
        password: str,
        test_db: AsyncSession
    ):
        """
        Property 2: User Profile Consistency
        For any user creation, the user should be created with consistent
        initial state and proper relationships.
        **Validates: Requirements 1.2, 7.1**
        """
        user_service = UserService(test_db)
        
        # Create user data
        user_data = UserCreate(
            email=email,
            username=username,
            full_name=full_name,
            password=password
        )
        
        # Create user
        user = await user_service.create_user(user_data)
        assert user is not None
        
        # Verify user data consistency
        assert user.email == user_data.email
        assert user.username == user_data.username
        assert user.full_name == user_data.full_name
        assert user.is_active is True  # Default value
        assert user.is_verified is False  # Default value
        assert user.last_login is None  # Initial state
        
        # Verify timestamps are set consistently
        assert user.created_at is not None
        assert user.updated_at is not None
        assert user.created_at == user.updated_at  # Should be equal on creation
        assert user.created_at <= datetime.utcnow()
        
        # Verify password is hashed (not stored in plain text)
        assert user.hashed_password != user_data.password
        assert len(user.hashed_password) > 0
        
        # Verify user can be retrieved consistently
        retrieved_user = await user_service.get_user_by_id(user.id)
        assert retrieved_user is not None
        assert retrieved_user.id == user.id
        assert retrieved_user.email == user.email
        assert retrieved_user.username == user.username

    @pytest.mark.asyncio
    @given(
        domain=st.text(min_size=1, max_size=50),
        skill_level=st.sampled_from(list(SkillLevel)),
        confidence_score=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=10, deadline=None)
    async def test_skill_assessment_consistency_property(
        self, 
        domain: str,
        skill_level: SkillLevel,
        confidence_score: int,
        test_db: AsyncSession
    ):
        """
        Property 2: User Profile Consistency
        For any skill assessment, the user's profile should be updated 
        to accurately reflect their new competency changes.
        **Validates: Requirements 1.2, 7.1**
        """
        user_service = UserService(test_db)
        
        # Create a test user first
        user_data = UserCreate(
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            password="testpassword123"
        )
        user = await user_service.create_user(user_data)
        assert user is not None
        
        # Create skill assessment
        assessment_data = SkillAssessmentCreate(
            domain=domain,
            skill_level=skill_level,
            confidence_score=confidence_score
        )
        
        assessment = await user_service.create_skill_assessment(user.id, assessment_data)
        assert assessment is not None
        
        # Verify assessment data consistency
        assert assessment.user_id == user.id
        assert assessment.domain == domain
        assert assessment.skill_level == skill_level
        assert assessment.confidence_score == confidence_score
        
        # Verify timestamps are set
        assert assessment.created_at is not None
        assert assessment.updated_at is not None
        assert assessment.last_assessed is not None
        assert assessment.created_at <= datetime.utcnow()
        assert assessment.updated_at <= datetime.utcnow()
        assert assessment.last_assessed <= datetime.utcnow()
        
        # Verify assessment can be retrieved
        retrieved_assessment = await user_service.get_skill_assessment(user.id, domain)
        assert retrieved_assessment is not None
        assert retrieved_assessment.id == assessment.id
        assert retrieved_assessment.skill_level == skill_level

    @pytest.mark.asyncio
    @given(
        activity_type=st.text(min_size=1, max_size=30),
        title=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=10, deadline=None)
    async def test_learning_activity_consistency_property(
        self, 
        activity_type: str,
        title: str,
        test_db: AsyncSession
    ):
        """
        Property 2: User Profile Consistency
        For any learning activity, the user's profile should be updated 
        to accurately reflect their new progress.
        **Validates: Requirements 1.2, 7.1**
        """
        user_service = UserService(test_db)
        
        # Create a test user first
        user_data = UserCreate(
            email="test2@example.com",
            username="testuser2",
            full_name="Test User 2",
            password="testpassword123"
        )
        user = await user_service.create_user(user_data)
        assert user is not None
        
        # Record initial state
        initial_activities = await user_service.get_user_learning_activities(user.id)
        initial_count = len(initial_activities)
        
        # Create learning activity
        activity_data = LearningActivityCreate(
            activity_type=activity_type,
            title=title
        )
        
        activity = await user_service.create_learning_activity(user.id, activity_data)
        assert activity is not None
        
        # Verify activity data consistency
        assert activity.user_id == user.id
        assert activity.activity_type == activity_type
        assert activity.title == title
        assert activity.status == "in_progress"  # Default value
        assert activity.completion_percentage == 0  # Default value
        
        # Verify timestamps are set
        assert activity.created_at is not None
        assert activity.updated_at is not None
        assert activity.created_at <= datetime.utcnow()
        assert activity.updated_at <= datetime.utcnow()
        
        # Verify profile consistency: activity should be recorded
        updated_activities = await user_service.get_user_learning_activities(user.id)
        assert len(updated_activities) == initial_count + 1
        
        # Verify the new activity is in the list
        activity_ids = [a.id for a in updated_activities]
        assert activity.id in activity_ids