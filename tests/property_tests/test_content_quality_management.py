"""Property-based tests for content quality management.

Feature: ai-learning-accelerator, Property 22: Content Quality Management
Validates: Requirements 6.1, 6.4
"""

import pytest
from datetime import datetime
from hypothesis import given, strategies as st, settings
from sqlalchemy.ext.asyncio import AsyncSession

from ai_learning_accelerator.models.content import LearningContent, ContentRating, QualityStatus
from ai_learning_accelerator.models.content import ContentType, ContentFormat
from ai_learning_accelerator.models.user import DifficultyLevel, User
from ai_learning_accelerator.services.content import ContentService
from ai_learning_accelerator.services.user import UserService
from ai_learning_accelerator.schemas.content import LearningContentCreate, ContentRatingCreate
from ai_learning_accelerator.schemas.user import UserCreate


class TestContentQualityManagement:
    """Property-based tests for content quality management."""

    @pytest.mark.asyncio
    @given(
        title=st.text(min_size=1, max_size=100),
        content_type=st.sampled_from(list(ContentType)),
        difficulty_level=st.sampled_from(list(DifficultyLevel)),
        quality_score=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=10, deadline=None)
    async def test_content_quality_validation_property(
        self, 
        title: str,
        content_type: ContentType,
        difficulty_level: DifficultyLevel,
        quality_score: float,
        test_db: AsyncSession
    ):
        """
        Property 22: Content Quality Management
        For any new content added to the knowledge base, the system should validate 
        quality and accuracy before integration.
        **Validates: Requirements 6.1, 6.4**
        """
        content_service = ContentService(test_db)
        
        # Create content data
        content_data = LearningContentCreate(
            title=title,
            description="Test content description",
            content_type=content_type,
            difficulty_level=difficulty_level,
            content_text="This is test content for quality validation."
        )
        
        # Create content
        content = await content_service.create_learning_content(content_data)
        assert content is not None
        
        # Verify content quality management properties
        assert content.title == title
        assert content.content_type == content_type
        assert content.difficulty_level == difficulty_level
        
        # Verify quality management defaults
        assert content.quality_status == QualityStatus.DRAFT  # Default status
        assert content.quality_score == 0.0  # Default quality score
        assert content.is_active is True  # Default active status
        assert content.view_count == 0  # Initial view count
        assert content.completion_count == 0  # Initial completion count
        assert content.average_rating == 0.0  # Initial rating
        assert content.rating_count == 0  # Initial rating count
        
        # Verify timestamps for quality tracking
        assert content.created_at is not None
        assert content.updated_at is not None
        assert content.created_at <= datetime.utcnow()
        assert content.updated_at <= datetime.utcnow()
        
        # Verify content can be retrieved for quality review
        retrieved_content = await content_service.get_learning_content(content.id)
        assert retrieved_content is not None
        assert retrieved_content.id == content.id
        assert retrieved_content.quality_status == QualityStatus.DRAFT

    @pytest.mark.asyncio
    @given(
        rating_value=st.integers(min_value=1, max_value=5),
        clarity_rating=st.integers(min_value=1, max_value=5),
        usefulness_rating=st.integers(min_value=1, max_value=5),
        accuracy_rating=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10, deadline=None)
    async def test_content_ranking_by_quality_property(
        self, 
        rating_value: int,
        clarity_rating: int,
        usefulness_rating: int,
        accuracy_rating: int,
        test_db: AsyncSession
    ):
        """
        Property 22: Content Quality Management
        For any content with user ratings, the system should maintain proper ranking 
        based on relevance, accuracy, and user ratings.
        **Validates: Requirements 6.1, 6.4**
        """
        content_service = ContentService(test_db)
        user_service = UserService(test_db)
        
        # Create a test user
        user_data = UserCreate(
            email="testuser@example.com",
            username="testuser",
            full_name="Test User",
            password="testpassword123"
        )
        user = await user_service.create_user(user_data)
        assert user is not None
        
        # Create test content
        content_data = LearningContentCreate(
            title="Test Content for Rating",
            description="Content to test rating system",
            content_type=ContentType.TUTORIAL,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            content_text="This is test content for rating validation."
        )
        
        content = await content_service.create_learning_content(content_data)
        assert content is not None
        
        # Create content rating
        rating_data = ContentRatingCreate(
            rating=rating_value,
            review_text="Test review for quality management",
            clarity_rating=clarity_rating,
            usefulness_rating=usefulness_rating,
            accuracy_rating=accuracy_rating
        )
        
        rating = await content_service.create_content_rating(
            content.id, user.id, rating_data
        )
        assert rating is not None
        
        # Verify rating data consistency
        assert rating.content_id == content.id
        assert rating.user_id == user.id
        assert rating.rating == rating_value
        assert rating.clarity_rating == clarity_rating
        assert rating.usefulness_rating == usefulness_rating
        assert rating.accuracy_rating == accuracy_rating
        
        # Verify rating quality management properties
        assert rating.is_verified is False  # Default verification status
        assert rating.helpful_votes == 0  # Initial helpful votes
        
        # Verify timestamps for quality tracking
        assert rating.created_at is not None
        assert rating.updated_at is not None
        assert rating.created_at <= datetime.utcnow()
        assert rating.updated_at <= datetime.utcnow()
        
        # Verify content rating statistics are updated
        updated_content = await content_service.get_learning_content(content.id)
        assert updated_content is not None
        assert updated_content.average_rating == float(rating_value)
        assert updated_content.rating_count == 1

    @pytest.mark.asyncio
    @given(
        initial_rating=st.integers(min_value=1, max_value=5),
        updated_rating=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10, deadline=None)
    async def test_content_quality_consistency_property(
        self, 
        initial_rating: int,
        updated_rating: int,
        test_db: AsyncSession
    ):
        """
        Property 22: Content Quality Management
        For any content quality updates, the system should maintain consistency 
        in quality metrics and ranking calculations.
        **Validates: Requirements 6.1, 6.4**
        """
        content_service = ContentService(test_db)
        user_service = UserService(test_db)
        
        # Create a test user
        user_data = UserCreate(
            email="qualityuser@example.com",
            username="qualityuser",
            full_name="Quality User",
            password="testpassword123"
        )
        user = await user_service.create_user(user_data)
        assert user is not None
        
        # Create test content
        content_data = LearningContentCreate(
            title="Quality Consistency Test Content",
            description="Content to test quality consistency",
            content_type=ContentType.ARTICLE,
            difficulty_level=DifficultyLevel.BEGINNER,
            content_text="This is test content for quality consistency validation."
        )
        
        content = await content_service.create_learning_content(content_data)
        assert content is not None
        
        # Create initial rating
        initial_rating_data = ContentRatingCreate(
            rating=initial_rating,
            review_text="Initial quality review"
        )
        
        rating = await content_service.create_content_rating(
            content.id, user.id, initial_rating_data
        )
        assert rating is not None
        assert rating.rating == initial_rating
        
        # Verify initial content statistics
        content_after_initial = await content_service.get_learning_content(content.id)
        assert content_after_initial.average_rating == float(initial_rating)
        assert content_after_initial.rating_count == 1
        
        # Update the rating (simulates quality re-evaluation)
        updated_rating_data = ContentRatingCreate(
            rating=updated_rating,
            review_text="Updated quality review"
        )
        
        updated_rating = await content_service.create_content_rating(
            content.id, user.id, updated_rating_data
        )
        assert updated_rating is not None
        assert updated_rating.id == rating.id  # Should update existing rating
        assert updated_rating.rating == updated_rating
        
        # Verify content statistics consistency after update
        content_after_update = await content_service.get_learning_content(content.id)
        assert content_after_update.average_rating == float(updated_rating)
        assert content_after_update.rating_count == 1  # Should still be 1 (updated, not added)
        
        # Verify quality management consistency
        assert content_after_update.id == content.id
        assert content_after_update.title == content.title
        assert content_after_update.quality_status == content.quality_status

    @pytest.mark.asyncio
    @given(
        view_increments=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=10, deadline=None)
    async def test_content_usage_tracking_property(
        self, 
        view_increments: int,
        test_db: AsyncSession
    ):
        """
        Property 22: Content Quality Management
        For any content usage, the system should accurately track usage metrics 
        for quality assessment and ranking.
        **Validates: Requirements 6.1, 6.4**
        """
        content_service = ContentService(test_db)
        
        # Create test content
        content_data = LearningContentCreate(
            title="Usage Tracking Test Content",
            description="Content to test usage tracking",
            content_type=ContentType.EXERCISE,
            difficulty_level=DifficultyLevel.ADVANCED,
            content_text="This is test content for usage tracking validation."
        )
        
        content = await content_service.create_learning_content(content_data)
        assert content is not None
        
        # Verify initial usage statistics
        assert content.view_count == 0
        assert content.completion_count == 0
        
        # Simulate multiple view increments
        for i in range(view_increments):
            success = await content_service.increment_view_count(content.id)
            assert success is True
        
        # Verify view count consistency
        updated_content = await content_service.get_learning_content(content.id)
        assert updated_content is not None
        assert updated_content.view_count == view_increments
        
        # Verify other statistics remain unchanged
        assert updated_content.completion_count == 0
        assert updated_content.average_rating == 0.0
        assert updated_content.rating_count == 0
        
        # Verify content identity consistency
        assert updated_content.id == content.id
        assert updated_content.title == content.title
        assert updated_content.quality_status == content.quality_status

    @pytest.mark.asyncio
    @given(
        num_ratings=st.integers(min_value=2, max_value=5),
        rating_values=st.lists(st.integers(min_value=1, max_value=5), min_size=2, max_size=5)
    )
    @settings(max_examples=5, deadline=None)
    async def test_multiple_ratings_average_property(
        self, 
        num_ratings: int,
        rating_values: list[int],
        test_db: AsyncSession
    ):
        """
        Property 22: Content Quality Management
        For any content with multiple ratings, the system should correctly calculate 
        and maintain average ratings for quality ranking.
        **Validates: Requirements 6.1, 6.4**
        """
        content_service = ContentService(test_db)
        user_service = UserService(test_db)
        
        # Ensure we have the right number of ratings
        rating_values = rating_values[:num_ratings]
        if len(rating_values) < num_ratings:
            rating_values.extend([3] * (num_ratings - len(rating_values)))  # Fill with neutral ratings
        
        # Create test content
        content_data = LearningContentCreate(
            title="Multiple Ratings Test Content",
            description="Content to test multiple ratings averaging",
            content_type=ContentType.QUIZ,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            content_text="This is test content for multiple ratings validation."
        )
        
        content = await content_service.create_learning_content(content_data)
        assert content is not None
        
        # Create multiple users and ratings
        created_ratings = []
        for i, rating_value in enumerate(rating_values):
            # Create unique user for each rating
            user_data = UserCreate(
                email=f"ratinguser{i}@example.com",
                username=f"ratinguser{i}",
                full_name=f"Rating User {i}",
                password="testpassword123"
            )
            user = await user_service.create_user(user_data)
            assert user is not None
            
            # Create rating
            rating_data = ContentRatingCreate(
                rating=rating_value,
                review_text=f"Rating {i+1} for quality test"
            )
            
            rating = await content_service.create_content_rating(
                content.id, user.id, rating_data
            )
            assert rating is not None
            created_ratings.append(rating)
        
        # Verify average rating calculation
        expected_average = sum(rating_values) / len(rating_values)
        updated_content = await content_service.get_learning_content(content.id)
        assert updated_content is not None
        assert abs(updated_content.average_rating - expected_average) < 0.01  # Allow small floating point differences
        assert updated_content.rating_count == len(rating_values)
        
        # Verify all ratings were created correctly
        content_ratings = await content_service.get_content_ratings(content.id)
        assert len(content_ratings) == len(rating_values)
        
        # Verify rating values match
        actual_ratings = [r.rating for r in content_ratings]
        assert sorted(actual_ratings) == sorted(rating_values)