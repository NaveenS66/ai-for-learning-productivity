"""Tests for feedback integration system."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.ai_learning_accelerator.models.content import LearningContent, ContentRating, ContentType, QualityStatus
from src.ai_learning_accelerator.models.user import User, DifficultyLevel
from src.ai_learning_accelerator.services.feedback_integration import (
    FeedbackIntegrationService, FeedbackSummary, ContentConflict, 
    ConflictSeverity, RankingUpdate
)


@pytest.fixture
async def sample_users(db_session):
    """Create sample users for testing."""
    users = []
    for i in range(5):
        user = User(
            username=f"testuser{i}",
            email=f"user{i}@test.com",
            hashed_password="hashed_password"
        )
        db_session.add(user)
        users.append(user)
    
    await db_session.commit()
    for user in users:
        await db_session.refresh(user)
    
    return users


@pytest.fixture
async def sample_content(db_session, sample_users):
    """Create sample content for testing."""
    content = LearningContent(
        title="Test Content for Feedback",
        description="Content for testing feedback integration",
        content_type=ContentType.TUTORIAL,
        difficulty_level=DifficultyLevel.BEGINNER,
        content_text="This is test content for feedback integration testing.",
        learning_objectives=["Test feedback"],
        tags=["test", "feedback"],
        author_id=sample_users[0].id,
        quality_status=QualityStatus.PUBLISHED,
        view_count=100,
        completion_count=80
    )
    
    db_session.add(content)
    await db_session.commit()
    await db_session.refresh(content)
    
    return content


@pytest.fixture
async def sample_ratings(db_session, sample_content, sample_users):
    """Create sample ratings for testing."""
    ratings = []
    
    # Create diverse ratings
    rating_data = [
        (sample_users[0].id, 5, "Excellent content! Very helpful and clear."),
        (sample_users[1].id, 4, "Good tutorial, learned a lot."),
        (sample_users[2].id, 2, "Content is outdated and has errors."),
        (sample_users[3].id, 1, "Terrible quality, very confusing."),
        (sample_users[4].id, 4, "Pretty good overall, could use more examples.")
    ]
    
    for user_id, rating, review in rating_data:
        content_rating = ContentRating(
            content_id=sample_content.id,
            user_id=user_id,
            rating=rating,
            review_text=review,
            clarity_rating=rating,
            usefulness_rating=rating,
            accuracy_rating=rating if rating > 2 else 1,
            difficulty_rating=3
        )
        ratings.append(content_rating)
        db_session.add(content_rating)
    
    await db_session.commit()
    for rating in ratings:
        await db_session.refresh(rating)
    
    return ratings


class TestFeedbackCollection:
    """Test feedback collection functionality."""
    
    async def test_collect_new_feedback(self, db_session, sample_content, sample_users):
        """Test collecting new feedback."""
        service = FeedbackIntegrationService(db_session)
        user = sample_users[0]
        
        rating = await service.collect_feedback(
            content_id=sample_content.id,
            user_id=user.id,
            rating=5,
            review_text="Great content!",
            feedback_categories={"clarity": 5, "usefulness": 4}
        )
        
        assert rating is not None
        assert rating.content_id == sample_content.id
        assert rating.user_id == user.id
        assert rating.rating == 5
        assert rating.review_text == "Great content!"
        assert rating.clarity_rating == 5
        assert rating.usefulness_rating == 4
    
    async def test_update_existing_feedback(self, db_session, sample_content, sample_users):
        """Test updating existing feedback."""
        service = FeedbackIntegrationService(db_session)
        user = sample_users[0]
        
        # Create initial feedback
        initial_rating = await service.collect_feedback(
            content_id=sample_content.id,
            user_id=user.id,
            rating=3,
            review_text="Initial review"
        )
        
        # Update feedback
        updated_rating = await service.collect_feedback(
            content_id=sample_content.id,
            user_id=user.id,
            rating=5,
            review_text="Updated review - much better!",
            feedback_categories={"clarity": 5}
        )
        
        assert updated_rating.id == initial_rating.id
        assert updated_rating.rating == 5
        assert updated_rating.review_text == "Updated review - much better!"
        assert updated_rating.clarity_rating == 5
    
    async def test_feedback_with_categories(self, db_session, sample_content, sample_users):
        """Test feedback collection with category ratings."""
        service = FeedbackIntegrationService(db_session)
        user = sample_users[0]
        
        categories = {
            "clarity": 4,
            "usefulness": 5,
            "accuracy": 3,
            "difficulty": 2
        }
        
        rating = await service.collect_feedback(
            content_id=sample_content.id,
            user_id=user.id,
            rating=4,
            feedback_categories=categories
        )
        
        assert rating.clarity_rating == 4
        assert rating.usefulness_rating == 5
        assert rating.accuracy_rating == 3
        assert rating.difficulty_rating == 2


class TestFeedbackSummary:
    """Test feedback summary functionality."""
    
    async def test_get_feedback_summary(self, db_session, sample_content, sample_ratings):
        """Test getting comprehensive feedback summary."""
        service = FeedbackIntegrationService(db_session)
        
        summary = await service.get_feedback_summary(sample_content.id)
        
        assert isinstance(summary, FeedbackSummary)
        assert summary.content_id == sample_content.id
        assert summary.total_ratings == 5
        assert 2.5 <= summary.average_rating <= 3.5  # Based on sample ratings
        assert summary.total_reviews == 5  # All sample ratings have reviews
        
        # Check rating distribution
        assert 1 in summary.rating_distribution
        assert 2 in summary.rating_distribution
        assert 4 in summary.rating_distribution
        assert 5 in summary.rating_distribution
    
    async def test_empty_feedback_summary(self, db_session, sample_content):
        """Test feedback summary for content with no feedback."""
        service = FeedbackIntegrationService(db_session)
        
        # Create content without ratings
        new_content = LearningContent(
            title="No Feedback Content",
            description="Content with no feedback",
            content_type=ContentType.ARTICLE,
            difficulty_level=DifficultyLevel.BEGINNER,
            content_text="Test content"
        )
        db_session.add(new_content)
        await db_session.commit()
        await db_session.refresh(new_content)
        
        summary = await service.get_feedback_summary(new_content.id)
        
        assert summary.total_ratings == 0
        assert summary.average_rating == 0.0
        assert summary.rating_distribution == {}
        assert summary.total_reviews == 0
        assert summary.sentiment_score == 0.0
        assert summary.common_themes == []
        assert summary.recent_feedback_trend == "stable"
    
    async def test_sentiment_calculation(self, db_session, sample_content, sample_users):
        """Test sentiment score calculation."""
        service = FeedbackIntegrationService(db_session)
        
        # Create ratings with clear sentiment
        positive_rating = ContentRating(
            content_id=sample_content.id,
            user_id=sample_users[0].id,
            rating=5,
            review_text="This is excellent and very helpful content!"
        )
        
        negative_rating = ContentRating(
            content_id=sample_content.id,
            user_id=sample_users[1].id,
            rating=1,
            review_text="This is terrible and confusing content."
        )
        
        db_session.add_all([positive_rating, negative_rating])
        await db_session.commit()
        
        # Test sentiment calculation
        ratings = [positive_rating, negative_rating]
        sentiment = await service._calculate_sentiment_score(ratings)
        
        # Should be close to neutral due to mixed sentiment
        assert -0.5 <= sentiment <= 0.5
    
    async def test_theme_extraction(self, db_session, sample_content, sample_users):
        """Test common theme extraction from reviews."""
        service = FeedbackIntegrationService(db_session)
        
        # Create ratings with specific themes
        theme_ratings = [
            ContentRating(
                content_id=sample_content.id,
                user_id=sample_users[0].id,
                rating=4,
                review_text="Content is clear and easy to understand."
            ),
            ContentRating(
                content_id=sample_content.id,
                user_id=sample_users[1].id,
                rating=3,
                review_text="The explanation is clear but could be more complete."
            ),
            ContentRating(
                content_id=sample_content.id,
                user_id=sample_users[2].id,
                rating=5,
                review_text="Very clear tutorial, helped me understand the topic."
            )
        ]
        
        db_session.add_all(theme_ratings)
        await db_session.commit()
        
        themes = await service._extract_common_themes(theme_ratings)
        
        # Should detect "clarity" as a common theme
        assert "clarity" in themes


class TestConflictDetection:
    """Test conflict detection functionality."""
    
    async def test_detect_rating_variance_conflict(self, db_session, sample_content, sample_users):
        """Test detection of rating variance conflicts."""
        service = FeedbackIntegrationService(db_session)
        
        # Create ratings with high variance
        high_variance_ratings = [
            ContentRating(content_id=sample_content.id, user_id=sample_users[0].id, rating=5),
            ContentRating(content_id=sample_content.id, user_id=sample_users[1].id, rating=1),
            ContentRating(content_id=sample_content.id, user_id=sample_users[2].id, rating=5),
            ContentRating(content_id=sample_content.id, user_id=sample_users[3].id, rating=1),
            ContentRating(content_id=sample_content.id, user_id=sample_users[4].id, rating=3)
        ]
        
        db_session.add_all(high_variance_ratings)
        await db_session.commit()
        
        conflicts = await service._analyze_content_conflicts(sample_content.id)
        
        # Should detect rating variance conflict
        variance_conflicts = [c for c in conflicts if c.conflict_type == "rating_variance"]
        assert len(variance_conflicts) > 0
        
        conflict = variance_conflicts[0]
        assert conflict.severity in [ConflictSeverity.LOW, ConflictSeverity.MEDIUM]
        assert "variance" in conflict.description.lower()
    
    async def test_detect_accuracy_dispute_conflict(self, db_session, sample_content, sample_users):
        """Test detection of accuracy dispute conflicts."""
        service = FeedbackIntegrationService(db_session)
        
        # Create ratings with accuracy disputes
        accuracy_dispute_ratings = [
            ContentRating(
                content_id=sample_content.id, 
                user_id=sample_users[0].id, 
                rating=3, 
                accuracy_rating=1
            ),
            ContentRating(
                content_id=sample_content.id, 
                user_id=sample_users[1].id, 
                rating=2, 
                accuracy_rating=1
            ),
            ContentRating(
                content_id=sample_content.id, 
                user_id=sample_users[2].id, 
                rating=4, 
                accuracy_rating=2
            ),
            ContentRating(
                content_id=sample_content.id, 
                user_id=sample_users[3].id, 
                rating=5, 
                accuracy_rating=5
            )
        ]
        
        db_session.add_all(accuracy_dispute_ratings)
        await db_session.commit()
        
        conflicts = await service._analyze_content_conflicts(sample_content.id)
        
        # Should detect accuracy dispute conflict
        accuracy_conflicts = [c for c in conflicts if c.conflict_type == "accuracy_dispute"]
        assert len(accuracy_conflicts) > 0
        
        conflict = accuracy_conflicts[0]
        assert conflict.severity == ConflictSeverity.HIGH
        assert "accuracy" in conflict.description.lower()
    
    async def test_no_conflicts_with_consistent_ratings(self, db_session, sample_content, sample_users):
        """Test that no conflicts are detected with consistent ratings."""
        service = FeedbackIntegrationService(db_session)
        
        # Create consistent ratings
        consistent_ratings = [
            ContentRating(
                content_id=sample_content.id, 
                user_id=sample_users[0].id, 
                rating=4, 
                accuracy_rating=4
            ),
            ContentRating(
                content_id=sample_content.id, 
                user_id=sample_users[1].id, 
                rating=4, 
                accuracy_rating=4
            ),
            ContentRating(
                content_id=sample_content.id, 
                user_id=sample_users[2].id, 
                rating=3, 
                accuracy_rating=4
            )
        ]
        
        db_session.add_all(consistent_ratings)
        await db_session.commit()
        
        conflicts = await service._analyze_content_conflicts(sample_content.id)
        
        # Should not detect significant conflicts
        assert len(conflicts) == 0


class TestConflictResolution:
    """Test conflict resolution functionality."""
    
    async def test_weighted_average_resolution(self, db_session, sample_content, sample_users):
        """Test weighted average conflict resolution."""
        service = FeedbackIntegrationService(db_session)
        
        # Create a conflict
        conflicting_ratings = [
            ContentRating(content_id=sample_content.id, user_id=sample_users[0].id, rating=5),
            ContentRating(content_id=sample_content.id, user_id=sample_users[1].id, rating=1),
            ContentRating(content_id=sample_content.id, user_id=sample_users[2].id, rating=4)
        ]
        
        db_session.add_all(conflicting_ratings)
        await db_session.commit()
        
        # Create conflict object
        conflict = ContentConflict(
            content_id=sample_content.id,
            conflict_type="rating_variance",
            severity=ConflictSeverity.MEDIUM,
            description="High variance in ratings",
            conflicting_ratings=[r.id for r in conflicting_ratings],
            suggested_resolution="Use weighted average",
            confidence=0.8
        )
        
        # Resolve conflict
        success = await service._resolve_with_weighted_average(conflict)
        assert success
        
        # Check that content rating was updated
        await db_session.refresh(sample_content)
        # The weighted average should be around 3.33
        assert 3.0 <= sample_content.average_rating <= 4.0
    
    async def test_expert_review_flagging(self, db_session, sample_content):
        """Test flagging content for expert review."""
        service = FeedbackIntegrationService(db_session)
        
        conflict = ContentConflict(
            content_id=sample_content.id,
            conflict_type="accuracy_dispute",
            severity=ConflictSeverity.HIGH,
            description="Multiple accuracy complaints",
            conflicting_ratings=[],
            suggested_resolution="Expert review needed",
            confidence=0.9
        )
        
        success = await service._flag_for_expert_review(conflict)
        assert success
        
        # Check that content status was updated
        await db_session.refresh(sample_content)
        assert sample_content.quality_status == QualityStatus.REVIEW


class TestRankingUpdates:
    """Test ranking update functionality."""
    
    async def test_calculate_new_ranking(self, db_session, sample_content, sample_ratings):
        """Test calculation of new ranking scores."""
        service = FeedbackIntegrationService(db_session)
        
        # Set initial ranking score
        sample_content.quality_score = 0.5
        await db_session.commit()
        
        ranking_update = await service._calculate_new_ranking(sample_content)
        
        if ranking_update:  # May be None if score didn't change significantly
            assert isinstance(ranking_update, RankingUpdate)
            assert ranking_update.content_id == sample_content.id
            assert 0.0 <= ranking_update.new_score <= 1.0
            assert ranking_update.old_score != ranking_update.new_score
            assert len(ranking_update.factors) > 0
            assert len(ranking_update.rationale) > 0
    
    async def test_apply_ranking_update(self, db_session, sample_content):
        """Test applying ranking updates to content."""
        service = FeedbackIntegrationService(db_session)
        
        ranking_update = RankingUpdate(
            content_id=sample_content.id,
            old_score=0.5,
            new_score=0.8,
            factors={"average_rating": 0.8, "rating_count": 0.6},
            rationale="Improved based on positive feedback"
        )
        
        success = await service._apply_ranking_update(ranking_update)
        assert success
        
        # Check that content score was updated
        await db_session.refresh(sample_content)
        assert sample_content.quality_score == 0.8
    
    async def test_bulk_ranking_updates(self, db_session, sample_content, sample_ratings):
        """Test bulk ranking updates."""
        service = FeedbackIntegrationService(db_session)
        
        result = await service.update_content_rankings(batch_size=10)
        
        assert isinstance(result, dict)
        assert "processed_count" in result
        assert "updated_count" in result
        assert result["processed_count"] >= 1  # At least our sample content


class TestFeedbackTrends:
    """Test feedback trend analysis."""
    
    async def test_feedback_trend_analysis(self, db_session, sample_content, sample_users):
        """Test analysis of feedback trends."""
        service = FeedbackIntegrationService(db_session)
        
        # Create recent positive ratings
        recent_date = datetime.utcnow() - timedelta(days=15)
        recent_ratings = [
            ContentRating(
                content_id=sample_content.id,
                user_id=sample_users[0].id,
                rating=5,
                created_at=recent_date
            ),
            ContentRating(
                content_id=sample_content.id,
                user_id=sample_users[1].id,
                rating=4,
                created_at=recent_date
            )
        ]
        
        # Create older negative ratings
        older_date = datetime.utcnow() - timedelta(days=45)
        older_ratings = [
            ContentRating(
                content_id=sample_content.id,
                user_id=sample_users[2].id,
                rating=2,
                created_at=older_date
            ),
            ContentRating(
                content_id=sample_content.id,
                user_id=sample_users[3].id,
                rating=1,
                created_at=older_date
            )
        ]
        
        db_session.add_all(recent_ratings + older_ratings)
        await db_session.commit()
        
        trend = await service._analyze_feedback_trend(sample_content.id)
        
        # Should detect improving trend
        assert trend == "improving"


class TestFeedbackIntegration:
    """Test integrated feedback functionality."""
    
    async def test_complete_feedback_workflow(self, db_session, sample_content, sample_users):
        """Test complete feedback collection and processing workflow."""
        service = FeedbackIntegrationService(db_session)
        
        # Collect feedback
        rating = await service.collect_feedback(
            content_id=sample_content.id,
            user_id=sample_users[0].id,
            rating=4,
            review_text="Good content with minor issues",
            feedback_categories={"clarity": 4, "accuracy": 3}
        )
        
        assert rating is not None
        
        # Get feedback summary
        summary = await service.get_feedback_summary(sample_content.id)
        assert summary.total_ratings >= 1
        
        # Check for conflicts
        conflicts = await service.detect_content_conflicts(sample_content.id)
        # May or may not have conflicts depending on existing data
        
        # Update rankings
        result = await service.update_content_rankings(batch_size=5)
        assert result["processed_count"] >= 1
    
    async def test_error_handling(self, db_session):
        """Test error handling in feedback service."""
        service = FeedbackIntegrationService(db_session)
        
        # Test with non-existent content
        fake_content_id = uuid4()
        fake_user_id = uuid4()
        
        # Should handle gracefully without raising exceptions
        summary = await service.get_feedback_summary(fake_content_id)
        assert summary.total_ratings == 0
        
        conflicts = await service.detect_content_conflicts(fake_content_id)
        assert conflicts == []
        
        suggestions = await service.generate_update_suggestions(fake_content_id)
        assert suggestions == []