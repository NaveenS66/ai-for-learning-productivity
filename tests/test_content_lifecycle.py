"""Tests for content lifecycle management."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from src.ai_learning_accelerator.models.content import (
    LearningContent, ContentRating, ContentType, ContentFormat, QualityStatus
)
from src.ai_learning_accelerator.models.user import User, DifficultyLevel
from src.ai_learning_accelerator.services.content_lifecycle import (
    ContentLifecycleService, ValidationSeverity, ValidationIssue,
    QualityAssessment, DeprecationAlert, UpdateSuggestion
)


@pytest.fixture
async def sample_content(db_session):
    """Create sample learning content for testing."""
    user = User(
        username="testauthor",
        email="author@test.com",
        hashed_password="hashed_password"
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    
    content = LearningContent(
        title="Sample Python Tutorial",
        description="A comprehensive tutorial on Python basics for beginners",
        content_type=ContentType.TUTORIAL,
        content_format=ContentFormat.MARKDOWN,
        difficulty_level=DifficultyLevel.BEGINNER,
        estimated_duration=60,
        learning_objectives=["Learn Python syntax", "Understand variables", "Write simple programs"],
        tags=["python", "programming", "tutorial"],
        topics=["programming", "python"],
        content_text="""
        # Python Basics Tutorial
        
        This tutorial covers the fundamentals of Python programming.
        
        ## Variables
        In Python, you can create variables like this:
        ```python
        name = "Alice"
        age = 25
        ```
        
        ## Functions
        Functions are defined using the def keyword:
        ```python
        def greet(name):
            return f"Hello, {name}!"
        ```
        """,
        author_id=user.id,
        quality_status=QualityStatus.PUBLISHED,
        view_count=100,
        completion_count=80,
        average_rating=4.2,
        rating_count=15
    )
    
    db_session.add(content)
    await db_session.commit()
    await db_session.refresh(content)
    
    return content


@pytest.fixture
async def poor_quality_content(db_session):
    """Create poor quality content for testing."""
    content = LearningContent(
        title="Bad",  # Too short
        description="Short",  # Too short
        content_type=ContentType.TUTORIAL,
        content_format=ContentFormat.MARKDOWN,
        difficulty_level=DifficultyLevel.ADVANCED,
        content_text="TODO: Write content here",  # Placeholder text
        learning_objectives=[],  # Empty
        tags=[],  # Empty
        quality_status=QualityStatus.DRAFT,
        view_count=50,
        completion_count=5,  # Low completion rate
        average_rating=2.0,  # Low rating
        rating_count=10
    )
    
    db_session.add(content)
    await db_session.commit()
    await db_session.refresh(content)
    
    return content


@pytest.fixture
async def old_content(db_session):
    """Create old content for deprecation testing."""
    old_date = datetime.utcnow() - timedelta(days=400)  # Over 1 year old
    
    content = LearningContent(
        title="Old JavaScript Tutorial",
        description="An outdated tutorial on JavaScript",
        content_type=ContentType.TUTORIAL,
        content_format=ContentFormat.MARKDOWN,
        difficulty_level=DifficultyLevel.INTERMEDIATE,
        content_text="This tutorial covers JavaScript ES5 features.",
        learning_objectives=["Learn JavaScript"],
        tags=["javascript"],
        quality_status=QualityStatus.PUBLISHED,
        view_count=200,
        completion_count=20,  # Low completion rate
        average_rating=2.3,  # Low rating
        rating_count=20,
        created_at=old_date
    )
    
    db_session.add(content)
    await db_session.commit()
    await db_session.refresh(content)
    
    return content


@pytest.fixture
async def content_with_ratings(db_session, sample_content):
    """Create content with various ratings."""
    user1 = User(username="user1", email="user1@test.com", hashed_password="hash")
    user2 = User(username="user2", email="user2@test.com", hashed_password="hash")
    user3 = User(username="user3", email="user3@test.com", hashed_password="hash")
    
    db_session.add_all([user1, user2, user3])
    await db_session.commit()
    
    # Add ratings with reviews
    ratings = [
        ContentRating(
            content_id=sample_content.id,
            user_id=user1.id,
            rating=5,
            review_text="Great tutorial! Very clear and helpful.",
            clarity_rating=5,
            usefulness_rating=5,
            accuracy_rating=5
        ),
        ContentRating(
            content_id=sample_content.id,
            user_id=user2.id,
            rating=2,
            review_text="Content is outdated and has some errors.",
            clarity_rating=3,
            usefulness_rating=2,
            accuracy_rating=1
        ),
        ContentRating(
            content_id=sample_content.id,
            user_id=user3.id,
            rating=4,
            review_text="Good content but could use more examples.",
            clarity_rating=4,
            usefulness_rating=4,
            accuracy_rating=4
        )
    ]
    
    db_session.add_all(ratings)
    await db_session.commit()
    
    return sample_content


class TestContentValidation:
    """Test content validation functionality."""
    
    async def test_validate_good_content(self, db_session, sample_content):
        """Test validation of good quality content."""
        service = ContentLifecycleService(db_session)
        
        issues = await service.validate_content(sample_content)
        
        # Should have minimal issues
        assert len(issues) <= 2  # Maybe some minor suggestions
        
        # No critical or error issues
        critical_errors = [i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
        assert len(critical_errors) == 0
    
    async def test_validate_poor_content(self, db_session, poor_quality_content):
        """Test validation of poor quality content."""
        service = ContentLifecycleService(db_session)
        
        issues = await service.validate_content(poor_quality_content)
        
        # Should have multiple issues
        assert len(issues) > 3
        
        # Should have error issues
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        assert len(error_issues) > 0
        
        # Check specific issues
        issue_messages = [issue.message for issue in issues]
        assert any("title" in msg.lower() for msg in issue_messages)
        assert any("description" in msg.lower() for msg in issue_messages)
        assert any("placeholder" in msg.lower() for msg in issue_messages)
    
    async def test_validate_markdown_format(self, db_session):
        """Test markdown format validation."""
        content = LearningContent(
            title="Markdown Test",
            description="Testing markdown validation",
            content_type=ContentType.TUTORIAL,
            content_format=ContentFormat.MARKDOWN,
            difficulty_level=DifficultyLevel.BEGINNER,
            content_text="```python\nprint('hello')\n# Missing closing backticks",
            learning_objectives=["Test markdown"]
        )
        
        db_session.add(content)
        await db_session.commit()
        
        service = ContentLifecycleService(db_session)
        issues = await service.validate_content(content)
        
        # Should detect unclosed code block
        format_issues = [i for i in issues if i.category == "format"]
        assert len(format_issues) > 0
        assert any("code block" in issue.message.lower() for issue in format_issues)


class TestQualityAssessment:
    """Test quality assessment functionality."""
    
    async def test_assess_good_content_quality(self, db_session, sample_content):
        """Test quality assessment of good content."""
        service = ContentLifecycleService(db_session)
        
        assessment = await service.assess_content_quality(sample_content)
        
        assert isinstance(assessment, QualityAssessment)
        assert 0.0 <= assessment.overall_score <= 1.0
        assert assessment.overall_score > 0.6  # Should be good quality
        assert assessment.is_publishable
        
        # Check category scores
        assert "clarity" in assessment.category_scores
        assert "accuracy" in assessment.category_scores
        assert "completeness" in assessment.category_scores
        assert "relevance" in assessment.category_scores
        assert "engagement" in assessment.category_scores
        
        # Should have recommendations
        assert isinstance(assessment.recommendations, list)
    
    async def test_assess_poor_content_quality(self, db_session, poor_quality_content):
        """Test quality assessment of poor content."""
        service = ContentLifecycleService(db_session)
        
        assessment = await service.assess_content_quality(poor_quality_content)
        
        assert assessment.overall_score < 0.6  # Should be poor quality
        assert not assessment.is_publishable
        
        # Should have many issues
        assert len(assessment.issues) > 3
        
        # Should have urgent recommendations
        urgent_recommendations = [r for r in assessment.recommendations if "URGENT" in r]
        assert len(urgent_recommendations) > 0
    
    async def test_category_score_calculation(self, db_session, sample_content):
        """Test individual category score calculation."""
        service = ContentLifecycleService(db_session)
        
        category_scores = await service._calculate_category_scores(sample_content)
        
        # All categories should be present
        expected_categories = ["clarity", "accuracy", "completeness", "relevance", "engagement"]
        for category in expected_categories:
            assert category in category_scores
            assert 0.0 <= category_scores[category] <= 1.0
        
        # Good content should have decent scores
        assert category_scores["clarity"] > 0.5
        assert category_scores["completeness"] > 0.5


class TestDeprecationDetection:
    """Test deprecation detection functionality."""
    
    async def test_detect_deprecated_content(self, db_session, old_content, poor_quality_content):
        """Test detection of deprecated content."""
        service = ContentLifecycleService(db_session)
        
        alerts = await service.detect_deprecated_content(limit=10)
        
        assert isinstance(alerts, list)
        
        # Should detect the old content
        old_content_alerts = [a for a in alerts if a.content_id == old_content.id]
        assert len(old_content_alerts) > 0
        
        alert = old_content_alerts[0]
        assert isinstance(alert, DeprecationAlert)
        assert alert.confidence > 0.4
        assert "months" in alert.reason.lower() or "rating" in alert.reason.lower()
    
    async def test_assess_deprecation_risk(self, db_session, old_content):
        """Test individual content deprecation risk assessment."""
        service = ContentLifecycleService(db_session)
        cutoff_date = datetime.utcnow() - timedelta(days=365)
        
        alert = await service._assess_deprecation_risk(old_content, cutoff_date)
        
        assert alert is not None
        assert isinstance(alert, DeprecationAlert)
        assert alert.content_id == old_content.id
        assert alert.confidence > 0.0
        assert len(alert.reason) > 0
        assert len(alert.suggested_action) > 0


class TestUpdateSuggestions:
    """Test update suggestion functionality."""
    
    async def test_generate_update_suggestions(self, db_session, content_with_ratings):
        """Test generation of update suggestions."""
        service = ContentLifecycleService(db_session)
        
        suggestions = await service.generate_update_suggestions(content_with_ratings.id)
        
        assert isinstance(suggestions, list)
        
        if suggestions:  # May not have suggestions for good content
            suggestion = suggestions[0]
            assert isinstance(suggestion, UpdateSuggestion)
            assert suggestion.content_id == content_with_ratings.id
            assert 1 <= suggestion.priority <= 5
            assert suggestion.estimated_effort in ["low", "medium", "high"]
            assert len(suggestion.description) > 0
            assert len(suggestion.rationale) > 0
    
    async def test_analyze_user_feedback(self, db_session, content_with_ratings):
        """Test user feedback analysis for suggestions."""
        service = ContentLifecycleService(db_session)
        
        # Get content with ratings loaded
        content = await service._get_content_with_ratings(content_with_ratings.id)
        suggestions = await service._analyze_user_feedback(content)
        
        # Should generate suggestions based on negative feedback
        if suggestions:
            error_suggestions = [s for s in suggestions if "error" in s.suggestion_type]
            outdated_suggestions = [s for s in suggestions if "outdated" in s.suggestion_type]
            
            # Should detect issues mentioned in reviews
            assert len(error_suggestions) > 0 or len(outdated_suggestions) > 0
    
    async def test_analyze_content_freshness(self, db_session, old_content):
        """Test content freshness analysis."""
        service = ContentLifecycleService(db_session)
        
        suggestions = service._analyze_content_freshness(old_content)
        
        # Should suggest freshness update for old content
        assert len(suggestions) > 0
        
        freshness_suggestions = [s for s in suggestions if "freshness" in s.suggestion_type]
        assert len(freshness_suggestions) > 0
        
        suggestion = freshness_suggestions[0]
        assert suggestion.priority > 0
        assert "months old" in suggestion.rationale


class TestContentLifecycleIntegration:
    """Test integrated content lifecycle functionality."""
    
    async def test_update_content_quality_score(self, db_session, sample_content):
        """Test updating content quality score."""
        service = ContentLifecycleService(db_session)
        
        # Store original score
        original_score = sample_content.quality_score
        
        # Update quality score
        success = await service.update_content_quality_score(sample_content.id)
        
        assert success
        
        # Refresh content and check score was updated
        await db_session.refresh(sample_content)
        assert sample_content.quality_score != original_score
        assert 0.0 <= sample_content.quality_score <= 1.0
    
    async def test_bulk_quality_assessment(self, db_session, sample_content, poor_quality_content):
        """Test bulk quality assessment."""
        service = ContentLifecycleService(db_session)
        
        result = await service.bulk_quality_assessment(limit=10)
        
        assert isinstance(result, dict)
        assert "processed_count" in result
        assert "updated_count" in result
        assert "average_quality_score" in result
        assert "publishable_count" in result
        assert "assessments" in result
        
        assert result["processed_count"] >= 2  # At least our test content
        assert 0.0 <= result["average_quality_score"] <= 1.0
        assert len(result["assessments"]) == result["processed_count"]
    
    async def test_error_handling(self, db_session):
        """Test error handling in lifecycle service."""
        service = ContentLifecycleService(db_session)
        
        # Test with non-existent content
        suggestions = await service.generate_update_suggestions(uuid4())
        assert suggestions == []
        
        # Test quality score update with non-existent content
        success = await service.update_content_quality_score(uuid4())
        assert not success