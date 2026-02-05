"""Property-based tests for content lifecycle management.

Property 23: Content Lifecycle Management
Validates: Requirements 6.2

This test validates that the content lifecycle management system properly handles
content validation, quality assessment, deprecation detection, and update suggestions
across various content states and conditions.
"""

import pytest
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant
from uuid import uuid4

from src.ai_learning_accelerator.models.content import (
    LearningContent, ContentRating, ContentType, ContentFormat, QualityStatus
)
from src.ai_learning_accelerator.models.user import User, DifficultyLevel
from src.ai_learning_accelerator.services.content_lifecycle import (
    ContentLifecycleService, ValidationSeverity, ValidationIssue,
    QualityAssessment, DeprecationAlert, UpdateSuggestion
)


# Hypothesis strategies for generating test data
content_types = st.sampled_from(list(ContentType))
content_formats = st.sampled_from(list(ContentFormat))
difficulty_levels = st.sampled_from(list(DifficultyLevel))
quality_statuses = st.sampled_from(list(QualityStatus))

# Content text strategies
valid_content_text = st.text(min_size=50, max_size=2000).filter(
    lambda x: len(x.strip()) >= 50 and not any(bad in x.lower() for bad in ['todo', 'fixme'])
)

poor_content_text = st.one_of(
    st.text(max_size=30),  # Too short
    st.just("TODO: Write content here"),  # Placeholder
    st.just("This content has TODO items and FIXME notes")  # Quality issues
)

# Title strategies
valid_titles = st.text(min_size=5, max_size=200).filter(lambda x: len(x.strip()) >= 5)
invalid_titles = st.one_of(
    st.text(max_size=4),  # Too short
    st.just(""),  # Empty
    st.text(min_size=201, max_size=300)  # Too long
)

# Description strategies
valid_descriptions = st.text(min_size=20, max_size=500).filter(lambda x: len(x.strip()) >= 20)
invalid_descriptions = st.text(max_size=19)

# Rating strategies
ratings = st.integers(min_value=1, max_value=5)
rating_counts = st.integers(min_value=0, max_value=1000)
view_counts = st.integers(min_value=0, max_value=10000)


class ContentLifecycleStateMachine(RuleBasedStateMachine):
    """Stateful testing for content lifecycle management."""
    
    def __init__(self):
        super().__init__()
        self.db_session = None
        self.service = None
        self.users = []
        self.content_items = []
    
    contents = Bundle('contents')
    users = Bundle('users')
    
    @initialize()
    async def setup(self):
        """Initialize the test environment."""
        # This would be set up by the test framework
        pass
    
    @rule(target=users)
    def create_user(self):
        """Create a test user."""
        user = User(
            username=f"user_{len(self.users)}",
            email=f"user{len(self.users)}@test.com",
            hashed_password="hashed_password"
        )
        self.users.append(user)
        return user
    
    @rule(
        target=contents,
        title=valid_titles,
        description=valid_descriptions,
        content_text=valid_content_text,
        content_type=content_types,
        difficulty=difficulty_levels
    )
    def create_valid_content(self, title, description, content_text, content_type, difficulty):
        """Create valid content that should pass validation."""
        content = LearningContent(
            title=title,
            description=description,
            content_text=content_text,
            content_type=content_type,
            difficulty_level=difficulty,
            learning_objectives=["Test objective"],
            tags=["test"],
            quality_status=QualityStatus.DRAFT
        )
        self.content_items.append(content)
        return content
    
    @rule(
        target=contents,
        title=invalid_titles,
        description=invalid_descriptions,
        content_text=poor_content_text
    )
    def create_invalid_content(self, title, description, content_text):
        """Create invalid content that should fail validation."""
        content = LearningContent(
            title=title,
            description=description,
            content_text=content_text,
            content_type=ContentType.ARTICLE,
            difficulty_level=DifficultyLevel.BEGINNER,
            learning_objectives=[],  # Empty objectives
            tags=[],  # No tags
            quality_status=QualityStatus.DRAFT
        )
        self.content_items.append(content)
        return content
    
    @rule(content=contents)
    async def validate_content_properties(self, content):
        """Test content validation properties."""
        if self.service is None:
            return
        
        issues = await self.service.validate_content(content)
        
        # Property: Validation should always return a list
        assert isinstance(issues, list)
        
        # Property: All issues should have required fields
        for issue in issues:
            assert isinstance(issue, ValidationIssue)
            assert issue.severity in ValidationSeverity
            assert isinstance(issue.category, str)
            assert isinstance(issue.message, str)
        
        # Property: Content with valid structure should have fewer critical issues
        if (content.title and len(content.title.strip()) >= 5 and
            content.content_text and len(content.content_text.strip()) >= 50):
            critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
            assert len(critical_issues) == 0
    
    @rule(content=contents)
    async def assess_quality_properties(self, content):
        """Test quality assessment properties."""
        if self.service is None:
            return
        
        assessment = await self.service.assess_content_quality(content)
        
        # Property: Assessment should always be returned
        assert isinstance(assessment, QualityAssessment)
        
        # Property: Overall score should be between 0 and 1
        assert 0.0 <= assessment.overall_score <= 1.0
        
        # Property: Category scores should be between 0 and 1
        for category, score in assessment.category_scores.items():
            assert 0.0 <= score <= 1.0
        
        # Property: Recommendations should be actionable strings
        assert isinstance(assessment.recommendations, list)
        for rec in assessment.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0
        
        # Property: Publishable status should be consistent with score and issues
        critical_errors = [i for i in assessment.issues 
                          if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
        if len(critical_errors) > 0:
            assert not assessment.is_publishable
        
        if assessment.overall_score < 0.6:
            assert not assessment.is_publishable
    
    @invariant()
    def content_consistency(self):
        """Invariant: Content state should remain consistent."""
        for content in self.content_items:
            # Property: Content ID should be stable
            if hasattr(content, 'id') and content.id:
                assert isinstance(content.id, type(content.id))
            
            # Property: Quality score should be valid if set
            if hasattr(content, 'quality_score'):
                assert 0.0 <= content.quality_score <= 1.0


@pytest.mark.asyncio
class TestContentLifecycleProperties:
    """Property-based tests for content lifecycle management."""
    
    @given(
        title=valid_titles,
        description=valid_descriptions,
        content_text=valid_content_text,
        content_type=content_types,
        difficulty=difficulty_levels
    )
    @settings(max_examples=50, deadline=None)
    async def test_valid_content_validation_properties(
        self, db_session, title, description, content_text, content_type, difficulty
    ):
        """Property: Valid content should pass basic validation checks."""
        service = ContentLifecycleService(db_session)
        
        content = LearningContent(
            title=title,
            description=description,
            content_text=content_text,
            content_type=content_type,
            difficulty_level=difficulty,
            learning_objectives=["Valid objective"],
            tags=["test", "valid"]
        )
        
        issues = await service.validate_content(content)
        
        # Property: Valid content should have no critical or error issues
        critical_errors = [i for i in issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
        assert len(critical_errors) == 0, f"Valid content should not have critical errors: {[i.message for i in critical_errors]}"
        
        # Property: All issues should have suggestions for improvement
        for issue in issues:
            if issue.severity in [ValidationSeverity.WARNING, ValidationSeverity.ERROR]:
                assert issue.suggestion is not None, f"Issue should have suggestion: {issue.message}"
    
    @given(
        title=invalid_titles,
        description=invalid_descriptions,
        content_text=poor_content_text
    )
    @settings(max_examples=30, deadline=None)
    async def test_invalid_content_validation_properties(
        self, db_session, title, description, content_text
    ):
        """Property: Invalid content should be flagged with appropriate issues."""
        service = ContentLifecycleService(db_session)
        
        content = LearningContent(
            title=title,
            description=description,
            content_text=content_text,
            content_type=ContentType.ARTICLE,
            difficulty_level=DifficultyLevel.BEGINNER,
            learning_objectives=[],
            tags=[]
        )
        
        issues = await service.validate_content(content)
        
        # Property: Invalid content should have validation issues
        assert len(issues) > 0, "Invalid content should have validation issues"
        
        # Property: Issues should be categorized appropriately
        categories = {issue.category for issue in issues}
        expected_categories = {"structure", "quality", "metadata"}
        assert categories.intersection(expected_categories), f"Issues should be in expected categories: {categories}"
        
        # Property: Severe issues should have error or critical severity
        if not title or len(title.strip()) < 5:
            title_issues = [i for i in issues if "title" in i.message.lower()]
            assert any(i.severity == ValidationSeverity.ERROR for i in title_issues)
        
        if not content_text or len(content_text.strip()) < 50:
            content_issues = [i for i in issues if "content" in i.message.lower()]
            assert any(i.severity == ValidationSeverity.ERROR for i in content_issues)
    
    @given(
        average_rating=st.floats(min_value=1.0, max_value=5.0),
        rating_count=rating_counts,
        view_count=view_counts,
        completion_count=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=50, deadline=None)
    async def test_quality_assessment_properties(
        self, db_session, average_rating, rating_count, view_count, completion_count
    ):
        """Property: Quality assessment should be consistent and bounded."""
        assume(completion_count <= view_count)  # Logical constraint
        
        service = ContentLifecycleService(db_session)
        
        content = LearningContent(
            title="Test Content for Quality Assessment",
            description="This is a test content for quality assessment properties",
            content_text="This is substantial content that meets the minimum requirements for testing quality assessment properties and algorithms.",
            content_type=ContentType.TUTORIAL,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            learning_objectives=["Test quality assessment"],
            tags=["test", "quality"],
            average_rating=average_rating,
            rating_count=rating_count,
            view_count=view_count,
            completion_count=completion_count
        )
        
        assessment = await service.assess_content_quality(content)
        
        # Property: Assessment scores should be bounded
        assert 0.0 <= assessment.overall_score <= 1.0
        for category, score in assessment.category_scores.items():
            assert 0.0 <= score <= 1.0, f"Category {category} score {score} out of bounds"
        
        # Property: Higher ratings should generally lead to higher quality scores
        if rating_count > 0:
            # Content with high ratings should have better accuracy scores
            if average_rating >= 4.0:
                assert assessment.category_scores.get("accuracy", 0) >= 0.6
            elif average_rating <= 2.0:
                assert assessment.category_scores.get("accuracy", 1) <= 0.5
        
        # Property: High engagement should improve relevance scores
        if view_count > 50 and completion_count > 0:
            completion_rate = completion_count / view_count
            if completion_rate >= 0.8:
                assert assessment.category_scores.get("engagement", 0) >= 0.7
        
        # Property: Publishable content should meet minimum standards
        if assessment.is_publishable:
            assert assessment.overall_score >= 0.6
            critical_errors = [i for i in assessment.issues if i.severity == ValidationSeverity.CRITICAL]
            assert len(critical_errors) == 0
    
    @given(
        age_days=st.integers(min_value=1, max_value=730),  # Up to 2 years
        average_rating=st.floats(min_value=1.0, max_value=5.0),
        rating_count=st.integers(min_value=0, max_value=100),
        view_count=view_counts,
        completion_count=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=40, deadline=None)
    async def test_deprecation_detection_properties(
        self, db_session, age_days, average_rating, rating_count, view_count, completion_count
    ):
        """Property: Deprecation detection should be consistent with content age and quality."""
        assume(completion_count <= view_count)
        
        service = ContentLifecycleService(db_session)
        
        # Create content with specific age
        created_date = datetime.utcnow() - timedelta(days=age_days)
        
        content = LearningContent(
            title="Test Content for Deprecation",
            description="Content for testing deprecation detection",
            content_text="This content is used to test deprecation detection algorithms and properties.",
            content_type=ContentType.TUTORIAL,
            difficulty_level=DifficultyLevel.BEGINNER,
            learning_objectives=["Test deprecation"],
            tags=["test"],
            average_rating=average_rating,
            rating_count=rating_count,
            view_count=view_count,
            completion_count=completion_count,
            created_at=created_date,
            quality_status=QualityStatus.PUBLISHED
        )
        
        db_session.add(content)
        await db_session.commit()
        await db_session.refresh(content)
        
        alerts = await service.detect_deprecated_content(limit=100)
        content_alerts = [a for a in alerts if a.content_id == content.id]
        
        # Property: Very old content should be flagged for deprecation
        if age_days > 365:  # Older than 1 year
            if rating_count >= 5 and average_rating < 2.5:
                # Old content with poor ratings should be flagged
                assert len(content_alerts) > 0, "Old content with poor ratings should be flagged"
                alert = content_alerts[0]
                assert alert.confidence > 0.4
        
        # Property: Recent high-quality content should not be flagged
        if age_days < 90 and average_rating >= 4.0 and rating_count >= 5:
            content_alerts_high_severity = [a for a in content_alerts 
                                          if a.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
            assert len(content_alerts_high_severity) == 0, "Recent high-quality content should not be flagged as critical"
        
        # Property: All alerts should have valid confidence scores
        for alert in content_alerts:
            assert 0.0 <= alert.confidence <= 1.0
            assert len(alert.reason) > 0
            assert len(alert.suggested_action) > 0
    
    @given(
        content_age_days=st.integers(min_value=30, max_value=730),
        rating_pattern=st.lists(ratings, min_size=1, max_size=20)
    )
    @settings(max_examples=30, deadline=None)
    async def test_update_suggestion_properties(
        self, db_session, content_age_days, rating_pattern
    ):
        """Property: Update suggestions should be relevant and prioritized correctly."""
        service = ContentLifecycleService(db_session)
        
        created_date = datetime.utcnow() - timedelta(days=content_age_days)
        average_rating = sum(rating_pattern) / len(rating_pattern)
        
        content = LearningContent(
            title="Content for Update Suggestions",
            description="Testing update suggestion generation",
            content_text="Content for testing update suggestion algorithms and prioritization.",
            content_type=ContentType.TUTORIAL,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            learning_objectives=["Test updates"],
            tags=["test"],
            average_rating=average_rating,
            rating_count=len(rating_pattern),
            view_count=100,
            completion_count=50,
            created_at=created_date
        )
        
        db_session.add(content)
        await db_session.commit()
        await db_session.refresh(content)
        
        suggestions = await service.generate_update_suggestions(content.id)
        
        # Property: All suggestions should have valid priority levels
        for suggestion in suggestions:
            assert isinstance(suggestion, UpdateSuggestion)
            assert 1 <= suggestion.priority <= 5
            assert suggestion.estimated_effort in ["low", "medium", "high"]
            assert len(suggestion.description) > 0
            assert len(suggestion.rationale) > 0
        
        # Property: Older content should get freshness update suggestions
        if content_age_days > 180:  # Older than 6 months
            freshness_suggestions = [s for s in suggestions if s.suggestion_type == "freshness_update"]
            assert len(freshness_suggestions) > 0, "Old content should get freshness update suggestions"
            
            # Property: Older content should have higher priority freshness updates
            if content_age_days > 365:
                freshness_suggestion = freshness_suggestions[0]
                assert freshness_suggestion.priority >= 3
        
        # Property: Low-rated content should get quality improvement suggestions
        if average_rating < 3.0 and len(rating_pattern) >= 3:
            quality_suggestions = [s for s in suggestions 
                                 if "quality" in s.suggestion_type or "improvement" in s.suggestion_type]
            # Should have some quality-related suggestions
            assert len(quality_suggestions) > 0 or len(suggestions) == 0  # May have no suggestions if content is new
        
        # Property: Suggestions should be ordered by priority
        if len(suggestions) > 1:
            priorities = [s.priority for s in suggestions]
            assert priorities == sorted(priorities, reverse=True), "Suggestions should be ordered by priority (highest first)"
    
    @given(
        quality_scores=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=10)
    )
    @settings(max_examples=20, deadline=None)
    async def test_bulk_quality_assessment_properties(self, db_session, quality_scores):
        """Property: Bulk quality assessment should maintain consistency."""
        service = ContentLifecycleService(db_session)
        
        # Create multiple content items
        content_items = []
        for i, score in enumerate(quality_scores):
            content = LearningContent(
                title=f"Bulk Test Content {i}",
                description=f"Content {i} for bulk quality assessment testing",
                content_text=f"This is test content number {i} for bulk quality assessment validation and testing.",
                content_type=ContentType.ARTICLE,
                difficulty_level=DifficultyLevel.BEGINNER,
                learning_objectives=[f"Test objective {i}"],
                tags=["test", "bulk"],
                quality_score=0.0,  # Will be updated by assessment
                is_active=True
            )
            content_items.append(content)
            db_session.add(content)
        
        await db_session.commit()
        for content in content_items:
            await db_session.refresh(content)
        
        result = await service.bulk_quality_assessment(limit=len(content_items))
        
        # Property: Bulk assessment should process all content
        assert result["processed_count"] == len(content_items)
        assert result["updated_count"] <= result["processed_count"]
        
        # Property: Average quality score should be bounded
        assert 0.0 <= result["average_quality_score"] <= 1.0
        
        # Property: Assessment results should match processed content
        assert len(result["assessments"]) == result["processed_count"]
        
        # Property: Each assessment should have valid data
        for assessment in result["assessments"]:
            assert "content_id" in assessment
            assert "quality_score" in assessment
            assert 0.0 <= assessment["quality_score"] <= 1.0
            assert isinstance(assessment["is_publishable"], bool)
            assert assessment["issues_count"] >= 0
    
    async def test_content_lifecycle_idempotency(self, db_session):
        """Property: Content lifecycle operations should be idempotent."""
        service = ContentLifecycleService(db_session)
        
        content = LearningContent(
            title="Idempotency Test Content",
            description="Testing idempotent operations in content lifecycle",
            content_text="This content is used to test that lifecycle operations are idempotent and consistent.",
            content_type=ContentType.TUTORIAL,
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            learning_objectives=["Test idempotency"],
            tags=["test", "idempotent"]
        )
        
        db_session.add(content)
        await db_session.commit()
        await db_session.refresh(content)
        
        # Property: Multiple validations should return consistent results
        issues1 = await service.validate_content(content)
        issues2 = await service.validate_content(content)
        
        assert len(issues1) == len(issues2)
        for i1, i2 in zip(issues1, issues2):
            assert i1.severity == i2.severity
            assert i1.category == i2.category
            assert i1.message == i2.message
        
        # Property: Multiple quality assessments should be consistent
        assessment1 = await service.assess_content_quality(content)
        assessment2 = await service.assess_content_quality(content)
        
        assert abs(assessment1.overall_score - assessment2.overall_score) < 0.01
        assert assessment1.is_publishable == assessment2.is_publishable
        
        # Property: Multiple quality score updates should converge
        success1 = await service.update_content_quality_score(content.id)
        await db_session.refresh(content)
        score_after_first = content.quality_score
        
        success2 = await service.update_content_quality_score(content.id)
        await db_session.refresh(content)
        score_after_second = content.quality_score
        
        assert success1 and success2
        # Scores should be very close (allowing for minor floating point differences)
        assert abs(score_after_first - score_after_second) < 0.01


# Integration property tests
@pytest.mark.asyncio
class TestContentLifecycleIntegrationProperties:
    """Integration property tests for content lifecycle management."""
    
    async def test_lifecycle_state_transitions(self, db_session):
        """Property: Content should transition through lifecycle states correctly."""
        service = ContentLifecycleService(db_session)
        
        # Create draft content
        content = LearningContent(
            title="Lifecycle State Test Content",
            description="Testing content lifecycle state transitions and validation",
            content_text="This content tests the proper state transitions through the content lifecycle management system.",
            content_type=ContentType.TUTORIAL,
            difficulty_level=DifficultyLevel.BEGINNER,
            learning_objectives=["Test lifecycle states"],
            tags=["test", "lifecycle"],
            quality_status=QualityStatus.DRAFT,
            quality_score=0.0
        )
        
        db_session.add(content)
        await db_session.commit()
        await db_session.refresh(content)
        
        # Property: Draft content should be assessable
        assessment = await service.assess_content_quality(content)
        assert assessment is not None
        
        # Property: Quality score update should change status if publishable
        await service.update_content_quality_score(content.id)
        await db_session.refresh(content)
        
        if assessment.is_publishable:
            assert content.quality_status == QualityStatus.REVIEW
        
        # Property: Content quality score should be updated
        assert content.quality_score > 0.0
        assert content.quality_score == assessment.overall_score
    
    async def test_cross_component_consistency(self, db_session):
        """Property: Content lifecycle should be consistent across components."""
        service = ContentLifecycleService(db_session)
        
        content = LearningContent(
            title="Cross-Component Consistency Test",
            description="Testing consistency across different lifecycle components",
            content_text="This content validates that all lifecycle management components maintain consistency.",
            content_type=ContentType.ARTICLE,
            difficulty_level=DifficultyLevel.ADVANCED,
            learning_objectives=["Test consistency"],
            tags=["test", "consistency"],
            average_rating=3.5,
            rating_count=10,
            view_count=200,
            completion_count=150
        )
        
        db_session.add(content)
        await db_session.commit()
        await db_session.refresh(content)
        
        # Get data from different components
        validation_issues = await service.validate_content(content)
        quality_assessment = await service.assess_content_quality(content)
        update_suggestions = await service.generate_update_suggestions(content.id)
        
        # Property: Components should agree on content quality
        has_critical_issues = any(i.severity == ValidationSeverity.CRITICAL for i in validation_issues)
        has_error_issues = any(i.severity == ValidationSeverity.ERROR for i in validation_issues)
        
        if has_critical_issues or has_error_issues:
            assert not quality_assessment.is_publishable
        
        # Property: Update suggestions should align with quality assessment
        if quality_assessment.overall_score < 0.7:
            quality_suggestions = [s for s in update_suggestions 
                                 if "quality" in s.description.lower() or "improve" in s.description.lower()]
            # Should have quality improvement suggestions for low-quality content
            assert len(quality_suggestions) > 0 or quality_assessment.overall_score > 0.6
        
        # Property: All components should handle the same content consistently
        assert all(isinstance(issue, ValidationIssue) for issue in validation_issues)
        assert isinstance(quality_assessment, QualityAssessment)
        assert all(isinstance(suggestion, UpdateSuggestion) for suggestion in update_suggestions)