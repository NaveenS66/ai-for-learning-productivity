"""Schemas for feedback integration."""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class FeedbackSubmissionRequest(BaseModel):
    """Request model for submitting feedback."""
    rating: int = Field(..., ge=1, le=5, description="Overall rating (1-5)")
    review_text: Optional[str] = Field(None, max_length=2000, description="Optional review text")
    categories: Optional[Dict[str, int]] = Field(None, description="Category-specific ratings")
    
    @validator('categories')
    def validate_categories(cls, v):
        if v is not None:
            valid_categories = {'clarity', 'usefulness', 'accuracy', 'difficulty'}
            for category, rating in v.items():
                if category not in valid_categories:
                    raise ValueError(f"Invalid category: {category}")
                if not (1 <= rating <= 5):
                    raise ValueError(f"Category rating must be between 1 and 5")
        return v


class FeedbackCollectionResponse(BaseModel):
    """Response model for feedback collection."""
    rating_id: UUID
    content_id: UUID
    user_id: UUID
    rating: int
    review_text: Optional[str]
    categories: Dict[str, Optional[int]]
    created_at: datetime
    updated_at: datetime


class FeedbackSummaryResponse(BaseModel):
    """Response model for feedback summary."""
    content_id: UUID
    total_ratings: int = Field(..., ge=0)
    average_rating: float = Field(..., ge=0.0, le=5.0)
    rating_distribution: Dict[int, int] = Field(..., description="Rating value to count mapping")
    total_reviews: int = Field(..., ge=0)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)")
    common_themes: List[str] = Field(..., description="Common themes in reviews")
    recent_feedback_trend: str = Field(..., description="Recent trend: improving, declining, stable")


class ContentConflictResponse(BaseModel):
    """Response model for content conflicts."""
    content_id: UUID
    conflict_type: str = Field(..., description="Type of conflict detected")
    severity: str = Field(..., description="Conflict severity level")
    description: str = Field(..., description="Description of the conflict")
    conflicting_ratings: List[UUID] = Field(..., description="IDs of conflicting ratings")
    suggested_resolution: str = Field(..., description="Suggested resolution approach")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in conflict detection")


class RankingUpdateResponse(BaseModel):
    """Response model for ranking updates."""
    content_id: UUID
    old_score: float = Field(..., ge=0.0, le=1.0)
    new_score: float = Field(..., ge=0.0, le=1.0)
    score_change: float = Field(..., description="Change in score")
    factors: Dict[str, float] = Field(..., description="Ranking factors and their values")
    rationale: str = Field(..., description="Explanation for the ranking change")


class FeedbackAnalyticsRequest(BaseModel):
    """Request model for feedback analytics."""
    content_ids: Optional[List[UUID]] = Field(None, description="Specific content IDs to analyze")
    date_from: Optional[datetime] = Field(None, description="Start date for analysis")
    date_to: Optional[datetime] = Field(None, description="End date for analysis")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_themes: bool = Field(default=True, description="Include theme extraction")
    include_trends: bool = Field(default=True, description="Include trend analysis")


class FeedbackAnalyticsResponse(BaseModel):
    """Response model for feedback analytics."""
    analysis_period: Dict[str, Optional[datetime]] = Field(..., description="Analysis time period")
    total_feedback_count: int = Field(..., ge=0)
    average_rating: float = Field(..., ge=0.0, le=5.0)
    rating_trends: List[Dict[str, any]] = Field(..., description="Rating trends over time")
    sentiment_analysis: Optional[Dict[str, float]] = Field(None, description="Sentiment analysis results")
    theme_analysis: Optional[Dict[str, int]] = Field(None, description="Theme frequency analysis")
    content_performance: List[Dict[str, any]] = Field(..., description="Individual content performance")


class ConflictResolutionRequest(BaseModel):
    """Request model for conflict resolution."""
    resolution_strategy: str = Field(..., description="Resolution strategy to use")
    additional_context: Optional[str] = Field(None, description="Additional context for resolution")
    notify_users: bool = Field(default=False, description="Whether to notify affected users")
    
    @validator('resolution_strategy')
    def validate_strategy(cls, v):
        valid_strategies = {'weighted_average', 'expert_review', 'community_vote', 'manual_review'}
        if v not in valid_strategies:
            raise ValueError(f"Invalid resolution strategy. Must be one of: {valid_strategies}")
        return v


class ConflictResolutionResponse(BaseModel):
    """Response model for conflict resolution."""
    conflict_id: str = Field(..., description="Identifier for the resolved conflict")
    content_id: UUID
    resolution_strategy: str
    resolution_status: str = Field(..., description="Status of resolution: resolved, pending, failed")
    resolution_details: Dict[str, any] = Field(..., description="Details of the resolution process")
    resolved_at: datetime
    resolved_by: UUID = Field(..., description="User who resolved the conflict")


class UserFeedbackStatsResponse(BaseModel):
    """Response model for user feedback statistics."""
    user_id: UUID
    total_ratings_given: int = Field(..., ge=0)
    average_rating_given: float = Field(..., ge=0.0, le=5.0)
    total_reviews_written: int = Field(..., ge=0)
    feedback_helpfulness_score: float = Field(..., ge=0.0, le=1.0, description="How helpful others find this user's feedback")
    most_active_categories: List[str] = Field(..., description="Categories user rates most frequently")
    feedback_consistency_score: float = Field(..., ge=0.0, le=1.0, description="Consistency of user's ratings")


class FeedbackModerationRequest(BaseModel):
    """Request model for feedback moderation."""
    rating_id: UUID
    moderation_action: str = Field(..., description="Moderation action to take")
    reason: str = Field(..., description="Reason for moderation")
    moderator_notes: Optional[str] = Field(None, description="Additional moderator notes")
    
    @validator('moderation_action')
    def validate_action(cls, v):
        valid_actions = {'approve', 'flag', 'hide', 'delete', 'edit'}
        if v not in valid_actions:
            raise ValueError(f"Invalid moderation action. Must be one of: {valid_actions}")
        return v


class FeedbackModerationResponse(BaseModel):
    """Response model for feedback moderation."""
    rating_id: UUID
    moderation_action: str
    moderation_status: str = Field(..., description="Status after moderation")
    moderated_at: datetime
    moderated_by: UUID
    reason: str
    previous_status: str = Field(..., description="Status before moderation")


class FeedbackQualityMetrics(BaseModel):
    """Metrics for feedback quality assessment."""
    content_id: UUID
    feedback_quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall feedback quality score")
    review_depth_score: float = Field(..., ge=0.0, le=1.0, description="Depth and detail of reviews")
    rating_consistency_score: float = Field(..., ge=0.0, le=1.0, description="Consistency of ratings")
    feedback_recency_score: float = Field(..., ge=0.0, le=1.0, description="Recency of feedback")
    user_credibility_score: float = Field(..., ge=0.0, le=1.0, description="Average credibility of reviewers")
    feedback_volume_score: float = Field(..., ge=0.0, le=1.0, description="Volume of feedback received")


class TrendingFeedbackResponse(BaseModel):
    """Response model for trending feedback."""
    period_days: int = Field(..., ge=1)
    trending_content: List[Dict[str, any]] = Field(..., description="Content trending by feedback activity")
    trending_themes: List[Dict[str, any]] = Field(default=[], description="Trending themes in feedback")
    sentiment_trends: Dict[str, float] = Field(default={}, description="Sentiment trends over the period")
    engagement_metrics: Dict[str, any] = Field(default={}, description="Engagement metrics for the period")