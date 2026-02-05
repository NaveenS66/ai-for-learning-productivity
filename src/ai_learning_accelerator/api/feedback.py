"""API endpoints for feedback integration."""

from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..services.feedback_integration import FeedbackIntegrationService
from ..services.content import ContentService
from ..schemas.feedback import (
    FeedbackSubmissionRequest,
    FeedbackSummaryResponse,
    ContentConflictResponse,
    RankingUpdateResponse,
    FeedbackCollectionResponse
)
from ..auth.dependencies import get_current_user
from ..models.user import User
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/submit/{content_id}", response_model=FeedbackCollectionResponse)
async def submit_feedback(
    content_id: UUID,
    feedback_request: FeedbackSubmissionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit feedback (rating and review) for content."""
    try:
        content_service = ContentService(db)
        feedback_service = FeedbackIntegrationService(db)
        
        # Verify content exists
        content = await content_service.get_learning_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Collect feedback
        rating = await feedback_service.collect_feedback(
            content_id=content_id,
            user_id=current_user.id,
            rating=feedback_request.rating,
            review_text=feedback_request.review_text,
            feedback_categories=feedback_request.categories
        )
        
        return FeedbackCollectionResponse(
            rating_id=rating.id,
            content_id=content_id,
            user_id=current_user.id,
            rating=rating.rating,
            review_text=rating.review_text,
            categories={
                "clarity": rating.clarity_rating,
                "usefulness": rating.usefulness_rating,
                "accuracy": rating.accuracy_rating,
                "difficulty": rating.difficulty_rating
            },
            created_at=rating.created_at,
            updated_at=rating.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error submitting feedback", error=str(e), content_id=str(content_id))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/summary/{content_id}", response_model=FeedbackSummaryResponse)
async def get_feedback_summary(
    content_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive feedback summary for content."""
    try:
        content_service = ContentService(db)
        feedback_service = FeedbackIntegrationService(db)
        
        # Verify content exists
        content = await content_service.get_learning_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Get feedback summary
        summary = await feedback_service.get_feedback_summary(content_id)
        
        return FeedbackSummaryResponse(
            content_id=summary.content_id,
            total_ratings=summary.total_ratings,
            average_rating=summary.average_rating,
            rating_distribution=summary.rating_distribution,
            total_reviews=summary.total_reviews,
            sentiment_score=summary.sentiment_score,
            common_themes=summary.common_themes,
            recent_feedback_trend=summary.recent_feedback_trend
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting feedback summary", error=str(e), content_id=str(content_id))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/conflicts", response_model=List[ContentConflictResponse])
async def get_content_conflicts(
    content_id: Optional[UUID] = Query(None, description="Specific content ID to check"),
    severity: Optional[str] = Query(None, description="Filter by conflict severity"),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get content conflicts that need resolution."""
    try:
        # Check if user has admin privileges for viewing conflicts
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        feedback_service = FeedbackIntegrationService(db)
        
        # Detect conflicts
        conflicts = await feedback_service.detect_content_conflicts(content_id)
        
        # Filter by severity if specified
        if severity:
            conflicts = [c for c in conflicts if c.severity.value == severity.lower()]
        
        # Limit results
        conflicts = conflicts[:limit]
        
        return [
            ContentConflictResponse(
                content_id=conflict.content_id,
                conflict_type=conflict.conflict_type,
                severity=conflict.severity.value,
                description=conflict.description,
                conflicting_ratings=conflict.conflicting_ratings,
                suggested_resolution=conflict.suggested_resolution,
                confidence=conflict.confidence
            )
            for conflict in conflicts
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting content conflicts", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/resolve-conflict/{content_id}")
async def resolve_content_conflict(
    content_id: UUID,
    resolution_strategy: str = Query(..., description="Resolution strategy: weighted_average, expert_review, community_vote"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Resolve a content conflict using specified strategy."""
    try:
        # Check if user has admin privileges
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        feedback_service = FeedbackIntegrationService(db)
        
        # Get conflicts for the content
        conflicts = await feedback_service.detect_content_conflicts(content_id)
        
        if not conflicts:
            raise HTTPException(status_code=404, detail="No conflicts found for this content")
        
        # Resolve the first conflict (in a real system, you might want to specify which conflict)
        conflict = conflicts[0]
        success = await feedback_service.resolve_content_conflict(conflict, resolution_strategy)
        
        if success:
            return {
                "message": "Conflict resolved successfully",
                "content_id": str(content_id),
                "strategy": resolution_strategy
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to resolve conflict")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error resolving content conflict", error=str(e), content_id=str(content_id))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/update-rankings", response_model=Dict[str, Any])
async def update_content_rankings(
    batch_size: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update content rankings based on feedback."""
    try:
        # Check if user has admin privileges
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        feedback_service = FeedbackIntegrationService(db)
        
        # Update rankings
        result = await feedback_service.update_content_rankings(batch_size)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating content rankings", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/user-feedback/{user_id}")
async def get_user_feedback_history(
    user_id: UUID,
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's feedback history."""
    try:
        # Users can only view their own feedback, admins can view any
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Access denied")
        
        from sqlalchemy import select
        from ..models.content import ContentRating
        
        # Get user's ratings
        stmt = select(ContentRating).where(
            ContentRating.user_id == user_id
        ).order_by(ContentRating.created_at.desc()).limit(limit)
        
        result = await db.execute(stmt)
        ratings = result.scalars().all()
        
        return [
            {
                "rating_id": str(rating.id),
                "content_id": str(rating.content_id),
                "rating": rating.rating,
                "review_text": rating.review_text,
                "categories": {
                    "clarity": rating.clarity_rating,
                    "usefulness": rating.usefulness_rating,
                    "accuracy": rating.accuracy_rating,
                    "difficulty": rating.difficulty_rating
                },
                "created_at": rating.created_at,
                "updated_at": rating.updated_at
            }
            for rating in ratings
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting user feedback history", error=str(e), user_id=str(user_id))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/trending-feedback")
async def get_trending_feedback(
    days: int = Query(default=7, ge=1, le=30),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get trending feedback and content based on recent activity."""
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import select, func, desc
        from ..models.content import ContentRating, LearningContent
        
        # Get content with most recent feedback
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        stmt = select(
            ContentRating.content_id,
            func.count(ContentRating.id).label('recent_ratings'),
            func.avg(ContentRating.rating).label('avg_rating')
        ).where(
            ContentRating.created_at >= cutoff_date
        ).group_by(
            ContentRating.content_id
        ).order_by(
            desc('recent_ratings')
        ).limit(limit)
        
        result = await db.execute(stmt)
        trending_data = result.fetchall()
        
        # Get content details
        trending_content = []
        for row in trending_data:
            content_stmt = select(LearningContent).where(LearningContent.id == row.content_id)
            content_result = await db.execute(content_stmt)
            content = content_result.scalar_one_or_none()
            
            if content:
                trending_content.append({
                    "content_id": str(content.id),
                    "title": content.title,
                    "recent_ratings": row.recent_ratings,
                    "recent_average_rating": float(row.avg_rating),
                    "overall_average_rating": content.average_rating,
                    "total_ratings": content.rating_count
                })
        
        return {
            "period_days": days,
            "trending_content": trending_content
        }
        
    except Exception as e:
        logger.error("Error getting trending feedback", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")