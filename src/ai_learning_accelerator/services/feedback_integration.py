"""Feedback integration service for rating and review management."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc, update
from sqlalchemy.orm import selectinload

from ..models.content import LearningContent, ContentRating, QualityStatus
from ..models.user import User
from ..logging_config import get_logger

logger = get_logger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback."""
    RATING = "rating"
    REVIEW = "review"
    REPORT = "report"
    SUGGESTION = "suggestion"


class ConflictSeverity(str, Enum):
    """Severity levels for content conflicts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeedbackSummary:
    """Summary of feedback for content."""
    content_id: UUID
    total_ratings: int
    average_rating: float
    rating_distribution: Dict[int, int]  # rating -> count
    total_reviews: int
    sentiment_score: float  # -1.0 to 1.0
    common_themes: List[str]
    recent_feedback_trend: str  # "improving", "declining", "stable"


@dataclass
class ContentConflict:
    """Represents a conflict in content feedback."""
    content_id: UUID
    conflict_type: str
    severity: ConflictSeverity
    description: str
    conflicting_ratings: List[UUID]  # Rating IDs involved in conflict
    suggested_resolution: str
    confidence: float  # 0.0 to 1.0


@dataclass
class RankingUpdate:
    """Represents a ranking update based on feedback."""
    content_id: UUID
    old_score: float
    new_score: float
    factors: Dict[str, float]  # factor -> weight
    rationale: str


class FeedbackIntegrationService:
    """Service for feedback integration and content ranking."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
        # Ranking weights for different feedback factors
        self.ranking_weights = {
            "average_rating": 0.30,
            "rating_count": 0.20,
            "review_sentiment": 0.15,
            "recency": 0.15,
            "user_credibility": 0.10,
            "engagement": 0.10
        }
        
        # Conflict detection thresholds
        self.conflict_thresholds = {
            "rating_variance": 2.0,  # High variance in ratings
            "sentiment_conflict": 0.5,  # Conflicting sentiment scores
            "accuracy_disputes": 3,  # Number of accuracy disputes
            "credibility_gap": 0.3  # Gap in user credibility scores
        }
    
    async def collect_feedback(
        self, 
        content_id: UUID, 
        user_id: UUID, 
        rating: int,
        review_text: Optional[str] = None,
        feedback_categories: Optional[Dict[str, int]] = None
    ) -> ContentRating:
        """Collect and process user feedback."""
        try:
            # Check if user already rated this content
            existing_rating = await self._get_user_rating(content_id, user_id)
            
            if existing_rating:
                # Update existing rating
                existing_rating.rating = rating
                existing_rating.review_text = review_text
                existing_rating.updated_at = datetime.utcnow()
                
                # Update category ratings if provided
                if feedback_categories:
                    existing_rating.clarity_rating = feedback_categories.get("clarity")
                    existing_rating.usefulness_rating = feedback_categories.get("usefulness")
                    existing_rating.accuracy_rating = feedback_categories.get("accuracy")
                    existing_rating.difficulty_rating = feedback_categories.get("difficulty")
                
                await self.db.commit()
                await self.db.refresh(existing_rating)
                
                logger.info("Feedback updated", content_id=str(content_id), user_id=str(user_id))
                feedback_rating = existing_rating
            else:
                # Create new rating
                new_rating = ContentRating(
                    content_id=content_id,
                    user_id=user_id,
                    rating=rating,
                    review_text=review_text,
                    clarity_rating=feedback_categories.get("clarity") if feedback_categories else None,
                    usefulness_rating=feedback_categories.get("usefulness") if feedback_categories else None,
                    accuracy_rating=feedback_categories.get("accuracy") if feedback_categories else None,
                    difficulty_rating=feedback_categories.get("difficulty") if feedback_categories else None
                )
                
                self.db.add(new_rating)
                await self.db.commit()
                await self.db.refresh(new_rating)
                
                logger.info("New feedback collected", content_id=str(content_id), user_id=str(user_id))
                feedback_rating = new_rating
            
            # Update content statistics
            await self._update_content_statistics(content_id)
            
            # Check for conflicts
            await self._detect_feedback_conflicts(content_id)
            
            # Update content ranking
            await self._update_content_ranking(content_id)
            
            return feedback_rating
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error collecting feedback", error=str(e), content_id=str(content_id))
            raise
    
    async def get_feedback_summary(self, content_id: UUID) -> FeedbackSummary:
        """Get comprehensive feedback summary for content."""
        try:
            # Get all ratings for content
            stmt = select(ContentRating).where(ContentRating.content_id == content_id)
            result = await self.db.execute(stmt)
            ratings = result.scalars().all()
            
            if not ratings:
                return FeedbackSummary(
                    content_id=content_id,
                    total_ratings=0,
                    average_rating=0.0,
                    rating_distribution={},
                    total_reviews=0,
                    sentiment_score=0.0,
                    common_themes=[],
                    recent_feedback_trend="stable"
                )
            
            # Calculate basic statistics
            total_ratings = len(ratings)
            average_rating = sum(r.rating for r in ratings) / total_ratings
            
            # Rating distribution
            rating_distribution = {}
            for rating in range(1, 6):
                rating_distribution[rating] = sum(1 for r in ratings if r.rating == rating)
            
            # Review statistics
            reviews = [r for r in ratings if r.review_text]
            total_reviews = len(reviews)
            
            # Sentiment analysis (simplified)
            sentiment_score = await self._calculate_sentiment_score(reviews)
            
            # Common themes extraction
            common_themes = await self._extract_common_themes(reviews)
            
            # Recent trend analysis
            recent_trend = await self._analyze_feedback_trend(content_id)
            
            return FeedbackSummary(
                content_id=content_id,
                total_ratings=total_ratings,
                average_rating=average_rating,
                rating_distribution=rating_distribution,
                total_reviews=total_reviews,
                sentiment_score=sentiment_score,
                common_themes=common_themes,
                recent_feedback_trend=recent_trend
            )
            
        except Exception as e:
            logger.error("Error getting feedback summary", error=str(e), content_id=str(content_id))
            return FeedbackSummary(
                content_id=content_id,
                total_ratings=0,
                average_rating=0.0,
                rating_distribution={},
                total_reviews=0,
                sentiment_score=0.0,
                common_themes=[],
                recent_feedback_trend="stable"
            )
    
    async def detect_content_conflicts(self, content_id: Optional[UUID] = None) -> List[ContentConflict]:
        """Detect conflicts in content feedback."""
        conflicts = []
        
        try:
            # Get content to analyze
            if content_id:
                content_ids = [content_id]
            else:
                # Get all active content with sufficient feedback
                stmt = select(LearningContent.id).where(
                    and_(
                        LearningContent.is_active == True,
                        LearningContent.rating_count >= 5
                    )
                )
                result = await self.db.execute(stmt)
                content_ids = [row[0] for row in result.fetchall()]
            
            for cid in content_ids:
                content_conflicts = await self._analyze_content_conflicts(cid)
                conflicts.extend(content_conflicts)
            
            logger.info("Conflict detection completed", conflicts_found=len(conflicts))
            return conflicts
            
        except Exception as e:
            logger.error("Error detecting content conflicts", error=str(e))
            return []
    
    async def resolve_content_conflict(
        self, 
        conflict: ContentConflict, 
        resolution_strategy: str = "weighted_average"
    ) -> bool:
        """Resolve a content conflict using specified strategy."""
        try:
            if resolution_strategy == "weighted_average":
                success = await self._resolve_with_weighted_average(conflict)
            elif resolution_strategy == "expert_review":
                success = await self._flag_for_expert_review(conflict)
            elif resolution_strategy == "community_vote":
                success = await self._initiate_community_vote(conflict)
            else:
                logger.warning("Unknown resolution strategy", strategy=resolution_strategy)
                return False
            
            if success:
                logger.info("Conflict resolved", content_id=str(conflict.content_id), strategy=resolution_strategy)
            
            return success
            
        except Exception as e:
            logger.error("Error resolving conflict", error=str(e), content_id=str(conflict.content_id))
            return False
    
    async def update_content_rankings(self, batch_size: int = 50) -> Dict[str, Any]:
        """Update content rankings based on feedback."""
        try:
            # Get content that needs ranking updates
            stmt = select(LearningContent).where(
                and_(
                    LearningContent.is_active == True,
                    LearningContent.rating_count > 0
                )
            ).limit(batch_size)
            
            result = await self.db.execute(stmt)
            content_list = result.scalars().all()
            
            updates = []
            updated_count = 0
            
            for content in content_list:
                ranking_update = await self._calculate_new_ranking(content)
                if ranking_update:
                    updates.append(ranking_update)
                    
                    # Apply the ranking update
                    if await self._apply_ranking_update(ranking_update):
                        updated_count += 1
            
            summary = {
                "processed_count": len(content_list),
                "updated_count": updated_count,
                "average_score_change": sum(u.new_score - u.old_score for u in updates) / max(len(updates), 1),
                "updates": [
                    {
                        "content_id": str(u.content_id),
                        "old_score": u.old_score,
                        "new_score": u.new_score,
                        "change": u.new_score - u.old_score
                    }
                    for u in updates
                ]
            }
            
            logger.info("Content rankings updated", **summary)
            return summary
            
        except Exception as e:
            logger.error("Error updating content rankings", error=str(e))
            return {"error": str(e), "processed_count": 0}
    
    async def _get_user_rating(self, content_id: UUID, user_id: UUID) -> Optional[ContentRating]:
        """Get existing user rating for content."""
        try:
            stmt = select(ContentRating).where(
                and_(
                    ContentRating.content_id == content_id,
                    ContentRating.user_id == user_id
                )
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error getting user rating", error=str(e))
            return None
    
    async def _update_content_statistics(self, content_id: UUID) -> None:
        """Update content rating statistics."""
        try:
            # Calculate new statistics
            stmt = select(
                func.count(ContentRating.id).label('rating_count'),
                func.avg(ContentRating.rating).label('avg_rating')
            ).where(ContentRating.content_id == content_id)
            
            result = await self.db.execute(stmt)
            stats = result.first()
            
            if stats:
                # Update content
                update_stmt = update(LearningContent).where(
                    LearningContent.id == content_id
                ).values(
                    rating_count=stats.rating_count,
                    average_rating=float(stats.avg_rating or 0.0)
                )
                
                await self.db.execute(update_stmt)
                await self.db.commit()
                
        except Exception as e:
            logger.error("Error updating content statistics", error=str(e), content_id=str(content_id))
    
    async def _calculate_sentiment_score(self, reviews: List[ContentRating]) -> float:
        """Calculate sentiment score from reviews (simplified implementation)."""
        if not reviews:
            return 0.0
        
        # Simple sentiment analysis based on keywords
        positive_keywords = ['good', 'great', 'excellent', 'helpful', 'clear', 'useful', 'amazing']
        negative_keywords = ['bad', 'poor', 'terrible', 'confusing', 'unclear', 'useless', 'awful']
        
        total_sentiment = 0.0
        
        for review in reviews:
            if not review.review_text:
                continue
                
            text = review.review_text.lower()
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            
            # Normalize by review rating
            rating_factor = (review.rating - 3) / 2  # Convert 1-5 to -1 to 1
            sentiment = (positive_count - negative_count) * 0.1 + rating_factor
            total_sentiment += max(-1.0, min(1.0, sentiment))
        
        return total_sentiment / len(reviews)
    
    async def _extract_common_themes(self, reviews: List[ContentRating]) -> List[str]:
        """Extract common themes from reviews (simplified implementation)."""
        if not reviews:
            return []
        
        # Simple keyword frequency analysis
        theme_keywords = {
            'clarity': ['clear', 'unclear', 'confusing', 'understandable'],
            'completeness': ['complete', 'incomplete', 'missing', 'thorough'],
            'accuracy': ['accurate', 'correct', 'wrong', 'error', 'mistake'],
            'usefulness': ['useful', 'helpful', 'practical', 'useless'],
            'difficulty': ['easy', 'hard', 'difficult', 'simple', 'complex']
        }
        
        theme_counts = {theme: 0 for theme in theme_keywords}
        
        for review in reviews:
            if not review.review_text:
                continue
                
            text = review.review_text.lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in text for keyword in keywords):
                    theme_counts[theme] += 1
        
        # Return themes mentioned in at least 20% of reviews
        threshold = len(reviews) * 0.2
        return [theme for theme, count in theme_counts.items() if count >= threshold]
    
    async def _analyze_feedback_trend(self, content_id: UUID) -> str:
        """Analyze recent feedback trend."""
        try:
            # Get ratings from last 30 days vs previous 30 days
            now = datetime.utcnow()
            recent_cutoff = now - timedelta(days=30)
            older_cutoff = now - timedelta(days=60)
            
            # Recent ratings
            recent_stmt = select(func.avg(ContentRating.rating)).where(
                and_(
                    ContentRating.content_id == content_id,
                    ContentRating.created_at >= recent_cutoff
                )
            )
            recent_result = await self.db.execute(recent_stmt)
            recent_avg = recent_result.scalar() or 0.0
            
            # Older ratings
            older_stmt = select(func.avg(ContentRating.rating)).where(
                and_(
                    ContentRating.content_id == content_id,
                    ContentRating.created_at >= older_cutoff,
                    ContentRating.created_at < recent_cutoff
                )
            )
            older_result = await self.db.execute(older_stmt)
            older_avg = older_result.scalar() or 0.0
            
            if recent_avg == 0.0 and older_avg == 0.0:
                return "stable"
            elif recent_avg > older_avg + 0.3:
                return "improving"
            elif recent_avg < older_avg - 0.3:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logger.error("Error analyzing feedback trend", error=str(e), content_id=str(content_id))
            return "stable"
    
    async def _detect_feedback_conflicts(self, content_id: UUID) -> None:
        """Detect and log feedback conflicts for content."""
        try:
            conflicts = await self._analyze_content_conflicts(content_id)
            
            for conflict in conflicts:
                if conflict.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]:
                    logger.warning(
                        "Content conflict detected",
                        content_id=str(content_id),
                        conflict_type=conflict.conflict_type,
                        severity=conflict.severity.value
                    )
                    
                    # Auto-resolve low-severity conflicts
                    if conflict.severity == ConflictSeverity.LOW:
                        await self.resolve_content_conflict(conflict, "weighted_average")
                        
        except Exception as e:
            logger.error("Error detecting feedback conflicts", error=str(e), content_id=str(content_id))
    
    async def _analyze_content_conflicts(self, content_id: UUID) -> List[ContentConflict]:
        """Analyze content for feedback conflicts."""
        conflicts = []
        
        try:
            # Get all ratings for content
            stmt = select(ContentRating).where(ContentRating.content_id == content_id)
            result = await self.db.execute(stmt)
            ratings = result.scalars().all()
            
            if len(ratings) < 3:  # Need minimum ratings to detect conflicts
                return conflicts
            
            # Check for rating variance conflicts
            rating_values = [r.rating for r in ratings]
            rating_variance = sum((r - sum(rating_values)/len(rating_values))**2 for r in rating_values) / len(rating_values)
            
            if rating_variance > self.conflict_thresholds["rating_variance"]:
                conflicts.append(ContentConflict(
                    content_id=content_id,
                    conflict_type="rating_variance",
                    severity=ConflictSeverity.MEDIUM if rating_variance > 3.0 else ConflictSeverity.LOW,
                    description=f"High variance in ratings (variance: {rating_variance:.2f})",
                    conflicting_ratings=[r.id for r in ratings],
                    suggested_resolution="Review content quality and consider expert evaluation",
                    confidence=min(1.0, rating_variance / 4.0)
                ))
            
            # Check for accuracy disputes
            accuracy_ratings = [r for r in ratings if r.accuracy_rating is not None]
            if len(accuracy_ratings) >= 3:
                low_accuracy_count = sum(1 for r in accuracy_ratings if r.accuracy_rating <= 2)
                if low_accuracy_count >= self.conflict_thresholds["accuracy_disputes"]:
                    conflicts.append(ContentConflict(
                        content_id=content_id,
                        conflict_type="accuracy_dispute",
                        severity=ConflictSeverity.HIGH,
                        description=f"Multiple accuracy complaints ({low_accuracy_count} users)",
                        conflicting_ratings=[r.id for r in accuracy_ratings if r.accuracy_rating <= 2],
                        suggested_resolution="Content needs expert review for accuracy",
                        confidence=0.8
                    ))
            
            return conflicts
            
        except Exception as e:
            logger.error("Error analyzing content conflicts", error=str(e), content_id=str(content_id))
            return []
    
    async def _calculate_new_ranking(self, content: LearningContent) -> Optional[RankingUpdate]:
        """Calculate new ranking score for content."""
        try:
            old_score = getattr(content, 'ranking_score', content.average_rating)
            
            # Get feedback summary
            feedback_summary = await self.get_feedback_summary(content.id)
            
            # Calculate ranking factors
            factors = {}
            
            # Average rating factor
            factors["average_rating"] = feedback_summary.average_rating / 5.0
            
            # Rating count factor (logarithmic scaling)
            import math
            factors["rating_count"] = min(1.0, math.log(feedback_summary.total_ratings + 1) / math.log(100))
            
            # Sentiment factor
            factors["review_sentiment"] = (feedback_summary.sentiment_score + 1.0) / 2.0
            
            # Recency factor (based on recent trend)
            if feedback_summary.recent_feedback_trend == "improving":
                factors["recency"] = 1.0
            elif feedback_summary.recent_feedback_trend == "declining":
                factors["recency"] = 0.3
            else:
                factors["recency"] = 0.7
            
            # User credibility factor (simplified)
            factors["user_credibility"] = 0.8  # Default value
            
            # Engagement factor
            if content.view_count > 0:
                completion_rate = content.completion_count / content.view_count
                factors["engagement"] = min(1.0, completion_rate * 2)
            else:
                factors["engagement"] = 0.5
            
            # Calculate weighted score
            new_score = sum(
                factors[factor] * self.ranking_weights[factor]
                for factor in factors
            )
            
            # Only create update if score changed significantly
            if abs(new_score - old_score) > 0.05:
                return RankingUpdate(
                    content_id=content.id,
                    old_score=old_score,
                    new_score=new_score,
                    factors=factors,
                    rationale=f"Updated based on {feedback_summary.total_ratings} ratings, "
                             f"avg: {feedback_summary.average_rating:.1f}, "
                             f"trend: {feedback_summary.recent_feedback_trend}"
                )
            
            return None
            
        except Exception as e:
            logger.error("Error calculating new ranking", error=str(e), content_id=str(content.id))
            return None
    
    async def _apply_ranking_update(self, ranking_update: RankingUpdate) -> bool:
        """Apply ranking update to content."""
        try:
            # Update content ranking score
            update_stmt = update(LearningContent).where(
                LearningContent.id == ranking_update.content_id
            ).values(
                # Assuming we add a ranking_score field to the model
                quality_score=ranking_update.new_score  # Using quality_score as proxy
            )
            
            await self.db.execute(update_stmt)
            await self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error("Error applying ranking update", error=str(e), content_id=str(ranking_update.content_id))
            return False
    
    async def _resolve_with_weighted_average(self, conflict: ContentConflict) -> bool:
        """Resolve conflict using weighted average approach."""
        try:
            # Get conflicting ratings
            stmt = select(ContentRating).where(ContentRating.id.in_(conflict.conflicting_ratings))
            result = await self.db.execute(stmt)
            ratings = result.scalars().all()
            
            if not ratings:
                return False
            
            # Calculate weighted average (could be enhanced with user credibility)
            total_weight = len(ratings)
            weighted_sum = sum(r.rating for r in ratings)
            adjusted_rating = weighted_sum / total_weight
            
            # Update content average rating
            update_stmt = update(LearningContent).where(
                LearningContent.id == conflict.content_id
            ).values(
                average_rating=adjusted_rating
            )
            
            await self.db.execute(update_stmt)
            await self.db.commit()
            
            logger.info("Conflict resolved with weighted average", 
                       content_id=str(conflict.content_id), 
                       adjusted_rating=adjusted_rating)
            
            return True
            
        except Exception as e:
            logger.error("Error resolving conflict with weighted average", error=str(e))
            return False
    
    async def _flag_for_expert_review(self, conflict: ContentConflict) -> bool:
        """Flag content for expert review."""
        try:
            # Update content status to require review
            update_stmt = update(LearningContent).where(
                LearningContent.id == conflict.content_id
            ).values(
                quality_status=QualityStatus.REVIEW
            )
            
            await self.db.execute(update_stmt)
            await self.db.commit()
            
            logger.info("Content flagged for expert review", content_id=str(conflict.content_id))
            return True
            
        except Exception as e:
            logger.error("Error flagging for expert review", error=str(e))
            return False
    
    async def _initiate_community_vote(self, conflict: ContentConflict) -> bool:
        """Initiate community voting for conflict resolution."""
        try:
            # This would typically involve creating a voting mechanism
            # For now, we'll just log the action
            logger.info("Community vote initiated for conflict resolution", 
                       content_id=str(conflict.content_id))
            
            # In a real implementation, this would:
            # 1. Create a voting record
            # 2. Notify community members
            # 3. Set up voting period
            # 4. Handle vote collection and resolution
            
            return True
            
        except Exception as e:
            logger.error("Error initiating community vote", error=str(e))
            return False