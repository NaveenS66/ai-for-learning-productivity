"""Learning engine service for personalized learning and skill assessment."""

import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import selectinload

# Temporary workaround for import issue
class LearningPathStatus(str, Enum):
    """Learning path status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

from ..models.learning import (
    LearningGoal, LearningPath, LearningPathItem, CompetencyArea, 
    SkillAssessmentDetail, LearningRecommendation,
    LearningGoalStatus, PathItemStatus, SkillAssessmentType,
    LearningRecommendationType
)
from ..models.user import User, UserProfile, SkillAssessment, SkillLevel, DifficultyLevel, LearningStyle
from ..models.content import LearningContent, ContentType
from ..services.user import UserService
from ..services.content import ContentService
from ..logging_config import get_logger

logger = get_logger(__name__)


class LearningEngine:
    """Core learning engine for personalized learning experiences."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.user_service = UserService(db)
        self.content_service = ContentService(db)
    
    # Skill Assessment Methods
    
    async def assess_user_skill_level(
        self, 
        user_id: UUID, 
        competency_area: str,
        assessment_type: SkillAssessmentType = SkillAssessmentType.AI_ASSESSMENT
    ) -> SkillLevel:
        """Assess user's skill level in a specific competency area using AI analysis."""
        try:
            # Get user's learning history and activities
            user_activities = await self.user_service.get_user_learning_activities(user_id)
            user_assessments = await self.user_service.get_user_skill_assessments(user_id)
            
            # Simple AI assessment algorithm (in production, this would use ML models)
            skill_indicators = []
            
            # Analyze learning activities
            relevant_activities = [
                activity for activity in user_activities 
                if competency_area.lower() in (activity.title.lower() + " " + (activity.description or ""))
            ]
            
            if relevant_activities:
                avg_completion = sum(a.completion_percentage for a in relevant_activities) / len(relevant_activities)
                avg_score = sum(a.score or 50 for a in relevant_activities) / len(relevant_activities)
                
                skill_indicators.append(avg_completion / 100.0)
                skill_indicators.append(avg_score / 100.0)
            
            # Analyze existing assessments
            relevant_assessments = [
                assessment for assessment in user_assessments
                if competency_area.lower() in assessment.domain.lower()
            ]
            
            if relevant_assessments:
                latest_assessment = max(relevant_assessments, key=lambda x: x.last_assessed)
                skill_indicators.append(self._skill_level_to_numeric(latest_assessment.skill_level))
                skill_indicators.append(latest_assessment.confidence_score / 100.0)
            
            # Calculate overall skill level
            if skill_indicators:
                avg_skill = sum(skill_indicators) / len(skill_indicators)
                assessed_level = self._numeric_to_skill_level(avg_skill)
            else:
                # Default to beginner for new areas
                assessed_level = SkillLevel.BEGINNER
            
            logger.info(
                "Skill assessment completed", 
                user_id=str(user_id), 
                competency_area=competency_area,
                assessed_level=assessed_level
            )
            
            return assessed_level
            
        except Exception as e:
            logger.error("Error assessing skill level", error=str(e), user_id=str(user_id))
            return SkillLevel.BEGINNER
    
    def _skill_level_to_numeric(self, skill_level: SkillLevel) -> float:
        """Convert skill level to numeric value (0.0-1.0)."""
        mapping = {
            SkillLevel.NOVICE: 0.0,
            SkillLevel.BEGINNER: 0.25,
            SkillLevel.INTERMEDIATE: 0.5,
            SkillLevel.ADVANCED: 0.75,
            SkillLevel.EXPERT: 1.0
        }
        return mapping.get(skill_level, 0.25)
    
    def _numeric_to_skill_level(self, numeric_value: float) -> SkillLevel:
        """Convert numeric value to skill level."""
        if numeric_value >= 0.9:
            return SkillLevel.EXPERT
        elif numeric_value >= 0.7:
            return SkillLevel.ADVANCED
        elif numeric_value >= 0.4:
            return SkillLevel.INTERMEDIATE
        elif numeric_value >= 0.15:
            return SkillLevel.BEGINNER
        else:
            return SkillLevel.NOVICE
    
    async def analyze_learning_preferences(self, user_id: UUID) -> Dict[str, Any]:
        """Analyze user's learning preferences based on activity patterns."""
        try:
            user_profile = await self.user_service.get_user_profile(user_id)
            user_activities = await self.user_service.get_user_learning_activities(user_id)
            
            preferences = {
                "learning_style": user_profile.learning_style if user_profile else LearningStyle.MULTIMODAL,
                "difficulty_preference": user_profile.difficulty_preference if user_profile else DifficultyLevel.INTERMEDIATE,
                "preferred_content_types": [],
                "optimal_session_length": 30,  # minutes
                "preferred_time_of_day": "morning",
                "engagement_patterns": {}
            }
            
            if user_activities:
                # Analyze content type preferences
                content_type_scores = {}
                for activity in user_activities:
                    content_type = activity.activity_type
                    score = (activity.completion_percentage / 100.0) * (activity.score or 50) / 50.0
                    
                    if content_type in content_type_scores:
                        content_type_scores[content_type].append(score)
                    else:
                        content_type_scores[content_type] = [score]
                
                # Calculate average scores and sort by preference
                avg_scores = {
                    content_type: sum(scores) / len(scores)
                    for content_type, scores in content_type_scores.items()
                }
                
                preferences["preferred_content_types"] = sorted(
                    avg_scores.keys(), 
                    key=lambda x: avg_scores[x], 
                    reverse=True
                )
                
                # Analyze session length preferences
                durations = [a.duration_minutes for a in user_activities if a.duration_minutes]
                if durations:
                    preferences["optimal_session_length"] = sum(durations) // len(durations)
            
            logger.info("Learning preferences analyzed", user_id=str(user_id))
            return preferences
            
        except Exception as e:
            logger.error("Error analyzing learning preferences", error=str(e), user_id=str(user_id))
            return {}
    
    # Learning Path Generation Methods
    
    async def generate_learning_path(
        self, 
        user_id: UUID, 
        goal_title: str,
        target_skill_level: SkillLevel,
        category: str,
        interests: Optional[List[str]] = None
    ) -> LearningPath:
        """Generate a personalized learning path for a user's goal with milestone tracking."""
        try:
            # Create learning goal with milestones
            goal = LearningGoal(
                user_id=user_id,
                title=goal_title,
                category=category,
                target_skill_level=target_skill_level,
                status=LearningGoalStatus.ACTIVE
            )
            
            # Generate milestones based on skill progression
            milestones = await self._generate_learning_milestones(
                current_skill=SkillLevel.BEGINNER,  # Will be updated after assessment
                target_skill=target_skill_level,
                category=category
            )
            goal.milestones = milestones
            
            self.db.add(goal)
            await self.db.commit()
            await self.db.refresh(goal)
            
            # Assess current skill level
            current_skill = await self.assess_user_skill_level(user_id, category)
            goal.current_skill_level = current_skill
            
            # Update milestones based on actual current skill
            updated_milestones = await self._generate_learning_milestones(
                current_skill=current_skill,
                target_skill=target_skill_level,
                category=category
            )
            goal.milestones = updated_milestones
            
            # Generate learning path
            path = LearningPath(
                user_id=user_id,
                goal_id=goal.id,
                name=f"Path to {goal_title}",
                description=f"Personalized learning path to achieve {target_skill_level} level in {category}",
                difficulty_level=self._skill_level_to_difficulty(target_skill_level),
                status=LearningPathStatus.ACTIVE,
                is_adaptive=True
            )
            
            self.db.add(path)
            await self.db.commit()
            await self.db.refresh(path)
            
            # Generate path items based on skill gap and milestones
            await self._generate_path_items_with_milestones(
                path, current_skill, target_skill_level, category, milestones, interests
            )
            
            logger.info(
                "Learning path generated with milestones", 
                user_id=str(user_id), 
                path_id=str(path.id),
                goal=goal_title,
                milestones_count=len(milestones)
            )
            
            return path
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error generating learning path", error=str(e), user_id=str(user_id))
            raise
    
    def _skill_level_to_difficulty(self, skill_level: SkillLevel) -> DifficultyLevel:
        """Convert skill level to difficulty level."""
        mapping = {
            SkillLevel.NOVICE: DifficultyLevel.BEGINNER,
            SkillLevel.BEGINNER: DifficultyLevel.BEGINNER,
            SkillLevel.INTERMEDIATE: DifficultyLevel.INTERMEDIATE,
            SkillLevel.ADVANCED: DifficultyLevel.ADVANCED,
            SkillLevel.EXPERT: DifficultyLevel.EXPERT
        }
        return mapping.get(skill_level, DifficultyLevel.INTERMEDIATE)
    
    async def _generate_learning_milestones(
        self,
        current_skill: SkillLevel,
        target_skill: SkillLevel,
        category: str
    ) -> List[Dict[str, Any]]:
        """Generate learning milestones based on skill progression."""
        milestones = []
        
        # Define skill progression levels
        skill_levels = [SkillLevel.NOVICE, SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED, SkillLevel.EXPERT]
        current_index = skill_levels.index(current_skill)
        target_index = skill_levels.index(target_skill)
        
        # Generate milestones for each skill level between current and target
        for i in range(current_index + 1, target_index + 1):
            skill_level = skill_levels[i]
            milestone = {
                "id": f"milestone_{i}",
                "title": f"Achieve {skill_level.value} level in {category}",
                "description": f"Demonstrate {skill_level.value} competency in {category}",
                "skill_level": skill_level.value,
                "order": i - current_index,
                "completion_criteria": self._get_completion_criteria_for_skill_level(skill_level, category),
                "estimated_duration_hours": self._estimate_milestone_duration(current_skill, skill_level),
                "is_completed": False,
                "completion_date": None
            }
            milestones.append(milestone)
        
        return milestones
    
    def _get_completion_criteria_for_skill_level(self, skill_level: SkillLevel, category: str) -> List[str]:
        """Get completion criteria for a specific skill level."""
        base_criteria = {
            SkillLevel.BEGINNER: [
                f"Complete foundational {category} concepts",
                "Demonstrate basic understanding through assessments",
                "Complete at least 3 practical exercises"
            ],
            SkillLevel.INTERMEDIATE: [
                f"Apply {category} concepts to solve real problems",
                "Complete intermediate-level projects",
                "Demonstrate problem-solving skills",
                "Pass intermediate assessment with 70%+ score"
            ],
            SkillLevel.ADVANCED: [
                f"Master advanced {category} techniques",
                "Complete complex projects independently",
                "Mentor others or contribute to community",
                "Pass advanced assessment with 80%+ score"
            ],
            SkillLevel.EXPERT: [
                f"Innovate and create new solutions in {category}",
                "Lead projects and teach others",
                "Contribute to field knowledge",
                "Demonstrate thought leadership"
            ]
        }
        return base_criteria.get(skill_level, [])
    
    def _estimate_milestone_duration(self, current_skill: SkillLevel, target_skill: SkillLevel) -> int:
        """Estimate hours needed to reach target skill level."""
        skill_gaps = {
            (SkillLevel.NOVICE, SkillLevel.BEGINNER): 20,
            (SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE): 40,
            (SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED): 60,
            (SkillLevel.ADVANCED, SkillLevel.EXPERT): 80
        }
        return skill_gaps.get((current_skill, target_skill), 30)
    
    async def _generate_path_items_with_milestones(
        self, 
        path: LearningPath, 
        current_skill: SkillLevel, 
        target_skill: SkillLevel,
        category: str,
        milestones: List[Dict[str, Any]],
        interests: Optional[List[str]] = None
    ) -> None:
        """Generate learning path items with milestone tracking and interest-based extensions."""
        try:
            # Find relevant content
            from ..schemas.content import ContentSearchRequest
            
            search_request = ContentSearchRequest(
                query=category,
                limit=100  # Get more content for better selection
            )
            
            content_list, _ = await self.content_service.search_learning_content(search_request)
            
            if not content_list:
                logger.warning("No content found for category", category=category)
                return
            
            # Sort content by difficulty and relevance
            sorted_content = sorted(
                content_list,
                key=lambda x: (
                    self._difficulty_to_numeric(x.difficulty_level),
                    -x.average_rating,
                    -x.view_count
                )
            )
            
            # Generate path items with progressive difficulty and milestone alignment
            current_numeric = self._skill_level_to_numeric(current_skill)
            target_numeric = self._skill_level_to_numeric(target_skill)
            
            items_created = 0
            order_index = 0
            
            # Create milestone items first
            for milestone in milestones:
                milestone_item = LearningPathItem(
                    learning_path_id=path.id,
                    title=milestone["title"],
                    description=milestone["description"],
                    item_type="milestone",
                    order_index=order_index,
                    status=PathItemStatus.LOCKED if order_index > 0 else PathItemStatus.AVAILABLE,
                    is_milestone=True,
                    is_required=True,
                    estimated_duration=milestone["estimated_duration_hours"] * 60,  # Convert to minutes
                    recommendation_score=1.0
                )
                
                self.db.add(milestone_item)
                items_created += 1
                order_index += 1
            
            # Add content items between milestones
            milestone_boundaries = [self._skill_level_to_numeric(SkillLevel(m["skill_level"])) for m in milestones]
            milestone_boundaries.insert(0, current_numeric)
            
            for i, boundary in enumerate(milestone_boundaries[:-1]):
                next_boundary = milestone_boundaries[i + 1]
                
                # Find content that fits between these skill levels
                relevant_content = [
                    content for content in sorted_content
                    if boundary <= self._difficulty_to_numeric(content.difficulty_level) <= next_boundary
                ]
                
                # Add 3-5 content items per milestone section
                content_count = min(5, len(relevant_content))
                for j, content in enumerate(relevant_content[:content_count]):
                    item = LearningPathItem(
                        learning_path_id=path.id,
                        content_id=content.id,
                        title=content.title,
                        description=content.description,
                        item_type="content",
                        order_index=order_index,
                        status=PathItemStatus.LOCKED if order_index > 0 else PathItemStatus.AVAILABLE,
                        estimated_duration=content.estimated_duration,
                        recommendation_score=min(1.0, content.average_rating / 5.0),
                        prerequisites=[f"milestone_{i}"] if i > 0 else []
                    )
                    
                    self.db.add(item)
                    items_created += 1
                    order_index += 1
            
            # Add interest-based extensions if provided
            if interests:
                await self._add_interest_based_extensions(path, interests, order_index)
                items_created += 3  # Approximate additional items
            
            # Update path totals
            path.total_items = items_created
            
            await self.db.commit()
            
            logger.info(
                "Path items generated with milestones", 
                path_id=str(path.id), 
                items_count=items_created,
                milestones_count=len(milestones)
            )
            
        except Exception as e:
            logger.error("Error generating path items with milestones", error=str(e), path_id=str(path.id))
    
    async def _add_interest_based_extensions(
        self,
        path: LearningPath,
        interests: List[str],
        start_order_index: int
    ) -> None:
        """Add interest-based extensions to the learning path."""
        try:
            order_index = start_order_index
            
            for interest in interests[:3]:  # Limit to top 3 interests
                from ..schemas.content import ContentSearchRequest
                
                # Search for content that combines the main category with the interest
                search_request = ContentSearchRequest(
                    query=f"{interest}",
                    limit=2
                )
                
                content_list, _ = await self.content_service.search_learning_content(search_request)
                
                for content in content_list:
                    extension_item = LearningPathItem(
                        learning_path_id=path.id,
                        content_id=content.id,
                        title=f"Extension: {content.title}",
                        description=f"Optional extension based on your interest in {interest}",
                        item_type="content",
                        order_index=order_index,
                        status=PathItemStatus.LOCKED,
                        is_required=False,  # Extensions are optional
                        estimated_duration=content.estimated_duration,
                        recommendation_score=0.8,  # High but not highest priority
                        prerequisites=["complete_main_path"]  # Unlock after main path completion
                    )
                    
                    self.db.add(extension_item)
                    order_index += 1
            
            logger.info(
                "Interest-based extensions added",
                path_id=str(path.id),
                interests=interests
            )
            
        except Exception as e:
            logger.error("Error adding interest-based extensions", error=str(e), path_id=str(path.id))
    
    def _difficulty_to_numeric(self, difficulty: DifficultyLevel) -> float:
        """Convert difficulty level to numeric value."""
        mapping = {
            DifficultyLevel.BEGINNER: 0.25,
            DifficultyLevel.INTERMEDIATE: 0.5,
            DifficultyLevel.ADVANCED: 0.75,
            DifficultyLevel.EXPERT: 1.0
        }
        return mapping.get(difficulty, 0.5)
    
    # Learning Recommendation Methods
    
    async def generate_learning_recommendations(
        self, 
        user_id: UUID, 
        limit: int = 10
    ) -> List[LearningRecommendation]:
        """Generate personalized learning recommendations for a user."""
        try:
            # Get user context
            user_profile = await self.user_service.get_user_profile(user_id)
            user_assessments = await self.user_service.get_user_skill_assessments(user_id)
            user_activities = await self.user_service.get_user_learning_activities(user_id, limit=20)
            
            recommendations = []
            
            # Skill-based recommendations
            skill_recs = await self._generate_skill_based_recommendations(
                user_id, user_assessments, limit // 2
            )
            recommendations.extend(skill_recs)
            
            # Interest-based recommendations
            if user_profile and user_profile.interests:
                interest_recs = await self._generate_interest_based_recommendations(
                    user_id, user_profile.interests, limit // 2
                )
                recommendations.extend(interest_recs)
            
            # Activity-based recommendations
            if user_activities:
                activity_recs = await self._generate_activity_based_recommendations(
                    user_id, user_activities, limit // 3
                )
                recommendations.extend(activity_recs)
            
            # Sort by relevance score and limit results
            recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
            recommendations = recommendations[:limit]
            
            # Save recommendations to database
            for rec in recommendations:
                self.db.add(rec)
            
            await self.db.commit()
            
            logger.info(
                "Learning recommendations generated", 
                user_id=str(user_id), 
                count=len(recommendations)
            )
            
            return recommendations
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error generating recommendations", error=str(e), user_id=str(user_id))
            return []
    
    async def _generate_skill_based_recommendations(
        self, 
        user_id: UUID, 
        assessments: List[SkillAssessment],
        limit: int
    ) -> List[LearningRecommendation]:
        """Generate recommendations based on skill gaps."""
        recommendations = []
        
        try:
            # Identify skill gaps (areas with lower confidence or skill levels)
            skill_gaps = []
            for assessment in assessments:
                if assessment.confidence_score < 70 or assessment.skill_level in [SkillLevel.NOVICE, SkillLevel.BEGINNER]:
                    skill_gaps.append({
                        'domain': assessment.domain,
                        'skill_level': assessment.skill_level,
                        'confidence': assessment.confidence_score
                    })
            
            # Generate recommendations for skill gaps
            for gap in skill_gaps[:limit]:
                from ..schemas.content import ContentSearchRequest
                
                search_request = ContentSearchRequest(
                    query=gap['domain'],
                    difficulty_level=self._skill_level_to_difficulty(gap['skill_level']),
                    limit=3
                )
                
                content_list, _ = await self.content_service.search_learning_content(search_request)
                
                for content in content_list:
                    relevance_score = 0.8 - (gap['confidence'] / 100.0) * 0.3  # Higher relevance for lower confidence
                    
                    rec = LearningRecommendation(
                        user_id=user_id,
                        content_id=content.id,
                        title=f"Improve {gap['domain']} skills",
                        description=f"Recommended to strengthen your {gap['domain']} competency",
                        recommendation_type=LearningRecommendationType.SKILL_BUILDING,
                        relevance_score=relevance_score,
                        confidence_score=0.85,
                        reasoning=f"Based on your {gap['skill_level']} level in {gap['domain']} with {gap['confidence']}% confidence"
                    )
                    
                    recommendations.append(rec)
                    
                    if len(recommendations) >= limit:
                        break
                
                if len(recommendations) >= limit:
                    break
            
        except Exception as e:
            logger.error("Error generating skill-based recommendations", error=str(e))
        
        return recommendations
    
    async def _generate_interest_based_recommendations(
        self, 
        user_id: UUID, 
        interests: List[str],
        limit: int
    ) -> List[LearningRecommendation]:
        """Generate recommendations based on user interests."""
        recommendations = []
        
        try:
            for interest in interests[:3]:  # Limit to top 3 interests
                from ..schemas.content import ContentSearchRequest
                
                search_request = ContentSearchRequest(
                    query=interest,
                    min_rating=3.0,
                    limit=3
                )
                
                content_list, _ = await self.content_service.search_learning_content(search_request)
                
                for content in content_list:
                    rec = LearningRecommendation(
                        user_id=user_id,
                        content_id=content.id,
                        title=f"Explore {interest}",
                        description=f"Based on your interest in {interest}",
                        recommendation_type=LearningRecommendationType.CONTENT,
                        relevance_score=0.7,
                        confidence_score=0.75,
                        reasoning=f"Matches your stated interest in {interest}"
                    )
                    
                    recommendations.append(rec)
                    
                    if len(recommendations) >= limit:
                        break
                
                if len(recommendations) >= limit:
                    break
            
        except Exception as e:
            logger.error("Error generating interest-based recommendations", error=str(e))
        
        return recommendations
    
    async def _generate_activity_based_recommendations(
        self, 
        user_id: UUID, 
        activities: List,
        limit: int
    ) -> List[LearningRecommendation]:
        """Generate recommendations based on recent learning activities."""
        recommendations = []
        
        try:
            # Analyze recent activity patterns
            recent_topics = []
            for activity in activities[:5]:  # Look at 5 most recent activities
                if activity.completion_percentage > 80:  # Successfully completed
                    recent_topics.append(activity.activity_type)
            
            if recent_topics:
                # Find related content
                most_common_topic = max(set(recent_topics), key=recent_topics.count)
                
                from ..schemas.content import ContentSearchRequest
                
                search_request = ContentSearchRequest(
                    query=most_common_topic,
                    limit=limit
                )
                
                content_list, _ = await self.content_service.search_learning_content(search_request)
                
                for content in content_list:
                    rec = LearningRecommendation(
                        user_id=user_id,
                        content_id=content.id,
                        title=f"Continue learning {most_common_topic}",
                        description=f"Build on your recent progress in {most_common_topic}",
                        recommendation_type=LearningRecommendationType.CONTENT,
                        relevance_score=0.6,
                        confidence_score=0.7,
                        reasoning=f"Based on your recent successful completion of {most_common_topic} activities"
                    )
                    
                    recommendations.append(rec)
                    
                    if len(recommendations) >= limit:
                        break
            
        except Exception as e:
            logger.error("Error generating activity-based recommendations", error=str(e))
        
        return recommendations
    
    # Adaptive Learning Methods
    
    async def update_learning_path_based_on_competency(
        self,
        path_id: UUID,
        completed_item_id: UUID,
        user_performance: Dict[str, Any]
    ) -> bool:
        """Update learning path based on demonstrated competency from completed modules."""
        try:
            # Get learning path with items
            stmt = select(LearningPath).where(LearningPath.id == path_id)\
                .options(selectinload(LearningPath.items), selectinload(LearningPath.goal))
            result = await self.db.execute(stmt)
            path = result.scalar_one_or_none()
            
            if not path:
                return False
            
            # Get completed item
            completed_item = next((item for item in path.items if item.id == completed_item_id), None)
            if not completed_item:
                return False
            
            # Analyze performance
            completion_score = user_performance.get('score', 75)
            time_efficiency = user_performance.get('time_efficiency', 1.0)  # actual_time / estimated_time
            difficulty_rating = user_performance.get('difficulty_rating', 3)  # 1-5 scale
            
            # Update competency assessment
            if completion_score >= 90 and time_efficiency <= 0.8 and difficulty_rating <= 2:
                # User is excelling - accelerate path
                await self._accelerate_learning_path(path, completed_item)
            elif completion_score >= 80 and time_efficiency <= 1.2:
                # User is progressing well - continue as planned
                await self._maintain_learning_pace(path, completed_item)
            elif completion_score < 60 or time_efficiency > 1.5 or difficulty_rating >= 4:
                # User is struggling - provide additional support
                await self._provide_additional_support(path, completed_item)
            
            # Update path progress
            await self._update_path_progress(path)
            
            # Check for milestone completion
            if completed_item.is_milestone:
                await self._handle_milestone_completion(path, completed_item, user_performance)
            
            await self.db.commit()
            
            logger.info(
                "Learning path updated based on competency",
                path_id=str(path_id),
                completed_item=completed_item.title,
                score=completion_score
            )
            
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error updating path based on competency", error=str(e), path_id=str(path_id))
            return False
    
    async def _accelerate_learning_path(self, path: LearningPath, completed_item: LearningPathItem) -> None:
        """Accelerate learning path for high-performing users."""
        # Skip redundant items
        for item in path.items:
            if (item.order_index > completed_item.order_index and 
                item.status == PathItemStatus.LOCKED and
                item.recommendation_score < 0.7 and
                not item.is_milestone):
                item.status = PathItemStatus.SKIPPED
                logger.info(f"Skipped redundant item: {item.title}")
        
        # Add advanced content
        await self._add_advanced_content_after_item(path, completed_item)
    
    async def _maintain_learning_pace(self, path: LearningPath, completed_item: LearningPathItem) -> None:
        """Maintain current learning pace."""
        # Unlock next items as planned
        next_items = [
            item for item in path.items 
            if item.order_index == completed_item.order_index + 1
        ]
        for item in next_items:
            if item.status == PathItemStatus.LOCKED:
                item.status = PathItemStatus.AVAILABLE
    
    async def _provide_additional_support(self, path: LearningPath, completed_item: LearningPathItem) -> None:
        """Provide additional support for struggling users."""
        # Add review materials
        await self._add_review_content_after_item(path, completed_item)
        
        # Adjust difficulty of upcoming items
        for item in path.items:
            if item.order_index > completed_item.order_index:
                item.difficulty_adjustment = max(-0.5, item.difficulty_adjustment - 0.2)
    
    async def _add_advanced_content_after_item(self, path: LearningPath, after_item: LearningPathItem) -> None:
        """Add advanced content for excelling learners."""
        try:
            from ..schemas.content import ContentSearchRequest
            
            # Search for advanced content in the same category
            if path.goal and path.goal.category:
                search_request = ContentSearchRequest(
                    query=f"advanced {path.goal.category}",
                    difficulty_level=DifficultyLevel.ADVANCED,
                    limit=2
                )
                
                content_list, _ = await self.content_service.search_learning_content(search_request)
                
                # Insert advanced items after the completed item
                for i, content in enumerate(content_list):
                    advanced_item = LearningPathItem(
                        learning_path_id=path.id,
                        content_id=content.id,
                        title=f"Advanced: {content.title}",
                        description=f"Advanced content added based on your excellent progress",
                        item_type="content",
                        order_index=after_item.order_index + 0.5 + (i * 0.1),  # Insert between existing items
                        status=PathItemStatus.AVAILABLE,
                        is_required=False,
                        estimated_duration=content.estimated_duration,
                        recommendation_score=0.9
                    )
                    
                    self.db.add(advanced_item)
                    path.total_items += 1
                    
                    logger.info(f"Added advanced content: {content.title}")
        
        except Exception as e:
            logger.error("Error adding advanced content", error=str(e))
    
    async def _add_review_content_after_item(self, path: LearningPath, after_item: LearningPathItem) -> None:
        """Add review content for struggling learners."""
        try:
            from ..schemas.content import ContentSearchRequest
            
            # Search for review/foundational content
            if path.goal and path.goal.category:
                search_request = ContentSearchRequest(
                    query=f"review {path.goal.category} fundamentals",
                    difficulty_level=DifficultyLevel.BEGINNER,
                    limit=2
                )
                
                content_list, _ = await self.content_service.search_learning_content(search_request)
                
                # Insert review items after the completed item
                for i, content in enumerate(content_list):
                    review_item = LearningPathItem(
                        learning_path_id=path.id,
                        content_id=content.id,
                        title=f"Review: {content.title}",
                        description=f"Review content to strengthen understanding",
                        item_type="review",
                        order_index=after_item.order_index + 0.5 + (i * 0.1),
                        status=PathItemStatus.AVAILABLE,
                        is_required=True,
                        estimated_duration=content.estimated_duration,
                        recommendation_score=0.8
                    )
                    
                    self.db.add(review_item)
                    path.total_items += 1
                    
                    logger.info(f"Added review content: {content.title}")
        
        except Exception as e:
            logger.error("Error adding review content", error=str(e))
    
    async def _handle_milestone_completion(
        self,
        path: LearningPath,
        milestone_item: LearningPathItem,
        performance: Dict[str, Any]
    ) -> None:
        """Handle milestone completion and update goal progress."""
        if not path.goal or not path.goal.milestones:
            return
        
        # Find and update the corresponding milestone
        milestones = path.goal.milestones
        for milestone in milestones:
            if milestone.get("title") == milestone_item.title:
                milestone["is_completed"] = True
                milestone["completion_date"] = datetime.utcnow().isoformat()
                milestone["actual_score"] = performance.get('score', 0)
                break
        
        # Update goal progress
        completed_milestones = sum(1 for m in milestones if m.get("is_completed", False))
        path.goal.progress_percentage = int((completed_milestones / len(milestones)) * 100)
        
        logger.info(
            "Milestone completed",
            milestone=milestone_item.title,
            goal_progress=path.goal.progress_percentage
        )
    
    async def detect_and_handle_progress_stalls(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Detect progress stalls and provide alternative approaches."""
        try:
            # Get user's active learning paths
            stmt = select(LearningPath).where(
                and_(
                    LearningPath.user_id == user_id,
                    LearningPath.status == LearningPathStatus.ACTIVE
                )
            ).options(selectinload(LearningPath.items), selectinload(LearningPath.goal))
            
            result = await self.db.execute(stmt)
            active_paths = result.scalars().all()
            
            stall_interventions = []
            
            for path in active_paths:
                stall_detected = await self._detect_stall_in_path(path)
                
                if stall_detected:
                    intervention = await self._create_stall_intervention(path, stall_detected)
                    stall_interventions.append(intervention)
                    
                    # Apply intervention
                    await self._apply_stall_intervention(path, intervention)
            
            if stall_interventions:
                await self.db.commit()
                
                logger.info(
                    "Progress stalls detected and handled",
                    user_id=str(user_id),
                    interventions_count=len(stall_interventions)
                )
            
            return stall_interventions
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error detecting progress stalls", error=str(e), user_id=str(user_id))
            return []
    
    async def _detect_stall_in_path(self, path: LearningPath) -> Optional[Dict[str, Any]]:
        """Detect if progress has stalled in a learning path."""
        # Get items that have been in progress for too long
        stalled_items = []
        current_time = datetime.utcnow()
        
        for item in path.items:
            if item.status == PathItemStatus.IN_PROGRESS:
                # Check if item has been in progress for more than expected duration + buffer
                expected_duration = timedelta(minutes=item.estimated_duration or 60)
                buffer_time = expected_duration * 0.5  # 50% buffer
                
                if item.updated_at and (current_time - item.updated_at) > (expected_duration + buffer_time):
                    stalled_items.append({
                        "item_id": item.id,
                        "title": item.title,
                        "stall_duration": (current_time - item.updated_at).days,
                        "difficulty_level": item.difficulty_adjustment
                    })
        
        # Check for repeated failures or low completion rates
        if len(stalled_items) > 0:
            return {
                "type": "item_stall",
                "stalled_items": stalled_items,
                "path_id": path.id,
                "severity": "high" if len(stalled_items) > 2 else "medium"
            }
        
        # Check for overall path progress stagnation
        if path.progress_percentage < 20 and (current_time - path.created_at).days > 14:
            return {
                "type": "path_stagnation",
                "path_id": path.id,
                "days_stagnant": (current_time - path.created_at).days,
                "severity": "high"
            }
        
        return None
    
    async def _create_stall_intervention(self, path: LearningPath, stall_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create intervention strategy for progress stall."""
        intervention = {
            "path_id": path.id,
            "stall_type": stall_info["type"],
            "severity": stall_info["severity"],
            "strategies": []
        }
        
        if stall_info["type"] == "item_stall":
            # Strategies for stalled items
            intervention["strategies"].extend([
                {
                    "type": "alternative_content",
                    "description": "Provide alternative learning materials with different teaching approaches"
                },
                {
                    "type": "difficulty_reduction",
                    "description": "Temporarily reduce difficulty and add foundational content"
                },
                {
                    "type": "learning_style_adaptation",
                    "description": "Adapt content to different learning styles (visual, auditory, kinesthetic)"
                }
            ])
        
        elif stall_info["type"] == "path_stagnation":
            # Strategies for overall path stagnation
            intervention["strategies"].extend([
                {
                    "type": "path_restructure",
                    "description": "Restructure path with smaller, more achievable milestones"
                },
                {
                    "type": "motivation_boost",
                    "description": "Add engaging, practical projects to boost motivation"
                },
                {
                    "type": "peer_support",
                    "description": "Connect with study groups or mentors"
                }
            ])
        
        return intervention
    
    async def _apply_stall_intervention(self, path: LearningPath, intervention: Dict[str, Any]) -> None:
        """Apply intervention strategies to address progress stall."""
        try:
            for strategy in intervention["strategies"]:
                if strategy["type"] == "alternative_content":
                    await self._add_alternative_content(path)
                elif strategy["type"] == "difficulty_reduction":
                    await self._reduce_path_difficulty(path)
                elif strategy["type"] == "learning_style_adaptation":
                    await self._adapt_to_learning_styles(path)
                elif strategy["type"] == "path_restructure":
                    await self._restructure_path_milestones(path)
                elif strategy["type"] == "motivation_boost":
                    await self._add_motivational_content(path)
            
            logger.info(
                "Stall intervention applied",
                path_id=str(path.id),
                strategies=[s["type"] for s in intervention["strategies"]]
            )
            
        except Exception as e:
            logger.error("Error applying stall intervention", error=str(e), path_id=str(path.id))
    
    async def _add_alternative_content(self, path: LearningPath) -> None:
        """Add alternative content with different teaching approaches."""
        # Implementation would search for content with different formats/approaches
        pass
    
    async def _reduce_path_difficulty(self, path: LearningPath) -> None:
        """Reduce overall path difficulty."""
        for item in path.items:
            if item.status in [PathItemStatus.AVAILABLE, PathItemStatus.LOCKED]:
                item.difficulty_adjustment = max(-0.5, item.difficulty_adjustment - 0.3)
    
    async def _adapt_to_learning_styles(self, path: LearningPath) -> None:
        """Adapt content to different learning styles."""
        # Implementation would add visual, auditory, and kinesthetic alternatives
        pass
    
    async def _restructure_path_milestones(self, path: LearningPath) -> None:
        """Restructure path with smaller, more achievable milestones."""
        # Implementation would break down large milestones into smaller ones
        pass
    
    async def _add_motivational_content(self, path: LearningPath) -> None:
        """Add engaging, practical projects to boost motivation."""
        # Implementation would add hands-on projects and real-world applications
        pass
    
    async def suggest_interest_based_path_extensions(
        self,
        user_id: UUID,
        path_id: UUID,
        new_interests: List[str]
    ) -> List[Dict[str, Any]]:
        """Suggest relevant extensions to learning path based on user interests."""
        try:
            # Get the learning path
            stmt = select(LearningPath).where(LearningPath.id == path_id)\
                .options(selectinload(LearningPath.goal))
            result = await self.db.execute(stmt)
            path = result.scalar_one_or_none()
            
            if not path:
                return []
            
            extensions = []
            
            for interest in new_interests:
                # Find content that combines the path's category with the new interest
                from ..schemas.content import ContentSearchRequest
                
                search_request = ContentSearchRequest(
                    query=f"{path.goal.category} {interest}",
                    limit=5
                )
                
                content_list, _ = await self.content_service.search_learning_content(search_request)
                
                if content_list:
                    extension = {
                        "interest": interest,
                        "title": f"Explore {interest} in {path.goal.category}",
                        "description": f"Extend your {path.goal.category} learning into {interest}",
                        "content_items": [
                            {
                                "id": str(content.id),
                                "title": content.title,
                                "description": content.description,
                                "difficulty": content.difficulty_level.value,
                                "duration": content.estimated_duration
                            }
                            for content in content_list[:3]
                        ],
                        "estimated_duration": sum(c.estimated_duration for c in content_list[:3]),
                        "relevance_score": 0.8
                    }
                    extensions.append(extension)
            
            logger.info(
                "Interest-based extensions suggested",
                user_id=str(user_id),
                path_id=str(path_id),
                extensions_count=len(extensions)
            )
            
            return extensions
            
        except Exception as e:
            logger.error("Error suggesting interest-based extensions", error=str(e))
            return []
    
    async def _update_path_progress(self, path: LearningPath) -> None:
        """Update overall path progress based on completed items."""
        if not path.items:
            return
        
        completed_items = sum(1 for item in path.items if item.status == PathItemStatus.COMPLETED)
        path.completed_items = completed_items
        path.progress_percentage = int((completed_items / path.total_items) * 100) if path.total_items > 0 else 0
    
    async def adapt_learning_path(self, path_id: UUID, user_performance_data: Dict[str, Any]) -> bool:
        """Adapt learning path based on user performance."""
        try:
            # Get learning path
            stmt = select(LearningPath).where(LearningPath.id == path_id)\
                .options(selectinload(LearningPath.items))
            result = await self.db.execute(stmt)
            path = result.scalar_one_or_none()
            
            if not path:
                return False
            
            # Analyze performance data
            avg_completion_rate = user_performance_data.get('avg_completion_rate', 0.8)
            avg_score = user_performance_data.get('avg_score', 75)
            time_spent_ratio = user_performance_data.get('time_spent_ratio', 1.0)  # actual/estimated
            
            # Determine adaptation strategy
            if avg_completion_rate < 0.6 or avg_score < 60:
                # User is struggling - add easier content or review materials
                await self._add_review_items(path)
                await self._adjust_difficulty_down(path)
            elif avg_completion_rate > 0.9 and avg_score > 85 and time_spent_ratio < 0.8:
                # User is excelling - add advanced content or skip basics
                await self._add_advanced_items(path)
                await self._skip_redundant_items(path)
            
            # Update path metadata
            path.adaptation_frequency = "weekly"
            await self.db.commit()
            
            logger.info("Learning path adapted", path_id=str(path_id))
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error adapting learning path", error=str(e), path_id=str(path_id))
            return False
    
    async def _add_review_items(self, path: LearningPath) -> None:
        """Add review items to help struggling learners."""
        try:
            if not path.goal or not path.goal.category:
                return
            
            from ..schemas.content import ContentSearchRequest
            
            search_request = ContentSearchRequest(
                query=f"review {path.goal.category} basics fundamentals",
                difficulty_level=DifficultyLevel.BEGINNER,
                limit=3
            )
            
            content_list, _ = await self.content_service.search_learning_content(search_request)
            
            # Find the best insertion point (after struggling items)
            insertion_index = max(item.order_index for item in path.items if item.status == PathItemStatus.COMPLETED) + 1
            
            for i, content in enumerate(content_list):
                review_item = LearningPathItem(
                    learning_path_id=path.id,
                    content_id=content.id,
                    title=f"Review: {content.title}",
                    description=f"Review material to strengthen understanding",
                    item_type="review",
                    order_index=insertion_index + i,
                    status=PathItemStatus.AVAILABLE,
                    is_required=True,
                    estimated_duration=content.estimated_duration,
                    recommendation_score=0.9
                )
                
                self.db.add(review_item)
                path.total_items += 1
            
            logger.info(f"Added {len(content_list)} review items to path {path.id}")
            
        except Exception as e:
            logger.error("Error adding review items", error=str(e))
    
    async def _adjust_difficulty_down(self, path: LearningPath) -> None:
        """Adjust path difficulty downward for struggling learners."""
        for item in path.items:
            if item.difficulty_adjustment > -0.5:
                item.difficulty_adjustment -= 0.1
    
    async def _add_advanced_items(self, path: LearningPath) -> None:
        """Add advanced items for excelling learners."""
        try:
            if not path.goal or not path.goal.category:
                return
            
            from ..schemas.content import ContentSearchRequest
            
            search_request = ContentSearchRequest(
                query=f"advanced {path.goal.category}",
                difficulty_level=DifficultyLevel.ADVANCED,
                limit=3
            )
            
            content_list, _ = await self.content_service.search_learning_content(search_request)
            
            # Add at the end of the path
            max_order = max(item.order_index for item in path.items) if path.items else 0
            
            for i, content in enumerate(content_list):
                advanced_item = LearningPathItem(
                    learning_path_id=path.id,
                    content_id=content.id,
                    title=f"Advanced: {content.title}",
                    description=f"Advanced content for accelerated learning",
                    item_type="content",
                    order_index=max_order + i + 1,
                    status=PathItemStatus.LOCKED,
                    is_required=False,
                    estimated_duration=content.estimated_duration,
                    recommendation_score=0.8
                )
                
                self.db.add(advanced_item)
                path.total_items += 1
            
            logger.info(f"Added {len(content_list)} advanced items to path {path.id}")
            
        except Exception as e:
            logger.error("Error adding advanced items", error=str(e))
    
    async def _skip_redundant_items(self, path: LearningPath) -> None:
        """Skip redundant items for advanced learners."""
        for item in path.items:
            if (item.status == PathItemStatus.AVAILABLE and 
                item.recommendation_score < 0.6 and 
                not item.is_milestone and
                not item.is_required):
                item.status = PathItemStatus.SKIPPED
                logger.info(f"Skipped redundant item: {item.title}")
    
    async def integrate_external_resources(
        self,
        path_id: UUID,
        external_resources: List[Dict[str, Any]]
    ) -> bool:
        """Integrate high-quality external resources into learning path."""
        try:
            # Get learning path
            stmt = select(LearningPath).where(LearningPath.id == path_id)\
                .options(selectinload(LearningPath.items))
            result = await self.db.execute(stmt)
            path = result.scalar_one_or_none()
            
            if not path:
                return False
            
            integrated_count = 0
            
            for resource in external_resources:
                # Validate resource quality
                if not self._validate_external_resource_quality(resource):
                    continue
                
                # Find appropriate insertion point based on difficulty and topic relevance
                insertion_point = self._find_optimal_insertion_point(path, resource)
                
                # Create learning path item for external resource
                external_item = LearningPathItem(
                    learning_path_id=path.id,
                    title=f"External: {resource['title']}",
                    description=f"High-quality external resource: {resource.get('description', '')}",
                    item_type="content",
                    order_index=insertion_point,
                    status=PathItemStatus.LOCKED,
                    is_required=False,
                    estimated_duration=resource.get('estimated_duration', 30),
                    recommendation_score=resource.get('quality_score', 0.8)
                )
                
                # Store external resource metadata
                external_item.prerequisites = resource.get('prerequisites', [])
                
                self.db.add(external_item)
                path.total_items += 1
                integrated_count += 1
            
            if integrated_count > 0:
                await self.db.commit()
                
                logger.info(
                    "External resources integrated",
                    path_id=str(path_id),
                    integrated_count=integrated_count
                )
            
            return integrated_count > 0
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error integrating external resources", error=str(e), path_id=str(path_id))
            return False
    
    def _validate_external_resource_quality(self, resource: Dict[str, Any]) -> bool:
        """Validate the quality of an external resource."""
        # Check required fields
        if not resource.get('title') or not resource.get('url'):
            return False
        
        # Check quality indicators
        quality_score = resource.get('quality_score', 0)
        user_rating = resource.get('user_rating', 0)
        
        # Minimum quality thresholds
        return quality_score >= 0.7 and user_rating >= 3.5
    
    def _find_optimal_insertion_point(self, path: LearningPath, resource: Dict[str, Any]) -> float:
        """Find the optimal insertion point for an external resource."""
        resource_difficulty = resource.get('difficulty_level', 'intermediate')
        resource_topics = resource.get('topics', [])
        
        # Find items with similar topics or difficulty
        similar_items = []
        for item in path.items:
            if (item.difficulty_adjustment and 
                self._difficulty_matches(resource_difficulty, item.difficulty_adjustment)):
                similar_items.append(item)
        
        if similar_items:
            # Insert after the last similar item
            return max(item.order_index for item in similar_items) + 0.5
        else:
            # Insert at the end
            return max(item.order_index for item in path.items) + 1 if path.items else 1
    
    def _difficulty_matches(self, resource_difficulty: str, item_difficulty_adjustment: float) -> bool:
        """Check if resource difficulty matches item difficulty."""
        difficulty_mapping = {
            'beginner': (-0.5, -0.1),
            'intermediate': (-0.1, 0.1),
            'advanced': (0.1, 0.5)
        }
        
        if resource_difficulty in difficulty_mapping:
            min_adj, max_adj = difficulty_mapping[resource_difficulty]
            return min_adj <= item_difficulty_adjustment <= max_adj
        
        return False