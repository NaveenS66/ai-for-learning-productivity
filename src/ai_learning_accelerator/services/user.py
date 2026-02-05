"""User service for user and profile management."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..models.user import User, UserProfile, SkillAssessment, LearningActivity
from ..schemas.user import (
    UserCreate, UserUpdate, UserProfileCreate, UserProfileUpdate,
    SkillAssessmentCreate, SkillAssessmentUpdate,
    LearningActivityCreate, LearningActivityUpdate
)
from ..utils.security import get_password_hash
from ..logging_config import get_logger

logger = get_logger(__name__)


class UserService:
    """Service for user management operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        try:
            # Check if user already exists
            existing_user = await self.get_user_by_email(user_data.email)
            if existing_user:
                raise ValueError("User with this email already exists")
            
            existing_username = await self.get_user_by_username(user_data.username)
            if existing_username:
                raise ValueError("User with this username already exists")
            
            # Create user
            hashed_password = get_password_hash(user_data.password)
            user = User(
                email=user_data.email,
                username=user_data.username,
                full_name=user_data.full_name,
                hashed_password=hashed_password
            )
            
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
            
            logger.info("User created successfully", user_id=str(user.id), username=user.username)
            return user
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating user", error=str(e), email=user_data.email)
            raise
    
    async def get_user_by_id(self, user_id: UUID, include_profile: bool = False) -> Optional[User]:
        """Get user by ID with optional profile."""
        try:
            query = select(User).where(User.id == user_id)
            
            if include_profile:
                query = query.options(
                    selectinload(User.profile),
                    selectinload(User.skill_assessments)
                )
            
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error("Error fetching user by ID", error=str(e), user_id=str(user_id))
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            stmt = select(User).where(User.email == email)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching user by email", error=str(e), email=email)
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            stmt = select(User).where(User.username == username)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching user by username", error=str(e), username=username)
            return None
    
    async def update_user(self, user_id: UUID, user_data: UserUpdate) -> Optional[User]:
        """Update user information."""
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return None
            
            # Update fields if provided
            update_data = user_data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(user, field, value)
            
            await self.db.commit()
            await self.db.refresh(user)
            
            logger.info("User updated successfully", user_id=str(user_id))
            return user
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error updating user", error=str(e), user_id=str(user_id))
            raise
    
    async def delete_user(self, user_id: UUID) -> bool:
        """Delete user (soft delete by setting inactive)."""
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return False
            
            user.is_active = False
            await self.db.commit()
            
            logger.info("User deactivated successfully", user_id=str(user_id))
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error deactivating user", error=str(e), user_id=str(user_id))
            return False
    
    async def create_user_profile(self, user_id: UUID, profile_data: UserProfileCreate) -> UserProfile:
        """Create user profile."""
        try:
            # Check if profile already exists
            existing_profile = await self.get_user_profile(user_id)
            if existing_profile:
                raise ValueError("User profile already exists")
            
            profile = UserProfile(
                user_id=user_id,
                **profile_data.model_dump()
            )
            
            self.db.add(profile)
            await self.db.commit()
            await self.db.refresh(profile)
            
            logger.info("User profile created successfully", user_id=str(user_id))
            return profile
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating user profile", error=str(e), user_id=str(user_id))
            raise
    
    async def get_user_profile(self, user_id: UUID) -> Optional[UserProfile]:
        """Get user profile."""
        try:
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching user profile", error=str(e), user_id=str(user_id))
            return None
    
    async def update_user_profile(self, user_id: UUID, profile_data: UserProfileUpdate) -> Optional[UserProfile]:
        """Update user profile."""
        try:
            profile = await self.get_user_profile(user_id)
            if not profile:
                return None
            
            # Update fields if provided
            update_data = profile_data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(profile, field, value)
            
            await self.db.commit()
            await self.db.refresh(profile)
            
            logger.info("User profile updated successfully", user_id=str(user_id))
            return profile
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error updating user profile", error=str(e), user_id=str(user_id))
            raise
    
    async def create_skill_assessment(self, user_id: UUID, assessment_data: SkillAssessmentCreate) -> SkillAssessment:
        """Create or update skill assessment."""
        try:
            # Check if assessment for this domain already exists
            existing = await self.get_skill_assessment(user_id, assessment_data.domain)
            
            if existing:
                # Update existing assessment
                existing.previous_level = existing.skill_level
                existing.skill_level = assessment_data.skill_level
                existing.confidence_score = assessment_data.confidence_score
                existing.assessment_method = assessment_data.assessment_method
                existing.evidence = assessment_data.evidence
                existing.progress_notes = assessment_data.progress_notes
                existing.last_assessed = datetime.utcnow()
                
                await self.db.commit()
                await self.db.refresh(existing)
                
                logger.info("Skill assessment updated", user_id=str(user_id), domain=assessment_data.domain)
                return existing
            else:
                # Create new assessment
                assessment = SkillAssessment(
                    user_id=user_id,
                    **assessment_data.model_dump()
                )
                
                self.db.add(assessment)
                await self.db.commit()
                await self.db.refresh(assessment)
                
                logger.info("Skill assessment created", user_id=str(user_id), domain=assessment_data.domain)
                return assessment
                
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating skill assessment", error=str(e), user_id=str(user_id))
            raise
    
    async def get_skill_assessment(self, user_id: UUID, domain: str) -> Optional[SkillAssessment]:
        """Get skill assessment for specific domain."""
        try:
            stmt = select(SkillAssessment).where(
                SkillAssessment.user_id == user_id,
                SkillAssessment.domain == domain
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching skill assessment", error=str(e), user_id=str(user_id), domain=domain)
            return None
    
    async def get_user_skill_assessments(self, user_id: UUID) -> List[SkillAssessment]:
        """Get all skill assessments for user."""
        try:
            stmt = select(SkillAssessment).where(SkillAssessment.user_id == user_id)
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error("Error fetching user skill assessments", error=str(e), user_id=str(user_id))
            return []
    
    async def create_learning_activity(self, user_id: UUID, activity_data: LearningActivityCreate) -> LearningActivity:
        """Create learning activity."""
        try:
            activity = LearningActivity(
                user_id=user_id,
                **activity_data.model_dump()
            )
            
            self.db.add(activity)
            await self.db.commit()
            await self.db.refresh(activity)
            
            logger.info("Learning activity created", user_id=str(user_id), activity_id=str(activity.id))
            return activity
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating learning activity", error=str(e), user_id=str(user_id))
            raise
    
    async def get_user_learning_activities(self, user_id: UUID, limit: int = 50) -> List[LearningActivity]:
        """Get user's learning activities."""
        try:
            stmt = select(LearningActivity).where(
                LearningActivity.user_id == user_id
            ).order_by(LearningActivity.created_at.desc()).limit(limit)
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error("Error fetching learning activities", error=str(e), user_id=str(user_id))
            return []