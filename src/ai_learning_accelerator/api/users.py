"""User management API endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_db
from ..models.user import User
from ..schemas.user import (
    UserResponse, UserUpdate, UserWithProfile,
    UserProfileCreate, UserProfileResponse, UserProfileUpdate,
    SkillAssessmentCreate, SkillAssessmentResponse, SkillAssessmentUpdate,
    LearningActivityCreate, LearningActivityResponse, LearningActivityUpdate
)
from ..services.auth import get_current_active_user
from ..services.user import UserService
from ..logging_config import get_logger

router = APIRouter(prefix="/users", tags=["users"])
logger = get_logger(__name__)


@router.get("/me", response_model=UserWithProfile)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> UserWithProfile:
    """Get current user's profile information."""
    user_service = UserService(db)
    user_with_profile = await user_service.get_user_by_id(current_user.id, include_profile=True)
    
    if not user_with_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserWithProfile.model_validate(user_with_profile)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> UserResponse:
    """Update current user's information."""
    user_service = UserService(db)
    
    try:
        updated_user = await user_service.update_user(current_user.id, user_data)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse.model_validate(updated_user)
        
    except Exception as e:
        logger.error("Error updating user", error=str(e), user_id=str(current_user.id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/me")
async def delete_current_user(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> dict:
    """Delete (deactivate) current user's account."""
    user_service = UserService(db)
    
    success = await user_service.delete_user(current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )
    
    return {"message": "User account deactivated successfully"}


# Profile endpoints
@router.post("/me/profile", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_user_profile(
    profile_data: UserProfileCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> UserProfileResponse:
    """Create user profile."""
    user_service = UserService(db)
    
    try:
        profile = await user_service.create_user_profile(current_user.id, profile_data)
        return UserProfileResponse.model_validate(profile)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error creating user profile", error=str(e), user_id=str(current_user.id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create profile"
        )


@router.get("/me/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> UserProfileResponse:
    """Get user profile."""
    user_service = UserService(db)
    profile = await user_service.get_user_profile(current_user.id)
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found"
        )
    
    return UserProfileResponse.model_validate(profile)


@router.put("/me/profile", response_model=UserProfileResponse)
async def update_user_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> UserProfileResponse:
    """Update user profile."""
    user_service = UserService(db)
    
    try:
        profile = await user_service.update_user_profile(current_user.id, profile_data)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        return UserProfileResponse.model_validate(profile)
        
    except Exception as e:
        logger.error("Error updating user profile", error=str(e), user_id=str(current_user.id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


# Skill assessment endpoints
@router.post("/me/skills", response_model=SkillAssessmentResponse, status_code=status.HTTP_201_CREATED)
async def create_skill_assessment(
    assessment_data: SkillAssessmentCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> SkillAssessmentResponse:
    """Create or update skill assessment."""
    user_service = UserService(db)
    
    try:
        assessment = await user_service.create_skill_assessment(current_user.id, assessment_data)
        return SkillAssessmentResponse.model_validate(assessment)
        
    except Exception as e:
        logger.error("Error creating skill assessment", error=str(e), user_id=str(current_user.id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create skill assessment"
        )


@router.get("/me/skills", response_model=List[SkillAssessmentResponse])
async def get_skill_assessments(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> List[SkillAssessmentResponse]:
    """Get all skill assessments for current user."""
    user_service = UserService(db)
    assessments = await user_service.get_user_skill_assessments(current_user.id)
    
    return [SkillAssessmentResponse.model_validate(assessment) for assessment in assessments]


@router.get("/me/skills/{domain}", response_model=SkillAssessmentResponse)
async def get_skill_assessment(
    domain: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> SkillAssessmentResponse:
    """Get skill assessment for specific domain."""
    user_service = UserService(db)
    assessment = await user_service.get_skill_assessment(current_user.id, domain)
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Skill assessment not found"
        )
    
    return SkillAssessmentResponse.model_validate(assessment)


# Learning activity endpoints
@router.post("/me/activities", response_model=LearningActivityResponse, status_code=status.HTTP_201_CREATED)
async def create_learning_activity(
    activity_data: LearningActivityCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> LearningActivityResponse:
    """Create learning activity."""
    user_service = UserService(db)
    
    try:
        activity = await user_service.create_learning_activity(current_user.id, activity_data)
        return LearningActivityResponse.model_validate(activity)
        
    except Exception as e:
        logger.error("Error creating learning activity", error=str(e), user_id=str(current_user.id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create learning activity"
        )


@router.get("/me/activities", response_model=List[LearningActivityResponse])
async def get_learning_activities(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> List[LearningActivityResponse]:
    """Get learning activities for current user."""
    user_service = UserService(db)
    activities = await user_service.get_user_learning_activities(current_user.id, limit)
    
    return [LearningActivityResponse.model_validate(activity) for activity in activities]