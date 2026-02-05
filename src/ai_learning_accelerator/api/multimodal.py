"""API endpoints for multi-modal content delivery."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_db
from ..services.content_adaptation import ContentAdaptationService
from ..schemas.multimodal import (
    ContentAdaptationRequest, ContentAdaptationResponse,
    VisualGenerationRequest, VisualContentResponse,
    InteractiveGenerationRequest, InteractiveContentResponse,
    AccessibilityGenerationRequest, AccessibilityAdaptationResponse,
    UserAdaptationPreferenceCreate, UserAdaptationPreferenceUpdate, UserAdaptationPreferenceResponse,
    AdaptationFeedbackCreate, AdaptationFeedbackResponse,
    MultiModalContentResponse, AdaptationAnalytics,
    ContentAdaptationBatchRequest
)
from ..models.multimodal import AdaptationMode, VisualType, InteractionType, AccessibilityFeature
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/multimodal", tags=["multimodal"])


@router.post("/adapt", response_model=List[ContentAdaptationResponse])
async def adapt_content(
    request: ContentAdaptationRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """Adapt content for multi-modal delivery."""
    try:
        service = ContentAdaptationService(db)
        
        # For now, we'll use a default user_id since we don't have auth context
        # In production, this would come from the authenticated user
        default_user_id = UUID("00000000-0000-0000-0000-000000000001")
        
        adaptations = await service.adapt_content_for_user(
            content_id=request.content_id,
            user_id=default_user_id,
            adaptation_modes=request.adaptation_modes
        )
        
        if not adaptations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found or adaptation failed"
            )
        
        return adaptations
        
    except Exception as e:
        logger.error("Error adapting content", error=str(e), content_id=str(request.content_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to adapt content"
        )


@router.post("/adapt/batch", response_model=List[ContentAdaptationResponse])
async def adapt_content_batch(
    request: ContentAdaptationBatchRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """Batch adapt multiple content items."""
    try:
        service = ContentAdaptationService(db)
        default_user_id = UUID("00000000-0000-0000-0000-000000000001")
        
        all_adaptations = []
        for content_id in request.content_ids:
            adaptations = await service.adapt_content_for_user(
                content_id=content_id,
                user_id=default_user_id,
                adaptation_modes=request.adaptation_modes
            )
            all_adaptations.extend(adaptations)
        
        return all_adaptations
        
    except Exception as e:
        logger.error("Error in batch content adaptation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to adapt content batch"
        )


@router.post("/visual/generate", response_model=VisualContentResponse)
async def generate_visual_content(
    request: VisualGenerationRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """Generate visual content from text."""
    try:
        service = ContentAdaptationService(db)
        
        # Get the content first
        content = await service._get_content(request.content_id)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        
        visual_content = await service.generate_text_to_visual(
            content=content,
            visual_type=request.visual_type,
            user_preferences=request.user_preferences
        )
        
        if not visual_content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to generate visual content"
            )
        
        return visual_content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating visual content", error=str(e), content_id=str(request.content_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate visual content"
        )


@router.post("/interactive/generate", response_model=InteractiveContentResponse)
async def generate_interactive_content(
    request: InteractiveGenerationRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """Generate interactive content examples."""
    try:
        service = ContentAdaptationService(db)
        
        # Get the content first
        content = await service._get_content(request.content_id)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        
        interactive_content = await service.generate_interactive_example(
            content=content,
            interaction_type=request.interaction_type,
            user_preferences=request.user_preferences
        )
        
        if not interactive_content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to generate interactive content"
            )
        
        return interactive_content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating interactive content", error=str(e), content_id=str(request.content_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate interactive content"
        )


@router.post("/accessibility/generate", response_model=AccessibilityAdaptationResponse)
async def generate_accessibility_adaptation(
    request: AccessibilityGenerationRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """Generate accessibility adaptations for content."""
    try:
        service = ContentAdaptationService(db)
        
        # Get the content first
        content = await service._get_content(request.content_id)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        
        accessibility_adaptation = await service.generate_accessibility_adaptation(
            content=content,
            accessibility_features=request.accessibility_features,
            user_preferences=request.user_preferences
        )
        
        if not accessibility_adaptation:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to generate accessibility adaptation"
            )
        
        return accessibility_adaptation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating accessibility adaptation", error=str(e), content_id=str(request.content_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate accessibility adaptation"
        )


@router.get("/content/{content_id}/multimodal", response_model=MultiModalContentResponse)
async def get_multimodal_content(
    content_id: UUID,
    user_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get complete multi-modal content package."""
    try:
        service = ContentAdaptationService(db)
        
        # Use default user if not provided
        if not user_id:
            user_id = UUID("00000000-0000-0000-0000-000000000001")
        
        # Get original content
        content = await service._get_content(content_id)
        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        
        # Get all adaptations for this content
        adaptations = await service.adapt_content_for_user(
            content_id=content_id,
            user_id=user_id
        )
        
        # Get user preferences
        user_prefs = await service._get_user_adaptation_preferences(user_id)
        
        response = MultiModalContentResponse(
            original_content={
                "id": str(content.id),
                "title": content.title,
                "content_type": content.content_type.value,
                "difficulty_level": content.difficulty_level.value,
                "content_text": content.content_text
            },
            adaptations=adaptations,
            user_preferences=user_prefs,
            recommendation_score=0.85,  # Placeholder score
            usage_analytics={}
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting multimodal content", error=str(e), content_id=str(content_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get multimodal content"
        )


@router.get("/adaptations/{adaptation_id}/analytics", response_model=AdaptationAnalytics)
async def get_adaptation_analytics(
    adaptation_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get analytics for a specific adaptation."""
    try:
        # This is a placeholder implementation
        # In production, this would query actual analytics data
        
        analytics = AdaptationAnalytics(
            adaptation_id=adaptation_id,
            total_usage=150,
            unique_users=45,
            average_rating=4.2,
            completion_rate=0.78,
            engagement_metrics={
                "average_time_spent": 320,
                "interaction_rate": 0.65,
                "return_rate": 0.42
            },
            accessibility_usage={
                "screen_reader": 12,
                "high_contrast": 8,
                "large_text": 15
            },
            performance_metrics={
                "load_time": 1.2,
                "error_rate": 0.02,
                "success_rate": 0.98
            },
            user_feedback_summary={
                "positive_feedback": 0.82,
                "common_issues": ["loading_time", "mobile_compatibility"],
                "improvement_requests": ["more_interactive_elements", "better_navigation"]
            },
            improvement_recommendations=[
                "Optimize loading performance for mobile devices",
                "Add more interactive elements for engagement",
                "Improve navigation accessibility"
            ]
        )
        
        return analytics
        
    except Exception as e:
        logger.error("Error getting adaptation analytics", error=str(e), adaptation_id=str(adaptation_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get adaptation analytics"
        )


@router.get("/modes", response_model=List[str])
async def get_adaptation_modes():
    """Get available adaptation modes."""
    return [mode.value for mode in AdaptationMode]


@router.get("/visual-types", response_model=List[str])
async def get_visual_types():
    """Get available visual content types."""
    return [vtype.value for vtype in VisualType]


@router.get("/interaction-types", response_model=List[str])
async def get_interaction_types():
    """Get available interaction types."""
    return [itype.value for itype in InteractionType]


@router.get("/accessibility-features", response_model=List[str])
async def get_accessibility_features():
    """Get available accessibility features."""
    return [feature.value for feature in AccessibilityFeature]


# User Preference Management Endpoints

@router.post("/preferences", response_model=UserAdaptationPreferenceResponse)
async def create_user_preferences(
    preferences: UserAdaptationPreferenceCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Create user adaptation preferences."""
    try:
        # This would be implemented with a proper service
        # For now, return a placeholder response
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User preference management not yet implemented"
        )
        
    except Exception as e:
        logger.error("Error creating user preferences", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user preferences"
        )


@router.get("/preferences/{user_id}", response_model=UserAdaptationPreferenceResponse)
async def get_user_preferences(
    user_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get user adaptation preferences."""
    try:
        service = ContentAdaptationService(db)
        preferences = await service._get_user_adaptation_preferences(user_id)
        
        if not preferences:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User preferences not found"
            )
        
        return preferences
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting user preferences", error=str(e), user_id=str(user_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user preferences"
        )


@router.put("/preferences/{user_id}", response_model=UserAdaptationPreferenceResponse)
async def update_user_preferences(
    user_id: UUID,
    preferences: UserAdaptationPreferenceUpdate,
    db: AsyncSession = Depends(get_async_db)
):
    """Update user adaptation preferences."""
    try:
        # This would be implemented with a proper service
        # For now, return a placeholder response
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="User preference updates not yet implemented"
        )
        
    except Exception as e:
        logger.error("Error updating user preferences", error=str(e), user_id=str(user_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user preferences"
        )


# Feedback Endpoints

@router.post("/feedback", response_model=AdaptationFeedbackResponse)
async def create_adaptation_feedback(
    feedback: AdaptationFeedbackCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Create feedback for an adaptation."""
    try:
        # This would be implemented with a proper service
        # For now, return a placeholder response
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Feedback creation not yet implemented"
        )
        
    except Exception as e:
        logger.error("Error creating adaptation feedback", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create adaptation feedback"
        )