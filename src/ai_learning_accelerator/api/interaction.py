"""API endpoints for multi-input interaction system."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_db
from ..services.interaction_service import InteractionService
from ..schemas.interaction import (
    InteractionSessionCreate, InteractionSessionResponse,
    VoiceInputRequest, VoiceInteractionResponse,
    GestureInputRequest, GestureInteractionResponse,
    InputFusionCreate, InputFusionResponse,
    InteractionFeedbackCreate, InteractionFeedbackResponse,
    InputCalibrationCreate, InputCalibrationUpdate, InputCalibrationResponse,
    MultiModalInputRequest, InteractionAnalytics, InteractionSessionSummary,
    InputCapabilities
)
from ..models.interaction import InputType, GestureType, VoiceCommand
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/interaction", tags=["interaction"])


@router.post("/sessions", response_model=InteractionSessionResponse)
async def create_interaction_session(
    session_data: InteractionSessionCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new interaction session."""
    try:
        service = InteractionService(db)
        
        session = await service.create_interaction_session(
            user_id=session_data.user_id,
            device_info=session_data.device_info,
            accessibility_settings=session_data.accessibility_settings
        )
        
        return session
        
    except Exception as e:
        logger.error("Error creating interaction session", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create interaction session"
        )


@router.get("/sessions/{session_id}", response_model=InteractionSessionResponse)
async def get_interaction_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get interaction session details."""
    try:
        # This would be implemented with proper session retrieval
        # For now, return a placeholder response
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Session retrieval not yet implemented"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting interaction session", error=str(e), session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get interaction session"
        )


@router.post("/voice", response_model=VoiceInteractionResponse)
async def process_voice_input(
    request: VoiceInputRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """Process voice input and extract intent."""
    try:
        service = InteractionService(db)
        
        voice_interaction = await service.process_voice_input(
            session_id=request.session_id,
            audio_data=request.audio_data,
            context=request.context
        )
        
        return voice_interaction
        
    except Exception as e:
        logger.error("Error processing voice input", error=str(e), session_id=str(request.session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process voice input"
        )


@router.post("/gesture", response_model=GestureInteractionResponse)
async def process_gesture_input(
    request: GestureInputRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """Process gesture input and recognize gestures."""
    try:
        service = InteractionService(db)
        
        gesture_interaction = await service.process_gesture_input(
            session_id=request.session_id,
            gesture_data=request.gesture_data,
            context=request.context
        )
        
        return gesture_interaction
        
    except Exception as e:
        logger.error("Error processing gesture input", error=str(e), session_id=str(request.session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process gesture input"
        )


@router.post("/multimodal", response_model=InputFusionResponse)
async def process_multimodal_input(
    request: MultiModalInputRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """Process multiple input modalities and fuse them."""
    try:
        service = InteractionService(db)
        
        # Process each input type
        interaction_ids = []
        
        for input_data in request.inputs:
            input_type = InputType(input_data.get("type"))
            
            if input_type == InputType.VOICE:
                voice_interaction = await service.process_voice_input(
                    session_id=request.session_id,
                    audio_data=input_data.get("data", {}),
                    context=request.context
                )
                interaction_ids.append(voice_interaction.interaction_id)
                
            elif input_type == InputType.GESTURE:
                gesture_interaction = await service.process_gesture_input(
                    session_id=request.session_id,
                    gesture_data=input_data.get("data", {}),
                    context=request.context
                )
                interaction_ids.append(gesture_interaction.interaction_id)
        
        # Fuse the inputs if multiple
        if len(interaction_ids) > 1:
            fusion_result = await service.fuse_multimodal_inputs(
                session_id=request.session_id,
                interaction_ids=interaction_ids,
                fusion_algorithm=request.fusion_algorithm
            )
            return fusion_result
        else:
            # Single input, create a simple fusion record
            fusion_result = await service.fuse_multimodal_inputs(
                session_id=request.session_id,
                interaction_ids=interaction_ids,
                fusion_algorithm="single_input"
            )
            return fusion_result
        
    except Exception as e:
        logger.error("Error processing multimodal input", error=str(e), session_id=str(request.session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process multimodal input"
        )


@router.post("/fusion", response_model=InputFusionResponse)
async def fuse_interactions(
    request: InputFusionCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Fuse existing interactions into unified intent."""
    try:
        service = InteractionService(db)
        
        fusion_result = await service.fuse_multimodal_inputs(
            session_id=request.session_id,
            interaction_ids=request.interaction_ids,
            fusion_algorithm=request.fusion_algorithm
        )
        
        return fusion_result
        
    except Exception as e:
        logger.error("Error fusing interactions", error=str(e), session_id=str(request.session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fuse interactions"
        )


@router.post("/calibration", response_model=InputCalibrationResponse)
async def calibrate_user_input(
    calibration_data: InputCalibrationCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Calibrate user input for improved recognition."""
    try:
        service = InteractionService(db)
        
        calibration = await service.calibrate_user_input(
            user_id=calibration_data.user_id,
            input_type=calibration_data.input_type,
            calibration_data=calibration_data.calibration_data
        )
        
        return calibration
        
    except Exception as e:
        logger.error("Error calibrating user input", error=str(e), user_id=str(calibration_data.user_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calibrate user input"
        )


@router.get("/calibration/{user_id}/{input_type}", response_model=InputCalibrationResponse)
async def get_user_calibration(
    user_id: UUID,
    input_type: InputType,
    db: AsyncSession = Depends(get_async_db)
):
    """Get user's input calibration settings."""
    try:
        service = InteractionService(db)
        
        calibration = await service._get_user_calibration(user_id, input_type)
        
        if not calibration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Calibration not found"
            )
        
        return calibration
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting user calibration", error=str(e), user_id=str(user_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user calibration"
        )


@router.put("/calibration/{user_id}/{input_type}", response_model=InputCalibrationResponse)
async def update_user_calibration(
    user_id: UUID,
    input_type: InputType,
    calibration_update: InputCalibrationUpdate,
    db: AsyncSession = Depends(get_async_db)
):
    """Update user's input calibration settings."""
    try:
        service = InteractionService(db)
        
        # Get existing calibration
        existing_calibration = await service._get_user_calibration(user_id, input_type)
        
        if not existing_calibration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Calibration not found"
            )
        
        # Update calibration
        update_data = calibration_update.model_dump(exclude_unset=True)
        calibration = await service.calibrate_user_input(
            user_id=user_id,
            input_type=input_type,
            calibration_data=update_data
        )
        
        return calibration
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating user calibration", error=str(e), user_id=str(user_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user calibration"
        )


@router.post("/feedback", response_model=InteractionFeedbackResponse)
async def create_interaction_feedback(
    feedback: InteractionFeedbackCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Create feedback for an interaction."""
    try:
        # This would be implemented with proper feedback service
        # For now, return a placeholder response
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Feedback creation not yet implemented"
        )
        
    except Exception as e:
        logger.error("Error creating interaction feedback", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create interaction feedback"
        )


@router.get("/analytics/{session_id}", response_model=InteractionAnalytics)
async def get_interaction_analytics(
    session_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get analytics for an interaction session."""
    try:
        # This is a placeholder implementation
        # In production, this would query actual analytics data
        
        analytics = InteractionAnalytics(
            session_id=session_id,
            total_interactions=25,
            successful_interactions=23,
            error_rate=0.08,
            average_confidence=0.82,
            input_type_distribution={
                "voice": 12,
                "gesture": 8,
                "text": 5
            },
            intent_distribution={
                "navigate": 8,
                "select": 6,
                "help": 4,
                "search": 3,
                "explain": 4
            },
            response_time_stats={
                "average": 1.2,
                "median": 0.9,
                "p95": 2.1
            },
            user_satisfaction=4.1,
            calibration_effectiveness={
                "voice": 0.15,
                "gesture": 0.12
            },
            improvement_recommendations=[
                "Consider voice calibration for better recognition",
                "Gesture sensitivity could be adjusted",
                "Multi-modal fusion is working well"
            ]
        )
        
        return analytics
        
    except Exception as e:
        logger.error("Error getting interaction analytics", error=str(e), session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get interaction analytics"
        )


@router.get("/sessions/{session_id}/summary", response_model=InteractionSessionSummary)
async def get_session_summary(
    session_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get summary of an interaction session."""
    try:
        # This is a placeholder implementation
        # In production, this would query actual session data
        
        summary = InteractionSessionSummary(
            session_id=session_id,
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            duration=1800,  # 30 minutes
            interaction_count=25,
            success_rate=0.92,
            primary_input_types=["voice", "gesture"],
            most_common_intents=["navigate", "select", "help"],
            average_confidence=0.82,
            errors_encountered=["speech_recognition_low_confidence", "gesture_ambiguous"],
            user_feedback_summary={
                "average_satisfaction": 4.1,
                "most_liked": "voice_commands",
                "needs_improvement": "gesture_sensitivity"
            },
            recommendations=[
                "Voice recognition is working well",
                "Consider adjusting gesture sensitivity",
                "Multi-modal interactions are effective"
            ]
        )
        
        return summary
        
    except Exception as e:
        logger.error("Error getting session summary", error=str(e), session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session summary"
        )


@router.get("/capabilities", response_model=InputCapabilities)
async def get_input_capabilities():
    """Get available input capabilities and recommendations."""
    try:
        capabilities = InputCapabilities(
            supported_input_types=["text", "voice", "gesture", "touch", "keyboard", "mouse"],
            voice_capabilities={
                "languages": ["en", "es", "fr", "de", "zh"],
                "commands": [cmd.value for cmd in VoiceCommand],
                "noise_reduction": True,
                "real_time_processing": True
            },
            gesture_capabilities={
                "gesture_types": [gesture.value for gesture in GestureType],
                "multi_touch": True,
                "pressure_sensitivity": False,
                "3d_gestures": False
            },
            accessibility_features=[
                "screen_reader_support",
                "high_contrast_mode",
                "large_text_support",
                "keyboard_navigation",
                "voice_commands"
            ],
            calibration_required=["voice", "gesture"],
            recommended_settings={
                "voice_sensitivity": 0.7,
                "gesture_sensitivity": 0.6,
                "multi_modal_fusion": True,
                "error_correction": True
            }
        )
        
        return capabilities
        
    except Exception as e:
        logger.error("Error getting input capabilities", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get input capabilities"
        )


@router.get("/input-types", response_model=List[str])
async def get_input_types():
    """Get available input types."""
    return [input_type.value for input_type in InputType]


@router.get("/voice-commands", response_model=List[str])
async def get_voice_commands():
    """Get available voice commands."""
    return [command.value for command in VoiceCommand]


@router.get("/gesture-types", response_model=List[str])
async def get_gesture_types():
    """Get available gesture types."""
    return [gesture.value for gesture in GestureType]