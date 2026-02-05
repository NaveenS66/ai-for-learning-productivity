"""Pydantic schemas for multi-input interaction API operations."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from ..models.interaction import InputType, InputModality, GestureType, VoiceCommand


class InteractionSessionBase(BaseModel):
    """Base interaction session schema."""
    device_info: Dict[str, Any] = Field(default_factory=dict)
    browser_info: Dict[str, Any] = Field(default_factory=dict)
    accessibility_settings: Dict[str, Any] = Field(default_factory=dict)
    preferred_input_types: List[str] = Field(default_factory=list)
    input_modality: InputModality = InputModality.UNIMODAL


class InteractionSessionCreate(InteractionSessionBase):
    """Schema for interaction session creation."""
    user_id: UUID


class InteractionSessionResponse(InteractionSessionBase):
    """Schema for interaction session responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    session_token: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[int] = None
    total_interactions: int
    successful_interactions: int
    error_count: int
    created_at: datetime
    updated_at: datetime


class UserInteractionBase(BaseModel):
    """Base user interaction schema."""
    input_type: InputType
    input_data: Dict[str, Any]
    content_context: Dict[str, Any] = Field(default_factory=dict)
    ui_context: Dict[str, Any] = Field(default_factory=dict)
    learning_context: Dict[str, Any] = Field(default_factory=dict)


class UserInteractionCreate(UserInteractionBase):
    """Schema for user interaction creation."""
    session_id: UUID


class UserInteractionResponse(UserInteractionBase):
    """Schema for user interaction responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    session_id: UUID
    processed_data: Dict[str, Any]
    timestamp: datetime
    processing_time: Optional[float] = None
    response_time: Optional[float] = None
    intent_recognized: Optional[str] = None
    confidence_score: float
    action_taken: Optional[str] = None
    success: bool
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class VoiceInteractionBase(BaseModel):
    """Base voice interaction schema."""
    audio_url: Optional[str] = None
    audio_duration: Optional[float] = None
    raw_transcript: Optional[str] = None
    processed_transcript: Optional[str] = None
    language_detected: str = "en"
    command_type: Optional[VoiceCommand] = None
    command_parameters: Dict[str, Any] = Field(default_factory=dict)
    natural_language_query: Optional[str] = None


class VoiceInteractionCreate(VoiceInteractionBase):
    """Schema for voice interaction creation."""
    session_id: UUID
    audio_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class VoiceInteractionResponse(VoiceInteractionBase):
    """Schema for voice interaction responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    interaction_id: UUID
    audio_quality: float
    speech_recognition_confidence: float
    intent_recognition_confidence: float
    noise_level: float
    response_text: Optional[str] = None
    response_audio_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class GestureInteractionBase(BaseModel):
    """Base gesture interaction schema."""
    gesture_type: GestureType
    start_position: Dict[str, Any] = Field(default_factory=dict)
    end_position: Dict[str, Any] = Field(default_factory=dict)
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    velocity: Optional[float] = None
    acceleration: Optional[float] = None
    target_element: Optional[str] = None
    gesture_area: Dict[str, Any] = Field(default_factory=dict)


class GestureInteractionCreate(GestureInteractionBase):
    """Schema for gesture interaction creation."""
    session_id: UUID
    gesture_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class GestureInteractionResponse(GestureInteractionBase):
    """Schema for gesture interaction responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    interaction_id: UUID
    gesture_data: Dict[str, Any]
    recognition_confidence: float
    gesture_quality: float
    action_triggered: Optional[str] = None
    feedback_provided: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class InputFusionBase(BaseModel):
    """Base input fusion schema."""
    input_types: List[str]
    interaction_ids: List[str]
    fusion_algorithm: str = "weighted_average"
    fusion_parameters: Dict[str, Any] = Field(default_factory=dict)
    combined_intent: Optional[str] = None
    combined_confidence: float = 0.0
    fusion_quality: float = 0.0
    conflicts_detected: List[Dict[str, Any]] = Field(default_factory=list)
    conflict_resolution: Dict[str, Any] = Field(default_factory=dict)
    final_action: Optional[str] = None
    action_parameters: Dict[str, Any] = Field(default_factory=dict)
    user_confirmation_required: bool = False


class InputFusionCreate(BaseModel):
    """Schema for input fusion creation."""
    session_id: UUID
    interaction_ids: List[UUID]
    fusion_algorithm: str = "weighted_average"


class InputFusionResponse(InputFusionBase):
    """Schema for input fusion responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    session_id: UUID
    fusion_timestamp: datetime
    created_at: datetime
    updated_at: datetime


class InteractionFeedbackBase(BaseModel):
    """Base interaction feedback schema."""
    satisfaction_rating: int = Field(..., ge=1, le=5)
    accuracy_rating: Optional[int] = Field(None, ge=1, le=5)
    speed_rating: Optional[int] = Field(None, ge=1, le=5)
    ease_of_use_rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_text: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    issues_encountered: List[str] = Field(default_factory=list)
    error_recovery_success: Optional[bool] = None
    feedback_context: Dict[str, Any] = Field(default_factory=dict)


class InteractionFeedbackCreate(InteractionFeedbackBase):
    """Schema for interaction feedback creation."""
    interaction_id: UUID
    user_id: UUID


class InteractionFeedbackResponse(InteractionFeedbackBase):
    """Schema for interaction feedback responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    interaction_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime


class InputCalibrationBase(BaseModel):
    """Base input calibration schema."""
    input_type: InputType
    calibration_data: Dict[str, Any]
    voice_profile: Dict[str, Any] = Field(default_factory=dict)
    speech_patterns: Dict[str, Any] = Field(default_factory=dict)
    accent_adaptation: Dict[str, Any] = Field(default_factory=dict)
    gesture_preferences: Dict[str, Any] = Field(default_factory=dict)
    gesture_sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    custom_gestures: Dict[str, Any] = Field(default_factory=dict)
    auto_calibration_enabled: bool = True


class InputCalibrationCreate(InputCalibrationBase):
    """Schema for input calibration creation."""
    user_id: UUID


class InputCalibrationUpdate(BaseModel):
    """Schema for input calibration updates."""
    calibration_data: Optional[Dict[str, Any]] = None
    voice_profile: Optional[Dict[str, Any]] = None
    speech_patterns: Optional[Dict[str, Any]] = None
    accent_adaptation: Optional[Dict[str, Any]] = None
    gesture_preferences: Optional[Dict[str, Any]] = None
    gesture_sensitivity: Optional[float] = Field(None, ge=0.0, le=1.0)
    custom_gestures: Optional[Dict[str, Any]] = None
    auto_calibration_enabled: Optional[bool] = None


class InputCalibrationResponse(InputCalibrationBase):
    """Schema for input calibration responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    accuracy_improvement: float
    speed_improvement: float
    user_satisfaction: float
    calibration_date: datetime
    calibration_sessions: int
    created_at: datetime
    updated_at: datetime


class VoiceInputRequest(BaseModel):
    """Schema for voice input processing requests."""
    session_id: UUID
    audio_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    language_hint: Optional[str] = "en"
    noise_reduction: bool = True
    real_time_processing: bool = False


class GestureInputRequest(BaseModel):
    """Schema for gesture input processing requests."""
    session_id: UUID
    gesture_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    sensitivity_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    custom_gesture_enabled: bool = True


class MultiModalInputRequest(BaseModel):
    """Schema for multi-modal input processing requests."""
    session_id: UUID
    inputs: List[Dict[str, Any]]
    fusion_algorithm: str = "weighted_average"
    context: Optional[Dict[str, Any]] = None
    require_confirmation: bool = False


class InteractionAnalytics(BaseModel):
    """Schema for interaction analytics."""
    session_id: UUID
    total_interactions: int
    successful_interactions: int
    error_rate: float
    average_confidence: float
    input_type_distribution: Dict[str, int]
    intent_distribution: Dict[str, int]
    response_time_stats: Dict[str, float]
    user_satisfaction: float
    calibration_effectiveness: Dict[str, float]
    improvement_recommendations: List[str]


class InteractionSessionSummary(BaseModel):
    """Schema for interaction session summary."""
    session_id: UUID
    user_id: UUID
    duration: int
    interaction_count: int
    success_rate: float
    primary_input_types: List[str]
    most_common_intents: List[str]
    average_confidence: float
    errors_encountered: List[str]
    user_feedback_summary: Dict[str, Any]
    recommendations: List[str]


class InputCapabilities(BaseModel):
    """Schema for device input capabilities."""
    supported_input_types: List[str]
    voice_capabilities: Dict[str, Any] = Field(default_factory=dict)
    gesture_capabilities: Dict[str, Any] = Field(default_factory=dict)
    accessibility_features: List[str] = Field(default_factory=list)
    calibration_required: List[str] = Field(default_factory=list)
    recommended_settings: Dict[str, Any] = Field(default_factory=dict)