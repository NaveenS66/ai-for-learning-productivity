"""Pydantic schemas for multi-modal content API operations."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from ..models.multimodal import (
    AdaptationMode, VisualType, InteractionType, AccessibilityFeature
)


class ContentAdaptationBase(BaseModel):
    """Base content adaptation schema."""
    adaptation_mode: AdaptationMode
    target_format: str
    adapted_content: Dict[str, Any]
    adaptation_metadata: Dict[str, Any] = Field(default_factory=dict)
    generation_method: Optional[str] = None
    generation_parameters: Dict[str, Any] = Field(default_factory=dict)


class ContentAdaptationCreate(ContentAdaptationBase):
    """Schema for content adaptation creation."""
    source_content_id: UUID


class ContentAdaptationUpdate(BaseModel):
    """Schema for content adaptation updates."""
    adaptation_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    adapted_content: Optional[Dict[str, Any]] = None
    adaptation_metadata: Optional[Dict[str, Any]] = None


class ContentAdaptationResponse(ContentAdaptationBase):
    """Schema for content adaptation responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    source_content_id: UUID
    adaptation_quality: float
    usage_count: int
    user_rating: float
    feedback_count: int
    created_at: datetime
    updated_at: datetime


class VisualContentBase(BaseModel):
    """Base visual content schema."""
    visual_type: VisualType
    title: str = Field(..., min_length=1, max_length=300)
    description: Optional[str] = None
    svg_content: Optional[str] = None
    image_url: Optional[str] = None
    interactive_data: Dict[str, Any] = Field(default_factory=dict)
    layout_config: Dict[str, Any] = Field(default_factory=dict)
    style_config: Dict[str, Any] = Field(default_factory=dict)
    responsive_config: Dict[str, Any] = Field(default_factory=dict)
    alt_text: Optional[str] = None
    aria_labels: Dict[str, Any] = Field(default_factory=dict)


class VisualContentCreate(VisualContentBase):
    """Schema for visual content creation."""
    adaptation_id: UUID


class VisualContentUpdate(BaseModel):
    """Schema for visual content updates."""
    title: Optional[str] = Field(None, min_length=1, max_length=300)
    description: Optional[str] = None
    svg_content: Optional[str] = None
    image_url: Optional[str] = None
    interactive_data: Optional[Dict[str, Any]] = None
    layout_config: Optional[Dict[str, Any]] = None
    style_config: Optional[Dict[str, Any]] = None
    responsive_config: Optional[Dict[str, Any]] = None
    alt_text: Optional[str] = None
    aria_labels: Optional[Dict[str, Any]] = None


class VisualContentResponse(VisualContentBase):
    """Schema for visual content responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    adaptation_id: UUID
    created_at: datetime
    updated_at: datetime


class InteractiveContentBase(BaseModel):
    """Base interactive content schema."""
    interaction_type: InteractionType
    title: str = Field(..., min_length=1, max_length=300)
    instructions: Optional[str] = None
    config_data: Dict[str, Any]
    initial_state: Dict[str, Any] = Field(default_factory=dict)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    learning_objectives: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    feedback_config: Dict[str, Any] = Field(default_factory=dict)
    hint_system: Dict[str, Any] = Field(default_factory=dict)
    completion_tracking: bool = True
    analytics_config: Dict[str, Any] = Field(default_factory=dict)


class InteractiveContentCreate(InteractiveContentBase):
    """Schema for interactive content creation."""
    adaptation_id: UUID


class InteractiveContentUpdate(BaseModel):
    """Schema for interactive content updates."""
    title: Optional[str] = Field(None, min_length=1, max_length=300)
    instructions: Optional[str] = None
    config_data: Optional[Dict[str, Any]] = None
    initial_state: Optional[Dict[str, Any]] = None
    validation_rules: Optional[Dict[str, Any]] = None
    learning_objectives: Optional[List[str]] = None
    success_criteria: Optional[List[str]] = None
    feedback_config: Optional[Dict[str, Any]] = None
    hint_system: Optional[Dict[str, Any]] = None
    completion_tracking: Optional[bool] = None
    analytics_config: Optional[Dict[str, Any]] = None


class InteractiveContentResponse(InteractiveContentBase):
    """Schema for interactive content responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    adaptation_id: UUID
    created_at: datetime
    updated_at: datetime


class AccessibilityAdaptationBase(BaseModel):
    """Base accessibility adaptation schema."""
    accessibility_features: List[str]
    target_disabilities: List[str] = Field(default_factory=list)
    screen_reader_content: Optional[str] = None
    high_contrast_css: Optional[str] = None
    large_text_css: Optional[str] = None
    audio_description: Optional[str] = None
    audio_url: Optional[str] = None
    sign_language_video_url: Optional[str] = None
    sign_language_description: Optional[str] = None
    simplified_text: Optional[str] = None
    reading_level: Optional[str] = None
    keyboard_shortcuts: Dict[str, Any] = Field(default_factory=dict)
    focus_indicators: Dict[str, Any] = Field(default_factory=dict)


class AccessibilityAdaptationCreate(AccessibilityAdaptationBase):
    """Schema for accessibility adaptation creation."""
    adaptation_id: UUID


class AccessibilityAdaptationUpdate(BaseModel):
    """Schema for accessibility adaptation updates."""
    accessibility_features: Optional[List[str]] = None
    target_disabilities: Optional[List[str]] = None
    screen_reader_content: Optional[str] = None
    high_contrast_css: Optional[str] = None
    large_text_css: Optional[str] = None
    audio_description: Optional[str] = None
    audio_url: Optional[str] = None
    sign_language_video_url: Optional[str] = None
    sign_language_description: Optional[str] = None
    simplified_text: Optional[str] = None
    reading_level: Optional[str] = None
    keyboard_shortcuts: Optional[Dict[str, Any]] = None
    focus_indicators: Optional[Dict[str, Any]] = None


class AccessibilityAdaptationResponse(AccessibilityAdaptationBase):
    """Schema for accessibility adaptation responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    adaptation_id: UUID
    created_at: datetime
    updated_at: datetime


class UserAdaptationPreferenceBase(BaseModel):
    """Base user adaptation preference schema."""
    preferred_modes: List[str] = Field(default_factory=list)
    visual_preferences: Dict[str, Any] = Field(default_factory=dict)
    interaction_preferences: Dict[str, Any] = Field(default_factory=dict)
    accessibility_needs: List[str] = Field(default_factory=list)
    learning_style: Optional[str] = None
    complexity_preference: str = "adaptive"
    pace_preference: str = "adaptive"
    preferred_formats: List[str] = Field(default_factory=list)
    avoided_formats: List[str] = Field(default_factory=list)
    device_preferences: Dict[str, Any] = Field(default_factory=dict)
    context_preferences: Dict[str, Any] = Field(default_factory=dict)


class UserAdaptationPreferenceCreate(UserAdaptationPreferenceBase):
    """Schema for user adaptation preference creation."""
    user_id: UUID


class UserAdaptationPreferenceUpdate(BaseModel):
    """Schema for user adaptation preference updates."""
    preferred_modes: Optional[List[str]] = None
    visual_preferences: Optional[Dict[str, Any]] = None
    interaction_preferences: Optional[Dict[str, Any]] = None
    accessibility_needs: Optional[List[str]] = None
    learning_style: Optional[str] = None
    complexity_preference: Optional[str] = None
    pace_preference: Optional[str] = None
    preferred_formats: Optional[List[str]] = None
    avoided_formats: Optional[List[str]] = None
    device_preferences: Optional[Dict[str, Any]] = None
    context_preferences: Optional[Dict[str, Any]] = None


class UserAdaptationPreferenceResponse(UserAdaptationPreferenceBase):
    """Schema for user adaptation preference responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime


class AdaptationFeedbackBase(BaseModel):
    """Base adaptation feedback schema."""
    rating: int = Field(..., ge=1, le=5)
    usefulness_rating: Optional[int] = Field(None, ge=1, le=5)
    clarity_rating: Optional[int] = Field(None, ge=1, le=5)
    accessibility_rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_text: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    usage_context: Dict[str, Any] = Field(default_factory=dict)
    device_info: Dict[str, Any] = Field(default_factory=dict)


class AdaptationFeedbackCreate(AdaptationFeedbackBase):
    """Schema for adaptation feedback creation."""
    adaptation_id: UUID
    user_id: UUID


class AdaptationFeedbackUpdate(BaseModel):
    """Schema for adaptation feedback updates."""
    rating: Optional[int] = Field(None, ge=1, le=5)
    usefulness_rating: Optional[int] = Field(None, ge=1, le=5)
    clarity_rating: Optional[int] = Field(None, ge=1, le=5)
    accessibility_rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_text: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    usage_context: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None


class AdaptationFeedbackResponse(AdaptationFeedbackBase):
    """Schema for adaptation feedback responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    adaptation_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime


class ContentAdaptationRequest(BaseModel):
    """Schema for content adaptation requests."""
    content_id: UUID
    adaptation_modes: Optional[List[AdaptationMode]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    target_formats: Optional[List[str]] = None


class ContentAdaptationBatchRequest(BaseModel):
    """Schema for batch content adaptation requests."""
    content_ids: List[UUID]
    adaptation_modes: List[AdaptationMode]
    user_preferences: Optional[Dict[str, Any]] = None
    priority: str = "normal"  # normal, high, low


class VisualGenerationRequest(BaseModel):
    """Schema for visual content generation requests."""
    content_id: UUID
    visual_type: VisualType
    user_preferences: Optional[Dict[str, Any]] = None
    style_preferences: Optional[Dict[str, Any]] = None
    size_constraints: Optional[Dict[str, Any]] = None


class InteractiveGenerationRequest(BaseModel):
    """Schema for interactive content generation requests."""
    content_id: UUID
    interaction_type: InteractionType
    user_preferences: Optional[Dict[str, Any]] = None
    difficulty_adjustment: Optional[str] = None
    learning_objectives: Optional[List[str]] = None


class AccessibilityGenerationRequest(BaseModel):
    """Schema for accessibility adaptation generation requests."""
    content_id: UUID
    accessibility_features: List[AccessibilityFeature]
    user_preferences: Optional[Dict[str, Any]] = None
    target_disabilities: Optional[List[str]] = None
    compliance_standards: Optional[List[str]] = None  # WCAG, Section 508, etc.


class MultiModalContentResponse(BaseModel):
    """Schema for complete multi-modal content responses."""
    original_content: Dict[str, Any]
    adaptations: List[ContentAdaptationResponse]
    visual_content: List[VisualContentResponse] = Field(default_factory=list)
    interactive_content: List[InteractiveContentResponse] = Field(default_factory=list)
    accessibility_adaptations: List[AccessibilityAdaptationResponse] = Field(default_factory=list)
    user_preferences: Optional[UserAdaptationPreferenceResponse] = None
    recommendation_score: float = Field(0.0, ge=0.0, le=1.0)
    usage_analytics: Dict[str, Any] = Field(default_factory=dict)


class AdaptationAnalytics(BaseModel):
    """Schema for adaptation analytics and metrics."""
    adaptation_id: UUID
    total_usage: int
    unique_users: int
    average_rating: float
    completion_rate: float
    engagement_metrics: Dict[str, Any]
    accessibility_usage: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    user_feedback_summary: Dict[str, Any]
    improvement_recommendations: List[str]


class ContentAdaptationSummary(BaseModel):
    """Schema for content adaptation summary."""
    content_id: UUID
    total_adaptations: int
    adaptation_modes: List[str]
    average_quality_score: float
    total_usage: int
    user_satisfaction: float
    accessibility_coverage: float
    last_updated: datetime
    recommendations: List[str]