"""Pydantic schemas for context analyzer API."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field

from ..models.context import (
    WorkspaceEventType, TechnologyType, KnowledgeGapSeverity,
    RecommendationType, NotificationPriority
)
from ..models.user import SkillLevel


class WorkspaceSessionCreate(BaseModel):
    """Schema for creating a workspace session."""
    workspace_path: str = Field(..., description="Path to the workspace root")
    session_name: Optional[str] = Field(None, description="Optional session name")
    project_name: Optional[str] = Field(None, description="Project name")
    primary_language: Optional[str] = Field(None, description="Primary programming language")
    ide_name: Optional[str] = Field(None, description="IDE being used")
    git_branch: Optional[str] = Field(None, description="Current git branch")


class WorkspaceSessionResponse(BaseModel):
    """Schema for workspace session response."""
    id: UUID
    user_id: UUID
    session_name: Optional[str]
    workspace_path: str
    project_name: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    last_activity_at: datetime
    total_duration_minutes: Optional[int]
    primary_language: Optional[str]
    detected_technologies: List[Dict[str, Any]]
    active_files: List[str]
    is_active: bool
    is_idle: bool
    idle_since: Optional[datetime]
    ide_name: Optional[str]
    git_branch: Optional[str]
    session_data: Dict[str, Any]

    class Config:
        from_attributes = True


class WorkspaceEventCreate(BaseModel):
    """Schema for creating a workspace event."""
    event_type: WorkspaceEventType
    event_description: Optional[str] = Field(None, description="Description of the event")
    file_path: Optional[str] = Field(None, description="File path related to event")
    file_extension: Optional[str] = Field(None, description="File extension")
    directory_path: Optional[str] = Field(None, description="Directory path related to event")
    event_data: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    before_state: Optional[Dict[str, Any]] = Field(None, description="State before the event")
    after_state: Optional[Dict[str, Any]] = Field(None, description="State after the event")


class WorkspaceEventResponse(BaseModel):
    """Schema for workspace event response."""
    id: UUID
    session_id: UUID
    user_id: UUID
    event_type: WorkspaceEventType
    event_description: Optional[str]
    file_path: Optional[str]
    file_extension: Optional[str]
    directory_path: Optional[str]
    event_data: Dict[str, Any]
    before_state: Optional[Dict[str, Any]]
    after_state: Optional[Dict[str, Any]]
    event_timestamp: datetime
    processing_duration_ms: Optional[int]
    triggered_analysis: bool
    generated_recommendations: int
    detected_patterns: List[Dict[str, Any]]

    class Config:
        from_attributes = True


class TechnologyStackResponse(BaseModel):
    """Schema for technology stack response."""
    id: UUID
    user_id: UUID
    workspace_path: str
    technology_name: str
    technology_type: TechnologyType
    version: Optional[str]
    detection_confidence: float
    detection_method: str
    detection_evidence: List[Dict[str, Any]]
    is_primary: bool
    usage_frequency: float
    last_used_at: Optional[datetime]
    first_detected_at: datetime
    last_updated_at: datetime
    is_active: bool
    user_proficiency: Optional[SkillLevel]
    learning_resources_suggested: int

    class Config:
        from_attributes = True


class KnowledgeGapResponse(BaseModel):
    """Schema for knowledge gap response."""
    id: UUID
    user_id: UUID
    technology_id: Optional[UUID]
    gap_title: str
    gap_description: str
    gap_category: str
    severity: KnowledgeGapSeverity
    confidence_score: float
    impact_score: float
    detected_from_events: List[str]
    related_files: List[str]
    context_data: Dict[str, Any]
    is_addressed: bool
    addressed_at: Optional[datetime]
    resolution_method: Optional[str]
    first_detected_at: datetime
    last_observed_at: datetime
    observation_count: int
    suggested_resources: List[Dict[str, Any]]
    learning_priority: float

    class Config:
        from_attributes = True


class ContextRecommendationCreate(BaseModel):
    """Schema for creating a context recommendation."""
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Recommendation description")
    recommendation_type: RecommendationType
    content: Dict[str, Any] = Field(default_factory=dict, description="Recommendation content")
    action_items: List[str] = Field(default_factory=list, description="Actionable items")
    resources: List[Dict[str, Any]] = Field(default_factory=list, description="Related resources")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    urgency_score: float = Field(..., ge=0.0, le=1.0, description="Urgency score")
    impact_score: float = Field(..., ge=0.0, le=1.0, description="Impact score")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Context data")
    triggering_events: List[str] = Field(default_factory=list, description="Triggering events")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    priority: NotificationPriority = Field(default=NotificationPriority.MEDIUM)


class ContextRecommendationResponse(BaseModel):
    """Schema for context recommendation response."""
    id: UUID
    user_id: UUID
    session_id: Optional[UUID]
    knowledge_gap_id: Optional[UUID]
    title: str
    description: str
    recommendation_type: RecommendationType
    content: Dict[str, Any]
    action_items: List[str]
    resources: List[Dict[str, Any]]
    relevance_score: float
    urgency_score: float
    impact_score: float
    context_data: Dict[str, Any]
    triggering_events: List[str]
    is_delivered: bool
    delivery_method: Optional[str]
    delivered_at: Optional[datetime]
    is_viewed: bool
    is_accepted: bool
    is_dismissed: bool
    user_feedback: Optional[str]
    expires_at: Optional[datetime]
    priority: NotificationPriority
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class RecommendationInteractionUpdate(BaseModel):
    """Schema for updating recommendation interaction."""
    is_viewed: Optional[bool] = Field(None, description="User has viewed the recommendation")
    is_accepted: Optional[bool] = Field(None, description="User has accepted the recommendation")
    is_dismissed: Optional[bool] = Field(None, description="User has dismissed the recommendation")
    user_feedback: Optional[str] = Field(None, description="User feedback on the recommendation")


class LearningOpportunityResponse(BaseModel):
    """Schema for learning opportunity response."""
    id: UUID
    user_id: UUID
    knowledge_gap_id: Optional[UUID]
    opportunity_title: str
    opportunity_description: str
    opportunity_category: str
    relevance_score: float
    difficulty_level: SkillLevel
    estimated_time_minutes: Optional[int]
    detected_from_context: Dict[str, Any]
    related_technologies: List[str]
    prerequisite_knowledge: List[str]
    optimal_timing: Optional[str]
    expires_at: Optional[datetime]
    learning_resources: List[Dict[str, Any]]
    practice_exercises: List[Dict[str, Any]]
    is_presented: bool
    is_accepted: bool
    is_completed: bool
    completion_date: Optional[datetime]
    priority_score: float
    is_intrusive: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ContextAnalysisResultResponse(BaseModel):
    """Schema for context analysis result response."""
    id: UUID
    user_id: UUID
    session_id: Optional[UUID]
    analysis_type: str
    analysis_trigger: str
    technologies_detected: List[Dict[str, Any]]
    knowledge_gaps_found: List[UUID]
    patterns_identified: List[Dict[str, Any]]
    recommendations_generated: List[UUID]
    analysis_duration_ms: int
    confidence_score: float
    complexity_score: float
    analysis_timestamp: datetime
    model_version: Optional[str]
    context_snapshot: Dict[str, Any]
    input_data_size: Optional[int]

    class Config:
        from_attributes = True


class WorkspacePatternResponse(BaseModel):
    """Schema for workspace pattern response."""
    id: UUID
    user_id: UUID
    pattern_name: str
    pattern_type: str
    pattern_description: str
    pattern_signature: str
    pattern_frequency: float
    pattern_confidence: float
    workspace_contexts: List[str]
    file_patterns: List[str]
    time_patterns: List[Dict[str, Any]]
    first_observed_at: datetime
    last_observed_at: datetime
    observation_count: int
    productivity_impact: Optional[float]
    learning_opportunities: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    is_active: bool
    is_beneficial: Optional[bool]

    class Config:
        from_attributes = True


class ContextInsights(BaseModel):
    """Schema for context insights."""
    current_focus: str = Field(..., description="Current work focus")
    technology_stack: List[str] = Field(..., description="Detected technologies")
    complexity_level: float = Field(..., ge=0.0, le=1.0, description="Work complexity level")
    potential_challenges: List[Dict[str, Any]] = Field(..., description="Potential challenges")
    recommended_resources: List[Dict[str, Any]] = Field(..., description="Recommended resources")
    knowledge_gaps: List[str] = Field(..., description="Identified knowledge gaps")
    learning_opportunities: List[str] = Field(..., description="Available learning opportunities")
    productivity_suggestions: List[str] = Field(..., description="Productivity improvement suggestions")


class RecommendationFilters(BaseModel):
    """Schema for filtering recommendations."""
    recommendation_type: Optional[RecommendationType] = Field(None, description="Filter by recommendation type")
    priority: Optional[NotificationPriority] = Field(None, description="Filter by priority")
    is_delivered: Optional[bool] = Field(None, description="Filter by delivery status")
    is_viewed: Optional[bool] = Field(None, description="Filter by view status")
    is_dismissed: Optional[bool] = Field(None, description="Filter by dismissal status")
    min_relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum relevance score")
    min_urgency_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum urgency score")
    created_after: Optional[datetime] = Field(None, description="Created after this date")
    created_before: Optional[datetime] = Field(None, description="Created before this date")


class WorkspaceMonitoringStatus(BaseModel):
    """Schema for workspace monitoring status."""
    session_id: UUID
    is_active: bool
    workspace_path: str
    started_at: datetime
    last_activity_at: datetime
    total_events: int
    active_recommendations: int
    detected_technologies: int
    identified_knowledge_gaps: int
    generated_learning_opportunities: int


class ProactiveRecommendationRequest(BaseModel):
    """Schema for requesting proactive recommendations."""
    workspace_context: Dict[str, Any] = Field(..., description="Current workspace context")
    user_activity_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="User activity patterns")
    focus_areas: List[str] = Field(default_factory=list, description="Areas of focus")
    exclude_categories: List[str] = Field(default_factory=list, description="Categories to exclude")
    max_recommendations: int = Field(default=5, ge=1, le=20, description="Maximum number of recommendations")
    urgency_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum urgency threshold")