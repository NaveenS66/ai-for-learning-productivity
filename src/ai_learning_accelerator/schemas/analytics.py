"""Analytics schemas for API operations."""

from datetime import datetime, date
from typing import List, Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator

from ..models.analytics import MetricType, VisualizationType, MilestoneType, PatternType


class LearningMetricBase(BaseModel):
    """Base schema for learning metrics."""
    metric_type: MetricType
    metric_name: str = Field(..., max_length=100)
    metric_value: float
    metric_unit: Optional[str] = Field(None, max_length=50)
    measurement_date: Optional[date] = None
    context_data: Dict[str, Any] = Field(default_factory=dict)
    baseline_value: Optional[float] = None
    previous_value: Optional[float] = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    data_quality: float = Field(default=1.0, ge=0.0, le=1.0)


class LearningMetricCreate(LearningMetricBase):
    """Schema for creating learning metrics."""
    session_id: Optional[UUID] = None
    content_id: Optional[UUID] = None


class LearningMetricUpdate(BaseModel):
    """Schema for updating learning metrics."""
    metric_value: Optional[float] = None
    metric_unit: Optional[str] = Field(None, max_length=50)
    context_data: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    data_quality: Optional[float] = Field(None, ge=0.0, le=1.0)


class LearningMetricResponse(LearningMetricBase):
    """Schema for learning metric responses."""
    id: UUID
    user_id: UUID
    session_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    measurement_timestamp: datetime
    improvement_percentage: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProgressVisualizationBase(BaseModel):
    """Base schema for progress visualizations."""
    visualization_name: str = Field(..., max_length=200)
    visualization_type: VisualizationType
    config_data: Dict[str, Any]
    data_sources: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    time_range: Dict[str, Any] = Field(default_factory=dict)
    chart_properties: Dict[str, Any] = Field(default_factory=dict)
    styling: Dict[str, Any] = Field(default_factory=dict)
    interactive_features: List[str] = Field(default_factory=list)
    refresh_frequency: int = Field(default=3600, gt=0)


class ProgressVisualizationCreate(ProgressVisualizationBase):
    """Schema for creating progress visualizations."""
    pass


class ProgressVisualizationUpdate(BaseModel):
    """Schema for updating progress visualizations."""
    visualization_name: Optional[str] = Field(None, max_length=200)
    config_data: Optional[Dict[str, Any]] = None
    data_sources: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    time_range: Optional[Dict[str, Any]] = None
    chart_properties: Optional[Dict[str, Any]] = None
    styling: Optional[Dict[str, Any]] = None
    interactive_features: Optional[List[str]] = None
    refresh_frequency: Optional[int] = Field(None, gt=0)
    is_favorite: Optional[bool] = None
    is_shared: Optional[bool] = None


class ProgressVisualizationResponse(ProgressVisualizationBase):
    """Schema for progress visualization responses."""
    id: UUID
    user_id: UUID
    generated_data: Dict[str, Any]
    last_updated: datetime
    view_count: int
    is_favorite: bool
    is_shared: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LearningMilestoneBase(BaseModel):
    """Base schema for learning milestones."""
    milestone_name: str = Field(..., max_length=200)
    milestone_type: MilestoneType
    description: Optional[str] = None
    criteria: Dict[str, Any]
    target_value: Optional[float] = None
    difficulty_level: float = Field(default=1.0, ge=1.0, le=5.0)
    estimated_time: Optional[int] = Field(None, gt=0)
    category: Optional[str] = Field(None, max_length=100)


class LearningMilestoneCreate(LearningMilestoneBase):
    """Schema for creating learning milestones."""
    pass


class LearningMilestoneUpdate(BaseModel):
    """Schema for updating learning milestones."""
    milestone_name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    criteria: Optional[Dict[str, Any]] = None
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    difficulty_level: Optional[float] = Field(None, ge=1.0, le=5.0)
    estimated_time: Optional[int] = Field(None, gt=0)
    category: Optional[str] = Field(None, max_length=100)
    reward_data: Optional[Dict[str, Any]] = None
    celebration_data: Optional[Dict[str, Any]] = None
    next_challenges: Optional[List[str]] = None


class LearningMilestoneResponse(LearningMilestoneBase):
    """Schema for learning milestone responses."""
    id: UUID
    user_id: UUID
    current_value: float
    is_achieved: bool
    achievement_date: Optional[datetime] = None
    progress_percentage: float
    recognition_given: bool
    reward_data: Dict[str, Any]
    celebration_data: Dict[str, Any]
    next_challenges: List[str]
    related_milestones: List[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LearningPatternBase(BaseModel):
    """Base schema for learning patterns."""
    pattern_name: str = Field(..., max_length=200)
    pattern_type: PatternType
    description: str
    pattern_data: Dict[str, Any]
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class LearningPatternCreate(LearningPatternBase):
    """Schema for creating learning patterns."""
    analysis_period: Dict[str, Any] = Field(default_factory=dict)
    data_points: int = Field(default=0, ge=0)


class LearningPatternUpdate(BaseModel):
    """Schema for updating learning patterns."""
    pattern_name: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None
    pattern_data: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency: Optional[float] = None
    strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    stability: Optional[float] = Field(None, ge=0.0, le=1.0)
    insights: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    validation_status: Optional[str] = None
    user_feedback: Optional[Dict[str, Any]] = None


class LearningPatternResponse(LearningPatternBase):
    """Schema for learning pattern responses."""
    id: UUID
    user_id: UUID
    discovery_date: datetime
    analysis_period: Dict[str, Any]
    data_points: int
    frequency: Optional[float] = None
    strength: float
    stability: float
    insights: List[str]
    recommendations: List[str]
    predicted_outcomes: Dict[str, Any]
    impact_score: float
    validation_status: str
    user_feedback: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class OptimizationSuggestionBase(BaseModel):
    """Base schema for optimization suggestions."""
    suggestion_title: str = Field(..., max_length=200)
    suggestion_type: str = Field(..., max_length=100)
    description: str
    rationale: Optional[str] = None
    expected_benefit: Optional[str] = None
    action_steps: List[str] = Field(default_factory=list)
    difficulty_level: float = Field(default=1.0, ge=1.0, le=5.0)
    estimated_effort: Optional[int] = Field(None, gt=0)


class OptimizationSuggestionCreate(OptimizationSuggestionBase):
    """Schema for creating optimization suggestions."""
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_pattern_id: Optional[UUID] = None
    context_data: Dict[str, Any] = Field(default_factory=dict)


class OptimizationSuggestionUpdate(BaseModel):
    """Schema for updating optimization suggestions."""
    status: Optional[str] = None
    user_response: Optional[str] = None
    implementation_date: Optional[datetime] = None
    effectiveness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=1.0)
    follow_up_needed: Optional[bool] = None


class OptimizationSuggestionResponse(OptimizationSuggestionBase):
    """Schema for optimization suggestion responses."""
    id: UUID
    user_id: UUID
    priority_score: float
    impact_score: float
    urgency_score: float
    status: str
    user_response: Optional[str] = None
    implementation_date: Optional[datetime] = None
    effectiveness_score: Optional[float] = None
    user_satisfaction: Optional[float] = None
    follow_up_needed: bool
    source_pattern_id: Optional[UUID] = None
    context_data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AnalyticsReportBase(BaseModel):
    """Base schema for analytics reports."""
    report_name: str = Field(..., max_length=200)
    report_type: str = Field(..., max_length=100)
    time_period: Dict[str, Any]
    metrics_included: List[str] = Field(default_factory=list)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsReportCreate(AnalyticsReportBase):
    """Schema for creating analytics reports."""
    pass


class AnalyticsReportUpdate(BaseModel):
    """Schema for updating analytics reports."""
    is_shared: Optional[bool] = None
    shared_with: Optional[List[str]] = None


class AnalyticsReportResponse(AnalyticsReportBase):
    """Schema for analytics report responses."""
    id: UUID
    user_id: UUID
    summary_data: Dict[str, Any]
    detailed_data: Dict[str, Any]
    visualizations: List[str]
    insights: List[str]
    recommendations: List[str]
    generation_date: datetime
    data_freshness: Optional[datetime] = None
    generation_time: Optional[float] = None
    view_count: int
    download_count: int
    is_shared: bool
    shared_with: List[str]
    data_quality_score: float
    completeness_score: float
    accuracy_validated: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CompetencyAssessmentBase(BaseModel):
    """Base schema for competency assessments."""
    skill_area: str = Field(..., max_length=100)
    competency_level: float = Field(..., ge=0.0, le=5.0)
    assessment_method: str = Field(..., max_length=100)
    evidence_data: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    validation_source: Optional[str] = Field(None, max_length=100)
    target_level: Optional[float] = Field(None, ge=0.0, le=5.0)
    target_date: Optional[date] = None
    learning_plan: Dict[str, Any] = Field(default_factory=dict)


class CompetencyAssessmentCreate(CompetencyAssessmentBase):
    """Schema for creating competency assessments."""
    domain_context: Dict[str, Any] = Field(default_factory=dict)
    assessment_context: Dict[str, Any] = Field(default_factory=dict)


class CompetencyAssessmentUpdate(BaseModel):
    """Schema for updating competency assessments."""
    competency_level: Optional[float] = Field(None, ge=0.0, le=5.0)
    evidence_data: Optional[List[Dict[str, Any]]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    validation_source: Optional[str] = Field(None, max_length=100)
    peer_validation: Optional[bool] = None
    target_level: Optional[float] = Field(None, ge=0.0, le=5.0)
    target_date: Optional[date] = None
    learning_plan: Optional[Dict[str, Any]] = None


class CompetencyAssessmentResponse(CompetencyAssessmentBase):
    """Schema for competency assessment responses."""
    id: UUID
    user_id: UUID
    assessment_date: datetime
    previous_level: Optional[float] = None
    improvement_rate: Optional[float] = None
    time_to_improve: Optional[int] = None
    peer_validation: bool
    domain_context: Dict[str, Any]
    assessment_context: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AnalyticsDashboardResponse(BaseModel):
    """Schema for analytics dashboard data."""
    user_id: UUID
    recent_metrics: List[LearningMetricResponse]
    active_milestones: List[LearningMilestoneResponse]
    key_patterns: List[LearningPatternResponse]
    top_suggestions: List[OptimizationSuggestionResponse]
    competency_overview: List[CompetencyAssessmentResponse]
    progress_summary: Dict[str, Any]
    generated_at: datetime

    class Config:
        from_attributes = True