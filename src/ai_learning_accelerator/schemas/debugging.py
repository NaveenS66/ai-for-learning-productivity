"""Pydantic schemas for debugging functionality."""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

from ..models.debugging import ErrorType, ComplexityLevel, SolutionStatus, DebuggingSessionStatus
from ..models.user import SkillLevel


# Request schemas

class CodeContextCreate(BaseModel):
    """Schema for creating code context."""
    workspace_root_path: str = Field(..., description="Root path of the workspace")
    project_type: str = Field(..., description="Type of project")
    primary_language: str = Field(..., description="Primary programming language")
    framework: Optional[str] = Field(None, description="Framework being used")
    dependencies: List[Dict[str, Any]] = Field(default_factory=list, description="Project dependencies")
    
    current_file_path: Optional[str] = Field(None, description="Path to current file")
    current_file_content: Optional[str] = Field(None, description="Content of current file")
    cursor_position: Optional[Dict[str, int]] = Field(None, description="Cursor position")
    selection_range: Optional[Dict[str, Any]] = Field(None, description="Selected text range")
    
    project_structure: Dict[str, Any] = Field(default_factory=dict, description="Project structure")
    git_branch: Optional[str] = Field(None, description="Current git branch")
    recent_commits: List[Dict[str, Any]] = Field(default_factory=list, description="Recent commits")
    changed_files: List[str] = Field(default_factory=list, description="Recently changed files")
    
    ide_name: Optional[str] = Field(None, description="IDE being used")
    ide_plugins: List[str] = Field(default_factory=list, description="IDE plugins")
    environment_settings: Dict[str, Any] = Field(default_factory=dict, description="Environment settings")


class ErrorContextCreate(BaseModel):
    """Schema for creating error context."""
    error_type: ErrorType = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    
    file_path: Optional[str] = Field(None, description="File where error occurred")
    line_number: Optional[int] = Field(None, description="Line number of error")
    column_number: Optional[int] = Field(None, description="Column number of error")
    
    surrounding_code: Optional[str] = Field(None, description="Code surrounding the error")
    error_context_data: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")


class DebugAnalysisRequest(BaseModel):
    """Schema for requesting debug analysis."""
    error_context: ErrorContextCreate = Field(..., description="Error context information")
    code_context: Optional[CodeContextCreate] = Field(None, description="Code context information")
    user_skill_level: Optional[SkillLevel] = Field(None, description="User's skill level")
    include_similar_issues: bool = Field(True, description="Include similar historical issues")
    max_solutions: int = Field(5, description="Maximum number of solutions to generate")


class SolutionFeedback(BaseModel):
    """Schema for solution feedback."""
    solution_id: UUID = Field(..., description="ID of the solution")
    status: SolutionStatus = Field(..., description="Solution status")
    effectiveness_rating: Optional[int] = Field(None, ge=1, le=5, description="Effectiveness rating (1-5)")
    user_feedback: Optional[str] = Field(None, description="User feedback text")
    time_spent_minutes: Optional[int] = Field(None, description="Time spent on this solution")


class DebuggingSessionCreate(BaseModel):
    """Schema for creating debugging session."""
    error_context_id: UUID = Field(..., description="ID of the error context")
    session_name: Optional[str] = Field(None, description="Optional session name")
    user_skill_level_at_start: Optional[SkillLevel] = Field(None, description="User's skill level at start")


class GuidanceStepCreate(BaseModel):
    """Schema for creating guidance step."""
    step_type: str = Field(..., description="Type of guidance step")
    title: str = Field(..., description="Step title")
    description: str = Field(..., description="Step description")
    instructions: List[str] = Field(default_factory=list, description="Detailed instructions")
    expected_outcome: Optional[str] = Field(None, description="Expected outcome")
    troubleshooting_tips: List[str] = Field(default_factory=list, description="Troubleshooting tips")


class PotentialIssueQuery(BaseModel):
    """Schema for querying potential issues."""
    code_context: CodeContextCreate = Field(..., description="Code context to analyze")
    include_low_probability: bool = Field(False, description="Include low probability issues")
    min_severity: float = Field(0.3, ge=0.0, le=1.0, description="Minimum severity threshold")


# Response schemas

class CodeContextResponse(BaseModel):
    """Schema for code context response."""
    id: UUID
    user_id: UUID
    workspace_root_path: str
    project_type: str
    primary_language: str
    framework: Optional[str]
    dependencies: List[Dict[str, Any]]
    
    current_file_path: Optional[str]
    cursor_position: Optional[Dict[str, int]]
    selection_range: Optional[Dict[str, Any]]
    
    git_branch: Optional[str]
    ide_name: Optional[str]
    ide_plugins: List[str]
    
    context_hash: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ErrorContextResponse(BaseModel):
    """Schema for error context response."""
    id: UUID
    user_id: UUID
    code_context_id: Optional[UUID]
    
    error_type: ErrorType
    error_message: str
    stack_trace: Optional[str]
    
    file_path: Optional[str]
    line_number: Optional[int]
    column_number: Optional[int]
    
    surrounding_code: Optional[str]
    error_context_data: Dict[str, Any]
    
    error_hash: Optional[str]
    frequency_count: int
    first_occurrence: datetime
    last_occurrence: datetime
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DebuggingSolutionResponse(BaseModel):
    """Schema for debugging solution response."""
    id: UUID
    user_id: UUID
    error_analysis_id: UUID
    
    title: str
    description: str
    solution_type: str
    
    steps: List[Dict[str, Any]]
    code_changes: List[Dict[str, Any]]
    
    likelihood_score: float
    difficulty_score: float
    risk_score: float
    overall_rank: int
    
    estimated_time_minutes: Optional[int]
    prerequisites: List[str]
    side_effects: List[str]
    
    status: SolutionStatus
    user_feedback: Optional[str]
    effectiveness_rating: Optional[int]
    
    success_count: int
    failure_count: int
    usage_count: int
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ErrorAnalysisResponse(BaseModel):
    """Schema for error analysis response."""
    id: UUID
    user_id: UUID
    error_context_id: UUID
    code_context_id: Optional[UUID]
    
    root_cause: str
    affected_components: List[str]
    complexity_level: ComplexityLevel
    
    confidence_score: float
    analysis_model_version: Optional[str]
    analysis_duration_ms: Optional[int]
    
    similar_issues: List[Dict[str, Any]]
    pattern_matches: List[Dict[str, Any]]
    
    insights: Dict[str, Any]
    suggested_investigation_steps: List[str]
    
    solutions: List[DebuggingSolutionResponse]
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DebuggingSessionResponse(BaseModel):
    """Schema for debugging session response."""
    id: UUID
    user_id: UUID
    error_context_id: UUID
    
    session_name: Optional[str]
    status: DebuggingSessionStatus
    
    started_at: datetime
    ended_at: Optional[datetime]
    total_duration_minutes: Optional[int]
    
    resolution_summary: Optional[str]
    successful_solution_id: Optional[UUID]
    lessons_learned: List[str]
    
    user_skill_level_at_start: Optional[SkillLevel]
    assistance_level_used: Optional[str]
    user_satisfaction_rating: Optional[int]
    
    actions_taken: List[Dict[str, Any]]
    time_spent_per_solution: Dict[str, int]
    context_switches: int
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class GuidanceStepResponse(BaseModel):
    """Schema for guidance step response."""
    id: UUID
    session_id: UUID
    user_id: UUID
    
    step_number: int
    step_type: str
    title: str
    description: str
    
    instructions: List[str]
    expected_outcome: Optional[str]
    troubleshooting_tips: List[str]
    
    is_completed: bool
    completion_time: Optional[datetime]
    user_notes: Optional[str]
    
    was_helpful: Optional[bool]
    difficulty_rating: Optional[int]
    time_spent_minutes: Optional[int]
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PotentialIssueResponse(BaseModel):
    """Schema for potential issue response."""
    id: UUID
    user_id: UUID
    code_context_id: UUID
    
    issue_type: ErrorType
    issue_description: str
    likelihood_score: float
    severity_score: float
    
    affected_files: List[str]
    code_locations: List[Dict[str, Any]]
    
    prevention_steps: List[str]
    monitoring_suggestions: List[str]
    
    prediction_confidence: float
    prediction_model_version: Optional[str]
    is_dismissed: bool
    
    became_actual_issue: bool
    actual_issue_date: Optional[datetime]
    prevention_actions_taken: List[str]
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ErrorPatternResponse(BaseModel):
    """Schema for error pattern response."""
    id: UUID
    pattern_name: str
    pattern_type: ErrorType
    pattern_signature: str
    
    common_causes: List[str]
    typical_contexts: List[str]
    complexity_level: ComplexityLevel
    
    recommended_solutions: List[Dict[str, Any]]
    prevention_tips: List[str]
    
    occurrence_count: int
    success_rate: float
    average_resolution_time: Optional[int]
    
    is_active: bool
    confidence_threshold: float
    last_updated: datetime
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Composite response schemas

class DebugAnalysisResult(BaseModel):
    """Complete debug analysis result."""
    error_context: ErrorContextResponse
    code_context: Optional[CodeContextResponse]
    analysis: ErrorAnalysisResponse
    
    class Config:
        from_attributes = True


class DebuggingSessionSummary(BaseModel):
    """Summary of debugging session."""
    session: DebuggingSessionResponse
    error_context: ErrorContextResponse
    guidance_steps: List[GuidanceStepResponse]
    successful_solution: Optional[DebuggingSolutionResponse]
    
    class Config:
        from_attributes = True


# Search and filter schemas

class ErrorSearchRequest(BaseModel):
    """Schema for searching errors."""
    error_types: Optional[List[ErrorType]] = Field(None, description="Filter by error types")
    complexity_levels: Optional[List[ComplexityLevel]] = Field(None, description="Filter by complexity")
    languages: Optional[List[str]] = Field(None, description="Filter by programming languages")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    resolved_only: Optional[bool] = Field(None, description="Only resolved issues")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Results offset")


class SolutionSearchRequest(BaseModel):
    """Schema for searching solutions."""
    solution_types: Optional[List[str]] = Field(None, description="Filter by solution types")
    min_likelihood: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum likelihood score")
    max_difficulty: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum difficulty score")
    successful_only: Optional[bool] = Field(None, description="Only successful solutions")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Results offset")


# Analytics schemas

class DebuggingAnalytics(BaseModel):
    """Debugging analytics summary."""
    user_id: UUID
    time_period: str
    
    total_errors: int
    resolved_errors: int
    resolution_rate: float
    
    average_resolution_time_minutes: float
    most_common_error_types: List[Dict[str, Any]]
    most_effective_solutions: List[Dict[str, Any]]
    
    skill_improvement_indicators: Dict[str, Any]
    productivity_metrics: Dict[str, Any]
    
    class Config:
        from_attributes = True