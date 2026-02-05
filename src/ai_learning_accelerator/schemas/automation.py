"""Pydantic schemas for automation engine API."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field

from ..models.automation import (
    ActionType, PatternType, AutomationComplexity, AutomationStatus,
    ExecutionStatus
)


class UserActionCreate(BaseModel):
    """Schema for creating a user action."""
    action_type: ActionType
    action_name: str = Field(..., description="Name or description of the action")
    action_description: Optional[str] = Field(None, description="Detailed description")
    workspace_path: Optional[str] = Field(None, description="Workspace path where action occurred")
    file_path: Optional[str] = Field(None, description="File path related to action")
    command: Optional[str] = Field(None, description="Command executed")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    tool_used: Optional[str] = Field(None, description="Tool or IDE used")
    duration_ms: Optional[int] = Field(None, description="Action duration in milliseconds")
    success: bool = Field(default=True, description="Action completed successfully")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class UserActionResponse(BaseModel):
    """Schema for user action response."""
    id: UUID
    user_id: UUID
    session_id: Optional[UUID]
    action_type: ActionType
    action_name: str
    action_description: Optional[str]
    workspace_path: Optional[str]
    file_path: Optional[str]
    command: Optional[str]
    parameters: Dict[str, Any]
    action_timestamp: datetime
    duration_ms: Optional[int]
    success: bool
    error_message: Optional[str]
    environment_data: Dict[str, Any]
    tool_used: Optional[str]
    sequence_number: Optional[int]
    pattern_signature: Optional[str]
    is_repetitive: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ActionPatternResponse(BaseModel):
    """Schema for action pattern response."""
    id: UUID
    user_id: UUID
    pattern_name: str
    pattern_type: PatternType
    pattern_signature: str
    action_sequence: List[str]
    frequency: int
    confidence_score: float
    workspace_contexts: List[str]
    file_patterns: List[str]
    time_patterns: List[Dict[str, Any]]
    first_observed_at: datetime
    last_observed_at: datetime
    average_duration_ms: Optional[int]
    automation_score: float
    complexity: Optional[AutomationComplexity]
    time_saving_potential: Optional[int]
    is_active: bool
    is_beneficial: Optional[bool]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AutomationOpportunityResponse(BaseModel):
    """Schema for automation opportunity response."""
    id: UUID
    user_id: UUID
    pattern_id: UUID
    title: str
    description: str
    category: str
    automation_score: float
    complexity: AutomationComplexity
    time_saving_potential: int
    frequency_per_week: float
    suggested_approach: Optional[str]
    required_tools: List[str]
    prerequisites: List[str]
    risk_level: str
    potential_issues: List[str]
    rollback_strategy: Optional[str]
    is_presented: bool
    user_interest_level: Optional[int]
    user_feedback: Optional[str]
    status: AutomationStatus
    priority_score: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AutomationOpportunityUpdate(BaseModel):
    """Schema for updating automation opportunity."""
    user_interest_level: Optional[int] = Field(None, ge=1, le=5, description="User interest level (1-5)")
    user_feedback: Optional[str] = Field(None, description="User feedback")
    status: Optional[AutomationStatus] = Field(None, description="Update status")


class AutomationScriptCreate(BaseModel):
    """Schema for creating automation script."""
    script_name: str = Field(..., description="Script name")
    script_description: Optional[str] = Field(None, description="Script description")
    script_type: str = Field(..., description="Type of script (bash, python, etc.)")
    script_content: str = Field(..., description="Actual script content")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Script configuration")
    environment_requirements: List[str] = Field(default_factory=list, description="Environment requirements")
    requires_confirmation: bool = Field(default=True, description="Requires user confirmation")


class AutomationScriptResponse(BaseModel):
    """Schema for automation script response."""
    id: UUID
    user_id: UUID
    opportunity_id: UUID
    script_name: str
    script_description: Optional[str]
    script_type: str
    script_content: str
    configuration: Dict[str, Any]
    environment_requirements: List[str]
    version: str
    created_by: str
    is_enabled: bool
    auto_execute: bool
    requires_confirmation: bool
    execution_count: int
    success_count: int
    total_time_saved_minutes: int
    status: AutomationStatus
    last_executed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AutomationExecutionCreate(BaseModel):
    """Schema for creating automation execution."""
    execution_trigger: str = Field(..., description="What triggered the execution")
    execution_context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")


class AutomationExecutionResponse(BaseModel):
    """Schema for automation execution response."""
    id: UUID
    script_id: UUID
    user_id: UUID
    execution_trigger: str
    execution_context: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    status: ExecutionStatus
    exit_code: Optional[int]
    output: Optional[str]
    error_output: Optional[str]
    cpu_usage_percent: Optional[float]
    memory_usage_mb: Optional[float]
    files_processed: Optional[int]
    user_satisfaction: Optional[int]
    time_saved_minutes: Optional[int]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AutomationMetricsResponse(BaseModel):
    """Schema for automation metrics response."""
    id: UUID
    user_id: UUID
    period_start: datetime
    period_end: datetime
    period_type: str
    patterns_detected: int
    opportunities_identified: int
    scripts_generated: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    total_time_saved_minutes: int
    average_time_saved_per_execution: float
    user_approval_rate: float
    average_satisfaction_rating: float
    pattern_detection_accuracy: float
    automation_success_rate: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserPreferenceUpdate(BaseModel):
    """Schema for updating user automation preferences."""
    auto_detect_patterns: Optional[bool] = Field(None, description="Automatically detect patterns")
    min_pattern_frequency: Optional[int] = Field(None, ge=1, le=20, description="Minimum frequency for pattern detection")
    min_automation_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum score for automation suggestions")
    notify_new_opportunities: Optional[bool] = Field(None, description="Notify about new opportunities")
    notification_frequency: Optional[str] = Field(None, description="Notification frequency")
    require_confirmation: Optional[bool] = Field(None, description="Require confirmation before execution")
    auto_execute_simple: Optional[bool] = Field(None, description="Auto-execute simple automations")
    max_execution_time_minutes: Optional[int] = Field(None, ge=1, le=120, description="Maximum execution time allowed")
    risk_tolerance: Optional[str] = Field(None, description="Risk tolerance (low, medium, high)")
    allow_file_modifications: Optional[bool] = Field(None, description="Allow automations to modify files")
    allow_system_commands: Optional[bool] = Field(None, description="Allow system command execution")
    learn_from_actions: Optional[bool] = Field(None, description="Learn patterns from user actions")
    share_anonymous_patterns: Optional[bool] = Field(None, description="Share anonymous pattern data")


class UserPreferenceResponse(BaseModel):
    """Schema for user automation preferences response."""
    id: UUID
    user_id: UUID
    auto_detect_patterns: bool
    min_pattern_frequency: int
    min_automation_score: float
    notify_new_opportunities: bool
    notification_frequency: str
    require_confirmation: bool
    auto_execute_simple: bool
    max_execution_time_minutes: int
    risk_tolerance: str
    allow_file_modifications: bool
    allow_system_commands: bool
    learn_from_actions: bool
    share_anonymous_patterns: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PatternFeedback(BaseModel):
    """Schema for pattern feedback."""
    is_beneficial: bool = Field(..., description="Whether the pattern is beneficial")
    user_feedback: Optional[str] = Field(None, description="Additional user feedback")


class AutomationSummary(BaseModel):
    """Schema for automation summary."""
    total_patterns: int = Field(..., description="Total patterns detected")
    active_patterns: int = Field(..., description="Active patterns")
    automation_opportunities: int = Field(..., description="Available automation opportunities")
    implemented_automations: int = Field(..., description="Implemented automations")
    total_time_saved_minutes: int = Field(..., description="Total time saved")
    success_rate: float = Field(..., description="Automation success rate")
    top_opportunities: List[AutomationOpportunityResponse] = Field(..., description="Top automation opportunities")


class PatternDetectionRequest(BaseModel):
    """Schema for requesting pattern detection."""
    force_detection: bool = Field(default=False, description="Force pattern detection even if below thresholds")
    time_window_hours: Optional[int] = Field(None, ge=1, le=168, description="Time window for analysis in hours")
    min_frequency: Optional[int] = Field(None, ge=1, le=50, description="Minimum frequency threshold")


class AutomationInsights(BaseModel):
    """Schema for automation insights."""
    productivity_gain: float = Field(..., description="Productivity gain percentage")
    most_automated_category: str = Field(..., description="Most automated category")
    time_saved_this_week: int = Field(..., description="Time saved this week in minutes")
    recommended_next_automation: Optional[str] = Field(None, description="Next recommended automation")
    pattern_trends: List[Dict[str, Any]] = Field(..., description="Pattern detection trends")
    automation_adoption_rate: float = Field(..., description="Rate of automation adoption")


class BulkActionImport(BaseModel):
    """Schema for bulk importing user actions."""
    actions: List[UserActionCreate] = Field(..., description="List of actions to import")
    session_id: Optional[UUID] = Field(None, description="Optional session ID for all actions")
    workspace_path: Optional[str] = Field(None, description="Default workspace path")


class AutomationHealthCheck(BaseModel):
    """Schema for automation system health check."""
    pattern_detection_active: bool = Field(..., description="Pattern detection is active")
    automation_engine_status: str = Field(..., description="Automation engine status")
    pending_opportunities: int = Field(..., description="Number of pending opportunities")
    active_scripts: int = Field(..., description="Number of active scripts")
    recent_executions: int = Field(..., description="Recent executions count")
    system_load: float = Field(..., description="System load percentage")
    last_pattern_detection: Optional[datetime] = Field(None, description="Last pattern detection run")


class WorkflowGenerationRequest(BaseModel):
    """Schema for requesting workflow generation from an opportunity."""
    script_type: Optional[str] = Field(None, description="Preferred script type (bash, python, powershell)")
    custom_parameters: Optional[Dict[str, Any]] = Field(None, description="Custom parameters for script generation")


class WorkflowExecutionRequest(BaseModel):
    """Schema for requesting workflow execution."""
    execution_context: Optional[Dict[str, Any]] = Field(None, description="Execution context and parameters")
    confirm_execution: bool = Field(default=True, description="Confirm execution before running")


class UserPreferenceCreate(BaseModel):
    """Schema for creating user automation preferences."""
    auto_detect_patterns: bool = Field(default=True, description="Automatically detect patterns")
    min_pattern_frequency: int = Field(default=3, ge=1, le=20, description="Minimum frequency for pattern detection")
    min_automation_score: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum score for automation suggestions")
    notify_new_opportunities: bool = Field(default=True, description="Notify about new opportunities")
    notification_frequency: str = Field(default="daily", description="Notification frequency")
    require_confirmation: bool = Field(default=True, description="Require confirmation before execution")
    auto_execute_simple: bool = Field(default=False, description="Auto-execute simple automations")
    max_execution_time_minutes: int = Field(default=30, ge=1, le=120, description="Maximum execution time allowed")
    risk_tolerance: str = Field(default="medium", description="Risk tolerance (low, medium, high)")
    allow_file_modifications: bool = Field(default=False, description="Allow automations to modify files")
    allow_system_commands: bool = Field(default=False, description="Allow system command execution")
    learn_from_actions: bool = Field(default=True, description="Learn patterns from user actions")
    share_anonymous_patterns: bool = Field(default=False, description="Share anonymous pattern data")