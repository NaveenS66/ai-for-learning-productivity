"""
Pydantic schemas for workflow orchestration.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class WorkflowTypeSchema(str, Enum):
    """Types of workflows supported by the orchestrator."""
    LEARNING_JOURNEY = "learning_journey"
    DEBUGGING_SESSION = "debugging_session"
    AUTOMATION_EXECUTION = "automation_execution"


class WorkflowStatusSchema(str, Enum):
    """Status of workflow execution."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LearningGoalSchema(BaseModel):
    """Schema for learning goals in workflows."""
    id: Optional[str] = None
    title: str = Field(..., description="Title of the learning goal")
    description: str = Field(..., description="Detailed description of the goal")
    domain: str = Field(..., description="Subject domain (e.g., 'python', 'machine-learning')")
    difficulty_level: str = Field(..., description="Target difficulty level")
    estimated_duration: Optional[int] = Field(None, description="Estimated time to complete in hours")
    prerequisites: List[str] = Field(default_factory=list, description="Required prerequisite skills")
    success_criteria: List[str] = Field(default_factory=list, description="Criteria for goal completion")


class ErrorContextSchema(BaseModel):
    """Schema for error context in debugging workflows."""
    error_type: str = Field(..., description="Type of error encountered")
    error_message: str = Field(..., description="Error message text")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    file_path: Optional[str] = Field(None, description="File where error occurred")
    line_number: Optional[int] = Field(None, description="Line number of error")
    code_snippet: Optional[str] = Field(None, description="Relevant code snippet")
    environment_info: Dict[str, Any] = Field(default_factory=dict, description="Environment details")


class UserActionSchema(BaseModel):
    """Schema for user actions in automation workflows."""
    action_type: str = Field(..., description="Type of action performed")
    timestamp: datetime = Field(..., description="When the action was performed")
    context: Dict[str, Any] = Field(default_factory=dict, description="Action context")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    result: Optional[Dict[str, Any]] = Field(None, description="Action result if available")


class WorkflowStepSchema(BaseModel):
    """Schema for individual workflow steps."""
    step_id: str = Field(..., description="Unique identifier for the step")
    name: str = Field(..., description="Human-readable step name")
    description: str = Field(..., description="Step description")
    component: str = Field(..., description="Component responsible for execution")
    action: str = Field(..., description="Action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    dependencies: List[str] = Field(default_factory=list, description="Required predecessor steps")
    timeout: Optional[int] = Field(None, description="Step timeout in seconds")
    retry_count: int = Field(0, description="Current retry count")
    max_retries: int = Field(3, description="Maximum retry attempts")
    status: Optional[str] = Field(None, description="Current step status")
    started_at: Optional[datetime] = Field(None, description="Step start time")
    completed_at: Optional[datetime] = Field(None, description="Step completion time")
    error_info: Optional[Dict[str, Any]] = Field(None, description="Error information if step failed")


class WorkflowContextSchema(BaseModel):
    """Schema for workflow execution context."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    user_id: str = Field(..., description="User who initiated the workflow")
    workflow_type: WorkflowTypeSchema = Field(..., description="Type of workflow")
    status: WorkflowStatusSchema = Field(..., description="Current workflow status")
    created_at: datetime = Field(..., description="Workflow creation time")
    updated_at: datetime = Field(..., description="Last update time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Workflow metadata")
    steps_completed: List[str] = Field(default_factory=list, description="Completed step IDs")
    current_step: Optional[str] = Field(None, description="Currently executing step")
    error_info: Optional[Dict[str, Any]] = Field(None, description="Error information if workflow failed")
    progress_percentage: Optional[float] = Field(None, description="Completion percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class StartLearningJourneySchema(BaseModel):
    """Schema for starting a learning journey workflow."""
    learning_goals: List[LearningGoalSchema] = Field(..., description="Learning objectives")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    priority: Optional[str] = Field("normal", description="Workflow priority")
    deadline: Optional[datetime] = Field(None, description="Target completion date")
    
    @validator('learning_goals')
    def validate_learning_goals(cls, v):
        if not v:
            raise ValueError("At least one learning goal is required")
        return v


class StartDebuggingSessionSchema(BaseModel):
    """Schema for starting a debugging session workflow."""
    error_context: ErrorContextSchema = Field(..., description="Error information")
    code_context: Optional[Dict[str, Any]] = Field(None, description="Code context")
    priority: Optional[str] = Field("normal", description="Workflow priority")
    user_skill_level: Optional[str] = Field(None, description="User's skill level for guidance adaptation")
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['low', 'normal', 'high', 'urgent']:
            raise ValueError("Priority must be one of: low, normal, high, urgent")
        return v


class StartAutomationExecutionSchema(BaseModel):
    """Schema for starting an automation execution workflow."""
    user_actions: List[UserActionSchema] = Field(..., description="User actions to analyze")
    automation_preferences: Optional[Dict[str, Any]] = Field(None, description="Automation preferences")
    min_pattern_frequency: Optional[int] = Field(3, description="Minimum pattern frequency for automation")
    risk_tolerance: Optional[str] = Field("medium", description="Risk tolerance for automation")
    
    @validator('user_actions')
    def validate_user_actions(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 user actions are required for pattern detection")
        return v
    
    @validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError("Risk tolerance must be one of: low, medium, high")
        return v


class WorkflowProgressSchema(BaseModel):
    """Schema for workflow progress information."""
    workflow_id: str = Field(..., description="Workflow identifier")
    total_steps: int = Field(..., description="Total number of steps")
    completed_steps: int = Field(..., description="Number of completed steps")
    current_step: Optional[str] = Field(None, description="Currently executing step")
    progress_percentage: float = Field(..., description="Completion percentage")
    estimated_remaining_time: Optional[int] = Field(None, description="Estimated remaining time in seconds")
    last_activity: datetime = Field(..., description="Last activity timestamp")


class WorkflowResultSchema(BaseModel):
    """Schema for workflow execution results."""
    workflow_id: str = Field(..., description="Workflow identifier")
    status: WorkflowStatusSchema = Field(..., description="Final workflow status")
    results: Dict[str, Any] = Field(default_factory=dict, description="Workflow results")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Execution metrics")
    duration: float = Field(..., description="Total execution time in seconds")
    steps_executed: List[str] = Field(default_factory=list, description="Successfully executed steps")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Errors encountered")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")


class WorkflowListSchema(BaseModel):
    """Schema for workflow list responses."""
    workflows: List[WorkflowContextSchema] = Field(..., description="List of workflows")
    total_count: int = Field(..., description="Total number of workflows")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(50, description="Number of items per page")
    has_next: bool = Field(False, description="Whether there are more pages")


class WorkflowMetricsSchema(BaseModel):
    """Schema for workflow execution metrics."""
    workflow_type: WorkflowTypeSchema = Field(..., description="Type of workflow")
    total_executions: int = Field(..., description="Total number of executions")
    successful_executions: int = Field(..., description="Number of successful executions")
    failed_executions: int = Field(..., description="Number of failed executions")
    average_duration: float = Field(..., description="Average execution time in seconds")
    success_rate: float = Field(..., description="Success rate as percentage")
    most_common_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Common error patterns")
    performance_trends: Dict[str, Any] = Field(default_factory=dict, description="Performance trend data")


class WorkflowTemplateSchema(BaseModel):
    """Schema for workflow templates."""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    workflow_type: WorkflowTypeSchema = Field(..., description="Type of workflow")
    steps: List[WorkflowStepSchema] = Field(..., description="Template steps")
    default_parameters: Dict[str, Any] = Field(default_factory=dict, description="Default parameters")
    estimated_duration: Optional[int] = Field(None, description="Estimated execution time in seconds")
    complexity_level: Optional[str] = Field(None, description="Template complexity level")
    tags: List[str] = Field(default_factory=list, description="Template tags")


class WorkflowEventSchema(BaseModel):
    """Schema for workflow events."""
    event_id: str = Field(..., description="Event identifier")
    workflow_id: str = Field(..., description="Associated workflow ID")
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(..., description="Event timestamp")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    source: str = Field(..., description="Event source component")
    severity: Optional[str] = Field(None, description="Event severity level")


class WorkflowConfigurationSchema(BaseModel):
    """Schema for workflow configuration."""
    max_concurrent_workflows: int = Field(10, description="Maximum concurrent workflows per user")
    default_timeout: int = Field(3600, description="Default workflow timeout in seconds")
    retry_policy: Dict[str, Any] = Field(default_factory=dict, description="Retry policy configuration")
    notification_settings: Dict[str, Any] = Field(default_factory=dict, description="Notification preferences")
    resource_limits: Dict[str, Any] = Field(default_factory=dict, description="Resource usage limits")
    cleanup_policy: Dict[str, Any] = Field(default_factory=dict, description="Cleanup policy settings")


# Response schemas for API endpoints
class WorkflowCreatedResponse(BaseModel):
    """Response schema for workflow creation."""
    workflow_id: str = Field(..., description="Created workflow ID")
    message: str = Field(..., description="Success message")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class WorkflowActionResponse(BaseModel):
    """Response schema for workflow actions (pause, resume, cancel)."""
    workflow_id: str = Field(..., description="Workflow ID")
    action: str = Field(..., description="Action performed")
    message: str = Field(..., description="Result message")
    new_status: WorkflowStatusSchema = Field(..., description="New workflow status")


class WorkflowCleanupResponse(BaseModel):
    """Response schema for workflow cleanup operations."""
    cleaned_workflows: int = Field(..., description="Number of workflows cleaned up")
    message: str = Field(..., description="Cleanup result message")
    cleanup_criteria: Dict[str, Any] = Field(default_factory=dict, description="Cleanup criteria used")


# Error schemas
class WorkflowErrorSchema(BaseModel):
    """Schema for workflow-specific errors."""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    step_id: Optional[str] = Field(None, description="Associated step ID")
    timestamp: datetime = Field(..., description="Error timestamp")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested remediation actions")