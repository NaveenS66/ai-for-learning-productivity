"""
Workflow API endpoints for orchestrating end-to-end user workflows.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.workflow_orchestrator import WorkflowOrchestrator, WorkflowType, WorkflowStatus
from ..models.learning import LearningGoal
from ..models.debugging import ErrorContext
from ..auth.dependencies import get_current_user, require_permissions
from ..models.user import User
from ..utils.exceptions import WorkflowError, ComponentError

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


class StartLearningJourneyRequest(BaseModel):
    """Request to start a learning journey workflow."""
    learning_goals: List[Dict[str, Any]] = Field(..., description="Learning objectives for the journey")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for the journey")


class StartDebuggingSessionRequest(BaseModel):
    """Request to start a debugging session workflow."""
    error_context: Dict[str, Any] = Field(..., description="Information about the error encountered")
    code_context: Optional[Dict[str, Any]] = Field(None, description="Code context information")


class StartAutomationExecutionRequest(BaseModel):
    """Request to start an automation execution workflow."""
    user_actions: List[Dict[str, Any]] = Field(..., description="User actions to analyze for automation")
    automation_preferences: Optional[Dict[str, Any]] = Field(None, description="Automation preferences")


class WorkflowStatusResponse(BaseModel):
    """Response containing workflow status information."""
    workflow_id: str
    user_id: str
    workflow_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    steps_completed: List[str]
    current_step: Optional[str]
    error_info: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class WorkflowListResponse(BaseModel):
    """Response containing list of workflows."""
    workflows: List[WorkflowStatusResponse]
    total_count: int


@router.post("/learning-journey", response_model=Dict[str, str])
async def start_learning_journey(
    request: StartLearningJourneyRequest,
    current_user: User = Depends(get_current_user),
    orchestrator: WorkflowOrchestrator = Depends(),
    _: None = Depends(require_permissions(["workflow:create"]))
):
    """
    Start a complete learning journey workflow.
    
    This endpoint initiates an end-to-end learning experience that includes:
    - Skill assessment
    - Learning goal analysis
    - Personalized learning path generation
    - Content adaptation
    - Progress monitoring setup
    - Content delivery
    """
    try:
        # Convert request data to LearningGoal objects
        learning_goals = [
            LearningGoal(**goal_data) for goal_data in request.learning_goals
        ]
        
        workflow_id = await orchestrator.start_learning_journey(
            user_id=current_user.id,
            learning_goals=learning_goals,
            preferences=request.preferences
        )
        
        return {
            "workflow_id": workflow_id,
            "message": "Learning journey workflow started successfully"
        }
        
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start learning journey: {str(e)}")


@router.post("/debugging-session", response_model=Dict[str, str])
async def start_debugging_session(
    request: StartDebuggingSessionRequest,
    current_user: User = Depends(get_current_user),
    orchestrator: WorkflowOrchestrator = Depends(),
    _: None = Depends(require_permissions(["workflow:create"]))
):
    """
    Start a debugging session workflow.
    
    This endpoint initiates an end-to-end debugging experience that includes:
    - Code context analysis
    - Error analysis and root cause identification
    - Solution generation and ranking
    - Debugging guidance adaptation
    - Step-by-step resolution assistance
    - Learning from successful sessions
    """
    try:
        # Convert request data to ErrorContext object
        error_context = ErrorContext(**request.error_context)
        
        workflow_id = await orchestrator.start_debugging_session(
            user_id=current_user.id,
            error_context=error_context,
            code_context=request.code_context
        )
        
        return {
            "workflow_id": workflow_id,
            "message": "Debugging session workflow started successfully"
        }
        
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start debugging session: {str(e)}")


@router.post("/automation-execution", response_model=Dict[str, str])
async def start_automation_execution(
    request: StartAutomationExecutionRequest,
    current_user: User = Depends(get_current_user),
    orchestrator: WorkflowOrchestrator = Depends(),
    _: None = Depends(require_permissions(["workflow:create"]))
):
    """
    Start an automation execution workflow.
    
    This endpoint initiates an end-to-end automation experience that includes:
    - Pattern detection in user actions
    - Automation opportunity evaluation
    - Workflow generation
    - Automation validation
    - Execution with monitoring
    - Performance tracking
    """
    try:
        workflow_id = await orchestrator.start_automation_execution(
            user_id=current_user.id,
            user_actions=request.user_actions,
            automation_preferences=request.automation_preferences
        )
        
        return {
            "workflow_id": workflow_id,
            "message": "Automation execution workflow started successfully"
        }
        
    except WorkflowError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start automation execution: {str(e)}")


@router.get("/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    orchestrator: WorkflowOrchestrator = Depends(),
    _: None = Depends(require_permissions(["workflow:read"]))
):
    """
    Get the current status of a workflow.
    
    Returns detailed information about workflow progress, including:
    - Current status and step
    - Completed steps
    - Error information (if any)
    - Execution metadata
    """
    try:
        context = await orchestrator.get_workflow_status(workflow_id)
        
        if not context:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check if user has access to this workflow
        if context.user_id != current_user.id and not current_user.has_permission("workflow:read_all"):
            raise HTTPException(status_code=403, detail="Access denied to this workflow")
        
        return WorkflowStatusResponse(
            workflow_id=context.workflow_id,
            user_id=context.user_id,
            workflow_type=context.workflow_type.value,
            status=context.status.value,
            created_at=context.created_at,
            updated_at=context.updated_at,
            steps_completed=context.steps_completed,
            current_step=context.current_step,
            error_info=context.error_info,
            metadata=context.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.post("/{workflow_id}/pause", response_model=Dict[str, str])
async def pause_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    orchestrator: WorkflowOrchestrator = Depends(),
    _: None = Depends(require_permissions(["workflow:control"]))
):
    """
    Pause an active workflow.
    
    Pauses workflow execution at the current step. The workflow can be
    resumed later from the same point.
    """
    try:
        # Check workflow ownership
        context = await orchestrator.get_workflow_status(workflow_id)
        if not context:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if context.user_id != current_user.id and not current_user.has_permission("workflow:control_all"):
            raise HTTPException(status_code=403, detail="Access denied to this workflow")
        
        success = await orchestrator.pause_workflow(workflow_id)
        
        if success:
            return {"message": "Workflow paused successfully"}
        else:
            raise HTTPException(status_code=400, detail="Workflow cannot be paused in current state")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause workflow: {str(e)}")


@router.post("/{workflow_id}/resume", response_model=Dict[str, str])
async def resume_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    orchestrator: WorkflowOrchestrator = Depends(),
    _: None = Depends(require_permissions(["workflow:control"]))
):
    """
    Resume a paused workflow.
    
    Resumes workflow execution from the point where it was paused.
    """
    try:
        # Check workflow ownership
        context = await orchestrator.get_workflow_status(workflow_id)
        if not context:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if context.user_id != current_user.id and not current_user.has_permission("workflow:control_all"):
            raise HTTPException(status_code=403, detail="Access denied to this workflow")
        
        success = await orchestrator.resume_workflow(workflow_id)
        
        if success:
            return {"message": "Workflow resumed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Workflow cannot be resumed in current state")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume workflow: {str(e)}")


@router.post("/{workflow_id}/cancel", response_model=Dict[str, str])
async def cancel_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    orchestrator: WorkflowOrchestrator = Depends(),
    _: None = Depends(require_permissions(["workflow:control"]))
):
    """
    Cancel an active workflow.
    
    Cancels workflow execution. This action cannot be undone.
    """
    try:
        # Check workflow ownership
        context = await orchestrator.get_workflow_status(workflow_id)
        if not context:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if context.user_id != current_user.id and not current_user.has_permission("workflow:control_all"):
            raise HTTPException(status_code=403, detail="Access denied to this workflow")
        
        success = await orchestrator.cancel_workflow(workflow_id)
        
        if success:
            return {"message": "Workflow cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Workflow cannot be cancelled in current state")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel workflow: {str(e)}")


@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    user_id: Optional[str] = None,
    workflow_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    orchestrator: WorkflowOrchestrator = Depends(),
    _: None = Depends(require_permissions(["workflow:read"]))
):
    """
    List workflows with optional filtering.
    
    Returns a list of workflows that the user has access to, with optional
    filtering by user, type, and status.
    """
    try:
        # Determine which user's workflows to show
        target_user_id = user_id
        if not current_user.has_permission("workflow:read_all"):
            target_user_id = current_user.id  # Users can only see their own workflows
        
        # Get workflows
        workflows = await orchestrator.get_active_workflows(target_user_id)
        
        # Apply filters
        if workflow_type:
            try:
                workflow_type_enum = WorkflowType(workflow_type)
                workflows = [w for w in workflows if w.workflow_type == workflow_type_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid workflow type: {workflow_type}")
        
        if status:
            try:
                status_enum = WorkflowStatus(status)
                workflows = [w for w in workflows if w.status == status_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        # Apply pagination
        total_count = len(workflows)
        workflows = workflows[offset:offset + limit]
        
        # Convert to response format
        workflow_responses = [
            WorkflowStatusResponse(
                workflow_id=w.workflow_id,
                user_id=w.user_id,
                workflow_type=w.workflow_type.value,
                status=w.status.value,
                created_at=w.created_at,
                updated_at=w.updated_at,
                steps_completed=w.steps_completed,
                current_step=w.current_step,
                error_info=w.error_info,
                metadata=w.metadata
            )
            for w in workflows
        ]
        
        return WorkflowListResponse(
            workflows=workflow_responses,
            total_count=total_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")


@router.post("/cleanup", response_model=Dict[str, Any])
async def cleanup_completed_workflows(
    older_than_hours: int = 24,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    orchestrator: WorkflowOrchestrator = Depends(),
    _: None = Depends(require_permissions(["workflow:admin"]))
):
    """
    Clean up completed workflows older than specified hours.
    
    This is an admin endpoint for cleaning up old workflow data to
    prevent memory and storage bloat.
    """
    try:
        # Run cleanup in background
        background_tasks.add_task(
            orchestrator.cleanup_completed_workflows,
            older_than_hours
        )
        
        return {
            "message": f"Cleanup task scheduled for workflows older than {older_than_hours} hours",
            "older_than_hours": older_than_hours
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule cleanup: {str(e)}")


# Dependency injection for WorkflowOrchestrator
async def get_workflow_orchestrator() -> WorkflowOrchestrator:
    """Dependency to get WorkflowOrchestrator instance."""
    # This would be injected by the dependency injection container
    # For now, we'll import and create it here
    from ..services.learning_engine import LearningEngine
    from ..services.debug_assistant import DebugAssistant
    from ..services.context_analyzer import ContextAnalyzer
    from ..services.pattern_detector import PatternDetector
    from ..services.workflow_generator import WorkflowGenerator
    from ..services.analytics_engine import AnalyticsEngine
    from ..services.interaction_service import InteractionService
    
    # These would normally be injected
    learning_engine = LearningEngine()
    debug_assistant = DebugAssistant()
    context_analyzer = ContextAnalyzer()
    pattern_detector = PatternDetector()
    workflow_generator = WorkflowGenerator()
    analytics_engine = AnalyticsEngine()
    interaction_service = InteractionService()
    
    return WorkflowOrchestrator(
        learning_engine=learning_engine,
        debug_assistant=debug_assistant,
        context_analyzer=context_analyzer,
        pattern_detector=pattern_detector,
        workflow_generator=workflow_generator,
        analytics_engine=analytics_engine,
        interaction_service=interaction_service
    )


# Override the dependency
router.dependencies = [Depends(get_workflow_orchestrator)]