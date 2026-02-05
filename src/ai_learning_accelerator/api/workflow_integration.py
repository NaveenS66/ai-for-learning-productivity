"""Workflow integration API endpoints."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..auth.dependencies import get_current_user
from ..models.user import User
from ..services.workflow_integration import workflow_integration_service
from ..integrations.workflow_detector import DetectedWorkflow, WorkflowType
from ..integrations.workflow_adapter import AdaptationType, IntegrationPoint

router = APIRouter(prefix="/workflow-integration", tags=["workflow-integration"])


# Request/Response Models

class WorkflowDetectionRequest(BaseModel):
    """Request to detect workflows in a project."""
    project_path: str = Field(..., description="Path to project directory")


class WorkflowDetectionResponse(BaseModel):
    """Response from workflow detection."""
    workflows: List[Dict[str, Any]] = Field(..., description="Detected workflows")
    total_count: int = Field(..., description="Total number of workflows detected")


class IntegrationAnalysisResponse(BaseModel):
    """Response from integration analysis."""
    total_workflows: int = Field(..., description="Total workflows analyzed")
    compatible_workflows: List[Dict[str, Any]] = Field(..., description="Compatible workflows")
    incompatible_workflows: List[Dict[str, Any]] = Field(..., description="Incompatible workflows")
    integration_opportunities: List[Dict[str, Any]] = Field(..., description="Integration opportunities")
    recommended_adaptations: List[Dict[str, Any]] = Field(..., description="Recommended adaptations")
    estimated_effort: str = Field(..., description="Estimated integration effort")
    potential_benefits: List[str] = Field(..., description="Potential benefits")
    average_compatibility_score: Optional[float] = Field(None, description="Average compatibility score")


class AdaptationConfigRequest(BaseModel):
    """Configuration for a workflow adaptation."""
    adaptation_type: AdaptationType = Field(..., description="Type of adaptation")
    integration_point: IntegrationPoint = Field(..., description="Integration point")
    name: str = Field(..., description="Adaptation name")
    description: Optional[str] = Field(None, description="Adaptation description")
    priority: int = Field(default=100, description="Execution priority")
    script_path: Optional[str] = Field(None, description="Path to adaptation script")
    command: Optional[str] = Field(None, description="Command to execute")
    config_changes: Dict[str, Any] = Field(default_factory=dict, description="Configuration changes")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    conditions: List[str] = Field(default_factory=list, description="Execution conditions")
    dependencies: List[str] = Field(default_factory=list, description="Adaptation dependencies")
    timeout: Optional[int] = Field(None, description="Execution timeout")


class IntegrationCreateRequest(BaseModel):
    """Request to create a workflow integration."""
    workflow_id: str = Field(..., description="Workflow ID to integrate")
    adaptations: List[AdaptationConfigRequest] = Field(..., description="Adaptations to apply")
    maintain_backward_compatibility: bool = Field(default=True, description="Maintain backward compatibility")
    enable_monitoring: bool = Field(default=True, description="Enable integration monitoring")
    rollback_on_partial_failure: bool = Field(default=False, description="Rollback on partial failure")
    rollback_on_compatibility_issue: bool = Field(default=False, description="Rollback on compatibility issues")


class IntegrationResponse(BaseModel):
    """Response from integration operations."""
    integration_id: str = Field(..., description="Integration ID")
    status: str = Field(..., description="Integration status")
    message: str = Field(..., description="Status message")


class IntegrationStatusResponse(BaseModel):
    """Response with integration status details."""
    integration: Dict[str, Any] = Field(..., description="Integration details")
    adaptation_results: List[Dict[str, Any]] = Field(..., description="Adaptation results")
    monitoring_active: bool = Field(..., description="Whether monitoring is active")


# API Endpoints

@router.post("/detect", response_model=WorkflowDetectionResponse)
async def detect_workflows(
    request: WorkflowDetectionRequest,
    current_user: User = Depends(get_current_user)
):
    """Detect workflows in a project directory."""
    try:
        workflows = await workflow_integration_service.detect_project_workflows(request.project_path)
        
        # Convert workflows to dict format for response
        workflow_dicts = []
        for workflow in workflows:
            workflow_dict = {
                "id": workflow.id,
                "name": workflow.name,
                "type": workflow.type.value,
                "description": workflow.description,
                "confidence": workflow.confidence,
                "source_files": workflow.source_files,
                "tools": [tool.value for tool in workflow.tools],
                "triggers": workflow.triggers,
                "integration_points": workflow.integration_points,
                "compatibility_issues": workflow.compatibility_issues,
                "project_path": workflow.project_path,
                "config_files": workflow.config_files,
                "documentation": workflow.documentation,
                "detected_at": workflow.detected_at.isoformat(),
                "steps": [
                    {
                        "id": step.id,
                        "name": step.name,
                        "description": step.description,
                        "command": step.command,
                        "tool": step.tool.value if step.tool else None,
                        "dependencies": step.dependencies,
                        "inputs": step.inputs,
                        "outputs": step.outputs,
                        "environment": step.environment,
                        "conditions": step.conditions,
                        "timeout": step.timeout,
                        "retry_count": step.retry_count,
                        "parallel": step.parallel
                    }
                    for step in workflow.steps
                ]
            }
            workflow_dicts.append(workflow_dict)
        
        return WorkflowDetectionResponse(
            workflows=workflow_dicts,
            total_count=len(workflows)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting workflows: {str(e)}"
        )


@router.post("/analyze", response_model=IntegrationAnalysisResponse)
async def analyze_integration_opportunities(
    workflows: List[Dict[str, Any]],
    current_user: User = Depends(get_current_user)
):
    """Analyze integration opportunities for detected workflows."""
    try:
        # Convert dict workflows back to DetectedWorkflow objects
        workflow_objects = []
        for workflow_dict in workflows:
            # This is a simplified conversion - in practice you'd want full reconstruction
            workflow = DetectedWorkflow(
                id=workflow_dict["id"],
                name=workflow_dict["name"],
                type=WorkflowType(workflow_dict["type"]),
                description=workflow_dict.get("description"),
                confidence=workflow_dict["confidence"],
                source_files=workflow_dict.get("source_files", []),
                project_path=workflow_dict.get("project_path")
            )
            workflow_objects.append(workflow)
        
        analysis = await workflow_integration_service.analyze_integration_opportunities(workflow_objects)
        
        return IntegrationAnalysisResponse(**analysis)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing integration opportunities: {str(e)}"
        )


@router.post("/create", response_model=IntegrationResponse)
async def create_integration(
    request: IntegrationCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new workflow integration."""
    try:
        # First, we need to get the workflow object
        # In practice, you'd store detected workflows and retrieve by ID
        # For now, we'll create a minimal workflow object
        workflow = DetectedWorkflow(
            id=request.workflow_id,
            name=f"Workflow_{request.workflow_id}",
            type=WorkflowType.CUSTOM,
            confidence=1.0
        )
        
        # Convert adaptation requests to config format
        integration_config = {
            "adaptations": [
                {
                    "adaptation_type": adaptation.adaptation_type.value,
                    "integration_point": adaptation.integration_point.value,
                    "name": adaptation.name,
                    "description": adaptation.description,
                    "priority": adaptation.priority,
                    "script_path": adaptation.script_path,
                    "command": adaptation.command,
                    "config_changes": adaptation.config_changes,
                    "environment_vars": adaptation.environment_vars,
                    "conditions": adaptation.conditions,
                    "dependencies": adaptation.dependencies,
                    "timeout": adaptation.timeout,
                    "created_by": current_user.id
                }
                for adaptation in request.adaptations
            ],
            "maintain_backward_compatibility": request.maintain_backward_compatibility,
            "enable_monitoring": request.enable_monitoring,
            "rollback_on_partial_failure": request.rollback_on_partial_failure,
            "rollback_on_compatibility_issue": request.rollback_on_compatibility_issue
        }
        
        integration_id = await workflow_integration_service.create_workflow_integration(
            workflow, integration_config
        )
        
        return IntegrationResponse(
            integration_id=integration_id,
            status="created",
            message="Workflow integration created successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating integration: {str(e)}"
        )


@router.post("/{integration_id}/apply", response_model=IntegrationResponse)
async def apply_integration(
    integration_id: str,
    current_user: User = Depends(get_current_user)
):
    """Apply a workflow integration."""
    try:
        success = await workflow_integration_service.apply_workflow_integration(integration_id)
        
        if success:
            return IntegrationResponse(
                integration_id=integration_id,
                status="applied",
                message="Workflow integration applied successfully"
            )
        else:
            return IntegrationResponse(
                integration_id=integration_id,
                status="failed",
                message="Failed to apply workflow integration"
            )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error applying integration: {str(e)}"
        )


@router.delete("/{integration_id}", response_model=IntegrationResponse)
async def remove_integration(
    integration_id: str,
    current_user: User = Depends(get_current_user)
):
    """Remove a workflow integration."""
    try:
        success = await workflow_integration_service.remove_workflow_integration(integration_id)
        
        if success:
            return IntegrationResponse(
                integration_id=integration_id,
                status="removed",
                message="Workflow integration removed successfully"
            )
        else:
            return IntegrationResponse(
                integration_id=integration_id,
                status="partially_removed",
                message="Workflow integration removed with some rollback failures"
            )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing integration: {str(e)}"
        )


@router.get("/active", response_model=List[Dict[str, Any]])
async def get_active_integrations(
    current_user: User = Depends(get_current_user)
):
    """Get all active workflow integrations."""
    try:
        integrations = await workflow_integration_service.get_active_integrations()
        return integrations
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting active integrations: {str(e)}"
        )


@router.get("/{integration_id}/status", response_model=IntegrationStatusResponse)
async def get_integration_status(
    integration_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a workflow integration."""
    try:
        status_info = await workflow_integration_service.get_integration_status(integration_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration {integration_id} not found"
            )
        
        return IntegrationStatusResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting integration status: {str(e)}"
        )


@router.post("/{integration_id}/compatibility-check", response_model=Dict[str, Any])
async def check_backward_compatibility(
    integration_id: str,
    current_user: User = Depends(get_current_user)
):
    """Check backward compatibility for a workflow integration."""
    try:
        compatibility_maintained = await workflow_integration_service.ensure_backward_compatibility(integration_id)
        
        return {
            "integration_id": integration_id,
            "backward_compatible": compatibility_maintained,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking compatibility: {str(e)}"
        )


@router.get("/running", response_model=WorkflowDetectionResponse)
async def detect_running_workflows(
    current_user: User = Depends(get_current_user)
):
    """Detect currently running workflows."""
    try:
        from ..integrations.workflow_detector import workflow_detector
        
        workflows = await workflow_detector.detect_running_workflows()
        
        # Convert workflows to dict format for response
        workflow_dicts = []
        for workflow in workflows:
            workflow_dict = {
                "id": workflow.id,
                "name": workflow.name,
                "type": workflow.type.value,
                "description": workflow.description,
                "confidence": workflow.confidence,
                "source_files": workflow.source_files,
                "tools": [tool.value for tool in workflow.tools],
                "triggers": workflow.triggers,
                "integration_points": workflow.integration_points,
                "compatibility_issues": workflow.compatibility_issues,
                "detected_at": workflow.detected_at.isoformat()
            }
            workflow_dicts.append(workflow_dict)
        
        return WorkflowDetectionResponse(
            workflows=workflow_dicts,
            total_count=len(workflows)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting running workflows: {str(e)}"
        )


# Import datetime for compatibility check endpoint
from datetime import datetime