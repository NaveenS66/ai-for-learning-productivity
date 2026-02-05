"""
Workflow Orchestrator Service

This service orchestrates end-to-end user workflows by coordinating multiple
AI Learning Accelerator components to deliver complete user experiences.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4

from ..utils.exceptions import AILearningAcceleratorException, IntegrationException

logger = logging.getLogger(__name__)


# Simple decorator for workflow monitoring
def monitor_workflow(func):
    """Decorator for monitoring workflow execution."""
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


# Simple function for tracking metrics
async def track_metrics(metrics: Dict[str, Any]):
    """Track workflow metrics."""
    logger.info(f"Workflow metrics: {metrics}")


class WorkflowType(Enum):
    """Types of end-to-end workflows supported by the orchestrator."""
    LEARNING_JOURNEY = "learning_journey"
    DEBUGGING_SESSION = "debugging_session"
    AUTOMATION_EXECUTION = "automation_execution"


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowContext:
    """Context information for workflow execution."""
    workflow_id: str
    user_id: str
    workflow_type: WorkflowType
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    steps_completed: List[str] = field(default_factory=list)
    current_step: Optional[str] = None
    error_info: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    name: str
    description: str
    component: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3


class WorkflowOrchestrator:
    """
    Orchestrates end-to-end user workflows by coordinating multiple
    AI Learning Accelerator components.
    """

    def __init__(
        self,
        learning_engine=None,
        debug_assistant=None,
        context_analyzer=None,
        pattern_detector=None,
        workflow_generator=None,
        analytics_engine=None,
        interaction_service=None
    ):
        self.learning_engine = learning_engine
        self.debug_assistant = debug_assistant
        self.context_analyzer = context_analyzer
        self.pattern_detector = pattern_detector
        self.workflow_generator = workflow_generator
        self.analytics_engine = analytics_engine
        self.interaction_service = interaction_service
        
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.workflow_templates: Dict[WorkflowType, List[WorkflowStep]] = {}
        
        self._initialize_workflow_templates()

    def _initialize_workflow_templates(self):
        """Initialize workflow templates for different workflow types."""
        
        # Learning Journey Workflow Template
        self.workflow_templates[WorkflowType.LEARNING_JOURNEY] = [
            WorkflowStep(
                step_id="assess_skills",
                name="Assess Current Skills",
                description="Evaluate user's current skill level and competencies",
                component="learning_engine",
                action="assess_skill_level",
                parameters={}
            ),
            WorkflowStep(
                step_id="analyze_goals",
                name="Analyze Learning Goals",
                description="Process and validate user's learning objectives",
                component="learning_engine",
                action="analyze_learning_goals",
                parameters={},
                dependencies=["assess_skills"]
            ),
            WorkflowStep(
                step_id="generate_path",
                name="Generate Learning Path",
                description="Create personalized learning path with milestones",
                component="learning_engine",
                action="generate_learning_path",
                parameters={},
                dependencies=["analyze_goals"]
            ),
            WorkflowStep(
                step_id="adapt_content",
                name="Adapt Content",
                description="Customize content based on user preferences and skill level",
                component="learning_engine",
                action="adapt_content_for_user",
                parameters={},
                dependencies=["generate_path"]
            ),
            WorkflowStep(
                step_id="setup_monitoring",
                name="Setup Progress Monitoring",
                description="Initialize progress tracking and analytics",
                component="analytics_engine",
                action="setup_learning_analytics",
                parameters={},
                dependencies=["adapt_content"]
            ),
            WorkflowStep(
                step_id="deliver_content",
                name="Deliver Learning Content",
                description="Present learning materials through appropriate channels",
                component="interaction_service",
                action="deliver_learning_content",
                parameters={},
                dependencies=["setup_monitoring"]
            )
        ]
        
        # Debugging Session Workflow Template
        self.workflow_templates[WorkflowType.DEBUGGING_SESSION] = [
            WorkflowStep(
                step_id="analyze_context",
                name="Analyze Code Context",
                description="Understand current code context and environment",
                component="context_analyzer",
                action="analyze_workspace",
                parameters={}
            ),
            WorkflowStep(
                step_id="analyze_error",
                name="Analyze Error",
                description="Process error information and identify root causes",
                component="debug_assistant",
                action="analyze_error",
                parameters={},
                dependencies=["analyze_context"]
            ),
            WorkflowStep(
                step_id="generate_solutions",
                name="Generate Solutions",
                description="Create ranked list of potential solutions",
                component="debug_assistant",
                action="suggest_solutions",
                parameters={},
                dependencies=["analyze_error"]
            ),
            WorkflowStep(
                step_id="adapt_guidance",
                name="Adapt Debugging Guidance",
                description="Customize guidance based on user skill level",
                component="debug_assistant",
                action="adapt_debugging_guidance",
                parameters={},
                dependencies=["generate_solutions"]
            ),
            WorkflowStep(
                step_id="guide_resolution",
                name="Guide Through Resolution",
                description="Provide step-by-step debugging assistance",
                component="debug_assistant",
                action="guide_through_debugging",
                parameters={},
                dependencies=["adapt_guidance"]
            ),
            WorkflowStep(
                step_id="learn_from_session",
                name="Learn from Session",
                description="Store successful patterns for future use",
                component="debug_assistant",
                action="learn_from_success",
                parameters={},
                dependencies=["guide_resolution"]
            )
        ]
        
        # Automation Execution Workflow Template
        self.workflow_templates[WorkflowType.AUTOMATION_EXECUTION] = [
            WorkflowStep(
                step_id="detect_patterns",
                name="Detect Automation Patterns",
                description="Identify repetitive patterns in user actions",
                component="pattern_detector",
                action="detect_repetitive_patterns",
                parameters={}
            ),
            WorkflowStep(
                step_id="evaluate_opportunities",
                name="Evaluate Automation Opportunities",
                description="Assess automation potential and benefits",
                component="pattern_detector",
                action="evaluate_automation_opportunities",
                parameters={},
                dependencies=["detect_patterns"]
            ),
            WorkflowStep(
                step_id="generate_workflow",
                name="Generate Automation Workflow",
                description="Create automated workflow for identified patterns",
                component="workflow_generator",
                action="generate_automation",
                parameters={},
                dependencies=["evaluate_opportunities"]
            ),
            WorkflowStep(
                step_id="validate_automation",
                name="Validate Automation",
                description="Test and validate generated automation workflow",
                component="workflow_generator",
                action="validate_automation",
                parameters={},
                dependencies=["generate_workflow"]
            ),
            WorkflowStep(
                step_id="execute_automation",
                name="Execute Automation",
                description="Run the automated workflow with monitoring",
                component="workflow_generator",
                action="execute_automation",
                parameters={},
                dependencies=["validate_automation"]
            ),
            WorkflowStep(
                step_id="monitor_execution",
                name="Monitor Execution",
                description="Track automation performance and results",
                component="analytics_engine",
                action="monitor_automation",
                parameters={},
                dependencies=["execute_automation"]
            )
        ]

    @monitor_workflow
    async def start_learning_journey(
        self,
        user_id: str,
        learning_goals: List[Dict[str, Any]],
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a complete learning journey workflow.
        
        Args:
            user_id: ID of the user starting the journey
            learning_goals: List of learning objectives
            preferences: Optional user preferences for the journey
            
        Returns:
            Workflow ID for tracking progress
        """
        workflow_id = str(uuid4())
        
        try:
            # Create workflow context
            context = WorkflowContext(
                workflow_id=workflow_id,
                user_id=user_id,
                workflow_type=WorkflowType.LEARNING_JOURNEY,
                status=WorkflowStatus.INITIATED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={
                    "learning_goals": learning_goals,
                    "preferences": preferences or {}
                }
            )
            
            self.active_workflows[workflow_id] = context
            
            # Start workflow execution
            await self._execute_workflow(workflow_id)
            
            logger.info(f"Started learning journey workflow {workflow_id} for user {user_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to start learning journey: {str(e)}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].status = WorkflowStatus.FAILED
                self.active_workflows[workflow_id].error_info = {"error": str(e)}
            raise AILearningAcceleratorException(f"Failed to start learning journey: {str(e)}")

    @monitor_workflow
    async def start_debugging_session(
        self,
        user_id: str,
        error_context: Dict[str, Any],
        code_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a debugging session workflow.
        
        Args:
            user_id: ID of the user requesting debugging help
            error_context: Information about the error encountered
            code_context: Optional code context information
            
        Returns:
            Workflow ID for tracking progress
        """
        workflow_id = str(uuid4())
        
        try:
            # Create workflow context
            context = WorkflowContext(
                workflow_id=workflow_id,
                user_id=user_id,
                workflow_type=WorkflowType.DEBUGGING_SESSION,
                status=WorkflowStatus.INITIATED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={
                    "error_context": error_context,
                    "code_context": code_context or {}
                }
            )
            
            self.active_workflows[workflow_id] = context
            
            # Start workflow execution
            await self._execute_workflow(workflow_id)
            
            logger.info(f"Started debugging session workflow {workflow_id} for user {user_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to start debugging session: {str(e)}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].status = WorkflowStatus.FAILED
                self.active_workflows[workflow_id].error_info = {"error": str(e)}
            raise AILearningAcceleratorException(f"Failed to start debugging session: {str(e)}")

    @monitor_workflow
    async def start_automation_execution(
        self,
        user_id: str,
        user_actions: List[Dict[str, Any]],
        automation_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start an automation execution workflow.
        
        Args:
            user_id: ID of the user requesting automation
            user_actions: List of user actions to analyze for automation
            automation_preferences: Optional automation preferences
            
        Returns:
            Workflow ID for tracking progress
        """
        workflow_id = str(uuid4())
        
        try:
            # Create workflow context
            context = WorkflowContext(
                workflow_id=workflow_id,
                user_id=user_id,
                workflow_type=WorkflowType.AUTOMATION_EXECUTION,
                status=WorkflowStatus.INITIATED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata={
                    "user_actions": user_actions,
                    "automation_preferences": automation_preferences or {}
                }
            )
            
            self.active_workflows[workflow_id] = context
            
            # Start workflow execution
            await self._execute_workflow(workflow_id)
            
            logger.info(f"Started automation execution workflow {workflow_id} for user {user_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to start automation execution: {str(e)}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].status = WorkflowStatus.FAILED
                self.active_workflows[workflow_id].error_info = {"error": str(e)}
            raise AILearningAcceleratorException(f"Failed to start automation execution: {str(e)}")

    async def _execute_workflow(self, workflow_id: str):
        """Execute a workflow by running its steps in order."""
        context = self.active_workflows[workflow_id]
        workflow_steps = self.workflow_templates[context.workflow_type]
        
        try:
            context.status = WorkflowStatus.IN_PROGRESS
            context.updated_at = datetime.utcnow()
            
            # Execute steps in dependency order
            for step in workflow_steps:
                if not self._can_execute_step(step, context.steps_completed):
                    continue
                    
                context.current_step = step.step_id
                context.updated_at = datetime.utcnow()
                
                # Execute step with retry logic
                success = await self._execute_step_with_retry(step, context)
                
                if success:
                    context.steps_completed.append(step.step_id)
                    logger.info(f"Completed step {step.step_id} in workflow {workflow_id}")
                else:
                    raise AILearningAcceleratorException(f"Step {step.step_id} failed after {step.max_retries} retries")
            
            # Mark workflow as completed
            context.status = WorkflowStatus.COMPLETED
            context.current_step = None
            context.updated_at = datetime.utcnow()
            
            # Track completion metrics
            await self._track_workflow_completion(context)
            
            logger.info(f"Completed workflow {workflow_id}")
            
        except Exception as e:
            context.status = WorkflowStatus.FAILED
            context.error_info = {"error": str(e), "step": context.current_step}
            context.updated_at = datetime.utcnow()
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            raise

    def _can_execute_step(self, step: WorkflowStep, completed_steps: List[str]) -> bool:
        """Check if a step can be executed based on its dependencies."""
        return all(dep in completed_steps for dep in step.dependencies)

    async def _execute_step_with_retry(self, step: WorkflowStep, context: WorkflowContext) -> bool:
        """Execute a workflow step with retry logic."""
        for attempt in range(step.max_retries + 1):
            try:
                # Prepare step parameters
                parameters = self._prepare_step_parameters(step, context)
                
                # Execute step based on component
                result = await self._execute_component_action(
                    step.component,
                    step.action,
                    parameters
                )
                
                # Store result in context
                if "step_results" not in context.metadata:
                    context.metadata["step_results"] = {}
                context.metadata["step_results"][step.step_id] = result
                
                return True
                
            except Exception as e:
                step.retry_count = attempt + 1
                logger.warning(f"Step {step.step_id} attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < step.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Step {step.step_id} failed after {step.max_retries + 1} attempts")
                    return False
        
        return False

    def _prepare_step_parameters(self, step: WorkflowStep, context: WorkflowContext) -> Dict[str, Any]:
        """Prepare parameters for step execution."""
        parameters = step.parameters.copy()
        
        # Add common parameters
        parameters["user_id"] = context.user_id
        parameters["workflow_id"] = context.workflow_id
        
        # Add workflow-specific parameters
        if context.workflow_type == WorkflowType.LEARNING_JOURNEY:
            parameters.update({
                "learning_goals": context.metadata.get("learning_goals", []),
                "preferences": context.metadata.get("preferences", {})
            })
        elif context.workflow_type == WorkflowType.DEBUGGING_SESSION:
            parameters.update({
                "error_context": context.metadata.get("error_context", {}),
                "code_context": context.metadata.get("code_context", {})
            })
        elif context.workflow_type == WorkflowType.AUTOMATION_EXECUTION:
            parameters.update({
                "user_actions": context.metadata.get("user_actions", []),
                "automation_preferences": context.metadata.get("automation_preferences", {})
            })
        
        # Add results from previous steps
        if "step_results" in context.metadata:
            parameters["previous_results"] = context.metadata["step_results"]
        
        return parameters

    async def _execute_component_action(
        self,
        component: str,
        action: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute an action on a specific component."""
        try:
            # For now, return mock results to allow testing
            # In production, this would call the actual component methods
            return {
                "component": component,
                "action": action,
                "status": "success",
                "result": f"Mock result for {component}.{action}",
                "parameters": parameters
            }
                
        except Exception as e:
            logger.error(f"Component action failed - {component}.{action}: {str(e)}")
            raise IntegrationException(f"Component action failed: {str(e)}")

    async def _track_workflow_completion(self, context: WorkflowContext):
        """Track metrics for completed workflow."""
        completion_time = (context.updated_at - context.created_at).total_seconds()
        
        await track_metrics({
            "workflow_type": context.workflow_type.value,
            "completion_time": completion_time,
            "steps_completed": len(context.steps_completed),
            "user_id": context.user_id,
            "success": context.status == WorkflowStatus.COMPLETED
        })

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Get the current status of a workflow."""
        return self.active_workflows.get(workflow_id)

    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause an active workflow."""
        if workflow_id in self.active_workflows:
            context = self.active_workflows[workflow_id]
            if context.status == WorkflowStatus.IN_PROGRESS:
                context.status = WorkflowStatus.PAUSED
                context.updated_at = datetime.utcnow()
                logger.info(f"Paused workflow {workflow_id}")
                return True
        return False

    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id in self.active_workflows:
            context = self.active_workflows[workflow_id]
            if context.status == WorkflowStatus.PAUSED:
                context.status = WorkflowStatus.IN_PROGRESS
                context.updated_at = datetime.utcnow()
                # Continue execution from current step
                await self._execute_workflow(workflow_id)
                logger.info(f"Resumed workflow {workflow_id}")
                return True
        return False

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow."""
        if workflow_id in self.active_workflows:
            context = self.active_workflows[workflow_id]
            if context.status in [WorkflowStatus.IN_PROGRESS, WorkflowStatus.PAUSED]:
                context.status = WorkflowStatus.CANCELLED
                context.updated_at = datetime.utcnow()
                logger.info(f"Cancelled workflow {workflow_id}")
                return True
        return False

    async def get_active_workflows(self, user_id: Optional[str] = None) -> List[WorkflowContext]:
        """Get list of active workflows, optionally filtered by user."""
        workflows = list(self.active_workflows.values())
        
        if user_id:
            workflows = [w for w in workflows if w.user_id == user_id]
        
        # Filter to only active workflows
        active_statuses = [WorkflowStatus.INITIATED, WorkflowStatus.IN_PROGRESS, WorkflowStatus.PAUSED]
        return [w for w in workflows if w.status in active_statuses]

    async def cleanup_completed_workflows(self, older_than_hours: int = 24):
        """Clean up completed workflows older than specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        completed_statuses = [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
        
        workflows_to_remove = []
        for workflow_id, context in self.active_workflows.items():
            if (context.status in completed_statuses and 
                context.updated_at < cutoff_time):
                workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            del self.active_workflows[workflow_id]
            logger.info(f"Cleaned up workflow {workflow_id}")
        
        return len(workflows_to_remove)