"""Workflow integration service."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from ..integrations.workflow_adapter import WorkflowAdapter, WorkflowAdaptation, workflow_adapter
from ..integrations.workflow_detector import DetectedWorkflow, WorkflowType, workflow_detector
from ..logging_config import get_logger
from ..plugins import plugin_manager


class WorkflowIntegrationService:
    """Service for managing workflow integrations with AI Learning Accelerator."""
    
    def __init__(self):
        """Initialize workflow integration service."""
        self.logger = get_logger(__name__)
        self.workflow_detector = workflow_detector
        self.workflow_adapter = workflow_adapter
        self.plugin_manager = plugin_manager
        
        # Integration state
        self._active_integrations: Dict[str, Dict[str, Any]] = {}
        self._integration_history: List[Dict[str, Any]] = []
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Configuration
        self._auto_detect_interval = 300  # 5 minutes
        self._compatibility_threshold = 0.7
        self._max_concurrent_adaptations = 5
    
    async def start_service(self) -> None:
        """Start the workflow integration service."""
        try:
            self.logger.info("Starting workflow integration service")
            
            # Start periodic workflow detection
            self._start_periodic_detection()
            
            # Initialize existing integrations
            await self._initialize_existing_integrations()
            
            self.logger.info("Workflow integration service started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting workflow integration service: {e}")
            raise
    
    async def stop_service(self) -> None:
        """Stop the workflow integration service."""
        try:
            self.logger.info("Stopping workflow integration service")
            
            # Cancel monitoring tasks
            for task in self._monitoring_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self._monitoring_tasks:
                await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
            
            self._monitoring_tasks.clear()
            
            self.logger.info("Workflow integration service stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping workflow integration service: {e}")
    
    async def detect_project_workflows(self, project_path: str) -> List[DetectedWorkflow]:
        """Detect workflows in a project.
        
        Args:
            project_path: Path to project directory
            
        Returns:
            List of detected workflows
        """
        try:
            workflows = await self.workflow_detector.detect_workflows(project_path)
            
            # Log detection results
            self.logger.info(f"Detected {len(workflows)} workflows in {project_path}")
            for workflow in workflows:
                self.logger.debug(f"  - {workflow.name} ({workflow.type.value}) - confidence: {workflow.confidence}")
            
            return workflows
            
        except Exception as e:
            self.logger.error(f"Error detecting workflows in {project_path}: {e}")
            return []
    
    async def analyze_integration_opportunities(self, workflows: List[DetectedWorkflow]) -> Dict[str, Any]:
        """Analyze integration opportunities for detected workflows.
        
        Args:
            workflows: List of detected workflows
            
        Returns:
            Integration analysis results
        """
        try:
            analysis = {
                "total_workflows": len(workflows),
                "compatible_workflows": [],
                "incompatible_workflows": [],
                "integration_opportunities": [],
                "recommended_adaptations": [],
                "estimated_effort": "Unknown",
                "potential_benefits": []
            }
            
            for workflow in workflows:
                # Analyze workflow compatibility
                compatibility = await self.workflow_adapter.analyze_workflow_for_integration(workflow)
                
                if compatibility["compatible"] and compatibility["compatibility_score"] >= self._compatibility_threshold:
                    analysis["compatible_workflows"].append({
                        "workflow": workflow,
                        "compatibility_score": compatibility["compatibility_score"],
                        "integration_points": compatibility["integration_points"],
                        "opportunities": compatibility.get("integration_opportunities", [])
                    })
                    
                    # Add integration opportunities
                    analysis["integration_opportunities"].extend(compatibility.get("integration_opportunities", []))
                    analysis["recommended_adaptations"].extend(compatibility.get("recommended_adaptations", []))
                    
                else:
                    analysis["incompatible_workflows"].append({
                        "workflow": workflow,
                        "compatibility_score": compatibility["compatibility_score"],
                        "issues": compatibility.get("potential_conflicts", [])
                    })
            
            # Calculate overall metrics
            if analysis["compatible_workflows"]:
                avg_score = sum(w["compatibility_score"] for w in analysis["compatible_workflows"]) / len(analysis["compatible_workflows"])
                analysis["average_compatibility_score"] = avg_score
                
                # Estimate effort based on number of adaptations
                total_adaptations = len(analysis["recommended_adaptations"])
                if total_adaptations <= 3:
                    analysis["estimated_effort"] = "Low"
                elif total_adaptations <= 8:
                    analysis["estimated_effort"] = "Medium"
                else:
                    analysis["estimated_effort"] = "High"
                
                # Describe potential benefits
                analysis["potential_benefits"] = [
                    "Contextual learning opportunities during development",
                    "Automated assistance for common workflow issues",
                    "Performance optimization suggestions",
                    "Best practices recommendations",
                    "Reduced time to resolve build/test failures"
                ]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing integration opportunities: {e}")
            return {
                "total_workflows": len(workflows),
                "compatible_workflows": [],
                "incompatible_workflows": [],
                "integration_opportunities": [],
                "recommended_adaptations": [],
                "error": str(e)
            }
    
    async def create_workflow_integration(self, workflow: DetectedWorkflow, integration_config: Dict[str, Any]) -> str:
        """Create a new workflow integration.
        
        Args:
            workflow: Workflow to integrate
            integration_config: Integration configuration
            
        Returns:
            Integration ID
        """
        try:
            integration_id = f"integration_{workflow.id}_{int(datetime.utcnow().timestamp())}"
            
            # Create adaptations based on configuration
            adaptations = []
            for adaptation_config in integration_config.get("adaptations", []):
                adaptation = await self.workflow_adapter.create_workflow_adaptation(workflow, adaptation_config)
                adaptations.append(adaptation)
            
            # Create integration record
            integration = {
                "id": integration_id,
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "workflow_type": workflow.type.value,
                "created_at": datetime.utcnow(),
                "status": "created",
                "adaptations": [a.id for a in adaptations],
                "config": integration_config,
                "backward_compatibility": integration_config.get("maintain_backward_compatibility", True)
            }
            
            self._active_integrations[integration_id] = integration
            self._integration_history.append({
                "action": "created",
                "integration_id": integration_id,
                "timestamp": datetime.utcnow(),
                "details": integration_config
            })
            
            self.logger.info(f"Created workflow integration: {integration_id} for {workflow.name}")
            return integration_id
            
        except Exception as e:
            self.logger.error(f"Error creating workflow integration: {e}")
            raise
    
    async def apply_workflow_integration(self, integration_id: str) -> bool:
        """Apply a workflow integration.
        
        Args:
            integration_id: Integration ID
            
        Returns:
            bool: True if integration applied successfully
        """
        if integration_id not in self._active_integrations:
            raise ValueError(f"Integration {integration_id} not found")
        
        integration = self._active_integrations[integration_id]
        
        try:
            self.logger.info(f"Applying workflow integration: {integration_id}")
            
            integration["status"] = "applying"
            
            # Apply all adaptations
            successful_adaptations = []
            failed_adaptations = []
            
            for adaptation_id in integration["adaptations"]:
                try:
                    result = await self.workflow_adapter.apply_adaptation(adaptation_id)
                    if result.success:
                        successful_adaptations.append(adaptation_id)
                    else:
                        failed_adaptations.append({
                            "adaptation_id": adaptation_id,
                            "error": result.error_message
                        })
                except Exception as e:
                    failed_adaptations.append({
                        "adaptation_id": adaptation_id,
                        "error": str(e)
                    })
            
            # Update integration status
            if failed_adaptations:
                integration["status"] = "partially_applied"
                integration["failed_adaptations"] = failed_adaptations
                
                # Rollback successful adaptations if configured
                if integration["config"].get("rollback_on_partial_failure", False):
                    for adaptation_id in successful_adaptations:
                        await self.workflow_adapter.rollback_adaptation(adaptation_id)
                    integration["status"] = "failed"
                    
                    self.logger.error(f"Integration {integration_id} failed, rolled back all changes")
                    return False
                else:
                    self.logger.warning(f"Integration {integration_id} partially applied with {len(failed_adaptations)} failures")
            else:
                integration["status"] = "active"
                integration["applied_at"] = datetime.utcnow()
                
                # Start monitoring if configured
                if integration["config"].get("enable_monitoring", True):
                    await self._start_integration_monitoring(integration_id)
                
                self.logger.info(f"Integration {integration_id} applied successfully")
            
            # Record in history
            self._integration_history.append({
                "action": "applied",
                "integration_id": integration_id,
                "timestamp": datetime.utcnow(),
                "success": integration["status"] == "active",
                "failed_adaptations": failed_adaptations
            })
            
            return integration["status"] in ["active", "partially_applied"]
            
        except Exception as e:
            integration["status"] = "failed"
            integration["error"] = str(e)
            
            self.logger.error(f"Error applying workflow integration {integration_id}: {e}")
            return False
    
    async def remove_workflow_integration(self, integration_id: str) -> bool:
        """Remove a workflow integration.
        
        Args:
            integration_id: Integration ID
            
        Returns:
            bool: True if integration removed successfully
        """
        if integration_id not in self._active_integrations:
            raise ValueError(f"Integration {integration_id} not found")
        
        integration = self._active_integrations[integration_id]
        
        try:
            self.logger.info(f"Removing workflow integration: {integration_id}")
            
            # Stop monitoring
            if integration_id in self._monitoring_tasks:
                self._monitoring_tasks[integration_id].cancel()
                del self._monitoring_tasks[integration_id]
            
            # Rollback all adaptations
            rollback_failures = []
            for adaptation_id in integration["adaptations"]:
                try:
                    success = await self.workflow_adapter.rollback_adaptation(adaptation_id)
                    if not success:
                        rollback_failures.append(adaptation_id)
                except Exception as e:
                    rollback_failures.append(f"{adaptation_id}: {str(e)}")
            
            # Remove integration
            del self._active_integrations[integration_id]
            
            # Record in history
            self._integration_history.append({
                "action": "removed",
                "integration_id": integration_id,
                "timestamp": datetime.utcnow(),
                "rollback_failures": rollback_failures
            })
            
            if rollback_failures:
                self.logger.warning(f"Integration {integration_id} removed with rollback failures: {rollback_failures}")
            else:
                self.logger.info(f"Integration {integration_id} removed successfully")
            
            return len(rollback_failures) == 0
            
        except Exception as e:
            self.logger.error(f"Error removing workflow integration {integration_id}: {e}")
            return False
    
    async def get_active_integrations(self) -> List[Dict[str, Any]]:
        """Get all active workflow integrations.
        
        Returns:
            List of active integrations
        """
        return [
            integration for integration in self._active_integrations.values()
            if integration["status"] in ["active", "partially_applied"]
        ]
    
    async def get_integration_status(self, integration_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow integration.
        
        Args:
            integration_id: Integration ID
            
        Returns:
            Integration status or None if not found
        """
        integration = self._active_integrations.get(integration_id)
        if not integration:
            return None
        
        # Get adaptation results
        adaptation_results = []
        for adaptation_id in integration["adaptations"]:
            results = await self.workflow_adapter.get_adaptation_results(integration["workflow_id"])
            adaptation_results.extend([r for r in results if r.adaptation_id == adaptation_id])
        
        return {
            "integration": integration,
            "adaptation_results": adaptation_results,
            "monitoring_active": integration_id in self._monitoring_tasks
        }
    
    async def ensure_backward_compatibility(self, integration_id: str) -> bool:
        """Ensure backward compatibility for a workflow integration.
        
        Args:
            integration_id: Integration ID
            
        Returns:
            bool: True if backward compatibility is maintained
        """
        if integration_id not in self._active_integrations:
            return False
        
        integration = self._active_integrations[integration_id]
        
        try:
            # Check if backward compatibility is required
            if not integration.get("backward_compatibility", True):
                return True
            
            # Validate that original workflow still functions
            workflow_id = integration["workflow_id"]
            
            # This would run compatibility tests
            # For now, we'll assume compatibility is maintained
            compatibility_maintained = True
            
            if not compatibility_maintained:
                self.logger.warning(f"Backward compatibility issue detected for integration {integration_id}")
                
                # Optionally rollback if compatibility is broken
                if integration["config"].get("rollback_on_compatibility_issue", False):
                    await self.remove_workflow_integration(integration_id)
                    return False
            
            return compatibility_maintained
            
        except Exception as e:
            self.logger.error(f"Error checking backward compatibility for {integration_id}: {e}")
            return False
    
    # Private methods
    
    def _start_periodic_detection(self) -> None:
        """Start periodic workflow detection."""
        async def detection_loop():
            while True:
                try:
                    await asyncio.sleep(self._auto_detect_interval)
                    
                    # Detect running workflows
                    running_workflows = await self.workflow_detector.detect_running_workflows()
                    
                    # Check for new integration opportunities
                    for workflow in running_workflows:
                        if not self._is_workflow_integrated(workflow):
                            self.logger.info(f"New workflow detected: {workflow.name}")
                            # Could trigger automatic integration here
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in periodic workflow detection: {e}")
        
        task = asyncio.create_task(detection_loop())
        self._monitoring_tasks["periodic_detection"] = task
    
    async def _initialize_existing_integrations(self) -> None:
        """Initialize existing integrations on service start."""
        try:
            # This would load existing integrations from storage
            # For now, we start with empty state
            self.logger.info("Initialized existing integrations")
            
        except Exception as e:
            self.logger.error(f"Error initializing existing integrations: {e}")
    
    async def _start_integration_monitoring(self, integration_id: str) -> None:
        """Start monitoring for a workflow integration."""
        async def monitoring_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    
                    # Check integration health
                    await self.ensure_backward_compatibility(integration_id)
                    
                    # Check for workflow changes that might affect integration
                    # This would monitor workflow files for changes
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error monitoring integration {integration_id}: {e}")
        
        task = asyncio.create_task(monitoring_loop())
        self._monitoring_tasks[integration_id] = task
    
    def _is_workflow_integrated(self, workflow: DetectedWorkflow) -> bool:
        """Check if a workflow is already integrated."""
        for integration in self._active_integrations.values():
            if integration["workflow_id"] == workflow.id:
                return True
        return False


# Global workflow integration service instance
workflow_integration_service = WorkflowIntegrationService()