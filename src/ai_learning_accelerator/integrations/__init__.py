"""Integration and workflow management system."""

from .workflow_detector import WorkflowDetector, workflow_detector
# Temporarily comment out workflow_adapter imports to fix circular import
# from .workflow_adapter import WorkflowAdapter, workflow_adapter
from .integration_manager import IntegrationManager, integration_manager

__all__ = [
    "WorkflowDetector",
    "workflow_detector",
    # "WorkflowAdapter", 
    # "workflow_adapter",
    "IntegrationManager",
    "integration_manager"
]