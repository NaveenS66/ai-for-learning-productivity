"""Tests for workflow integration system."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ai_learning_accelerator.integrations.workflow_adapter import (
    AdaptationType,
    IntegrationPoint,
    WorkflowAdapter,
    WorkflowAdaptation,
)
from src.ai_learning_accelerator.integrations.workflow_detector import (
    DetectedWorkflow,
    WorkflowDetector,
    WorkflowTool,
    WorkflowType,
)
from src.ai_learning_accelerator.services.workflow_integration import (
    WorkflowIntegrationService,
)


class TestWorkflowDetector:
    """Test workflow detection functionality."""
    
    @pytest.fixture
    def workflow_detector(self):
        """Create workflow detector instance."""
        return WorkflowDetector()
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            yield project_path
    
    @pytest.mark.asyncio
    async def test_detect_maven_workflow(self, workflow_detector, temp_project_dir):
        """Test detection of Maven workflow."""
        # Create pom.xml file
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
</project>"""
        
        pom_file = temp_project_dir / "pom.xml"
        pom_file.write_text(pom_content)
        
        # Detect workflows
        workflows = await workflow_detector.detect_workflows(str(temp_project_dir))
        
        # Verify Maven workflow detected
        assert len(workflows) > 0
        maven_workflow = next((w for w in workflows if w.type == WorkflowType.BUILD_SYSTEM), None)
        assert maven_workflow is not None
        assert maven_workflow.name == "Maven Build"
        assert WorkflowTool.MAVEN in maven_workflow.tools
        assert "pom.xml" in maven_workflow.source_files
    
    @pytest.mark.asyncio
    async def test_detect_npm_workflow(self, workflow_detector, temp_project_dir):
        """Test detection of NPM workflow."""
        # Create package.json file
        package_content = {
            "name": "test-project",
            "version": "1.0.0",
            "scripts": {
                "build": "webpack --mode production",
                "test": "jest",
                "start": "node server.js"
            },
            "devDependencies": {
                "webpack": "^5.0.0",
                "jest": "^27.0.0"
            }
        }
        
        package_file = temp_project_dir / "package.json"
        package_file.write_text(json.dumps(package_content, indent=2))
        
        # Detect workflows
        workflows = await workflow_detector.detect_workflows(str(temp_project_dir))
        
        # Verify NPM workflow detected
        assert len(workflows) > 0
        npm_workflow = next((w for w in workflows if w.type == WorkflowType.BUILD_SYSTEM), None)
        assert npm_workflow is not None
        assert npm_workflow.name == "NPM/Yarn Build"
        assert WorkflowTool.NPM in npm_workflow.tools
        assert "package.json" in npm_workflow.source_files
    
    @pytest.mark.asyncio
    async def test_detect_github_actions_workflow(self, workflow_detector, temp_project_dir):
        """Test detection of GitHub Actions workflow."""
        # Create GitHub Actions workflow file
        workflow_content = """name: CI
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '16'
    - name: Install dependencies
      run: npm install
    - name: Run tests
      run: npm test
"""
        
        github_dir = temp_project_dir / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        workflow_file = github_dir / "ci.yml"
        workflow_file.write_text(workflow_content)
        
        # Detect workflows
        workflows = await workflow_detector.detect_workflows(str(temp_project_dir))
        
        # Verify GitHub Actions workflow detected
        assert len(workflows) > 0
        ci_workflow = next((w for w in workflows if w.type == WorkflowType.CI_CD), None)
        assert ci_workflow is not None
        assert "GitHub Actions" in ci_workflow.name
        assert WorkflowTool.GITHUB_ACTIONS in ci_workflow.tools
    
    @pytest.mark.asyncio
    async def test_analyze_workflow_compatibility(self, workflow_detector):
        """Test workflow compatibility analysis."""
        # Create test workflow
        workflow = DetectedWorkflow(
            name="Test Workflow",
            type=WorkflowType.BUILD_SYSTEM,
            confidence=0.9,
            tools=[WorkflowTool.MAVEN]
        )
        
        # Analyze compatibility
        analysis = await workflow_detector.analyze_workflow_compatibility(workflow)
        
        # Verify analysis results
        assert "compatible" in analysis
        assert "compatibility_score" in analysis
        assert "integration_points" in analysis
        assert "potential_conflicts" in analysis
        assert "recommendations" in analysis
        assert "required_adaptations" in analysis
        
        # Maven should be compatible
        assert analysis["compatible"] is True
        assert analysis["compatibility_score"] > 0.5


class TestWorkflowAdapter:
    """Test workflow adaptation functionality."""
    
    @pytest.fixture
    def workflow_adapter(self):
        """Create workflow adapter instance."""
        return WorkflowAdapter()
    
    @pytest.fixture
    def test_workflow(self):
        """Create test workflow."""
        return DetectedWorkflow(
            name="Test Build Workflow",
            type=WorkflowType.BUILD_SYSTEM,
            confidence=0.9,
            tools=[WorkflowTool.MAVEN],
            config_files=["pom.xml"]
        )
    
    @pytest.mark.asyncio
    async def test_analyze_workflow_for_integration(self, workflow_adapter, test_workflow):
        """Test workflow integration analysis."""
        analysis = await workflow_adapter.analyze_workflow_for_integration(test_workflow)
        
        # Verify analysis structure
        assert "compatible" in analysis
        assert "integration_opportunities" in analysis
        assert "recommended_adaptations" in analysis
        
        # Build system should have integration opportunities
        opportunities = analysis["integration_opportunities"]
        assert len(opportunities) > 0
        
        # Should have build failure learning opportunity
        build_failure_opportunity = next(
            (op for op in opportunities if "build_failure" in op["type"]), None
        )
        assert build_failure_opportunity is not None
    
    @pytest.mark.asyncio
    async def test_create_workflow_adaptation(self, workflow_adapter, test_workflow):
        """Test creating workflow adaptation."""
        adaptation_config = {
            "adaptation_type": AdaptationType.HOOK_INJECTION.value,
            "integration_point": IntegrationPoint.ON_FAILURE.value,
            "name": "Build Failure Learning Hook",
            "description": "Provide learning content when builds fail",
            "priority": 90,
            "command": "python learning_hook.py",
            "conditions": ["build_failed"],
            "timeout": 30
        }
        
        adaptation = await workflow_adapter.create_workflow_adaptation(
            test_workflow, adaptation_config
        )
        
        # Verify adaptation created
        assert adaptation.workflow_id == test_workflow.id
        assert adaptation.adaptation_type == AdaptationType.HOOK_INJECTION
        assert adaptation.integration_point == IntegrationPoint.ON_FAILURE
        assert adaptation.name == "Build Failure Learning Hook"
        assert adaptation.priority == 90
    
    @pytest.mark.asyncio
    async def test_apply_adaptation_hook_injection(self, workflow_adapter, test_workflow):
        """Test applying hook injection adaptation."""
        adaptation_config = {
            "adaptation_type": AdaptationType.HOOK_INJECTION.value,
            "integration_point": IntegrationPoint.ON_FAILURE.value,
            "name": "Test Hook",
            "script_path": "/tmp/test_hook.py"
        }
        
        adaptation = await workflow_adapter.create_workflow_adaptation(
            test_workflow, adaptation_config
        )
        
        # Mock the hook injection implementation
        with patch.object(workflow_adapter, '_apply_hook_injection', return_value=True):
            result = await workflow_adapter.apply_adaptation(adaptation.id)
        
        # Verify result
        assert result.success is True
        assert result.adaptation_id == adaptation.id
        assert result.workflow_id == test_workflow.id


class TestWorkflowIntegrationService:
    """Test workflow integration service functionality."""
    
    @pytest.fixture
    def integration_service(self):
        """Create workflow integration service instance."""
        service = WorkflowIntegrationService()
        # Mock the detector and adapter to avoid initialization issues
        service.workflow_detector = MagicMock()
        service.workflow_adapter = MagicMock()
        return service
    
    @pytest.fixture
    def test_workflows(self):
        """Create test workflows."""
        return [
            DetectedWorkflow(
                name="Maven Build",
                type=WorkflowType.BUILD_SYSTEM,
                confidence=0.9,
                tools=[WorkflowTool.MAVEN]
            ),
            DetectedWorkflow(
                name="Jest Testing",
                type=WorkflowType.TESTING,
                confidence=0.8,
                tools=[WorkflowTool.JEST]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_detect_project_workflows(self, integration_service, test_workflows):
        """Test project workflow detection."""
        # Mock workflow detector
        integration_service.workflow_detector.detect_workflows = AsyncMock(
            return_value=test_workflows
        )
        
        workflows = await integration_service.detect_project_workflows("/test/project")
        
        # Verify workflows detected
        assert len(workflows) == 2
        assert workflows[0].name == "Maven Build"
        assert workflows[1].name == "Jest Testing"
    
    @pytest.mark.asyncio
    async def test_analyze_integration_opportunities(self, integration_service, test_workflows):
        """Test integration opportunity analysis."""
        # Mock workflow adapter
        integration_service.workflow_adapter.analyze_workflow_for_integration = AsyncMock(
            return_value={
                "compatible": True,
                "compatibility_score": 0.9,
                "integration_points": ["build_failure", "test_results"],
                "integration_opportunities": [
                    {
                        "type": "build_failure_learning",
                        "description": "Provide learning content when builds fail",
                        "integration_point": "on_failure",
                        "priority": 90
                    }
                ],
                "recommended_adaptations": [
                    {
                        "name": "build_failure_adaptation",
                        "description": "Build failure learning adaptation",
                        "priority": 90
                    }
                ]
            }
        )
        
        analysis = await integration_service.analyze_integration_opportunities(test_workflows)
        
        # Verify analysis results
        assert analysis["total_workflows"] == 2
        assert len(analysis["compatible_workflows"]) == 2
        assert len(analysis["integration_opportunities"]) == 2  # One per workflow
        assert analysis["estimated_effort"] in ["Low", "Medium", "High"]
        assert len(analysis["potential_benefits"]) > 0
    
    @pytest.mark.asyncio
    async def test_create_workflow_integration(self, integration_service, test_workflows):
        """Test creating workflow integration."""
        workflow = test_workflows[0]
        
        # Mock workflow adapter
        mock_adaptation = WorkflowAdaptation(
            workflow_id=workflow.id,
            adaptation_type=AdaptationType.HOOK_INJECTION,
            integration_point=IntegrationPoint.ON_FAILURE,
            name="Test Adaptation"
        )
        
        integration_service.workflow_adapter.create_workflow_adaptation = AsyncMock(
            return_value=mock_adaptation
        )
        
        integration_config = {
            "adaptations": [
                {
                    "adaptation_type": AdaptationType.HOOK_INJECTION.value,
                    "integration_point": IntegrationPoint.ON_FAILURE.value,
                    "name": "Test Adaptation"
                }
            ],
            "maintain_backward_compatibility": True,
            "enable_monitoring": True
        }
        
        integration_id = await integration_service.create_workflow_integration(
            workflow, integration_config
        )
        
        # Verify integration created
        assert integration_id is not None
        assert integration_id in integration_service._active_integrations
        
        integration = integration_service._active_integrations[integration_id]
        assert integration["workflow_id"] == workflow.id
        assert integration["status"] == "created"
        assert len(integration["adaptations"]) == 1
    
    @pytest.mark.asyncio
    async def test_apply_workflow_integration(self, integration_service, test_workflows):
        """Test applying workflow integration."""
        workflow = test_workflows[0]
        
        # Create integration first
        integration_service._active_integrations["test_integration"] = {
            "id": "test_integration",
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "workflow_type": workflow.type.value,
            "status": "created",
            "adaptations": ["adaptation_1"],
            "config": {"enable_monitoring": True}
        }
        
        # Mock successful adaptation application
        from src.ai_learning_accelerator.integrations.workflow_adapter import AdaptationResult
        mock_result = AdaptationResult(
            adaptation_id="adaptation_1",
            workflow_id=workflow.id,
            success=True
        )
        
        integration_service.workflow_adapter.apply_adaptation = AsyncMock(
            return_value=mock_result
        )
        
        success = await integration_service.apply_workflow_integration("test_integration")
        
        # Verify integration applied
        assert success is True
        integration = integration_service._active_integrations["test_integration"]
        assert integration["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_remove_workflow_integration(self, integration_service, test_workflows):
        """Test removing workflow integration."""
        workflow = test_workflows[0]
        
        # Create active integration
        integration_service._active_integrations["test_integration"] = {
            "id": "test_integration",
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "workflow_type": workflow.type.value,
            "status": "active",
            "adaptations": ["adaptation_1"],
            "config": {}
        }
        
        # Mock successful rollback
        integration_service.workflow_adapter.rollback_adaptation = AsyncMock(
            return_value=True
        )
        
        success = await integration_service.remove_workflow_integration("test_integration")
        
        # Verify integration removed
        assert success is True
        assert "test_integration" not in integration_service._active_integrations
    
    @pytest.mark.asyncio
    async def test_get_active_integrations(self, integration_service):
        """Test getting active integrations."""
        # Create test integrations
        integration_service._active_integrations.update({
            "active_1": {"status": "active", "name": "Active 1"},
            "inactive_1": {"status": "created", "name": "Inactive 1"},
            "active_2": {"status": "partially_applied", "name": "Active 2"},
            "failed_1": {"status": "failed", "name": "Failed 1"}
        })
        
        active_integrations = await integration_service.get_active_integrations()
        
        # Verify only active integrations returned
        assert len(active_integrations) == 2
        statuses = [integration["status"] for integration in active_integrations]
        assert "active" in statuses
        assert "partially_applied" in statuses
        assert "created" not in statuses
        assert "failed" not in statuses


@pytest.mark.asyncio
async def test_workflow_integration_end_to_end():
    """Test end-to-end workflow integration scenario."""
    # Create temporary project with Maven setup
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create pom.xml
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
</project>"""
        
        pom_file = project_path / "pom.xml"
        pom_file.write_text(pom_content)
        
        # Initialize services
        detector = WorkflowDetector()
        adapter = WorkflowAdapter()
        integration_service = WorkflowIntegrationService()
        integration_service.workflow_detector = detector
        integration_service.workflow_adapter = adapter
        
        # Detect workflows
        workflows = await detector.detect_workflows(str(project_path))
        assert len(workflows) > 0
        
        # Analyze integration opportunities
        analysis = await integration_service.analyze_integration_opportunities(workflows)
        assert analysis["total_workflows"] > 0
        
        # The workflow should be compatible
        if analysis["compatible_workflows"]:
            workflow = analysis["compatible_workflows"][0]["workflow"]
            
            # Create integration configuration
            integration_config = {
                "adaptations": [
                    {
                        "adaptation_type": AdaptationType.NOTIFICATION_INTEGRATION.value,
                        "integration_point": IntegrationPoint.ON_FAILURE.value,
                        "name": "Build Failure Notification",
                        "description": "Notify when build fails",
                        "priority": 90
                    }
                ],
                "maintain_backward_compatibility": True,
                "enable_monitoring": False  # Disable monitoring for test
            }
            
            # Create integration
            integration_id = await integration_service.create_workflow_integration(
                workflow, integration_config
            )
            assert integration_id is not None
            
            # Verify integration was created
            integrations = await integration_service.get_active_integrations()
            # Note: Integration won't be "active" until applied, so check all integrations
            assert integration_id in integration_service._active_integrations