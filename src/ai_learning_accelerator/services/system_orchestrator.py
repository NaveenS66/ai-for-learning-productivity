"""System orchestrator for coordinating all AI Learning Accelerator components."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..logging_config import get_logger
from ..utils.exceptions import AILearningAcceleratorException, ErrorSeverity
from ..utils.resilience import ResilientServiceMixin
from ..utils.monitoring import monitoring_system

from .learning_engine import LearningEngine
from .debug_assistant import DebugAssistant
from .context_analyzer import ContextAnalyzer
from .pattern_detector import PatternDetector
from .workflow_generator import WorkflowGenerator
from .content_adaptation import ContentAdaptationService
from .interaction_service import InteractionService
from .analytics_engine import AnalyticsEngine
from .engagement_tracker import EngagementTracker
from .privacy_service import PrivacyService
from .encryption_service import EncryptionService
from .content_lifecycle import ContentLifecycleService
from .feedback_integration import FeedbackIntegrationService
from .workflow_integration import workflow_integration_service

logger = get_logger(__name__)


class SystemOrchestrator(ResilientServiceMixin):
    """Orchestrates all system components and manages inter-service communication."""
    
    def __init__(self):
        super().__init__("system_orchestrator")
        
        # Core AI services
        self.learning_engine = LearningEngine()
        self.debug_assistant = DebugAssistant()
        self.context_analyzer = ContextAnalyzer()
        
        # Automation services
        self.pattern_detector = PatternDetector()
        self.workflow_generator = WorkflowGenerator()
        
        # Content and interaction services
        self.content_adaptation = ContentAdaptationService()
        self.interaction_service = InteractionService()
        
        # Analytics and tracking
        self.analytics_engine = AnalyticsEngine()
        self.engagement_tracker = EngagementTracker()
        
        # Security and privacy
        self.privacy_service = PrivacyService()
        self.encryption_service = EncryptionService()
        
        # Content management
        self.content_lifecycle = ContentLifecycleService()
        self.feedback_integration = FeedbackIntegrationService()
        
        # Workflow integration
        self.workflow_integration = workflow_integration_service
        
        # Service registry
        self.services = {
            "learning_engine": self.learning_engine,
            "debug_assistant": self.debug_assistant,
            "context_analyzer": self.context_analyzer,
            "pattern_detector": self.pattern_detector,
            "workflow_generator": self.workflow_generator,
            "content_adaptation": self.content_adaptation,
            "interaction_service": self.interaction_service,
            "analytics_engine": self.analytics_engine,
            "engagement_tracker": self.engagement_tracker,
            "privacy_service": self.privacy_service,
            "encryption_service": self.encryption_service,
            "content_lifecycle": self.content_lifecycle,
            "feedback_integration": self.feedback_integration,
            "workflow_integration": self.workflow_integration
        }
        
        # Service dependencies
        self.service_dependencies = {
            "learning_engine": ["privacy_service", "analytics_engine"],
            "debug_assistant": ["context_analyzer", "content_lifecycle"],
            "context_analyzer": ["privacy_service"],
            "pattern_detector": ["context_analyzer", "privacy_service"],
            "workflow_generator": ["pattern_detector", "privacy_service"],
            "content_adaptation": ["learning_engine", "privacy_service"],
            "interaction_service": ["content_adaptation", "privacy_service"],
            "analytics_engine": ["engagement_tracker", "privacy_service"],
            "engagement_tracker": ["privacy_service"],
            "content_lifecycle": ["feedback_integration"],
            "feedback_integration": ["analytics_engine"]
        }
        
        self._initialized = False
        self._startup_tasks = []
    
    async def initialize_system(self) -> Dict[str, Any]:
        """Initialize all system components in dependency order."""
        if self._initialized:
            return {"status": "already_initialized"}
        
        logger.info("Starting system initialization")
        
        try:
            # Initialize services in dependency order
            initialization_order = self._calculate_initialization_order()
            
            results = {}
            for service_name in initialization_order:
                service = self.services[service_name]
                
                logger.info(f"Initializing service: {service_name}")
                
                try:
                    # Initialize service if it has an initialize method
                    if hasattr(service, 'initialize'):
                        result = await service.initialize()
                        results[service_name] = {"status": "initialized", "result": result}
                    else:
                        results[service_name] = {"status": "no_initialization_required"}
                    
                    # Record successful initialization
                    await monitoring_system.record_ai_model_metrics(
                        model_name=service_name,
                        operation="initialization",
                        duration=0.1,
                        success=True
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to initialize service {service_name}: {e}")
                    results[service_name] = {"status": "failed", "error": str(e)}
                    
                    # Record failed initialization
                    await monitoring_system.record_ai_model_metrics(
                        model_name=service_name,
                        operation="initialization",
                        duration=0.1,
                        success=False
                    )
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._initialized = True
            
            logger.info("System initialization completed")
            
            return {
                "status": "initialized",
                "timestamp": datetime.utcnow().isoformat(),
                "services": results,
                "total_services": len(self.services),
                "successful_initializations": len([r for r in results.values() if r["status"] == "initialized"]),
                "failed_initializations": len([r for r in results.values() if r["status"] == "failed"])
            }
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise AILearningAcceleratorException(
                message=f"System initialization failed: {str(e)}",
                error_code="SYSTEM_INITIALIZATION_FAILED",
                category="system",
                severity=ErrorSeverity.CRITICAL
            )
    
    def _calculate_initialization_order(self) -> List[str]:
        """Calculate the order to initialize services based on dependencies."""
        # Topological sort of service dependencies
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                # Circular dependency detected - use default order
                return
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            
            # Visit dependencies first
            for dependency in self.service_dependencies.get(service_name, []):
                if dependency in self.services:
                    visit(dependency)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
        
        # Visit all services
        for service_name in self.services.keys():
            if service_name not in visited:
                visit(service_name)
        
        return order
    
    async def _start_background_tasks(self):
        """Start background tasks for system maintenance."""
        # Context monitoring task
        self._startup_tasks.append(
            asyncio.create_task(self._context_monitoring_loop())
        )
        
        # Analytics aggregation task
        self._startup_tasks.append(
            asyncio.create_task(self._analytics_aggregation_loop())
        )
        
        # Content lifecycle maintenance task
        self._startup_tasks.append(
            asyncio.create_task(self._content_maintenance_loop())
        )
        
        logger.info(f"Started {len(self._startup_tasks)} background tasks")
    
    async def _context_monitoring_loop(self):
        """Background task for continuous context monitoring."""
        while True:
            try:
                # This would normally monitor active user sessions
                # For now, just log that monitoring is active
                logger.debug("Context monitoring loop active")
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Context monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _analytics_aggregation_loop(self):
        """Background task for analytics data aggregation."""
        while True:
            try:
                # Aggregate analytics data periodically
                logger.debug("Analytics aggregation loop active")
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
            except Exception as e:
                logger.error(f"Analytics aggregation loop error: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _content_maintenance_loop(self):
        """Background task for content lifecycle maintenance."""
        while True:
            try:
                # Check for outdated content and maintenance needs
                logger.debug("Content maintenance loop active")
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Content maintenance loop error: {e}")
                await asyncio.sleep(3600)  # Wait same time on error
    
    async def process_learning_request(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete learning request through the system."""
        logger.info(f"Processing learning request for user {user_id}")
        
        try:
            # Step 1: Check privacy settings
            privacy_check = await self.privacy_service.check_data_access(
                user_id=user_id,
                data_type="learning_request",
                operation="process"
            )
            
            if not privacy_check.get("allowed", False):
                raise AILearningAcceleratorException(
                    message="Privacy settings do not allow this operation",
                    error_code="PRIVACY_DENIED",
                    category="privacy",
                    severity=ErrorSeverity.HIGH
                )
            
            # Step 2: Analyze context
            context_analysis = await self.context_analyzer.analyze_workspace(
                workspace_data=request_data.get("workspace", {})
            )
            
            # Step 3: Generate learning recommendations
            learning_recommendations = await self.learning_engine.generate_recommendations(
                user_id=user_id,
                context=context_analysis,
                preferences=request_data.get("preferences", {})
            )
            
            # Step 4: Adapt content for user
            adapted_content = await self.content_adaptation.adapt_content(
                content=learning_recommendations.get("content", {}),
                user_preferences=request_data.get("preferences", {}),
                context=context_analysis
            )
            
            # Step 5: Track engagement
            await self.engagement_tracker.track_interaction(
                user_id=user_id,
                interaction_type="learning_request",
                content_id=adapted_content.get("id"),
                context=context_analysis
            )
            
            # Step 6: Update analytics
            await self.analytics_engine.record_learning_activity(
                user_id=user_id,
                activity_type="learning_request",
                content=adapted_content,
                context=context_analysis
            )
            
            return {
                "status": "success",
                "recommendations": learning_recommendations,
                "adapted_content": adapted_content,
                "context": context_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Learning request processing failed: {e}")
            
            # Record failure in analytics
            await self.analytics_engine.record_error(
                user_id=user_id,
                error_type="learning_request_failed",
                error_details=str(e)
            )
            
            raise
    
    async def process_debugging_request(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete debugging request through the system."""
        logger.info(f"Processing debugging request for user {user_id}")
        
        try:
            # Step 1: Privacy check
            privacy_check = await self.privacy_service.check_data_access(
                user_id=user_id,
                data_type="code_analysis",
                operation="debug"
            )
            
            if not privacy_check.get("allowed", False):
                raise AILearningAcceleratorException(
                    message="Privacy settings do not allow code analysis",
                    error_code="PRIVACY_DENIED",
                    category="privacy",
                    severity=ErrorSeverity.HIGH
                )
            
            # Step 2: Analyze code context
            code_context = await self.context_analyzer.analyze_code_context(
                code_data=request_data.get("code", {}),
                error_data=request_data.get("error", {})
            )
            
            # Step 3: Debug analysis
            debug_analysis = await self.debug_assistant.analyze_error(
                error_context=request_data.get("error", {}),
                code_context=code_context
            )
            
            # Step 4: Generate solutions
            solutions = await self.debug_assistant.generate_solutions(
                analysis=debug_analysis,
                user_skill_level=request_data.get("skill_level", "intermediate")
            )
            
            # Step 5: Adapt solutions for user
            adapted_solutions = await self.content_adaptation.adapt_debugging_content(
                solutions=solutions,
                user_preferences=request_data.get("preferences", {}),
                skill_level=request_data.get("skill_level", "intermediate")
            )
            
            # Step 6: Track debugging session
            await self.engagement_tracker.track_interaction(
                user_id=user_id,
                interaction_type="debugging_session",
                content_id=debug_analysis.get("id"),
                context=code_context
            )
            
            # Step 7: Update analytics
            await self.analytics_engine.record_debugging_activity(
                user_id=user_id,
                error_type=debug_analysis.get("error_type"),
                solutions_provided=len(adapted_solutions),
                context=code_context
            )
            
            return {
                "status": "success",
                "analysis": debug_analysis,
                "solutions": adapted_solutions,
                "context": code_context,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Debugging request processing failed: {e}")
            
            # Record failure
            await self.analytics_engine.record_error(
                user_id=user_id,
                error_type="debugging_request_failed",
                error_details=str(e)
            )
            
            raise
    
    async def process_automation_request(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process automation pattern detection and workflow generation."""
        logger.info(f"Processing automation request for user {user_id}")
        
        try:
            # Step 1: Privacy check
            privacy_check = await self.privacy_service.check_data_access(
                user_id=user_id,
                data_type="user_actions",
                operation="analyze"
            )
            
            if not privacy_check.get("allowed", False):
                raise AILearningAcceleratorException(
                    message="Privacy settings do not allow action analysis",
                    error_code="PRIVACY_DENIED",
                    category="privacy",
                    severity=ErrorSeverity.HIGH
                )
            
            # Step 2: Detect patterns
            patterns = await self.pattern_detector.detect_patterns(
                user_actions=request_data.get("actions", []),
                context=request_data.get("context", {})
            )
            
            # Step 3: Generate automation workflows
            workflows = []
            for pattern in patterns:
                workflow = await self.workflow_generator.generate_workflow(
                    pattern=pattern,
                    user_preferences=request_data.get("preferences", {})
                )
                workflows.append(workflow)
            
            # Step 4: Integrate with existing workflows
            integrated_workflows = await self.workflow_integration.integrate_workflows(
                new_workflows=workflows,
                existing_context=request_data.get("existing_workflows", {})
            )
            
            # Step 5: Track automation activity
            await self.engagement_tracker.track_interaction(
                user_id=user_id,
                interaction_type="automation_request",
                content_id=f"patterns_{len(patterns)}",
                context={"patterns_found": len(patterns)}
            )
            
            # Step 6: Update analytics
            await self.analytics_engine.record_automation_activity(
                user_id=user_id,
                patterns_detected=len(patterns),
                workflows_generated=len(workflows),
                integration_success=len(integrated_workflows)
            )
            
            return {
                "status": "success",
                "patterns": patterns,
                "workflows": workflows,
                "integrated_workflows": integrated_workflows,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Automation request processing failed: {e}")
            
            # Record failure
            await self.analytics_engine.record_error(
                user_id=user_id,
                error_type="automation_request_failed",
                error_details=str(e)
            )
            
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        service_statuses = {}
        
        for service_name, service in self.services.items():
            try:
                if hasattr(service, 'get_status'):
                    status = await service.get_status()
                else:
                    status = {"status": "running", "initialized": True}
                
                service_statuses[service_name] = status
                
            except Exception as e:
                service_statuses[service_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Get monitoring system status
        monitoring_status = monitoring_system.get_system_status()
        
        # Calculate overall health
        healthy_services = len([s for s in service_statuses.values() if s.get("status") == "running"])
        total_services = len(service_statuses)
        health_ratio = healthy_services / total_services if total_services > 0 else 0
        
        overall_status = "healthy"
        if health_ratio < 0.5:
            overall_status = "critical"
        elif health_ratio < 0.8:
            overall_status = "degraded"
        elif health_ratio < 1.0:
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "health_ratio": health_ratio,
            "initialized": self._initialized,
            "services": service_statuses,
            "monitoring": monitoring_status,
            "background_tasks": len(self._startup_tasks),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def shutdown_system(self) -> Dict[str, Any]:
        """Gracefully shutdown all system components."""
        logger.info("Starting system shutdown")
        
        try:
            # Cancel background tasks
            for task in self._startup_tasks:
                task.cancel()
            
            # Wait for tasks to complete cancellation
            if self._startup_tasks:
                await asyncio.gather(*self._startup_tasks, return_exceptions=True)
            
            # Shutdown services in reverse dependency order
            shutdown_order = list(reversed(self._calculate_initialization_order()))
            
            results = {}
            for service_name in shutdown_order:
                service = self.services[service_name]
                
                try:
                    if hasattr(service, 'shutdown'):
                        await service.shutdown()
                        results[service_name] = {"status": "shutdown"}
                    else:
                        results[service_name] = {"status": "no_shutdown_required"}
                        
                except Exception as e:
                    logger.error(f"Failed to shutdown service {service_name}: {e}")
                    results[service_name] = {"status": "failed", "error": str(e)}
            
            self._initialized = False
            
            logger.info("System shutdown completed")
            
            return {
                "status": "shutdown",
                "timestamp": datetime.utcnow().isoformat(),
                "services": results
            }
            
        except Exception as e:
            logger.error(f"System shutdown failed: {e}")
            raise AILearningAcceleratorException(
                message=f"System shutdown failed: {str(e)}",
                error_code="SYSTEM_SHUTDOWN_FAILED",
                category="system",
                severity=ErrorSeverity.HIGH
            )


# Global system orchestrator instance
system_orchestrator = SystemOrchestrator()