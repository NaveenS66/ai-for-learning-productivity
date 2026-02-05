"""Context analyzer service for real-time workspace monitoring and recommendations."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..models.context import (
    WorkspaceSession, WorkspaceEvent, TechnologyStack, KnowledgeGap,
    ContextRecommendation, WorkspacePattern, ContextAnalysisResult,
    LearningOpportunity, WorkspaceEventType, TechnologyType,
    KnowledgeGapSeverity, RecommendationType, NotificationPriority
)
from ..models.user import User, SkillLevel
from ..models.content import LearningContent
from ..database import get_async_db

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """Service for analyzing workspace context and generating recommendations."""
    
    def __init__(self):
        self.file_watchers: Dict[str, Any] = {}
        self.active_sessions: Dict[UUID, WorkspaceSession] = {}
        self.technology_patterns = self._load_technology_patterns()
        self.knowledge_gap_patterns = self._load_knowledge_gap_patterns()
    
    def _load_technology_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load technology detection patterns."""
        return {
            "python": {
                "files": [".py", "requirements.txt", "pyproject.toml", "setup.py"],
                "frameworks": {
                    "django": ["manage.py", "settings.py", "urls.py"],
                    "flask": ["app.py", "flask", "werkzeug"],
                    "fastapi": ["fastapi", "uvicorn", "pydantic"],
                    "pytest": ["pytest", "conftest.py", "test_*.py"]
                }
            },
            "javascript": {
                "files": [".js", ".jsx", "package.json", "node_modules"],
                "frameworks": {
                    "react": ["react", "jsx", "components"],
                    "vue": ["vue", ".vue", "vuex"],
                    "angular": ["angular", "@angular", "ng"],
                    "express": ["express", "app.js", "server.js"]
                }
            },
            "typescript": {
                "files": [".ts", ".tsx", "tsconfig.json"],
                "frameworks": {
                    "react": ["react", "tsx", "@types/react"],
                    "angular": ["@angular", "ng", "typescript"],
                    "nest": ["@nestjs", "nest", "decorators"]
                }
            }
        }
    
    def _load_knowledge_gap_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load knowledge gap detection patterns."""
        return {
            "error_patterns": {
                "import_error": {
                    "keywords": ["ImportError", "ModuleNotFoundError", "cannot import"],
                    "gap_type": "dependency_management",
                    "severity": KnowledgeGapSeverity.MEDIUM
                },
                "syntax_error": {
                    "keywords": ["SyntaxError", "invalid syntax", "unexpected token"],
                    "gap_type": "language_fundamentals",
                    "severity": KnowledgeGapSeverity.HIGH
                },
                "type_error": {
                    "keywords": ["TypeError", "AttributeError", "has no attribute"],
                    "gap_type": "type_system",
                    "severity": KnowledgeGapSeverity.MEDIUM
                }
            },
            "file_patterns": {
                "no_tests": {
                    "condition": "has_source_files_but_no_tests",
                    "gap_type": "testing_practices",
                    "severity": KnowledgeGapSeverity.MEDIUM
                },
                "no_documentation": {
                    "condition": "has_complex_code_but_no_docs",
                    "gap_type": "documentation_practices",
                    "severity": KnowledgeGapSeverity.LOW
                },
                "large_functions": {
                    "condition": "functions_over_50_lines",
                    "gap_type": "code_organization",
                    "severity": KnowledgeGapSeverity.LOW
                }
            }
        }
    
    async def start_workspace_monitoring(
        self, 
        user_id: UUID, 
        workspace_path: str,
        session_name: Optional[str] = None
    ) -> WorkspaceSession:
        """Start monitoring a workspace session."""
        async with get_async_db() as session:
            # Create new workspace session
            workspace_session = WorkspaceSession(
                user_id=user_id,
                session_name=session_name,
                workspace_path=workspace_path,
                started_at=datetime.utcnow(),
                is_active=True
            )
            
            session.add(workspace_session)
            await session.commit()
            await session.refresh(workspace_session)
            
            # Store in active sessions
            self.active_sessions[workspace_session.id] = workspace_session
            
            # Start file system monitoring
            await self._start_file_monitoring(workspace_session)
            
            # Perform initial context analysis
            await self._perform_initial_analysis(workspace_session)
            
            logger.info(f"Started workspace monitoring for user {user_id} at {workspace_path}")
            return workspace_session
    
    async def _start_file_monitoring(self, workspace_session: WorkspaceSession):
        """Start file system monitoring for the workspace."""
        # In a real implementation, this would set up file watchers
        # For now, we'll simulate with periodic checks
        workspace_path = workspace_session.workspace_path
        
        # Detect initial technology stack
        technologies = await self._detect_technologies(workspace_path)
        
        async with get_async_db() as session:
            for tech_info in technologies:
                tech_stack = TechnologyStack(
                    user_id=workspace_session.user_id,
                    workspace_path=workspace_path,
                    technology_name=tech_info["name"],
                    technology_type=tech_info["type"],
                    version=tech_info.get("version"),
                    detection_confidence=tech_info["confidence"],
                    detection_method="file_analysis",
                    detection_evidence=tech_info["evidence"],
                    is_primary=tech_info.get("is_primary", False)
                )
                session.add(tech_stack)
            
            await session.commit()
    
    async def _detect_technologies(self, workspace_path: str) -> List[Dict[str, Any]]:
        """Detect technologies used in the workspace."""
        technologies = []
        
        # Simulate technology detection based on file patterns
        # In a real implementation, this would scan the actual filesystem
        
        # Example detection for Python projects
        technologies.append({
            "name": "Python",
            "type": TechnologyType.PROGRAMMING_LANGUAGE,
            "version": "3.9+",
            "confidence": 0.95,
            "evidence": ["*.py files", "requirements.txt"],
            "is_primary": True
        })
        
        technologies.append({
            "name": "FastAPI",
            "type": TechnologyType.FRAMEWORK,
            "version": "0.68+",
            "confidence": 0.90,
            "evidence": ["fastapi imports", "main.py"],
            "is_primary": True
        })
        
        technologies.append({
            "name": "SQLAlchemy",
            "type": TechnologyType.LIBRARY,
            "version": "1.4+",
            "confidence": 0.85,
            "evidence": ["sqlalchemy imports", "models directory"],
            "is_primary": False
        })
        
        return technologies
    
    async def _perform_initial_analysis(self, workspace_session: WorkspaceSession):
        """Perform initial context analysis for the workspace."""
        analysis_start = datetime.utcnow()
        
        # Detect knowledge gaps
        knowledge_gaps = await self._detect_knowledge_gaps(workspace_session)
        
        # Generate initial recommendations
        recommendations = await self._generate_recommendations(workspace_session, knowledge_gaps)
        
        # Store analysis results
        analysis_duration = (datetime.utcnow() - analysis_start).total_seconds() * 1000
        
        async with get_async_db() as session:
            analysis_result = ContextAnalysisResult(
                user_id=workspace_session.user_id,
                session_id=workspace_session.id,
                analysis_type="initial_workspace_analysis",
                analysis_trigger="workspace_start",
                technologies_detected=[],  # Will be populated from TechnologyStack
                knowledge_gaps_found=[gap.id for gap in knowledge_gaps],
                recommendations_generated=[rec.id for rec in recommendations],
                analysis_duration_ms=int(analysis_duration),
                confidence_score=0.8,
                complexity_score=0.6
            )
            
            session.add(analysis_result)
            await session.commit()
    
    async def _detect_knowledge_gaps(self, workspace_session: WorkspaceSession) -> List[KnowledgeGap]:
        """Detect knowledge gaps based on workspace analysis."""
        gaps = []
        
        async with get_async_db() as session:
            # Get user's skill profile
            user_query = select(User).where(User.id == workspace_session.user_id)
            user_result = await session.execute(user_query)
            user = user_result.scalar_one_or_none()
            
            if not user:
                return gaps
            
            # Simulate gap detection based on technology stack and user profile
            # In a real implementation, this would analyze actual code and patterns
            
            # Example: Detect testing gap
            testing_gap = KnowledgeGap(
                user_id=workspace_session.user_id,
                gap_title="Testing Best Practices",
                gap_description="Limited test coverage detected in the project. Consider implementing comprehensive unit and integration tests.",
                gap_category="testing_practices",
                severity=KnowledgeGapSeverity.MEDIUM,
                confidence_score=0.75,
                impact_score=0.80,
                detected_from_events=["project_analysis"],
                related_files=["src/", "tests/"],
                suggested_resources=[
                    {
                        "title": "Python Testing with pytest",
                        "type": "tutorial",
                        "url": "https://docs.pytest.org/",
                        "difficulty": "intermediate"
                    }
                ],
                learning_priority=0.7
            )
            
            session.add(testing_gap)
            gaps.append(testing_gap)
            
            # Example: Detect documentation gap
            docs_gap = KnowledgeGap(
                user_id=workspace_session.user_id,
                gap_title="API Documentation",
                gap_description="API endpoints lack comprehensive documentation. Consider adding OpenAPI/Swagger documentation.",
                gap_category="documentation_practices",
                severity=KnowledgeGapSeverity.LOW,
                confidence_score=0.65,
                impact_score=0.60,
                detected_from_events=["api_analysis"],
                related_files=["src/ai_learning_accelerator/api/"],
                suggested_resources=[
                    {
                        "title": "FastAPI Documentation Guide",
                        "type": "documentation",
                        "url": "https://fastapi.tiangolo.com/tutorial/",
                        "difficulty": "beginner"
                    }
                ],
                learning_priority=0.5
            )
            
            session.add(docs_gap)
            gaps.append(docs_gap)
            
            await session.commit()
            
        return gaps
    
    async def _generate_recommendations(
        self, 
        workspace_session: WorkspaceSession, 
        knowledge_gaps: List[KnowledgeGap]
    ) -> List[ContextRecommendation]:
        """Generate context-aware recommendations."""
        recommendations = []
        
        async with get_async_db() as session:
            for gap in knowledge_gaps:
                # Generate recommendation based on knowledge gap
                recommendation = ContextRecommendation(
                    user_id=workspace_session.user_id,
                    session_id=workspace_session.id,
                    knowledge_gap_id=gap.id,
                    title=f"Improve {gap.gap_category.replace('_', ' ').title()}",
                    description=f"Based on your current project, we recommend focusing on {gap.gap_title.lower()}.",
                    recommendation_type=RecommendationType.LEARNING_RESOURCE,
                    content={
                        "gap_details": gap.gap_description,
                        "resources": gap.suggested_resources,
                        "estimated_time": "30-60 minutes",
                        "difficulty": "intermediate"
                    },
                    action_items=[
                        f"Review {gap.gap_title} best practices",
                        "Apply recommendations to current project",
                        "Set up automated checks if applicable"
                    ],
                    resources=gap.suggested_resources,
                    relevance_score=gap.learning_priority,
                    urgency_score=self._calculate_urgency_score(gap.severity),
                    impact_score=gap.impact_score,
                    context_data={
                        "workspace_path": workspace_session.workspace_path,
                        "detected_technologies": [],  # Will be populated
                        "analysis_timestamp": datetime.utcnow().isoformat()
                    },
                    triggering_events=["workspace_analysis", "knowledge_gap_detection"],
                    priority=self._map_severity_to_priority(gap.severity)
                )
                
                session.add(recommendation)
                recommendations.append(recommendation)
            
            # Generate proactive recommendations
            proactive_recs = await self._generate_proactive_recommendations(workspace_session)
            recommendations.extend(proactive_recs)
            
            await session.commit()
            
        return recommendations
    
    async def _generate_proactive_recommendations(
        self, 
        workspace_session: WorkspaceSession
    ) -> List[ContextRecommendation]:
        """Generate proactive recommendations based on patterns and best practices."""
        recommendations = []
        
        async with get_async_db() as session:
            # Example: Recommend code review practices
            code_review_rec = ContextRecommendation(
                user_id=workspace_session.user_id,
                session_id=workspace_session.id,
                title="Code Review Best Practices",
                description="Consider implementing systematic code review practices to improve code quality and knowledge sharing.",
                recommendation_type=RecommendationType.BEST_PRACTICE,
                content={
                    "practices": [
                        "Use pull request templates",
                        "Implement automated code quality checks",
                        "Schedule regular code review sessions"
                    ],
                    "tools": ["GitHub PR templates", "pre-commit hooks", "SonarQube"],
                    "benefits": [
                        "Improved code quality",
                        "Knowledge sharing",
                        "Bug prevention"
                    ]
                },
                action_items=[
                    "Set up pull request templates",
                    "Configure automated linting",
                    "Establish review guidelines"
                ],
                relevance_score=0.7,
                urgency_score=0.4,
                impact_score=0.8,
                context_data={
                    "workspace_path": workspace_session.workspace_path,
                    "recommendation_type": "proactive_best_practice"
                },
                triggering_events=["proactive_analysis"],
                priority=NotificationPriority.MEDIUM
            )
            
            session.add(code_review_rec)
            recommendations.append(code_review_rec)
            
            # Example: Recommend security practices
            security_rec = ContextRecommendation(
                user_id=workspace_session.user_id,
                session_id=workspace_session.id,
                title="Security Best Practices",
                description="Enhance your application security with these recommended practices and tools.",
                recommendation_type=RecommendationType.TOOL_SUGGESTION,
                content={
                    "security_practices": [
                        "Input validation and sanitization",
                        "Secure authentication implementation",
                        "Regular dependency updates"
                    ],
                    "tools": ["bandit", "safety", "OWASP ZAP"],
                    "resources": [
                        {
                            "title": "OWASP Top 10",
                            "url": "https://owasp.org/www-project-top-ten/",
                            "type": "security_guide"
                        }
                    ]
                },
                action_items=[
                    "Run security scan with bandit",
                    "Review authentication implementation",
                    "Set up dependency vulnerability monitoring"
                ],
                relevance_score=0.8,
                urgency_score=0.6,
                impact_score=0.9,
                context_data={
                    "workspace_path": workspace_session.workspace_path,
                    "recommendation_type": "proactive_security"
                },
                triggering_events=["proactive_analysis"],
                priority=NotificationPriority.HIGH
            )
            
            session.add(security_rec)
            recommendations.append(security_rec)
            
        return recommendations
    
    def _calculate_urgency_score(self, severity: KnowledgeGapSeverity) -> float:
        """Calculate urgency score based on knowledge gap severity."""
        severity_mapping = {
            KnowledgeGapSeverity.LOW: 0.2,
            KnowledgeGapSeverity.MEDIUM: 0.5,
            KnowledgeGapSeverity.HIGH: 0.8,
            KnowledgeGapSeverity.CRITICAL: 1.0
        }
        return severity_mapping.get(severity, 0.5)
    
    def _map_severity_to_priority(self, severity: KnowledgeGapSeverity) -> NotificationPriority:
        """Map knowledge gap severity to notification priority."""
        severity_mapping = {
            KnowledgeGapSeverity.LOW: NotificationPriority.LOW,
            KnowledgeGapSeverity.MEDIUM: NotificationPriority.MEDIUM,
            KnowledgeGapSeverity.HIGH: NotificationPriority.HIGH,
            KnowledgeGapSeverity.CRITICAL: NotificationPriority.URGENT
        }
        return severity_mapping.get(severity, NotificationPriority.MEDIUM)
    
    async def process_workspace_event(
        self, 
        session_id: UUID, 
        event_type: WorkspaceEventType,
        event_data: Dict[str, Any]
    ) -> Optional[List[ContextRecommendation]]:
        """Process a workspace event and generate recommendations if needed."""
        async with get_async_db() as session:
            # Get workspace session
            session_query = select(WorkspaceSession).where(WorkspaceSession.id == session_id)
            session_result = await session.execute(session_query)
            workspace_session = session_result.scalar_one_or_none()
            
            if not workspace_session:
                logger.warning(f"Workspace session {session_id} not found")
                return None
            
            # Create workspace event
            workspace_event = WorkspaceEvent(
                session_id=session_id,
                user_id=workspace_session.user_id,
                event_type=event_type,
                event_description=event_data.get("description", ""),
                file_path=event_data.get("file_path"),
                file_extension=event_data.get("file_extension"),
                directory_path=event_data.get("directory_path"),
                event_data=event_data,
                event_timestamp=datetime.utcnow()
            )
            
            session.add(workspace_event)
            
            # Update session activity
            workspace_session.last_activity_at = datetime.utcnow()
            workspace_session.is_idle = False
            workspace_session.idle_since = None
            
            # Check if event should trigger analysis
            should_analyze = self._should_trigger_analysis(event_type, event_data)
            
            recommendations = []
            if should_analyze:
                workspace_event.triggered_analysis = True
                
                # Perform contextual analysis
                new_recommendations = await self._analyze_event_context(workspace_event)
                recommendations.extend(new_recommendations)
                
                workspace_event.generated_recommendations = len(new_recommendations)
            
            await session.commit()
            
            return recommendations if recommendations else None
    
    def _should_trigger_analysis(self, event_type: WorkspaceEventType, event_data: Dict[str, Any]) -> bool:
        """Determine if an event should trigger context analysis."""
        # Trigger analysis for significant events
        trigger_events = {
            WorkspaceEventType.FILE_CREATED,
            WorkspaceEventType.FILE_MODIFIED,
            WorkspaceEventType.BUILD_FAILED,
            WorkspaceEventType.TEST_FAILED,
            WorkspaceEventType.DEPENDENCY_ADDED,
            WorkspaceEventType.CONFIGURATION_CHANGED
        }
        
        return event_type in trigger_events
    
    async def _analyze_event_context(self, workspace_event: WorkspaceEvent) -> List[ContextRecommendation]:
        """Analyze event context and generate relevant recommendations."""
        recommendations = []
        
        # Example: File modification analysis
        if workspace_event.event_type == WorkspaceEventType.FILE_MODIFIED:
            if workspace_event.file_path and workspace_event.file_path.endswith('.py'):
                # Python file modified - check for common issues
                rec = await self._generate_python_file_recommendation(workspace_event)
                if rec:
                    recommendations.append(rec)
        
        # Example: Build failure analysis
        elif workspace_event.event_type == WorkspaceEventType.BUILD_FAILED:
            rec = await self._generate_build_failure_recommendation(workspace_event)
            if rec:
                recommendations.append(rec)
        
        return recommendations
    
    async def _generate_python_file_recommendation(
        self, 
        workspace_event: WorkspaceEvent
    ) -> Optional[ContextRecommendation]:
        """Generate recommendation for Python file modifications."""
        async with get_async_db() as session:
            recommendation = ContextRecommendation(
                user_id=workspace_event.user_id,
                session_id=workspace_event.session_id,
                title="Python Code Quality Check",
                description="Consider running code quality checks on your modified Python file.",
                recommendation_type=RecommendationType.CODE_IMPROVEMENT,
                content={
                    "file_path": workspace_event.file_path,
                    "suggestions": [
                        "Run pylint or flake8 for style checking",
                        "Add type hints for better code clarity",
                        "Consider adding docstrings for functions"
                    ],
                    "tools": ["pylint", "black", "mypy"]
                },
                action_items=[
                    f"Run linter on {workspace_event.file_path}",
                    "Fix any style issues found",
                    "Add type annotations if missing"
                ],
                relevance_score=0.6,
                urgency_score=0.3,
                impact_score=0.5,
                context_data={
                    "file_path": workspace_event.file_path,
                    "event_type": workspace_event.event_type.value,
                    "timestamp": workspace_event.event_timestamp.isoformat()
                },
                triggering_events=[workspace_event.event_type.value],
                priority=NotificationPriority.LOW
            )
            
            session.add(recommendation)
            await session.commit()
            
            return recommendation
    
    async def _generate_build_failure_recommendation(
        self, 
        workspace_event: WorkspaceEvent
    ) -> Optional[ContextRecommendation]:
        """Generate recommendation for build failures."""
        async with get_async_db() as session:
            recommendation = ContextRecommendation(
                user_id=workspace_event.user_id,
                session_id=workspace_event.session_id,
                title="Build Failure Analysis",
                description="Your build failed. Here are some common troubleshooting steps.",
                recommendation_type=RecommendationType.WARNING,
                content={
                    "failure_context": workspace_event.event_data,
                    "troubleshooting_steps": [
                        "Check for syntax errors in recent changes",
                        "Verify all dependencies are installed",
                        "Review build logs for specific error messages",
                        "Ensure environment variables are set correctly"
                    ],
                    "common_causes": [
                        "Missing dependencies",
                        "Syntax errors",
                        "Configuration issues",
                        "Environment problems"
                    ]
                },
                action_items=[
                    "Review build logs carefully",
                    "Check recent file changes",
                    "Verify dependency installation",
                    "Test in clean environment if needed"
                ],
                relevance_score=0.9,
                urgency_score=0.8,
                impact_score=0.7,
                context_data={
                    "event_type": workspace_event.event_type.value,
                    "failure_data": workspace_event.event_data,
                    "timestamp": workspace_event.event_timestamp.isoformat()
                },
                triggering_events=[workspace_event.event_type.value],
                priority=NotificationPriority.HIGH
            )
            
            session.add(recommendation)
            await session.commit()
            
            return recommendation
    
    async def get_active_recommendations(
        self, 
        user_id: UUID, 
        limit: int = 10,
        priority_filter: Optional[NotificationPriority] = None
    ) -> List[ContextRecommendation]:
        """Get active recommendations for a user."""
        async with get_async_db() as session:
            query = select(ContextRecommendation).where(
                and_(
                    ContextRecommendation.user_id == user_id,
                    ContextRecommendation.is_dismissed == False,
                    or_(
                        ContextRecommendation.expires_at.is_(None),
                        ContextRecommendation.expires_at > datetime.utcnow()
                    )
                )
            ).order_by(
                desc(ContextRecommendation.urgency_score),
                desc(ContextRecommendation.relevance_score)
            ).limit(limit)
            
            if priority_filter:
                query = query.where(ContextRecommendation.priority == priority_filter)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def mark_recommendation_delivered(
        self, 
        recommendation_id: UUID, 
        delivery_method: str
    ) -> bool:
        """Mark a recommendation as delivered."""
        async with get_async_db() as session:
            query = select(ContextRecommendation).where(ContextRecommendation.id == recommendation_id)
            result = await session.execute(query)
            recommendation = result.scalar_one_or_none()
            
            if recommendation:
                recommendation.is_delivered = True
                recommendation.delivery_method = delivery_method
                recommendation.delivered_at = datetime.utcnow()
                await session.commit()
                return True
            
            return False
    
    async def update_recommendation_interaction(
        self, 
        recommendation_id: UUID,
        is_viewed: Optional[bool] = None,
        is_accepted: Optional[bool] = None,
        is_dismissed: Optional[bool] = None,
        user_feedback: Optional[str] = None
    ) -> bool:
        """Update user interaction with a recommendation."""
        async with get_async_db() as session:
            query = select(ContextRecommendation).where(ContextRecommendation.id == recommendation_id)
            result = await session.execute(query)
            recommendation = result.scalar_one_or_none()
            
            if recommendation:
                if is_viewed is not None:
                    recommendation.is_viewed = is_viewed
                if is_accepted is not None:
                    recommendation.is_accepted = is_accepted
                if is_dismissed is not None:
                    recommendation.is_dismissed = is_dismissed
                if user_feedback is not None:
                    recommendation.user_feedback = user_feedback
                
                await session.commit()
                return True
            
            return False
    
    async def generate_learning_opportunities(
        self, 
        user_id: UUID, 
        workspace_session_id: Optional[UUID] = None
    ) -> List[LearningOpportunity]:
        """Generate non-intrusive learning opportunities based on context."""
        opportunities = []
        
        async with get_async_db() as session:
            # Get user's knowledge gaps
            gaps_query = select(KnowledgeGap).where(
                and_(
                    KnowledgeGap.user_id == user_id,
                    KnowledgeGap.is_addressed == False
                )
            ).order_by(desc(KnowledgeGap.learning_priority))
            
            gaps_result = await session.execute(gaps_query)
            knowledge_gaps = gaps_result.scalars().all()
            
            for gap in knowledge_gaps[:5]:  # Limit to top 5 gaps
                opportunity = LearningOpportunity(
                    user_id=user_id,
                    knowledge_gap_id=gap.id,
                    opportunity_title=f"Learn {gap.gap_title}",
                    opportunity_description=f"Improve your skills in {gap.gap_category.replace('_', ' ')} based on your current project needs.",
                    opportunity_category=gap.gap_category,
                    relevance_score=gap.learning_priority,
                    difficulty_level=self._estimate_difficulty_level(gap),
                    estimated_time_minutes=self._estimate_learning_time(gap),
                    detected_from_context={
                        "gap_severity": gap.severity.value,
                        "confidence": gap.confidence_score,
                        "impact": gap.impact_score
                    },
                    related_technologies=[],  # Will be populated from technology stack
                    learning_resources=gap.suggested_resources,
                    optimal_timing="during_break",
                    priority_score=gap.learning_priority,
                    is_intrusive=False  # Non-intrusive by default
                )
                
                session.add(opportunity)
                opportunities.append(opportunity)
            
            await session.commit()
            
        return opportunities
    
    def _estimate_difficulty_level(self, gap: KnowledgeGap) -> SkillLevel:
        """Estimate difficulty level for a knowledge gap."""
        if gap.severity == KnowledgeGapSeverity.LOW:
            return SkillLevel.BEGINNER
        elif gap.severity == KnowledgeGapSeverity.MEDIUM:
            return SkillLevel.INTERMEDIATE
        else:
            return SkillLevel.ADVANCED
    
    def _estimate_learning_time(self, gap: KnowledgeGap) -> int:
        """Estimate learning time in minutes for a knowledge gap."""
        base_time = {
            KnowledgeGapSeverity.LOW: 15,
            KnowledgeGapSeverity.MEDIUM: 30,
            KnowledgeGapSeverity.HIGH: 60,
            KnowledgeGapSeverity.CRITICAL: 120
        }
        return base_time.get(gap.severity, 30)
    
    async def stop_workspace_monitoring(self, session_id: UUID) -> bool:
        """Stop monitoring a workspace session."""
        async with get_async_db() as session:
            query = select(WorkspaceSession).where(WorkspaceSession.id == session_id)
            result = await session.execute(query)
            workspace_session = result.scalar_one_or_none()
            
            if workspace_session and workspace_session.is_active:
                workspace_session.is_active = False
                workspace_session.ended_at = datetime.utcnow()
                
                # Calculate total duration
                if workspace_session.started_at:
                    duration = workspace_session.ended_at - workspace_session.started_at
                    workspace_session.total_duration_minutes = int(duration.total_seconds() / 60)
                
                # Remove from active sessions
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                
                await session.commit()
                
                logger.info(f"Stopped workspace monitoring for session {session_id}")
                return True
            
            return False


# Global context analyzer instance
context_analyzer = ContextAnalyzer()
