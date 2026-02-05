"""API endpoints for context analyzer and recommendation engine."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_db
from ..models.user import User
from ..models.context import NotificationPriority, RecommendationType
from ..schemas.context import (
    WorkspaceSessionCreate, WorkspaceSessionResponse,
    WorkspaceEventCreate, WorkspaceEventResponse,
    ContextRecommendationResponse, RecommendationInteractionUpdate,
    LearningOpportunityResponse, ContextInsights,
    RecommendationFilters, WorkspaceMonitoringStatus,
    ProactiveRecommendationRequest, KnowledgeGapResponse,
    TechnologyStackResponse, ContextAnalysisResultResponse
)
from ..services.context_analyzer import context_analyzer
from ..services.auth import get_current_user

router = APIRouter(prefix="/context", tags=["context"])


@router.post("/sessions", response_model=WorkspaceSessionResponse)
async def start_workspace_session(
    session_data: WorkspaceSessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Start a new workspace monitoring session."""
    try:
        workspace_session = await context_analyzer.start_workspace_monitoring(
            user_id=current_user.id,
            workspace_path=session_data.workspace_path,
            session_name=session_data.session_name
        )
        
        return WorkspaceSessionResponse.from_orm(workspace_session)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workspace session: {str(e)}"
        )


@router.delete("/sessions/{session_id}")
async def stop_workspace_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Stop a workspace monitoring session."""
    success = await context_analyzer.stop_workspace_monitoring(session_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace session not found or already stopped"
        )
    
    return {"message": "Workspace session stopped successfully"}


@router.post("/sessions/{session_id}/events", response_model=List[ContextRecommendationResponse])
async def process_workspace_event(
    session_id: UUID,
    event_data: WorkspaceEventCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Process a workspace event and get generated recommendations."""
    try:
        recommendations = await context_analyzer.process_workspace_event(
            session_id=session_id,
            event_type=event_data.event_type,
            event_data=event_data.dict(exclude={"event_type"})
        )
        
        if recommendations:
            return [ContextRecommendationResponse.from_orm(rec) for rec in recommendations]
        
        return []
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process workspace event: {str(e)}"
        )


@router.get("/recommendations", response_model=List[ContextRecommendationResponse])
async def get_recommendations(
    current_user: User = Depends(get_current_user),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of recommendations"),
    priority: Optional[NotificationPriority] = Query(default=None, description="Filter by priority"),
    recommendation_type: Optional[RecommendationType] = Query(default=None, description="Filter by type"),
    include_dismissed: bool = Query(default=False, description="Include dismissed recommendations"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get active recommendations for the current user."""
    try:
        recommendations = await context_analyzer.get_active_recommendations(
            user_id=current_user.id,
            limit=limit,
            priority_filter=priority
        )
        
        # Filter by recommendation type if specified
        if recommendation_type:
            recommendations = [r for r in recommendations if r.recommendation_type == recommendation_type]
        
        # Filter dismissed recommendations if not included
        if not include_dismissed:
            recommendations = [r for r in recommendations if not r.is_dismissed]
        
        return [ContextRecommendationResponse.from_orm(rec) for rec in recommendations]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )


@router.patch("/recommendations/{recommendation_id}/interaction")
async def update_recommendation_interaction(
    recommendation_id: UUID,
    interaction_data: RecommendationInteractionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update user interaction with a recommendation."""
    try:
        success = await context_analyzer.update_recommendation_interaction(
            recommendation_id=recommendation_id,
            is_viewed=interaction_data.is_viewed,
            is_accepted=interaction_data.is_accepted,
            is_dismissed=interaction_data.is_dismissed,
            user_feedback=interaction_data.user_feedback
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Recommendation not found"
            )
        
        return {"message": "Recommendation interaction updated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update recommendation interaction: {str(e)}"
        )


@router.post("/recommendations/{recommendation_id}/deliver")
async def mark_recommendation_delivered(
    recommendation_id: UUID,
    delivery_method: str = Query(..., description="Method used to deliver the recommendation"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Mark a recommendation as delivered."""
    try:
        success = await context_analyzer.mark_recommendation_delivered(
            recommendation_id=recommendation_id,
            delivery_method=delivery_method
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Recommendation not found"
            )
        
        return {"message": "Recommendation marked as delivered"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark recommendation as delivered: {str(e)}"
        )


@router.get("/learning-opportunities", response_model=List[LearningOpportunityResponse])
async def get_learning_opportunities(
    current_user: User = Depends(get_current_user),
    workspace_session_id: Optional[UUID] = Query(default=None, description="Filter by workspace session"),
    limit: int = Query(default=10, ge=1, le=20, description="Maximum number of opportunities"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get non-intrusive learning opportunities for the current user."""
    try:
        opportunities = await context_analyzer.generate_learning_opportunities(
            user_id=current_user.id,
            workspace_session_id=workspace_session_id
        )
        
        # Limit results
        opportunities = opportunities[:limit]
        
        return [LearningOpportunityResponse.from_orm(opp) for opp in opportunities]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get learning opportunities: {str(e)}"
        )


@router.get("/insights", response_model=ContextInsights)
async def get_context_insights(
    current_user: User = Depends(get_current_user),
    workspace_path: Optional[str] = Query(default=None, description="Workspace path for context"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get context insights for the current user's workspace."""
    try:
        # This would typically analyze the current workspace context
        # For now, we'll return a simulated response
        
        insights = ContextInsights(
            current_focus="Python web development with FastAPI",
            technology_stack=["Python", "FastAPI", "SQLAlchemy", "PostgreSQL"],
            complexity_level=0.7,
            potential_challenges=[
                {
                    "challenge": "Database migration complexity",
                    "severity": "medium",
                    "description": "Complex database schema changes may require careful migration planning"
                },
                {
                    "challenge": "API performance optimization",
                    "severity": "low",
                    "description": "Consider implementing caching for frequently accessed endpoints"
                }
            ],
            recommended_resources=[
                {
                    "title": "FastAPI Performance Best Practices",
                    "type": "documentation",
                    "url": "https://fastapi.tiangolo.com/advanced/",
                    "relevance": 0.9
                },
                {
                    "title": "SQLAlchemy Migration Strategies",
                    "type": "tutorial",
                    "url": "https://alembic.sqlalchemy.org/",
                    "relevance": 0.8
                }
            ],
            knowledge_gaps=["Testing best practices", "API documentation"],
            learning_opportunities=[
                "Property-based testing with Hypothesis",
                "OpenAPI documentation generation",
                "Database performance optimization"
            ],
            productivity_suggestions=[
                "Set up automated code formatting with Black",
                "Implement pre-commit hooks for code quality",
                "Add comprehensive API documentation"
            ]
        )
        
        return insights
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get context insights: {str(e)}"
        )


@router.get("/knowledge-gaps", response_model=List[KnowledgeGapResponse])
async def get_knowledge_gaps(
    current_user: User = Depends(get_current_user),
    include_addressed: bool = Query(default=False, description="Include addressed gaps"),
    severity_filter: Optional[str] = Query(default=None, description="Filter by severity"),
    limit: int = Query(default=20, ge=1, le=50, description="Maximum number of gaps"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get knowledge gaps identified for the current user."""
    try:
        # This would query the actual knowledge gaps from the database
        # For now, we'll return an empty list as a placeholder
        return []
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get knowledge gaps: {str(e)}"
        )


@router.get("/technology-stack", response_model=List[TechnologyStackResponse])
async def get_technology_stack(
    current_user: User = Depends(get_current_user),
    workspace_path: Optional[str] = Query(default=None, description="Filter by workspace path"),
    active_only: bool = Query(default=True, description="Only return active technologies"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get detected technology stack for the current user."""
    try:
        # This would query the actual technology stack from the database
        # For now, we'll return an empty list as a placeholder
        return []
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get technology stack: {str(e)}"
        )


@router.get("/monitoring-status", response_model=List[WorkspaceMonitoringStatus])
async def get_monitoring_status(
    current_user: User = Depends(get_current_user),
    active_only: bool = Query(default=True, description="Only return active sessions"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get workspace monitoring status for the current user."""
    try:
        # This would query the actual workspace sessions and their status
        # For now, we'll return an empty list as a placeholder
        return []
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring status: {str(e)}"
        )


@router.post("/proactive-recommendations", response_model=List[ContextRecommendationResponse])
async def generate_proactive_recommendations(
    request_data: ProactiveRecommendationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Generate proactive recommendations based on current context."""
    try:
        # This would analyze the provided context and generate proactive recommendations
        # For now, we'll return an empty list as a placeholder
        return []
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate proactive recommendations: {str(e)}"
        )


@router.get("/analysis-results", response_model=List[ContextAnalysisResultResponse])
async def get_analysis_results(
    current_user: User = Depends(get_current_user),
    session_id: Optional[UUID] = Query(default=None, description="Filter by session ID"),
    analysis_type: Optional[str] = Query(default=None, description="Filter by analysis type"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of results"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get context analysis results for the current user."""
    try:
        # This would query the actual analysis results from the database
        # For now, we'll return an empty list as a placeholder
        return []
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analysis results: {str(e)}"
        )
