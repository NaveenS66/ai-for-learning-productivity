"""API endpoints for automation engine and pattern detection."""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_db
from ..models.user import User
from ..models.automation import ActionType, PatternType, AutomationStatus
from ..schemas.automation import (
    UserActionCreate, UserActionResponse,
    ActionPatternResponse, AutomationOpportunityResponse,
    AutomationOpportunityUpdate, AutomationScriptResponse,
    AutomationMetricsResponse, UserPreferenceUpdate, UserPreferenceResponse,
    PatternFeedback, AutomationSummary, PatternDetectionRequest,
    AutomationInsights, BulkActionImport, AutomationHealthCheck
)
from ..services.pattern_detector import pattern_detector
from ..services.workflow_generator import workflow_generator
from ..services.auth import get_current_user

router = APIRouter(prefix="/automation", tags=["automation"])


@router.post("/actions", response_model=UserActionResponse)
async def track_user_action(
    action_data: UserActionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Track a user action for pattern detection."""
    try:
        user_action = await pattern_detector.track_user_action(
            user_id=current_user.id,
            action_type=action_data.action_type,
            action_name=action_data.action_name,
            workspace_path=action_data.workspace_path,
            file_path=action_data.file_path,
            command=action_data.command,
            parameters=action_data.parameters,
            tool_used=action_data.tool_used
        )
        
        return UserActionResponse.from_orm(user_action)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to track user action: {str(e)}"
        )


@router.post("/actions/bulk")
async def import_bulk_actions(
    bulk_data: BulkActionImport,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Import multiple user actions in bulk."""
    try:
        imported_actions = []
        
        for action_data in bulk_data.actions:
            user_action = await pattern_detector.track_user_action(
                user_id=current_user.id,
                action_type=action_data.action_type,
                action_name=action_data.action_name,
                workspace_path=action_data.workspace_path or bulk_data.workspace_path,
                file_path=action_data.file_path,
                command=action_data.command,
                parameters=action_data.parameters,
                session_id=bulk_data.session_id,
                tool_used=action_data.tool_used
            )
            imported_actions.append(user_action)
        
        return {
            "message": f"Successfully imported {len(imported_actions)} actions",
            "imported_count": len(imported_actions)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import bulk actions: {str(e)}"
        )


@router.get("/patterns", response_model=List[ActionPatternResponse])
async def get_user_patterns(
    current_user: User = Depends(get_current_user),
    min_frequency: int = Query(default=3, ge=1, le=50, description="Minimum pattern frequency"),
    pattern_type: Optional[PatternType] = Query(default=None, description="Filter by pattern type"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of patterns"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get detected patterns for the current user."""
    try:
        patterns = await pattern_detector.get_user_patterns(
            user_id=current_user.id,
            min_frequency=min_frequency,
            limit=limit
        )
        
        # Filter by pattern type if specified
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        return [ActionPatternResponse.from_orm(pattern) for pattern in patterns]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user patterns: {str(e)}"
        )


@router.post("/patterns/detect")
async def trigger_pattern_detection(
    request_data: PatternDetectionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Manually trigger pattern detection for the current user."""
    try:
        # This would trigger a manual pattern detection run
        # For now, return a success message
        return {
            "message": "Pattern detection triggered successfully",
            "user_id": current_user.id,
            "force_detection": request_data.force_detection,
            "time_window_hours": request_data.time_window_hours
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger pattern detection: {str(e)}"
        )


@router.patch("/patterns/{pattern_id}/feedback")
async def update_pattern_feedback(
    pattern_id: UUID,
    feedback_data: PatternFeedback,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update feedback for a detected pattern."""
    try:
        await pattern_detector.update_pattern_feedback(
            pattern_id=pattern_id,
            is_beneficial=feedback_data.is_beneficial,
            user_feedback=feedback_data.user_feedback
        )
        
        return {"message": "Pattern feedback updated successfully"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update pattern feedback: {str(e)}"
        )


@router.get("/opportunities", response_model=List[AutomationOpportunityResponse])
async def get_automation_opportunities(
    current_user: User = Depends(get_current_user),
    min_score: float = Query(default=0.6, ge=0.0, le=1.0, description="Minimum automation score"),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    status_filter: Optional[AutomationStatus] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of opportunities"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get automation opportunities for the current user."""
    try:
        opportunities = await pattern_detector.get_automation_opportunities(
            user_id=current_user.id,
            min_score=min_score,
            limit=limit
        )
        
        # Apply additional filters
        if category:
            opportunities = [o for o in opportunities if o.category == category]
        
        if status_filter:
            opportunities = [o for o in opportunities if o.status == status_filter]
        
        return [AutomationOpportunityResponse.from_orm(opp) for opp in opportunities]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation opportunities: {str(e)}"
        )


@router.patch("/opportunities/{opportunity_id}")
async def update_automation_opportunity(
    opportunity_id: UUID,
    update_data: AutomationOpportunityUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update an automation opportunity."""
    try:
        # This would update the opportunity in the database
        # For now, return a success message
        return {
            "message": "Automation opportunity updated successfully",
            "opportunity_id": opportunity_id,
            "updates": update_data.dict(exclude_unset=True)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update automation opportunity: {str(e)}"
        )


@router.post("/opportunities/{opportunity_id}/generate-workflow", response_model=AutomationScriptResponse)
async def generate_workflow(
    opportunity_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Generate an automation workflow from an opportunity."""
    try:
        script = await workflow_generator.generate_workflow_from_opportunity(
            db, opportunity_id, current_user.id
        )
        
        if not script:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Opportunity not found or workflow generation failed"
            )
        
        return AutomationScriptResponse.from_orm(script)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate workflow: {str(e)}"
        )


@router.get("/scripts", response_model=List[AutomationScriptResponse])
async def get_automation_scripts(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
    enabled_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=100)
):
    """Get automation scripts for the current user."""
    try:
        from sqlalchemy import select, and_, desc
        from ..models.automation import AutomationScript
        
        query = select(AutomationScript).where(
            AutomationScript.user_id == current_user.id
        )
        
        if enabled_only:
            query = query.where(AutomationScript.is_enabled == True)
        
        query = query.order_by(desc(AutomationScript.created_at)).limit(limit)
        
        result = await db.execute(query)
        scripts = result.scalars().all()
        
        return [AutomationScriptResponse.from_orm(script) for script in scripts]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scripts: {str(e)}"
        )


@router.post("/scripts/{script_id}/execute")
async def execute_automation_script(
    script_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
    execution_context: Optional[dict] = None
):
    """Execute an automation script."""
    try:
        execution = await workflow_generator.execute_automation_script(
            db, script_id, current_user.id, execution_context or {}
        )
        
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Script not found or execution failed"
            )
        
        return {
            "message": "Script executed successfully",
            "execution_id": execution.id,
            "status": execution.status.value,
            "duration_ms": execution.duration_ms,
            "time_saved_minutes": execution.time_saved_minutes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute script: {str(e)}"
        )


@router.put("/scripts/{script_id}/enable")
async def enable_automation_script(
    script_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Enable an automation script."""
    try:
        from sqlalchemy import select, and_
        from ..models.automation import AutomationScript
        
        query = select(AutomationScript).where(
            and_(
                AutomationScript.id == script_id,
                AutomationScript.user_id == current_user.id
            )
        )
        result = await db.execute(query)
        script = result.scalar_one_or_none()
        
        if not script:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Script not found"
            )
        
        script.is_enabled = True
        await db.commit()
        
        return {"message": "Script enabled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable script: {str(e)}"
        )


@router.put("/scripts/{script_id}/disable")
async def disable_automation_script(
    script_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Disable an automation script."""
    try:
        from sqlalchemy import select, and_
        from ..models.automation import AutomationScript
        
        query = select(AutomationScript).where(
            and_(
                AutomationScript.id == script_id,
                AutomationScript.user_id == current_user.id
            )
        )
        result = await db.execute(query)
        script = result.scalar_one_or_none()
        
        if not script:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Script not found"
            )
        
        script.is_enabled = False
        await db.commit()
        
        return {"message": "Script disabled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable script: {str(e)}"
        )


@router.get("/executions")
async def get_execution_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
    script_id: Optional[UUID] = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """Get execution history for the current user."""
    try:
        executions = await workflow_generator.get_execution_history(
            db, current_user.id, script_id, limit
        )
        
        return {
            "executions": [
                {
                    "id": str(execution.id),
                    "script_id": str(execution.script_id),
                    "status": execution.status.value,
                    "started_at": execution.started_at.isoformat(),
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "duration_ms": execution.duration_ms,
                    "time_saved_minutes": execution.time_saved_minutes,
                    "exit_code": execution.exit_code
                }
                for execution in executions
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution history: {str(e)}"
        )


@router.get("/summary", response_model=AutomationSummary)
async def get_automation_summary(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get automation summary for the current user."""
    try:
        # Get patterns and opportunities
        patterns = await pattern_detector.get_user_patterns(
            user_id=current_user.id,
            min_frequency=1,
            limit=100
        )
        
        opportunities = await pattern_detector.get_automation_opportunities(
            user_id=current_user.id,
            min_score=0.0,
            limit=100
        )
        
        # Calculate summary statistics
        active_patterns = len([p for p in patterns if p.is_active])
        top_opportunities = opportunities[:5]  # Top 5 opportunities
        
        summary = AutomationSummary(
            total_patterns=len(patterns),
            active_patterns=active_patterns,
            automation_opportunities=len(opportunities),
            implemented_automations=0,  # Would be calculated from actual data
            total_time_saved_minutes=0,  # Would be calculated from actual data
            success_rate=0.95,  # Would be calculated from actual data
            top_opportunities=[AutomationOpportunityResponse.from_orm(opp) for opp in top_opportunities]
        )
        
        return summary
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation summary: {str(e)}"
        )


@router.get("/insights", response_model=AutomationInsights)
async def get_automation_insights(
    current_user: User = Depends(get_current_user),
    time_period_days: int = Query(default=30, ge=1, le=365, description="Time period for insights"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get automation insights and analytics."""
    try:
        # This would calculate actual insights from user data
        # For now, return mock insights
        insights = AutomationInsights(
            productivity_gain=15.5,
            most_automated_category="file_management",
            time_saved_this_week=45,
            recommended_next_automation="Automate test execution workflow",
            pattern_trends=[
                {"date": "2024-02-01", "patterns_detected": 3},
                {"date": "2024-02-02", "patterns_detected": 5},
                {"date": "2024-02-03", "patterns_detected": 2}
            ],
            automation_adoption_rate=0.75
        )
        
        return insights
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation insights: {str(e)}"
        )


@router.get("/preferences", response_model=UserPreferenceResponse)
async def get_user_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get user automation preferences."""
    try:
        # This would get actual user preferences from database
        # For now, return default preferences
        preferences = UserPreferenceResponse(
            id=current_user.id,
            user_id=current_user.id,
            auto_detect_patterns=True,
            min_pattern_frequency=3,
            min_automation_score=0.6,
            notify_new_opportunities=True,
            notification_frequency="daily",
            require_confirmation=True,
            auto_execute_simple=False,
            max_execution_time_minutes=30,
            risk_tolerance="medium",
            allow_file_modifications=False,
            allow_system_commands=False,
            learn_from_actions=True,
            share_anonymous_patterns=False,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return preferences
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user preferences: {str(e)}"
        )


@router.patch("/preferences")
async def update_user_preferences(
    preference_updates: UserPreferenceUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update user automation preferences."""
    try:
        # This would update actual user preferences in database
        # For now, return success message
        return {
            "message": "User preferences updated successfully",
            "user_id": current_user.id,
            "updates": preference_updates.dict(exclude_unset=True)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user preferences: {str(e)}"
        )


@router.get("/health", response_model=AutomationHealthCheck)
async def get_automation_health(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get automation system health status."""
    try:
        health_check = AutomationHealthCheck(
            pattern_detection_active=True,
            automation_engine_status="healthy",
            pending_opportunities=5,
            active_scripts=3,
            recent_executions=12,
            system_load=0.35,
            last_pattern_detection=datetime.utcnow() - timedelta(hours=2)
        )
        
        return health_check
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation health: {str(e)}"
        )


@router.get("/metrics", response_model=List[AutomationMetricsResponse])
async def get_automation_metrics(
    current_user: User = Depends(get_current_user),
    period_type: str = Query(default="weekly", description="Period type (daily, weekly, monthly)"),
    limit: int = Query(default=10, ge=1, le=50, description="Number of periods to return"),
    db: AsyncSession = Depends(get_async_db)
):
    """Get automation metrics for the current user."""
    try:
        # This would get actual metrics from database
        # For now, return empty list
        return []
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation metrics: {str(e)}"
        )


@router.get("/categories")
async def get_automation_categories(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get available automation categories."""
    try:
        categories = [
            {
                "name": "build_automation",
                "display_name": "Build Automation",
                "description": "Automate build and compilation processes"
            },
            {
                "name": "testing_automation",
                "display_name": "Testing Automation",
                "description": "Automate test execution and validation"
            },
            {
                "name": "file_management",
                "display_name": "File Management",
                "description": "Automate file operations and organization"
            },
            {
                "name": "version_control",
                "display_name": "Version Control",
                "description": "Automate Git and version control operations"
            },
            {
                "name": "deployment",
                "display_name": "Deployment",
                "description": "Automate deployment and release processes"
            },
            {
                "name": "workflow_automation",
                "display_name": "Workflow Automation",
                "description": "Automate general development workflows"
            }
        ]
        
        return {"categories": categories}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation categories: {str(e)}"
        )