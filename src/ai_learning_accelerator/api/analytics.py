"""Analytics API endpoints."""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_db
from ..models.analytics import (
    LearningMetric, ProgressVisualization, LearningMilestone, LearningPattern,
    OptimizationSuggestion, AnalyticsReport, CompetencyAssessment,
    MetricType, VisualizationType, MilestoneType, PatternType
)
from ..models.user import User
from ..schemas.analytics import (
    LearningMetricCreate, LearningMetricUpdate, LearningMetricResponse,
    ProgressVisualizationCreate, ProgressVisualizationUpdate, ProgressVisualizationResponse,
    LearningMilestoneCreate, LearningMilestoneUpdate, LearningMilestoneResponse,
    LearningPatternResponse, LearningPatternUpdate,
    OptimizationSuggestionResponse, OptimizationSuggestionUpdate,
    AnalyticsReportCreate, AnalyticsReportResponse,
    CompetencyAssessmentCreate, CompetencyAssessmentUpdate, CompetencyAssessmentResponse,
    AnalyticsDashboardResponse
)
from ..services.analytics_engine import AnalyticsEngine
from ..services.auth import get_current_user
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.post("/metrics", response_model=LearningMetricResponse)
async def create_learning_metric(
    metric_data: LearningMetricCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new learning metric."""
    try:
        analytics_engine = AnalyticsEngine(db)
        
        metric = await analytics_engine.record_learning_metric(
            user_id=current_user.id,
            metric_type=metric_data.metric_type,
            metric_name=metric_data.metric_name,
            metric_value=metric_data.metric_value,
            metric_unit=metric_data.metric_unit,
            session_id=metric_data.session_id,
            content_id=metric_data.content_id,
            context_data=metric_data.context_data
        )
        
        return metric
        
    except Exception as e:
        logger.error(f"Failed to create learning metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to create learning metric")


@router.get("/metrics", response_model=List[LearningMetricResponse])
async def get_learning_metrics(
    metric_type: Optional[MetricType] = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get learning metrics for the current user."""
    try:
        query = select(LearningMetric).where(LearningMetric.user_id == current_user.id)
        
        if metric_type:
            query = query.where(LearningMetric.metric_type == metric_type)
        
        query = query.order_by(desc(LearningMetric.measurement_timestamp)).offset(offset).limit(limit)
        
        result = await db.execute(query)
        metrics = result.scalars().all()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get learning metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning metrics")


@router.get("/metrics/{metric_id}", response_model=LearningMetricResponse)
async def get_learning_metric(
    metric_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific learning metric."""
    try:
        result = await db.execute(
            select(LearningMetric).where(
                and_(
                    LearningMetric.id == metric_id,
                    LearningMetric.user_id == current_user.id
                )
            )
        )
        metric = result.scalar_one_or_none()
        
        if not metric:
            raise HTTPException(status_code=404, detail="Learning metric not found")
        
        return metric
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get learning metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning metric")


@router.put("/metrics/{metric_id}", response_model=LearningMetricResponse)
async def update_learning_metric(
    metric_id: UUID,
    metric_update: LearningMetricUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update a learning metric."""
    try:
        result = await db.execute(
            select(LearningMetric).where(
                and_(
                    LearningMetric.id == metric_id,
                    LearningMetric.user_id == current_user.id
                )
            )
        )
        metric = result.scalar_one_or_none()
        
        if not metric:
            raise HTTPException(status_code=404, detail="Learning metric not found")
        
        # Update fields
        update_data = metric_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(metric, field, value)
        
        await db.commit()
        await db.refresh(metric)
        
        return metric
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update learning metric: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update learning metric")


@router.post("/visualizations", response_model=ProgressVisualizationResponse)
async def create_progress_visualization(
    visualization_data: ProgressVisualizationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new progress visualization."""
    try:
        analytics_engine = AnalyticsEngine(db)
        
        visualization = await analytics_engine.generate_progress_visualization(
            user_id=current_user.id,
            visualization_name=visualization_data.visualization_name,
            visualization_type=visualization_data.visualization_type,
            config_data=visualization_data.config_data,
            time_range=visualization_data.time_range
        )
        
        return visualization
        
    except Exception as e:
        logger.error(f"Failed to create progress visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to create progress visualization")


@router.get("/visualizations", response_model=List[ProgressVisualizationResponse])
async def get_progress_visualizations(
    visualization_type: Optional[VisualizationType] = None,
    limit: int = Query(default=20, le=50),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get progress visualizations for the current user."""
    try:
        query = select(ProgressVisualization).where(ProgressVisualization.user_id == current_user.id)
        
        if visualization_type:
            query = query.where(ProgressVisualization.visualization_type == visualization_type)
        
        query = query.order_by(desc(ProgressVisualization.last_updated)).offset(offset).limit(limit)
        
        result = await db.execute(query)
        visualizations = result.scalars().all()
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Failed to get progress visualizations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get progress visualizations")


@router.get("/visualizations/{visualization_id}", response_model=ProgressVisualizationResponse)
async def get_progress_visualization(
    visualization_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific progress visualization."""
    try:
        result = await db.execute(
            select(ProgressVisualization).where(
                and_(
                    ProgressVisualization.id == visualization_id,
                    ProgressVisualization.user_id == current_user.id
                )
            )
        )
        visualization = result.scalar_one_or_none()
        
        if not visualization:
            raise HTTPException(status_code=404, detail="Progress visualization not found")
        
        # Increment view count
        visualization.view_count += 1
        await db.commit()
        
        return visualization
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to get progress visualization")


@router.put("/visualizations/{visualization_id}", response_model=ProgressVisualizationResponse)
async def update_progress_visualization(
    visualization_id: UUID,
    visualization_update: ProgressVisualizationUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update a progress visualization."""
    try:
        result = await db.execute(
            select(ProgressVisualization).where(
                and_(
                    ProgressVisualization.id == visualization_id,
                    ProgressVisualization.user_id == current_user.id
                )
            )
        )
        visualization = result.scalar_one_or_none()
        
        if not visualization:
            raise HTTPException(status_code=404, detail="Progress visualization not found")
        
        # Update fields
        update_data = visualization_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(visualization, field, value)
        
        # Update last_updated timestamp
        visualization.last_updated = datetime.utcnow()
        
        await db.commit()
        await db.refresh(visualization)
        
        return visualization
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update progress visualization: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update progress visualization")


@router.post("/milestones", response_model=LearningMilestoneResponse)
async def create_learning_milestone(
    milestone_data: LearningMilestoneCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new learning milestone."""
    try:
        analytics_engine = AnalyticsEngine(db)
        
        milestone = await analytics_engine.create_learning_milestone(
            user_id=current_user.id,
            milestone_name=milestone_data.milestone_name,
            milestone_type=milestone_data.milestone_type,
            criteria=milestone_data.criteria,
            target_value=milestone_data.target_value,
            description=milestone_data.description
        )
        
        return milestone
        
    except Exception as e:
        logger.error(f"Failed to create learning milestone: {e}")
        raise HTTPException(status_code=500, detail="Failed to create learning milestone")


@router.get("/milestones", response_model=List[LearningMilestoneResponse])
async def get_learning_milestones(
    milestone_type: Optional[MilestoneType] = None,
    achieved: Optional[bool] = None,
    limit: int = Query(default=20, le=50),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get learning milestones for the current user."""
    try:
        query = select(LearningMilestone).where(LearningMilestone.user_id == current_user.id)
        
        if milestone_type:
            query = query.where(LearningMilestone.milestone_type == milestone_type)
        
        if achieved is not None:
            query = query.where(LearningMilestone.is_achieved == achieved)
        
        query = query.order_by(desc(LearningMilestone.progress_percentage)).offset(offset).limit(limit)
        
        result = await db.execute(query)
        milestones = result.scalars().all()
        
        return milestones
        
    except Exception as e:
        logger.error(f"Failed to get learning milestones: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning milestones")


@router.get("/milestones/{milestone_id}", response_model=LearningMilestoneResponse)
async def get_learning_milestone(
    milestone_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific learning milestone."""
    try:
        result = await db.execute(
            select(LearningMilestone).where(
                and_(
                    LearningMilestone.id == milestone_id,
                    LearningMilestone.user_id == current_user.id
                )
            )
        )
        milestone = result.scalar_one_or_none()
        
        if not milestone:
            raise HTTPException(status_code=404, detail="Learning milestone not found")
        
        return milestone
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get learning milestone: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning milestone")


@router.put("/milestones/{milestone_id}", response_model=LearningMilestoneResponse)
async def update_learning_milestone(
    milestone_id: UUID,
    milestone_update: LearningMilestoneUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update a learning milestone."""
    try:
        result = await db.execute(
            select(LearningMilestone).where(
                and_(
                    LearningMilestone.id == milestone_id,
                    LearningMilestone.user_id == current_user.id
                )
            )
        )
        milestone = result.scalar_one_or_none()
        
        if not milestone:
            raise HTTPException(status_code=404, detail="Learning milestone not found")
        
        # Update fields
        update_data = milestone_update.model_dump(exclude_unset=True)
        
        # Handle current_value update specially to trigger progress calculation
        if "current_value" in update_data:
            analytics_engine = AnalyticsEngine(db)
            milestone = await analytics_engine.update_milestone_progress(
                milestone_id=milestone_id,
                current_value=update_data["current_value"]
            )
            # Remove current_value from update_data to avoid double update
            del update_data["current_value"]
        
        # Update remaining fields
        for field, value in update_data.items():
            setattr(milestone, field, value)
        
        await db.commit()
        await db.refresh(milestone)
        
        return milestone
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update learning milestone: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update learning milestone")


@router.post("/patterns/detect")
async def detect_learning_patterns(
    analysis_period_days: int = Query(default=30, ge=7, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Detect learning patterns for the current user."""
    try:
        analytics_engine = AnalyticsEngine(db)
        
        patterns = await analytics_engine.detect_learning_patterns(
            user_id=current_user.id,
            analysis_period_days=analysis_period_days
        )
        
        return {"patterns_detected": len(patterns), "patterns": patterns}
        
    except Exception as e:
        logger.error(f"Failed to detect learning patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to detect learning patterns")


@router.get("/patterns", response_model=List[LearningPatternResponse])
async def get_learning_patterns(
    pattern_type: Optional[PatternType] = None,
    limit: int = Query(default=20, le=50),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get learning patterns for the current user."""
    try:
        query = select(LearningPattern).where(LearningPattern.user_id == current_user.id)
        
        if pattern_type:
            query = query.where(LearningPattern.pattern_type == pattern_type)
        
        query = query.order_by(desc(LearningPattern.confidence_score)).offset(offset).limit(limit)
        
        result = await db.execute(query)
        patterns = result.scalars().all()
        
        return patterns
        
    except Exception as e:
        logger.error(f"Failed to get learning patterns: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning patterns")


@router.get("/patterns/{pattern_id}", response_model=LearningPatternResponse)
async def get_learning_pattern(
    pattern_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific learning pattern."""
    try:
        result = await db.execute(
            select(LearningPattern).where(
                and_(
                    LearningPattern.id == pattern_id,
                    LearningPattern.user_id == current_user.id
                )
            )
        )
        pattern = result.scalar_one_or_none()
        
        if not pattern:
            raise HTTPException(status_code=404, detail="Learning pattern not found")
        
        return pattern
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get learning pattern: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning pattern")


@router.put("/patterns/{pattern_id}", response_model=LearningPatternResponse)
async def update_learning_pattern(
    pattern_id: UUID,
    pattern_update: LearningPatternUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update a learning pattern."""
    try:
        result = await db.execute(
            select(LearningPattern).where(
                and_(
                    LearningPattern.id == pattern_id,
                    LearningPattern.user_id == current_user.id
                )
            )
        )
        pattern = result.scalar_one_or_none()
        
        if not pattern:
            raise HTTPException(status_code=404, detail="Learning pattern not found")
        
        # Update fields
        update_data = pattern_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(pattern, field, value)
        
        await db.commit()
        await db.refresh(pattern)
        
        return pattern
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update learning pattern: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update learning pattern")


@router.post("/suggestions/generate")
async def generate_optimization_suggestions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Generate optimization suggestions for the current user."""
    try:
        analytics_engine = AnalyticsEngine(db)
        
        suggestions = await analytics_engine.generate_optimization_suggestions(
            user_id=current_user.id
        )
        
        return {"suggestions_generated": len(suggestions), "suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"Failed to generate optimization suggestions: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate optimization suggestions")


@router.get("/suggestions", response_model=List[OptimizationSuggestionResponse])
async def get_optimization_suggestions(
    status: Optional[str] = None,
    limit: int = Query(default=20, le=50),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get optimization suggestions for the current user."""
    try:
        query = select(OptimizationSuggestion).where(OptimizationSuggestion.user_id == current_user.id)
        
        if status:
            query = query.where(OptimizationSuggestion.status == status)
        
        query = query.order_by(desc(OptimizationSuggestion.priority_score)).offset(offset).limit(limit)
        
        result = await db.execute(query)
        suggestions = result.scalars().all()
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Failed to get optimization suggestions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get optimization suggestions")


@router.get("/suggestions/{suggestion_id}", response_model=OptimizationSuggestionResponse)
async def get_optimization_suggestion(
    suggestion_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific optimization suggestion."""
    try:
        result = await db.execute(
            select(OptimizationSuggestion).where(
                and_(
                    OptimizationSuggestion.id == suggestion_id,
                    OptimizationSuggestion.user_id == current_user.id
                )
            )
        )
        suggestion = result.scalar_one_or_none()
        
        if not suggestion:
            raise HTTPException(status_code=404, detail="Optimization suggestion not found")
        
        return suggestion
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization suggestion: {e}")
        raise HTTPException(status_code=500, detail="Failed to get optimization suggestion")


@router.put("/suggestions/{suggestion_id}", response_model=OptimizationSuggestionResponse)
async def update_optimization_suggestion(
    suggestion_id: UUID,
    suggestion_update: OptimizationSuggestionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update an optimization suggestion."""
    try:
        result = await db.execute(
            select(OptimizationSuggestion).where(
                and_(
                    OptimizationSuggestion.id == suggestion_id,
                    OptimizationSuggestion.user_id == current_user.id
                )
            )
        )
        suggestion = result.scalar_one_or_none()
        
        if not suggestion:
            raise HTTPException(status_code=404, detail="Optimization suggestion not found")
        
        # Update fields
        update_data = suggestion_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(suggestion, field, value)
        
        await db.commit()
        await db.refresh(suggestion)
        
        return suggestion
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update optimization suggestion: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update optimization suggestion")


@router.post("/reports", response_model=AnalyticsReportResponse)
async def create_analytics_report(
    report_data: AnalyticsReportCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new analytics report."""
    try:
        analytics_engine = AnalyticsEngine(db)
        
        report = await analytics_engine.generate_analytics_report(
            user_id=current_user.id,
            report_name=report_data.report_name,
            report_type=report_data.report_type,
            time_period=report_data.time_period,
            metrics_included=report_data.metrics_included
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to create analytics report: {e}")
        raise HTTPException(status_code=500, detail="Failed to create analytics report")


@router.get("/reports", response_model=List[AnalyticsReportResponse])
async def get_analytics_reports(
    report_type: Optional[str] = None,
    limit: int = Query(default=20, le=50),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get analytics reports for the current user."""
    try:
        query = select(AnalyticsReport).where(AnalyticsReport.user_id == current_user.id)
        
        if report_type:
            query = query.where(AnalyticsReport.report_type == report_type)
        
        query = query.order_by(desc(AnalyticsReport.generation_date)).offset(offset).limit(limit)
        
        result = await db.execute(query)
        reports = result.scalars().all()
        
        return reports
        
    except Exception as e:
        logger.error(f"Failed to get analytics reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics reports")


@router.get("/reports/{report_id}", response_model=AnalyticsReportResponse)
async def get_analytics_report(
    report_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific analytics report."""
    try:
        result = await db.execute(
            select(AnalyticsReport).where(
                and_(
                    AnalyticsReport.id == report_id,
                    AnalyticsReport.user_id == current_user.id
                )
            )
        )
        report = result.scalar_one_or_none()
        
        if not report:
            raise HTTPException(status_code=404, detail="Analytics report not found")
        
        # Increment view count
        report.view_count += 1
        await db.commit()
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analytics report: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics report")


@router.post("/competency", response_model=CompetencyAssessmentResponse)
async def create_competency_assessment(
    assessment_data: CompetencyAssessmentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new competency assessment."""
    try:
        analytics_engine = AnalyticsEngine(db)
        
        assessment = await analytics_engine.assess_competency(
            user_id=current_user.id,
            skill_area=assessment_data.skill_area,
            assessment_method=assessment_data.assessment_method,
            evidence_data=assessment_data.evidence_data
        )
        
        return assessment
        
    except Exception as e:
        logger.error(f"Failed to create competency assessment: {e}")
        raise HTTPException(status_code=500, detail="Failed to create competency assessment")


@router.get("/competency", response_model=List[CompetencyAssessmentResponse])
async def get_competency_assessments(
    skill_area: Optional[str] = None,
    limit: int = Query(default=20, le=50),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get competency assessments for the current user."""
    try:
        query = select(CompetencyAssessment).where(CompetencyAssessment.user_id == current_user.id)
        
        if skill_area:
            query = query.where(CompetencyAssessment.skill_area == skill_area)
        
        query = query.order_by(desc(CompetencyAssessment.assessment_date)).offset(offset).limit(limit)
        
        result = await db.execute(query)
        assessments = result.scalars().all()
        
        return assessments
        
    except Exception as e:
        logger.error(f"Failed to get competency assessments: {e}")
        raise HTTPException(status_code=500, detail="Failed to get competency assessments")


@router.get("/competency/{assessment_id}", response_model=CompetencyAssessmentResponse)
async def get_competency_assessment(
    assessment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific competency assessment."""
    try:
        result = await db.execute(
            select(CompetencyAssessment).where(
                and_(
                    CompetencyAssessment.id == assessment_id,
                    CompetencyAssessment.user_id == current_user.id
                )
            )
        )
        assessment = result.scalar_one_or_none()
        
        if not assessment:
            raise HTTPException(status_code=404, detail="Competency assessment not found")
        
        return assessment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get competency assessment: {e}")
        raise HTTPException(status_code=500, detail="Failed to get competency assessment")


@router.put("/competency/{assessment_id}", response_model=CompetencyAssessmentResponse)
async def update_competency_assessment(
    assessment_id: UUID,
    assessment_update: CompetencyAssessmentUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Update a competency assessment."""
    try:
        result = await db.execute(
            select(CompetencyAssessment).where(
                and_(
                    CompetencyAssessment.id == assessment_id,
                    CompetencyAssessment.user_id == current_user.id
                )
            )
        )
        assessment = result.scalar_one_or_none()
        
        if not assessment:
            raise HTTPException(status_code=404, detail="Competency assessment not found")
        
        # Update fields
        update_data = assessment_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(assessment, field, value)
        
        await db.commit()
        await db.refresh(assessment)
        
        return assessment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update competency assessment: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update competency assessment")


@router.get("/dashboard", response_model=AnalyticsDashboardResponse)
async def get_analytics_dashboard(
    days: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get analytics dashboard data for the current user."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get recent metrics
        metrics_result = await db.execute(
            select(LearningMetric)
            .where(
                and_(
                    LearningMetric.user_id == current_user.id,
                    LearningMetric.measurement_timestamp >= start_date
                )
            )
            .order_by(desc(LearningMetric.measurement_timestamp))
            .limit(10)
        )
        recent_metrics = metrics_result.scalars().all()
        
        # Get active milestones
        milestones_result = await db.execute(
            select(LearningMilestone)
            .where(
                and_(
                    LearningMilestone.user_id == current_user.id,
                    LearningMilestone.is_achieved == False
                )
            )
            .order_by(desc(LearningMilestone.progress_percentage))
            .limit(5)
        )
        active_milestones = milestones_result.scalars().all()
        
        # Get key patterns
        patterns_result = await db.execute(
            select(LearningPattern)
            .where(
                and_(
                    LearningPattern.user_id == current_user.id,
                    LearningPattern.discovery_date >= start_date
                )
            )
            .order_by(desc(LearningPattern.confidence_score))
            .limit(3)
        )
        key_patterns = patterns_result.scalars().all()
        
        # Get top suggestions
        suggestions_result = await db.execute(
            select(OptimizationSuggestion)
            .where(
                and_(
                    OptimizationSuggestion.user_id == current_user.id,
                    OptimizationSuggestion.status == "pending"
                )
            )
            .order_by(desc(OptimizationSuggestion.priority_score))
            .limit(5)
        )
        top_suggestions = suggestions_result.scalars().all()
        
        # Get competency overview
        competency_result = await db.execute(
            select(CompetencyAssessment)
            .where(CompetencyAssessment.user_id == current_user.id)
            .order_by(desc(CompetencyAssessment.assessment_date))
            .limit(10)
        )
        competency_overview = competency_result.scalars().all()
        
        # Generate progress summary
        analytics_engine = AnalyticsEngine(db)
        progress_summary = await analytics_engine._generate_report_summary(
            current_user.id,
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        )
        
        dashboard_data = AnalyticsDashboardResponse(
            user_id=current_user.id,
            recent_metrics=recent_metrics,
            active_milestones=active_milestones,
            key_patterns=key_patterns,
            top_suggestions=top_suggestions,
            competency_overview=competency_overview,
            progress_summary=progress_summary,
            generated_at=datetime.utcnow()
        )
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics dashboard")