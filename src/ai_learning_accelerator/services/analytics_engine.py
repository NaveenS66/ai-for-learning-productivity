"""Learning analytics engine service."""

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import statistics
import math

from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.analytics import (
    LearningMetric, ProgressVisualization, LearningMilestone, LearningPattern,
    OptimizationSuggestion, AnalyticsReport, CompetencyAssessment,
    MetricType, VisualizationType, MilestoneType, PatternType
)
from ..models.user import User, LearningActivity
from ..models.content import LearningContent
from ..models.learning import LearningPath, LearningGoal
from ..database import get_async_db
from ..logging_config import get_logger

logger = get_logger(__name__)


class AnalyticsEngine:
    """Core analytics engine for learning data analysis."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def record_learning_metric(
        self,
        user_id: UUID,
        metric_type: MetricType,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        session_id: Optional[UUID] = None,
        content_id: Optional[UUID] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> LearningMetric:
        """Record a new learning metric."""
        try:
            # Get previous value for comparison
            previous_metric = await self._get_latest_metric(user_id, metric_type, metric_name)
            previous_value = previous_metric.metric_value if previous_metric else None
            
            # Calculate improvement percentage
            improvement_percentage = None
            if previous_value and previous_value > 0:
                improvement_percentage = ((metric_value - previous_value) / previous_value) * 100
            
            # Create new metric
            metric = LearningMetric(
                user_id=user_id,
                session_id=session_id,
                content_id=content_id,
                metric_type=metric_type,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                measurement_date=date.today(),
                context_data=context_data or {},
                previous_value=previous_value,
                improvement_percentage=improvement_percentage
            )
            
            self.db.add(metric)
            await self.db.commit()
            await self.db.refresh(metric)
            
            logger.info(f"Recorded learning metric: {metric_type.value} = {metric_value} for user {user_id}")
            return metric
            
        except Exception as e:
            logger.error(f"Failed to record learning metric: {e}")
            await self.db.rollback()
            raise
    
    async def generate_progress_visualization(
        self,
        user_id: UUID,
        visualization_name: str,
        visualization_type: VisualizationType,
        config_data: Dict[str, Any],
        time_range: Optional[Dict[str, Any]] = None
    ) -> ProgressVisualization:
        """Generate a progress visualization."""
        try:
            # Set default time range if not provided
            if not time_range:
                time_range = {
                    "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                    "end_date": datetime.now().isoformat()
                }
            
            # Generate visualization data based on type
            generated_data = await self._generate_visualization_data(
                user_id, visualization_type, config_data, time_range
            )
            
            visualization = ProgressVisualization(
                user_id=user_id,
                visualization_name=visualization_name,
                visualization_type=visualization_type,
                config_data=config_data,
                time_range=time_range,
                generated_data=generated_data,
                last_updated=datetime.utcnow()
            )
            
            self.db.add(visualization)
            await self.db.commit()
            await self.db.refresh(visualization)
            
            logger.info(f"Generated progress visualization: {visualization_name} for user {user_id}")
            return visualization
            
        except Exception as e:
            logger.error(f"Failed to generate progress visualization: {e}")
            await self.db.rollback()
            raise
    
    async def create_learning_milestone(
        self,
        user_id: UUID,
        milestone_name: str,
        milestone_type: MilestoneType,
        criteria: Dict[str, Any],
        target_value: Optional[float] = None,
        description: Optional[str] = None
    ) -> LearningMilestone:
        """Create a new learning milestone."""
        try:
            milestone = LearningMilestone(
                user_id=user_id,
                milestone_name=milestone_name,
                milestone_type=milestone_type,
                description=description,
                criteria=criteria,
                target_value=target_value,
                current_value=0.0,
                progress_percentage=0.0
            )
            
            self.db.add(milestone)
            await self.db.commit()
            await self.db.refresh(milestone)
            
            logger.info(f"Created learning milestone: {milestone_name} for user {user_id}")
            return milestone
            
        except Exception as e:
            logger.error(f"Failed to create learning milestone: {e}")
            await self.db.rollback()
            raise
    
    async def update_milestone_progress(
        self,
        milestone_id: UUID,
        current_value: float
    ) -> LearningMilestone:
        """Update milestone progress."""
        try:
            result = await self.db.execute(
                select(LearningMilestone).where(LearningMilestone.id == milestone_id)
            )
            milestone = result.scalar_one_or_none()
            
            if not milestone:
                raise ValueError(f"Milestone {milestone_id} not found")
            
            milestone.current_value = current_value
            
            # Calculate progress percentage
            if milestone.target_value and milestone.target_value > 0:
                milestone.progress_percentage = min(
                    (current_value / milestone.target_value) * 100, 100.0
                )
            else:
                milestone.progress_percentage = 100.0 if current_value > 0 else 0.0
            
            # Check if milestone is achieved
            if milestone.progress_percentage >= 100.0 and not milestone.is_achieved:
                milestone.is_achieved = True
                milestone.achievement_date = datetime.utcnow()
                
                # Generate next challenges
                milestone.next_challenges = await self._generate_next_challenges(milestone)
                
                logger.info(f"Milestone achieved: {milestone.milestone_name} for user {milestone.user_id}")
            
            await self.db.commit()
            await self.db.refresh(milestone)
            
            return milestone
            
        except Exception as e:
            logger.error(f"Failed to update milestone progress: {e}")
            await self.db.rollback()
            raise
    
    async def detect_learning_patterns(
        self,
        user_id: UUID,
        analysis_period_days: int = 30
    ) -> List[LearningPattern]:
        """Detect learning patterns for a user."""
        try:
            patterns = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=analysis_period_days)
            
            # Detect different types of patterns
            schedule_pattern = await self._detect_schedule_pattern(user_id, start_date, end_date)
            if schedule_pattern:
                patterns.append(schedule_pattern)
            
            content_pattern = await self._detect_content_preference_pattern(user_id, start_date, end_date)
            if content_pattern:
                patterns.append(content_pattern)
            
            difficulty_pattern = await self._detect_difficulty_progression_pattern(user_id, start_date, end_date)
            if difficulty_pattern:
                patterns.append(difficulty_pattern)
            
            engagement_pattern = await self._detect_engagement_pattern(user_id, start_date, end_date)
            if engagement_pattern:
                patterns.append(engagement_pattern)
            
            # Save patterns to database
            for pattern in patterns:
                self.db.add(pattern)
            
            await self.db.commit()
            
            logger.info(f"Detected {len(patterns)} learning patterns for user {user_id}")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect learning patterns: {e}")
            await self.db.rollback()
            raise
    
    async def generate_optimization_suggestions(
        self,
        user_id: UUID,
        patterns: Optional[List[LearningPattern]] = None
    ) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on patterns."""
        try:
            if not patterns:
                # Get recent patterns for the user
                result = await self.db.execute(
                    select(LearningPattern)
                    .where(LearningPattern.user_id == user_id)
                    .where(LearningPattern.discovery_date >= datetime.now() - timedelta(days=30))
                    .order_by(desc(LearningPattern.confidence_score))
                )
                patterns = result.scalars().all()
            
            suggestions = []
            
            for pattern in patterns:
                pattern_suggestions = await self._generate_suggestions_from_pattern(pattern)
                suggestions.extend(pattern_suggestions)
            
            # Add general optimization suggestions
            general_suggestions = await self._generate_general_suggestions(user_id)
            suggestions.extend(general_suggestions)
            
            # Prioritize suggestions
            suggestions = await self._prioritize_suggestions(suggestions)
            
            # Save suggestions to database
            for suggestion in suggestions:
                self.db.add(suggestion)
            
            await self.db.commit()
            
            logger.info(f"Generated {len(suggestions)} optimization suggestions for user {user_id}")
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate optimization suggestions: {e}")
            await self.db.rollback()
            raise
    
    async def generate_analytics_report(
        self,
        user_id: UUID,
        report_name: str,
        report_type: str,
        time_period: Dict[str, Any],
        metrics_included: Optional[List[str]] = None
    ) -> AnalyticsReport:
        """Generate a comprehensive analytics report."""
        try:
            start_time = datetime.utcnow()
            
            # Generate report data
            summary_data = await self._generate_report_summary(user_id, time_period)
            detailed_data = await self._generate_detailed_analytics(user_id, time_period, metrics_included)
            visualizations = await self._generate_report_visualizations(user_id, time_period)
            insights = await self._generate_report_insights(user_id, summary_data, detailed_data)
            recommendations = await self._generate_report_recommendations(user_id, insights)
            
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            report = AnalyticsReport(
                user_id=user_id,
                report_name=report_name,
                report_type=report_type,
                time_period=time_period,
                metrics_included=metrics_included or [],
                summary_data=summary_data,
                detailed_data=detailed_data,
                visualizations=visualizations,
                insights=insights,
                recommendations=recommendations,
                generation_time=generation_time,
                data_freshness=datetime.utcnow()
            )
            
            self.db.add(report)
            await self.db.commit()
            await self.db.refresh(report)
            
            logger.info(f"Generated analytics report: {report_name} for user {user_id}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            await self.db.rollback()
            raise
    
    async def assess_competency(
        self,
        user_id: UUID,
        skill_area: str,
        assessment_method: str,
        evidence_data: List[Dict[str, Any]]
    ) -> CompetencyAssessment:
        """Assess user competency in a skill area."""
        try:
            # Calculate competency level based on evidence
            competency_level = await self._calculate_competency_level(evidence_data, assessment_method)
            
            # Get previous assessment for comparison
            previous_assessment = await self._get_latest_competency_assessment(user_id, skill_area)
            previous_level = previous_assessment.competency_level if previous_assessment else None
            
            # Calculate improvement metrics
            improvement_rate = None
            time_to_improve = None
            if previous_assessment and previous_level:
                improvement_rate = competency_level - previous_level
                time_to_improve = (datetime.utcnow() - previous_assessment.assessment_date).days
            
            # Calculate confidence score based on evidence quality
            confidence_score = await self._calculate_assessment_confidence(evidence_data, assessment_method)
            
            assessment = CompetencyAssessment(
                user_id=user_id,
                skill_area=skill_area,
                competency_level=competency_level,
                assessment_method=assessment_method,
                evidence_data=evidence_data,
                previous_level=previous_level,
                improvement_rate=improvement_rate,
                time_to_improve=time_to_improve,
                confidence_score=confidence_score
            )
            
            self.db.add(assessment)
            await self.db.commit()
            await self.db.refresh(assessment)
            
            logger.info(f"Assessed competency: {skill_area} = {competency_level} for user {user_id}")
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to assess competency: {e}")
            await self.db.rollback()
            raise
    
    # Private helper methods
    
    async def _get_latest_metric(
        self,
        user_id: UUID,
        metric_type: MetricType,
        metric_name: str
    ) -> Optional[LearningMetric]:
        """Get the latest metric of a specific type for a user."""
        result = await self.db.execute(
            select(LearningMetric)
            .where(
                and_(
                    LearningMetric.user_id == user_id,
                    LearningMetric.metric_type == metric_type,
                    LearningMetric.metric_name == metric_name
                )
            )
            .order_by(desc(LearningMetric.measurement_timestamp))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def _generate_visualization_data(
        self,
        user_id: UUID,
        visualization_type: VisualizationType,
        config_data: Dict[str, Any],
        time_range: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate data for a specific visualization type."""
        start_date = datetime.fromisoformat(time_range["start_date"])
        end_date = datetime.fromisoformat(time_range["end_date"])
        
        if visualization_type == VisualizationType.LINE_CHART:
            return await self._generate_line_chart_data(user_id, config_data, start_date, end_date)
        elif visualization_type == VisualizationType.BAR_CHART:
            return await self._generate_bar_chart_data(user_id, config_data, start_date, end_date)
        elif visualization_type == VisualizationType.PIE_CHART:
            return await self._generate_pie_chart_data(user_id, config_data, start_date, end_date)
        elif visualization_type == VisualizationType.HEATMAP:
            return await self._generate_heatmap_data(user_id, config_data, start_date, end_date)
        elif visualization_type == VisualizationType.PROGRESS_BAR:
            return await self._generate_progress_bar_data(user_id, config_data, start_date, end_date)
        elif visualization_type == VisualizationType.RADAR_CHART:
            return await self._generate_radar_chart_data(user_id, config_data, start_date, end_date)
        elif visualization_type == VisualizationType.TIMELINE:
            return await self._generate_timeline_data(user_id, config_data, start_date, end_date)
        elif visualization_type == VisualizationType.DASHBOARD:
            return await self._generate_dashboard_data(user_id, config_data, start_date, end_date)
        else:
            return {"error": f"Unsupported visualization type: {visualization_type}"}
    
    async def _generate_line_chart_data(
        self,
        user_id: UUID,
        config_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate line chart data."""
        metric_type = config_data.get("metric_type", MetricType.COMPLETION_RATE)
        
        result = await self.db.execute(
            select(LearningMetric)
            .where(
                and_(
                    LearningMetric.user_id == user_id,
                    LearningMetric.metric_type == metric_type,
                    LearningMetric.measurement_timestamp >= start_date,
                    LearningMetric.measurement_timestamp <= end_date
                )
            )
            .order_by(asc(LearningMetric.measurement_timestamp))
        )
        metrics = result.scalars().all()
        
        return {
            "type": "line",
            "data": {
                "labels": [m.measurement_timestamp.isoformat() for m in metrics],
                "datasets": [{
                    "label": metric_type.value.replace("_", " ").title(),
                    "data": [m.metric_value for m in metrics],
                    "borderColor": "rgb(75, 192, 192)",
                    "tension": 0.1
                }]
            }
        }
    
    async def _generate_bar_chart_data(
        self,
        user_id: UUID,
        config_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate bar chart data."""
        # Group metrics by type and calculate averages
        result = await self.db.execute(
            select(
                LearningMetric.metric_type,
                func.avg(LearningMetric.metric_value).label("avg_value")
            )
            .where(
                and_(
                    LearningMetric.user_id == user_id,
                    LearningMetric.measurement_timestamp >= start_date,
                    LearningMetric.measurement_timestamp <= end_date
                )
            )
            .group_by(LearningMetric.metric_type)
        )
        data = result.all()
        
        return {
            "type": "bar",
            "data": {
                "labels": [d.metric_type.value.replace("_", " ").title() for d in data],
                "datasets": [{
                    "label": "Average Values",
                    "data": [float(d.avg_value) for d in data],
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 1
                }]
            }
        }
    
    async def _generate_pie_chart_data(
        self,
        user_id: UUID,
        config_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate pie chart data."""
        # Show distribution of learning activities by type
        result = await self.db.execute(
            select(
                LearningActivity.activity_type,
                func.count(LearningActivity.id).label("count")
            )
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= start_date,
                    LearningActivity.created_at <= end_date
                )
            )
            .group_by(LearningActivity.activity_type)
        )
        data = result.all()
        
        return {
            "type": "pie",
            "data": {
                "labels": [d.activity_type for d in data],
                "datasets": [{
                    "data": [d.count for d in data],
                    "backgroundColor": [
                        "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0",
                        "#9966FF", "#FF9F40", "#FF6384", "#C9CBCF"
                    ]
                }]
            }
        }
    
    async def _generate_heatmap_data(
        self,
        user_id: UUID,
        config_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate heatmap data."""
        # Learning activity heatmap by day of week and hour
        result = await self.db.execute(
            select(
                func.extract('dow', LearningActivity.created_at).label('day_of_week'),
                func.extract('hour', LearningActivity.created_at).label('hour'),
                func.count(LearningActivity.id).label('activity_count')
            )
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= start_date,
                    LearningActivity.created_at <= end_date
                )
            )
            .group_by('day_of_week', 'hour')
        )
        data = result.all()
        
        # Create 7x24 matrix (days x hours)
        heatmap_data = [[0 for _ in range(24)] for _ in range(7)]
        for d in data:
            day = int(d.day_of_week)
            hour = int(d.hour)
            count = d.activity_count
            heatmap_data[day][hour] = count
        
        return {
            "type": "heatmap",
            "data": heatmap_data,
            "labels": {
                "days": ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                "hours": [f"{i:02d}:00" for i in range(24)]
            }
        }
    
    async def _generate_progress_bar_data(
        self,
        user_id: UUID,
        config_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate progress bar data."""
        # Get milestone progress
        result = await self.db.execute(
            select(LearningMilestone)
            .where(
                and_(
                    LearningMilestone.user_id == user_id,
                    LearningMilestone.created_at >= start_date,
                    LearningMilestone.created_at <= end_date
                )
            )
            .order_by(desc(LearningMilestone.progress_percentage))
        )
        milestones = result.scalars().all()
        
        return {
            "type": "progress_bars",
            "data": [
                {
                    "label": m.milestone_name,
                    "progress": m.progress_percentage,
                    "target": 100.0,
                    "achieved": m.is_achieved
                }
                for m in milestones
            ]
        }
    
    async def _generate_radar_chart_data(
        self,
        user_id: UUID,
        config_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate radar chart data."""
        # Show competency levels across different skill areas
        result = await self.db.execute(
            select(CompetencyAssessment)
            .where(
                and_(
                    CompetencyAssessment.user_id == user_id,
                    CompetencyAssessment.assessment_date >= start_date,
                    CompetencyAssessment.assessment_date <= end_date
                )
            )
            .order_by(desc(CompetencyAssessment.assessment_date))
        )
        assessments = result.scalars().all()
        
        # Get latest assessment for each skill area
        skill_levels = {}
        for assessment in assessments:
            if assessment.skill_area not in skill_levels:
                skill_levels[assessment.skill_area] = assessment.competency_level
        
        return {
            "type": "radar",
            "data": {
                "labels": list(skill_levels.keys()),
                "datasets": [{
                    "label": "Competency Level",
                    "data": list(skill_levels.values()),
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "pointBackgroundColor": "rgba(54, 162, 235, 1)"
                }]
            },
            "options": {
                "scales": {
                    "r": {
                        "min": 0,
                        "max": 5
                    }
                }
            }
        }
    
    async def _generate_timeline_data(
        self,
        user_id: UUID,
        config_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate timeline data."""
        # Get learning milestones and achievements
        result = await self.db.execute(
            select(LearningMilestone)
            .where(
                and_(
                    LearningMilestone.user_id == user_id,
                    LearningMilestone.created_at >= start_date,
                    LearningMilestone.created_at <= end_date
                )
            )
            .order_by(asc(LearningMilestone.created_at))
        )
        milestones = result.scalars().all()
        
        timeline_events = []
        for milestone in milestones:
            timeline_events.append({
                "date": milestone.created_at.isoformat(),
                "title": f"Milestone Created: {milestone.milestone_name}",
                "type": "milestone_created",
                "achieved": False
            })
            
            if milestone.is_achieved and milestone.achievement_date:
                timeline_events.append({
                    "date": milestone.achievement_date.isoformat(),
                    "title": f"Milestone Achieved: {milestone.milestone_name}",
                    "type": "milestone_achieved",
                    "achieved": True
                })
        
        # Sort by date
        timeline_events.sort(key=lambda x: x["date"])
        
        return {
            "type": "timeline",
            "events": timeline_events
        }
    
    async def _generate_dashboard_data(
        self,
        user_id: UUID,
        config_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate dashboard data."""
        # Combine multiple visualization types
        dashboard_data = {
            "summary": await self._generate_report_summary(user_id, {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }),
            "line_chart": await self._generate_line_chart_data(user_id, {"metric_type": MetricType.COMPLETION_RATE}, start_date, end_date),
            "bar_chart": await self._generate_bar_chart_data(user_id, {}, start_date, end_date),
            "progress_bars": await self._generate_progress_bar_data(user_id, {}, start_date, end_date),
            "radar_chart": await self._generate_radar_chart_data(user_id, {}, start_date, end_date)
        }
        
        return dashboard_data
    
    async def _generate_next_challenges(self, milestone: LearningMilestone) -> List[str]:
        """Generate next challenges based on achieved milestone."""
        challenges = []
        
        if milestone.milestone_type == MilestoneType.SKILL_MASTERY:
            challenges.extend([
                f"Apply {milestone.milestone_name} in a real project",
                f"Teach {milestone.milestone_name} to someone else",
                f"Find advanced applications of {milestone.milestone_name}"
            ])
        elif milestone.milestone_type == MilestoneType.COURSE_COMPLETION:
            challenges.extend([
                "Apply course concepts to a personal project",
                "Take an advanced course in the same domain",
                "Share your learning experience with others"
            ])
        elif milestone.milestone_type == MilestoneType.PROJECT_COMPLETION:
            challenges.extend([
                "Optimize the project for better performance",
                "Add new features to the project",
                "Start a more complex project"
            ])
        
        return challenges[:3]  # Return top 3 challenges
    
    async def _detect_schedule_pattern(
        self,
        user_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[LearningPattern]:
        """Detect learning schedule patterns."""
        result = await self.db.execute(
            select(
                func.extract('dow', LearningActivity.created_at).label('day_of_week'),
                func.extract('hour', LearningActivity.created_at).label('hour'),
                func.count(LearningActivity.id).label('activity_count')
            )
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= start_date,
                    LearningActivity.created_at <= end_date
                )
            )
            .group_by('day_of_week', 'hour')
            .having(func.count(LearningActivity.id) >= 3)  # At least 3 activities
        )
        data = result.all()
        
        if not data:
            return None
        
        # Analyze patterns
        peak_hours = []
        peak_days = []
        
        # Find peak hours
        hour_counts = {}
        for d in data:
            hour = int(d.hour)
            count = d.activity_count
            hour_counts[hour] = hour_counts.get(hour, 0) + count
        
        if hour_counts:
            max_count = max(hour_counts.values())
            peak_hours = [hour for hour, count in hour_counts.items() if count >= max_count * 0.8]
        
        # Find peak days
        day_counts = {}
        for d in data:
            day = int(d.day_of_week)
            count = d.activity_count
            day_counts[day] = day_counts.get(day, 0) + count
        
        if day_counts:
            max_count = max(day_counts.values())
            peak_days = [day for day, count in day_counts.items() if count >= max_count * 0.8]
        
        # Calculate confidence based on consistency
        total_activities = sum(d.activity_count for d in data)
        peak_activities = sum(d.activity_count for d in data if int(d.hour) in peak_hours and int(d.day_of_week) in peak_days)
        confidence_score = peak_activities / total_activities if total_activities > 0 else 0.0
        
        if confidence_score < 0.3:  # Low confidence threshold
            return None
        
        day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        
        pattern = LearningPattern(
            user_id=user_id,
            pattern_name="Learning Schedule Pattern",
            pattern_type=PatternType.LEARNING_SCHEDULE,
            description=f"User tends to learn most actively on {', '.join([day_names[d] for d in peak_days])} between {min(peak_hours):02d}:00-{max(peak_hours):02d}:00",
            pattern_data={
                "peak_hours": peak_hours,
                "peak_days": peak_days,
                "hour_distribution": hour_counts,
                "day_distribution": day_counts
            },
            confidence_score=confidence_score,
            analysis_period={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            data_points=len(data),
            frequency=len(peak_hours) * len(peak_days),
            strength=confidence_score,
            stability=min(confidence_score * 1.2, 1.0)
        )
        
        return pattern
    
    async def _detect_content_preference_pattern(
        self,
        user_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[LearningPattern]:
        """Detect content preference patterns."""
        result = await self.db.execute(
            select(
                LearningContent.content_type,
                LearningContent.difficulty_level,
                func.count(LearningActivity.id).label('activity_count'),
                func.avg(LearningActivity.completion_percentage).label('avg_completion')
            )
            .join(LearningActivity, LearningActivity.content_id == LearningContent.id)
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= start_date,
                    LearningActivity.created_at <= end_date
                )
            )
            .group_by(LearningContent.content_type, LearningContent.difficulty_level)
        )
        data = result.all()
        
        if not data:
            return None
        
        # Analyze preferences
        content_preferences = {}
        difficulty_preferences = {}
        
        for d in data:
            content_type = d.content_type
            difficulty = d.difficulty_level
            count = d.activity_count
            completion = float(d.avg_completion) if d.avg_completion else 0.0
            
            # Weight by both frequency and completion rate
            score = count * (completion / 100.0)
            
            content_preferences[content_type] = content_preferences.get(content_type, 0) + score
            difficulty_preferences[difficulty] = difficulty_preferences.get(difficulty, 0) + score
        
        if not content_preferences:
            return None
        
        # Find top preferences
        top_content_types = sorted(content_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
        preferred_difficulty = max(difficulty_preferences.items(), key=lambda x: x[1])[0] if difficulty_preferences else None
        
        total_score = sum(content_preferences.values())
        confidence_score = top_content_types[0][1] / total_score if total_score > 0 else 0.0
        
        if confidence_score < 0.2:
            return None
        
        pattern = LearningPattern(
            user_id=user_id,
            pattern_name="Content Preference Pattern",
            pattern_type=PatternType.CONTENT_PREFERENCE,
            description=f"User prefers {top_content_types[0][0]} content at difficulty level {preferred_difficulty}",
            pattern_data={
                "content_preferences": dict(top_content_types),
                "difficulty_preference": preferred_difficulty,
                "all_preferences": content_preferences
            },
            confidence_score=confidence_score,
            analysis_period={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            data_points=len(data),
            strength=confidence_score,
            stability=min(confidence_score * 1.1, 1.0)
        )
        
        return pattern
    
    async def _detect_difficulty_progression_pattern(
        self,
        user_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[LearningPattern]:
        """Detect difficulty progression patterns."""
        result = await self.db.execute(
            select(
                LearningContent.difficulty_level,
                LearningActivity.created_at,
                LearningActivity.completion_percentage
            )
            .join(LearningActivity, LearningActivity.content_id == LearningContent.id)
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= start_date,
                    LearningActivity.created_at <= end_date
                )
            )
            .order_by(asc(LearningActivity.created_at))
        )
        data = result.all()
        
        if len(data) < 5:  # Need at least 5 data points
            return None
        
        # Analyze progression
        difficulties = [d.difficulty_level for d in data]
        timestamps = [d.created_at for d in data]
        completions = [d.completion_percentage for d in data]
        
        # Calculate trend
        n = len(difficulties)
        sum_x = sum(range(n))
        sum_y = sum(difficulties)
        sum_xy = sum(i * difficulties[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        # Linear regression slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        
        # Determine progression type
        if slope > 0.1:
            progression_type = "increasing"
        elif slope < -0.1:
            progression_type = "decreasing"
        else:
            progression_type = "stable"
        
        # Calculate confidence based on R-squared
        mean_y = sum_y / n
        ss_tot = sum((difficulties[i] - mean_y) ** 2 for i in range(n))
        ss_res = sum((difficulties[i] - (slope * i + (sum_y - slope * sum_x) / n)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        confidence_score = max(0, r_squared)
        
        if confidence_score < 0.3:
            return None
        
        pattern = LearningPattern(
            user_id=user_id,
            pattern_name="Difficulty Progression Pattern",
            pattern_type=PatternType.DIFFICULTY_PROGRESSION,
            description=f"User shows {progression_type} difficulty progression with slope {slope:.2f}",
            pattern_data={
                "progression_type": progression_type,
                "slope": slope,
                "r_squared": r_squared,
                "difficulty_range": [min(difficulties), max(difficulties)],
                "avg_completion": statistics.mean(completions)
            },
            confidence_score=confidence_score,
            analysis_period={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            data_points=len(data),
            strength=abs(slope),
            stability=confidence_score
        )
        
        return pattern
    
    async def _detect_engagement_pattern(
        self,
        user_id: UUID,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[LearningPattern]:
        """Detect engagement patterns."""
        result = await self.db.execute(
            select(
                func.date(LearningActivity.created_at).label('activity_date'),
                func.count(LearningActivity.id).label('activity_count'),
                func.avg(LearningActivity.completion_percentage).label('avg_completion'),
                func.sum(LearningActivity.time_spent).label('total_time')
            )
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= start_date,
                    LearningActivity.created_at <= end_date
                )
            )
            .group_by(func.date(LearningActivity.created_at))
            .order_by(func.date(LearningActivity.created_at))
        )
        data = result.all()
        
        if len(data) < 7:  # Need at least a week of data
            return None
        
        # Calculate engagement metrics
        daily_activities = [d.activity_count for d in data]
        daily_completions = [float(d.avg_completion) if d.avg_completion else 0.0 for d in data]
        daily_times = [d.total_time if d.total_time else 0 for d in data]
        
        # Calculate consistency (inverse of coefficient of variation)
        if len(daily_activities) > 1:
            activity_consistency = 1 - (statistics.stdev(daily_activities) / statistics.mean(daily_activities)) if statistics.mean(daily_activities) > 0 else 0
            completion_consistency = 1 - (statistics.stdev(daily_completions) / statistics.mean(daily_completions)) if statistics.mean(daily_completions) > 0 else 0
        else:
            activity_consistency = 1.0
            completion_consistency = 1.0
        
        # Overall engagement score
        avg_activities = statistics.mean(daily_activities)
        avg_completion = statistics.mean(daily_completions)
        avg_time = statistics.mean(daily_times)
        
        engagement_score = (avg_activities / 10.0 + avg_completion / 100.0 + min(avg_time / 3600.0, 1.0)) / 3.0
        consistency_score = (activity_consistency + completion_consistency) / 2.0
        
        confidence_score = min(engagement_score * consistency_score, 1.0)
        
        if confidence_score < 0.2:
            return None
        
        # Determine engagement level
        if engagement_score > 0.7:
            engagement_level = "high"
        elif engagement_score > 0.4:
            engagement_level = "medium"
        else:
            engagement_level = "low"
        
        pattern = LearningPattern(
            user_id=user_id,
            pattern_name="Engagement Pattern",
            pattern_type=PatternType.ENGAGEMENT_PATTERN,
            description=f"User shows {engagement_level} engagement with {consistency_score:.2f} consistency",
            pattern_data={
                "engagement_level": engagement_level,
                "engagement_score": engagement_score,
                "consistency_score": consistency_score,
                "avg_daily_activities": avg_activities,
                "avg_completion_rate": avg_completion,
                "avg_daily_time": avg_time
            },
            confidence_score=confidence_score,
            analysis_period={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            data_points=len(data),
            strength=engagement_score,
            stability=consistency_score
        )
        
        return pattern
    
    async def _generate_suggestions_from_pattern(self, pattern: LearningPattern) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions from a specific pattern."""
        suggestions = []
        
        if pattern.pattern_type == PatternType.LEARNING_SCHEDULE:
            peak_hours = pattern.pattern_data.get("peak_hours", [])
            peak_days = pattern.pattern_data.get("peak_days", [])
            
            if peak_hours and peak_days:
                suggestions.append(OptimizationSuggestion(
                    user_id=pattern.user_id,
                    suggestion_title="Optimize Learning Schedule",
                    suggestion_type="schedule_optimization",
                    description=f"Schedule your most challenging learning sessions during your peak hours ({min(peak_hours):02d}:00-{max(peak_hours):02d}:00) on your most active days.",
                    rationale="Based on your activity patterns, you're most engaged during these times.",
                    expected_benefit="Improved focus and retention during learning sessions.",
                    action_steps=[
                        "Block calendar time during peak hours",
                        "Save easier content for off-peak times",
                        "Set reminders for optimal learning windows"
                    ],
                    priority_score=0.8,
                    impact_score=0.7,
                    urgency_score=0.5,
                    source_pattern_id=pattern.id
                ))
        
        elif pattern.pattern_type == PatternType.CONTENT_PREFERENCE:
            preferred_content = list(pattern.pattern_data.get("content_preferences", {}).keys())
            
            if preferred_content:
                suggestions.append(OptimizationSuggestion(
                    user_id=pattern.user_id,
                    suggestion_title="Leverage Content Preferences",
                    suggestion_type="content_optimization",
                    description=f"Focus on {preferred_content[0]} content types where you show highest engagement.",
                    rationale="You complete these content types more consistently.",
                    expected_benefit="Higher completion rates and better learning outcomes.",
                    action_steps=[
                        f"Prioritize {preferred_content[0]} content in your learning path",
                        "Use preferred content types for difficult concepts",
                        "Gradually introduce other content types"
                    ],
                    priority_score=0.7,
                    impact_score=0.6,
                    urgency_score=0.4,
                    source_pattern_id=pattern.id
                ))
        
        elif pattern.pattern_type == PatternType.DIFFICULTY_PROGRESSION:
            progression_type = pattern.pattern_data.get("progression_type", "stable")
            
            if progression_type == "increasing":
                suggestions.append(OptimizationSuggestion(
                    user_id=pattern.user_id,
                    suggestion_title="Continue Difficulty Progression",
                    suggestion_type="difficulty_optimization",
                    description="You're successfully progressing to harder content. Consider accelerating your learning path.",
                    rationale="Your difficulty progression shows consistent improvement.",
                    expected_benefit="Faster skill development and mastery.",
                    action_steps=[
                        "Increase difficulty level more quickly",
                        "Take on challenging projects",
                        "Seek advanced learning materials"
                    ],
                    priority_score=0.6,
                    impact_score=0.8,
                    urgency_score=0.6,
                    source_pattern_id=pattern.id
                ))
            elif progression_type == "stable":
                suggestions.append(OptimizationSuggestion(
                    user_id=pattern.user_id,
                    suggestion_title="Challenge Yourself More",
                    suggestion_type="difficulty_optimization",
                    description="You're staying at the same difficulty level. Try gradually increasing challenge.",
                    rationale="Stable difficulty might indicate readiness for more challenge.",
                    expected_benefit="Accelerated learning and skill development.",
                    action_steps=[
                        "Try content one level higher in difficulty",
                        "Mix in some challenging exercises",
                        "Set progressive difficulty goals"
                    ],
                    priority_score=0.5,
                    impact_score=0.6,
                    urgency_score=0.3,
                    source_pattern_id=pattern.id
                ))
        
        elif pattern.pattern_type == PatternType.ENGAGEMENT_PATTERN:
            engagement_level = pattern.pattern_data.get("engagement_level", "medium")
            consistency_score = pattern.pattern_data.get("consistency_score", 0.5)
            
            if engagement_level == "low":
                suggestions.append(OptimizationSuggestion(
                    user_id=pattern.user_id,
                    suggestion_title="Boost Learning Engagement",
                    suggestion_type="engagement_optimization",
                    description="Your engagement levels could be improved with some adjustments.",
                    rationale="Low engagement patterns detected in your learning activities.",
                    expected_benefit="Increased motivation and better learning outcomes.",
                    action_steps=[
                        "Try different content formats",
                        "Set smaller, achievable goals",
                        "Find learning partners or communities",
                        "Gamify your learning experience"
                    ],
                    priority_score=0.9,
                    impact_score=0.8,
                    urgency_score=0.7,
                    source_pattern_id=pattern.id
                ))
            
            if consistency_score < 0.5:
                suggestions.append(OptimizationSuggestion(
                    user_id=pattern.user_id,
                    suggestion_title="Improve Learning Consistency",
                    suggestion_type="consistency_optimization",
                    description="Your learning schedule shows inconsistency. Regular practice yields better results.",
                    rationale="Inconsistent learning patterns detected.",
                    expected_benefit="Better retention and steady progress.",
                    action_steps=[
                        "Set a regular learning schedule",
                        "Start with shorter, daily sessions",
                        "Use habit tracking tools",
                        "Create accountability systems"
                    ],
                    priority_score=0.7,
                    impact_score=0.7,
                    urgency_score=0.5,
                    source_pattern_id=pattern.id
                ))
        
        return suggestions
    
    async def _generate_general_suggestions(self, user_id: UUID) -> List[OptimizationSuggestion]:
        """Generate general optimization suggestions."""
        suggestions = []
        
        # Get user's recent activity
        result = await self.db.execute(
            select(func.count(LearningActivity.id))
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= datetime.now() - timedelta(days=7)
                )
            )
        )
        recent_activity_count = result.scalar() or 0
        
        if recent_activity_count < 3:
            suggestions.append(OptimizationSuggestion(
                user_id=user_id,
                suggestion_title="Increase Learning Frequency",
                suggestion_type="frequency_optimization",
                description="Try to engage in learning activities more regularly for better results.",
                rationale="Low activity detected in recent days.",
                expected_benefit="Better retention and steady progress.",
                action_steps=[
                    "Set aside 15-30 minutes daily for learning",
                    "Choose a consistent time each day",
                    "Start with easier content to build momentum"
                ],
                priority_score=0.6,
                impact_score=0.5,
                urgency_score=0.4
            ))
        
        # Check for incomplete milestones
        result = await self.db.execute(
            select(func.count(LearningMilestone.id))
            .where(
                and_(
                    LearningMilestone.user_id == user_id,
                    LearningMilestone.is_achieved == False,
                    LearningMilestone.progress_percentage > 50
                )
            )
        )
        near_complete_milestones = result.scalar() or 0
        
        if near_complete_milestones > 0:
            suggestions.append(OptimizationSuggestion(
                user_id=user_id,
                suggestion_title="Complete Pending Milestones",
                suggestion_type="milestone_completion",
                description=f"You have {near_complete_milestones} milestones that are more than 50% complete.",
                rationale="Completing milestones provides motivation and clear progress markers.",
                expected_benefit="Sense of achievement and momentum for continued learning.",
                action_steps=[
                    "Review your pending milestones",
                    "Focus on the closest to completion",
                    "Break remaining work into small tasks"
                ],
                priority_score=0.7,
                impact_score=0.6,
                urgency_score=0.6
            ))
        
        return suggestions
    
    async def _prioritize_suggestions(self, suggestions: List[OptimizationSuggestion]) -> List[OptimizationSuggestion]:
        """Prioritize suggestions based on impact, urgency, and priority scores."""
        for suggestion in suggestions:
            # Calculate overall priority score
            suggestion.priority_score = (
                suggestion.priority_score * 0.4 +
                suggestion.impact_score * 0.4 +
                suggestion.urgency_score * 0.2
            )
        
        # Sort by priority score (descending)
        suggestions.sort(key=lambda x: x.priority_score, reverse=True)
        
        return suggestions
    
    async def _generate_report_summary(self, user_id: UUID, time_period: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary data for analytics report."""
        start_date = datetime.fromisoformat(time_period["start_date"])
        end_date = datetime.fromisoformat(time_period["end_date"])
        
        # Total activities
        result = await self.db.execute(
            select(func.count(LearningActivity.id))
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= start_date,
                    LearningActivity.created_at <= end_date
                )
            )
        )
        total_activities = result.scalar() or 0
        
        # Average completion rate
        result = await self.db.execute(
            select(func.avg(LearningActivity.completion_percentage))
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= start_date,
                    LearningActivity.created_at <= end_date
                )
            )
        )
        avg_completion = float(result.scalar() or 0.0)
        
        # Total time spent
        result = await self.db.execute(
            select(func.sum(LearningActivity.time_spent))
            .where(
                and_(
                    LearningActivity.user_id == user_id,
                    LearningActivity.created_at >= start_date,
                    LearningActivity.created_at <= end_date
                )
            )
        )
        total_time = result.scalar() or 0
        
        # Milestones achieved
        result = await self.db.execute(
            select(func.count(LearningMilestone.id))
            .where(
                and_(
                    LearningMilestone.user_id == user_id,
                    LearningMilestone.is_achieved == True,
                    LearningMilestone.achievement_date >= start_date,
                    LearningMilestone.achievement_date <= end_date
                )
            )
        )
        milestones_achieved = result.scalar() or 0
        
        return {
            "total_activities": total_activities,
            "average_completion_rate": avg_completion,
            "total_time_spent": total_time,
            "milestones_achieved": milestones_achieved,
            "period_days": (end_date - start_date).days,
            "activities_per_day": total_activities / max((end_date - start_date).days, 1),
            "time_per_day": total_time / max((end_date - start_date).days, 1)
        }
    
    async def _generate_detailed_analytics(
        self,
        user_id: UUID,
        time_period: Dict[str, Any],
        metrics_included: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate detailed analytics data."""
        start_date = datetime.fromisoformat(time_period["start_date"])
        end_date = datetime.fromisoformat(time_period["end_date"])
        
        detailed_data = {}
        
        # Learning metrics breakdown
        if not metrics_included or "learning_metrics" in metrics_included:
            result = await self.db.execute(
                select(
                    LearningMetric.metric_type,
                    func.avg(LearningMetric.metric_value).label('avg_value'),
                    func.min(LearningMetric.metric_value).label('min_value'),
                    func.max(LearningMetric.metric_value).label('max_value'),
                    func.count(LearningMetric.id).label('count')
                )
                .where(
                    and_(
                        LearningMetric.user_id == user_id,
                        LearningMetric.measurement_timestamp >= start_date,
                        LearningMetric.measurement_timestamp <= end_date
                    )
                )
                .group_by(LearningMetric.metric_type)
            )
            metrics_data = result.all()
            
            detailed_data["learning_metrics"] = {
                d.metric_type.value: {
                    "average": float(d.avg_value),
                    "minimum": float(d.min_value),
                    "maximum": float(d.max_value),
                    "count": d.count
                }
                for d in metrics_data
            }
        
        # Content engagement breakdown
        if not metrics_included or "content_engagement" in metrics_included:
            result = await self.db.execute(
                select(
                    LearningContent.content_type,
                    func.count(LearningActivity.id).label('activity_count'),
                    func.avg(LearningActivity.completion_percentage).label('avg_completion'),
                    func.sum(LearningActivity.time_spent).label('total_time')
                )
                .join(LearningActivity, LearningActivity.content_id == LearningContent.id)
                .where(
                    and_(
                        LearningActivity.user_id == user_id,
                        LearningActivity.created_at >= start_date,
                        LearningActivity.created_at <= end_date
                    )
                )
                .group_by(LearningContent.content_type)
            )
            content_data = result.all()
            
            detailed_data["content_engagement"] = {
                d.content_type: {
                    "activities": d.activity_count,
                    "avg_completion": float(d.avg_completion) if d.avg_completion else 0.0,
                    "total_time": d.total_time or 0
                }
                for d in content_data
            }
        
        return detailed_data
    
    async def _generate_report_visualizations(self, user_id: UUID, time_period: Dict[str, Any]) -> List[str]:
        """Generate list of recommended visualizations for the report."""
        return [
            "learning_progress_timeline",
            "skill_competency_radar",
            "activity_heatmap",
            "milestone_progress_bars",
            "content_type_distribution"
        ]
    
    async def _generate_report_insights(
        self,
        user_id: UUID,
        summary_data: Dict[str, Any],
        detailed_data: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from analytics data."""
        insights = []
        
        # Activity insights
        activities_per_day = summary_data.get("activities_per_day", 0)
        if activities_per_day > 2:
            insights.append("You maintain a high level of learning activity with consistent daily engagement.")
        elif activities_per_day > 1:
            insights.append("You have moderate learning activity. Consider increasing frequency for better results.")
        else:
            insights.append("Your learning activity is low. Try to establish a more regular learning routine.")
        
        # Completion rate insights
        avg_completion = summary_data.get("average_completion_rate", 0)
        if avg_completion > 80:
            insights.append("Excellent completion rates indicate strong engagement and persistence.")
        elif avg_completion > 60:
            insights.append("Good completion rates, but there's room for improvement in finishing activities.")
        else:
            insights.append("Low completion rates suggest content might be too difficult or not engaging enough.")
        
        # Time management insights
        time_per_day = summary_data.get("time_per_day", 0)
        if time_per_day > 3600:  # More than 1 hour
            insights.append("You dedicate substantial time to learning each day, which is excellent for skill development.")
        elif time_per_day > 1800:  # More than 30 minutes
            insights.append("You maintain a good daily learning time commitment.")
        else:
            insights.append("Consider increasing your daily learning time for more effective skill development.")
        
        # Milestone insights
        milestones_achieved = summary_data.get("milestones_achieved", 0)
        if milestones_achieved > 0:
            insights.append(f"You achieved {milestones_achieved} milestones, showing clear progress toward your goals.")
        else:
            insights.append("No milestones achieved in this period. Consider setting more achievable short-term goals.")
        
        return insights
    
    async def _generate_report_recommendations(self, user_id: UUID, insights: List[str]) -> List[str]:
        """Generate recommendations based on insights."""
        recommendations = []
        
        # Get recent optimization suggestions
        result = await self.db.execute(
            select(OptimizationSuggestion)
            .where(
                and_(
                    OptimizationSuggestion.user_id == user_id,
                    OptimizationSuggestion.status == "pending"
                )
            )
            .order_by(desc(OptimizationSuggestion.priority_score))
            .limit(5)
        )
        suggestions = result.scalars().all()
        
        for suggestion in suggestions:
            recommendations.append(f"{suggestion.suggestion_title}: {suggestion.description}")
        
        # Add general recommendations if no specific suggestions
        if not recommendations:
            recommendations.extend([
                "Set up a regular learning schedule to improve consistency",
                "Focus on completing started activities before beginning new ones",
                "Track your progress with specific, measurable milestones",
                "Experiment with different content types to find what works best for you"
            ])
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def _calculate_competency_level(
        self,
        evidence_data: List[Dict[str, Any]],
        assessment_method: str
    ) -> float:
        """Calculate competency level based on evidence."""
        if not evidence_data:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for evidence in evidence_data:
            score = evidence.get("score", 0.0)
            weight = evidence.get("weight", 1.0)
            max_score = evidence.get("max_score", 5.0)
            
            # Normalize score to 0-5 scale
            normalized_score = (score / max_score) * 5.0 if max_score > 0 else 0.0
            
            total_score += normalized_score * weight
            total_weight += weight
        
        competency_level = total_score / total_weight if total_weight > 0 else 0.0
        
        # Apply assessment method modifier
        if assessment_method == "peer_review":
            competency_level *= 1.1  # Peer review gets slight boost
        elif assessment_method == "self_assessment":
            competency_level *= 0.9  # Self assessment gets slight reduction
        elif assessment_method == "automated_test":
            competency_level *= 1.0  # Automated tests are neutral
        
        return min(competency_level, 5.0)  # Cap at 5.0
    
    async def _get_latest_competency_assessment(
        self,
        user_id: UUID,
        skill_area: str
    ) -> Optional[CompetencyAssessment]:
        """Get the latest competency assessment for a skill area."""
        result = await self.db.execute(
            select(CompetencyAssessment)
            .where(
                and_(
                    CompetencyAssessment.user_id == user_id,
                    CompetencyAssessment.skill_area == skill_area
                )
            )
            .order_by(desc(CompetencyAssessment.assessment_date))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def _calculate_assessment_confidence(
        self,
        evidence_data: List[Dict[str, Any]],
        assessment_method: str
    ) -> float:
        """Calculate confidence score for competency assessment."""
        if not evidence_data:
            return 0.0
        
        # Base confidence on number and quality of evidence
        evidence_count = len(evidence_data)
        quality_scores = [e.get("quality", 1.0) for e in evidence_data]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        # Calculate confidence based on evidence count and quality
        count_factor = min(evidence_count / 5.0, 1.0)  # Normalize to 5 pieces of evidence
        quality_factor = avg_quality
        
        base_confidence = (count_factor + quality_factor) / 2.0
        
        # Adjust based on assessment method
        method_multipliers = {
            "automated_test": 1.0,
            "peer_review": 0.9,
            "self_assessment": 0.7,
            "project_evaluation": 0.8,
            "portfolio_review": 0.85
        }
        
        method_multiplier = method_multipliers.get(assessment_method, 0.8)
        confidence_score = base_confidence * method_multiplier
        
        return min(confidence_score, 1.0)