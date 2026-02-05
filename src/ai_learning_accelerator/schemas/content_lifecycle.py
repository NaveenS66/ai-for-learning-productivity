"""Schemas for content lifecycle management."""

from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field

from ..services.content_lifecycle import ValidationSeverity


class ValidationIssueResponse(BaseModel):
    """Response model for validation issues."""
    severity: ValidationSeverity
    category: str
    message: str
    suggestion: Optional[str] = None
    location: Optional[str] = None


class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score (0.0-1.0)")
    category_scores: Dict[str, float] = Field(..., description="Quality scores by category")
    issues: List[ValidationIssueResponse] = Field(..., description="Validation issues found")
    recommendations: List[str] = Field(..., description="Quality improvement recommendations")
    is_publishable: bool = Field(..., description="Whether content is ready for publication")


class DeprecationAlertResponse(BaseModel):
    """Response model for deprecation alerts."""
    content_id: UUID
    reason: str = Field(..., description="Reason for deprecation alert")
    severity: ValidationSeverity
    suggested_action: str = Field(..., description="Suggested action to take")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in deprecation assessment")


class UpdateSuggestionResponse(BaseModel):
    """Response model for update suggestions."""
    content_id: UUID
    suggestion_type: str = Field(..., description="Type of update suggestion")
    description: str = Field(..., description="Description of suggested update")
    priority: int = Field(..., ge=1, le=5, description="Priority level (1-5, 5 being highest)")
    estimated_effort: str = Field(..., description="Estimated effort level (low/medium/high)")
    rationale: str = Field(..., description="Rationale for the suggestion")


class ContentAssessmentSummary(BaseModel):
    """Summary of individual content assessment."""
    content_id: str
    title: str
    quality_score: float = Field(..., ge=0.0, le=1.0)
    is_publishable: bool
    issues_count: int = Field(..., ge=0)


class BulkQualityAssessmentResponse(BaseModel):
    """Response model for bulk quality assessment."""
    processed_count: int = Field(..., ge=0, description="Number of content items processed")
    updated_count: int = Field(..., ge=0, description="Number of content items updated")
    average_quality_score: float = Field(..., ge=0.0, le=1.0, description="Average quality score")
    publishable_count: int = Field(..., ge=0, description="Number of publishable content items")
    assessments: List[ContentAssessmentSummary] = Field(..., description="Individual assessment summaries")
    error: Optional[str] = Field(None, description="Error message if assessment failed")


class ContentValidationRequest(BaseModel):
    """Request model for content validation."""
    check_structure: bool = Field(default=True, description="Check basic content structure")
    check_quality: bool = Field(default=True, description="Check content quality aspects")
    check_metadata: bool = Field(default=True, description="Check metadata completeness")
    check_format: bool = Field(default=True, description="Check format-specific requirements")


class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment."""
    include_recommendations: bool = Field(default=True, description="Include improvement recommendations")
    detailed_scoring: bool = Field(default=True, description="Include detailed category scoring")
    check_user_feedback: bool = Field(default=True, description="Include user feedback analysis")


class DeprecationDetectionRequest(BaseModel):
    """Request model for deprecation detection."""
    age_threshold_months: Optional[int] = Field(default=12, ge=1, le=60, description="Age threshold in months")
    rating_threshold: Optional[float] = Field(default=2.5, ge=1.0, le=5.0, description="Minimum rating threshold")
    engagement_threshold: Optional[float] = Field(default=0.1, ge=0.0, le=1.0, description="Minimum engagement rate")
    include_accuracy_analysis: bool = Field(default=True, description="Include accuracy complaint analysis")


class UpdateSuggestionRequest(BaseModel):
    """Request model for update suggestions."""
    analyze_feedback: bool = Field(default=True, description="Analyze user feedback for suggestions")
    check_freshness: bool = Field(default=True, description="Check content freshness")
    analyze_engagement: bool = Field(default=True, description="Analyze engagement patterns")
    max_suggestions: int = Field(default=10, ge=1, le=50, description="Maximum number of suggestions to return")


class ContentLifecycleStats(BaseModel):
    """Statistics for content lifecycle management."""
    total_content_count: int = Field(..., ge=0)
    published_count: int = Field(..., ge=0)
    draft_count: int = Field(..., ge=0)
    review_count: int = Field(..., ge=0)
    deprecated_count: int = Field(..., ge=0)
    average_quality_score: float = Field(..., ge=0.0, le=1.0)
    content_needing_review: int = Field(..., ge=0)
    outdated_content_count: int = Field(..., ge=0)


class ContentMaintenanceReport(BaseModel):
    """Comprehensive content maintenance report."""
    report_date: str = Field(..., description="Report generation date")
    stats: ContentLifecycleStats
    deprecation_alerts: List[DeprecationAlertResponse]
    quality_issues: List[ValidationIssueResponse]
    update_suggestions: List[UpdateSuggestionResponse]
    recommendations: List[str] = Field(..., description="Overall maintenance recommendations")