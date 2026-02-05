"""API endpoints for content lifecycle management."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..services.content_lifecycle import ContentLifecycleService
from ..services.content import ContentService
from ..schemas.content_lifecycle import (
    QualityAssessmentResponse,
    ValidationIssueResponse,
    DeprecationAlertResponse,
    UpdateSuggestionResponse,
    BulkQualityAssessmentResponse
)
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/content-lifecycle", tags=["content-lifecycle"])


@router.post("/validate/{content_id}", response_model=List[ValidationIssueResponse])
async def validate_content(
    content_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Validate content and return validation issues."""
    try:
        content_service = ContentService(db)
        lifecycle_service = ContentLifecycleService(db)
        
        # Get content
        content = await content_service.get_learning_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Validate content
        issues = await lifecycle_service.validate_content(content)
        
        return [
            ValidationIssueResponse(
                severity=issue.severity,
                category=issue.category,
                message=issue.message,
                suggestion=issue.suggestion,
                location=issue.location
            )
            for issue in issues
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error validating content", error=str(e), content_id=str(content_id))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/assess-quality/{content_id}", response_model=QualityAssessmentResponse)
async def assess_content_quality(
    content_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Perform comprehensive quality assessment of content."""
    try:
        content_service = ContentService(db)
        lifecycle_service = ContentLifecycleService(db)
        
        # Get content
        content = await content_service.get_learning_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Assess quality
        assessment = await lifecycle_service.assess_content_quality(content)
        
        return QualityAssessmentResponse(
            overall_score=assessment.overall_score,
            category_scores=assessment.category_scores,
            issues=[
                ValidationIssueResponse(
                    severity=issue.severity,
                    category=issue.category,
                    message=issue.message,
                    suggestion=issue.suggestion,
                    location=issue.location
                )
                for issue in assessment.issues
            ],
            recommendations=assessment.recommendations,
            is_publishable=assessment.is_publishable
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error assessing content quality", error=str(e), content_id=str(content_id))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/deprecated-content", response_model=List[DeprecationAlertResponse])
async def get_deprecated_content(
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """Get list of content that may need deprecation or updates."""
    try:
        lifecycle_service = ContentLifecycleService(db)
        
        # Detect deprecated content
        alerts = await lifecycle_service.detect_deprecated_content(limit=limit)
        
        return [
            DeprecationAlertResponse(
                content_id=alert.content_id,
                reason=alert.reason,
                severity=alert.severity,
                suggested_action=alert.suggested_action,
                confidence=alert.confidence
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error("Error detecting deprecated content", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/update-suggestions/{content_id}", response_model=List[UpdateSuggestionResponse])
async def get_update_suggestions(
    content_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get update suggestions for specific content."""
    try:
        content_service = ContentService(db)
        lifecycle_service = ContentLifecycleService(db)
        
        # Check if content exists
        content = await content_service.get_learning_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Generate update suggestions
        suggestions = await lifecycle_service.generate_update_suggestions(content_id)
        
        return [
            UpdateSuggestionResponse(
                content_id=suggestion.content_id,
                suggestion_type=suggestion.suggestion_type,
                description=suggestion.description,
                priority=suggestion.priority,
                estimated_effort=suggestion.estimated_effort,
                rationale=suggestion.rationale
            )
            for suggestion in suggestions
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating update suggestions", error=str(e), content_id=str(content_id))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/update-quality-score/{content_id}")
async def update_content_quality_score(
    content_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Update quality score for specific content."""
    try:
        content_service = ContentService(db)
        lifecycle_service = ContentLifecycleService(db)
        
        # Check if content exists
        content = await content_service.get_learning_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Update quality score
        success = await lifecycle_service.update_content_quality_score(content_id)
        
        if success:
            return {"message": "Quality score updated successfully", "content_id": str(content_id)}
        else:
            raise HTTPException(status_code=500, detail="Failed to update quality score")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating quality score", error=str(e), content_id=str(content_id))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/bulk-quality-assessment", response_model=BulkQualityAssessmentResponse)
async def bulk_quality_assessment(
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db)
):
    """Perform bulk quality assessment on content."""
    try:
        lifecycle_service = ContentLifecycleService(db)
        
        # Perform bulk assessment
        result = await lifecycle_service.bulk_quality_assessment(limit=limit)
        
        return BulkQualityAssessmentResponse(**result)
        
    except Exception as e:
        logger.error("Error in bulk quality assessment", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")