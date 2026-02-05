"""Privacy control and data boundary API endpoints."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..database import get_async_db
from ..models.user import User
from ..models.privacy import (
    PrivacySetting, DataBoundary, ConsentRecord, PrivacyViolation, PrivacyAuditLog,
    PrivacyScope, ComplianceFramework, DataProcessingPurpose, PrivacyViolationType
)
from ..schemas.privacy import (
    CreatePrivacySettingRequest, UpdatePrivacySettingRequest, PrivacySettingResponse, PrivacySettingListResponse,
    CreateDataBoundaryRequest, DataBoundaryResponse, DataBoundaryListResponse,
    RecordConsentRequest, ConsentRecordResponse, ConsentRecordListResponse,
    CheckDataAccessRequest, DataAccessCheckResponse,
    ReportViolationRequest, PrivacyViolationResponse, PrivacyViolationListResponse,
    PrivacyAuditLogResponse, PrivacyAuditLogListResponse,
    PrivacyComplianceStatusResponse, PrivacyDashboardResponse
)
from ..models.privacy import DataSharingLevel, DataProcessingPurpose
from ..services.privacy_service import privacy_service
from ..services.auth import get_current_user

router = APIRouter(prefix="/privacy", tags=["privacy"])


# Privacy Settings Management
@router.post("/settings", response_model=PrivacySettingResponse, status_code=status.HTTP_201_CREATED)
async def create_privacy_setting(
    request: CreatePrivacySettingRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new privacy setting."""
    try:
        privacy_setting = await privacy_service.create_privacy_setting(
            db=db,
            user_id=current_user.id,
            setting_name=request.setting_name,
            category=request.category,
            scope=request.scope,
            setting_value=request.setting_value,
            data_sharing_level=request.data_sharing_level,
            allowed_purposes=request.allowed_purposes,
            blocked_purposes=request.blocked_purposes,
            description=request.description
        )
        
        # Apply additional settings
        if request.workspace_restrictions:
            privacy_setting.workspace_restrictions = request.workspace_restrictions
        if request.project_restrictions:
            privacy_setting.project_restrictions = request.project_restrictions
        if request.content_type_restrictions:
            privacy_setting.content_type_restrictions = request.content_type_restrictions
        if request.expires_at:
            privacy_setting.expires_at = request.expires_at
        if request.compliance_frameworks:
            privacy_setting.compliance_frameworks = [f.value for f in request.compliance_frameworks]
        
        await db.commit()
        await db.refresh(privacy_setting)
        
        return PrivacySettingResponse.from_orm(privacy_setting)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create privacy setting: {str(e)}"
        )


@router.get("/settings", response_model=PrivacySettingListResponse)
async def list_privacy_settings(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    scope: Optional[PrivacyScope] = Query(None, description="Filter by scope"),
    category: Optional[str] = Query(None, description="Filter by category"),
    is_enabled: Optional[bool] = Query(None, description="Filter by enabled status"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """List user's privacy settings."""
    try:
        # Build query
        query = select(PrivacySetting).where(PrivacySetting.user_id == current_user.id)
        
        if scope:
            query = query.where(PrivacySetting.scope == scope)
        if category:
            query = query.where(PrivacySetting.category == category)
        if is_enabled is not None:
            query = query.where(PrivacySetting.is_enabled == is_enabled)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        settings = result.scalars().all()
        
        return PrivacySettingListResponse(
            settings=[PrivacySettingResponse.from_orm(setting) for setting in settings],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list privacy settings: {str(e)}"
        )


@router.get("/settings/{setting_id}", response_model=PrivacySettingResponse)
async def get_privacy_setting(
    setting_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Get privacy setting by ID."""
    try:
        result = await db.execute(
            select(PrivacySetting).where(
                and_(
                    PrivacySetting.id == setting_id,
                    PrivacySetting.user_id == current_user.id
                )
            )
        )
        setting = result.scalar_one_or_none()
        
        if not setting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Privacy setting not found"
            )
        
        return PrivacySettingResponse.from_orm(setting)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get privacy setting: {str(e)}"
        )


@router.put("/settings/{setting_id}", response_model=PrivacySettingResponse)
async def update_privacy_setting(
    setting_id: UUID,
    request: UpdatePrivacySettingRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Update privacy setting."""
    try:
        # Convert request to dict, excluding None values
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        updated_setting = await privacy_service.update_privacy_setting(
            db=db,
            user_id=current_user.id,
            setting_id=setting_id,
            updates=updates
        )
        
        if not updated_setting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Privacy setting not found"
            )
        
        return PrivacySettingResponse.from_orm(updated_setting)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update privacy setting: {str(e)}"
        )


# Data Boundary Management
@router.post("/boundaries", response_model=DataBoundaryResponse, status_code=status.HTTP_201_CREATED)
async def create_data_boundary(
    request: CreateDataBoundaryRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new data boundary."""
    try:
        data_boundary = await privacy_service.create_data_boundary(
            db=db,
            user_id=current_user.id,
            boundary_name=request.boundary_name,
            boundary_type=request.boundary_type,
            scope=request.scope,
            inclusion_rules=request.inclusion_rules,
            exclusion_rules=request.exclusion_rules,
            data_types=request.data_types,
            description=request.description,
            access_rules=request.access_rules,
            geographic_restrictions=request.geographic_restrictions
        )
        
        # Apply additional settings
        if request.sensitivity_levels:
            data_boundary.sensitivity_levels = request.sensitivity_levels
        if request.classification_tags:
            data_boundary.classification_tags = request.classification_tags
        if request.jurisdictional_requirements:
            data_boundary.jurisdictional_requirements = request.jurisdictional_requirements
        if request.enforcement_level:
            data_boundary.enforcement_level = request.enforcement_level
        if request.violation_actions:
            data_boundary.violation_actions = request.violation_actions
        if request.effective_date:
            data_boundary.effective_date = request.effective_date
        if request.expiration_date:
            data_boundary.expiration_date = request.expiration_date
        if request.compliance_frameworks:
            data_boundary.compliance_frameworks = [f.value for f in request.compliance_frameworks]
        
        await db.commit()
        await db.refresh(data_boundary)
        
        return DataBoundaryResponse.from_orm(data_boundary)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create data boundary: {str(e)}"
        )


@router.get("/boundaries", response_model=DataBoundaryListResponse)
async def list_data_boundaries(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    boundary_type: Optional[str] = Query(None, description="Filter by boundary type"),
    scope: Optional[PrivacyScope] = Query(None, description="Filter by scope"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """List user's data boundaries."""
    try:
        # Build query
        query = select(DataBoundary).where(DataBoundary.user_id == current_user.id)
        
        if boundary_type:
            query = query.where(DataBoundary.boundary_type == boundary_type)
        if scope:
            query = query.where(DataBoundary.scope == scope)
        if is_active is not None:
            query = query.where(DataBoundary.is_active == is_active)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        boundaries = result.scalars().all()
        
        return DataBoundaryListResponse(
            boundaries=[DataBoundaryResponse.from_orm(boundary) for boundary in boundaries],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list data boundaries: {str(e)}"
        )


# Consent Management
@router.post("/consent", response_model=ConsentRecordResponse, status_code=status.HTTP_201_CREATED)
async def record_consent(
    request: RecordConsentRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Record user consent for data processing."""
    try:
        consent_record = await privacy_service.record_consent(
            db=db,
            user_id=current_user.id,
            purpose=request.purpose,
            data_types=request.data_types,
            description=request.description,
            is_granted=request.is_granted,
            consent_method=request.consent_method,
            is_explicit=request.is_explicit,
            expires_at=request.expires_at,
            legal_basis=request.legal_basis
        )
        
        return ConsentRecordResponse.from_orm(consent_record)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record consent: {str(e)}"
        )


@router.delete("/consent/{consent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_consent(
    consent_id: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Revoke user consent."""
    try:
        success = await privacy_service.revoke_consent(
            db=db,
            user_id=current_user.id,
            consent_id=consent_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Consent record not found or already revoked"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke consent: {str(e)}"
        )


@router.get("/consent", response_model=ConsentRecordListResponse)
async def list_consent_records(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    purpose: Optional[DataProcessingPurpose] = Query(None, description="Filter by purpose"),
    is_granted: Optional[bool] = Query(None, description="Filter by granted status"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """List user's consent records."""
    try:
        # Build query
        query = select(ConsentRecord).where(ConsentRecord.user_id == current_user.id)
        
        if purpose:
            query = query.where(ConsentRecord.purpose == purpose)
        if is_granted is not None:
            query = query.where(ConsentRecord.is_granted == is_granted)
        
        # Order by most recent first
        query = query.order_by(ConsentRecord.granted_at.desc())
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        consents = result.scalars().all()
        
        return ConsentRecordListResponse(
            consents=[ConsentRecordResponse.from_orm(consent) for consent in consents],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list consent records: {str(e)}"
        )


# Data Access Control
@router.post("/check-access", response_model=DataAccessCheckResponse)
async def check_data_access(
    request: CheckDataAccessRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Check if data access is permitted."""
    try:
        allowed, violations = await privacy_service.check_data_access_permission(
            db=db,
            user_id=current_user.id,
            data_type=request.data_type,
            operation=request.operation,
            purpose=request.purpose,
            context=request.context
        )
        
        # Get counts for response
        privacy_settings = await privacy_service.get_user_privacy_settings(db, current_user.id)
        boundaries = await privacy_service.get_user_data_boundaries(db, current_user.id)
        
        return DataAccessCheckResponse(
            allowed=allowed,
            violations=violations,
            privacy_settings_checked=len(privacy_settings),
            boundaries_checked=len(boundaries),
            consent_verified=allowed  # Simplified - actual implementation would be more detailed
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check data access: {str(e)}"
        )


# Privacy Violations
@router.post("/violations", response_model=PrivacyViolationResponse, status_code=status.HTTP_201_CREATED)
async def report_privacy_violation(
    request: ReportViolationRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Report a privacy violation."""
    try:
        violation = await privacy_service.detect_privacy_violation(
            db=db,
            violation_type=request.violation_type,
            description=request.description,
            severity_level=request.severity_level,
            affected_user_id=request.affected_user_id,
            source_user_id=current_user.id,
            source_component=request.source_component,
            source_operation=request.source_operation,
            affected_data_types=request.affected_data_types,
            violation_context=request.violation_context
        )
        
        return PrivacyViolationResponse.from_orm(violation)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to report privacy violation: {str(e)}"
        )


@router.get("/violations", response_model=PrivacyViolationListResponse)
async def list_privacy_violations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    violation_type: Optional[PrivacyViolationType] = Query(None, description="Filter by violation type"),
    severity_level: Optional[str] = Query(None, description="Filter by severity level"),
    is_resolved: Optional[bool] = Query(None, description="Filter by resolution status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """List privacy violations."""
    try:
        # Build query - show violations where user is affected or is the source
        query = select(PrivacyViolation).where(
            or_(
                PrivacyViolation.user_id == current_user.id,
                PrivacyViolation.source_user_id == current_user.id
            )
        )
        
        if violation_type:
            query = query.where(PrivacyViolation.violation_type == violation_type)
        if severity_level:
            query = query.where(PrivacyViolation.severity_level == severity_level)
        if is_resolved is not None:
            query = query.where(PrivacyViolation.is_resolved == is_resolved)
        if start_date:
            query = query.where(PrivacyViolation.detected_at >= start_date)
        if end_date:
            query = query.where(PrivacyViolation.detected_at <= end_date)
        
        # Order by most recent first
        query = query.order_by(PrivacyViolation.detected_at.desc())
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        violations = result.scalars().all()
        
        return PrivacyViolationListResponse(
            violations=[PrivacyViolationResponse.from_orm(violation) for violation in violations],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list privacy violations: {str(e)}"
        )


# Privacy Audit and Compliance
@router.get("/audit", response_model=PrivacyAuditLogListResponse)
async def list_privacy_audit_logs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    event_category: Optional[str] = Query(None, description="Filter by event category"),
    privacy_compliant: Optional[bool] = Query(None, description="Filter by compliance status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """List privacy audit logs."""
    try:
        # Build query
        query = select(PrivacyAuditLog).where(
            or_(
                PrivacyAuditLog.user_id == current_user.id,
                PrivacyAuditLog.user_id.is_(None)  # System events
            )
        )
        
        if event_type:
            query = query.where(PrivacyAuditLog.event_type == event_type)
        if event_category:
            query = query.where(PrivacyAuditLog.event_category == event_category)
        if privacy_compliant is not None:
            query = query.where(PrivacyAuditLog.privacy_compliant == privacy_compliant)
        if start_date:
            query = query.where(PrivacyAuditLog.timestamp >= start_date)
        if end_date:
            query = query.where(PrivacyAuditLog.timestamp <= end_date)
        
        # Order by most recent first
        query = query.order_by(PrivacyAuditLog.timestamp.desc())
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        logs = result.scalars().all()
        
        return PrivacyAuditLogListResponse(
            logs=[PrivacyAuditLogResponse.from_orm(log) for log in logs],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list privacy audit logs: {str(e)}"
        )


@router.get("/compliance", response_model=PrivacyComplianceStatusResponse)
async def get_privacy_compliance_status(
    framework: Optional[ComplianceFramework] = Query(None, description="Compliance framework"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Get privacy compliance status."""
    try:
        compliance_status = await privacy_service.get_privacy_compliance_status(
            db=db,
            user_id=current_user.id,
            framework=framework
        )
        
        # Add recommendations based on compliance score
        recommendations = []
        if compliance_status["compliance_score"] < 70:
            recommendations.append("Review and update privacy settings")
        if compliance_status["unresolved_violations"] > 0:
            recommendations.append("Address unresolved privacy violations")
        if compliance_status["active_consents"] < 3:
            recommendations.append("Review consent requirements for data processing")
        
        compliance_status["recommendations"] = recommendations
        
        return PrivacyComplianceStatusResponse(**compliance_status)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get privacy compliance status: {str(e)}"
        )


@router.get("/dashboard", response_model=PrivacyDashboardResponse)
async def get_privacy_dashboard(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Get privacy dashboard overview."""
    try:
        # Get counts
        settings_count = len(await privacy_service.get_user_privacy_settings(db, current_user.id))
        boundaries_count = len(await privacy_service.get_user_data_boundaries(db, current_user.id))
        
        # Get active consents count
        consents_result = await db.execute(
            select(func.count()).select_from(ConsentRecord).where(
                and_(
                    ConsentRecord.user_id == current_user.id,
                    ConsentRecord.is_granted == True,
                    or_(
                        ConsentRecord.expires_at.is_(None),
                        ConsentRecord.expires_at > datetime.utcnow()
                    )
                )
            )
        )
        active_consents = consents_result.scalar()
        
        # Get recent violations count
        violations_result = await db.execute(
            select(func.count()).select_from(PrivacyViolation).where(
                and_(
                    PrivacyViolation.user_id == current_user.id,
                    PrivacyViolation.detected_at >= datetime.utcnow() - timedelta(days=30)
                )
            )
        )
        recent_violations = violations_result.scalar()
        
        # Get compliance status
        compliance_status = await privacy_service.get_privacy_compliance_status(db, current_user.id)
        
        # Get default data sharing level (from most recent setting)
        settings = await privacy_service.get_user_privacy_settings(db, current_user.id)
        data_sharing_level = settings[0].data_sharing_level if settings else "selective"
        
        return PrivacyDashboardResponse(
            user_id=str(current_user.id),
            privacy_settings_count=settings_count,
            data_boundaries_count=boundaries_count,
            active_consents_count=active_consents,
            recent_violations_count=recent_violations,
            compliance_score=compliance_status["compliance_score"],
            data_sharing_level=data_sharing_level,
            privacy_alerts=[]  # Would be populated with actual alerts
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get privacy dashboard: {str(e)}"
        )


@router.post("/profiles/granular", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_granular_privacy_profile(
    profile_name: str,
    data_categories: List[str],
    processing_purposes: List[DataProcessingPurpose],
    sharing_preferences: Dict[str, DataSharingLevel],
    retention_preferences: Dict[str, int],
    geographic_restrictions: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Create a comprehensive granular privacy profile."""
    try:
        created_settings = await privacy_service.create_granular_privacy_profile(
            db=db,
            user_id=current_user.id,
            profile_name=profile_name,
            data_categories=data_categories,
            processing_purposes=processing_purposes,
            sharing_preferences=sharing_preferences,
            retention_preferences=retention_preferences,
            geographic_restrictions=geographic_restrictions
        )
        
        return {
            "profile_name": profile_name,
            "created_settings": {k: str(v.id) for k, v in created_settings.items()},
            "settings_count": len(created_settings),
            "message": f"Granular privacy profile '{profile_name}' created successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create granular privacy profile: {str(e)}"
        )


@router.put("/preferences/bulk", response_model=Dict[str, Any])
async def update_privacy_preferences_bulk(
    preference_updates: Dict[str, Any],
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Update multiple privacy preferences in bulk."""
    try:
        results = await privacy_service.update_privacy_preferences_bulk(
            db=db,
            user_id=current_user.id,
            preference_updates=preference_updates
        )
        
        return {
            "user_id": str(current_user.id),
            "update_results": results,
            "success": len(results["validation_errors"]) == 0,
            "message": "Bulk privacy preferences update completed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update privacy preferences: {str(e)}"
        )


@router.post("/boundaries/enforce", response_model=Dict[str, Any])
async def enforce_data_boundary_real_time(
    data_access_request: Dict[str, Any],
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Perform real-time data boundary enforcement check."""
    try:
        allowed, violations, enforcement_details = await privacy_service.enforce_data_boundary_real_time(
            db=db,
            user_id=current_user.id,
            data_access_request=data_access_request
        )
        
        return {
            "user_id": str(current_user.id),
            "access_allowed": allowed,
            "violations": violations,
            "enforcement_details": enforcement_details,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enforce data boundaries: {str(e)}"
        )


@router.get("/compliance/detailed", response_model=Dict[str, Any])
async def get_detailed_privacy_compliance(
    framework: Optional[ComplianceFramework] = Query(None, description="Compliance framework"),
    include_recommendations: bool = Query(True, description="Include compliance recommendations"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed privacy compliance status with framework-specific checks."""
    try:
        compliance_status = await privacy_service.get_privacy_compliance_status(
            db=db,
            user_id=current_user.id,
            framework=framework
        )
        
        # Add recommendations if requested
        if include_recommendations:
            recommendations = []
            
            if compliance_status["compliance_score"] < 70:
                recommendations.append("Review and update privacy settings to improve compliance")
            
            if compliance_status["unresolved_violations"] > 0:
                recommendations.append("Address unresolved privacy violations immediately")
            
            if compliance_status["active_consents"] < 3:
                recommendations.append("Review consent requirements for data processing activities")
            
            if compliance_status["active_data_boundaries"] == 0:
                recommendations.append("Consider creating data boundaries to protect sensitive information")
            
            # Framework-specific recommendations
            if framework == ComplianceFramework.GDPR:
                gdpr_details = compliance_status.get("compliance_details", {}).get("framework_specific", {})
                if not gdpr_details.get("explicit_consent", False):
                    recommendations.append("Ensure explicit consent is obtained for all data processing")
                if not gdpr_details.get("data_minimization", False):
                    recommendations.append("Implement data minimization boundaries")
                if not gdpr_details.get("data_portability", False):
                    recommendations.append("Enable data portability settings")
            
            compliance_status["recommendations"] = recommendations
        
        return compliance_status
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get detailed compliance status: {str(e)}"
        )

