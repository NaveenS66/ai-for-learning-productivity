"""Privacy control and data boundary API schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ..models.privacy import (
    DataSharingLevel, PrivacyScope, ComplianceFramework,
    DataProcessingPurpose, PrivacyViolationType
)


# Base schemas
class PrivacySettingBase(BaseModel):
    """Base schema for privacy settings."""
    setting_name: str = Field(..., description="Name of the privacy setting")
    scope: PrivacyScope = Field(..., description="Scope of the setting")
    category: str = Field(..., description="Category of privacy setting")
    description: Optional[str] = Field(None, description="Description of the setting")


class DataBoundaryBase(BaseModel):
    """Base schema for data boundaries."""
    boundary_name: str = Field(..., description="Name of the data boundary")
    boundary_type: str = Field(..., description="Type of boundary")
    scope: PrivacyScope = Field(..., description="Boundary scope")
    description: str = Field(..., description="Boundary description")


class ConsentRecordBase(BaseModel):
    """Base schema for consent records."""
    purpose: DataProcessingPurpose = Field(..., description="Purpose of data processing")
    description: str = Field(..., description="Description of what user is consenting to")
    data_types: List[str] = Field(..., description="Types of data covered by consent")


# Request schemas
class CreatePrivacySettingRequest(PrivacySettingBase):
    """Request schema for creating privacy settings."""
    setting_value: Dict[str, Any] = Field(..., description="Setting value and configuration")
    data_sharing_level: DataSharingLevel = Field(DataSharingLevel.SELECTIVE, description="Data sharing level")
    allowed_purposes: Optional[List[DataProcessingPurpose]] = Field(default_factory=list, description="Allowed data processing purposes")
    blocked_purposes: Optional[List[DataProcessingPurpose]] = Field(default_factory=list, description="Blocked data processing purposes")
    workspace_restrictions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Workspace-specific restrictions")
    project_restrictions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Project-specific restrictions")
    content_type_restrictions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Content type restrictions")
    expires_at: Optional[datetime] = Field(None, description="Setting expiration timestamp")
    compliance_frameworks: Optional[List[ComplianceFramework]] = Field(default_factory=list, description="Applicable compliance frameworks")


class UpdatePrivacySettingRequest(BaseModel):
    """Request schema for updating privacy settings."""
    setting_value: Optional[Dict[str, Any]] = Field(None, description="Setting value and configuration")
    data_sharing_level: Optional[DataSharingLevel] = Field(None, description="Data sharing level")
    allowed_purposes: Optional[List[DataProcessingPurpose]] = Field(None, description="Allowed data processing purposes")
    blocked_purposes: Optional[List[DataProcessingPurpose]] = Field(None, description="Blocked data processing purposes")
    is_enabled: Optional[bool] = Field(None, description="Setting is enabled")
    workspace_restrictions: Optional[Dict[str, Any]] = Field(None, description="Workspace-specific restrictions")
    project_restrictions: Optional[Dict[str, Any]] = Field(None, description="Project-specific restrictions")
    content_type_restrictions: Optional[Dict[str, Any]] = Field(None, description="Content type restrictions")
    expires_at: Optional[datetime] = Field(None, description="Setting expiration timestamp")


class CreateDataBoundaryRequest(DataBoundaryBase):
    """Request schema for creating data boundaries."""
    inclusion_rules: Dict[str, Any] = Field(..., description="Rules for what data is included")
    exclusion_rules: Dict[str, Any] = Field(..., description="Rules for what data is excluded")
    data_types: List[str] = Field(..., description="Types of data covered by boundary")
    access_rules: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Access control rules")
    sensitivity_levels: Optional[List[str]] = Field(default_factory=list, description="Data sensitivity levels")
    classification_tags: Optional[List[str]] = Field(default_factory=list, description="Data classification tags")
    geographic_restrictions: Optional[List[str]] = Field(default_factory=list, description="Geographic restrictions")
    jurisdictional_requirements: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Jurisdictional requirements")
    enforcement_level: str = Field("strict", description="Enforcement level")
    violation_actions: Optional[List[str]] = Field(default_factory=list, description="Actions to take on violations")
    effective_date: datetime = Field(default_factory=datetime.utcnow, description="When boundary becomes effective")
    expiration_date: Optional[datetime] = Field(None, description="Boundary expiration date")
    compliance_frameworks: Optional[List[ComplianceFramework]] = Field(default_factory=list, description="Applicable compliance frameworks")


class RecordConsentRequest(ConsentRecordBase):
    """Request schema for recording consent."""
    is_granted: bool = Field(..., description="Consent is granted")
    consent_method: str = Field(..., description="Method of consent collection")
    is_explicit: bool = Field(True, description="Consent was explicitly given")
    expires_at: Optional[datetime] = Field(None, description="When consent expires")
    legal_basis: Optional[str] = Field(None, description="Legal basis for processing")
    consent_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Context when consent was given")


class CheckDataAccessRequest(BaseModel):
    """Request schema for checking data access permission."""
    data_type: str = Field(..., description="Type of data being accessed")
    operation: str = Field(..., description="Operation being performed")
    purpose: DataProcessingPurpose = Field(..., description="Purpose of data processing")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class ReportViolationRequest(BaseModel):
    """Request schema for reporting privacy violations."""
    violation_type: PrivacyViolationType = Field(..., description="Type of violation")
    description: str = Field(..., description="Description of the violation")
    severity_level: str = Field(..., description="Severity level")
    affected_user_id: Optional[UUID] = Field(None, description="Affected user")
    source_component: Optional[str] = Field(None, description="Component that caused violation")
    source_operation: Optional[str] = Field(None, description="Operation that caused violation")
    affected_data_types: Optional[List[str]] = Field(default_factory=list, description="Types of data affected")
    violation_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Context of the violation")


# Response schemas
class PrivacySettingResponse(PrivacySettingBase):
    """Response schema for privacy settings."""
    id: UUID = Field(..., description="Setting ID")
    user_id: UUID = Field(..., description="User who owns the settings")
    setting_value: Dict[str, Any] = Field(..., description="Setting value and configuration")
    default_value: Dict[str, Any] = Field(default_factory=dict, description="Default value for the setting")
    is_enabled: bool = Field(..., description="Setting is enabled")
    data_sharing_level: DataSharingLevel = Field(..., description="Data sharing level")
    allowed_purposes: List[str] = Field(default_factory=list, description="Allowed data processing purposes")
    blocked_purposes: List[str] = Field(default_factory=list, description="Blocked data processing purposes")
    workspace_restrictions: Dict[str, Any] = Field(default_factory=dict, description="Workspace-specific restrictions")
    project_restrictions: Dict[str, Any] = Field(default_factory=dict, description="Project-specific restrictions")
    content_type_restrictions: Dict[str, Any] = Field(default_factory=dict, description="Content type restrictions")
    created_at: datetime = Field(..., description="Setting creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Setting expiration timestamp")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Applicable compliance frameworks")
    
    class Config:
        from_attributes = True


class DataBoundaryResponse(DataBoundaryBase):
    """Response schema for data boundaries."""
    id: UUID = Field(..., description="Boundary ID")
    user_id: UUID = Field(..., description="User who owns the boundary")
    inclusion_rules: Dict[str, Any] = Field(..., description="Rules for what data is included")
    exclusion_rules: Dict[str, Any] = Field(..., description="Rules for what data is excluded")
    access_rules: Dict[str, Any] = Field(default_factory=dict, description="Access control rules")
    data_types: List[str] = Field(..., description="Types of data covered by boundary")
    sensitivity_levels: List[str] = Field(default_factory=list, description="Data sensitivity levels")
    classification_tags: List[str] = Field(default_factory=list, description="Data classification tags")
    geographic_restrictions: List[str] = Field(default_factory=list, description="Geographic restrictions")
    jurisdictional_requirements: Dict[str, Any] = Field(default_factory=dict, description="Jurisdictional requirements")
    is_active: bool = Field(..., description="Boundary is active")
    enforcement_level: str = Field(..., description="Enforcement level")
    violation_actions: List[str] = Field(default_factory=list, description="Actions to take on violations")
    created_at: datetime = Field(..., description="Boundary creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    effective_date: datetime = Field(..., description="When boundary becomes effective")
    expiration_date: Optional[datetime] = Field(None, description="Boundary expiration date")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Applicable compliance frameworks")
    
    class Config:
        from_attributes = True


class ConsentRecordResponse(ConsentRecordBase):
    """Response schema for consent records."""
    id: UUID = Field(..., description="Consent record ID")
    user_id: UUID = Field(..., description="User providing consent")
    consent_id: str = Field(..., description="Unique consent identifier")
    is_granted: bool = Field(..., description="Consent is granted")
    is_explicit: bool = Field(..., description="Consent was explicitly given")
    consent_method: str = Field(..., description="Method of consent collection")
    granted_at: Optional[datetime] = Field(None, description="When consent was granted")
    revoked_at: Optional[datetime] = Field(None, description="When consent was revoked")
    expires_at: Optional[datetime] = Field(None, description="When consent expires")
    last_confirmed_at: Optional[datetime] = Field(None, description="Last consent confirmation")
    consent_context: Dict[str, Any] = Field(default_factory=dict, description="Context when consent was given")
    legal_basis: Optional[str] = Field(None, description="Legal basis for processing")
    withdrawal_method: Optional[str] = Field(None, description="Method for withdrawing consent")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Applicable compliance frameworks")
    
    class Config:
        from_attributes = True


class PrivacyViolationResponse(BaseModel):
    """Response schema for privacy violations."""
    id: UUID = Field(..., description="Violation ID")
    user_id: Optional[UUID] = Field(None, description="Affected user")
    violation_id: str = Field(..., description="Unique violation identifier")
    violation_type: PrivacyViolationType = Field(..., description="Type of violation")
    description: str = Field(..., description="Description of the violation")
    severity_level: str = Field(..., description="Severity level")
    affected_data_types: List[str] = Field(default_factory=list, description="Types of data affected")
    detected_at: datetime = Field(..., description="When violation was detected")
    detection_method: str = Field(..., description="How violation was detected")
    violation_context: Dict[str, Any] = Field(default_factory=dict, description="Context of the violation")
    source_component: Optional[str] = Field(None, description="Component that caused violation")
    source_operation: Optional[str] = Field(None, description="Operation that caused violation")
    source_user_id: Optional[UUID] = Field(None, description="User who triggered violation")
    is_resolved: bool = Field(..., description="Violation has been resolved")
    resolution_actions: List[str] = Field(default_factory=list, description="Actions taken to resolve violation")
    resolved_at: Optional[datetime] = Field(None, description="When violation was resolved")
    resolved_by: Optional[UUID] = Field(None, description="Who resolved the violation")
    impact_assessment: Dict[str, Any] = Field(default_factory=dict, description="Assessment of violation impact")
    affected_users_count: int = Field(..., description="Number of users affected")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Affected compliance frameworks")
    
    class Config:
        from_attributes = True


class PrivacyAuditLogResponse(BaseModel):
    """Response schema for privacy audit logs."""
    id: UUID = Field(..., description="Audit log ID")
    user_id: Optional[UUID] = Field(None, description="User associated with event")
    event_type: str = Field(..., description="Type of privacy event")
    event_category: str = Field(..., description="Event category")
    timestamp: datetime = Field(..., description="Event timestamp")
    description: str = Field(..., description="Event description")
    affected_data_types: List[str] = Field(default_factory=list, description="Types of data affected")
    component: Optional[str] = Field(None, description="System component involved")
    operation: Optional[str] = Field(None, description="Operation performed")
    privacy_settings_applied: Dict[str, Any] = Field(default_factory=dict, description="Privacy settings that were applied")
    boundaries_checked: List[str] = Field(default_factory=list, description="Data boundaries that were checked")
    consent_verified: Dict[str, Any] = Field(default_factory=dict, description="Consent verification results")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Applicable compliance frameworks")
    success: bool = Field(..., description="Event was successful")
    privacy_compliant: bool = Field(..., description="Event was privacy compliant")
    violations_detected: List[str] = Field(default_factory=list, description="Privacy violations detected")
    
    class Config:
        from_attributes = True


# Utility response schemas
class DataAccessCheckResponse(BaseModel):
    """Response schema for data access permission checks."""
    allowed: bool = Field(..., description="Access is allowed")
    violations: List[str] = Field(default_factory=list, description="List of violations found")
    checked_at: datetime = Field(default_factory=datetime.utcnow, description="When check was performed")
    privacy_settings_checked: int = Field(..., description="Number of privacy settings checked")
    boundaries_checked: int = Field(..., description="Number of boundaries checked")
    consent_verified: bool = Field(..., description="Consent was verified")


class PrivacyComplianceStatusResponse(BaseModel):
    """Response schema for privacy compliance status."""
    user_id: str = Field(..., description="User ID")
    compliance_score: float = Field(..., description="Overall compliance score (0-100)")
    unresolved_violations: int = Field(..., description="Number of unresolved violations")
    active_consents: int = Field(..., description="Number of active consents")
    active_privacy_settings: int = Field(..., description="Number of active privacy settings")
    last_assessed: str = Field(..., description="Last assessment timestamp")
    framework: str = Field(..., description="Compliance framework")
    recommendations: List[str] = Field(default_factory=list, description="Compliance improvement recommendations")


class PrivacyDashboardResponse(BaseModel):
    """Response schema for privacy dashboard."""
    user_id: str = Field(..., description="User ID")
    privacy_settings_count: int = Field(..., description="Total privacy settings")
    data_boundaries_count: int = Field(..., description="Total data boundaries")
    active_consents_count: int = Field(..., description="Active consents")
    recent_violations_count: int = Field(..., description="Recent violations (last 30 days)")
    compliance_score: float = Field(..., description="Overall compliance score")
    data_sharing_level: DataSharingLevel = Field(..., description="Current data sharing level")
    last_privacy_review: Optional[datetime] = Field(None, description="Last privacy review date")
    privacy_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Active privacy alerts")


# List response schemas
class PrivacySettingListResponse(BaseModel):
    """Response schema for privacy setting lists."""
    settings: List[PrivacySettingResponse] = Field(default_factory=list, description="List of privacy settings")
    total: int = Field(..., description="Total number of settings")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class DataBoundaryListResponse(BaseModel):
    """Response schema for data boundary lists."""
    boundaries: List[DataBoundaryResponse] = Field(default_factory=list, description="List of data boundaries")
    total: int = Field(..., description="Total number of boundaries")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class ConsentRecordListResponse(BaseModel):
    """Response schema for consent record lists."""
    consents: List[ConsentRecordResponse] = Field(default_factory=list, description="List of consent records")
    total: int = Field(..., description="Total number of consent records")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class PrivacyViolationListResponse(BaseModel):
    """Response schema for privacy violation lists."""
    violations: List[PrivacyViolationResponse] = Field(default_factory=list, description="List of privacy violations")
    total: int = Field(..., description="Total number of violations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class PrivacyAuditLogListResponse(BaseModel):
    """Response schema for privacy audit log lists."""
    logs: List[PrivacyAuditLogResponse] = Field(default_factory=list, description="List of audit logs")
    total: int = Field(..., description="Total number of logs")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")