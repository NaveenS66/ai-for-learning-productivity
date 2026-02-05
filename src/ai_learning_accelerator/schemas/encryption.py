"""Encryption and security API schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ..models.encryption import (
    EncryptionAlgorithm, KeyType, DataClassification, EncryptionStatus
)


# Base schemas
class EncryptionKeyBase(BaseModel):
    """Base schema for encryption keys."""
    key_name: str = Field(..., description="Unique key identifier")
    key_type: KeyType = Field(..., description="Type of encryption key")
    algorithm: EncryptionAlgorithm = Field(..., description="Encryption algorithm")
    purpose: str = Field(..., description="Purpose of the key")
    description: Optional[str] = Field(None, description="Key description")
    classification: DataClassification = Field(DataClassification.CONFIDENTIAL, description="Data classification")


class EncryptedDataBase(BaseModel):
    """Base schema for encrypted data."""
    data_name: str = Field(..., description="Data identifier")
    data_type: str = Field(..., description="Type of encrypted data")
    classification: DataClassification = Field(..., description="Data classification level")


class EncryptionOperationBase(BaseModel):
    """Base schema for encryption operations."""
    operation_type: str = Field(..., description="Type of operation")
    operation_id: str = Field(..., description="Unique operation identifier")


class DataEncryptionPolicyBase(BaseModel):
    """Base schema for data encryption policies."""
    policy_name: str = Field(..., description="Policy name")
    policy_version: str = Field(..., description="Policy version")
    description: str = Field(..., description="Policy description")
    data_types: List[str] = Field(..., description="Data types covered by policy")
    data_classifications: List[DataClassification] = Field(..., description="Data classifications covered")


# Request schemas
class CreateEncryptionKeyRequest(EncryptionKeyBase):
    """Request schema for creating encryption keys."""
    key_size: Optional[int] = Field(None, description="Key size in bits")
    expires_at: Optional[datetime] = Field(None, description="Key expiration timestamp")
    access_permissions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Key access permissions")
    usage_restrictions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Key usage restrictions")


class EncryptDataRequest(BaseModel):
    """Request schema for encrypting data."""
    data: Any = Field(..., description="Data to encrypt")
    data_name: str = Field(..., description="Data identifier")
    data_type: str = Field(..., description="Type of data")
    classification: DataClassification = Field(..., description="Data classification level")
    key_name: Optional[str] = Field(None, description="Specific key to use for encryption")
    algorithm: Optional[EncryptionAlgorithm] = Field(None, description="Encryption algorithm to use")
    compress: bool = Field(True, description="Compress data before encryption")
    access_permissions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Data access permissions")
    sharing_restrictions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Data sharing restrictions")
    expires_at: Optional[datetime] = Field(None, description="Data expiration timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DecryptDataRequest(BaseModel):
    """Request schema for decrypting data."""
    encrypted_data_id: UUID = Field(..., description="ID of encrypted data to decrypt")
    verify_permissions: bool = Field(True, description="Verify user permissions before decryption")


class RotateKeyRequest(BaseModel):
    """Request schema for key rotation."""
    key_name: str = Field(..., description="Name of key to rotate")


class CreatePolicyRequest(DataEncryptionPolicyBase):
    """Request schema for creating encryption policies."""
    user_roles: Optional[List[str]] = Field(default_factory=list, description="User roles subject to policy")
    required_algorithm: EncryptionAlgorithm = Field(..., description="Required encryption algorithm")
    minimum_key_size: int = Field(..., description="Minimum key size in bits")
    key_rotation_interval: Optional[int] = Field(None, description="Key rotation interval in days")
    encryption_at_rest: bool = Field(True, description="Require encryption at rest")
    encryption_in_transit: bool = Field(True, description="Require encryption in transit")
    encryption_in_use: bool = Field(False, description="Require encryption in use")
    access_controls: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Access control requirements")
    sharing_restrictions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Data sharing restrictions")
    compliance_frameworks: Optional[List[str]] = Field(default_factory=list, description="Applicable compliance frameworks")
    effective_date: datetime = Field(..., description="Policy effective date")
    expiration_date: Optional[datetime] = Field(None, description="Policy expiration date")


# Response schemas
class EncryptionKeyResponse(EncryptionKeyBase):
    """Response schema for encryption keys."""
    id: UUID = Field(..., description="Key ID")
    key_size: int = Field(..., description="Key size in bits")
    key_hash: str = Field(..., description="Hash of key for verification")
    created_by: UUID = Field(..., description="User who created the key")
    created_at: datetime = Field(..., description="Key creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Key expiration timestamp")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    is_active: bool = Field(..., description="Key is active and usable")
    is_revoked: bool = Field(..., description="Key has been revoked")
    version: int = Field(..., description="Key version number")
    parent_key_id: Optional[UUID] = Field(None, description="Parent key for rotation")
    access_permissions: Dict[str, Any] = Field(default_factory=dict, description="Key access permissions")
    usage_restrictions: Dict[str, Any] = Field(default_factory=dict, description="Key usage restrictions")
    
    class Config:
        from_attributes = True


class EncryptedDataResponse(EncryptedDataBase):
    """Response schema for encrypted data."""
    id: UUID = Field(..., description="Encrypted data ID")
    encryption_key_id: UUID = Field(..., description="Encryption key used")
    algorithm: EncryptionAlgorithm = Field(..., description="Encryption algorithm used")
    content_hash: str = Field(..., description="Hash of original content")
    content_size: int = Field(..., description="Size of original content")
    compressed: bool = Field(..., description="Content was compressed before encryption")
    owner_id: UUID = Field(..., description="Data owner")
    created_at: datetime = Field(..., description="Data creation timestamp")
    last_accessed_at: Optional[datetime] = Field(None, description="Last access timestamp")
    expires_at: Optional[datetime] = Field(None, description="Data expiration timestamp")
    status: EncryptionStatus = Field(..., description="Encryption status")
    integrity_verified: bool = Field(..., description="Data integrity verified")
    last_integrity_check: Optional[datetime] = Field(None, description="Last integrity check timestamp")
    access_permissions: Dict[str, Any] = Field(default_factory=dict, description="Data access permissions")
    sharing_restrictions: Dict[str, Any] = Field(default_factory=dict, description="Data sharing restrictions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        from_attributes = True


class EncryptionOperationResponse(EncryptionOperationBase):
    """Response schema for encryption operations."""
    id: UUID = Field(..., description="Operation ID")
    data_id: Optional[UUID] = Field(None, description="Associated encrypted data")
    key_id: Optional[UUID] = Field(None, description="Key used in operation")
    algorithm: Optional[EncryptionAlgorithm] = Field(None, description="Algorithm used")
    user_id: UUID = Field(..., description="User performing operation")
    status: EncryptionStatus = Field(..., description="Operation status")
    started_at: datetime = Field(..., description="Operation start time")
    completed_at: Optional[datetime] = Field(None, description="Operation completion time")
    duration_ms: Optional[int] = Field(None, description="Operation duration in milliseconds")
    success: bool = Field(..., description="Operation was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    data_size: Optional[int] = Field(None, description="Size of data processed")
    
    class Config:
        from_attributes = True


class DataEncryptionPolicyResponse(DataEncryptionPolicyBase):
    """Response schema for data encryption policies."""
    id: UUID = Field(..., description="Policy ID")
    user_roles: List[str] = Field(default_factory=list, description="User roles subject to policy")
    required_algorithm: EncryptionAlgorithm = Field(..., description="Required encryption algorithm")
    minimum_key_size: int = Field(..., description="Minimum key size in bits")
    key_rotation_interval: Optional[int] = Field(None, description="Key rotation interval in days")
    encryption_at_rest: bool = Field(..., description="Require encryption at rest")
    encryption_in_transit: bool = Field(..., description="Require encryption in transit")
    encryption_in_use: bool = Field(..., description="Require encryption in use")
    access_controls: Dict[str, Any] = Field(default_factory=dict, description="Access control requirements")
    sharing_restrictions: Dict[str, Any] = Field(default_factory=dict, description="Data sharing restrictions")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Applicable compliance frameworks")
    created_by: UUID = Field(..., description="Policy creator")
    created_at: datetime = Field(..., description="Policy creation timestamp")
    effective_date: datetime = Field(..., description="Policy effective date")
    expiration_date: Optional[datetime] = Field(None, description="Policy expiration date")
    is_active: bool = Field(..., description="Policy is active")
    is_enforced: bool = Field(..., description="Policy is enforced")
    
    class Config:
        from_attributes = True


class EncryptionAuditLogResponse(BaseModel):
    """Response schema for encryption audit logs."""
    id: UUID = Field(..., description="Audit log ID")
    event_type: str = Field(..., description="Type of audit event")
    event_category: str = Field(..., description="Event category")
    severity_level: str = Field(..., description="Event severity level")
    user_id: Optional[UUID] = Field(None, description="User associated with event")
    resource_type: str = Field(..., description="Type of resource accessed")
    resource_id: Optional[str] = Field(None, description="Resource identifier")
    operation: str = Field(..., description="Operation performed")
    timestamp: datetime = Field(..., description="Event timestamp")
    success: bool = Field(..., description="Operation was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    event_data: Dict[str, Any] = Field(default_factory=dict, description="Detailed event information")
    compliance_tags: List[str] = Field(default_factory=list, description="Compliance framework tags")
    
    class Config:
        from_attributes = True


# Utility response schemas
class DecryptDataResponse(BaseModel):
    """Response schema for data decryption."""
    data: Any = Field(..., description="Decrypted data")
    data_type: str = Field(..., description="Type of decrypted data")
    original_size: int = Field(..., description="Size of original data")
    decryption_timestamp: datetime = Field(..., description="When data was decrypted")


class EncryptionStatsResponse(BaseModel):
    """Response schema for encryption statistics."""
    total_keys: int = Field(..., description="Total number of encryption keys")
    active_keys: int = Field(..., description="Number of active keys")
    total_encrypted_data: int = Field(..., description="Total number of encrypted data records")
    total_operations: int = Field(..., description="Total number of encryption operations")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    data_by_classification: Dict[str, int] = Field(default_factory=dict, description="Data count by classification")
    operations_by_algorithm: Dict[str, int] = Field(default_factory=dict, description="Operations by algorithm")


class EncryptionHealthResponse(BaseModel):
    """Response schema for encryption system health."""
    status: str = Field(..., description="Overall system status")
    master_key_status: str = Field(..., description="Master key status")
    key_rotation_status: str = Field(..., description="Key rotation status")
    encryption_performance: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    security_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="Security alerts")
    compliance_status: Dict[str, str] = Field(default_factory=dict, description="Compliance status by framework")


# List response schemas
class EncryptionKeyListResponse(BaseModel):
    """Response schema for encryption key lists."""
    keys: List[EncryptionKeyResponse] = Field(default_factory=list, description="List of encryption keys")
    total: int = Field(..., description="Total number of keys")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class EncryptedDataListResponse(BaseModel):
    """Response schema for encrypted data lists."""
    data: List[EncryptedDataResponse] = Field(default_factory=list, description="List of encrypted data")
    total: int = Field(..., description="Total number of data records")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class EncryptionOperationListResponse(BaseModel):
    """Response schema for encryption operation lists."""
    operations: List[EncryptionOperationResponse] = Field(default_factory=list, description="List of operations")
    total: int = Field(..., description="Total number of operations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class EncryptionAuditLogListResponse(BaseModel):
    """Response schema for encryption audit log lists."""
    logs: List[EncryptionAuditLogResponse] = Field(default_factory=list, description="List of audit logs")
    total: int = Field(..., description="Total number of logs")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")