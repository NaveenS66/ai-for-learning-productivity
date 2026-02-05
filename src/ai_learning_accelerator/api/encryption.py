"""Encryption and data security API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..database import get_async_db
from ..models.user import User
from ..models.encryption import (
    EncryptionKey, EncryptedData, EncryptionOperation, DataEncryptionPolicy,
    EncryptionAuditLog, DataClassification, EncryptionStatus, EncryptionAlgorithm
)
from ..schemas.encryption import (
    CreateEncryptionKeyRequest, EncryptionKeyResponse, EncryptionKeyListResponse,
    EncryptDataRequest, EncryptedDataResponse, EncryptedDataListResponse,
    DecryptDataRequest, DecryptDataResponse,
    RotateKeyRequest, CreatePolicyRequest, DataEncryptionPolicyResponse,
    EncryptionOperationResponse, EncryptionOperationListResponse,
    EncryptionAuditLogResponse, EncryptionAuditLogListResponse,
    EncryptionStatsResponse, EncryptionHealthResponse
)
from ..services.encryption_service import encryption_service
from ..services.auth import get_current_user

router = APIRouter(prefix="/encryption", tags=["encryption"])


# Encryption Key Management
@router.post("/keys", response_model=EncryptionKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_encryption_key(
    request: CreateEncryptionKeyRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new encryption key."""
    try:
        encryption_key = await encryption_service.create_encryption_key(
            db=db,
            key_name=request.key_name,
            key_type=request.key_type,
            algorithm=request.algorithm,
            purpose=request.purpose,
            created_by=current_user.id,
            key_size=request.key_size,
            expires_at=request.expires_at
        )
        
        return EncryptionKeyResponse.from_orm(encryption_key)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create encryption key: {str(e)}"
        )


@router.get("/keys", response_model=EncryptionKeyListResponse)
async def list_encryption_keys(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    key_type: Optional[str] = Query(None, description="Filter by key type"),
    algorithm: Optional[EncryptionAlgorithm] = Query(None, description="Filter by algorithm"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """List encryption keys with filtering and pagination."""
    try:
        # Build query
        query = select(EncryptionKey).where(EncryptionKey.created_by == current_user.id)
        
        if key_type:
            query = query.where(EncryptionKey.key_type == key_type)
        if algorithm:
            query = query.where(EncryptionKey.algorithm == algorithm)
        if is_active is not None:
            query = query.where(EncryptionKey.is_active == is_active)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        keys = result.scalars().all()
        
        return EncryptionKeyListResponse(
            keys=[EncryptionKeyResponse.from_orm(key) for key in keys],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list encryption keys: {str(e)}"
        )


@router.get("/keys/{key_id}", response_model=EncryptionKeyResponse)
async def get_encryption_key(
    key_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Get encryption key by ID."""
    try:
        result = await db.execute(
            select(EncryptionKey).where(
                and_(
                    EncryptionKey.id == key_id,
                    EncryptionKey.created_by == current_user.id
                )
            )
        )
        key = result.scalar_one_or_none()
        
        if not key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Encryption key not found"
            )
        
        return EncryptionKeyResponse.from_orm(key)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get encryption key: {str(e)}"
        )


@router.post("/keys/{key_name}/rotate", response_model=EncryptionKeyResponse)
async def rotate_encryption_key(
    key_name: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Rotate an encryption key."""
    try:
        new_key = await encryption_service.rotate_key(
            db=db,
            key_name=key_name,
            user_id=current_user.id
        )
        
        return EncryptionKeyResponse.from_orm(new_key)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rotate encryption key: {str(e)}"
        )


# Data Encryption Operations
@router.post("/data/encrypt", response_model=EncryptedDataResponse, status_code=status.HTTP_201_CREATED)
async def encrypt_data(
    request: EncryptDataRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Encrypt data and store it securely."""
    try:
        encrypted_data = await encryption_service.encrypt_data(
            db=db,
            data=request.data,
            data_name=request.data_name,
            data_type=request.data_type,
            classification=request.classification,
            owner_id=current_user.id,
            key_name=request.key_name,
            algorithm=request.algorithm,
            compress=request.compress
        )
        
        # Update access permissions if provided
        if request.access_permissions:
            encrypted_data.access_permissions = request.access_permissions
        if request.sharing_restrictions:
            encrypted_data.sharing_restrictions = request.sharing_restrictions
        if request.expires_at:
            encrypted_data.expires_at = request.expires_at
        if request.metadata:
            encrypted_data.metadata = request.metadata
        
        await db.commit()
        await db.refresh(encrypted_data)
        
        return EncryptedDataResponse.from_orm(encrypted_data)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encrypt data: {str(e)}"
        )


@router.post("/data/decrypt", response_model=DecryptDataResponse)
async def decrypt_data(
    request: DecryptDataRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Decrypt data and return original content."""
    try:
        # Get encrypted data info first
        result = await db.execute(
            select(EncryptedData).where(EncryptedData.id == request.encrypted_data_id)
        )
        encrypted_data = result.scalar_one_or_none()
        
        if not encrypted_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Encrypted data not found"
            )
        
        # Decrypt data
        decrypted_data = await encryption_service.decrypt_data(
            db=db,
            encrypted_data_id=request.encrypted_data_id,
            user_id=current_user.id,
            verify_permissions=request.verify_permissions
        )
        
        return DecryptDataResponse(
            data=decrypted_data,
            data_type=encrypted_data.data_type,
            original_size=encrypted_data.content_size,
            decryption_timestamp=datetime.utcnow()
        )
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to decrypt data: {str(e)}"
        )


@router.get("/data", response_model=EncryptedDataListResponse)
async def list_encrypted_data(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    classification: Optional[DataClassification] = Query(None, description="Filter by classification"),
    status: Optional[EncryptionStatus] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """List encrypted data with filtering and pagination."""
    try:
        # Build query
        query = select(EncryptedData).where(EncryptedData.owner_id == current_user.id)
        
        if data_type:
            query = query.where(EncryptedData.data_type == data_type)
        if classification:
            query = query.where(EncryptedData.classification == classification)
        if status:
            query = query.where(EncryptedData.status == status)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        data = result.scalars().all()
        
        return EncryptedDataListResponse(
            data=[EncryptedDataResponse.from_orm(item) for item in data],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list encrypted data: {str(e)}"
        )


@router.get("/data/{data_id}", response_model=EncryptedDataResponse)
async def get_encrypted_data(
    data_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Get encrypted data by ID."""
    try:
        result = await db.execute(
            select(EncryptedData).where(
                and_(
                    EncryptedData.id == data_id,
                    EncryptedData.owner_id == current_user.id
                )
            )
        )
        data = result.scalar_one_or_none()
        
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Encrypted data not found"
            )
        
        return EncryptedDataResponse.from_orm(data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get encrypted data: {str(e)}"
        )


@router.delete("/data/{data_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_encrypted_data(
    data_id: UUID,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Securely delete encrypted data."""
    try:
        success = await encryption_service.secure_delete_data(
            db=db,
            encrypted_data_id=data_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Encrypted data not found"
            )
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete encrypted data: {str(e)}"
        )


# Encryption Operations and Audit
@router.get("/operations", response_model=EncryptionOperationListResponse)
async def list_encryption_operations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    status: Optional[EncryptionStatus] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """List encryption operations with filtering and pagination."""
    try:
        # Build query
        query = select(EncryptionOperation).where(EncryptionOperation.user_id == current_user.id)
        
        if operation_type:
            query = query.where(EncryptionOperation.operation_type == operation_type)
        if status:
            query = query.where(EncryptionOperation.status == status)
        if start_date:
            query = query.where(EncryptionOperation.started_at >= start_date)
        if end_date:
            query = query.where(EncryptionOperation.started_at <= end_date)
        
        # Order by most recent first
        query = query.order_by(EncryptionOperation.started_at.desc())
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        operations = result.scalars().all()
        
        return EncryptionOperationListResponse(
            operations=[EncryptionOperationResponse.from_orm(op) for op in operations],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list encryption operations: {str(e)}"
        )


@router.get("/audit", response_model=EncryptionAuditLogListResponse)
async def list_audit_logs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    event_category: Optional[str] = Query(None, description="Filter by event category"),
    severity_level: Optional[str] = Query(None, description="Filter by severity level"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """List encryption audit logs with filtering and pagination."""
    try:
        # Build query
        query = select(EncryptionAuditLog).where(
            or_(
                EncryptionAuditLog.user_id == current_user.id,
                EncryptionAuditLog.user_id.is_(None)  # System events
            )
        )
        
        if event_type:
            query = query.where(EncryptionAuditLog.event_type == event_type)
        if event_category:
            query = query.where(EncryptionAuditLog.event_category == event_category)
        if severity_level:
            query = query.where(EncryptionAuditLog.severity_level == severity_level)
        if start_date:
            query = query.where(EncryptionAuditLog.timestamp >= start_date)
        if end_date:
            query = query.where(EncryptionAuditLog.timestamp <= end_date)
        
        # Order by most recent first
        query = query.order_by(EncryptionAuditLog.timestamp.desc())
        
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
        
        return EncryptionAuditLogListResponse(
            logs=[EncryptionAuditLogResponse.from_orm(log) for log in logs],
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list audit logs: {str(e)}"
        )


# Statistics and Health
@router.get("/stats", response_model=EncryptionStatsResponse)
async def get_encryption_stats(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Get encryption system statistics."""
    try:
        # Get key statistics
        total_keys_result = await db.execute(
            select(func.count()).select_from(EncryptionKey).where(
                EncryptionKey.created_by == current_user.id
            )
        )
        total_keys = total_keys_result.scalar()
        
        active_keys_result = await db.execute(
            select(func.count()).select_from(EncryptionKey).where(
                and_(
                    EncryptionKey.created_by == current_user.id,
                    EncryptionKey.is_active == True
                )
            )
        )
        active_keys = active_keys_result.scalar()
        
        # Get data statistics
        total_data_result = await db.execute(
            select(func.count()).select_from(EncryptedData).where(
                EncryptedData.owner_id == current_user.id
            )
        )
        total_encrypted_data = total_data_result.scalar()
        
        # Get operation statistics
        total_ops_result = await db.execute(
            select(func.count()).select_from(EncryptionOperation).where(
                EncryptionOperation.user_id == current_user.id
            )
        )
        total_operations = total_ops_result.scalar()
        
        successful_ops_result = await db.execute(
            select(func.count()).select_from(EncryptionOperation).where(
                and_(
                    EncryptionOperation.user_id == current_user.id,
                    EncryptionOperation.success == True
                )
            )
        )
        successful_operations = successful_ops_result.scalar()
        
        failed_operations = total_operations - successful_operations
        
        # Get data by classification
        classification_result = await db.execute(
            select(
                EncryptedData.classification,
                func.count()
            ).where(
                EncryptedData.owner_id == current_user.id
            ).group_by(EncryptedData.classification)
        )
        data_by_classification = {
            str(row[0]): row[1] for row in classification_result.fetchall()
        }
        
        # Get operations by algorithm
        algorithm_result = await db.execute(
            select(
                EncryptionOperation.algorithm,
                func.count()
            ).where(
                EncryptionOperation.user_id == current_user.id
            ).group_by(EncryptionOperation.algorithm)
        )
        operations_by_algorithm = {
            str(row[0]): row[1] for row in algorithm_result.fetchall() if row[0]
        }
        
        return EncryptionStatsResponse(
            total_keys=total_keys,
            active_keys=active_keys,
            total_encrypted_data=total_encrypted_data,
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            data_by_classification=data_by_classification,
            operations_by_algorithm=operations_by_algorithm
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get encryption statistics: {str(e)}"
        )


@router.get("/health", response_model=EncryptionHealthResponse)
async def get_encryption_health(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Get encryption system health status."""
    try:
        # Check master key status
        master_key_status = "healthy"
        try:
            encryption_service._get_or_create_master_key()
        except Exception:
            master_key_status = "error"
        
        # Check key rotation status
        key_rotation_status = "healthy"
        expired_keys_result = await db.execute(
            select(func.count()).select_from(EncryptionKey).where(
                and_(
                    EncryptionKey.created_by == current_user.id,
                    EncryptionKey.expires_at < datetime.utcnow(),
                    EncryptionKey.is_active == True
                )
            )
        )
        expired_keys = expired_keys_result.scalar()
        if expired_keys > 0:
            key_rotation_status = "warning"
        
        # Get recent failed operations
        recent_failures_result = await db.execute(
            select(func.count()).select_from(EncryptionOperation).where(
                and_(
                    EncryptionOperation.user_id == current_user.id,
                    EncryptionOperation.success == False,
                    EncryptionOperation.started_at >= datetime.utcnow() - timedelta(hours=24)
                )
            )
        )
        recent_failures = recent_failures_result.scalar()
        
        # Determine overall status
        if master_key_status == "error" or recent_failures > 10:
            overall_status = "error"
        elif key_rotation_status == "warning" or recent_failures > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        security_alerts = []
        if expired_keys > 0:
            security_alerts.append({
                "type": "expired_keys",
                "message": f"{expired_keys} encryption keys have expired",
                "severity": "warning"
            })
        
        if recent_failures > 0:
            security_alerts.append({
                "type": "operation_failures",
                "message": f"{recent_failures} encryption operations failed in the last 24 hours",
                "severity": "warning" if recent_failures <= 5 else "error"
            })
        
        return EncryptionHealthResponse(
            status=overall_status,
            master_key_status=master_key_status,
            key_rotation_status=key_rotation_status,
            encryption_performance={
                "recent_failures": recent_failures,
                "expired_keys": expired_keys
            },
            security_alerts=security_alerts,
            compliance_status={
                "data_encryption": "compliant",
                "key_management": "compliant" if key_rotation_status == "healthy" else "warning"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get encryption health: {str(e)}"
        )