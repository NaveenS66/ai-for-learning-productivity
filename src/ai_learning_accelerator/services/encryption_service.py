"""Comprehensive data encryption service."""

import os
import json
import gzip
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.encryption import (
    EncryptionKey, EncryptedData, EncryptionOperation, DataEncryptionPolicy,
    EncryptionAuditLog, EncryptionAlgorithm, KeyType, DataClassification,
    EncryptionStatus
)
from ..database import get_async_db


class EncryptionService:
    """Comprehensive encryption service for data protection."""
    
    def __init__(self):
        """Initialize encryption service."""
        self.backend = default_backend()
        self._master_key = self._get_or_create_master_key()
        
    def _get_or_create_master_key(self) -> bytes:
        """Get or create the master encryption key."""
        master_key_path = os.getenv("MASTER_KEY_PATH", ".master_key")
        
        if os.path.exists(master_key_path):
            with open(master_key_path, "rb") as f:
                return f.read()
        else:
            # Generate new master key
            master_key = secrets.token_bytes(32)  # 256-bit key
            with open(master_key_path, "wb") as f:
                f.write(master_key)
            os.chmod(master_key_path, 0o600)  # Restrict permissions
            return master_key
    
    async def create_encryption_key(
        self,
        db: AsyncSession,
        key_name: str,
        key_type: KeyType,
        algorithm: EncryptionAlgorithm,
        purpose: str,
        created_by: UUID,
        key_size: Optional[int] = None,
        expires_at: Optional[datetime] = None
    ) -> EncryptionKey:
        """Create a new encryption key."""
        
        # Generate key material based on algorithm
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            key_material = secrets.token_bytes(32)  # 256-bit key
            key_size = key_size or 256
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            key_material = secrets.token_bytes(32)  # 256-bit key
            key_size = key_size or 256
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_material = secrets.token_bytes(32)  # 256-bit key
            key_size = key_size or 256
        elif algorithm == EncryptionAlgorithm.RSA_2048:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=self.backend
            )
            key_material = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            key_size = 2048
        elif algorithm == EncryptionAlgorithm.RSA_4096:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=self.backend
            )
            key_material = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            key_size = 4096
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Encrypt key material with master key
        encrypted_key_data = self._encrypt_with_master_key(key_material)
        
        # Create key hash for verification
        key_hash = hashlib.sha256(key_material).hexdigest()
        
        # Create encryption key record
        encryption_key = EncryptionKey(
            key_name=key_name,
            key_type=key_type,
            algorithm=algorithm,
            key_data=encrypted_key_data,
            key_size=key_size,
            key_hash=key_hash,
            purpose=purpose,
            created_by=created_by,
            expires_at=expires_at
        )
        
        db.add(encryption_key)
        await db.commit()
        await db.refresh(encryption_key)
        
        # Log key creation
        await self._log_audit_event(
            db, "key_created", "key_management", "info",
            user_id=created_by,
            resource_type="encryption_key",
            resource_id=str(encryption_key.id),
            operation="create_key",
            success=True,
            event_data={
                "key_name": key_name,
                "algorithm": algorithm.value,
                "key_size": key_size,
                "purpose": purpose
            }
        )
        
        return encryption_key
    
    async def encrypt_data(
        self,
        db: AsyncSession,
        data: Union[str, bytes, Dict[str, Any]],
        data_name: str,
        data_type: str,
        classification: DataClassification,
        owner_id: UUID,
        key_name: Optional[str] = None,
        algorithm: Optional[EncryptionAlgorithm] = None,
        compress: bool = True
    ) -> EncryptedData:
        """Encrypt data and store it securely."""
        
        operation_id = str(uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, dict):
                data_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
            else:
                data_bytes = data
            
            # Compress data if requested
            original_size = len(data_bytes)
            if compress and original_size > 1024:  # Only compress if > 1KB
                data_bytes = gzip.compress(data_bytes)
            
            # Get or create encryption key
            if key_name:
                encryption_key = await self._get_key_by_name(db, key_name)
            else:
                # Use default key for data classification
                encryption_key = await self._get_default_key_for_classification(
                    db, classification, owner_id
                )
            
            if not encryption_key or not encryption_key.is_active:
                raise ValueError("No active encryption key available")
            
            # Decrypt key material
            key_material = self._decrypt_with_master_key(encryption_key.key_data)
            
            # Encrypt data based on algorithm
            if encryption_key.algorithm == EncryptionAlgorithm.AES_256_GCM:
                encrypted_content, iv, auth_tag = self._encrypt_aes_gcm(data_bytes, key_material)
            elif encryption_key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                encrypted_content, iv, auth_tag = self._encrypt_chacha20_poly1305(data_bytes, key_material)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {encryption_key.algorithm}")
            
            # Create content hash
            content_hash = hashlib.sha256(data_bytes).hexdigest()
            
            # Create encrypted data record
            encrypted_data = EncryptedData(
                data_name=data_name,
                data_type=data_type,
                classification=classification,
                encryption_key_id=encryption_key.id,
                algorithm=encryption_key.algorithm,
                initialization_vector=iv,
                authentication_tag=auth_tag,
                encrypted_content=encrypted_content,
                content_hash=content_hash,
                content_size=original_size,
                compressed=compress and len(data_bytes) < original_size,
                owner_id=owner_id,
                status=EncryptionStatus.ENCRYPTED
            )
            
            db.add(encrypted_data)
            await db.commit()
            await db.refresh(encrypted_data)
            
            # Log encryption operation
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self._log_encryption_operation(
                db, "encrypt", operation_id, encrypted_data.id,
                encryption_key.id, encryption_key.algorithm, owner_id,
                EncryptionStatus.ENCRYPTED, start_time, duration_ms,
                True, data_size=original_size
            )
            
            return encrypted_data
            
        except Exception as e:
            # Log failed operation
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self._log_encryption_operation(
                db, "encrypt", operation_id, None,
                encryption_key.id if 'encryption_key' in locals() else None,
                algorithm or EncryptionAlgorithm.AES_256_GCM, owner_id,
                EncryptionStatus.FAILED, start_time, duration_ms,
                False, error_message=str(e)
            )
            raise
    
    async def decrypt_data(
        self,
        db: AsyncSession,
        encrypted_data_id: UUID,
        user_id: UUID,
        verify_permissions: bool = True
    ) -> Union[str, bytes, Dict[str, Any]]:
        """Decrypt data and return original content."""
        
        operation_id = str(uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Get encrypted data record
            encrypted_data = await self._get_encrypted_data(db, encrypted_data_id)
            if not encrypted_data:
                raise ValueError("Encrypted data not found")
            
            # Verify permissions
            if verify_permissions and not await self._check_decrypt_permission(
                db, encrypted_data, user_id
            ):
                raise PermissionError("Insufficient permissions to decrypt data")
            
            # Get encryption key
            encryption_key = await self._get_key_by_id(db, encrypted_data.encryption_key_id)
            if not encryption_key or not encryption_key.is_active:
                raise ValueError("Encryption key not available")
            
            # Decrypt key material
            key_material = self._decrypt_with_master_key(encryption_key.key_data)
            
            # Decrypt data based on algorithm
            if encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM:
                decrypted_bytes = self._decrypt_aes_gcm(
                    encrypted_data.encrypted_content,
                    key_material,
                    encrypted_data.initialization_vector,
                    encrypted_data.authentication_tag
                )
            elif encrypted_data.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                decrypted_bytes = self._decrypt_chacha20_poly1305(
                    encrypted_data.encrypted_content,
                    key_material,
                    encrypted_data.initialization_vector,
                    encrypted_data.authentication_tag
                )
            else:
                raise ValueError(f"Unsupported decryption algorithm: {encrypted_data.algorithm}")
            
            # Decompress if needed
            if encrypted_data.compressed:
                decrypted_bytes = gzip.decompress(decrypted_bytes)
            
            # Verify content integrity
            content_hash = hashlib.sha256(decrypted_bytes).hexdigest()
            if content_hash != encrypted_data.content_hash:
                raise ValueError("Data integrity check failed")
            
            # Update access timestamp
            encrypted_data.last_accessed_at = datetime.utcnow()
            await db.commit()
            
            # Log decryption operation
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self._log_encryption_operation(
                db, "decrypt", operation_id, encrypted_data.id,
                encryption_key.id, encrypted_data.algorithm, user_id,
                EncryptionStatus.DECRYPTED, start_time, duration_ms,
                True, data_size=encrypted_data.content_size
            )
            
            # Convert back to original format based on data type
            if encrypted_data.data_type in ["json", "dict"]:
                return json.loads(decrypted_bytes.decode('utf-8'))
            elif encrypted_data.data_type == "text":
                return decrypted_bytes.decode('utf-8')
            else:
                return decrypted_bytes
                
        except Exception as e:
            # Log failed operation
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self._log_encryption_operation(
                db, "decrypt", operation_id, encrypted_data_id,
                None, None, user_id,
                EncryptionStatus.FAILED, start_time, duration_ms,
                False, error_message=str(e)
            )
            raise
    
    def _encrypt_with_master_key(self, data: bytes) -> bytes:
        """Encrypt data with master key using AES-GCM."""
        aesgcm = AESGCM(self._master_key)
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        ciphertext = aesgcm.encrypt(nonce, data, None)
        return nonce + ciphertext
    
    def _decrypt_with_master_key(self, encrypted_data: bytes) -> bytes:
        """Decrypt data with master key using AES-GCM."""
        aesgcm = AESGCM(self._master_key)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        return aesgcm.decrypt(nonce, ciphertext, None)
    
    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> Tuple[bytes, bytes, bytes]:
        """Encrypt data using AES-256-GCM."""
        aesgcm = AESGCM(key)
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        ciphertext = aesgcm.encrypt(nonce, data, None)
        # For GCM, the authentication tag is included in ciphertext
        return ciphertext[:-16], nonce, ciphertext[-16:]
    
    def _decrypt_aes_gcm(self, ciphertext: bytes, key: bytes, nonce: bytes, auth_tag: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        aesgcm = AESGCM(key)
        # Reconstruct full ciphertext with auth tag
        full_ciphertext = ciphertext + auth_tag
        return aesgcm.decrypt(nonce, full_ciphertext, None)
    
    def _encrypt_chacha20_poly1305(self, data: bytes, key: bytes) -> Tuple[bytes, bytes, bytes]:
        """Encrypt data using ChaCha20-Poly1305."""
        chacha = ChaCha20Poly1305(key)
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        ciphertext = chacha.encrypt(nonce, data, None)
        # For ChaCha20-Poly1305, the authentication tag is included in ciphertext
        return ciphertext[:-16], nonce, ciphertext[-16:]
    
    def _decrypt_chacha20_poly1305(self, ciphertext: bytes, key: bytes, nonce: bytes, auth_tag: bytes) -> bytes:
        """Decrypt data using ChaCha20-Poly1305."""
        chacha = ChaCha20Poly1305(key)
        # Reconstruct full ciphertext with auth tag
        full_ciphertext = ciphertext + auth_tag
        return chacha.decrypt(nonce, full_ciphertext, None)
    
    async def _get_key_by_name(self, db: AsyncSession, key_name: str) -> Optional[EncryptionKey]:
        """Get encryption key by name."""
        from sqlalchemy import select
        result = await db.execute(
            select(EncryptionKey).where(
                EncryptionKey.key_name == key_name,
                EncryptionKey.is_active == True,
                EncryptionKey.is_revoked == False
            )
        )
        return result.scalar_one_or_none()
    
    async def _get_key_by_id(self, db: AsyncSession, key_id: UUID) -> Optional[EncryptionKey]:
        """Get encryption key by ID."""
        from sqlalchemy import select
        result = await db.execute(
            select(EncryptionKey).where(EncryptionKey.id == key_id)
        )
        return result.scalar_one_or_none()
    
    async def _get_encrypted_data(self, db: AsyncSession, data_id: UUID) -> Optional[EncryptedData]:
        """Get encrypted data by ID."""
        from sqlalchemy import select
        result = await db.execute(
            select(EncryptedData).where(EncryptedData.id == data_id)
        )
        return result.scalar_one_or_none()
    
    async def _get_default_key_for_classification(
        self,
        db: AsyncSession,
        classification: DataClassification,
        user_id: UUID
    ) -> Optional[EncryptionKey]:
        """Get default encryption key for data classification."""
        # For now, create a default key if none exists
        key_name = f"default_{classification.value}_key"
        
        existing_key = await self._get_key_by_name(db, key_name)
        if existing_key:
            return existing_key
        
        # Create default key
        algorithm = EncryptionAlgorithm.AES_256_GCM
        if classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            algorithm = EncryptionAlgorithm.CHACHA20_POLY1305
        
        return await self.create_encryption_key(
            db, key_name, KeyType.DATA_ENCRYPTION_KEY, algorithm,
            f"Default key for {classification.value} data", user_id
        )
    
    async def _check_decrypt_permission(
        self,
        db: AsyncSession,
        encrypted_data: EncryptedData,
        user_id: UUID
    ) -> bool:
        """Check if user has permission to decrypt data."""
        # Owner can always decrypt
        if encrypted_data.owner_id == user_id:
            return True
        
        # Check access permissions
        permissions = encrypted_data.access_permissions or {}
        allowed_users = permissions.get("allowed_users", [])
        
        return str(user_id) in allowed_users
    
    async def _log_encryption_operation(
        self,
        db: AsyncSession,
        operation_type: str,
        operation_id: str,
        data_id: Optional[UUID],
        key_id: Optional[UUID],
        algorithm: Optional[EncryptionAlgorithm],
        user_id: UUID,
        status: EncryptionStatus,
        started_at: datetime,
        duration_ms: int,
        success: bool,
        error_message: Optional[str] = None,
        data_size: Optional[int] = None
    ):
        """Log encryption operation."""
        operation = EncryptionOperation(
            operation_type=operation_type,
            operation_id=operation_id,
            data_id=data_id,
            key_id=key_id,
            algorithm=algorithm,
            user_id=user_id,
            status=status,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            data_size=data_size
        )
        
        db.add(operation)
        await db.commit()
    
    async def _log_audit_event(
        self,
        db: AsyncSession,
        event_type: str,
        event_category: str,
        severity_level: str,
        user_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        operation: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None
    ):
        """Log audit event."""
        audit_log = EncryptionAuditLog(
            event_type=event_type,
            event_category=event_category,
            severity_level=severity_level,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            operation=operation,
            success=success,
            error_message=error_message,
            event_data=event_data or {}
        )
        
        db.add(audit_log)
        await db.commit()
    
    async def rotate_key(
        self,
        db: AsyncSession,
        key_name: str,
        user_id: UUID
    ) -> EncryptionKey:
        """Rotate an encryption key."""
        # Get current key
        current_key = await self._get_key_by_name(db, key_name)
        if not current_key:
            raise ValueError("Key not found")
        
        # Create new key with incremented version
        new_key = await self.create_encryption_key(
            db, key_name, current_key.key_type, current_key.algorithm,
            current_key.purpose, user_id, current_key.key_size,
            current_key.expires_at
        )
        
        new_key.version = current_key.version + 1
        new_key.parent_key_id = current_key.id
        
        # Deactivate old key
        current_key.is_active = False
        
        await db.commit()
        
        # Log key rotation
        await self._log_audit_event(
            db, "key_rotated", "key_management", "info",
            user_id=user_id,
            resource_type="encryption_key",
            resource_id=str(new_key.id),
            operation="rotate_key",
            success=True,
            event_data={
                "old_key_id": str(current_key.id),
                "new_key_id": str(new_key.id),
                "key_name": key_name
            }
        )
        
        return new_key
    
    async def secure_delete_data(
        self,
        db: AsyncSession,
        encrypted_data_id: UUID,
        user_id: UUID
    ) -> bool:
        """Securely delete encrypted data."""
        encrypted_data = await self._get_encrypted_data(db, encrypted_data_id)
        if not encrypted_data:
            return False
        
        # Verify permissions
        if encrypted_data.owner_id != user_id:
            raise PermissionError("Only data owner can delete encrypted data")
        
        # Log deletion
        await self._log_audit_event(
            db, "data_deleted", "data_management", "info",
            user_id=user_id,
            resource_type="encrypted_data",
            resource_id=str(encrypted_data_id),
            operation="secure_delete",
            success=True,
            event_data={
                "data_name": encrypted_data.data_name,
                "data_type": encrypted_data.data_type,
                "classification": encrypted_data.classification.value
            }
        )
        
        # Delete from database
        await db.delete(encrypted_data)
        await db.commit()
        
        return True


# Global encryption service instance
encryption_service = EncryptionService()