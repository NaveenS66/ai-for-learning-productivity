"""Tests for encryption functionality."""

import pytest
from uuid import uuid4

from src.ai_learning_accelerator.services.encryption_service import encryption_service
from src.ai_learning_accelerator.models.encryption import (
    EncryptionAlgorithm, KeyType, DataClassification
)


def test_encryption_service_initialization():
    """Test that encryption service initializes correctly."""
    assert encryption_service is not None
    assert encryption_service._master_key is not None
    assert len(encryption_service._master_key) == 32  # 256-bit key


def test_master_key_encryption_decryption():
    """Test master key encryption and decryption."""
    test_data = b"Hello, World! This is a test message."
    
    # Encrypt with master key
    encrypted_data = encryption_service._encrypt_with_master_key(test_data)
    assert encrypted_data != test_data
    assert len(encrypted_data) > len(test_data)  # Should be larger due to nonce + ciphertext
    
    # Decrypt with master key
    decrypted_data = encryption_service._decrypt_with_master_key(encrypted_data)
    assert decrypted_data == test_data


def test_aes_gcm_encryption_decryption():
    """Test AES-GCM encryption and decryption."""
    import secrets
    
    test_data = b"This is a test message for AES-GCM encryption."
    key = secrets.token_bytes(32)  # 256-bit key
    
    # Encrypt
    ciphertext, nonce, auth_tag = encryption_service._encrypt_aes_gcm(test_data, key)
    assert ciphertext != test_data
    assert len(nonce) == 12  # 96-bit nonce
    assert len(auth_tag) == 16  # 128-bit auth tag
    
    # Decrypt
    decrypted_data = encryption_service._decrypt_aes_gcm(ciphertext, key, nonce, auth_tag)
    assert decrypted_data == test_data


def test_chacha20_poly1305_encryption_decryption():
    """Test ChaCha20-Poly1305 encryption and decryption."""
    import secrets
    
    test_data = b"This is a test message for ChaCha20-Poly1305 encryption."
    key = secrets.token_bytes(32)  # 256-bit key
    
    # Encrypt
    ciphertext, nonce, auth_tag = encryption_service._encrypt_chacha20_poly1305(test_data, key)
    assert ciphertext != test_data
    assert len(nonce) == 12  # 96-bit nonce
    assert len(auth_tag) == 16  # 128-bit auth tag
    
    # Decrypt
    decrypted_data = encryption_service._decrypt_chacha20_poly1305(ciphertext, key, nonce, auth_tag)
    assert decrypted_data == test_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])