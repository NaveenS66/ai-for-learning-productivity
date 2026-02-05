"""Tests for security utilities."""

from datetime import timedelta

import pytest
from hypothesis import given, strategies as st

from ai_learning_accelerator.utils.security import (
    create_access_token,
    generate_api_key,
    get_password_hash,
    verify_password,
    verify_token,
)


class TestPasswordHashing:
    """Test password hashing functionality."""
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)
    
    @given(st.text(min_size=1, max_size=72))  # Limit to bcrypt max length
    def test_password_hashing_property(self, password: str):
        """Property test: any password should hash and verify correctly."""
        hashed = get_password_hash(password)
        assert verify_password(password, hashed)
        assert hashed != password  # Hash should be different from original


class TestJWTTokens:
    """Test JWT token functionality."""
    
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification."""
        data = {"sub": "test_user", "role": "user"}
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "test_user"
        assert payload["role"] == "user"
        assert "exp" in payload
    
    def test_token_with_custom_expiry(self):
        """Test token creation with custom expiry."""
        data = {"sub": "test_user"}
        expires_delta = timedelta(minutes=60)
        token = create_access_token(data, expires_delta)
        
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "test_user"
    
    def test_invalid_token_verification(self):
        """Test verification of invalid tokens."""
        assert verify_token("invalid_token") is None
        assert verify_token("") is None
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(min_size=1, max_size=50), st.integers()),
        min_size=1,
        max_size=5
    ))
    def test_token_roundtrip_property(self, data: dict):
        """Property test: any data should roundtrip through token creation/verification."""
        token = create_access_token(data)
        payload = verify_token(token)
        
        assert payload is not None
        for key, value in data.items():
            assert payload[key] == value


class TestAPIKeyGeneration:
    """Test API key generation."""
    
    def test_api_key_generation(self):
        """Test API key generation."""
        key1 = generate_api_key()
        key2 = generate_api_key()
        
        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert len(key1) > 0
        assert len(key2) > 0
        assert key1 != key2  # Should generate unique keys
    
    def test_api_key_uniqueness(self):
        """Test that generated API keys are unique."""
        keys = [generate_api_key() for _ in range(100)]
        assert len(set(keys)) == 100  # All keys should be unique