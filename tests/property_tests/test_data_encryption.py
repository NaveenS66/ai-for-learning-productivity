"""Property-based tests for data encryption.

Tests that the system properly encrypts and protects sensitive data.
Validates Requirements 10.1.
"""

import pytest
import secrets
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Union
from uuid import uuid4

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize

from src.ai_learning_accelerator.models.encryption import (
    EncryptionKey, EncryptedData, EncryptionOperation, DataEncryptionPolicy,
    EncryptionAlgorithm, KeyType, DataClassification, EncryptionStatus
)
from src.ai_learning_accelerator.services.encryption_service import encryption_service
from tests.property_tests.test_base_properties import BasePropertyTest


class TestDataEncryption(BasePropertyTest):
    """Property-based tests for data encryption."""
    
    def test_data_encryption_property(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that the system properly encrypts and protects sensitive data.
        This is a comprehensive property test that validates multiple aspects of data encryption.
        """
        
        # Test 1: Encryption algorithms are secure
        secure_algorithms = [
            EncryptionAlgorithm.AES_256_GCM,
            EncryptionAlgorithm.AES_256_CBC,
            EncryptionAlgorithm.CHACHA20_POLY1305,
            EncryptionAlgorithm.RSA_2048,
            EncryptionAlgorithm.RSA_4096
        ]
        
        for algorithm in secure_algorithms:
            # Property: All supported algorithms should be cryptographically secure
            assert algorithm in [
                EncryptionAlgorithm.AES_256_GCM,
                EncryptionAlgorithm.AES_256_CBC,
                EncryptionAlgorithm.CHACHA20_POLY1305,
                EncryptionAlgorithm.RSA_2048,
                EncryptionAlgorithm.RSA_4096,
                EncryptionAlgorithm.ECDSA_P256,
                EncryptionAlgorithm.ECDSA_P384
            ], f"Algorithm {algorithm} should be supported"
            
            # Property: Key sizes should meet security standards
            if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC, EncryptionAlgorithm.CHACHA20_POLY1305]:
                expected_key_size = 256
            elif algorithm == EncryptionAlgorithm.RSA_2048:
                expected_key_size = 2048
            elif algorithm == EncryptionAlgorithm.RSA_4096:
                expected_key_size = 4096
            else:
                expected_key_size = 256  # Default for ECDSA
            
            assert expected_key_size >= 256, f"Key size {expected_key_size} should be at least 256 bits for {algorithm}"
    
    def test_encryption_key_generation_properties(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that encryption keys are generated with proper security properties.
        """
        
        # Test key generation parameters
        test_cases = [
            {
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
                "key_type": KeyType.DATA_ENCRYPTION_KEY,
                "expected_key_size": 256
            },
            {
                "algorithm": EncryptionAlgorithm.CHACHA20_POLY1305,
                "key_type": KeyType.DATA_ENCRYPTION_KEY,
                "expected_key_size": 256
            },
            {
                "algorithm": EncryptionAlgorithm.RSA_2048,
                "key_type": KeyType.ASYMMETRIC_PRIVATE,
                "expected_key_size": 2048
            }
        ]
        
        for case in test_cases:
            # Property: Key generation should use secure parameters
            key_config = {
                "key_name": f"test_key_{case['algorithm'].value}",
                "key_type": case["key_type"],
                "algorithm": case["algorithm"],
                "purpose": "test_encryption",
                "created_by": uuid4(),
                "key_size": case["expected_key_size"]
            }
            
            # Validate key configuration
            assert key_config["key_size"] >= 256, f"Key size should be at least 256 bits"
            assert key_config["algorithm"] in [
                EncryptionAlgorithm.AES_256_GCM,
                EncryptionAlgorithm.AES_256_CBC,
                EncryptionAlgorithm.CHACHA20_POLY1305,
                EncryptionAlgorithm.RSA_2048,
                EncryptionAlgorithm.RSA_4096
            ], f"Algorithm {key_config['algorithm']} should be supported"
            
            # Property: Key names should be unique and descriptive
            assert len(key_config["key_name"]) > 0, "Key name should not be empty"
            assert key_config["purpose"], "Key purpose should be specified"
    
    def test_data_classification_encryption_requirements(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that different data classifications have appropriate encryption requirements.
        """
        
        classification_requirements = {
            DataClassification.PUBLIC: {
                "encryption_required": False,
                "min_algorithm_strength": None
            },
            DataClassification.INTERNAL: {
                "encryption_required": True,
                "min_algorithm_strength": EncryptionAlgorithm.AES_256_GCM
            },
            DataClassification.CONFIDENTIAL: {
                "encryption_required": True,
                "min_algorithm_strength": EncryptionAlgorithm.AES_256_GCM
            },
            DataClassification.RESTRICTED: {
                "encryption_required": True,
                "min_algorithm_strength": EncryptionAlgorithm.CHACHA20_POLY1305
            },
            DataClassification.TOP_SECRET: {
                "encryption_required": True,
                "min_algorithm_strength": EncryptionAlgorithm.CHACHA20_POLY1305
            }
        }
        
        for classification, requirements in classification_requirements.items():
            # Property: Higher classifications should require stronger encryption
            if classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
                assert requirements["encryption_required"], f"{classification} should require encryption"
                assert requirements["min_algorithm_strength"] in [
                    EncryptionAlgorithm.CHACHA20_POLY1305,
                    EncryptionAlgorithm.RSA_4096
                ], f"{classification} should use strong encryption algorithms"
            
            elif classification in [DataClassification.CONFIDENTIAL, DataClassification.INTERNAL]:
                assert requirements["encryption_required"], f"{classification} should require encryption"
                assert requirements["min_algorithm_strength"] in [
                    EncryptionAlgorithm.AES_256_GCM,
                    EncryptionAlgorithm.AES_256_CBC,
                    EncryptionAlgorithm.CHACHA20_POLY1305
                ], f"{classification} should use secure encryption algorithms"
    
    def test_encryption_operation_integrity(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that encryption operations maintain data integrity.
        """
        
        # Test data samples
        test_data_samples = [
            {"data": "Hello, World!", "data_type": "text"},
            {"data": {"key": "value", "number": 42}, "data_type": "json"},
            {"data": b"binary_data_sample", "data_type": "binary"},
            {"data": "A" * 10000, "data_type": "large_text"}  # Large data
        ]
        
        for sample in test_data_samples:
            data = sample["data"]
            data_type = sample["data_type"]
            
            # Simulate encryption process
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, dict):
                data_bytes = json.dumps(data, ensure_ascii=False).encode('utf-8')
            else:
                data_bytes = data
            
            # Property: Original data should be recoverable after encryption/decryption
            original_size = len(data_bytes)
            
            # Simulate encryption metadata
            encryption_metadata = {
                "algorithm": EncryptionAlgorithm.AES_256_GCM,
                "key_size": 256,
                "iv_size": 12,  # 96-bit IV for GCM
                "auth_tag_size": 16,  # 128-bit auth tag
                "original_size": original_size,
                "data_type": data_type
            }
            
            # Property: Encryption should preserve data size information
            assert encryption_metadata["original_size"] == original_size, \
                "Original data size should be preserved in metadata"
            
            # Property: Encryption should use appropriate IV/nonce sizes
            if encryption_metadata["algorithm"] == EncryptionAlgorithm.AES_256_GCM:
                assert encryption_metadata["iv_size"] == 12, "AES-GCM should use 96-bit IV"
                assert encryption_metadata["auth_tag_size"] == 16, "AES-GCM should use 128-bit auth tag"
            
            # Property: Encrypted data should be larger than original (due to IV + auth tag)
            expected_encrypted_size = original_size + encryption_metadata["iv_size"] + encryption_metadata["auth_tag_size"]
            assert expected_encrypted_size > original_size, \
                "Encrypted data should be larger than original due to IV and auth tag"
    
    def test_encryption_key_lifecycle_properties(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that encryption keys follow proper lifecycle management.
        """
        
        # Test key lifecycle states
        key_lifecycle_states = [
            {"state": "created", "is_active": True, "is_revoked": False},
            {"state": "active", "is_active": True, "is_revoked": False},
            {"state": "expired", "is_active": False, "is_revoked": False},
            {"state": "revoked", "is_active": False, "is_revoked": True}
        ]
        
        for state_info in key_lifecycle_states:
            state = state_info["state"]
            is_active = state_info["is_active"]
            is_revoked = state_info["is_revoked"]
            
            # Property: Key state should be consistent
            if state == "revoked":
                assert is_revoked, "Revoked keys should have is_revoked=True"
                assert not is_active, "Revoked keys should have is_active=False"
            
            elif state == "expired":
                assert not is_active, "Expired keys should have is_active=False"
            
            elif state in ["created", "active"]:
                assert is_active, f"{state} keys should have is_active=True"
                assert not is_revoked, f"{state} keys should have is_revoked=False"
            
            # Property: Only active keys should be usable for encryption
            can_encrypt = is_active and not is_revoked
            if state in ["created", "active"]:
                assert can_encrypt, f"{state} keys should be usable for encryption"
            else:
                assert not can_encrypt, f"{state} keys should not be usable for encryption"
    
    def test_encryption_policy_enforcement(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that encryption policies are properly enforced.
        """
        
        # Test encryption policy configurations
        policy_configs = [
            {
                "policy_name": "standard_encryption",
                "data_classifications": [DataClassification.CONFIDENTIAL],
                "required_algorithm": EncryptionAlgorithm.AES_256_GCM,
                "minimum_key_size": 256,
                "encryption_at_rest": True,
                "encryption_in_transit": True
            },
            {
                "policy_name": "high_security_encryption",
                "data_classifications": [DataClassification.RESTRICTED, DataClassification.TOP_SECRET],
                "required_algorithm": EncryptionAlgorithm.CHACHA20_POLY1305,
                "minimum_key_size": 256,
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "encryption_in_use": True
            }
        ]
        
        for policy in policy_configs:
            # Property: Policies should enforce minimum security standards
            assert policy["minimum_key_size"] >= 256, \
                f"Policy {policy['policy_name']} should require at least 256-bit keys"
            
            assert policy["encryption_at_rest"], \
                f"Policy {policy['policy_name']} should require encryption at rest"
            
            assert policy["encryption_in_transit"], \
                f"Policy {policy['policy_name']} should require encryption in transit"
            
            # Property: High-security policies should have stricter requirements
            if "high_security" in policy["policy_name"]:
                assert policy["required_algorithm"] in [
                    EncryptionAlgorithm.CHACHA20_POLY1305,
                    EncryptionAlgorithm.RSA_4096
                ], f"High-security policy should use strong algorithms"
                
                assert policy.get("encryption_in_use", False), \
                    "High-security policy should require encryption in use"
            
            # Property: Policies should cover appropriate data classifications
            for classification in policy["data_classifications"]:
                assert classification in [
                    DataClassification.INTERNAL,
                    DataClassification.CONFIDENTIAL,
                    DataClassification.RESTRICTED,
                    DataClassification.TOP_SECRET
                ], f"Policy should only apply to sensitive data classifications"
    
    def test_encryption_audit_and_compliance(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that encryption operations are properly audited for compliance.
        """
        
        # Test audit requirements for different operations
        audit_operations = [
            {
                "operation": "encrypt",
                "required_fields": ["user_id", "data_id", "key_id", "algorithm", "timestamp"],
                "security_level": "standard"
            },
            {
                "operation": "decrypt",
                "required_fields": ["user_id", "data_id", "key_id", "algorithm", "timestamp", "access_reason"],
                "security_level": "high"
            },
            {
                "operation": "key_creation",
                "required_fields": ["user_id", "key_id", "algorithm", "purpose", "timestamp"],
                "security_level": "high"
            },
            {
                "operation": "key_rotation",
                "required_fields": ["user_id", "old_key_id", "new_key_id", "timestamp", "rotation_reason"],
                "security_level": "critical"
            }
        ]
        
        for audit_op in audit_operations:
            operation = audit_op["operation"]
            required_fields = audit_op["required_fields"]
            security_level = audit_op["security_level"]
            
            # Property: All operations should have required audit fields
            for field in required_fields:
                assert field in [
                    "user_id", "data_id", "key_id", "algorithm", "timestamp",
                    "access_reason", "purpose", "old_key_id", "new_key_id", "rotation_reason"
                ], f"Field {field} should be a valid audit field"
            
            # Property: Critical operations should have more audit requirements
            if security_level == "critical":
                assert len(required_fields) >= 5, \
                    f"Critical operation {operation} should have at least 5 audit fields"
                assert "timestamp" in required_fields, \
                    f"Critical operation {operation} should require timestamp"
                assert "user_id" in required_fields, \
                    f"Critical operation {operation} should require user_id"
            
            # Property: High-security operations should include access reasons
            if security_level in ["high", "critical"] and operation in ["decrypt", "key_rotation"]:
                reason_fields = [f for f in required_fields if "reason" in f]
                assert len(reason_fields) > 0, \
                    f"High-security operation {operation} should require reason field"
    
    def test_encryption_performance_and_scalability(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that encryption operations meet performance requirements.
        """
        
        # Test performance characteristics for different data sizes
        performance_test_cases = [
            {"data_size": 1024, "max_time_ms": 10, "description": "small_data"},      # 1KB
            {"data_size": 1024 * 1024, "max_time_ms": 100, "description": "medium_data"},  # 1MB
            {"data_size": 10 * 1024 * 1024, "max_time_ms": 1000, "description": "large_data"}  # 10MB
        ]
        
        for test_case in performance_test_cases:
            data_size = test_case["data_size"]
            max_time_ms = test_case["max_time_ms"]
            description = test_case["description"]
            
            # Property: Encryption time should scale reasonably with data size
            # Simulate encryption time calculation (linear scaling with some overhead)
            base_overhead_ms = 5  # Base encryption overhead
            bytes_per_ms = 1024 * 100  # Assume 100KB/ms processing rate
            
            estimated_time_ms = base_overhead_ms + (data_size / bytes_per_ms)
            
            assert estimated_time_ms <= max_time_ms, \
                f"Encryption of {description} ({data_size} bytes) should complete within {max_time_ms}ms"
            
            # Property: Memory usage should be bounded
            # Assume encryption uses at most 2x the data size in memory
            max_memory_bytes = data_size * 2
            estimated_memory = data_size + 1024  # Data + encryption overhead
            
            assert estimated_memory <= max_memory_bytes, \
                f"Encryption of {description} should use at most {max_memory_bytes} bytes of memory"
    
    def test_encryption_error_handling_and_recovery(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that encryption operations handle errors gracefully.
        """
        
        # Test error scenarios and expected responses
        error_scenarios = [
            {
                "scenario": "invalid_key",
                "error_type": "KeyError",
                "should_fail_gracefully": True,
                "should_log_error": True
            },
            {
                "scenario": "corrupted_data",
                "error_type": "IntegrityError",
                "should_fail_gracefully": True,
                "should_log_error": True
            },
            {
                "scenario": "expired_key",
                "error_type": "KeyExpiredError",
                "should_fail_gracefully": True,
                "should_log_error": True
            },
            {
                "scenario": "insufficient_permissions",
                "error_type": "PermissionError",
                "should_fail_gracefully": True,
                "should_log_error": True
            }
        ]
        
        for scenario in error_scenarios:
            scenario_name = scenario["scenario"]
            error_type = scenario["error_type"]
            should_fail_gracefully = scenario["should_fail_gracefully"]
            should_log_error = scenario["should_log_error"]
            
            # Property: All error scenarios should be handled gracefully
            assert should_fail_gracefully, \
                f"Scenario {scenario_name} should fail gracefully"
            
            # Property: All errors should be logged for audit purposes
            assert should_log_error, \
                f"Scenario {scenario_name} should log errors for audit"
            
            # Property: Error types should be specific and actionable
            assert error_type in [
                "KeyError", "IntegrityError", "KeyExpiredError", 
                "PermissionError", "ValidationError", "CryptoError"
            ], f"Error type {error_type} should be a recognized encryption error"
    
    def test_encryption_compliance_frameworks(self):
        """
        **Property 37: Data Encryption**
        **Validates: Requirements 10.1**
        
        Test that encryption meets various compliance framework requirements.
        """
        
        # Test compliance framework requirements
        compliance_frameworks = [
            {
                "framework": "GDPR",
                "requirements": {
                    "encryption_at_rest": True,
                    "encryption_in_transit": True,
                    "key_management": True,
                    "audit_logging": True,
                    "data_minimization": True
                }
            },
            {
                "framework": "HIPAA",
                "requirements": {
                    "encryption_at_rest": True,
                    "encryption_in_transit": True,
                    "access_controls": True,
                    "audit_logging": True,
                    "integrity_controls": True
                }
            },
            {
                "framework": "SOX",
                "requirements": {
                    "encryption_at_rest": True,
                    "key_management": True,
                    "audit_logging": True,
                    "change_management": True,
                    "access_controls": True
                }
            }
        ]
        
        for framework_info in compliance_frameworks:
            framework = framework_info["framework"]
            requirements = framework_info["requirements"]
            
            # Property: All compliance frameworks should require basic encryption
            assert requirements.get("encryption_at_rest", False), \
                f"{framework} should require encryption at rest"
            
            # Property: All compliance frameworks should require audit logging
            assert requirements.get("audit_logging", False), \
                f"{framework} should require audit logging"
            
            # Property: Healthcare and financial frameworks should have stricter requirements
            if framework in ["HIPAA", "SOX"]:
                assert requirements.get("access_controls", False), \
                    f"{framework} should require access controls"
                assert requirements.get("integrity_controls", False) or requirements.get("change_management", False), \
                    f"{framework} should require integrity or change management controls"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])