"""Tests for privacy control functionality."""

import pytest
from uuid import uuid4

from src.ai_learning_accelerator.services.privacy_service import privacy_service
from src.ai_learning_accelerator.models.privacy import (
    DataSharingLevel, PrivacyScope, DataProcessingPurpose, PrivacyViolationType
)


def test_privacy_service_initialization():
    """Test that privacy service initializes correctly."""
    assert privacy_service is not None
    assert hasattr(privacy_service, '_violation_handlers')
    assert hasattr(privacy_service, '_compliance_checkers')
    assert hasattr(privacy_service, '_boundary_validators')


def test_data_sharing_levels():
    """Test data sharing level enumeration."""
    levels = [
        DataSharingLevel.NONE,
        DataSharingLevel.MINIMAL,
        DataSharingLevel.SELECTIVE,
        DataSharingLevel.STANDARD,
        DataSharingLevel.FULL
    ]
    
    assert len(levels) == 5
    assert DataSharingLevel.NONE.value == "none"
    assert DataSharingLevel.SELECTIVE.value == "selective"


def test_privacy_scope_enumeration():
    """Test privacy scope enumeration."""
    scopes = [
        PrivacyScope.GLOBAL,
        PrivacyScope.WORKSPACE,
        PrivacyScope.PROJECT,
        PrivacyScope.SESSION,
        PrivacyScope.CONTENT_TYPE
    ]
    
    assert len(scopes) == 5
    assert PrivacyScope.GLOBAL.value == "global"
    assert PrivacyScope.WORKSPACE.value == "workspace"


def test_data_processing_purposes():
    """Test data processing purpose enumeration."""
    purposes = [
        DataProcessingPurpose.LEARNING_ANALYTICS,
        DataProcessingPurpose.PERSONALIZATION,
        DataProcessingPurpose.DEBUGGING_ASSISTANCE,
        DataProcessingPurpose.AUTOMATION,
        DataProcessingPurpose.CONTENT_RECOMMENDATION,
        DataProcessingPurpose.PROGRESS_TRACKING,
        DataProcessingPurpose.SYSTEM_IMPROVEMENT,
        DataProcessingPurpose.RESEARCH,
        DataProcessingPurpose.MARKETING
    ]
    
    assert len(purposes) == 9
    assert DataProcessingPurpose.LEARNING_ANALYTICS.value == "learning_analytics"
    assert DataProcessingPurpose.PERSONALIZATION.value == "personalization"


def test_privacy_violation_types():
    """Test privacy violation type enumeration."""
    violation_types = [
        PrivacyViolationType.UNAUTHORIZED_ACCESS,
        PrivacyViolationType.DATA_SHARING_VIOLATION,
        PrivacyViolationType.BOUNDARY_CROSSING,
        PrivacyViolationType.CONSENT_VIOLATION,
        PrivacyViolationType.RETENTION_VIOLATION,
        PrivacyViolationType.PURPOSE_VIOLATION
    ]
    
    assert len(violation_types) == 6
    assert PrivacyViolationType.UNAUTHORIZED_ACCESS.value == "unauthorized_access"
    assert PrivacyViolationType.CONSENT_VIOLATION.value == "consent_violation"


def test_boundary_rule_evaluation():
    """Test boundary rule evaluation logic."""
    # Test simple rule evaluation
    rules = {
        "data_types": ["user_data", "learning_data"],
        "operations": ["read", "analyze"]
    }
    
    # Should pass - data type and operation match
    result = privacy_service._evaluate_boundary_rules(
        rules, "user_data", "read", {}
    )
    assert result is True
    
    # Should fail - data type doesn't match
    result = privacy_service._evaluate_boundary_rules(
        rules, "system_data", "read", {}
    )
    assert result is False
    
    # Should fail - operation doesn't match
    result = privacy_service._evaluate_boundary_rules(
        rules, "user_data", "write", {}
    )
    assert result is False


def test_compliance_score_calculation():
    """Test privacy compliance score calculation."""
    # Perfect compliance
    score = privacy_service._calculate_compliance_score(0, 5, 3)
    assert score == 100.0  # Base 100 + 10 (consents) + 3 (settings) - 0 (violations)
    
    # With violations
    score = privacy_service._calculate_compliance_score(2, 3, 2)
    assert score == 100.0 - 20 + 6 + 2  # 88.0
    
    # High violations
    score = privacy_service._calculate_compliance_score(10, 0, 0)
    assert score == 50.0  # Capped violation penalty
    
    # Minimum score
    score = privacy_service._calculate_compliance_score(20, 0, 0)
    assert score == 50.0  # Capped violation penalty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])