"""Custom exceptions for AI Learning Accelerator."""

from typing import Any, Dict, Optional
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    AI_MODEL = "ai_model"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    PRIVACY = "privacy"
    SECURITY = "security"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"


class AILearningAcceleratorException(Exception):
    """Base exception for AI Learning Accelerator."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        recoverable: bool = True,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.user_message = user_message or self._generate_user_message()
        self.recoverable = recoverable
    
    def _generate_user_message(self) -> str:
        """Generate user-friendly error message."""
        if self.category == ErrorCategory.AI_MODEL:
            return "We're experiencing issues with our AI services. Please try again in a moment."
        elif self.category == ErrorCategory.DATABASE:
            return "We're having trouble accessing your data. Please try again shortly."
        elif self.category == ErrorCategory.AUTHENTICATION:
            return "Please check your login credentials and try again."
        elif self.category == ErrorCategory.AUTHORIZATION:
            return "You don't have permission to perform this action."
        elif self.category == ErrorCategory.VALIDATION:
            return "Please check your input and try again."
        elif self.category == ErrorCategory.NETWORK:
            return "Network connection issue. Please check your connection and try again."
        else:
            return "Something went wrong. Please try again or contact support if the problem persists."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "recoverable": self.recoverable,
        }


# AI Model Exceptions
class AIModelException(AILearningAcceleratorException):
    """Base exception for AI model errors."""
    
    def __init__(self, message: str, model_name: str, **kwargs):
        super().__init__(
            message=message,
            error_code="AI_MODEL_ERROR",
            category=ErrorCategory.AI_MODEL,
            **kwargs
        )
        self.details["model_name"] = model_name


class ModelInferenceException(AIModelException):
    """Exception for model inference failures."""
    
    def __init__(self, model_name: str, input_data: Optional[Dict] = None, **kwargs):
        super().__init__(
            message=f"Model inference failed for {model_name}",
            model_name=model_name,
            error_code="MODEL_INFERENCE_FAILED",
            **kwargs
        )
        if input_data:
            self.details["input_data"] = input_data


class ModelLoadException(AIModelException):
    """Exception for model loading failures."""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Failed to load model {model_name}",
            model_name=model_name,
            error_code="MODEL_LOAD_FAILED",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if model_path:
            self.details["model_path"] = model_path


class ModelTimeoutException(AIModelException):
    """Exception for model timeout errors."""
    
    def __init__(self, model_name: str, timeout_seconds: float, **kwargs):
        super().__init__(
            message=f"Model {model_name} timed out after {timeout_seconds} seconds",
            model_name=model_name,
            error_code="MODEL_TIMEOUT",
            **kwargs
        )
        self.details["timeout_seconds"] = timeout_seconds


# Database Exceptions
class DatabaseException(AILearningAcceleratorException):
    """Base exception for database errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            category=ErrorCategory.DATABASE,
            **kwargs
        )


class DatabaseConnectionException(DatabaseException):
    """Exception for database connection failures."""
    
    def __init__(self, database_url: Optional[str] = None, **kwargs):
        super().__init__(
            message="Failed to connect to database",
            error_code="DATABASE_CONNECTION_FAILED",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )
        if database_url:
            self.details["database_url"] = database_url


class DataIntegrityException(DatabaseException):
    """Exception for data integrity violations."""
    
    def __init__(self, table: str, constraint: str, **kwargs):
        super().__init__(
            message=f"Data integrity violation in table {table}: {constraint}",
            error_code="DATA_INTEGRITY_VIOLATION",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.details.update({"table": table, "constraint": constraint})


# Authentication/Authorization Exceptions
class AuthenticationException(AILearningAcceleratorException):
    """Base exception for authentication errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            category=ErrorCategory.AUTHENTICATION,
            **kwargs
        )


class InvalidCredentialsException(AuthenticationException):
    """Exception for invalid credentials."""
    
    def __init__(self, **kwargs):
        super().__init__(
            message="Invalid credentials provided",
            error_code="INVALID_CREDENTIALS",
            user_message="Invalid username or password. Please try again.",
            **kwargs
        )


class TokenExpiredException(AuthenticationException):
    """Exception for expired tokens."""
    
    def __init__(self, **kwargs):
        super().__init__(
            message="Authentication token has expired",
            error_code="TOKEN_EXPIRED",
            user_message="Your session has expired. Please log in again.",
            **kwargs
        )


class AuthorizationException(AILearningAcceleratorException):
    """Base exception for authorization errors."""
    
    def __init__(self, message: str, required_permission: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            category=ErrorCategory.AUTHORIZATION,
            **kwargs
        )
        if required_permission:
            self.details["required_permission"] = required_permission


class InsufficientPermissionsException(AuthorizationException):
    """Exception for insufficient permissions."""
    
    def __init__(self, required_permission: str, **kwargs):
        super().__init__(
            message=f"Insufficient permissions: {required_permission} required",
            required_permission=required_permission,
            error_code="INSUFFICIENT_PERMISSIONS",
            **kwargs
        )


# Privacy and Security Exceptions
class PrivacyViolationException(AILearningAcceleratorException):
    """Exception for privacy boundary violations."""
    
    def __init__(self, violation_type: str, user_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=f"Privacy violation detected: {violation_type}",
            error_code="PRIVACY_VIOLATION",
            category=ErrorCategory.PRIVACY,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )
        self.details["violation_type"] = violation_type
        if user_id:
            self.details["user_id"] = user_id


class SecurityException(AILearningAcceleratorException):
    """Base exception for security errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class EncryptionException(SecurityException):
    """Exception for encryption/decryption failures."""
    
    def __init__(self, operation: str, **kwargs):
        super().__init__(
            message=f"Encryption operation failed: {operation}",
            error_code="ENCRYPTION_FAILED",
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.details["operation"] = operation


# Integration Exceptions
class IntegrationException(AILearningAcceleratorException):
    """Base exception for integration errors."""
    
    def __init__(self, message: str, service_name: str, **kwargs):
        super().__init__(
            message=message,
            error_code="INTEGRATION_ERROR",
            category=ErrorCategory.INTEGRATION,
            **kwargs
        )
        self.details["service_name"] = service_name


class ExternalServiceException(IntegrationException):
    """Exception for external service failures."""
    
    def __init__(self, service_name: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message=f"External service {service_name} is unavailable",
            service_name=service_name,
            error_code="EXTERNAL_SERVICE_UNAVAILABLE",
            **kwargs
        )
        if status_code:
            self.details["status_code"] = status_code


class PluginException(IntegrationException):
    """Exception for plugin errors."""
    
    def __init__(self, plugin_name: str, operation: str, **kwargs):
        super().__init__(
            message=f"Plugin {plugin_name} failed during {operation}",
            service_name=plugin_name,
            error_code="PLUGIN_ERROR",
            **kwargs
        )
        self.details["operation"] = operation


# Validation Exceptions
class ValidationException(AILearningAcceleratorException):
    """Base exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            **kwargs
        )
        if field:
            self.details["field"] = field


class InvalidInputException(ValidationException):
    """Exception for invalid input data."""
    
    def __init__(self, field: str, value: Any, expected: str, **kwargs):
        super().__init__(
            message=f"Invalid value for {field}: expected {expected}, got {type(value).__name__}",
            field=field,
            error_code="INVALID_INPUT",
            **kwargs
        )
        self.details.update({"value": str(value), "expected": expected})


# Configuration Exceptions
class ConfigurationException(AILearningAcceleratorException):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )
        if config_key:
            self.details["config_key"] = config_key


class MissingConfigurationException(ConfigurationException):
    """Exception for missing configuration values."""
    
    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            message=f"Missing required configuration: {config_key}",
            config_key=config_key,
            error_code="MISSING_CONFIGURATION",
            **kwargs
        )


# Network Exceptions
class NetworkException(AILearningAcceleratorException):
    """Base exception for network errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            category=ErrorCategory.NETWORK,
            **kwargs
        )


class TimeoutException(NetworkException):
    """Exception for network timeouts."""
    
    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(
            message=f"Operation {operation} timed out after {timeout_seconds} seconds",
            error_code="NETWORK_TIMEOUT",
            **kwargs
        )
        self.details.update({"operation": operation, "timeout_seconds": timeout_seconds})


class ConnectionException(NetworkException):
    """Exception for connection failures."""
    
    def __init__(self, host: str, port: Optional[int] = None, **kwargs):
        super().__init__(
            message=f"Failed to connect to {host}" + (f":{port}" if port else ""),
            error_code="CONNECTION_FAILED",
            **kwargs
        )
        self.details["host"] = host
        if port:
            self.details["port"] = port