"""Comprehensive error handling system for AI Learning Accelerator."""

import traceback
import uuid
from typing import Any, Dict, Optional, Union
from datetime import datetime

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from pydantic import ValidationError

from ..logging_config import get_logger
from .exceptions import (
    AILearningAcceleratorException,
    ErrorSeverity,
    ErrorCategory,
    DatabaseException,
    DatabaseConnectionException,
    DataIntegrityException,
    ValidationException,
    InvalidInputException,
    AuthenticationException,
    AuthorizationException,
    PrivacyViolationException,
    SecurityException,
    IntegrationException,
    NetworkException,
    ConfigurationException
)

logger = get_logger(__name__)


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: list = []
        self.max_recent_errors = 100
    
    async def handle_exception(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle any exception and return appropriate JSON response."""
        
        # Generate unique error ID for tracking
        error_id = str(uuid.uuid4())
        
        # Convert exception to our standard format
        error_info = self._process_exception(exc, error_id, request)
        
        # Log the error
        await self._log_error(error_info, request, exc)
        
        # Track error statistics
        self._track_error(error_info)
        
        # Send alerts for critical errors
        if error_info["severity"] == ErrorSeverity.CRITICAL:
            await self._send_alert(error_info, request)
        
        # Return appropriate HTTP response
        return self._create_error_response(error_info)
    
    def _process_exception(
        self,
        exc: Exception,
        error_id: str,
        request: Request
    ) -> Dict[str, Any]:
        """Process exception into standardized error information."""
        
        # Handle our custom exceptions
        if isinstance(exc, AILearningAcceleratorException):
            return {
                "error_id": error_id,
                "error_code": exc.error_code,
                "message": exc.message,
                "user_message": exc.user_message,
                "category": exc.category.value,
                "severity": exc.severity.value,
                "details": exc.details,
                "recoverable": exc.recoverable,
                "http_status": self._get_http_status(exc),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Handle SQLAlchemy exceptions
        elif isinstance(exc, SQLAlchemyError):
            return self._handle_sqlalchemy_error(exc, error_id)
        
        # Handle FastAPI validation errors
        elif isinstance(exc, (RequestValidationError, ValidationError)):
            return self._handle_validation_error(exc, error_id)
        
        # Handle HTTP exceptions
        elif isinstance(exc, (HTTPException, StarletteHTTPException)):
            return self._handle_http_error(exc, error_id)
        
        # Handle unknown exceptions
        else:
            return self._handle_unknown_error(exc, error_id)
    
    def _handle_sqlalchemy_error(self, exc: SQLAlchemyError, error_id: str) -> Dict[str, Any]:
        """Handle SQLAlchemy database errors."""
        if isinstance(exc, IntegrityError):
            # Data integrity violation
            constraint = str(exc.orig) if hasattr(exc, 'orig') else "unknown"
            return DataIntegrityException(
                table="unknown",
                constraint=constraint
            ).to_dict()
        
        elif isinstance(exc, OperationalError):
            # Database connection or operational error
            return DatabaseConnectionException().to_dict()
        
        else:
            # Generic database error
            return DatabaseException(
                message=f"Database error: {str(exc)}",
                details={"original_error": str(exc)}
            ).to_dict()
    
    def _handle_validation_error(
        self,
        exc: Union[RequestValidationError, ValidationError],
        error_id: str
    ) -> Dict[str, Any]:
        """Handle validation errors."""
        if isinstance(exc, RequestValidationError):
            errors = exc.errors()
            field = errors[0]["loc"][-1] if errors and errors[0]["loc"] else "unknown"
            message = errors[0]["msg"] if errors else "Validation error"
        else:
            errors = exc.errors()
            field = errors[0]["loc"][-1] if errors and errors[0]["loc"] else "unknown"
            message = errors[0]["msg"] if errors else "Validation error"
        
        return InvalidInputException(
            field=str(field),
            value="invalid",
            expected="valid input",
            details={"validation_errors": errors}
        ).to_dict()
    
    def _handle_http_error(
        self,
        exc: Union[HTTPException, StarletteHTTPException],
        error_id: str
    ) -> Dict[str, Any]:
        """Handle HTTP exceptions."""
        if exc.status_code == 401:
            return AuthenticationException(
                message="Authentication required",
                details={"detail": exc.detail}
            ).to_dict()
        
        elif exc.status_code == 403:
            return AuthorizationException(
                message="Access forbidden",
                details={"detail": exc.detail}
            ).to_dict()
        
        elif exc.status_code == 404:
            return ValidationException(
                message="Resource not found",
                details={"detail": exc.detail}
            ).to_dict()
        
        else:
            return {
                "error_id": error_id,
                "error_code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail),
                "user_message": str(exc.detail),
                "category": ErrorCategory.BUSINESS_LOGIC.value,
                "severity": ErrorSeverity.MEDIUM.value,
                "details": {},
                "recoverable": True,
                "http_status": exc.status_code,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _handle_unknown_error(self, exc: Exception, error_id: str) -> Dict[str, Any]:
        """Handle unknown/unexpected errors."""
        return {
            "error_id": error_id,
            "error_code": "UNKNOWN_ERROR",
            "message": f"Unexpected error: {str(exc)}",
            "user_message": "An unexpected error occurred. Please try again or contact support.",
            "category": ErrorCategory.BUSINESS_LOGIC.value,
            "severity": ErrorSeverity.HIGH.value,
            "details": {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)
            },
            "recoverable": True,
            "http_status": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_http_status(self, exc: AILearningAcceleratorException) -> int:
        """Get appropriate HTTP status code for custom exception."""
        if exc.category == ErrorCategory.AUTHENTICATION:
            return status.HTTP_401_UNAUTHORIZED
        elif exc.category == ErrorCategory.AUTHORIZATION:
            return status.HTTP_403_FORBIDDEN
        elif exc.category == ErrorCategory.VALIDATION:
            return status.HTTP_400_BAD_REQUEST
        elif exc.category == ErrorCategory.DATABASE:
            if exc.severity == ErrorSeverity.CRITICAL:
                return status.HTTP_503_SERVICE_UNAVAILABLE
            else:
                return status.HTTP_500_INTERNAL_SERVER_ERROR
        elif exc.category == ErrorCategory.AI_MODEL:
            return status.HTTP_503_SERVICE_UNAVAILABLE
        elif exc.category == ErrorCategory.INTEGRATION:
            return status.HTTP_502_BAD_GATEWAY
        elif exc.category == ErrorCategory.NETWORK:
            return status.HTTP_503_SERVICE_UNAVAILABLE
        elif exc.category == ErrorCategory.PRIVACY:
            return status.HTTP_403_FORBIDDEN
        elif exc.category == ErrorCategory.SECURITY:
            return status.HTTP_403_FORBIDDEN
        else:
            return status.HTTP_500_INTERNAL_SERVER_ERROR
    
    async def _log_error(
        self,
        error_info: Dict[str, Any],
        request: Request,
        exc: Exception
    ):
        """Log error with appropriate level and context."""
        severity = ErrorSeverity(error_info["severity"])
        
        # Prepare log context
        log_context = {
            "error_id": error_info["error_id"],
            "error_code": error_info["error_code"],
            "category": error_info["category"],
            "severity": error_info["severity"],
            "recoverable": error_info["recoverable"],
            "http_status": error_info["http_status"],
            "request_method": request.method,
            "request_path": str(request.url.path),
            "request_query": dict(request.query_params),
            "user_id": getattr(request.state, "user_id", None),
            "user_agent": request.headers.get("user-agent"),
            "client_ip": request.client.host if request.client else None
        }
        
        # Add exception details for high/critical errors
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            log_context["exception_type"] = type(exc).__name__
            log_context["exception_message"] = str(exc)
            log_context["traceback"] = traceback.format_exc()
        
        # Log with appropriate level
        if severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **log_context)
        elif severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", **log_context)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", **log_context)
        else:
            logger.info("Low severity error occurred", **log_context)
    
    def _track_error(self, error_info: Dict[str, Any]):
        """Track error statistics."""
        error_code = error_info["error_code"]
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        
        # Add to recent errors (keep only last N)
        self.recent_errors.append({
            "error_id": error_info["error_id"],
            "error_code": error_code,
            "category": error_info["category"],
            "severity": error_info["severity"],
            "timestamp": error_info["timestamp"]
        })
        
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
    
    async def _send_alert(self, error_info: Dict[str, Any], request: Request):
        """Send alert for critical errors."""
        # This would integrate with alerting systems (email, Slack, PagerDuty, etc.)
        # For now, just log the alert
        logger.critical(
            "ALERT: Critical error requires immediate attention",
            error_id=error_info["error_id"],
            error_code=error_info["error_code"],
            message=error_info["message"],
            request_path=str(request.url.path),
            user_id=getattr(request.state, "user_id", None)
        )
    
    def _create_error_response(self, error_info: Dict[str, Any]) -> JSONResponse:
        """Create JSON error response."""
        # Remove internal details from response
        response_data = {
            "error": {
                "error_id": error_info["error_id"],
                "error_code": error_info["error_code"],
                "message": error_info["user_message"],
                "category": error_info["category"],
                "recoverable": error_info["recoverable"],
                "timestamp": error_info["timestamp"]
            }
        }
        
        # Add retry information for recoverable errors
        if error_info["recoverable"]:
            response_data["error"]["retry_after"] = 60  # seconds
        
        # Add support contact for high/critical errors
        if error_info["severity"] in [ErrorSeverity.HIGH.value, ErrorSeverity.CRITICAL.value]:
            response_data["error"]["support_contact"] = "support@ailearningaccelerator.com"
            response_data["error"]["support_reference"] = error_info["error_id"]
        
        return JSONResponse(
            status_code=error_info["http_status"],
            content=response_data
        )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_counts_by_code": self.error_counts,
            "recent_errors_count": len(self.recent_errors),
            "most_common_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def get_recent_errors(self, limit: int = 50) -> list:
        """Get recent errors."""
        return self.recent_errors[-limit:]
    
    def clear_stats(self):
        """Clear error statistics."""
        self.error_counts.clear()
        self.recent_errors.clear()


# Global error handler instance
error_handler = ErrorHandler()


# Exception handler functions for FastAPI
async def ai_learning_accelerator_exception_handler(
    request: Request,
    exc: AILearningAcceleratorException
) -> JSONResponse:
    """Handle custom AI Learning Accelerator exceptions."""
    return await error_handler.handle_exception(request, exc)


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI validation exceptions."""
    return await error_handler.handle_exception(request, exc)


async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    return await error_handler.handle_exception(request, exc)


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle all other exceptions."""
    return await error_handler.handle_exception(request, exc)


# Health check function for error handling system
def get_error_handler_health() -> Dict[str, Any]:
    """Get health status of error handling system."""
    stats = error_handler.get_error_stats()
    recent_critical_errors = [
        err for err in error_handler.get_recent_errors(10)
        if err["severity"] == ErrorSeverity.CRITICAL.value
    ]
    
    return {
        "status": "healthy" if len(recent_critical_errors) == 0 else "degraded",
        "total_errors": stats["total_errors"],
        "recent_critical_errors": len(recent_critical_errors),
        "error_rate": len(error_handler.recent_errors) / max(error_handler.max_recent_errors, 1),
        "most_common_error": stats["most_common_errors"][0] if stats["most_common_errors"] else None
    }