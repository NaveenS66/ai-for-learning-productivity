"""Logging configuration for AI Learning Accelerator."""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import Processor

from .config import get_settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Configure structlog
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
    ]
    
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_request_middleware(request_data: Dict[str, Any]) -> None:
    """Log HTTP request data."""
    logger = get_logger("api.request")
    logger.info(
        "HTTP request",
        method=request_data.get("method"),
        path=request_data.get("path"),
        query_params=request_data.get("query_params"),
        user_id=request_data.get("user_id"),
    )


def log_response_middleware(response_data: Dict[str, Any]) -> None:
    """Log HTTP response data."""
    logger = get_logger("api.response")
    logger.info(
        "HTTP response",
        status_code=response_data.get("status_code"),
        processing_time=response_data.get("processing_time"),
        user_id=response_data.get("user_id"),
    )