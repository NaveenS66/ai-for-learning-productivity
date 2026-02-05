"""Main FastAPI application."""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError, HTTPException

from .api.health import router as health_router
from .api.auth import router as auth_router
from .api.users import router as users_router
from .api.context import router as context_router
from .api.automation import router as automation_router
from .api.multimodal import router as multimodal_router
from .api.interaction import router as interaction_router
from .api.analytics import router as analytics_router
from .api.encryption import router as encryption_router
from .api.privacy import router as privacy_router
from .api.plugins import router as plugins_router
from .api.workflow_integration import router as workflow_integration_router
from .api.content_lifecycle import router as content_lifecycle_router
from .api.feedback import router as feedback_router
from .api.workflows import router as workflows_router
from .config import get_settings
from .database import close_db, init_db
from .logging_config import configure_logging, get_logger, log_request_middleware, log_response_middleware
from .utils.exceptions import AILearningAcceleratorException
from .utils.error_handler import (
    ai_learning_accelerator_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler,
    get_error_handler_health
)
from .utils.monitoring import monitoring_system, register_health_check
from .utils.circuit_breaker import circuit_breaker_registry

# Configure logging first
configure_logging()
logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting AI Learning Accelerator", version=settings.app_version)
    
    # Initialize monitoring system
    try:
        await monitoring_system.start()
        logger.info("Monitoring system initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize monitoring system", error=str(e))
        # Continue without monitoring
    
    # Register health checks
    register_health_check("error_handler", get_error_handler_health)
    register_health_check("circuit_breakers", lambda: {
        "status": "healthy",
        "breakers": circuit_breaker_registry.get_all_stats()
    })
    register_health_check("monitoring", lambda: {
        "status": "healthy" if monitoring_system._started else "stopped",
        "system_status": monitoring_system.get_system_status()
    })
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise
    
    # Initialize system orchestrator
    try:
        from .services.system_orchestrator import system_orchestrator
        orchestrator_result = await system_orchestrator.initialize_system()
        
        if orchestrator_result["status"] == "initialized":
            logger.info("System orchestrator initialized successfully")
            logger.info(f"Services: {orchestrator_result['successful_initializations']}/{orchestrator_result['total_services']} initialized")
        else:
            logger.warning("System orchestrator initialization had issues")
            logger.warning(f"Result: {orchestrator_result}")
    except Exception as e:
        logger.error("Failed to initialize system orchestrator", error=str(e))
        # Continue without full orchestration
    
    # Initialize plugin and webhook managers
    try:
        from .plugins import plugin_manager, webhook_manager
        from .services.workflow_integration import workflow_integration_service
        from pathlib import Path
        
        # Add default plugin directories
        plugin_manager.add_plugin_directory(Path("plugins"))
        plugin_manager.add_plugin_directory(Path("./plugins"))
        
        # Start webhook manager
        await webhook_manager.start()
        
        # Start workflow integration service
        await workflow_integration_service.start_service()
        
        logger.info("Plugin, webhook, and workflow integration services initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        # Don't raise - continue without plugin support
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI Learning Accelerator")
    
    # Shutdown system orchestrator
    try:
        from .services.system_orchestrator import system_orchestrator
        shutdown_result = await system_orchestrator.shutdown_system()
        logger.info("System orchestrator shut down successfully")
    except Exception as e:
        logger.error("Error shutting down system orchestrator", error=str(e))
    
    # Shutdown monitoring system
    try:
        await monitoring_system.stop()
        logger.info("Monitoring system shut down successfully")
    except Exception as e:
        logger.error("Error shutting down monitoring system", error=str(e))
    
    # Shutdown plugin and webhook managers
    try:
        from .plugins import plugin_manager, webhook_manager
        from .services.workflow_integration import workflow_integration_service
        
        await workflow_integration_service.stop_service()
        await plugin_manager.shutdown()
        await webhook_manager.stop()
        logger.info("All services shut down successfully")
    except Exception as e:
        logger.error("Error shutting down services", error=str(e))
    
    await close_db()


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered solution for accelerated learning and development productivity",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Add exception handlers
app.add_exception_handler(AILearningAcceleratorException, ai_learning_accelerator_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Response:
    """Log HTTP requests and responses with monitoring."""
    start_time = time.time()
    
    # Log request
    log_request_middleware({
        "method": request.method,
        "path": str(request.url.path),
        "query_params": dict(request.query_params),
        "user_id": getattr(request.state, "user_id", None),
    })
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Log response
    log_response_middleware({
        "status_code": response.status_code,
        "processing_time": f"{processing_time:.4f}s",
        "user_id": getattr(request.state, "user_id", None),
    })
    
    # Record metrics
    try:
        await monitoring_system.record_request_metrics(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration=processing_time
        )
    except Exception as e:
        logger.warning(f"Failed to record request metrics: {e}")
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(processing_time)
    
    return response


# Include routers
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(users_router, prefix=settings.api_prefix)
app.include_router(context_router, prefix=settings.api_prefix)
app.include_router(automation_router, prefix=settings.api_prefix)
app.include_router(multimodal_router, prefix=settings.api_prefix)
app.include_router(interaction_router, prefix=settings.api_prefix)
app.include_router(analytics_router, prefix=settings.api_prefix)
app.include_router(encryption_router, prefix=settings.api_prefix)
app.include_router(privacy_router, prefix=settings.api_prefix)
app.include_router(plugins_router, prefix=settings.api_prefix)
app.include_router(workflow_integration_router, prefix=settings.api_prefix)
app.include_router(content_lifecycle_router, prefix=settings.api_prefix)
app.include_router(feedback_router, prefix=settings.api_prefix)
app.include_router(workflows_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to AI Learning Accelerator",
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "Documentation not available in production",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ai_learning_accelerator.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_config=None,  # Use our custom logging
    )