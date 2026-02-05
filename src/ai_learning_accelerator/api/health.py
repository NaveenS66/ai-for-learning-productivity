"""Health check endpoints."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_db
from ..logging_config import get_logger
from ..utils.monitoring import monitoring_system
from ..utils.circuit_breaker import circuit_breaker_registry
from ..utils.error_handler import error_handler
from ..utils.fallback import fallback_registry

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger(__name__)


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-learning-accelerator",
        "version": "0.1.0"
    }


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_async_db)) -> Dict[str, Any]:
    """Readiness check including database connectivity."""
    try:
        # Test database connection
        await db.execute("SELECT 1")
        db_status = "connected"
        logger.info("Readiness check passed")
    except Exception as e:
        db_status = f"error: {str(e)}"
        logger.error("Readiness check failed", error=str(e))
    
    return {
        "status": "ready" if db_status == "connected" else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-learning-accelerator",
        "version": "0.1.0",
        "checks": {
            "database": db_status
        }
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check for container orchestration."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-learning-accelerator"
    }


@router.get("/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_async_db)) -> Dict[str, Any]:
    """Detailed health check with all system components."""
    
    # Test database connection
    try:
        await db.execute("SELECT 1")
        database_status = "healthy"
        database_error = None
    except Exception as e:
        database_status = "unhealthy"
        database_error = str(e)
    
    # Get monitoring system status
    system_status = monitoring_system.get_system_status()
    
    # Get circuit breaker stats
    circuit_breaker_stats = circuit_breaker_registry.get_all_stats()
    
    # Get error handler stats
    error_stats = error_handler.get_error_stats()
    
    # Get fallback handler stats
    fallback_stats = fallback_registry.get_all_stats()
    
    # Get health check results
    health_results = await monitoring_system.health.run_all_health_checks()
    
    # Determine overall status
    overall_status = "healthy"
    if database_status != "healthy":
        overall_status = "unhealthy"
    elif system_status["health"]["status"] != "healthy":
        overall_status = system_status["health"]["status"]
    elif system_status["alerts"]["critical_count"] > 0:
        overall_status = "critical"
    elif system_status["alerts"]["error_count"] > 0:
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-learning-accelerator",
        "version": "0.1.0",
        "components": {
            "database": {
                "status": database_status,
                "error": database_error
            },
            "monitoring": {
                "status": "healthy" if monitoring_system._started else "stopped",
                "metrics": system_status["metrics"],
                "alerts": system_status["alerts"]
            },
            "circuit_breakers": {
                "status": "healthy" if not any(
                    stats["state"] == "open" 
                    for stats in circuit_breaker_stats.values()
                ) else "degraded",
                "breakers": circuit_breaker_stats
            },
            "error_handling": {
                "status": "healthy" if error_stats["total_errors"] == 0 else "active",
                "stats": error_stats
            },
            "fallback_system": {
                "status": "healthy",
                "handlers": fallback_stats
            }
        },
        "health_checks": health_results
    }


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics."""
    return {
        "system_status": monitoring_system.get_system_status(),
        "circuit_breakers": circuit_breaker_registry.get_all_stats(),
        "error_stats": error_handler.get_error_stats(),
        "fallback_stats": fallback_registry.get_all_stats()
    }


@router.get("/alerts")
async def get_alerts() -> Dict[str, Any]:
    """Get current alerts."""
    active_alerts = monitoring_system.alerts.get_active_alerts()
    recent_alerts = monitoring_system.alerts.get_alerts(since=monitoring_system.get_system_status()["timestamp"] - 3600)  # Last hour
    
    return {
        "active_alerts": [alert.to_dict() for alert in active_alerts],
        "recent_alerts": [alert.to_dict() for alert in recent_alerts],
        "summary": {
            "active_count": len(active_alerts),
            "recent_count": len(recent_alerts),
            "critical_count": len([a for a in active_alerts if a.severity.value == "critical"]),
            "error_count": len([a for a in active_alerts if a.severity.value == "error"])
        }
    }


@router.post("/alerts/{alert_name}/resolve")
async def resolve_alert(alert_name: str) -> Dict[str, Any]:
    """Resolve an alert by name."""
    await monitoring_system.alerts.resolve_alert(alert_name)
    return {"message": f"Alert '{alert_name}' resolved"}


@router.post("/circuit-breakers/reset")
async def reset_all_circuit_breakers() -> Dict[str, Any]:
    """Reset all circuit breakers."""
    await circuit_breaker_registry.reset_all()
    return {"message": "All circuit breakers reset"}


@router.post("/circuit-breakers/{breaker_name}/reset")
async def reset_circuit_breaker(breaker_name: str) -> Dict[str, Any]:
    """Reset a specific circuit breaker."""
    await circuit_breaker_registry.reset_breaker(breaker_name)
    return {"message": f"Circuit breaker '{breaker_name}' reset"}


@router.delete("/error-stats")
async def clear_error_stats() -> Dict[str, Any]:
    """Clear error statistics."""
    error_handler.clear_stats()
    return {"message": "Error statistics cleared"}


@router.delete("/fallback-caches")
async def clear_fallback_caches() -> Dict[str, Any]:
    """Clear all fallback caches."""
    fallback_registry.clear_all_caches()
    return {"message": "All fallback caches cleared"}