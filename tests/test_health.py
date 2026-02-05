"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


def test_health_check(client: TestClient):
    """Test basic health check endpoint."""
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ai-learning-accelerator"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_readiness_check(async_client: AsyncClient):
    """Test readiness check endpoint."""
    response = await async_client.get("/api/v1/health/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ready"
    assert data["service"] == "ai-learning-accelerator"
    assert "checks" in data
    assert "database" in data["checks"]


def test_liveness_check(client: TestClient):
    """Test liveness check endpoint."""
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "alive"
    assert data["service"] == "ai-learning-accelerator"
    assert "timestamp" in data


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert data["version"] == "0.1.0"