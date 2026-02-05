"""Pytest configuration and fixtures."""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ai_learning_accelerator.database import Base, get_async_db
from ai_learning_accelerator.main import app

# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_SYNC_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    # Create async engine for testing
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(test_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with test database."""
    
    async def override_get_db():
        yield test_db
    
    app.dependency_overrides[get_async_db] = override_get_db
    
    async with AsyncClient(base_url="http://test") as client:
        client.app = app
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "full_name": "Test User",
        "password": "testpassword123"
    }


@pytest.fixture
def sample_learning_content():
    """Sample learning content for testing."""
    return {
        "title": "Introduction to Python",
        "description": "Learn Python basics",
        "content_type": "tutorial",
        "difficulty": "beginner",
        "estimated_duration": 60,
        "tags": ["python", "programming", "basics"]
    }