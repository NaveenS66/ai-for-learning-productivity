"""Property-based tests for base functionality."""

import pytest
import asyncio
from contextlib import asynccontextmanager
from hypothesis import given, strategies as st
from sqlalchemy import Column, String
from sqlalchemy.ext.asyncio import AsyncSession

from ai_learning_accelerator.models.base import BaseModel
from ai_learning_accelerator.database import get_async_db


class TestModel(BaseModel):
    """Test model for property testing."""
    __tablename__ = "test_model"
    
    name = Column(String(100), nullable=False)
    description = Column(String(500))


class BasePropertyTest:
    """Base class for property-based tests with database support."""
    
    @asynccontextmanager
    async def get_db_session(self):
        """Get a database session for testing."""
        async for db in get_async_db():
            try:
                yield db
            finally:
                await db.rollback()  # Rollback any changes made during testing


class TestBaseModelProperties(BasePropertyTest):
    """Property-based tests for BaseModel functionality."""
    
    @given(st.text(min_size=1, max_size=100))
    def test_model_creation_property(self, name: str):
        """Property test: BaseModel instances should be created with valid data."""
        model = TestModel(name=name)
        
        assert model.name == name
        assert model.id is not None
        assert model.created_at is not None
        assert model.updated_at is not None
        assert model.created_at == model.updated_at  # Should be equal on creation
    
    @given(
        st.text(min_size=1, max_size=100),
        st.one_of(st.none(), st.text(max_size=500))
    )
    def test_model_to_dict_property(self, name: str, description):
        """Property test: to_dict should preserve all model data."""
        model = TestModel(name=name, description=description)
        model_dict = model.to_dict()
        
        assert isinstance(model_dict, dict)
        assert model_dict["name"] == name
        assert model_dict["description"] == description
        assert "id" in model_dict
        assert "created_at" in model_dict
        assert "updated_at" in model_dict
    
    def test_model_repr_property(self):
        """Property test: __repr__ should include model class and id."""
        model = TestModel(name="test")
        repr_str = repr(model)
        
        assert "TestModel" in repr_str
        assert str(model.id) in repr_str
        assert repr_str.startswith("<")
        assert repr_str.endswith(">")