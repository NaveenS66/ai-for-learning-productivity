#!/usr/bin/env python3
"""Debug learning module loading."""

import sys
sys.path.insert(0, 'src')

print("Step 1: Importing base dependencies...")
try:
    from datetime import datetime
    from enum import Enum
    from typing import List, Optional
    from uuid import UUID
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")
    sys.exit(1)

print("Step 2: Importing SQLAlchemy...")
try:
    from sqlalchemy import Boolean, Column, DateTime, Enum as SQLEnum, ForeignKey, Integer, String, Text, JSON, Float
    from sqlalchemy.dialects.postgresql import UUID as PGUUID
    from sqlalchemy.orm import relationship
    print("✓ SQLAlchemy imports successful")
except Exception as e:
    print(f"✗ SQLAlchemy imports failed: {e}")
    sys.exit(1)

print("Step 3: Importing base model...")
try:
    from ai_learning_accelerator.models.base import BaseModel
    print("✓ BaseModel import successful")
except Exception as e:
    print(f"✗ BaseModel import failed: {e}")
    sys.exit(1)

print("Step 4: Importing user models...")
try:
    from ai_learning_accelerator.models.user import DifficultyLevel, SkillLevel
    print("✓ User models import successful")
except Exception as e:
    print(f"✗ User models import failed: {e}")
    sys.exit(1)

print("Step 5: Defining LearningPathStatus enum...")
try:
    class LearningPathStatus(str, Enum):
        """Learning path status."""
        ACTIVE = "active"
        COMPLETED = "completed"
        PAUSED = "paused"
        CANCELLED = "cancelled"
    print(f"✓ LearningPathStatus defined: {LearningPathStatus}")
except Exception as e:
    print(f"✗ LearningPathStatus definition failed: {e}")
    sys.exit(1)

print("Step 6: Loading full learning module...")
try:
    import ai_learning_accelerator.models.learning
    print("✓ Learning module loaded")
except Exception as e:
    print(f"✗ Learning module loading failed: {e}")
    import traceback
    traceback.print_exc()