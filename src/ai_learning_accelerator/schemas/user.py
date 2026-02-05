"""Pydantic schemas for user-related API operations."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, ConfigDict

from ..models.user import LearningStyle, DifficultyLevel, DataSharingLevel, SkillLevel


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: str = Field(..., min_length=1, max_length=200)


class UserCreate(UserBase):
    """Schema for user creation."""
    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """Schema for user updates."""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, min_length=1, max_length=200)
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """Schema for user responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class UserProfileBase(BaseModel):
    """Base user profile schema."""
    timezone: str = "UTC"
    learning_style: LearningStyle = LearningStyle.MULTIMODAL
    difficulty_preference: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    content_formats: List[str] = Field(default_factory=list)
    primary_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    project_types: List[str] = Field(default_factory=list)
    data_sharing: DataSharingLevel = DataSharingLevel.ANONYMOUS
    analytics_opt_in: bool = True
    content_personalization: bool = True
    bio: Optional[str] = None
    goals: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)


class UserProfileCreate(UserProfileBase):
    """Schema for user profile creation."""
    pass


class UserProfileUpdate(BaseModel):
    """Schema for user profile updates."""
    timezone: Optional[str] = None
    learning_style: Optional[LearningStyle] = None
    difficulty_preference: Optional[DifficultyLevel] = None
    content_formats: Optional[List[str]] = None
    primary_languages: Optional[List[str]] = None
    frameworks: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    project_types: Optional[List[str]] = None
    data_sharing: Optional[DataSharingLevel] = None
    analytics_opt_in: Optional[bool] = None
    content_personalization: Optional[bool] = None
    bio: Optional[str] = None
    goals: Optional[List[str]] = None
    interests: Optional[List[str]] = None


class UserProfileResponse(UserProfileBase):
    """Schema for user profile responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime


class SkillAssessmentBase(BaseModel):
    """Base skill assessment schema."""
    domain: str = Field(..., min_length=1, max_length=100)
    skill_level: SkillLevel
    confidence_score: int = Field(50, ge=0, le=100)
    assessment_method: Optional[str] = None
    evidence: dict = Field(default_factory=dict)
    progress_notes: Optional[str] = None


class SkillAssessmentCreate(SkillAssessmentBase):
    """Schema for skill assessment creation."""
    pass


class SkillAssessmentUpdate(BaseModel):
    """Schema for skill assessment updates."""
    skill_level: Optional[SkillLevel] = None
    confidence_score: Optional[int] = Field(None, ge=0, le=100)
    assessment_method: Optional[str] = None
    evidence: Optional[dict] = None
    progress_notes: Optional[str] = None


class SkillAssessmentResponse(SkillAssessmentBase):
    """Schema for skill assessment responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    previous_level: Optional[SkillLevel] = None
    last_assessed: datetime
    created_at: datetime
    updated_at: datetime


class LearningActivityBase(BaseModel):
    """Base learning activity schema."""
    activity_type: str = Field(..., min_length=1, max_length=50)
    content_id: Optional[str] = Field(None, max_length=100)
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    duration_minutes: Optional[int] = Field(None, ge=0)
    difficulty_level: Optional[DifficultyLevel] = None
    tags: List[str] = Field(default_factory=list)
    activity_metadata: dict = Field(default_factory=dict)


class LearningActivityCreate(LearningActivityBase):
    """Schema for learning activity creation."""
    pass


class LearningActivityUpdate(BaseModel):
    """Schema for learning activity updates."""
    status: Optional[str] = Field(None, max_length=20)
    completion_percentage: Optional[int] = Field(None, ge=0, le=100)
    score: Optional[int] = None
    duration_minutes: Optional[int] = Field(None, ge=0)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    activity_metadata: Optional[dict] = None


class LearningActivityResponse(LearningActivityBase):
    """Schema for learning activity responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    status: str
    completion_percentage: int
    score: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class Token(BaseModel):
    """JWT token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    user_id: Optional[UUID] = None


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str
    password: str


class UserWithProfile(UserResponse):
    """User response with profile included."""
    profile: Optional[UserProfileResponse] = None
    skill_assessments: List[SkillAssessmentResponse] = Field(default_factory=list)