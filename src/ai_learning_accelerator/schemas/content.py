"""Pydantic schemas for content-related API operations."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from ..models.content import ContentType, ContentFormat, QualityStatus
from ..models.user import DifficultyLevel


class LearningContentBase(BaseModel):
    """Base learning content schema."""
    title: str = Field(..., min_length=1, max_length=300)
    description: Optional[str] = None
    content_type: ContentType
    content_format: ContentFormat = ContentFormat.TEXT
    difficulty_level: DifficultyLevel
    estimated_duration: Optional[int] = Field(None, ge=1)
    language: str = "en"
    prerequisites: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    content_text: Optional[str] = None
    content_data: dict = Field(default_factory=dict)


class LearningContentCreate(LearningContentBase):
    """Schema for learning content creation."""
    pass


class LearningContentUpdate(BaseModel):
    """Schema for learning content updates."""
    title: Optional[str] = Field(None, min_length=1, max_length=300)
    description: Optional[str] = None
    content_type: Optional[ContentType] = None
    content_format: Optional[ContentFormat] = None
    difficulty_level: Optional[DifficultyLevel] = None
    estimated_duration: Optional[int] = Field(None, ge=1)
    language: Optional[str] = None
    prerequisites: Optional[List[str]] = None
    learning_objectives: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    content_text: Optional[str] = None
    content_data: Optional[dict] = None
    quality_status: Optional[QualityStatus] = None
    review_notes: Optional[str] = None
    version: Optional[str] = None
    is_active: Optional[bool] = None


class LearningContentResponse(LearningContentBase):
    """Schema for learning content responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    quality_status: QualityStatus
    quality_score: float
    review_notes: Optional[str] = None
    author_id: Optional[UUID] = None
    version: str
    is_active: bool
    view_count: int
    completion_count: int
    average_rating: float
    rating_count: int
    created_at: datetime
    updated_at: datetime


class ContentRatingBase(BaseModel):
    """Base content rating schema."""
    rating: int = Field(..., ge=1, le=5)
    review_text: Optional[str] = None
    clarity_rating: Optional[int] = Field(None, ge=1, le=5)
    usefulness_rating: Optional[int] = Field(None, ge=1, le=5)
    accuracy_rating: Optional[int] = Field(None, ge=1, le=5)
    difficulty_rating: Optional[int] = Field(None, ge=1, le=5)


class ContentRatingCreate(ContentRatingBase):
    """Schema for content rating creation."""
    pass


class ContentRatingUpdate(BaseModel):
    """Schema for content rating updates."""
    rating: Optional[int] = Field(None, ge=1, le=5)
    review_text: Optional[str] = None
    clarity_rating: Optional[int] = Field(None, ge=1, le=5)
    usefulness_rating: Optional[int] = Field(None, ge=1, le=5)
    accuracy_rating: Optional[int] = Field(None, ge=1, le=5)
    difficulty_rating: Optional[int] = Field(None, ge=1, le=5)


class ContentRatingResponse(ContentRatingBase):
    """Schema for content rating responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    content_id: UUID
    user_id: UUID
    is_verified: bool
    helpful_votes: int
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseBase(BaseModel):
    """Base knowledge base schema."""
    title: str = Field(..., min_length=1, max_length=300)
    slug: str = Field(..., min_length=1, max_length=200)
    category: str = Field(..., min_length=1, max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    summary: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    content_format: ContentFormat = ContentFormat.MARKDOWN
    keywords: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    external_links: List[str] = Field(default_factory=list)


class KnowledgeBaseCreate(KnowledgeBaseBase):
    """Schema for knowledge base entry creation."""
    pass


class KnowledgeBaseUpdate(BaseModel):
    """Schema for knowledge base entry updates."""
    title: Optional[str] = Field(None, min_length=1, max_length=300)
    slug: Optional[str] = Field(None, min_length=1, max_length=200)
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    summary: Optional[str] = Field(None, min_length=1)
    content: Optional[str] = Field(None, min_length=1)
    content_format: Optional[ContentFormat] = None
    keywords: Optional[List[str]] = None
    related_topics: Optional[List[str]] = None
    external_links: Optional[List[str]] = None
    quality_status: Optional[QualityStatus] = None
    review_due: Optional[datetime] = None
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_featured: Optional[bool] = None


class KnowledgeBaseResponse(KnowledgeBaseBase):
    """Schema for knowledge base entry responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    quality_status: QualityStatus
    last_reviewed: Optional[datetime] = None
    review_due: Optional[datetime] = None
    access_count: int
    relevance_score: float
    is_featured: bool
    author_id: Optional[UUID] = None
    last_updated_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime


class ContentCollectionBase(BaseModel):
    """Base content collection schema."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    collection_type: str = "manual"
    difficulty_level: Optional[DifficultyLevel] = None
    estimated_duration: Optional[int] = Field(None, ge=1)
    tags: List[str] = Field(default_factory=list)
    is_public: bool = True
    is_sequential: bool = False
    curator_notes: Optional[str] = None


class ContentCollectionCreate(ContentCollectionBase):
    """Schema for content collection creation."""
    pass


class ContentCollectionUpdate(BaseModel):
    """Schema for content collection updates."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    collection_type: Optional[str] = None
    difficulty_level: Optional[DifficultyLevel] = None
    estimated_duration: Optional[int] = Field(None, ge=1)
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    is_sequential: Optional[bool] = None
    curator_notes: Optional[str] = None


class CollectionItemBase(BaseModel):
    """Base collection item schema."""
    content_id: UUID
    order_index: int = Field(..., ge=0)
    is_required: bool = True
    notes: Optional[str] = None
    custom_title: Optional[str] = Field(None, max_length=300)


class CollectionItemCreate(CollectionItemBase):
    """Schema for collection item creation."""
    pass


class CollectionItemUpdate(BaseModel):
    """Schema for collection item updates."""
    order_index: Optional[int] = Field(None, ge=0)
    is_required: Optional[bool] = None
    notes: Optional[str] = None
    custom_title: Optional[str] = Field(None, max_length=300)


class CollectionItemResponse(CollectionItemBase):
    """Schema for collection item responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    collection_id: UUID
    created_at: datetime
    updated_at: datetime


class ContentCollectionResponse(ContentCollectionBase):
    """Schema for content collection responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    owner_id: UUID
    subscriber_count: int
    completion_rate: float
    items: List[CollectionItemResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ContentSearchRequest(BaseModel):
    """Schema for content search requests."""
    query: Optional[str] = None
    content_type: Optional[ContentType] = None
    difficulty_level: Optional[DifficultyLevel] = None
    tags: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    min_rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    language: Optional[str] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class ContentSearchResponse(BaseModel):
    """Schema for content search responses."""
    total_count: int
    items: List[LearningContentResponse]
    facets: dict = Field(default_factory=dict)  # Search facets for filtering