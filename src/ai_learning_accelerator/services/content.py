"""Content service for learning content and knowledge base management."""

from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import selectinload

from ..models.content import (
    LearningContent, ContentRating, KnowledgeBase, 
    ContentCollection, CollectionItem, QualityStatus
)
from ..schemas.content import (
    LearningContentCreate, LearningContentUpdate,
    ContentRatingCreate, ContentRatingUpdate,
    KnowledgeBaseCreate, KnowledgeBaseUpdate,
    ContentCollectionCreate, ContentCollectionUpdate,
    CollectionItemCreate, CollectionItemUpdate,
    ContentSearchRequest
)
from ..logging_config import get_logger

logger = get_logger(__name__)


class ContentService:
    """Service for learning content management operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    # Learning Content Methods
    
    async def create_learning_content(
        self, 
        content_data: LearningContentCreate, 
        author_id: Optional[UUID] = None
    ) -> LearningContent:
        """Create new learning content."""
        try:
            content = LearningContent(
                author_id=author_id,
                **content_data.model_dump()
            )
            
            self.db.add(content)
            await self.db.commit()
            await self.db.refresh(content)
            
            logger.info("Learning content created", content_id=str(content.id), title=content.title)
            return content
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating learning content", error=str(e), title=content_data.title)
            raise
    
    async def get_learning_content(self, content_id: UUID) -> Optional[LearningContent]:
        """Get learning content by ID."""
        try:
            stmt = select(LearningContent).where(LearningContent.id == content_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching learning content", error=str(e), content_id=str(content_id))
            return None
    
    async def update_learning_content(
        self, 
        content_id: UUID, 
        content_data: LearningContentUpdate
    ) -> Optional[LearningContent]:
        """Update learning content."""
        try:
            content = await self.get_learning_content(content_id)
            if not content:
                return None
            
            # Update fields if provided
            update_data = content_data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(content, field, value)
            
            await self.db.commit()
            await self.db.refresh(content)
            
            logger.info("Learning content updated", content_id=str(content_id))
            return content
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error updating learning content", error=str(e), content_id=str(content_id))
            raise
    
    async def search_learning_content(
        self, 
        search_request: ContentSearchRequest
    ) -> Tuple[List[LearningContent], int]:
        """Search learning content with filters."""
        try:
            query = select(LearningContent).where(LearningContent.is_active == True)
            
            # Apply filters
            if search_request.query:
                search_term = f"%{search_request.query}%"
                query = query.where(
                    or_(
                        LearningContent.title.ilike(search_term),
                        LearningContent.description.ilike(search_term),
                        LearningContent.content_text.ilike(search_term)
                    )
                )
            
            if search_request.content_type:
                query = query.where(LearningContent.content_type == search_request.content_type)
            
            if search_request.difficulty_level:
                query = query.where(LearningContent.difficulty_level == search_request.difficulty_level)
            
            if search_request.min_rating:
                query = query.where(LearningContent.average_rating >= search_request.min_rating)
            
            if search_request.language:
                query = query.where(LearningContent.language == search_request.language)
            
            if search_request.tags:
                # PostgreSQL JSON array contains check
                for tag in search_request.tags:
                    query = query.where(LearningContent.tags.op('@>')([tag]))
            
            if search_request.topics:
                # PostgreSQL JSON array contains check
                for topic in search_request.topics:
                    query = query.where(LearningContent.topics.op('@>')([topic]))
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            count_result = await self.db.execute(count_query)
            total_count = count_result.scalar()
            
            # Apply ordering, limit, and offset
            query = query.order_by(desc(LearningContent.average_rating), desc(LearningContent.created_at))
            query = query.offset(search_request.offset).limit(search_request.limit)
            
            result = await self.db.execute(query)
            content_list = result.scalars().all()
            
            return content_list, total_count
            
        except Exception as e:
            logger.error("Error searching learning content", error=str(e))
            return [], 0
    
    async def increment_view_count(self, content_id: UUID) -> bool:
        """Increment view count for content."""
        try:
            content = await self.get_learning_content(content_id)
            if content:
                content.view_count += 1
                await self.db.commit()
                return True
            return False
        except Exception as e:
            logger.error("Error incrementing view count", error=str(e), content_id=str(content_id))
            return False
    
    # Content Rating Methods
    
    async def create_content_rating(
        self, 
        content_id: UUID, 
        user_id: UUID, 
        rating_data: ContentRatingCreate
    ) -> ContentRating:
        """Create or update content rating."""
        try:
            # Check if rating already exists
            existing_rating = await self.get_user_content_rating(content_id, user_id)
            
            if existing_rating:
                # Update existing rating
                update_data = rating_data.model_dump(exclude_unset=True)
                for field, value in update_data.items():
                    setattr(existing_rating, field, value)
                
                await self.db.commit()
                await self.db.refresh(existing_rating)
                
                # Update content average rating
                await self._update_content_rating_stats(content_id)
                
                logger.info("Content rating updated", content_id=str(content_id), user_id=str(user_id))
                return existing_rating
            else:
                # Create new rating
                rating = ContentRating(
                    content_id=content_id,
                    user_id=user_id,
                    **rating_data.model_dump()
                )
                
                self.db.add(rating)
                await self.db.commit()
                await self.db.refresh(rating)
                
                # Update content average rating
                await self._update_content_rating_stats(content_id)
                
                logger.info("Content rating created", content_id=str(content_id), user_id=str(user_id))
                return rating
                
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating content rating", error=str(e), content_id=str(content_id))
            raise
    
    async def get_user_content_rating(self, content_id: UUID, user_id: UUID) -> Optional[ContentRating]:
        """Get user's rating for specific content."""
        try:
            stmt = select(ContentRating).where(
                and_(ContentRating.content_id == content_id, ContentRating.user_id == user_id)
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching user content rating", error=str(e))
            return None
    
    async def get_content_ratings(self, content_id: UUID, limit: int = 50) -> List[ContentRating]:
        """Get ratings for content."""
        try:
            stmt = select(ContentRating).where(ContentRating.content_id == content_id)\
                .order_by(desc(ContentRating.created_at)).limit(limit)
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error("Error fetching content ratings", error=str(e), content_id=str(content_id))
            return []
    
    async def _update_content_rating_stats(self, content_id: UUID) -> None:
        """Update content rating statistics."""
        try:
            # Calculate average rating and count
            stmt = select(
                func.avg(ContentRating.rating).label('avg_rating'),
                func.count(ContentRating.id).label('rating_count')
            ).where(ContentRating.content_id == content_id)
            
            result = await self.db.execute(stmt)
            stats = result.first()
            
            if stats:
                content = await self.get_learning_content(content_id)
                if content:
                    content.average_rating = float(stats.avg_rating or 0.0)
                    content.rating_count = stats.rating_count or 0
                    await self.db.commit()
                    
        except Exception as e:
            logger.error("Error updating content rating stats", error=str(e), content_id=str(content_id))
    
    # Knowledge Base Methods
    
    async def create_knowledge_base_entry(
        self, 
        entry_data: KnowledgeBaseCreate, 
        author_id: Optional[UUID] = None
    ) -> KnowledgeBase:
        """Create knowledge base entry."""
        try:
            entry = KnowledgeBase(
                author_id=author_id,
                last_updated_by=author_id,
                **entry_data.model_dump()
            )
            
            self.db.add(entry)
            await self.db.commit()
            await self.db.refresh(entry)
            
            logger.info("Knowledge base entry created", entry_id=str(entry.id), title=entry.title)
            return entry
            
        except Exception as e:
            await self.db.rollback()
            logger.error("Error creating knowledge base entry", error=str(e), title=entry_data.title)
            raise
    
    async def get_knowledge_base_entry(self, entry_id: UUID) -> Optional[KnowledgeBase]:
        """Get knowledge base entry by ID."""
        try:
            stmt = select(KnowledgeBase).where(KnowledgeBase.id == entry_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching knowledge base entry", error=str(e), entry_id=str(entry_id))
            return None
    
    async def get_knowledge_base_entry_by_slug(self, slug: str) -> Optional[KnowledgeBase]:
        """Get knowledge base entry by slug."""
        try:
            stmt = select(KnowledgeBase).where(KnowledgeBase.slug == slug)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching knowledge base entry by slug", error=str(e), slug=slug)
            return None
    
    async def search_knowledge_base(
        self, 
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[KnowledgeBase], int]:
        """Search knowledge base entries."""
        try:
            stmt = select(KnowledgeBase).where(
                KnowledgeBase.quality_status.in_([QualityStatus.APPROVED, QualityStatus.PUBLISHED])
            )
            
            if query:
                search_term = f"%{query}%"
                stmt = stmt.where(
                    or_(
                        KnowledgeBase.title.ilike(search_term),
                        KnowledgeBase.summary.ilike(search_term),
                        KnowledgeBase.content.ilike(search_term)
                    )
                )
            
            if category:
                stmt = stmt.where(KnowledgeBase.category == category)
            
            # Get total count
            count_stmt = select(func.count()).select_from(stmt.subquery())
            count_result = await self.db.execute(count_stmt)
            total_count = count_result.scalar()
            
            # Apply ordering, limit, and offset
            stmt = stmt.order_by(desc(KnowledgeBase.relevance_score), desc(KnowledgeBase.created_at))
            stmt = stmt.offset(offset).limit(limit)
            
            result = await self.db.execute(stmt)
            entries = result.scalars().all()
            
            return entries, total_count
            
        except Exception as e:
            logger.error("Error searching knowledge base", error=str(e))
            return [], 0
    
    async def increment_knowledge_base_access(self, entry_id: UUID) -> bool:
        """Increment access count for knowledge base entry."""
        try:
            entry = await self.get_knowledge_base_entry(entry_id)
            if entry:
                entry.access_count += 1
                await self.db.commit()
                return True
            return False
        except Exception as e:
            logger.error("Error incrementing access count", error=str(e), entry_id=str(entry_id))
            return False