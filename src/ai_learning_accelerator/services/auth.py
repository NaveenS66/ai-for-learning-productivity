"""Authentication service for user management."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..config import get_settings
from ..database import get_async_db
from ..models.user import User
from ..schemas.user import TokenData
from ..utils.security import verify_password, verify_token, create_access_token
from ..logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()
security = HTTPBearer()


class AuthService:
    """Authentication service for user operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password."""
        try:
            # Try to find user by username or email
            stmt = select(User).where(
                (User.username == username) | (User.email == username)
            )
            result = await self.db.execute(stmt)
            user = result.scalar_one_or_none()
            
            if not user:
                logger.warning("Authentication failed: user not found", username=username)
                return None
            
            if not user.is_active:
                logger.warning("Authentication failed: user inactive", username=username)
                return None
            
            if not verify_password(password, user.hashed_password):
                logger.warning("Authentication failed: invalid password", username=username)
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            await self.db.commit()
            
            logger.info("User authenticated successfully", username=username, user_id=str(user.id))
            return user
            
        except Exception as e:
            logger.error("Authentication error", error=str(e), username=username)
            return None
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        try:
            stmt = select(User).where(User.id == user_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching user by ID", error=str(e), user_id=str(user_id))
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            stmt = select(User).where(User.username == username)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching user by username", error=str(e), username=username)
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            stmt = select(User).where(User.email == email)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error fetching user by email", error=str(e), email=email)
            return None
    
    def create_access_token_for_user(self, user: User) -> str:
        """Create access token for user."""
        data = {
            "sub": user.username,
            "user_id": str(user.id),
            "email": user.email
        }
        return create_access_token(data)


async def get_auth_service(db: AsyncSession = Depends(get_async_db)) -> AuthService:
    """Dependency to get authentication service."""
    return AuthService(db)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify token
        payload = verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        
        username: str = payload.get("sub")
        user_id_str: str = payload.get("user_id")
        
        if username is None or user_id_str is None:
            raise credentials_exception
        
        token_data = TokenData(username=username, user_id=UUID(user_id_str))
        
    except Exception as e:
        logger.warning("Token validation failed", error=str(e))
        raise credentials_exception
    
    # Get user from database
    user = await auth_service.get_user_by_id(token_data.user_id)
    if user is None:
        logger.warning("User not found for valid token", user_id=user_id_str)
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user (alias for clarity)."""
    return current_user

# Global auth service instance for convenience
auth_service = None


async def get_auth_service_instance(db: AsyncSession = Depends(get_async_db)) -> AuthService:
    """Get global auth service instance."""
    global auth_service
    if auth_service is None:
        auth_service = AuthService(db)
    return auth_service