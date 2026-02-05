"""Authentication and authorization dependencies for FastAPI."""

from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_async_db
from ..models.user import User
from ..services.auth import AuthService, get_auth_service

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        auth_service: Authentication service
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    try:
        from ..utils.security import verify_token
        
        # Verify JWT token
        payload = verify_token(credentials.credentials)
        user_id = payload.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        user = await auth_service.get_user_by_id(UUID(user_id))
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is disabled",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user.
    
    Args:
        current_user: Current user from token
        
    Returns:
        Current active user
        
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def require_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require admin privileges.
    
    Args:
        current_user: Current active user
        
    Returns:
        Current user if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


async def require_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require superuser privileges.
    
    Args:
        current_user: Current active user
        
    Returns:
        Current user if superuser
        
    Raises:
        HTTPException: If user is not superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser privileges required"
        )
    return current_user


def require_permissions(*permissions: str):
    """Require specific permissions.
    
    Args:
        permissions: Required permissions
        
    Returns:
        Dependency function
    """
    async def permission_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        # Check if user has required permissions
        user_permissions = set(current_user.permissions or [])
        required_permissions = set(permissions)
        
        if not required_permissions.issubset(user_permissions):
            missing_permissions = required_permissions - user_permissions
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(missing_permissions)}"
            )
        
        return current_user
    
    return permission_checker


def require_roles(*roles: str):
    """Require specific roles.
    
    Args:
        roles: Required roles
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        # Check if user has required roles
        user_roles = set(current_user.roles or [])
        required_roles = set(roles)
        
        if not required_roles.intersection(user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(required_roles)}"
            )
        
        return current_user
    
    return role_checker


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[User]:
    """Get current user if authenticated, otherwise None.
    
    Args:
        credentials: Optional HTTP authorization credentials
        auth_service: Authentication service
        
    Returns:
        Current user or None if not authenticated
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, auth_service)
    except HTTPException:
        return None