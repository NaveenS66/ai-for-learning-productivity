"""Authentication API endpoints."""

from datetime import timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_async_db
from ..schemas.user import Token, LoginRequest, UserCreate, UserResponse
from ..services.auth import AuthService, get_auth_service
from ..services.user import UserService
from ..logging_config import get_logger

router = APIRouter(prefix="/auth", tags=["authentication"])
logger = get_logger(__name__)
settings = get_settings()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_async_db)
) -> UserResponse:
    """Register a new user."""
    try:
        user_service = UserService(db)
        user = await user_service.create_user(user_data)
        
        logger.info("User registered successfully", user_id=str(user.id), username=user.username)
        return UserResponse.model_validate(user)
        
    except ValueError as e:
        logger.warning("Registration failed", error=str(e), email=user_data.email)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Registration error", error=str(e), email=user_data.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login(
    login_data: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> Token:
    """Authenticate user and return access token."""
    user = await auth_service.authenticate_user(login_data.username, login_data.password)
    
    if not user:
        logger.warning("Login failed", username=login_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth_service.create_access_token_for_user(user)
    
    logger.info("User logged in successfully", user_id=str(user.id), username=user.username)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
) -> Token:
    """OAuth2 compatible token endpoint."""
    user = await auth_service.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth_service.create_access_token_for_user(user)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    auth_service: AuthService = Depends(get_auth_service)
) -> Token:
    """Refresh access token (placeholder for future implementation)."""
    # TODO: Implement refresh token logic
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Refresh token not implemented yet"
    )


@router.post("/logout")
async def logout() -> Dict[str, Any]:
    """Logout user (placeholder for future implementation)."""
    # TODO: Implement token blacklisting if needed
    return {"message": "Successfully logged out"}