import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from config import settings

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

class User(BaseModel):
    """User model for authentication"""
    id: str
    email: str
    user_tier: str = "free"
    is_active: bool = True

class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    email: str
    user_tier: str
    exp: datetime

def create_access_token(user_id: str, email: str, user_tier: str = "free") -> str:
    """Create JWT access token"""
    try:
        payload = {
            "user_id": user_id,
            "email": email,
            "user_tier": user_tier,
            "exp": datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow(),
            "iss": settings.SERVICE_NAME
        }
        
        token = jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        
        return token
        
    except Exception as e:
        logger.error(f"Failed to create access token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create access token"
        )

def verify_token(token: str) -> TokenData:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": True}
        )
        
        return TokenData(
            user_id=payload["user_id"],
            email=payload["email"],
            user_tier=payload.get("user_tier", "free"),
            exp=datetime.fromtimestamp(payload["exp"])
        )
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    try:
        token_data = verify_token(credentials.credentials)
        
        user = User(
            id=token_data.user_id,
            email=token_data.email,
            user_tier=token_data.user_tier,
            is_active=True
        )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get current user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current user with admin privileges"""
    if current_user.user_tier not in ["enterprise", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

async def get_premium_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current user with premium privileges"""
    if current_user.user_tier not in ["premium", "enterprise", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_user

# Mock user database for development
MOCK_USERS = {
    "test_user_123": {
        "id": "test_user_123",
        "email": "test@example.com",
        "user_tier": "premium",
        "password_hash": "mock_hash"
    },
    "free_user_456": {
        "id": "free_user_456", 
        "email": "free@example.com",
        "user_tier": "free",
        "password_hash": "mock_hash"
    },
    "enterprise_user_789": {
        "id": "enterprise_user_789",
        "email": "enterprise@example.com", 
        "user_tier": "enterprise",
        "password_hash": "mock_hash"
    }
}

def create_test_token(user_id: str) -> str:
    """Create test token for development"""
    if user_id not in MOCK_USERS:
        raise ValueError(f"Unknown test user: {user_id}")
    
    user_data = MOCK_USERS[user_id]
    return create_access_token(
        user_id=user_data["id"],
        email=user_data["email"],
        user_tier=user_data["user_tier"]
    )
