# utils/auth.py
"""
Production-Ready Authentication & Authorization
Handles OAuth2, JWT tokens, and user permissions
"""

from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, validator
import logging
import secrets
import hashlib

from utils.db import get_db_connection, execute_query
from config import settings

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 30  # 30 days

# OAuth2 schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
bearer_scheme = HTTPBearer()


# ============================================================================
# MODELS
# ============================================================================

class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data"""
    user_id: Optional[int] = None
    email: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    """User model"""
    id: int
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime
    
    @validator('email')
    def email_must_be_lowercase(cls, v):
        return v.lower()


class UserCreate(BaseModel):
    """User creation model"""
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v
    
    @validator('email')
    def email_must_be_lowercase(cls, v):
        return v.lower()


# ============================================================================
# PASSWORD UTILITIES
# ============================================================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def generate_password_reset_token() -> str:
    """Generate a secure random token for password resets"""
    return secrets.token_urlsafe(32)


# ============================================================================
# JWT TOKEN UTILITIES
# ============================================================================

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Payload data to encode
        expires_delta: Optional expiration time delta
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key.get_secret_value(),
        algorithm=ALGORITHM
    )
    
    return encoded_jwt


def create_refresh_token(user_id: int) -> str:
    """
    Create a refresh token
    
    Args:
        user_id: User ID
        
    Returns:
        Encoded refresh token
    """
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "sub": str(user_id),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key.get_secret_value(),
        algorithm=ALGORITHM
    )
    
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================================
# USER OPERATIONS
# ============================================================================

def get_user_by_email(email: str) -> Optional[User]:
    """
    Get user by email address
    
    Args:
        email: User email address
        
    Returns:
        User object or None if not found
    """
    try:
        result = execute_query(
            """
            SELECT id, email, full_name, is_active, is_admin, created_at
            FROM users
            WHERE email = %s
            """,
            (email.lower(),),
            fetch='one'
        )
        
        if result:
            return User(**result)
        return None
        
    except Exception as e:
        logger.error(f"Error getting user by email: {e}", exc_info=True)
        return None


def get_user_by_id(user_id: int) -> Optional[User]:
    """
    Get user by ID
    
    Args:
        user_id: User ID
        
    Returns:
        User object or None if not found
    """
    try:
        result = execute_query(
            """
            SELECT id, email, full_name, is_active, is_admin, created_at
            FROM users
            WHERE id = %s
            """,
            (user_id,),
            fetch='one'
        )
        
        if result:
            return User(**result)
        return None
        
    except Exception as e:
        logger.error(f"Error getting user by ID: {e}", exc_info=True)
        return None


def authenticate_user(email: str, password: str) -> Optional[User]:
    """
    Authenticate user with email and password
    
    Args:
        email: User email
        password: Plain text password
        
    Returns:
        User object if authenticated, None otherwise
    """
    try:
        result = execute_query(
            """
            SELECT id, email, password_hash, full_name, is_active, is_admin, created_at
            FROM users
            WHERE email = %s
            """,
            (email.lower(),),
            fetch='one'
        )
        
        if not result:
            logger.warning(f"User not found: {email}")
            return None
        
        if not result['is_active']:
            logger.warning(f"Inactive user attempted login: {email}")
            return None
        
        if not verify_password(password, result['password_hash']):
            logger.warning(f"Invalid password for user: {email}")
            return None
        
        # Remove password_hash from result
        user_data = {k: v for k, v in result.items() if k != 'password_hash'}
        return User(**user_data)
        
    except Exception as e:
        logger.error(f"Error authenticating user: {e}", exc_info=True)
        return None


def create_user(user_data: UserCreate) -> Optional[User]:
    """
    Create a new user
    
    Args:
        user_data: User creation data
        
    Returns:
        Created User object or None on error
    """
    try:
        # Check if user already exists
        existing_user = get_user_by_email(user_data.email)
        if existing_user:
            logger.warning(f"User already exists: {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        password_hash = hash_password(user_data.password)
        
        # Insert user
        result = execute_query(
            """
            INSERT INTO users (email, password_hash, full_name, is_active, is_admin)
            VALUES (%s, %s, %s, TRUE, FALSE)
            RETURNING id, email, full_name, is_active, is_admin, created_at
            """,
            (user_data.email.lower(), password_hash, user_data.full_name),
            fetch='one'
        )
        
        if result:
            logger.info(f"Created new user: {user_data.email}")
            return User(**result)
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}", exc_info=True)
        return None


# ============================================================================
# DEPENDENCY FUNCTIONS
# ============================================================================

async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    FastAPI dependency to get current authenticated user from JWT token
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        Current User object
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
        
        token_type = payload.get("type")
        if token_type != "access":
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_id(int(user_id))
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user
    
    Args:
        current_user: Current user from get_current_user dependency
        
    Returns:
        Active User object
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current admin user
    
    Args:
        current_user: Current user from get_current_user dependency
        
    Returns:
        Admin User object
        
    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_permissions(required_scopes: List[str]):
    """
    Dependency factory for permission checking
    
    Args:
        required_scopes: List of required permission scopes
        
    Returns:
        Dependency function that checks permissions
    """
    async def permission_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        # For now, just check if user is active
        # In future, implement granular permissions
        if not current_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return permission_checker


# ============================================================================
# TOKEN OPERATIONS
# ============================================================================

def login(email: str, password: str) -> Optional[Token]:
    """
    Login user and create tokens
    
    Args:
        email: User email
        password: User password
        
    Returns:
        Token object with access and refresh tokens
    """
    user = authenticate_user(email, password)
    if not user:
        return None
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email}
    )
    refresh_token = create_refresh_token(user.id)
    
    # Log successful login
    logger.info(f"User logged in: {user.email}")
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
    )


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """
    Create new access token from refresh token
    
    Args:
        refresh_token: Refresh token
        
    Returns:
        New access token or None if invalid
    """
    try:
        payload = decode_token(refresh_token)
        
        # Verify it's a refresh token
        if payload.get("type") != "refresh":
            logger.warning("Invalid token type for refresh")
            return None
        
        user_id = int(payload.get("sub"))
        user = get_user_by_id(user_id)
        
        if not user or not user.is_active:
            logger.warning(f"Cannot refresh token for inactive user: {user_id}")
            return None
        
        # Create new access token
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        
        return access_token
        
    except Exception as e:
        logger.error(f"Error refreshing access token: {e}", exc_info=True)
        return None


# ============================================================================
# API KEY MANAGEMENT (for service-to-service auth)
# ============================================================================

def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'User',
    'UserCreate',
    'Token',
    'TokenData',
    'hash_password',
    'verify_password',
    'create_access_token',
    'create_refresh_token',
    'decode_token',
    'get_user_by_email',
    'get_user_by_id',
    'authenticate_user',
    'create_user',
    'get_current_user',
    'get_current_active_user',
    'get_current_admin_user',
    'require_permissions',
    'login',
    'refresh_access_token',
    'generate_api_key',
    'hash_api_key',
]
