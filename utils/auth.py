# utils/auth.py
"""
Authentication and authorization utilities.

Provides JWT-based authentication, permission checking, and user management.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from functools import wraps

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging

from utils.db import get_connection
from utils.exceptions import UnauthorizedError

logger = logging.getLogger(__name__)

# Configuration (should come from environment variables)
SECRET_KEY = "your-secret-key-change-in-production"  # Use env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token security
security = HTTPBearer()


class TokenData:
    """Data extracted from JWT token."""
    
    def __init__(self, user_id: int, email: str, permissions: List[str]):
        self.user_id = user_id
        self.email = email
        self.permissions = permissions


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Bcrypt hashed password
        
    Returns:
        bool: True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Data to encode in the token
        
    Returns:
        str: Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        dict: Decoded token payload
        
    Raises:
        JWTError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"Invalid token: {e}")
        raise


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Get the current authenticated user from JWT token.
    
    Dependency for FastAPI endpoints requiring authentication.
    
    Args:
        credentials: HTTP bearer credentials
        
    Returns:
        dict: User information
        
    Raises:
        HTTPException 401: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = decode_token(token)
        
        # Validate token type
        if payload.get("type") != "access":
            raise credentials_exception
        
        user_id: int = payload.get("sub")
        email: str = payload.get("email")
        
        if user_id is None or email is None:
            raise credentials_exception
        
        # Fetch user from database
        user = get_user_by_id(user_id)
        if user is None:
            raise credentials_exception
        
        # Check if user is active
        if not user.get("is_active", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user account"
            )
        
        return user
        
    except JWTError:
        raise credentials_exception
    except Exception as e:
        logger.error(f"Error in authentication: {e}")
        raise credentials_exception


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Get the current user if authenticated, None otherwise.
    
    Useful for endpoints that work both with and without authentication.
    
    Args:
        request: FastAPI request
        credentials: HTTP bearer credentials (optional)
        
    Returns:
        Optional[dict]: User information or None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch user from database by ID.
    
    Args:
        user_id: User's database ID
        
    Returns:
        Optional[dict]: User data or None if not found
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    id,
                    email,
                    is_active,
                    created_at,
                    last_login
                FROM users
                WHERE id = %s
                """,
                (user_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return {
                "id": row[0],
                "email": row[1],
                "is_active": row[2],
                "created_at": row[3],
                "last_login": row[4]
            }
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()


def get_user_permissions(user_id: int) -> List[str]:
    """
    Get list of permissions for a user.
    
    Args:
        user_id: User's database ID
        
    Returns:
        List[str]: List of permission names
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT p.name
                FROM permissions p
                JOIN role_permissions rp ON rp.permission_id = p.id
                JOIN user_roles ur ON ur.role_id = rp.role_id
                WHERE ur.user_id = %s
                """,
                (user_id,)
            )
            
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching permissions for user {user_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()


def has_permission(user: Dict[str, Any], permission: str) -> bool:
    """
    Check if user has a specific permission.
    
    Args:
        user: User object
        permission: Permission name to check
        
    Returns:
        bool: True if user has the permission
    """
    user_permissions = get_user_permissions(user["id"])
    return permission in user_permissions or "admin" in user_permissions


def require_permissions(*permissions: str):
    """
    Decorator to require specific permissions for an endpoint.
    
    Args:
        *permissions: Permission names required
        
    Example:
        @router.post("/admin/users")
        @require_permissions("user:create", "admin")
        async def create_user():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs (injected by Depends)
            current_user = kwargs.get("current_user")
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check if user has any of the required permissions
            user_has_permission = any(
                has_permission(current_user, perm)
                for perm in permissions
            )
            
            if not user_has_permission:
                logger.warning(
                    f"Permission denied for user {current_user['id']}",
                    extra={
                        "user_id": current_user["id"],
                        "required_permissions": permissions,
                        "endpoint": func.__name__
                    }
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permission. Required: {', '.join(permissions)}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user with email and password.
    
    Args:
        email: User's email
        password: Plain text password
        
    Returns:
        Optional[dict]: User data if authentication succeeds, None otherwise
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    id,
                    email,
                    password_hash,
                    is_active
                FROM users
                WHERE email = %s
                """,
                (email,)
            )
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Authentication failed: user not found - {email}")
                return None
            
            user_id, email, password_hash, is_active = row
            
            if not is_active:
                logger.warning(f"Authentication failed: inactive user - {email}")
                return None
            
            if not verify_password(password, password_hash):
                logger.warning(f"Authentication failed: invalid password - {email}")
                return None
            
            # Update last login
            cursor.execute(
                """
                UPDATE users
                SET last_login = NOW()
                WHERE id = %s
                """,
                (user_id,)
            )
            conn.commit()
            
            logger.info(f"User authenticated successfully: {email}")
            
            return {
                "id": user_id,
                "email": email,
                "is_active": is_active
            }
            
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        return None
    finally:
        if conn:
            conn.close()


# Middleware to attach user to request state
async def auth_middleware(request: Request, call_next):
    """
    Middleware to extract and validate authentication for all requests.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/handler
    """
    # Try to get user from bearer token
    auth_header = request.headers.get("Authorization")
    
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = decode_token(token)
            user_id = payload.get("sub")
            
            if user_id:
                user = get_user_by_id(user_id)
                if user:
                    request.state.user = user
        except Exception as e:
            logger.debug(f"Failed to extract user from token: {e}")
    
    response = await call_next(request)
    return response
