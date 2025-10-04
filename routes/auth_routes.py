"""
Authentication Service Layer - Production Ready
================================================
Business logic for authentication and session management

Features:
✅ OAuth2 token management
✅ Session management with Redis
✅ User CRUD operations
✅ Security validations
✅ Concurrent session limiting
✅ IP-based security
"""

import asyncio
import secrets
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import httpx
import aioredis
import asyncpg
from google.oauth2.credentials import Credentials
from fastapi import Response

from utils.db import DatabasePool, track_query, QueryType
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SecurityConfig:
    """Security configuration"""
    SESSION_TOKEN_BYTES: int = 32
    SESSION_EXPIRY_DAYS: int = 14
    MAX_SESSIONS_PER_USER: int = 5
    SESSION_COOKIE_NAME: str = "yt_optimizer_session"
    SECURE_COOKIE: bool = settings.is_production
    HTTPONLY_COOKIE: bool = True
    SAMESITE_POLICY: str = "none" if settings.is_production else "lax"


# ============================================================================
# MODELS
# ============================================================================

class UserInfo:
    """User information model"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.google_id = kwargs.get('google_id')
        self.email = kwargs.get('email')
        self.name = kwargs.get('name')
        self.permission_level = kwargs.get('permission_level')
        self.is_free_trial = kwargs.get('is_free_trial', False)
        self.created_at = kwargs.get('created_at')
        self.last_login = kwargs.get('last_login')


class SessionInfo:
    """Session information model"""
    def __init__(self, **kwargs):
        self.session_id = kwargs.get('session_id')
        self.user_id = kwargs.get('user_id')
        self.created_at = kwargs.get('created_at')
        self.expires_at = kwargs.get('expires_at')
        self.ip_address = kwargs.get('ip_address')
        self.user_agent = kwargs.get('user_agent')


# ============================================================================
# AUTHENTICATION SERVICE
# ============================================================================

class AuthService:
    """Authentication service"""
    
    def __init__(self, pool: DatabasePool):
        self.pool = pool
        self.http_timeout = 30.0
    
    async def exchange_code_for_tokens(
        self,
        code: str,
        redirect_uri: str
    ) -> Credentials:
        """Exchange authorization code for access tokens"""
        try:
            # Load client config
            with open(settings.client_secret_file, 'r') as f:
                client_config = json.load(f)['web']
            
            client_id = client_config['client_id']
            client_secret = client_config['client_secret']
            token_uri = client_config['token_uri']
            
            # Exchange code for tokens
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                response = await client.post(
                    token_uri,
                    data={
                        'code': code,
                        'client_id': client_id,
                        'client_secret': client_secret,
                        'redirect_uri': redirect_uri,
                        'grant_type': 'authorization_code'
                    }
                )
            
            if response.status_code != 200:
                logger.error(f"Token exchange failed: {response.text}")
                raise ValueError(f"Token exchange failed: {response.status_code}")
            
            token_data = response.json()
            
            # Create credentials object
            credentials = Credentials(
                token=token_data.get('access_token'),
                refresh_token=token_data.get('refresh_token'),
                token_uri=token_uri,
                client_id=client_id,
                client_secret=client_secret,
                scopes=token_data.get('scope', '').split()
            )
            
            return credentials
            
        except Exception as e:
            logger.error(f"Error exchanging code: {e}")
            raise
    
    async def get_user_info(self, credentials: Credentials) -> Dict[str, Any]:
        """Get user info from Google"""
        try:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                response = await client.get(
                    "https://www.googleapis.com/oauth2/v1/userinfo",
                    headers={"Authorization": f"Bearer {credentials.token}"}
                )
            
            if response.status_code != 200:
                raise ValueError(f"Failed to get user info: {response.status_code}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            raise
    
    @track_query(QueryType.INSERT, 'create_user')
    async def create_or_update_user(
        self,
        google_id: str,
        email: str,
        name: str,
        credentials: Credentials,
        permission_level: str,
        free_trial: bool
    ) -> UserInfo:
        """Create or update user"""
        try:
            # Prepare scopes
            scopes = credentials.scopes if hasattr(credentials, 'scopes') else []
            scopes_array = "{" + ",".join(scopes) + "}" if scopes else "{}"
            
            # Prepare token expiry
            token_expiry = (
                credentials.expiry.isoformat()
                if hasattr(credentials, 'expiry') and credentials.expiry
                else None
            )
            
            async with self.pool.transaction() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO users (
                        google_id,
                        email,
                        name,
                        permission_level,
                        is_free_trial,
                        token,
                        refresh_token,
                        token_uri,
                        client_id,
                        client_secret,
                        scopes,
                        token_expiry,
                        last_login,
                        created_at,
                        updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW(), NOW(), NOW())
                    ON CONFLICT (google_id) DO UPDATE SET
                        email = EXCLUDED.email,
                        name = EXCLUDED.name,
                        permission_level = EXCLUDED.permission_level,
                        is_free_trial = EXCLUDED.is_free_trial,
                        token = EXCLUDED.token,
                        refresh_token = EXCLUDED.refresh_token,
                        token_uri = EXCLUDED.token_uri,
                        client_id = EXCLUDED.client_id,
                        client_secret = EXCLUDED.client_secret,
                        scopes = EXCLUDED.scopes,
                        token_expiry = EXCLUDED.token_expiry,
                        last_login = NOW(),
                        updated_at = NOW()
                    RETURNING id, google_id, email, name, permission_level, 
                              is_free_trial, created_at, last_login
                """,
                    google_id,
                    email,
                    name,
                    permission_level,
                    free_trial,
                    credentials.token,
                    credentials.refresh_token,
                    credentials.token_uri,
                    credentials.client_id,
                    credentials.client_secret,
                    scopes_array,
                    token_expiry
                )
            
            return UserInfo(**dict(row))
            
        except Exception as e:
            logger.error(f"Error creating/updating user: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            return True
        except Exception as e:
            logger.error(f"Auth service health check failed: {e}")
            return False


# ============================================================================
# SESSION SERVICE
# ============================================================================

class SessionService:
    """Session management service"""
    
    def __init__(
        self,
        pool: DatabasePool,
        redis_client: Optional[aioredis.Redis] = None
    ):
        self.pool = pool
        self.redis = redis_client
        self.config = SecurityConfig()
    
    @track_query(QueryType.INSERT, 'create_session')
    async def create_session(
        self,
        user_id: int,
        ip_address: str,
        user_agent: str
    ) -> str:
        """
        Create new session with security features
        
        Security:
        - Limits concurrent sessions per user
        - Tracks IP and user agent
        - Cryptographically secure tokens
        """
        try:
            # Generate secure session token
            session_token = secrets.token_urlsafe(
                self.config.SESSION_TOKEN_BYTES
            )
            
            # Calculate expiry
            expires_at = datetime.now(timezone.utc) + timedelta(
                days=self.config.SESSION_EXPIRY_DAYS
            )
            
            # Create session ID
            session_id = hashlib.sha256(session_token.encode()).hexdigest()
            
            async with self.pool.transaction() as conn:
                # Check existing session count
                session_count = await conn.fetchval("""
                    SELECT COUNT(*)
                    FROM user_sessions
                    WHERE user_id = $1
                      AND expires_at > NOW()
                      AND is_valid = TRUE
                """, user_id)
                
                # Remove oldest session if at limit
                if session_count >= self.config.MAX_SESSIONS_PER_USER:
                    await conn.execute("""
                        UPDATE user_sessions
                        SET is_valid = FALSE,
                            invalidated_at = NOW(),
                            invalidation_reason = 'max_sessions_exceeded'
                        WHERE id = (
                            SELECT id
                            FROM user_sessions
                            WHERE user_id = $1
                              AND is_valid = TRUE
                            ORDER BY created_at ASC
                            LIMIT 1
                        )
                    """, user_id)
                
                # Create new session
                await conn.execute("""
                    INSERT INTO user_sessions (
                        session_id,
                        session_token,
                        user_id,
                        ip_address,
                        user_agent,
                        created_at,
                        expires_at,
                        is_valid
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, TRUE)
                """,
                    session_id,
                    session_token,
                    user_id,
                    ip_address,
                    user_agent,
                    datetime.now(timezone.utc),
                    expires_at
                )
            
            # Cache in Redis if available
            if self.redis:
                await self._cache_session(session_token, user_id, expires_at)
            
            logger.info(f"Created session for user {user_id}")
            return session_token
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    async def validate_session(
        self,
        token: str,
        ip_address: str,
        user_agent: str
    ) -> Optional[SessionInfo]:
        """
        Validate session with security checks
        
        Security:
        - Token validation
        - Expiry check
        - IP address verification (warning only)
        - User agent verification (warning only)
        """
        try:
            # Check Redis cache first
            if self.redis:
                cached = await self._get_cached_session(token)
                if cached:
                    return SessionInfo(**cached)
            
            session_id = hashlib.sha256(token.encode()).hexdigest()
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT 
                        session_id,
                        user_id,
                        ip_address,
                        user_agent,
                        created_at,
                        expires_at
                    FROM user_sessions
                    WHERE session_id = $1
                      AND expires_at > NOW()
                      AND is_valid = TRUE
                """, session_id)
                
                if not row:
                    return None
                
                session = SessionInfo(**dict(row))
                
                # Security checks (warning only, don't block)
                if session.ip_address != ip_address:
                    logger.warning(
                        f"IP mismatch for session {session_id}: "
                        f"{session.ip_address} vs {ip_address}"
                    )
                
                if session.user_agent != user_agent:
                    logger.warning(
                        f"User agent mismatch for session {session_id}"
                    )
                
                return session
                
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    async def get_user_from_session(
        self,
        token: str
    ) -> Optional[UserInfo]:
        """Get user from session token"""
        try:
            session_id = hashlib.sha256(token.encode()).hexdigest()
            
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT 
                        u.id,
                        u.google_id,
                        u.email,
                        u.name,
                        u.permission_level,
                        u.is_free_trial,
                        u.created_at,
                        u.last_login
                    FROM users u
                    INNER JOIN user_sessions s ON u.id = s.user_id
                    WHERE s.session_id = $1
                      AND s.expires_at > NOW()
                      AND s.is_valid = TRUE
                """, session_id)
                
                if not row:
                    return None
                
                return UserInfo(**dict(row))
                
        except Exception as e:
            logger.error(f"Error getting user from session: {e}")
            return None
    
    async def invalidate_session(
        self,
        token: str,
        reason: str
    ) -> bool:
        """Invalidate a session"""
        try:
            session_id = hashlib.sha256(token.encode()).hexdigest()
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE user_sessions
                    SET is_valid = FALSE,
                        invalidated_at = NOW(),
                        invalidation_reason = $2
                    WHERE session_id = $1
                """, session_id, reason)
            
            # Remove from Redis
            if self.redis:
                await self.redis.delete(f"session:{session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating session: {e}")
            return False
    
    async def set_session_cookie(
        self,
        response: Response,
        session_token: str
    ) -> None:
        """Set session cookie"""
        response.set_cookie(
            key=self.config.SESSION_COOKIE_NAME,
            value=session_token,
            max_age=self.config.SESSION_EXPIRY_DAYS * 24 * 60 * 60,
            path="/",
            httponly=self.config.HTTPONLY_COOKIE,
            secure=self.config.SECURE_COOKIE,
            samesite=self.config.SAMESITE_POLICY
        )
    
    async def clear_session_cookie(self, response: Response) -> None:
        """Clear session cookie"""
        response.delete_cookie(
            key=self.config.SESSION_COOKIE_NAME,
            path="/",
            secure=self.config.SECURE_COOKIE,
            samesite=self.config.SAMESITE_POLICY
        )
    
    async def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all active sessions for a user"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        session_id,
                        ip_address,
                        user_agent,
                        created_at,
                        expires_at
                    FROM user_sessions
                    WHERE user_id = $1
                      AND expires_at > NOW()
                      AND is_valid = TRUE
                    ORDER BY created_at DESC
                """, user_id)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    async def revoke_user_session(
        self,
        user_id: int,
        session_id: str
    ) -> bool:
        """Revoke a specific user session"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE user_sessions
                    SET is_valid = FALSE,
                        invalidated_at = NOW(),
                        invalidation_reason = 'user_revoked'
                    WHERE session_id = $1
                      AND user_id = $2
                """, session_id, user_id)
                
                return result != "UPDATE 0"
                
        except Exception as e:
            logger.error(f"Error revoking session: {e}")
            return False
    
    async def count_active_sessions(self) -> int:
        """Count all active sessions"""
        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval("""
                    SELECT COUNT(*)
                    FROM user_sessions
                    WHERE expires_at > NOW()
                      AND is_valid = TRUE
                """)
                return count or 0
        except Exception as e:
            logger.error(f"Error counting sessions: {e}")
            return 0
    
    async def _cache_session(
        self,
        token: str,
        user_id: int,
        expires_at: datetime
    ) -> None:
        """Cache session in Redis"""
        if not self.redis:
            return
        
        try:
            session_id = hashlib.sha256(token.encode()).hexdigest()
            ttl = int((expires_at - datetime.now(timezone.utc)).total_seconds())
            
            await self.redis.setex(
                f"session:{session_id}",
                ttl,
                json.dumps({'user_id': user_id})
            )
        except Exception as e:
            logger.warning(f"Error caching session: {e}")
    
    async def _get_cached_session(self, token: str) -> Optional[Dict]:
        """Get session from Redis cache"""
        if not self.redis:
            return None
        
        try:
            session_id = hashlib.sha256(token.encode()).hexdigest()
            cached = await self.redis.get(f"session:{session_id}")
            
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error getting cached session: {e}")
        
        return None
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            return True
        except Exception as e:
            logger.error(f"Session service health check failed: {e}")
            return False
