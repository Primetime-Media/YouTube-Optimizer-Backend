"""
Authentication Utilities - COMPLETE REWRITE
===========================================
ALL 8 CRITICAL ERRORS FIXED

Major Changes:
✅ #1: Removed get_connection() - now uses get_pool() from asyncpg
✅ #2: Fixed dual Request import conflict
✅ #3: Removed non-existent client_secret_file reference
✅ #4-8: All functions converted to async with asyncpg
✅ All database operations now use asyncpg API
✅ All functions properly use async/await patterns
✅ Proper connection pool usage throughout
"""

import logging
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleRequest  # ✅ FIXED: Renamed import
from utils.db import get_pool  # ✅ FIXED: Use get_pool instead of get_connection
from typing import Dict, Optional
import datetime
from fastapi import Cookie
from pydantic import BaseModel
from config import get_settings
import re

# Models for user data
class User(BaseModel):
    id: int
    google_id: str
    email: str
    name: Optional[str] = None
    permission_level: Optional[str] = "readwrite"
    is_free_trial: Optional[bool] = False

logger = logging.getLogger(__name__)


# ============================================================================
# APP CREDENTIALS
# ============================================================================

async def get_app_credentials() -> Optional[Credentials]:  # ✅ FIXED: Now async
    """
    Retrieve app-level credentials from environment variables.
    
    Returns:
        Credentials object or None if not found
    """
    try:
        settings = get_settings()
        
        # ✅ FIXED: Use existing config field instead of non-existent one
        service_account_file = settings.GOOGLE_VISION_CREDENTIALS
        
        if not service_account_file:
            logger.warning("No service account file configured")
            return None
            
        return service_account.Credentials.from_service_account_file(service_account_file)
            
    except Exception as e:
        logger.error(f"Error retrieving app credentials: {e}")
        return None


# ============================================================================
# USER CREDENTIALS
# ============================================================================

async def get_user_credentials(user_id: int, auto_refresh: bool = True) -> Optional[Credentials]:
    """
    Retrieve user credentials from the database and convert to a Google OAuth2 Credentials object.
    Optionally refreshes the token if it's expired or about to expire.
    
    Args:
        user_id: Database user ID
        auto_refresh: Whether to automatically refresh the token if needed
        
    Returns:
        Credentials object or None if not found
    """
    try:
        # ✅ FIXED: Use asyncpg connection pool
        async with get_pool().acquire() as conn:
            # ✅ FIXED: Use asyncpg API
            row = await conn.fetchrow("""
                SELECT token, refresh_token, token_uri, client_id, client_secret, scopes, token_expiry
                FROM users
                WHERE id = $1
            """, user_id)
            
            if not row:
                logger.warning(f"No credentials found for user {user_id}")
                return None
            
            # ✅ FIXED: asyncpg returns dict-like records
            token = row['token']
            refresh_token = row['refresh_token']
            token_uri = row['token_uri']
            client_id = row['client_id']
            client_secret = row['client_secret']
            scopes = row['scopes']
            token_expiry = row['token_expiry']
            
            # Convert scopes to a Python list (asyncpg handles arrays natively)
            if scopes:
                if isinstance(scopes, list):
                    scopes_list = scopes
                elif isinstance(scopes, str):
                    scopes_list = scopes.replace("{", "").replace("}", "").split(",")
                else:
                    logger.warning(f"Unknown scope format for user {user_id}: {type(scopes)}")
                    scopes_list = []
            else:
                scopes_list = []
            
            # Parse token expiry if available
            expiry = None
            if token_expiry:
                if isinstance(token_expiry, str):
                    expiry = datetime.datetime.fromisoformat(token_expiry)
                else:
                    expiry = token_expiry
            
            # Create credentials object
            credentials = Credentials(
                token=token,
                refresh_token=refresh_token,
                token_uri=token_uri,
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes_list,
                expiry=expiry
            )
            
            # Check if token is expired or about to expire (within 5 minutes)
            if auto_refresh and credentials.refresh_token:
                should_refresh = False
                
                if credentials.expired:
                    logger.info(f"Token for user {user_id} is expired, refreshing")
                    should_refresh = True
                elif expiry:
                    now = datetime.datetime.now(expiry.tzinfo) if expiry.tzinfo else datetime.datetime.now()
                    time_until_expiry = expiry - now
                    if time_until_expiry.total_seconds() < 300:
                        logger.info(f"Token for user {user_id} expires in {time_until_expiry.total_seconds()}s, refreshing")
                        should_refresh = True
                
                if should_refresh:
                    try:
                        # ✅ FIXED: Use GoogleRequest (renamed import)
                        request = GoogleRequest()
                        credentials.refresh(request)
                        
                        # Update the refreshed credentials in the database
                        await update_user_credentials(user_id, credentials)
                        logger.info(f"Successfully refreshed token for user {user_id}")
                    except Exception as e:
                        logger.error(f"Error refreshing token for user {user_id}: {e}")
            
            return credentials
            
    except Exception as e:
        logger.error(f"Error retrieving user credentials: {e}")
        return None


async def update_user_credentials(user_id: int, credentials: Credentials) -> bool:
    """
    Update stored credentials after token refresh.
    
    Args:
        user_id: Database user ID
        credentials: Updated credentials object
        
    Returns:
        bool: True if update was successful
    """
    try:
        # ✅ FIXED: asyncpg handles arrays natively
        scopes_list = credentials.scopes if credentials.scopes else []
        
        # Format token expiry to a timestamp
        token_expiry = credentials.expiry.isoformat() if hasattr(credentials, 'expiry') and credentials.expiry else None
        
        # ✅ FIXED: Use asyncpg connection pool and transaction
        async with get_pool().acquire() as conn:
            async with conn.transaction():  # ✅ FIXED: asyncpg transaction API
                result = await conn.execute("""
                    UPDATE users SET
                        token = $1,
                        refresh_token = $2,
                        token_uri = $3,
                        client_id = $4,
                        client_secret = $5,
                        scopes = $6,
                        token_expiry = $7,
                        updated_at = NOW()
                    WHERE id = $8
                """, 
                    credentials.token,
                    credentials.refresh_token,
                    credentials.token_uri,
                    credentials.client_id,
                    credentials.client_secret,
                    scopes_list,  # ✅ FIXED: asyncpg handles list natively
                    token_expiry,
                    user_id
                )
                
                # ✅ FIXED: Check result format for asyncpg
                if result == "UPDATE 1":
                    logger.info(f"Updated credentials for user {user_id}")
                    return True
                else:
                    logger.warning(f"No user found with ID {user_id}, credentials not updated")
                    return False
                
    except Exception as e:
        logger.error(f"Error updating credentials for user {user_id}: {e}")
        return False


async def get_credentials_dict(user_id: int) -> Optional[Dict]:
    """
    Retrieve user credentials from the database as a dictionary format
    
    Args:
        user_id: Database user ID
        
    Returns:
        dict: Credentials as a dictionary or None if not found
    """
    try:
        logger.info(f"Getting credentials data for user {user_id}")
        
        # ✅ FIXED: Use asyncpg connection pool
        async with get_pool().acquire() as conn:
            row = await conn.fetchrow("""
                SELECT token, refresh_token, token_uri, client_id, client_secret, scopes, token_expiry
                FROM users
                WHERE id = $1
            """, user_id)
            
            if not row:
                logger.warning(f"No credentials found for user {user_id}")
                return None
            
            # ✅ FIXED: asyncpg returns dict-like records
            scopes = row['scopes']
            token_expiry = row['token_expiry']
            
            # Convert scopes to a Python list
            if scopes:
                if isinstance(scopes, list):
                    scopes_list = scopes
                elif isinstance(scopes, str):
                    scopes_list = scopes.replace("{", "").replace("}", "").split(",")
                else:
                    logger.warning(f"Unknown scope format for user {user_id}: {type(scopes)}")
                    scopes_list = []
            else:
                scopes_list = []
            
            # Convert token_expiry to string if needed
            expiry_str = None
            if token_expiry:
                if isinstance(token_expiry, datetime.datetime):
                    expiry_str = token_expiry.isoformat()
                else:
                    expiry_str = str(token_expiry)
            
            return {
                "token": row['token'],
                "refresh_token": row['refresh_token'],
                "token_uri": row['token_uri'],
                "client_id": row['client_id'],
                "client_secret": row['client_secret'],
                "scopes": scopes_list,
                "token_expiry": expiry_str
            }
            
    except Exception as e:
        logger.error(f"Error retrieving user credentials dict: {e}")
        return None


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def validate_session_token(session_token: str) -> bool:
    """
    Validate session token format and security requirements
    
    Security checks:
    - Ensures token is not empty
    - Validates minimum length (32+ characters)
    - Checks for valid URL-safe base64 characters only
    - Prevents injection attacks through malformed tokens
    """
    if not session_token:
        return False
    
    # Check token length
    if len(session_token) < 32:
        return False
    
    # Check for valid characters (URL-safe base64)
    if not re.match(r'^[A-Za-z0-9_-]+$', session_token):
        return False
    
    return True


async def cleanup_invalid_sessions() -> int:
    """
    Clean up any invalid session tokens in the database
    
    Returns:
        int: Number of invalid sessions cleaned up
    """
    try:
        # ✅ FIXED: Use asyncpg connection pool
        async with get_pool().acquire() as conn:
            # Get all session tokens
            rows = await conn.fetch(
                "SELECT id, session_token FROM users WHERE session_token IS NOT NULL"
            )
            
            invalid_count = 0
            
            # ✅ FIXED: Use asyncpg transaction
            async with conn.transaction():
                for row in rows:
                    user_id = row['id']
                    session_token = row['session_token']
                    
                    if not validate_session_token(session_token):
                        # Clear invalid session token
                        await conn.execute("""
                            UPDATE users 
                            SET session_token = NULL, session_expires = NULL
                            WHERE id = $1
                        """, user_id)
                        invalid_count += 1
            
            if invalid_count > 0:
                logger.warning(f"Cleaned up {invalid_count} invalid session tokens")
            else:
                logger.info("No invalid session tokens found")
            
            return invalid_count
                
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")
        return 0


async def get_user_from_session(
    session_token: Optional[str] = Cookie(None, alias='youtube_optimizer_session')
) -> Optional[User]:
    """
    Retrieve user from session token with security validation
    
    Security features:
    - Validates session token format before database lookup
    - Checks session expiry
    - Uses secure token validation
    - Prevents injection attacks
    
    Args:
        session_token: The session token cookie value
        
    Returns:
        User object or None if session is invalid
    """
    if not session_token:
        return None
    
    # Validate session token format before database lookup
    if not validate_session_token(session_token):
        logger.warning(f"Invalid session token format: {session_token[:10] if session_token else None}...")
        return None
    
    try:
        # ✅ FIXED: Use asyncpg connection pool
        async with get_pool().acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, google_id, email, name, permission_level, is_free_trial, session_expires
                FROM users
                WHERE session_token = $1
            """, session_token)
            
            if not row:
                logger.info(f"No user found with session token {session_token[:10] if session_token else None}...")
                return None
            
            # Check for session expiry
            session_expires = row['session_expires']
            
            if session_expires:
                now = datetime.datetime.now(session_expires.tzinfo) if hasattr(session_expires, 'tzinfo') else datetime.datetime.now()
                if now > session_expires:
                    logger.info(f"Session expired for user {row['id']}")
                    return None
            
            return User(
                id=row['id'],
                google_id=row['google_id'],
                email=row['email'],
                name=row['name'],
                permission_level=row['permission_level'] or "readwrite",
                is_free_trial=row['is_free_trial'] or False
            )
            
    except Exception as e:
        logger.error(f"Error retrieving user from session: {e}")
        return None
