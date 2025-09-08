import logging
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from utils.db import get_connection
from typing import Dict, Optional
import datetime
from fastapi import Cookie, Request
from pydantic import BaseModel
from config import get_settings

# Models for user data
class User(BaseModel):
    id: int
    google_id: str
    email: str
    name: Optional[str] = None
    permission_level: Optional[str] = "readwrite"
    is_free_trial: Optional[bool] = False

logger = logging.getLogger(__name__)

def get_app_credentials() -> Optional[Credentials]:
    """
    Retrieve app-level credentials from environment variables.
    
    Returns:
        Credentials object or None if not found
    """
    try:
        settings = get_settings()

        service_account_file = settings.client_secret_file
        return service_account.Credentials.from_service_account_file(service_account_file)
            
    except Exception as e:
        logger.error(f"Error retrieving app credentials: {e}")
        return None
    
def get_user_credentials(user_id: int, auto_refresh: bool = True) -> Optional[Credentials]:
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
        conn = get_connection()
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT token, refresh_token, token_uri, client_id, client_secret, scopes, token_expiry
                FROM users
                WHERE id = %s
            """, (user_id,))
            
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No credentials found for user {user_id}")
                return None
                
            token, refresh_token, token_uri, client_id, client_secret, scopes, token_expiry = result
            
            # Convert scopes to a Python list (handle both string and list formats)
            if scopes:
                if isinstance(scopes, list):
                    # Already a list, use as is
                    scopes_list = scopes
                elif isinstance(scopes, str):
                    # PostgreSQL array format string like "{scope1,scope2}"
                    scopes_list = scopes.replace("{", "").replace("}", "").split(",")
                else:
                    # Unknown format, log warning and use empty list
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
                
                # Check if expired
                if credentials.expired:
                    logger.info(f"Token for user {user_id} is expired, refreshing")
                    should_refresh = True
                # Check if about to expire
                elif expiry:
                    now = datetime.datetime.now(expiry.tzinfo) if expiry.tzinfo else datetime.datetime.now()
                    time_until_expiry = expiry - now
                    if time_until_expiry.total_seconds() < 300:  # Less than 5 minutes
                        logger.info(f"Token for user {user_id} expires in {time_until_expiry.total_seconds()}s, refreshing")
                        should_refresh = True
                
                # Refresh the token if needed
                if should_refresh:
                    try:
                        request = Request()
                        credentials.refresh(request)
                        
                        # Update the refreshed credentials in the database
                        update_user_credentials(user_id, credentials)
                        logger.info(f"Successfully refreshed token for user {user_id}")
                    except Exception as e:
                        logger.error(f"Error refreshing token for user {user_id}: {e}")
                
            return credentials
            
    except Exception as e:
        logger.error(f"Error retrieving user credentials: {e}")
        return None
    finally:
        conn.close()


def update_user_credentials(user_id: int, credentials: Credentials) -> bool:
    """
    Update stored credentials after token refresh.
    
    Args:
        user_id: Database user ID
        credentials: Updated credentials object
        
    Returns:
        bool: True if update was successful
    """
    try:
        conn = get_connection()
        
        # Convert scopes list to array format for PostgreSQL
        scopes_array = "{" + ",".join(credentials.scopes) + "}" if credentials.scopes else "{}"
        
        # Format token expiry to a timestamp
        token_expiry = credentials.expiry.isoformat() if hasattr(credentials, 'expiry') and credentials.expiry else None
        
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE users SET
                    token = %s,
                    refresh_token = %s,
                    token_uri = %s,
                    client_id = %s,
                    client_secret = %s,
                    scopes = %s,
                    token_expiry = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                credentials.token,
                credentials.refresh_token,
                credentials.token_uri,
                credentials.client_id,
                credentials.client_secret,
                scopes_array,
                token_expiry,
                user_id
            ))
            
            conn.commit()
            
            # Check if update was successful
            if cursor.rowcount > 0:
                logger.info(f"Updated credentials for user {user_id}")
                return True
            else:
                logger.warning(f"No user found with ID {user_id}, credentials not updated")
                return False
                
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating credentials for user {user_id}: {e}")
        return False
    finally:
        conn.close()


def get_credentials_dict(user_id: int) -> Optional[Dict]:
    """
    Retrieve user credentials from the database as a dictionary format
    
    Args:
        user_id: Database user ID
        
    Returns:
        dict: Credentials as a dictionary or None if not found
    """
    try:
        logger.info(f"Getting credentials data for user {user_id}")
        conn = get_connection()
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT token, refresh_token, token_uri, client_id, client_secret, scopes, token_expiry
                FROM users
                WHERE id = %s
            """, (user_id,))
            
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"No credentials found for user {user_id}")
                return None
                
            token, refresh_token, token_uri, client_id, client_secret, scopes, token_expiry = result
            
            # Convert scopes to a Python list (handle both string and list formats)
            if scopes:
                if isinstance(scopes, list):
                    # Already a list, use as is
                    scopes_list = scopes
                elif isinstance(scopes, str):
                    # PostgreSQL array format string like "{scope1,scope2}"
                    scopes_list = scopes.replace("{", "").replace("}", "").split(",")
                else:
                    # Unknown format, log warning and use empty list
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
            
            # Return as dictionary
            return {
                "token": token,
                "refresh_token": refresh_token,
                "token_uri": token_uri,
                "client_id": client_id,
                "client_secret": client_secret,
                "scopes": scopes_list,
                "token_expiry": expiry_str
            }
            
    except Exception as e:
        logger.error(f"Error retrieving user credentials dict: {e}")
        return None
    finally:
        conn.close()


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
    
    # Check token length (secrets.token_urlsafe(32) produces ~43 characters)
    if len(session_token) < 32:
        return False
    
    # Check for valid characters (URL-safe base64)
    import re
    if not re.match(r'^[A-Za-z0-9_-]+$', session_token):
        return False
    
    return True

def cleanup_invalid_sessions() -> int:
    """
    Clean up any invalid session tokens in the database
    
    Returns:
        int: Number of invalid sessions cleaned up
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Get all session tokens
            cursor.execute("SELECT id, session_token FROM users WHERE session_token IS NOT NULL")
            users = cursor.fetchall()
            
            invalid_count = 0
            for user_id, session_token in users:
                if not validate_session_token(session_token):
                    # Clear invalid session token
                    cursor.execute("""
                        UPDATE users 
                        SET session_token = NULL, session_expires = NULL
                        WHERE id = %s
                    """, (user_id,))
                    invalid_count += 1
            
            if invalid_count > 0:
                conn.commit()
                logger.warning(f"Cleaned up {invalid_count} invalid session tokens")
            else:
                logger.info("No invalid session tokens found")
            
            return invalid_count
                
    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")
        return 0
    finally:
        conn.close()

def get_user_from_session(session_token: Optional[str] = Cookie(None, alias='youtube_optimizer_session')) -> Optional[User]:
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
        
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, google_id, email, name, permission_level, is_free_trial, session_expires
                FROM users
                WHERE session_token = %s
            """, (session_token,))
            
            result = cursor.fetchone()
            if not result:
                logger.info(f"No user found with session token {session_token[:10] if session_token else None}...")
                return None
                
            # Check for session expiry
            id, google_id, email, name, permission_level, is_free_trial, session_expires = result
            
            if session_expires:
                now = datetime.datetime.now(session_expires.tzinfo) if hasattr(session_expires, 'tzinfo') else datetime.datetime.now()
                if now > session_expires:
                    logger.info(f"Session expired for user {id}")
                    return None
            
            return User(
                id=id,
                google_id=google_id,
                email=email,
                name=name,
                permission_level=permission_level or "readwrite",
                is_free_trial=is_free_trial or False
            )
    except Exception as e:
        logger.error(f"Error retrieving user from session: {e}")
        return None
    finally:
        conn.close()
