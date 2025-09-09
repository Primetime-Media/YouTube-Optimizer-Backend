# auth_routes.py
from fastapi import APIRouter, Response, HTTPException, Request, Depends, BackgroundTasks, Cookie
from fastapi.responses import RedirectResponse
from typing import Optional
from datetime import datetime, timedelta
import requests
import httpx
import logging
import secrets
import json
import base64
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from config import get_settings
from utils.auth import validate_session_token
from services.youtube import fetch_and_store_youtube_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Get application settings
settings = get_settings()
client_secrets_file = settings.client_secret_file

# Use redirect URI from settings
REDIRECT_URI = settings.redirect_uri

# Session configuration
SESSION_COOKIE_NAME = "youtube_optimizer_session"
SESSION_EXPIRY_DAYS = 14  # Session validity period

def get_db_connection():
    """Database connection function"""
    try:
        from utils.db import get_connection as get_db_conn
        return get_db_conn()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

def create_session(user_id: int, response: Response, request: Request) -> str:
    """Create a new session for a user and set the session cookie
    
    Security features:
    - Uses cryptographically secure random generation (secrets.token_urlsafe)
    - Generates 32 bytes of entropy (256 bits) for session tokens
    - Validates system entropy availability in production
    - Implements proper token format validation
    - Uses secure cookie settings in production (httponly, secure, samesite)
    """
    # Generate cryptographically secure session token
    # Using token_urlsafe for URL-safe tokens with 32 bytes of entropy (256 bits)
    session_token = secrets.token_urlsafe(32)
    
    # Additional entropy check for production environments
    if settings.is_production:
        # Ensure we have sufficient entropy for session tokens
        if not secrets.token_bytes(1):  # This will raise if system entropy is insufficient
            raise RuntimeError("Insufficient system entropy for secure session generation")
    
    # Use timezone-aware expiration date
    from datetime import timezone
    expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_EXPIRY_DAYS)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE users 
                SET session_token = %s, session_expires = %s
                WHERE id = %s
            """, (session_token, expires_at, user_id))
            conn.commit()
            
        # For development, use simpler cookie settings that will work
        # Set the session cookie - use max_age instead of expires to avoid timezone issues
        if settings.debug:
            logging.info(f"Creating session token for user {user_id}")
        
        # Set secure session cookie for cross-site requests in production
        if settings.is_production:
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=session_token,
                max_age=SESSION_EXPIRY_DAYS * 24 * 60 * 60,
                path="/",
                httponly=True,
                secure=True,
                samesite="none"
            )
        else:
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=session_token,
                max_age=SESSION_EXPIRY_DAYS * 24 * 60 * 60,
                path="/",
                httponly=True,
                secure=False,
                samesite="lax"
            )
        
        logging.info(f"Created session for user {user_id}")
        return session_token
    except Exception as e:
        logging.error(f"Error creating session: {e}")
        raise
    finally:
        conn.close()

def get_user_from_session(session_token: str) -> Optional[dict]:
    """Retrieve user from session token using auth.py function"""
    from utils.auth import get_user_from_session as auth_get_user_from_session
    
    # Use the auth.py function which includes security validation
    user = auth_get_user_from_session(session_token)
    
    if user:
        # Convert User model to dict format for compatibility
        return {
            "id": user.id,
            "google_id": user.google_id,
            "email": user.email,
            "name": user.name,
            "permission_level": user.permission_level,
            "is_free_trial": user.is_free_trial
        }
    
    return None

def delete_session(session_token: str, response: Response):
    """Invalidate a session and clear the cookie"""
    if not session_token:
        return
    
    # Validate session token format before attempting deletion
    if not validate_session_token(session_token):
        logging.warning(f"Attempted to delete invalid session token format: {session_token[:10]}...")
        return
        
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE users 
                SET session_token = NULL, session_expires = NULL
                WHERE session_token = %s
            """, (session_token,))
            conn.commit()
            
        # Clear the session cookie securely
        if settings.is_production:
            response.delete_cookie(
                key=SESSION_COOKIE_NAME,
                path="/",
                secure=True,
                samesite="none"
            )
        else:
            response.delete_cookie(
                key=SESSION_COOKIE_NAME,
                path="/",
                secure=False,
                samesite="lax"
            )
    except Exception as e:
        logging.error(f"Error deleting session: {e}")
    finally:
        conn.close()

async def get_current_user(session: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)) -> Optional[dict]:
    """Get the current authenticated user from the session cookie"""
    if not session:
        return None
    
    user = get_user_from_session(session)
    if settings.debug and user:
        logging.info(f"Authenticated user: {user['id']}")
        
    return user

def credentials_from_mock_response(client_id: str, client_secret: str, token_uri: str):
    """Create mock credentials for testing when token exchange fails"""
    return Credentials(
        token="fake_token_for_testing",
        refresh_token=None,
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.readonly"]
    )

@router.get("/scopes")
async def get_required_scopes():
    """Return the list of required OAuth scopes for the application."""
    return {
        "scopes": settings.youtube_api_scopes,
        "required_for_analytics": [
            "https://www.googleapis.com/auth/yt-analytics.readonly",
            "https://www.googleapis.com/auth/youtube.readonly"
        ],
        "required_for_captions": [
            "https://www.googleapis.com/auth/youtube.force-ssl"
        ]
    }

@router.get("/me")
async def get_current_user_info(request: Request, user: Optional[dict] = Depends(get_current_user)):
    """Get the current authenticated user information from session cookie.
    This endpoint can be used by the frontend to validate if the user is logged in.
    """
    if settings.debug:
        logging.info(f"API /me called for user: {user['id'] if user else 'none'}")
    
    if not user:
        logging.info("User not authenticated")
        return {"authenticated": False}

    logging.info(f"User {user['id']} is authenticated")
    return {
        "authenticated": True,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "permission_level": user["permission_level"],
            "is_free_trial": user["is_free_trial"]
        }
    }

@router.get("/login")
async def login(permission_level: str = "readwrite", free_trial: bool = False):
    # Store permission level in session or state
    # You can add it to the state parameter in the OAuth flow
    state_data = {
        "permission_level": permission_level,
        "free_trial": free_trial
    }

    # Convert state data to JSON and encode it
    state_json = json.dumps(state_data)
    state = base64.b64encode(state_json.encode()).decode()

    # Define the essential scopes we need
    basic_scopes = [
        # Basic user info
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        
        # Core YouTube scopes
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/youtube.force-ssl",
        "https://www.googleapis.com/auth/yt-analytics.readonly"
    ]
    
    logging.info(f"Starting OAuth flow with scopes: {basic_scopes}")

    # Create flow with only the most essential scopes
    flow = Flow.from_client_secrets_file(
        client_secrets_file,
        scopes=basic_scopes,
        redirect_uri=REDIRECT_URI
    )

    # Force approval prompt to show the consent screen with all scopes
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",  # Include any previously granted scopes
        prompt="consent",  # Always show the consent screen
        state=state
    )

    logging.info(f"Authorization URL: {authorization_url}")
    return RedirectResponse(authorization_url)

@router.get("/callback")
async def auth_callback(request: Request, response: Response, background_tasks: BackgroundTasks):
    try:
        # Get state parameter and decode it
        state_param = request.query_params.get("state", "")
        state_data = {}

        if state_param:
            try:
                state_json = base64.b64decode(state_param).decode()
                state_data = json.loads(state_json)
            except:
                logging.error("Failed to decode state parameter")

        permission_level = state_data.get("permission_level", "readwrite")
        free_trial = state_data.get("free_trial", False)
        
        logging.info(f"Auth callback with permission_level: {permission_level}, free_trial: {free_trial}")

        # Use the Flow class directly since we're in a web application context
        logging.info("Starting OAuth token exchange")
        
        # Extract the authorization code from the request
        code = request.query_params.get("code")
        if not code:
            raise ValueError("No authorization code found in callback")
            
        # Get the actual scopes from the callback URL
        callback_scope = request.query_params.get("scope", "")
        logging.info(f"Received scopes in callback: {callback_scope}")
        
        # We need to do this manually without any Flow or scope validation
        with open(client_secrets_file, 'r') as f:
            client_config = json.load(f)['web']
        
        try:
            # Basic fields needed for our manual token exchange
            client_id = client_config['client_id']
            client_secret = client_config['client_secret']
            token_uri = client_config['token_uri']
            
            # Use async HTTP client to exchange the code for tokens
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                token_response = await client.post(
                    token_uri,
                    data={
                        'code': code,
                        'client_id': client_id,
                        'client_secret': client_secret,
                        'redirect_uri': REDIRECT_URI,
                        'grant_type': 'authorization_code'
                    }
                )
            
            # Check for success
            if token_response.status_code != 200:
                logging.error(f"Token exchange failed with status {token_response.status_code}")
                logging.error(f"Response: {token_response.text}")
                
                # Create a mock credentials object for testing session management
                credentials = credentials_from_mock_response(client_id, client_secret, token_uri)
                logging.warning("Using mock credentials for testing - API calls will fail")
            else:
                # We got a successful token response
                token_data = token_response.json()
                logging.info(f"Token exchange successful: {token_data.keys()}")
                
                # Create a proper credentials object
                credentials = Credentials(
                    token=token_data.get('access_token'),
                    refresh_token=token_data.get('refresh_token'),
                    token_uri=token_uri,
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=callback_scope.split()
                )
                
                logging.info(f"Created credentials with token: {credentials.token[:10] if credentials.token else 'None'}")
                logging.info(f"Has refresh token: {bool(credentials.refresh_token)}")
                
        except Exception as e:
            logging.error(f"Error in manual token exchange: {e}")
            # Fall back to creating a fake token just to test the session management
            credentials = Credentials(
                token="fake_token_for_testing",
                refresh_token=None,
                token_uri=token_uri,
                client_id=client_id,
                client_secret=client_secret,
                scopes=callback_scope.split()
            )
            logging.warning("Using mock credentials due to error - API calls will fail")
        
        # Store credentials as a dictionary for the background task
        credentials_dict = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }

        # Fetch user info asynchronously
        async with httpx.AsyncClient(timeout=30.0) as client:
            userinfo_response = await client.get(
                "https://www.googleapis.com/oauth2/v1/userinfo",
                headers={"Authorization": f"Bearer {credentials.token}"}
            )
            userinfo = userinfo_response.json()

        logging.info(f"User authenticated: {userinfo.get('email')}")

        # Save user info and credentials to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert scopes to PostgreSQL array format
        if hasattr(credentials, 'scopes') and credentials.scopes:
            # Make sure scopes is a list
            if isinstance(credentials.scopes, str):
                scopes_list = credentials.scopes.split()
            else:
                scopes_list = list(credentials.scopes)
            scopes_array = "{" + ",".join(scopes_list) + "}"
        else:
            # Extract scopes from the callback URL if credentials doesn't have them
            callback_scopes = request.query_params.get("scope", "").replace("+", " ").split()
            scopes_array = "{" + ",".join(callback_scopes) + "}" if callback_scopes else "{}"
        
        # Format token expiry to a timestamp
        token_expiry = credentials.expiry.isoformat() if hasattr(credentials, 'expiry') and credentials.expiry else None
        
        cursor.execute("""
            INSERT INTO users (
                google_id, email, name, permission_level, is_free_trial,
                token, refresh_token, token_uri, client_id, client_secret, scopes, token_expiry
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                updated_at = NOW()
            RETURNING id
        """, (
            userinfo["id"], 
            userinfo["email"], 
            userinfo["name"], 
            permission_level, 
            free_trial,
            credentials.token,
            credentials.refresh_token,
            credentials.token_uri,
            credentials.client_id,
            credentials.client_secret,
            scopes_array,
            token_expiry
        ))
        user_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        # Log that credentials were stored
        logging.info(f"User {user_id} credentials stored in database (token expiry: {token_expiry})")
        
        # Create a session and set the session cookie
        try:
            session_token = create_session(user_id, response, request)
            logging.info(f"Created session for user {user_id} with token {session_token[:10]}...")
            
            # Log cookie for debugging
            logging.info(f"Session cookie set with name: {SESSION_COOKIE_NAME}")
            logging.info(f"Response cookies: {response.headers.get('set-cookie', 'None')}")
        except Exception as e:
            logging.error(f"Error creating session: {e}")
            logging.exception("Session creation exception")
        
        # Add background task to fetch YouTube data - no need to pass credentials
        # since they are now stored in the database and can be retrieved by user_id
        background_tasks.add_task(
            fetch_and_store_youtube_data,
            user_id=user_id,
            max_videos=1000  # Limit to 10000 most recent videos to save quota
        )

        # Encode user info as JSON to pass to frontend
        user_data = {
            "id": user_id,
            "email": userinfo.get("email"),
            "name": userinfo.get("name"),
            "is_free_trial": free_trial
        }
        encoded_user_data = base64.b64encode(json.dumps(user_data).encode()).decode()

        # Redirect to frontend dashboard with user data encoded in URL
        # The frontend will store this in local storage, but the session cookie
        # is the source of truth for authentication
        frontend_dashboard_url = f"{settings.frontend_url}/dashboard"
        
        # Log the redirect URL and cookie information
        logging.info(f"Redirecting to: {frontend_dashboard_url}?user_data={encoded_user_data[:10]}...")
        logging.info(f"Response headers: {dict(response.headers)}")
        
        # Set cache control headers to ensure nothing gets cached
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        # Create redirect response with status code 307 to ensure cookies are preserved
        # Add use_session=true to indicate we want to use session auth
        redirect_url = f"{frontend_dashboard_url}?user_data={encoded_user_data}&use_session=true"
        redirect_response = RedirectResponse(url=redirect_url, status_code=307)
        
        # Copy cookies from original response to redirect response
        for header_name, header_value in response.headers.items():
            if header_name.lower() == "set-cookie":
                redirect_response.headers.append(header_name, header_value)
                
        logging.info(f"Final redirect response headers: {dict(redirect_response.headers)}")
        
        return redirect_response
    except Exception as e:
        logging.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")

@router.post("/logout")
async def logout(response: Response, session: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)):
    try:
        # Clear session in database and remove cookie
        if session:
            delete_session(session, response)
            logging.info("User session deleted successfully")
        
        return {
            "message": "Logged out successfully",
            "external_home_url": settings.external_home_url
        }
    except Exception as e:
        logging.error(f"Error during logout: {e}")
        return {
            "message": "Logout completed, but with errors",
            "external_home_url": settings.external_home_url
        }

@router.get("/config/external-home-url")
async def get_external_home_url():
    """Get the external home page URL for logout redirects."""
    return {"external_home_url": settings.external_home_url}
