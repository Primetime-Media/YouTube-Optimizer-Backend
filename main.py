from fastapi import FastAPI, Response, HTTPException, Request, Depends, BackgroundTasks, Cookie
from fastapi.security import HTTPBearer
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import os
import uvicorn
import requests
import logging
import secrets
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from utils.db import get_connection, init_db
from services.youtube import fetch_and_store_youtube_data, fetch_video_transcript
from services.llm_optimization import get_comprehensive_optimization
from routes.analytics import router as analytics_router
from routes.channel_routes import router as channel_router
from routes.scheduler_routes import router as scheduler_router
from routes.video_routes import router as video_router
from config import get_settings
from utils.auth import get_user_credentials
from services.scheduler import initialize_scheduler

# Get application settings
settings = get_settings()
client_secrets_file = settings.client_secret_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress googleapiclient.discovery_cache warnings
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

load_dotenv()

# Use redirect URI from settings
REDIRECT_URI = settings.redirect_uri

# Only allow insecure transport in development
if not settings.is_production:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Session configuration
SESSION_COOKIE_NAME = "youtube_optimizer_session"
SESSION_EXPIRY_DAYS = 14  # Session validity period
SESSION_SECRET = os.getenv("SESSION_SECRET", secrets.token_hex(32))

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None
)

# Configure CORS
allowed_origins = [settings.frontend_url]
if not settings.is_production:
    allowed_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])

logging.info(f"Allowed origins: {allowed_origins}")
logging.info(f"Environment: {settings.environment}")
logging.info(f"Frontend URL: {settings.frontend_url}")
logging.info(f"Backend URL: {settings.backend_url}")
logging.info(f"Is production: {settings.is_production}")
logging.info(f"Redirect URI: {REDIRECT_URI}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["content-type"]
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Include routers
app.include_router(analytics_router) 
app.include_router(channel_router)
app.include_router(scheduler_router)
app.include_router(video_router)

@app.on_event("startup")
async def startup_db_client():
    """Initialize the database and scheduler system on startup."""
    init_db()
    initialize_scheduler()  # Just initializes scheduler resources, doesn't run any jobs
    logging.info("Database and scheduler system initialized with session management support")

# Database connection function
def get_db_connection():
    try:
        return get_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

# Session management functions
def create_session(user_id: int, response: Response, request: Request) -> str:
    """Create a new session for a user and set the session cookie"""
    session_token = secrets.token_urlsafe(32)
    
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
        
        # Set secure session cookie
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_token,
            max_age=SESSION_EXPIRY_DAYS * 24 * 60 * 60,
            path="/",
            httponly=True,
            secure=settings.is_production,
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
    """Retrieve user from session token"""
    if not session_token:
        return None
        
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Check if this token exists
            if settings.debug:
                cursor.execute("SELECT COUNT(*) FROM users WHERE session_token = %s", (session_token,))
                count = cursor.fetchone()[0]
                logging.info(f"Found {count} users with matching session token")
            
            cursor.execute("""
                SELECT id, google_id, email, name, permission_level, is_free_trial, session_expires
                FROM users
                WHERE session_token = %s
            """, (session_token,))
            
            result = cursor.fetchone()
            if not result:
                logging.info(f"No user found with session token {session_token[:10]}...")
                return None
                
            # Check if session has expired
            session_expires = result[6]
            if session_expires:
                # Handle timezone-aware comparison
                current_time = datetime.now()
                if session_expires.tzinfo is not None:
                    # session_expires is timezone-aware, make current_time timezone-aware
                    from datetime import timezone
                    current_time = datetime.now(timezone.utc)
                    if session_expires.tzinfo != timezone.utc:
                        session_expires = session_expires.astimezone(timezone.utc)
                else:
                    # session_expires is naive, use naive current_time
                    current_time = datetime.now()
                
                if current_time > session_expires:
                    logging.info(f"Session token expired at {session_expires}")
                    return None
            
            logging.info(f"Found valid session for user ID {result[0]}")
            return {
                "id": result[0],
                "google_id": result[1],
                "email": result[2],
                "name": result[3],
                "permission_level": result[4],
                "is_free_trial": result[5]
            }
    except Exception as e:
        logging.error(f"Error retrieving user from session: {e}")
        logging.exception("Session retrieval exception")
        return None
    finally:
        conn.close()

def delete_session(session_token: str, response: Response):
    """Invalidate a session and clear the cookie"""
    if not session_token:
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
        response.delete_cookie(
            key=SESSION_COOKIE_NAME,
            path="/",
            secure=not settings.debug,
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

"""
# OAuth flow variable will be created dynamically in the login function
"""

@app.get("/")
async def root():
    return {"message": "YouTube Optimizer API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Quick database connectivity check
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database connection failed")


@app.get("/api/scopes")
async def get_required_scopes():
    """Return the list of required OAuth scopes for the application."""
    settings = get_settings()
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
    
@app.get("/api/me")
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

@app.post("/videos/{video_id}/fetch-transcript")
async def fetch_and_store_transcript(video_id: str, user_id: int):
    """Fetch and store transcript for a specific video."""
    try:
        # First check if we already have the transcript
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT transcript, has_captions
                    FROM youtube_videos
                    WHERE video_id = %s
                """, (video_id,))
                
                result = cursor.fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail="Video not found")
                    
                existing_transcript, has_captions = result
                
                # If transcript already exists, return it
                if existing_transcript:
                    return {
                        "video_id": video_id,
                        "has_transcript": True,
                        "transcript_length": len(existing_transcript),
                        "has_captions": has_captions,
                        "message": "Transcript already exists for this video"
                    }
        finally:
            conn.close()
        
        credentials = get_user_credentials(user_id)
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="No valid credentials found. Please log in again."
            )
            
        # Fetch transcript
        transcript_data = fetch_video_transcript(credentials, video_id)
        transcript = transcript_data.get("transcript")
        has_captions = transcript_data.get("has_captions", False)
        
        if not transcript:
            return {
                "video_id": video_id,
                "has_transcript": False,
                "error": transcript_data.get("error", "No transcript available for this video"),
                "has_captions": has_captions
            }
            
        # Store in database
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE youtube_videos
                    SET transcript = %s, has_captions = %s
                    WHERE video_id = %s
                """, (transcript, has_captions, video_id))
                conn.commit()
        finally:
            conn.close()
            
        return {
            "video_id": video_id,
            "has_transcript": True,
            "transcript_length": len(transcript),
            "has_captions": has_captions,
            "message": "Successfully fetched and stored transcript"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching transcript: {str(e)}")



@app.get("/login")
async def login(permission_level: str = "readwrite", free_trial: bool = False):
    # Store permission level in session or state
    # You can add it to the state parameter in the OAuth flow
    state_data = {
        "permission_level": permission_level,
        "free_trial": free_trial
    }

    # Convert state data to JSON and encode it
    import json
    import base64
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


@app.get("/auth/callback")
async def auth_callback(request: Request, response: Response, background_tasks: BackgroundTasks):
    try:
        # Get state parameter and decode it
        import json
        import base64
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
        import json
        with open(client_secrets_file, 'r') as f:
            client_config = json.load(f)['web']
        
        try:
            # Basic fields needed for our manual token exchange
            client_id = client_config['client_id']
            client_secret = client_config['client_secret']
            token_uri = client_config['token_uri']
            
            # Use requests to directly exchange the code for tokens
            # without any scope validation
            token_response = requests.post(
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

        # Fetch user info
        userinfo = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"}
        ).json()

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
            max_videos=10  # Limit to 10 most recent videos to save quota
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

# Authentication dependency
async def verify_user_auth(
    user_id: int,
    current_user: Optional[dict] = Depends(get_current_user)
) -> dict:
    """Verify that the user is authenticated and has access to the requested user_id resources."""
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    
    # If the requested user_id doesn't match the authenticated user,
    # they don't have permission to access the data
    if current_user["id"] != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    return current_user

@app.get("/youtube-data/{user_id}")
async def get_youtube_data(user_id: int, _: dict = Depends(verify_user_auth)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all channel data for the user
        cursor.execute("""
            SELECT c.id, c.channel_id, c.kind, c.etag, c.title, c.description, 
                   c.custom_url, c.published_at, c.view_count, c.subscriber_count,
                   c.hidden_subscriber_count, c.video_count, c.thumbnail_url_default,
                   c.thumbnail_url_medium, c.thumbnail_url_high, c.uploads_playlist_id,
                   c.banner_url, c.privacy_status, c.is_linked, c.long_uploads_status,
       c.is_monetization_enabled, c.topic_ids, c.topic_categories,
       c.overall_good_standing, c.community_guidelines_good_standing,
       c.copyright_strikes_good_standing, c.content_id_claims_good_standing,
       c.branding_settings, c.audit_details, c.topic_details, c.status_details,
       c.created_at, c.updated_at
            FROM youtube_channels c
            WHERE c.user_id = %s
            ORDER BY c.title
        """, (user_id,))
            
        channels = []
        for channel_data in cursor.fetchall():
            channel = {
                "id": channel_data[0],
                "channel_id": channel_data[1],
                "kind": channel_data[2],
                "etag": channel_data[3],
                "title": channel_data[4],
                "description": channel_data[5],
                "custom_url": channel_data[6],
                "published_at": channel_data[7],
                "view_count": channel_data[8],
                "subscriber_count": channel_data[9],
                "hidden_subscriber_count": channel_data[10],
                "video_count": channel_data[11],
                "thumbnails": {
                    "default": channel_data[12],
                    "medium": channel_data[13],
                    "high": channel_data[14]
                },
                "uploads_playlist_id": channel_data[15],
                "optimization": {
                    "banner_url": channel_data[16],
                    "privacy_status": channel_data[17],
                    "is_linked": channel_data[18],
                    "long_uploads_status": channel_data[19],
                    "is_monetization_enabled": channel_data[20],
                    "topic_ids": channel_data[21],
                    "topic_categories": channel_data[22],
                    "channel_standing": {
                        "overall_good_standing": channel_data[23],
                        "community_guidelines_good_standing": channel_data[24],
                        "copyright_strikes_good_standing": channel_data[25],
                        "content_id_claims_good_standing": channel_data[26]
                    }
                },
                "raw_data": {
                    "branding_settings": channel_data[27],
                    "audit_details": channel_data[28],
                    "topic_details": channel_data[29],
                    "status_details": channel_data[30]
                },
                "created_at": channel_data[31],
                "updated_at": channel_data[32]
            }
            
            # Get videos data - join with youtube_channels to filter by user_id
            cursor.execute("""
                SELECT v.id, v.video_id, v.kind, v.etag, v.playlist_item_id, v.title, 
                       v.description, v.published_at, v.channel_title, v.tags, v.playlist_id,
                       v.position, v.thumbnail_url_default, v.thumbnail_url_medium, 
                       v.thumbnail_url_high, v.thumbnail_url_standard, v.thumbnail_url_maxres,
                       v.view_count, v.like_count, v.comment_count, v.duration, 
                       v.is_optimized, v.created_at, v.updated_at, v.queued_for_optimization, v.optimizations_completed
                FROM youtube_videos v
                JOIN youtube_channels c ON v.channel_id = c.id
                WHERE c.user_id = %s
                ORDER BY v.published_at DESC
            """, (user_id,))
            
            videos_data = cursor.fetchall()
            
            videos = []
            for video in videos_data:
                videos.append({
                    "id": video[0],
                    "video_id": video[1],
                    "kind": video[2],
                    "etag": video[3],
                    "playlist_item_id": video[4],
                    "title": video[5],
                    "description": video[6],
                    "published_at": video[7],
                    "channel_title": video[8],
                    "tags": video[9],
                    "playlist_id": video[10],
                    "position": video[11],
                    "thumbnails": {
                        "default": video[12],
                        "medium": video[13],
                        "high": video[14],
                        "standard": video[15],
                        "maxres": video[16]
                    },
                    "view_count": video[17],
                    "like_count": video[18],
                    "comment_count": video[19],
                    "duration": video[20],
                    "is_optimized": video[21],
                    "created_at": video[22],
                    "updated_at": video[23],
                    "queued_for_optimization": video[24],
                    "optimizations_completed": video[25]
                })
        
        conn.close()
        
        return {
            "status": "success",
            "data": {
                "channel": channel,
                "videos": videos,
                "video_count": len(videos)
            }
        }
        
    except Exception as e:
        logging.error(f"Error fetching YouTube data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching YouTube data: {str(e)}")

@app.post("/refresh-youtube-data/{user_id}")
async def refresh_youtube_data(user_id: int, background_tasks: BackgroundTasks):
    try:
        # Check if user exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        # We need credentials for this user
        # In a real app, you'd store refresh tokens securely and use them here
        # For now, we'll return a message that they need to log in again
        
        return {
            "status": "error", 
            "message": "For security reasons, please log in again to refresh your YouTube data"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error refreshing YouTube data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refreshing YouTube data: {str(e)}")

@app.post("/logout")
async def logout(response: Response, session: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)):
    try:
        # Clear session in database and remove cookie
        if session:
            delete_session(session, response)
            logging.info("User session deleted successfully")
        
        return {"message": "Logged out successfully"}
    except Exception as e:
        logging.error(f"Error during logout: {e}")
        return {"message": "Logout completed, but with errors"}

class VideoOptimizationStatus(BaseModel):
    is_optimized: bool

@app.put("/video/{video_id}/optimization-status")
async def update_video_optimization_status(video_id: str, status: VideoOptimizationStatus):
    """Update the optimization status of a video."""
    try:
        conn = get_db_connection()
        
        with conn.cursor() as cursor:
            # Update the is_optimized field for the video
            cursor.execute("""
                UPDATE youtube_videos
                SET is_optimized = %s, updated_at = NOW()
                WHERE video_id = %s
                RETURNING id
            """, (status.is_optimized, video_id))
            
            result = cursor.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Video not found")
            
            conn.commit()
        
        conn.close()
        
        return {
            "status": "success",
            "message": f"Video optimization status updated to {status.is_optimized}",
            "video_id": video_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating video optimization status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating video status: {str(e)}")

@app.get("/optimized-videos/{user_id}")
async def get_optimized_videos(user_id: int):
    """Get all optimized videos for a user."""
    try:
        conn = get_db_connection()
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT v.id, v.video_id, v.title, v.thumbnail_url_medium, 
                       v.view_count, v.like_count, v.published_at
                FROM youtube_videos v
                JOIN youtube_channels c ON v.channel_id = c.id
                WHERE c.user_id = %s AND v.is_optimized = TRUE
                ORDER BY v.published_at DESC
            """, (user_id,))
            
            videos_data = cursor.fetchall()
            
            videos = []
            for video in videos_data:
                videos.append({
                    "id": video[0],
                    "video_id": video[1],
                    "title": video[2],
                    "thumbnail": video[3],
                    "view_count": video[4],
                    "like_count": video[5],
                    "published_at": video[6]
                })
        
        conn.close()
        
        return {
            "status": "success",
            "data": {
                "videos": videos,
                "count": len(videos)
            }
        }
        
    except Exception as e:
        logging.error(f"Error fetching optimized videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching optimized videos: {str(e)}")


class ComprehensiveOptimizationResponse(BaseModel):
    original_title: str
    optimized_title: str
    original_description: str
    optimized_description: str
    original_tags: list[str] = []
    optimized_tags: list[str] = []
    optimization_notes: str

@app.post("/videos/{video_id}/optimize-all", response_model=ComprehensiveOptimizationResponse)
async def optimize_video_comprehensive(video_id: str):
    """Generate comprehensive optimizations (title, description, tags) for a video using Claude 3.7."""
    try:
        conn = get_db_connection()
        
        # Get video data including transcript
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT title, description, tags, transcript, has_captions
                FROM youtube_videos
                WHERE video_id = %s
            """, (video_id,))
            
            video_data = cursor.fetchone()
            if not video_data:
                raise HTTPException(status_code=404, detail="Video not found")
            
            original_title = video_data[0]
            original_description = video_data[1] or ""
            original_tags = video_data[2] or []
            stored_transcript = video_data[3]  # May be None
            stored_has_captions = video_data[4] or False
            
            # Get the user's credentials to fetch the transcript if needed
            user_id = None
            try:
                # Try to get the user ID from the database
                cursor.execute("""
                    SELECT c.user_id
                    FROM youtube_videos v
                    JOIN youtube_channels c ON v.channel_id = c.id
                    WHERE v.video_id = %s
                """, (video_id,))
                user_result = cursor.fetchone()
                if user_result:
                    user_id = user_result[0]
            except Exception as e:
                logging.warning(f"Error getting user_id for video {video_id}: {e}")
            
            transcript = stored_transcript
            has_captions = stored_has_captions
            
            # If we don't have a transcript stored, try to fetch it on demand
            if not transcript and user_id:
                try:
                    credentials = get_user_credentials(user_id)
                    
                    if credentials:
                        # Fetch transcript directly from YouTube
                        transcript_data = fetch_video_transcript(credentials, video_id)
                        transcript = transcript_data.get("transcript")
                        has_captions = transcript_data.get("has_captions", False)
                        
                        logging.info(f"Fetched transcript on demand for video {video_id}")
                except Exception as e:
                    logging.error(f"Error fetching transcript on demand: {e}")
            
            # Safely check transcript length
            transcript_length = 0
            if transcript is not None:
                transcript_length = len(transcript)
            logging.info(f"Video {video_id} has captions: {has_captions}, transcript length: {transcript_length}")
        
        conn.close()

        # Get comprehensive optimization with all available data including transcript
        logging.info(f"Generating comprehensive optimization for video {video_id}")
        result = get_comprehensive_optimization(
            original_title=original_title,
            original_description=original_description,
            original_tags=original_tags,
            transcript=transcript,
            has_captions=has_captions
        )
        
        # Normalize result to ensure all fields are the correct type
        normalized_result = {
            "original_title": str(result.get("original_title", "")),
            "optimized_title": str(result.get("optimized_title", "")),
            "original_description": str(result.get("original_description", "")),
            "optimized_description": str(result.get("optimized_description", "")),
            "original_tags": [str(tag) for tag in result.get("original_tags", []) if tag],
            "optimized_tags": [str(tag) for tag in result.get("optimized_tags", []) if tag],
            "optimization_notes": str(result.get("optimization_notes", ""))
        }
        
        # Ensure we have valid lists
        if not normalized_result["original_tags"]:
            normalized_result["original_tags"] = []
        if not normalized_result["optimized_tags"]:
            normalized_result["optimized_tags"] = []
        
        # Store the transcript in the database for future use if we fetched it on demand
        if transcript and not stored_transcript and user_id:
            try:
                store_conn = get_db_connection()
                with store_conn.cursor() as store_cursor:
                    store_cursor.execute("""
                        UPDATE youtube_videos
                        SET transcript = %s, has_captions = %s
                        WHERE video_id = %s
                    """, (transcript, has_captions, video_id))
                    store_conn.commit()
                    logging.info(f"Stored transcript for video {video_id} for future use")
            except Exception as store_e:
                logging.error(f"Error storing transcript: {store_e}")
            finally:
                store_conn.close()
            
        return ComprehensiveOptimizationResponse(**normalized_result)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error generating comprehensive optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error optimizing video: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)