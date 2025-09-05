from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn
import logging
import secrets

# Local imports
from utils.db import init_db
from routes.analytics import router as analytics_router
from routes.auth_routes import router as auth_router
from routes.channel_routes import router as channel_router
from routes.health_routes import router as health_router
from routes.scheduler_routes import router as scheduler_router
from routes.video_routes import router as video_router
from config import get_settings
from utils.auth import cleanup_invalid_sessions
from services.scheduler import initialize_scheduler
from utils.logging_config import setup_logging

# Load environment variables from .env file
load_dotenv()

# Load application settings
settings = get_settings()

# Configure logging with rotation and console output
logger = setup_logging(
    log_level="INFO",
    log_dir="logs",
    console_output=True,
    max_file_size=10 * 1024 * 1024,  # 10MB per log file
    backup_count=5,  # Keep last 5 log files
)

# Suppress noisy googleapiclient warnings
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

# Security: Ensure session secret is properly configured in production
if settings.is_production and not os.getenv("SESSION_SECRET"):
    raise ValueError("SESSION_SECRET environment variable must be set in production")

# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    # Hide docs in production for security
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# -------------------------------
#  CORS CONFIGURATION
# -------------------------------
# Allow only trusted domains (frontend URL in prod, localhost in dev)
allowed_origins = [settings.frontend_url]
if not settings.is_production:
    allowed_origins.extend(
        [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

logging.info(f"Allowed origins: {allowed_origins}")
logging.info(f"Environment: {settings.environment}")
logging.info(f"Frontend URL: {settings.frontend_url}")
logging.info(f"Backend URL: {settings.backend_url}")
logging.info(f"Is production: {settings.is_production}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  #  Only allow listed domains
    allow_credentials=True,  #  Needed for cookies/session auth
    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "OPTIONS",
    ],  #  Restrict to required methods
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Cookie",
    ],  #  Restrict headers to necessary ones
    expose_headers=["set-cookie"],  # Allow frontend to read set-cookie header
)


# -------------------------------
#  SECURITY HEADERS
# -------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Middleware to inject common security headers into every response.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Enforce HTTPS in production
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        # Optional: uncomment for stronger XSS/CSRF protection
        # response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response


# -------------------------------
#  ROUTERS
# -------------------------------
# Register API routers
app.include_router(analytics_router)
app.include_router(auth_router)
app.include_router(channel_router)
app.include_router(health_router)
app.include_router(scheduler_router)
app.include_router(video_router)


# -------------------------------
#   STARTUP TASKS
# -------------------------------
@app.on_event("startup")
async def startup_db_client():
    """
    Initialize database, scheduler, and session cleanup on startup.
    """
    # Verify system entropy for cryptographic security
    try:
        secrets.token_bytes(32)  # Ensure OS entropy is available
        logging.info("System entropy verified for secure session generation")
    except Exception as e:
        logging.error(f"System entropy check failed: {e}")
        if settings.is_production:
            raise RuntimeError("Insufficient system entropy for secure operations")

    # Initialize database
    init_db()

    # Initialize scheduler resources (does not start jobs)
    initialize_scheduler()

    # Clean up invalid/expired session tokens
    cleanup_invalid_sessions()

    logging.info(
        " Database and scheduler system initialized with secure session management support"
    )


# -------------------------------
#   MAIN ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="0.0.0.0", port=8080, reload=not settings.is_production
    )
