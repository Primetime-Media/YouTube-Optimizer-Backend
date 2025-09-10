"""
YouTube Optimizer Backend - Main Application Entry Point

FastAPI application with OAuth2 authentication, CORS configuration,
security headers, and database initialization.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn
import logging
import secrets
# Local imports - organized by functionality
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

# Load environment variables and configure logging
load_dotenv()
settings = get_settings()

logger = setup_logging(
    log_level="INFO",
    log_dir="logs",
    console_output=True,
    max_file_size=10 * 1024 * 1024,  # 10MB per log file
    backup_count=5,
)

# Suppress noisy warnings
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

# Security validation
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

# CORS Configuration - Allow only trusted domains for security
allowed_origins = [settings.frontend_url]
if not settings.is_production:
    allowed_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])

logging.info(f"Allowed origins: {allowed_origins}")

# Add CORS middleware with restrictive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Cookie"],
    expose_headers=["set-cookie"],
)


# Security Headers Middleware - Add protective headers to all responses
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to protect against XSS, clickjacking, and MIME sniffing."""
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Enforce HTTPS in production
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# Register API routers for different functionality domains
app.include_router(analytics_router)    # Analytics endpoints
app.include_router(auth_router)         # Authentication
app.include_router(channel_router)      # Channel management
app.include_router(health_router)       # Health checks
app.include_router(scheduler_router)    # Background tasks
app.include_router(video_router)        # Video optimization


# Application startup - Initialize database, scheduler, and security
@app.on_event("startup")
async def startup_db_client():
    """Initialize database, scheduler, and perform security validations on startup."""
    # Verify system entropy for secure token generation
    try:
        secrets.token_bytes(32)
        logging.info("System entropy verified")
    except Exception as e:
        logging.error(f"System entropy check failed: {e}")
        if settings.is_production:
            raise RuntimeError("Insufficient system entropy for secure operations")

    # Initialize systems
    try:
        init_db()
        initialize_scheduler()
        cleanup_invalid_sessions()
        logging.info("Database and scheduler initialized")
    except ConnectionError as e:
        logging.error(f"Database initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize database: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during initialization: {e}")
        raise RuntimeError(f"Failed to initialize application: {e}")


# Main entry point - Run application directly with uvicorn
if __name__ == "__main__":
    """Start the FastAPI application on host 0.0.0.0:8080 with auto-reload in dev."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=not settings.is_production
    )
