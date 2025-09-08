"""
YouTube Optimizer Backend - Main Application Entry Point

This module serves as the main entry point for the YouTube Optimizer Backend API.
It initializes the FastAPI application, configures middleware, sets up routing,
and handles application lifecycle events.

Key responsibilities:
- FastAPI application initialization with security configurations
- CORS middleware setup for frontend integration
- Security headers middleware for enhanced protection
- Database and scheduler initialization
- Route registration for all API endpoints
- Application startup and shutdown event handling

Author: YouTube Optimizer Team
Version: 1.0.0
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

# =============================================================================
# ENVIRONMENT AND CONFIGURATION SETUP
# =============================================================================

# Load environment variables from .env file
# This must be done before importing any modules that depend on environment variables
load_dotenv()

# Load application settings from configuration module
# Settings include database URLs, API keys, security configurations, etc.
settings = get_settings()

# Configure application logging with rotation and console output
# This ensures proper log management for production and development environments
logger = setup_logging(
    log_level="INFO",  # Set to DEBUG for development, INFO for production
    log_dir="logs",    # Directory where log files will be stored
    console_output=True,  # Enable console output for development
    max_file_size=10 * 1024 * 1024,  # 10MB per log file before rotation
    backup_count=5,  # Keep last 5 log files for historical reference
)

# Suppress noisy googleapiclient warnings that clutter the logs
# These warnings are related to API discovery cache and don't affect functionality
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

# Security validation: Ensure session secret is properly configured in production
# This prevents security vulnerabilities from missing session secrets
if settings.is_production and not os.getenv("SESSION_SECRET"):
    raise ValueError("SESSION_SECRET environment variable must be set in production")

# =============================================================================
# FASTAPI APPLICATION INITIALIZATION
# =============================================================================

# Initialize FastAPI application with security-conscious configuration
app = FastAPI(
    title=settings.app_name,  # Application name from settings
    debug=settings.debug,     # Debug mode based on environment
    # Hide API documentation in production for security
    # This prevents exposure of API structure to potential attackers
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# =============================================================================
# CORS (Cross-Origin Resource Sharing) CONFIGURATION
# =============================================================================

# Configure CORS to allow only trusted domains for security
# This prevents unauthorized cross-origin requests from malicious websites
allowed_origins = [settings.frontend_url]  # Primary frontend URL from settings

# Add development origins for local development
# These are only allowed in non-production environments
if not settings.is_production:
    allowed_origins.extend([
        "http://localhost:3000",    # React development server
        "http://127.0.0.1:3000",   # Alternative localhost address
    ])

# Log CORS configuration for debugging and monitoring
logging.info(f"Allowed origins: {allowed_origins}")
logging.info(f"Environment: {settings.environment}")
logging.info(f"Frontend URL: {settings.frontend_url}")
logging.info(f"Backend URL: {settings.backend_url}")
logging.info(f"Is production: {settings.is_production}")

# Add CORS middleware with restrictive security settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Only allow explicitly listed domains
    allow_credentials=True,         # Required for cookie/session authentication
    allow_methods=[                 # Restrict to only necessary HTTP methods
        "GET",
        "POST", 
        "PUT",
        "DELETE",
        "OPTIONS",  # Required for preflight requests
    ],
    allow_headers=[                 # Restrict to only necessary headers
        "Authorization",            # For API authentication
        "Content-Type",            # For request body type specification
        "Cookie",                  # For session management
    ],
    expose_headers=["set-cookie"], # Allow frontend to read set-cookie header
)


# =============================================================================
# SECURITY HEADERS MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Security middleware to inject protective headers into every HTTP response.
    
    This middleware adds several security headers that help protect against
    common web vulnerabilities including XSS, clickjacking, and MIME type sniffing.
    
    Args:
        request: The incoming HTTP request
        call_next: The next middleware/handler in the chain
        
    Returns:
        HTTP response with security headers added
    """
    # Process the request through the application
    response = await call_next(request)
    
    # Add security headers to protect against common vulnerabilities
    response.headers["X-Content-Type-Options"] = "nosniff"  # Prevent MIME type sniffing
    response.headers["X-Frame-Options"] = "DENY"            # Prevent clickjacking attacks
    response.headers["X-XSS-Protection"] = "1; mode=block"  # Enable XSS filtering
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"  # Control referrer info
    
    # Additional security headers for production environments
    if settings.is_production:
        # Enforce HTTPS connections in production
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"  # 1 year HSTS with subdomain inclusion
        )
        # Optional: Uncomment for stronger XSS/CSRF protection
        # This would restrict all resources to same-origin only
        # response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response


# =============================================================================
# API ROUTE REGISTRATION
# =============================================================================

# Register all API routers with the FastAPI application
# Each router handles a specific domain of functionality
app.include_router(analytics_router)    # Analytics and reporting endpoints
app.include_router(auth_router)         # Authentication and user management
app.include_router(channel_router)      # YouTube channel management and optimization
app.include_router(health_router)       # Health check and system status endpoints
app.include_router(scheduler_router)    # Background task scheduling and management
app.include_router(video_router)        # Video optimization and management endpoints


# =============================================================================
# APPLICATION STARTUP EVENT HANDLER
# =============================================================================

@app.on_event("startup")
async def startup_db_client():
    """
    Application startup event handler that initializes all required systems.
    
    This function is called when the FastAPI application starts up and is responsible
    for initializing the database, scheduler, and performing security validations.
    
    The startup process includes:
    1. System entropy verification for cryptographic security
    2. Database initialization and connection setup
    3. Background scheduler initialization
    4. Session cleanup of invalid/expired tokens
    
    Raises:
        RuntimeError: If system entropy is insufficient in production
        Exception: If database or scheduler initialization fails
    """
    # Verify system entropy for cryptographic security
    # This ensures we can generate secure random tokens and session IDs
    try:
        secrets.token_bytes(32)  # Test generation of 32-byte random token
        logging.info("System entropy verified for secure session generation")
    except Exception as e:
        logging.error(f"System entropy check failed: {e}")
        # In production, insufficient entropy is a critical security issue
        if settings.is_production:
            raise RuntimeError("Insufficient system entropy for secure operations")

    # Initialize database connection and create tables if needed
    # This sets up the PostgreSQL connection and ensures all required tables exist
    init_db()

    # Initialize background scheduler resources
    # This prepares the APScheduler for handling background optimization tasks
    # Note: This initializes resources but doesn't start any jobs yet
    initialize_scheduler()

    # Clean up any invalid or expired session tokens from previous runs
    # This helps maintain database hygiene and prevents token accumulation
    cleanup_invalid_sessions()

    logging.info(
        "Database and scheduler system initialized with secure session management support"
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Main entry point for running the application directly.
    
    This is used when running the application with 'python main.py' instead of
    using uvicorn directly. The application will start on host 0.0.0.0 and port 8080.
    
    Configuration:
    - host="0.0.0.0": Listen on all available network interfaces
    - port=8080: Use port 8080 for the HTTP server
    - reload: Enable auto-reload in development, disable in production
    """
    uvicorn.run(
        "main:app",                           # Application module and instance
        host="0.0.0.0",                      # Listen on all interfaces
        port=8080,                           # HTTP server port
        reload=not settings.is_production    # Auto-reload only in development
    )
