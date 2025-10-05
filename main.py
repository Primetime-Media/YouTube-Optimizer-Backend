# main.py
"""
Production-Ready FastAPI Application
YouTube Video Optimization SaaS Platform
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import Dict, Any
import os

from config import settings
from utils.db import init_database, cleanup_database, check_database_health
from utils.logging_config import setup_logging

# Import routers
from routes import (
    video_routes,
    channel_routes,
    analytics_routes,
    auth_routes,
    health_routes,
    scheduler_routes
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

setup_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    
    Handles:
    - Database connection pool initialization
    - Resource cleanup on shutdown
    """
    # Startup
    logger.info("🚀 Starting YouTube Optimizer API...")
    
    try:
        # Initialize database connection pool
        init_database()
        logger.info("✅ Database connection pool initialized")
        
        # Verify database health
        health = check_database_health()
        if not health['healthy']:
            logger.error(f"❌ Database health check failed: {health.get('error')}")
            raise RuntimeError("Database not healthy")
        
        logger.info(
            f"✅ Database health check passed "
            f"(response time: {health['response_time_ms']}ms)"
        )
        
        logger.info("✅ Application startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down YouTube Optimizer API...")
    
    try:
        cleanup_database()
        logger.info("✅ Database cleanup complete")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}", exc_info=True)
    
    logger.info("✅ Application shutdown complete")


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================

app = FastAPI(
    title="YouTube Video Optimizer API",
    description="AI-powered YouTube video optimization platform",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan,
    openapi_url="/openapi.json" if settings.environment != "production" else None
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)

# Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted Host (security)
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )


# Request logging and timing middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all requests with timing information
    """
    start_time = time.time()
    
    # Generate request ID
    request_id = str(time.time_ns())
    
    # Log request
    logger.info(
        f"Request started: {request.method} {request.url.path} "
        f"[ID: {request_id}]"
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{process_time:.4f}s"
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"[ID: {request_id}] "
            f"Status: {response.status_code} "
            f"Time: {process_time:.4f}s"
        )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"[ID: {request_id}] "
            f"Error: {str(e)} "
            f"Time: {process_time:.4f}s",
            exc_info=True
        )
        raise


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Add security headers to all responses
    """
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    if settings.environment == "production":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'"
        )
    
    return response


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors with detailed error messages
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {errors}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Request validation failed",
            "errors": errors
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle all unhandled exceptions
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True
    )
    
    if settings.environment == "production":
        # Don't expose internal errors in production
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal error occurred. Please try again later."
            }
        )
    else:
        # Show detailed errors in development
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )


# ============================================================================
# ROUTES
# ============================================================================

# Health check routes (no auth required)
app.include_router(
    health_routes.router,
    tags=["Health"]
)

# Authentication routes
app.include_router(
    auth_routes.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

# Video optimization routes
app.include_router(
    video_routes.router,
    prefix="/api/v1/videos",
    tags=["Videos"]
)

# Channel management routes
app.include_router(
    channel_routes.router,
    prefix="/api/v1/channels",
    tags=["Channels"]
)

# Analytics routes
app.include_router(
    analytics_routes.router,
    prefix="/api/v1/analytics",
    tags=["Analytics"]
)

# Scheduler routes
app.include_router(
    scheduler_routes.router,
    prefix="/api/v1/scheduler",
    tags=["Scheduler"]
)


# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint - API information
    """
    return {
        "name": "YouTube Video Optimizer API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs" if settings.environment != "production" else "disabled",
        "health": "/health"
    }


@app.get("/info")
async def info() -> Dict[str, Any]:
    """
    Get API information and configuration
    """
    return {
        "name": "YouTube Video Optimizer API",
        "version": "1.0.0",
        "environment": settings.environment,
        "features": {
            "video_optimization": True,
            "channel_optimization": True,
            "analytics": True,
            "scheduling": True,
            "thumbnail_generation": True
        },
        "limits": {
            "max_upload_size_mb": 100,
            "max_videos_per_batch": 100,
            "rate_limit_per_minute": 60
        }
    }


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.environment != "production" else False,
        log_level="info",
        access_log=True
    )
