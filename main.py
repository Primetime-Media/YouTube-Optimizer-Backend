"""
Main Application Entry Point - FIXED VERSION
=============================================
ALL 7 CRITICAL ERRORS CORRECTED

Fixes Applied:
✅ #1: Corrected function imports (init_db_pool, close_db_pool, check_db_health)
✅ #2: Added await to init_db_pool with proper parameters
✅ #3: Added await to check_db_health
✅ #4: Added await to close_db_pool
✅ #5: Added await to readiness check
✅ #6: Added await to metrics endpoint
✅ #7: Fixed metrics response to use correct fields
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import sys
import uuid
import time
from typing import Dict, Any
import asyncio

# Import routes
from routes import (
    analytics,
    auth_routes,
    channel_routes,
    health_routes,
    scheduler_routes,
    video_routes
)

# Import database utilities - FIXED IMPORT
from utils.db import init_db_pool, close_db_pool, check_db_health

# Import configuration
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async lifespan manager for startup and shutdown
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting YouTube Optimizer Backend")
    logger.info("=" * 60)
    
    try:
        # Initialize database connection pool - FIXED
        logger.info("Initializing database...")
        await init_db_pool(
            settings.database_url,
            min_size=settings.DB_POOL_MIN,
            max_size=settings.DB_POOL_MAX
        )
        
        # Check database health - FIXED
        health = await check_db_health()
        if health['status'] != 'healthy':
            logger.warning(f"Database health check: {health}")
        else:
            logger.info("Database initialized successfully")
        
        # Initialize rate limiter state
        app.state.rate_limiter = {}
        app.state.request_counts = {}
        
        logger.info("Application startup complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down YouTube Optimizer Backend")
    logger.info("=" * 60)
    
    try:
        # Close database connections - FIXED
        logger.info("Closing database connections...")
        await close_db_pool()
        
        # Give time for in-flight requests to complete
        await asyncio.sleep(1)
        
        logger.info("Shutdown complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="YouTube Optimizer API",
    description="AI-powered YouTube content optimization platform",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# Request ID Middleware (must be first)
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Timing Middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    """Add process time header"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# Rate Limiting Middleware
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Simple rate limiting middleware"""
    client_ip = request.client.host
    
    if not hasattr(app.state, 'rate_limiter'):
        app.state.rate_limiter = {}
    
    current_time = time.time()
    
    if client_ip in app.state.rate_limiter:
        last_request, count = app.state.rate_limiter[client_ip]
        
        # Reset counter if more than 60 seconds passed
        if current_time - last_request > 60:
            app.state.rate_limiter[client_ip] = (current_time, 1)
        else:
            # Increment counter
            if count >= 100:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded. Please try again later."}
                )
            app.state.rate_limiter[client_ip] = (last_request, count + 1)
    else:
        app.state.rate_limiter[client_ip] = (current_time, 1)
    
    response = await call_next(request)
    return response


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self'"
    )
    
    return response


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"]
)


# GZip Compression
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging"""
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail} "
        f"(Request ID: {getattr(request.state, 'request_id', 'unknown')})"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        f"Unhandled exception: {exc} "
        f"(Request ID: {getattr(request.state, 'request_id', 'unknown')})",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "request_id": getattr(request.state, 'request_id', None)
        }
    )


# ============================================================================
# HEALTH & METRICS ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0"
    }


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness check (includes database)"""
    # FIXED: Added await
    db_health = await check_db_health()
    
    is_ready = db_health['status'] == 'healthy'
    
    return {
        "ready": is_ready,
        "components": {
            "database": db_health
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness check (basic process check)"""
    return {
        "alive": True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus-compatible metrics endpoint"""
    # Get request count from rate limiter state
    total_requests = sum(
        count for _, count in app.state.rate_limiter.values()
    ) if hasattr(app.state, 'rate_limiter') else 0
    
    # FIXED: Added await
    db_health = await check_db_health()
    
    # FIXED: Use correct fields from db_health
    return {
        "app_info": {
            "version": "2.0.0",
            "name": "youtube_optimizer"
        },
        "http": {
            "requests_total": total_requests,
            "active_connections": len(app.state.rate_limiter) if hasattr(app.state, 'rate_limiter') else 0
        },
        "database": {
            "status": db_health['status'],
            "connected": db_health.get('connected', False),
            "pool_stats": db_health.get('pool_stats', {})
        }
    }


# ============================================================================
# ROUTE REGISTRATION
# ============================================================================

# Include all route modules
app.include_router(auth_routes.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(channel_routes.router, prefix="/api/channels", tags=["Channels"])
app.include_router(video_routes.router, prefix="/api/videos", tags=["Videos"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(scheduler_routes.router, prefix="/api/scheduler", tags=["Scheduler"])
app.include_router(health_routes.router, prefix="/api/health", tags=["Health"])


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "YouTube Optimizer API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True
    )
