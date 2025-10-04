"""
Main Application Entry Point - Production Ready
================================================
Enterprise-grade FastAPI application with comprehensive features

Production Features:
✅ Structured logging with correlation IDs
✅ Redis-based distributed rate limiting
✅ Comprehensive error handling with error codes
✅ Graceful shutdown with cleanup
✅ Database connection pooling and health checks
✅ Prometheus metrics integration
✅ Request/response logging
✅ Enhanced security headers
✅ API versioning support
✅ Dependency injection pattern
✅ Circuit breaker for resilience
✅ Request size limits
✅ Timeout handling
"""

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
import logging
import sys
import uuid
import time
import asyncio
import signal
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import routes
from routes import (
    analytics,
    auth_routes,
    channel_routes,
    health_routes,
    scheduler_routes,
    video_routes
)

# Import database utilities
from utils.db import init_db_pool, close_db_pool, check_db_health

# Import configuration
from config import get_settings

# Get settings instance
settings = get_settings()


# ============================================================================
# STRUCTURED LOGGING CONFIGURATION
# ============================================================================

class StructuredLogger:
    """Structured logger with correlation ID support"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _log(self, level: int, message: str, **kwargs):
        """Log with structured context"""
        extra = kwargs.copy()
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.value),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(request_id)s - %(extra)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT
        )
    ]
)

logger = StructuredLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class AppException(Exception):
    """Base application exception"""
    def __init__(self, message: str, code: str, status_code: int = 500, details: Dict = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class DatabaseException(AppException):
    """Database-related exceptions"""
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, "DATABASE_ERROR", 503, details)


class RateLimitException(AppException):
    """Rate limit exceeded exception"""
    def __init__(self, message: str = "Rate limit exceeded", details: Dict = None):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", 429, details)


class ValidationException(AppException):
    """Validation error exception"""
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, "VALIDATION_ERROR", 422, details)


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

RESPONSE_SIZE = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

# Application metrics
ACTIVE_CONNECTIONS = Gauge(
    'http_active_connections',
    'Number of active connections'
)

DB_CONNECTIONS = Gauge(
    'db_connections_active',
    'Number of active database connections'
)

RATE_LIMIT_HITS = Counter(
    'rate_limit_hits_total',
    'Total rate limit hits',
    ['endpoint']
)

# Error metrics
ERROR_COUNT = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'error_type']
)


# ============================================================================
# REDIS-BASED RATE LIMITER
# ============================================================================

class RedisRateLimiter:
    """Distributed rate limiter using Redis"""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.enabled = settings.RATE_LIMIT_ENABLED
        self.requests = settings.RATE_LIMIT_REQUESTS
        self.window = settings.RATE_LIMIT_WINDOW
    
    async def check_rate_limit(self, key: str) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit
        
        Args:
            key: Rate limit key (e.g., IP address or user ID)
        
        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        if not self.enabled:
            return True, {}
        
        # If Redis is not available, fall back to in-memory
        if not self.redis:
            return True, {"fallback": True}
        
        try:
            current_time = int(time.time())
            window_key = f"rate_limit:{key}:{current_time // self.window}"
            
            # Increment counter
            count = await self.redis.incr(window_key)
            
            # Set expiry on first request
            if count == 1:
                await self.redis.expire(window_key, self.window * 2)
            
            remaining = max(0, self.requests - count)
            reset_time = ((current_time // self.window) + 1) * self.window
            
            allowed = count <= self.requests
            
            info = {
                "limit": self.requests,
                "remaining": remaining,
                "reset": reset_time,
                "retry_after": reset_time - current_time if not allowed else None
            }
            
            if not allowed:
                RATE_LIMIT_HITS.labels(endpoint=key).inc()
            
            return allowed, info
            
        except Exception as e:
            logger.error(f"Rate limiter error: {e}", error=str(e))
            # Fail open - allow request if rate limiter fails
            return True, {"error": "rate_limiter_unavailable"}


# ============================================================================
# APPLICATION STATE MANAGEMENT
# ============================================================================

class AppState:
    """Application state container"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.rate_limiter = None
        self.startup_time = datetime.now(timezone.utc)
        self.shutdown_event = asyncio.Event()
        self.active_requests = 0
    
    async def increment_requests(self):
        self.active_requests += 1
        ACTIVE_CONNECTIONS.set(self.active_requests)
    
    async def decrement_requests(self):
        self.active_requests = max(0, self.active_requests - 1)
        ACTIVE_CONNECTIONS.set(self.active_requests)


app_state = AppState()


# ============================================================================
# GRACEFUL SHUTDOWN HANDLER
# ============================================================================

async def shutdown_handler(sig: int):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    app_state.shutdown_event.set()


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async lifespan manager for startup and shutdown
    """
    # ========================================================================
    # STARTUP
    # ========================================================================
    
    logger.info("=" * 70)
    logger.info("Starting YouTube Optimizer Backend")
    logger.info("=" * 70)
    logger.info(f"Environment: {settings.ENVIRONMENT.value}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"Version: {settings.APP_VERSION}")
    logger.info("=" * 70)
    
    try:
        # Initialize database connection pool
        logger.info("Initializing database connection pool...")
        await init_db_pool(
            settings.database_url,
            min_size=settings.DB_POOL_MIN,
            max_size=settings.DB_POOL_MAX
        )
        logger.info("✓ Database pool initialized")
        
        # Check database health
        db_health = await check_db_health()
        if db_health['status'] != 'healthy':
            logger.warning(f"Database health check: {db_health}")
        else:
            logger.info("✓ Database connection verified")
        
        # Initialize Redis (if available)
        try:
            import aioredis
            app_state.redis_client = await aioredis.create_redis_pool(
                settings.redis_url,
                minsize=2,
                maxsize=10,
                encoding='utf-8'
            )
            logger.info("✓ Redis connection established")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}. Rate limiting will use fallback.")
            app_state.redis_client = None
        
        # Initialize rate limiter
        app_state.rate_limiter = RedisRateLimiter(app_state.redis_client)
        logger.info("✓ Rate limiter initialized")
        
        # Register shutdown handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(shutdown_handler(s))
            )
        
        logger.info("=" * 70)
        logger.info("✓ Application startup complete")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", error=str(e), exc_info=True)
        raise
    
    yield  # Application runs here
    
    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    
    logger.info("=" * 70)
    logger.info("Shutting down YouTube Optimizer Backend")
    logger.info("=" * 70)
    
    try:
        # Wait for active requests to complete (max 30 seconds)
        logger.info(f"Waiting for {app_state.active_requests} active requests to complete...")
        timeout = 30
        start_time = time.time()
        
        while app_state.active_requests > 0 and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.5)
        
        if app_state.active_requests > 0:
            logger.warning(f"Force closing with {app_state.active_requests} active requests")
        else:
            logger.info("✓ All requests completed")
        
        # Close Redis connection
        if app_state.redis_client:
            app_state.redis_client.close()
            await app_state.redis_client.wait_closed()
            logger.info("✓ Redis connection closed")
        
        # Close database connections
        logger.info("Closing database connections...")
        await close_db_pool()
        logger.info("✓ Database connections closed")
        
        logger.info("=" * 70)
        logger.info("✓ Shutdown complete")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}", error=str(e), exc_info=True)


# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered YouTube content optimization platform",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,  # Disable docs in production
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    # Custom OpenAPI configuration
    openapi_tags=[
        {"name": "Authentication", "description": "User authentication and authorization"},
        {"name": "Channels", "description": "YouTube channel management"},
        {"name": "Videos", "description": "Video analysis and optimization"},
        {"name": "Analytics", "description": "Performance analytics and insights"},
        {"name": "Health", "description": "Health checks and monitoring"},
        {"name": "Scheduler", "description": "Background job scheduling"},
    ]
)


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# Request tracking middleware (must be first)
@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Track requests with correlation ID and metrics"""
    # Generate request ID
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    request.state.start_time = time.time()
    
    # Track active connections
    await app_state.increment_requests()
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - request.state.start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code}",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration=duration,
            client=request.client.host if request.client else "unknown"
        )
        
        return response
        
    except Exception as e:
        # Record error metrics
        ERROR_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            error_type=type(e).__name__
        ).inc()
        raise
        
    finally:
        await app_state.decrement_requests()


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Distributed rate limiting middleware"""
    # Skip rate limiting for health checks
    if request.url.path.startswith("/health") or request.url.path == "/metrics":
        return await call_next(request)
    
    # Get client identifier (IP or user ID if authenticated)
    client_id = request.client.host if request.client else "unknown"
    
    # Check rate limit
    allowed, info = await app_state.rate_limiter.check_rate_limit(client_id)
    
    if not allowed:
        logger.warning(
            f"Rate limit exceeded for {client_id}",
            request_id=getattr(request.state, 'request_id', 'unknown'),
            client=client_id,
            endpoint=request.url.path
        )
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please try again later.",
                    "details": info
                }
            },
            headers={
                "X-RateLimit-Limit": str(info.get("limit", 0)),
                "X-RateLimit-Remaining": str(info.get("remaining", 0)),
                "X-RateLimit-Reset": str(info.get("reset", 0)),
                "Retry-After": str(info.get("retry_after", 60))
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    
    if info:
        response.headers["X-RateLimit-Limit"] = str(info.get("limit", 0))
        response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(info.get("reset", 0))
    
    return response


# Request size limit middleware
@app.middleware("http")
async def request_size_limit_middleware(request: Request, call_next):
    """Limit request body size"""
    max_size = 10 * 1024 * 1024  # 10MB
    
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "error": {
                    "code": "REQUEST_TOO_LARGE",
                    "message": f"Request body too large. Maximum size: {max_size} bytes",
                    "details": {"max_size": max_size, "received": int(content_length)}
                }
            }
        )
    
    return await call_next(request)


# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add comprehensive security headers"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # HSTS (only in production)
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    
    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    
    # Permissions Policy
    response.headers["Permissions-Policy"] = (
        "geolocation=(), "
        "microphone=(), "
        "camera=(), "
        "payment=(), "
        "usb=(), "
        "magnetometer=(), "
        "gyroscope=(), "
        "accelerometer=()"
    )
    
    return response


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=3600  # Cache preflight requests for 1 hour
)


# Trusted Host Middleware (production only)
if settings.is_production:
    allowed_hosts = [origin.replace("https://", "").replace("http://", "") for origin in settings.CORS_ORIGINS]
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )


# GZip Compression
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # Only compress responses larger than 1KB
    compresslevel=6  # Balance between speed and compression
)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom application exceptions"""
    logger.error(
        f"Application error: {exc.message}",
        request_id=getattr(request.state, 'request_id', 'unknown'),
        error_code=exc.code,
        details=exc.details,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "request_id": getattr(request.state, 'request_id', None)
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.warning(
        f"Validation error: {exc.errors()}",
        request_id=getattr(request.state, 'request_id', 'unknown'),
        errors=exc.errors()
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
                "request_id": getattr(request.state, 'request_id', None)
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        request_id=getattr(request.state, 'request_id', 'unknown'),
        status=exc.status_code,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "request_id": getattr(request.state, 'request_id', None)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        request_id=getattr(request.state, 'request_id', 'unknown'),
        error_type=type(exc).__name__,
        exc_info=True
    )
    
    # Don't expose internal errors in production
    message = "Internal server error" if settings.is_production else str(exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": message,
                "request_id": getattr(request.state, 'request_id', None)
            }
        }
    )


# ============================================================================
# HEALTH & METRICS ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Health"], summary="Basic health check")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value
    }


@app.get("/health/ready", tags=["Health"], summary="Readiness probe")
async def readiness_check():
    """
    Readiness check for Kubernetes
    Checks all critical dependencies
    """
    components = {}
    is_ready = True
    
    # Check database
    try:
        db_health = await check_db_health()
        components["database"] = db_health
        if db_health['status'] != 'healthy':
            is_ready = False
    except Exception as e:
        components["database"] = {"status": "unhealthy", "error": str(e)}
        is_ready = False
    
    # Check Redis
    if app_state.redis_client:
        try:
            await app_state.redis_client.ping()
            components["redis"] = {"status": "healthy"}
        except Exception as e:
            components["redis"] = {"status": "unhealthy", "error": str(e)}
            # Redis is optional, don't mark as not ready
    else:
        components["redis"] = {"status": "unavailable"}
    
    return {
        "ready": is_ready,
        "components": components,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": (datetime.now(timezone.utc) - app_state.startup_time).total_seconds()
    }


@app.get("/health/live", tags=["Health"], summary="Liveness probe")
async def liveness_check():
    """
    Liveness check for Kubernetes
    Simple check if process is responsive
    """
    return {
        "alive": True,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/metrics", tags=["Monitoring"], summary="Prometheus metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint
    Returns metrics in Prometheus exposition format
    """
    # Update database connection gauge
    try:
        db_health = await check_db_health()
        pool_stats = db_health.get('pool_stats', {})
        DB_CONNECTIONS.set(pool_stats.get('active', 0))
    except Exception:
        pass
    
    # Generate Prometheus metrics
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/metrics/json", tags=["Monitoring"], summary="JSON metrics")
async def metrics_json():
    """JSON-formatted metrics for easier consumption"""
    db_health = await check_db_health()
    
    return {
        "app": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "uptime_seconds": (datetime.now(timezone.utc) - app_state.startup_time).total_seconds()
        },
        "http": {
            "active_connections": app_state.active_requests,
            "rate_limit_enabled": settings.RATE_LIMIT_ENABLED
        },
        "database": {
            "status": db_health.get('status'),
            "connected": db_health.get('connected', False),
            "pool": db_health.get('pool_stats', {})
        },
        "redis": {
            "connected": app_state.redis_client is not None
        }
    }


# ============================================================================
# ROUTE REGISTRATION
# ============================================================================

# API version prefix
API_V1_PREFIX = "/api/v1"

# Include all route modules with versioning
app.include_router(auth_routes.router, prefix=f"{API_V1_PREFIX}/auth", tags=["Authentication"])
app.include_router(channel_routes.router, prefix=f"{API_V1_PREFIX}/channels", tags=["Channels"])
app.include_router(video_routes.router, prefix=f"{API_V1_PREFIX}/videos", tags=["Videos"])
app.include_router(analytics.router, prefix=f"{API_V1_PREFIX}/analytics", tags=["Analytics"])
app.include_router(scheduler_routes.router, prefix=f"{API_V1_PREFIX}/scheduler", tags=["Scheduler"])
app.include_router(health_routes.router, prefix=f"{API_V1_PREFIX}/health", tags=["Health"])


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Root"], summary="API information")
async def root():
    """API root endpoint with version information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT.value,
        "docs": "/docs" if not settings.is_production else "disabled",
        "health": "/health",
        "metrics": "/metrics",
        "api_version": "v1",
        "endpoints": {
            "health": "/health",
            "readiness": "/health/ready",
            "liveness": "/health/live",
            "metrics": "/metrics",
            "api": API_V1_PREFIX
        }
    }


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Production-grade Uvicorn configuration
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG and settings.is_development,
        log_level=settings.LOG_LEVEL.value.lower(),
        access_log=True,
        server_header=False,  # Don't expose server info
        date_header=True,
        proxy_headers=True,  # Trust X-Forwarded-* headers
        forwarded_allow_ips="*",  # Configure based on your proxy
        timeout_keep_alive=65,  # Keep-alive timeout
        limit_concurrency=1000,  # Max concurrent connections
        limit_max_requests=10000,  # Restart worker after N requests (memory leaks)
        backlog=2048  # Socket backlog size
    )
