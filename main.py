# main.py
"""
Main FastAPI Application - YouTube Channel Optimizer
====================================================
Production-ready FastAPI application with comprehensive features:
- Multi-route architecture (channels, videos, scheduler, health)
- AI-powered LLM optimization using Anthropic Claude
- Authentication & authorization
- Rate limiting
- Metrics & monitoring
- Error tracking
- Database connection pooling
- Background tasks
- Comprehensive middleware stack

Author: YouTube Optimizer Team
Version: 1.0.0
"""

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Callable, Any
import logging
import sys
import time
import traceback
from datetime import datetime
import uuid
import os

from config import get_settings, setup_logging
from utils.exceptions import BaseAPIException

# Import routes
from routes import channel_routes, video_routes, scheduler_routes, health_routes

# Import utilities (conditional import based on availability)
try:
    from utils.auth import auth_middleware
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Auth middleware not available, continuing without authentication")
    auth_middleware = None

try:
    from utils.rate_limiter import RateLimiter, add_rate_limit_headers
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Rate limiter not available")
    RateLimiter = None
    add_rate_limit_headers = None

try:
    from utils.metrics import MetricsCollector, MetricType
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Metrics collector not available")
    MetricsCollector = None
    MetricType = None

# Initialize settings
settings = get_settings()

# Setup logging
setup_logging(settings)
logger = logging.getLogger(__name__)

# Log AI/LLM configuration status
if settings.ANTHROPIC_API_KEY:
    logger.info("✓ Anthropic Claude AI enabled for content optimization")
else:
    logger.warning("⚠ Anthropic API key not configured - LLM optimization features will be limited")

if settings.SERPAPI_API_KEY:
    logger.info("✓ SerpAPI enabled for Google Trends integration")
else:
    logger.warning("⚠ SerpAPI key not configured - Trending keywords features will be limited")

# Initialize utilities
if MetricsCollector:
    metrics = MetricsCollector(
        backend=settings.METRICS_BACKEND if settings.METRICS_ENABLED else "console",
        namespace=settings.METRICS_NAMESPACE,
        enabled=settings.METRICS_ENABLED
    )
else:
    metrics = None

if RateLimiter:
    rate_limiter = RateLimiter(
        redis_url=settings.REDIS_URL if settings.RATE_LIMITING_ENABLED else None,
        enabled=settings.RATE_LIMITING_ENABLED
    )
else:
    rate_limiter = None


# ============================================================================
# APPLICATION LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events for proper resource management.
    """
    # ========================================
    # STARTUP
    # ========================================
    startup_time = time.time()
    
    logger.info("=" * 80)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info("=" * 80)
    
    # Run comprehensive startup validation
    try:
        from utils.startup_checks import validate_startup
        
        validation_passed = await validate_startup(settings)
        
        if not validation_passed:
            logger.critical("Startup validation failed! Application cannot start safely.")
            if settings.is_production():
                logger.critical("Exiting due to failed startup checks in production mode.")
                sys.exit(1)
            else:
                logger.warning("Continuing despite failed checks (development mode)")
        
    except ImportError:
        logger.warning("Startup validation module not available, skipping checks")
    except Exception as e:
        logger.error(f"Error during startup validation: {e}", exc_info=True)
        if settings.is_production():
            sys.exit(1)
    
    # Initialize Sentry for error tracking
    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.logging import LoggingIntegration
            
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.SENTRY_ENVIRONMENT or settings.ENVIRONMENT,
                traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
                profiles_sample_rate=settings.SENTRY_PROFILES_SAMPLE_RATE,
                integrations=[
                    FastApiIntegration(),
                    LoggingIntegration(
                        level=logging.INFO,
                        event_level=logging.ERROR
                    )
                ],
                release=settings.APP_VERSION,
                send_default_pii=False,
                attach_stacktrace=True,
            )
            logger.info("✓ Sentry error tracking initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Sentry: {e}")
    
    # Initialize database connection pool
    try:
        from utils.db import init_db_pool, test_db_connection
        
        init_db_pool(
            host=settings.DATABASE_HOST,
            port=settings.DATABASE_PORT,
            database=settings.DATABASE_NAME,
            user=settings.DATABASE_USER,
            password=settings.DATABASE_PASSWORD,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW
        )
        
        # Test connection
        if test_db_connection():
            logger.info("✓ Database connection pool initialized and tested")
            if metrics:
                metrics.increment("app.startup.database_connected")
        else:
            logger.error("✗ Database connection test failed")
            if metrics:
                metrics.increment("app.startup.database_failed")
            
    except Exception as e:
        logger.error(f"✗ Failed to initialize database: {e}")
        if metrics:
            metrics.increment("app.startup.database_error")
        if settings.is_production():
            raise  # Fail fast in production
    
    # Initialize Redis connection
    if settings.REDIS_URL:
        try:
            import redis
            redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            redis_client.ping()
            logger.info("✓ Redis connection initialized")
            if metrics:
                metrics.increment("app.startup.redis_connected")
        except Exception as e:
            logger.warning(f"⚠ Redis connection failed (non-critical): {e}")
            if metrics:
                metrics.increment("app.startup.redis_failed")
    
    # Validate AI/LLM configuration
    if settings.ENABLE_LLM_OPTIMIZATION:
        if not settings.ANTHROPIC_API_KEY:
            logger.warning("⚠ LLM optimization enabled but ANTHROPIC_API_KEY not set")
            logger.warning("⚠ AI-powered optimization features will not be available")
        else:
            try:
                import anthropic
                # Test Anthropic connection
                client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                logger.info("✓ Anthropic Claude client initialized successfully")
                logger.info(f"✓ Default model: {settings.ANTHROPIC_DEFAULT_MODEL}")
                logger.info(f"✓ Fallback models: {', '.join(settings.ANTHROPIC_FALLBACK_MODELS)}")
            except Exception as e:
                logger.error(f"✗ Failed to initialize Anthropic client: {e}")
    
    # Validate Google Trends configuration
    if not settings.SERPAPI_API_KEY:
        logger.warning("⚠ SERPAPI_API_KEY not set - trending keywords features limited")
    else:
        logger.info("✓ SerpAPI configured for Google Trends integration")
    
    # Initialize background task scheduler (if enabled)
    if settings.SCHEDULER_ENABLED:
        try:
            logger.info("✓ Task scheduler initialized")
            if metrics:
                metrics.increment("app.startup.scheduler_initialized")
        except Exception as e:
            logger.warning(f"⚠ Task scheduler initialization failed: {e}")
    
    # Record startup metrics
    startup_duration = time.time() - startup_time
    if metrics:
        metrics.histogram("app.startup.duration", startup_duration)
        metrics.gauge("app.info", 1, tags={
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT
        })
    
    logger.info(f"✓ Application started successfully in {startup_duration:.2f}s")
    logger.info("=" * 80)
    
    yield
    
    # ========================================
    # SHUTDOWN
    # ========================================
    logger.info("=" * 80)
    logger.info("Shutting down application...")
    
    # Close database connections
    try:
        from utils.db import close_db_pool
        close_db_pool()
        logger.info("✓ Database connections closed")
    except Exception as e:
        logger.error(f"✗ Error closing database: {e}")
    
    # Close Redis connections
    if settings.REDIS_URL:
        try:
            logger.info("✓ Redis connections closed")
        except Exception as e:
            logger.error(f"✗ Error closing Redis: {e}")
    
    if metrics:
        metrics.increment("app.shutdown")
    logger.info("✓ Application shutdown complete")
    logger.info("=" * 80)


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## YouTube Channel Optimizer API
    
    AI-powered YouTube channel and video optimization platform using Anthropic's Claude AI.
    
    ### Features
    * 🤖 **AI-Powered Optimization** - Uses Claude AI for intelligent content optimization
    * 📊 **Channel Metadata Optimization** - Enhance channel descriptions and keywords
    * 🎬 **Video Optimization** - Optimize titles, descriptions, tags with AI
    * 📈 **Trending Keywords** - Integration with Google Trends via SerpAPI
    * 🌍 **Multilingual Support** - Optimize content in multiple languages
    * 📅 **Automated Scheduling** - Schedule regular optimizations
    * 🔄 **Batch Processing** - Optimize multiple videos at once
    * 📊 **Statistical Analysis** - Data-driven optimization decisions
    * 🔐 **Secure Authentication** - JWT-based security
    
    ### AI Capabilities
    * Title optimization using Claude AI
    * Description enhancement with SEO keywords
    * Hashtag generation and trending analysis
    * Competitor keyword extraction
    * Multi-language content translation
    * Automatic chapter generation from transcripts
    
    ### Rate Limits
    * Channel Optimization: 10 requests/minute
    * Video Operations: 30 requests/minute
    * Batch Operations: 5 requests/hour
    * AI Optimization: 5 requests/minute
    
    ### Authentication
    All endpoints (except health checks) require JWT authentication.
    Include the token in the `Authorization` header:
    ```
    Authorization: Bearer <your-token>
    ```
    
    ### AI Models Available
    * Primary: {model}
    * Fallbacks: {fallbacks}
    """.format(
        model=settings.ANTHROPIC_DEFAULT_MODEL,
        fallbacks=", ".join(settings.ANTHROPIC_FALLBACK_MODELS[:2])
    ),
    docs_url="/docs" if settings.ENABLE_SWAGGER_UI else None,
    redoc_url="/redoc" if settings.ENABLE_REDOC else None,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
    debug=settings.DEBUG,
    # OpenAPI customization
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check and system status endpoints"
        },
        {
            "name": "Channel Optimization",
            "description": "AI-powered YouTube channel metadata optimization"
        },
        {
            "name": "Video Operations",
            "description": "Video management and AI optimization endpoints"
        },
        {
            "name": "Optimization Scheduler",
            "description": "Scheduled and automated optimization tasks"
        },
        {
            "name": "Authentication",
            "description": "User authentication and authorization"
        }
    ],
    responses={
        401: {
            "description": "Unauthorized - Invalid or missing authentication",
            "content": {
                "application/json": {
                    "example": {
                        "error": "unauthorized",
                        "message": "Invalid or missing authentication token"
                    }
                }
            }
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "content": {
                "application/json": {
                    "example": {
                        "error": "forbidden",
                        "message": "Insufficient permissions for this operation"
                    }
                }
            }
        },
        429: {
            "description": "Too Many Requests - Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "error": "rate_limit_exceeded",
                        "message": "Rate limit exceeded. Try again later.",
                        "retry_after": 60
                    }
                }
            }
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "internal_server_error",
                        "message": "An unexpected error occurred"
                    }
                }
            }
        }
    }
)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(BaseAPIException)
async def api_exception_handler(request: Request, exc: BaseAPIException):
    """Handle custom API exceptions with proper formatting."""
    logger.warning(
        f"API Exception: {exc.error_code}",
        extra={
            "error_code": exc.error_code,
            "message": exc.message,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    if metrics:
        metrics.increment("app.exceptions.api", tags={
            "error_code": exc.error_code,
            "endpoint": request.url.path
        })
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(
        f"Validation Error: {exc.errors()}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "errors": exc.errors()
        }
    )
    
    if metrics:
        metrics.increment("app.exceptions.validation", tags={
            "endpoint": request.url.path
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled Exception: {str(exc)}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc()
        },
        exc_info=True
    )
    
    if metrics:
        metrics.increment("app.exceptions.unhandled", tags={
            "endpoint": request.url.path,
            "exception_type": type(exc).__name__
        })
    
    # Don't expose internal errors in production
    if settings.is_production():
        error_message = "An unexpected error occurred. Please try again later."
        details = {"request_id": request_id}
    else:
        error_message = str(exc)
        details = {
            "request_id": request_id,
            "type": type(exc).__name__,
            "traceback": traceback.format_exc().split('\n')
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": error_message,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Callable):
    """Add unique request ID to all requests."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next: Callable):
    """Track request timing and record metrics."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        if metrics:
            metrics.record_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                duration=duration,
                user_id=getattr(request.state, "user", {}).get("id")
            )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        # Log slow requests
        if duration > 1.0:  # Log requests slower than 1 second
            logger.warning(
                f"Slow request: {request.method} {request.url.path}",
                extra={
                    "duration": duration,
                    "status_code": response.status_code,
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        if metrics:
            metrics.record_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=500,
                duration=duration
            )
        raise


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next: Callable):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    if settings.is_production():
        response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
    expose_headers=[
        "X-Request-ID",
        "X-Response-Time",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset"
    ]
)

# Add GZip compression
if settings.ENABLE_COMPRESSION:
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,
        compresslevel=6
    )

# Add trusted host middleware (production only)
if settings.is_production() and settings.ALLOWED_HOSTS != ["*"]:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# Add custom middleware
if auth_middleware:
    app.middleware("http")(auth_middleware)

if settings.ENABLE_RATE_LIMITING and add_rate_limit_headers:
    app.middleware("http")(add_rate_limit_headers)


# ============================================================================
# INCLUDE ROUTERS
# ============================================================================

# Health check routes (no prefix, public)
app.include_router(
    health_routes.router,
    tags=["Health"]
)

# API v1 routes
app.include_router(
    channel_routes.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Channel Optimization"]
)

app.include_router(
    video_routes.router,
    prefix=f"{settings.API_V1_PREFIX}/videos",
    tags=["Video Operations"]
)

app.include_router(
    scheduler_routes.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Optimization Scheduler"]
)


# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    if settings.ENABLE_SWAGGER_UI:
        return RedirectResponse(url="/docs")
    else:
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "status": "operational",
            "documentation": "/redoc" if settings.ENABLE_REDOC else None,
            "ai_features": {
                "llm_optimization": settings.ENABLE_LLM_OPTIMIZATION,
                "multilingual_support": settings.ENABLE_MULTILINGUAL_SUPPORT,
                "anthropic_configured": bool(settings.ANTHROPIC_API_KEY),
                "serpapi_configured": bool(settings.SERPAPI_API_KEY)
            }
        }


@app.get("/info", tags=["Root"])
async def app_info():
    """Get application information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "features": {
            "swagger_ui": settings.ENABLE_SWAGGER_UI,
            "redoc": settings.ENABLE_REDOC,
            "rate_limiting": settings.ENABLE_RATE_LIMITING,
            "metrics": settings.ENABLE_METRICS,
            "caching": settings.ENABLE_CACHING,
            "llm_optimization": settings.ENABLE_LLM_OPTIMIZATION,
            "multilingual": settings.ENABLE_MULTILINGUAL_SUPPORT
        },
        "ai_configuration": {
            "anthropic_enabled": bool(settings.ANTHROPIC_API_KEY),
            "default_model": settings.ANTHROPIC_DEFAULT_MODEL if settings.ANTHROPIC_API_KEY else None,
            "serpapi_enabled": bool(settings.SERPAPI_API_KEY),
            "statistical_analysis": settings.USE_STATISTICAL_ANALYSIS
        },
        "api": {
            "version": "v1",
            "prefix": settings.API_V1_PREFIX,
            "documentation": "/docs" if settings.ENABLE_SWAGGER_UI else None
        }
    }


# ============================================================================
# PROMETHEUS METRICS ENDPOINT
# ============================================================================

if settings.METRICS_BACKEND == "prometheus" and settings.METRICS_ENABLED:
    try:
        from prometheus_client import make_asgi_app
        
        # Mount Prometheus metrics app
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
        
        logger.info(f"✓ Prometheus metrics endpoint available at /metrics")
    except ImportError:
        logger.warning("⚠ Prometheus client not installed, metrics endpoint disabled")


# ============================================================================
# STARTUP EVENT LOGGING
# ============================================================================

@app.on_event("startup")
async def log_routes():
    """Log all registered routes on startup."""
    if settings.DEBUG:
        logger.info("Registered routes:")
        for route in app.routes:
            if hasattr(route, "methods"):
                logger.info(f"  {', '.join(route.methods)} {route.path}")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": settings.HOST,
        "port": settings.PORT,
        "reload": settings.DEBUG,
        "log_level": settings.LOG_LEVEL.lower(),
        "access_log": True,
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
    }
    
    # Use workers in production (but not with reload)
    if settings.is_production() and not settings.DEBUG:
        uvicorn_config["workers"] = settings.WORKERS
    
    logger.info("Starting Uvicorn server...")
    logger.info(f"Server will be available at http://{settings.HOST}:{settings.PORT}")
    logger.info(f"Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    if settings.ANTHROPIC_API_KEY:
        logger.info(f"AI Features: ✓ ENABLED")
    else:
        logger.warning(f"AI Features: ⚠ LIMITED (no Anthropic API key)")
    
    uvicorn.run(**uvicorn_config)
