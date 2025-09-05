from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uvicorn
import logging
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
import secrets

# Get application settings
settings = get_settings()

# Configure logging with automatic folder creation
from utils.logging_config import setup_logging
logger = setup_logging(
    log_level="INFO",
    log_dir="logs",
    console_output=True,
    max_file_size=10 * 1024 * 1024,  # 10MB
    backup_count=5
)

# Suppress googleapiclient.discovery_cache warnings
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

load_dotenv()

# Security: Ensure proper session secret configuration in production
# The SESSION_SECRET environment variable should be set to a cryptographically
# secure random value in production environments. This prevents predictable
# session generation and ensures proper security.
if settings.is_production and not os.getenv("SESSION_SECRET"):
    raise ValueError("SESSION_SECRET environment variable must be set in production")

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None
)

# Configure CORS
allowed_origins = [settings.frontend_url]
if not settings.is_production:
    allowed_origins.extend([
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
    ])

logging.info(f"Allowed origins: {allowed_origins}")
logging.info(f"Environment: {settings.environment}")
logging.info(f"Frontend URL: {settings.frontend_url}")
logging.info(f"Backend URL: {settings.backend_url}")
logging.info(f"Is production: {settings.is_production}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],  # Add Cookie header
    expose_headers=["content-type"]  # Expose set-cookie header
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
app.include_router(auth_router)
app.include_router(channel_router)
app.include_router(health_router)
app.include_router(scheduler_router)
app.include_router(video_router)

@app.on_event("startup")
async def startup_db_client():
    """Initialize the database and scheduler system on startup."""
    # Verify system entropy for secure session generation
    try:
        # Test entropy availability
        test_token = secrets.token_bytes(32)
        logging.info("System entropy verified for secure session generation")
    except Exception as e:
        logging.error(f"System entropy check failed: {e}")
        if settings.is_production:
            raise RuntimeError("Insufficient system entropy for secure operations")
    
    init_db()
    initialize_scheduler()  # Just initializes scheduler resources, doesn't run any jobs
    
    # Clean up any invalid session tokens in the database
    cleanup_invalid_sessions()
    
    logging.info("Database and scheduler system initialized with secure session management support")










if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)