"""
Configuration Settings Module
==============================
Production-ready configuration with environment variables

Features:
- Pydantic settings validation
- Environment-based configuration
- Type safety
- Default values
- Sensitive data handling
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
import os
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings with validation
    
    All settings can be overridden via environment variables
    """
    
    # ========================================================================
    # Application Settings
    # ========================================================================
    
    APP_NAME: str = "YouTube Optimizer"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # ========================================================================
    # Database Settings
    # ========================================================================
    
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=5432, env="DB_PORT")
    DB_NAME: str = Field(default="youtube_optimizer", env="DB_NAME")
    DB_USER: str = Field(default="postgres", env="DB_USER")
    DB_PASSWORD: str = Field(default="", env="DB_PASSWORD")
    
    # Connection pool settings
    DB_POOL_MIN: int = Field(default=2, env="DB_POOL_MIN")
    DB_POOL_MAX: int = Field(default=10, env="DB_POOL_MAX")
    DB_TIMEOUT: int = Field(default=30, env="DB_TIMEOUT")
    
    # Encryption key for sensitive data
    DB_ENCRYPTION_KEY: Optional[str] = Field(default=None, env="DB_ENCRYPTION_KEY")
    
    # ========================================================================
    # Security Settings
    # ========================================================================
    
    SECRET_KEY: str = Field(..., env="SECRET_KEY")  # Required
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")  # Required
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_MINUTES: int = Field(default=60, env="JWT_EXPIRATION_MINUTES")
    
    # Password hashing
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_SPECIAL: bool = True
    
    # ========================================================================
    # CORS Settings
    # ========================================================================
    
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # ========================================================================
    # API Keys
    # ========================================================================
    
    # YouTube Data API
    YOUTUBE_API_KEY: Optional[str] = Field(default=None, env="YOUTUBE_API_KEY")
    
    # Anthropic Claude API
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    CLAUDE_MODEL: str = Field(default="claude-sonnet-4-5-20250929", env="CLAUDE_MODEL")
    CLAUDE_MAX_TOKENS: int = Field(default=4096, env="CLAUDE_MAX_TOKENS")
    
    # OpenAI API (for fallback)
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4-1", env="OPENAI_MODEL")
    
    # Google Cloud Vision API
    GOOGLE_VISION_CREDENTIALS: Optional[str] = Field(default=None, env="GOOGLE_VISION_CREDENTIALS")
    
    # Stripe (for payments)
    STRIPE_API_KEY: Optional[str] = Field(default=None, env="STRIPE_API_KEY")
    STRIPE_WEBHOOK_SECRET: Optional[str] = Field(default=None, env="STRIPE_WEBHOOK_SECRET")
    
    # SendGrid (for emails)
    SENDGRID_API_KEY: Optional[str] = Field(default=None, env="SENDGRID_API_KEY")
    SENDGRID_FROM_EMAIL: str = Field(default="noreply@youtubeoptimizer.com", env="SENDGRID_FROM_EMAIL")
    
    # SerpAPI (for Google Trends)
    SERPAPI_KEY: Optional[str] = Field(default=None, env="SERPAPI_KEY")
    
    # ========================================================================
    # Rate Limiting
    # ========================================================================
    
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # ========================================================================
    # Caching
    # ========================================================================
    
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # ========================================================================
    # Background Jobs
    # ========================================================================
    
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        env="CELERY_BROKER_URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/0",
        env="CELERY_RESULT_BACKEND"
    )
    
    # ========================================================================
    # Logging
    # ========================================================================
    
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="app.log", env="LOG_FILE")
    LOG_MAX_BYTES: int = Field(default=10485760, env="LOG_MAX_BYTES")  # 10MB
    LOG_BACKUP_COUNT: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # ========================================================================
    # Feature Flags
    # ========================================================================
    
    FEATURE_A_B_TESTING: bool = Field(default=True, env="FEATURE_A_B_TESTING")
    FEATURE_COMPETITOR_ANALYSIS: bool = Field(default=True, env="FEATURE_COMPETITOR_ANALYSIS")
    FEATURE_THUMBNAIL_OPTIMIZATION: bool = Field(default=True, env="FEATURE_THUMBNAIL_OPTIMIZATION")
    FEATURE_HASHTAG_OPTIMIZATION: bool = Field(default=True, env="FEATURE_HASHTAG_OPTIMIZATION")
    
    # ========================================================================
    # Optimization Settings
    # ========================================================================
    
    # Statistical thresholds
    CONFIDENCE_THRESHOLD: float = Field(default=0.6, env="CONFIDENCE_THRESHOLD")
    UPLIFT_THRESHOLD: float = Field(default=0.03, env="UPLIFT_THRESHOLD")  # 3%
    
    # Cooling-off periods (days)
    COOLOFF_FIRST_OPT: int = Field(default=7, env="COOLOFF_FIRST_OPT")
    COOLOFF_REPEAT_OPT: int = Field(default=14, env="COOLOFF_REPEAT_OPT")
    COOLOFF_NO_OPT: int = Field(default=7, env="COOLOFF_NO_OPT")
    
    # View thresholds
    LOW_VIEW_THRESHOLD: int = Field(default=100, env="LOW_VIEW_THRESHOLD")
    MEDIUM_VIEW_THRESHOLD: int = Field(default=1000, env="MEDIUM_VIEW_THRESHOLD")
    
    # ========================================================================
    # Email Settings
    # ========================================================================
    
    EMAIL_ENABLED: bool = Field(default=True, env="EMAIL_ENABLED")
    EMAIL_NOTIFICATIONS_ENABLED: bool = Field(default=True, env="EMAIL_NOTIFICATIONS_ENABLED")
    
    # ========================================================================
    # Monitoring
    # ========================================================================
    
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    HEALTH_CHECK_ENABLED: bool = Field(default=True, env="HEALTH_CHECK_ENABLED")
    
    # ========================================================================
    # Pydantic Config
    # ========================================================================
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    # ========================================================================
    # Computed Properties
    # ========================================================================
    
    @property
    def database_url(self) -> str:
        """Get full database URL"""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    @property
    def redis_url(self) -> str:
        """Get full Redis URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT.lower() in ["development", "dev"]
    
    # ========================================================================
    # Validation Methods
    # ========================================================================
    
    def validate_required_keys(self) -> List[str]:
        """
        Validate that all required API keys are set
        
        Returns:
            List of missing keys
        """
        missing_keys = []
        
        required_keys = {
            "YOUTUBE_API_KEY": self.YOUTUBE_API_KEY,
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY,
            "SECRET_KEY": self.SECRET_KEY,
            "JWT_SECRET_KEY": self.JWT_SECRET_KEY,
        }
        
        for key_name, key_value in required_keys.items():
            if not key_value:
                missing_keys.append(key_name)
        
        return missing_keys
    
    def log_configuration(self) -> None:
        """Log current configuration (safe - no secrets)"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("Configuration Loaded")
        logger.info("=" * 60)
        logger.info(f"Environment: {self.ENVIRONMENT}")
        logger.info(f"Debug Mode: {self.DEBUG}")
        logger.info(f"Host: {self.HOST}:{self.PORT}")
        logger.info(f"Database: {self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}")
        logger.info(f"Redis: {self.REDIS_HOST}:{self.REDIS_PORT}")
        logger.info(f"Claude Model: {self.CLAUDE_MODEL}")
        logger.info(f"Rate Limiting: {self.RATE_LIMIT_ENABLED}")
        logger.info(f"A/B Testing: {self.FEATURE_A_B_TESTING}")
        
        # Check for missing keys
        missing = self.validate_required_keys()
        if missing:
            logger.warning(f"Missing API keys: {', '.join(missing)}")
        else:
            logger.info("All required API keys configured")
        
        logger.info("=" * 60)


# ============================================================================
# Settings Instance (Cached)
# ============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Returns:
        Settings instance
    """
    return Settings()


# Create global settings instance
settings = get_settings()


# ============================================================================
# Configuration Validation on Import
# ============================================================================

if __name__ != "__main__":
    # Log configuration when module is imported (not when run directly)
    settings.log_configuration()
