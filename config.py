# config.py
"""
Application Configuration Management
====================================
Centralized configuration with environment variable support,
validation, and type safety using Pydantic Settings.

Environment variables can be set in .env file or system environment.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional, List, Dict, Any
from functools import lru_cache
import os
import logging
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden via environment variables.
    For nested configs, use double underscore notation (e.g., DATABASE__HOST)
    """
    
    # ============================================================================
    # APPLICATION SETTINGS
    # ============================================================================
    APP_NAME: str = Field(
        default="YouTube Channel Optimizer API",
        description="Application name"
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version"
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    ENVIRONMENT: str = Field(
        default="production",
        description="Environment: development, staging, production"
    )
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent,
        description="Base directory of the application"
    )
    
    # ============================================================================
    # API CONFIGURATION
    # ============================================================================
    API_V1_PREFIX: str = Field(
        default="/api/v1",
        description="API version 1 prefix"
    )
    HOST: str = Field(
        default="0.0.0.0",
        description="Host to bind the server"
    )
    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port to bind the server"
    )
    WORKERS: int = Field(
        default=4,
        ge=1,
        description="Number of worker processes"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["*"],
        description="Allowed hosts for the application"
    )
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8000",
            "https://localhost:3000"
        ],
        description="Allowed CORS origins"
    )
    CORS_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    CORS_METHODS: List[str] = Field(
        default=["*"],
        description="Allowed CORS methods"
    )
    CORS_HEADERS: List[str] = Field(
        default=["*"],
        description="Allowed CORS headers"
    )
    
    # ============================================================================
    # SECURITY SETTINGS
    # ============================================================================
    SECRET_KEY: str = Field(
        ...,  # Required field
        min_length=32,
        description="Secret key for JWT and encryption (MUST be set in environment)"
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        description="Access token expiration in minutes"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
        description="Refresh token expiration in days"
    )
    PASSWORD_MIN_LENGTH: int = Field(
        default=8,
        ge=6,
        description="Minimum password length"
    )
    CLOUD_SCHEDULER_SECRET: str = Field(
        default="your-secret-key-here",
        description="Secret for Cloud Scheduler authentication"
    )
    
    # ============================================================================
    # DATABASE CONFIGURATION
    # ============================================================================
    DATABASE_HOST: str = Field(
        default="localhost",
        description="Database host"
    )
    DATABASE_PORT: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="Database port"
    )
    DATABASE_NAME: str = Field(
        default="youtube_optimizer",
        description="Database name"
    )
    DATABASE_USER: str = Field(
        default="postgres",
        description="Database user"
    )
    DATABASE_PASSWORD: str = Field(
        ...,  # Required field
        description="Database password (MUST be set in environment)"
    )
    DATABASE_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        description="Database connection pool size"
    )
    DATABASE_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        description="Database connection pool max overflow"
    )
    DATABASE_POOL_TIMEOUT: int = Field(
        default=30,
        ge=1,
        description="Database connection pool timeout in seconds"
    )
    DATABASE_SSL_MODE: str = Field(
        default="prefer",
        description="Database SSL mode: disable, allow, prefer, require, verify-ca, verify-full"
    )
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL from components."""
        return (
            f"postgresql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )
    
    # ============================================================================
    # REDIS CONFIGURATION
    # ============================================================================
    REDIS_HOST: str = Field(
        default="localhost",
        description="Redis host"
    )
    REDIS_PORT: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port"
    )
    REDIS_DB: int = Field(
        default=0,
        ge=0,
        description="Redis database number"
    )
    REDIS_PASSWORD: Optional[str] = Field(
        default=None,
        description="Redis password (optional)"
    )
    REDIS_SSL: bool = Field(
        default=False,
        description="Use SSL for Redis connection"
    )
    
    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL from components."""
        protocol = "rediss" if self.REDIS_SSL else "redis"
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"{protocol}://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    CACHE_TTL: int = Field(
        default=300,
        ge=0,
        description="Default cache TTL in seconds"
    )
    
    # ============================================================================
    # RATE LIMITING
    # ============================================================================
    RATE_LIMITING_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    RATE_LIMIT_STORAGE: str = Field(
        default="redis",
        description="Rate limit storage backend: redis or memory"
    )
    DEFAULT_RATE_LIMIT: str = Field(
        default="100/minute",
        description="Default rate limit for endpoints"
    )
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format: json or text"
    )
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Log file path (optional, logs to stdout if not set)"
    )
    LOG_ROTATION: str = Field(
        default="1 day",
        description="Log rotation interval"
    )
    LOG_RETENTION: str = Field(
        default="30 days",
        description="Log retention period"
    )
    
    # ============================================================================
    # METRICS & MONITORING
    # ============================================================================
    METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    METRICS_BACKEND: str = Field(
        default="prometheus",
        description="Metrics backend: prometheus, statsd, cloudwatch, datadog, console"
    )
    METRICS_PORT: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Metrics endpoint port"
    )
    METRICS_NAMESPACE: str = Field(
        default="youtube_optimizer",
        description="Metrics namespace/prefix"
    )
    
    # ============================================================================
    # AI/LLM CONFIGURATION (ANTHROPIC CLAUDE)
    # ============================================================================
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key for Claude AI"
    )
    ANTHROPIC_DEFAULT_MODEL: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Default Claude model to use"
    )
    ANTHROPIC_FALLBACK_MODELS: List[str] = Field(
        default=[
            "claude-3-5-haiku-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022"
        ],
        description="Fallback models if primary fails"
    )
    ANTHROPIC_MAX_RETRIES: int = Field(
        default=3,
        ge=1,
        description="Maximum retry attempts for Anthropic API"
    )
    ANTHROPIC_TIMEOUT: int = Field(
        default=60,
        ge=10,
        description="Anthropic API timeout in seconds"
    )
    
    # ============================================================================
    # GOOGLE TRENDS & SERPAPI CONFIGURATION
    # ============================================================================
    SERPAPI_API_KEY: Optional[str] = Field(
        default=None,
        description="SerpAPI key for Google Trends data"
    )
    SERPAPI_TIMEOUT: int = Field(
        default=30,
        ge=5,
        description="SerpAPI timeout in seconds"
    )
    TRENDS_DEFAULT_TIMEFRAME: str = Field(
        default="now 7-d",
        description="Default timeframe for Google Trends queries"
    )
    TRENDS_MAX_KEYWORDS_PER_BATCH: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum keywords per Google Trends batch"
    )
    
    # ============================================================================
    # YOUTUBE API CONFIGURATION
    # ============================================================================
    YOUTUBE_API_KEY: Optional[str] = Field(
        default=None,
        description="YouTube Data API key"
    )
    YOUTUBE_CLIENT_ID: Optional[str] = Field(
        default=None,
        description="YouTube OAuth client ID"
    )
    YOUTUBE_CLIENT_SECRET: Optional[str] = Field(
        default=None,
        description="YouTube OAuth client secret"
    )
    YOUTUBE_REDIRECT_URI: Optional[str] = Field(
        default=None,
        description="YouTube OAuth redirect URI"
    )
    YOUTUBE_QUOTA_PER_DAY: int = Field(
        default=10000,
        ge=0,
        description="YouTube API daily quota limit"
    )
    YOUTUBE_API_SERVICE_NAME: str = Field(
        default="youtube",
        description="YouTube API service name"
    )
    YOUTUBE_API_VERSION: str = Field(
        default="v3",
        description="YouTube API version"
    )
    
    # ============================================================================
    # OPTIMIZATION SETTINGS
    # ============================================================================
    MAX_OPTIMIZATION_RETRIES: int = Field(
        default=3,
        ge=0,
        description="Maximum optimization retries on failure"
    )
    OPTIMIZATION_TIMEOUT: int = Field(
        default=300,
        ge=1,
        description="Optimization timeout in seconds"
    )
    AUTO_APPLY_OPTIMIZATIONS: bool = Field(
        default=False,
        description="Automatically apply optimizations to YouTube"
    )
    BATCH_OPTIMIZATION_LIMIT: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum videos per batch optimization"
    )
    USE_STATISTICAL_ANALYSIS: bool = Field(
        default=True,
        description="Use statistical analysis for optimization decisions"
    )
    MIN_OPTIMIZATION_CONFIDENCE: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to apply optimization"
    )
    OPTIMIZATION_COOLING_PERIOD_DAYS: int = Field(
        default=7,
        ge=1,
        description="Days to wait between optimizations for same video"
    )
    MAX_HASHTAGS_PER_VIDEO: int = Field(
        default=15,
        ge=1,
        le=30,
        description="Maximum number of hashtags per video"
    )
    MAINTAIN_ORIGINAL_DESCRIPTION: bool = Field(
        default=True,
        description="Keep original description and append optimizations"
    )
    
    # ============================================================================
    # BACKGROUND TASKS & CELERY
    # ============================================================================
    CELERY_BROKER_URL: Optional[str] = Field(
        default=None,
        description="Celery broker URL (e.g., Redis or RabbitMQ)"
    )
    CELERY_RESULT_BACKEND: Optional[str] = Field(
        default=None,
        description="Celery result backend URL"
    )
    CELERY_TASK_TIME_LIMIT: int = Field(
        default=3600,
        ge=1,
        description="Celery task time limit in seconds"
    )
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(
        default=3000,
        ge=1,
        description="Celery task soft time limit in seconds"
    )
    
    # ============================================================================
    # SCHEDULER SETTINGS
    # ============================================================================
    SCHEDULER_ENABLED: bool = Field(
        default=True,
        description="Enable scheduled tasks"
    )
    MONTHLY_OPTIMIZATION_DAY: int = Field(
        default=1,
        ge=1,
        le=28,
        description="Day of month to run monthly optimizations"
    )
    SCHEDULER_TIMEZONE: str = Field(
        default="UTC",
        description="Timezone for scheduled tasks"
    )
    
    # ============================================================================
    # ERROR TRACKING (SENTRY)
    # ============================================================================
    SENTRY_DSN: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    SENTRY_ENVIRONMENT: Optional[str] = Field(
        default=None,
        description="Sentry environment name"
    )
    SENTRY_TRACES_SAMPLE_RATE: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sentry traces sample rate"
    )
    SENTRY_PROFILES_SAMPLE_RATE: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sentry profiles sample rate"
    )
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    ENABLE_SWAGGER_UI: bool = Field(
        default=True,
        description="Enable Swagger UI documentation"
    )
    ENABLE_REDOC: bool = Field(
        default=True,
        description="Enable ReDoc documentation"
    )
    ENABLE_RATE_LIMITING: bool = Field(
        default=True,
        description="Enable rate limiting globally"
    )
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    ENABLE_CACHING: bool = Field(
        default=True,
        description="Enable caching"
    )
    ENABLE_COMPRESSION: bool = Field(
        default=True,
        description="Enable response compression"
    )
    ENABLE_CSRF_PROTECTION: bool = Field(
        default=False,
        description="Enable CSRF protection"
    )
    ENABLE_LLM_OPTIMIZATION: bool = Field(
        default=True,
        description="Enable AI-powered LLM optimization"
    )
    ENABLE_MULTILINGUAL_SUPPORT: bool = Field(
        default=True,
        description="Enable multilingual content optimization"
    )
    
    # ============================================================================
    # AWS CONFIGURATION (for CloudWatch, S3, etc.)
    # ============================================================================
    AWS_REGION: Optional[str] = Field(
        default=None,
        description="AWS region"
    )
    AWS_ACCESS_KEY_ID: Optional[str] = Field(
        default=None,
        description="AWS access key ID"
    )
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(
        default=None,
        description="AWS secret access key"
    )
    S3_BUCKET_NAME: Optional[str] = Field(
        default=None,
        description="S3 bucket for file storage"
    )
    
    # ============================================================================
    # EMAIL CONFIGURATION
    # ============================================================================
    SMTP_HOST: Optional[str] = Field(
        default=None,
        description="SMTP host for sending emails"
    )
    SMTP_PORT: int = Field(
        default=587,
        ge=1,
        le=65535,
        description="SMTP port"
    )
    SMTP_USER: Optional[str] = Field(
        default=None,
        description="SMTP username"
    )
    SMTP_PASSWORD: Optional[str] = Field(
        default=None,
        description="SMTP password"
    )
    SMTP_TLS: bool = Field(
        default=True,
        description="Use TLS for SMTP"
    )
    EMAIL_FROM: Optional[str] = Field(
        default=None,
        description="Default sender email address"
    )
    
    # ============================================================================
    # VALIDATORS
    # ============================================================================
    @field_validator('ENVIRONMENT')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        allowed = ['development', 'staging', 'production']
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v_upper
    
    @field_validator('METRICS_BACKEND')
    @classmethod
    def validate_metrics_backend(cls, v: str) -> str:
        """Validate metrics backend."""
        allowed = ['prometheus', 'statsd', 'cloudwatch', 'datadog', 'console']
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Metrics backend must be one of {allowed}")
        return v_lower
    
    # ============================================================================
    # PYDANTIC CONFIGURATION
    # ============================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra fields in .env
        validate_default=True
    )
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.ENVIRONMENT == "staging"
    
    def get_database_url(self, driver: str = "postgresql") -> str:
        """Get database URL with custom driver."""
        return (
            f"{driver}://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (excluding sensitive data)."""
        sensitive_fields = {
            'SECRET_KEY', 'DATABASE_PASSWORD', 'REDIS_PASSWORD',
            'YOUTUBE_API_KEY', 'YOUTUBE_CLIENT_SECRET',
            'AWS_SECRET_ACCESS_KEY', 'SMTP_PASSWORD',
            'SENTRY_DSN', 'CLOUD_SCHEDULER_SECRET',
            'ANTHROPIC_API_KEY', 'SERPAPI_API_KEY'
        }
        
        return {
            key: '***REDACTED***' if key in sensitive_fields else value
            for key, value in self.model_dump().items()
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures settings are loaded only once.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Convenience function to reload settings (useful for testing)
def reload_settings() -> Settings:
    """
    Reload settings by clearing cache.
    
    Returns:
        Settings: Fresh settings instance
    """
    get_settings.cache_clear()
    return get_settings()


# Export settings instance
settings = get_settings()


# Logging setup
def setup_logging(settings: Settings):
    """
    Configure application logging based on settings.
    
    Args:
        settings: Application settings
    """
    import sys
    
    # Standard Python logging configuration
    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(filename)s:%(lineno)d - %(message)s'
    )
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Set third-party loggers to WARNING
    for logger_name in ['urllib3', 'googleapiclient', 'google', 'asyncio', 'anthropic']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
