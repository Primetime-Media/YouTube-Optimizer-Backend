"""
Configuration Settings Module - Production Ready
================================================
Enterprise-grade configuration with comprehensive validation

Features:
- Pydantic v2 settings with strict validation
- Environment-based configuration
- Type safety with enums
- Secure defaults
- Comprehensive validation
- Sensitive data handling
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator, SecretStr, PostgresDsn, RedisDsn
from typing import List, Optional, Dict, Any
from enum import Enum
from functools import lru_cache
import secrets
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS FOR CONSTRAINED CHOICES
# ============================================================================

class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class JWTAlgorithm(str, Enum):
    """Supported JWT algorithms"""
    HS256 = "HS256"
    HS384 = "HS384"
    HS512 = "HS512"
    RS256 = "RS256"


# ============================================================================
# MAIN SETTINGS CLASS
# ============================================================================

class Settings(BaseSettings):
    """
    Application settings with comprehensive validation
    
    All settings can be overridden via environment variables.
    Required fields will raise validation errors if not provided.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra fields in .env
        validate_default=True,
        str_strip_whitespace=True
    )
    
    # ========================================================================
    # APPLICATION SETTINGS
    # ========================================================================
    
    APP_NAME: str = Field(
        default="YouTube Optimizer",
        description="Application name"
    )
    APP_VERSION: str = Field(
        default="2.0.0",
        description="Application version"
    )
    DEBUG: bool = Field(
        default=False,
        description="Debug mode - should be False in production"
    )
    ENVIRONMENT: Environment = Field(
        default=Environment.PRODUCTION,
        description="Application environment"
    )
    
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    
    # ========================================================================
    # DATABASE SETTINGS
    # ========================================================================
    
    DB_HOST: str = Field(
        default="localhost",
        min_length=1,
        description="Database host"
    )
    DB_PORT: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="Database port"
    )
    DB_NAME: str = Field(
        default="youtube_optimizer",
        min_length=1,
        description="Database name"
    )
    DB_USER: str = Field(
        default="postgres",
        min_length=1,
        description="Database user"
    )
    DB_PASSWORD: SecretStr = Field(
        ...,  # Required in production
        description="Database password (required)"
    )
    
    # Connection pool settings
    DB_POOL_MIN: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Minimum database pool size"
    )
    DB_POOL_MAX: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum database pool size"
    )
    DB_TIMEOUT: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Database timeout in seconds"
    )
    
    # Encryption key for sensitive data
    DB_ENCRYPTION_KEY: Optional[SecretStr] = Field(
        default=None,
        min_length=32,
        description="32+ character encryption key for sensitive data"
    )
    
    @field_validator("DB_POOL_MAX")
    @classmethod
    def validate_pool_max(cls, v: int, info) -> int:
        """Ensure max pool size >= min pool size"""
        if "DB_POOL_MIN" in info.data and v < info.data["DB_POOL_MIN"]:
            raise ValueError("DB_POOL_MAX must be >= DB_POOL_MIN")
        return v
    
    # ========================================================================
    # SECURITY SETTINGS
    # ========================================================================
    
    SECRET_KEY: SecretStr = Field(
        ...,  # Required
        min_length=32,
        description="Application secret key (min 32 characters)"
    )
    JWT_SECRET_KEY: SecretStr = Field(
        ...,  # Required
        min_length=32,
        description="JWT secret key (min 32 characters)"
    )
    JWT_ALGORITHM: JWTAlgorithm = Field(
        default=JWTAlgorithm.HS256,
        description="JWT signing algorithm"
    )
    JWT_EXPIRATION_MINUTES: int = Field(
        default=60,
        ge=5,
        le=43200,  # Max 30 days
        description="JWT token expiration in minutes"
    )
    
    # Password requirements
    PASSWORD_MIN_LENGTH: int = Field(
        default=8,
        ge=8,
        le=128,
        description="Minimum password length"
    )
    PASSWORD_REQUIRE_SPECIAL: bool = Field(
        default=True,
        description="Require special characters in passwords"
    )
    
    # ========================================================================
    # CORS SETTINGS
    # ========================================================================
    
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v: List[str]) -> List[str]:
        """Validate CORS origins format"""
        for origin in v:
            if not origin.startswith(("http://", "https://")):
                raise ValueError(f"Invalid CORS origin: {origin}. Must start with http:// or https://")
        return v
    
    # ========================================================================
    # API KEYS
    # ========================================================================
    
    # YouTube Data API
    YOUTUBE_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="YouTube Data API key"
    )
    
    # Anthropic Claude API
    ANTHROPIC_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Anthropic API key"
    )
    CLAUDE_MODEL: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Claude model identifier"
    )
    CLAUDE_MAX_TOKENS: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens for Claude responses"
    )
    
    # OpenAI API (fallback)
    OPENAI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4-turbo",
        description="OpenAI model identifier"
    )
    
    # Google Cloud Vision API
    GOOGLE_VISION_CREDENTIALS: Optional[str] = Field(
        default=None,
        description="Path to Google Cloud credentials JSON"
    )
    
    # Stripe (payments)
    STRIPE_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Stripe API key"
    )
    STRIPE_WEBHOOK_SECRET: Optional[SecretStr] = Field(
        default=None,
        description="Stripe webhook secret"
    )
    
    # SendGrid (emails)
    SENDGRID_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="SendGrid API key"
    )
    SENDGRID_FROM_EMAIL: str = Field(
        default="noreply@youtubeoptimizer.com",
        description="SendGrid from email address"
    )
    
    @field_validator("SENDGRID_FROM_EMAIL")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation"""
        if "@" not in v or "." not in v.split("@")[1]:
            raise ValueError("Invalid email address format")
        return v.lower()
    
    # SerpAPI (Google Trends)
    SERPAPI_KEY: Optional[SecretStr] = Field(
        default=None,
        description="SerpAPI key"
    )
    
    # ========================================================================
    # RATE LIMITING
    # ========================================================================
    
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Max requests per window"
    )
    RATE_LIMIT_WINDOW: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Rate limit window in seconds"
    )
    
    # ========================================================================
    # CACHING (REDIS)
    # ========================================================================
    
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
        le=15,
        description="Redis database number"
    )
    REDIS_PASSWORD: Optional[SecretStr] = Field(
        default=None,
        description="Redis password"
    )
    CACHE_TTL: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds"
    )
    
    # ========================================================================
    # BACKGROUND JOBS (CELERY)
    # ========================================================================
    
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL"
    )
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    LOG_FILE: str = Field(
        default="app.log",
        description="Log file path"
    )
    LOG_MAX_BYTES: int = Field(
        default=10485760,  # 10MB
        ge=1048576,  # Min 1MB
        le=104857600,  # Max 100MB
        description="Max log file size in bytes"
    )
    LOG_BACKUP_COUNT: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of log backups to keep"
    )
    
    # ========================================================================
    # FEATURE FLAGS
    # ========================================================================
    
    FEATURE_A_B_TESTING: bool = Field(
        default=True,
        description="Enable A/B testing features"
    )
    FEATURE_COMPETITOR_ANALYSIS: bool = Field(
        default=True,
        description="Enable competitor analysis"
    )
    FEATURE_THUMBNAIL_OPTIMIZATION: bool = Field(
        default=True,
        description="Enable thumbnail optimization"
    )
    FEATURE_HASHTAG_OPTIMIZATION: bool = Field(
        default=True,
        description="Enable hashtag optimization"
    )
    
    # ========================================================================
    # OPTIMIZATION SETTINGS
    # ========================================================================
    
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Statistical confidence threshold"
    )
    UPLIFT_THRESHOLD: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Minimum uplift threshold (3%)"
    )
    
    # Cooling-off periods (days)
    COOLOFF_FIRST_OPT: int = Field(
        default=7,
        ge=1,
        le=365,
        description="First optimization cooloff in days"
    )
    COOLOFF_REPEAT_OPT: int = Field(
        default=14,
        ge=1,
        le=365,
        description="Repeat optimization cooloff in days"
    )
    COOLOFF_NO_OPT: int = Field(
        default=7,
        ge=1,
        le=365,
        description="No optimization cooloff in days"
    )
    
    # View thresholds
    LOW_VIEW_THRESHOLD: int = Field(
        default=100,
        ge=1,
        description="Low view count threshold"
    )
    MEDIUM_VIEW_THRESHOLD: int = Field(
        default=1000,
        ge=1,
        description="Medium view count threshold"
    )
    
    @field_validator("MEDIUM_VIEW_THRESHOLD")
    @classmethod
    def validate_view_thresholds(cls, v: int, info) -> int:
        """Ensure medium threshold > low threshold"""
        if "LOW_VIEW_THRESHOLD" in info.data and v <= info.data["LOW_VIEW_THRESHOLD"]:
            raise ValueError("MEDIUM_VIEW_THRESHOLD must be > LOW_VIEW_THRESHOLD")
        return v
    
    # ========================================================================
    # EMAIL SETTINGS
    # ========================================================================
    
    EMAIL_ENABLED: bool = Field(
        default=True,
        description="Enable email functionality"
    )
    EMAIL_NOTIFICATIONS_ENABLED: bool = Field(
        default=True,
        description="Enable email notifications"
    )
    
    # ========================================================================
    # MONITORING
    # ========================================================================
    
    METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    HEALTH_CHECK_ENABLED: bool = Field(
        default=True,
        description="Enable health check endpoints"
    )
    
    # ========================================================================
    # MODEL VALIDATORS
    # ========================================================================
    
    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Validate critical production settings"""
        if self.ENVIRONMENT == Environment.PRODUCTION:
            # Production must not be in debug mode
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production environment")
            
            # Production must have HTTPS in CORS origins
            for origin in self.CORS_ORIGINS:
                if origin.startswith("http://") and "localhost" not in origin:
                    logger.warning(
                        f"Non-HTTPS CORS origin in production: {origin}"
                    )
        
        return self
    
    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        """Warn about missing critical API keys"""
        critical_keys = {
            "YOUTUBE_API_KEY": self.YOUTUBE_API_KEY,
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY,
        }
        
        missing = [name for name, value in critical_keys.items() if not value]
        
        if missing and self.ENVIRONMENT == Environment.PRODUCTION:
            logger.warning(
                f"Missing critical API keys in production: {', '.join(missing)}"
            )
        
        return self
    
    # ========================================================================
    # COMPUTED PROPERTIES
    # ========================================================================
    
    @property
    def database_url(self) -> str:
        """Get full database URL (with exposed password)"""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD.get_secret_value()}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    @property
    def database_url_safe(self) -> str:
        """Get database URL with masked password for logging"""
        return (
            f"postgresql://{self.DB_USER}:****"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    @property
    def redis_url(self) -> str:
        """Get full Redis URL"""
        if self.REDIS_PASSWORD:
            password = self.REDIS_PASSWORD.get_secret_value()
            return f"redis://:{password}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def redis_url_safe(self) -> str:
        """Get Redis URL with masked password for logging"""
        if self.REDIS_PASSWORD:
            return f"redis://:****@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT in [Environment.DEVELOPMENT, Environment.TESTING]
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_required_api_keys(self) -> Dict[str, bool]:
        """
        Get status of required API keys
        
        Returns:
            Dict mapping key names to their configured status
        """
        return {
            "YOUTUBE_API_KEY": self.YOUTUBE_API_KEY is not None,
            "ANTHROPIC_API_KEY": self.ANTHROPIC_API_KEY is not None,
            "SECRET_KEY": True,  # Always present (required)
            "JWT_SECRET_KEY": True,  # Always present (required)
            "DB_PASSWORD": True,  # Always present (required)
        }
    
    def get_optional_api_keys(self) -> Dict[str, bool]:
        """
        Get status of optional API keys
        
        Returns:
            Dict mapping key names to their configured status
        """
        return {
            "OPENAI_API_KEY": self.OPENAI_API_KEY is not None,
            "STRIPE_API_KEY": self.STRIPE_API_KEY is not None,
            "SENDGRID_API_KEY": self.SENDGRID_API_KEY is not None,
            "SERPAPI_KEY": self.SERPAPI_KEY is not None,
            "GOOGLE_VISION_CREDENTIALS": self.GOOGLE_VISION_CREDENTIALS is not None,
        }
    
    def log_configuration(self) -> None:
        """Log current configuration (safe - no secrets)"""
        logger.info("=" * 70)
        logger.info("Configuration Loaded")
        logger.info("=" * 70)
        logger.info(f"Environment: {self.ENVIRONMENT.value}")
        logger.info(f"Debug Mode: {self.DEBUG}")
        logger.info(f"Host: {self.HOST}:{self.PORT}")
        logger.info(f"Database: {self.database_url_safe}")
        logger.info(f"Redis: {self.redis_url_safe}")
        logger.info(f"Claude Model: {self.CLAUDE_MODEL}")
        logger.info(f"Rate Limiting: {self.RATE_LIMIT_ENABLED} ({self.RATE_LIMIT_REQUESTS}/{self.RATE_LIMIT_WINDOW}s)")
        logger.info(f"Log Level: {self.LOG_LEVEL.value}")
        logger.info("")
        
        # API Keys Status
        logger.info("Required API Keys:")
        for key, status in self.get_required_api_keys().items():
            logger.info(f"  {key}: {'✓ Configured' if status else '✗ Missing'}")
        
        logger.info("")
        logger.info("Optional API Keys:")
        for key, status in self.get_optional_api_keys().items():
            logger.info(f"  {key}: {'✓ Configured' if status else '- Not set'}")
        
        logger.info("")
        logger.info("Feature Flags:")
        logger.info(f"  A/B Testing: {self.FEATURE_A_B_TESTING}")
        logger.info(f"  Competitor Analysis: {self.FEATURE_COMPETITOR_ANALYSIS}")
        logger.info(f"  Thumbnail Optimization: {self.FEATURE_THUMBNAIL_OPTIMIZATION}")
        logger.info(f"  Hashtag Optimization: {self.FEATURE_HASHTAG_OPTIMIZATION}")
        
        logger.info("=" * 70)
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors
        
        Returns:
            List of validation messages
        """
        warnings = []
        
        # Check production settings
        if self.is_production:
            if self.DEBUG:
                warnings.append("⚠️  DEBUG is enabled in production")
            
            if not self.YOUTUBE_API_KEY:
                warnings.append("⚠️  YOUTUBE_API_KEY not set in production")
            
            if not self.ANTHROPIC_API_KEY:
                warnings.append("⚠️  ANTHROPIC_API_KEY not set in production")
            
            # Check CORS
            for origin in self.CORS_ORIGINS:
                if origin.startswith("http://") and "localhost" not in origin:
                    warnings.append(f"⚠️  Non-HTTPS CORS origin: {origin}")
        
        # Check pool settings
        if self.DB_POOL_MAX < self.DB_POOL_MIN:
            warnings.append("⚠️  DB_POOL_MAX is less than DB_POOL_MIN")
        
        # Check thresholds
        if self.MEDIUM_VIEW_THRESHOLD <= self.LOW_VIEW_THRESHOLD:
            warnings.append("⚠️  MEDIUM_VIEW_THRESHOLD should be > LOW_VIEW_THRESHOLD")
        
        return warnings


# ============================================================================
# SETTINGS INSTANCE (CACHED)
# ============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Returns:
        Validated Settings instance
    
    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        settings_instance = Settings()
        
        # Log configuration (only in non-production or when explicitly enabled)
        if settings_instance.is_development or settings_instance.DEBUG:
            settings_instance.log_configuration()
        
        # Validate and show warnings
        warnings = settings_instance.validate_configuration()
        if warnings:
            logger.warning("Configuration warnings:")
            for warning in warnings:
                logger.warning(f"  {warning}")
        
        return settings_instance
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


# Create global settings instance
# Note: This will be created when first accessed, not at import time
def create_settings() -> Settings:
    """Create settings instance (call this explicitly in your app startup)"""
    return get_settings()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # This runs only when the file is executed directly
    print("Loading configuration...")
    
    try:
        settings = get_settings()
        settings.log_configuration()
        
        print("\nConfiguration loaded successfully!")
        print(f"Environment: {settings.ENVIRONMENT.value}")
        print(f"Database URL: {settings.database_url_safe}")
        print(f"Redis URL: {settings.redis_url_safe}")
        
        # Check for warnings
        warnings = settings.validate_configuration()
        if warnings:
            print("\n⚠️  Configuration Warnings:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print("\n✓ No configuration warnings")
            
    except Exception as e:
        print(f"\n❌ Configuration Error: {e}")
        import sys
        sys.exit(1)
