"""
Configuration Module

This module defines the application configuration using Pydantic settings
for the YouTube Optimizer platform. It handles environment variable loading,
validation, and provides type-safe configuration management.

Key functionalities:
- Environment variable loading and validation
- Type-safe configuration with Pydantic
- Database connection string construction
- OAuth and API key management
- Environment-specific settings (development/production)
- Cached settings for performance optimization

The configuration supports both explicit database URLs and constructed URLs
from individual components, providing flexibility for different deployment scenarios.

Author: YouTube Optimizer Team
Version: 1.0.0
"""

import os
from dotenv import load_dotenv
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings configuration using Pydantic BaseSettings.
    
    This class defines all configuration parameters for the YouTube Optimizer
    application, including database settings, API keys, OAuth configuration,
    and environment-specific settings.
    
    All settings can be overridden via environment variables or .env file.
    Sensitive data like passwords and API keys are handled as SecretStr for
    security.
    """

    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    
    app_name: str = Field(default="YouTube Optimizer")  # Application display name
    debug: bool = Field(default=False)                  # Debug mode flag
    environment: str = Field(default="development")     # Environment: development, production

    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    
    postgres_user: str = Field(default="")             # PostgreSQL username
    postgres_password: SecretStr = Field(default="")   # PostgreSQL password (sensitive)
    postgres_db: str = Field(default="")               # PostgreSQL database name
    postgres_host: str = Field(default="localhost")    # PostgreSQL host
    postgres_port: int = Field(default=5432)           # PostgreSQL port
    
    database_url: str | None = None                    # Optional explicit database URL

    # =============================================================================
    # OAUTH AND GOOGLE API CONFIGURATION
    # =============================================================================
    
    client_secret_file: str | None = Field(default=None)  # Google OAuth client secret file path

    # =============================================================================
    # URL CONFIGURATION
    # =============================================================================
    
    frontend_url: str = Field(default="http://localhost:3000")    # Frontend application URL
    backend_url: str = Field(default="http://localhost:8080")     # Backend API URL
    external_home_url: str = Field(default="http://localhost:3000")  # External homepage URL

    # =============================================================================
    # API KEYS AND EXTERNAL SERVICES
    # =============================================================================
    
    anthropic_api_key: SecretStr | None = None  # Anthropic Claude API key (sensitive)

    # =============================================================================
    # YOUTUBE API SCOPES
    # =============================================================================
    
    # Required OAuth scopes for YouTube API access
    # These scopes define what permissions the application requests from users
    youtube_api_scopes: list[str] = Field(
        default=[
            "https://www.googleapis.com/auth/userinfo.email",           # User email access
            "https://www.googleapis.com/auth/userinfo.profile",         # User profile access
            "https://www.googleapis.com/auth/youtube.upload",           # Upload videos
            "https://www.googleapis.com/auth/youtube",                  # Full YouTube access
            "https://www.googleapis.com/auth/youtube.readonly",         # Read-only YouTube access
            "https://www.googleapis.com/auth/youtube.force-ssl",        # Force SSL for YouTube
            "https://www.googleapis.com/auth/yt-analytics.readonly",    # YouTube Analytics read-only
            "https://www.googleapis.com/auth/youtubepartner",           # YouTube Partner access
            "https://www.googleapis.com/auth/youtubepartner-channel-audit",  # Channel audit access
            "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",  # Monetary analytics
        ]
    )

    # =============================================================================
    # COMPUTED PROPERTIES
    # =============================================================================

    @property
    def redirect_uri(self) -> str:
        """
        Dynamic redirect URI based on backend URL.
        
        This property constructs the OAuth redirect URI dynamically based on
        the configured backend URL, ensuring consistency across environments.
        
        Returns:
            str: Complete OAuth redirect URI for Google authentication
        """
        return f"{self.backend_url}/auth/callback"

    @property
    def is_production(self) -> bool:
        """
        Check if running in production environment.
        
        This property determines if the application is running in production
        mode, which affects security settings, logging levels, and feature flags.
        
        Returns:
            bool: True if running in production, False otherwise
        """
        return self.environment.lower() == "production"

    @property
    def resolved_database_url(self) -> str:
        """
        Return a fully constructed database URL.
        
        This property provides the complete PostgreSQL connection string,
        either from the explicit database_url setting or constructed from
        individual database components.
        
        Returns:
            str: Complete PostgreSQL connection string
        """
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # =============================================================================
    # PYDANTIC CONFIGURATION
    # =============================================================================
    
    model_config = {
        "env_file": ".env",           # Load from .env file
        "env_file_encoding": "utf-8", # UTF-8 encoding for .env file
        "case_sensitive": False,      # Case-insensitive environment variable matching
        "extra": "ignore",            # Ignore extra fields not defined in model
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Return application settings as a cached instance.
    
    This function uses LRU caching to ensure settings are loaded only once
    per application instance, improving performance and consistency.
    
    Returns:
        Settings: Cached application settings instance
    """
    return Settings()
