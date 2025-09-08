"""
Configuration Module

Application configuration using Pydantic settings for environment variable loading,
validation, and type-safe configuration management with database and OAuth settings.
"""

from dotenv import load_dotenv
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings configuration with database, OAuth, and API key management."""

    # Application settings
    app_name: str = Field(default="YouTube Optimizer")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")

    # Database configuration
    postgres_user: str = Field(default="")
    postgres_password: SecretStr = Field(default="")
    postgres_db: str = Field(default="")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    database_url: str | None = None

    # OAuth and Google API configuration
    client_secret_file: str | None = Field(default=None)

    # URL configuration
    frontend_url: str = Field(default="http://localhost:3000")
    backend_url: str = Field(default="http://localhost:8080")
    external_home_url: str = Field(default="http://localhost:3000")

    # API keys
    anthropic_api_key: SecretStr | None = None

    # YouTube API scopes for OAuth permissions
    youtube_api_scopes: list[str] = Field(
        default=[
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube",
            "https://www.googleapis.com/auth/youtube.readonly",
            "https://www.googleapis.com/auth/youtube.force-ssl",
            "https://www.googleapis.com/auth/yt-analytics.readonly",
            "https://www.googleapis.com/auth/youtubepartner",
            "https://www.googleapis.com/auth/youtubepartner-channel-audit",
            "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
        ]
    )

    @property
    def redirect_uri(self) -> str:
        """Dynamic redirect URI based on backend URL for OAuth authentication."""
        return f"{self.backend_url}/auth/callback"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def resolved_database_url(self) -> str:
        """Return complete PostgreSQL connection string from settings or components."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance for performance."""
    return Settings()
