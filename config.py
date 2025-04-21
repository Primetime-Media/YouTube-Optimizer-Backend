import os
from dotenv import load_dotenv
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    app_name: str = Field(default="YouTube Optimizer")
    debug: bool = Field(default=False)
    database_url: str = Field(default="")
    client_secret_file: str = Field(
        default="client_secret_941974948417-la4udombfq14du8vea6b8jqmo6d8nbv8.apps.googleusercontent.com.json"
    )
    frontend_url: str = Field(default="http://localhost:3000")
    redirect_uri: str = Field(default="http://localhost:8080/auth/callback")
    
    # Database settings
    postgres_user: str = Field(default="")
    postgres_password: str = Field(default="")
    postgres_db: str = Field(default="")
    
    # API keys
    anthropic_api_key: str = Field(default="")
    
    # YouTube API settings
    youtube_api_scopes: list[str] = Field(
        default=[
            # Basic user info
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            
            # Owner scopes - these ensure user quota is used
            "https://www.googleapis.com/auth/youtube.channel-memberships.creator",
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube",  # Full access scope
            "https://www.googleapis.com/auth/youtube.readonly",
            "https://www.googleapis.com/auth/youtube.force-ssl",
            "https://www.googleapis.com/auth/yt-analytics.readonly",
            
            # Partner and advanced content owner scopes
            "https://www.googleapis.com/auth/youtubepartner",  # Partner scope if available
            "https://www.googleapis.com/auth/youtubepartner-channel-audit",
            
            # Analytics monetary scope
            "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"
            
            # The following scopes are invalid or restricted:
            # "https://www.googleapis.com/auth/youtube.content.id"
            # "https://www.googleapis.com/auth/youtube.content.administration"
            # "https://www.googleapis.com/auth/youtube.reporting.readonly"
        ]
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Allow extra fields in environment
    }
        
@lru_cache()
def get_settings():
    """
    Return application settings as a cached instance.
    Uses LRU cache for performance.
    """
    return Settings()