# utils/exceptions.py
"""
Custom exception classes for the application.

Provides domain-specific exceptions with proper error codes and messages.
"""

from typing import Optional, Dict, Any


class BaseAPIException(Exception):
    """Base exception class for all API exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ChannelNotFoundError(BaseAPIException):
    """Raised when a requested channel doesn't exist."""
    
    def __init__(self, message: str = "Channel not found", **kwargs):
        super().__init__(message, error_code="CHANNEL_NOT_FOUND", **kwargs)


class OptimizationNotFoundError(BaseAPIException):
    """Raised when a requested optimization doesn't exist."""
    
    def __init__(self, message: str = "Optimization not found", **kwargs):
        super().__init__(message, error_code="OPTIMIZATION_NOT_FOUND", **kwargs)


class OptimizationCreationError(BaseAPIException):
    """Raised when optimization creation fails."""
    
    def __init__(self, message: str = "Failed to create optimization", **kwargs):
        super().__init__(message, error_code="OPTIMIZATION_CREATION_FAILED", **kwargs)


class UnauthorizedError(BaseAPIException):
    """Raised when user lacks permissions for the requested action."""
    
    def __init__(self, message: str = "Unauthorized access", **kwargs):
        super().__init__(message, error_code="UNAUTHORIZED", **kwargs)


class ValidationError(BaseAPIException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str = "Validation error", **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)


class RateLimitExceeded(BaseAPIException):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if retry_after:
            details["retry_after"] = retry_after
        kwargs["details"] = details
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED", **kwargs)


class DatabaseError(BaseAPIException):
    """Raised when database operations fail."""
    
    def __init__(self, message: str = "Database error", **kwargs):
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)


class YouTubeAPIError(BaseAPIException):
    """Raised when YouTube API operations fail."""
    
    def __init__(
        self,
        message: str = "YouTube API error",
        youtube_error_code: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if youtube_error_code:
            details["youtube_error_code"] = youtube_error_code
        kwargs["details"] = details
        super().__init__(message, error_code="YOUTUBE_API_ERROR", **kwargs)


class ServiceUnavailableError(BaseAPIException):
    """Raised when external service is unavailable."""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        service_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if service_name:
            details["service"] = service_name
        kwargs["details"] = details
        super().__init__(message, error_code="SERVICE_UNAVAILABLE", **kwargs)
