# utils/rate_limiter.py
"""
Rate limiting utilities using Redis for distributed rate limiting.

Provides decorator-based rate limiting for API endpoints.
"""

from functools import wraps
from typing import Optional, Callable
from fastapi import Request, HTTPException, status
from datetime import datetime, timedelta
import redis
import hashlib
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Distributed rate limiter using Redis.
    
    Implements sliding window rate limiting to prevent API abuse.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize rate limiter.
        
        Args:
            redis_url: Redis connection URL (default from config)
            enabled: Whether rate limiting is enabled
        """
        self.enabled = enabled
        
        if self.enabled and redis_url:
            try:
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Rate limiter initialized with Redis")
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis for rate limiting: {e}. "
                    "Rate limiting will be disabled."
                )
                self.enabled = False
                self.redis_client = None
        else:
            self.redis_client = None
            if not self.enabled:
                logger.info("Rate limiting is disabled")
    
    def _get_identifier(self, request: Request) -> str:
        """
        Get unique identifier for the request (user ID or IP).
        
        Args:
            request: FastAPI request object
            
        Returns:
            str: Unique identifier for rate limiting
        """
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.get('id', 'anonymous')}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            ip = forwarded_for.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"
    
    def _parse_rate(self, rate: str) -> tuple[int, int]:
        """
        Parse rate limit string (e.g., "10/minute" or "100/hour").
        
        Args:
            rate: Rate limit string
            
        Returns:
            tuple: (limit, window_seconds)
        """
        parts = rate.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid rate format: {rate}")
        
        limit = int(parts[0])
        period = parts[1].lower()
        
        period_seconds = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
        
        if period not in period_seconds:
            raise ValueError(f"Invalid period: {period}")
        
        return limit, period_seconds[period]
    
    def _check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ) -> tuple[bool, dict]:
        """
        Check if request is within rate limit using sliding window.
        
        Args:
            key: Redis key for this limit
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            tuple: (is_allowed, metadata)
        """
        if not self.enabled or not self.redis_client:
            return True, {}
        
        try:
            now = datetime.utcnow().timestamp()
            window_start = now - window
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiry
            pipe.expire(key, window + 1)
            
            # Execute pipeline
            results = pipe.execute()
            
            current_count = results[1]
            is_allowed = current_count < limit
            
            # Calculate reset time
            if current_count > 0:
                oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    reset_time = int(oldest[0][1] + window)
                else:
                    reset_time = int(now + window)
            else:
                reset_time = int(now + window)
            
            metadata = {
                "limit": limit,
                "remaining": max(0, limit - current_count - 1),
                "reset": reset_time,
                "retry_after": int(reset_time - now) if not is_allowed else None
            }
            
            return is_allowed, metadata
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if Redis is down
            return True, {}
    
    def limit(self, rate: str):
        """
        Decorator to apply rate limiting to an endpoint.
        
        Args:
            rate: Rate limit string (e.g., "10/minute")
            
        Example:
            @router.get("/api/resource")
            @rate_limiter.limit("10/minute")
            async def get_resource():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request from args/kwargs
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                
                if not request:
                    request = kwargs.get("request")
                
                if not request or not self.enabled:
                    # No rate limiting if request not found or disabled
                    return await func(*args, **kwargs)
                
                # Parse rate limit
                limit, window = self._parse_rate(rate)
                
                # Get identifier
                identifier = self._get_identifier(request)
                
                # Create unique key for this endpoint and identifier
                endpoint = f"{request.method}:{request.url.path}"
                key_parts = f"{endpoint}:{identifier}"
                key = f"ratelimit:{hashlib.md5(key_parts.encode()).hexdigest()}"
                
                # Check rate limit
                is_allowed, metadata = self._check_rate_limit(key, limit, window)
                
                # Add rate limit headers to response
                if hasattr(request.state, "rate_limit_metadata"):
                    request.state.rate_limit_metadata = metadata
                
                if not is_allowed:
                    logger.warning(
                        f"Rate limit exceeded for {identifier} on {endpoint}",
                        extra={
                            "identifier": identifier,
                            "endpoint": endpoint,
                            "limit": limit,
                            "window": window
                        }
                    )
                    
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail={
                            "error": "rate_limit_exceeded",
                            "message": f"Rate limit exceeded. Try again in {metadata.get('retry_after', window)} seconds.",
                            "limit": metadata.get("limit"),
                            "retry_after": metadata.get("retry_after")
                        },
                        headers={
                            "X-RateLimit-Limit": str(metadata.get("limit", limit)),
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(metadata.get("reset", "")),
                            "Retry-After": str(metadata.get("retry_after", window))
                        }
                    )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator


# Middleware to add rate limit headers to all responses
async def add_rate_limit_headers(request: Request, call_next):
    """
    Middleware to add rate limit headers to responses.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/handler
    """
    response = await call_next(request)
    
    # Add rate limit headers if metadata is available
    if hasattr(request.state, "rate_limit_metadata"):
        metadata = request.state.rate_limit_metadata
        if metadata:
            response.headers["X-RateLimit-Limit"] = str(metadata.get("limit", ""))
            response.headers["X-RateLimit-Remaining"] = str(metadata.get("remaining", ""))
            response.headers["X-RateLimit-Reset"] = str(metadata.get("reset", ""))
    
    return response
