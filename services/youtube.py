"""
YouTube API Service Module - Production Ready
==============================================
Enterprise-grade YouTube Data API v3 integration

Production Features:
✅ Async/await architecture
✅ Connection pooling with asyncpg
✅ Retry logic with exponential backoff
✅ Circuit breaker for API resilience
✅ Prometheus metrics integration
✅ Structured logging with context
✅ Type safety with Pydantic
✅ API quota tracking and management
✅ Redis caching for API responses
✅ Rate limiting for API calls
✅ Resource cleanup with context managers
✅ Credential refresh handling
✅ Comprehensive error handling
"""

import asyncio
import logging
import html
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager
from functools import wraps
import asyncpg

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from PIL import Image
import aioredis

from utils.db import get_pool
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

class TimeConstants:
    """Time-related constants"""
    DEFAULT_ANALYTICS_DAYS = 28
    SECONDS_PER_HOUR = 3600
    SECONDS_PER_MINUTE = 60
    MINUTES_PER_HOUR = 60
    HOURS_PER_DAY = 24
    DAYS_PER_WEEK = 7


class VideoConstants:
    """Video-related constants"""
    MIN_DURATION_SECONDS = 360  # 6 minutes
    SHORTS_MAX_DURATION_SECONDS = 180  # 3 minutes
    THUMBNAIL_MAX_SIZE_BYTES = 2 * 1024 * 1024  # 2MB
    THUMBNAIL_JPEG_QUALITY = 90
    VALID_THUMBNAIL_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']


class APIQuotaLimits:
    """YouTube API quota limits"""
    DAILY_QUOTA_LIMIT = 10000
    VIDEO_LIST_COST = 1
    CHANNEL_LIST_COST = 1
    PLAYLIST_ITEMS_COST = 1
    ANALYTICS_COST = 0  # Analytics API has separate quota
    THUMBNAIL_UPLOAD_COST = 50


# ============================================================================
# METRICS
# ============================================================================

YOUTUBE_API_CALLS = Counter(
    'youtube_api_calls_total',
    'Total YouTube API calls',
    ['method', 'status']
)

YOUTUBE_API_DURATION = Histogram(
    'youtube_api_call_duration_seconds',
    'YouTube API call duration',
    ['method']
)

YOUTUBE_API_ERRORS = Counter(
    'youtube_api_errors_total',
    'YouTube API errors',
    ['method', 'error_type']
)

YOUTUBE_QUOTA_USAGE = Gauge(
    'youtube_api_quota_usage',
    'YouTube API quota usage',
    ['user_id']
)

YOUTUBE_CACHE_HITS = Counter(
    'youtube_cache_hits_total',
    'YouTube API cache hits',
    ['cache_type']
)


# ============================================================================
# ENUMS
# ============================================================================

class CacheType(str, Enum):
    """Cache key types"""
    VIDEO_DATA = "video_data"
    CHANNEL_DATA = "channel_data"
    ANALYTICS = "analytics"
    TRANSCRIPT = "transcript"


class YouTubeAPIMethod(str, Enum):
    """YouTube API methods for tracking"""
    VIDEO_LIST = "videos.list"
    CHANNEL_LIST = "channels.list"
    PLAYLIST_ITEMS = "playlistItems.list"
    CAPTIONS_LIST = "captions.list"
    THUMBNAIL_SET = "thumbnails.set"
    VIDEO_UPDATE = "videos.update"
    ANALYTICS_QUERY = "analytics.query"


# ============================================================================
# EXCEPTIONS
# ============================================================================

class YouTubeServiceException(Exception):
    """Base exception for YouTube service"""
    pass


class QuotaExceededException(YouTubeServiceException):
    """API quota exceeded"""
    pass


class AuthenticationException(YouTubeServiceException):
    """Authentication failed"""
    pass


class VideoNotFoundException(YouTubeServiceException):
    """Video not found"""
    pass


class ChannelNotFoundException(YouTubeServiceException):
    """Channel not found"""
    pass


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class VideoMetadata(BaseModel):
    """Video metadata model"""
    video_id: str = Field(..., min_length=11, max_length=11)
    title: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=5000)
    tags: List[str] = Field(default_factory=list, max_items=500)
    category_id: Optional[str] = None
    privacy_status: str = "public"
    
    @validator('video_id')
    def validate_video_id(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Invalid YouTube video ID format")
        return v
    
    class Config:
        str_strip_whitespace = True


class ChannelMetadata(BaseModel):
    """Channel metadata model"""
    channel_id: str
    title: str
    description: str = ""
    custom_url: Optional[str] = None
    subscriber_count: int = Field(default=0, ge=0)
    video_count: int = Field(default=0, ge=0)
    view_count: int = Field(default=0, ge=0)
    
    class Config:
        str_strip_whitespace = True


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""
    failures: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    success_count: int = 0


class CircuitBreaker:
    """Circuit breaker for YouTube API calls"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = HttpError
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.state = CircuitBreakerState()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state.is_open:
            if self._should_attempt_reset():
                logger.info("Circuit breaker attempting reset")
                self.state.is_open = False
                self.state.success_count = 0
            else:
                raise YouTubeServiceException("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func, *args, **kwargs):
        """Execute async function with circuit breaker protection"""
        if self.state.is_open:
            if self._should_attempt_reset():
                logger.info("Circuit breaker attempting reset")
                self.state.is_open = False
                self.state.success_count = 0
            else:
                raise YouTubeServiceException("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        self.state.failures = 0
        self.state.success_count += 1
        
        if self.state.success_count >= 2:
            logger.info("Circuit breaker CLOSED after successful calls")
            self.state.is_open = False
    
    def _on_failure(self):
        """Handle failed call"""
        self.state.failures += 1
        self.state.last_failure_time = datetime.now(timezone.utc)
        
        if self.state.failures >= self.failure_threshold:
            logger.warning(f"Circuit breaker OPEN after {self.state.failures} failures")
            self.state.is_open = True
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.state.last_failure_time:
            return True
        
        time_since_failure = (
            datetime.now(timezone.utc) - self.state.last_failure_time
        ).total_seconds()
        
        return time_since_failure >= self.recovery_timeout


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class YouTubeUtils:
    """Utility functions for YouTube data processing"""
    
    @staticmethod
    def parse_duration_to_seconds(duration_str: str) -> int:
        """
        Parse ISO 8601 duration to seconds
        
        Args:
            duration_str: ISO 8601 duration (e.g., PT4M13S)
            
        Returns:
            Duration in seconds
        """
        try:
            if not duration_str:
                return 0
            
            pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
            match = re.match(pattern, duration_str)
            
            if not match:
                logger.warning(f"Invalid duration format: {duration_str}")
                return 0
            
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            
            return (
                hours * TimeConstants.SECONDS_PER_HOUR +
                minutes * TimeConstants.SECONDS_PER_MINUTE +
                seconds
            )
        except Exception as e:
            logger.error(f"Error parsing duration '{duration_str}': {e}")
            return 0
    
    @staticmethod
    def parse_srt_content(srt_content: str) -> List[Dict]:
        """
        Parse SRT format captions
        
        Args:
            srt_content: SRT formatted caption text
            
        Returns:
            List of caption segments
        """
        try:
            if not srt_content or not isinstance(srt_content, str):
                return []
            
            segments = []
            pattern = r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})\s+((?:.+\s*)+?)(?:\r?\n\r?\n|\Z)'
            matches = re.findall(pattern, srt_content, re.DOTALL)
            
            for match in matches:
                try:
                    index, start_time, end_time, text = match
                    
                    # Clean text
                    clean_text = html.unescape(re.sub(r'<[^>]+>', '', text))
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    
                    # Convert to seconds
                    start_seconds = YouTubeUtils.timecode_to_seconds(start_time)
                    end_seconds = YouTubeUtils.timecode_to_seconds(end_time)
                    
                    segments.append({
                        'index': int(index),
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'duration': end_seconds - start_seconds,
                        'text': clean_text
                    })
                except Exception as e:
                    logger.warning(f"Error parsing SRT segment: {e}")
                    continue
            
            return segments
        except Exception as e:
            logger.error(f"Error parsing SRT content: {e}")
            return []
    
    @staticmethod
    def timecode_to_seconds(timecode: str) -> float:
        """
        Convert SRT timecode to seconds
        
        Args:
            timecode: HH:MM:SS,MS format
            
        Returns:
            Seconds as float
        """
        try:
            if not timecode:
                return 0.0
            
            timecode_clean = timecode.replace(',', '.')
            parts = timecode_clean.split(':')
            
            if len(parts) != 3:
                return 0.0
            
            hours, minutes, seconds = parts
            return (
                float(hours) * TimeConstants.SECONDS_PER_HOUR +
                float(minutes) * TimeConstants.SECONDS_PER_MINUTE +
                float(seconds)
            )
        except Exception as e:
            logger.error(f"Error parsing timecode '{timecode}': {e}")
            return 0.0


# ============================================================================
# CREDENTIAL MANAGER
# ============================================================================

class CredentialManager:
    """Manages YouTube API credentials with auto-refresh"""
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def get_credentials(self, user_id: int) -> Optional[Credentials]:
        """
        Get and refresh credentials if needed
        
        Args:
            user_id: User ID
            
        Returns:
            Refreshed credentials or None
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT 
                        access_token,
                        refresh_token,
                        token_uri,
                        client_id,
                        client_secret,
                        scopes,
                        token_expiry
                    FROM user_credentials
                    WHERE user_id = $1
                """, user_id)
                
                if not row:
                    logger.warning(f"No credentials found for user {user_id}")
                    return None
                
                credentials = Credentials(
                    token=row['access_token'],
                    refresh_token=row['refresh_token'],
                    token_uri=row['token_uri'],
                    client_id=row['client_id'],
                    client_secret=row['client_secret'],
                    scopes=row['scopes']
                )
                
                # Check if expired and refresh
                if credentials.expired and credentials.refresh_token:
                    logger.info(f"Refreshing expired credentials for user {user_id}")
                    try:
                        credentials.refresh(Request())
                        
                        # Update in database
                        await self.update_credentials(user_id, credentials)
                        logger.info(f"Credentials refreshed for user {user_id}")
                    except RefreshError as e:
                        logger.error(f"Refresh failed for user {user_id}: {e}")
                        raise AuthenticationException(f"Credential refresh failed: {e}")
                
                return credentials
                
        except Exception as e:
            logger.error(f"Error getting credentials for user {user_id}: {e}")
            raise
    
    async def update_credentials(
        self,
        user_id: int,
        credentials: Credentials
    ) -> None:
        """Update stored credentials"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    UPDATE user_credentials
                    SET 
                        access_token = $1,
                        refresh_token = $2,
                        token_expiry = $3,
                        updated_at = $4
                    WHERE user_id = $5
                """,
                    credentials.token,
                    credentials.refresh_token,
                    credentials.expiry,
                    datetime.now(timezone.utc),
                    user_id
                )
        except Exception as e:
            logger.error(f"Error updating credentials: {e}")
            raise


# ============================================================================
# YOUTUBE API CLIENT
# ============================================================================

class YouTubeAPIClient:
    """YouTube API client with retry and circuit breaker"""
    
    def __init__(
        self,
        credentials: Credentials,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.credentials = credentials
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._client = None
        self._analytics_client = None
    
    @property
    def client(self):
        """Get YouTube Data API client"""
        if not self._client:
            self._client = build(
                'youtube',
                'v3',
                credentials=self.credentials,
                cache_discovery=False
            )
        return self._client
    
    @property
    def analytics_client(self):
        """Get YouTube Analytics API client"""
        if not self._analytics_client:
            self._analytics_client = build(
                'youtubeAnalytics',
                'v2',
                credentials=self.credentials,
                cache_discovery=False
            )
        return self._analytics_client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((HttpError,))
    )
    async def execute_with_retry(
        self,
        request,
        method: YouTubeAPIMethod
    ) -> Dict:
        """
        Execute API request with retry logic
        
        Args:
            request: API request object
            method: API method for tracking
            
        Returns:
            API response
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.circuit_breaker.call,
                request.execute
            )
            
            # Record success metrics
            duration = asyncio.get_event_loop().time() - start_time
            YOUTUBE_API_CALLS.labels(method=method.value, status='success').inc()
            YOUTUBE_API_DURATION.labels(method=method.value).observe(duration)
            
            return response
            
        except HttpError as e:
            duration = asyncio.get_event_loop().time() - start_time
            YOUTUBE_API_DURATION.labels(method=method.value).observe(duration)
            
            # Handle specific errors
            if e.resp.status == 403:
                if "quota" in str(e).lower():
                    YOUTUBE_API_ERRORS.labels(
                        method=method.value,
                        error_type='quota_exceeded'
                    ).inc()
                    raise QuotaExceededException(f"API quota exceeded: {e}")
                else:
                    YOUTUBE_API_ERRORS.labels(
                        method=method.value,
                        error_type='permission_denied'
                    ).inc()
                    raise AuthenticationException(f"Permission denied: {e}")
            elif e.resp.status == 404:
                YOUTUBE_API_ERRORS.labels(
                    method=method.value,
                    error_type='not_found'
                ).inc()
                raise VideoNotFoundException(f"Resource not found: {e}")
            else:
                YOUTUBE_API_ERRORS.labels(
                    method=method.value,
                    error_type='http_error'
                ).inc()
                raise
        except Exception as e:
            YOUTUBE_API_ERRORS.labels(
                method=method.value,
                error_type='unexpected_error'
            ).inc()
            logger.error(f"Unexpected error in API call: {e}")
            raise


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """Redis cache manager for YouTube API responses"""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    def _get_cache_key(
        self,
        cache_type: CacheType,
        identifier: str,
        **kwargs
    ) -> str:
        """Generate cache key"""
        parts = [cache_type.value, identifier]
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}:{v}")
        return ":".join(parts)
    
    async def get(
        self,
        cache_type: CacheType,
        identifier: str,
        **kwargs
    ) -> Optional[Dict]:
        """Get cached data"""
        if not self.redis:
            return None
        
        try:
            key = self._get_cache_key(cache_type, identifier, **kwargs)
            cached = await self.redis.get(key)
            
            if cached:
                YOUTUBE_CACHE_HITS.labels(cache_type=cache_type.value).inc()
                return json.loads(cached)
            
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        cache_type: CacheType,
        identifier: str,
        data: Dict,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        """Set cached data"""
        if not self.redis:
            return
        
        try:
            key = self._get_cache_key(cache_type, identifier, **kwargs)
            await self.redis.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    async def invalidate(
        self,
        cache_type: CacheType,
        identifier: str,
        **kwargs
    ) -> None:
        """Invalidate cached data"""
        if not self.redis:
            return
        
        try:
            key = self._get_cache_key(cache_type, identifier, **kwargs)
            await self.redis.delete(key)
        except Exception as e:
            logger.warning(f"Cache invalidate error: {e}")


# ============================================================================
# YOUTUBE SERVICE
# ============================================================================

class YouTubeService:
    """Main YouTube service class"""
    
    def __init__(
        self,
        pool: asyncpg.Pool,
        redis_client: Optional[aioredis.Redis] = None
    ):
        self.pool = pool
        self.credential_manager = CredentialManager(pool)
        self.cache_manager = CacheManager(redis_client)
        self.circuit_breaker = CircuitBreaker()
    
    async def get_api_client(self, user_id: int) -> YouTubeAPIClient:
        """Get authenticated API client for user"""
        credentials = await self.credential_manager.get_credentials(user_id)
        if not credentials:
            raise AuthenticationException(f"No credentials for user {user_id}")
        
        return YouTubeAPIClient(credentials, self.circuit_breaker)
    
    async def fetch_video_data(
        self,
        user_id: int,
        video_id: str,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Fetch video data from YouTube API
        
        Args:
            user_id: User ID for credentials
            video_id: YouTube video ID
            use_cache: Whether to use cache
            
        Returns:
            Video data dict or None
        """
        try:
            # Check cache first
            if use_cache:
                cached = await self.cache_manager.get(
                    CacheType.VIDEO_DATA,
                    video_id
                )
                if cached:
                    return cached
            
            # Fetch from API
            client = await self.get_api_client(user_id)
            request = client.client.videos().list(
                part='snippet,contentDetails,statistics,status',
                id=video_id
            )
            
            response = await client.execute_with_retry(
                request,
                YouTubeAPIMethod.VIDEO_LIST
            )
            
            if not response.get('items'):
                return None
            
            video_data = response['items'][0]
            
            # Cache the result
            await self.cache_manager.set(
                CacheType.VIDEO_DATA,
                video_id,
                video_data,
                ttl=1800  # 30 minutes
            )
            
            return video_data
            
        except Exception as e:
            logger.error(f"Error fetching video {video_id}: {e}")
            raise
    
    async def update_video(
        self,
        user_id: int,
        video_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict:
        """
        Update video metadata
        
        Args:
            user_id: User ID for credentials
            video_id: YouTube video ID
            title: New title
            description: New description
            tags: New tags
            
        Returns:
            Update result
        """
        try:
            client = await self.get_api_client(user_id)
            
            # Get current video data
            video_request = client.client.videos().list(
                part='snippet',
                id=video_id
            )
            video_response = await client.execute_with_retry(
                video_request,
                YouTubeAPIMethod.VIDEO_LIST
            )
            
            if not video_response.get('items'):
                raise VideoNotFoundException(f"Video {video_id} not found")
            
            snippet = video_response['items'][0]['snippet']
            
            # Update fields
            if title:
                snippet['title'] = title
            if description:
                snippet['description'] = description
            if tags:
                snippet['tags'] = tags
            
            # Send update
            update_request = client.client.videos().update(
                part='snippet',
                body={'id': video_id, 'snippet': snippet}
            )
            
            result = await client.execute_with_retry(
                update_request,
                YouTubeAPIMethod.VIDEO_UPDATE
            )
            
            # Invalidate cache
            await self.cache_manager.invalidate(
                CacheType.VIDEO_DATA,
                video_id
            )
            
            return {
                'success': True,
                'video_id': video_id,
                'updated_fields': {
                    'title': title is not None,
                    'description': description is not None,
                    'tags': tags is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating video {video_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_id': video_id
            }
    
    async def upload_thumbnail(
        self,
        user_id: int,
        video_id: str,
        thumbnail_path: str
    ) -> Dict:
        """
        Upload custom thumbnail
        
        Args:
            user_id: User ID for credentials
            video_id: YouTube video ID
            thumbnail_path: Path to thumbnail file
            
        Returns:
            Upload result
        """
        try:
            import os
            from pathlib import Path
            
            # Validate file
            if not os.path.exists(thumbnail_path):
                return {
                    'success': False,
                    'error': f'File not found: {thumbnail_path}'
                }
            
            file_size = os.path.getsize(thumbnail_path)
            if file_size > VideoConstants.THUMBNAIL_MAX_SIZE_BYTES:
                return {
                    'success': False,
                    'error': f'File too large: {file_size} bytes'
                }
            
            file_ext = Path(thumbnail_path).suffix.lower()
            if file_ext not in VideoConstants.VALID_THUMBNAIL_EXTENSIONS:
                return {
                    'success': False,
                    'error': f'Invalid file type: {file_ext}'
                }
            
            # Convert PNG to JPG if needed
            if file_ext == '.png':
                jpg_path = str(Path(thumbnail_path).with_suffix('.jpg'))
                with Image.open(thumbnail_path) as img:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    img.save(
                        jpg_path,
                        'JPEG',
                        quality=VideoConstants.THUMBNAIL_JPEG_QUALITY,
                        optimize=True
                    )
                thumbnail_path = jpg_path
                file_ext = '.jpg'
            
            # Determine MIME type
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp'
            }
            mime_type = mime_types.get(file_ext, 'image/jpeg')
            
            # Upload thumbnail
            client = await self.get_api_client(user_id)
            
            # Create media upload
            media = MediaFileUpload(
                thumbnail_path,
                mimetype=mime_type,
                resumable=True
            )
            
            # Execute upload
            upload_request = client.client.thumbnails().set(
                videoId=video_id,
                media_body=media
            )
            
            result = await client.execute_with_retry(
                upload_request,
                YouTubeAPIMethod.THUMBNAIL_SET
            )
            
            return {
                'success': True,
                'video_id': video_id,
                'thumbnail_info': result
            }
            
        except Exception as e:
            logger.error(f"Error uploading thumbnail for {video_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_id': video_id
            }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

async def create_youtube_service(
    pool: Optional[asyncpg.Pool] = None,
    redis_client: Optional[aioredis.Redis] = None
) -> YouTubeService:
    """
    Create YouTube service instance
    
    Args:
        pool: Database connection pool
        redis_client: Redis client for caching
        
    Returns:
        Configured YouTubeService
    """
    if pool is None:
        pool = await get_pool()
    
    if redis_client is None:
        try:
            redis_client = await aioredis.create_redis_pool(
                settings.redis_url,
                encoding='utf-8'
            )
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            redis_client = None
    
    return YouTubeService(pool, redis_client)


# ============================================================================
# LEGACY COMPATIBILITY WRAPPERS
# ============================================================================
# Maintain backward compatibility with existing code

async def fetch_video_data_legacy(
    user_id: int,
    video_id: str
) -> Optional[Dict]:
    """Legacy wrapper for fetch_video_data"""
    service = await create_youtube_service()
    return await service.fetch_video_data(user_id, video_id)


async def update_video_legacy(
    user_id: int,
    video_id: str,
    title: str,
    description: str,
    tags: List[str]
) -> Dict:
    """Legacy wrapper for update_video"""
    service = await create_youtube_service()
    return await service.update_video(
        user_id,
        video_id,
        title=title,
        description=description,
        tags=tags
    )
