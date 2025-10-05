# services/youtube.py
"""
Production-Ready YouTube API Service
Handles YouTube Data API v3 and Analytics API v2 integration with enterprise-grade reliability
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import logging
import time
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from utils.db import get_connection
from config import settings

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

# API Configuration
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
ANALYTICS_API_SERVICE_NAME = "youtubeAnalytics"
ANALYTICS_API_VERSION = "v2"

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0

# Rate Limiting
API_QUOTA_LIMIT = 10000  # Daily quota
QUOTA_COSTS = {
    'videos.list': 1,
    'channels.list': 1,
    'search.list': 100,
    'videos.update': 50,
    'thumbnails.set': 50,
    'captions.list': 50,
}

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=5)


# ============================================================================
# DECORATORS FOR RELIABILITY
# ============================================================================

def retry_on_error(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """Retry decorator for handling transient API errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except HttpError as e:
                    last_exception = e
                    error_code = e.resp.status
                    
                    # Don't retry on client errors (4xx except 429)
                    if 400 <= error_code < 500 and error_code != 429:
                        logger.error(f"Client error in {func.__name__}: {e}")
                        raise
                    
                    # Log and retry on server errors or rate limits
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= BACKOFF_MULTIPLIER
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                
                except Exception as e:
                    last_exception = e
                    logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                    raise
            
            raise last_exception
        
        return wrapper
    return decorator


def log_quota_usage(operation: str, cost: int = 1):
    """Decorator to log API quota usage"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"API Call: {operation} (Quota cost: {cost})")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# CREDENTIAL MANAGEMENT
# ============================================================================

class YouTubeCredentialManager:
    """Manages OAuth2 credentials with refresh capabilities"""
    
    @staticmethod
    def get_credentials(user_id: int) -> Optional[Credentials]:
        """
        Retrieve and refresh credentials for a user
        
        Args:
            user_id: User ID to fetch credentials for
            
        Returns:
            Valid OAuth2 credentials or None if not found
        """
        conn = None
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT access_token, refresh_token, token_uri, 
                           client_id, client_secret, scopes, expiry
                    FROM youtube_credentials
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (user_id,))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"No credentials found for user {user_id}")
                    return None
                
                creds = Credentials(
                    token=row[0],
                    refresh_token=row[1],
                    token_uri=row[2],
                    client_id=row[3],
                    client_secret=row[4],
                    scopes=row[5].split(',') if row[5] else []
                )
                
                # Refresh if expired
                if creds.expired and creds.refresh_token:
                    logger.info(f"Refreshing expired credentials for user {user_id}")
                    creds.refresh(Request())
                    YouTubeCredentialManager.update_credentials(user_id, creds)
                
                return creds
                
        except Exception as e:
            logger.error(f"Error retrieving credentials for user {user_id}: {e}", exc_info=True)
            return None
        finally:
            if conn:
                conn.close()
    
    @staticmethod
    def update_credentials(user_id: int, creds: Credentials) -> bool:
        """Update refreshed credentials in database"""
        conn = None
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE youtube_credentials
                    SET access_token = %s,
                        expiry = %s,
                        updated_at = NOW()
                    WHERE user_id = %s
                """, (creds.token, creds.expiry, user_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating credentials: {e}", exc_info=True)
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()


# ============================================================================
# YOUTUBE API CLIENT
# ============================================================================

class YouTubeService:
    """Production-ready YouTube API service with error handling and retry logic"""
    
    def __init__(self, user_id: int):
        """
        Initialize YouTube service for a user
        
        Args:
            user_id: User ID to authenticate with
        """
        self.user_id = user_id
        self.credentials = YouTubeCredentialManager.get_credentials(user_id)
        
        if not self.credentials:
            raise ValueError(f"No valid credentials found for user {user_id}")
        
        self.youtube = None
        self.analytics = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize YouTube Data and Analytics API clients"""
        try:
            self.youtube = build(
                YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION,
                credentials=self.credentials,
                cache_discovery=False
            )
            
            self.analytics = build(
                ANALYTICS_API_SERVICE_NAME,
                ANALYTICS_API_VERSION,
                credentials=self.credentials,
                cache_discovery=False
            )
            
            logger.info(f"YouTube API clients initialized for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # CHANNEL OPERATIONS
    # ========================================================================
    
    @retry_on_error()
    @log_quota_usage("channels.list", cost=1)
    def get_channel_details(self, channel_id: str) -> Optional[Dict]:
        """
        Get detailed channel information
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            Channel details dict or None if not found
        """
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics,contentDetails,brandingSettings",
                id=channel_id
            )
            response = request.execute()
            
            if not response.get('items'):
                logger.warning(f"Channel not found: {channel_id}")
                return None
            
            return response['items'][0]
            
        except HttpError as e:
            logger.error(f"HTTP error getting channel {channel_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting channel {channel_id}: {e}", exc_info=True)
            raise
    
    @retry_on_error()
    @log_quota_usage("search.list", cost=100)
    def search_channel(self, handle: str) -> Optional[Dict]:
        """
        Search for channel by handle/username
        
        Args:
            handle: Channel handle (e.g., @username)
            
        Returns:
            Search result or None if not found
        """
        try:
            # Clean handle
            query = handle.lstrip('@')
            
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="channel",
                maxResults=1
            )
            response = request.execute()
            
            if not response.get('items'):
                logger.warning(f"No channel found for handle: {handle}")
                return None
            
            return response['items'][0]
            
        except Exception as e:
            logger.error(f"Error searching for channel {handle}: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # VIDEO OPERATIONS
    # ========================================================================
    
    @retry_on_error()
    @log_quota_usage("videos.list", cost=1)
    def get_video_details(self, video_ids: List[str]) -> List[Dict]:
        """
        Get detailed information for multiple videos (batch operation)
        
        Args:
            video_ids: List of video IDs (max 50 per request)
            
        Returns:
            List of video detail dicts
        """
        if not video_ids:
            return []
        
        try:
            # YouTube API allows max 50 IDs per request
            all_videos = []
            for i in range(0, len(video_ids), 50):
                batch = video_ids[i:i+50]
                
                request = self.youtube.videos().list(
                    part="snippet,statistics,contentDetails,status",
                    id=','.join(batch)
                )
                response = request.execute()
                all_videos.extend(response.get('items', []))
                
                # Rate limiting between batches
                if len(video_ids) > 50:
                    time.sleep(0.5)
            
            logger.info(f"Retrieved {len(all_videos)} videos from {len(video_ids)} IDs")
            return all_videos
            
        except Exception as e:
            logger.error(f"Error getting video details: {e}", exc_info=True)
            raise
    
    @retry_on_error()
    @log_quota_usage("videos.update", cost=50)
    def update_video_metadata(
        self,
        video_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category_id: Optional[str] = None
    ) -> bool:
        """
        Update video metadata
        
        Args:
            video_id: Video ID to update
            title: New title (optional)
            description: New description (optional)
            tags: New tags list (optional)
            category_id: New category ID (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First get current video details
            current_video = self.get_video_details([video_id])
            if not current_video:
                logger.error(f"Video not found: {video_id}")
                return False
            
            current = current_video[0]
            snippet = current['snippet'].copy()
            
            # Update only provided fields
            if title:
                snippet['title'] = title
            if description is not None:  # Allow empty string
                snippet['description'] = description
            if tags is not None:
                snippet['tags'] = tags
            if category_id:
                snippet['categoryId'] = category_id
            
            # Prepare update request
            request = self.youtube.videos().update(
                part="snippet",
                body={
                    "id": video_id,
                    "snippet": snippet
                }
            )
            
            response = request.execute()
            logger.info(f"Successfully updated video {video_id}")
            return True
            
        except HttpError as e:
            logger.error(f"HTTP error updating video {video_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating video {video_id}: {e}", exc_info=True)
            return False
    
    @retry_on_error()
    @log_quota_usage("thumbnails.set", cost=50)
    def set_video_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """
        Upload custom thumbnail for a video
        
        Args:
            video_id: Video ID
            thumbnail_path: Path to thumbnail image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            media = MediaFileUpload(
                thumbnail_path,
                mimetype='image/jpeg',
                resumable=True
            )
            
            request = self.youtube.thumbnails().set(
                videoId=video_id,
                media_body=media
            )
            
            response = request.execute()
            logger.info(f"Successfully set thumbnail for video {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting thumbnail for {video_id}: {e}", exc_info=True)
            return False
    
    # ========================================================================
    # PLAYLIST & VIDEO LISTING
    # ========================================================================
    
    @retry_on_error()
    def get_channel_videos(
        self,
        channel_id: str,
        max_results: int = 50,
        published_after: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get videos from a channel's upload playlist
        
        Args:
            channel_id: Channel ID
            max_results: Maximum number of videos to retrieve
            published_after: Only get videos published after this date
            
        Returns:
            List of video dicts
        """
        try:
            # Get uploads playlist ID
            channel = self.get_channel_details(channel_id)
            if not channel:
                return []
            
            uploads_playlist_id = channel['contentDetails']['relatedPlaylists']['uploads']
            
            # Get videos from playlist
            videos = []
            page_token = None
            
            while len(videos) < max_results:
                request = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=page_token
                )
                
                response = request.execute()
                items = response.get('items', [])
                
                # Filter by publish date if specified
                if published_after:
                    items = [
                        item for item in items
                        if datetime.fromisoformat(
                            item['snippet']['publishedAt'].replace('Z', '+00:00')
                        ) > published_after
                    ]
                
                videos.extend(items)
                page_token = response.get('nextPageToken')
                
                if not page_token:
                    break
            
            logger.info(f"Retrieved {len(videos)} videos from channel {channel_id}")
            return videos[:max_results]
            
        except Exception as e:
            logger.error(f"Error getting channel videos: {e}", exc_info=True)
            return []
    
    # ========================================================================
    # ANALYTICS OPERATIONS
    # ========================================================================
    
    @retry_on_error()
    def get_video_analytics(
        self,
        video_id: str,
        start_date: datetime,
        end_date: datetime,
        metrics: List[str] = None
    ) -> Optional[Dict]:
        """
        Get analytics data for a specific video
        
        Args:
            video_id: Video ID
            start_date: Start date for analytics
            end_date: End date for analytics
            metrics: List of metrics to retrieve
            
        Returns:
            Analytics data dict or None on error
        """
        if metrics is None:
            metrics = [
                'views', 'likes', 'comments', 'shares',
                'estimatedMinutesWatched', 'averageViewDuration',
                'subscribersGained'
            ]
        
        try:
            request = self.analytics.reports().query(
                ids='channel==MINE',
                startDate=start_date.strftime('%Y-%m-%d'),
                endDate=end_date.strftime('%Y-%m-%d'),
                metrics=','.join(metrics),
                dimensions='video',
                filters=f'video=={video_id}',
                sort='-views'
            )
            
            response = request.execute()
            
            if not response.get('rows'):
                logger.warning(f"No analytics data for video {video_id}")
                return None
            
            # Parse response into dict
            headers = [h['name'] for h in response.get('columnHeaders', [])]
            row = response['rows'][0]
            
            analytics = dict(zip(headers, row))
            logger.info(f"Retrieved analytics for video {video_id}")
            
            return analytics
            
        except HttpError as e:
            logger.error(f"HTTP error getting analytics for {video_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting analytics for {video_id}: {e}", exc_info=True)
            return None
    
    @retry_on_error()
    def get_channel_analytics(
        self,
        start_date: datetime,
        end_date: datetime,
        metrics: List[str] = None
    ) -> Optional[Dict]:
        """
        Get aggregated analytics for entire channel
        
        Args:
            start_date: Start date
            end_date: End date
            metrics: List of metrics to retrieve
            
        Returns:
            Channel analytics dict or None on error
        """
        if metrics is None:
            metrics = [
                'views', 'likes', 'comments', 'shares',
                'estimatedMinutesWatched', 'averageViewDuration',
                'subscribersGained', 'subscribersLost'
            ]
        
        try:
            request = self.analytics.reports().query(
                ids='channel==MINE',
                startDate=start_date.strftime('%Y-%m-%d'),
                endDate=end_date.strftime('%Y-%m-%d'),
                metrics=','.join(metrics)
            )
            
            response = request.execute()
            
            if not response.get('rows'):
                logger.warning("No analytics data for channel")
                return None
            
            headers = [h['name'] for h in response.get('columnHeaders', [])]
            row = response['rows'][0]
            
            analytics = dict(zip(headers, row))
            logger.info(f"Retrieved channel analytics from {start_date} to {end_date}")
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting channel analytics: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # CAPTION/TRANSCRIPT OPERATIONS
    # ========================================================================
    
    @retry_on_error()
    @log_quota_usage("captions.list", cost=50)
    def get_video_captions(self, video_id: str) -> Optional[List[Dict]]:
        """
        Get available captions/subtitles for a video
        
        Args:
            video_id: Video ID
            
        Returns:
            List of caption track dicts or None on error
        """
        try:
            request = self.youtube.captions().list(
                part="snippet",
                videoId=video_id
            )
            
            response = request.execute()
            captions = response.get('items', [])
            
            logger.info(f"Found {len(captions)} caption tracks for video {video_id}")
            return captions
            
        except HttpError as e:
            if e.resp.status == 403:
                logger.warning(f"No permission to access captions for {video_id}")
                return None
            logger.error(f"HTTP error getting captions for {video_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting captions for {video_id}: {e}", exc_info=True)
            return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_user_channel_id(user_id: int) -> Optional[str]:
    """Get YouTube channel ID for a user from database"""
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT channel_id
                FROM channels
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))
            
            row = cursor.fetchone()
            return row[0] if row else None
            
    except Exception as e:
        logger.error(f"Error getting channel ID for user {user_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()


async def fetch_video_analytics_async(
    user_id: int,
    video_id: str,
    days_back: int = 30
) -> Optional[Dict]:
    """Async wrapper for fetching video analytics"""
    loop = asyncio.get_event_loop()
    
    def _fetch():
        service = YouTubeService(user_id)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        return service.get_video_analytics(video_id, start_date, end_date)
    
    return await loop.run_in_executor(executor, _fetch)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'YouTubeService',
    'YouTubeCredentialManager',
    'get_user_channel_id',
    'fetch_video_analytics_async',
]
