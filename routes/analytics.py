from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
import logging
from config import get_settings
from services.youtube import (
    fetch_video_analytics, 
    fetch_granular_view_data, 
    fetch_video_timeseries_data,
    fetch_and_store_youtube_analytics
)
from utils.db import get_connection
from utils.auth import get_user_credentials, get_credentials_dict
from google.oauth2.credentials import Credentials

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.get("/video/{video_id}")
async def get_video_analytics(
    video_id: str,
    user_id: int,
    metrics: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get analytics data for a specific YouTube video.
    
    - video_id: YouTube video ID
    - metrics: Optional list of metrics to retrieve (e.g., views,likes,comments)
    - dimensions: Optional list of dimensions to group by (e.g., day,country)
    - start_date: Optional start date in YYYY-MM-DD format 
    - end_date: Optional end date in YYYY-MM-DD format
    """
    try:
        # Get user credentials from database
        credentials = get_user_credentials(user_id)
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="No valid credentials found. Please log in again."
            )
        
        # Set default metrics if not provided
        if not metrics:
            metrics = [
                "views", "estimatedMinutesWatched", "averageViewDuration", 
                "likes", "dislikes", "shares", "comments", 
                "subscribersGained", "subscribersLost",
            ]
            
        analytics_data = fetch_video_analytics(
            credentials,
            video_id,
            metrics,
            dimensions,
            start_date,
            end_date
        )
        logger.info(f"Successfully fetched analytics data for video {video_id}")
        logger.info(f"Analytics data for video {video_id}: {analytics_data}")
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error getting video analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{video_id}/timeseries")
async def get_video_timeseries(
    video_id: str,
    interval: str = "day",
    refresh: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Get timeseries data for a YouTube video. Derives user context from video_id.
    
    - video_id: YouTube video ID
    - interval: Time interval for data points (only 'day' is supported as YouTube no longer provides more granular data)
    - refresh: If True, force refresh data from YouTube API
    - start_date: Optional start date in YYYY-MM-DD format
    - end_date: Optional end date in YYYY-MM-DD format
    
    This endpoint retrieves daily timeseries data for the video. If refresh=True, 
    it will fetch fresh data from the YouTube API regardless of cache status.
    """
    try:
        logger.info(f"Received request for video timeseries data: video_id={video_id}, interval={interval}, refresh={refresh}, start_date={start_date}, end_date={end_date}")
        
        # If interval is not 'day', log warning and use 'day' anyway (YouTube API limitation)
        if interval != 'day':
            logger.warning(f"Interval '{interval}' is not supported by YouTube API, using 'day' instead")
            interval = 'day'
        
        # Try to get data from the database, with force_refresh if requested
        db_data = fetch_video_timeseries_data(video_id, interval, force_refresh=refresh)
        
        # If we have data and not forcing refresh, return it
        if not db_data.get('error') and not refresh:
            logger.info(f"Returning cached timeseries data for video {video_id}")
            return db_data
        
        # --- Derive user_id from video_id ---
        conn = get_connection()
        derived_user_id = None
        try:
            with conn.cursor() as cursor:
                # Find channel_id from youtube_videos using youtube_video_id
                cursor.execute("""
                    SELECT channel_id
                    FROM youtube_videos
                    WHERE video_id = %s
                """, (video_id,))
                video_result = cursor.fetchone()
                if not video_result:
                    # If video not in our DB, we can't get user, maybe return cached error or 404
                    logger.warning(f"Video {video_id} not found in database for timeseries fetch.")
                    # Return the cache error if it exists
                    if db_data.get('error'):
                         return db_data
                    else:
                        # Or raise 404 if no cache error either
                        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in database.")

                channel_id = video_result[0]

                # Find user_id from youtube_channels using channel_id
                cursor.execute("""
                    SELECT user_id
                    FROM youtube_channels
                    WHERE id = %s
                """, (channel_id,))
                channel_result = cursor.fetchone()
                if not channel_result:
                     # This implies data inconsistency if video exists but channel doesn't
                     logger.error(f"Channel {channel_id} for video {video_id} not found in database.")
                     raise HTTPException(status_code=500, detail="Internal data inconsistency: Channel not found.")

                derived_user_id = channel_result[0]
        finally:
            conn.close()

        if derived_user_id is None:
             # Should have been caught above, but as a safeguard
             raise HTTPException(status_code=500, detail="Could not determine user owner for the video.")

        # Get user credentials using the derived_user_id
        credentials = get_user_credentials(derived_user_id)
        credentials_dict = get_credentials_dict(derived_user_id)
        
        if not credentials or not credentials_dict:
            # This might happen if the derived user has no valid credentials stored
            logger.warning(f"No valid credentials found for derived user_id {derived_user_id} for video {video_id}.")
            # Depending on desired behavior, either return the cache error or raise 401/specific error
            if db_data.get('error'):
                 return db_data
            else:
                raise HTTPException(
                        status_code=401,
                        detail="Could not retrieve valid credentials for the video owner."
                )
        
        # Fetch data directly using the fetch_and_store function
        logger.info(f"Fetching fresh data from YouTube API for video {video_id}")
        analytics_data = await fetch_and_store_youtube_analytics(
            derived_user_id,
            video_id,
            credentials_dict,
            interval
        )
        
        return analytics_data
            
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        logger.error(f"Error getting video timeseries data for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/video/{video_id}/refresh")
async def refresh_video_analytics(
    video_id: str,
    user_id: int,
    interval: str = "day",
    background_tasks: BackgroundTasks = None
):
    """
    Force refresh of analytics data for a specific video.
    
    - **video_id**: YouTube video ID
    - **interval**: Time interval for data points (only 'day' is supported as YouTube no longer provides more granular data)
    
    This triggers a background task to fetch fresh analytics data from YouTube.
    """
    try:
        # Log warning if using unsupported interval
        if interval != 'day':
            logger.warning(f"Interval '{interval}' is not supported by YouTube API, using 'day' instead")
            interval = 'day'
            
        # Get user credentials from database
        credentials = get_user_credentials(user_id)
        credentials_dict = get_credentials_dict(user_id)
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="No valid credentials found. Please log in again."
            )
        
        if background_tasks:
            # Queue the background task
            background_tasks.add_task(
                fetch_and_store_youtube_analytics,
                user_id,
                video_id,
                credentials_dict,
                interval
            )
            
            return {
                "video_id": video_id,
                "status": "refreshing",
                "message": "Analytics refresh has been triggered in the background."
            }
        else:
            # Fetch directly if no background tasks available
            result = await fetch_and_store_youtube_analytics(
                user_id,
                video_id,
                credentials_dict,
                interval
            )
            
            return {
                "video_id": video_id,
                "status": "refreshed",
                "result": result
            }
            
    except Exception as e:
        logger.error(f"Error refreshing video analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))