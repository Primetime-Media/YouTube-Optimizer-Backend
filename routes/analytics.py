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
    
    - **video_id**: YouTube video ID
    - **metrics**: Optional list of metrics to retrieve (e.g., views,likes,comments)
    - **dimensions**: Optional list of dimensions to group by (e.g., day,country)
    - **start_date**: Optional start date in YYYY-MM-DD format 
    - **end_date**: Optional end date in YYYY-MM-DD format
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
                "subscribersGained", "subscribersLost"
            ]
            
        analytics_data = fetch_video_analytics(
            credentials,
            video_id,
            metrics,
            dimensions,
            start_date,
            end_date
        )
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error getting video analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{video_id}/timeseries")
async def get_video_timeseries(
    video_id: str,
    user_id: int,
    interval: str = "30m",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Get granular timeseries data for a YouTube video.
    
    - **video_id**: YouTube video ID
    - **interval**: Time interval for data points ('1m' for minute-by-minute, '30m' for half-hour)
    - **start_date**: Optional start date in YYYY-MM-DD format
    - **end_date**: Optional end date in YYYY-MM-DD format
    
    This endpoint first attempts to retrieve stored timeseries data. If none exists,
    it will fetch new data from the YouTube API in the background and return an empty result.
    """
    try:
        # First try to get data from the database
        db_data = fetch_video_timeseries_data(video_id, interval)
        
        # If we have data, return it
        if not db_data.get('error'):
            return db_data
        
        # Get user credentials from database
        credentials = get_user_credentials(user_id)
        credentials_dict = get_credentials_dict(user_id)
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="No valid credentials found. Please log in again."
            )
            
        # If no data in database, trigger background fetch and inform the client
        if background_tasks:
            # Queue the background task to fetch and store the data
            background_tasks.add_task(
                fetch_and_store_youtube_analytics,
                user_id, 
                video_id,
                credentials_dict,
                interval
            )
            
            return {
                "video_id": video_id,
                "status": "fetching",
                "message": "No data available yet. Data fetch has been triggered in the background."
            }
        else:
            # If no background tasks available, fetch directly
            # This will slow down the response but provide immediate results
            analytics_data = fetch_granular_view_data(
                credentials,
                video_id,
                interval,
                start_date,
                end_date
            )
            
            # Store the data in the background later
            return analytics_data
            
    except Exception as e:
        logger.error(f"Error getting video timeseries data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/video/{video_id}/refresh")
async def refresh_video_analytics(
    video_id: str,
    user_id: int,
    interval: str = "30m",
    background_tasks: BackgroundTasks = None
):
    """
    Force refresh of analytics data for a specific video.
    
    - **video_id**: YouTube video ID
    - **interval**: Time interval for data points ('1m' for minute-by-minute, '30m' for half-hour)
    
    This triggers a background task to fetch fresh analytics data from YouTube.
    """
    try:
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