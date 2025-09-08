from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
import logging
import re
from datetime import datetime
from services.youtube import (
    fetch_video_analytics, 
    fetch_video_timeseries_data,
    fetch_and_store_youtube_analytics
)
from utils.db import get_connection
from utils.auth import get_user_credentials, get_credentials_dict

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# Valid YouTube Analytics API metrics and dimensions
VALID_METRICS = {
    "views", "estimatedMinutesWatched", "averageViewDuration",
    "likes", "dislikes", "shares", "comments", 
    "subscribersGained", "subscribersLost", "estimatedRevenue",
    "estimatedAdRevenue", "cpm", "impressions", "impressionBasedCpm",
    "annotationClicks", "annotationClickThroughRate", "annotationClosableImpressions",
    "cardClicks", "cardClickRate", "cardImpressions", "cardTeaserClicks",
    "cardTeaserClickRate", "cardTeaserImpressions", "conversions", "conversionRate",
    "costPerConversion", "costPerView", "earnings", "grossRevenue", "monetizedPlaybacks",
    "playbackBasedCpm", "redViews", "redViewsPercentage", "revenuePerMille",
    "subscribersGainedFromRed", "subscribersLostFromRed", "uniqueViewers",
    "viewsPerUniqueViewer", "averageViewPercentage", "relativeRetentionPerformance"
}

VALID_DIMENSIONS = {
    "day", "month", "year", "country", "province", "continent",
    "ageGroup", "gender", "deviceType", "operatingSystem", "browser",
    "subscribedStatus", "youtubeProduct", "insightTrafficSourceType",
    "insightTrafficSourceDetail", "sharingService", "liveOrOnDemand",
    "subscribedStatus", "youtubeProduct", "creatorContentType",
    "uploaderType", "uploaderTypeCode", "video", "playlist", "channel"
}

def validate_video_id(video_id: str) -> bool:
    """Validate YouTube video ID format"""
    return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))

def validate_date_format(date_str: str) -> bool:
    """Validate date format YYYY-MM-DD"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_user_owns_video(user_id: int, video_id: str) -> bool:
    """Check if user owns the video"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 1 FROM youtube_videos v
                JOIN youtube_channels c ON v.channel_id = c.id
                WHERE v.video_id = %s AND c.user_id = %s
            """, (video_id, user_id))
            return cursor.fetchone() is not None
    except Exception as e:
        logger.error(f"Error validating user ownership for video {video_id}: {e}")
        return False
    finally:
        conn.close()

@router.get("/video/{video_id}")
async def get_video_analytics(
    video_id: str,
    user_id: int = Query(..., description="Database user ID (must be positive integer)"),
    metrics: Optional[str] = Query(None, description="Comma-separated list of metrics (e.g., 'views,likes,comments')"),
    dimensions: Optional[str] = Query(None, description="Comma-separated list of dimensions (e.g., 'day,country')"),
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format")
):
    """
    Get analytics data for a specific YouTube video.
    
    - video_id: YouTube video ID (11 characters, alphanumeric with hyphens/underscores)
    - user_id: Database user ID (must be positive integer)
    - metrics: Optional list of metrics to retrieve (e.g., views,likes,comments)
    - dimensions: Optional list of dimensions to group by (e.g., day,country)
    - start_date: Optional start date in YYYY-MM-DD format 
    - end_date: Optional end date in YYYY-MM-DD format
    """
    try:
        # Debug logging for parameters
        logger.info(f"Received parameters - video_id: {video_id}, user_id: {user_id}, metrics: {metrics}, dimensions: {dimensions}")
        
        # Input validation
        if not validate_video_id(video_id):
            raise HTTPException(
                status_code=400, 
                detail="Invalid video_id format. Must be 11 characters, alphanumeric with hyphens/underscores."
            )
        
        if user_id <= 0:
            raise HTTPException(
                status_code=400, 
                detail="Invalid user_id. Must be a positive integer."
            )
        
        if start_date and not validate_date_format(start_date):
            raise HTTPException(
                status_code=400, 
                detail="start_date must be in YYYY-MM-DD format"
            )
        
        if end_date and not validate_date_format(end_date):
            raise HTTPException(
                status_code=400, 
                detail="end_date must be in YYYY-MM-DD format"
            )
        
        # Process and validate metrics
        metrics_list = None
        if metrics:
            # Split comma-separated string into list
            metrics_list = [m.strip() for m in metrics.split(',') if m.strip()]
            
            if metrics_list:
                invalid_metrics = set(metrics_list) - VALID_METRICS
                if invalid_metrics:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid metrics: {', '.join(invalid_metrics)}. Valid metrics: {', '.join(sorted(VALID_METRICS))}"
                    )
        
        # Process and validate dimensions
        dimensions_list = None
        if dimensions:
            # Split comma-separated string into list
            dimensions_list = [d.strip() for d in dimensions.split(',') if d.strip()]
            
            if dimensions_list:
                invalid_dimensions = set(dimensions_list) - VALID_DIMENSIONS
                if invalid_dimensions:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid dimensions: {', '.join(invalid_dimensions)}. Valid dimensions: {', '.join(sorted(VALID_DIMENSIONS))}"
                    )
        
        # Check if user owns the video
        if not validate_user_owns_video(user_id, video_id):
            raise HTTPException(
                status_code=403, 
                detail="You don't have permission to access this video's analytics"
            )
        
        # Get user credentials from database
        credentials = get_user_credentials(user_id)
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="No valid credentials found. Please log in again."
            )
        
        # Set default metrics if not provided
        if not metrics_list:
            metrics_list = [
                "views", "estimatedMinutesWatched", "averageViewDuration", 
                "likes", "dislikes", "shares", "comments", 
                "subscribersGained", "subscribersLost",
            ]
            
        analytics_data = fetch_video_analytics(
            credentials,
            video_id,
            metrics_list,
            dimensions_list,
            start_date,
            end_date
        )
        
        # Check if the response contains an error
        if isinstance(analytics_data, dict) and 'error' in analytics_data:
            status_code = analytics_data.get('status_code', 500)
            raise HTTPException(status_code=status_code, detail=analytics_data['error'])
        
        logger.info(f"Successfully fetched analytics data for video {video_id}")
        return analytics_data
        
    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Error getting video analytics: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

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
        
        # --- Derive user_id from video_id using optimized single query ---
        conn = get_connection()
        derived_user_id = None
        try:
            with conn.cursor() as cursor:
                # Single optimized query to get user_id from video_id
                cursor.execute("""
                    SELECT c.user_id 
                    FROM youtube_videos v
                    JOIN youtube_channels c ON v.channel_id = c.id
                    WHERE v.video_id = %s
                """, (video_id,))
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Video {video_id} not found in database for timeseries fetch.")
                    # Return the cache error if it exists
                    if db_data.get('error'):
                         return db_data
                    else:
                        # Or raise 404 if no cache error either
                        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in database.")

                derived_user_id = result[0]
        finally:
            if conn and not conn.closed:
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
            # Always raise HTTP exception for consistency
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
        
        # Check if the response contains an error
        if isinstance(analytics_data, dict) and 'error' in analytics_data:
            status_code = analytics_data.get('status_code', 500)
            raise HTTPException(status_code=status_code, detail=analytics_data['error'])
        
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
            
        # Input validation
        if not validate_video_id(video_id):
            raise HTTPException(
                status_code=400, 
                detail="Invalid video_id format. Must be 11 characters, alphanumeric with hyphens/underscores."
            )
        
        if user_id <= 0:
            raise HTTPException(
                status_code=400, 
                detail="Invalid user_id. Must be a positive integer."
            )
        
        # Check if user owns the video
        if not validate_user_owns_video(user_id, video_id):
            raise HTTPException(
                status_code=403, 
                detail="You don't have permission to access this video's analytics"
            )
        
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
            
            # Check if the result contains an error
            if isinstance(result, dict) and 'error' in result:
                status_code = result.get('status_code', 500)
                raise HTTPException(status_code=status_code, detail=result['error'])
            
            return {
                "video_id": video_id,
                "status": "refreshed",
                "result": result
            }
            
    except Exception as e:
        logger.error(f"Error refreshing video analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))