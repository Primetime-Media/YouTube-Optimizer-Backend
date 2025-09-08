"""
Video Routes Module

This module defines all API endpoints related to video optimization and management
for the YouTube Optimizer platform. It handles video data retrieval, optimization
generation, application of optimizations to YouTube, and analytics tracking.

Key functionalities:
- Video data retrieval and management
- AI-powered video optimization generation
- Optimization application to YouTube videos
- Video analytics and performance tracking
- Competitor analysis integration
- Background task processing for long-running operations

The routes integrate with multiple services including LLM optimization, YouTube API,
competitor analysis, and database operations to provide comprehensive video optimization.

Author: YouTube Optimizer Team
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Service imports - organized by functionality
from services.video import (
    get_video_data,
    create_optimization,
    generate_video_optimization,
    get_video_optimizations
)
from services.optimizer import apply_optimization_to_youtube_video, get_optimization_status
from services.llm_optimization import should_optimize_video, enhanced_should_optimize_video, get_comprehensive_optimization
from services.youtube import fetch_video_timeseries_data, fetch_and_store_youtube_analytics
from services.competitor_analysis import get_competitor_analysis

# Utility imports
from utils.auth import get_user_credentials, get_user_from_session, User
from utils.db import get_connection
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Create FastAPI router with prefix and tags for API documentation
router = APIRouter(prefix="/video", tags=["Video Optimization"])

# =============================================================================
# PYDANTIC MODELS FOR API REQUEST/RESPONSE VALIDATION
# =============================================================================

class VideoOptimizationStatusResponse(BaseModel):
    """
    Response model for video optimization status information.
    
    This model represents the current status of a video optimization process,
    including progress tracking and optimization details for completed optimizations.
    """
    id: int                                    # Unique optimization ID
    video_id: int                             # Associated video ID
    status: str                               # Current optimization status
    progress: int                             # Progress percentage (0-100)
    created_at: Optional[datetime] = None     # When optimization was created
    updated_at: Optional[datetime] = None     # When optimization was last updated
    
    # Optional fields included for completed optimizations
    original_title: Optional[str] = None      # Original video title
    optimized_title: Optional[str] = None     # AI-optimized title
    original_description: Optional[str] = None # Original video description
    optimized_description: Optional[str] = None # AI-optimized description
    original_tags: Optional[List[str]] = None # Original video tags
    optimized_tags: Optional[List[str]] = None # AI-optimized tags
    optimization_notes: Optional[str] = None  # AI-generated optimization notes
    is_applied: Optional[bool] = None         # Whether optimization was applied to YouTube
    applied_at: Optional[datetime] = None     # When optimization was applied

class ComprehensiveVideoOptimizationResponse(BaseModel):
    id: int
    original_title: Optional[str] = None
    optimized_title: Optional[str] = None
    original_description: Optional[str] = None
    optimized_description: Optional[str] = None
    original_tags: Optional[List[str]] = None
    optimized_tags: Optional[List[str]] = None
    optimization_notes: Optional[str] = None
    status: str
    progress: int

class QueueOptimizationRequest(BaseModel):
    youtube_video_ids: List[str] = Field(..., min_items=1)

@router.post("/{youtube_video_id}/optimize")
async def optimize_video(youtube_video_id: str, background_tasks: BackgroundTasks):
    """
    (DEPRECATED) This endpoint is now used for queuing optimizations.

    Start an optimization job for a YouTube video

    Args:
        youtube_video_id: The YouTube video ID (not database ID)
        
    Returns:
        dict: Contains job ID and initial status
    """
    try:
        # Fetch video data from database
        video = get_video_data(youtube_video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get user_id from the video's channel
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT c.user_id
                FROM youtube_videos v
                JOIN youtube_channels c ON v.channel_id = c.id
                WHERE v.video_id = %s
            """, (youtube_video_id,))
            
            user_result = cursor.fetchone()
            if not user_result:
                raise HTTPException(status_code=404, detail="Video channel not found")
            
            user_id = user_result[0]
            
        # Create a new optimization record for this request
        optimization_id = create_optimization(video["id"])
        if optimization_id == 0:
            raise HTTPException(status_code=500, detail="Failed to create optimization record")
            
        # Run the optimization in the background with the pre-created optimization ID
        background_tasks.add_task(generate_video_optimization, video, user_id, optimization_id)
        
        # Return the job ID and status for tracking
        return {
            "id": optimization_id,
            "video_id": video["id"],
            "youtube_video_id": youtube_video_id,
            "status": "pending",
            "progress": 0,
            "message": "Optimization started"
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error starting video optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting optimization: {str(e)}")


@router.get("/optimization/{optimization_id}/status", response_model=VideoOptimizationStatusResponse)
async def get_optimization_status_endpoint(optimization_id: int):
    """
    Get the current status of a video optimization job
    
    Args:
        optimization_id: The ID of the optimization job
        
    Returns:
        VideoOptimizationStatusResponse: Contains status and progress information
    """
    try:
        status = get_optimization_status(optimization_id)
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
            
        return status
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting optimization status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@router.get("/{youtube_video_id}/optimizations")
async def get_video_optimizations_endpoint(youtube_video_id: str, limit: int = 5):
    """
    Get recent optimization results for a video
    
    Args:
        youtube_video_id: The YouTube video ID
        limit: Maximum number of results to return (default: 5)
        
    Returns:
        list: Recent optimization records
    """
    try:
        # First, get the database ID for the video
        video = get_video_data(youtube_video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
            
        optimizations = get_video_optimizations(video["id"], limit)
        return {
            "video_id": youtube_video_id,
            "optimizations": optimizations,
            "count": len(optimizations)
        }
    except Exception as e:
        logging.error(f"Error getting video optimizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting optimizations: {str(e)}")


@router.post("/optimization/{optimization_id}/apply")
async def apply_optimization_to_video(optimization_id: int, user = Depends(get_user_from_session)):
    """
    Apply an optimization directly to YouTube video metadata
    
    Args:
        optimization_id: The ID of the optimization to apply
        
    Returns:
        dict: Results of the update operation
    """
    try:
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
            
        # Get the video_id for this optimization
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT v.channel_id
                    FROM video_optimizations o
                    JOIN youtube_videos v ON v.id = o.video_id
                    WHERE o.id = %s
                """, (optimization_id,))
                
                result = cursor.fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail="Optimization not found")
                
                channel_id = result[0]
                
                # Get the user_id for this channel
                cursor.execute("""
                    SELECT user_id
                    FROM youtube_channels
                    WHERE id = %s
                """, (channel_id,))
                
                user_result = cursor.fetchone()
                if not user_result:
                    raise HTTPException(status_code=404, detail="Channel not found")
                
                video_owner_id = user_result[0]
                
                # Verify the user owns this video
                if user.id != video_owner_id:
                    raise HTTPException(status_code=403, detail="You don't have permission to update this video")
        finally:
            conn.close()
            
        # Apply the optimization using the shared function
        result = apply_optimization_to_youtube_video(optimization_id, user.id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to apply optimization")
            )
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying optimization: {str(e)}")
    
@router.get("/")
@router.get("/performance/all")
async def get_all_videos_performance(
    interval: str = "1d",
    limit: int = 100,
    offset: int = 0,
    refresh: bool = False,
):
    """
    Get timeseries performance data for all videos across all users and channels.
    Includes metrics like view count, click-through rate, and subscriber change.
    Analyzes videos by channel for better performance insights.
    
    Args:
        interval: Time interval for data points ('30m' for half-hour)
        limit: Maximum number of videos to return
        offset: Offset for pagination
        refresh: If true, forces a refresh of analytics data from YouTube API
        
    Returns:
        dict: Contains performance data for all videos grouped by channel
    """
    logger.info(f"Starting get_all_videos_performance: interval={interval}, limit={limit}, offset={offset}, refresh={refresh}")
    try:
        # Get all channels first
        logger.info("Fetching list of all channels...")
        conn = get_connection()
        channels = []
        total_count = 0
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, title, user_id
                    FROM youtube_channels
                    ORDER BY title
                """)
                channels = cursor.fetchall()
                logger.info(f"Found {len(channels)} channels.")

                # Get total count of videos for pagination info
                cursor.execute("SELECT COUNT(*) FROM youtube_videos")
                total_count_result = cursor.fetchone()
                total_count = total_count_result[0] if total_count_result else 0
                logger.info(f"Total videos across all channels: {total_count}")
        except Exception as db_err:
            logger.error(f"Database error fetching channels or total count: {db_err}")
            raise HTTPException(status_code=500, detail="Database error retrieving channel list.")
        finally:
            conn.close()

        # Process data by channel
        all_channel_data = []
        all_videos = []
        total_videos_processed_across_channels = 0 # Tracks total videos considered across channels visited

        logger.info(f"Processing {len(channels)} channels to gather videos up to limit={limit} with offset={offset}")
        # Loop through channels and get performance data
        for channel_id, channel_title, channel_user_id in channels:
            logger.info(f"Processing channel ID: {channel_id}, Title: {channel_title}")

            # Optimization: Skip further channel processing if we've already gathered enough videos
            if len(all_videos) >= limit:
                logger.info(f"Reached limit ({limit}), stopping channel processing.")
                break

            # --- Fetching Data for the Current Channel ---
            try:
                logger.info(f"Calling get_channel_videos_performance for channel {channel_id}...")
                # IMPORTANT: Ensure BackgroundTasks is not passed here if get_channel_videos_performance doesn't accept it
                # Or handle it appropriately if it's needed downstream
                channel_data = await get_channel_videos_performance(
                    channel_id=channel_id,
                    interval=interval,
                    refresh=refresh,
                )
                logger.debug(f"Received data for channel {channel_id}. Video count: {channel_data.get('count', 0)}")
            except Exception as channel_fetch_err:
                logger.error(f"Failed to get performance data for channel {channel_id}: {channel_fetch_err}")
                # Decide whether to skip this channel or raise an error
                continue # Skip this channel on error

            # Skip channels with no videos
            channel_video_count = channel_data.get("count", 0)
            if channel_video_count == 0:
                logger.debug(f"Channel {channel_id} has no videos, skipping.")
                continue

            channel_videos = channel_data.get("videos", [])

            # --- Pagination Logic for Videos within this Channel ---
            # Calculate how many videos from *previous* channels we've already processed/skipped for offset
            videos_processed_before_this_channel = total_videos_processed_across_channels

            # Check if this entire channel falls before the offset window
            if videos_processed_before_this_channel + channel_video_count <= offset:
                logger.debug(f"Channel {channel_id} (videos {videos_processed_before_this_channel + 1} to {videos_processed_before_this_channel + channel_video_count}) falls completely before offset {offset}. Skipping.")
                total_videos_processed_across_channels += channel_video_count
                continue

            # Calculate how many videos to skip *within this channel* due to the offset
            videos_to_skip_in_this_channel = max(0, offset - videos_processed_before_this_channel)

            # Calculate how many videos to take *from this channel*
            remaining_limit_slots = limit - len(all_videos)
            videos_available_in_this_channel_after_skip = channel_video_count - videos_to_skip_in_this_channel
            videos_to_take_from_this_channel = min(remaining_limit_slots, videos_available_in_this_channel_after_skip)

            logger.debug(f"Channel {channel_id}: Skipping {videos_to_skip_in_this_channel} videos, Taking {videos_to_take_from_this_channel} videos.")

            if videos_to_take_from_this_channel <= 0:
                 # This case should ideally be caught by the earlier offset check, but included for safety
                 logger.debug(f"No videos to take from channel {channel_id} after applying offset/limit.")
                 total_videos_processed_across_channels += channel_video_count
                 continue

            # Get the slice of videos for this channel that falls within our pagination window
            start_index = videos_to_skip_in_this_channel
            end_index = start_index + videos_to_take_from_this_channel
            channel_videos_slice = channel_videos[start_index:end_index]

            # Add these videos to our results
            all_videos.extend(channel_videos_slice)
            logger.info(f"Added {len(channel_videos_slice)} videos from channel {channel_id}. Total videos collected: {len(all_videos)}.")

            # Update total videos processed across all channels visited so far
            total_videos_processed_across_channels += channel_video_count

            # Add channel summary (only if we took videos from it, or maybe always?)
            # Let's add it if we processed it, even if offset skipped its videos
            all_channel_data.append({
                "channel_id": channel_id,
                "channel_title": channel_title,
                "user_id": channel_user_id,
                "video_count": channel_video_count, # Total videos in this channel
                "subscriber_count": channel_data.get("channel_subscriber_count", 0),
            })

            if len(all_videos) >= limit:
                logger.info(f"Reached limit ({limit}) after processing channel {channel_id}. Stopping.")
                break

        logger.info(f"Finished processing channels. Returning {len(all_videos)} videos and {len(all_channel_data)} channel summaries.")
        return {
            "videos": all_videos,
            "channels": all_channel_data,
            "videos_count": len(all_videos),
            "channels_count": len(all_channel_data), # Number of channels we actually processed data for
            "total_videos": total_count, # Total videos across all channels in DB
            "offset": offset,
            "limit": limit,
            "refreshing_data": refresh # Indicate if refresh was attempted
        }

    except HTTPException as http_exc:
        logger.error(f"HTTPException in get_all_videos_performance: {http_exc.detail}")
        raise # Re-raise HTTPException
    except Exception as e:
        logger.exception(f"Unexpected error in get_all_videos_performance: {str(e)}") # Use logger.exception to include traceback
        raise HTTPException(status_code=500, detail=f"Error retrieving performance data: {str(e)}")


@router.get("/performance/by-channel/{channel_id}")
async def get_channel_videos_performance(
    channel_id: int,
    interval: str = "1d",
    refresh: bool = False,
    include_optimizations: bool = True,
    optimization_limit: int = 3,
    optimization_confidence_threshold: float = 0.5
):
    """
    Get timeseries performance data for all videos in a specific channel.
    Includes metrics like view count, click-through rate, and subscriber change.
    Also retrieves optimization data for analysis and recommendations.
    
    Args:
        channel_id: The database ID of the channel
        interval: Time interval for data points ('30m' for half-hour)
        refresh: If true, forces a refresh of analytics data from YouTube API
        include_optimizations: If true, includes optimization history for each video
        optimization_limit: Maximum number of optimizations to include per video
        optimization_confidence_threshold: Minimum confidence threshold for including an optimization
        
    Returns:
        dict: Contains performance data for all videos in the channel with optimization history
    """
    logger.info(f"Starting get_channel_videos_performance for channel_id={channel_id}, interval={interval}, refresh={refresh}")
    try:
        # Verify the user owns this channel or is admin
        conn = get_connection()
        channel_owner_id = None
        videos = []
        channel_title = None
        channel_subscriber_count = 0
        
        try:
            with conn.cursor() as cursor:
                logger.info(f"Fetching user_id for channel {channel_id}")
                cursor.execute("""
                    SELECT user_id
                    FROM youtube_channels
                    WHERE id = %s
                """, (channel_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Channel {channel_id} not found in database")
                    raise HTTPException(status_code=404, detail="Channel not found")
                
                channel_owner_id = result[0]
                logger.info(f"Channel {channel_id} owned by user {channel_owner_id}")
                
                # Get all videos for this channel
                logger.info(f"Fetching videos for channel {channel_id}")
                cursor.execute("""
                    SELECT v.id, v.video_id, v.title, v.view_count, v.published_at,
                           c.title as channel_title, c.subscriber_count,
                           v.is_optimized, v.last_optimized_at, v.last_optimization_id
                    FROM youtube_videos v
                    JOIN youtube_channels c ON v.channel_id = c.id
                    WHERE v.channel_id = %s AND v.queued_for_optimization = true
                    ORDER BY v.published_at DESC
                """, (channel_id,))
                
                videos = cursor.fetchall()
                
                if not videos:
                    logger.info(f"No videos found for channel {channel_id}")
                    return {
                        "channel_id": channel_id,
                        "videos": [],
                        "count": 0,
                        "message": "No videos found for this channel"
                    }
                
                logger.info(f"Found {len(videos)} videos for channel {channel_id}")
                
                # Store channel info from the first video result
                if videos and len(videos[0]) > 6:
                    channel_title = videos[0][5]
                    channel_subscriber_count = videos[0][6]
                    logger.info(f"Channel title: {channel_title}, subscribers: {channel_subscriber_count}")
        except Exception as db_err:
            logger.error(f"Database error while fetching channel/video data: {db_err}")
            if conn:
                conn.rollback()
            raise
        finally:
            conn.close()
            logger.debug("Database connection closed")
        
        # Process videos and fetch timeseries data
        refresh_tasks = []
        
        from utils.auth import get_credentials_dict
        
        # Get credentials for the channel owner if refresh is requested
        credentials_dict = None
        if refresh and channel_owner_id:
            logger.info(f"Refresh requested, getting credentials for user {channel_owner_id}")
            try:
                credentials_dict = get_credentials_dict(channel_owner_id)
                if credentials_dict:
                    logger.info(f"Successfully retrieved credentials for user {channel_owner_id}")
                else:
                    logger.warning(f"Failed to retrieve valid credentials for user {channel_owner_id}")
            except Exception as cred_err:
                logger.error(f"Error retrieving credentials for user {channel_owner_id}: {cred_err}")
        
        video_optimizations_applied = 0
        logger.info(f"Processing {len(videos)} videos for performance data")
        for i, video_data in enumerate(videos):
            logger.info(f"Processing video {i+1}/{len(videos)}")
            
            try:
                video_db_id = video_data[0]
                video_id = video_data[1]
                title = video_data[2]
                view_count = video_data[3]
                published_at = video_data[4]
                is_optimized = video_data[7] if len(video_data) > 7 else False
                last_optimized_at = video_data[8] if len(video_data) > 8 else None
                last_optimization_id = video_data[9] if len(video_data) > 9 else None
                
                logger.info(f"Video DB ID: {video_db_id}, YouTube ID: {video_id}, Title: {title}")

                # Get optimization data if requested
                optimizations = []
                if include_optimizations:
                    logger.info(f"Fetching optimization history for video {video_db_id}")
                    try:
                        optimizations = get_video_optimizations(video_db_id, optimization_limit)
                        logger.info(f"Found {len(optimizations)} optimizations for video {video_id}")
                    except Exception as opt_err:
                        logger.warning(f"Error getting optimizations for video {video_id}: {opt_err}")

                num_optimizations_applied = len([opt for opt in optimizations if opt.get("is_applied")])

                # If the video has already been optimized within the last 24 hours, skip it
                if last_optimized_at and (datetime.now() - last_optimized_at).total_seconds() < 86400:
                    logger.info(f"Video {video_id} has already been optimized within the last 24 hours, skipping")
                    continue

                analytics_data = {}
                
                # Refresh analytics data if requested
                if refresh and credentials_dict:
                    logger.info(f"Refreshing analytics data for video {video_id}")
                    try:
                        analytics_data = await fetch_and_store_youtube_analytics(
                            channel_owner_id,
                            video_id,
                            credentials_dict,
                            interval
                        )
                        logger.info(f"Successfully refreshed analytics for video {video_id}")
                    except Exception as refresh_err:
                        logger.error(f"Error refreshing analytics for video {video_id}: {refresh_err}")
                        refresh_tasks.append({
                            "video_id": video_id,
                            "user_id": channel_owner_id,
                            "status": "failed",
                            "error": str(refresh_err)
                        })
                else:
                    # Get cached timeseries data
                    logger.info(f"Fetching timeseries data for video {video_id} with interval {interval}")
                    analytics_data = fetch_video_timeseries_data(video_id, interval)

                    """
                    # Extract analytics data from timeseries_data if available
                    if timeseries_data and isinstance(timeseries_data, dict):
                        analytics_data = {
                            "likes": timeseries_data.get("likes", {}).get("latest", 0),
                            "comments": timeseries_data.get("comments", {}).get("latest", 0),
                            "ctr": timeseries_data.get("ctr", {}).get("latest", 0),
                            "avg_view_duration": timeseries_data.get("average_view_duration", {}).get("latest", 0),
                        }
                    """
                
                # Get detailed video data
                detailed_video_data = get_video_data(video_id)

                if not detailed_video_data:
                    logger.warning(f"No detailed data found for video {video_id}")
                    continue

                # Fetch transcript if missing
                if not detailed_video_data.get('transcript'):
                    # Use existing credentials_dict or fetch fresh ones
                    transcript_credentials_dict = credentials_dict
                    if not transcript_credentials_dict:
                        logger.info(f"No credentials available, fetching credentials for user {channel_owner_id}")
                        try:
                            transcript_credentials_dict = get_credentials_dict(channel_owner_id)
                        except Exception as cred_err:
                            logger.warning(f"Failed to get credentials for transcript fetching: {cred_err}")
                    
                    if transcript_credentials_dict:
                        logger.info(f"Fetching transcript for video {video_id}")
                        try:
                            from services.youtube import fetch_video_transcript
                            from google.oauth2.credentials import Credentials
                            
                            # Convert credentials dict to Credentials object
                            creds = Credentials(
                                token=transcript_credentials_dict['token'],
                                refresh_token=transcript_credentials_dict['refresh_token'],
                                token_uri=transcript_credentials_dict['token_uri'],
                                client_id=transcript_credentials_dict['client_id'],
                                client_secret=transcript_credentials_dict['client_secret'],
                                scopes=transcript_credentials_dict['scopes']
                            )
                            
                            transcript_data = fetch_video_transcript(creds, video_id)
                            if transcript_data.get('transcript'):
                                detailed_video_data['transcript'] = transcript_data['transcript']
                                logger.info(f"Successfully fetched transcript for video {video_id}")
                            else:
                                logger.info(f"No transcript available for video {video_id}")
                        except Exception as transcript_err:
                            logger.warning(f"Failed to fetch transcript for video {video_id}: {transcript_err}")
                    else:
                        logger.info(f"No valid credentials available for transcript fetching for video {video_id}")

                # Call LLM to determine if optimization is needed
                logger.info(f"Calling LLM to evaluate if video {video_id} needs optimization")
                optimization_decision = await enhanced_should_optimize_video(
                    detailed_video_data,
                    channel_subscriber_count,
                    analytics_data,
                    optimizations
                )

                if not optimization_decision["should_optimize"] or optimization_decision["confidence"] < optimization_confidence_threshold:
                    logger.info(f"Video {video_id} does not need optimization")
                    continue

                competitor_analysis = get_competitor_analysis(channel_owner_id, detailed_video_data)

                # Create a new optimization record for this request
                optimization_id = create_optimization(video_db_id, num_optimizations_applied + 1)
                if optimization_id == 0:
                    raise Exception("Failed to create optimization record")

                # Run the optimization with the pre-created optimization ID
                video_optimization_details = generate_video_optimization(
                    video=detailed_video_data,
                    user_id=channel_owner_id,
                    optimization_id=optimization_id,
                    optimization_decision_data=optimization_decision,
                    analytics_data=analytics_data,
                    competitor_analytics_data=competitor_analysis,
                    apply_optimization=True,
                    prev_optimizations=optimizations
                )

                if video_optimization_details['is_applied']:
                    logger.info(f"Optimization applied for video {video_id} with optimization ID {optimization_id}")
                    video_optimizations_applied += 1

            except Exception as video_err:
                logger.error(f"Error processing video {video_data[1] if len(video_data) > 1 else 'unknown'}: {video_err}")
                # Continue processing other videos

        return video_optimizations_applied
        
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions with their original status code
        logger.error(f"HTTP exception in get_channel_videos_performance: {http_exc.detail}")
        raise
    except Exception as e:
        # For other exceptions, log with traceback and return 500
        logger.exception(f"Unexpected error in get_channel_videos_performance for channel {channel_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving performance data: {str(e)}")


@router.post("/queue-optimizations")
async def queue_videos_for_optimization(
    request_body: QueueOptimizationRequest, # Accept list in body
    user: User = Depends(get_user_from_session)
):
    """
    Marks a list of videos to be optimized by a background process at a later time.
    Does not start the optimization immediately. Checks ownership for each video.

    Args:
        request_body: Contains a list of YouTube video IDs to queue.
        user: The authenticated user making the request.

    Returns:
        dict: Summary of the queuing operation.
    """
    logger.info(f"Received request to queue {len(request_body.youtube_video_ids)} videos for optimization by user {user.id}")
    if not user:
        logger.warning("Queue optimization attempt failed: Authentication required.")
        raise HTTPException(status_code=401, detail="Authentication required")

    conn = None
    successfully_queued = []
    failed_to_queue = []

    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            for youtube_video_id in request_body.youtube_video_ids:
                try:
                    # 1. Find the video and its owner user_id
                    logger.debug(f"Checking ownership for video {youtube_video_id} and user {user.id}")
                    cursor.execute("""
                        SELECT v.id, c.user_id
                        FROM youtube_videos v
                        JOIN youtube_channels c ON v.channel_id = c.id
                        WHERE v.video_id = %s
                    """, (youtube_video_id,))

                    video_result = cursor.fetchone()
                    if not video_result:
                        logger.warning(f"Video {youtube_video_id} not found in database. Skipping queue.")
                        failed_to_queue.append({"video_id": youtube_video_id, "reason": "Not found"})
                        continue # Skip to the next video ID

                    video_db_id, video_owner_id = video_result

                    # 2. Verify the requesting user owns this video
                    # Add admin check if needed: and not user.is_admin
                    if user.id != video_owner_id:
                         logger.warning(f"User {user.id} does not own video {youtube_video_id} (owner: {video_owner_id}). Skipping queue.")
                         failed_to_queue.append({"video_id": youtube_video_id, "reason": "Permission denied"})
                         continue # Skip to the next video ID

                    # 3. Update the video record to mark it for optimization
                    logger.info(f"Updating video {video_db_id} (YouTube ID: {youtube_video_id}) to set queued_for_optimization=TRUE")
                    cursor.execute("""
                        UPDATE youtube_videos
                        SET queued_for_optimization = TRUE,
                            updated_at = NOW()
                        WHERE id = %s
                    """, (video_db_id,))

                    # Add to success list *before* commit in case commit fails later
                    successfully_queued.append({"video_id": youtube_video_id, "db_id": video_db_id})

                except Exception as video_err:                    
                    logger.error(f"Error processing video {youtube_video_id} in batch queue: {video_err}")
                    failed_to_queue.append({"video_id": youtube_video_id, "reason": f"Internal error: {str(video_err)}"})
                    conn.rollback() # Rollback changes for this specific video if an error occurred during its processing
                else:                     
                     conn.commit()

            logger.info(f"Batch queue summary: {len(successfully_queued)} succeeded, {len(failed_to_queue)} failed.")

        return {
            "message": f"Queued {len(successfully_queued)} out of {len(request_body.youtube_video_ids)} videos for optimization.",
            "successfully_queued": successfully_queued,
            "failed_to_queue": failed_to_queue
        }

    except HTTPException as http_exc:
        # This catches HTTPExceptions raised before the loop or if DB connection fails
        if conn: conn.rollback()
        logger.error(f"HTTPException queuing video batch: {http_exc.detail}")
        raise # Re-raise the specific HTTP exception
    except Exception as e:
        # This catches unexpected errors (like DB connection issues before loop)
        if conn: conn.rollback()
        logger.exception(f"Unexpected error queuing video batch: {str(e)}") # Log full traceback
        raise HTTPException(status_code=500, detail="Failed to process video queue request.")
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed for batch queue-optimization.")

# Additional YouTube-related endpoints moved from main.py

@router.post("/{video_id}/fetch-transcript")
async def fetch_and_store_transcript(video_id: str, user_id: int):
    """Fetch and store transcript for a specific video."""
    try:
        # First check if we already have the transcript
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT transcript, has_captions
                    FROM youtube_videos
                    WHERE video_id = %s
                """, (video_id,))
                
                result = cursor.fetchone()
                if not result:
                    raise HTTPException(status_code=404, detail="Video not found")
                    
                existing_transcript, has_captions = result
                
                # If transcript already exists, return it
                if existing_transcript:
                    return {
                        "video_id": video_id,
                        "has_transcript": True,
                        "transcript_length": len(existing_transcript),
                        "has_captions": has_captions,
                        "message": "Transcript already exists for this video"
                    }
        finally:
            conn.close()
        
        credentials = get_user_credentials(user_id)
        
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="No valid credentials found. Please log in again."
            )
            
        # Fetch transcript
        from services.youtube import fetch_video_transcript
        transcript_data = fetch_video_transcript(credentials, video_id)
        transcript = transcript_data.get("transcript")
        has_captions = transcript_data.get("has_captions", False)
        
        if not transcript:
            return {
                "video_id": video_id,
                "has_transcript": False,
                "error": transcript_data.get("error", "No transcript available for this video"),
                "has_captions": has_captions
            }
            
        # Store in database
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE youtube_videos
                    SET transcript = %s, has_captions = %s
                    WHERE video_id = %s
                """, (transcript, has_captions, video_id))
                conn.commit()
        finally:
            conn.close()
            
        return {
            "video_id": video_id,
            "has_transcript": True,
            "transcript_length": len(transcript),
            "has_captions": has_captions,
            "message": "Successfully fetched and stored transcript"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching transcript: {str(e)}")

@router.get("/youtube-data/{user_id}")
async def get_youtube_data(user_id: int, _: dict = Depends(get_user_from_session)):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all channel data for the user
        cursor.execute("""
            SELECT c.id, c.channel_id, c.kind, c.etag, c.title, c.description, 
                   c.custom_url, c.published_at, c.view_count, c.subscriber_count,
                   c.hidden_subscriber_count, c.video_count, c.thumbnail_url_default,
                   c.thumbnail_url_medium, c.thumbnail_url_high, c.uploads_playlist_id,
                   c.banner_url, c.privacy_status, c.is_linked, c.long_uploads_status,
       c.is_monetization_enabled, c.topic_ids, c.topic_categories,
       c.overall_good_standing, c.community_guidelines_good_standing,
       c.copyright_strikes_good_standing, c.content_id_claims_good_standing,
       c.branding_settings, c.audit_details, c.topic_details, c.status_details,
       c.created_at, c.updated_at
            FROM youtube_channels c
            WHERE c.user_id = %s
            ORDER BY c.title
        """, (user_id,))
            
        channels = []
        for channel_data in cursor.fetchall():
            channel = {
                "id": channel_data[0],
                "channel_id": channel_data[1],
                "kind": channel_data[2],
                "etag": channel_data[3],
                "title": channel_data[4],
                "description": channel_data[5],
                "custom_url": channel_data[6],
                "published_at": channel_data[7],
                "view_count": channel_data[8],
                "subscriber_count": channel_data[9],
                "hidden_subscriber_count": channel_data[10],
                "video_count": channel_data[11],
                "thumbnails": {
                    "default": channel_data[12],
                    "medium": channel_data[13],
                    "high": channel_data[14]
                },
                "uploads_playlist_id": channel_data[15],
                "optimization": {
                    "banner_url": channel_data[16],
                    "privacy_status": channel_data[17],
                    "is_linked": channel_data[18],
                    "long_uploads_status": channel_data[19],
                    "is_monetization_enabled": channel_data[20],
                    "topic_ids": channel_data[21],
                    "topic_categories": channel_data[22],
                    "channel_standing": {
                        "overall_good_standing": channel_data[23],
                        "community_guidelines_good_standing": channel_data[24],
                        "copyright_strikes_good_standing": channel_data[25],
                        "content_id_claims_good_standing": channel_data[26]
                    }
                },
                "raw_data": {
                    "branding_settings": channel_data[27],
                    "audit_details": channel_data[28],
                    "topic_details": channel_data[29],
                    "status_details": channel_data[30]
                },
                "created_at": channel_data[31],
                "updated_at": channel_data[32]
            }
            
            # Get videos data - join with youtube_channels to filter by user_id
            cursor.execute("""
                SELECT v.id, v.video_id, v.kind, v.etag, v.playlist_item_id, v.title, 
                       v.description, v.published_at, v.channel_title, v.tags, v.playlist_id,
                       v.position, v.thumbnail_url_default, v.thumbnail_url_medium, 
                       v.thumbnail_url_high, v.thumbnail_url_standard, v.thumbnail_url_maxres,
                       v.view_count, v.like_count, v.comment_count, v.duration, 
                       v.is_optimized, v.created_at, v.updated_at, v.queued_for_optimization, v.optimizations_completed
                FROM youtube_videos v
                JOIN youtube_channels c ON v.channel_id = c.id
                WHERE c.user_id = %s
                ORDER BY v.published_at DESC
            """, (user_id,))
            
            videos_data = cursor.fetchall()
            
            videos = []
            for video in videos_data:
                videos.append({
                    "id": video[0],
                    "video_id": video[1],
                    "kind": video[2],
                    "etag": video[3],
                    "playlist_item_id": video[4],
                    "title": video[5],
                    "description": video[6],
                    "published_at": video[7],
                    "channel_title": video[8],
                    "tags": video[9],
                    "playlist_id": video[10],
                    "position": video[11],
                    "thumbnails": {
                        "default": video[12],
                        "medium": video[13],
                        "high": video[14],
                        "standard": video[15],
                        "maxres": video[16]
                    },
                    "view_count": video[17],
                    "like_count": video[18],
                    "comment_count": video[19],
                    "duration": video[20],
                    "is_optimized": video[21],
                    "created_at": video[22],
                    "updated_at": video[23],
                    "queued_for_optimization": video[24],
                    "optimizations_completed": video[25]
                })
        
        conn.close()
        
        return {
            "status": "success",
            "data": {
                "channel": channel,
                "videos": videos,
                "video_count": len(videos)
            }
        }
        
    except Exception as e:
        logging.error(f"Error fetching YouTube data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching YouTube data: {str(e)}")

@router.post("/refresh-youtube-data/{user_id}")
async def refresh_youtube_data(user_id: int, background_tasks: BackgroundTasks):
    try:
        # Check if user exists
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        # We need credentials for this user
        # In a real app, you'd store refresh tokens securely and use them here
        # For now, we'll return a message that they need to log in again
        
        return {
            "status": "error", 
            "message": "For security reasons, please log in again to refresh your YouTube data"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error refreshing YouTube data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refreshing YouTube data: {str(e)}")

class VideoOptimizationStatus(BaseModel):
    is_optimized: bool

@router.put("/{video_id}/optimization-status")
async def update_video_optimization_status(video_id: str, status: VideoOptimizationStatus):
    """Update the optimization status of a video."""
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            # Update the is_optimized field for the video
            cursor.execute("""
                UPDATE youtube_videos
                SET is_optimized = %s, updated_at = NOW()
                WHERE video_id = %s
                RETURNING id
            """, (status.is_optimized, video_id))
            
            result = cursor.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Video not found")
            
            conn.commit()
        
        conn.close()
        
        return {
            "status": "success",
            "message": f"Video optimization status updated to {status.is_optimized}",
            "video_id": video_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating video optimization status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating video status: {str(e)}")

@router.get("/optimized-videos/{user_id}")
async def get_optimized_videos(user_id: int):
    """Get all optimized videos for a user."""
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT v.id, v.video_id, v.title, v.thumbnail_url_medium, 
                       v.view_count, v.like_count, v.published_at
                FROM youtube_videos v
                JOIN youtube_channels c ON v.channel_id = c.id
                WHERE c.user_id = %s AND v.is_optimized = TRUE
                ORDER BY v.published_at DESC
            """, (user_id,))
            
            videos_data = cursor.fetchall()
            
            videos = []
            for video in videos_data:
                videos.append({
                    "id": video[0],
                    "video_id": video[1],
                    "title": video[2],
                    "thumbnail": video[3],
                    "view_count": video[4],
                    "like_count": video[5],
                    "published_at": video[6]
                })
        
        conn.close()
        
        return {
            "status": "success",
            "data": {
                "videos": videos,
                "count": len(videos)
            }
        }
        
    except Exception as e:
        logging.error(f"Error fetching optimized videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching optimized videos: {str(e)}")

class ComprehensiveOptimizationResponse(BaseModel):
    original_title: str
    optimized_title: str
    original_description: str
    optimized_description: str
    original_tags: list[str] = []
    optimized_tags: list[str] = []
    optimization_notes: str

@router.post("/{video_id}/optimize-all", response_model=ComprehensiveOptimizationResponse)
async def optimize_video_comprehensive(video_id: str):
    """Generate comprehensive optimizations (title, description, tags) for a video using Claude 3.7."""
    try:
        conn = get_connection()
        
        # Get video data including transcript
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT title, description, tags, transcript, has_captions
                FROM youtube_videos
                WHERE video_id = %s
            """, (video_id,))
            
            video_data = cursor.fetchone()
            if not video_data:
                raise HTTPException(status_code=404, detail="Video not found")
            
            original_title = video_data[0]
            original_description = video_data[1] or ""
            original_tags = video_data[2] or []
            stored_transcript = video_data[3]  # May be None
            stored_has_captions = video_data[4] or False
            
            # Get the user's credentials to fetch the transcript if needed
            user_id = None
            try:
                cursor.execute("""
                    SELECT c.user_id
                    FROM youtube_videos v
                    JOIN youtube_channels c ON v.channel_id = c.id
                    WHERE v.video_id = %s
                """, (video_id,))
                user_result = cursor.fetchone()
                if user_result:
                    user_id = user_result[0]
            except Exception as e:
                logging.warning(f"Error getting user_id for video {video_id}: {e}")
            
            transcript = stored_transcript
            has_captions = stored_has_captions
            
            # If we don't have transcript stored, fetch on demand
            if not transcript and user_id:
                try:
                    credentials = get_user_credentials(user_id)
                    if credentials:
                        from services.youtube import fetch_video_transcript
                        transcript_data = fetch_video_transcript(credentials, video_id)
                        transcript = transcript_data.get("transcript")
                        has_captions = transcript_data.get("has_captions", False)
                        logging.info(f"Fetched transcript on demand for video {video_id}")
                except Exception as e:
                    logging.error(f"Error fetching transcript on demand: {e}")
            
            transcript_length = len(transcript) if transcript else 0
            logging.info(f"Video {video_id} has captions: {has_captions}, transcript length: {transcript_length}")
        
        conn.close()

        # Call optimization pipeline
        logging.info(f"Generating comprehensive optimization for video {video_id}")
        result = get_comprehensive_optimization(
            original_title=original_title,
            original_description=original_description,
            original_tags=original_tags,
            transcript=transcript,
            has_captions=has_captions
        )

        # Normalize LLM result
        if isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], dict):
                result = result[0]
            else:
                raise ValueError(f"Unexpected result format: {result}")
        elif not isinstance(result, dict):
            raise ValueError(f"Expected dict, got {type(result)}: {result}")

        # Normalize output fields
        normalized_result = {
            "original_title": str(result.get("original_title", "")),
            "optimized_title": str(result.get("optimized_title", "")),
            "original_description": str(result.get("original_description", "")),
            "optimized_description": str(result.get("optimized_description", "")),
            "original_tags": [str(tag) for tag in result.get("original_tags", []) if tag],
            "optimized_tags": [str(tag) for tag in result.get("optimized_tags", []) if tag],
            "optimization_notes": str(result.get("optimization_notes", "")),
        }

        # Ensure valid list fields
        normalized_result["original_tags"] = normalized_result["original_tags"] or []
        normalized_result["optimized_tags"] = normalized_result["optimized_tags"] or []

        # Persist results in DB
        try:
            update_conn = get_connection()
            with update_conn.cursor() as update_cursor:
                update_cursor.execute("""
                    UPDATE youtube_videos
                    SET transcript = COALESCE(%s, transcript),
                        has_captions = %s,
                        optimized_title = %s,
                        optimized_description = %s,
                        optimized_tags = %s
                    WHERE video_id = %s
                """, (
                    transcript,
                    has_captions,
                    normalized_result["optimized_title"],
                    normalized_result["optimized_description"],
                    normalized_result["optimized_tags"],
                    video_id
                ))
                update_conn.commit()
                logging.info(f"Updated optimizations in DB for video {video_id}")

        except Exception as store_e:
            logging.error(f"Error storing optimization results: {store_e}")
        finally:
            update_conn.close()

        return ComprehensiveOptimizationResponse(**normalized_result)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error generating comprehensive optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error optimizing video: {str(e)}")