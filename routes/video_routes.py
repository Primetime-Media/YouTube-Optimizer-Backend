# video_routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from services.video import (
    get_video_data,
    create_optimization,
    generate_video_optimization,
    get_video_optimizations
)
from services.optimizer import apply_optimization_to_youtube_video, get_optimization_status
from services.llm_optimization import should_optimize_video, enhanced_should_optimize_video
from services.youtube import fetch_video_timeseries_data, fetch_and_store_youtube_analytics
from services.competitor_analysis import get_competitor_analysis
from utils.auth import get_user_credentials, get_user_from_session, User
from utils.db import get_connection
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["Video Optimization"])

class VideoOptimizationStatusResponse(BaseModel):
    id: int
    video_id: int
    status: str
    progress: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Optional fields that will be included for completed optimizations
    original_title: Optional[str] = None
    optimized_title: Optional[str] = None
    original_description: Optional[str] = None
    optimized_description: Optional[str] = None
    original_tags: Optional[List[str]] = None
    optimized_tags: Optional[List[str]] = None
    optimization_notes: Optional[str] = None
    is_applied: Optional[bool] = None
    applied_at: Optional[datetime] = None

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
            
        # Create a new optimization record for this request
        optimization_id = create_optimization(video["id"])
        if optimization_id == 0:
            raise HTTPException(status_code=500, detail="Failed to create optimization record")
            
        # Run the optimization in the background with the pre-created optimization ID
        background_tasks.add_task(generate_video_optimization, video, optimization_id)
        
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