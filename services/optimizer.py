from typing import Dict, Optional
import logging
from services.channel import get_optimization_status as get_channel_optimization_status
from services.channel import apply_channel_optimization
from services.youtube import update_youtube_channel_branding, build_youtube_client, update_youtube_video
from utils.auth import get_user_credentials
from utils.db import get_connection

logger = logging.getLogger(__name__)

async def apply_optimization_to_youtube_channel(
    optimization_id: int, 
    user_id: int,
    only_description: bool = False,
    only_keywords: bool = True
) -> Dict:
    """
    Apply an optimization to a YouTube channel by making the actual API calls
    
    Args:
        optimization_id: The ID of the optimization to apply
        user_id: The ID of the user who owns the channel
        only_description: If true, only update the description
        only_keywords: If true, only update the keywords
        
    Returns:
        dict: Results of the update operation with format:
            {
                "success": bool,
                "message": str,
                "error": Optional[str]
            }
    """
    logger.info(f"Applying optimization {optimization_id} to YouTube channel")
    
    try:
        # Get the optimization record
        optimization = get_channel_optimization_status(optimization_id)
        if "error" in optimization:
            logger.error(f"Error retrieving optimization {optimization_id}: {optimization['error']}")
            return {
                "success": False,
                "error": optimization["error"],
                "message": "Failed to retrieve optimization"
            }

        if optimization["status"] != "completed":
            error_msg = f"Cannot apply incomplete optimization (status: {optimization['status']})"
            logger.error(f"{error_msg} for optimization {optimization_id}")
            return {
                "success": False,
                "error": error_msg,
                "message": "Optimization not ready to apply"
            }

        # Get user credentials
        logger.info(f"Retrieving credentials for user {user_id}")
        credentials = get_user_credentials(user_id)

        if not credentials:
            error_msg = f"No valid credentials for user {user_id}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "Authentication failed"
            }

        # Build YouTube API client
        logger.info("Building YouTube API client")
        youtube_client = build_youtube_client(credentials)

        # Update the channel on YouTube
        logger.info(f"Updating YouTube channel with optimization {optimization_id}")
        update_result = update_youtube_channel_branding(
            youtube_client,
            optimization["channel_id"],
            optimization["optimized_description"],
            optimization["optimized_keywords"],
            optimization_id=optimization_id,
            only_description=only_description,
            only_keywords=only_keywords
        )

        if update_result["success"]:
            # Mark optimization as applied in database
            logger.info(f"YouTube update successful, marking optimization {optimization_id} as applied")
            apply_channel_optimization(optimization_id)

            return {
                "success": True,
                "message": "Channel updated successfully",
                "optimization_id": optimization_id,
                "channel_id": optimization["channel_id"]
            }
        else:
            error_msg = update_result.get('error', 'Unknown error')
            logger.error(f"Failed to update YouTube channel: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "message": "Failed to update YouTube channel"
            }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error applying optimization to YouTube: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "message": "Error applying optimization"
        }

def apply_optimization_to_youtube_video(
    optimization_id: int, 
    user_id: int,
    only_title: bool = False,
    only_description: bool = False,
    only_tags: bool = False,
    thumbnail_file: str = None
) -> Dict:
    """
    Apply an optimization to a YouTube video.
    
    Args:
        optimization_id: The ID of the optimization to apply
        user_id: The ID of the user who owns the video
        only_title: If true, only update the title
        only_description: If true, only update the description
        only_tags: If true, only update the tags
        
    Returns:
        dict: Results of the update operation with format:
            {
                "success": bool,
                "message": str,
                "error": Optional[str]
            }
    """
    logger.info(f"Applying optimization {optimization_id} to YouTube video")
    
    try:
        # Get the optimization record
        optimization = get_optimization_status(optimization_id)
        if "error" in optimization:
            logger.error(f"Error retrieving optimization {optimization_id}: {optimization['error']}")
            return {
                "success": False,
                "error": optimization["error"],
                "message": "Failed to retrieve optimization"
            }

        if optimization["status"] != "completed":
            error_msg = f"Cannot apply incomplete optimization (status: {optimization['status']})"
            logger.error(f"{error_msg} for optimization {optimization_id}")
            return {
                "success": False,
                "error": error_msg,
                "message": "Optimization not ready to apply"
            }

        # Get user credentials
        logger.info(f"Retrieving credentials for user {user_id}")
        credentials = get_user_credentials(user_id)

        if not credentials:
            error_msg = f"No valid credentials for user {user_id}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "Authentication failed"
            }

        # Get the video_id from database using the db_video_id
        conn = get_connection()
        youtube_video_id = None
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT video_id
                    FROM youtube_videos
                    WHERE id = %s
                """, (optimization["video_id"],))
                
                result = cursor.fetchone()
                if not result:
                    error_msg = f"Video with ID {optimization['video_id']} not found in database"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "message": "Video not found"
                    }
                
                youtube_video_id = result[0]
        finally:
            conn.close()

        # Build YouTube API client
        logger.info("Building YouTube API client")
        youtube_client = build_youtube_client(credentials)

        # Update the video on YouTube
        logger.info(f"Updating YouTube video with optimization {optimization_id}")
        update_result = update_youtube_video(
            youtube_client,
            youtube_video_id,
            optimization["optimized_title"],
            optimization["optimized_description"],
            optimization["optimized_tags"],
            optimization_id=optimization_id,
            only_title=only_title,
            only_description=only_description,
            only_tags=only_tags,
            thumbnail_file=thumbnail_file
        )

        if update_result["success"]:
            # Mark optimization as applied in database
            logger.info(f"YouTube update successful, marking optimization {optimization_id} as applied")
            apply_video_optimization(optimization_id)

            return {
                "success": True,
                "message": "Video updated successfully",
                "optimization_id": optimization_id,
                "video_id": youtube_video_id
            }
        else:
            error_msg = update_result.get('error', 'Unknown error')
            logger.error(f"Failed to update YouTube video: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "message": "Failed to update YouTube video"
            }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error applying optimization to YouTube video: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "message": "Error applying optimization"
        }


def apply_video_optimization(optimization_id: int) -> bool:
    """
    Mark a video optimization as applied

    Args:
        optimization_id: The ID of the optimization to apply

    Returns:
        bool: True if successfully applied, False otherwise
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE video_optimizations
                SET is_applied = TRUE, applied_at = NOW(), updated_at = NOW()
                WHERE id = %s
                RETURNING video_id
            """, (optimization_id,))

            result = cursor.fetchone()
            if not result:
                logger.warning(f"No optimization found with ID {optimization_id}")
                return False

            video_id = result[0]
            conn.commit()

            logger.info(f"Applied video optimization {optimization_id} for video with DB ID {video_id}")
            return True

    except Exception as e:
        logger.error(f"Error applying video optimization: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_optimization_status(optimization_id: int) -> Dict:
    """
    Get the current status of a video optimization

    Args:
        optimization_id: The ID of the optimization record

    Returns:
        dict: Status information including progress percentage
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    video_id,
                    status,
                    progress,
                    created_at,
                    updated_at
                FROM video_optimizations
                WHERE id = %s
            """, (optimization_id,))

            result = cursor.fetchone()
            if not result:
                logger.warning(f"No optimization found with ID {optimization_id}")
                return {
                    "error": "Optimization not found",
                    "id": optimization_id
                }

            video_id, status, progress, created_at, updated_at = result

            status_info = {
                "id": optimization_id,
                "video_id": video_id,
                "status": status,
                "progress": progress,
                "created_at": created_at,
                "updated_at": updated_at
            }

            # If optimization is complete, include the full results
            if status == "completed" and progress == 100:
                # Get the complete optimization data
                cursor.execute("""
                    SELECT 
                        original_title,
                        optimized_title,
                        original_description,
                        optimized_description,
                        original_tags,
                        optimized_tags,
                        optimization_notes,
                        is_applied,
                        applied_at
                    FROM video_optimizations
                    WHERE id = %s
                """, (optimization_id,))

                complete_data = cursor.fetchone()
                if complete_data:
                    (
                        original_title,
                        optimized_title,
                        original_description,
                        optimized_description,
                        original_tags,
                        optimized_tags,
                        optimization_notes,
                        is_applied,
                        applied_at
                    ) = complete_data

                    # Add result data to the response
                    status_info.update({
                        "original_title": original_title,
                        "optimized_title": optimized_title,
                        "original_description": original_description,
                        "optimized_description": optimized_description,
                        "original_tags": original_tags,
                        "optimized_tags": optimized_tags,
                        "optimization_notes": optimization_notes,
                        "is_applied": is_applied,
                        "applied_at": applied_at
                    })

            return status_info

    except Exception as e:
        logger.error(f"Error getting optimization status: {str(e)}")
        return {
            "error": f"Error getting optimization status: {str(e)}",
            "id": optimization_id
        }
    finally:
        if conn:
            conn.close()