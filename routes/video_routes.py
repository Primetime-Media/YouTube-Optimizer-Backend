# video_routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from services.video import (
    get_video_data,
    create_optimization,
    generate_video_optimization,
    get_optimization_status,
    get_video_optimizations
)
from services.optimizer import apply_optimization_to_youtube_video
from utils.auth import get_user_credentials, get_user_from_session
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


@router.post("/{youtube_video_id}/optimize")
async def optimize_video(youtube_video_id: str, background_tasks: BackgroundTasks):
    """
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
        result = await apply_optimization_to_youtube_video(optimization_id, user.id)
        
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