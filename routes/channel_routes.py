# channel_endpoints.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from services.channel import (
    get_channel_data,
    create_optimization,
    generate_channel_optimization,
    get_optimization_status,
    get_channel_optimizations,
)
from services.optimizer import apply_optimization_to_youtube_channel
from utils.db import get_connection
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/channel", tags=["Channel Optimization"])

class ChannelOptimizationStatusResponse(BaseModel):
    id: int
    channel_id: int
    status: str
    progress: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Optional fields that will be included for completed optimizations
    original_description: Optional[str] = None
    optimized_description: Optional[str] = None
    original_keywords: Optional[str] = None
    optimized_keywords: Optional[str] = None
    optimization_notes: Optional[str] = None
    is_applied: Optional[bool] = None
    applied_at: Optional[datetime] = None

class ComprehensiveChannelOptimizationResponse(BaseModel):
    id: int
    original_description: Optional[str] = None
    optimized_description: Optional[str] = None
    original_keywords: Optional[str] = None
    optimized_keywords: Optional[str] = None
    optimization_notes: Optional[str] = None
    status: str
    progress: int


@router.post("/{channel_id}/optimize")
async def optimize_channel(channel_id: int, background_tasks: BackgroundTasks):
    """
    Start an optimization job for a YouTube channel

    Args:
        channel_id: The database ID of the channel

    Returns:
        dict: Contains job ID and initial status
    """
    logger.info(f"Starting optimization for channel {channel_id}")
    try:
        # Fetch channel data from database
        channel = get_channel_data(channel_id)
        if not channel:
            logger.warning(f"Channel {channel_id} not found in database")
            raise HTTPException(status_code=404, detail="Channel not found")

        # Create a new optimization record for this request
        optimization_id = create_optimization(channel_id)
        if optimization_id == 0:
            logger.error(f"Failed to create optimization record for channel {channel_id}")
            raise HTTPException(status_code=500, detail="Failed to create optimization record")

        logger.info(f"Created optimization record {optimization_id} for channel {channel_id}")
        # Run the optimization in the background with the pre-created optimization ID
        background_tasks.add_task(generate_channel_optimization, channel, optimization_id)

        # Return the job ID and status for tracking
        return {
            "id": optimization_id,
            "channel_id": channel_id,
            "status": "pending",
            "progress": 0,
            "message": "Optimization started"
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error starting channel optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting optimization: {str(e)}")


@router.get("/optimization/{optimization_id}/status", response_model=ChannelOptimizationStatusResponse)
async def get_optimization_status_endpoint(optimization_id: int):
    """
    Get the current status of a channel optimization job

    Args:
        optimization_id: The ID of the optimization job

    Returns:
        ChannelOptimizationStatusResponse: Contains status and progress information
    """
    logger.info(f"Getting channel optimization status for ID {optimization_id}")
    try:
        status = get_optimization_status(optimization_id)
        if "error" in status:
            logger.warning(f"Channel optimization {optimization_id} not found: {status['error']}")
            raise HTTPException(status_code=404, detail=status["error"])
            
        return status
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting optimization status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")


@router.get("/{channel_id}/optimizations")
async def get_channel_optimizations_endpoint(channel_id: int, limit: int = 5):
    """
    Get recent optimization results for a channel

    Args:
        channel_id: The database ID of the channel
        limit: Maximum number of results to return (default: 5)

    Returns:
        list: Recent optimization records
    """
    logger.info(f"Getting optimizations for channel {channel_id} with limit {limit}")
    try:
        optimizations = get_channel_optimizations(channel_id, limit)
        logger.info(f"Found {len(optimizations)} optimizations for channel {channel_id}")
        return {
            "channel_id": channel_id,
            "optimizations": optimizations,
            "count": len(optimizations)
        }
    except Exception as e:
        logging.error(f"Error getting channel optimizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting optimizations: {str(e)}")


@router.post("/optimization/{optimization_id}/apply")
async def apply_optimization_to_channel(
    optimization_id: int, 
    only_description: bool = False, 
    only_keywords: bool = False
):
    """
    Apply an optimization directly to YouTube channel metadata

    Args:
        optimization_id: The ID of the optimization to apply
        only_description: If true, only update the description (query parameter)
        only_keywords: If true, only update the keywords (query parameter)

    Returns:
        dict: Results of the update operation
    """
    logger.info(f"Applying channel optimization {optimization_id} (description: {only_description}, keywords: {only_keywords})")
    try:
        # Get the user_id for this optimization's channel
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT c.user_id
                    FROM channel_optimizations o
                    JOIN youtube_channels c ON c.id = o.channel_id
                    WHERE o.id = %s
                """, (optimization_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Channel optimization {optimization_id} not found")
                    raise HTTPException(status_code=404, detail="Optimization not found")
                
                user_id = result[0]
                logger.info(f"Found user {user_id} for channel optimization {optimization_id}")
        finally:
            conn.close()

        # Apply the optimization using the shared function, passing the flags
        result = await apply_optimization_to_youtube_channel(
            optimization_id,
            user_id,
            only_description=only_description,
            only_keywords=only_keywords
        )
        
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