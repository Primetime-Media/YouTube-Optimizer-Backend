from fastapi import APIRouter, HTTPException, Depends, Header, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import os
from utils.db import get_connection
from services.scheduler import setup_monthly_optimization_for_channel, process_monthly_optimizations

# Setup logger for this module
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scheduler", tags=["Optimization Scheduler"])

CLOUD_SCHEDULER_SECRET = os.getenv("CLOUD_SCHEDULER_SECRET", "your-secret-key-here")

class ScheduleResponse(BaseModel):
    success: bool
    message: str
    schedule_id: Optional[int] = None
    channel_id: Optional[int] = None
    next_run: Optional[datetime] = None
    error: Optional[str] = None

@router.post("/channel/{channel_id}/monthly", response_model=ScheduleResponse)
async def setup_monthly_optimization(channel_id: int):
    """
    Set up automatic monthly optimization for a channel
    
    This will schedule the channel for monthly optimization that:
    1. Creates a new channel optimization
    2. Automatically applies it when complete
    
    Args:
        channel_id: The database ID of the channel
        
    Returns:
        ScheduleResponse: Result of the scheduling operation
    """
    result = setup_monthly_optimization_for_channel(channel_id, auto_apply=True)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to set up schedule"))
        
    return result

async def verify_cloud_scheduler_auth(request: Request, x_scheduler_auth: Optional[str] = Header(None)):
    """Verify Cloud Scheduler authentication"""
    logger.info("Verifying Cloud Scheduler authentication...")
    # Check secret key header
    if x_scheduler_auth != CLOUD_SCHEDULER_SECRET:
        logger.warning(f"Unauthorized scheduler access attempt. Provided Header: {x_scheduler_auth}")
        raise HTTPException(status_code=403, detail="Unauthorized access to scheduler job")
    
    # For Cloud Run authentication, could also verify:
    # - Cloud Run Job service identity
    # - JWT token in Authorization header
    # - Request source (User-Agent/IP)
    
    logger.info("Cloud Scheduler authentication successful.")
    return True

@router.post("/run-monthly-optimizations")
async def run_monthly_optimizations(authorized: bool = Depends(verify_cloud_scheduler_auth)):
    """
    Run monthly optimization jobs for all channels that are due
    
    This endpoint is meant to be called by Cloud Scheduler (every day or hour)
    to check if any channels are due for optimization.
    
    Required header: X-Scheduler-Auth with the secret key
    
    Returns:
        Dict: Results of the job execution
    """
    logger.info("Starting monthly optimization process via scheduler endpoint...")
    # Process all pending optimizations
    result = await process_monthly_optimizations()
    logger.info(f"Monthly optimization process finished. Result: {result}")
    return result

@router.get("/status/{channel_id}")
async def get_channel_optimization_status(channel_id: int):
    """
    Get the optimization schedule status for a channel
    
    Args:
        channel_id: The database ID of the channel
        
    Returns:
        Dict: Schedule status including past runs
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Get schedule info
                cursor.execute("""
                    SELECT 
                        id, is_active, auto_apply, last_run, next_run, created_at, updated_at
                    FROM channel_optimization_schedules
                    WHERE channel_id = %s
                """, (channel_id,))
                
                result = cursor.fetchone()
                if not result:
                    return {
                        "success": True,
                        "channel_id": channel_id,
                        "has_schedule": False,
                        "message": "No optimization schedule found for this channel"
                    }
                    
                schedule_id, is_active, auto_apply, last_run, next_run, created_at, updated_at = result
                
                # Get history of recent runs
                cursor.execute("""
                    SELECT 
                        id, start_time, end_time, status, optimization_id, applied, error_message
                    FROM scheduler_run_history
                    WHERE schedule_id = %s
                    ORDER BY start_time DESC
                    LIMIT 5
                """, (schedule_id,))
                
                history = []
                for history_row in cursor.fetchall():
                    run_id, start_time, end_time, status, optimization_id, applied, error_message = history_row
                    
                    history.append({
                        "id": run_id,
                        "start_time": start_time.isoformat() if start_time else None,
                        "end_time": end_time.isoformat() if end_time else None,
                        "status": status,
                        "optimization_id": optimization_id,
                        "applied": applied,
                        "error_message": error_message
                    })
                
                return {
                    "success": True,
                    "channel_id": channel_id,
                    "has_schedule": True,
                    "schedule_id": schedule_id,
                    "is_active": is_active,
                    "auto_apply": auto_apply,
                    "last_run": last_run.isoformat() if last_run else None,
                    "next_run": next_run.isoformat() if next_run else None,
                    "created_at": created_at.isoformat() if created_at else None,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                    "history": history
                }
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"Error getting optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")
