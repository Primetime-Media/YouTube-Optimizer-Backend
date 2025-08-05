import logging
import asyncio
from datetime import datetime, timedelta
from utils.db import get_connection
from services.channel import (
    get_channel_data,
    create_optimization,
    generate_channel_optimization,
    get_optimization_status,
    apply_channel_optimization
)
from services.optimizer import apply_optimization_to_youtube_channel

logger = logging.getLogger(__name__)

def initialize_scheduler():
    """
    Initialize the scheduler system.
    
    For Cloud Run compatibility, we don't use APScheduler.
    Instead, this function just sets up any required resources.
    Actual scheduling is handled by Cloud Scheduler calling an endpoint.
    """
    logger.info("Scheduler system initialized for Cloud Run")

def setup_monthly_optimization_for_channel(channel_id, auto_apply=True):
    """
    Create a monthly optimization schedule for a channel
    
    Args:
        channel_id: The database ID of the channel
        auto_apply: Whether to automatically apply optimizations
        
    Returns:
        dict: Result of the operation
    """
    try:
        # Check if channel exists
        conn = get_connection()
        schedule_id = None
        
        try:
            with conn.cursor() as cursor:
                # Verify channel exists
                cursor.execute("SELECT id FROM youtube_channels WHERE id = %s", (channel_id,))
                if not cursor.fetchone():
                    return {
                        "success": False,
                        "error": f"Channel {channel_id} not found"
                    }
                
                # Check if schedule already exists
                cursor.execute("""
                    SELECT id, is_active 
                    FROM channel_optimization_schedules 
                    WHERE channel_id = %s
                """, (channel_id,))
                
                existing = cursor.fetchone()
                
                if existing:
                    schedule_id, is_active = existing
                    
                    # If already active, just return success
                    if is_active:
                        return {
                            "success": True,
                            "schedule_id": schedule_id,
                            "message": "Monthly optimization already scheduled",
                            "channel_id": channel_id
                        }
                    
                    # Reactivate if exists but inactive
                    cursor.execute("""
                        UPDATE channel_optimization_schedules
                        SET 
                            is_active = TRUE,
                            auto_apply = %s,
                            updated_at = NOW(),
                            next_run = NOW() + INTERVAL '30 days'
                        WHERE id = %s
                        RETURNING id
                    """, (auto_apply, schedule_id))
                    
                    schedule_id = cursor.fetchone()[0]
                    conn.commit()
                else:
                    # Create new schedule record
                    cursor.execute("""
                        INSERT INTO channel_optimization_schedules
                        (channel_id, auto_apply, is_active, next_run)
                        VALUES (%s, %s, TRUE, NOW() + INTERVAL '30 days')
                        RETURNING id
                    """, (channel_id, auto_apply))
                    
                    schedule_id = cursor.fetchone()[0]
                    conn.commit()
        finally:
            conn.close()
            
        # Calculate next run time (30 days from now)
        next_run = datetime.now() + timedelta(days=30)
            
        return {
            "success": True,
            "schedule_id": schedule_id,
            "channel_id": channel_id,
            "next_run": next_run.isoformat(),
            "message": "Monthly optimization scheduled successfully"
        }
        
    except Exception as e:
        logger.error(f"Error setting up monthly optimization: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def record_scheduler_run(schedule_id):
    """Record the start of a scheduler run"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO scheduler_run_history
                (schedule_id, start_time, status)
                VALUES (%s, NOW(), 'running')
                RETURNING id
            """, (schedule_id,))
            run_id = cursor.fetchone()[0]
            conn.commit()
            return run_id
    finally:
        conn.close()

def update_scheduler_run(run_id, optimization_id=None, applied=False):
    """Update a scheduler run record"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            updates = []
            params = []
            
            if optimization_id is not None:
                updates.append("optimization_id = %s")
                params.append(optimization_id)
                
            if applied:
                updates.append("applied = TRUE")
                
            if updates:
                cursor.execute(f"""
                    UPDATE scheduler_run_history
                    SET {", ".join(updates)}
                    WHERE id = %s
                """, params + [run_id])
                conn.commit()
    finally:
        conn.close()

def complete_scheduler_run(run_id, status="completed", error_message=None):
    """Mark a scheduler run as complete"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE scheduler_run_history
                SET 
                    end_time = NOW(),
                    status = %s,
                    error_message = %s
                WHERE id = %s
            """, (status, error_message, run_id))
            conn.commit()
    finally:
        conn.close()

async def process_monthly_optimizations():
    """
    Process all channels due for optimization.
    This function is meant to be called by a Cloud Run Job.
    """
    logger.info("Starting scheduled monthly optimization processing")
    processed_count = 0
    failed_count = 0
    
    conn = get_connection()
    try:
        # Get channels due for optimization
        logger.info("Querying database for channels due for optimization...")
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT s.id, s.channel_id, s.auto_apply, c.user_id
                FROM channel_optimization_schedules s
                JOIN youtube_channels c ON c.id = s.channel_id
                LEFT JOIN channel_optimizations co ON co.channel_id = s.channel_id AND co.status = 'completed'
                WHERE s.is_active = TRUE AND 
                      (s.next_run IS NULL OR s.next_run <= NOW()) AND
                      co.id IS NULL
            """)
            
            schedules = cursor.fetchall()
            
        logger.info(f"Found {len(schedules)} channels due for optimization")
        
        # Process each schedule
        for schedule_data in schedules:
            schedule_id, channel_id, auto_apply, user_id = schedule_data
            run_id = None # Initialize run_id
            try:
                # Record that we're starting this run
                run_id = record_scheduler_run(schedule_id)
                logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Starting processing.")
                
                # Get channel data
                logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Fetching channel data...")
                channel = get_channel_data(channel_id)
                if not channel:
                    logger.error(f"[Schedule {schedule_id}, Channel {channel_id}] Channel data not found")
                    complete_scheduler_run(run_id, "error", f"Channel {channel_id} not found")
                    failed_count += 1
                    continue
                    
                # Create optimization record
                logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Creating optimization record...")
                optimization_id = create_optimization(channel_id)
                if optimization_id == 0:
                    logger.error(f"[Schedule {schedule_id}, Channel {channel_id}] Failed to create optimization record")
                    complete_scheduler_run(run_id, "error", "Failed to create optimization record")
                    failed_count += 1
                    continue
                
                # Update run record with optimization ID
                update_scheduler_run(run_id, optimization_id=optimization_id)
                logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Created optimization record ID: {optimization_id}")
                
                # Generate the optimization
                logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Generating optimization {optimization_id}...")
                # NOTE: generate_channel_optimization is currently synchronous.
                # If it's long-running, this could block processing of other schedules.
                # Consider making it async or running in a separate thread/process for true concurrency.
                optimization_result = generate_channel_optimization(channel, optimization_id) # Assume it returns a dict
                
                # Check if generate_channel_optimization returned an error
                if isinstance(optimization_result, dict) and "error" in optimization_result:
                    error_msg = optimization_result["error"]
                    logger.error(f"[Schedule {schedule_id}, Channel {channel_id}] Failed to generate optimization {optimization_id}: {error_msg}")
                    complete_scheduler_run(run_id, "error", f"Failed to generate optimization: {error_msg}")
                    failed_count += 1
                    continue
                
                logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Optimization {optimization_id} generated successfully.")
                
                # Apply if auto-apply is enabled
                if auto_apply:
                    logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Auto-apply enabled for optimization {optimization_id}. Applying...")
                    # NOTE: Applying waits synchronously. Consider if this is desired.
                    
                    # Apply the optimization using the shared function
                    apply_result = await apply_optimization_to_youtube_channel(optimization_id, user_id)
                    
                    if not apply_result["success"]:
                        error_msg = apply_result.get('error')
                        logger.error(f"[Schedule {schedule_id}, Channel {channel_id}] Failed to apply optimization {optimization_id}: {error_msg}")
                        complete_scheduler_run(run_id, "error", f"Failed to apply optimization: {error_msg}")
                        failed_count += 1
                        continue

                    update_scheduler_run(run_id, applied=True)
                    logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Successfully applied optimization {optimization_id}.")
                
                # Update last run time and schedule next run
                logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Updating schedule's next run time.")
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE channel_optimization_schedules
                        SET last_run = NOW(),
                            next_run = NOW() + INTERVAL '30 days'
                        WHERE id = %s
                    """, (schedule_id,))
                    conn.commit()
                
                # Complete run record
                complete_scheduler_run(run_id)
                processed_count += 1
                logger.info(f"[Schedule {schedule_id}, Channel {channel_id}] Completed processing for optimization {optimization_id}.")
                
            except Exception as e:
                logger.error(f"[Schedule {schedule_id}, Channel {channel_id}] Error processing: {e}", exc_info=True) # Add stack trace
                failed_count += 1
                if run_id: # Check if run_id was created before error
                    try:
                        complete_scheduler_run(run_id, "error", str(e))
                    except Exception as inner_e:
                        logger.error(f"[Schedule {schedule_id}, Channel {channel_id}] Failed to mark run {run_id} as error: {inner_e}")
    except Exception as e:
        logger.error(f"Error querying or connecting during monthly optimization processing: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
    
    logger.info(f"Finished scheduled monthly optimization processing. Processed: {processed_count}, Failed: {failed_count}")
    return {"success": True, "message": f"Monthly optimization process completed. Processed: {processed_count}, Failed: {failed_count}"}