import logging
from typing import Dict, Optional, List
from utils.db import get_connection
from services.llm_optimization import get_comprehensive_optimization

logger = logging.getLogger(__name__)

def get_video_data(video_id: str) -> Optional[Dict]:
    """
    Retrieve video data from the database using the video_id
    
    Args:
        video_id: The YouTube video ID
        
    Returns:
        dict: The video data or None if not found
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, description, tags, transcript, has_captions
                FROM youtube_videos
                WHERE video_id = %s
            """, (video_id,))
            
            result = cursor.fetchone()
            if not result:
                return None
                
            db_id, title, description, tags, transcript, has_captions = result
            
            return {
                "id": db_id,  # Database ID
                "video_id": video_id,  # YouTube video ID
                "title": title,
                "description": description,
                "tags": tags,
                "transcript": transcript,
                "has_captions": has_captions
            }
    except Exception as e:
        logger.error(f"Error fetching video data: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def create_optimization(db_video_id: int) -> int:
    """
    Create a new optimization record for a video
    
    Args:
        db_video_id: The database ID of the video
        
    Returns:
        int: The ID of the new optimization record
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Always create a new optimization record
                cursor.execute("""
                    INSERT INTO video_optimizations (
                        video_id,
                        status,
                        progress,
                        created_by
                    ) VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (
                    db_video_id,
                    'pending',
                    0,
                    'youtube-optimizer-system'
                ))
                
                optimization_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"Created new optimization record {optimization_id} for video with DB ID {db_video_id}")
                return optimization_id
        except Exception as e:
            # Roll back transaction on error
            if not conn.autocommit:
                conn.rollback()
            raise
    except Exception as e:
        logger.error(f"Error creating video optimization record: {str(e)}")
        return 0
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def update_optimization_progress(optimization_id: int, progress: int, status: str = None) -> bool:
    """
    Update the progress of a video optimization
    
    Args:
        optimization_id: The ID of the optimization record
        progress: Progress percentage (0-100)
        status: Optional new status
        
    Returns:
        bool: Success status
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            if status:
                cursor.execute("""
                    UPDATE video_optimizations
                    SET progress = %s, status = %s, updated_at = NOW()
                    WHERE id = %s
                """, (progress, status, optimization_id))
            else:
                cursor.execute("""
                    UPDATE video_optimizations
                    SET progress = %s, updated_at = NOW()
                    WHERE id = %s
                """, (progress, optimization_id))
            
            conn.commit()
            logger.info(f"Updated video optimization {optimization_id} progress to {progress}%")
            return True
            
    except Exception as e:
        logger.error(f"Error updating video optimization progress: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def generate_video_optimization(video: Dict, optimization_id: int) -> Dict:
    """
    Generate optimized title, description, and tags for a YouTube video
    
    Args:
        video: Dict containing video data with title, description, tags, and transcript
        optimization_id: The ID of the pre-created optimization record to use
        
    Returns:
        dict: Contains optimized title, description, and tags
    """
    logger.info(f"Starting video optimization for video: {video.get('title', 'Unknown')} with optimization ID: {optimization_id}")
    
    title = video.get("title", "")
    description = video.get("description", "")
    tags = video.get("tags", [])
    transcript = video.get("transcript")
    has_captions = video.get("has_captions", False)
    db_video_id = video.get("id")
    
    # Verify optimization record exists
    if optimization_id == 0:
        logger.error(f"Invalid optimization ID for video {db_video_id}")
        return {
            "error": "Invalid optimization ID",
            "id": 0
        }
    
    # Update status to in_progress
    update_optimization_progress(optimization_id, 10, "in_progress")
    
    # Safely check transcript length
    transcript_length = 0
    if transcript is not None:
        transcript_length = len(transcript)
    logger.debug(f"Video data summary - Title: {title}, Description length: {len(description)}, Tags: {len(tags)}, Transcript length: {transcript_length}")
    
    # Update progress - data extraction complete
    update_optimization_progress(optimization_id, 25)
    
    try:
        # Call the optimization function with all data
        result = get_comprehensive_optimization(
            original_title=title,
            original_description=description,
            original_tags=tags,
            transcript=transcript,
            has_captions=has_captions
        )
        
        # Update progress - LLM processing complete
        update_optimization_progress(optimization_id, 75)
        
        logger.info(f"Video optimization completed successfully for {title}")
        
        # Store the results in the existing optimization record
        store_optimization_results(optimization_id, db_video_id, result)
        
        # Add optimization ID to the result
        result["id"] = optimization_id
        
        return {
            "id": optimization_id,
            "original_title": result.get("original_title", ""),
            "optimized_title": result.get("optimized_title", ""),
            "original_description": result.get("original_description", ""),
            "optimized_description": result.get("optimized_description", ""),
            "original_tags": result.get("original_tags", []),
            "optimized_tags": result.get("optimized_tags", []),
            "optimization_notes": result.get("optimization_notes", "")
        }
    except Exception as e:
        logger.error(f"Error during video optimization: {str(e)}")
        update_optimization_progress(optimization_id, 0, "failed")
        return {
            "error": f"Optimization failed: {str(e)}",
            "id": optimization_id
        }

def store_optimization_results(optimization_id: int, db_video_id: int, optimization_data: Dict) -> bool:
    """
    Store optimization results in an existing optimization record
    
    Args:
        optimization_id: The ID of the existing optimization record
        db_video_id: The database ID of the video
        optimization_data: Dict containing optimization results
        
    Returns:
        bool: Success status
    """
    try:
        logger.info(f"Storing optimization results for optimization ID {optimization_id}, video {db_video_id}")
        
        # Update the existing record with optimization results
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE video_optimizations
                SET 
                    original_title = %s,
                    optimized_title = %s,
                    original_description = %s,
                    optimized_description = %s,
                    original_tags = %s,
                    optimized_tags = %s,
                    optimization_notes = %s,
                    status = 'completed',
                    progress = 100,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id
            """, (
                optimization_data.get("original_title", ""),
                optimization_data.get("optimized_title", ""),
                optimization_data.get("original_description", ""),
                optimization_data.get("optimized_description", ""),
                optimization_data.get("original_tags", []),
                optimization_data.get("optimized_tags", []),
                optimization_data.get("optimization_notes", ""),
                optimization_id
            ))
            
            result = cursor.fetchone()
            if not result:
                logger.error(f"Failed to update optimization record {optimization_id}")
                return False
                
            conn.commit()
            
            # Update the video record to mark it as optimized
            cursor.execute("""
                UPDATE youtube_videos
                SET 
                    is_optimized = TRUE,
                    last_optimized_at = NOW(),
                    last_optimization_id = %s
                WHERE id = %s
            """, (optimization_id, db_video_id))
            conn.commit()
            
            logger.info(f"Successfully stored results for optimization ID {optimization_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error storing optimization results: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_video_optimizations(db_video_id: int, limit: int = 5) -> List[Dict]:
    """
    Get the most recent optimizations for a video
    
    Args:
        db_video_id: The database ID of the video
        limit: Maximum number of records to return (default: 5)
        
    Returns:
        list: Recent optimization records for the video
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    id, 
                    original_title,
                    optimized_title,
                    original_description,
                    optimized_description,
                    original_tags,
                    optimized_tags,
                    optimization_notes,
                    is_applied,
                    applied_at,
                    status,
                    progress,
                    created_at
                FROM video_optimizations
                WHERE video_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (db_video_id, limit))
            
            optimizations = []
            for row in cursor.fetchall():
                (
                    optimization_id, 
                    original_title,
                    optimized_title,
                    original_description,
                    optimized_description,
                    original_tags,
                    optimized_tags,
                    optimization_notes,
                    is_applied,
                    applied_at,
                    status,
                    progress,
                    created_at
                ) = row
                
                optimizations.append({
                    "id": optimization_id,
                    "original_title": original_title,
                    "optimized_title": optimized_title,
                    "original_description": original_description,
                    "optimized_description": optimized_description,
                    "original_tags": original_tags,
                    "optimized_tags": optimized_tags,
                    "optimization_notes": optimization_notes,
                    "is_applied": is_applied,
                    "applied_at": applied_at,
                    "status": status,
                    "progress": progress,
                    "created_at": created_at
                })
            
            return optimizations
    except Exception as e:
        logger.error(f"Error getting video optimizations: {str(e)}")
        return []
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