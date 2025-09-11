import logging
from typing import Dict, Optional, List

from utils.db import get_connection
from services.llm_optimization import get_comprehensive_optimization
from services.optimizer import apply_optimization_to_youtube_video
from services.thumbnail_optimizer import do_thumbnail_optimization

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
                SELECT id, title, description, tags, transcript, has_captions, like_count, comment_count, category_name, category_id
                FROM youtube_videos
                WHERE video_id = %s
            """, (video_id,))
            
            result = cursor.fetchone()
            if not result:
                return None
                
            db_id, title, description, tags, transcript, has_captions, like_count, comment_count, category_name, category_id = result

            return {
                "id": db_id,  # Database ID
                "video_id": video_id,  # YouTube video ID
                "title": title,
                "description": description,
                "tags": tags,
                "transcript": transcript,
                "has_captions": has_captions,
                "like_count": like_count,
                "comment_count": comment_count,
                "category_name": category_name,
                "category_id": category_id
            }
    except Exception as e:
        logger.error(f"Error fetching video data: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def create_optimization(db_video_id: int, optimization_step: int = 1) -> int:
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
                # Delete any existing optimization for this video and step, regardless of status
                cursor.execute("""
                    DELETE FROM video_optimizations
                    WHERE video_id = %s AND optimization_step = %s
                """, (db_video_id, optimization_step))
                logger.info(f"Deleted existing optimizations (any status) for video_id {db_video_id}, step {optimization_step}")

                # Always create a new optimization record
                cursor.execute("""
                    INSERT INTO video_optimizations (
                        video_id,
                        status,
                        progress,
                        optimization_step,
                        created_by
                    ) VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    db_video_id,
                    'pending',
                    0,
                    optimization_step,
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

def cleanup_generated_files(video_id: str) -> None:
    """
    Clean up generated video file and extracted thumbnails after optimization
    
    Args:
        video_id: The YouTube video ID used for file naming
    """
    import os
    import glob
    from pathlib import Path
    
    try:
        # Clean up video file
        video_filename = f"{video_id}.mp4"
        if os.path.exists(video_filename):
            os.remove(video_filename)
            logger.info(f"Cleaned up video file: {video_filename}")
        
        # Clean up extracted thumbnails directory
        extracted_thumbnails_dir = "extracted_thumbnails"
        if os.path.exists(extracted_thumbnails_dir):
            # Find all thumbnail files related to this video
            thumbnail_pattern = f"{extracted_thumbnails_dir}/orig_{video_id}_*.jpg"
            thumbnail_files = glob.glob(thumbnail_pattern)
            
            for thumbnail_file in thumbnail_files:
                try:
                    os.remove(thumbnail_file)
                    logger.info(f"Cleaned up thumbnail file: {thumbnail_file}")
                except OSError as e:
                    logger.warning(f"Failed to delete thumbnail file {thumbnail_file}: {e}")
            
            # Also clean up any thumbnails directory files that might not match the pattern
            try:
                thumbnails_pattern = f"{extracted_thumbnails_dir}/*{video_id}*.jpg"
                additional_files = glob.glob(thumbnails_pattern)
                for file_path in additional_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleaned up additional thumbnail file: {file_path}")
            except Exception as e:
                logger.warning(f"Error cleaning up additional thumbnail files: {e}")
        
        
        logger.info(f"Cleanup completed for video {video_id}")
        
    except Exception as e:
        logger.error(f"Error during cleanup for video {video_id}: {str(e)}")

def generate_video_optimization(
        video: Dict,
        user_id: int,
        optimization_id: int,
        optimization_decision_data: Optional[Dict] = None,
        analytics_data: Optional[Dict] = None,
        competitor_analytics_data: Optional[Dict] = None,
        apply_optimization: bool = False,
        prev_optimizations: Optional[List[Dict]] = None
) -> Dict:
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
    like_count = video.get("like_count", 0)
    comment_count = video.get("comment_count", 0)
    category_name = video.get("category_name", "")
    video_id = video.get("video_id", "")  # YouTube video ID for cleanup
    
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
        # PARALLELIZE THUMBNAIL AND CONTENT OPTIMIZATION
        import asyncio
        import concurrent.futures
        
        def run_thumbnail_optimization():
            try:
                return do_thumbnail_optimization(
                    video_id=video.get("video_id"),
                    original_title=title,
                    original_description=description,
                    original_tags=tags,
                    transcript=transcript,
                    competitor_analytics_data=competitor_analytics_data,
                    category_name=category_name,
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(f"Thumbnail optimization failed for video {video.get('video_id')}: {str(e)}")
                return None
        
        def run_content_optimization():
            return get_comprehensive_optimization(
                original_title=title,
                original_description=description,
                original_tags=tags,
                transcript=transcript,
                has_captions=has_captions,
                like_count=like_count,
                comment_count=comment_count,
                optimization_decision_data=optimization_decision_data or {},
                analytics_data=analytics_data or {},
                competitor_analytics_data=competitor_analytics_data or {},
                category_name=category_name,
                user_id=user_id,
                prev_optimizations=prev_optimizations or []
            )
        
        # Run both optimizations in parallel
        logger.info("Starting parallel optimization processes...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            thumbnail_future = executor.submit(run_thumbnail_optimization)
            content_future = executor.submit(run_content_optimization)
            
            # Wait for both to complete
            thumbnail_optimization_result = thumbnail_future.result()
            result = content_future.result()
        
        logger.info("Parallel optimization processes completed")

        if not result:
            raise RuntimeError("LLM optimization failed to generate results")

        best_optimization = max(result, key=lambda x: x['optimization_score'])

        best_optimization.update({
            'original_title': title,
            'original_description': description,
            'original_tags': tags
        })

        if thumbnail_optimization_result and isinstance(thumbnail_optimization_result, dict):
            best_optimization.update({"thumbnail_optimization_file": thumbnail_optimization_result.get("optimized_thumbnail", {}).get("optimized_thumbnail")})

        # Update progress - LLM processing complete
        update_optimization_progress(optimization_id, 75)
        
        logger.info(f"Video optimization completed successfully for {title}")
        
        # Store the results in the existing optimization record
        optimization_results_stored = store_optimization_results(optimization_id, db_video_id, best_optimization)

        if not optimization_results_stored:
            raise RuntimeError("Failed to store optimization results in database")
        
        # Add optimization ID to the result
        best_optimization["id"] = optimization_id

        import asyncio

        optimization_applied = False
        if apply_optimization:
            optimization_result = apply_optimization_to_youtube_video(optimization_id, user_id, thumbnail_file=best_optimization.get("thumbnail_optimization_file"))
            optimization_applied = optimization_result.get('sucesss')

        return {
            "id": optimization_id,
            "original_title": best_optimization.get("original_title", ""),
            "optimized_title": best_optimization.get("optimized_title", ""),
            "original_description": best_optimization.get("original_description", ""),
            "optimized_description": best_optimization.get("optimized_description", ""),
            "original_tags": best_optimization.get("original_tags", []),
            "optimized_tags": best_optimization.get("optimized_tags", []),
            "optimization_notes": best_optimization.get("optimization_notes", ""),
            "is_applied": optimization_applied
        }
    except Exception as e:
        logger.error(f"Error during video optimization: {str(e)}")
        update_optimization_progress(optimization_id, 0, "failed")
        return {
            "error": f"Optimization failed: {str(e)}",
            "id": optimization_id,
            "is_applied": False
        }
    finally:
        # Clean up generated files after optimization completes (success or failure)
        if video_id:
            cleanup_generated_files(video_id)

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
                WHERE video_id = %s AND is_applied = TRUE
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
