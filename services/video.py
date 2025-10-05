# services/video.py
"""
Production-Ready Video Service
Handles video optimization operations with comprehensive error handling
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from decimal import Decimal
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from utils.db import get_connection
from services.llm_optimization import get_comprehensive_optimization
from services.thumbnail_optimizer import generate_thumbnail_recommendation
from services.youtube import YouTubeService
from config import settings

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Progress tracking constants
PROGRESS_STARTED = 0
PROGRESS_ANALYZING = 10
PROGRESS_OPTIMIZING = 25
PROGRESS_GENERATING_THUMBNAIL = 50
PROGRESS_APPLYING = 75
PROGRESS_COMPLETED = 100

# Thread pool for parallel operations
executor = ThreadPoolExecutor(max_workers=5)


# ============================================================================
# OPTIMIZATION MANAGEMENT
# ============================================================================

def create_optimization(
    video_id: int,
    user_id: int,
    optimization_type: str = 'comprehensive',
    auto_apply: bool = False
) -> Optional[int]:
    """
    Create a new optimization record
    
    Args:
        video_id: Database video ID
        user_id: User ID requesting optimization
        optimization_type: Type of optimization
        auto_apply: Whether to automatically apply changes
        
    Returns:
        Optimization ID or None on error
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO video_optimizations (
                    video_id,
                    user_id,
                    optimization_type,
                    status,
                    progress,
                    auto_apply,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                RETURNING id
            """, (
                video_id,
                user_id,
                optimization_type,
                'pending',
                PROGRESS_STARTED,
                auto_apply
            ))
            
            optimization_id = cursor.fetchone()[0]
            conn.commit()
            
            logger.info(
                f"Created optimization {optimization_id} for video {video_id}"
            )
            return optimization_id
            
    except Exception as e:
        logger.error(f"Error creating optimization: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()


def update_optimization_progress(
    optimization_id: int,
    progress: int,
    status: str = 'in_progress',
    error_message: Optional[str] = None
) -> bool:
    """
    Update optimization progress
    
    Args:
        optimization_id: Optimization ID
        progress: Progress percentage (0-100)
        status: Current status
        error_message: Error message if failed
        
    Returns:
        True if successful, False otherwise
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            if error_message:
                cursor.execute("""
                    UPDATE video_optimizations
                    SET progress = %s,
                        status = %s,
                        error_message = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (progress, status, error_message, optimization_id))
            else:
                cursor.execute("""
                    UPDATE video_optimizations
                    SET progress = %s,
                        status = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (progress, status, optimization_id))
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(
            f"Error updating optimization {optimization_id}: {e}",
            exc_info=True
        )
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def save_optimization_results(
    optimization_id: int,
    recommendations: Dict[str, Any],
    thumbnail_data: Optional[Dict] = None
) -> bool:
    """
    Save optimization recommendations to database
    
    Args:
        optimization_id: Optimization ID
        recommendations: LLM optimization results
        thumbnail_data: Thumbnail optimization data
        
    Returns:
        True if successful, False otherwise
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE video_optimizations
                SET recommended_title = %s,
                    recommended_description = %s,
                    recommended_tags = %s,
                    optimization_reasoning = %s,
                    thumbnail_recommendations = %s,
                    quality_score = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                recommendations.get('optimized_title'),
                recommendations.get('optimized_description'),
                json.dumps(recommendations.get('optimized_tags', [])),
                recommendations.get('reasoning'),
                json.dumps(thumbnail_data) if thumbnail_data else None,
                recommendations.get('quality_score', 0.0),
                optimization_id
            ))
            
            conn.commit()
            logger.info(f"Saved optimization results for {optimization_id}")
            return True
            
    except Exception as e:
        logger.error(
            f"Error saving optimization results for {optimization_id}: {e}",
            exc_info=True
        )
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


# ============================================================================
# VIDEO DATA RETRIEVAL
# ============================================================================

def get_video_data(video_id: int) -> Optional[Dict]:
    """
    Get comprehensive video data from database
    
    Args:
        video_id: Database video ID
        
    Returns:
        Video data dict or None if not found
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    v.id,
                    v.youtube_video_id,
                    v.user_id,
                    v.channel_id,
                    v.title,
                    v.description,
                    v.tags,
                    v.published_at,
                    v.duration,
                    v.category_id,
                    v.view_count,
                    v.like_count,
                    v.comment_count,
                    v.thumbnail_url,
                    c.channel_name,
                    c.channel_handle,
                    c.subscriber_count
                FROM videos v
                LEFT JOIN channels c ON v.channel_id = c.id
                WHERE v.id = %s
            """, (video_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Video {video_id} not found")
                return None
            
            return {
                'id': row[0],
                'youtube_video_id': row[1],
                'user_id': row[2],
                'channel_id': row[3],
                'title': row[4],
                'description': row[5],
                'tags': row[6],
                'published_at': row[7],
                'duration': row[8],
                'category_id': row[9],
                'view_count': row[10],
                'like_count': row[11],
                'comment_count': row[12],
                'thumbnail_url': row[13],
                'channel_name': row[14],
                'channel_handle': row[15],
                'subscriber_count': row[16]
            }
            
    except Exception as e:
        logger.error(f"Error getting video data for {video_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()


def get_video_transcript(video_id: int) -> Optional[str]:
    """
    Get video transcript from database
    
    Args:
        video_id: Database video ID
        
    Returns:
        Transcript text or None if not available
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT transcript_text
                FROM video_transcripts
                WHERE video_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (video_id,))
            
            row = cursor.fetchone()
            return row[0] if row else None
            
    except Exception as e:
        logger.error(f"Error getting transcript for video {video_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()


# ============================================================================
# OPTIMIZATION EXECUTION
# ============================================================================

async def optimize_video_async(
    optimization_id: int,
    video_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Execute video optimization asynchronously
    
    Args:
        optimization_id: Optimization ID
        video_id: Database video ID
        user_id: User ID
        
    Returns:
        Optimization results dict
    """
    try:
        # Update progress: Starting
        update_optimization_progress(
            optimization_id,
            PROGRESS_ANALYZING,
            'in_progress'
        )
        
        # Get video data
        video_data = get_video_data(video_id)
        if not video_data:
            update_optimization_progress(
                optimization_id,
                0,
                'failed',
                'Video not found'
            )
            return {'success': False, 'error': 'Video not found'}
        
        # Get transcript if available
        transcript = get_video_transcript(video_id)
        
        # Run optimizations in parallel
        logger.info(f"Starting parallel optimization for video {video_id}")
        
        async def run_content_optimization():
            """Run LLM content optimization"""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                executor,
                get_comprehensive_optimization,
                video_data,
                transcript
            )
        
        async def run_thumbnail_optimization():
            """Run thumbnail optimization"""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                executor,
                generate_thumbnail_recommendation,
                video_data.get('youtube_video_id'),
                video_data.get('title')
            )
        
        # Update progress: Optimizing
        update_optimization_progress(
            optimization_id,
            PROGRESS_OPTIMIZING,
            'in_progress'
        )
        
        # Execute both optimizations concurrently
        content_result, thumbnail_result = await asyncio.gather(
            run_content_optimization(),
            run_thumbnail_optimization(),
            return_exceptions=True
        )
        
        # Handle exceptions from parallel execution
        if isinstance(content_result, Exception):
            logger.error(f"Content optimization failed: {content_result}")
            content_result = None
        
        if isinstance(thumbnail_result, Exception):
            logger.error(f"Thumbnail optimization failed: {thumbnail_result}")
            thumbnail_result = None
        
        # Update progress: Generating recommendations
        update_optimization_progress(
            optimization_id,
            PROGRESS_GENERATING_THUMBNAIL,
            'in_progress'
        )
        
        # Save results
        if content_result:
            save_optimization_results(
                optimization_id,
                content_result,
                thumbnail_result
            )
        
        # Mark as completed
        update_optimization_progress(
            optimization_id,
            PROGRESS_COMPLETED,
            'completed'
        )
        
        logger.info(f"Completed optimization {optimization_id} for video {video_id}")
        
        return {
            'success': True,
            'optimization_id': optimization_id,
            'content_optimization': content_result,
            'thumbnail_optimization': thumbnail_result
        }
        
    except Exception as e:
        logger.error(
            f"Error in optimize_video_async for {video_id}: {e}",
            exc_info=True
        )
        update_optimization_progress(
            optimization_id,
            0,
            'failed',
            str(e)
        )
        return {'success': False, 'error': str(e)}


def optimize_video(
    video_id: int,
    user_id: int,
    auto_apply: bool = False
) -> Optional[int]:
    """
    Create and execute video optimization (sync wrapper)
    
    Args:
        video_id: Database video ID
        user_id: User ID
        auto_apply: Whether to auto-apply changes
        
    Returns:
        Optimization ID or None on error
    """
    # Create optimization record
    optimization_id = create_optimization(
        video_id,
        user_id,
        auto_apply=auto_apply
    )
    
    if not optimization_id:
        return None
    
    # Start async optimization in background
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Schedule the task
    loop.create_task(
        optimize_video_async(optimization_id, video_id, user_id)
    )
    
    return optimization_id


# ============================================================================
# APPLYING OPTIMIZATIONS
# ============================================================================

def apply_optimization(optimization_id: int, user_id: int) -> bool:
    """
    Apply optimization recommendations to YouTube video
    
    Args:
        optimization_id: Optimization ID to apply
        user_id: User ID (for authorization)
        
    Returns:
        True if successful, False otherwise
    """
    conn = None
    try:
        # Get optimization data
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    vo.video_id,
                    vo.user_id,
                    vo.recommended_title,
                    vo.recommended_description,
                    vo.recommended_tags,
                    v.youtube_video_id,
                    v.title AS current_title,
                    v.description AS current_description,
                    v.tags AS current_tags
                FROM video_optimizations vo
                JOIN videos v ON vo.video_id = v.id
                WHERE vo.id = %s
            """, (optimization_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.error(f"Optimization {optimization_id} not found")
                return False
            
            # Verify user owns this optimization
            if row[1] != user_id:
                logger.error(
                    f"User {user_id} not authorized for optimization {optimization_id}"
                )
                return False
            
            video_id = row[0]
            youtube_video_id = row[5]
            recommended_title = row[2]
            recommended_description = row[3]
            recommended_tags = json.loads(row[4]) if row[4] else None
            
            # Store current values as backup
            current_title = row[6]
            current_description = row[7]
            current_tags = row[8]
        
        # Update progress: Applying
        update_optimization_progress(
            optimization_id,
            PROGRESS_APPLYING,
            'applying'
        )
        
        # Apply changes to YouTube
        youtube_service = YouTubeService(user_id)
        success = youtube_service.update_video_metadata(
            youtube_video_id,
            title=recommended_title,
            description=recommended_description,
            tags=recommended_tags
        )
        
        if not success:
            update_optimization_progress(
                optimization_id,
                PROGRESS_APPLYING,
                'failed',
                'Failed to update YouTube video'
            )
            return False
        
        # Record the change
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE video_optimizations
                SET applied_at = NOW(),
                    status = 'applied',
                    progress = %s,
                    previous_title = %s,
                    previous_description = %s,
                    previous_tags = %s
                WHERE id = %s
            """, (
                PROGRESS_COMPLETED,
                current_title,
                current_description,
                current_tags,
                optimization_id
            ))
            
            # Update video record
            cursor.execute("""
                UPDATE videos
                SET title = %s,
                    description = %s,
                    tags = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                recommended_title,
                recommended_description,
                json.dumps(recommended_tags) if recommended_tags else None,
                video_id
            ))
            
            conn.commit()
        
        logger.info(f"Successfully applied optimization {optimization_id}")
        return True
        
    except Exception as e:
        logger.error(
            f"Error applying optimization {optimization_id}: {e}",
            exc_info=True
        )
        if conn:
            conn.rollback()
        update_optimization_progress(
            optimization_id,
            PROGRESS_APPLYING,
            'failed',
            str(e)
        )
        return False
    finally:
        if conn:
            conn.close()


def rollback_optimization(optimization_id: int, user_id: int) -> bool:
    """
    Rollback an applied optimization to previous values
    
    Args:
        optimization_id: Optimization ID to rollback
        user_id: User ID (for authorization)
        
    Returns:
        True if successful, False otherwise
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    vo.video_id,
                    vo.user_id,
                    vo.previous_title,
                    vo.previous_description,
                    vo.previous_tags,
                    v.youtube_video_id
                FROM video_optimizations vo
                JOIN videos v ON vo.video_id = v.id
                WHERE vo.id = %s AND vo.applied_at IS NOT NULL
            """, (optimization_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.error(
                    f"Optimization {optimization_id} not found or not applied"
                )
                return False
            
            # Verify authorization
            if row[1] != user_id:
                logger.error(f"User {user_id} not authorized")
                return False
            
            video_id = row[0]
            youtube_video_id = row[5]
            previous_title = row[2]
            previous_description = row[3]
            previous_tags = json.loads(row[4]) if row[4] else None
        
        # Rollback on YouTube
        youtube_service = YouTubeService(user_id)
        success = youtube_service.update_video_metadata(
            youtube_video_id,
            title=previous_title,
            description=previous_description,
            tags=previous_tags
        )
        
        if not success:
            logger.error(f"Failed to rollback optimization {optimization_id}")
            return False
        
        # Update database
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE video_optimizations
                SET rolled_back_at = NOW(),
                    status = 'rolled_back'
                WHERE id = %s
            """, (optimization_id,))
            
            # Restore video record
            cursor.execute("""
                UPDATE videos
                SET title = %s,
                    description = %s,
                    tags = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                previous_title,
                previous_description,
                json.dumps(previous_tags) if previous_tags else None,
                video_id
            ))
            
            conn.commit()
        
        logger.info(f"Successfully rolled back optimization {optimization_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error rolling back optimization: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'create_optimization',
    'update_optimization_progress',
    'save_optimization_results',
    'get_video_data',
    'get_video_transcript',
    'optimize_video',
    'optimize_video_async',
    'apply_optimization',
    'rollback_optimization',
]
