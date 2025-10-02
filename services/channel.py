"""
Channel Service Module - COMPLETE FIXED VERSION
================================================
9 Critical Errors Fixed - Production Ready

Key Fixes Applied:
1. Connection leak prevention (initialized before try)
2. SQL injection prevention (parameterized query)
3. Transaction rollbacks added
4. NULL checks for all results
5. Comprehensive error handling
6. Proper resource cleanup
"""

import logging
from typing import Dict, Optional, List
from utils.db import get_connection
from services.llm_optimization import get_channel_optimization

logger = logging.getLogger(__name__)


def get_channel_data(channel_id: int) -> Optional[Dict]:
    """
    Retrieve channel data from the database using the channel_id
    
    FIXES:
    - #1: Initialize conn = None before try block
    - #2: NULL check for branding_settings
    
    Args:
        channel_id: The database ID of the channel
        
    Returns:
        dict: The channel data including branding settings or None if not found
    """
    conn = None  # ✅ FIX: Initialize to avoid NameError
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT title, branding_settings
                FROM youtube_channels
                WHERE id = %s
            """, (channel_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Channel {channel_id} not found")
                return None
            
            title, branding_settings = result
            
            # ✅ FIX: Add NULL check
            if branding_settings is None:
                branding_settings = {}
            
            # Get the latest 3 videos for this channel
            latest_videos = get_latest_videos_for_channel(channel_id, limit=3)
            
            # Get video IDs to exclude from random selection
            latest_video_ids = [video["video_id"] for video in latest_videos]
            
            # Get 3 random videos, excluding the latest ones
            random_videos = get_random_videos_for_channel(
                channel_id, 
                limit=3, 
                exclude_ids=latest_video_ids
            )
            
            return {
                "channel_id": channel_id,
                "title": title,
                "branding_settings": branding_settings,
                "latest_videos": latest_videos,
                "random_videos": random_videos
            }
            
    except Exception as e:
        logger.error(f"Error fetching channel data for channel {channel_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


def get_latest_videos_for_channel(channel_id: int, limit: int = 3) -> List[Dict]:
    """
    Get the latest videos for a channel
    
    FIXES:
    - #3: Initialize conn = None
    - #4: NULL check for results
    
    Args:
        channel_id: The database ID of the channel
        limit: Number of videos to retrieve
        
    Returns:
        list: List of video dictionaries
    """
    conn = None  # ✅ FIX
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT youtube_video_id AS video_id, title, views, likes, comments
                FROM youtube_videos
                WHERE channel_id = %s
                ORDER BY published_at DESC
                LIMIT %s
            """, (channel_id, limit))
            
            results = cursor.fetchall()
            
            # ✅ FIX: Check for NULL results
            if not results:
                return []
            
            return [
                {
                    "video_id": row[0],
                    "title": row[1],
                    "views": row[2] or 0,
                    "likes": row[3] or 0,
                    "comments": row[4] or 0
                }
                for row in results
            ]
            
    except Exception as e:
        logger.error(f"Error fetching latest videos for channel {channel_id}: {e}", exc_info=True)
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


def get_random_videos_for_channel(
    channel_id: int,
    limit: int = 3,
    exclude_ids: Optional[List[str]] = None
) -> List[Dict]:
    """
    Get random videos for a channel, excluding specified video IDs
    
    FIXES:
    - #5: Initialize conn = None
    - #6: SQL injection prevention with parameterized query
    - #7: NULL check for results
    
    Args:
        channel_id: The database ID of the channel
        limit: Number of videos to retrieve
        exclude_ids: List of video IDs to exclude
        
    Returns:
        list: List of video dictionaries
    """
    conn = None  # ✅ FIX
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # ✅ FIX: Use parameterized query instead of string formatting
            if exclude_ids:
                placeholders = ','.join(['%s'] * len(exclude_ids))
                query = f"""
                    SELECT youtube_video_id AS video_id, title, views, likes, comments
                    FROM youtube_videos
                    WHERE channel_id = %s
                    AND youtube_video_id NOT IN ({placeholders})
                    ORDER BY RANDOM()
                    LIMIT %s
                """
                params = (channel_id, *exclude_ids, limit)
            else:
                query = """
                    SELECT youtube_video_id AS video_id, title, views, likes, comments
                    FROM youtube_videos
                    WHERE channel_id = %s
                    ORDER BY RANDOM()
                    LIMIT %s
                """
                params = (channel_id, limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # ✅ FIX: Check for NULL results
            if not results:
                return []
            
            return [
                {
                    "video_id": row[0],
                    "title": row[1],
                    "views": row[2] or 0,
                    "likes": row[3] or 0,
                    "comments": row[4] or 0
                }
                for row in results
            ]
            
    except Exception as e:
        logger.error(f"Error fetching random videos for channel {channel_id}: {e}", exc_info=True)
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


def create_optimization(
    channel_id: int,
    optimization_data: Dict
) -> Optional[int]:
    """
    Create a new channel optimization record
    
    FIXES:
    - #8: Initialize conn = None
    - #9: Add transaction rollback on error
    
    Args:
        channel_id: The database ID of the channel
        optimization_data: Optimization data to store
        
    Returns:
        int: The ID of the created optimization or None
    """
    conn = None  # ✅ FIX
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO channel_optimizations 
                (channel_id, optimization_data, created_at)
                VALUES (%s, %s, NOW())
                RETURNING id
            """, (channel_id, optimization_data))
            
            result = cursor.fetchone()
            conn.commit()  # ✅ FIX: Explicit commit
            
            if result:
                optimization_id = result[0]
                logger.info(f"Created optimization {optimization_id} for channel {channel_id}")
                return optimization_id
            
            return None
            
    except Exception as e:
        if conn:
            conn.rollback()  # ✅ FIX: Rollback on error
        logger.error(f"Error creating optimization for channel {channel_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


def update_optimization_status(
    optimization_id: int,
    status: str,
    error_message: Optional[str] = None
) -> bool:
    """
    Update the status of an optimization
    
    Args:
        optimization_id: The ID of the optimization
        status: New status value
        error_message: Optional error message
        
    Returns:
        bool: True if update successful
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            if error_message:
                cursor.execute("""
                    UPDATE channel_optimizations
                    SET status = %s, error_message = %s, updated_at = NOW()
                    WHERE id = %s
                """, (status, error_message, optimization_id))
            else:
                cursor.execute("""
                    UPDATE channel_optimizations
                    SET status = %s, updated_at = NOW()
                    WHERE id = %s
                """, (status, optimization_id))
            
            conn.commit()
            logger.info(f"Updated optimization {optimization_id} status to {status}")
            return True
            
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error updating optimization status: {e}", exc_info=True)
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


def get_channel_optimizations(
    channel_id: int,
    limit: int = 10
) -> List[Dict]:
    """
    Get optimization history for a channel
    
    Args:
        channel_id: The database ID of the channel
        limit: Number of optimizations to retrieve
        
    Returns:
        list: List of optimization records
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, optimization_data, status, created_at, updated_at
                FROM channel_optimizations
                WHERE channel_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (channel_id, limit))
            
            results = cursor.fetchall()
            
            if not results:
                return []
            
            return [
                {
                    "id": row[0],
                    "optimization_data": row[1],
                    "status": row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                    "updated_at": row[4].isoformat() if row[4] else None
                }
                for row in results
            ]
            
    except Exception as e:
        logger.error(f"Error fetching optimizations for channel {channel_id}: {e}", exc_info=True)
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
