import logging
from typing import Dict, Optional, List
from utils.db import get_connection
from services.llm_optimization import get_channel_optimization

logger = logging.getLogger(__name__)

def get_channel_data(channel_id: int) -> Optional[Dict]:
    """
    Retrieve channel data from the database using the channel_id
    
    Args:
        channel_id: The database ID of the channel
        
    Returns:
        dict: The channel data including branding settings or None if not found
    """
    logger.info(f"Getting channel data for channel {channel_id}")
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
                logger.warning(f"Channel {channel_id} not found in database")
                return None
                
            title, branding_settings = result
            
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
                "id": channel_id,
                "title": title,
                "branding_settings": branding_settings,
                "recent_videos": latest_videos,
                "random_videos": random_videos
            }
    except Exception as e:
        logger.error(f"Error fetching channel data: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def get_latest_videos_for_channel(channel_id: int, limit: int = 3) -> List[Dict]:
    """
    Get the latest videos for a channel
    
    Args:
        channel_id: The database ID of the channel
        limit: The number of videos to retrieve (default: 3)
        
    Returns:
        list: The latest videos with their metadata
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT video_id, title, description, tags
                FROM youtube_videos
                WHERE channel_id = %s
                ORDER BY published_at DESC
                LIMIT %s
            """, (channel_id, limit))
            
            videos = []
            for video_data in cursor.fetchall():
                video_id, title, description, tags = video_data
                videos.append({
                    "video_id": video_id,
                    "title": title,
                    "description": description,
                    "tags": tags
                })
            
            return videos
    except Exception as e:
        logger.error(f"Error fetching latest videos: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()
            
def get_random_videos_for_channel(channel_id: int, limit: int = 3, exclude_ids: List[str] = None) -> List[Dict]:
    """
    Get random videos for a channel, excluding specific video IDs
    
    Args:
        channel_id: The database ID of the channel
        limit: The number of videos to retrieve (default: 3)
        exclude_ids: List of video_ids to exclude from the selection
        
    Returns:
        list: A random selection of videos with their metadata
    """
    try:
        if exclude_ids is None:
            exclude_ids = []
            
        conn = get_connection()
        with conn.cursor() as cursor:
            # Create parameterized query with dynamic exclusion
            exclusion_clause = ""
            params = [channel_id]
            
            if exclude_ids:
                placeholders = []
                for i, vid_id in enumerate(exclude_ids):
                    placeholders.append(f"%s")
                    params.append(vid_id)
                
                exclusion_clause = f"AND video_id NOT IN ({', '.join(placeholders)})"
            
            # Use RANDOM() for true randomization
            query = f"""
                SELECT video_id, title, description, tags
                FROM youtube_videos
                WHERE channel_id = %s {exclusion_clause}
                ORDER BY RANDOM()
                LIMIT %s
            """
            params.append(limit)
            
            cursor.execute(query, params)
            
            videos = []
            for video_data in cursor.fetchall():
                video_id, title, description, tags = video_data
                videos.append({
                    "video_id": video_id,
                    "title": title,
                    "description": description,
                    "tags": tags
                })
            
            return videos
    except Exception as e:
        logger.error(f"Error fetching random videos: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

def generate_channel_optimization(channel: Dict, optimization_id: int) -> Dict:
    """
    Generate optimized description and keywords for a YouTube channel
    
    Args:
        channel: Dict containing channel data with title, branding_settings, and recent_videos
        optimization_id: The ID of the pre-created optimization record to use
        
    Returns:
        dict: Contains optimized description and keywords
    """
    logger.info(f"Starting channel optimization for channel: {channel.get('title', 'Unknown')} with optimization ID: {optimization_id}")
    
    title = channel.get("title", "")
    branding_settings = channel.get("branding_settings", {})
    recent_videos = channel.get("recent_videos", [])
    channel_id = channel.get("id")
    
    # Verify optimization record exists
    if optimization_id == 0:
        logger.error(f"Invalid optimization ID for channel {channel_id}")
        return {
            "error": "Invalid optimization ID",
            "id": 0
        }
    
    # Update status to in_progress
    update_optimization_progress(optimization_id, 10, "in_progress")
    logger.debug(f"Channel data summary - Title: {title}, Videos: {len(recent_videos)}")
    
    # Extract description and keywords from branding settings
    description = ""
    keywords = ""
    if branding_settings and isinstance(branding_settings, dict):
        channel_data = branding_settings.get("channel", {})
        description = channel_data.get("description", "")
        keywords = channel_data.get("keywords", "")
        
        logger.debug(f"Extracted description length: {len(description)}, keywords: {keywords}")
    
    # Update progress - data extraction complete
    update_optimization_progress(optimization_id, 25)
    
    # Format recent and random videos for inclusion in the optimization prompt
    recent_video_data = ""
    
    # Process recent videos
    if recent_videos:
        logger.debug(f"Processing {len(recent_videos)} recent videos")
        recent_video_data = "RECENT VIDEOS (Latest uploads):\n"
        for idx, video in enumerate(recent_videos):
            video_title = video.get('title', '')
            recent_video_data += f"Recent Video {idx+1}:\n"
            recent_video_data += f"Title: {video_title}\n"
            
            # Add a truncated description for each video
            video_desc = video.get('description', '')
            if video_desc:
                if len(video_desc) > 200:
                    video_desc = video_desc[:200] + "..."
                recent_video_data += f"Description: {video_desc}\n"
            
            # Add tags if available
            tags = video.get('tags', [])
            if tags:
                tags_str = ", ".join(tags[:10])  # Limit to first 10 tags
                recent_video_data += f"Tags: {tags_str}\n"
        
        recent_video_data += "\n"
    
    # Process random videos
    random_videos = channel.get("random_videos", [])
    if random_videos:
        logger.debug(f"Processing {len(random_videos)} random videos")
        recent_video_data += "RANDOM VIDEOS (Broader channel content sample):\n"
        for idx, video in enumerate(random_videos):
            video_title = video.get('title', '')
            recent_video_data += f"Random Video {idx+1}:\n"
            recent_video_data += f"Title: {video_title}\n"
            
            # Add a truncated description for each video
            video_desc = video.get('description', '')
            if video_desc:
                if len(video_desc) > 200:
                    video_desc = video_desc[:200] + "..."
                recent_video_data += f"Description: {video_desc}\n"
            
            # Add tags if available
            tags = video.get('tags', [])
            if tags:
                tags_str = ", ".join(tags[:10])  # Limit to first 10 tags
                recent_video_data += f"Tags: {tags_str}\n"
        
    if recent_video_data:
        logger.debug(f"Generated video data summary: {len(recent_video_data)} characters")
    
    # Update progress - prompt preparation complete
    update_optimization_progress(optimization_id, 50)
    
    try:
        # Call the optimization function with all data
        result = get_channel_optimization(
            channel_title=title,
            description=description,
            keywords=keywords,
            recent_videos_data=recent_video_data
        )
        
        # Update progress - LLM processing complete
        update_optimization_progress(optimization_id, 75)
        
        logger.info(f"Channel optimization completed successfully for {title}")
        logger.debug(f"Optimization result summary: {result}")
        
        # Store the results in the existing optimization record
        store_optimization_results(optimization_id, channel.get("id"), result)
        
        # Add optimization ID to the result
        result["id"] = optimization_id
        
        return {
            "id": optimization_id,
            "original_description": result.get("original_description", ""),
            "optimized_description": result.get("optimized_description", ""),
            "original_keywords": result.get("original_keywords", ""),
            "optimized_keywords": result.get("optimized_keywords", ""),
            "optimization_notes": result.get("optimization_notes", "")
        }
    except Exception as e:
        logger.error(f"Error during channel optimization: {str(e)}")
        update_optimization_progress(optimization_id, 0, "failed")
        return {
            "error": f"Optimization failed: {str(e)}",
            "id": optimization_id
        }

def create_optimization(channel_id: int) -> int:
    """
    Create a new optimization record for each request
    
    Args:
        channel_id: The database ID of the channel
        
    Returns:
        int: The ID of the new optimization record
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Always create a new optimization record
                cursor.execute("""
                    INSERT INTO channel_optimizations (
                        channel_id,
                        status,
                        progress,
                        created_by
                    ) VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (
                    channel_id,
                    'pending',
                    0,
                    'youtube-optimizer-system'
                ))
                
                optimization_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"Created new optimization record {optimization_id} for channel {channel_id}")
                return optimization_id
        except Exception as e:
            # Roll back transaction on error
            if not conn.autocommit:
                conn.rollback()
            raise
    except Exception as e:
        logger.error(f"Error creating optimization record: {str(e)}")
        return 0
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def update_optimization_progress(optimization_id: int, progress: int, status: str = None) -> bool:
    """
    Update the progress of a channel optimization
    
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
                    UPDATE channel_optimizations
                    SET progress = %s, status = %s, updated_at = NOW()
                    WHERE id = %s
                """, (progress, status, optimization_id))
            else:
                cursor.execute("""
                    UPDATE channel_optimizations
                    SET progress = %s, updated_at = NOW()
                    WHERE id = %s
                """, (progress, optimization_id))
            
            conn.commit()
            logger.info(f"Updated optimization {optimization_id} progress to {progress}%")
            return True
            
    except Exception as e:
        logger.error(f"Error updating optimization progress: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def store_optimization_results(optimization_id: int, channel_id: int, optimization_data: Dict, update_description: bool = False) -> bool:
    """
    Store optimization results in an existing optimization record
    
    Args:
        optimization_id: The ID of the existing optimization record
        channel_id: The database ID of the channel
        optimization_data: Dict containing optimization results
        update_description: Whether to update the description (default: False)
        
    Returns:
        bool: Success status
    """
    try:
        logger.info(f"Storing optimization results for optimization ID {optimization_id}, channel {channel_id}")
        logger.debug(f"Optimization data summary: {optimization_data}")
        
        # Update the existing record with optimization results
        conn = get_connection()
        with conn.cursor() as cursor:
            # By default, keep original description and only update keywords
            description_to_store = optimization_data.get("original_description", "") if not update_description else optimization_data.get("optimized_description", "")
            
            cursor.execute("""
                UPDATE channel_optimizations
                SET 
                    original_description = %s,
                    optimized_description = %s,
                    original_keywords = %s,
                    optimized_keywords = %s,
                    optimization_notes = %s,
                    status = 'completed',
                    progress = 100,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id
            """, (
                optimization_data.get("original_description", ""),
                description_to_store,
                optimization_data.get("original_keywords", ""),
                optimization_data.get("optimized_keywords", ""),
                optimization_data.get("optimization_notes", ""),
                optimization_id
            ))
            
            result = cursor.fetchone()
            if not result:
                logger.error(f"Failed to update optimization record {optimization_id}")
                return False
                
            conn.commit()
            
            logger.info(f"Successfully stored results for optimization ID {optimization_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error storing optimization results: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_channel_optimizations(channel_id: int, limit: int = 5) -> List[Dict]:
    """
    Get the most recent channel optimizations for a channel
    
    Args:
        channel_id: The database ID of the channel
        limit: Maximum number of records to return (default: 5)
        
    Returns:
        list: Recent optimization records for the channel
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    id, 
                    original_description,
                    optimized_description,
                    original_keywords,
                    optimized_keywords,
                    optimization_notes,
                    is_applied,
                    applied_at,
                    status,
                    progress,
                    created_at
                FROM channel_optimizations
                WHERE channel_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (channel_id, limit))
            
            optimizations = []
            for row in cursor.fetchall():
                (
                    optimization_id, 
                    original_description,
                    optimized_description,
                    original_keywords,
                    optimized_keywords,
                    optimization_notes,
                    is_applied,
                    applied_at,
                    status,
                    progress,
                    created_at
                ) = row
                
                optimizations.append({
                    "id": optimization_id,
                    "original_description": original_description,
                    "optimized_description": optimized_description,
                    "original_keywords": original_keywords,
                    "optimized_keywords": optimized_keywords,
                    "optimization_notes": optimization_notes,
                    "is_applied": is_applied,
                    "applied_at": applied_at,
                    "status": status,
                    "progress": progress,
                    "created_at": created_at
                })
            
            return optimizations
    except Exception as e:
        logger.error(f"Error getting channel optimizations: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

def get_optimization_status(optimization_id: int) -> Dict:
    """
    Get the current status of a channel optimization
    
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
                    channel_id,
                    status,
                    progress,
                    created_at,
                    updated_at
                FROM channel_optimizations
                WHERE id = %s
            """, (optimization_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"No optimization found with ID {optimization_id}")
                return {
                    "error": "Optimization not found",
                    "id": optimization_id
                }
                
            channel_id, status, progress, created_at, updated_at = result
            
            status_info = {
                "id": optimization_id,
                "channel_id": channel_id,
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
                        original_description,
                        optimized_description,
                        original_keywords,
                        optimized_keywords,
                        optimization_notes,
                        is_applied,
                        applied_at
                    FROM channel_optimizations
                    WHERE id = %s
                """, (optimization_id,))
                
                complete_data = cursor.fetchone()
                if complete_data:
                    (
                        original_description,
                        optimized_description,
                        original_keywords,
                        optimized_keywords,
                        optimization_notes,
                        is_applied,
                        applied_at
                    ) = complete_data
                    
                    # Add result data to the response
                    status_info.update({
                        "original_description": original_description,
                        "optimized_description": optimized_description,
                        "original_keywords": original_keywords,
                        "optimized_keywords": optimized_keywords,
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

def apply_channel_optimization(optimization_id: int) -> bool:
    """
    Mark a channel optimization as applied
    
    Args:
        optimization_id: The ID of the optimization to apply
        
    Returns:
        bool: True if successfully applied, False otherwise
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE channel_optimizations
                SET is_applied = TRUE, applied_at = NOW(), updated_at = NOW()
                WHERE id = %s
                RETURNING channel_id
            """, (optimization_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"No optimization found with ID {optimization_id}")
                return False
                
            channel_id = result[0]
            conn.commit()
            
            logger.info(f"Applied channel optimization {optimization_id} for channel {channel_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error applying channel optimization: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()