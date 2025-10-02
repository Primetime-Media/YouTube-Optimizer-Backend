"""
Video Routes Module - COMPLETE PRODUCTION-READY
================================================
All Errors Fixed - Complete Video Management Routes

Features:
- Video listing and retrieval
- Video optimization workflows
- Batch operations
- Status tracking
- Progress monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import logging
from pydantic import BaseModel, Field

from utils.db import get_connection
from services.video import optimize_video, get_video_status
from services.youtube_service import sync_video_from_youtube

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class VideoBase(BaseModel):
    """Base video model"""
    youtube_video_id: str = Field(..., description="YouTube video ID")
    title: str
    description: Optional[str] = None
    channel_id: int


class VideoCreate(VideoBase):
    """Model for creating a new video"""
    pass


class VideoResponse(VideoBase):
    """Video response model"""
    id: int
    status: str = "pending"
    views: int = 0
    likes: int = 0
    comments: int = 0
    published_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class OptimizationRequest(BaseModel):
    """Request model for video optimization"""
    auto_apply: bool = Field(False, description="Automatically apply optimization to YouTube")
    force: bool = Field(False, description="Force optimization even if recently optimized")


class OptimizationResponse(BaseModel):
    """Optimization response model"""
    video_id: int
    optimization_id: int
    status: str
    message: str
    progress: int = 0


class BatchOptimizationRequest(BaseModel):
    """Request model for batch optimization"""
    video_ids: List[int] = Field(..., description="List of video IDs to optimize")
    auto_apply: bool = Field(False, description="Automatically apply optimizations")


# ============================================================================
# VIDEO CRUD ENDPOINTS
# ============================================================================

@router.get("/", response_model=List[VideoResponse])
async def list_videos(
    channel_id: Optional[int] = Query(None, description="Filter by channel ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Number of videos to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List videos with optional filtering
    
    Args:
        channel_id: Optional channel ID filter
        status: Optional status filter
        limit: Maximum number of videos to return
        offset: Pagination offset
        
    Returns:
        List of videos
    """
    conn = None
    try:
        conn = get_connection()
        
        # Build query with filters
        query = """
            SELECT 
                id, youtube_video_id, title, description, channel_id, status,
                views, likes, comments, published_at, created_at, updated_at
            FROM videos
            WHERE 1=1
        """
        params = []
        
        if channel_id is not None:
            query += " AND channel_id = %s"
            params.append(channel_id)
        
        if status is not None:
            query += " AND status = %s"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(params))
            
            videos = []
            for row in cursor.fetchall():
                videos.append(VideoResponse(
                    id=row[0],
                    youtube_video_id=row[1],
                    title=row[2],
                    description=row[3],
                    channel_id=row[4],
                    status=row[5] or 'pending',
                    views=row[6] or 0,
                    likes=row[7] or 0,
                    comments=row[8] or 0,
                    published_at=row[9],
                    created_at=row[10],
                    updated_at=row[11]
                ))
        
        return videos
        
    except Exception as e:
        logger.error(f"Error listing videos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(video_id: int):
    """
    Get a specific video by ID
    
    Args:
        video_id: Database ID of the video
        
    Returns:
        Video details
    """
    conn = None
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    id, youtube_video_id, title, description, channel_id, status,
                    views, likes, comments, published_at, created_at, updated_at
                FROM videos
                WHERE id = %s
            """, (video_id,))
            
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Video not found")
            
            return VideoResponse(
                id=row[0],
                youtube_video_id=row[1],
                title=row[2],
                description=row[3],
                channel_id=row[4],
                status=row[5] or 'pending',
                views=row[6] or 0,
                likes=row[7] or 0,
                comments=row[8] or 0,
                published_at=row[9],
                created_at=row[10],
                updated_at=row[11]
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch video: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.post("/", response_model=VideoResponse, status_code=201)
async def create_video(video: VideoCreate):
    """
    Create a new video entry
    
    Args:
        video: Video creation data
        
    Returns:
        Created video
    """
    conn = None
    try:
        conn = get_connection()
        
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO videos 
                (youtube_video_id, title, description, channel_id, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                RETURNING id, youtube_video_id, title, description, channel_id, 
                          status, views, likes, comments, published_at, created_at, updated_at
            """, (
                video.youtube_video_id,
                video.title,
                video.description,
                video.channel_id
            ))
            
            row = cursor.fetchone()
            conn.commit()
            
            if not row:
                raise HTTPException(status_code=500, detail="Failed to create video")
            
            return VideoResponse(
                id=row[0],
                youtube_video_id=row[1],
                title=row[2],
                description=row[3],
                channel_id=row[4],
                status=row[5] or 'pending',
                views=row[6] or 0,
                likes=row[7] or 0,
                comments=row[8] or 0,
                published_at=row[9],
                created_at=row[10],
                updated_at=row[11]
            )
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error creating video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create video: {str(e)}")
    finally:
        if conn:
            conn.close()


# ============================================================================
# VIDEO OPTIMIZATION ENDPOINTS
# ============================================================================

@router.post("/{video_id}/optimize", response_model=OptimizationResponse)
async def optimize_video_endpoint(
    video_id: int,
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Optimize a video's metadata
    
    Args:
        video_id: Database ID of the video
        request: Optimization request parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Optimization response with status
    """
    conn = None
    try:
        # Verify video exists
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, youtube_video_id, channel_id, status
                FROM videos
                WHERE id = %s
            """, (video_id,))
            
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Video not found")
            
            video_status = row[3]
            
            # Check if already optimizing
            if video_status == 'optimizing' and not request.force:
                raise HTTPException(
                    status_code=400,
                    detail="Video is already being optimized. Use force=true to override."
                )
        
        # Create optimization record
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO optimizations 
                (video_id, status, auto_apply, created_at)
                VALUES (%s, 'pending', %s, NOW())
                RETURNING id
            """, (video_id, request.auto_apply))
            
            optimization_id = cursor.fetchone()[0]
            
            # Update video status
            cursor.execute("""
                UPDATE videos
                SET status = 'optimizing', updated_at = NOW()
                WHERE id = %s
            """, (video_id,))
            
            conn.commit()
        
        # Start optimization in background
        background_tasks.add_task(
            optimize_video,
            video_id=video_id,
            optimization_id=optimization_id,
            auto_apply=request.auto_apply
        )
        
        return OptimizationResponse(
            video_id=video_id,
            optimization_id=optimization_id,
            status='pending',
            message='Optimization started',
            progress=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error starting optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/{video_id}/optimization-status")
async def get_optimization_status(video_id: int):
    """
    Get current optimization status for a video
    
    Args:
        video_id: Database ID of the video
        
    Returns:
        Current optimization status
    """
    try:
        status = await get_video_status(video_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="No optimization found for this video")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching optimization status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch status: {str(e)}")


@router.post("/batch-optimize")
async def batch_optimize_videos(
    request: BatchOptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Optimize multiple videos in batch
    
    Args:
        request: Batch optimization request
        background_tasks: FastAPI background tasks
        
    Returns:
        Batch operation status
    """
    conn = None
    try:
        if not request.video_ids:
            raise HTTPException(status_code=400, detail="No video IDs provided")
        
        if len(request.video_ids) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 videos per batch")
        
        conn = get_connection()
        optimization_ids = []
        
        # Create optimization records for all videos
        with conn.cursor() as cursor:
            for video_id in request.video_ids:
                # Verify video exists
                cursor.execute("SELECT id FROM videos WHERE id = %s", (video_id,))
                if not cursor.fetchone():
                    logger.warning(f"Video {video_id} not found, skipping")
                    continue
                
                # Create optimization
                cursor.execute("""
                    INSERT INTO optimizations 
                    (video_id, status, auto_apply, created_at)
                    VALUES (%s, 'pending', %s, NOW())
                    RETURNING id
                """, (video_id, request.auto_apply))
                
                optimization_id = cursor.fetchone()[0]
                optimization_ids.append((video_id, optimization_id))
                
                # Update video status
                cursor.execute("""
                    UPDATE videos
                    SET status = 'optimizing', updated_at = NOW()
                    WHERE id = %s
                """, (video_id,))
            
            conn.commit()
        
        # Start optimizations in background
        for video_id, optimization_id in optimization_ids:
            background_tasks.add_task(
                optimize_video,
                video_id=video_id,
                optimization_id=optimization_id,
                auto_apply=request.auto_apply
            )
        
        return {
            'status': 'success',
            'message': f'Started optimization for {len(optimization_ids)} videos',
            'optimizations': [
                {'video_id': vid, 'optimization_id': opt_id}
                for vid, opt_id in optimization_ids
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error in batch optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch optimization failed: {str(e)}")
    finally:
        if conn:
            conn.close()


# ============================================================================
# VIDEO SYNC ENDPOINTS
# ============================================================================

@router.post("/{video_id}/sync")
async def sync_video(video_id: int):
    """
    Sync video data from YouTube
    
    Args:
        video_id: Database ID of the video
        
    Returns:
        Sync status
    """
    conn = None
    try:
        conn = get_connection()
        
        # Get video's YouTube ID
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT youtube_video_id
                FROM videos
                WHERE id = %s
            """, (video_id,))
            
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Video not found")
            
            youtube_video_id = row[0]
        
        # Sync from YouTube
        synced_data = await sync_video_from_youtube(youtube_video_id)
        
        if not synced_data:
            raise HTTPException(status_code=500, detail="Failed to sync video from YouTube")
        
        # Update database
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE videos
                SET 
                    title = %s,
                    description = %s,
                    views = %s,
                    likes = %s,
                    comments = %s,
                    published_at = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                synced_data.get('title'),
                synced_data.get('description'),
                synced_data.get('views', 0),
                synced_data.get('likes', 0),
                synced_data.get('comments', 0),
                synced_data.get('published_at'),
                video_id
            ))
            
            conn.commit()
        
        return {
            'status': 'success',
            'message': 'Video synced successfully',
            'synced_data': synced_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error syncing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to sync video: {str(e)}")
    finally:
        if conn:
            conn.close()
