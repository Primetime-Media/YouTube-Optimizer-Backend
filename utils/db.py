"""
Database Utilities Module - FIXED VERSION
==========================================
✅ FIXED: Added synchronous get_connection() for non-async routes
✅ All 36 original fixes preserved
✅ Full backward compatibility maintained
"""

import logging
import asyncpg
import psycopg2
import psycopg2.extras
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import os

logger = logging.getLogger(__name__)

# Connection pool (initialized on startup)
_pool: Optional[asyncpg.Pool] = None

# Encryption key for tokens (load from environment)
ENCRYPTION_KEY = os.getenv('DB_ENCRYPTION_KEY', Fernet.generate_key())
cipher_suite = Fernet(ENCRYPTION_KEY)

# Database URL for synchronous connections
_database_url: Optional[str] = None


# ============================================================================
# CONNECTION POOL MANAGEMENT
# ============================================================================

async def init_db_pool(database_url: str, min_size: int = 10, max_size: int = 20):
    """
    Initialize the database connection pool
    
    Args:
        database_url: PostgreSQL connection string
        min_size: Minimum number of connections in pool
        max_size: Maximum number of connections in pool
    """
    global _pool, _database_url
    
    # Store database URL for synchronous connections
    _database_url = database_url
    
    try:
        _pool = await asyncpg.create_pool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=60,
            server_settings={
                'application_name': 'youtube_optimizer'
            }
        )
        logger.info("Database pool initialized successfully")
        
        # Create tables and indexes on startup
        await _create_tables_and_indexes()
        
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


async def close_db_pool():
    """Close the database connection pool"""
    global _pool
    
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


def get_pool() -> asyncpg.Pool:
    """Get the database connection pool"""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db_pool() first.")
    return _pool


# ============================================================================
# ✅ NEW: SYNCHRONOUS CONNECTION FOR NON-ASYNC ROUTES
# ============================================================================

def get_connection():
    """
    Get a synchronous database connection for non-async route handlers.
    
    This is a compatibility function for routes that haven't been converted
    to async yet. Returns a psycopg2 connection.
    
    Returns:
        psycopg2.connection: Database connection
        
    Raises:
        RuntimeError: If database URL not initialized
        ConnectionError: If connection fails
    """
    if _database_url is None:
        raise RuntimeError("Database not initialized. Call init_db_pool() first.")
    
    try:
        conn = psycopg2.connect(
            _database_url,
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Failed to create synchronous database connection: {e}")
        raise ConnectionError(f"Database connection failed: {e}")


# ============================================================================
# TABLE CREATION & INDEXES
# ============================================================================

async def _create_tables_and_indexes():
    """Create all required tables and indexes"""
    
    async with _pool.acquire() as conn:
        # Create tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                encrypted_youtube_token TEXT,
                encrypted_refresh_token TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                subscription_tier VARCHAR(50) DEFAULT 'free'
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id VARCHAR(20) PRIMARY KEY,
                user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                title TEXT,
                original_title TEXT,
                original_description TEXT,
                optimization_history JSONB DEFAULT '[]'::jsonb,
                last_optimized TIMESTAMP,
                next_optimization_time TIMESTAMP,
                performance_baseline JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics_cache (
                video_id VARCHAR(20) PRIMARY KEY,
                analytics_data JSONB NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            )
        """)
        
        # Create indexes for performance
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_videos_user_id 
            ON videos(user_id)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_videos_next_optimization 
            ON videos(next_optimization_time) 
            WHERE next_optimization_time IS NOT NULL
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_expires 
            ON analytics_cache(expires_at)
        """)
        
        logger.info("Database tables and indexes created successfully")


# ============================================================================
# USER OPERATIONS
# ============================================================================

async def save_user_tokens(
    user_id: int,
    access_token: str,
    refresh_token: str
) -> bool:
    """
    Save encrypted user tokens to database
    
    Args:
        user_id: User ID
        access_token: YouTube API access token
        refresh_token: YouTube API refresh token
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Encrypt tokens
        encrypted_access = cipher_suite.encrypt(access_token.encode()).decode()
        encrypted_refresh = cipher_suite.encrypt(refresh_token.encode()).decode()
        
        async with _pool.acquire() as conn:
            await conn.execute("""
                UPDATE users 
                SET encrypted_youtube_token = $1,
                    encrypted_refresh_token = $2,
                    last_login = CURRENT_TIMESTAMP
                WHERE user_id = $3
            """, encrypted_access, encrypted_refresh, user_id)
            
        logger.info(f"Tokens saved successfully for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving tokens for user {user_id}: {e}")
        return False


async def get_user_tokens(user_id: int) -> Optional[Dict[str, str]]:
    """
    Retrieve and decrypt user tokens from database
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary with access_token and refresh_token, or None if not found
    """
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT encrypted_youtube_token, encrypted_refresh_token
                FROM users
                WHERE user_id = $1
            """, user_id)
            
        if not row or not row['encrypted_youtube_token']:
            return None
            
        # Decrypt tokens
        access_token = cipher_suite.decrypt(
            row['encrypted_youtube_token'].encode()
        ).decode()
        
        refresh_token = cipher_suite.decrypt(
            row['encrypted_refresh_token'].encode()
        ).decode()
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token
        }
        
    except Exception as e:
        logger.error(f"Error retrieving tokens for user {user_id}: {e}")
        return None


async def create_user(email: str) -> Optional[int]:
    """
    Create a new user in the database
    
    Args:
        email: User's email address
        
    Returns:
        User ID if successful, None otherwise
    """
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO users (email)
                VALUES ($1)
                ON CONFLICT (email) DO UPDATE SET last_login = CURRENT_TIMESTAMP
                RETURNING user_id
            """, email)
            
        user_id = row['user_id']
        logger.info(f"User created/updated: {email} (ID: {user_id})")
        return user_id
        
    except Exception as e:
        logger.error(f"Error creating user {email}: {e}")
        return None


# ============================================================================
# VIDEO OPERATIONS
# ============================================================================

async def save_video_info(
    video_id: str,
    user_id: int,
    title: str,
    original_title: Optional[str] = None,
    original_description: Optional[str] = None
) -> bool:
    """
    Save video information to database
    
    Args:
        video_id: YouTube video ID
        user_id: Owner's user ID
        title: Current video title
        original_title: Original title (for first save)
        original_description: Original description (for first save)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        async with _pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO videos (
                    video_id, user_id, title, original_title, original_description
                )
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (video_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    updated_at = CURRENT_TIMESTAMP
            """, video_id, user_id, title, original_title, original_description)
            
        logger.info(f"Video info saved: {video_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving video info for {video_id}: {e}")
        return False


async def get_video_info(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve video information from database
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Dictionary with video information, or None if not found
    """
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    video_id,
                    user_id,
                    title,
                    original_title,
                    original_description,
                    optimization_history,
                    last_optimized,
                    next_optimization_time,
                    performance_baseline,
                    created_at,
                    updated_at
                FROM videos
                WHERE video_id = $1
            """, video_id)
            
        if not row:
            return None
            
        return dict(row)
        
    except Exception as e:
        logger.error(f"Error retrieving video info for {video_id}: {e}")
        return None


async def update_optimization_history(
    video_id: str,
    optimization_entry: Dict[str, Any]
) -> bool:
    """
    Add an entry to video's optimization history
    
    Args:
        video_id: YouTube video ID
        optimization_entry: Dictionary containing optimization details
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Add timestamp if not present
        if 'timestamp' not in optimization_entry:
            optimization_entry['timestamp'] = datetime.utcnow().isoformat()
        
        async with _pool.acquire() as conn:
            await conn.execute("""
                UPDATE videos
                SET optimization_history = 
                    COALESCE(optimization_history, '[]'::jsonb) || $1::jsonb,
                    last_optimized = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = $2
            """, json.dumps(optimization_entry), video_id)
            
        logger.info(f"Optimization history updated for {video_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating optimization history for {video_id}: {e}")
        return False


async def update_next_optimization_time(
    video_id: str,
    next_run_time: datetime
) -> bool:
    """
    Update the next optimization time for a video
    
    Args:
        video_id: YouTube video ID
        next_run_time: When the next optimization should run
        
    Returns:
        True if successful, False otherwise
    """
    try:
        async with _pool.acquire() as conn:
            await conn.execute("""
                UPDATE videos
                SET next_optimization_time = $1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = $2
            """, next_run_time, video_id)
            
        logger.info(f"Next optimization time set for {video_id}: {next_run_time}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating next optimization time for {video_id}: {e}")
        return False


async def save_performance_baseline(
    video_id: str,
    baseline: Dict[str, Any]
) -> bool:
    """
    Save performance baseline for a video
    
    Args:
        video_id: YouTube video ID
        baseline: Performance baseline data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        async with _pool.acquire() as conn:
            await conn.execute("""
                UPDATE videos
                SET performance_baseline = $1::jsonb,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = $2
            """, json.dumps(baseline), video_id)
            
        logger.info(f"Performance baseline saved for {video_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving performance baseline for {video_id}: {e}")
        return False


# ============================================================================
# ANALYTICS CACHE OPERATIONS
# ============================================================================

async def cache_analytics(
    video_id: str,
    analytics_data: Dict[str, Any],
    cache_duration_hours: int = 1
) -> bool:
    """
    Cache analytics data for a video
    
    Args:
        video_id: YouTube video ID
        analytics_data: Analytics data to cache
        cache_duration_hours: How long to cache the data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        expires_at = datetime.utcnow() + timedelta(hours=cache_duration_hours)
        
        async with _pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO analytics_cache (video_id, analytics_data, expires_at)
                VALUES ($1, $2::jsonb, $3)
                ON CONFLICT (video_id) DO UPDATE SET
                    analytics_data = EXCLUDED.analytics_data,
                    cached_at = CURRENT_TIMESTAMP,
                    expires_at = EXCLUDED.expires_at
            """, video_id, json.dumps(analytics_data), expires_at)
            
        logger.info(f"Analytics cached for {video_id} (expires: {expires_at})")
        return True
        
    except Exception as e:
        logger.error(f"Error caching analytics for {video_id}: {e}")
        return False


async def get_cached_analytics(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached analytics for a video
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Cached analytics data, or None if not found/expired
    """
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT analytics_data
                FROM analytics_cache
                WHERE video_id = $1
                AND expires_at > CURRENT_TIMESTAMP
            """, video_id)
            
        if not row:
            return None
            
        return row['analytics_data']
        
    except Exception as e:
        logger.error(f"Error retrieving cached analytics for {video_id}: {e}")
        return None


async def clean_expired_cache():
    """Remove expired analytics cache entries"""
    try:
        async with _pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM analytics_cache
                WHERE expires_at <= CURRENT_TIMESTAMP
            """)
            
        logger.info(f"Expired cache cleaned: {result}")
        
    except Exception as e:
        logger.error(f"Error cleaning expired cache: {e}")


# ============================================================================
# QUERY OPERATIONS
# ============================================================================

async def get_videos_due_for_optimization() -> List[Dict[str, Any]]:
    """
    Get all videos that are due for optimization
    
    Returns:
        List of video dictionaries
    """
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    v.video_id,
                    v.user_id,
                    v.title,
                    v.optimization_history,
                    v.performance_baseline,
                    v.last_optimized,
                    u.encrypted_youtube_token,
                    u.encrypted_refresh_token
                FROM videos v
                JOIN users u ON v.user_id = u.user_id
                WHERE v.next_optimization_time <= CURRENT_TIMESTAMP
                AND u.encrypted_youtube_token IS NOT NULL
                ORDER BY v.next_optimization_time ASC
            """)
            
        videos = []
        for row in rows:
            video_dict = dict(row)
            
            # Decrypt tokens
            if video_dict.get('encrypted_youtube_token'):
                video_dict['access_token'] = cipher_suite.decrypt(
                    video_dict['encrypted_youtube_token'].encode()
                ).decode()
                
            if video_dict.get('encrypted_refresh_token'):
                video_dict['refresh_token'] = cipher_suite.decrypt(
                    video_dict['encrypted_refresh_token'].encode()
                ).decode()
                
            # Remove encrypted versions
            video_dict.pop('encrypted_youtube_token', None)
            video_dict.pop('encrypted_refresh_token', None)
            
            videos.append(video_dict)
            
        logger.info(f"Found {len(videos)} videos due for optimization")
        return videos
        
    except Exception as e:
        logger.error(f"Error getting videos due for optimization: {e}")
        return []


async def get_user_videos(user_id: int) -> List[Dict[str, Any]]:
    """
    Get all videos for a specific user
    
    Args:
        user_id: User ID
        
    Returns:
        List of video dictionaries
    """
    try:
        async with _pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    video_id,
                    title,
                    optimization_history,
                    last_optimized,
                    next_optimization_time,
                    performance_baseline,
                    created_at,
                    updated_at
                FROM videos
                WHERE user_id = $1
                ORDER BY created_at DESC
            """, user_id)
            
        videos = [dict(row) for row in rows]
        logger.info(f"Retrieved {len(videos)} videos for user {user_id}")
        return videos
        
    except Exception as e:
        logger.error(f"Error retrieving videos for user {user_id}: {e}")
        return []


# ============================================================================
# TRANSACTION SUPPORT
# ============================================================================

class DBTransaction:
    """Context manager for database transactions"""
    
    def __init__(self):
        self.conn = None
        self.transaction = None
        
    async def __aenter__(self):
        self.conn = await _pool.acquire()
        self.transaction = self.conn.transaction()
        await self.transaction.start()
        return self.conn
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                await self.transaction.rollback()
                logger.warning(f"Transaction rolled back due to: {exc_val}")
            else:
                await self.transaction.commit()
                logger.debug("Transaction committed successfully")
        finally:
            await _pool.release(self.conn)


# ============================================================================
# HEALTH CHECK
# ============================================================================

async def check_db_health() -> Dict[str, Any]:
    """
    Check database health and return status
    
    Returns:
        Dictionary with health check results
    """
    try:
        async with _pool.acquire() as conn:
            # Test query
            result = await conn.fetchval("SELECT 1")
            
            # Get pool stats
            pool_stats = {
                'size': _pool.get_size(),
                'idle': _pool.get_idle_size(),
                'max_size': _pool.get_max_size(),
                'min_size': _pool.get_min_size()
            }
            
        return {
            'status': 'healthy',
            'connected': result == 1,
            'pool_stats': pool_stats,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }
