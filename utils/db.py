"""
Database Utilities Module

PostgreSQL connection pooling, database initialization, and database operation helpers
with thread-safe connection management and resource cleanup.
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2_pool import ThreadSafeConnectionPool
from dotenv import load_dotenv
import logging
import threading
import atexit

# Load environment variables
load_dotenv()

# Import config here to avoid circular imports
from config import get_settings

# Export the context manager for easy importing
__all__ = ['get_connection', 'return_connection', 'DatabaseConnection', 'close_connection_pool']

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Global connection pool management
_connection_pool = None
_pool_lock = threading.Lock()

def get_connection_pool():
    """
    Get or create database connection pool using thread-safe singleton pattern.
    Configured for high-concurrency applications with 1-100 connections.
    """
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            # Double-check locking pattern for thread safety
            if _connection_pool is None:
                try:
                    settings = get_settings()
                    logger.info(f"Creating connection pool for database at {settings.database_url}")
                    
                    # Create connection pool with optimized configuration
                    _connection_pool = ThreadSafeConnectionPool(
                        minconn=1,        # Minimum connections in pool
                        maxconn=100,      # High limit - handles unlimited concurrent users
                        idle_timeout=30,  # Close idle connections quickly (30 seconds)
                        dsn=settings.database_url
                    )
                    
                    # Register cleanup function to close pool on application exit
                    atexit.register(close_connection_pool)
                    
                    logger.info("Connection pool created successfully")
                except Exception as e:
                    logger.error(f"Error creating connection pool: {e}")
                    raise
    
    return _connection_pool

def get_connection():
    """Get a connection from the pool."""
    try:
        pool = get_connection_pool()
        conn = pool.getconn()
        return conn
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}")
        raise

def return_connection(conn):
    """Return a connection to the pool."""
    try:
        if conn and not conn.closed:
            pool = get_connection_pool()
            pool.putconn(conn)
        else:
            logger.warning("Attempted to return closed or invalid connection")
    except Exception as e:
        logger.error(f"Error returning connection to pool: {e}")

def close_connection_pool():
    """Close the connection pool and all connections."""
    global _connection_pool
    
    if _connection_pool:
        try:
            _connection_pool.clear()
            logger.info("Connection pool closed successfully")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")
        finally:
            _connection_pool = None

def get_pool_status():
    """Get current connection pool status for monitoring."""
    global _connection_pool
    
    if _connection_pool:
        return {
            'min_connections': _connection_pool.minconn,
            'max_connections': _connection_pool.maxconn,
            'connections_in_use': len(_connection_pool.connections_in_use),
            'idle_connections': len(_connection_pool.idle_connections),
            'total_connections': len(_connection_pool.connections_in_use) + len(_connection_pool.idle_connections),
            'idle_timeout': _connection_pool.idle_timeout
        }
    return None

def cleanup_idle_connections():
    """Manually cleanup idle connections if needed."""
    global _connection_pool
    
    if _connection_pool:
        try:
            # The pool automatically handles idle connection cleanup
            # This function is for manual cleanup if needed
            status = get_pool_status()
            if status:
                logger.info(f"Pool status: {status['connections_in_use']} in use, {status['idle_connections']} idle")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

class DatabaseConnection:
    """
    Context manager for database connections.
    
    This class provides automatic connection management with the connection pool.
    Connections are automatically returned to the pool when the context exits.
    
    Example usage:
        from utils.db import DatabaseConnection
        
        with DatabaseConnection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users")
                results = cursor.fetchall()
        # Connection is automatically returned to pool
    """
    
    def __init__(self):
        self.conn = None
    
    def __enter__(self):
        self.conn = get_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            return_connection(self.conn)
            self.conn = None

def create_timeseries_table(conn):
    """Create the table for storing timeseries analytics data."""
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_timeseries_data (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES youtube_videos(id),
                timestamp TIMESTAMP NOT NULL,
                views INTEGER NOT NULL DEFAULT 0,
                estimated_minutes_watched FLOAT,
                average_view_percentage FLOAT,
                raw_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(video_id, timestamp)
            )
        """)
        
        # Create indexes for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_timeseries_video_timestamp 
            ON video_timeseries_data(video_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_timeseries_video_id
            ON video_timeseries_data(video_id)
        """)
        
        conn.commit()


def init_db():
    """Initialize the database with required tables."""
    conn = None
    try:
        conn = get_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        #delete_all_tables_except_users()

        with conn.cursor() as cursor:

            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    google_id VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    name VARCHAR(255),
                    permission_level VARCHAR(50) DEFAULT 'readwrite',
                    is_free_trial BOOLEAN DEFAULT FALSE,
                    token TEXT,
                    refresh_token TEXT,
                    token_uri VARCHAR(255),
                    client_id VARCHAR(255),
                    client_secret VARCHAR(255),
                    scopes TEXT[],
                    token_expiry TIMESTAMP,
                    -- Session management columns
                    session_token VARCHAR(255),
                    session_expires TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Index for session token lookups
                CREATE INDEX IF NOT EXISTS idx_users_session_token
                ON users(session_token);
            """)
            
            # Create YouTube channels table with enhanced fields for optimization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS youtube_channels (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    channel_id VARCHAR(255) UNIQUE NOT NULL,
                    kind VARCHAR(50),
                    etag VARCHAR(100),
                    title VARCHAR(255) NOT NULL,
                    description TEXT,
                    custom_url VARCHAR(255),
                    published_at TIMESTAMP,
                    view_count BIGINT DEFAULT 0,
                    subscriber_count INTEGER DEFAULT 0,
                    hidden_subscriber_count BOOLEAN DEFAULT FALSE,
                    video_count INTEGER DEFAULT 30,
                    thumbnail_url_default TEXT,
                    thumbnail_url_medium TEXT,
                    thumbnail_url_high TEXT,
                    uploads_playlist_id VARCHAR(255),
                    
                    -- Channel optimization fields
                    banner_url TEXT,                           -- Channel banner URL
                    privacy_status VARCHAR(50),                -- public, private, unlisted
                    is_linked BOOLEAN DEFAULT FALSE,           -- Whether channel is linked to a Google+ account
                    long_uploads_status VARCHAR(50),           -- Status for uploading long videos
                    is_monetization_enabled BOOLEAN DEFAULT FALSE, -- Channel monetization status
                    
                    -- Topic details for content categorization
                    topic_ids TEXT[],                          -- YouTube topic IDs
                    topic_categories TEXT[],                   -- Topic categories as URLs
                    
                    -- Channel standing info for compliance
                    overall_good_standing BOOLEAN DEFAULT TRUE,
                    community_guidelines_good_standing BOOLEAN DEFAULT TRUE,
                    copyright_strikes_good_standing BOOLEAN DEFAULT TRUE,
                    content_id_claims_good_standing BOOLEAN DEFAULT TRUE,
                    
                    -- Raw data for future use
                    branding_settings JSONB,                   -- All branding settings as JSON
                    audit_details JSONB,                       -- Audit details as JSON
                    topic_details JSONB,                       -- Topic details as JSON
                    status_details JSONB,                      -- Status information as JSON
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Channel optimization tracking fields
                    is_optimized BOOLEAN DEFAULT FALSE,
                    last_optimized_at TIMESTAMP,
                    last_optimization_id INTEGER
                );
            """)
            
            # Create YouTube videos table with enhanced optimization fields
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS youtube_videos (
                    id SERIAL PRIMARY KEY,
                    channel_id INTEGER REFERENCES youtube_channels(id),
                    video_id VARCHAR(255) UNIQUE NOT NULL,
                    kind VARCHAR(50),
                    etag VARCHAR(100),
                    playlist_item_id VARCHAR(255),
                    title VARCHAR(255) NOT NULL,
                    description TEXT,
                    published_at TIMESTAMP,
                    channel_title VARCHAR(255),
                    playlist_id VARCHAR(255),
                    position INTEGER,
                    tags TEXT[],
                    thumbnail_url_default TEXT,
                    thumbnail_url_medium TEXT,
                    thumbnail_url_high TEXT,
                    thumbnail_url_standard TEXT,
                    thumbnail_url_maxres TEXT,
                    view_count BIGINT DEFAULT 0,
                    like_count BIGINT DEFAULT 0,
                    comment_count INTEGER DEFAULT 0,
                    duration VARCHAR(20),
                    
                    -- Transcript and caption data
                    transcript TEXT,
                    has_captions BOOLEAN DEFAULT FALSE,
                    caption_language VARCHAR(10),
                    
                    -- Optimization status
                    is_optimized BOOLEAN DEFAULT FALSE,
                    last_optimized_at TIMESTAMP,
                    last_optimization_id INTEGER,
                    queued_for_optimization BOOLEAN DEFAULT FALSE,
                    optimizations_completed INTEGER DEFAULT 0,
                    
                    -- Video status and visibility
                    privacy_status VARCHAR(50),  -- public, private, unlisted
                    upload_status VARCHAR(50),   -- uploaded, processed, etc.
                    license VARCHAR(50),         -- youtube, creative_commons
                    embeddable BOOLEAN,          -- If the video can be embedded
                    public_stats_viewable BOOLEAN, -- If stats are publicly viewable
                    
                    -- Content details
                    definition VARCHAR(20),      -- hd, sd
                    dimension VARCHAR(20),       -- 2d, 3d
                    has_custom_thumbnail BOOLEAN DEFAULT FALSE,
                    projection VARCHAR(20),      -- rectangular, 360
                    
                    -- Video category
                    category_id VARCHAR(50),
                    category_name VARCHAR(255),
                    
                    -- Topic details for content categorization (similar to channel)
                    topic_ids TEXT[],            -- YouTube topic IDs
                    topic_categories TEXT[],     -- Topic categories as URLs
                    relevance_score FLOAT,       -- Custom score for optimization relevance
                    
                    -- Optimization performance metrics
                    pre_optimization_view_count BIGINT,
                    post_optimization_view_count BIGINT,
                    optimization_improvement_percent FLOAT,
                    
                    -- Advanced analytics storage
                    retention_graph_data JSONB,  -- Audience retention data
                    viewer_demographics JSONB,   -- Audience demographics 
                    traffic_sources JSONB,       -- Where views come from
                    last_analytics_refresh TIMESTAMP WITH TIME ZONE,
                    
                    -- Raw data for future use
                    content_details JSONB,       -- All content details as JSON
                    status_details JSONB,        -- Status information as JSON
                    topic_details JSONB,         -- Topic details as JSON
                    
                    -- Optimization results
                    optimized_title VARCHAR(255),
                    optimized_description TEXT,
                    optimized_tags TEXT[],
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            
            # Create channel optimizations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS channel_optimizations (
                    id SERIAL PRIMARY KEY,
                    channel_id INTEGER REFERENCES youtube_channels(id) NOT NULL,
                    original_description TEXT,
                    optimized_description TEXT,
                    original_keywords TEXT,
                    optimized_keywords TEXT,
                    optimization_notes TEXT,
                    is_applied BOOLEAN DEFAULT FALSE,
                    applied_at TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'pending',  -- pending, in_progress, completed, failed
                    progress INTEGER DEFAULT 0,            -- 0-100 percentage of completion
                    optimization_score FLOAT,              -- Optional score to rate quality of optimization
                    created_by VARCHAR(255),               -- User or system that created the optimization
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Create indexes for better query performance
                CREATE INDEX IF NOT EXISTS idx_channel_optimization_channel_id
                ON channel_optimizations(channel_id);

                CREATE INDEX IF NOT EXISTS idx_channel_optimization_created_at
                ON channel_optimizations(created_at);
                
                CREATE INDEX IF NOT EXISTS idx_channel_optimization_status
                ON channel_optimizations(status);

                -- Add comment to explain table usage
                COMMENT ON TABLE channel_optimizations IS 'Stores history of channel description and keyword optimizations';
            """)
            
            # Create timeseries analytics table
            create_timeseries_table(conn)
            
            # Create video optimizations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_optimizations (
                    id SERIAL PRIMARY KEY,
                    video_id INTEGER REFERENCES youtube_videos(id) NOT NULL,
                    original_title TEXT,
                    optimized_title TEXT,
                    original_description TEXT,
                    optimized_description TEXT,
                    original_tags TEXT[],
                    optimized_tags TEXT[],
                    optimization_notes TEXT,
                    is_applied BOOLEAN DEFAULT FALSE,
                    applied_at TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'pending',  -- pending, in_progress, completed, failed
                    progress INTEGER DEFAULT 0,            -- 0-100 percentage of completion
                    optimization_score FLOAT,              -- Optional score to rate quality of optimization
                    
                    optimization_step INTEGER NOT NULL,
                    created_by VARCHAR(255),               -- User or system that created the optimization
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(video_id, optimization_step)
                );

                -- Create indexes for better query performance
                CREATE INDEX IF NOT EXISTS idx_video_optimization_video_id
                ON video_optimizations(video_id);

                CREATE INDEX IF NOT EXISTS idx_video_optimization_created_at
                ON video_optimizations(created_at);
                
                CREATE INDEX IF NOT EXISTS idx_video_optimization_status
                ON video_optimizations(status);

                -- Add comment to explain table usage
                COMMENT ON TABLE video_optimizations IS 'Stores history of video title, description and tag optimizations';
            """)
            
            # Initialize scheduler tables
            init_scheduler_tables(cursor)
            
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        if conn:
            return_connection(conn)
        
def init_scheduler_tables(cursor):
    """Initialize tables for scheduler functionality"""
    # Table to store scheduling configuration per channel
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS channel_optimization_schedules (
            id SERIAL PRIMARY KEY,
            channel_id INTEGER REFERENCES youtube_channels(id),
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            auto_apply BOOLEAN NOT NULL DEFAULT TRUE,
            last_run TIMESTAMP WITH TIME ZONE,
            next_run TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(channel_id)
        )
    """)
    
    # Table to log scheduled runs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scheduler_run_history (
            id SERIAL PRIMARY KEY,
            schedule_id INTEGER REFERENCES channel_optimization_schedules(id),
            start_time TIMESTAMP WITH TIME ZONE,
            end_time TIMESTAMP WITH TIME ZONE,
            status VARCHAR(50),
            optimization_id INTEGER,
            applied BOOLEAN DEFAULT FALSE,
            error_message TEXT
        )
    """)

def delete_all_tables_except_users():
    """
    Deletes all database tables except the 'users' table.
    Useful for resetting the database during development.
    """
    conn = None
    try:
        conn = get_connection()
        # Set isolation level to AUTOCOMMIT for DROP TABLE statements
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cursor:
            # Get a list of all tables in the current database
            # This query works for PostgreSQL
            cursor.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            # Filter out the 'users' table
            tables_to_delete = [table for table in tables if table != 'users']
            
            if not tables_to_delete:
                logger.info("No tables found to delete (excluding 'users').")
                return

            logger.info(f"Deleting tables: {', '.join(tables_to_delete)}")

            # Drop each table with CASCADE to remove dependent objects (like foreign keys)
            for table in tables_to_delete:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                    logger.info(f"Dropped table: {table}")
                except Exception as e:
                    logger.error(f"Error dropping table {table}: {e}")
                    # Continue with other tables even if one fails
                    pass 
            
            logger.info("Finished attempting to delete tables.")

    except Exception as e:
        logger.error(f"Error deleting tables: {e}")
        # No rollback needed with AUTOCOMMIT, but log the error
        raise # Re-raise the exception
    finally:
        if conn:
            return_connection(conn)
            logger.info("Database connection returned to pool.")

if __name__ == "__main__":
    pass
    # Run this file directly to initialize the database
    #init_db()
    #delete_all_tables_except_users()