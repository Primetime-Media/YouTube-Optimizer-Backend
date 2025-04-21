import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import logging

load_dotenv()

# Import config here to avoid circular imports
from config import get_settings

# Initialize logging
logger = logging.getLogger(__name__)

def get_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        settings = get_settings()
        print(f"Connecting to database at {settings.database_url}")
        conn = psycopg2.connect(settings.database_url)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

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

def run_migrations(conn):
    """Run database migrations for schema updates."""
    try:
        with conn.cursor() as cursor:
            # Migration: Add last_optimization_id to youtube_videos if it doesn't exist
            cursor.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_name='youtube_videos' AND column_name='last_optimization_id'
                    ) THEN
                        ALTER TABLE youtube_videos ADD COLUMN last_optimization_id INTEGER;
                    END IF;
                END $$;
            """)
            
            conn.commit()
            logger.info("Database migrations completed successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error running database migrations: {e}")
        raise

def init_db():
    """Initialize the database with required tables."""
    conn = get_connection()
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    try:
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
                    
                    -- Raw data for future use
                    content_details JSONB,       -- All content details as JSON
                    status_details JSONB,        -- Status information as JSON
                    topic_details JSONB,         -- Topic details as JSON
                    
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
                    created_by VARCHAR(255),               -- User or system that created the optimization
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        conn.close()
        
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

if __name__ == "__main__":
    # Run this file directly to initialize the database
    init_db()