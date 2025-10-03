-- ============================================================================
-- ADD OPTIMIZATION_HISTORY TABLE
-- Migration Script - Run this to add the missing table
-- ============================================================================

BEGIN;

-- ============================================================================
-- CREATE OPTIMIZATION_HISTORY TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS optimization_history (
    id SERIAL PRIMARY KEY,
    
    -- Foreign key to youtube_videos
    video_id INTEGER NOT NULL,
    
    -- Status of the optimization
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    -- Possible values: pending, in_progress, completed, failed
    
    -- Metrics before optimization (JSONB for flexibility)
    metrics_before JSONB,
    -- Example: {"views": 1000, "likes": 50, "comments": 10, "ctr": 5.2}
    
    -- Metrics after optimization (JSONB for flexibility)
    metrics_after JSONB,
    -- Example: {"views": 1500, "likes": 85, "comments": 18, "ctr": 6.8}
    
    -- Additional metadata (JSONB)
    metadata JSONB,
    -- Can store: optimization_type, model_used, temperature, etc.
    
    -- Optimization details (for reference)
    optimization_notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Add foreign key constraint
    CONSTRAINT fk_optimization_history_video
        FOREIGN KEY (video_id)
        REFERENCES youtube_videos(id)
        ON DELETE CASCADE
);

-- ============================================================================
-- CREATE INDEXES
-- ============================================================================

-- Index on video_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_optimization_history_video_id 
    ON optimization_history(video_id);

-- Index on status for filtering
CREATE INDEX IF NOT EXISTS idx_optimization_history_status 
    ON optimization_history(status);

-- Index on created_at for sorting by date
CREATE INDEX IF NOT EXISTS idx_optimization_history_created_at 
    ON optimization_history(created_at DESC);

-- Composite index for common query pattern (video + date)
CREATE INDEX IF NOT EXISTS idx_optimization_history_video_created 
    ON optimization_history(video_id, created_at DESC);

-- GIN index on JSONB columns for efficient querying
CREATE INDEX IF NOT EXISTS idx_optimization_history_metrics_before 
    ON optimization_history USING GIN(metrics_before);

CREATE INDEX IF NOT EXISTS idx_optimization_history_metrics_after 
    ON optimization_history USING GIN(metrics_after);

-- ============================================================================
-- CREATE TRIGGER FOR UPDATED_AT
-- ============================================================================

-- Create or replace the trigger function if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to optimization_history
CREATE TRIGGER update_optimization_history_updated_at 
    BEFORE UPDATE ON optimization_history
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- MIGRATE DATA FROM video_optimizations (if needed)
-- ============================================================================

-- Optionally populate optimization_history with historical data
-- from video_optimizations table
INSERT INTO optimization_history (
    video_id,
    status,
    metrics_before,
    metrics_after,
    metadata,
    optimization_notes,
    created_at,
    updated_at,
    completed_at
)
SELECT 
    vo.video_id,
    vo.status,
    -- Construct metrics_before from existing data
    jsonb_build_object(
        'views', vo.views_before,
        'likes', vo.likes_before,
        'comments', vo.comments_before
    ) as metrics_before,
    -- Construct metrics_after from existing data
    jsonb_build_object(
        'views', vo.views_after,
        'likes', vo.likes_after,
        'comments', vo.comments_after
    ) as metrics_after,
    -- Store optimization metadata
    jsonb_build_object(
        'optimization_type', vo.optimization_type,
        'temperature', vo.temperature,
        'model_version', vo.model_version
    ) as metadata,
    vo.optimization_notes,
    vo.created_at,
    vo.updated_at,
    vo.applied_at as completed_at
FROM video_optimizations vo
WHERE vo.is_applied = TRUE
    AND NOT EXISTS (
        -- Don't duplicate if already migrated
        SELECT 1 FROM optimization_history oh
        WHERE oh.video_id = vo.video_id
        AND oh.created_at = vo.created_at
    )
ORDER BY vo.created_at;

-- ============================================================================
-- CREATE VIEW FOR EASY QUERYING
-- ============================================================================

-- Create a view that joins optimization_history with video details
CREATE OR REPLACE VIEW optimization_history_with_videos AS
SELECT 
    oh.id,
    oh.video_id,
    v.video_id as youtube_video_id,
    v.title as video_title,
    oh.status,
    oh.metrics_before,
    oh.metrics_after,
    oh.metadata,
    oh.optimization_notes,
    oh.created_at,
    oh.updated_at,
    oh.completed_at,
    -- Calculate improvement metrics
    CASE 
        WHEN (oh.metrics_before->>'views')::numeric > 0 
        THEN ((oh.metrics_after->>'views')::numeric - (oh.metrics_before->>'views')::numeric) 
             / (oh.metrics_before->>'views')::numeric * 100
        ELSE 0 
    END as views_improvement_percent,
    CASE 
        WHEN (oh.metrics_before->>'likes')::numeric > 0 
        THEN ((oh.metrics_after->>'likes')::numeric - (oh.metrics_before->>'likes')::numeric) 
             / (oh.metrics_before->>'likes')::numeric * 100
        ELSE 0 
    END as likes_improvement_percent,
    CASE 
        WHEN (oh.metrics_before->>'comments')::numeric > 0 
        THEN ((oh.metrics_after->>'comments')::numeric - (oh.metrics_before->>'comments')::numeric) 
             / (oh.metrics_before->>'comments')::numeric * 100
        ELSE 0 
    END as comments_improvement_percent
FROM optimization_history oh
JOIN youtube_videos v ON oh.video_id = v.id;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to add a new optimization history entry
CREATE OR REPLACE FUNCTION add_optimization_history(
    p_video_id INTEGER,
    p_status VARCHAR(50),
    p_metrics_before JSONB DEFAULT NULL,
    p_metrics_after JSONB DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL,
    p_notes TEXT DEFAULT NULL
) RETURNS INTEGER AS $$
DECLARE
    v_history_id INTEGER;
BEGIN
    INSERT INTO optimization_history (
        video_id,
        status,
        metrics_before,
        metrics_after,
        metadata,
        optimization_notes,
        created_at
    ) VALUES (
        p_video_id,
        p_status,
        p_metrics_before,
        p_metrics_after,
        p_metadata,
        p_notes,
        NOW()
    )
    RETURNING id INTO v_history_id;
    
    RETURN v_history_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update optimization history status
CREATE OR REPLACE FUNCTION update_optimization_history_status(
    p_history_id INTEGER,
    p_status VARCHAR(50),
    p_metrics_after JSONB DEFAULT NULL
) RETURNS BOOLEAN AS $$
DECLARE
    v_success BOOLEAN;
BEGIN
    UPDATE optimization_history
    SET 
        status = p_status,
        metrics_after = COALESCE(p_metrics_after, metrics_after),
        completed_at = CASE WHEN p_status = 'completed' THEN NOW() ELSE completed_at END,
        updated_at = NOW()
    WHERE id = p_history_id;
    
    GET DIAGNOSTICS v_success = ROW_COUNT;
    RETURN v_success > 0;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================

-- Grant permissions to your application user (adjust username as needed)
-- Replace 'youtube_admin' with your actual database user
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'youtube_admin') THEN
        GRANT ALL PRIVILEGES ON optimization_history TO youtube_admin;
        GRANT USAGE, SELECT ON SEQUENCE optimization_history_id_seq TO youtube_admin;
        GRANT SELECT ON optimization_history_with_videos TO youtube_admin;
    END IF;
END $$;

-- ============================================================================
-- VERIFY MIGRATION
-- ============================================================================

-- Check that table was created
DO $$
DECLARE
    v_table_exists BOOLEAN;
    v_row_count INTEGER;
BEGIN
    -- Check if table exists
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'optimization_history'
    ) INTO v_table_exists;
    
    IF v_table_exists THEN
        RAISE NOTICE 'SUCCESS: optimization_history table created';
        
        -- Check row count
        SELECT COUNT(*) INTO v_row_count FROM optimization_history;
        RAISE NOTICE 'optimization_history contains % rows', v_row_count;
        
        -- Check indexes
        RAISE NOTICE 'Indexes created:';
        FOR rec IN (
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'optimization_history'
        ) LOOP
            RAISE NOTICE '  - %', rec.indexname;
        END LOOP;
    ELSE
        RAISE EXCEPTION 'FAILED: optimization_history table not created';
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- ROLLBACK SCRIPT (Keep this commented, use if you need to undo)
-- ============================================================================

/*
BEGIN;

-- Drop view
DROP VIEW IF EXISTS optimization_history_with_videos;

-- Drop functions
DROP FUNCTION IF EXISTS add_optimization_history(INTEGER, VARCHAR, JSONB, JSONB, JSONB, TEXT);
DROP FUNCTION IF EXISTS update_optimization_history_status(INTEGER, VARCHAR, JSONB);

-- Drop table (CASCADE will drop dependent objects)
DROP TABLE IF EXISTS optimization_history CASCADE;

COMMIT;
*/

-- ============================================================================
-- USAGE EXAMPLES
-- ============================================================================

/*
-- Example 1: Add a new optimization history entry
SELECT add_optimization_history(
    123,  -- video_id
    'pending',
    '{"views": 1000, "likes": 50, "comments": 10}'::jsonb,
    NULL,
    '{"optimization_type": "ai_generated", "temperature": 0.7}'::jsonb,
    'Initial optimization attempt'
);

-- Example 2: Update optimization status
SELECT update_optimization_history_status(
    1,  -- history_id
    'completed',
    '{"views": 1500, "likes": 85, "comments": 18}'::jsonb
);

-- Example 3: Query optimization history for a video
SELECT * FROM optimization_history
WHERE video_id = 123
ORDER BY created_at DESC;

-- Example 4: Get optimization history with improvement calculations
SELECT * FROM optimization_history_with_videos
WHERE video_id = 123
ORDER BY created_at DESC;

-- Example 5: Find most successful optimizations
SELECT 
    youtube_video_id,
    video_title,
    views_improvement_percent,
    likes_improvement_percent
FROM optimization_history_with_videos
WHERE status = 'completed'
ORDER BY views_improvement_percent DESC
LIMIT 10;
*/
