# Database Migration Guide

**Version:** 2.0  
**Last Updated:** October 2, 2025  
**Status:** Production Ready

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Backup Your Data](#backup-your-data)
- [Migration Options](#migration-options)
- [Fresh Installation](#fresh-installation)
- [Updating Existing Database](#updating-existing-database)
- [Post-Migration Verification](#post-migration-verification)
- [Rollback Procedures](#rollback-procedures)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This guide covers all database setup and migration procedures for the YouTube Content Optimization Platform, including:

- Creating the database from scratch
- Migrating from previous versions
- Adding new tables and columns
- Updating indexes and constraints
- Data migration procedures

---

## ✅ Prerequisites

### **Required Software**

- PostgreSQL 15.0 or higher
- psql command-line tool
- Database admin access (postgres user or equivalent)

### **Before You Begin**

1. ✅ Stop your application
2. ✅ Backup your existing database (if applicable)
3. ✅ Verify PostgreSQL is running
4. ✅ Have database credentials ready

### **Check PostgreSQL Status**

```bash
# Linux/macOS
sudo systemctl status postgresql

# Or check connection
psql -U postgres -c "SELECT version();"
```

---

## 💾 Backup Your Data

**CRITICAL:** Always backup before migrations!

### **Full Database Backup**

```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Full database backup
pg_dump -U postgres -d youtube_optimizer \
  -F c -b -v \
  -f "backups/$(date +%Y%m%d)/youtube_optimizer_$(date +%H%M%S).backup"

# Or as SQL file
pg_dump -U postgres -d youtube_optimizer \
  > "backups/$(date +%Y%m%d)/youtube_optimizer_$(date +%H%M%S).sql"
```

### **Table-Specific Backups**

```bash
# Backup specific tables
pg_dump -U postgres -d youtube_optimizer \
  -t users -t youtube_channels -t youtube_videos \
  > "backups/$(date +%Y%m%d)/critical_tables.sql"
```

### **Verify Backup**

```bash
# Check backup file exists and is not empty
ls -lh backups/$(date +%Y%m%d)/

# Verify backup integrity
pg_restore --list "backups/$(date +%Y%m%d)/youtube_optimizer_*.backup"
```

---

## 🆕 Fresh Installation

Use this if you're setting up the database for the first time.

### **Step 1: Create Database**

```bash
# Connect as postgres user
psql -U postgres

# Create database
CREATE DATABASE youtube_optimizer
  WITH 
  ENCODING = 'UTF8'
  LC_COLLATE = 'en_US.UTF-8'
  LC_CTYPE = 'en_US.UTF-8'
  TEMPLATE = template0;

# Create dedicated user (optional but recommended)
CREATE USER youtube_admin WITH ENCRYPTED PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE youtube_optimizer TO youtube_admin;

# Exit psql
\q
```

### **Step 2: Apply Complete Schema**

Save this as `database/schema.sql`:

```sql
-- ============================================================================
-- YOUTUBE OPTIMIZER DATABASE SCHEMA
-- Version: 2.0
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- USERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    google_id VARCHAR(255) UNIQUE,
    
    -- OAuth Credentials (encrypted)
    access_token TEXT,
    refresh_token TEXT,
    token_type VARCHAR(50),
    token_expiry TIMESTAMP WITH TIME ZONE,
    
    -- User metadata
    name VARCHAR(255),
    picture_url TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    
    -- Indexes
    CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_google_id ON users(google_id);
CREATE INDEX idx_users_token_expiry ON users(token_expiry);

-- ============================================================================
-- YOUTUBE CHANNELS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS youtube_channels (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- YouTube identifiers
    channel_id VARCHAR(255) UNIQUE NOT NULL,
    kind VARCHAR(100),
    etag VARCHAR(255),
    
    -- Basic info
    title VARCHAR(255),
    description TEXT,
    custom_url VARCHAR(255),
    published_at TIMESTAMP WITH TIME ZONE,
    
    -- Statistics
    view_count BIGINT DEFAULT 0,
    subscriber_count INTEGER DEFAULT 0,
    hidden_subscriber_count BOOLEAN DEFAULT FALSE,
    video_count INTEGER DEFAULT 0,
    
    -- Thumbnails
    thumbnail_url_default TEXT,
    thumbnail_url_medium TEXT,
    thumbnail_url_high TEXT,
    banner_url TEXT,
    
    -- Content details
    uploads_playlist_id VARCHAR(255),
    
    -- Channel status
    privacy_status VARCHAR(50),
    is_linked BOOLEAN DEFAULT FALSE,
    long_uploads_status VARCHAR(50),
    is_monetization_enabled BOOLEAN DEFAULT FALSE,
    
    -- Topics and categories
    topic_ids TEXT[],
    topic_categories TEXT[],
    
    -- Channel standing
    overall_good_standing BOOLEAN DEFAULT TRUE,
    community_guidelines_good_standing BOOLEAN DEFAULT TRUE,
    copyright_strikes_good_standing BOOLEAN DEFAULT TRUE,
    content_id_claims_good_standing BOOLEAN DEFAULT TRUE,
    
    -- Raw API data (JSONB for flexibility)
    branding_settings JSONB,
    audit_details JSONB,
    topic_details JSONB,
    status_details JSONB,
    
    -- Optimization tracking
    is_optimized BOOLEAN DEFAULT FALSE,
    last_optimized_at TIMESTAMP WITH TIME ZONE,
    last_optimization_id INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_user_channel UNIQUE(user_id, channel_id)
);

CREATE INDEX idx_channels_user_id ON youtube_channels(user_id);
CREATE INDEX idx_channels_channel_id ON youtube_channels(channel_id);
CREATE INDEX idx_channels_last_optimized ON youtube_channels(last_optimized_at);
CREATE INDEX idx_channels_branding_settings ON youtube_channels USING GIN(branding_settings);

-- ============================================================================
-- YOUTUBE VIDEOS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS youtube_videos (
    id SERIAL PRIMARY KEY,
    channel_id INTEGER NOT NULL REFERENCES youtube_channels(id) ON DELETE CASCADE,
    
    -- YouTube identifiers
    video_id VARCHAR(255) UNIQUE NOT NULL,
    kind VARCHAR(100),
    etag VARCHAR(255),
    playlist_item_id VARCHAR(255),
    
    -- Basic info
    title TEXT,
    description TEXT,
    published_at TIMESTAMP WITH TIME ZONE,
    channel_title VARCHAR(255),
    
    -- Playlist info
    playlist_id VARCHAR(255),
    position INTEGER,
    
    -- Tags
    tags TEXT[],
    
    -- Thumbnails
    thumbnail_url_default TEXT,
    thumbnail_url_medium TEXT,
    thumbnail_url_high TEXT,
    thumbnail_url_standard TEXT,
    thumbnail_url_maxres TEXT,
    
    -- Statistics
    view_count BIGINT DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    
    -- Content details
    duration VARCHAR(50),  -- ISO 8601 format (PT4M13S)
    
    -- Transcript
    transcript TEXT,
    has_captions BOOLEAN DEFAULT FALSE,
    caption_language VARCHAR(10),
    
    -- Video status
    privacy_status VARCHAR(50),
    upload_status VARCHAR(50),
    license VARCHAR(50),
    embeddable BOOLEAN DEFAULT TRUE,
    public_stats_viewable BOOLEAN DEFAULT TRUE,
    
    -- Technical details
    definition VARCHAR(10),  -- 'hd' or 'sd'
    dimension VARCHAR(10),   -- '2d' or '3d'
    has_custom_thumbnail BOOLEAN DEFAULT FALSE,
    projection VARCHAR(50),  -- 'rectangular' or '360'
    
    -- Category
    category_id VARCHAR(50),
    category_name VARCHAR(100),
    
    -- Topics
    topic_ids TEXT[],
    topic_categories TEXT[],
    
    -- Raw API data
    content_details JSONB,
    status_details JSONB,
    topic_details JSONB,
    
    -- Optimization tracking
    is_optimized BOOLEAN DEFAULT FALSE,
    last_optimized_at TIMESTAMP WITH TIME ZONE,
    last_optimization_id INTEGER,
    last_analytics_refresh TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_channel_video UNIQUE(channel_id, video_id)
);

CREATE INDEX idx_videos_channel_id ON youtube_videos(channel_id);
CREATE INDEX idx_videos_video_id ON youtube_videos(video_id);
CREATE INDEX idx_videos_published_at ON youtube_videos(published_at DESC);
CREATE INDEX idx_videos_view_count ON youtube_videos(view_count DESC);
CREATE INDEX idx_videos_is_optimized ON youtube_videos(is_optimized);
CREATE INDEX idx_videos_last_optimized ON youtube_videos(last_optimized_at);
CREATE INDEX idx_videos_tags ON youtube_videos USING GIN(tags);
CREATE INDEX idx_videos_content_details ON youtube_videos USING GIN(content_details);

-- ============================================================================
-- VIDEO OPTIMIZATIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS video_optimizations (
    id SERIAL PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES youtube_videos(id) ON DELETE CASCADE,
    
    -- Original content
    original_title TEXT,
    original_description TEXT,
    original_tags TEXT[],
    
    -- Optimized content
    optimized_title TEXT,
    optimized_description TEXT,
    optimized_tags TEXT[],
    
    -- Optimization metadata
    optimization_type VARCHAR(50) DEFAULT 'ai_generated',  -- ai_generated, manual, a_b_test
    optimization_notes TEXT,
    temperature FLOAT DEFAULT 0.7,
    model_version VARCHAR(50),
    
    -- Application status
    is_applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMP WITH TIME ZONE,
    
    -- Performance tracking
    views_before INTEGER DEFAULT 0,
    views_after INTEGER,
    likes_before INTEGER DEFAULT 0,
    likes_after INTEGER,
    comments_before INTEGER DEFAULT 0,
    comments_after INTEGER,
    
    -- Status and progress
    status VARCHAR(50) DEFAULT 'pending',  -- pending, in_progress, completed, failed, applied
    progress INTEGER DEFAULT 0,  -- 0-100
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_optimizations_video_id ON video_optimizations(video_id);
CREATE INDEX idx_optimizations_is_applied ON video_optimizations(is_applied);
CREATE INDEX idx_optimizations_status ON video_optimizations(status);
CREATE INDEX idx_optimizations_created_at ON video_optimizations(created_at DESC);

-- ============================================================================
-- VIDEO TIMESERIES DATA TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS video_timeseries_data (
    id SERIAL PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES youtube_videos(id) ON DELETE CASCADE,
    
    -- Time dimension
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Metrics
    views INTEGER NOT NULL DEFAULT 0,
    estimated_minutes_watched FLOAT,
    average_view_percentage FLOAT,
    
    -- Raw data for future analysis
    raw_data JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint
    CONSTRAINT unique_video_timestamp UNIQUE(video_id, timestamp)
);

CREATE INDEX idx_timeseries_video_id ON video_timeseries_data(video_id);
CREATE INDEX idx_timeseries_timestamp ON video_timeseries_data(timestamp DESC);
CREATE INDEX idx_timeseries_video_timestamp ON video_timeseries_data(video_id, timestamp DESC);
CREATE INDEX idx_timeseries_raw_data ON video_timeseries_data USING GIN(raw_data);

-- ============================================================================
-- CHANNEL OPTIMIZATIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS channel_optimizations (
    id SERIAL PRIMARY KEY,
    channel_id INTEGER NOT NULL REFERENCES youtube_channels(id) ON DELETE CASCADE,
    
    -- Original content
    original_description TEXT,
    original_keywords TEXT,
    
    -- Optimized content
    optimized_description TEXT,
    optimized_keywords TEXT,
    
    -- Optimization metadata
    optimization_notes TEXT,
    temperature FLOAT DEFAULT 0.7,
    model_version VARCHAR(50),
    
    -- Application status
    is_applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMP WITH TIME ZONE,
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_channel_optimizations_channel_id ON channel_optimizations(channel_id);
CREATE INDEX idx_channel_optimizations_is_applied ON channel_optimizations(is_applied);
CREATE INDEX idx_channel_optimizations_created_at ON channel_optimizations(created_at DESC);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Updated timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_youtube_channels_updated_at BEFORE UPDATE ON youtube_channels
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_youtube_videos_updated_at BEFORE UPDATE ON youtube_videos
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_video_optimizations_updated_at BEFORE UPDATE ON video_optimizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_channel_optimizations_updated_at BEFORE UPDATE ON channel_optimizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Video performance summary view
CREATE OR REPLACE VIEW video_performance_summary AS
SELECT 
    v.id,
    v.video_id,
    v.title,
    v.view_count,
    v.like_count,
    v.comment_count,
    v.published_at,
    v.is_optimized,
    v.last_optimized_at,
    COUNT(vo.id) as optimization_count,
    MAX(vo.created_at) as last_optimization_date,
    SUM(CASE WHEN vo.is_applied THEN 1 ELSE 0 END) as applied_optimizations
FROM youtube_videos v
LEFT JOIN video_optimizations vo ON v.id = vo.video_id
GROUP BY v.id, v.video_id, v.title, v.view_count, v.like_count, v.comment_count, 
         v.published_at, v.is_optimized, v.last_optimized_at;

-- Channel summary view
CREATE OR REPLACE VIEW channel_summary AS
SELECT 
    c.id,
    c.channel_id,
    c.title,
    c.subscriber_count,
    c.video_count,
    c.view_count,
    c.is_optimized,
    u.email as user_email,
    COUNT(v.id) as total_videos,
    SUM(CASE WHEN v.is_optimized THEN 1 ELSE 0 END) as optimized_videos
FROM youtube_channels c
JOIN users u ON c.user_id = u.id
LEFT JOIN youtube_videos v ON c.id = v.channel_id
GROUP BY c.id, c.channel_id, c.title, c.subscriber_count, c.video_count, 
         c.view_count, c.is_optimized, u.email;

-- ============================================================================
-- COMPLETION
-- ============================================================================

-- Grant permissions to youtube_admin user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO youtube_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO youtube_admin;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO youtube_admin;

-- Verify schema
SELECT table_name, table_type 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
```

### **Step 3: Apply Schema**

```bash
# Apply the schema
psql -U postgres -d youtube_optimizer -f database/schema.sql

# Verify tables were created
psql -U postgres -d youtube_optimizer -c "\dt"
```

### **Step 4: Verify Installation**

```bash
# Check table count
psql -U postgres -d youtube_optimizer -c "
SELECT COUNT(*) as table_count 
FROM information_schema.tables 
WHERE table_schema = 'public';
"

# Expected: 7 tables (users, youtube_channels, youtube_videos, 
#                     video_optimizations, video_timeseries_data,
#                     channel_optimizations, plus system tables)
```

---

## 🔄 Updating Existing Database

Use this if you're upgrading from a previous version.

### **Migration Path**

**Version Detection:**

```bash
# Check if you have the timeseries table
psql -U postgres -d youtube_optimizer -c "\d video_timeseries_data"

# If exists: You're on v1.5+
# If not exists: You're on v1.0-1.4
```

### **Migration Script**

Save as `database/migrate_to_v2.sql`:

```sql
-- ============================================================================
-- MIGRATION TO VERSION 2.0
-- ============================================================================

BEGIN;

-- Add new columns to existing tables if they don't exist
DO $$ 
BEGIN
    -- youtube_channels new columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='youtube_channels' AND column_name='banner_url') THEN
        ALTER TABLE youtube_channels ADD COLUMN banner_url TEXT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='youtube_channels' AND column_name='branding_settings') THEN
        ALTER TABLE youtube_channels ADD COLUMN branding_settings JSONB;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='youtube_channels' AND column_name='is_monetization_enabled') THEN
        ALTER TABLE youtube_channels ADD COLUMN is_monetization_enabled BOOLEAN DEFAULT FALSE;
    END IF;
    
    -- youtube_videos new columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='youtube_videos' AND column_name='has_custom_thumbnail') THEN
        ALTER TABLE youtube_videos ADD COLUMN has_custom_thumbnail BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='youtube_videos' AND column_name='content_details') THEN
        ALTER TABLE youtube_videos ADD COLUMN content_details JSONB;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='youtube_videos' AND column_name='last_analytics_refresh') THEN
        ALTER TABLE youtube_videos ADD COLUMN last_analytics_refresh TIMESTAMP WITH TIME ZONE;
    END IF;
    
    -- video_optimizations new columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='video_optimizations' AND column_name='temperature') THEN
        ALTER TABLE video_optimizations ADD COLUMN temperature FLOAT DEFAULT 0.7;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='video_optimizations' AND column_name='model_version') THEN
        ALTER TABLE video_optimizations ADD COLUMN model_version VARCHAR(50);
    END IF;
END $$;

-- Create video_timeseries_data table if it doesn't exist
CREATE TABLE IF NOT EXISTS video_timeseries_data (
    id SERIAL PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES youtube_videos(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    views INTEGER NOT NULL DEFAULT 0,
    estimated_minutes_watched FLOAT,
    average_view_percentage FLOAT,
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_video_timestamp UNIQUE(video_id, timestamp)
);

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_timeseries_video_id ON video_timeseries_data(video_id);
CREATE INDEX IF NOT EXISTS idx_timeseries_timestamp ON video_timeseries_data(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_timeseries_video_timestamp ON video_timeseries_data(video_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_channels_branding_settings ON youtube_channels USING GIN(branding_settings);
CREATE INDEX IF NOT EXISTS idx_videos_content_details ON youtube_videos USING GIN(content_details);

-- Create channel_optimizations table if it doesn't exist
CREATE TABLE IF NOT EXISTS channel_optimizations (
    id SERIAL PRIMARY KEY,
    channel_id INTEGER NOT NULL REFERENCES youtube_channels(id) ON DELETE CASCADE,
    original_description TEXT,
    original_keywords TEXT,
    optimized_description TEXT,
    optimized_keywords TEXT,
    optimization_notes TEXT,
    temperature FLOAT DEFAULT 0.7,
    model_version VARCHAR(50),
    is_applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_channel_optimizations_channel_id ON channel_optimizations(channel_id);

-- Create or replace views
CREATE OR REPLACE VIEW video_performance_summary AS
SELECT 
    v.id,
    v.video_id,
    v.title,
    v.view_count,
    v.like_count,
    v.comment_count,
    v.published_at,
    v.is_optimized,
    v.last_optimized_at,
    COUNT(vo.id) as optimization_count,
    MAX(vo.created_at) as last_optimization_date,
    SUM(CASE WHEN vo.is_applied THEN 1 ELSE 0 END) as applied_optimizations
FROM youtube_videos v
LEFT JOIN video_optimizations vo ON v.id = vo.video_id
GROUP BY v.id, v.video_id, v.title, v.view_count, v.like_count, v.comment_count, 
         v.published_at, v.is_optimized, v.last_optimized_at;

COMMIT;

-- Display success message
SELECT 'Migration to v2.0 completed successfully!' as status;
```

### **Apply Migration**

```bash
# 1. Backup first!
pg_dump -U postgres youtube_optimizer > backup_before_migration.sql

# 2. Apply migration
psql -U postgres -d youtube_optimizer -f database/migrate_to_v2.sql

# 3. Verify
psql -U postgres -d youtube_optimizer -c "
SELECT table_name, column_name, data_type 
FROM information_schema.columns 
WHERE table_name IN ('youtube_channels', 'youtube_videos', 'video_optimizations')
ORDER BY table_name, ordinal_position;
"
```

---

## ✅ Post-Migration Verification

### **Step 1: Table Structure Check**

```sql
-- Connect to database
psql -U postgres -d youtube_optimizer

-- Check all tables exist
\dt

-- Expected output:
-- users
-- youtube_channels
-- youtube_videos
-- video_optimizations
-- video_timeseries_data
-- channel_optimizations
```

### **Step 2: Column Verification**

```sql
-- Verify youtube_channels columns
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'youtube_channels'
ORDER BY ordinal_position;

-- Verify youtube_videos columns
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'youtube_videos'
ORDER BY ordinal_position;
```

### **Step 3: Index Check**

```sql
-- List all indexes
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
```

### **Step 4: Test Queries**

```sql
-- Test basic queries
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM youtube_channels;
SELECT COUNT(*) FROM youtube_videos;

-- Test views
SELECT * FROM video_performance_summary LIMIT 5;
SELECT * FROM channel_summary LIMIT 5;
```

### **Step 5: Application Test**

```bash
# Start application
python main.py

# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "database": "connected",
#   "version": "2.0"
# }
```

---

## ⏮️ Rollback Procedures

If something goes wrong, you can rollback to your backup.

### **Rollback from SQL Backup**

```bash
# 1. Drop the database
psql -U postgres -c "DROP DATABASE youtube_optimizer;"

# 2. Recreate database
psql -U postgres -c "CREATE DATABASE youtube_optimizer;"

# 3. Restore from backup
psql -U postgres -d youtube_optimizer < backup_before_migration.sql

# 4. Verify restoration
psql -U postgres -d youtube_optimizer -c "SELECT COUNT(*) FROM users;"
```

### **Rollback from Binary Backup**

```bash
# 1. Drop the database
psql -U postgres -c "DROP DATABASE youtube_optimizer;"

# 2. Recreate database
psql -U postgres -c "CREATE DATABASE youtube_optimizer;"

# 3. Restore from binary backup
pg_restore -U postgres -d youtube_optimizer \
  "backups/20251002/youtube_optimizer_100000.backup"

# 4. Verify
psql -U postgres -d youtube_optimizer -c "\dt"
```

---

## 🐛 Troubleshooting

### **Issue: Permission Denied**

```
ERROR: permission denied for table users
```

**Solution:**
```sql
-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO youtube_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO youtube_admin;
```

### **Issue: Column Already Exists**

```
ERROR: column "banner_url" of relation "youtube_channels" already exists
```

**Solution:**
This is normal during migration - the script checks for existing columns.

### **Issue: Foreign Key Constraint Violation**

```
ERROR: update or delete on table "youtube_channels" violates foreign key constraint
```

**Solution:**
```sql
-- Temporarily disable constraints
SET CONSTRAINTS ALL DEFERRED;

-- Run your migration
-- ...

-- Re-enable constraints
SET CONSTRAINTS ALL IMMEDIATE;
```

### **Issue: Connection Refused**

```
psql: could not connect to server: Connection refused
```

**Solution:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Start if stopped
sudo systemctl start postgresql

# Check port
sudo netstat -plnt | grep 5432
```

### **Issue: Disk Space**

```
ERROR: could not extend file: No space left on device
```

**Solution:**
```bash
# Check disk space
df -h

# Find large files
du -sh /var/lib/postgresql/*

# Clean old logs
sudo journalctl --vacuum-time=7d
```

---

## 📊 Database Maintenance

### **Regular Maintenance Tasks**

```sql
-- Vacuum and analyze (run weekly)
VACUUM ANALYZE;

-- Reindex (run monthly)
REINDEX DATABASE youtube_optimizer;

-- Update statistics
ANALYZE;

-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### **Performance Tuning**

```sql
-- Check slow queries
SELECT
    query,
    calls,
    total_time,
    mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND schemaname = 'public';
```

---

## 📝 Checklist

Use this checklist for migrations:

- [ ] Backup current database
- [ ] Verify backup integrity
- [ ] Stop application
- [ ] Apply migration script
- [ ] Verify table structure
- [ ] Verify indexes
- [ ] Test basic queries
- [ ] Start application
- [ ] Test health endpoint
- [ ] Test critical features
- [ ] Monitor logs for errors
- [ ] Keep backup for 30 days

---

**Migration Support:** If you encounter issues not covered here, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue on GitHub.
