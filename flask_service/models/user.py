import logging
from datetime import datetime, timezone, timedelta
from utils.db import get_connection

logger = logging.getLogger(__name__)

class UserModel:
    """User model for database operations."""
    
    @staticmethod
    def store_user_data(user_data, auth_info, metadata):
        """Store user authentication data in the database."""
        conn = get_connection()
        
        try:
            with conn.cursor() as cursor:
                # Convert scopes to PostgreSQL array format
                granted_scopes = auth_info.get('granted_scopes', [])
                if isinstance(granted_scopes, list):
                    scopes_array = "{" + ",".join(granted_scopes) + "}"
                else:
                    scopes_array = "{}"
                
                # Calculate token expiry
                expires_in = auth_info.get('expires_in', 3600)
                token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                
                # Insert or update user
                cursor.execute("""
                    INSERT INTO users (
                        google_id, email, name, permission_level, is_free_trial,
                        token, refresh_token, token_uri, client_id, client_secret, 
                        scopes, token_expiry, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (google_id) DO UPDATE SET
                        email = EXCLUDED.email,
                        name = EXCLUDED.name,
                        token = EXCLUDED.token,
                        refresh_token = EXCLUDED.refresh_token,
                        scopes = EXCLUDED.scopes,
                        token_expiry = EXCLUDED.token_expiry,
                        updated_at = EXCLUDED.updated_at
                    RETURNING id
                """, (
                    user_data['google_id'],
                    user_data['email'], 
                    user_data['name'],
                    'readwrite',  # Default permission level
                    False,  # Default is_free_trial
                    auth_info['google_access_token'],
                    None,  # No refresh token in the provided data
                    'https://oauth2.googleapis.com/token',  # Standard Google token URI
                    None,  # Client ID not provided in auth data
                    None,  # Client secret not provided in auth data
                    scopes_array,
                    token_expiry,
                    datetime.now(timezone.utc)
                ))
                
                user_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"User {user_id} stored/updated in database")
                return user_id
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing user data: {e}")
            raise
        finally:
            conn.close()
    
    @staticmethod
    def queue_user_videos_for_optimization(user_id, limit_to_30_days=True):
        """Queue user videos for optimization with optional 30-day limit."""
        conn = get_connection()
        
        try:
            with conn.cursor() as cursor:
                # Build query based on 30-day limit flag
                if limit_to_30_days:
                    # Get videos published in the last 30 days that aren't already queued or optimized
                    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
                    cursor.execute("""
                        SELECT v.id, v.video_id
                        FROM youtube_videos v
                        JOIN youtube_channels c ON v.channel_id = c.id  
                        WHERE c.user_id = %s 
                        AND v.queued_for_optimization = FALSE 
                        AND v.is_optimized = FALSE
                        AND v.published_at >= %s
                    """, (user_id, thirty_days_ago))
                else:
                    # Get all videos that aren't already queued or optimized
                    cursor.execute("""
                        SELECT v.id, v.video_id
                        FROM youtube_videos v
                        JOIN youtube_channels c ON v.channel_id = c.id  
                        WHERE c.user_id = %s 
                        AND v.queued_for_optimization = FALSE 
                        AND v.is_optimized = FALSE
                    """, (user_id,))
                
                videos = cursor.fetchall()
                
                if not videos:
                    logger.info(f"No videos to queue for user {user_id}")
                    return 0
                
                # Mark videos as queued for optimization
                video_ids = [video[0] for video in videos]
                cursor.execute("""
                    UPDATE youtube_videos 
                    SET queued_for_optimization = TRUE, updated_at = NOW()
                    WHERE id = ANY(%s)
                """, (video_ids,))
                
                conn.commit()
                
                logger.info(f"Queued {len(videos)} videos for optimization for user {user_id}")
                return len(videos)
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Error queueing videos for optimization: {e}")
            raise
        finally:
            conn.close()