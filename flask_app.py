from flask import Flask, request, jsonify
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from utils.db import get_connection
from services.youtube import fetch_and_store_youtube_data
# Note: queue_videos_for_optimization is implemented locally in this file

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "youtube-optimizer-flask"})

@app.route('/api/auth/process', methods=['POST'])
def process_auth_data():
    """
    Process authentication data from the homepage.
    Stores user data, fetches YouTube videos, and queues them for optimization.
    """
    try:
        # Get the JSON data from the request
        auth_data = request.get_json()
        
        if not auth_data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract user and auth data
        user_data = auth_data.get('user', {})
        auth_info = auth_data.get('auth', {})
        metadata = auth_data.get('metadata', {})
        
        # Validate required fields
        required_user_fields = ['google_id', 'email', 'name']
        required_auth_fields = ['google_access_token', 'google_id_token']
        
        for field in required_user_fields:
            if not user_data.get(field):
                return jsonify({"error": f"Missing required user field: {field}"}), 400
        
        for field in required_auth_fields:
            if not auth_info.get(field):
                return jsonify({"error": f"Missing required auth field: {field}"}), 400
        
        logger.info(f"Processing auth data for user: {user_data.get('email')}")
        
        # Store user in database
        user_id = store_user_data(user_data, auth_info, metadata)
        
        # Fetch and store YouTube data in background
        try:
            fetch_and_store_youtube_data(user_id=user_id, max_videos=1000)
            logger.info(f"YouTube data fetch initiated for user {user_id}")
        except Exception as e:
            logger.error(f"Error fetching YouTube data: {e}")
        
        # Queue videos for optimization
        try:
            queued_count = queue_user_videos_for_optimization(user_id)
            logger.info(f"Queued {queued_count} videos for optimization for user {user_id}")
        except Exception as e:
            logger.error(f"Error queueing videos for optimization: {e}")
            queued_count = 0
        
        return jsonify({
            "status": "success",
            "message": "User authenticated and data processed successfully",
            "user_id": user_id,
            "email": user_data.get('email'),
            "videos_queued": queued_count
        })
        
    except Exception as e:
        logger.error(f"Error processing auth data: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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

def queue_user_videos_for_optimization(user_id):
    """Queue all user videos for optimization."""
    conn = get_connection()
    
    try:
        with conn.cursor() as cursor:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)