"""
YouTube Optimizer - Flask Authentication Service

Lightweight Flask service for user authentication and initial data setup.
Handles OAuth processing, user storage, and video queueing for optimization.
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from utils.db import get_connection
from services.youtube import fetch_and_store_youtube_data
# Note: queue_videos_for_optimization is implemented locally in this file

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize Flask application
app = Flask(__name__)
logger = logging.getLogger(__name__)

# Health check endpoint for monitoring and load balancers
@app.route('/health', methods=['GET'])
def health_check():
    """Return service health status for monitoring systems."""
    return jsonify({
        "status": "healthy", 
        "service": "youtube-optimizer-flask"
    })

# =============================================================================
# AUTHENTICATION PROCESSING ENDPOINT
# =============================================================================

@app.route('/api/auth/process', methods=['POST'])
def process_auth_data():
    """
    Process authentication data from the frontend homepage.
    
    This endpoint handles the complete user onboarding process:
    1. Validates incoming authentication data from Google OAuth
    2. Stores user credentials and metadata in the database
    3. Fetches and stores the user's YouTube channel and video data
    4. Queues recent videos for optimization processing
    
    The process is designed to be fault-tolerant, with each step handling
    errors gracefully to ensure partial success doesn't break the entire flow.
    
    Expected Request Body:
        {
            "user": {
                "google_id": "string",
                "email": "string", 
                "name": "string"
            },
            "auth": {
                "google_access_token": "string",
                "google_id_token": "string",
                "granted_scopes": ["string"],
                "expires_in": number
            },
            "metadata": {
                "additional_user_data": "any"
            }
        }
    
    Returns:
        JSON response with processing status and user information
        
    Raises:
        400: If required fields are missing
        500: If internal processing errors occur
    """
    try:
        # Extract JSON data from the HTTP request
        auth_data = request.get_json()
        
        # Validate that request contains data
        if not auth_data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract structured data from the request
        user_data = auth_data.get('user', {})      # User profile information
        auth_info = auth_data.get('auth', {})      # OAuth tokens and credentials
        metadata = auth_data.get('metadata', {})   # Additional metadata
        
        # Validate required user profile fields
        required_user_fields = ['google_id', 'email', 'name']
        for field in required_user_fields:
            if not user_data.get(field):
                return jsonify({"error": f"Missing required user field: {field}"}), 400
        
        # Validate required authentication fields
        required_auth_fields = ['google_access_token', 'google_id_token']
        for field in required_auth_fields:
            if not auth_info.get(field):
                return jsonify({"error": f"Missing required auth field: {field}"}), 400
        
        logger.info(f"Processing auth data for user: {user_data.get('email')}")
        
        # Store user data in the database
        # This includes user profile, OAuth credentials, and permissions
        user_id = store_user_data(user_data, auth_info, metadata)
        
        # Fetch and store YouTube data in the background
        # This is a potentially long-running operation, so we handle errors gracefully
        try:
            fetch_and_store_youtube_data(user_id=user_id, max_videos=1000)
            logger.info(f"YouTube data fetch initiated for user {user_id}")
        except Exception as e:
            logger.error(f"Error fetching YouTube data: {e}")
            # Continue processing even if YouTube data fetch fails
        
        # Queue recent videos for optimization processing
        # Only videos from the last 30 days are queued to focus on recent content
        try:
            queued_count = queue_user_videos_for_optimization(user_id)
            logger.info(f"Queued {queued_count} videos for optimization for user {user_id}")
        except Exception as e:
            logger.error(f"Error queueing videos for optimization: {e}")
            queued_count = 0  # Set to 0 if queueing fails
        
        # Return success response with user information
        return jsonify({
            "status": "success",
            "message": "User authenticated and data processed successfully",
            "user_id": user_id,
            "email": user_data.get('email'),
            "videos_queued": queued_count
        })
        
    except Exception as e:
        # Log the error and return a generic error response
        logger.error(f"Error processing auth data: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# =============================================================================
# DATABASE HELPER FUNCTIONS
# =============================================================================

def store_user_data(user_data, auth_info, metadata):
    """
    Store user authentication data in the PostgreSQL database.
    
    This function handles the complete user data storage process, including:
    - Converting OAuth scopes to PostgreSQL array format
    - Calculating token expiration times
    - Inserting new users or updating existing ones
    - Handling database transactions with proper rollback on errors
    
    Args:
        user_data (dict): User profile information (google_id, email, name)
        auth_info (dict): OAuth authentication data (tokens, scopes, expiry)
        metadata (dict): Additional user metadata (currently unused)
        
    Returns:
        int: The database user ID of the stored/updated user
        
    Raises:
        Exception: If database operations fail
    """
    conn = get_connection()
    
    try:
        with conn.cursor() as cursor:
            # Convert OAuth scopes to PostgreSQL array format
            # PostgreSQL arrays are represented as {item1,item2,item3}
            granted_scopes = auth_info.get('granted_scopes', [])
            if isinstance(granted_scopes, list):
                scopes_array = "{" + ",".join(granted_scopes) + "}"
            else:
                scopes_array = "{}"  # Empty array if no scopes provided
            
            # Calculate token expiration time
            # Default to 1 hour if expires_in is not provided
            expires_in = auth_info.get('expires_in', 3600)
            token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            
            # Insert new user or update existing user if google_id already exists
            # This uses PostgreSQL's ON CONFLICT clause for upsert functionality
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
                user_data['google_id'],                    # Unique Google user identifier
                user_data['email'],                        # User's email address
                user_data['name'],                         # User's display name
                'readwrite',                               # Default permission level
                False,                                     # Default is_free_trial status
                auth_info['google_access_token'],          # OAuth access token
                None,                                      # No refresh token in provided data
                'https://oauth2.googleapis.com/token',     # Standard Google token URI
                None,                                      # Client ID not provided in auth data
                None,                                      # Client secret not provided in auth data
                scopes_array,                              # OAuth scopes as PostgreSQL array
                token_expiry,                              # Calculated token expiration time
                datetime.now(timezone.utc)                 # Current timestamp for updated_at
            ))
            
            # Get the user ID from the database
            user_id = cursor.fetchone()[0]
            conn.commit()  # Commit the transaction
            
            logger.info(f"User {user_id} stored/updated in database")
            return user_id
            
    except Exception as e:
        # Rollback the transaction on any error
        conn.rollback()
        logger.error(f"Error storing user data: {e}")
        raise
    finally:
        # Always close the database connection
        conn.close()

def queue_user_videos_for_optimization(user_id):
    """
    Queue user's recent videos for optimization processing.
    
    This function identifies videos that are eligible for optimization and marks them
    as queued in the database. Only videos from the last 30 days are considered
    to focus optimization efforts on recent, relevant content.
    
    The function filters out videos that are:
    - Already queued for optimization
    - Already optimized
    - Published more than 30 days ago
    
    Args:
        user_id (int): The database user ID to queue videos for
        
    Returns:
        int: Number of videos queued for optimization
        
    Raises:
        Exception: If database operations fail
    """
    conn = get_connection()
    
    try:
        with conn.cursor() as cursor:
            # Calculate the cutoff date for recent videos (30 days ago)
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            
            # Find videos that are eligible for optimization
            # This query joins videos with channels to filter by user
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
            
            # If no videos are found, return early
            if not videos:
                logger.info(f"No videos to queue for user {user_id}")
                return 0
            
            # Extract video IDs for the update query
            video_ids = [video[0] for video in videos]
            
            # Mark all eligible videos as queued for optimization
            cursor.execute("""
                UPDATE youtube_videos 
                SET queued_for_optimization = TRUE, updated_at = NOW()
                WHERE id = ANY(%s)
            """, (video_ids,))
            
            # Commit the transaction
            conn.commit()
            
            logger.info(f"Queued {len(videos)} videos for optimization for user {user_id}")
            return len(videos)
            
    except Exception as e:
        # Rollback the transaction on any error
        conn.rollback()
        logger.error(f"Error queueing videos for optimization: {e}")
        raise
    finally:
        # Always close the database connection
        conn.close()

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    """
    Main entry point for running the Flask application directly.
    
    This is used when running the application with 'python flask_app.py' instead of
    using a WSGI server like gunicorn. The application will start on host 0.0.0.0 
    and port 5001.
    
    Configuration:
    - host='0.0.0.0': Listen on all available network interfaces
    - port=5001: Use port 5001 for the HTTP server
    - debug=True: Enable debug mode for development
    """
    app.run(host='0.0.0.0', port=5001, debug=True)