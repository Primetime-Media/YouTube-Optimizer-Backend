import logging
from models.user import UserModel
from services.youtube import fetch_and_store_youtube_data

logger = logging.getLogger(__name__)

class AuthService:
    """Service for handling authentication and user onboarding."""
    
    @staticmethod
    def process_user_authentication(auth_data):
        """
        Process authentication data from the homepage.
        Stores user data, fetches YouTube videos, and queues them for optimization.
        """
        # Extract user and auth data
        user_data = auth_data.get('user', {})
        auth_info = auth_data.get('auth', {})
        metadata = auth_data.get('metadata', {})
        
        # Extract 30-day limit flag (defaults to False for now)
        limit_to_30_days = auth_data.get('limit_to_30_days', False)
        
        # Validate required fields
        required_user_fields = ['google_id', 'email', 'name']
        required_auth_fields = ['google_access_token', 'google_id_token']
        
        for field in required_user_fields:
            if not user_data.get(field):
                raise ValueError(f"Missing required user field: {field}")
        
        for field in required_auth_fields:
            if not auth_info.get(field):
                raise ValueError(f"Missing required auth field: {field}")
        
        logger.info(f"Processing auth data for user: {user_data.get('email')}")
        
        # Store user in database
        user_id = UserModel.store_user_data(user_data, auth_info, metadata)
        
        # Fetch and store YouTube data in background
        try:
            fetch_and_store_youtube_data(user_id=user_id, max_videos=1000)
            logger.info(f"YouTube data fetch initiated for user {user_id}")
        except Exception as e:
            logger.error(f"Error fetching YouTube data: {e}")
        
        # Queue videos for optimization
        try:
            queued_count = UserModel.queue_user_videos_for_optimization(user_id, limit_to_30_days)
            logger.info(f"Queued {queued_count} videos for optimization for user {user_id} (30-day limit: {limit_to_30_days})")
        except Exception as e:
            logger.error(f"Error queueing videos for optimization: {e}")
            queued_count = 0
        
        return {
            "status": "success",
            "message": "User authenticated and data processed successfully",
            "user_id": user_id,
            "email": user_data.get('email'),
            "videos_queued": queued_count
        }