import logging
from flask import Blueprint, request, jsonify
from services.auth_service import AuthService

logger = logging.getLogger(__name__)
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/auth/process', methods=['POST'])
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
        
        # Process authentication through service
        result = AuthService.process_user_authentication(auth_data)
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing auth data: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500