import logging
from flask import Blueprint, request, jsonify, redirect, current_app
from services.oauth_service import OAuthService
from services.auth_service import AuthService

logger = logging.getLogger(__name__)
oauth_bp = Blueprint('oauth', __name__)

@oauth_bp.route('/auth/google')
def google_login():
    """Initiate Google OAuth flow with YouTube scopes."""
    try:
        auth_url = OAuthService.initiate_oauth_flow()
        return redirect(auth_url)
    except ValueError as e:
        logger.error(f"OAuth initiation error: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=oauth_config_error")
    except Exception as e:
        logger.error(f"Unexpected error initiating OAuth: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=unexpected_error")

@oauth_bp.route('/auth/google/callback')
def google_callback():
    """Handle Google OAuth callback."""
    try:
        # Get callback parameters
        code = request.args.get('code')
        state = request.args.get('state')
        error = request.args.get('error')
        error_description = request.args.get('error_description')
        
        # Handle OAuth callback
        user_info, tokens = OAuthService.handle_oauth_callback(code, state, error, error_description)
        
        # Create API payload and process through existing auth service
        try:
            api_payload = OAuthService.create_api_payload()
            result = AuthService.process_user_authentication(api_payload)
            
            logger.info(f"User authentication processed successfully: {result}")
            
            frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
            return redirect(f"{frontend_url}?auth=success&youtube=enabled")
            
        except Exception as e:
            logger.error(f"Error processing user authentication: {e}")
            frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
            return redirect(f"{frontend_url}?auth=failed&reason=processing_error")
        
    except ValueError as e:
        logger.error(f"OAuth callback error: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=oauth_error")
    except Exception as e:
        logger.error(f"Unexpected error in OAuth callback: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?auth=failed&reason=unexpected_error")

@oauth_bp.route('/auth/status')
def auth_status():
    """Get current authentication status."""
    try:
        status = OAuthService.get_auth_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting auth status: {e}")
        return jsonify({'error': 'Failed to get auth status'}), 500

@oauth_bp.route('/auth/logout')
def logout():
    """Logout and clear session."""
    try:
        user_email = OAuthService.logout()
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?logout=success")
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        return redirect(f"{frontend_url}?logout=error")

@oauth_bp.route('/youtube/test-access')
def test_youtube_access():
    """Test YouTube API access with current session."""
    try:
        result = OAuthService.test_youtube_access()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error testing YouTube access: {e}")
        return jsonify({
            'youtube_access': False,
            'error': str(e),
            'test_successful': False
        }), 500

@oauth_bp.route('/auth/process-session', methods=['POST'])
def process_session_auth():
    """Process authentication using session data (alternative to /api/auth/process)."""
    try:
        # Check if user is authenticated in session
        status = OAuthService.get_auth_status()
        if not status.get('authenticated'):
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Create API payload from session
        api_payload = OAuthService.create_api_payload()
        
        # Process through existing auth service
        result = AuthService.process_user_authentication(api_payload)
        
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Session auth processing error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error processing session auth: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500