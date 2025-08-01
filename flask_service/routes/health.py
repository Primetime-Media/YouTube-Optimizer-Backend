import os
from datetime import datetime
from flask import Blueprint, jsonify, current_app

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "service": "youtube-optimizer-flask-oauth",
        "version": "2.0.0",
        "oauth_enabled": True,
        "timestamp": datetime.now().isoformat()
    })

@health_bp.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify server and OAuth configuration."""
    config = current_app.config
    
    return jsonify({
        'message': 'Flask OAuth service is running!',
        'server_status': 'healthy',
        'oauth_config': {
            'client_secret_file': config.get('CLIENT_SECRET_FILE'),
            'has_oauth_config': bool(config.google_oauth_config.get('client_id')),
            'frontend_url': config.get('FRONTEND_URL'),
            'youtube_scopes_count': len(config.get('YOUTUBE_SCOPES', [])),
            'session_configured': bool(config.get('SECRET_KEY'))
        },
        'available_endpoints': {
            'oauth': [
                'GET /auth/google - Start OAuth flow',
                'GET /auth/google/callback - OAuth callback',
                'GET /auth/status - Check auth status',
                'GET /auth/logout - Logout',
                'GET /youtube/test-access - Test YouTube API',
                'POST /auth/process-session - Process session auth'
            ],
            'existing': [
                'POST /api/auth/process - Process auth data',
                'GET /health - Health check',
                'GET /test - This test endpoint'
            ]
        },
        'youtube_scopes': config.get('YOUTUBE_SCOPES', []),
        'timestamp': datetime.now().isoformat()
    })