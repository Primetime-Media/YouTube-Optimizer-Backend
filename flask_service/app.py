import logging
from flask import Flask
from flask_cors import CORS
from config import get_config
from routes.health import health_bp
from routes.auth import auth_bp
from routes.oauth import oauth_bp

def create_app():
    """Flask application factory."""
    app = Flask(__name__)
    config = get_config()
    app.config.from_object(config)
    
    # Add instance attributes to Flask config
    app.config.google_oauth_config = config.google_oauth_config
    
    # Configure CORS
    CORS(app, supports_credentials=True, origins=[
        config.FRONTEND_URL,
        'http://localhost:3000',  # Development fallback
        'http://127.0.0.1:3000',  # Alternative localhost
        'https://www.primetime.media',  # Production frontend
        'https://primetime.media'  # Production frontend (without www)
    ])
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Log configuration status
    logger = logging.getLogger(__name__)
    logger.info("Flask service starting with OAuth support")
    logger.info(f"Frontend URL: {config.FRONTEND_URL}")
    logger.info(f"Client secret file: {config.CLIENT_SECRET_FILE}")
    
    # Test OAuth config loading
    try:
        oauth_config = config.google_oauth_config
        logger.info(f"Google Client ID configured: {bool(oauth_config.get('client_id'))}")
        logger.info(f"YouTube scopes enabled: {len(config.YOUTUBE_SCOPES)} scopes")
    except Exception as e:
        logger.error(f"Error loading OAuth config: {e}")
    
    # Add root route
    @app.route('/')
    def root():
        """Root endpoint."""
        return {
            "service": "youtube-optimizer-flask-oauth",
            "version": "2.0.0", 
            "status": "running",
            "endpoints": {
                "health": "/health",
                "test": "/test",
                "oauth_start": "/auth/google",
                "oauth_callback": "/auth/google/callback",
                "auth_status": "/auth/status"
            }
        }
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(oauth_bp)
    
    return app