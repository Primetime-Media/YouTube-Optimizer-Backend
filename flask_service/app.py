import logging
from flask import Flask
from config import get_config
from routes.health import health_bp
from routes.auth import auth_bp

def create_app():
    """Flask application factory."""
    app = Flask(__name__)
    config = get_config()
    app.config.from_object(config)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(auth_bp)
    
    return app