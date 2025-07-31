import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Flask application configuration."""
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    DATABASE_URL = os.getenv('DATABASE_URL')
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5001))

def get_config():
    """Get configuration object."""
    return Config()