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
    
    # OAuth Configuration
    CLIENT_SECRET_FILE = os.getenv('CLIENT_SECRET_FILE', '../client_secret_941974948417-la4udombfq14du8vea6b8jqmo6d8nbv8.apps.googleusercontent.com.json')
    FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')
    
    # Load Google OAuth credentials from JSON file
    GOOGLE_CLIENT_ID = None
    GOOGLE_CLIENT_SECRET = None
    
    def __init__(self):
        """Initialize config and load OAuth credentials."""
        self.google_oauth_config = self._load_oauth_config()
    
    def _load_oauth_config(self):
        """Load Google OAuth config from JSON file."""
        import json
        import os.path
        try:
            # Get absolute path to client secret file
            if not os.path.isabs(self.CLIENT_SECRET_FILE):
                # If relative path, make it relative to the current working directory
                client_secret_path = os.path.join(os.getcwd(), self.CLIENT_SECRET_FILE)
            else:
                client_secret_path = self.CLIENT_SECRET_FILE
            
            print(f"Trying to load OAuth config from: {client_secret_path}")
            
            with open(client_secret_path, 'r') as f:
                credentials = json.load(f)
                web_config = credentials.get('web', {})
                oauth_config = {
                    'client_id': web_config.get('client_id'),
                    'client_secret': web_config.get('client_secret'),
                    'auth_uri': web_config.get('auth_uri', self.GOOGLE_AUTH_URL),
                    'token_uri': web_config.get('token_uri', self.GOOGLE_TOKEN_URL)
                }
                print(f"OAuth config loaded successfully. Client ID: {oauth_config['client_id'][:10]}...")
                return oauth_config
        except Exception as e:
            print(f"Error loading OAuth config: {e}")
            print(f"Falling back to environment variables")
            return {
                'client_id': os.getenv('GOOGLE_CLIENT_ID'),
                'client_secret': os.getenv('GOOGLE_CLIENT_SECRET'),
                'auth_uri': self.GOOGLE_AUTH_URL,
                'token_uri': self.GOOGLE_TOKEN_URL
            }
    
    # Session Configuration
    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Google OAuth URLs
    GOOGLE_AUTH_URL = 'https://accounts.google.com/o/oauth2/auth'
    GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token'
    GOOGLE_USERINFO_URL = 'https://openidconnect.googleapis.com/v1/userinfo'
    
    # YouTube API scopes
    YOUTUBE_SCOPES = [
        'openid',
        'email',
        'profile',
        'https://www.googleapis.com/auth/youtube.readonly',
        'https://www.googleapis.com/auth/yt-analytics.readonly',
        'https://www.googleapis.com/auth/yt-analytics-monetary.readonly',
        'https://www.googleapis.com/auth/youtube'
    ]
    
    # Stripe Configuration
    STRIPE_PUBLIC_KEY = os.getenv('STRIPE_PUBLIC_KEY')
    STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
    STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')

def get_config():
    """Get configuration object."""
    return Config()