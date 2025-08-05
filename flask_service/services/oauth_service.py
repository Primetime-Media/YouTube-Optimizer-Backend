import os
import urllib.parse
import secrets
import logging
import requests
from datetime import datetime
from flask import current_app, session, url_for

logger = logging.getLogger(__name__)

class OAuthService:
    """Service for handling Google OAuth flow."""
    
    @staticmethod
    def initiate_oauth_flow():
        """Initiate Google OAuth flow with YouTube scopes."""
        # Clear any existing session
        session.clear()
        
        # Get OAuth configuration
        oauth_config = current_app.config.google_oauth_config
        client_id = oauth_config.get('client_id')
        if not client_id:
            logger.error("Google OAuth client_id not configured")
            raise ValueError("OAuth not configured")
        
        # Generate state parameter for security
        state = secrets.token_urlsafe(32)
        session['oauth_state'] = state
        
        # Build redirect URI - use HTTP for development, HTTPS for production
        scheme = 'http' if current_app.config['DEBUG'] else 'https'
        redirect_uri = url_for('oauth.google_callback', _external=True, _scheme=scheme)
        
        # Build Google OAuth URL with YouTube scopes
        params = {
            'client_id': client_id,
            'redirect_uri': redirect_uri,
            'scope': ' '.join(current_app.config['YOUTUBE_SCOPES']),
            'response_type': 'code',
            'state': state,
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        auth_url = f"{oauth_config.get('auth_uri')}?{urllib.parse.urlencode(params)}"
        
        logger.info(f"Starting OAuth flow with YouTube scopes")
        logger.info(f"Redirect URI: {redirect_uri}")
        logger.info(f"State: {state}")
        
        return auth_url
    
    @staticmethod
    def handle_oauth_callback(code, state, error=None, error_description=None):
        """Handle OAuth callback and exchange code for tokens."""
        logger.info(f"OAuth callback received:")
        logger.info(f"Code: {bool(code)}")
        logger.info(f"State: {bool(state)}")
        logger.info(f"Error: {error}")
        
        # Handle OAuth errors
        if error:
            logger.error(f"OAuth error from Google: {error} - {error_description}")
            raise ValueError(f"OAuth error: {error}")
        
        if not code:
            logger.error("No authorization code received from Google")
            raise ValueError("No authorization code received")
        
        # Verify state parameter (CSRF protection)
        expected_state = session.get('oauth_state')
        if state != expected_state:
            logger.error(f"State parameter mismatch. Expected: {expected_state}, Got: {state}")
            raise ValueError("State parameter mismatch")
        
        logger.info("State verification passed")
        
        # Exchange code for tokens
        tokens = OAuthService._exchange_code_for_tokens(code)
        
        # Get user information
        user_info = OAuthService._get_user_info(tokens['access_token'])
        
        # Store in session
        session['user_data'] = user_info
        session['google_tokens'] = tokens
        
        # Clear the OAuth state
        session.pop('oauth_state', None)
        
        logger.info(f"Session stored successfully for user: {user_info.get('email')}")
        logger.info(f"YouTube API access granted with scopes: {tokens.get('scope', 'Unknown')}")
        
        return user_info, tokens
    
    @staticmethod
    def _exchange_code_for_tokens(code):
        """Exchange authorization code for access tokens."""
        oauth_config = current_app.config.google_oauth_config
        scheme = 'http' if current_app.config['DEBUG'] else 'https'
        redirect_uri = url_for('oauth.google_callback', _external=True, _scheme=scheme)
        token_data = {
            'client_id': oauth_config.get('client_id'),
            'client_secret': oauth_config.get('client_secret'),
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': redirect_uri
        }
        
        logger.info("Exchanging authorization code for tokens...")
        
        token_response = requests.post(
            oauth_config.get('token_uri'),
            data=token_data,
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            },
            timeout=10
        )
        
        if token_response.status_code != 200:
            logger.error(f"Token exchange failed: {token_response.status_code}")
            logger.error(f"Response: {token_response.text}")
            raise ValueError("Token exchange failed")
        
        tokens = token_response.json()
        access_token = tokens.get('access_token')
        
        if not access_token:
            logger.error("No access token in response")
            logger.error(f"Token response: {tokens}")
            raise ValueError("No access token received")
        
        logger.info("Access token received successfully")
        logger.info(f"Token type: {tokens.get('token_type', 'Bearer')}")
        logger.info(f"Expires in: {tokens.get('expires_in')} seconds")
        logger.info(f"Has ID token: {bool(tokens.get('id_token'))}")
        logger.info(f"Has refresh token: {bool(tokens.get('refresh_token'))}")
        logger.info(f"Granted scopes: {tokens.get('scope', 'No scope info')}")
        
        return tokens
    
    @staticmethod
    def _get_user_info(access_token):
        """Get user information from Google."""
        logger.info("Fetching user information from Google...")
        
        user_response = requests.get(
            current_app.config['GOOGLE_USERINFO_URL'],
            headers={
                'Authorization': f"Bearer {access_token}",
                'Accept': 'application/json'
            },
            timeout=10
        )
        
        if user_response.status_code != 200:
            logger.error(f"User info request failed: {user_response.status_code}")
            logger.error(f"Response: {user_response.text}")
            raise ValueError("Failed to get user information")
        
        user_info = user_response.json()
        
        logger.info(f"User information retrieved:")
        logger.info(f"Email: {user_info.get('email')}")
        logger.info(f"Name: {user_info.get('name')}")
        logger.info(f"Google ID: {user_info.get('sub')}")
        logger.info(f"Email verified: {user_info.get('email_verified')}")
        
        if not user_info or not user_info.get('sub'):
            logger.error("Invalid user info received from Google")
            logger.error(f"User info: {user_info}")
            raise ValueError("Invalid user information")
        
        return user_info
    
    @staticmethod
    def get_auth_status():
        """Get current authentication status."""
        user_data = session.get('user_data')
        tokens = session.get('google_tokens', {})
        
        if user_data:
            # Parse granted scopes
            granted_scopes = tokens.get('scope', '').split(' ') if tokens.get('scope') else []
            youtube_scopes_granted = [scope for scope in granted_scopes if 'youtube' in scope or 'yt-analytics' in scope]
            
            return {
                'authenticated': True,
                'user': {
                    'name': user_data.get('name'),
                    'email': user_data.get('email'),
                    'picture': user_data.get('picture'),
                    'google_id': user_data.get('sub'),
                    'email_verified': user_data.get('email_verified')
                },
                'tokens': {
                    'has_access_token': bool(tokens.get('access_token')),
                    'has_refresh_token': bool(tokens.get('refresh_token')),
                    'has_id_token': bool(tokens.get('id_token')),
                    'token_type': tokens.get('token_type'),
                    'expires_in': tokens.get('expires_in')
                },
                'youtube_access': {
                    'enabled': len(youtube_scopes_granted) > 0,
                    'granted_scopes': youtube_scopes_granted,
                    'all_granted_scopes': granted_scopes
                }
            }
        
        return {
            'authenticated': False,
            'session_exists': len(session) > 0,
            'youtube_access': {'enabled': False}
        }
    
    @staticmethod
    def logout():
        """Logout and clear session."""
        user_email = session.get('user_data', {}).get('email', 'unknown')
        session_keys = list(session.keys())
        
        session.clear()
        
        logger.info(f"User logged out: {user_email}")
        logger.info(f"Cleared session keys: {session_keys}")
        
        return user_email
    
    @staticmethod
    def create_api_payload():
        """Create API payload from session data for processing."""
        user_data = session.get('user_data')
        tokens = session.get('google_tokens', {})
        
        if not user_data or not tokens:
            raise ValueError("No user data or tokens in session")
        
        api_payload = {
            "user": {
                "google_id": user_data.get('sub'),
                "email": user_data.get('email'),
                "name": user_data.get('name'),
                "picture_url": user_data.get('picture'),
                "email_verified": user_data.get('email_verified', False),
                "locale": user_data.get('locale', 'en')
            },
            "auth": {
                "google_access_token": tokens.get('access_token'),
                "google_id_token": tokens.get('id_token'),
                "expires_in": tokens.get('expires_in'),
                "token_type": tokens.get('token_type', 'Bearer'),
                "granted_scopes": tokens.get('scope', '').split(' '),
                "youtube_access": True
            },
            "metadata": {
                "platform": "flask_service",
                "auth_timestamp": datetime.now().isoformat(),
                "auth_method": "oauth_youtube_scopes"
            }
        }
        
        return api_payload
    
    @staticmethod
    def test_youtube_access():
        """Test YouTube API access with current tokens."""
        tokens = session.get('google_tokens', {})
        access_token = tokens.get('access_token')
        
        if not access_token:
            return {'error': 'No access token', 'youtube_access': False}
        
        try:
            youtube_response = requests.get(
                'https://www.googleapis.com/youtube/v3/channels',
                params={'part': 'snippet', 'mine': 'true'},
                headers={'Authorization': f'Bearer {access_token}'},
                timeout=10
            )
            
            if youtube_response.status_code == 200:
                channels = youtube_response.json()
                return {
                    'youtube_access': True,
                    'channels_found': len(channels.get('items', [])),
                    'api_quota_used': True,
                    'test_successful': True
                }
            else:
                return {
                    'youtube_access': False,
                    'error': f'YouTube API returned {youtube_response.status_code}',
                    'response': youtube_response.text
                }
        
        except Exception as e:
            return {
                'youtube_access': False,
                'error': str(e),
                'test_successful': False
            }