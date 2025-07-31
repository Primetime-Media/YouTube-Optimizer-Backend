import os
import logging
import html
import re
import json
import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from utils.db import get_connection
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import timezone

logger = logging.getLogger(__name__)

def build_youtube_client(credentials):
    """
    Build a YouTube API client with user credentials to maximize use of their quota.
    
    Args:
        credentials: User OAuth2 credentials
        
    Returns:
        YouTube API client object
    """
    try:
        logger.info("Building YouTube API client with user credentials")
        return build(
            'youtube', 
            'v3', 
            credentials=credentials,
            cache_discovery=False,  # Disable cache to ensure fresh credentials are used
        )
    except Exception as e:
        logger.error(f"Error building YouTube client: {e}")
        raise

def fetch_and_store_youtube_data(user_id: int, max_videos: int = 1000):
    """
    Simplified version of YouTube data fetching for Flask service.
    This function triggers the background process but doesn't implement the full logic.
    The actual implementation would be handled by the main FastAPI service.
    """
    logger.info(f"YouTube data fetch requested for user {user_id}, max_videos: {max_videos}")
    # In a real implementation, this might trigger a background job
    # or call the main FastAPI service to handle the heavy lifting
    return True