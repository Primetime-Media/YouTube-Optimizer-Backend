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
from flask import session

logger = logging.getLogger(__name__)

def parse_duration_to_seconds(duration_str):
    """
    Parse ISO 8601 duration format to total seconds
    Example: PT4M13S -> 253 seconds, PT1H2M10S -> 3730 seconds
    """
    if not duration_str:
        return 0
    
    # Parse ISO 8601 duration format
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0) 
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds

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

def _get_credentials_from_session():
    """
    Get user credentials from session for YouTube API calls.
    """
    tokens = session.get('google_tokens', {})
    access_token = tokens.get('access_token')
    refresh_token = tokens.get('refresh_token')
    
    if not access_token:
        raise ValueError("No access token in session")
    
    # Create credentials object
    credentials = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri='https://oauth2.googleapis.com/token',
        client_id=session.get('client_id'),
        client_secret=session.get('client_secret')
    )
    
    return credentials

def _fetch_channel_info(youtube_client):
    """
    Fetch the user's YouTube channel information.
    """
    try:
        channels_response = youtube_client.channels().list(
            part="snippet,contentDetails,statistics",
            mine=True,
            maxResults=1
        ).execute()
        
        if not channels_response['items']:
            logger.warning("No YouTube channel found for this user")
            return None
            
        return channels_response['items'][0]
        
    except HttpError as e:
        logger.error(f"Error fetching channel info: {e}")
        raise

def _fetch_videos(youtube_client, channel_info, max_videos=1000):
    """
    Fetch the user's uploaded videos.
    """
    try:
        uploads_playlist_id = channel_info['contentDetails']['relatedPlaylists']['uploads']
        
        videos = []
        next_page_token = None
        
        while len(videos) < max_videos:
            # Get video IDs from uploads playlist
            playlist_response = youtube_client.playlistItems().list(
                part='contentDetails',
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_videos - len(videos)),
                pageToken=next_page_token
            ).execute()
            
            video_ids = [item['contentDetails']['videoId'] for item in playlist_response['items']]
            
            if not video_ids:
                break
                
            # Get detailed video information
            videos_response = youtube_client.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(video_ids)
            ).execute()
            
            videos.extend(videos_response['items'])
            
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
                
        logger.info(f"Fetched {len(videos)} videos")
        return videos
        
    except HttpError as e:
        logger.error(f"Error fetching videos: {e}")
        raise

def _store_channel_data(cursor, user_id, channel_info):
    """
    Store channel data in database.
    """
    snippet = channel_info.get('snippet', {})
    statistics = channel_info.get('statistics', {})
    content_details = channel_info.get('contentDetails', {})
    
    cursor.execute("""
        INSERT INTO youtube_channels (
            user_id, channel_id, title, description, 
            published_at, view_count, subscriber_count, 
            video_count, uploads_playlist_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (channel_id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            view_count = EXCLUDED.view_count,
            subscriber_count = EXCLUDED.subscriber_count,
            video_count = EXCLUDED.video_count
        RETURNING id
    """, (
        user_id,
        channel_info['id'],
        snippet.get('title', ''),
        snippet.get('description', ''),
        snippet.get('publishedAt'),
        int(statistics.get('viewCount', 0)),
        int(statistics.get('subscriberCount', 0)),
        int(statistics.get('videoCount', 0)),
        content_details.get('relatedPlaylists', {}).get('uploads')
    ))
    
    result = cursor.fetchone()
    return result[0] if result else None

def _store_video_data(cursor, channel_db_id, videos):
    """
    Store video data in database, filtering out shorts.
    """
    stored_count = 0
    
    for video in videos:
        try:
            snippet = video.get('snippet', {})
            statistics = video.get('statistics', {})
            content_details = video.get('contentDetails', {})
            
            # Filter out shorts (videos under 60 seconds)
            duration_str = content_details.get('duration', '')
            duration_seconds = parse_duration_to_seconds(duration_str)
            
            if duration_seconds < 60:  # Skip shorts
                logger.info(f"Skipping short video {video.get('id', 'unknown')}: {duration_seconds}s")
                continue
            
            cursor.execute("""
                INSERT INTO youtube_videos (
                    channel_id, video_id, title, description,
                    published_at, view_count, like_count,
                    comment_count, duration
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (video_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    view_count = EXCLUDED.view_count,
                    like_count = EXCLUDED.like_count,
                    comment_count = EXCLUDED.comment_count,
                    duration = EXCLUDED.duration
            """, (
                channel_db_id,
                video['id'],
                snippet.get('title', ''),
                snippet.get('description', ''),
                snippet.get('publishedAt'),
                int(statistics.get('viewCount', 0)),
                int(statistics.get('likeCount', 0)),
                int(statistics.get('commentCount', 0)),
                content_details.get('duration', '')  # Store as ISO 8601 string like FastAPI
            ))
            
            stored_count += 1
            
        except Exception as e:
            logger.error(f"Error storing video {video.get('id', 'unknown')}: {e}")
            continue
    
    return stored_count

def fetch_and_store_youtube_data(user_id: int, max_videos: int = 1000):
    """
    Fetch and store YouTube data for a user using session credentials.
    
    Args:
        user_id: Database user ID
        max_videos: Maximum number of videos to fetch
    """
    try:
        logger.info(f"Fetching YouTube data for user {user_id}, max_videos: {max_videos}")
        
        # Get credentials from session
        credentials = _get_credentials_from_session()
        
        # Build YouTube client
        youtube_client = build_youtube_client(credentials)
        
        # Fetch channel info
        channel_info = _fetch_channel_info(youtube_client)
        if not channel_info:
            logger.error(f"Could not fetch channel info for user {user_id}")
            return False
        
        # Fetch videos
        videos = _fetch_videos(youtube_client, channel_info, max_videos)
        
        # Store in database
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Store channel data
                channel_db_id = _store_channel_data(cursor, user_id, channel_info)
                
                if not channel_db_id:
                    logger.error(f"Failed to store channel data for user {user_id}")
                    return False
                
                # Store video data
                stored_count = _store_video_data(cursor, channel_db_id, videos)
                
                conn.commit()
                
                logger.info(f"Successfully stored {stored_count} videos for user {user_id}")
                return True
                
        finally:
            conn.close()
            
    except HttpError as e:
        if e.resp.status == 403 and "quota" in str(e).lower():
            logger.error(f"YouTube API quota exceeded for user {user_id}: {e}")
        else:
            logger.error(f"YouTube API error for user {user_id}: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Error fetching and storing YouTube data for user {user_id}: {e}")
        return False