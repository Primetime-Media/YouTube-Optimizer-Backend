import logging
import re
import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from utils.db import get_connection
from typing import Dict
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
                
            # Get detailed video information (include status part for comprehensive data)
            videos_response = youtube_client.videos().list(
                part='snippet,statistics,contentDetails,status',
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
    Store channel data in database using FastAPI-compatible comprehensive schema.
    """
    snippet = channel_info.get('snippet', {})
    statistics = channel_info.get('statistics', {})
    content_details = channel_info.get('contentDetails', {})
    
    # Parse thumbnails like FastAPI does
    thumbnails = snippet.get('thumbnails', {})
    
    # Extract enhanced channel data (minimal version, can be expanded)
    banner_url = None
    privacy_status = None
    is_linked = False
    long_uploads_status = None
    is_monetization_enabled = False
    
    # Topic details
    topic_ids = []
    topic_categories = []
    
    # Channel standing info (defaults)
    overall_good_standing = True
    community_guidelines_good_standing = True
    copyright_strikes_good_standing = True
    content_id_claims_good_standing = True
    
    # Raw JSON data storage
    branding_settings = None
    audit_details = None
    topic_details = None
    status_details = None
    
    # Use the exact same INSERT statement as FastAPI
    cursor.execute("""
        INSERT INTO youtube_channels (
            user_id, channel_id, kind, etag, title, description, 
            custom_url, published_at, view_count, subscriber_count, 
            hidden_subscriber_count, video_count, 
            thumbnail_url_default, thumbnail_url_medium, thumbnail_url_high,
            uploads_playlist_id,
            
            -- Enhanced channel optimization fields
            banner_url, privacy_status, is_linked, long_uploads_status,
            is_monetization_enabled, topic_ids, topic_categories,
            overall_good_standing, community_guidelines_good_standing,
            copyright_strikes_good_standing, content_id_claims_good_standing,
            branding_settings, audit_details, topic_details, status_details
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (channel_id) DO UPDATE SET
            kind = EXCLUDED.kind,
            etag = EXCLUDED.etag,
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            custom_url = EXCLUDED.custom_url,
            published_at = EXCLUDED.published_at,
            view_count = EXCLUDED.view_count,
            subscriber_count = EXCLUDED.subscriber_count,
            hidden_subscriber_count = EXCLUDED.hidden_subscriber_count,
            video_count = EXCLUDED.video_count,
            thumbnail_url_default = EXCLUDED.thumbnail_url_default,
            thumbnail_url_medium = EXCLUDED.thumbnail_url_medium,
            thumbnail_url_high = EXCLUDED.thumbnail_url_high,
            uploads_playlist_id = EXCLUDED.uploads_playlist_id,
            banner_url = EXCLUDED.banner_url,
            privacy_status = EXCLUDED.privacy_status,
            is_linked = EXCLUDED.is_linked,
            long_uploads_status = EXCLUDED.long_uploads_status,
            is_monetization_enabled = EXCLUDED.is_monetization_enabled,
            topic_ids = EXCLUDED.topic_ids,
            topic_categories = EXCLUDED.topic_categories,
            overall_good_standing = EXCLUDED.overall_good_standing,
            community_guidelines_good_standing = EXCLUDED.community_guidelines_good_standing,
            copyright_strikes_good_standing = EXCLUDED.copyright_strikes_good_standing,
            content_id_claims_good_standing = EXCLUDED.content_id_claims_good_standing,
            branding_settings = EXCLUDED.branding_settings,
            audit_details = EXCLUDED.audit_details,
            topic_details = EXCLUDED.topic_details,
            status_details = EXCLUDED.status_details,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
    """, (
        user_id,
        channel_info['id'],
        channel_info.get('kind', 'youtube#channel'),
        channel_info.get('etag', ''),
        snippet.get('title', ''),
        snippet.get('description', ''),
        snippet.get('customUrl'),
        snippet.get('publishedAt'),
        int(statistics.get('viewCount', 0)),
        int(statistics.get('subscriberCount', 0)),
        statistics.get('hiddenSubscriberCount', False),
        int(statistics.get('videoCount', 0)),
        thumbnails.get('default', {}).get('url'),
        thumbnails.get('medium', {}).get('url'),
        thumbnails.get('high', {}).get('url'),
        content_details.get('relatedPlaylists', {}).get('uploads'),
        # Enhanced fields
        banner_url,
        privacy_status,
        is_linked,
        long_uploads_status,
        is_monetization_enabled,
        topic_ids,
        topic_categories,
        overall_good_standing,
        community_guidelines_good_standing,
        copyright_strikes_good_standing,
        content_id_claims_good_standing,
        branding_settings,
        audit_details,
        topic_details,
        status_details
    ))
    
    result = cursor.fetchone()
    return result[0] if result else None

def _store_video_data(cursor, db_channel_id, videos):
    """
    Store video data in database using FastAPI-compatible comprehensive schema, filtering out shorts.
    """
    stored_count = 0
    
    for video in videos:
        try:
            snippet = video.get('snippet', {})
            statistics = video.get('statistics', {})
            content_details = video.get('contentDetails', {})
            status = video.get('status', {})
            
            # Filter out shorts (videos up to 3 minutes as of Oct 2024)
            duration_str = content_details.get('duration', '')
            duration_seconds = parse_duration_to_seconds(duration_str)
            
            if duration_seconds <= 180:  # Skip shorts (YouTube Shorts max is now 3 minutes)
                logger.info(f"Skipping potential short video {video.get('id', 'unknown')}: {duration_seconds}s (≤3min)")
                continue
            
            # Parse thumbnails like FastAPI does
            thumbnails = snippet.get('thumbnails', {})
            
            # Enhanced video fields (defaults)
            privacy_status = status.get('privacyStatus', 'public')
            upload_status = status.get('uploadStatus', 'processed')
            license = status.get('license', 'youtube')
            embeddable = status.get('embeddable', True)
            public_stats_viewable = status.get('publicStatsViewable', True)
            
            # Content details
            definition = content_details.get('definition', 'hd')
            dimension = content_details.get('dimension', '2d')
            has_custom_thumbnail = content_details.get('hasCustomThumbnail', False)
            projection = content_details.get('projection', 'rectangular')
            
            # Category
            category_id = snippet.get('categoryId', '22')  # Default to People & Blogs
            category_name = ''  # Could be looked up from category_id
            
            # Topic details
            topic_ids = []
            topic_categories = []
            
            # Raw JSON data storage
            import json
            content_details_json = json.dumps(content_details) if content_details else None
            status_details_json = json.dumps(status) if status else None
            topic_details_json = None
            
            # Use the exact same INSERT statement as FastAPI
            cursor.execute("""
                INSERT INTO youtube_videos (
                    channel_id, video_id, kind, etag, playlist_item_id,
                    title, description, published_at, channel_title,
                    playlist_id, position, tags,
                    thumbnail_url_default, thumbnail_url_medium, 
                    thumbnail_url_high, thumbnail_url_standard, 
                    thumbnail_url_maxres,
                    view_count, like_count, comment_count, duration,
                    transcript, has_captions, caption_language,
                    
                    -- Enhanced video fields
                    privacy_status, upload_status, license, embeddable, 
                    public_stats_viewable, definition, dimension, 
                    has_custom_thumbnail, projection, category_id, category_name,
                    topic_ids, topic_categories, 
                    content_details, status_details, topic_details
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (video_id) DO UPDATE SET
                    kind = EXCLUDED.kind,
                    etag = EXCLUDED.etag,
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    channel_title = EXCLUDED.channel_title,
                    playlist_id = EXCLUDED.playlist_id,
                    position = EXCLUDED.position,
                    tags = EXCLUDED.tags,
                    thumbnail_url_default = EXCLUDED.thumbnail_url_default,
                    thumbnail_url_medium = EXCLUDED.thumbnail_url_medium,
                    thumbnail_url_high = EXCLUDED.thumbnail_url_high,
                    thumbnail_url_standard = EXCLUDED.thumbnail_url_standard,
                    thumbnail_url_maxres = EXCLUDED.thumbnail_url_maxres,
                    view_count = EXCLUDED.view_count,
                    like_count = EXCLUDED.like_count,
                    comment_count = EXCLUDED.comment_count,
                    duration = EXCLUDED.duration,
                    privacy_status = EXCLUDED.privacy_status,
                    upload_status = EXCLUDED.upload_status,
                    license = EXCLUDED.license,
                    embeddable = EXCLUDED.embeddable,
                    public_stats_viewable = EXCLUDED.public_stats_viewable,
                    definition = EXCLUDED.definition,
                    dimension = EXCLUDED.dimension,
                    has_custom_thumbnail = EXCLUDED.has_custom_thumbnail,
                    projection = EXCLUDED.projection,
                    category_id = EXCLUDED.category_id,
                    category_name = EXCLUDED.category_name,
                    topic_ids = EXCLUDED.topic_ids,
                    topic_categories = EXCLUDED.topic_categories,
                    content_details = EXCLUDED.content_details,
                    status_details = EXCLUDED.status_details,
                    topic_details = EXCLUDED.topic_details,
                    updated_at = CURRENT_TIMESTAMP
                """, (
                db_channel_id,
                video['id'],
                video.get('kind', 'youtube#video'),
                video.get('etag', ''),
                None,  # playlist_item_id - not available in this context
                snippet.get('title', ''),
                snippet.get('description', ''),
                snippet.get('publishedAt'),
                snippet.get('channelTitle', ''),
                None,  # playlist_id - not available in this context
                None,  # position - not available in this context
                snippet.get('tags', []),
                thumbnails.get('default', {}).get('url'),
                thumbnails.get('medium', {}).get('url'),
                thumbnails.get('high', {}).get('url'),
                thumbnails.get('standard', {}).get('url'),
                thumbnails.get('maxres', {}).get('url'),
                int(statistics.get('viewCount', 0)),
                int(statistics.get('likeCount', 0)),
                int(statistics.get('commentCount', 0)),
                content_details.get('duration', ''),
                None,  # transcript - will be fetched separately
                False,  # has_captions - will be updated when transcript is fetched
                None,  # caption_language - will be updated when transcript is fetched
                # Enhanced fields
                privacy_status,
                upload_status,
                license,
                embeddable,
                public_stats_viewable,
                definition,
                dimension,
                has_custom_thumbnail,
                projection,
                category_id,
                category_name,
                topic_ids,
                topic_categories,
                content_details_json,
                status_details_json,
                topic_details_json
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
                db_channel_id = _store_channel_data(cursor, user_id, channel_info)
                
                if not db_channel_id:
                    logger.error(f"Failed to store channel data for user {user_id}")
                    return False
                
                # Store video data
                stored_count = _store_video_data(cursor, db_channel_id, videos)
                
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