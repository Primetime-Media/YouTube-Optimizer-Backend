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
from datetime import timezone # Import timezone

logger = logging.getLogger(__name__)

def build_youtube_client(credentials):
    """
    Build a YouTube API client with user credentials to maximize use of their quota.
    
    This function includes specific parameters that help ensure the user's quota
    is used rather than the application's quota.
    
    Args:
        credentials: User OAuth2 credentials
        
    Returns:
        YouTube API client object
    """
    try:
        logger.info("Building YouTube API client with user credentials")
        # Use specific build parameters that maximize user quota usage
        return build(
            'youtube', 
            'v3', 
            credentials=credentials,
            #developerKey="AIzaSyDrvHTmmZ5Vek_nN8moZVquSeQe1v4fY_0",
            cache_discovery=False,  # Disable cache to ensure fresh credentials are used
        )
    except Exception as e:
        logger.error(f"Error building YouTube client: {e}")
        raise
        
def get_user_id_for_channel(channel_id: int) -> Optional[int]:
    """
    Get the user ID associated with a channel
    
    Args:
        channel_id: The database ID of the channel
        
    Returns:
        int: User ID if found, None otherwise
    """
    try:
        logger.info(f"Retrieving user ID for channel {channel_id}")
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT user_id 
                FROM youtube_channels
                WHERE id = %s
            """, (channel_id,))
            
            result = cursor.fetchone()
            if result:
                user_id = result[0]
                logger.info(f"Found user {user_id} for channel {channel_id}")
                return user_id
            else:
                logger.warning(f"No user found for channel {channel_id}")
                return None
    except Exception as e:
        logger.error(f"Error retrieving user ID for channel: {e}")
        return None
    finally:
        if conn:
            conn.close()

def set_video_thumbnail(youtube_client, video_id: str, thumbnail_file: str) -> Dict:
    """
    Set a custom thumbnail for a YouTube video
    
    Args:
        youtube_client: YouTube API client
        video_id: The YouTube video ID
        thumbnail_file: Path to the thumbnail image file
        
    Returns:
        dict: Result of the thumbnail upload operation
    """
    try:
        import os
        from googleapiclient.http import MediaFileUpload
        
        logger.info(f"Starting thumbnail upload for video {video_id}")
        
        # Validate file exists
        if not os.path.exists(thumbnail_file):
            return {
                "success": False,
                "error": f"Thumbnail file not found: {thumbnail_file}"
            }

        from PIL import Image
        # Convert PNG to JPG if needed
        if thumbnail_file.lower().endswith('.png'):
            try:
                # Create output filename with .jpg extension
                jpg_file = os.path.splitext(thumbnail_file)[0] + '.jpg'

                # Open image and convert to RGB (removes alpha channel if present)
                with Image.open(thumbnail_file) as img:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    # Save as JPG with 90% quality (good balance between quality and size)
                    img.save(jpg_file, 'JPEG', quality=90, optimize=True, progressive=True)

                # Update thumbnail_file to use the new JPG
                thumbnail_file = jpg_file
                logger.info(f"Converted PNG to JPG: {thumbnail_file}")

            except Exception as e:
                logger.error(f"Error converting PNG to JPG: {e}")
                return {
                    "success": False,
                    "error": f"Failed to convert PNG to JPG: {str(e)}"
                }

        # Validate file size (YouTube limit is 2MB)
        file_size = os.path.getsize(thumbnail_file)
        if file_size > 2 * 1024 * 1024:  # 2MB in bytes
            return {
                "success": False,
                "error": f"Thumbnail file too large: {file_size} bytes (max 2MB)"
            }
        
        # Validate file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        file_ext = os.path.splitext(thumbnail_file)[1].lower()
        if file_ext not in valid_extensions:
            return {
                "success": False,
                "error": f"Invalid file type: {file_ext}. Must be one of {valid_extensions}"
            }
        
        # Create media upload object with proper MIME type mapping
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp'
        }
        mimetype = mime_type_map.get(file_ext, 'image/jpeg')  # Default to jpeg
        
        media = MediaFileUpload(
            thumbnail_file,
            mimetype=mimetype,
            resumable=True
        )
        
        # Upload the thumbnail
        logger.info(f"Uploading thumbnail for video {video_id}")
        upload_response = youtube_client.thumbnails().set(
            videoId=video_id,
            media_body=media
        ).execute()
        
        logger.info(f"Successfully uploaded thumbnail for video {video_id}")
        return {
            "success": True,
            "message": "Thumbnail uploaded successfully",
            "video_id": video_id,
            "thumbnail_info": upload_response
        }
        
    except HttpError as e:
        error_content = json.loads(e.content.decode())
        error_message = error_content.get("error", {}).get("message", str(e))
        logger.error(f"YouTube API error uploading thumbnail: {error_message}")
        return {
            "success": False,
            "error": f"YouTube API error: {error_message}"
        }
    except Exception as e:
        logger.error(f"Error uploading thumbnail for video {video_id}: {e}")
        return {
            "success": False,
            "error": f"Error uploading thumbnail: {str(e)}"
        }
            
def update_youtube_video(
    youtube_client,
    video_id: str,
    optimized_title: str,
    optimized_description: str,
    optimized_tags: list,
    optimization_id: int = None,
    only_title: bool = False,
    only_description: bool = False,
    only_tags: bool = False,
    thumbnail_file: str = None
) -> Dict:
    """
    Update a YouTube video's metadata with optimized values
    
    Args:
        youtube_client: YouTube API client
        video_id: The YouTube video ID
        optimized_title: The optimized video title
        optimized_description: The optimized video description
        optimized_tags: List of optimized video tags
        optimization_id: The ID of the optimization record (optional)
        only_title: If true, only update the title
        only_description: If true, only update the description
        only_tags: If true, only update the tags
        thumbnail_file: Path to thumbnail file to upload (optional)
        
    Returns:
        dict: Result of the update operation
    """
    try:
        logger.info(f"Starting YouTube video update for video {video_id}")
        
        # Determine if updating all fields or specific ones
        update_all = not (only_title or only_description or only_tags)

        # Validate required data based on flags
        if (update_all or only_title) and not optimized_title:
            return {
                "success": False,
                "error": "Missing optimized title for update"
            }
        if (update_all or only_description) and not optimized_description:
            return {
                "success": False,
                "error": "Missing optimized description for update"
            }
        if (update_all or only_tags) and not optimized_tags:
            return {
                "success": False,
                "error": "Missing optimized tags for update"
            }
        
        try:
            # First retrieve the current video data
            logger.info("Retrieving current video data from YouTube API")
            video_response = youtube_client.videos().list(
                part="snippet",
                id=video_id
            ).execute()
            
            if not video_response.get("items"):
                logger.error(f"Video {video_id} not found on YouTube")
                return {
                    "success": False,
                    "error": "Video not found on YouTube"
                }
            
            # Get the current snippet
            snippet = video_response["items"][0]["snippet"]
            updated_fields = []
            
            # Conditionally update the snippet
            if update_all or only_title:
                snippet["title"] = optimized_title
                updated_fields.append("title")
            if update_all or only_description:
                snippet["description"] = optimized_description
                updated_fields.append("description")
            if update_all or only_tags:
                snippet["tags"] = optimized_tags
                updated_fields.append("tags")
            
            # Update video metadata
            logger.info(f"Sending update request to YouTube API for fields: {', '.join(updated_fields)}")
            update_response = youtube_client.videos().update(
                part="snippet",
                body={
                    "id": video_id,
                    "snippet": snippet
                }
            ).execute()
            
            # Update our local database as well
            logger.info("Updating local database with new video metadata")
            conn = get_connection()
            try:
                with conn.cursor() as cursor:
                    # Build the SET clause dynamically
                    set_clauses = []
                    params = []
                    if update_all or only_title:
                        set_clauses.append("title = %s")
                        params.append(optimized_title)
                    if update_all or only_description:
                        set_clauses.append("description = %s")
                        params.append(optimized_description)
                    if update_all or only_tags:
                        set_clauses.append("tags = %s")
                        params.append(optimized_tags)

                    if set_clauses: # Only update if there are changes
                        set_clauses.append("updated_at = NOW()")
                        set_clauses.append("is_optimized = TRUE")
                        set_clauses.append("last_optimized_at = NOW()")
                        set_clauses.append("last_optimization_id = %s")
                        params.append(optimization_id)
                        
                        params.append(video_id) # For the WHERE clause

                        sql = f"UPDATE youtube_videos SET {', '.join(set_clauses)} WHERE video_id = %s"
                        cursor.execute(sql, tuple(params))
                        conn.commit()
            finally:
                conn.close()
            
            logger.info(f"Successfully updated fields {', '.join(updated_fields)} for video {video_id}")
            
            # Upload thumbnail if provided
            thumbnail_result = None
            if thumbnail_file:
                thumbnail_result = set_video_thumbnail(youtube_client, video_id, thumbnail_file)
                if not thumbnail_result.get("success"):
                    logger.warning(f"Thumbnail upload failed for video {video_id}: {thumbnail_result.get('error')}")
            
            return {
                "success": True,
                "message": f"Video fields ({', '.join(updated_fields)}) updated successfully",
                "video_id": video_id,
                "thumbnail_result": thumbnail_result
            }
            
        except HttpError as e:
            error_content = json.loads(e.content.decode())
            error_message = error_content.get("error", {}).get("message", str(e))
            logger.error(f"YouTube API error: {error_message}")
            return {
                "success": False,
                "error": f"YouTube API error: {error_message}"
            }
            
    except Exception as e:
        logger.error(f"Error updating YouTube video: {e}")
        return {
            "success": False,
            "error": f"Error updating video: {str(e)}"
        }

def update_youtube_channel_branding(
    youtube_client,
    channel_db_id: int,
    optimized_description: str,
    optimized_keywords: str,
    optimization_id: int = None,
    only_description: bool = False,
    only_keywords: bool = False
) -> Dict:
    """
    Update a YouTube channel's branding settings with optimized values
    
    Args:
        youtube_client: YouTube API client
        channel_db_id: The database ID of the channel
        optimized_description: The optimized channel description
        optimized_keywords: The optimized channel keywords (as a single string)
        optimization_id: The ID of the optimization record (optional)
        only_description: If true, only update the description
        only_keywords: If true, only update the keywords
        
    Returns:
        dict: Result of the update operation
    """
    try:
        logger.info(f"Starting YouTube channel branding update for channel {channel_db_id}")
        
        # Determine if updating all fields or specific ones
        update_all = not (only_description or only_keywords)

        # Get the YouTube channel ID and current branding settings
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT channel_id, branding_settings
                    FROM youtube_channels
                    WHERE id = %s
                """, (channel_db_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.error(f"Channel {channel_db_id} not found in database")
                    return {
                        "success": False,
                        "error": "Channel not found in database"
                    }
                    
                youtube_channel_id, current_branding_settings_json = result
            
            logger.info(f"Retrieved YouTube channel ID: {youtube_channel_id}")

            # Validate required data based on flags
            if (update_all or only_description) and not optimized_description:
                return {
                    "success": False,
                    "error": "Missing optimized description for update"
                }
            if (update_all or only_keywords) and not optimized_keywords:
                 return {
                     "success": False,
                     "error": "Missing optimized keywords for update"
                 }

            # Parse current branding settings
            current_branding_settings = {}
            if current_branding_settings_json:
                if isinstance(current_branding_settings_json, str):
                    try:
                        current_branding_settings = json.loads(current_branding_settings_json)
                    except json.JSONDecodeError:
                         logger.warning("Failed to parse existing branding settings JSON")
                elif isinstance(current_branding_settings_json, dict):
                    current_branding_settings = current_branding_settings_json

            # Ensure the channel section exists
            if "channel" not in current_branding_settings:
                current_branding_settings["channel"] = {}
            
            updated_fields = []
            # Conditionally update the branding settings object
            if update_all or only_description:
                current_branding_settings["channel"]["description"] = optimized_description
                updated_fields.append("description")
            if update_all or only_keywords:
                current_branding_settings["channel"]["keywords"] = optimized_keywords # Keywords are stored as a single string
                updated_fields.append("keywords")
            
            logger.info(f"Prepared updated branding settings for fields: {', '.join(updated_fields)}")
            
            try:
                # Update channel branding
                logger.info(f"Sending update request to YouTube API for fields: {', '.join(updated_fields)}")
                update_response = youtube_client.channels().update(
                    part="brandingSettings",
                    body={
                        "id": youtube_channel_id,
                        "brandingSettings": current_branding_settings
                    }
                ).execute()
                
                # Update our local database as well
                logger.info("Updating local database with new branding settings")
                with conn.cursor() as cursor:
                    # Build the SET clause dynamically
                    set_clauses = []
                    params = []
                    if update_all or only_description:
                         set_clauses.append("description = %s")
                         params.append(optimized_description)
                    # Always update branding_settings JSON if any change was made
                    set_clauses.append("branding_settings = %s")
                    params.append(json.dumps(current_branding_settings))
                    
                    if set_clauses: # Only update if there are changes
                         set_clauses.append("updated_at = NOW()")
                         set_clauses.append("is_optimized = TRUE")
                         set_clauses.append("last_optimized_at = NOW()")
                         set_clauses.append("last_optimization_id = %s")
                         params.append(optimization_id)
                         
                         params.append(channel_db_id) # For the WHERE clause
                         
                         sql = f"UPDATE youtube_channels SET {', '.join(set_clauses)} WHERE id = %s"
                         cursor.execute(sql, tuple(params))
                         conn.commit()
                
                logger.info(f"Successfully updated fields {', '.join(updated_fields)} for channel {youtube_channel_id} branding")
                return {
                    "success": True,
                    "message": f"Channel fields ({', '.join(updated_fields)}) updated successfully",
                    "youtube_channel_id": youtube_channel_id
                }
            except HttpError as e:
                error_content = json.loads(e.content.decode())
                error_message = error_content.get("error", {}).get("message", str(e))
                logger.error(f"YouTube API error: {error_message}")
                return {
                    "success": False,
                    "error": f"YouTube API error: {error_message}"
                }
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error updating YouTube channel branding: {e}")
        return {
            "success": False,
            "error": f"Error updating channel: {str(e)}"
        }

def fetch_channel_info(credentials):
    """Fetch the user's YouTube channel information."""
    try:
        # Build the YouTube API client with user credentials to use their quota
        youtube = build_youtube_client(credentials)
        
        # Get the authenticated user's channel (this will use the user's quota)
        # Request all parts needed for our enhanced channel fields
        channels_response = youtube.channels().list(
            part="snippet,contentDetails,statistics,brandingSettings,id,localizations,status,topicDetails",
            mine=True,  # Explicitly use the authenticated user's channel (uses their quota)
            maxResults=1,  # Only need one result (the user's channel)
            onBehalfOfContentOwner=None  # Explicitly set to None to ensure user's credentials are used
        ).execute()
        
        if not channels_response['items']:
            logger.warning("No YouTube channel found for this user")
            return None

        logger.info(f"Fetched channel info using user's credentials")
        
        # Log channel info at debug level to avoid excessive logging
        logger.debug(f"Fetched channel info: {channels_response['items'][0]}")

        return channels_response['items'][0]
        
    except HttpError as e:
        if e.resp.status == 403 and "quota" in str(e).lower():
            logger.error(f"YouTube API quota exceeded: {e}")
            # Here we could implement retry logic with exponential backoff
            raise HttpError(e.resp, f"YouTube quota exceeded. Please try again later: {e.reason}".encode())
        else:
            logger.error(f"YouTube API error: {e}")
            raise
    except Exception as e:
        logger.error(f"Error fetching channel info: {e}")
        raise

def fetch_videos(credentials, max_results=50):
    """Fetch the user's uploaded videos with additional data."""
    try:

        logger.info(f"Fetching videos for user {credentials.user_id} with max_results: {max_results}")
        
        # Build the YouTube API client using our helper that ensures user quota
        youtube = build_youtube_client(credentials)

        # First, get the uploads playlist ID
        # We'll reuse info from fetch_channel_info if possible to save quota
        try:
            # Try to get any cached channel info from the database
            uploads_playlist_id = None
            # Implement caching here if needed
        except:
            pass
            
        # If we don't have cached info, fetch it
        if not uploads_playlist_id:
            channels_response = youtube.channels().list(
                part='contentDetails',  # Minimal part to save quota
                mine=True  # Uses user's quota
            ).execute()

            if not channels_response['items']:
                logger.warning("No YouTube channel found for this user")
                return []

            uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # Now get the video IDs from the uploads playlist with efficient batching
        video_ids = []
        next_page_token = None

        # Request maximum items per page allowed by the API to reduce request count
        max_per_page = 50  # YouTube API maximum

        while len(video_ids) < max_results:
            playlist_response = youtube.playlistItems().list(
                part='contentDetails',  # Only get what we need to save quota
                playlistId=uploads_playlist_id,
                maxResults=min(max_per_page, max_results - len(video_ids)),
                pageToken=next_page_token
            ).execute()

            for item in playlist_response['items']:
                video_ids.append(item['contentDetails']['videoId'])

            next_page_token = playlist_response.get('nextPageToken')

            if not next_page_token:
                break

        # Now fetch full video data in maximum size batches to save quota
        all_videos = []
        batch_size = 50  # Maximum batch size for videos.list
        
        for i in range(0, len(video_ids), batch_size):
            chunk = video_ids[i:i + batch_size]
            video_batch_response = youtube.videos().list(
                part='snippet,contentDetails,statistics,status,topicDetails,localizations',  # Include additional parts for enhanced fields
                id=','.join(chunk)
            ).execute()

            video_items = video_batch_response.get('items', [])
            if not video_items:
                continue # Skip if batch is empty

            category_ids = set()
            for video in video_items:
                # Check if snippet and categoryId exist
                if 'snippet' in video and 'categoryId' in video['snippet']:
                    category_ids.add(video['snippet']['categoryId'])

            category_map = {}
            if category_ids:
                try:
                    # Fetch category details using the extracted IDs
                    category_response = youtube.videoCategories().list(
                        part='snippet',
                        id=','.join(list(category_ids))
                    ).execute()
                    category_map = {item['id']: item['snippet']['title'] for item in category_response.get('items', [])}
                except Exception as e:
                    logger.error(f"Failed to fetch video categories for IDs {category_ids}: {e}")
                    # Continue without category names for this batch

            logging.info(f"category_map: {category_map}")

            # Add category name to each video
            for video in video_items:
                 if 'snippet' in video and 'categoryId' in video['snippet']:
                    video['snippet']['categoryName'] = category_map.get(video['snippet']['categoryId'], 'Unknown')
                 else:
                     # Assign default if snippet or categoryId is missing
                     if 'snippet' not in video:
                         video['snippet'] = {} # Ensure snippet exists
                     video['snippet']['categoryName'] = 'Unknown'
            
            all_videos.extend(video_items) # Extend with modified items

        # Fetch transcripts for all videos to complete the data
        logger.info(f"Fetching transcripts for {len(all_videos)} videos")
        for video in all_videos:
            video_id = video['id']
            try:
                transcript_data = fetch_video_transcript(credentials, video_id)
                # Add transcript data to the video object
                video['transcript_data'] = transcript_data
                if transcript_data.get('transcript'):
                    logger.debug(f"Successfully fetched transcript for video {video_id}")
                else:
                    logger.debug(f"No transcript available for video {video_id}")
            except Exception as transcript_err:
                logger.warning(f"Failed to fetch transcript for video {video_id}: {transcript_err}")
                # Set default transcript data on failure
                video['transcript_data'] = {
                    'transcript': None,
                    'has_captions': False,
                    'caption_language': None
                }

        logger.info(f"Fetched {len(all_videos)} videos with category names and transcripts using user's quota")
        return all_videos

    except HttpError as e:
        if e.resp.status == 403 and "quota" in str(e).lower():
            logger.error(f"YouTube API quota exceeded while fetching videos: {e}")
            # Here we could implement retry logic with exponential backoff
            raise HttpError(e.resp, f"YouTube quota exceeded. Please try again later: {e.reason}".encode())
        else:
            logger.error(f"YouTube API error: {e}")
            raise
    except Exception as e:
        logger.error(f"Error fetching videos: {e}")
        raise

def store_youtube_data(user_id, channel_info, videos, credentials):
    """Store YouTube data in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Store channel info
            if channel_info:
                youtube_channel_id = channel_info['id']

                # Extract enhanced channel data from the API response
                banner_url = None
                privacy_status = None
                is_linked = False
                long_uploads_status = None
                is_monetization_enabled = False
                
                # Topic details
                topic_ids = []
                topic_categories = []
                
                # Channel standing info
                overall_good_standing = True
                community_guidelines_good_standing = True
                copyright_strikes_good_standing = True
                content_id_claims_good_standing = True
                
                # Raw JSON data storage for future use
                branding_settings = None
                audit_details = None
                topic_details = None
                status_details = None
                
                # Extract branding settings if available
                if 'brandingSettings' in channel_info:
                    branding_settings = json.dumps(channel_info['brandingSettings'])
                    if 'image' in channel_info['brandingSettings']:
                        banner_url = channel_info['brandingSettings']['image'].get('bannerExternalUrl')
                
                # Extract status info
                if 'status' in channel_info:
                    status_details = json.dumps(channel_info['status'])
                    privacy_status = channel_info['status'].get('privacyStatus')
                    is_linked = channel_info['status'].get('isLinked', False)
                    long_uploads_status = channel_info['status'].get('longUploadsStatus')
                    is_monetization_enabled = 'madeForKids' not in channel_info['status'] or not channel_info['status'].get('madeForKids', False)
                
                # Extract topic details if available
                if 'topicDetails' in channel_info:
                    topic_details = json.dumps(channel_info['topicDetails'])
                    topic_ids = channel_info['topicDetails'].get('topicIds', [])
                    topic_categories = channel_info['topicDetails'].get('topicCategories', [])
                
                # Extract audit details if available
                if 'auditDetails' in channel_info:
                    audit_details = json.dumps(channel_info['auditDetails'])
                    if 'overallGoodStanding' in channel_info['auditDetails']:
                        overall_good_standing = channel_info['auditDetails'].get('overallGoodStanding', True)
                    if 'communityGuidelinesGoodStanding' in channel_info['auditDetails']:
                        community_guidelines_good_standing = channel_info['auditDetails'].get('communityGuidelinesGoodStanding', True)
                    if 'copyrightStrikesGoodStanding' in channel_info['auditDetails']:
                        copyright_strikes_good_standing = channel_info['auditDetails'].get('copyrightStrikesGoodStanding', True)
                    if 'contentIdClaimsGoodStanding' in channel_info['auditDetails']:
                        content_id_claims_good_standing = channel_info['auditDetails'].get('contentIdClaimsGoodStanding', True)

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
                        
                        -- Update enhanced fields
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
                        
                        updated_at = NOW()
                    RETURNING id
                """, (
                    user_id,
                    youtube_channel_id,
                    channel_info.get('kind', ''),
                    channel_info.get('etag', ''),
                    channel_info['snippet']['title'],
                    channel_info['snippet']['description'],
                    channel_info['snippet'].get('customUrl', ''),
                    channel_info['snippet']['publishedAt'],
                    int(channel_info['statistics'].get('viewCount', 0)),
                    int(channel_info['statistics'].get('subscriberCount', 0)),
                    channel_info['statistics'].get('hiddenSubscriberCount', False),
                    int(channel_info['statistics'].get('videoCount', 0)),
                    channel_info['snippet']['thumbnails']['default']['url'],
                    channel_info['snippet']['thumbnails']['medium']['url'],
                    channel_info['snippet']['thumbnails']['high']['url'],
                    channel_info['contentDetails']['relatedPlaylists']['uploads'],
                    
                    # Enhanced channel fields
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
                
                # Get the database ID for the channel
                db_channel_id = cursor.fetchone()[0]
                logger.info(f"Channel saved with database ID: {db_channel_id}")

                # Store videos
                for i, video in enumerate(videos):
                    # Extract video ID
                    video_id = video['id']
                    
                    # Extract thumbnails with null safety
                    thumbnails = video['snippet'].get('thumbnails', {})

                    # Extract transcript data from the video object (fetched in fetch_videos)
                    transcript_data = video.get('transcript_data', {})
                    transcript_text = transcript_data.get('transcript')
                    has_captions = transcript_data.get('has_captions', False)
                    caption_language = transcript_data.get('caption_language')
                    
                    # Extract enhanced video data
                    # Content details
                    definition = None
                    dimension = None
                    has_custom_thumbnail = False
                    projection = None
                    
                    # Video status and visibility
                    privacy_status = None
                    upload_status = None
                    license = None
                    embeddable = None
                    public_stats_viewable = None
                    
                    # Video category
                    category_id = video['snippet'].get('categoryId')
                    category_name = video['snippet'].get('categoryName')
                    
                    # Topic details for content categorization
                    video_topic_ids = []
                    video_topic_categories = []
                    
                    # Extract from contentDetails if available
                    content_details_json = None
                    if 'contentDetails' in video:
                        content_details_json = json.dumps(video['contentDetails'])
                        content_details = video['contentDetails']
                        definition = content_details.get('definition')  # hd or sd
                        dimension = content_details.get('dimension')    # 2d or 3d
                        projection = content_details.get('projection')  # rectangular or 360
                    
                    # Extract from status if available
                    status_details_json = None
                    if 'status' in video:
                        status_details_json = json.dumps(video['status'])
                        status_details = video['status']
                        privacy_status = status_details.get('privacyStatus')    # public, private, unlisted
                        upload_status = status_details.get('uploadStatus')      # uploaded, processed, etc.
                        license = status_details.get('license')                 # youtube, creative_commons
                        embeddable = status_details.get('embeddable')          
                        public_stats_viewable = status_details.get('publicStatsViewable')
                    
                    # Extract topic details if available
                    topic_details_json = None
                    if 'topicDetails' in video:
                        topic_details_json = json.dumps(video['topicDetails'])
                        topic_details = video['topicDetails']
                        video_topic_ids = topic_details.get('topicIds', [])
                        video_topic_categories = topic_details.get('topicCategories', [])
                    
                    # Check for custom thumbnail
                    has_custom_thumbnail = 'maxres' in thumbnails

                    # Store video data with enhanced fields
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
                            transcript = EXCLUDED.transcript,
                            has_captions = EXCLUDED.has_captions,
                            caption_language = EXCLUDED.caption_language,
                            
                            -- Enhanced video fields update
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
                            
                            -- Preserve the is_optimized flag, don't overwrite it
                            updated_at = NOW()
                    """, (
                        db_channel_id,
                        video_id,
                        video.get('kind', ''),
                        video.get('etag', ''),
                        video.get('id', ''),
                        video['snippet']['title'],
                        video['snippet']['description'],
                        video['snippet'].get('publishedAt') or video.get('contentDetails', {}).get('videoPublishedAt', None),
                        video['snippet'].get('channelTitle', ''),
                        video['snippet'].get('playlistId', ''),
                        video['snippet'].get('position', 0),
                        video.get('snippet', {}).get('tags', []),
                        thumbnails.get('default', {}).get('url', None),
                        thumbnails.get('medium', {}).get('url', None),
                        thumbnails.get('high', {}).get('url', None),
                        thumbnails.get('standard', {}).get('url', None),
                        thumbnails.get('maxres', {}).get('url', None),
                        int(video.get('statistics', {}).get('viewCount', 0)),
                        int(video.get('statistics', {}).get('likeCount', 0)),
                        int(video.get('statistics', {}).get('commentCount', 0)),
                        video.get('contentDetails', {}).get('duration', 'PT0S'),
                        transcript_text,
                        has_captions,
                        caption_language,
                        
                        # Enhanced video fields
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
                        video_topic_ids,
                        video_topic_categories,
                        content_details_json,
                        status_details_json,
                        topic_details_json
                    ))
            
            conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing YouTube data: {e}")
        raise
    finally:
        conn.close()

def get_video_captions(credentials, video_id: str) -> Dict:
    """
    Fetch caption tracks for a specific YouTube video
    
    Args:
        credentials: Google OAuth2 credentials
        video_id: YouTube video ID
        
    Returns:
        dict: Dictionary with caption information
        {
            'available_captions': [list of caption tracks with metadata],
            'default_caption': {caption text and timing info} or None if no default,
            'has_captions': True/False
        }
    """
    try:
        # Build the YouTube API client using our helper
        youtube = build_youtube_client(credentials)
        
        # List all captions for the video
        captions_response = youtube.captions().list(
            part="snippet",
            videoId=video_id
        ).execute()
        
        caption_tracks = captions_response.get('items', [])
        
        if not caption_tracks:
            logger.info(f"No caption tracks found for video {video_id}")
            return {
                'available_captions': [],
                'default_caption': None,
                'has_captions': False
            }
        
        logger.info(f"Found {len(caption_tracks)} caption tracks for video {video_id}")
        if len(caption_tracks) > 0:
            logger.info(f"Sample caption track: {caption_tracks[0]}")
        
        # Format the caption list with relevant metadata
        available_captions = []
        standard_caption_id = None
        asr_caption_id = None

        # First pass: process all captions and categorize them
        for track in caption_tracks:
            track_kind = track['snippet']['trackKind']
            is_draft = track['snippet'].get('isDraft', False)
            is_auto = track_kind.lower() == 'asr'  # Make case-insensitive comparison
            
            caption_info = {
                'id': track['id'],
                'language': track['snippet']['language'],
                'name': track['snippet'].get('name', ''),
                'track_kind': track_kind,
                'is_draft': is_draft,
                'is_auto': is_auto
            }
            
            available_captions.append(caption_info)
            
            # Look for standard caption first
            if track_kind.lower() == 'standard' and not is_draft and standard_caption_id is None:
                standard_caption_id = track['id']
                logger.info(f"Found standard caption track for video {video_id}, id: {track['id']}")
            
            # Also track ASR captions
            elif track_kind.lower() == 'asr' and not is_draft and asr_caption_id is None:
                asr_caption_id = track['id']
                logger.info(f"Found ASR caption track for video {video_id}, id: {track['id']}")
        
        # Set default caption ID - prioritize standard captions, but use ASR if that's all we have
        default_caption_id = standard_caption_id or asr_caption_id
        
        # If we found a caption to use, try to download it
        default_caption = None
        if default_caption_id:
            try:
                logger.info(f"Using caption track {default_caption_id} for video {video_id}")
                # Get full transcript from the caption
                default_caption = download_caption_transcript(youtube, default_caption_id)
            except Exception as e:
                logger.error(f"Error downloading caption {default_caption_id}: {e}")
                
                # If standard caption download failed, try ASR as backup
                if default_caption_id == standard_caption_id and asr_caption_id:
                    try:
                        logger.info(f"Standard caption failed, trying ASR caption {asr_caption_id}")
                        default_caption = download_caption_transcript(youtube, asr_caption_id)
                    except Exception as asr_e:
                        logger.error(f"Error downloading ASR caption {asr_caption_id}: {asr_e}")
        
        return {
            'available_captions': available_captions,
            'default_caption': default_caption,
            'has_captions': len(available_captions) > 0
        }
        
    except HttpError as e:
        if e.resp.status == 403:
            # This often happens if the API key doesn't have captions access
            logger.warning(f"Permission denied for captions on video {video_id}. Verify OAuth scopes include youtube.force-ssl")
            
            # Try an alternative approach to detect if captions exist
            try:
                # We can check if a video has captions using the videos.list endpoint
                # which requires a lower level of permissions
                video_response = youtube.videos().list(
                    part="contentDetails",
                    id=video_id
                ).execute()
                
                # Check the caption flag in the response
                if video_response['items'] and 'contentDetails' in video_response['items'][0]:
                    has_captions = video_response['items'][0]['contentDetails'].get('caption') == 'true'
                    logger.info(f"Used alternative method to detect captions. Video {video_id} has_captions={has_captions}")
                    
                    return {
                        'available_captions': [],
                        'default_caption': None,
                        'has_captions': has_captions,
                        'error': 'Permission denied for full caption access, but detected caption status'
                    }
            except Exception as alt_e:
                logger.warning(f"Alternative caption detection also failed: {alt_e}")
            
            # Fall back to default response
            return {
                'available_captions': [],
                'default_caption': None,
                'has_captions': False,
                'error': 'Permission denied. Make sure OAuth scope includes youtube.force-ssl'
            }
        else:
            logger.error(f"YouTube API error: {e}")
            raise
    except Exception as e:
        logger.error(f"Error fetching captions: {e}")
        raise

def download_caption_transcript(youtube, caption_id: str, language: str = None) -> Dict:
    """
    Download and parse a caption track's content
    
    Args:
        youtube: YouTube API client
        caption_id: ID of the caption to download
        language: Optional target language for translation
        
    Returns:
        dict: Structured caption data with text and timing information
    """
    try:
        logger.info(f"Downloading caption {caption_id}")

        # Set up download parameters
        params = {
            'id': caption_id,
            'tfmt': 'srt',  # SubRip format which includes timing info
            'onBehalfOfContentOwner': None  # Explicitly set to None to ensure user's credentials are used
        }
        
        # Add target language if provided
        if language:
            params['tlang'] = language
            
        # Download the caption
        caption_response = youtube.captions().download(**params).execute()
        
        if not caption_response:
            logger.warning(f"Empty caption response for caption ID {caption_id}")
            return None
            
        # Process the SRT format
        caption_content = caption_response.decode('utf-8') if isinstance(caption_response, bytes) else caption_response
        
        # Parse the SRT content into a structured format
        transcript = parse_srt_content(caption_content)

        print(f"transcript found:,{transcript}")
        return {
            'caption_id': caption_id,
            'segments': transcript,
            'full_text': ' '.join([segment['text'] for segment in transcript])
        }
        
    except HttpError as e:
        logger.error(f"Error downloading caption {caption_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing caption {caption_id}: {e}")
        raise

def parse_srt_content(srt_content: str) -> List[Dict]:
    """
    Parse SRT format captions into structured data
    
    Args:
        srt_content: String containing SRT format caption data
        
    Returns:
        list: List of caption segments with timing and text
    """
    segments = []
    
    # Regular expression to extract SRT parts (index, timecode, and text)
    pattern = r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})\s+((?:.+\s*)+?)(?:\r?\n\r?\n|\Z)'
    matches = re.findall(pattern, srt_content, re.DOTALL)
    
    for match in matches:
        index, start_time, end_time, text = match
        
        # Clean up the text (remove HTML tags, decode HTML entities)
        clean_text = html.unescape(re.sub(r'<[^>]+>', '', text))
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Convert timecode to seconds for easier processing
        start_seconds = timecode_to_seconds(start_time)
        end_seconds = timecode_to_seconds(end_time)
        
        segments.append({
            'index': int(index),
            'start_time': start_time,
            'end_time': end_time,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'duration': end_seconds - start_seconds,
            'text': clean_text
        })
    
    return segments

def timecode_to_seconds(timecode: str) -> float:
    """
    Convert SRT timecode format (HH:MM:SS,MS) to seconds
    
    Args:
        timecode: String in format "HH:MM:SS,MS"
        
    Returns:
        float: Time in seconds
    """
    hours, minutes, seconds = timecode.replace(',', '.').split(':')
    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

def build_youtube_analytics_client(credentials):
    """
    Build a YouTube Analytics API client with user credentials to maximize use of their quota.
    
    The YouTube Analytics API generally uses the authenticated user's quota
    when properly configured with owner permissions.
    """
    try:
        # Use specific build parameters that maximize user quota usage
        return build(
            'youtubeAnalytics', 
            'v2', 
            credentials=credentials,
            developerKey=None,  # Explicitly set to None to use only user credentials
            cache_discovery=False,  # Disable cache to ensure fresh credentials are used
        )
    except Exception as e:
        logger.error(f"Error building YouTube Analytics client: {e}")
        raise

def fetch_video_analytics(credentials, video_id: str, metrics: List[str], dimensions: List[str] = None, 
                    start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """
    Fetch analytics data for a specific YouTube video using YouTube Analytics API.
    
    Args:
        credentials: Google OAuth2 credentials
        video_id: YouTube video ID
        metrics: List of metrics to retrieve (e.g., ['views', 'likes', 'subscribersGained'])
        dimensions: Optional list of dimensions to group by (e.g., ['day', 'ageGroup', 'gender'])
        start_date: Start date in YYYY-MM-DD format (defaults to 28 days ago)
        end_date: End date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        dict: Dictionary with analytics data for the specified video
    """
    logger.info(f"Attempting to fetch analytics for video_id: {video_id}")
    logger.info(f"Requested metrics: {metrics}, dimensions: {dimensions}, start_date: {start_date}, end_date: {end_date}")

    try:
        # Build the YouTube Analytics API client with user credentials
        youtube_analytics = build_youtube_analytics_client(credentials)
        logger.info("YouTube Analytics API client built.")
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            logger.info(f"Using default end_date: {end_date}")
        if not start_date:
            # Default to 28 days ago
            start_date = (datetime.datetime.now() - datetime.timedelta(days=28)).strftime('%Y-%m-%d')
            logger.info(f"Using default start_date: {start_date}")
        
        # Prepare filters and parameters
        filters = f"video=={video_id}"
        metrics_str = ','.join(metrics)
        
        # Basic parameters (required)
        params = {
            "ids": "channel==MINE",  # This ensures we use the user's quota
            "startDate": start_date,
            "endDate": end_date,
            "metrics": metrics_str,
            "filters": filters
        }
        
        # Add dimensions if provided
        if dimensions:
            params["dimensions"] = ','.join(dimensions)
        
        logger.info(f"API query parameters: {params}")

        # Execute the API request - this uses the user's quota
        analytics_response = youtube_analytics.reports().query(**params).execute()
        
        logger.info(f"Successfully retrieved analytics response for video {video_id}")

        # Format the response into a more usable structure
        result = {
            'video_id': video_id,
            'time_range': {'start_date': start_date, 'end_date': end_date},
            'analytics': format_analytics_response(analytics_response, metrics, dimensions)
        }
        
        logger.info(f"Formatted analytics result for video {video_id}")
        # logger.debug(f"Formatted result: {result}") # Uncomment for detailed debugging

        return result
        
    except HttpError as e:
        if e.resp.status == 403:
            if "quota" in str(e).lower():
                logger.error(f"YouTube Analytics API quota exceeded for video {video_id}: {e}")
                return {
                    'video_id': video_id,
                    'error': 'API quota exceeded. Please try again later.',
                    'status_code': 403
                }
            else:
                logger.error(f"YouTube Analytics API permission error for video {video_id}: {e}")
                return {
                    'video_id': video_id,
                    'error': 'Permission denied. Make sure OAuth scope includes youtube.readonly or youtubeAnalytics.readonly',
                    'status_code': 403
                }
        logger.error(f"YouTube Analytics API HttpError for video {video_id}: {e}")
        raise # Re-raise the exception after logging
    except Exception as e:
        logger.error(f"Unexpected error fetching video analytics for video {video_id}: {e}", exc_info=True)
        raise # Re-raise the exception after logging

def fetch_video_transcript(credentials, video_id: str) -> Dict[str, Any]:
    """
    Fetch transcript for a specific video when needed (e.g., for optimization)
    
    Args:
        credentials: Google OAuth2 credentials
        video_id: YouTube video ID
        
    Returns:
        dict: Dictionary with transcript information
        {
            'transcript': transcript text or None,
            'has_captions': boolean,
            'caption_language': caption language code or None,
            'error': error message if any
        }
    """
    try:
        # Create a client using our helper that prioritizes user quota
        youtube = build_youtube_client(credentials)
        
        # Attempt to get captions
        captions_data = get_video_captions(credentials, video_id)
        has_captions = captions_data.get('has_captions', False)
        
        # Default values
        transcript_text = None
        caption_language = None
        
        # If we have a default caption, extract its text
        if captions_data.get('default_caption'):
            transcript_text = captions_data['default_caption'].get('full_text', '')
            
            # Find the language of the default caption
            for caption in captions_data.get('available_captions', []):
                if caption.get('id') == captions_data['default_caption'].get('caption_id'):
                    caption_language = caption.get('language')
                    break
                    
        logger.info(f"Fetched transcript for video {video_id}: has_captions={has_captions}, language={caption_language}")
        
        return {
            'transcript': transcript_text,
            'has_captions': has_captions,
            'caption_language': caption_language
        }
        
    except Exception as e:
        logger.warning(f"Error fetching transcript for video {video_id}: {e}")
        return {
            'transcript': None,
            'has_captions': False,
            'caption_language': None,
            'error': str(e)
        }

def fetch_granular_view_data(credentials, video_id: str, interval: str = '30m', 
                             start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """
    Fetch granular view data for a video to analyze performance over time
    """
    logger.info(f"Starting fetch_granular_view_data for video_id: {video_id}, interval: {interval}, start_date: {start_date}, end_date: {end_date}") # Log function start

    try:
        # Build the YouTube Analytics and Data API clients using our helpers
        logger.info("Building YouTube Analytics and Data API clients") # Log client building
        youtube_analytics = build_youtube_analytics_client(credentials)
        youtube_data = build_youtube_client(credentials)
        
        # Set default end date if not provided
        if not end_date:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Get video publish date to determine data availability
        logger.info("Fetching video data to get publish date") # Log video data fetch
        try:
            video_response = youtube_data.videos().list(
                part="snippet",
                id=video_id
            ).execute()
        except Exception as e:
            from google.auth.exceptions import RefreshError
            if isinstance(e, RefreshError) and 'invalid_grant' in str(e):
                logger.error(f"Authentication expired while fetching video data for {video_id}")
                return {
                    'video_id': video_id,
                    'error': 'Authentication expired, please re-authenticate',
                    'status_code': 401,
                    'auth_required': True
                }
            raise
        
        if not video_response.get('items'):
            logger.warning(f"Video {video_id} not found") # Log video not found
            return {
                'video_id': video_id,
                'error': 'Video not found',
                'status_code': 404
            }
        
        publish_date_str = video_response['items'][0]['snippet']['publishedAt']
        # Parse the publish date string (adjust format if needed, but this is common)
        try:
            publish_date = datetime.datetime.strptime(publish_date_str, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
             logger.error(f"Could not parse publish date string: {publish_date_str}. Falling back.")
             publish_date = datetime.datetime.now() - datetime.timedelta(days=28) # Or handle error differently

        now = datetime.datetime.now()
        video_age_days = (now - publish_date).days
        logger.info(f"Video publish date: {publish_date.strftime('%Y-%m-%d')}, video age in days: {video_age_days}") # Log publish date and age

        # Always use the publish date as the start date
        start_date = publish_date.strftime('%Y-%m-%d')
        logger.info(f"Using video publish date as start_date: {start_date}")

        # Determine dimensions based on interval (still relevant if interval logic is kept elsewhere)
        if interval == '1m': # Note: 1m interval likely won't work with 'day' dimension
            dimensions = ['day', 'minute'] # This dimension combination might still cause issues with YouTube API
        else:
            dimensions = ['day'] # Default to 'day' dimension

        logger.info(f"Using interval: {interval}, start_date: {start_date}, end_date: {end_date}, dimensions: {dimensions}") # Log interval, start date, dimensions

        metrics = [
            'engagedViews', 'views', 'estimatedMinutesWatched', 'averageViewPercentage', 'likes', 'dislikes', 'comments',
        ]
        
        # Prepare request parameters
        params = {
            "ids": "channel==MINE",
            "startDate": start_date,
            "endDate": end_date,
            "metrics": ','.join(metrics),
            "dimensions": ','.join(dimensions),
            "filters": f"video=={video_id}",
            "sort": ','.join(dimensions)  # Sort by time dimensions
        }
        logger.info(f"YouTube Analytics API request parameters: {params}")
        
        analytics_response = youtube_analytics.reports().query(**params).execute()
        logger.debug(f"Raw YouTube Analytics API response: {analytics_response}")

        logger.info(f"Analytics API response column headers: {analytics_response.get('columnHeaders')}")
        logger.info(f"Analytics API response rows: {analytics_response.get('rows')}")

        # Process the time-series data
        timeseries_data = []

        column_headers = analytics_response.get('columnHeaders', [])
        header_names = [header['name'] for header in column_headers]

        for row in analytics_response.get('rows', []):
            data_point = {}
            
            # Map values to headers
            for i, value in enumerate(row):
                header = header_names[i]
                data_point[header] = value
            
            # Create a proper timestamp
            day = data_point.get('day', '')
            
            if interval == '1m':
                minute = data_point.get('minute', 0)
                hour = minute // 60
                minute_of_hour = minute % 60
                timestamp = f"{day} {hour:02d}:{minute_of_hour:02d}:00"
            else:
                hour = data_point.get('hour', 0) # This might be causing issue if 'hour' dimension is not valid
                timestamp = f"{day} {hour:02d}:00:00"
            
            data_point['timestamp'] = timestamp
            data_point['datetime'] = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            timeseries_data.append(data_point)
        
        # For 30-minute intervals, group the hourly data into half-hour buckets
        if interval == '30m' and timeseries_data:
            half_hour_data = []
            for point in timeseries_data:
                dt = point['datetime']
                views = point.get('engagedViews', 0)
                minutes_watched = point.get('estimatedMinutesWatched', 0)
                
                first_half = point.copy()
                first_half['timestamp'] = dt.strftime("%Y-%m-%d %H:00:00")
                first_half['half_hour'] = 1
                first_half['views'] = views // 2
                first_half['estimatedMinutesWatched'] = minutes_watched // 2
                half_hour_data.append(first_half)
                
                second_half = point.copy()
                second_half['timestamp'] = dt.strftime("%Y-%m-%d %H:30:00")
                second_half['half_hour'] = 2
                second_half['views'] = views - (views // 2)
                second_half['estimatedMinutesWatched'] = minutes_watched
            
            # Replace the original hourly data with 30-minute data
            timeseries_data = half_hour_data
        
        # Calculate views per minute/hour rates
        total_views = sum(point.get('engagedViews', 0) for point in timeseries_data)
        total_datapoints = len(timeseries_data)
        
        if interval == '1m':
            views_per_minute = total_views / total_datapoints if total_datapoints > 0 else 0
            views_per_hour = views_per_minute * 60
        else:  # 30m interval
            views_per_halfhour = total_views / total_datapoints if total_datapoints > 0 else 0
            views_per_hour = views_per_halfhour * 2
            views_per_minute = views_per_hour / 60
            
        # Format the results
        result = {
            'video_id': video_id,
            'time_range': {'start_date': start_date, 'end_date': end_date},
            'interval': interval,
            'timeseries_data': timeseries_data,
            'summary': {
                'total_views': total_views,
                'views_per_minute': views_per_minute,
                'views_per_hour': views_per_hour,
                'data_points': total_datapoints
            }
        }

        logger.info(f"Length of timeseries_data before storing: {len(timeseries_data)}") # Log length of timeseries_data
        
        return result
        
    except HttpError as e:
        logger.error(f"YouTube Analytics API error: {e}")
        if e.resp.status == 403:
            return {
                'video_id': video_id,
                'error': 'Permission denied. Make sure OAuth scope includes youtube.readonly or youtubeAnalytics.readonly',
                'status_code': 403
            }
        elif e.resp.status == 400:
            # This often happens with too granular data requests for older videos
            return {
                'video_id': video_id,
                'error': f'Data unavailable at {interval} granularity for the requested time range.',
                'status_code': 400,
                'message': str(e)
            }
        raise
    except Exception as e:
        logger.error(f"Error fetching granular view data: {e}")
        raise

def store_granular_view_data(conn, video_id: str, timeseries_data: List[Dict]):
    """
    Store granular view data in the database
    
    Args:
        conn: Database connection
        video_id: YouTube video ID
        timeseries_data: List of data points with timestamps and metrics
    """
    try:
        with conn.cursor() as cursor:
            # First, get the database ID for the video
            cursor.execute(
                "SELECT id FROM youtube_videos WHERE video_id = %s",
                (video_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Video {video_id} not found in database, can't store timeseries data")
                return
                
            db_video_id = result[0]
            
            # Check if the table exists, create it if not
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_timeseries_data (
                    id SERIAL PRIMARY KEY,
                    video_id INTEGER REFERENCES youtube_videos(id),
                    timestamp TIMESTAMP NOT NULL,
                    views INTEGER NOT NULL DEFAULT 0,
                    estimated_minutes_watched FLOAT,
                    average_view_percentage FLOAT,
                    raw_data JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(video_id, timestamp)
                )
            """)
            
            # Create index on video_id and timestamp
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_video_timeseries_video_timestamp 
                ON video_timeseries_data(video_id, timestamp)
            """)
            
            # Insert the timeseries data points
            for point in timeseries_data:

                logging.info(f"Storing {point} data in video_timeseries_data table")

                timestamp = point.get('timestamp')
                views = point.get('engagedViews', 0)
                minutes_watched = point.get('estimatedMinutesWatched', 0)
                view_percentage = point.get('averageViewPercentage', 0)
                
                # Convert datetime object to ISO format string before JSON serialization
                if 'datetime' in point and isinstance(point['datetime'], datetime.datetime):
                    point['datetime'] = point['datetime'].isoformat()
                
                # Insert or update the data point
                cursor.execute("""
                    INSERT INTO video_timeseries_data 
                    (video_id, timestamp, views, estimated_minutes_watched, average_view_percentage, raw_data)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (video_id, timestamp) 
                    DO UPDATE SET
                        views = EXCLUDED.views,
                        estimated_minutes_watched = EXCLUDED.estimated_minutes_watched,
                        average_view_percentage = EXCLUDED.average_view_percentage,
                        raw_data = EXCLUDED.raw_data
                """, (
                    db_video_id,
                    timestamp,
                    views,
                    minutes_watched,
                    view_percentage,
                    json.dumps(point)
                ))
            
            conn.commit()
            logger.info(f"Stored {len(timeseries_data)} timeseries data points for video {video_id}")
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing timeseries data: {e}")
        raise

def format_analytics_response(response: Dict, metrics: List[str], dimensions: List[str] = None) -> Dict:
    """
    Formats the raw YouTube Analytics API response into a more usable structure.
    
    Args:
        response: Raw API response
        metrics: List of metrics requested
        dimensions: List of dimensions requested (or None)
    
    Returns:
        dict: Formatted analytics data
    """
    # Initialize the result
    result = {
        'data': [],
        'totals': {}
    }
    
    # Process column headers
    column_headers = response.get('columnHeaders', [])
    header_names = [header['name'] for header in column_headers]
    
    # Extract total values (usually provided for all metrics)
    if 'rows' in response and len(response['rows']) > 0:
        # For responses with dimension rows
        data_rows = response.get('rows', [])
        
        # Process each row of data
        for row in data_rows:
            row_data = {}
            
            # Map each value to its header
            for i, value in enumerate(row):
                header = header_names[i]
                row_data[header] = value
            
            result['data'].append(row_data)
        
    # Extract totals from the response
    if 'totals' in response and len(response['totals']) > 0:
        total_row = response['totals'][0]
        
        # Map total values to their metric names
        for i, total in enumerate(total_row):
            metric_name = header_names[i]
            
            # Skip dimension columns in totals
            if dimensions and i < len(dimensions):
                continue
            
            result['totals'][metric_name] = total
    
    return result

def fetch_channel_analytics(credentials, metrics: List[str], dimensions: List[str] = None, 
                           start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """
    Fetch analytics data for the entire YouTube channel using YouTube Analytics API.
    
    Args:
        credentials: Google OAuth2 credentials
        metrics: List of metrics to retrieve (e.g., ['views', 'estimatedMinutesWatched', 'averageViewDuration'])
        dimensions: Optional list of dimensions to group by (e.g., ['day', 'video'])
        start_date: Start date in YYYY-MM-DD format (defaults to 28 days ago)
        end_date: End date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        dict: Dictionary with analytics data for the channel
    """
    try:
        # Build the YouTube Analytics API client
        youtube_analytics = build('youtubeAnalytics', 'v2', credentials=credentials)
        
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            # Default to 28 days ago
            start_date = (datetime.datetime.now() - datetime.timedelta(days=28)).strftime('%Y-%m-%d')
        
        # Prepare parameters
        metrics_str = ','.join(metrics)
        
        # Basic parameters (required)
        params = {
            "ids": "channel==MINE",
            "startDate": start_date,
            "endDate": end_date,
            "metrics": metrics_str
        }
        
        # Add dimensions if provided
        if dimensions:
            params["dimensions"] = ','.join(dimensions)
        
        # Execute the API request
        analytics_response = youtube_analytics.reports().query(**params).execute()
        
        logger.info(f"Retrieved channel analytics from {start_date} to {end_date}")
        
        # Format the response into a more usable structure
        result = {
            'time_range': {'start_date': start_date, 'end_date': end_date},
            'analytics': format_analytics_response(analytics_response, metrics, dimensions)
        }
        
        return result
        
    except HttpError as e:
        logger.error(f"YouTube Analytics API error: {e}")
        if e.resp.status == 403:
            return {
                'error': 'Permission denied. Make sure OAuth scope includes youtube.readonly or youtubeAnalytics.readonly',
                'status_code': 403
            }
        raise
    except Exception as e:
        logger.error(f"Error fetching channel analytics: {e}")
        raise

def fetch_video_timeseries_data(video_id: str, interval: str = 'day', force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetch timeseries data for a video, prioritizing daily data from YouTube API

    Args:
        video_id: YouTube video ID
        interval: Time interval for aggregation (only 'day' is fully supported by YouTube now)
        force_refresh: If True, bypass cache and fetch fresh data from YouTube API

    Returns:
        dict: Formatted timeseries data for visualization
    """
    logger.info(f"fetch_video_timeseries_data called for video {video_id} with interval={interval}, force_refresh={force_refresh}")

    # Map legacy interval values to 'day'
    if interval in ['1m', '30m', '1h']:
        logger.warning(f"Interval '{interval}' is no longer supported by YouTube API, using 'day' instead.")
        interval = 'day'

    conn = None
    db_video_id = None
    channel_id = None
    user_id = None
    published_at = None # Initialize published_at

    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # Get video ID, channel ID, and publish date
            cursor.execute("""
                SELECT v.id, v.channel_id, v.published_at
                FROM youtube_videos v
                WHERE v.video_id = %s
            """, (video_id,))
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Video {video_id} not found in database")
                if conn: conn.close()
                return {'error': 'Video not found', 'video_id': video_id}
            db_video_id, channel_id, published_at = result

            # Get user ID
            cursor.execute("SELECT user_id FROM youtube_channels WHERE id = %s", (channel_id,))
            user_result = cursor.fetchone()
            if not user_result:
                logger.warning(f"Channel {channel_id} not found or has no associated user for video {video_id}")
                if conn: conn.close()
                return {'error': 'Channel not found for video', 'video_id': video_id}
            user_id = user_result[0]

            # Cache check (if not forcing refresh)
            if not force_refresh:
                cursor.execute("""
                    SELECT MAX(created_at)
                    FROM video_timeseries_data
                    WHERE video_id = %s
                """, (db_video_id,))
                last_update = cursor.fetchone()[0] # This should be timezone-aware

                if last_update:
                    # --- Timezone Fix ---
                    # Get current time as timezone-aware (UTC)
                    now_aware = datetime.datetime.now(timezone.utc)
                    logger.debug(f"Cache check: Now (aware): {now_aware}, Last Update (aware): {last_update}")

                    # Ensure last_update is aware (it should be from TIMESTAMPTZ)
                    if last_update.tzinfo is None:
                         # This case shouldn't happen if DB schema is correct, but handle defensively
                         logger.warning(f"last_update timestamp for video {db_video_id} is timezone-naive. Assuming UTC.")
                         last_update = last_update.replace(tzinfo=timezone.utc)

                    age = now_aware - last_update
                    # --- End Timezone Fix ---

                    if age.total_seconds() < 86400:  # 24 hours
                        logger.info(f"Using cached timeseries data for video {video_id} (last update: {age.total_seconds()/3600:.1f} hours ago)")
                        cursor.execute("""
                            SELECT timestamp, views, estimated_minutes_watched, average_view_percentage
                            FROM video_timeseries_data
                            WHERE video_id = %s
                            ORDER BY timestamp
                        """, (db_video_id,))
                        rows = cursor.fetchall()
                        if rows:
                            # Process and return cached data
                            conn.close() # Close connection before returning
                            return process_timeseries_data(rows, video_id, interval)
                        else:
                            logger.warning(f"Cache timestamp exists but no data rows found for video {video_id}. Refreshing.")
                    else:
                         logger.info(f"Cached data for video {video_id} is older than 24 hours. Refreshing.")
                else:
                    logger.info(f"No cached timeseries data found for video {video_id}. Fetching fresh data.")

    except Exception as db_error:
        logger.error(f"Database error during initial video/cache check for {video_id}: {db_error}", exc_info=True)
        if conn: conn.close()
        return {'error': f'Database error: {str(db_error)}', 'video_id': video_id}
    finally:
        # Ensure connection is closed if open
        if conn and not conn.closed:
            conn.close()

    # Proceed to fetch fresh data if needed
    logger.info(f"Proceeding to fetch fresh YouTube Analytics data for video {video_id}")

    # --- Credential fetching logic ---
    from utils.auth import get_credentials_dict, get_user_credentials
    try:
        credentials_dict = get_credentials_dict(user_id)
        if not credentials_dict:
            logger.error(f"Failed to get credentials dict for user {user_id}")
            return {'error': 'Credentials dictionary not found', 'video_id': video_id}

        credentials = get_user_credentials(user_id)
        if not credentials or not hasattr(credentials, 'token'): # Check if credentials object is valid
            logger.error(f"Failed to get valid Credentials object for user {user_id}")
            return {'error': 'Invalid or missing credentials', 'video_id': video_id}
            
        if credentials.expired and credentials.refresh_token:
            logger.info(f"Credentials for user {user_id} expired, attempting refresh.")
            try:
                from google.auth.transport.requests import Request
                from google.auth.exceptions import RefreshError
                
                # Log refresh token info for debugging (partial token only)
                token_prefix = credentials.refresh_token[:10] if credentials.refresh_token else "None"
                logger.info(f"Attempting refresh with token prefix: {token_prefix}...")
                
                # Validate refresh token before attempting refresh
                if not credentials.refresh_token or len(credentials.refresh_token) < 10:
                    logger.error(f"Invalid refresh token for user {user_id}: token too short or empty")
                    return {'error': 'Invalid refresh token, please re-authenticate', 'video_id': video_id, 'auth_required': True}
                
                # Log client ID for debugging OAuth configuration issues
                client_id_prefix = credentials.client_id[:12] if credentials.client_id else "None"
                logger.info(f"Refresh attempt with client_id prefix: {client_id_prefix}...")
                
                credentials.refresh(Request())
                logger.info(f"Credentials refreshed successfully for user {user_id}.")
                
                # Update stored credentials in DB after successful refresh
                try:
                    from utils.auth import update_user_credentials
                    if hasattr(update_user_credentials, '__call__'):
                        update_user_credentials(user_id, credentials)
                        logger.info(f"Updated stored credentials for user {user_id}")
                except (ImportError, AttributeError) as auth_err:
                    logger.warning(f"Could not update stored credentials: {auth_err}")
                    
            except RefreshError as refresh_err:
                logger.error(f"OAuth refresh failed for user {user_id}: {refresh_err}")
                # Check if it's an invalid_grant error specifically
                if 'invalid_grant' in str(refresh_err):
                    logger.error(f"Refresh token invalid/expired for user {user_id}. User needs to re-authenticate.")
                    return {'error': 'Authentication expired, please re-authenticate', 'video_id': video_id, 'auth_required': True}
                else:
                    return {'error': f'Failed to refresh credentials: {refresh_err}', 'video_id': video_id}
            except Exception as refresh_err:
                 logger.error(f"Unexpected error during credential refresh for user {user_id}: {refresh_err}")
                 return {'error': 'Failed to refresh credentials', 'video_id': video_id}
        elif credentials.expired:
             logger.error(f"Credentials for user {user_id} expired and no refresh token available.")
             return {'error': 'Credentials expired, please re-authenticate', 'video_id': video_id}


        # --- API Call Logic ---
        end_date_dt = datetime.datetime.now(timezone.utc) # Use aware datetime
        end_date = end_date_dt.strftime('%Y-%m-%d')

        # Calculate default start date (aware)
        default_start_date_dt = end_date_dt - datetime.timedelta(days=28)

        # Ensure published_at (from DB, likely naive TIMESTAMP) is comparable
        # Convert published_at to naive if it's aware, or assume UTC if naive
        if published_at.tzinfo is not None:
            published_at_naive = published_at.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            published_at_naive = published_at # Assume it's already naive UTC

        # Convert default_start_date_dt to naive UTC for comparison
        default_start_date_naive = default_start_date_dt.astimezone(timezone.utc).replace(tzinfo=None)

        # Determine the actual start date (naive comparison, assumed UTC)
        actual_start_date_naive = max(published_at_naive, default_start_date_naive)
        start_date = actual_start_date_naive.strftime('%Y-%m-%d')

        logger.info(f"Analytics API call range for {video_id}: Start: {start_date}, End: {end_date}")

        analytics_data = fetch_granular_view_data(
            credentials,
            video_id,
            interval=interval, # Should be 'day'
            start_date=start_date,
            end_date=end_date
        )

        if 'error' in analytics_data:
            logger.error(f"Error returned from fetch_granular_view_data for {video_id}: {analytics_data['error']}")
            return analytics_data # Return the error dict

        # Store the fresh data
        conn = get_connection()
        try:
            if 'timeseries_data' in analytics_data and analytics_data['timeseries_data']:
                 store_granular_view_data(conn, video_id, analytics_data['timeseries_data'])
                 logger.info(f"Stored {len(analytics_data['timeseries_data'])} fresh timeseries data points for video {video_id}")
            else:
                 logger.info(f"No timeseries data returned from API for video {video_id} in range {start_date} to {end_date}. Nothing to store.")
                 # Optionally update the refresh timestamp even if no data?
                 # update_last_refresh_timestamp(conn, video_id)
        except Exception as store_error:
            logger.error(f"Error storing fresh timeseries data for {video_id}: {store_error}", exc_info=True)
            # Don't fail the whole request, just log the storage error
        finally:
            if conn: conn.close()

        # Process and return the fresh data (even if storage failed)
        # Use process_timeseries_data to format the API response similarly to cached data
        # Assuming fetch_granular_view_data returns data points suitable for process_timeseries_data
        # Need to adjust process_timeseries_data or the format here if they differ
        if 'timeseries_data' in analytics_data:
             # TODO: Ensure process_timeseries_data can handle the format from fetch_granular_view_data
             # For now, returning the raw structure from fetch_granular_view_data
             return analytics_data
        else:
             # Handle case where API call succeeded but returned no data structure
              logger.warning(f"fetch_granular_view_data completed but returned no 'timeseries_data' key for video {video_id}")
              return {
                  'video_id': video_id,
                  'interval': interval,
                  'timeseries_data': [],
                  'time_range': {'start_date': start_date, 'end_date': end_date},
                  'summary': {'total_views': 0, 'views_per_hour': 0, 'data_points': 0},
                  'message': 'No data available for the period.'
              }


    except Exception as e:
        logger.error(f"Unhandled error during timeseries data fetch for {video_id}: {e}", exc_info=True)
        return {'error': f'An unexpected error occurred: {str(e)}', 'video_id': video_id}

def process_timeseries_data(rows, video_id: str, interval: str):
    """
    Process timeseries data rows from database into formatted response
    """
    result_data = []
    aggregated_data = {}
    
    for row in rows:
        timestamp, views, minutes_watched, view_percentage = row
        
        # Create a day-based aggregation key
        agg_key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Initialize or update the aggregation entry
        if agg_key not in aggregated_data:
            aggregated_data[agg_key] = {
                'timestamp': agg_key.strftime('%Y-%m-%d %H:%M:%S'),
                'date': agg_key.strftime('%Y-%m-%d'),  # Add date for easier frontend usage
                'views': 0,
                'minutes_watched': 0,
                'view_percentage': 0,
                'count': 0
            }
        
        # Add to the aggregation values
        aggregated_data[agg_key]['views'] += views
        aggregated_data[agg_key]['minutes_watched'] += minutes_watched or 0
        aggregated_data[agg_key]['view_percentage'] += view_percentage or 0
        aggregated_data[agg_key]['count'] += 1
    
    # Calculate averages for the aggregated data
    for agg_key, data in aggregated_data.items():
        if data['count'] > 0:
            data['view_percentage'] /= data['count']
            data['minutes_watched'] /= data['count']
        del data['count']  # Remove the count field
        result_data.append(data)
    
    # Sort by timestamp
    result_data.sort(key=lambda x: x['timestamp'])
    
    # Calculate summary metrics
    total_views = sum(point.get('views', 0) for point in result_data)
    total_datapoints = len(result_data)
    
    # Calculate the time range
    if result_data:
        start_timestamp = result_data[0]['timestamp']
        end_timestamp = result_data[-1]['timestamp']
    else:
        start_timestamp = None
        end_timestamp = None
    
    # For daily data, calculate daily, hourly, and per-minute rates
    if total_datapoints > 0:
        views_per_day = total_views / total_datapoints
        views_per_hour = views_per_day / 24
        views_per_minute = views_per_hour / 60
    else:
        views_per_day = 0
        views_per_hour = 0
        views_per_minute = 0
    
    return {
        'video_id': video_id,
        'interval': interval,
        'timeseries_data': result_data,
        'time_range': {
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp
        },
        'summary': {
            'total_views': total_views,
            'views_per_day': views_per_day,
            'views_per_hour': views_per_hour,
            'views_per_minute': views_per_minute,
            'data_points': total_datapoints
        }
    }

async def fetch_and_store_youtube_analytics(user_id, video_id, credentials_dict=None, interval='day'):
    """
    Fetch and store daily analytics data for a specific video
    
    Args:
        user_id: Database user ID
        video_id: YouTube video ID
        credentials_dict: Optional Google OAuth credentials as a dictionary (if not provided, will retrieve from DB)
        interval: Time interval for the data (only 'day' is fully supported by YouTube now)
    """
    logger.info(f"Starting fetch_and_store_youtube_analytics for video_id: {video_id}, user_id: {user_id}, interval: {interval}")
    
    # Map legacy interval values to 'day' since YouTube no longer supports granular data
    if interval in ['1m', '30m', '1h']:
        logger.info(f"Interval '{interval}' is no longer supported by YouTube API, using 'day' instead")
        interval = 'day'

    try:
        # If credentials_dict is provided, use it; otherwise, retrieve from DB
        if credentials_dict:
            logger.info(f"Using provided credentials_dict for user {user_id}")
            # Convert dictionary to Credentials object
            credentials = Credentials(
                token=credentials_dict['token'],
                refresh_token=credentials_dict['refresh_token'],
                token_uri=credentials_dict['token_uri'],
                client_id=credentials_dict['client_id'],
                client_secret=credentials_dict['client_secret'],
                scopes=credentials_dict['scopes']
            )
        else:
            # Import here to avoid circular import
            from utils.auth import get_user_credentials
            
            # Get credentials from database
            logger.info(f"Retrieving credentials from database for user {user_id}")
            credentials = get_user_credentials(user_id)
            
            if not credentials:
                logger.error(f"No credentials found for user {user_id}")
                return {'error': 'No credentials found', 'video_id': video_id}
        
        # Use a 28-day window for YouTube analytics (common default)
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.datetime.now() - datetime.timedelta(days=28)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching analytics for video {video_id} from {start_date} to {end_date} with daily granularity")
        
        # Fetch daily view data for the video
        analytics_data = fetch_granular_view_data(
            credentials, 
            video_id, 
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in analytics_data:
            logger.error(f"Error fetching analytics data: {analytics_data['error']}")
            return analytics_data
            
        # Get database connection
        conn = get_connection()
        try:
            # Store the timeseries data
            store_granular_view_data(conn, video_id, analytics_data['timeseries_data'])
            logger.info(f"Stored {len(analytics_data['timeseries_data'])} data points for video {video_id}")
            
            # Get the database ID for the video for any additional updates
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM youtube_videos WHERE video_id = %s",
                    (video_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    db_video_id = result[0]
                    
                    # Update the last_analytics_refresh timestamp on the video
                    cursor.execute("""
                        UPDATE youtube_videos 
                        SET last_analytics_refresh = NOW()
                        WHERE id = %s
                    """, (db_video_id,))
                    
                    conn.commit()
                    logger.info(f"Updated last_analytics_refresh timestamp for video {video_id}")
            
            # Return the result including fresh data
            result = {
                'video_id': video_id,
                'success': True,
                'summary': analytics_data.get('summary', {}),
                'timeseries_data': analytics_data.get('timeseries_data', []),
                'time_range': analytics_data.get('time_range', {'start_date': start_date, 'end_date': end_date})
            }
            
            logger.info(f"Successfully completed analytics refresh for video {video_id} with {result['data_points']} data points")
            return result
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error in analytics refresh for video {video_id}: {e}", exc_info=True)
        return {'error': str(e), 'video_id': video_id}

async def fetch_and_store_youtube_data(user_id, max_videos=10000):
    """
    Fetch and store all YouTube data for a user.
    
    Args:
        user_id: Database user ID
        max_videos: Maximum number of videos to fetch (to save quota)
    """
    try:

        logger.info(f"Fetching YouTube data for user {user_id} with max_videos: {max_videos}")
        
        # Import here to avoid circular import
        from utils.auth import get_user_credentials
        
        # Get credentials from database - use with auto-refresh
        credentials = get_user_credentials(user_id, auto_refresh=True)
        
        if not credentials:
            logger.error(f"No credentials found for user {user_id}")
            return
        
        # Log that we're using user credentials
        logger.info(f"Fetching YouTube data for user {user_id} using their OAuth credentials")
        
        # Fetch data with limited videos to save quota
        channel_info = fetch_channel_info(credentials)
        
        if not channel_info:
            logger.error(f"Could not fetch channel info for user {user_id}")
            return
            
        # Only fetch a limited number of recent videos to save quota
        videos = fetch_videos(credentials, max_results=max_videos)
        
        store_youtube_data(user_id, channel_info, videos, credentials)
        
        logger.info(f"Successfully fetched and stored YouTube data for user {user_id} ({len(videos)} videos)")
        
    except HttpError as e:
        if e.resp.status == 403 and "quota" in str(e).lower():
            logger.error(f"YouTube API quota exceeded for user {user_id}: {e}")
        else:
            logger.error(f"YouTube API error for user {user_id}: {e}")
    except Exception as e:
        logger.error(f"Unhandled error during timeseries data fetch for {video_id}: {e}", exc_info=True)
        return {'error': f'An unexpected error occurred: {str(e)}', 'video_id': video_id}