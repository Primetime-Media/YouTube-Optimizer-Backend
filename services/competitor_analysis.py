from utils.auth import get_user_credentials
from services.youtube import build_youtube_client
from services.llm_optimization import generate_search_query, find_relevant_topic_id, extract_competitor_keywords
import logging
from datetime import datetime, timedelta, timezone
import math

logger = logging.getLogger(__name__)

def get_competitor_analysis(user_id: int, video_data: dict):
    credentials = get_user_credentials(user_id)

    if not credentials:
        logger.error(f"No credentials found for user {user_id}")
        return {'error': 'No credentials found', 'user_id': user_id}

    logger.info("Building YouTube API client")
    youtube_client = build_youtube_client(credentials)

    competitor_video_data = search_competitors(
        youtube_client,
        user_id,
        video_data
    )

    if competitor_video_data:
        # Sort competitors by performance score (combining view_to_subscriber_ratio and subscriber count)
        competitor_video_data = sort_competitors_by_performance(competitor_video_data)
        # Extract keywords from top competitors, comparing with user's video
        competitor_keywords = extract_competitor_keywords(competitor_video_data, video_data)
    else:
        competitor_keywords = {
            "keywords": [],
            "explanation": "No competitor videos found to analyze.",
            "relevance": "",
            "opportunities": ""
        }

    return {
        "competitor_videos": competitor_video_data,
        "competitor_keywords": competitor_keywords
    }


def get_channel_subscriber_counts(client: object, channel_ids: list[str]) -> dict:
    """
    Fetches subscriber counts for a list of channel IDs using the YouTube API.

    Args:
        client: The authenticated YouTube Data API client.
        channel_ids: A list of YouTube channel IDs.

    Returns:
        A dictionary mapping channel_id to subscriber_count.
        Returns an empty dictionary if the API call fails or no IDs are provided.
        Subscriber counts might be None if hidden by the channel owner.
    """
    if not channel_ids:
        return {}

    subscriber_counts = {}
    # The API allows up to 50 IDs per request
    for i in range(0, len(channel_ids), 50):
        batch_ids = channel_ids[i:i+50]
        try:
            # Join the batch of IDs into a comma-separated string
            ids_string = ",".join(batch_ids)
            response = client.channels().list(
                part="id,statistics",
                id=ids_string
            ).execute()

            for item in response.get("items", []):
                channel_id = item.get("id")
                stats = item.get("statistics", {})
                # Subscriber count can be hidden
                sub_count = int(stats.get("subscriberCount", 0)) if stats.get("hiddenSubscriberCount") is False else None
                if channel_id:
                    subscriber_counts[channel_id] = sub_count

        except Exception as e:
            logger.error(f"Failed to fetch subscriber counts for batch starting with {batch_ids[0]}: {e}")
            # Continue to next batch if one fails, or return partial results

    return subscriber_counts


def get_video_view_counts(client: object, video_ids: list[str]) -> dict:
    """
    Fetches view counts for a list of video IDs using the YouTube API.

    Args:
        client: The authenticated YouTube Data API client.
        video_ids: A list of YouTube video IDs.

    Returns:
        A dictionary mapping video_id to view_count.
        Returns an empty dictionary if the API call fails or no IDs are provided.
        View counts will be integers.
    """
    if not video_ids:
        return {}

    view_counts = {}
    # API allows up to 50 IDs per request for videos.list
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        try:
            ids_string = ",".join(batch_ids)
            response = client.videos().list(
                part="id,statistics",
                id=ids_string
            ).execute()

            for item in response.get("items", []):
                video_id = item.get("id")
                stats = item.get("statistics", {})
                view_count = int(stats.get("viewCount", 0))  # Default to 0 if missing
                if video_id:
                    view_counts[video_id] = view_count

        except Exception as e:
            logger.error(f"Failed to fetch view counts for video batch starting with {batch_ids[0]}: {e}")
            # Consider how to handle partial failures - currently continues

    return view_counts


def search_competitors(client: object, user_id: int, video_data: dict):

    title = video_data['title']
    description = video_data['description']
    tags = video_data.get('tags', [])  # Use .get for potentially missing keys like tags
    category_id = video_data['category_id']
    category_name = video_data['category_name']
    transcript = video_data['transcript']

    # Generate search query to get more relevant competitors
    query = generate_search_query(title, description, tags, category_name)
    if query:
        logger.info(f"Generated search query for user {user_id}: {query}")

    topic_data = find_relevant_topic_id(title, description, transcript, tags, category_name)
    topic_id = topic_data['topic_id']

    try:
        # Calculate the date 7 days ago in UTC for recent results
        now_utc = datetime.now(timezone.utc)
        seven_days_ago = now_utc - timedelta(days=7)
        # Format as RFC 3339 timestamp (required by YouTube API)
        published_after_time = seven_days_ago.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Initialize competitor_videos list
        competitor_videos = []
        
        # Search using both the generated query (if available) and the category ID
        search_args = {
            "part": "id,snippet",
            "type": "video",
            "videoCategoryId": category_id,
            "order": "viewCount",
            "relevanceLanguage": "en",
            "regionCode": "US",
            "maxResults": 20,
            "publishedAfter": published_after_time
        }
        if topic_id:
            search_args["topicId"] = topic_id

        # Retry loop with proper initialization
        for attempt in range(3):
            try:
                search_response = client.search().list(**search_args).execute()
                videos = search_response.get("items", [])
                
                if not videos:
                    if attempt < 2:  # Don't raise on last attempt
                        logger.warning(f"No competitors found on attempt {attempt + 1}, retrying...")
                        continue
                    else:
                        logger.warning("No competitors found after 3 attempts")
                        break
                
                competitor_videos = videos
                logger.info(f"Found {len(competitor_videos)} raw competitors published after {published_after_time} for user {user_id}")
                break
                
            except Exception as e:
                logger.error(f"Error fetching competitors for user {user_id} on attempt {attempt + 1}: {e}")
                if attempt == 2:  # Last attempt
                    logger.error("All retry attempts failed")

        # Search using query if available (append results)
        if query and competitor_videos:  # Only search with query if we have initial results
            try:
                query_search_args = search_args.copy()
                query_search_args["q"] = query
                search_response = client.search().list(**query_search_args).execute()
                additional_videos = search_response.get("items", [])
                competitor_videos.extend(additional_videos)
                logger.info(f"Added {len(additional_videos)} videos from query search, total: {len(competitor_videos)}")
            except Exception as e:
                logger.error(f"Error in query-based search: {e}")

        if not competitor_videos:
            logger.warning(f"No competitor videos found for user {user_id}")
            return []

        # Deduplicate videos by video_id
        seen_video_ids = set()
        deduplicated_videos = []
        for video in competitor_videos:
            try:
                if "snippet" not in video or "id" not in video or "videoId" not in video["id"]:
                    logger.warning(f"Skipping competitor video due to missing data")
                    continue
                
                video_id = video["id"]["videoId"]
                
                # Skip if we've already seen this video
                if video_id in seen_video_ids:
                    continue
                    
                seen_video_ids.add(video_id)
                deduplicated_videos.append(video)
                
            except Exception as e:
                logger.error(f"Error processing video in deduplication: {e}")
                continue
        
        competitor_videos = deduplicated_videos
        logger.info(f"After deduplication: {len(competitor_videos)} unique videos")

        # Initial formatting & collecting IDs
        formatted_competitor_video_data = []
        channel_ids_to_fetch = set()
        video_ids_to_fetch = []  # Use a list for video IDs as they are the primary key
        
        for video in competitor_videos:
            try:
                video_id = video["id"]["videoId"]
                channel_id = video['snippet']['channelId']

                formatted_data = {
                    "video_id": video_id,
                    "title": video["snippet"].get("title", "N/A"),
                    "description": video["snippet"].get("description", ""),
                    "thumbnail_urls": video["snippet"].get("thumbnails", {}),
                    "published_at": video["snippet"].get("publishedAt"),
                    'channel_id': channel_id,
                    'channel_title': video['snippet'].get('channelTitle', "N/A"),
                    # Placeholders for counts to be added later
                    'view_count': None,
                    'subscriber_count': None
                }
                formatted_competitor_video_data.append(formatted_data)
                if channel_id:
                    channel_ids_to_fetch.add(channel_id)
                video_ids_to_fetch.append(video_id)
                
            except KeyError as e:
                logger.error(f"KeyError processing competitor video snippet: {e}")
                continue

        # --- Fetch additional data ---
        subscriber_map = {}
        if channel_ids_to_fetch:
            logger.info(f"Fetching subscriber counts for {len(channel_ids_to_fetch)} unique channels...")
            subscriber_map = get_channel_subscriber_counts(client, list(channel_ids_to_fetch))

        # Fetch detailed video stats (views, likes, comments)
        video_stats_map = {}
        if video_ids_to_fetch:
            logger.info(f"Fetching detailed stats for {len(video_ids_to_fetch)} videos...")
            try:
                # Get detailed video stats including viewCount, likeCount, commentCount, and tags
                video_stats_request = client.videos().list(
                    part="statistics,contentDetails,snippet",
                    id=",".join(video_ids_to_fetch),
                    regionCode="US"
                ).execute()
                
                for video_item in video_stats_request.get("items", []):
                    video_id = video_item["id"]
                    statistics = video_item.get("statistics", {})
                    snippet = video_item.get("snippet", {})
                    
                    # Check if the video is in English using API language fields
                    default_lang = snippet.get("defaultLanguage", "")
                    audio_lang = snippet.get("defaultAudioLanguage", "")
                    is_english = default_lang.startswith("en") or audio_lang.startswith("en")
                    
                    video_stats_map[video_id] = {
                        "view_count": int(statistics.get("viewCount", 0)),
                        "like_count": int(statistics.get("likeCount", 0)),
                        "comment_count": int(statistics.get("commentCount", 0)),
                        "duration": video_item.get("contentDetails", {}).get("duration", ""),
                        "full_title": snippet.get("title", ""),
                        "full_description": snippet.get("description", ""),
                        "tags": snippet.get("tags", []),
                        "is_english": is_english,
                        "default_language": default_lang,
                        "default_audio_language": audio_lang
                    }
            except Exception as e:
                logger.error(f"Error fetching detailed video stats: {e}")

        # --- Augment the data and filter for channels with >100K subscribers ---
        filtered_competitor_data = []
        min_subscriber_threshold = 100000
        
        for video_data_item in formatted_competitor_video_data:
            ch_id = video_data_item.get('channel_id')
            vid_id = video_data_item.get('video_id')
            
            # Add subscriber count
            subscriber_count = subscriber_map.get(ch_id) if ch_id else None
            video_data_item['subscriber_count'] = subscriber_count
            
            # Only process videos from channels with >100K subscribers
            if subscriber_count is None or subscriber_count <= min_subscriber_threshold:
                continue
            
            # Add video stats
            video_stats = video_stats_map.get(vid_id, {})
            view_count = video_stats.get("view_count", 0)
            like_count = video_stats.get("like_count", 0)
            comment_count = video_stats.get("comment_count", 0)
            
            # Get title and check for English content if language flag is not set
            title = video_stats.get("full_title", video_data_item['title'])
            description = video_stats.get("full_description", video_data_item['description'])
            
            # Check if video is in English
            is_english = video_stats.get("is_english", False)
            
            # If language flag not set, perform enhanced language check
            if not is_english:
                # Try ASCII encoding check
                try:
                    # Check for mostly ASCII characters (allows some accented chars)
                    ascii_chars = sum(1 for c in title if ord(c) < 128)
                    ascii_ratio = ascii_chars / len(title) if len(title) > 0 else 0
                    
                    # Check for common English words
                    common_english_words = [
                        'the', 'and', 'for', 'to', 'in', 'of', 'a', 'how', 'what', 'why',
                        'with', 'on', 'this', 'that', 'my', 'your', 'best', 'top', 'new'
                    ]
                    title_words = title.lower().split()
                    desc_words = description.lower().split()
                    
                    english_word_count = sum(1 for word in title_words if word in common_english_words)
                    english_word_count += sum(1 for word in desc_words[:50] if word in common_english_words)
                    
                    # Consider it English if:
                    # - High ASCII ratio (>80%) AND has English words, OR
                    # - Has multiple (3+) common English words
                    is_english = (ascii_ratio > 0.8 and english_word_count >= 2) or english_word_count >= 3
                    
                except Exception as lang_check_error:
                    logger.debug(f"Language check failed for video {vid_id}: {lang_check_error}")
                    is_english = False
            
            # Skip non-English videos
            if not is_english:
                continue
            
            # Update with full details from the videos.list API call
            video_data_item['view_count'] = view_count
            video_data_item['like_count'] = like_count
            video_data_item['comment_count'] = comment_count
            
            # Add full title, description and tags from the detailed API response
            video_data_item['title'] = video_stats.get("full_title", video_data_item['title'])
            video_data_item['description'] = video_stats.get("full_description", video_data_item['description'])
            video_data_item['tags'] = video_stats.get("tags", [])
            video_data_item['duration'] = video_stats.get("duration", "")
            
            # View/Subscriber ratio
            video_data_item['view_to_subscriber_ratio'] = round(view_count / subscriber_count, 4) if subscriber_count else 0
            
            # Engagement rates (likes + comments per view)
            engagement_count = like_count + comment_count
            video_data_item['engagement_count'] = engagement_count
            video_data_item['engagement_rate'] = round(engagement_count / view_count * 100, 2) if view_count else 0
            
            # Average views per day
            try:
                published_at = video_data_item.get("published_at")
                if published_at:
                    published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    days_since_upload = max(1, (now_utc - published_date).days)  # Ensure at least 1 day
                    video_data_item['days_since_upload'] = days_since_upload
                    video_data_item['views_per_day'] = round(view_count / days_since_upload, 2)
                else:
                    video_data_item['days_since_upload'] = None
                    video_data_item['views_per_day'] = None
            except Exception as date_error:
                logger.error(f"Error calculating days since upload: {date_error}")
                video_data_item['days_since_upload'] = None
                video_data_item['views_per_day'] = None
            
            filtered_competitor_data.append(video_data_item)
        
        # Limit to top 20 if we have more than that
        filtered_competitor_data = filtered_competitor_data[:20]
        
        logger.info(f"Returning {len(filtered_competitor_data)} competitors with >100K subscribers and calculated metrics.")
        
        return filtered_competitor_data

    except Exception as e:
        logger.error(f"YouTube API search or processing failed for user {user_id}, category {category_id}, query '{query}': {e}")
        return []


def sort_competitors_by_performance(competitor_videos):
    """
    Sort competitor videos by a performance score that combines:
    - view_to_subscriber_ratio (how well video performs relative to channel size)
    - subscriber_count (gives weight to established channels)
    
    A higher score means better performance.
    """
    if not competitor_videos:
        return []
    
    def calculate_performance_score(video):
        # Get base metrics
        view_sub_ratio = video.get('view_to_subscriber_ratio', 0)
        subscriber_count = video.get('subscriber_count', 0)
        engagement_rate = video.get('engagement_rate', 0)
        views_per_day = video.get('views_per_day', 0)
        
        # Normalize subscriber count (log scale to prevent domination by huge channels)
        # 1M subscribers = 6, 100K = 5, etc.
        sub_score = math.log10(max(subscriber_count, 1000)) - 3
        
        # Combined score - view/sub ratio is given highest weight
        # The formula prioritizes videos that significantly outperform their channel's typical reach
        # while still giving credit to larger channels
        performance_score = (
            view_sub_ratio * 0.6 + 
            sub_score * 0.2 + 
            engagement_rate * 0.1 + 
            (views_per_day / 1000000 * 0.1 if views_per_day else 0)
        )
        
        return performance_score
    
    # Add performance score to each video
    for video in competitor_videos:
        video['performance_score'] = calculate_performance_score(video)
    
    # Sort by performance score (descending)
    sorted_videos = sorted(competitor_videos, key=lambda x: x.get('performance_score', 0), reverse=True)
    
    return sorted_videos
