"""
Competitor Analysis Service Module - COMPLETE FIXED VERSION
============================================================
7 Critical Errors Fixed - Production Ready

Key Fixes Applied:
1. Undefined variable error fixed (video_ids initialization)
2. NULL checks added throughout
3. Proper error handling
4. Video deduplication logic
5. Language detection improvements
"""

import logging
from typing import Dict, List, Optional, Set
from googleapiclient.discovery import build
import os

logger = logging.getLogger(__name__)

# Initialize YouTube API client
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = None

if YOUTUBE_API_KEY:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
else:
    logger.warning("YOUTUBE_API_KEY not set - competitor analysis will be limited")


def find_competitors_by_keywords(
    channel_keywords: List[str],
    max_results: int = 5
) -> List[Dict]:
    """
    Find competitor channels based on keywords
    
    FIXES:
    - #1: NULL check for youtube client
    - #2: NULL check for search results
    
    Args:
        channel_keywords: List of keywords to search
        max_results: Maximum number of competitors to return
        
    Returns:
        list: List of competitor channel data
    """
    if not youtube:
        logger.error("YouTube API client not initialized")
        return []
    
    if not channel_keywords:
        logger.warning("No keywords provided for competitor search")
        return []
    
    competitors = []
    seen_channel_ids = set()
    
    try:
        # Search for each keyword
        for keyword in channel_keywords[:3]:  # Limit to top 3 keywords
            try:
                request = youtube.search().list(
                    part="snippet",
                    q=keyword,
                    type="channel",
                    maxResults=max_results,
                    relevanceLanguage="en"
                )
                
                response = request.execute()
                
                # ✅ FIX: Check for NULL response or items
                if not response or 'items' not in response:
                    logger.warning(f"No results for keyword: {keyword}")
                    continue
                
                for item in response['items']:
                    channel_id = item['snippet']['channelId']
                    
                    # Skip duplicates
                    if channel_id in seen_channel_ids:
                        continue
                    
                    seen_channel_ids.add(channel_id)
                    
                    competitors.append({
                        'channel_id': channel_id,
                        'title': item['snippet'].get('title', 'Unknown'),
                        'description': item['snippet'].get('description', ''),
                        'thumbnail': item['snippet'].get('thumbnails', {}).get('default', {}).get('url', ''),
                        'keyword': keyword
                    })
                    
                    if len(competitors) >= max_results:
                        break
                
                if len(competitors) >= max_results:
                    break
                    
            except Exception as e:
                logger.error(f"Error searching for keyword '{keyword}': {e}")
                continue
        
        logger.info(f"Found {len(competitors)} competitor channels")
        return competitors[:max_results]
        
    except Exception as e:
        logger.error(f"Error in competitor search: {e}", exc_info=True)
        return []


def get_competitor_video_titles(
    competitor_channel_ids: List[str],
    max_videos_per_channel: int = 10
) -> List[Dict]:
    """
    Get recent video titles from competitor channels
    
    FIXES:
    - #3: NULL check for youtube client
    - #4: NULL check for API responses
    - #5: Initialize video_ids = set() before use
    
    Args:
        competitor_channel_ids: List of competitor channel IDs
        max_videos_per_channel: Maximum videos to retrieve per channel
        
    Returns:
        list: List of video data with titles and metadata
    """
    if not youtube:
        logger.error("YouTube API client not initialized")
        return []
    
    if not competitor_channel_ids:
        logger.warning("No competitor channel IDs provided")
        return []
    
    all_videos = []
    video_ids: Set[str] = set()  # ✅ FIX: Initialize before use
    
    try:
        for channel_id in competitor_channel_ids:
            try:
                # Get uploads playlist ID
                channel_request = youtube.channels().list(
                    part="contentDetails",
                    id=channel_id
                )
                
                channel_response = channel_request.execute()
                
                # ✅ FIX: Check for NULL response
                if not channel_response or 'items' not in channel_response:
                    logger.warning(f"No channel data for ID: {channel_id}")
                    continue
                
                if not channel_response['items']:
                    continue
                
                uploads_playlist_id = (
                    channel_response['items'][0]
                    .get('contentDetails', {})
                    .get('relatedPlaylists', {})
                    .get('uploads')
                )
                
                if not uploads_playlist_id:
                    logger.warning(f"No uploads playlist for channel: {channel_id}")
                    continue
                
                # Get videos from uploads playlist
                playlist_request = youtube.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=max_videos_per_channel
                )
                
                playlist_response = playlist_request.execute()
                
                # ✅ FIX: Check for NULL response
                if not playlist_response or 'items' not in playlist_response:
                    logger.warning(f"No videos in playlist: {uploads_playlist_id}")
                    continue
                
                for item in playlist_response['items']:
                    video_id = item['snippet'].get('resourceId', {}).get('videoId')
                    
                    if not video_id:
                        continue
                    
                    # ✅ FIX: Check duplicates using initialized set
                    if video_id in video_ids:
                        continue
                    
                    video_ids.add(video_id)
                    
                    all_videos.append({
                        'video_id': video_id,
                        'title': item['snippet'].get('title', 'Unknown'),
                        'description': item['snippet'].get('description', ''),
                        'channel_id': channel_id,
                        'published_at': item['snippet'].get('publishedAt', ''),
                        'thumbnail': item['snippet'].get('thumbnails', {}).get('default', {}).get('url', '')
                    })
                    
            except Exception as e:
                logger.error(f"Error fetching videos for channel {channel_id}: {e}")
                continue
        
        logger.info(f"Retrieved {len(all_videos)} competitor videos")
        return all_videos
        
    except Exception as e:
        logger.error(f"Error in competitor video retrieval: {e}", exc_info=True)
        return []


def analyze_competitor_strategies(
    competitor_videos: List[Dict]
) -> Dict:
    """
    Analyze competitor content strategies
    
    FIXES:
    - #6: NULL check for input
    - #7: Improved language detection
    
    Args:
        competitor_videos: List of competitor video data
        
    Returns:
        dict: Analysis results with common patterns
    """
    # ✅ FIX: Check for NULL or empty input
    if not competitor_videos:
        logger.warning("No competitor videos to analyze")
        return {
            'common_keywords': [],
            'avg_title_length': 0,
            'language_patterns': {},
            'description_patterns': []
        }
    
    try:
        # Analyze titles
        all_titles = [v.get('title', '') for v in competitor_videos if v.get('title')]
        all_descriptions = [v.get('description', '') for v in competitor_videos if v.get('description')]
        
        # Calculate average title length
        title_lengths = [len(title) for title in all_titles if title]
        avg_title_length = sum(title_lengths) / len(title_lengths) if title_lengths else 0
        
        # Extract common keywords from titles
        word_freq = {}
        for title in all_titles:
            words = title.lower().split()
            for word in words:
                # Filter out very common words
                if len(word) > 3 and word.isalnum():
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        common_keywords = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        # ✅ FIX: Improved language detection
        language_patterns = {}
        for video in competitor_videos:
            title = video.get('title', '')
            description = video.get('description', '')
            
            # Simple language detection heuristics
            if not title:
                continue
            
            # Detect common patterns
            if any(char in title for char in '日本語中文한국어'):
                lang = 'CJK'  # Chinese/Japanese/Korean
            elif any(ord(char) > 127 for char in title):
                lang = 'Non-English'
            else:
                lang = 'English'
            
            language_patterns[lang] = language_patterns.get(lang, 0) + 1
        
        # Analyze description patterns
        description_patterns = []
        avg_desc_length = sum(len(d) for d in all_descriptions) / len(all_descriptions) if all_descriptions else 0
        
        if avg_desc_length > 200:
            description_patterns.append("Long descriptions (200+ chars)")
        
        # Check for common description elements
        has_links = sum(1 for d in all_descriptions if 'http' in d.lower())
        if has_links > len(all_descriptions) * 0.5:
            description_patterns.append("Links in majority of descriptions")
        
        has_hashtags = sum(1 for d in all_descriptions if '#' in d)
        if has_hashtags > len(all_descriptions) * 0.3:
            description_patterns.append("Frequent hashtag usage")
        
        return {
            'common_keywords': [{'word': word, 'count': count} for word, count in common_keywords],
            'avg_title_length': round(avg_title_length, 1),
            'language_patterns': language_patterns,
            'description_patterns': description_patterns,
            'total_videos_analyzed': len(competitor_videos)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing competitor strategies: {e}", exc_info=True)
        return {
            'common_keywords': [],
            'avg_title_length': 0,
            'language_patterns': {},
            'description_patterns': [],
            'error': str(e)
        }


def get_full_competitor_analysis(
    channel_keywords: List[str],
    max_competitors: int = 5,
    max_videos_per_competitor: int = 10
) -> Dict:
    """
    Perform complete competitor analysis
    
    Args:
        channel_keywords: Keywords to find competitors
        max_competitors: Maximum number of competitors to analyze
        max_videos_per_competitor: Maximum videos per competitor
        
    Returns:
        dict: Complete competitor analysis results
    """
    try:
        # Find competitors
        competitors = find_competitors_by_keywords(
            channel_keywords,
            max_results=max_competitors
        )
        
        if not competitors:
            return {
                'success': False,
                'message': 'No competitors found',
                'competitors': [],
                'analysis': {}
            }
        
        # Get competitor videos
        competitor_ids = [c['channel_id'] for c in competitors]
        competitor_videos = get_competitor_video_titles(
            competitor_ids,
            max_videos_per_channel=max_videos_per_competitor
        )
        
        # Analyze strategies
        analysis = analyze_competitor_strategies(competitor_videos)
        
        return {
            'success': True,
            'competitors': competitors,
            'competitor_videos': competitor_videos,
            'analysis': analysis
        }
        
    except Exception as e:
        logger.error(f"Error in full competitor analysis: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'competitors': [],
            'analysis': {}
        }
