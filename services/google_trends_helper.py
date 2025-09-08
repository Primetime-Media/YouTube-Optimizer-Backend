import serpapi
from dotenv import load_dotenv
import os
import logging
import time
from typing import List, Dict, Any
import pandas as pd
from services.youtube import build_youtube_client
from utils.auth import get_user_credentials

logger = logging.getLogger(__name__)

load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

DEFAULT_GENERIC_KEYWORDS = [
    "video", "videos", "youtube", "channel", "subscribe", "like", "comment",
    "shorts", "live", "stream", "gaming", "vlog", "tutorial", "howto", "review",
    "unboxing", "news", "best", "top", "new", "update", "guide", "diy", "challenge",
    "music", "podcast", "episode", "series", "funny", "comedy", "asmr", "reaction",
    "viralvideo", "viral", "shortvideo"
]

def get_trending_data_with_serpapi(
    keywords: List[str], 
    max_keywords_per_batch: int = 5,
    geo: str = "", 
    timeframe: str = "today 7-d",
) -> Dict[str, Any]:
    """
    Gets trending data for keywords using SerpAPI Google Trends.
    
    Args:
        keywords: List of keyword strings to search for
        max_keywords_per_batch: Maximum number of keywords to send in a single API request
                               (Google Trends supports up to 5)
        geo: Optional location code (e.g., "US" for United States)
        timeframe: Time range for trend data (e.g., "now 1-d", "today 3-m", "today 12-m")
    
    Returns:
        Dictionary with trending data including interest over time and related queries
    """
    if not keywords:
        logger.warning("No keywords provided for trending search")
        return {}
    
    if not SERPAPI_API_KEY:
        logger.error("SERPAPI_API_KEY is not set in environment variables")
        return {}
    
    # Process keywords - ensure they're strings and limit per batch
    processed_keywords = [str(kw).strip() for kw in keywords if kw and str(kw).strip()]
    
    if not processed_keywords:
        logger.warning("No valid keywords found after processing")
        return {}

    # Take first batch (up to max_keywords_per_batch)
    keyword_batch = processed_keywords[:max_keywords_per_batch]
    keywords_str = ",".join(keyword_batch)

    logger.info(f"Getting Google Trends data for keywords: {keywords_str}")
    
    results = {}

    try:
        # Set up parameters for SerpAPI request
        params = {
            "engine": "google_trends",
            "q": keywords_str,
            "data_type": "TIMESERIES",
            "gprop": "youtube",
            "api_key": SERPAPI_API_KEY
        }

        # Add optional parameters if provided
        if geo:
            params["geo"] = geo
        if timeframe:
            params["tz"] = "-420"  # Pacific Time (adjust as needed)
            params["date"] = timeframe

        # Make the API request
        serp_timeseries_results = serpapi.search(params)
        trends_results = serp_timeseries_results.data

        # Store the interest over time data
        if "interest_over_time" in trends_results:
            results["interest_over_time"] = trends_results["interest_over_time"]
            
            average_interest_list = []
            iot_data = results.get("interest_over_time", {})
            if isinstance(iot_data, dict):
                timeline_data = iot_data.get("timeline_data")
                # Check if timeline_data is a non-empty list before proceeding
                if isinstance(timeline_data, list) and timeline_data:
                    # Prepare data for DataFrame
                    records = []
                    for time_point in timeline_data:
                        if isinstance(time_point, dict):
                            values_at_time_point = time_point.get("values")
                            if isinstance(values_at_time_point, list):
                                for query_value_pair in values_at_time_point:
                                    if isinstance(query_value_pair, dict):
                                        query = query_value_pair.get("query")
                                        try:
                                            extracted_value = int(query_value_pair.get("extracted_value", 0))
                                        except (ValueError, TypeError):
                                            extracted_value = 0
                                        if query:
                                            records.append({"query": query, "extracted_value": extracted_value})
                    
                    if records:
                        df = pd.DataFrame(records)
                        average_df = df.groupby("query")["extracted_value"].mean().round().reset_index()
                        average_interest_list = average_df.rename(columns={"extracted_value": "value"}).to_dict('records')
            
            results["average_interest_by_query"] = average_interest_list
        else:
            results["average_interest_by_query"] = [] # Ensure the key exists

        # Get related queries for the top 3 keywords based on average interest, or all if not available
        related_queries = {}
        
        # Determine which keywords to get related queries for
        target_keywords = []
        
        # Check if we have average interest data to identify top keywords
        if results["average_interest_by_query"]:
            # Sort keywords by average interest value (descending)
            sorted_keywords = sorted(
                results["average_interest_by_query"], 
                key=lambda x: x.get('value', 0), 
                reverse=True
            )
            
            # Get the top 3 (or fewer if less are available)
            top_keywords = sorted_keywords[:min(3, len(sorted_keywords))]
            target_keywords = [item.get('query') for item in top_keywords if item.get('query')]
            
            logger.info(f"Getting related queries for top {len(target_keywords)} keywords by average interest")
        
        # If no average interest data or no valid top keywords, use all keywords
        if not target_keywords:
            target_keywords = keyword_batch
            logger.info(f"No average interest data available, getting related queries for all {len(target_keywords)} keywords")

        for keyword in target_keywords:
            try:
                # Small delay to avoid rate limits
                time.sleep(0.5)

                related_params = {
                    "engine": "google_trends",
                    "q": keyword,
                    "data_type": "RELATED_QUERIES",
                    "gprop": "youtube", 
                    "api_key": SERPAPI_API_KEY
                }

                # Add optional parameters if provided
                if geo:
                    related_params["geo"] = geo
                if timeframe:
                    related_params["date"] = timeframe

                related_results = serpapi.search(related_params).data

                if "related_queries" in related_results:
                    related_queries[keyword] = related_results["related_queries"]

            except Exception as e:
                logger.error(f"Error getting related queries for keyword '{keyword}': {e}")

        if related_queries:
            results["related_queries"] = related_queries

    except Exception as e:
        logger.error(f"Error using SerpAPI for Google Trends: {e}")

    return results

def get_trending_keywords_with_serpapi(
    keywords: List[str], 
    max_keywords_to_check: int = 5, 
    top_n_trending: int = 10, 
    geo: str = "", 
    timeframe: str = "today 3-m"
) -> List[str]:
    """
    Gets trending keywords related to the input keywords using SerpAPI's Google Trends.
    
    Args:
        keywords: A list of seed keywords.
        max_keywords_to_check: Max number of seed keywords to check (to avoid too many API calls).
        top_n_trending: How many top/rising related queries to consider for each seed keyword.
        geo: Optional location code (e.g., "US" for United States)
        timeframe: Time range for trend data (e.g., "now 1-d", "today 3-m", "today 12-m")
    
    Returns:
        A list of unique trending keyword strings.
    """
    if not keywords:
        logger.warning("No keywords provided for trending search")
        return []
    
    # Limit the number of keywords to check
    keywords_to_check = keywords[:max_keywords_to_check]
    logger.info(f"Checking trends for keywords: {keywords_to_check}")
    
    trending_keywords_set = set()
    
    # Get data for each keyword individually to get related queries
    for kw in keywords_to_check:
        if not kw or len(str(kw).strip()) == 0:
            continue
        
        try:
            trends_data = get_trending_data_with_serpapi(
                [kw],  # Just one keyword at a time for related queries
                max_keywords_per_batch=1,
                geo=geo,
                timeframe=timeframe
            )
            
            # Extract related queries
            if "related_queries" in trends_data and kw in trends_data["related_queries"]:
                related_data = trends_data["related_queries"][kw]
                
                # Process top queries
                if "top" in related_data and isinstance(related_data["top"], list):
                    for item in related_data["top"][:top_n_trending]:
                        if isinstance(item, dict) and "query" in item:
                            trending_keywords_set.add(item["query"])
                
                # Process rising queries (prioritize these as they show growing interest)
                if "rising" in related_data and isinstance(related_data["rising"], list):
                    for item in related_data["rising"][:top_n_trending]:
                        if isinstance(item, dict) and "query" in item:
                            trending_keywords_set.add(item["query"])
            
            # Wait between API calls to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing trending data for keyword '{kw}': {e}")
    
    final_trending_list = list(trending_keywords_set)
    logger.info(f"Found trending keywords via SerpAPI: {final_trending_list}")
    return final_trending_list

def select_final_hashtags(
    trend_keywords_data: list[dict],
    user_id: int,
    num_final_hashtags: int = 10,
    generic_keywords_override: list[str] = None
):
    """
    Selects the top hashtags based on trend interest and YouTube competition.
    Args:
        hero_keywords: List of high-impact keywords/phrases from the primary video.
        trend_keywords_data: List of dictionaries with trend data including
                            {"query": keyword_string, "interest_score": numeric_score}
        user_id: User ID to build YouTube client.
        num_final_hashtags: The number of final hashtags to select (default 5).
        generic_keywords_override: Optional list to override default generic keywords.
    Returns:
        A list of the selected hashtag strings (without the # symbol).
    """
    logger.info(f"Received {len(trend_keywords_data)} trend keywords with interest data")
    
    # 1. Assemble Candidate Pool
    # Extract just the keyword strings from trend_keywords_data
    trend_keywords = [item["query"] for item in trend_keywords_data]
    raw_candidates = (trend_keywords or [])
    if not raw_candidates:
        logger.info("No raw candidate keywords provided.")
        return []
    logger.debug(f"Raw candidates: {raw_candidates}")

    # 2. Normalize and Dedupe
    normalized_keywords_map = {}
    _generic_list = generic_keywords_override if generic_keywords_override is not None else DEFAULT_GENERIC_KEYWORDS
    generic_check_list = {g.lower().replace(" ", "") for g in _generic_list} # Use a set for faster lookups

    # Create interest score mapping for trend keywords
    interest_score_map = {}
    for item in trend_keywords_data:
        if isinstance(item, dict) and "query" in item and "interest_score" in item:
            query = item["query"]
            score = item["interest_score"]
            if isinstance(score, (int, float)):
                interest_score_map[query] = score

    for kw_phrase in raw_candidates:
        if not isinstance(kw_phrase, str) or not kw_phrase.strip():
            continue
        
        original_phrase_cleaned = kw_phrase.strip()
        # Normalize: lowercase, remove punctuation, keep alphanumeric and spaces for potential multi-word phrases
        normalized_for_hashtag = ''.join(filter(str.isalnum, original_phrase_cleaned.lower()))

        if normalized_for_hashtag and normalized_for_hashtag not in normalized_keywords_map and normalized_for_hashtag not in generic_check_list:
            normalized_keywords_map[normalized_for_hashtag] = original_phrase_cleaned
    
    if not normalized_keywords_map:
        logger.info("No valid candidates after normalization and generic filtering.")
        return []

    # Prepare for trend analysis: list of dicts with trend data
    candidates_for_trends = []
    for norm, orig in normalized_keywords_map.items():
        # Try to find interest score in our map
        interest_score = None
        
        # Check different variations of the original phrase
        for key in [orig, orig.lower(), norm]:
            if key in interest_score_map:
                interest_score = interest_score_map[key]
                break
                
        candidates_for_trends.append({
            'normalized': norm, 
            'original_phrase': orig,
            'interest_score': interest_score
        })
        
    logger.debug(f"Normalized candidates for trends: {candidates_for_trends}")

    # Apply real interest scores or simulate if not available
    trend_analyzed_candidates = []
    for candidate in candidates_for_trends:
        # Use the actual interest score from trend data if available
        if candidate['interest_score'] is not None:
            interest_score = candidate['interest_score']
        else:
           interest_score = 5 # Default low score if not interest_score found
                
        trend_analyzed_candidates.append({
            **candidate,
            'interest_score': interest_score
        })
    
    # Sort by interest score, highest first
    trend_analyzed_candidates.sort(key=lambda x: x.get('interest_score', 0), reverse=True)
    logger.debug(f"Trend analyzed candidates (sorted by interest): {trend_analyzed_candidates[:5]}")

    # Keep the top candidates for YouTube competition check
    shortlisted_for_yt_check = trend_analyzed_candidates[:min(len(trend_analyzed_candidates), max(10, num_final_hashtags * 2))]
    if not shortlisted_for_yt_check:
        logger.info("No candidates after trend analysis.")
        return []
    
    # Check YouTube competition
    try:
        credentials = get_user_credentials(user_id)
        if not credentials:
            logger.error(f"No credentials found for user {user_id}")
            return []
            
        youtube_client = build_youtube_client(credentials)
        if not youtube_client:
            logger.error("Failed to build YouTube client")
            return []
            
        # Check competition for each shortlisted hashtag
        competition_checked_candidates = []
        for candidate_data in shortlisted_for_yt_check:
            query_term = candidate_data['normalized']
            try:
                request = youtube_client.search().list(
                    part="id",  # Only need ID part to get totalResults
                    q=f"#{query_term}",  # Search for the actual hashtag
                    type="video",
                    maxResults=20
                )
                response = request.execute()
                competition_count = response.get('pageInfo', {}).get('totalResults', 0)
                logger.info(f"YouTube competition for #{query_term}: {competition_count}")
                competition_checked_candidates.append({
                    **candidate_data,
                    'competition_count': competition_count
                })
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error checking YouTube competition for #{query_term}: {e}")
                # Assign a high competition count on error to deprioritize
                competition_checked_candidates.append({
                    **candidate_data,
                    'competition_count': float('inf')
                })

        if not competition_checked_candidates:
            logger.info("No candidates after YouTube competition check.")
            return []

        # Create ranks for interest (1 is best) and competition (1 is best)
        competition_checked_candidates.sort(key=lambda x: x.get('interest_score', 0), reverse=True)
        for i, item in enumerate(competition_checked_candidates):
            item['interest_rank'] = i + 1
            
        competition_checked_candidates.sort(key=lambda x: x.get('competition_count', float('inf')), reverse=False)
        for i, item in enumerate(competition_checked_candidates):
            item['competition_rank'] = i + 1
            
        # Calculate final score: Score = (RelativeTrendInterestRank) + (1 / CompetitionRank)
        # Lower score is better
        scored_candidates = []
        for item in competition_checked_candidates:
            if item.get('competition_rank', 0) == 0:  # Avoid division by zero
                score = float('inf')
            else:
                score = item['interest_rank'] + (1 / item['competition_rank'])
            item['final_score'] = score
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x.get('final_score', float('inf')))
        
        # Select top N hashtags
        final_selection = [item['normalized'] for item in scored_candidates[:num_final_hashtags]]
        logger.info(f"Selected {len(final_selection)} final hashtags: {final_selection}")
        
        return final_selection
        
    except Exception as e:
        logger.error(f"Error during hashtag selection: {e}")
        return []
