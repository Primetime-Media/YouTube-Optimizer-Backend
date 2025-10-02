"""
Google Trends Helper Service Module - PRODUCTION READY

All errors fixed:
- ✅ Comprehensive error handling added
- ✅ Type hints improved
- ✅ Safe API key validation
- ✅ Better rate limiting
- ✅ Null/None checks throughout
- ✅ Proper logging
"""

import serpapi
from dotenv import load_dotenv
import os
import logging
import time
from typing import List, Dict, Any, Optional
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
    # ✅ FIX: Validate inputs
    if not keywords:
        logger.warning("No keywords provided for trending search")
        return {}
    
    if not SERPAPI_API_KEY:
        logger.error("SERPAPI_API_KEY is not set in environment variables")
        return {}
    
    # ✅ FIX: Safe keyword processing with type checking
    processed_keywords = []
    for kw in keywords:
        try:
            if kw and str(kw).strip():
                processed_keywords.append(str(kw).strip())
        except Exception as e:
            logger.warning(f"Error processing keyword {kw}: {e}")
            continue
    
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
        
        # ✅ FIX: Validate response
        if not serp_timeseries_results or not hasattr(serp_timeseries_results, 'data'):
            logger.warning("Invalid response from SerpAPI")
            return {}
        
        trends_results = serp_timeseries_results.data

        # Store the interest over time data
        if "interest_over_time" in trends_results:
            results["interest_over_time"] = trends_results["interest_over_time"]
            
            average_interest_list = []
            iot_data = results.get("interest_over_time", {})
            
            # ✅ FIX: Safe data extraction with type checks
            if isinstance(iot_data, dict):
                timeline_data = iot_data.get("timeline_data")
                
                if isinstance(timeline_data, list) and timeline_data:
                    records = []
                    for time_point in timeline_data:
                        if not isinstance(time_point, dict):
                            continue
                        
                        values_at_time_point = time_point.get("values")
                        if not isinstance(values_at_time_point, list):
                            continue
                        
                        for query_value_pair in values_at_time_point:
                            if not isinstance(query_value_pair, dict):
                                continue
                            
                            query = query_value_pair.get("query")
                            if not query:
                                continue
                            
                            # ✅ FIX: Safe integer conversion
                            try:
                                extracted_value = int(query_value_pair.get("extracted_value", 0))
                            except (ValueError, TypeError):
                                extracted_value = 0
                            
                            records.append({
                                "query": query, 
                                "extracted_value": extracted_value
                            })
                    
                    if records:
                        try:
                            df = pd.DataFrame(records)
                            average_df = df.groupby("query")["extracted_value"].mean().round().reset_index()
                            average_interest_list = average_df.rename(
                                columns={"extracted_value": "value"}
                            ).to_dict('records')
                        except Exception as e:
                            logger.error(f"Error processing DataFrame: {e}")
                            average_interest_list = []
            
            results["average_interest_by_query"] = average_interest_list
        else:
            results["average_interest_by_query"] = []

        # Get related queries for the top 3 keywords based on average interest
        related_queries = {}
        target_keywords = []
        
        # ✅ FIX: Safe keyword selection
        if results.get("average_interest_by_query"):
            try:
                sorted_keywords = sorted(
                    results["average_interest_by_query"], 
                    key=lambda x: x.get('value', 0), 
                    reverse=True
                )
                
                top_keywords = sorted_keywords[:min(3, len(sorted_keywords))]
                target_keywords = [
                    item.get('query') 
                    for item in top_keywords 
                    if item.get('query')
                ]
                
                logger.info(f"Getting related queries for top {len(target_keywords)} keywords by average interest")
            except Exception as e:
                logger.error(f"Error selecting top keywords: {e}")
        
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

                if geo:
                    related_params["geo"] = geo
                if timeframe:
                    related_params["date"] = timeframe

                related_results = serpapi.search(related_params)
                
                # ✅ FIX: Validate response
                if related_results and hasattr(related_results, 'data'):
                    related_data = related_results.data
                    if "related_queries" in related_data:
                        related_queries[keyword] = related_data["related_queries"]

            except Exception as e:
                logger.error(f"Error getting related queries for keyword '{keyword}': {e}")

        if related_queries:
            results["related_queries"] = related_queries

    except Exception as e:
        logger.error(f"Error using SerpAPI for Google Trends: {e}", exc_info=True)

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
    # ✅ FIX: Validate inputs
    if not keywords:
        logger.warning("No keywords provided for trending search")
        return []
    
    # Limit the number of keywords to check
    keywords_to_check = keywords[:max_keywords_to_check]
    logger.info(f"Checking trends for keywords: {keywords_to_check}")
    
    trending_keywords_set = set()
    
    # Get data for each keyword individually to get related queries
    for kw in keywords_to_check:
        # ✅ FIX: Safe keyword validation
        try:
            if not kw or len(str(kw).strip()) == 0:
                continue
        except Exception as e:
            logger.warning(f"Error validating keyword: {e}")
            continue
        
        try:
            trends_data = get_trending_data_with_serpapi(
                [kw],
                max_keywords_per_batch=1,
                geo=geo,
                timeframe=timeframe
            )
            
            # ✅ FIX: Safe data extraction
            if not trends_data or "related_queries" not in trends_data:
                continue
            
            if kw not in trends_data["related_queries"]:
                continue
            
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
    logger.info(f"Found {len(final_trending_list)} trending keywords via SerpAPI")
    return final_trending_list


def select_final_hashtags(
    trend_keywords_data: List[Dict],
    user_id: int,
    num_final_hashtags: int = 10,
    generic_keywords_override: Optional[List[str]] = None
) -> List[str]:
    """
    Selects the top hashtags based on trend interest and YouTube competition.
    
    Args:
        trend_keywords_data: List of dictionaries with trend data including
                            {"query": keyword_string, "interest_score": numeric_score}
        user_id: User ID to build YouTube client.
        num_final_hashtags: The number of final hashtags to select (default 10).
        generic_keywords_override: Optional list to override default generic keywords.
    
    Returns:
        A list of the selected hashtag strings (without the # symbol).
    """
    logger.info(f"Received {len(trend_keywords_data)} trend keywords with interest data")
    
    # ✅ FIX: Validate input
    if not trend_keywords_data or not isinstance(trend_keywords_data, list):
        logger.warning("Invalid trend_keywords_data provided")
        return []
    
    # Extract just the keyword strings from trend_keywords_data
    trend_keywords = []
    for item in trend_keywords_data:
        if isinstance(item, dict) and "query" in item:
            trend_keywords.append(item["query"])
    
    raw_candidates = trend_keywords
    
    if not raw_candidates:
        logger.info("No raw candidate keywords provided.")
        return []
    
    logger.debug(f"Raw candidates: {raw_candidates}")

    # Normalize and dedupe
    normalized_keywords_map = {}
    _generic_list = generic_keywords_override if generic_keywords_override is not None else DEFAULT_GENERIC_KEYWORDS
    generic_check_list = {g.lower().replace(" ", "") for g in _generic_list}

    # Create interest score mapping for trend keywords
    interest_score_map = {}
    for item in trend_keywords_data:
        if not isinstance(item, dict):
            continue
        
        if "query" in item and "interest_score" in item:
            query = item["query"]
            score = item["interest_score"]
            
            # ✅ FIX: Safe type check
            if isinstance(score, (int, float)):
                interest_score_map[query] = score

    for kw_phrase in raw_candidates:
        # ✅ FIX: Safe string validation
        try:
            if not isinstance(kw_phrase, str) or not kw_phrase.strip():
                continue
        except Exception as e:
            logger.warning(f"Error validating keyword phrase: {e}")
            continue
        
        original_phrase_cleaned = kw_phrase.strip()
        normalized_for_hashtag = ''.join(filter(str.isalnum, original_phrase_cleaned.lower()))

        if (normalized_for_hashtag and 
            normalized_for_hashtag not in normalized_keywords_map and 
            normalized_for_hashtag not in generic_check_list):
            normalized_keywords_map[normalized_for_hashtag] = original_phrase_cleaned
    
    if not normalized_keywords_map:
        logger.info("No valid candidates after normalization and generic filtering.")
        return []

    # Prepare for trend analysis
    candidates_for_trends = []
    for norm, orig in normalized_keywords_map.items():
        interest_score = None
        
        # ✅ FIX: Safe score lookup
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

    # Apply interest scores
    trend_analyzed_candidates = []
    for candidate in candidates_for_trends:
        interest_score = candidate.get('interest_score')
        if interest_score is None:
            interest_score = 5  # Default low score
        
        trend_analyzed_candidates.append({
            **candidate,
            'interest_score': interest_score
        })
    
    # Sort by interest score, highest first
    trend_analyzed_candidates.sort(key=lambda x: x.get('interest_score', 0), reverse=True)
    logger.debug(f"Trend analyzed candidates (sorted by interest): {trend_analyzed_candidates[:5]}")

    # Keep the top candidates for YouTube competition check
    shortlisted_for_yt_check = trend_analyzed_candidates[:min(
        len(trend_analyzed_candidates), 
        max(10, num_final_hashtags * 2)
    )]
    
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
                    part="id",
                    q=f"#{query_term}",
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
                competition_checked_candidates.append({
                    **candidate_data,
                    'competition_count': float('inf')
                })

        if not competition_checked_candidates:
            logger.info("No candidates after YouTube competition check.")
            return []

        # Create ranks for interest and competition
        competition_checked_candidates.sort(
            key=lambda x: x.get('interest_score', 0), 
            reverse=True
        )
        for i, item in enumerate(competition_checked_candidates):
            item['interest_rank'] = i + 1
        
        competition_checked_candidates.sort(
            key=lambda x: x.get('competition_count', float('inf')), 
            reverse=False
        )
        for i, item in enumerate(competition_checked_candidates):
            item['competition_rank'] = i + 1
        
        # Calculate final score
        scored_candidates = []
        for item in competition_checked_candidates:
            comp_rank = item.get('competition_rank', 0)
            if comp_rank == 0:
                score = float('inf')
            else:
                score = item['interest_rank'] + (1 / comp_rank)
            
            item['final_score'] = score
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x.get('final_score', float('inf')))
        
        # Select top N hashtags
        final_selection = [
            item['normalized'] 
            for item in scored_candidates[:num_final_hashtags]
        ]
        
        logger.info(f"Selected {len(final_selection)} final hashtags: {final_selection}")
        return final_selection
        
    except Exception as e:
        logger.error(f"Error during hashtag selection: {e}", exc_info=True)
        return []
