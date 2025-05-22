import os
import logging
import anthropic
from dotenv import load_dotenv
import json
import re
from typing import Dict, List, Optional, Any
from services.google_trends_helper import get_trending_data_with_serpapi, select_final_hashtags

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Anthropic client
try:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY is not set in environment variables")

    client = anthropic.Anthropic(api_key=anthropic_api_key)
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {e}")
    client = None

# Default list of generic keywords to filter out
DEFAULT_GENERIC_KEYWORDS = [
    "video", "videos", "youtube", "channel", "subscribe", "like", "comment",
    "shorts", "live", "stream", "gaming", "vlog", "tutorial", "howto", "review",
    "unboxing", "news", "best", "top", "new", "update", "guide", "diy", "challenge",
    "music", "podcast", "episode", "series", "funny", "comedy", "asmr", "reaction"
]

def get_comprehensive_optimization(
    original_title: str,
    original_description: str = "",
    original_tags: Optional[List[str]] = None,
    transcript: str = "",
    has_captions: bool = False,
    like_count: int = 0,
    comment_count: int = 0,
    optimization_decision_data: Dict = {},
    analytics_data: Dict = {},
    competitor_analytics_data: Dict = {},
    category_name: str = "",
    model: str = "claude-3-7-sonnet-20250219",
    max_retries: int = 3,
    user_id: Optional[int] = None
) -> Dict:
    """
    Generate comprehensive optimizations for a YouTube video including
    title, description, and tags - all in one Claude request

    Args:
        original_title: The original video title
        original_description: The original video description
        original_tags: The original video tags
        model: The Claude model to use
        max_retries: Maximum number of retry attempts

    Returns:
        dict: Contains optimized title, description, tags and notes
    """
    if not client:
        logger.error("Anthropic client not initialized.")
        return {
            "original_title": original_title,
            "optimized_title": original_title,
            "original_description": original_description,
            "optimized_description": original_description,
            "original_tags": original_tags or [],
            "optimized_tags": original_tags or [],
            "optimization_notes": 'No optimization performed since Anthropic client not initialized.'
        }

    # Safely check transcript length
    transcript_length = 0
    if transcript is not None:
        transcript_length = len(transcript)
    else:
        transcript = ""
    logger.info(f"Transcript length: {transcript_length}")

    # Prepare tags string
    tags_str = ", ".join(original_tags) if original_tags else "No tags available"

    # Create an array to collect error messages from retry attempts
    retry_errors = []

    # Extract video keywords from video data
    video_keywords = extract_keywords_with_llm(
        original_title,
        original_description,
        transcript,
        original_tags,
        min_keywords=3,
        max_keywords=5
    )

    # Get trending keywords based on the extracted keywords using SerpAPI/Google Trends
    trending_data = get_trending_data_with_serpapi(
        keywords=video_keywords,
        max_keywords_per_batch=5,
        timeframe="now 7-d",
    )
    
    # Get complete competitor video keywords using SerpAPI/Google Trends
    competitor_trending_data = get_trending_data_with_serpapi(
        keywords=competitor_analytics_data.get('competitor_keywords', {}).get('keywords', []),
        max_keywords_per_batch=5,
        timeframe="now 7-d",
    )
    
    # Function to extract keywords from trending data
    def extract_keywords_from_trend_data(trend_data):
        try:
            extracted_keywords = []
            if not trend_data or not isinstance(trend_data, dict):
                return []

            # Extract from related_queries with scores
            related_queries = trend_data.get("related_queries", {})
            if related_queries and isinstance(related_queries, dict):
                for keyword, queries_data in related_queries.items():
                    # Extract from top queries
                    top_queries = queries_data.get("top", [])
                    if isinstance(top_queries, list):
                        for query_item in top_queries:
                            if isinstance(query_item, dict) and "query" in query_item:
                                query_text = query_item["query"]
                                # Extract value if available (represents interest)
                                value = query_item.get("value", 0)
                                if isinstance(value, str):
                                    try:
                                        value = int(value.replace("%", "").strip())
                                    except (ValueError, TypeError):
                                        value = 50  # Default value if parsing fails

                                # Add to results if not already in list
                                if not any(k["query"] == query_text for k in extracted_keywords):
                                    extracted_keywords.append({
                                        "query": query_text,
                                        "interest_score": value
                                    })

                    # Extract from rising queries (these are typically high interest)
                    rising_queries = queries_data.get("rising", [])
                    if isinstance(rising_queries, list):
                        for query_item in rising_queries:
                            if isinstance(query_item, dict) and "query" in query_item:
                                query_text = query_item["query"]
                                # Extract value - for rising queries this is often higher
                                value = query_item.get("value", 0)
                                if isinstance(value, str) and "+" in value:
                                    # Handle values like "+130%"
                                    try:
                                        value = int(value.replace("%", "").replace("+", "").strip())
                                        # Scale rising values to prioritize them (they show growth)
                                        value = min(100, 75 + value // 5)  # Cap at 100, but prioritize rising terms
                                    except (ValueError, TypeError):
                                        value = 80  # Default high value for rising queries

                                # Add to results if not already in list or update score if higher
                                existing_item = next((k for k in extracted_keywords if k["query"] == query_text), None)
                                if existing_item:
                                    existing_item["interest_score"] = max(existing_item["interest_score"], value)
                                else:
                                    extracted_keywords.append({
                                        "query": query_text,
                                        "interest_score": value
                                    })

        
            # Extract from average interest data
            avg_interest = trend_data.get("average_interest_by_query", [])
            if isinstance(avg_interest, list):
                for item in avg_interest:
                    if isinstance(item, dict) and "query" in item:
                        query_text = item["query"]
                        value = item.get("value", 50)  # Get interest value or default to 50

                        # Add to results if not already in list or update score if higher
                        existing_item = next((k for k in extracted_keywords if k["query"] == query_text), None)
                        if existing_item:
                            existing_item["interest_score"] = max(existing_item["interest_score"], value)
                        else:
                            extracted_keywords.append({
                                "query": query_text,
                                "interest_score": value
                            })
        except Exception as e:
            logger.error(f"Error while extracting keywords from trend data: {e}")
            extracted_keywords = []
                    
        return extracted_keywords
    
    # Extract keywords from both data sources
    trending_keywords_data = extract_keywords_from_trend_data(trending_data)
    competitor_trending_keywords_data = extract_keywords_from_trend_data(competitor_trending_data)
    
    # Extract just the keywords for logging
    trending_keywords = [item["query"] for item in trending_keywords_data]
    competitor_trending_keywords = [item["query"] for item in competitor_trending_keywords_data]
    
    logger.info(f"Extracted {len(trending_keywords)} trending keywords: {trending_keywords[:10]}")
    logger.info(f"Extracted {len(competitor_trending_keywords)} competitor trending keywords: {competitor_trending_keywords[:10]}")

    # Process and select optimal hashtags from hero keywords (video) and trend keywords (combined)
    optimized_hashtags = []
    try:
        # Combine trending data, preserving interest scores
        combined_trend_keywords_data = trending_keywords_data + competitor_trending_keywords_data
        
        # Filter out generic keywords using LLM
        if combined_trend_keywords_data:
            video_data = {
                "title": original_title,
                "description": original_description,
                "tags": original_tags,
                "transcript": transcript,
                "category_name": category_name,
            }
            # Extract just the keyword strings for LLM filtering
            keyword_strings = [item["query"] for item in combined_trend_keywords_data]
            filtered_keyword_strings = filter_generic_keywords_with_llm(keywords=keyword_strings, video_data=video_data)
            
            # Only keep keyword data for non-generic keywords
            combined_trend_keywords_data = [
                item for item in combined_trend_keywords_data 
                if item["query"] in filtered_keyword_strings
            ]
            logger.info(f"After generic keyword filtering: {len(combined_trend_keywords_data)} keywords remain")
        
        # Now pass the filtered keywords to select_final_hashtags
        if combined_trend_keywords_data and user_id:
            optimized_hashtags = select_final_hashtags(
                trend_keywords_data=combined_trend_keywords_data,
                user_id=user_id,
                num_final_hashtags=10
            )
            logger.info(f"Selected {len(optimized_hashtags)} optimized hashtags: {optimized_hashtags}")
        else:
            if not combined_trend_keywords_data:
                logger.warning("No trend keywords available for hashtag optimization")
            if not user_id:
                logger.warning("No user_id provided for hashtag optimization")
    except Exception as e:
        logger.error(f"Error during hashtag optimization: {e}")

    # Create sections to add to the prompt
    trending_keywords_prompt_section = ""
    if trending_keywords:
        trending_keywords_str = ", ".join(trending_keywords)
        trending_keywords_prompt_section = f"""
        TRENDING KEYWORDS TO CONSIDER (based on video content and Google Trends):
        {trending_keywords_str}
        Incorporate these trending keywords naturally if they fit the content's core topic and optimization goals.
        """
        logger.info(f"Adding trending keywords to prompt: {trending_keywords_str}")
    else:
        logger.info("No trending keywords to add to prompt or Pytrends unavailable.")
    
    # Add optimized hashtags section if available
    optimized_hashtags_prompt_section = ""
    if optimized_hashtags:
        # Format the hashtags with the # symbol for visibility
        formatted_hashtags = [f"#{tag}" for tag in optimized_hashtags]
        hashtags_str = ", ".join(formatted_hashtags)
        optimized_hashtags_prompt_section = f"""
        OPTIMIZED HASHTAGS (selected based on trend interest, relevance, and competition analysis):
        {hashtags_str}

        Consider these OPTIMIZED HASHTAGS when crafting the new TITLE, DESCRIPTION, and TAGS:
        - TITLE: If natural and concise, consider incorporating 1-2 of the most impactful hashtags directly into the title.
        - DESCRIPTION: Strategically weave relevant hashtags from this list into the description, especially in the first few lines and near relevant calls-to-action. Aim for 3-5 well-placed hashtags.
        - TAGS: Ensure that the final list of video tags includes these optimized hashtags, as they are pre-vetted for performance.
        These hashtags have been carefully selected. Prioritize their use where appropriate.
        """
        logger.info(f"Adding optimized hashtags to prompt: {hashtags_str}")
    
    # Try multiple times with slightly different prompts
    for attempt in range(max_retries):
        try:
            # Log retry attempts
            if attempt > 0:
                logger.info(f"Retry attempt {attempt} for comprehensive optimization")

            # Adjust temperature slightly on retries to get different outputs
            temperature = 0.7 + (attempt * 0.1)

            system_message = """You are a YouTube SEO expert who specializes in optimizing videos for maximum engagement, 
            searchability and audience retention. You understand what makes content perform well and how to optimize 
            metadata to increase views, watch time, and subscriber conversion.
            
            IMPORTANT: Your response MUST be valid JSON with properly escaped characters."""

            # Adjust the prompt slightly for each retry
            json_emphasis = ""
            if attempt > 0:
                json_emphasis = """
                EXTREMELY IMPORTANT: Format your response as VALID JSON.
                Do not include ANY control characters in strings that would break JSON parsing.
                Make sure all quotes are properly escaped and all values are valid JSON.
                """

            # Format transcript for prompt inclusion
            transcript_section = ""
            if transcript and len(transcript) > 0:
                # If transcript is very long, use the first ~1500 characters
                if len(transcript) > 2000:
                    transcript_text = transcript[:1500] + "... [transcript truncated]"
                else:
                    transcript_text = transcript

                transcript_section = f"""
                VIDEO TRANSCRIPT:
                {transcript_text}
                """

            prompt = f"""
            I need comprehensive optimization for the following YouTube video:

            ORIGINAL TITLE: {original_title}

            ORIGINAL DESCRIPTION:
            {original_description if original_description else "No description provided"}

            ORIGINAL TAGS: {tags_str}
            
            {transcript_section}
            
            {trending_keywords_prompt_section}
            
            {optimized_hashtags_prompt_section}

            ADDITIONAL CONTEXT & GOALS:
            - Video Category: {category_name if category_name else 'Unknown'}
            - Current Likes: {like_count}
            - Current Comments: {comment_count}
            - Reason for Optimization: {optimization_decision_data.get('reasons', 'N/A') if optimization_decision_data else 'N/A'}

            Use this context to inform your suggestions:
            1.  Tailor keywords and tone to the VIDEO CATEGORY.
            2.  Consider the CURRENT ENGAGEMENT metrics when crafting calls-to-action and assessing the need for change in the description.
            3.  Directly address the REASON FOR OPTIMIZATION in your final 'optimization_notes', explaining how your suggestions tackle the identified issues.

            Please provide optimized versions of the title, description, and tags, following these guidelines:

            For the TITLE:
            - Between 40-60 characters for optimal performance
            - Include high-value keywords for searchability
            - Create curiosity and emotional appeal 
            - Maintain the original meaning and intent
            - Use strategic capitalization, symbols, or emojis if appropriate
            - Consider including 1-2 of the most relevant OPTIMIZED HASHTAGS if they fit naturally and enhance discoverability.

            For the DESCRIPTION:
            - Front-load important keywords and OPTIMIZED HASHTAGS in the first 2-3 sentences.
            - Include a clear call-to-action (subscribe, like, comment)
            - Add timestamps for longer videos (if applicable)
            - Incorporate relevant OPTIMIZED HASHTAGS (3-5 is optimal) throughout the description, especially near related content or calls-to-action.
            - Keep it engaging but concise
            - EXTREMELY IMPORTANT: DO NOT include placeholder text like "[Link to Playlist]" or "[Social Media Links]" - ONLY include actual, real URLs that appear in the original description
            - If you don't have actual URLs or links, do not mention them at all
            - Avoid any placeholders - the description should be ready to publish as-is

            For the TAGS:
            - Focus on trending tags that are relevant to the content.
            - Crucially, ensure the provided OPTIMIZED HASHTAGS are included in your list of tags.**
            - Each tag should not contain any whitespace or punctuation; they should be single words or joined phrases (e.g., tag, thisisatag).
            - Include exact and phrase match keywords.
            - Focus on specific, niche tags rather than broad ones.
            - Start with most important keywords.
            - Include misspellings of important terms if relevant.
            - 10-15 well-chosen tags is ideal (including the OPTIMIZED HASHTAGS).
            
            {json_emphasis}

            Please provide THREE distinct optimization variations. Return your response as a JSON list containing three objects, each following this format. I5 MUST BE PARSEABLE BY PYTHON json.loads():
            [
              {{
                  "optimized_title": "Variation 1 optimized title here",
                  "optimized_description": "Variation 1 optimized description here",
                  "optimized_tags": ["v1_tag1", "v1_tag2", ...],
                  "optimization_notes": "Explanation for Variation 1 changes",
                  "optimization_score": <score between 0.0 and 1.0>
              }},
              {{
                  "optimized_title": "Variation 2 optimized title here",
                  "optimized_description": "Variation 2 optimized description here",
                  "optimized_tags": ["v2_tag1", "v2_tag2", ...],
                  "optimization_notes": "Explanation for Variation 2 changes",
                  "optimization_score": <score between 0.0 and 1.0>
              }},
              {{
                  "optimized_title": "Variation 3 optimized title here",
                  "optimized_description": "Variation 3 optimized description here",
                  "optimized_tags": ["v3_tag1", "v3_tag2", ...],
                  "optimization_notes": "Explanation for Variation 3 changes",
                  "optimization_score": <score between 0.0 and 1.0>
              }}
            ]

            """

            tools = [
                {
                    "name": "provide_comprehensive_video_optimizations",
                    "description": "Provides three distinct sets of optimized title, description, and tags for a YouTube video, along with notes explaining the reasoning for each variation.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "optimizations": {
                                "type": "array",
                                "description": "A list containing exactly three optimization objects.",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "optimized_title": {
                                            "type": "string",
                                            "description": "The optimized video title for this variation."
                                        },
                                        "optimized_description": {
                                            "type": "string",
                                            "description": "The optimized video description for this variation."
                                        },
                                        "optimized_tags": {
                                            "type": "array",
                                            "description": "A list of optimized video tags (strings) for this variation. Each tag should be a single word or joined phrase.",
                                            "items": {"type": "string"}
                                        },
                                        "optimization_notes": {
                                            "type": "string",
                                            "description": "Detailed explanation of the changes made for this specific variation and why they will improve performance."
                                        },
                                        "optimization_score": {
                                            "type": "number",
                                            "description": "A score from 0.0 to 1.0 indicating the estimated effectiveness of this optimization variation.",
                                            "minimum": 0,
                                            "maximum": 1
                                        }
                                    },
                                    "required": ["optimized_title", "optimized_description", "optimized_tags", "optimization_notes", "optimization_score"]
                                }
                            }
                        },
                        "required": ["optimizations"]
                    }
                }
            ]

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=temperature,
                system=system_message,
                tools=tools,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            generated_optimizations = None
            for content in response.content:
                if content.type == "tool_use" and content.name == "provide_comprehensive_video_optimizations":
                    generated_optimizations = content.input
                    if generated_optimizations.get('optimizations'):
                        return generated_optimizations['optimizations']

            """
            # Parse the response
            response_text = message.content[0].text

            # Two parsing methods to try
            parsing_methods = [
                {
                    "name": "json_loads",
                    "func": lambda text: json.loads(text[text.find('['):text.rfind(']')])
                },
                {
                    "name": "json_pattern_match",
                    "func": lambda text: parse_json_with_pattern(text)
                },
                {
                    "name": "text_extraction",
                    "func": lambda text: extract_optimization_from_text(
                    text, original_title, original_description, original_tags
                )
                }
            ]

            # Try each parsing method
            last_error = None
            for method in parsing_methods:
                try:
                    logger.info(f"Trying parsing method: {method['name']}")
                    result = method["func"](response_text)

                    # --- MODIFIED VALIDATION ---
                    # Check if result is a list of 3 valid optimization dicts
                    is_list_of_3 = isinstance(result, list) and len(result) == 3
                    all_items_valid = False
                    if is_list_of_3:
                        # Check each item in the list using the original validation function
                        all_items_valid = all(is_valid_optimization_result(item) for item in result)

                    # If we got a valid list of 3 results, return it
                    if is_list_of_3 and all_items_valid:
                        logger.info(f"Successfully parsed response as list of 3 with method: {method['name']}")
                        return result # Return the list of 3 dicts
                    else:
                        # Log why validation failed
                        if not is_list_of_3:
                             logger.warning(f"Parsing method {method['name']} did not return a list of 3 items. Result type: {type(result)}, Length: {len(result) if isinstance(result, list) else 'N/A'}")
                        elif not all_items_valid:
                             logger.warning(f"Parsing method {method['name']} returned a list, but not all 3 items were valid.")
                        # Raise an error to try the next parsing method or fail the attempt
                        raise ValueError("Parsed result is not a list of 3 valid optimization objects.")
                    # --- END MODIFIED VALIDATION ---

                except Exception as e:
                    last_error = e
                    logger.warning(f"Parsing method {method['name']} failed: {e}")

            # If we got here, all parsing methods failed for this attempt
            retry_errors.append(f"Attempt {attempt+1}: All parsing methods failed. Last error: {last_error}")
            """
        except Exception as e:
            logger.error(f"Retry attempt {attempt+1} failed with error: {e}")
            retry_errors.append(f"Attempt {attempt+1}: {str(e)}")

    # If we get here, all retries failed
    logger.error(f"All {max_retries} optimization attempts failed")
    return {}

def parse_json_with_pattern(response_text: str) -> Dict:
    """Parse JSON from text with multiple fallback methods"""
    # Look for JSON pattern
    json_match = re.search(r'({[\s\S]*})', response_text)

    if not json_match:
        raise ValueError("No JSON pattern found in response")

    json_str = json_match.group(1)

    # Try multiple JSON parsing approaches
    try:
        # First attempt: Basic JSON loading
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Second attempt: Clean control characters
        logger.warning("Basic JSON parsing failed, trying with control character cleanup")
        # Replace control characters with appropriate escaped versions
        clean_json = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        # Remove other control characters
        clean_json = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1F\x7F]', '', clean_json)

        try:
            return json.loads(clean_json)
        except json.JSONDecodeError:
            # Third attempt: More aggressive cleaning
            logger.warning("Control character cleanup failed, trying more aggressive cleaning")
            # Strip all whitespace and try to fix common JSON errors
            clean_json = re.sub(r'[\s\t\n\r]+', ' ', clean_json)
            clean_json = clean_json.replace('}, }', '}}').replace('},}', '}}')
            clean_json = clean_json.replace('], }', ']}').replace(',]}', ']}')

            return json.loads(clean_json)

def check_for_placeholders(text: str) -> bool:
    """Check if text contains placeholder patterns like [Text] or {Text}"""
    placeholder_patterns = [
        r'\[([^\]]+)\]',  # [Text]
        r'\{([^}]+)\}',   # {Text}
        r'<([^>]+)>',     # <Text>
        r'\(placeholder\)',
        r'\[placeholder\]',
        r'link to',
        r'social media links',
        r'insert',
        r'add your',
        r'add a',
        r'your url',
        r'your website',
        r'your social'
    ]

    # Check each pattern
    for pattern in placeholder_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning(f"Found placeholder matching pattern '{pattern}' in text: {text[:50]}...")
            return True

    return False

def get_channel_optimization(
    channel_title: str,
    description: str = "",
    keywords: str = "",
    recent_videos_data: str = "",
    model: str = "claude-3-7-sonnet-20250219",
    max_retries: int = 3
) -> Dict:
    """
    Generate optimizations for a YouTube channel including
    description and keywords - using Claude

    Args:
        channel_title: The channel title
        description: The current channel description
        keywords: The current channel keywords
        recent_videos_data: Data about recent videos on the channel (optional)
        model: The Claude model to use
        max_retries: Maximum number of retry attempts

    Returns:
        dict: Contains optimized description, keywords and notes
    """
    if not client:
        logger.error("Anthropic client not initialized. Using fallback optimization.")
        return fallback_channel_optimization(channel_title, description, keywords)

    # Create an array to collect error messages from retry attempts
    retry_errors = []

    # Try multiple times with slightly different prompts
    for attempt in range(max_retries):
        try:
            # Log retry attempts
            if attempt > 0:
                logger.info(f"Retry attempt {attempt} for channel optimization")

            # Adjust temperature slightly on retries to get different outputs
            temperature = 0.7 + (attempt * 0.1)

            system_message = """You are a YouTube SEO expert who specializes in optimizing YouTube channels 
            for maximum discovery, engagement, and brand consistency. You understand what makes channels 
            perform well and how to optimize metadata to increase visibility, subscriber growth, and brand recognition.
            
            IMPORTANT: Your response MUST be valid JSON with properly escaped characters."""

            # Adjust the prompt slightly for each retry
            json_emphasis = ""
            if attempt > 0:
                json_emphasis = """
                EXTREMELY IMPORTANT: Format your response as VALID JSON.
                Do not include ANY control characters in strings that would break JSON parsing.
                Make sure all quotes are properly escaped and all values are valid JSON.
                """

            # Add videos section if available
            recent_videos_section = ""
            if recent_videos_data:
                recent_videos_section = f"""
                The channel has the following video samples to help you understand 
                the channel's content and theme. This includes both the most recent uploads
                AND a random selection of older videos to give you a more comprehensive view.
                Use this diverse selection to create more targeted and relevant description and keywords
                that accurately represent the full scope of the channel's content:
                
                {recent_videos_data}
                """

            prompt = f"""
            I need to optimize the following YouTube channel's branding settings:
            
            CHANNEL TITLE: {channel_title}
            
            CURRENT DESCRIPTION:
            {description if description else "No description provided"}
            
            CURRENT KEYWORDS:
            {keywords if keywords else "No keywords provided"}
            
            {recent_videos_section}
            
            Please provide an optimized version of the description and keywords, following these guidelines:
            
            For the DESCRIPTION:
            - If the current description is inadequate, missing, or doesn’t represent the channel well, create a brand-new, compelling, and scannable description based on the channel title and any recent video data provided.
            - Otherwise, analyze the current description. Identify its strengths (e.g., clear focus, engaging tone) and weaknesses (e.g., boring tone, missing keywords). Then, craft a new version that keeps the strengths and fixes the weaknesses.
            - Create a compelling, scannable channel description of at least 250 characters and at most 500 characters
            - Mix video data with the channel’s bigger niche, audience, and unique hook. Think YouTube ecosystem and what sets it apart.
            - Optimize the first 100-150 characters for search by including the main and secondary keywords.
            - Ensure the description includes:
              - A clear statement of what the channel is about.
              - Benefits for viewers (e.g., "Learn quick fitness tips").
              - Upload frequency if known (e.g., "New videos every Tuesday").
              - A call to action (e.g., "Subscribe for weekly workouts!").
            - Include calls to action, such as 'Subscribe for weekly workouts!'
            - Use formatting like line breaks and bullet points (using '*', '+', or '-') to improve readability.
            - Use emojis strategically to improve readability and engagement
            - Include links to social media or website if available
            - Analyze both recent uploads AND random older videos to help identify consistent themes across the channel's content history but do not overly focus on these videos and consider the overall channel niche and audience.
            - Check for video outliers that may not fit the channel's theme and avoid including them in the description. These instances are rare but can be important to note.
            - The description should highlight both consistent themes and the full variety of content on the channel
            - Pay special attention to where recent videos and older/random videos overlap in theme - these are likely core channel topics
            - Avoid placeholder text - the description should be ready to publish as-is
            
            For the KEYWORDS:
            - Include 5-7 highly relevant keywords totaling 50-75 characters.
            - Focus on the main topic and popular related terms.
            - Use quotes for multi-word keywords, e.g., 'Home Yoga'
            - Include misspellings of important terms if relevant.
            - Focus on specific, niche keywords rather than broad ones
            - If recent videos data is provided, extract high-value keywords from it to capture the channel's full scope.
            - Format properly with quotes and spacing (e.g. "keyword1" "keyword phrase" "keyword3")
            - Start with the most important keywords
            
            {json_emphasis}
            
            Please return your response in the following JSON format:
            {{
                "optimized_description": "Your optimized description here",
                "optimized_keywords": "Your optimized keywords here with proper quotation marks",
                "optimization_notes": "Detailed explanation of the changes made and why they will improve performance"
            }}
            """

            # Call Claude API
            message = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=temperature,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse the response
            response_text = message.content[0].text

            # Try parsing using existing methods
            try:
                # First try parsing as JSON
                result = parse_json_with_pattern(response_text)

                # Validate the result
                if is_valid_channel_optimization_result(result):
                    logger.info("Successfully parsed channel optimization response as JSON")
                    return {
                        "original_description": description,
                        "optimized_description": result.get("optimized_description", description),
                        "original_keywords": keywords,
                        "optimized_keywords": result.get("optimized_keywords", keywords),
                        "optimization_notes": result.get("optimization_notes", "No notes provided")
                    }
            except Exception as json_error:
                logger.error(f"JSON parsing failed for channel optimization: {json_error}")
                # Fall back to regex extraction

            # If JSON parsing failed, try regex extraction
            desc_match = re.search(r'optimized_description[\s:"\']+([^"\']+|[^"]+|[^\']+)', response_text, re.IGNORECASE | re.DOTALL)
            keywords_match = re.search(r'optimized_keywords[\s:"\']+([^"\']+|[^"]+|[^\']+)', response_text, re.IGNORECASE | re.DOTALL)
            notes_match = re.search(r'optimization_notes[\s:"\']+([^"\']+|[^"]+|[^\']+)', response_text, re.IGNORECASE | re.DOTALL)

            optimized_description = description
            if desc_match:
                optimized_description = desc_match.group(1).strip()

            optimized_keywords = keywords
            if keywords_match:
                optimized_keywords = keywords_match.group(1).strip()

            notes = "No notes provided"
            if notes_match:
                notes = notes_match.group(1).strip()
                # Limit length and clean up
                notes = notes[:500].replace('\n', ' ').strip()

            # Check for placeholders
            if check_for_placeholders(optimized_description) or check_for_placeholders(optimized_keywords):
                logger.warning("Found placeholders in channel optimization results, trying again")
                continue

            return {
                "original_description": description,
                "optimized_description": optimized_description,
                "original_keywords": keywords,
                "optimized_keywords": optimized_keywords,
                "optimization_notes": notes
            }

        except Exception as e:
            logger.error(f"Retry attempt {attempt+1} failed with error: {e}")
            retry_errors.append(f"Attempt {attempt+1}: {str(e)}")

    # If we get here, all retries failed
    logger.error(f"All {max_retries} channel optimization attempts failed")
    return fallback_channel_optimization(channel_title, description, keywords)

def fallback_channel_optimization(channel_title: str, description: str, keywords: str) -> Dict:
    """
    Fallback function for channel optimization if Claude API is unavailable

    Args:
        channel_title: The channel title
        description: The channel description
        keywords: The channel keywords

    Returns:
        dict: Contains optimized description and keywords with notes
    """
    optimization_notes = []
    optimized_description = description
    optimized_keywords = keywords

    # Basic description optimization
    if len(description) < 100:
        optimization_notes.append("Description is too short. Added more channel information.")
        # Add some basic improvements
        channel_title_capitalized = ' '.join(word.capitalize() for word in channel_title.split())
        optimized_description = f"Welcome to {channel_title_capitalized}! 🌟\n\n{description}\n\nSubscribe and hit the notification bell 🔔 to never miss a new upload!"

    # Ensure description has emojis
    if "🔔" not in optimized_description and "🌟" not in optimized_description:
        optimization_notes.append("Added emojis to increase engagement.")
        if "subscribe" in optimized_description.lower():
            optimized_description = optimized_description.replace("Subscribe", "Subscribe 🔔")

    # Add line breaks for better readability
    if "\n" not in optimized_description:
        optimization_notes.append("Added line breaks for better readability.")
        sentences = re.split(r'(?<=[.!?])\s+', optimized_description)
        if len(sentences) > 1:
            optimized_description = "\n\n".join(sentences)

    # Basic keywords optimization
    if not keywords or len(keywords) < 10:
        optimization_notes.append("Keywords missing or insufficient. Added basic keywords based on channel title.")
        words = channel_title.lower().split()
        # Add some basic keyword structure
        keyword_pieces = []
        for word in words:
            if len(word) > 3:  # Only use meaningful words
                keyword_pieces.append(f'"{word}"')
        if channel_title:
            keyword_pieces.append(f'"{channel_title}"')

        # Add some common YouTube keywords
        keyword_pieces.extend(['"youtube channel"', '"new videos"', '"subscribe"'])

        # Join all keywords
        optimized_keywords = " ".join(keyword_pieces)

    # Ensure keywords are in quotes
    if '"' not in optimized_keywords:
        optimization_notes.append("Added proper quotation marks to keywords.")
        words = re.split(r'\s+', optimized_keywords)
        quoted_words = []
        for word in words:
            if word and not word.startswith('"') and not word.endswith('"'):
                quoted_words.append(f'"{word}"')
            else:
                quoted_words.append(word)
        optimized_keywords = " ".join(quoted_words)

    # If no changes were made, add a note
    if not optimization_notes:
        optimization_notes.append("Channel settings appear to already follow best practices.")

    # Create a string from the list of notes
    notes_string = "\n".join(optimization_notes)

    return {
        "original_description": description,
        "optimized_description": optimized_description,
        "original_keywords": keywords,
        "optimized_keywords": optimized_keywords,
        "optimization_notes": notes_string
    }

def is_valid_channel_optimization_result(result: Dict) -> bool:
    """Check if a channel optimization result has the required fields and no placeholders"""
    required_fields = ["optimized_description", "optimized_keywords"]

    # Check that all required fields are present and non-empty
    for field in required_fields:
        if field not in result or not result[field]:
            logger.warning(f"Missing or empty required field: {field}")
            return False

    # Check for placeholders in description and keywords
    if check_for_placeholders(result.get("optimized_description", "")):
        logger.warning("Found placeholders in optimized_description")
        return False

    if check_for_placeholders(result.get("optimized_keywords", "")):
        logger.warning("Found placeholders in optimized_keywords")
        return False

    return True

def is_valid_optimization_result(result: Dict) -> bool:
    """Check if an optimization result has the required fields and no placeholders"""
    required_fields = ["optimized_title", "optimized_description", "optimized_tags"]

    # Check that all required fields are present and non-empty
    for field in required_fields:
        if field not in result or not result[field]:
            logger.warning(f"Missing or empty required field: {field}")
            return False

    # Check that optimized_tags is a list
    if not isinstance(result.get("optimized_tags", []), list):
        logger.warning("optimized_tags is not a list")
        return False

    # Check for placeholders in title and description
    if check_for_placeholders(result.get("optimized_title", "")):
        logger.warning("Found placeholders in optimized_title")
        return False

    if check_for_placeholders(result.get("optimized_description", "")):
        logger.warning("Found placeholders in optimized_description")
        return False

    return True


async def should_optimize_video(
        video_data: Dict[str, Any],
        channel_subscriber_count: int,
        analytics_data: Dict[str, Any],
        past_optimizations: List[Dict[str, Any]] = [],
        model: str = "claude-3-7-sonnet-20250219"
) -> Dict[str, Any]:
    """
    Uses LLM to determine if a video should be optimized based on performance metrics

    Args:
        video_data: Dictionary containing video information (title, description, tags, view_count)
        channel_subscriber_count: Number of subscribers for the channel
        analytics_data: Dictionary containing analytics metrics for the video
        past_optimizations: List of previous optimizations for this video

    Returns:
        dict: Contains decision and reasoning
            {
                "should_optimize": bool,
                "reasons": str,
                "confidence": float
            }
    """
    logger.info(f"Evaluating if video '{video_data.get('title', 'Unknown')}' should be optimized")

    if past_optimizations:
        #TODO: Integrate past optimizations analysis
        past_optimizations_analysis = analyze_past_optimizations(past_optimizations, video_data, channel_subscriber_count, analytics_data)

    past_optimizations_prompt = '\n'.join([
        (lambda o:
         f"- Date: {o.get('created_at', 'Unknown')}\n"
         f"  Title: {o.get('optimized_title', 'Not changed')}\n"
         f"  Description: {o.get('optimized_description', 'Not changed')}\n"
         f"  Tags: {', '.join(o.get('optimized_tags', [])) if isinstance(o.get('optimized_tags'), list) else 'Not changed'}"
         )(opt)
        for opt in past_optimizations
    ]) if past_optimizations else 'No previous optimizations'

    recent_performance_data_prompt = '\n'.join([
        f"{day['day']}: {day['views']} views, {day['averageViewPercentage']}% watched"
        for day in analytics_data.get('data_points', [])
    ])

    system_message = """You are a YouTube SEO expert AI assistant. Your task is to analyze video performance data and decide if the video's metadata (title, description, tags) should be optimized to improve performance. Provide a clear 'YES' or 'NO' decision, a brief reasoning based *only* on the provided data, and a confidence score between 0.0 and 1.0. Respond strictly in JSON format."""

    prompt = f"""
    Analyze the video data provided and respond with only "YES" or "NO" regarding whether 
    the video should be optimized based on its current performance and metadata quality.

    Consider the following factors:

    VIDEO INFORMATION:
    - Title: {video_data.get('title', 'N/A')}
    - Description: {video_data.get('description', 'N/A')}
    - Tags: {', '.join(video_data.get('tags', [])) if video_data.get('tags') else 'No tags'}
    - View Count: {video_data.get('view_count', 0)}
    - Published At: {video_data.get('published_at', 'Unknown')}

    CHANNEL INFORMATION:
    - Subscriber Count: {channel_subscriber_count}

    ENGAGEMENT METRICS:
    - Likes: {video_data.get('like_count', 0)}
    - Comments: {video_data.get('comments', 0)}
    - Recent Performance: {recent_performance_data_prompt if recent_performance_data_prompt else 'No recent performance data'}

    PAST OPTIMIZATIONS:
    {
    past_optimizations_prompt if past_optimizations else 'No previous optimizations'
    }

    KEY DECISION FACTORS:
    1. Is the video underperforming relative to the channel's typical performance?
    2. Are there clear SEO issues with the title, description or tags?
    3. Could the video benefit from updated metadata based on current trends?
    4. If the video was recently optimized, has it been enough time to evaluate performance?
    5. Does the engagement rate (likes/views, comments/views) suggest improvements are needed?

    Your response MUST be in the following JSON format:
    [
        {{
            "decision": "YES" or "NO",
            "reasoning": "Reasoning for decision",
            "confidence": 0.0 to 1.0
        }}
    ]
    """

    for _ in range(3):

        try:
            logger.info("Calling Claude API to determine optimization need")
            message = client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.5,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse the response
            response_text = message.content[0].text
            logger.debug(f"Claude response for optimization check: {response_text}")

            # Look for JSON array pattern (e.g., [{...}]) as the primary target
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                json_str = json_match.group(0)
                # Attempt to parse the found string as a JSON array
                response_list = json.loads(json_str)

                # Check if the list is not empty and contains at least one dictionary
                if isinstance(response_list, list) and len(response_list) > 0 and isinstance(response_list[0], dict):
                    response_data = response_list[0]  # Get the first object from the array
                    decision = response_data.get("decision", "NO").upper()
                    should_optimize = decision == "YES"
                    reasons = response_data.get("reasoning", "LLM provided no reasoning.")
                    confidence = float(response_data.get("confidence", 0.5))  # Ensure confidence is float
                    logger.info(f"Claude decision (from array): {decision}, Confidence: {confidence}, Reasoning: {reasons}")
                else:
                    # Log an error if the structure is not the expected list of dicts
                    logger.error(f"Parsed JSON array is not valid or empty: {json_str}")
                    raise ValueError("Parsed JSON array is not valid or empty.")

            else:
                # Fallback: If no JSON array is found, attempt to find a JSON object pattern
                logger.warning("No JSON array pattern found in response, attempting to find JSON object pattern...")
                json_match_obj = re.search(r'{[\s\S]*}', response_text)
                if json_match_obj:
                    json_str_obj = json_match_obj.group(0)
                    response_data = json.loads(json_str_obj)  # Assume it's just the object
                    # Ensure the parsed data is a dictionary
                    if isinstance(response_data, dict):
                        decision = response_data.get("decision", "NO").upper()
                        should_optimize = decision == "YES"
                        reasons = response_data.get("reasoning", "LLM provided no reasoning.")
                        confidence = float(response_data.get("confidence", 0.5))  # Ensure confidence is float
                        logger.info(f"Claude decision (from object fallback): {decision}, Confidence: {confidence}, Reasoning: {reasons}")
                    else:
                        logger.error(f"Fallback JSON object pattern parsed but is not a dictionary: {json_str_obj}")
                        raise ValueError("Fallback JSON object pattern parsed but is not a dictionary.")
                else:
                    # Log an error if neither pattern is found
                    logger.error(f"No JSON pattern (array or object) found in response: {response_text}")
                    raise ValueError("No JSON pattern (array or object) found in response.")

            # Return statement remains the same
            return {
                "should_optimize": should_optimize,
                "reasons": reasons,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error determining if video should be optimized via Claude: {str(e)}")

    return {
        "should_optimize": False,
        "reasons": f"Error during optimization evaluation",
        "confidence": 0.0
    }

def analyze_past_optimizations(
        past_optimizations: List[Dict[str, Any]],
        video_data: Dict[str, Any],
        channel_subscriber_count: int,
        analytics_data: Dict[str, Any]
) -> None:
    """
    Analyze past optimizations to determine if they were effective

    Args:
        past_optimizations: List of previous optimizations for this video
        video_data: Dictionary containing video information (title, description, tags)
        channel_subscriber_count: Number of subscribers for the channel
        analytics_data: Dictionary containing analytics metrics for the video
    """
    # Placeholder for future analysis logic
    pass

def generate_search_query(title: str, description: str, tags: list[str], category_name: str) -> str:
    """
    Generates a concise YouTube search query using Claude Haiku based on video details,
    focusing on the general niche within the category.

    Args:
        title: The video title.
        description: The video description.
        tags: A list of video tags.
        category_name: The name of the video's category (e.g., "Gaming", "How-to & Style").

    Returns:
        A concise, general search query string for the niche, or an empty string if generation fails.
    """
    if not client:
        print("Anthropic client not available.")
        return ""

    tags_string = ", ".join(tags) if tags else "None"

    prompt = f"""Based on the details of the following YouTube video, generate a general search query (3-7 words) to find best-performing competitor videos within the same niche and category ({category_name}). The query should represent the broader topic area, not just the specifics of this single video. Output ONLY the search query string.

    Video Category: {category_name}
    Video Title: {title}
    Video Description: {description}
    Video Tags: {tags_string}
    
    ---
    Example 1:
    Input Category: Gaming
    Input Title: My Best Apex Legends Win Yet! Clutch Moments
    Input Description: Watch me dominate in Apex Legends season 20. Solo queue madness.
    Input Tags: Apex Legends, FPS, Battle Royale, Gaming, Clutch, Win
    Output Query: popular battle royale gameplay
    
    Example 2:
    Input Category: How-to & Style
    Input Title: Easy Sourdough Bread Recipe for Beginners
    Input Description: Step-by-step guide to making delicious sourdough bread at home. No prior experience needed!
    Input Tags: Sourdough, Baking, Recipe, How-to, Bread, Cooking, DIY
    Output Query: beginner baking tutorials
    
    Example 3:
    Input Category: Science & Technology
    Input Title: Explaining Quantum Entanglement Simply
    Input Description: A clear explanation of quantum entanglement and its implications. Physics concepts made easy.
    Input Tags: Quantum Physics, Entanglement, Science, Education, Physics, Technology
    Output Query: science education explainer videos
    ---
    
    Now, generate the query for the provided video details:
    Category: {category_name}
    Title: {title}
    Description: {description}
    Tags: {tags_string}
    
    Output Query:"""

    try:
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=50,
            temperature=0.6,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        if message.content and isinstance(message.content, list) and len(message.content) > 0:
             query = message.content[0].text.strip().strip('\"\'')
             print(f"Generated general YouTube search query: {query}")
             return query
        else:
             print("Claude Haiku response format not as expected.")
             return ""

    except Exception as e:
        print(f"An unexpected error occurred during general search query generation: {e}")
        return ""


def extract_keywords_with_llm(
        title: str,
        description: str,
        transcript: str = "",
        tags: list = None,
        min_keywords: int = 3,
        max_keywords: int = 5
) -> list[str]:
    """
    Uses an LLM to extract the most relevant keywords from video content.

    Args:
        client: Anthropic client
        title: Video title
        description: Video description
        transcript: Video transcript (optional)
        tags: Existing video tags (optional)
        min_keywords: Minimum number of keywords to extract
        max_keywords: Maximum number of keywords to extract

    Returns:
        List of relevant keywords as strings
    """
    tags_str = ", ".join(tags) if tags else "No tags"

    # Format transcript for prompt inclusion
    transcript_section = ""
    if transcript and len(transcript) > 0:
        # If transcript is very long, truncate it
        if len(transcript) > 2000:
            transcript_text = transcript[:1500] + "... [transcript truncated]"
        else:
            transcript_text = transcript
        transcript_section = f"VIDEO TRANSCRIPT:\n{transcript_text}"

    system_message = """You are an expert YouTube SEO and competitor analyst. 
    Your critical task is to extract 1-2 word topical keywords from video content. These keywords are essential for researching competitor videos using Google and YouTube Trends. 
    The primary goal is to identify terms that define the video's broad niche or category.
    
    Key constraints for keywords:
    - Length: Strictly 1 or 2 words. NO THREE-WORD OR LONGER PHRASES.
    - Generality: Keywords must be general enough to have search volume on Google/YouTube Trends. Avoid hyper-specific terms.
    - Purpose: To find a range of competitor videos in the same general content area.
    
    Provide ONLY the keywords as a comma-separated list, sorted from most to least optimal. No other text."""

    prompt = f"""
    Analyze the provided YouTube video content and extract {min_keywords}-{max_keywords} highly concise (1-2 words ONLY) SEO keywords. These keywords are for discovering competitor videos within the same niche by searching Google/YouTube Trends.
    
    Critical Requirements for Keywords:
    1.  Niche/Category Identifier: Each keyword (or two-word pair) must represent the video's main category or primary subject niche. Think about what broad terms someone would search for to find similar types of content.
    2.  Trend-Searchable: Keywords MUST be 1-2 words. This is vital for getting results in Google/YouTube Trends. Longer phrases will likely yield no data.
    3.  Competitor Discovery Focus: The aim is to find *other channels and videos* operating in the same content space.
    
    Keyword Format and Length:
    -   Strictly 1 or 2 words per keyword.
    -   If a concept seems to need more, find a more general 1-2 word term that encompasses it.
    
    Examples to Guide You:
    
    *   Video Content: "A detailed tutorial on how to bake a sourdough bread starter from scratch, covering flour types, hydration, and feeding schedules."
        *   GOOD Keywords (1-2 words, category-focused): `Sourdough Baking`, `Bread Making`, `Baking Tutorial`, `Home Baking`
        *   BAD Keywords (Too specific, too long): `sourdough bread starter from scratch`, `flour types for sourdough`, `hydration and feeding schedules for baking`
    
    *   Video Content: "Gameplay of the final boss battle in the Elden Ring: Shadow of the Erdtree DLC, showing strategies for defeating Radahn."
        *   GOOD Keywords (1-2 words, category-focused): `Elden Ring`, `Gaming DLC`, `Boss Guide`, `Action RPG`
        *   BAD Keywords (Too specific, too long): `Elden Ring Shadow of the Erdtree final boss`, `strategies for defeating Radahn Elden Ring`, `Radahn boss battle gameplay`
    
    Video Details for Analysis:
    VIDEO TITLE: {title}
    VIDEO DESCRIPTION:
    {description}
    EXISTING TAGS: {tags_str}
    {transcript_section}
    
    Return ONLY a comma-separated list of {min_keywords}-{max_keywords} keywords (each 1-2 words), sorted from most to least optimal for competitor and trends research. NO other text.
    """

    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=100,
            temperature=0.3,  # Low temperature for consistency
            system=system_message,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract response text and clean it
        if response.content and len(response.content) > 0:
            keywords_text = response.content[0].text.strip()
            # Split by commas and strip whitespace from each keyword
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            # Filter out empty strings or very short keywords
            keywords = [kw for kw in keywords if len(kw) > 2]

            logger.info(f"Extracted keywords via LLM: {keywords}")
            return keywords[:max_keywords]  # Ensure we don't exceed max_keywords

        return []

    except Exception as e:
        logger.error(f"Error extracting keywords with LLM: {e}")
        # Fall back to frequency-based extraction if LLM fails
        return []


def find_relevant_topic_id(
    title: str,
    description: str = "",
    transcript: str = "",
    tags: Optional[List[str]] = None,
    category_name: str = "",
    model: str = "claude-3-5-haiku-20241022"
) -> Dict[str, Any]:
    """
    Identifies the most relevant YouTube topic ID for a video based on its content.

    Args:
        title: The video title
        description: The video description (optional)
        transcript: The video transcript (optional)
        tags: List of video tags (optional)
        model: The Claude model to use

    Returns:
        dict: Contains the identified topic_id, name, parent_topic, and confidence score
    """
    if not client:
        logger.error("Anthropic client not initialized for topic identification.")
        return {
            "topic_id": None,
            "topic_name": None,
            "parent_topic": None,
            "confidence": 0,
            "notes": "Unable to identify topic: Anthropic client not initialized."
        }

    # Format tags for prompt
    tags_str = ", ".join(tags) if tags else "No tags available"

    # Truncate transcript if it's too long
    truncated_transcript = ""
    if transcript:
        max_transcript_length = 4000  # Characters
        truncated_transcript = (transcript[:max_transcript_length] + "...") if len(transcript) > max_transcript_length else transcript

    # YouTube topic IDs organized by category
    topic_data = """
    Music topics:
    /m/04rlf - Music (parent topic)
    /m/02mscn - Christian music
    /m/0ggq0m - Classical music
    /m/01lyv - Country
    /m/02lkt - Electronic music
    /m/0glt670 - Hip hop music
    /m/05rwpb - Independent music
    /m/03_d0 - Jazz
    /m/028sqc - Music of Asia
    /m/0g293 - Music of Latin America
    /m/064t9 - Pop music
    /m/06cqb - Reggae
    /m/06j6l - Rhythm and blues
    /m/06by7 - Rock music
    /m/0gywn - Soul music
    
    Gaming topics:
    /m/0bzvm2 - Gaming (parent topic)
    /m/025zzc - Action game
    /m/02ntfj - Action-adventure game
    /m/0b1vjn - Casual game
    /m/02hygl - Music video game
    /m/04q1x3q - Puzzle video game
    /m/01sjng - Racing video game
    /m/0403l3g - Role-playing video game
    /m/021bp2 - Simulation video game
    /m/022dc6 - Sports game
    /m/03hf_rm - Strategy video game
    
    Sports topics:
    /m/06ntj - Sports (parent topic)
    /m/0jm_ - American football
    /m/018jz - Baseball
    /m/018w8 - Basketball
    /m/01cgz - Boxing
    /m/09xp_ - Cricket
    /m/02vx4 - Football
    /m/037hz - Golf
    /m/03tmr - Ice hockey
    /m/01h7lh - Mixed martial arts
    /m/0410tth - Motorsport
    /m/07bs0 - Tennis
    /m/07_53 - Volleyball
    
    Entertainment topics:
    /m/02jjt - Entertainment (parent topic)
    /m/09kqc - Humor
    /m/02vxn - Movies
    /m/05qjc - Performing arts
    /m/066wd - Professional wrestling
    /m/0f2f9 - TV shows
    
    Lifestyle topics:
    /m/019_rr - Lifestyle (parent topic)
    /m/032tl - Fashion
    /m/027x7n - Fitness
    /m/02wbm - Food
    /m/03glg - Hobby
    /m/068hy - Pets
    /m/041xxh - Physical attractiveness [Beauty]
    /m/07c1v - Technology
    /m/07bxq - Tourism
    /m/07yv9 - Vehicles
    
    Society topics:
    /m/098wr - Society (parent topic)
    /m/09s1f - Business
    /m/0kt51 - Health
    /m/01h6rj - Military
    /m/05qt0 - Politics
    /m/06bvp - Religion
    
    Other topics:
    /m/01k8wb - Knowledge
    """

    system_prompt = """
    You are a YouTube content categorization expert. Your task is to analyze video content 
    and identify the most relevant YouTube topic ID that matches the video's subject matter.
    
    You will be presented with:
    1. A video title
    2. A video description
    3. Video tags
    4. Possibly a video transcript
    5. A video category name
    
    Your goal is to determine which ONE topic ID from YouTube's standardized list best represents 
    the video content. You should consider all available information but prioritize the most 
    distinctive and definitive content signals.
    
    IMPORTANT SELECTION GUIDELINES:
    - If the content clearly fits a specific subtopic, choose that subtopic's ID.
    - If the content spans multiple subtopics within a category, prefer the parent topic ID.
    - If you're uncertain between multiple specific subtopics, it's better to select the parent topic.
    - Parent topics often provide better discoverability when content doesn't fit neatly into a subtopic.
    
    You must return your analysis in JSON format with:
    - The single most relevant topic_id
    - The corresponding topic_name
    - The parent_topic category
    - A confidence score from 0-100 (how certain you are of the match)
    - Brief notes explaining your reasoning
    """

    user_prompt = f"""
    Please analyze this YouTube video content and determine the most relevant topic ID:
    
    VIDEO TITLE: {title}

    VIDEO DESCRIPTION: 
    {description}
    
    VIDEO TAGS: {tags_str}
    
    {"VIDEO TRANSCRIPT EXCERPT: " + truncated_transcript if truncated_transcript else "No transcript available"}
    
    VIDEO CATEGORY: {category_name}
    
    AVAILABLE YOUTUBE TOPIC IDs:
    {topic_data}
    
    Return your analysis as valid JSON with the following structure:
    {{
        "topic_id": "/m/XXXXX",  
        "topic_name": "Name of the specific topic", 
        "parent_topic": "Name of the parent category",
        "confidence": 85,  
        "notes": "Brief explanation of why this topic was chosen"
    }}
    
    Remember: If the content spans multiple subtopics or you're uncertain between specific subtopics, 
    prefer using the parent topic ID (e.g., "/m/04rlf" for Music rather than a specific music genre).
    
    Don't include any text outside the JSON. Choose only ONE topic ID, not multiple.
    """
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Parse the response
        response_text = message.content[0].text

        # Extract JSON from response
        json_match = re.search(r'({[\s\S]*})', response_text)

        if json_match:
            try:
                json_str = json_match.group(1)
                response_data = json.loads(json_str)

                # Validate and return the result
                return {
                    "topic_id": response_data.get("topic_id"),
                    "topic_name": response_data.get("topic_name"),
                    "parent_topic": response_data.get("parent_topic"),
                    "confidence": response_data.get("confidence", 0),
                    "notes": response_data.get("notes", "No explanation provided")
                }
            except Exception as json_error:
                logger.error(f"JSON parsing failed in topic identification: {json_error}")

        # Fallback if JSON parsing fails
        return {
            "topic_id": None,
            "topic_name": None,
            "parent_topic": None,
            "confidence": 0,
            "notes": f"Failed to parse topic identification response: {response_text[:100]}..."
        }

    except Exception as e:
        logger.error(f"Error in topic identification: {e}")
        return {
            "topic_id": None,
            "topic_name": None,
            "parent_topic": None,
            "confidence": 0,
            "notes": f"Error during topic identification: {str(e)}"
        }

def extract_competitor_keywords(top_competitor_videos, video_data, max_keywords=5, model="claude-3-7-sonnet-20250219",top_n_competitors:int = 5):
    """
    Analyzes top-performing competitor videos in relation to the user's video to extract
    the most relevant and effective keywords.
    
    Args:
        top_competitor_videos: List of top-performing competitor videos with title, description, tags, etc.
        video_data: Dictionary containing the user's video information (title, description, tags)
        max_keywords: Maximum number of keywords to return
        model: The Claude model to use
        
    Returns:
        dict: Contains a list of keywords and explanation of their effectiveness
    """
    if not client:
        logger.error("Anthropic client not initialized for competitor keyword extraction.")
        return {
            "keywords": [],
            "explanation": "Unable to extract keywords: Anthropic client not initialized."
        }
        
    # Format the user's video data
    user_title = video_data.get('title', 'No title')
    user_description = video_data.get('description', 'No description')[:300] + "..." if video_data.get('description', '') else 'No description'
    user_tags = ", ".join(video_data.get('tags', [])) if video_data.get('tags') else "No tags"
    
    # Format the competitor video data for the prompt
    competitor_content = ""
    for idx, video in enumerate(top_competitor_videos[:top_n_competitors], 1):  # Limit to 5 videos max
        title = video.get('title', 'No title')
        description = video.get('description', 'No description')[:300] + "..." if video.get('description', '') else 'No description'
        tags = ", ".join(video.get('tags', [])) if video.get('tags') else "No tags"
        view_count = video.get('view_count', 0)
        views_per_day = video.get('views_per_day', 0)
        engagement_rate = video.get('engagement_rate', 0)
        performance_score = video.get('performance_score', 0)
        
        competitor_content += f"""
        VIDEO {idx}:
        Title: {title}
        Description excerpt: {description}
        Tags: {tags}
        Views: {view_count:,}
        Views per day: {views_per_day:,.2f}
        Engagement rate: {engagement_rate:.2f}%
        Performance score: {performance_score:.2f}
        
        """
    
    system_message = """
    You are a YouTube keyword analysis expert. Your task is to analyze high-performing competitor videos
    in relation to a specific user video, and identify the most effective keywords that could improve 
    the user's video performance. The goal is to find keywords suitable for further research in Google/YouTube Trends.
    
    You should:
    1. Identify patterns and common themes across the successful competitor videos.
    2. Extract keywords that appear to drive engagement and views.
    3. Consider both explicit keywords (in tags) and implicit keywords (in titles/descriptions).
    4. Focus on keywords that are specific enough to target a niche but broad enough to capture audience interest and be searchable in trend tools. Avoid overly generic (e.g., \"video\") or excessively niche/long-tail keywords.
    5. Prioritize keywords that appear in multiple successful videos.
    6. Consider the relevance of these keywords to the user's specific content.
    7. Identify keyword gaps and opportunities that the user could leverage.
    8. ABSOLUTELY DO NOT include hashtag symbols (#) in the keywords.
    """
    
    prompt = f"""
    I need you to analyze high-performing competitor YouTube videos in relation to a specific user video, 
    and identify the most effective keywords that could improve the user's video performance. These keywords will be used for further research in Google/YouTube Trends.
    
    USER'S VIDEO:
    Title: {user_title}
    Description excerpt: {user_description}
    Tags: {user_tags}
    
    TOP COMPETITOR VIDEOS:
    {competitor_content}
    
    Please analyze both the user's content and the competitor videos to:
    1. Identify 3-5 highly effective keywords or keyword phrases (2-4 words ideally) that would work well for the user's video. These keywords should be suitable for searching in trend analysis tools – not too broad, not too specific.
    2. Explain why these keywords are effective in this niche.
    3. Explain how these keywords relate to both the top-performing competitors and the user's content.
    4. Note any keyword gaps or opportunities the user could exploit.
    5. Ensure NO hashtag symbols (#) are included in the keywords.
    
    Return your analysis as JSON with:
    {{
        "keywords": ["keyword phrase 1", "keyword phrase 2", "keyword phrase 3"],
        "explanation": "These keywords are effective because...",
        "relevance": "How these keywords relate to the user's content...",
        "opportunities": "Keyword gaps or opportunities the user could exploit..."
    }}
    """
    
    try:
        # Call the Claude API
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.3,
            system=system_message,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse the response
        if response.content and len(response.content) > 0:
            response_text = response.content[0].text.strip()
            
            # Extract JSON
            result = parse_json_with_pattern(response_text)
            
            if result and "keywords" in result and isinstance(result["keywords"], list):
                # Ensure we don't exceed max_keywords and remove any hashtags
                processed_keywords = []
                for kw in result["keywords"]:
                    if isinstance(kw, str):
                        processed_keywords.append(kw.replace("#", "").strip())
                
                result["keywords"] = [kw for kw in processed_keywords if kw][:max_keywords]
                logger.info(f"Extracted competitor keywords relevant to user video: {result['keywords']}")
                return result
        
        # Fallback if parsing fails
        logger.warning("Couldn't parse competitor keywords from model response")
        return {
            "keywords": [],
            "explanation": "Failed to extract keywords from competitor analysis."
        }
    
    except Exception as e:
        logger.error(f"Error extracting competitor keywords: {e}")
        return {
            "keywords": [],
            "explanation": f"Error extracting keywords: {str(e)}"
        }

def filter_generic_keywords_with_llm(
    keywords: List[str],
    video_data: dict,
    model: str = "claude-3-7-sonnet-20250219",
) -> List[str]:
    """
    Uses Claude to identify and filter out keywords that:
    1. Are too generic or would make poor hashtags
    2. Are irrelevant to the video's content (if video_data is provided)
    
    Args:
        keywords: List of keyword strings to evaluate
        model: The Claude model to use
        video_data: Optional dictionary containing video content (title, description, tags, transcript, category_name)
        
    Returns:
        List of keywords with generic and/or irrelevant ones filtered out
    """
    if not client or not keywords:
        logger.warning("No Anthropic client or empty keywords list - skipping LLM keyword filtering")
        return keywords
        
    try:
        title = video_data.get('title', '')
        description = video_data.get('description', '')
        tags = video_data.get('tags', [])
        tags_str = ", ".join(tags) if isinstance(tags, list) else tags
        transcript_excerpt = video_data.get('transcript', '')[:500]
        category = video_data.get('category_name', '')

        video_content_prompt_section = f"""
        
        Here is information about the video content. Please ensure the keywords you select are relevant to this content:
        VIDEO TITLE: {title}
        VIDEO DESCRIPTION: {description}
        VIDEO TAGS: {tags_str}
        VIDEO CATEGORY: {category}
        {f"VIDEO TRANSCRIPT EXCERPT: {transcript_excerpt}" if transcript_excerpt else ""}
        """

        prompt = f"""
        You are a YouTube SEO expert. I need your help to identify which of these potential hashtags should be REMOVED because they are:
        1. Too generic (like "video" or "trending")
        2. Platform-specific but not content-specific (like "youtube" or "shorts")
        3. Common engagement terms (like "subscribe" or "comment")
        4. Vague descriptors that don't help categorize content (like "amazing" or "best")
        5. IRRELEVANT TO THE VIDEO CONTENT (this is very important)
        {video_content_prompt_section}

        Keywords to evaluate:
        {keywords}

        Please respond with ONLY a JSON array of the keywords that should be REMOVED. Be strict and aggressive in your filtering. Pay special attention to keywords that have no clear connection to the video's content or topic.
        """
        
        # Make the API call to Claude
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        if not response or not response.content:
            logger.error("Empty response from Anthropic API")
            return keywords
            
        response_text = response.content[0].text if response.content else ""
        
        try:
            json_pattern = r'\[[\s\S]*?\]'
            json_match = re.search(json_pattern, response_text)
            
            if json_match:
                removed_keywords = json.loads(json_match.group(0))
                log_message_prefix = "LLM identified"
                if video_data:
                    log_message_prefix += f" {len(removed_keywords)} generic or irrelevant keywords to remove: {removed_keywords}"
                else:
                    log_message_prefix += f" {len(removed_keywords)} generic keywords to remove: {removed_keywords}"
                logger.info(log_message_prefix)
                
                filtered_keywords = [kw for kw in keywords if kw not in removed_keywords]
                
                logger.info(f"Filtered from {len(keywords)} to {len(filtered_keywords)} keywords")
                return filtered_keywords
            else:
                logger.warning("Could not extract JSON array from LLM response for keyword filtering. Response: " + response_text[:200])
                return keywords
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response for keyword filtering: {e}. Response: " + response_text[:200])
            return keywords
            
    except Exception as e:
        logger.error(f"Error in LLM keyword filtering: {e}")
        return keywords
