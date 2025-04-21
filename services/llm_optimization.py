import os
import logging
import anthropic
from dotenv import load_dotenv
import json
import re
from typing import Dict, List, Optional

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

def get_llm_optimized_title(
    original_title: str, 
    description: str = "", 
    tags: list = None,
    model: str = "claude-3-7-sonnet-20250219"
):
    """
    Generate an optimized YouTube title using Claude 3.7
    
    Args:
        original_title: The original video title
        description: The video description (optional)
        tags: List of video tags (optional)
        model: The Claude model to use (default: claude-3-7-sonnet-20250219)
        
    Returns:
        dict: Contains original_title, optimized_title, and optimization_notes
    """
    if not client:
        logger.error("Anthropic client not initialized. Using fallback optimization.")
        return fallback_title_optimization(original_title, tags or [])
    
    # Prepare tags string if tags were provided
    tags_str = ", ".join(tags) if tags else "No tags available"
    
    try:
        system_message = "You are a YouTube SEO expert who specializes in optimizing video titles for maximum engagement and searchability. Your goal is to create titles that drive clicks while maintaining the core meaning."

        prompt = f"""
        Original Video Title: {original_title}

        {f"Video Description: {description}" if description else "No description provided"} 

        {f"Video Tags: {tags_str}" if tags_str else "No tags provided"}

        Please create an optimized version of the title that:
        1. Is between 40-60 characters long (YouTube's sweet spot)
        2. Includes high-value keywords
        3. Creates curiosity and emotional appeal
        4. Maintains the original meaning and intent
        5. Uses strategic capitalization and symbols if appropriate (like brackets, emojis, etc.)
        6. Is formatted for optimal CTR (click-through rate)
        7. Strongly consider using hashtags in title as they may help with discoverability with YouTube's algorithm. You may consider keeping some of the original title hashtags if you believe they are optimal for discoverability or select your own.
        8. IMPORTANT: The optimized title must be ready to use exactly as returned - do not include any placeholders like [keyword] or [topic]
        
        Please return your response in the following JSON format:
        {{
            "optimized_title": "Your optimized title here",
            "optimization_notes": "Detailed explanation of the changes made and why they will improve performance"
        }}
        """

        # Call the Claude API
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.7,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response
        response_text = message.content[0].text
        
        # Look for JSON pattern
        json_match = re.search(r'({[\s\S]*})', response_text)
        
        if json_match:
            try:
                json_str = json_match.group(1)
                
                # First attempt: Basic JSON loading
                try:
                    response_data = json.loads(json_str)
                except json.JSONDecodeError:
                    # Second attempt: Clean control characters
                    logger.warning("Basic JSON parsing failed, trying with control character cleanup")
                    # Replace control characters with appropriate escaped versions
                    clean_json = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    # Remove other control characters
                    clean_json = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1F\x7F]', '', clean_json)
                    response_data = json.loads(clean_json)
                
                return {
                    "original_title": original_title,
                    "optimized_title": response_data.get("optimized_title", original_title),
                    "optimization_notes": response_data.get("optimization_notes", "No notes provided")
                }
            except Exception as json_error:
                logger.error(f"JSON parsing failed: {json_error}")
                # Fall back to regex extraction
        
        # If JSON parsing failed or no JSON was found, extract with regex
        title_match = re.search(r'optimized_title[\s:"\']+([^\n"\']+)', response_text, re.IGNORECASE)
        notes_match = re.search(r'optimization_notes[\s:"\']+([^"\']+|[^"]+|[^\']+)', response_text, re.IGNORECASE | re.DOTALL)
        
        notes = ""
        if notes_match:
            notes = notes_match.group(1).strip()
            # Limit length and clean up
            notes = notes[:500].replace('\n', ' ').strip()
        
        return {
            "original_title": original_title,
            "optimized_title": title_match.group(1).strip() if title_match else original_title,
            "optimization_notes": notes if notes else "No notes provided"
        }
            
    except Exception as e:
        logger.error(f"Error in Claude title optimization: {e}")
        return fallback_title_optimization(original_title, tags or [])

def fallback_title_optimization(original_title: str, tags: list = None):
    """
    Fallback function for title optimization if Claude API is unavailable
    
    Args:
        original_title: The original video title
        tags: List of video tags (optional)
        
    Returns:
        dict: Contains original_title, optimized_title, and optimization_notes
    """
    tags = tags or []
    optimized_title = original_title
    optimization_notes = []
    
    # Check length
    if len(original_title) < 40:
        optimization_notes.append("Title is too short. Added more relevant keywords.")
        
        # Add relevant keywords from tags
        for tag in tags[:2]:  # Use first two tags
            if tag.lower() not in original_title.lower() and len(optimized_title) + len(tag) + 3 <= 60:
                optimized_title += f" - {tag}"
                
    elif len(original_title) > 60:
        optimized_title = original_title[:57] + "..."
        optimization_notes.append("Title is too long. Shortened to fit YouTube's optimal length.")
        
    # Check for brackets/parentheses for emphasis
    if "[" not in original_title and "(" not in original_title:
        # Find a keyword from tags to emphasize
        for tag in tags:
            if tag.lower() in optimized_title.lower() and len(tag) > 3:
                # Don't make it too long
                if len(optimized_title) + 2 <= 60:
                    optimized_title = optimized_title.replace(tag, f"[{tag}]")
                    optimization_notes.append(f"Added emphasis to keyword '{tag}'.")
                break
    
    # Capitalize first letter of each word for better appearance
    if not any(x.isupper() for x in optimized_title):
        optimized_title = ' '.join(word.capitalize() for word in optimized_title.split())
        optimization_notes.append("Added capitalization for better appearance.")
        
    # If no changes were made, add a note
    if not optimization_notes:
        optimization_notes.append("Title appears to already follow best practices.")
        
    # Create a string from the list of notes
    notes_string = "\n".join(optimization_notes)
    
    return {
        "original_title": original_title,
        "optimized_title": optimized_title,
        "optimization_notes": notes_string
    }
    
def extract_optimization_from_text(response_text: str, original_title: str, original_description: str, original_tags: Optional[List[str]]) -> Dict:
    """
    Attempt to extract optimization data from raw text when JSON parsing fails
    
    Args:
        response_text: The raw response from Claude
        original_title: The original video title
        original_description: The original video description
        original_tags: The original video tags
        
    Returns:
        dict: Best effort extraction of optimized content
    """
    # Try to extract title
    title_match = re.search(r'optimized_title[\s:"\']+([^\n"\']+)', response_text, re.IGNORECASE)
    optimized_title = title_match.group(1).strip() if title_match else original_title
    
    # Try to extract description
    desc_start = re.search(r'optimized_description[\s:"\']+', response_text, re.IGNORECASE)
    desc_end = re.search(r'optimized_tags|tags[\s:"\']+\[', response_text, re.IGNORECASE)
    
    optimized_description = original_description
    if desc_start and desc_end:
        start_pos = desc_start.end()
        end_pos = desc_end.start()
        if start_pos < end_pos:
            extracted_desc = response_text[start_pos:end_pos].strip()
            if extracted_desc and len(extracted_desc) > 20:  # Sanity check
                optimized_description = extracted_desc.strip('"\'')
    
    # Try to extract tags
    tags_match = re.search(r'optimized_tags[\s:"\']+\[(.*?)\]', response_text, re.DOTALL | re.IGNORECASE)
    optimized_tags = original_tags or []
    if tags_match:
        tags_text = tags_match.group(1)
        # Extract words in quotes
        tag_matches = re.findall(r'["\'](.*?)["\']', tags_text)
        if tag_matches:
            optimized_tags = tag_matches
    
    # Construct a notes section
    notes = "The optimization was extracted from text since JSON parsing failed. "
    notes += "The results may not be complete or fully formatted."
    
    return {
        "original_title": original_title,
        "optimized_title": optimized_title,
        "original_description": original_description,
        "optimized_description": optimized_description,
        "original_tags": original_tags or [],
        "optimized_tags": optimized_tags,
        "optimization_notes": notes
    }

def get_comprehensive_optimization(
    original_title: str,
    original_description: str = "",
    original_tags: Optional[List[str]] = None,
    transcript: str = "",
    has_captions: bool = False,
    model: str = "claude-3-7-sonnet-20250219",
    max_retries: int = 3
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
        logger.error("Anthropic client not initialized. Using fallback optimization.")
        result = fallback_title_optimization(original_title, original_tags or [])
        return {
            "original_title": original_title,
            "optimized_title": result["optimized_title"],
            "original_description": original_description,
            "optimized_description": original_description,
            "original_tags": original_tags or [],
            "optimized_tags": original_tags or [],
            "optimization_notes": result["optimization_notes"]
        }

    # Safely check transcript length
    transcript_length = 0
    if transcript is not None:
        transcript_length = len(transcript)
    logger.info(f"Transcript length: {transcript_length}")

    # Prepare tags string
    tags_str = ", ".join(original_tags) if original_tags else "No tags available"
    
    # Create an array to collect error messages from retry attempts
    retry_errors = []
    
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
            I need a comprehensive optimization for the following YouTube video:

            ORIGINAL TITLE: {original_title}

            ORIGINAL DESCRIPTION:
            {original_description if original_description else "No description provided"}

            ORIGINAL TAGS: {tags_str}
            
            {transcript_section}

            Please provide optimized versions of the title, description, and tags, following these guidelines:

            For the TITLE:
            - Between 40-60 characters for optimal performance
            - Include high-value keywords for searchability
            - Create curiosity and emotional appeal 
            - Maintain the original meaning and intent
            - Use strategic capitalization, symbols, or emojis if appropriate
            - Consider including key hashtags if relevant

            For the DESCRIPTION:
            - Front-load important keywords in the first 2-3 sentences
            - Include a clear call-to-action (subscribe, like, comment)
            - Add timestamps for longer videos (if applicable)
            - Incorporate relevant hashtags (3-5 is optimal)
            - Keep it engaging but concise
            - EXTREMELY IMPORTANT: DO NOT include placeholder text like "[Link to Playlist]" or "[Social Media Links]" - ONLY include actual, real URLs that appear in the original description
            - If you don't have actual URLs or links, do not mention them at all
            - Avoid any placeholders - the description should be ready to publish as-is

            For the TAGS:
            - Focus on trending tags that are relevant to the content
            - Each tag should not contain any whitespace or punctuation, they should be single words or joined phrases (e.g. tag, thisisatag)
            - Include exact and phrase match keywords
            - Focus on specific, niche tags rather than broad ones
            - Start with most important keywords
            - Include misspellings of important terms if relevant
            - 10-15 well-chosen tags is ideal

            {json_emphasis}

            Please return your response in the following JSON format:
            {{
                "optimized_title": "Your optimized title here",
                "optimized_description": "Your optimized description here",
                "optimized_tags": ["tag1", "tag2", "tag3", ...],
                "optimization_notes": "Detailed explanation of the changes made across all elements and why they will improve performance"
            }}
            """

            # Call Claude API 
            message = client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=temperature,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            response_text = message.content[0].text
            
            # Two parsing methods to try
            parsing_methods = [
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
                    
                    # If we got a valid result, return it
                    if is_valid_optimization_result(result):
                        logger.info(f"Successfully parsed response with method: {method['name']}")
                        return result
                except Exception as e:
                    last_error = e
                    logger.warning(f"Parsing method {method['name']} failed: {e}")
            
            # If we got here, all parsing methods failed for this attempt
            retry_errors.append(f"Attempt {attempt+1}: All parsing methods failed. Last error: {last_error}")
            
        except Exception as e:
            logger.error(f"Retry attempt {attempt+1} failed with error: {e}")
            retry_errors.append(f"Attempt {attempt+1}: {str(e)}")
    
    # If we get here, all retries failed
    logger.error(f"All {max_retries} optimization attempts failed")
    result = fallback_title_optimization(original_title, original_tags or [])
    
    return {
        "original_title": original_title,
        "optimized_title": result["optimized_title"],
        "original_description": original_description,
        "optimized_description": original_description,
        "original_tags": original_tags or [],
        "optimized_tags": original_tags or [],
        "optimization_notes": f"Error with Claude optimization after {max_retries} attempts. Errors: {'; '.join(retry_errors)}. {result.get('optimization_notes', '')}"
    }

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