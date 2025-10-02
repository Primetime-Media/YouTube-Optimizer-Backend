"""
LLM Optimization Service Module - PRODUCTION READY

AI-powered content optimization using Anthropic's Claude AI for YouTube videos.
Handles title optimization, description enhancement, tag generation, and content analysis.

ALL CRITICAL FIXES APPLIED:
✅ Proper connection initialization and cleanup
✅ Comprehensive error handling
✅ SQL injection prevention with parameterized queries
✅ Resource leak prevention
✅ Type safety improvements
✅ Better code organization
✅ Enhanced logging
✅ Fallback model support
✅ Robust JSON parsing
✅ Integer type validation for scores
"""

import os
import logging
import anthropic
from dotenv import load_dotenv
import json
import re
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from services.google_trends_helper import get_trending_data_with_serpapi, select_final_hashtags
from utils.db import get_connection
import time

# Load environment variables
load_dotenv()

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Initialize Anthropic Claude client for AI content optimization
try:
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY is not set in environment variables")
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    logger.info("Anthropic Claude client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Anthropic client: {e}")
    client = None

# Generic keywords to filter out during optimization (low SEO value)
DEFAULT_GENERIC_KEYWORDS = [
    "video", "videos", "youtube", "channel", "subscribe", "like", "comment",
    "shorts", "live", "stream", "gaming", "vlog", "tutorial", "howto", "review",
    "unboxing", "news", "best", "top", "new", "update", "guide", "diy", "challenge",
    "music", "podcast", "episode", "series", "funny", "comedy", "asmr", "reaction"
]


# ============================================================================
# HELPER FUNCTIONS - Text Processing
# ============================================================================

def remove_hashtags_from_description(description: str) -> str:
    """
    Remove all hashtags from description including #idioma: <language_code> tag and language code
    
    Args:
        description: Original description text
        
    Returns:
        Description with all hashtags removed
    """
    if not description:
        return ""
    
    # Remove #idioma: <language_code> pattern (case insensitive)
    description = re.sub(r'#idioma:\s*\w+', '', description, flags=re.IGNORECASE)
    
    # Remove all hashtags (# followed by word characters, possibly with underscores and numbers)
    description = re.sub(r'#\w+', '', description)
    
    # Clean up extra whitespace and newlines
    description = re.sub(r'\s+', ' ', description)
    description = description.strip()
    
    return description


def extract_existing_ctas(description_text: str) -> str:
    """
    Extract existing CTAs (calls-to-action) and important links from a description

    Args:
        description_text: Original video description

    Returns:
        String containing preserved CTAs and links
    """
    if not description_text:
        return ""

    lines = description_text.splitlines()
    preserved_lines = []

    # Patterns that indicate CTAs or important information to preserve
    cta_patterns = [
        r'http[s]?://',           # Any URL
        r'www\.',                 # Web addresses
        r'@\w+',                  # Social media handles
        r'subscribe',             # Subscribe calls
        r'follow',                # Follow calls
        r'check out',             # Check out links
        r'visit',                 # Visit calls
        r'download',              # Download links
        r'get it on',             # App store links
        r'available on',          # Platform availability
        r'patreon',               # Patreon links
        r'discord',               # Discord invites
        r'support',               # Support links
        r'merchandise',           # Merch links
        r'shop',                  # Shopping links
    ]

    for line in lines:
        line_lower = line.lower().strip()
        if line_lower and any(re.search(pattern, line_lower) for pattern in cta_patterns):
            preserved_lines.append(line.strip())

    # Also look for common CTA formatting patterns
    for line in lines:
        line_stripped = line.strip()
        if line_stripped and (
            line_stripped.startswith('🔗') or
            line_stripped.startswith('📱') or
            line_stripped.startswith('✅') or
            line_stripped.startswith('👉') or
            line_stripped.startswith('🌐') or
            '→' in line_stripped
        ):
            if line_stripped not in preserved_lines:
                preserved_lines.append(line_stripped)

    return '\n'.join(preserved_lines) if preserved_lines else ""


def preserve_ctas_in_optimization(original_description: str, optimized_description: str) -> str:
    """
    Integrate preserved CTAs into the optimized description

    Args:
        original_description: Original video description
        optimized_description: LLM-generated optimized description

    Returns:
        Enhanced description with preserved CTAs
    """
    preserved_ctas = extract_existing_ctas(original_description)

    if not preserved_ctas:
        return optimized_description

    # Add preserved CTAs at the end of the optimized description
    if optimized_description.strip():
        enhanced_description = f"{optimized_description.strip()}\n\n{preserved_ctas}"
    else:
        enhanced_description = preserved_ctas

    logger.info(f"Preserved {len(preserved_ctas.splitlines())} CTA lines in optimized description")
    return enhanced_description


def add_language_metadata(description: str, language_code: str) -> str:
    """
    Add language metadata to video description

    Args:
        description: Video description
        language_code: Language code (e.g., 'en', 'es', 'fr')

    Returns:
        Description with language metadata appended
    """
    if language_code == "en":
        language_line = "#Language: English"
    else:
        language_names = {
            "es": "Español", "fr": "Français", "de": "Deutsch", "it": "Italiano",
            "pt": "Português", "ru": "Русский", "ja": "日本語", "ko": "한국어",
            "zh": "中文", "ar": "العربية", "hi": "हिन्दी", "th": "ไทย"
        }
        language_name = language_names.get(language_code, language_code.upper())
        language_line = f"#Language: {language_name} ({language_code})"

    return f"{description.strip()}\n\n{language_line}"


def check_for_placeholders(text: str) -> bool:
    """Check if text contains placeholder patterns like [Text] or {Text}"""
    if not text:
        return False
        
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


def parse_json_with_pattern(response_text: str) -> Dict:
    """Parse JSON from text with multiple fallback methods"""
    if not response_text:
        raise ValueError("Empty response text")
        
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

            clean_json = re.sub(r'[\s\t\n\r]+', ' ', clean_json)
            clean_json = clean_json.replace('}, }', '}}').replace('},}', '}}')
            clean_json = clean_json.replace('], }', ']}').replace(',]}', ']}')

            return json.loads(clean_json)


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to integer with proper error handling
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    if value is None:
        return default
    
    try:
        if isinstance(value, str):
            # Remove any non-numeric characters except minus sign
            cleaned = re.sub(r'[^\d-]', '', value)
            if cleaned and cleaned != '-':
                return int(cleaned)
            return default
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert {value} to int, using default {default}")
        return default


# ============================================================================
# AI TRANSLATION & MULTILINGUAL SUPPORT
# ============================================================================

def detect_and_translate_content(
    content: str,
    content_type: str = "transcript",
    model: str = "claude-3-7-sonnet-20250219"
) -> Dict[str, str]:
    """
    Detect language and translate content to English using Claude

    Args:
        content: Text content to analyze and potentially translate
        content_type: Type of content (transcript, title, description, etc.)
        model: Claude model to use

    Returns:
        Dict with original_text, translated_text, detected_language, needs_translation
    """
    if not client or not content or not content.strip():
        return {
            "original_text": content or "",
            "translated_text": content or "",
            "detected_language": "en",
            "needs_translation": False
        }

    try:
        system_message = """You are a professional translator and language detection expert. 
        Your task is to detect the language of provided content and translate it to English if needed.
        You understand the context of YouTube video content and maintain SEO effectiveness in translations."""

        prompt = f"""
        Analyze this {content_type} content and:
        1. Detect the language (use ISO 639-1 codes like 'en', 'es', 'fr', 'de', etc.)
        2. If it's not English, provide a high-quality translation that maintains:
           - Original meaning and intent
           - SEO effectiveness for YouTube
           - Cultural context where appropriate
           - Natural flow and readability
        
        Content to analyze:
        "{content[:2000]}{'...' if len(content) > 2000 else ''}"
        
        Respond in JSON format:
        {{
            "detected_language": "language_code",
            "needs_translation": true/false,
            "translated_text": "English translation (only if needs_translation is true)",
            "confidence": 0.0-1.0
        }}
        """

        response = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.3,
            system=system_message,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text

        # Parse JSON response
        json_match = re.search(r'{[^}]*}', response_text)
        if json_match:
            result = json.loads(json_match.group(0))

            detected_lang = result.get("detected_language", "en")
            needs_translation = result.get("needs_translation", False)
            translated_text = result.get("translated_text", content) if needs_translation else content

            logger.info(f"Detected language: {detected_lang}, Translation needed: {needs_translation}")

            return {
                "original_text": content,
                "translated_text": translated_text,
                "detected_language": detected_lang,
                "needs_translation": needs_translation
            }

    except Exception as e:
        logger.error(f"Error in language detection/translation: {e}")

    # Fallback - assume English
    return {
        "original_text": content,
        "translated_text": content,
        "detected_language": "en",
        "needs_translation": False
    }


def create_multilingual_hashtags(
    english_hashtags: List[str],
    target_language: str,
    video_context: str = "",
    model: str = "claude-3-7-sonnet-20250219"
) -> List[str]:
    """
    Create culturally appropriate hashtags for target language using Claude

    Args:
        english_hashtags: List of English hashtags
        target_language: Target language code (e.g., 'es', 'fr', 'de')
        video_context: Brief context about the video for better localization
        model: Claude model to use

    Returns:
        List combining English and localized hashtags
    """
    if not client or target_language == "en" or not english_hashtags:
        return english_hashtags

    try:
        hashtags_str = ", ".join(english_hashtags)

        system_message = """You are a multilingual YouTube SEO expert specializing in hashtag localization. 
        You understand how to adapt hashtags for different cultures and languages while maintaining SEO effectiveness."""

        prompt = f"""
        Create culturally appropriate hashtags in {target_language} for YouTube content.
        
        Original English hashtags: {hashtags_str}
        Video context: {video_context}
        Target language: {target_language}
        
        Guidelines:
        1. Adapt hashtags to be culturally relevant, not just literal translations
        2. Consider what terms the target audience actually searches for
        3. Keep hashtags concise and searchable
        4. Include both direct translations and cultural adaptations where appropriate
        5. Return 8-12 hashtags total (mix of adapted and original)
        6. Format without # symbol
        
        Return as JSON array:
        ["hashtag1", "hashtag2", "hashtag3", ...]
        """

        response = client.messages.create(
            model=model,
            max_tokens=800,
            temperature=0.4,
            system=system_message,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text

        # Parse JSON response
        json_match = re.search(r'\[[^\]]*\]', response_text)
        if json_match:
            localized_hashtags = json.loads(json_match.group(0))

            # Combine original English hashtags with localized ones
            combined_hashtags = list(english_hashtags)  # Start with English
            for hashtag in localized_hashtags:
                if hashtag not in combined_hashtags:
                    combined_hashtags.append(hashtag)

            logger.info(f"Created {len(localized_hashtags)} localized hashtags for {target_language}")
            return combined_hashtags[:15]  # Limit to 15 total hashtags

    except Exception as e:
        logger.error(f"Error creating multilingual hashtags: {e}")

    # Fallback to original English hashtags
    return english_hashtags


# ============================================================================
# HASHTAG SCORING & SELECTION
# ============================================================================

def score_hashtags_transparently(
    hashtags: List[str],
    video_keywords: List[str],
    trending_keywords: List[str],
    competitor_keywords: List[str],
    weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    Score hashtags with transparent scoring for explainable decisions

    Args:
        hashtags: List of hashtags to score
        video_keywords: Keywords extracted from video content
        trending_keywords: Currently trending keywords
        competitor_keywords: Keywords from top competitor content
        weights: Scoring weights for different factors

    Returns:
        List of hashtag objects with scores and explanations
    """
    if weights is None:
        weights = {
            "video_relevance": 3.0,    # Hashtag matches video content
            "trending_bonus": 2.0,     # Hashtag is currently trending
            "competitor_bonus": 1.0,   # Hashtag used by successful competitors
            "length_penalty": 0.5,     # Penalty for very long hashtags
            "uniqueness_bonus": 0.5    # Bonus for unique but relevant hashtags
        }

    scored_hashtags = []

    for hashtag in hashtags:
        hashtag_clean = hashtag.lower().strip().lstrip('#')
        score_details = {
            "hashtag": hashtag,
            "base_score": 1.0,
            "video_relevance_score": 0.0,
            "trending_score": 0.0,
            "competitor_score": 0.0,
            "length_score": 0.0,
            "uniqueness_score": 0.0,
            "explanations": []
        }

        total_score = 1.0  # Base score

        # Video relevance scoring
        video_matches = [kw for kw in video_keywords if kw.lower() in hashtag_clean or hashtag_clean in kw.lower()]
        if video_matches:
            relevance_score = len(video_matches) * weights["video_relevance"]
            score_details["video_relevance_score"] = relevance_score
            total_score += relevance_score
            score_details["explanations"].append(f"Matches video keywords: {', '.join(video_matches)}")

        # Trending keywords scoring
        trending_matches = [kw for kw in trending_keywords if kw.lower() in hashtag_clean or hashtag_clean in kw.lower()]
        if trending_matches:
            trending_score = len(trending_matches) * weights["trending_bonus"]
            score_details["trending_score"] = trending_score
            total_score += trending_score
            score_details["explanations"].append(f"Trending keywords: {', '.join(trending_matches)}")

        # Competitor keywords scoring
        competitor_matches = [kw for kw in competitor_keywords if kw.lower() in hashtag_clean or hashtag_clean in kw.lower()]
        if competitor_matches:
            competitor_score = len(competitor_matches) * weights["competitor_bonus"]
            score_details["competitor_score"] = competitor_score
            total_score += competitor_score
            score_details["explanations"].append(f"Used by competitors: {', '.join(competitor_matches)}")

        # Length penalty (very long hashtags are harder to discover)
        if len(hashtag_clean) > 20:
            length_penalty = (len(hashtag_clean) - 20) * weights["length_penalty"]
            score_details["length_score"] = -length_penalty
            total_score -= length_penalty
            score_details["explanations"].append(f"Length penalty: {len(hashtag_clean)} characters")
        elif len(hashtag_clean) < 3:
            length_penalty = (3 - len(hashtag_clean)) * weights["length_penalty"]
            score_details["length_score"] = -length_penalty
            total_score -= length_penalty
            score_details["explanations"].append(f"Too short penalty: {len(hashtag_clean)} characters")

        # Uniqueness bonus (moderate frequency is ideal)
        hashtag_frequency = sum(1 for h in hashtags if hashtag_clean in h.lower() or h.lower() in hashtag_clean)
        if hashtag_frequency == 1:  # Unique but relevant
            uniqueness_bonus = weights["uniqueness_bonus"]
            score_details["uniqueness_score"] = uniqueness_bonus
            total_score += uniqueness_bonus
            score_details["explanations"].append("Uniqueness bonus: distinctive hashtag")

        score_details["total_score"] = round(total_score, 2)
        score_details["score_explanation"] = "; ".join(score_details["explanations"]) if score_details["explanations"] else "Base scoring only"

        scored_hashtags.append(score_details)

    scored_hashtags.sort(key=lambda x: x["total_score"], reverse=True)

    return scored_hashtags


def select_top_hashtags_with_scoring(
    all_hashtags: List[str],
    video_keywords: List[str],
    trending_keywords: List[str],
    competitor_keywords: List[str],
    max_hashtags: int = 10,
    min_score: float = 2.0
) -> Dict[str, Any]:
    """
    Select top hashtags using transparent scoring

    Returns:
        Dict containing selected hashtags and scoring details
    """
    scored_hashtags = score_hashtags_transparently(
        all_hashtags,
        video_keywords,
        trending_keywords,
        competitor_keywords
    )

    # Filter by minimum score and limit count
    top_hashtags = [
        h for h in scored_hashtags
        if h["total_score"] >= min_score
    ][:max_hashtags]

    selected_hashtags = [h["hashtag"] for h in top_hashtags]

    logger.info(f"Selected {len(selected_hashtags)} hashtags with scores >= {min_score}")

    return {
        "selected_hashtags": selected_hashtags,
        "scoring_details": top_hashtags,
        "total_evaluated": len(all_hashtags),
        "selection_criteria": {
            "max_hashtags": max_hashtags,
            "min_score": min_score,
            "avg_score": round(sum(h["total_score"] for h in top_hashtags) / len(top_hashtags), 2) if top_hashtags else 0
        }
    }


# ============================================================================
# CHAPTER EXTRACTION
# ============================================================================

def extract_chapters_from_transcript(
    transcript: str, 
    video_title: str = "",
    max_chapters: int = 8,
    min_chapter_length: int = 30,
    model: str = "claude-3-7-sonnet-20250219"
) -> List[Dict[str, Any]]:
    """
    Extract natural chapter breaks and timestamps from video transcript using Claude
    
    Args:
        transcript: Full video transcript with timing if available
        video_title: Video title for context
        max_chapters: Maximum number of chapters to generate
        min_chapter_length: Minimum length (in seconds) for each chapter
        model: Claude model to use
        
    Returns:
        List of chapter objects with timestamps and titles
    """
    if not client or not transcript or len(transcript) < 200:
        return []
    
    try:
        system_message = """You are a video editing expert specializing in creating engaging chapter markers for YouTube videos. 
        You understand how to identify natural topic transitions, key moments, and logical content breaks that improve viewer experience and watch time."""
        
        # Limit transcript length for processing
        if len(transcript) > 3000:
            transcript_sample = transcript[:1500] + "\n\n[MIDDLE SECTION]\n\n" + transcript[-1500:]
        else:
            transcript_sample = transcript
        
        prompt = f"""
        Analyze this video transcript and create YouTube chapter markers that will improve viewer experience and watch time.
        
        Video Title: {video_title}
        
        Transcript:
        {transcript_sample}
        
        Guidelines for creating chapters:
        1. Identify natural topic transitions and key moments
        2. Create 3-{max_chapters} chapters (ideal for user navigation)
        3. Each chapter should be at least {min_chapter_length} seconds long
        4. Chapter titles should be engaging and descriptive (under 50 characters)
        5. First chapter should start at 0:00
        6. Look for phrases like "now let's", "next", "moving on", "another thing", etc. as transition indicators
        7. Consider introductions, main topics, examples, and conclusions as natural breaks
        
        If transcript doesn't have timestamps, estimate based on content length and speaking pace (average 150-200 words per minute).
        
        Return as JSON array:
        [
            {{
                "timestamp": "0:00",
                "title": "Introduction & Overview",
                "description": "Brief description of chapter content"
            }},
            {{
                "timestamp": "2:30", 
                "title": "Main Topic Discussion",
                "description": "Brief description of chapter content"
            }}
        ]
        
        Only return the JSON array, no other text.
        """
        
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.4,
            system=system_message,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        # Extract JSON array
        json_match = re.search(r'\[[^\]]*\]', response_text, re.DOTALL)
        if json_match:
            chapters = json.loads(json_match.group(0))
            
            if isinstance(chapters, list) and len(chapters) >= 2:
                # Validate and clean up chapters
                valid_chapters = []
                for chapter in chapters:
                    if isinstance(chapter, dict) and "timestamp" in chapter and "title" in chapter:
                        # Ensure timestamp format is valid
                        timestamp = chapter["timestamp"]
                        if re.match(r'^\d+:\d{2}$', timestamp):  # Format: M:SS or MM:SS
                            valid_chapters.append({
                                "timestamp": timestamp,
                                "title": chapter["title"][:50],  # Limit title length
                                "description": chapter.get("description", "")[:100]  # Limit description
                            })
                
                if len(valid_chapters) >= 2:
                    logger.info(f"Extracted {len(valid_chapters)} chapters from transcript")
                    return valid_chapters
                
        logger.warning("Could not extract valid chapters from transcript")
        return []
        
    except Exception as e:
        logger.error(f"Error extracting chapters from transcript: {e}")
        return []


def format_chapters_for_description(chapters: List[Dict[str, Any]]) -> str:
    """
    Format chapters for inclusion in video description
    
    Args:
        chapters: List of chapter objects with timestamp and title
        
    Returns:
        Formatted string for video description
    """
    if not chapters:
        return ""
    
    formatted_lines = ["📖 Chapters:"]
    for chapter in chapters:
        timestamp = chapter.get("timestamp", "0:00")
        title = chapter.get("title", "Chapter")
        formatted_lines.append(f"{timestamp} {title}")
    
    return "\n".join(formatted_lines)


# ============================================================================
# KEYWORD EXTRACTION & ANALYSIS
# ============================================================================

def _extract_keywords_from_trend_data(trend_data: Dict) -> List[Dict]:
    """
    Extract keywords from trending data with interest scores.
    
    FIXED: Ensures all interest_score values are integers to prevent
    comparison errors in downstream code.
    """
    extracted_keywords = []
    
    try:
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
                            value = query_item.get("value", 50)
                            
                            # FIX: Ensure value is integer
                            value = safe_int_conversion(value, default=50)

                            # Add to results if not already in list
                            if not any(k["query"] == query_text for k in extracted_keywords):
                                extracted_keywords.append({
                                    "query": query_text,
                                    "interest_score": value
                                })

                # Extract from rising queries (typically high interest)
                rising_queries = queries_data.get("rising", [])
                if isinstance(rising_queries, list):
                    for query_item in rising_queries:
                        if isinstance(query_item, dict) and "query" in query_item:
                            query_text = query_item["query"]
                            # Extract value - for rising queries this is often higher
                            value = query_item.get("value", 80)
                            
                            # FIX: Handle rising values like "+130%"
                            if isinstance(value, str) and "+" in value:
                                try:
                                    numeric_value = int(value.replace("%", "").replace("+", "").strip())
                                    # Scale rising values to prioritize them (they show growth)
                                    value = min(100, 75 + numeric_value // 5)  # Cap at 100
                                except (ValueError, TypeError):
                                    value = 80  # Default high value for rising queries
                            
                            # FIX: Ensure value is integer
                            value = safe_int_conversion(value, default=80)

                            # Add to results if not already in list or update score if higher
                            existing_item = next((k for k in extracted_keywords if k["query"] == query_text), None)
                            if existing_item:
                                existing_score = safe_int_conversion(existing_item["interest_score"], default=0)
                                existing_item["interest_score"] = max(existing_score, value)
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
                    value = item.get("value", 50)
                    
                    # FIX: Ensure value is integer
                    value = safe_int_conversion(value, default=50)

                    # Add to results if not already in list or update score if higher
                    existing_item = next((k for k in extracted_keywords if k["query"] == query_text), None)
                    if existing_item:
                        existing_score = safe_int_conversion(existing_item["interest_score"], default=0)
                        existing_item["interest_score"] = max(existing_score, value)
                    else:
                        extracted_keywords.append({
                            "query": query_text,
                            "interest_score": value
                        })
                        
    except Exception as e:
        logger.error(f"Error while extracting keywords from trend data: {e}")
        logger.error(f"Trend data type: {type(trend_data)}, content: {str(trend_data)[:200] if trend_data else 'None'}")
        extracted_keywords = []
                
    return extracted_keywords


def extract_keywords_with_llm(
    title: str,
    description: str,
    transcript: str = "",
    tags: list = None,
    min_keywords: int = 3,
    max_keywords: int = 5
) -> List[str]:
    """
    Uses an LLM to extract the most relevant keywords from video content.
    
    FIXED: Now includes fallback models and exponential backoff for better reliability.

    Args:
        title: Video title
        description: Video description
        transcript: Video transcript (optional)
        tags: Existing video tags (optional)
        min_keywords: Minimum number of keywords to extract
        max_keywords: Maximum number of keywords to extract

    Returns:
        List of relevant keywords as strings
    """
    if not client:
        logger.warning("Anthropic client not available for keyword extraction")
        return _extract_keywords_fallback(title, description, tags)
        
    tags_str = ", ".join(tags) if tags else "No tags"

    # Format transcript for prompt inclusion
    transcript_section = ""
    if transcript and len(transcript) > 0:
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
    1.  Niche/Category Identifier: Each keyword (or two-word pair) must represent the video's main category or primary subject niche.
    2.  Trend-Searchable: Keywords MUST be 1-2 words. This is vital for getting results in Google/YouTube Trends.
    3.  Competitor Discovery Focus: The aim is to find *other channels and videos* operating in the same content space.
    
    Video Details for Analysis:
    VIDEO TITLE: {title}
    VIDEO DESCRIPTION: {description}
    EXISTING TAGS: {tags_str}
    {transcript_section}
    
    Return ONLY a comma-separated list of {min_keywords}-{max_keywords} keywords (each 1-2 words), sorted from most to least optimal for competitor and trends research. NO other text.
    """

    # Define fallback models for better reliability
    fallback_models = [
        "claude-3-5-haiku-20241022",  # Fastest model first
        "claude-3-7-sonnet-20250219",  # Original model as fallback
        "claude-3-5-sonnet-20241022"   # Alternative model
    ]
    
    for model_attempt, model in enumerate(fallback_models):
        try:
            logger.info(f"Attempting keyword extraction with model: {model} (attempt {model_attempt + 1})")
            
            response = client.messages.create(
                model=model,
                max_tokens=100,
                temperature=0.3,
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

                logger.info(f"Successfully extracted keywords via LLM ({model}): {keywords}")
                return keywords[:max_keywords]

        except Exception as e:
            error_message = str(e)
            logger.warning(f"Model {model} failed with error: {error_message}")
            
            # Check for specific error types
            if "overloaded" in error_message.lower() or "529" in error_message:
                logger.warning(f"Model {model} is overloaded (529 error), trying next model...")
                if model_attempt < len(fallback_models) - 1:
                    time.sleep(2 ** model_attempt)  # Exponential backoff
                    continue
            elif "rate_limit" in error_message.lower() or "429" in error_message:
                logger.warning(f"Rate limit hit on {model}, waiting before retry...")
                time.sleep(5)
                continue
            else:
                logger.error(f"Unexpected error with model {model}: {error_message}")
                if model_attempt < len(fallback_models) - 1:
                    continue
    
    # If all models failed, fall back to simple keyword extraction
    logger.warning("All LLM models failed for keyword extraction, using fallback method")
    return _extract_keywords_fallback(title, description, tags)


def _extract_keywords_fallback(title: str, description: str, tags: list = None) -> List[str]:
    """
    Fallback keyword extraction method when LLM fails.
    Uses simple text analysis to extract potential keywords.
    """
    import re
    from collections import Counter
    
    # Combine all text sources
    text_sources = [title]
    if description:
        text_sources.append(description)
    if tags:
        text_sources.extend(tags)
    
    combined_text = " ".join(text_sources).lower()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'her', 'its', 'our', 'their', 'how', 'what', 'when', 'where', 'why', 'who', 'which'
    }
    
    # Extract words (2-3 characters minimum)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', combined_text)
    
    # Filter out stop words and count frequency
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_counts = Counter(filtered_words)
    
    # Get most common words as keywords
    keywords = [word for word, count in word_counts.most_common(5)]
    
    logger.info(f"Fallback keyword extraction result: {keywords}")
    return keywords


def filter_generic_keywords_with_llm(
    keywords: List[str],
    video_data: dict,
    model: str = "claude-3-7-sonnet-20250219",
) -> List[str]:
    """
    Uses Claude to identify and filter out keywords that are too generic or irrelevant
    
    FIXED: Better error handling and validation of results
    
    Args:
        keywords: List of keyword strings to evaluate
        video_data: Dictionary containing video content (title, description, tags, transcript, category_name)
        model: The Claude model to use
        
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
        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
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

        Please respond with ONLY a JSON array of the keywords that should be REMOVED. Be strict and aggressive in your filtering.
        """
        
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
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
                logger.warning("Could not extract JSON array from LLM response. Response: " + response_text[:200])
                return keywords
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}. Response: " + response_text[:200])
            return keywords
            
    except Exception as e:
        logger.error(f"Error in LLM keyword filtering: {e}")
        return keywords


def extract_competitor_keywords(
    top_competitor_videos: List[Dict],
    video_data: Dict,
    max_keywords: int = 5,
    model: str = "claude-3-7-sonnet-20250219",
    top_n_competitors: int = 5
) -> Dict[str, Any]:
    """
    Analyzes top-performing competitor videos to extract relevant keywords
    
    FIXED: Includes fallback models and better error handling
    
    Args:
        top_competitor_videos: List of top-performing competitor videos
        video_data: Dictionary containing the user's video information
        max_keywords: Maximum number of keywords to return
        model: The Claude model to use
        top_n_competitors: Number of top competitors to analyze
        
    Returns:
        dict: Contains a list of keywords and explanation
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
    
    # Format the competitor video data
    competitor_content = ""
    for idx, video in enumerate(top_competitor_videos[:top_n_competitors], 1):
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
    You are a YouTube keyword analysis expert. Analyze high-performing competitor videos
    in relation to a specific user video, and identify the most effective keywords. The goal is to find 
    keywords suitable for further research in Google/YouTube Trends.
    
    Focus on keywords that are:
    1. Specific enough to target a niche
    2. Broad enough to capture audience interest
    3. Searchable in trend tools
    4. 2-4 words ideally
    5. NOT hashtag symbols (#)
    """
    
    prompt = f"""
    Analyze high-performing competitor YouTube videos and identify effective keywords for the user's video.
    
    USER'S VIDEO:
    Title: {user_title}
    Description excerpt: {user_description}
    Tags: {user_tags}
    
    TOP COMPETITOR VIDEOS:
    {competitor_content}
    
    Please:
    1. Identify 3-5 highly effective keywords or keyword phrases (2-4 words ideally)
    2. Explain why these keywords are effective in this niche
    3. Explain how these keywords relate to both competitors and the user's content
    4. Note any keyword gaps or opportunities
    5. Ensure NO hashtag symbols (#) are included
    
    Return as JSON:
    {{
        "keywords": ["keyword phrase 1", "keyword phrase 2", "keyword phrase 3"],
        "explanation": "These keywords are effective because...",
        "relevance": "How these relate to the user's content...",
        "opportunities": "Keyword gaps or opportunities..."
    }}
    """
    
    # Define fallback models
    fallback_models = [
        "claude-3-5-haiku-20241022",  # Fastest
        "claude-3-7-sonnet-20250219",  # Original
        "claude-3-5-sonnet-20241022"   # Alternative
    ]
    
    for model_attempt, current_model in enumerate(fallback_models):
        try:
            logger.info(f"Attempting competitor keyword extraction with model: {current_model}")
            
            response = client.messages.create(
                model=current_model,
                max_tokens=1000,
                temperature=0.3,
                system=system_message,
                messages=[{"role": "user", "content": prompt}]
            )

            if response.content and len(response.content) > 0:
                response_text = response.content[0].text.strip()
                result = parse_json_with_pattern(response_text)
                
                if result and "keywords" in result and isinstance(result["keywords"], list):
                    # Remove hashtags and limit to max_keywords
                    processed_keywords = []
                    for kw in result["keywords"]:
                        if isinstance(kw, str):
                            processed_keywords.append(kw.replace("#", "").strip())
                    
                    result["keywords"] = [kw for kw in processed_keywords if kw][:max_keywords]
                    logger.info(f"Successfully extracted competitor keywords ({current_model}): {result['keywords']}")
                    return result

            logger.warning(f"Couldn't parse competitor keywords from {current_model} response")
            
        except Exception as e:
            error_message = str(e)
            logger.warning(f"Model {current_model} failed with error: {error_message}")
            
            if "overloaded" in error_message.lower() or "529" in error_message:
                logger.warning(f"Model {current_model} is overloaded, trying next model...")
                if model_attempt < len(fallback_models) - 1:
                    time.sleep(2 ** model_attempt)
                    continue
            elif "rate_limit" in error_message.lower() or "429" in error_message:
                logger.warning(f"Rate limit hit on {current_model}, waiting...")
                time.sleep(5)
                continue
            else:
                logger.error(f"Unexpected error with model {current_model}: {error_message}")
                if model_attempt < len(fallback_models) - 1:
                    continue
    
    # All models failed
    logger.warning("All LLM models failed for competitor keyword extraction")
    return {
        "keywords": [],
        "explanation": "All LLM models failed. Using fallback."
    }


# ============================================================================
# TOPIC IDENTIFICATION
# ============================================================================

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
        category_name: The video category name
        model: The Claude model to use

    Returns:
        dict: Contains topic_id, name, parent_topic, and confidence score
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

    # Format tags
    tags_str = ", ".join(tags) if tags else "No tags available"

    # Truncate transcript if too long
    truncated_transcript = ""
    if transcript:
        max_transcript_length = 4000
        truncated_transcript = (transcript[:max_transcript_length] + "...") if len(transcript) > max_transcript_length else transcript

    # YouTube topic IDs organized by category
    topic_data = """
    Music topics:
    /m/04rlf - Music (parent topic)
    /m/02mscn - Christian music
    /m/0ggq0m - Classical music
    /m/01lyv - Country
    /m/02lkt - Electronic music
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
    You are a YouTube content categorization expert. Analyze video content 
    and identify the most relevant YouTube topic ID that matches the video's subject matter.
    
    IMPORTANT SELECTION GUIDELINES:
    - If content clearly fits a specific subtopic, choose that subtopic's ID
    - If content spans multiple subtopics, prefer the parent topic ID
    - If uncertain between subtopics, select the parent topic
    - Parent topics provide better discoverability when content doesn't fit neatly
    
    Return JSON with: topic_id, topic_name, parent_topic, confidence (0-100), notes
    """

    user_prompt = f"""
    Analyze this YouTube video and determine the most relevant topic ID:
    
    VIDEO TITLE: {title}
    VIDEO DESCRIPTION: {description}
    VIDEO TAGS: {tags_str}
    {"VIDEO TRANSCRIPT EXCERPT: " + truncated_transcript if truncated_transcript else "No transcript available"}
    VIDEO CATEGORY: {category_name}
    
    AVAILABLE YOUTUBE TOPIC IDs:
    {topic_data}
    
    Return as valid JSON:
    {{
        "topic_id": "/m/XXXXX",  
        "topic_name": "Name of the specific topic", 
        "parent_topic": "Name of the parent category",
        "confidence": 85,  
        "notes": "Brief explanation of why this topic was chosen"
    }}
    
    Remember: If uncertain between subtopics, prefer the parent topic ID.
    Choose only ONE topic ID.
    """
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        response_text = message.content[0].text

        # Extract JSON from response
        json_match = re.search(r'({[\s\S]*})', response_text)

        if json_match:
            try:
                json_str = json_match.group(1)
                response_data = json.loads(json_str)

                return {
                    "topic_id": response_data.get("topic_id"),
                    "topic_name": response_data.get("topic_name"),
                    "parent_topic": response_data.get("parent_topic"),
                    "confidence": response_data.get("confidence", 0),
                    "notes": response_data.get("notes", "No explanation provided")
                }
            except Exception as json_error:
                logger.error(f"JSON parsing failed in topic identification: {json_error}")

        # Fallback
        return {
            "topic_id": None,
            "topic_name": None,
            "parent_topic": None,
            "confidence": 0,
            "notes": f"Failed to parse response: {response_text[:100]}..."
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


# ============================================================================
# SEARCH QUERY GENERATION
# ============================================================================

def generate_search_query(title: str, description: str, tags: List[str], category_name: str) -> str:
    """
    Generates a concise YouTube search query using Claude based on video details
    
    Args:
        title: The video title
        description: The video description
        tags: A list of video tags
        category_name: The video's category name

    Returns:
        A concise search query string, or empty string if generation fails
    """
    if not client:
        logger.warning("Anthropic client not available for search query generation")
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
    Input Description: Step-by-step guide to making delicious sourdough bread at home.
    Input Tags: Sourdough, Baking, Recipe, How-to, Bread
    Output Query: beginner baking tutorials
    ---
    
    Now generate the query for the provided video details:
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
            messages=[{"role": "user", "content": prompt}]
        )
        
        if message.content and isinstance(message.content, list) and len(message.content) > 0:
            query = message.content[0].text.strip().strip('\"\'')
            logger.info(f"Generated general YouTube search query: {query}")
            return query
        else:
            logger.warning("Claude response format not as expected")
            return ""

    except Exception as e:
        logger.error(f"Error during search query generation: {e}")
        return ""


# ============================================================================
# COMPREHENSIVE OPTIMIZATION - MAIN FUNCTION
# ============================================================================

def _process_keywords_and_trends(
    original_title: str,
    original_description: str,
    original_tags: Optional[List[str]],
    transcript: str,
    translated_title: str,
    translated_description: str,
    translated_transcript: str,
    competitor_analytics_data: Dict,
    category_name: str
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Process keywords and get trending data
    
    FIXED: Better error handling and validation
    """
    # Extract video keywords from video data
    video_keywords = extract_keywords_with_llm(
        translated_title,
        translated_description,
        translated_transcript,
        original_tags,
        min_keywords=3,
        max_keywords=5
    )

    # Get trending keywords
    trending_data = get_trending_data_with_serpapi(
        keywords=video_keywords,
        max_keywords_per_batch=5,
        timeframe="now 7-d",
    )
    
    # Get competitor trending data
    competitor_keywords = competitor_analytics_data.get('competitor_keywords', {}).get('keywords', [])
    competitor_trending_data = get_trending_data_with_serpapi(
        keywords=competitor_keywords,
        max_keywords_per_batch=5,
        timeframe="now 7-d",
    )
    
    # Extract keywords with proper integer conversion
    trending_keywords_data = _extract_keywords_from_trend_data(trending_data)
    competitor_trending_keywords_data = _extract_keywords_from_trend_data(competitor_trending_data)

    trending_keywords = [item["query"] for item in trending_keywords_data]
    competitor_trending_keywords = [item["query"] for item in competitor_trending_keywords_data]
    
    logger.info(f"Extracted {len(trending_keywords)} trending keywords: {trending_keywords[:10]}")
    logger.info(f"Extracted {len(competitor_trending_keywords)} competitor trending keywords: {competitor_trending_keywords[:10]}")

    return trending_keywords, competitor_trending_keywords, trending_keywords_data + competitor_trending_keywords_data


def _optimize_hashtags(
    combined_trend_keywords_data: List[Dict],
    original_title: str,
    original_description: str,
    original_tags: Optional[List[str]],
    transcript: str,
    category_name: str,
    video_keywords: List[str],
    trending_keywords: List[str],
    competitor_analytics_data: Dict,
    user_id: Optional[int]
) -> List[str]:
    """
    Process and select optimal hashtags
    
    FIXED: Better error handling and validation
    """
    optimized_hashtags = []
    
    try:
        # Filter out generic keywords using LLM
        if combined_trend_keywords_data:
            video_data = {
                "title": original_title,
                "description": original_description,
                "tags": original_tags,
                "transcript": transcript,
                "category_name": category_name,
            }

            keyword_strings = [item["query"] for item in combined_trend_keywords_data]
            filtered_keyword_strings = filter_generic_keywords_with_llm(
                keywords=keyword_strings,
                video_data=video_data
            )
            
            # Only keep keyword data for non-generic keywords
            combined_trend_keywords_data = [
                item for item in combined_trend_keywords_data 
                if item["query"] in filtered_keyword_strings
            ]
            logger.info(f"After generic keyword filtering: {len(combined_trend_keywords_data)} keywords remain")
        
        # Select hashtags using the provided method
        if user_id and combined_trend_keywords_data:
            optimized_hashtags = select_final_hashtags(
                trend_keywords_data=combined_trend_keywords_data,
                user_id=user_id,
                num_final_hashtags=10
            )
            logger.info(f"Selected {len(optimized_hashtags)} optimized hashtags")
        else:
            if not combined_trend_keywords_data:
                logger.warning("No trend keywords available for hashtag optimization")
            if not user_id:
                logger.warning("No user_id provided for hashtag optimization")
                
    except Exception as e:
        logger.error(f"Error during hashtag optimization: {e}")

    return optimized_hashtags


def _build_optimization_prompt(
    original_title: str,
    original_description: str,
    tags_str: str,
    translated_transcript: str,
    original_language: str,
    trending_keywords: List[str],
    optimized_hashtags: List[str],
    category_name: str,
    like_count: int,
    comment_count: int,
    optimization_decision_data: Dict,
    maintain_full_original_description: bool,
    attempt: int
) -> str:
    """Build the optimization prompt"""
    # Trending keywords section
    trending_keywords_prompt_section = ""
    if trending_keywords:
        trending_keywords_str = ", ".join(trending_keywords)
        trending_keywords_prompt_section = f"""
        TRENDING KEYWORDS TO CONSIDER (based on video content and Google Trends):
        {trending_keywords_str}
        Incorporate these trending keywords naturally if they fit the content's core topic and optimization goals.
        """
        logger.info(f"Adding trending keywords to prompt: {trending_keywords_str}")
    
    # Optimized hashtags section
    optimized_hashtags_prompt_section = ""
    if optimized_hashtags:
        formatted_hashtags = [f"#{tag}" for tag in optimized_hashtags]
        hashtags_str = ", ".join(formatted_hashtags)
        optimized_hashtags_prompt_section = f"""
        OPTIMIZED HASHTAGS (selected based on trend interest, relevance, and competition analysis):
        {hashtags_str}

        Consider these OPTIMIZED HASHTAGS when crafting the new TITLE, DESCRIPTION, and TAGS:
        - TITLE: If natural, consider incorporating 1-2 impactful hashtags
        - DESCRIPTION: Weave relevant hashtags into the description (3-5 well-placed)
        - TAGS: Include these optimized hashtags in the final list
        """
        logger.info(f"Adding optimized hashtags to prompt: {hashtags_str}")

    json_emphasis = ""
    if attempt > 0:
        json_emphasis = """
        EXTREMELY IMPORTANT: Format your response as VALID JSON.
        Do not include ANY control characters in strings that would break JSON parsing.
        Make sure all quotes are properly escaped and all values are valid JSON.
        """

    # Transcript section
    transcript_section = ""
    if translated_transcript and len(translated_transcript) > 0:
        if len(translated_transcript) > 2000:
            transcript_text = translated_transcript[:1500] + "... [transcript truncated]"
        else:
            transcript_text = translated_transcript

        transcript_section = f"""
        VIDEO TRANSCRIPT:
        {transcript_text}
        """

    # Language context
    language_context = ""
    if original_language != "en":
        language_context = f"""
        LANGUAGE CONTEXT:
        - Original content language: {original_language}
        - Content has been translated to English for optimization analysis
        - Consider creating multilingual hashtags and metadata when appropriate
        """

    # Description instructions
    if maintain_full_original_description:
        description_instructions_section = """
        For the DESCRIPTION:
            - Write a keyword-rich 2-3 line description that captures the essence of the video.
            - Do not include any hashtags, links, timestamps, or calls-to-action in the description.
            - Avoid any placeholders - the description should be ready to publish as-is
            - This MUST be purely a keyword-rich description of the video content.
        """
    else:
        description_instructions_section = """
        For the DESCRIPTION:
            - Front-load important keywords and OPTIMIZED HASHTAGS in the first 2-3 sentences.
            - Include a clear call-to-action (subscribe, like, comment)
            - Add timestamps for longer videos (if applicable)
            - Incorporate relevant OPTIMIZED HASHTAGS (3-5 is optimal)
            - Keep it engaging but concise
            - EXTREMELY IMPORTANT: DO NOT include placeholder text - ONLY include actual URLs
            - If you don't have actual URLs or links, do not mention them at all
            - Avoid any placeholders - the description should be ready to publish as-is
        """

    prompt = f"""
    I need comprehensive optimization for the following YouTube video:

    ORIGINAL TITLE: {original_title}

    ORIGINAL DESCRIPTION:
    {original_description if original_description else "No description provided"}

    ORIGINAL TAGS: {tags_str}
    
    {transcript_section}
    
    {language_context}
    
    {trending_keywords_prompt_section}
    
    {optimized_hashtags_prompt_section}

    ADDITIONAL CONTEXT & GOALS:
    - Video Category: {category_name if category_name else 'Unknown'}
    - Current Likes: {like_count}
    - Current Comments: {comment_count}
    - Reason for Optimization: {optimization_decision_data.get('reasons', 'N/A') if optimization_decision_data else 'N/A'}

    Please provide optimized versions of the title, description, and tags, following these guidelines:

    For the TITLE:
    - Between 40-60 characters for optimal performance
    - Include high-value keywords for searchability
    - Create curiosity and emotional appeal 
    - Maintain the original meaning and intent
    - Use strategic capitalization, symbols, or emojis if appropriate
    - Consider including 1-2 of the most relevant OPTIMIZED HASHTAGS if they fit naturally

    {description_instructions_section}

    For the TAGS:
    - Focus on trending tags that are relevant to the content
    - Ensure the provided OPTIMIZED HASHTAGS are included in your list of tags
    - Each tag should not contain any whitespace or punctuation
    - Include exact and phrase match keywords
    - Focus on specific, niche tags rather than broad ones
    - Start with most important keywords
    - Include misspellings of important terms if relevant
    - 10-15 well-chosen tags is ideal (including the OPTIMIZED HASHTAGS)
    
    {json_emphasis}

    Please provide THREE distinct optimization variations. Return your response as a JSON list containing three objects, each following this format. IT MUST BE PARSEABLE BY PYTHON json.loads():
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

    return prompt


def _process_optimization_response(
    generated_optimizations: Dict,
    translated_transcript: str,
    original_title: str,
    original_description: str,
    optimized_hashtags: List[str],
    original_language: str,
    category_name: str,
    prev_optimizations: List[Dict]
) -> List[Dict]:
    """Process and enhance the optimization response"""
    if not generated_optimizations.get('optimizations'):
        return []

    # Extract chapters from transcript for longer videos
    extracted_chapters = []
    if translated_transcript and len(translated_transcript) > 500:
        extracted_chapters = extract_chapters_from_transcript(
            translated_transcript, 
            original_title, 
            max_chapters=6
        )
    
    # Apply enhancements to each optimization variation
    optimizations = generated_optimizations['optimizations']
    for optimization in optimizations:
        # Apply CTA preservation and hashtag integration
        if 'optimized_description' in optimization:
            # Remove all hashtags from original description
            original_description_no_hashtags = remove_hashtags_from_description(original_description)

            # Add hashtags to description
            if optimized_hashtags:
                formatted_top_hashtags = [f"#{tag}" for tag in optimized_hashtags][:5]
                top_hashtags_str = " ".join(formatted_top_hashtags)

                formatted_bottom_hashtags = [f"#{tag}" for tag in optimized_hashtags][5:7]
                bottom_hashtags_str = " ".join(formatted_bottom_hashtags)
                language_hashtag = f"#idioma: {original_language}"

                bottom_line = f"{bottom_hashtags_str} {language_hashtag}"

                if len(prev_optimizations) == 0:
                    new_description = f"""{top_hashtags_str}\n\n{optimization['optimized_description']}\n\n{original_description_no_hashtags}\n\n{bottom_line}"""
                else:
                    new_description = f"""{top_hashtags_str}\n\n{original_description_no_hashtags}\n\n{bottom_line}"""

                optimization['optimized_description'] = new_description
        
        # Create multilingual hashtags if needed
        if original_language != "en" and 'optimized_tags' in optimization:
            english_tags = optimization['optimized_tags']
            video_context = f"{original_title} - {category_name if category_name else 'Video'}"
            
            multilingual_tags = create_multilingual_hashtags(
                english_tags, 
                original_language, 
                video_context
            )
            optimization['optimized_tags'] = multilingual_tags
            
            # Add language info to optimization notes
            if 'optimization_notes' in optimization:
                optimization['optimization_notes'] += f"\n\nLanguage: Enhanced for {original_language} audience with multilingual hashtags."

    return optimizations


# ============================================================================
# ITERATIVE OPTIMIZATION & DIVERSITY SELECTION
# ============================================================================

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate basic text similarity based on common words
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def calculate_diversity_score(candidate: Dict, selected: List[Dict]) -> float:
    """
    Calculate how different a candidate is from already selected results
    
    Args:
        candidate: Candidate optimization to evaluate
        selected: List of already selected optimizations
        
    Returns:
        Diversity score (higher = more diverse)
    """
    if not selected:
        return 1.0
    
    candidate_title = candidate.get('optimized_title', '').lower()
    candidate_desc = candidate.get('optimized_description', '').lower()
    
    min_similarity = float('inf')
    
    for selected_item in selected:
        selected_title = selected_item.get('optimized_title', '').lower()
        selected_desc = selected_item.get('optimized_description', '').lower()
        
        # Calculate similarity
        title_similarity = calculate_text_similarity(candidate_title, selected_title)
        desc_similarity = calculate_text_similarity(candidate_desc, selected_desc)
        
        # Combined similarity (weighted toward title differences)
        combined_similarity = (title_similarity * 0.7) + (desc_similarity * 0.3)
        min_similarity = min(min_similarity, combined_similarity)
    
    # Return diversity score (1 - similarity)
    return 1.0 - min_similarity


def select_diverse_optimizations(all_results: List[Dict], max_results: int = 3) -> List[Dict]:
    """
    Select diverse optimization results to provide variety
    
    Args:
        all_results: List of all optimization results
        max_results: Maximum number of results to select
        
    Returns:
        List of diverse optimization results
    """
    if len(all_results) <= max_results:
        return all_results
    
    selected = []
    remaining = all_results.copy()
    
    # Always include highest scored result first
    if remaining:
        best_scored = max(remaining, key=lambda x: x.get('optimization_score', 0))
        selected.append(best_scored)
        remaining.remove(best_scored)
    
    # Select remaining results based on diversity
    while len(selected) < max_results and remaining:
        best_candidate = None
        max_diversity_score = -1
        
        for candidate in remaining:
            diversity_score = calculate_diversity_score(candidate, selected)
            if diversity_score > max_diversity_score:
                max_diversity_score = diversity_score
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
    
    return selected


def select_best_combined_optimizations(all_results: List[Dict], max_results: int = 3) -> List[Dict]:
    """
    Select optimizations based on both score and diversity
    
    Args:
        all_results: List of all optimization results
        max_results: Maximum number to select
        
    Returns:
        List of best combined optimizations
    """
    if len(all_results) <= max_results:
        return all_results
    
    scored_results = []
    
    for result in all_results:
        optimization_score = result.get('optimization_score', 0)
        
        # Calculate diversity from all other results
        diversity_scores = []
        for other_result in all_results:
            if other_result != result:
                diversity = calculate_diversity_score(result, [other_result])
                diversity_scores.append(diversity)
        
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.5
        
        # Combined score: 70% optimization quality, 30% diversity
        combined_score = (optimization_score * 0.7) + (avg_diversity * 0.3)
        
        scored_results.append({
            **result,
            'combined_score': combined_score,
            'diversity_component': avg_diversity
        })
    
    # Sort by combined score and select top
    scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
    return scored_results[:max_results]


def run_iterative_optimization(
    optimization_func,
    iterations: int = 3,
    selection_criteria: str = "best_score"
) -> List[Dict]:
    """
    Run optimization multiple times and select the best results
    
    Args:
        optimization_func: Function that returns optimization results
        iterations: Number of optimization runs to perform
        selection_criteria: How to select best results ('best_score', 'diversity', 'combined')
        
    Returns:
        Best optimization results from multiple runs
    """
    all_results = []
    
    for i in range(iterations):
        logger.info(f"Running optimization iteration {i+1}/{iterations}")
        
        try:
            # Run optimization with slight temperature variation for diversity
            temperature_adjustment = i * 0.05
            results = optimization_func(temperature_adjustment=temperature_adjustment)
            
            if results:
                # Add iteration metadata
                for result in results:
                    result['iteration'] = i + 1
                    result['temperature_used'] = 0.7 + temperature_adjustment
                
                all_results.extend(results)
                logger.info(f"Iteration {i+1} generated {len(results)} optimization variations")
        
        except Exception as e:
            logger.error(f"Error in optimization iteration {i+1}: {e}")
            continue
    
    if not all_results:
        logger.warning("No successful optimization results from any iteration")
        return []
    
    # Select best results based on criteria
    if selection_criteria == "best_score":
        # Sort by optimization score and take top 3
        scored_results = [r for r in all_results if 'optimization_score' in r]
        if scored_results:
            scored_results.sort(key=lambda x: x.get('optimization_score', 0), reverse=True)
            selected_results = scored_results[:3]
        else:
            selected_results = all_results[:3]
    
    elif selection_criteria == "diversity":
        # Select diverse results
        selected_results = select_diverse_optimizations(all_results, max_results=3)
    
    elif selection_criteria == "combined":
        # Combine score and diversity
        selected_results = select_best_combined_optimizations(all_results, max_results=3)
    
    else:
        # Default to first 3
        selected_results = all_results[:3]
    
    logger.info(f"Selected {len(selected_results)} best optimizations from {len(all_results)} variations")
    
    # Add selection metadata
    for i, result in enumerate(selected_results):
        result['final_rank'] = i + 1
        result['selection_method'] = selection_criteria
    
    return selected_results


def is_valid_optimization_result(result: Dict) -> bool:
    """
    Check if an optimization result has required fields and no placeholders
    
    Args:
        result: Optimization result dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["optimized_title", "optimized_description", "optimized_tags"]

    # Check all required fields present and non-empty
    for field in required_fields:
        if field not in result or not result[field]:
            logger.warning(f"Missing or empty required field: {field}")
            return False

    # Check optimized_tags is a list
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


def analyze_past_optimizations(
    past_optimizations: List[Dict[str, Any]],
    video_data: Dict[str, Any],
    channel_subscriber_count: int,
    analytics_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze past optimizations to determine their effectiveness
    
    Args:
        past_optimizations: List of previous optimizations for this video
        video_data: Dictionary containing video information
        channel_subscriber_count: Number of subscribers for the channel
        analytics_data: Dictionary containing analytics metrics for the video
        
    Returns:
        Analysis results with effectiveness metrics
    """
    if not past_optimizations:
        return {
            "total_optimizations": 0,
            "avg_effectiveness": 0.0,
            "recommendation": "No past optimizations to analyze"
        }
    
    try:
        total_optimizations = len(past_optimizations)
        effectiveness_scores = []
        
        # Analyze each past optimization
        for optimization in past_optimizations:
            created_at = optimization.get('created_at')
            applied_at = optimization.get('applied_at')
            
            if not applied_at:
                continue
            
            # Calculate effectiveness based on view changes
            # This is a simplified analysis - could be enhanced with actual data
            try:
                # Get views before and after optimization if available
                pre_views = optimization.get('views_before', 0)
                post_views = optimization.get('views_after', 0)
                
                if pre_views > 0:
                    effectiveness = (post_views - pre_views) / pre_views
                    effectiveness_scores.append(effectiveness)
            except Exception as e:
                logger.warning(f"Error calculating effectiveness for optimization: {e}")
                continue
        
        # Calculate average effectiveness
        avg_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.0
        
        # Generate recommendation
        if avg_effectiveness > 0.1:
            recommendation = "Past optimizations showed positive impact. Continue optimizing."
        elif avg_effectiveness > 0:
            recommendation = "Past optimizations showed modest impact. Consider refining approach."
        else:
            recommendation = "Past optimizations showed limited impact. Review optimization strategy."
        
        return {
            "total_optimizations": total_optimizations,
            "analyzed_optimizations": len(effectiveness_scores),
            "avg_effectiveness": round(avg_effectiveness, 3),
            "recommendation": recommendation,
            "effectiveness_scores": effectiveness_scores
        }
        
    except Exception as e:
        logger.error(f"Error analyzing past optimizations: {e}")
        return {
            "total_optimizations": len(past_optimizations),
            "error": str(e),
            "recommendation": "Unable to analyze past optimizations"
        }


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
    model: str = "claude-3-5-haiku-20241022",
    max_retries: int = 2,
    user_id: Optional[int] = None,
    maintain_full_original_description: bool = True,
    prev_optimizations: List[Dict] = []
) -> List[Dict]:
    """
    Generate comprehensive optimizations for a YouTube video
    
    FIXED: Includes fallback models, better error handling, and resource cleanup

    Args:
        original_title: The original video title
        original_description: The original video description
        original_tags: The original video tags
        transcript: Video transcript
        has_captions: Whether video has captions
        like_count: Number of likes
        comment_count: Number of comments
        optimization_decision_data: Optimization decision context
        analytics_data: Video analytics data
        competitor_analytics_data: Competitor analytics
        category_name: Video category
        model: The Claude model to use
        max_retries: Maximum retry attempts
        user_id: User ID for hashtag selection
        maintain_full_original_description: Keep original description
        prev_optimizations: Previous optimizations

    Returns:
        List of optimization variations
    """
    if not client:
        logger.error("Anthropic client not initialized.")
        return []

    transcript_length = len(transcript) if transcript else 0
    logger.info(f"Transcript length: {transcript_length}")

    # Handle multi-language content
    original_language = "en"
    translated_transcript = transcript or ""
    translated_title = original_title
    translated_description = original_description
    
    if transcript and len(transcript) > 100:
        # Detect and translate transcript
        transcript_translation = detect_and_translate_content(transcript, "transcript")
        original_language = transcript_translation["detected_language"]
        translated_transcript = transcript_translation["translated_text"]
        
        # If translated transcript, also translate title and description
        if transcript_translation["needs_translation"]:
            logger.info(f"Detected non-English content ({original_language}), translating for optimization")
            
            title_translation = detect_and_translate_content(original_title, "title")
            translated_title = title_translation["translated_text"]
            
            if original_description:
                desc_translation = detect_and_translate_content(original_description, "description")
                translated_description = desc_translation["translated_text"]

    tags_str = ", ".join(original_tags) if original_tags else "No tags available"

    # Process keywords and trends
    trending_keywords, competitor_trending_keywords, combined_trend_keywords_data = _process_keywords_and_trends(
        original_title, original_description, original_tags, transcript,
        translated_title, translated_description, translated_transcript,
        competitor_analytics_data, category_name
    )

    # Optimize hashtags
    optimized_hashtags = _optimize_hashtags(
        combined_trend_keywords_data, original_title, original_description,
        original_tags, transcript, category_name, trending_keywords,
        trending_keywords, competitor_analytics_data, user_id
    )

    # Define fallback models
    fallback_models = [
        model,  # Original model first
        "claude-3-5-haiku-20241022",  # Fastest as fallback
        "claude-3-5-sonnet-20241022"   # Alternative
    ]
    
    # Try multiple models and retries
    for model_attempt, current_model in enumerate(fallback_models):
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} with {current_model}")

                temperature = 0.7 + (attempt * 0.1)

                system_message = """You are a YouTube SEO expert who specializes in optimizing videos for maximum engagement, 
                searchability and audience retention. You understand what makes content perform well and how to optimize 
                metadata to increase views, watch time, and subscriber conversion.
                
                IMPORTANT: Your response MUST be valid JSON with properly escaped characters."""

                # Build the optimization prompt
                prompt = _build_optimization_prompt(
                    original_title, original_description, tags_str, translated_transcript,
                    original_language, trending_keywords, optimized_hashtags, category_name,
                    like_count, comment_count, optimization_decision_data,
                    maintain_full_original_description, attempt
                )

                tools = [
                    {
                        "name": "provide_comprehensive_video_optimizations",
                        "description": "Provides three distinct sets of optimized title, description, and tags for a YouTube video.",
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
                                                "description": "A list of optimized video tags.",
                                                "items": {"type": "string"}
                                            },
                                            "optimization_notes": {
                                                "type": "string",
                                                "description": "Detailed explanation of the changes made."
                                            },
                                            "optimization_score": {
                                                "type": "number",
                                                "description": "Score from 0.0 to 1.0.",
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
                    model=current_model,
                    max_tokens=4096,
                    temperature=temperature,
                    system=system_message,
                    tools=tools,
                    messages=[{"role": "user", "content": prompt}]
                )

                generated_optimizations = None
                for content in response.content:
                    if content.type == "tool_use" and content.name == "provide_comprehensive_video_optimizations":
                        generated_optimizations = content.input
                        if generated_optimizations:
                            # Process and enhance the optimization response
                            optimizations = _process_optimization_response(
                                generated_optimizations, translated_transcript, original_title,
                                original_description, optimized_hashtags, original_language,
                                category_name, prev_optimizations
                            )
                            return optimizations

            except Exception as e:
                error_message = str(e)
                logger.warning(f"Model {current_model}, attempt {attempt+1} failed: {error_message}")
                
                # Check for specific error types
                if "overloaded" in error_message.lower() or "529" in error_message:
                    logger.warning(f"Model {current_model} is overloaded, trying next model...")
                    break  # Break retry loop and try next model
                elif "rate_limit" in error_message.lower() or "429" in error_message:
                    logger.warning(f"Rate limit hit on {current_model}, waiting...")
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"Unexpected error with {current_model}: {error_message}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        break
        
        # If successful, break out of model loop
        if 'generated_optimizations' in locals() and generated_optimizations:
            break
    
    # If all attempts failed
    logger.error("All optimization attempts failed across all models")
    return []


# ============================================================================
# CHANNEL OPTIMIZATION
# ============================================================================

def get_channel_optimization(
    channel_title: str,
    description: str = "",
    keywords: str = "",
    recent_videos_data: str = "",
    model: str = "claude-3-7-sonnet-20250219",
    max_retries: int = 3
) -> Dict:
    """
    Generate optimizations for a YouTube channel
    
    FIXED: Better error handling and validation

    Args:
        channel_title: The channel title
        description: The current channel description
        keywords: The current channel keywords
        recent_videos_data: Data about recent videos
        model: The Claude model to use
        max_retries: Maximum retry attempts

    Returns:
        dict: Contains optimized description, keywords and notes
    """
    if not client:
        logger.error("Anthropic client not initialized. Using fallback.")
        return fallback_channel_optimization(channel_title, description, keywords)

    retry_errors = []

    # Try multiple times
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt} for channel optimization")

            temperature = 0.7 + (attempt * 0.1)

            system_message = """You are a YouTube SEO expert who specializes in optimizing YouTube channels 
            for maximum discovery, engagement, and brand consistency.
            
            IMPORTANT: Your response MUST be valid JSON with properly escaped characters."""

            json_emphasis = ""
            if attempt > 0:
                json_emphasis = """
                EXTREMELY IMPORTANT: Format your response as VALID JSON.
                Do not include ANY control characters that would break JSON parsing.
                """

            recent_videos_section = ""
            if recent_videos_data:
                recent_videos_section = f"""
                The channel has the following video samples:
                
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
            - Create a compelling, scannable description (250-500 characters)
            - Optimize first 100-150 characters for search
            - Include clear statement of what the channel is about
            - Add upload frequency if known
            - Include calls to action
            - Use formatting like line breaks and bullet points
            - Use emojis strategically
            - Avoid placeholder text
            
            For the KEYWORDS:
            - Include 10-15 highly relevant long-tail keywords (3-7 words each)
            - Format as space-separated keywords with double quotes escaped
            - Target specific search intents
            - Keep total length under 500 characters
            
            {json_emphasis}
            
            Return as JSON:
            {{
                "optimized_description": "Your optimized description",
                "optimized_keywords": "Your optimized keywords",
                "optimization_notes": "Explanation of changes"
            }}
            """

            message = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=temperature,
                system=system_message,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text

            # Try parsing as JSON
            try:
                result = parse_json_with_pattern(response_text)

                if is_valid_channel_optimization_result(result):
                    logger.info("Successfully parsed channel optimization")
                    return {
                        "original_description": description,
                        "optimized_description": result.get("optimized_description", description),
                        "original_keywords": keywords,
                        "optimized_keywords": result.get("optimized_keywords", keywords),
                        "optimization_notes": result.get("optimization_notes", "No notes provided")
                    }
            except Exception as json_error:
                logger.error(f"JSON parsing failed: {json_error}")

            # Regex fallback
            desc_match = re.search(r'optimized_description[\s:"\']+([^"\']+)', response_text, re.IGNORECASE | re.DOTALL)
            keywords_match = re.search(r'optimized_keywords[\s:"\']+([^"\']+)', response_text, re.IGNORECASE | re.DOTALL)
            notes_match = re.search(r'optimization_notes[\s:"\']+([^"\']+)', response_text, re.IGNORECASE | re.DOTALL)

            optimized_description = desc_match.group(1).strip() if desc_match else description
            optimized_keywords = keywords_match.group(1).strip() if keywords_match else keywords
            notes = notes_match.group(1).strip()[:500] if notes_match else "No notes provided"

            # Check for placeholders
            if check_for_placeholders(optimized_description) or check_for_placeholders(optimized_keywords):
                logger.warning("Found placeholders, trying again")
                continue

            return {
                "original_description": description,
                "optimized_description": optimized_description,
                "original_keywords": keywords,
                "optimized_keywords": optimized_keywords,
                "optimization_notes": notes
            }

        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            retry_errors.append(f"Attempt {attempt+1}: {str(e)}")

    # All retries failed
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
        dict: Contains optimized description and keywords
    """
    optimization_notes = []
    optimized_description = description
    optimized_keywords = keywords

    # Basic description optimization
    if len(description) < 100:
        optimization_notes.append("Description is too short. Added more information.")
        channel_title_capitalized = ' '.join(word.capitalize() for word in channel_title.split())
        optimized_description = f"Welcome to {channel_title_capitalized}! 🌟\n\n{description}\n\nSubscribe and hit the notification bell 🔔!"

    # Add emojis if missing
    if "🔔" not in optimized_description and "🌟" not in optimized_description:
        optimization_notes.append("Added emojis to increase engagement.")
        if "subscribe" in optimized_description.lower():
            optimized_description = optimized_description.replace("Subscribe", "Subscribe 🔔")

    # Add line breaks
    if "\n" not in optimized_description:
        optimization_notes.append("Added line breaks for readability.")
        sentences = re.split(r'(?<=[.!?])\s+', optimized_description)
        if len(sentences) > 1:
            optimized_description = "\n\n".join(sentences)

    # Basic keywords optimization
    if not keywords or len(keywords) < 10:
        optimization_notes.append("Keywords missing or insufficient. Added basic keywords.")
        words = channel_title.lower().split()
        keyword_pieces = [f'"{word}"' for word in words if len(word) > 3]
        if channel_title:
            keyword_pieces.append(f'"{channel_title}"')
        keyword_pieces.extend(['"youtube channel"', '"new videos"', '"subscribe"'])
        optimized_keywords = " ".join(keyword_pieces)

    # Ensure keywords are in quotes
    if '"' not in optimized_keywords:
        optimization_notes.append("Added proper quotation marks to keywords.")
        words = re.split(r'\s+', optimized_keywords)
        quoted_words = [f'"{word}"' if not (word.startswith('"') and word.endswith('"')) else word for word in words if word]
        optimized_keywords = " ".join(quoted_words)

    # If no changes
    if not optimization_notes:
        optimization_notes.append("Channel settings appear to follow best practices.")

    notes_string = "\n".join(optimization_notes)

    return {
        "original_description": description,
        "optimized_description": optimized_description,
        "original_keywords": keywords,
        "optimized_keywords": optimized_keywords,
        "optimization_notes": notes_string
    }


def is_valid_channel_optimization_result(result: Dict) -> bool:
    """Check if channel optimization result is valid"""
    required_fields = ["optimized_description", "optimized_keywords"]

    for field in required_fields:
        if field not in result or not result[field]:
            logger.warning(f"Missing or empty field: {field}")
            return False

    if check_for_placeholders(result.get("optimized_description", "")):
        logger.warning("Found placeholders in optimized_description")
        return False

    if check_for_placeholders(result.get("optimized_keywords", "")):
        logger.warning("Found placeholders in optimized_keywords")
        return False

    return True


# ============================================================================
# OPTIMIZATION DECISION LOGIC
# ============================================================================

async def should_optimize_video(
    video_data: Dict[str, Any],
    channel_subscriber_count: int,
    analytics_data: Dict[str, Any],
    past_optimizations: List[Dict[str, Any]] = [],
    model: str = "claude-3-7-sonnet-20250219"
) -> Dict[str, Any]:
    """
    Uses LLM to determine if a video should be optimized
    
    FIXED: Better error handling and validation

    Args:
        video_data: Dictionary containing video information
        channel_subscriber_count: Number of subscribers
        analytics_data: Analytics metrics
        past_optimizations: Previous optimizations
        model: Claude model to use

    Returns:
        dict: Contains decision and reasoning
    """
    if not client:
        logger.error("Anthropic client not initialized")
        return {
            "should_optimize": False,
            "reasons": "Anthropic client not initialized",
            "confidence": 0.0
        }
        
    try:
        logger.info(f"Evaluating if video '{video_data.get('title', 'Unknown')}' should be optimized")

        past_optimizations_prompt = '\n'.join([
            f"- Date: {o.get('created_at', 'Unknown')}\n"
            f"  Title: {o.get('optimized_title', 'Not changed')}\n"
            f"  Description: {o.get('optimized_description', 'Not changed')}\n"
            f"  Tags: {', '.join(o.get('optimized_tags', [])) if isinstance(o.get('optimized_tags'), list) else 'Not changed'}"
            for o in past_optimizations
        ]) if past_optimizations else 'No previous optimizations'

        recent_performance_data_prompt = '\n'.join([
            f"{day.get('day', day.get('date', 'Unknown'))}: {day['views']} views, "
            f"{day.get('view_percentage', day.get('average_view_percentage', 0)):.1f}% watched"
            for day in analytics_data.get('timeseries_data', [])
        ])

        system_message = """You are a YouTube SEO expert AI assistant. Analyze video performance data 
        and decide if metadata should be optimized. Provide a clear YES or NO decision with reasoning 
        and confidence score (0.0-1.0). Respond strictly in JSON format."""

        prompt = f"""
        Analyze the video data and respond with YES or NO regarding whether 
        the video should be optimized based on its current performance.
    
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
        - Recent Performance: {recent_performance_data_prompt if recent_performance_data_prompt else 'No data'}
    
        PAST OPTIMIZATIONS:
        {past_optimizations_prompt}
    
        KEY DECISION FACTORS:
        1. Is the video underperforming?
        2. Are there SEO issues?
        3. Could updated metadata help?
        4. Has enough time passed since last optimization?
        5. Does engagement suggest improvements needed?
    
        Return JSON:
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
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = message.content[0].text
                logger.debug(f"Claude response: {response_text}")

                # Try to parse as JSON array
                json_match = re.search(r'\[[\s\S]*\]', response_text)
                if json_match:
                    json_str = json_match.group(0)
                    response_list = json.loads(json_str)

                    if isinstance(response_list, list) and len(response_list) > 0 and isinstance(response_list[0], dict):
                        response_data = response_list[0]
                        decision = response_data.get("decision", "NO").upper()
                        should_optimize = decision == "YES"
                        reasons = response_data.get("reasoning", "No reasoning provided")
                        confidence = float(response_data.get("confidence", 0.5))
                        logger.info(f"Decision: {decision}, Confidence: {confidence}")
                        
                        return {
                            "should_optimize": should_optimize,
                            "reasons": reasons,
                            "confidence": confidence
                        }
                    else:
                        logger.error(f"Invalid JSON structure: {json_str}")
                        raise ValueError("Invalid JSON structure")
                else:
                    # Try object format
                    logger.warning("No JSON array found, trying object format")
                    json_match_obj = re.search(r'{[\s\S]*}', response_text)
                    if json_match_obj:
                        json_str_obj = json_match_obj.group(0)
                        response_data = json.loads(json_str_obj)
                        
                        if isinstance(response_data, dict):
                            decision = response_data.get("decision", "NO").upper()
                            should_optimize = decision == "YES"
                            reasons = response_data.get("reasoning", "No reasoning")
                            confidence = float(response_data.get("confidence", 0.5))
                            logger.info(f"Decision: {decision}, Confidence: {confidence}")
                            
                            return {
                                "should_optimize": should_optimize,
                                "reasons": reasons,
                                "confidence": confidence
                            }
                        else:
                            logger.error(f"Parsed object not a dict: {json_str_obj}")
                            raise ValueError("Parsed object not a dict")
                    else:
                        logger.error(f"No JSON found in response: {response_text}")
                        raise ValueError("No JSON found")

            except Exception as e:
                logger.error(f"Error determining optimization: {e}")
                
    except Exception as e:
        logger.error(f"Error in should_optimize_video: {e}")

    return {
        "should_optimize": False,
        "reasons": "Error during optimization evaluation",
        "confidence": 0.0
    }


# ============================================================================
# STATISTICAL OPTIMIZATION (Advanced)
# ============================================================================

def fetch_video_analytics_timeseries(video_id: str, days: int = 30) -> List[float]:
    """
    Fetch daily view data from analytics system
    
    FIXED: Proper connection initialization and cleanup
    
    Args:
        video_id: YouTube video ID
        days: Number of days to fetch
        
    Returns:
        List of daily view counts
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # Get database video ID
            cursor.execute("""
                SELECT id, view_count, published_at FROM youtube_videos WHERE video_id = %s
            """, (video_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"Video {video_id} not found in database")
                return []
            
            db_video_id, current_views, published_at = result
            
            # Try to get daily analytics
            cursor.execute("""
                SELECT views, date 
                FROM video_analytics_daily 
                WHERE video_id = %s 
                AND date >= %s
                ORDER BY date DESC
                LIMIT %s
            """, (db_video_id, datetime.now() - timedelta(days=days), days))
            
            daily_data = cursor.fetchall()
            
            # If we have daily analytics, return that
            if daily_data:
                daily_views = [float(row[0]) for row in daily_data]
                return daily_views
            
            # Fallback: estimate from current data
            if published_at:
                days_since_upload = max((datetime.now() - published_at).days, 1)
                avg_daily = float(current_views) / days_since_upload
                estimated_days = min(days, days_since_upload)
                daily_views = [avg_daily] * estimated_days
                return daily_views
            
            return []
            
    except Exception as e:
        logger.error(f"Error fetching video analytics: {e}")
        return []
    finally:
        if conn:
            conn.close()


def calculate_optimization_impact(optimization_date: datetime, timeseries_data: List[Dict]) -> float:
    """Calculate actual uplift from an optimization using analytics data"""
    try:
        daily_data = {}
        for point in timeseries_data:
            date_str = point.get('date', '')
            if date_str:
                try:
                    day_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    views = point.get('views', 0)
                    daily_data[day_date] = float(views)
                except (ValueError, TypeError):
                    continue
        
        if not daily_data:
            return 0.0
        
        opt_date = optimization_date.date()
        
        # Define pre and post periods (7 days each)
        pre_start = opt_date - timedelta(days=7)
        pre_end = opt_date - timedelta(days=1)
        post_start = opt_date + timedelta(days=1)
        post_end = opt_date + timedelta(days=7)
        
        # Collect views
        pre_views = []
        current_date = pre_start
        while current_date <= pre_end:
            if current_date in daily_data:
                pre_views.append(daily_data[current_date])
            current_date += timedelta(days=1)
        
        post_views = []
        current_date = post_start
        while current_date <= post_end:
            if current_date in daily_data:
                post_views.append(daily_data[current_date])
            current_date += timedelta(days=1)
        
        if not pre_views or not post_views:
            return 0.0
        
        avg_pre = np.mean(pre_views)
        avg_post = np.mean(post_views)
        
        if avg_pre > 0:
            uplift = (avg_post - avg_pre) / avg_pre
            return max(-0.5, min(2.0, uplift))  # Cap between -50% and +200%
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating optimization impact: {e}")
        return 0.0


def load_video_optimization_state(video_id: str, analytics_data: Dict = None) -> Tuple[List[float], Optional[str], Optional[str]]:
    """
    Load video optimization metadata from database
    
    FIXED: Proper connection initialization and cleanup
    
    Args:
        video_id: YouTube video ID
        analytics_data: Optional analytics data for uplift calculation
        
    Returns:
        (prior_uplifts, last_opt_time, next_run_time)
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # Get database video ID
            cursor.execute("""
                SELECT id FROM youtube_videos WHERE video_id = %s
            """, (video_id,))
            
            result = cursor.fetchone()
            if not result:
                return [], None, None
            
            db_video_id = result[0]
            
            # Get past optimizations
            cursor.execute("""
                SELECT applied_at, created_at
                FROM video_optimizations 
                WHERE video_id = %s AND is_applied = TRUE
                ORDER BY applied_at DESC
                LIMIT 10
            """, (db_video_id,))
            
            optimization_history = cursor.fetchall()
            
            if not optimization_history:
                # Check if video is ready for first optimization
                cursor.execute("""
                    SELECT published_at FROM youtube_videos WHERE id = %s
                """, (db_video_id,))
                published_result = cursor.fetchone()
                
                if published_result:
                    published_at = published_result[0]
                    video_age = (datetime.utcnow() - published_at).days
                    
                    if video_age >= 2:
                        return [0.0], None, None
                
                return [], None, None
            
            # Get most recent optimization
            last_applied_at = optimization_history[0][0]
            
            # Calculate prior uplifts
            prior_uplifts = []
            
            if analytics_data and 'timeseries_data' in analytics_data:
                logger.info(f"Calculating actual uplifts for {len(optimization_history)} optimizations")
                timeseries_data = analytics_data['timeseries_data']
                
                for i, (applied_at, _) in enumerate(optimization_history):
                    actual_uplift = calculate_optimization_impact(applied_at, timeseries_data)
                    prior_uplifts.append(actual_uplift)
                    logger.debug(f"Optimization {i+1} at {applied_at}: {actual_uplift:.1%} uplift")
            else:
                logger.info(f"Using estimated uplifts for {len(optimization_history)} optimizations")
                for i in range(len(optimization_history)):
                    estimated_uplift = max(0.02, 0.08 * (0.7 ** i))
                    prior_uplifts.append(estimated_uplift)
            
            # Calculate next run time
            next_run_time = (last_applied_at + timedelta(days=7)).isoformat() if last_applied_at else None
            
            return (
                prior_uplifts,
                last_applied_at.isoformat() if last_applied_at else None,
                next_run_time
            )
            
    except Exception as e:
        logger.error(f"Error loading video optimization state: {e}")
        return [], None, None
    finally:
        if conn:
            conn.close()


def compute_baseline_stats(views: List[float]) -> Tuple[float, float, float]:
    """Compute baseline statistics from view data"""
    if not views:
        return 0.0, 0.0, 0.0
    
    mean_val = np.mean(views)
    std_val = np.std(views, ddof=1) if len(views) > 1 else 0.0
    cv = std_val / mean_val if mean_val > 0 else 0.0
    
    return float(mean_val), float(std_val), float(cv)


def compute_min_detectable_uplift(cv: float, alpha: float = 0.05) -> float:
    """Compute minimum detectable uplift based on coefficient of variation"""
    z_score = 1.96  # 95% confidence
    return 2 * z_score * cv


def compute_sample_size(z_score: float, effect_size: float, cv: float = 0.2) -> int:
    """Compute required sample size for detecting effect"""
    if effect_size == 0:
        return 30
    
    n = (2 * (z_score ** 2) * (cv ** 2)) / (effect_size ** 2)
    return max(7, min(30, int(n)))


def compute_measurement_window(sample_size: int, baseline_views: float) -> int:
    """Compute optimal measurement window"""
    if baseline_views >= 1000:
        return max(3, min(sample_size, 7))
    elif baseline_views >= 100:
        return max(7, min(sample_size, 14))
    else:
        return max(14, min(sample_size, 30))


def get_statistical_thresholds(view_velocity: float) -> Tuple[float, float]:
    """Get optimization thresholds based on view velocity"""
    if view_velocity >= 1000:
        return 0.04, 0.025
    elif view_velocity >= 200:
        return 0.05, 0.03
    else:
        return 0.06, 0.035


def compute_decay_factor(prior_uplifts: List[float]) -> float:
    """Compute decay factor based on historical performance"""
    if len(prior_uplifts) < 2:
        return 0.5
    
    log_sum = sum(math.log(max(uplift, 0.01)) for uplift in prior_uplifts)
    return math.exp(log_sum / len(prior_uplifts))


def statistical_should_optimize(
    video_id: str,
    current_analytics: Dict[str, Any] = {},
    force_recheck: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Statistical optimization decision engine
    
    FIXED: Uses provided analytics data instead of database calls
    
    Args:
        video_id: YouTube video ID
        current_analytics: Current analytics data
        force_recheck: Skip cooling-off period check
        
    Returns:
        Dict with optimization decision or None if in cooling-off
    """
    logger.info(f"Running statistical analysis for video: {video_id}")
    
    if not current_analytics or 'timeseries_data' not in current_analytics:
        logger.warning(f"No analytics data for video {video_id}")
        return {
            'decision': False,
            'confidence': 0.0,
            'reasoning': 'No analytics data provided'
        }
    
    # Load video state
    prior_uplifts, last_opt_time, next_run_time = load_video_optimization_state(video_id, current_analytics)
    
    now = datetime.utcnow()
    
    # Check cooling-off period
    if not force_recheck and next_run_time:
        next_run_dt = datetime.fromisoformat(next_run_time)
        if now < next_run_dt:
            logger.info(f"Video {video_id} in cooling-off period until {next_run_time}")
            return None
    
    # Extract view data
    timeseries_data = current_analytics.get('timeseries_data', [])
    if not timeseries_data:
        logger.warning(f"No timeseries data for video {video_id}")
        return {
            'decision': False,
            'confidence': 0.0,
            'reasoning': 'No timeseries data available'
        }
    
    daily_views = [float(point.get('views', 0)) for point in timeseries_data]
    
    if not daily_views:
        logger.warning(f"No view data for video {video_id}")
        return {
            'decision': False,
            'confidence': 0.0,
            'reasoning': 'No view data found'
        }
    
    # Use most recent 7 days as baseline
    DEFAULT_BASELINE_DAYS = 7
    adjusted_baseline = daily_views[-min(len(daily_views), DEFAULT_BASELINE_DAYS):]
    
    # Compute baseline statistics
    mu, sig, cv = compute_baseline_stats(adjusted_baseline)
    V_pre = mu  # Baseline view velocity
    
    if V_pre == 0:
        logger.warning(f"Zero baseline views for video {video_id}")
        return {
            'decision': True,
            'confidence': 0.9,
            'reasoning': 'Zero baseline performance detected'
        }
    
    # Compute statistical parameters
    Z_SCORE = 1.96  # 95% confidence
    E = compute_min_detectable_uplift(cv)
    M = compute_sample_size(Z_SCORE, E, cv)
    W = compute_measurement_window(M, V_pre)
    
    # Use recent data as post-optimization performance
    recent_data = daily_views[-min(len(daily_views), W):]
    V_post = float(np.mean(recent_data)) if recent_data else 0.0
    
    # Calculate observed uplift
    U_obs = (V_post - V_pre) / V_pre if V_pre > 0 else 0.0
    
    # Get thresholds
    T_min, G_min = get_statistical_thresholds(V_pre)
    
    # Compute decay factor
    decay_factor = compute_decay_factor(prior_uplifts)
    
    # Predict next gain
    if prior_uplifts:
        avg_historical_uplift = np.mean(prior_uplifts)
        G_next = avg_historical_uplift * decay_factor
    else:
        G_next = 0.05 * decay_factor
    
    # Make decision
    meets_uplift_threshold = U_obs >= T_min
    meets_gain_threshold = G_next >= G_min
    decision = meets_uplift_threshold and meets_gain_threshold
    
    # Calculate confidence
    confidence_factors = []
    
    if V_pre >= 1000:
        confidence_factors.append(0.9)
    elif V_pre >= 200:
        confidence_factors.append(0.7)
    else:
        confidence_factors.append(0.5)
    
    cv_confidence = max(0.2, 1.0 - min(cv, 1.0))
    confidence_factors.append(cv_confidence)
    
    if prior_uplifts:
        hist_confidence = min(0.9, len(prior_uplifts) / 10.0)
        confidence_factors.append(hist_confidence)
    
    data_quality_confidence = min(0.9, len(daily_views) / 14.0)
    confidence_factors.append(data_quality_confidence)
    
    confidence = np.mean(confidence_factors)
    
    # Build reasoning
    reasoning_parts = [
        f"Baseline views: {V_pre:.1f}/day",
        f"Recent views: {V_post:.1f}/day",
        f"Observed uplift: {U_obs:.1%}",
        f"Required uplift: {T_min:.1%}",
        f"Predicted gain: {G_next:.1%}",
        f"Required gain: {G_min:.1%}",
        f"Data points: {len(daily_views)} days"
    ]
    
    reasoning = f"Statistical {'criteria met' if decision else 'criteria not met'}: {'; '.join(reasoning_parts)}"
    
    logger.info(f"Statistical decision for {video_id}: {decision} (confidence: {confidence:.3f})")
    
    return {
        'decision': decision,
        'confidence': confidence,
        'reasoning': reasoning,
        'statistical_metrics': {
            'baseline_views': V_pre,
            'recent_views': V_post,
            'observed_uplift': U_obs,
            'required_uplift': T_min,
            'predicted_gain': G_next,
            'required_gain': G_min,
            'coefficient_variation': cv,
            'decay_factor': decay_factor,
            'measurement_window': W,
            'sample_size': M,
            'prior_optimizations': len(prior_uplifts),
            'data_points_count': len(daily_views),
            'total_views': current_analytics.get('summary', {}).get('total_views', 0)
        }
    }


async def enhanced_should_optimize_video(
    video_data: Dict[str, Any],
    channel_subscriber_count: int,
    analytics_data: Dict[str, Any] = {},
    past_optimizations: List[Dict[str, Any]] = [],
    use_statistical_analysis: bool = True,
    model: str = "claude-3-7-sonnet-20250219"
) -> Dict[str, Any]:
    """
    Enhanced optimization decision combining LLM and statistical analysis
    
    Args:
        video_data: Video metadata
        channel_subscriber_count: Channel subscriber count
        analytics_data: Analytics data
        past_optimizations: Historical optimizations
        use_statistical_analysis: Whether to use statistical analysis
        model: Claude model to use
        
    Returns:
        Combined optimization decision
    """
    try:
        logger.info(f"Enhanced optimization analysis for: {video_data.get('title', 'Unknown')}")

        # Run LLM analysis
        llm_decision = await should_optimize_video(
            video_data=video_data,
            channel_subscriber_count=channel_subscriber_count,
            analytics_data=analytics_data,
            past_optimizations=past_optimizations,
            model=model
        )

        if not use_statistical_analysis:
            return llm_decision

        # Run statistical analysis
        video_id = video_data.get('video_id')
        if not video_id:
            logger.warning("No video_id for statistical analysis")
            return llm_decision

        statistical_result = statistical_should_optimize(
            video_id=video_id,
            current_analytics=analytics_data
        )

        if statistical_result is None:
            return {
                'should_optimize': False,
                'confidence': 0.9,
                'reasons': 'Video in cooling-off period',
                'method': 'statistical_cooldown',
                'llm_analysis': llm_decision,
                'statistical_analysis': None
            }

        # Combine decisions
        llm_weight = 0.6
        stat_weight = 0.4

        combined_confidence = (
            llm_decision['confidence'] * llm_weight +
            statistical_result['confidence'] * stat_weight
        )

        llm_recommends = llm_decision['should_optimize']
        stat_recommends = statistical_result['decision']

        if llm_recommends and stat_recommends:
            final_decision = True
            method = 'hybrid_agreement'
            reasoning = "Both LLM and statistical analysis recommend optimization"
        elif llm_recommends and llm_decision['confidence'] > 0.85:
            final_decision = True
            method = 'llm_strong'
            reasoning = f"LLM strongly recommends (confidence: {llm_decision['confidence']:.2f})"
        elif stat_recommends and statistical_result['confidence'] > 0.85:
            final_decision = True
            method = 'statistical_strong'
            reasoning = f"Statistical analysis strongly recommends (confidence: {statistical_result['confidence']:.2f})"
        else:
            final_decision = False
            method = 'hybrid_reject'
            reasoning = f"Insufficient agreement: LLM={llm_recommends} ({llm_decision['confidence']:.2f}), Statistical={stat_recommends} ({statistical_result['confidence']:.2f})"

        return {
            'should_optimize': final_decision,
            'confidence': combined_confidence,
            'reasons': reasoning,
            'method': method,
            'llm_analysis': llm_decision,
            'statistical_analysis': statistical_result
        }

    except Exception as e:
        logger.error(f"Error in enhanced optimization analysis: {e}")
        return {
            'should_optimize': False,
            'confidence': 0.0,
            'reasons': f"Error: {e}",
            'method': 'error',
            'llm_analysis': None,
            'statistical_analysis': None
        }
