import requests
import time
import json
import os
import logging
from dotenv import load_dotenv
from pathlib import Path
import subprocess
import shutil
from google import genai
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from google.genai.types import (
    GenerateContentResponse,
    FileState,
    GenerateContentConfig
)
import cv2
import numpy as np
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI
import base64
import openai
from google.cloud import vision

client = vision.ImageAnnotatorClient()

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class SceneTime(BaseModel):
    timestamp_seconds: float = Field(..., description="The timestamp in seconds (e.g., 123.45) into the video for the suggested thumbnail frame.")
    description: str = Field(..., description="A brief explanation of why this specific moment is a strong candidate for a click-worthy YouTube thumbnail, considering factors like emotion, clarity, action, branding, and YouTube best practices.")

class SuggestedThumbnails(BaseModel):
    suggested_scenes: List[SceneTime] = Field(..., description="A list of suggested scenes with timestamps and descriptions for optimal YouTube thumbnails.")

class CompetitorThumbnailDescription(BaseModel):
    """Model for storing a single competitor thumbnail description."""
    video_id: str = Field(..., description="The ID of the competitor video")
    description: str = Field(..., description="Detailed analysis of the thumbnail")
    strengths: List[str] = Field(..., description="List of thumbnail strengths")
    weaknesses: List[str] = Field(..., description="List of potential weaknesses or improvements")
    suggested_improvements: List[str] = Field(..., description="Suggested improvements for the thumbnail")

class EvaluationMetric(BaseModel):
    """Model for individual evaluation metrics with score and feedback."""
    score: int = Field(..., ge=1, le=10, description="Numerical score from 1-10 for this metric")
    feedback: str = Field(..., description="Detailed feedback explaining the score")

class ThumbnailEvaluation(BaseModel):
    """Comprehensive evaluation of a YouTube thumbnail."""
    mobile_legibility: EvaluationMetric = Field(..., description="Evaluation of how well the thumbnail works on mobile devices")
    color_contrast: EvaluationMetric = Field(..., description="Assessment of color usage and contrast in the thumbnail")
    curiosity_gap: EvaluationMetric = Field(..., description="Evaluation of how well the thumbnail creates curiosity")
    emotional_alignment: EvaluationMetric = Field(..., description="How well the emotional tone matches the video content")
    content_relevance: EvaluationMetric = Field(..., description="Relevance of the thumbnail to the actual video content")
    text_visibility: EvaluationMetric = Field(..., description="Readability and effectiveness of any text in the thumbnail")
    composition: EvaluationMetric = Field(..., description="Overall composition, visual balance, and cinematic quality")
    emotional_impact: EvaluationMetric = Field(..., description="Strength of emotional response the thumbnail evokes")
    uniqueness_trend_fit: EvaluationMetric = Field(..., description="Originality and alignment with platform trends")
    seasonal_alignment: EvaluationMetric = Field(..., description="Relevance to current seasons or trends if applicable")
    overall_score: int = Field(..., ge=1, le=10, description="Overall score for the thumbnail (1-10)")
    overall_impression: str = Field(..., description="General impression and summary of the thumbnail's effectiveness")
    improvement_suggestions: List[str] = Field(default_factory=list, description="List of specific suggestions for improving the thumbnail")
    total_score: int = Field(..., ge=1, le=100, description="Total cumulative score out of 100 points")
    ctr_uplift_range: str = Field(..., description="Estimated range of CTR improvement this thumbnail might provide")
    confidence_level: str = Field(..., description="Confidence level in this evaluation (Low, Medium, High)")
    margin_compliance_violations: List[str] = Field(default_factory=list, description="List of elements too close to thumbnail margins")

def generate_content_from_file(
    file_path: str,
    prompt: str,
    model_name: str = "gemini-2.0-flash-001",
    generation_config: Optional[Dict[str, Any]] = None
) -> GenerateContentResponse | None:
    """
    Generate content using Gemini API with file upload and automatic fallback models.
    
    Args:
        file_path: Path to the file to upload
        prompt: The prompt to send to the model
        model_name: Primary model to use
        generation_config: Optional generation configuration
        
    Returns:
        GenerateContentResponse or None if all attempts fail
    """
    client_obj = None
    uploaded_file_name = None
    MAX_PROCESSING_TIME_SECONDS = 180
    POLL_INTERVAL_SECONDS = 5

    try:
        if not GEMINI_API_KEY:
            logging.error("GEMINI_API_KEY is not set in environment variables")
            return None

        client_obj = genai.Client(api_key=GEMINI_API_KEY)

        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None
        
        logging.info(f"Uploading file: {file_path} for model {model_name}")
        uploaded_file = client_obj.files.upload(file=file_path) 
        uploaded_file_name = uploaded_file.name
        logging.info(f"File uploaded: {uploaded_file_name}. Waiting for processing...")

        # Poll for file to become active
        time_waited = 0
        while time_waited < MAX_PROCESSING_TIME_SECONDS:
            processed_file = client_obj.files.get(name=uploaded_file_name)
            current_state = processed_file.state
            logging.info(f"File '{uploaded_file_name}' state: {current_state}")

            if current_state == FileState.ACTIVE:
                logging.info(f"File '{uploaded_file_name}' is now ACTIVE.")
                break
            elif current_state == FileState.FAILED:
                logging.error(f"File '{uploaded_file_name}' processing FAILED.")
                try:
                    client_obj.files.delete(name=uploaded_file_name)
                    logging.info(f"Deleted failed file: {uploaded_file_name}")
                except Exception as del_err:
                    logging.warning(f"Could not delete failed file: {del_err}")
                return None
            elif current_state != FileState.PROCESSING:
                logging.warning(f"File in unexpected state: {str(current_state)}")

            time.sleep(POLL_INTERVAL_SECONDS)
            time_waited += POLL_INTERVAL_SECONDS
        else:
            logging.error(f"Timeout waiting for file to become ACTIVE")
            try:
                client_obj.files.delete(name=uploaded_file_name)
                logging.info(f"Deleted timed-out file: {uploaded_file_name}")
            except Exception as del_err:
                logging.warning(f"Could not delete timed-out file: {del_err}")
            return None

        logging.info(f"Generating content with model '{model_name}'...")
        
        response_mime_type = "text/plain"
        response_schema = None
        
        if generation_config:
            response_mime_type = generation_config.get("response_mime_type", "text/plain")
            response_schema = generation_config.get("response_schema")

        # Define fallback models with clearer logic
        fallback_models = [model_name]
        default_fallbacks = [
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-pro-preview-06-05",
            "gemini-2.0-flash",
            "gemini-2.5-flash-preview-05-20"
        ]
        
        # Add fallbacks only if not already in list
        for fallback in default_fallbacks:
            if fallback not in fallback_models:
                fallback_models.append(fallback)
        
        response = None
        for model_idx, current_model in enumerate(fallback_models):
            logging.info(f"Trying model {model_idx + 1}/{len(fallback_models)}: {current_model}")
            
            for retry in range(3):
                try:
                    config_args = {
                        "response_mime_type": response_mime_type,
                    }
                    
                    if response_schema:
                        config_args["response_schema"] = response_schema
                    
                    # Add safety settings
                    safety_categories = [
                        'HARM_CATEGORY_HATE_SPEECH',
                        'HARM_CATEGORY_HARASSMENT',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                        'HARM_CATEGORY_DANGEROUS_CONTENT',
                        'HARM_CATEGORY_CIVIC_INTEGRITY'
                    ]
                    
                    config_args["safety_settings"] = [
                        {"category": cat, "threshold": "BLOCK_NONE"}
                        for cat in safety_categories
                    ]
                    
                    response = client_obj.models.generate_content(
                        model=current_model,
                        contents=[uploaded_file, prompt],
                        config=GenerateContentConfig(**config_args)
                    )
                    
                    # Check if response was blocked or empty
                    if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                            logging.warning(f"Content blocked by {current_model}")
                            response = None
                            break
                    
                    if response and (not hasattr(response, 'candidates') or not response.candidates):
                        logging.warning(f"{current_model} returned empty candidates")
                        response = None
                        break
                    
                    logging.info(f"Successfully generated content with {current_model}")
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    logging.error(f"{current_model} attempt {retry + 1} failed: {e}")
                    
                    if "overloaded" in error_msg.lower() or "503" in error_msg:
                        logging.warning(f"{current_model} is overloaded")
                        break
                    
                    if retry < 2:
                        time.sleep(2 ** retry)
            
            if response:
                break
        
        if not response:
            logging.error(f"All models failed: {fallback_models}")
            return None

        # Clean up uploaded file
        if uploaded_file_name:
            try:
                client_obj.files.delete(name=uploaded_file_name)
                logging.info(f"Deleted uploaded file: {uploaded_file_name}")
            except Exception as e:
                logging.warning(f"Could not delete file: {e}")
        
        return response
        
    except Exception as e:
        logging.error(f"Error in generate_content_from_file: {e}")
        if client_obj and uploaded_file_name:
            try:
                client_obj.files.delete(name=uploaded_file_name)
                logging.info(f"Cleaned up file after error: {uploaded_file_name}")
            except Exception as cleanup_err:
                logging.warning(f"Could not cleanup file: {cleanup_err}")
        return None

def get_competitor_thumbnail_descriptions(
    competitor_analytics_data: Dict[str, Any],
    model_name: str = "gemini-2.0-flash",
    max_workers: int = 5
) -> Dict[str, Any]:
    """
    Generate thumbnail descriptions for competitor thumbnails in parallel.
    
    Args:
        competitor_analytics_data: Dictionary containing competitor video data with thumbnails          
        model_name: Name of the Gemini model to use for analysis
        max_workers: Maximum number of concurrent thumbnail analysis tasks
        
    Returns:
        Updated competitor_analytics_data with thumbnail descriptions
    """
    if not competitor_analytics_data or not competitor_analytics_data.get("competitor_videos"):
        return competitor_analytics_data
    
    competitor_videos = competitor_analytics_data["competitor_videos"]
    
    def process_single_thumbnail(video_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single competitor's thumbnail and generate analysis."""
        video_id = video_data.get("video_id")
        if not video_id or not video_data.get("thumbnail_urls"):
            return None
            
        # Get the highest quality thumbnail available
        thumbnail_url = (
            video_data["thumbnail_urls"].get("high", {}) or 
            video_data["thumbnail_urls"].get("medium", {}) or 
            video_data["thumbnail_urls"].get("default", {})
        ).get("url")
        
        if not thumbnail_url:
            return None
            
        # Prepare the prompt with video metrics
        metrics = [
            f"Title: {video_data.get('title', 'N/A')}",
            f"Description: {video_data.get('description', 'N/A')}",
            f"Tags: {video_data.get('tags', 'N/A')}",
            f"Views: {video_data.get('view_count', 0):,}",
            f"Likes: {video_data.get('like_count', 0):,}",
            f"Comments: {video_data.get('comment_count', 0):,}",
            f"Engagement Rate: {video_data.get('engagement_rate', 0):.2f}%"
        ]
        
        if video_data.get("days_since_upload") is not None:
            metrics.extend([
                f"Days Since Upload: {video_data['days_since_upload']}",
                f"Views Per Day: {video_data.get('views_per_day', 0):,.2f}"
            ])
        
        metrics_text = "\n".join(metrics)
        
        prompt = f"""
        Analyze this YouTube thumbnail and the provided video metrics. Generate a detailed analysis that includes:
        
        1. A description of the visual elements and composition
        2. The emotional appeal and messaging
        3. How the thumbnail likely contributes to the video's performance
        4. The most effective design patterns or techniques used
        5. Any potential weaknesses or areas for improvement
        
        Video Metrics:
        {metrics_text}
        
        Format your response as a JSON object with these exact keys:
        {{  
            "video_id": "{video_id}",
            "description": "Detailed analysis of the thumbnail",
            "strengths": ["strength 1", "strength 2", ...],
            "weaknesses": ["weakness 1", "weakness 2", ...],
            "suggested_improvements": ["suggestion 1", "suggestion 2", ...]
        }}
        """
        
        temp_file_path = None
        try:
            # Download the thumbnail image
            response = requests.get(thumbnail_url, timeout=10)
            response.raise_for_status()
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            # Generate content using Gemini
            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": CompetitorThumbnailDescription
            }
            
            gemini_response = generate_content_from_file(
                file_path=temp_file_path,
                prompt=prompt,
                model_name=model_name,
                generation_config=generation_config
            )
            
            if not gemini_response:
                return None
            
            try:
                parsed_response: CompetitorThumbnailDescription = gemini_response.parsed
                if not parsed_response.video_id:
                    parsed_response.video_id = video_id
                
                video_data['thumbnail_descriptions'] = parsed_response
                logging.info(f"Successfully analyzed thumbnail for video {video_id}")
                return video_data
            except (ValidationError, json.JSONDecodeError) as e:
                logging.error(f"Error parsing thumbnail analysis for {video_id}: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Error processing thumbnail for {video_id}: {e}")
            return None
        finally:
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logging.debug(f"Could not delete temp file {temp_file_path}: {e}")
    
    # Process thumbnails in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(process_single_thumbnail, video_data): video_data.get("video_id", idx)
            for idx, video_data in enumerate(competitor_videos)
        }
        
        for future in as_completed(future_to_video):
            video_id = future_to_video[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logging.error(f"Error in thumbnail processing future for {video_id}: {e}")
    
    return competitor_analytics_data


def do_thumbnail_optimization(
    video_id: str,
    original_title: str,
    original_description: str,
    original_tags: List[str],
    transcript: str,
    competitor_analytics_data: Dict[str, Any],
    category_name: str,
    user_id: int
):
    """
    Main function to optimize thumbnails for a video.
    
    Args:
        video_id: YouTube video ID
        original_title: Video title
        original_description: Video description
        original_tags: List of video tags
        transcript: Video transcript
        competitor_analytics_data: Competitor analysis data
        category_name: Video category
        user_id: User ID for file organization
        
    Returns:
        Best optimized thumbnail result or None on failure
    """
    try:
        logging.info(f"Starting thumbnail optimization for video {video_id}")
        competitor_analytics_data = get_competitor_thumbnail_descriptions(competitor_analytics_data)    

        local_video_filename = f"{video_id}.mp4"

        extracted_frames = download_youtube_video_and_extract_frames(video_id, local_video_filename)

        if not extracted_frames:
            logging.error("No frames were extracted from the video")
            return None

        def get_video_thumbnail_description(video_data: dict) -> str:
            """Extract thumbnail description from competitor video data."""
            try:
                thumbnail_description = video_data.get('thumbnail_descriptions', {})
                if not thumbnail_description:
                    return "No thumbnail description"
                
                return f"""
                Video Title: {video_data.get('title', 'No title')}
                Video Description: {video_data.get('description', 'No video description')}
                Thumbnail Description: {thumbnail_description.description}
                Thumbnail Strengths: {', '.join(thumbnail_description.strengths)}
                Thumbnail Weaknesses: {', '.join(thumbnail_description.weaknesses)}
                Thumbnail Suggested Improvements: {', '.join(thumbnail_description.suggested_improvements)}
                """
            except Exception as e:
                logging.error(f"Error getting video thumbnail description: {e}")
                return "No thumbnail description"
        
        # Prepare competitor insights
        competitor_insights = "\n\n".join(
            get_video_thumbnail_description(comp)
            for comp in competitor_analytics_data.get('competitor_videos', [])
        )

        def generate_thumbnail_transformation_prompt(
            frame_path: str,
            original_title: str,
            category_name: str,
            original_tags: List[str],
            transcript: str,
            competitor_insights: str
        ) -> Optional[str]:
            """
            Generate a detailed image transformation prompt using Gemini based on the frame and competitor insights.
            
            Args:
                frame_path: Path to the original frame
                original_title: Original video title
                category_name: Video category
                original_tags: List of video tags
                transcript: Video transcript
                competitor_insights: Insights from competitor thumbnails
                
            Returns:
                Generated transformation prompt or None if failed
            """
            try:
                system_prompt = """You are an expert YouTube thumbnail designer with deep knowledge of visual psychology and platform best practices. Your task is to analyze the provided frame and generate a detailed, specific prompt to transform it into a high-performing YouTube thumbnail.

                CORE CONSIDERATIONS:
                1. VIDEO CONTEXT: Title, category, and tags
                2. COMPETITOR ANALYSIS: Successful patterns and differentiators
                3. FRAME COMPOSITION: Current strengths and areas for improvement
                4. YOUTUBE BEST PRACTICES: Platform-specific optimizations

                EMOTIONAL TRIGGERS TO INCORPORATE:
                - Curiosity: Create information gaps that drive clicks
                - Intrigue: Add mysterious or surprising elements
                - Awe: Use scale or impressive visuals
                - Urgency: Imply time-sensitivity
                - Relatability: Show authentic human emotions

                YOUTUBE THUMBNAIL BEST PRACTICES:
                - Clear focal point visible at small sizes
                - High contrast for mobile viewing
                - Minimal text (3-5 words max if needed)
                - Brand consistency
                - Avoid bottom-right corner (timestamp area)
                - 16:9 aspect ratio optimization
                - Faces and emotions drive engagement
                - Use of directional cues to guide attention
                - DO NOT include hashtags
                """

                # Create the transcript section separately
                transcript_section = ""
                if transcript and len(transcript) > 10:
                    transcript_section = f"Consider key moments or themes from the transcript for contextual relevance and potential visual cues:\n{transcript[:300]}..."
                
                prompt = f"""
                You are tasked with analyzing this video frame and creating a precise prompt for an AI image generation model to transform it into a high-CTR YouTube thumbnail.
                Remember to leverage principles of visual psychology and YouTube platform best practices, including:
                - Clear focal point, high contrast, minimal text (3-5 words max if needed), brand consistency, avoiding the bottom-right timestamp area.
                - Prioritizing faces and emotions for engagement, and using directional cues.
                - Evoking emotions like curiosity (information gaps), intrigue (mysterious elements), awe (impressive visuals), urgency (time-sensitivity), or relatability (authentic human emotions).

                # CONTEXT
                Video Title: {original_title}
                Video Category: {category_name}
                Key Tags: {', '.join(original_tags) if original_tags else 'None provided'}

                # OBJECTIVE
                Create a detailed prompt that will guide an image generation model to transform this frame into a thumbnail that:
                1. Maximizes click-through rate (CTR)
                2. Stands out in YouTube's competitive feed
                3. Accurately represents the video content
                4. Creates curiosity without misleading viewers

                # TRANSFORMATION REQUIREMENTS
                - Output dimensions: 1920×1080 pixels (16:9 aspect ratio)
                - Photorealistic, high-quality appearance (DSLR-like)
                - Mobile-optimized visibility (clear at small sizes)
                - Vibrant colors that avoid YouTube UI clash (minimize dominant red/white/black)
                - Strong focal point with cinematic lighting
                - Rule of thirds composition with optimal negative space
                - Emotional impact through expressions, color, and composition
                - Text limited to 3-5 words maximum, if needed (must be extremely legible)
                - DO NOT include hashtags or other social media elements

                # ANALYZE FIRST, THEN CREATE
                First, analyze the provided frame for:
                - Current strengths (what to preserve or enhance)
                - Weaknesses (what to improve or change)
                - Subject positioning and focus (how to make it more compelling)
                - Lighting and color profile (opportunities for enhancement)
                - Emotional impact (current vs. desired)
                - Potential for creating curiosity or intrigue

                Then, review these competitor insights to understand what works in this niche and identify differentiation opportunities:
                {competitor_insights}

                {transcript_section}

                # OUTPUT INSTRUCTIONS
                Based on your analysis, create a detailed, specific prompt for an image generation model that will:

                1. DEFINE CORE TRANSFORMATION: Start with a clear, concise statement of what the thumbnail should depict and the overall desired mood/style.

                2. SPECIFY VISUAL ENHANCEMENTS:
                - Subject treatment: Detail specific adjustments to the main subject
                - Background modifications: Specify how to treat the background
                - Color adjustments: Be specific about the color palette and its application
                - Lighting effects: Describe the desired lighting setup
                - Depth adjustments: Specify how to create or enhance depth

                3. DETAIL TEXT OVERLAY (if applicable, max 3-5 words):
                - Exact words: Provide the precise text
                - Font specifications: Suggest font characteristics
                - Positioning and effects: Describe placement and visual treatments for legibility

                4. ADDRESS EMOTIONAL IMPACT:
                - Primary emotion to evoke
                - Visual cues for emotion
                - Facial expression adjustments (if people are present)

                5. ENSURE TECHNICAL QUALITY & DIFFERENTIATION:
                - Mobile visibility optimizations
                - Feed differentiation elements
                - Professional polish specifications

                Your output should be ONLY the transformation prompt, ready to be sent directly to an image generation model. Be extremely specific with technical details while maintaining clarity. Focus on measurable enhancements rather than vague directives.
                """
                
                # Generate the transformation prompt
                response = generate_content_from_file(
                    file_path=frame_path,
                    model_name="gemini-2.5-pro-preview-05-06",
                    prompt=prompt
                )
                
                if not response:
                    logging.error(f"Failed to generate transformation prompt for {frame_path}")
                    return None
                
                # Clean and return the generated prompt
                generated_prompt = response.text.strip()
                if not generated_prompt:
                    raise ValueError("Empty prompt generated")
                    
                return generated_prompt
                
            except Exception as e:
                logging.error(f"Error generating transformation prompt: {str(e)}")
                return None
        
        # Process frames in parallel
        results = []
        with ThreadPoolExecutor(max_workers=min(5, len(extracted_frames))) as executor:
            future_to_frame = {
                executor.submit(
                    process_single_thumbnail,
                    frame_path,
                    generate_thumbnail_transformation_prompt(
                        frame_path, original_title, category_name, 
                        original_tags, transcript, competitor_insights
                    ),
                    original_title,
                    original_description,
                    user_id
                ): frame_path
                for frame_path in extracted_frames
            }
            
            for future in as_completed(future_to_frame):
                frame_path = future_to_frame[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logging.error(f"Error processing frame {frame_path}: {e}")

        if not results:
            raise ValueError("No optimized thumbnails generated")

        # Sort by evaluation score
        results.sort(
            key=lambda r: getattr(r.get('evaluation', {}), 'total_score', 0),
            reverse=True
        )

        # Filter for compliance
        final_results = [
            r for r in results 
            if len(getattr(r.get('evaluation', {}), 'margin_compliance_violations', [])) < 2
        ]

        if not final_results:
            logging.warning("No thumbnails met compliance requirements, using best available")
            final_results = results

        return final_results[0] if final_results else None

    except Exception as e:
        logging.error(f"Error in thumbnail optimization: {e}")
        return None

def process_single_thumbnail(
    frame_path: str,
    prompt: str,
    video_title: str,
    video_description: str,
    user_id: int
) -> Optional[Dict]:
    """
    Process a single thumbnail frame with OpenAI and evaluate it.
    
    Args:
        frame_path: Path to the original frame
        prompt: Transformation prompt
        video_title: Video title for context
        video_description: Video description for context
        user_id: User ID for file organization
        
    Returns:
        Dictionary with thumbnail data or None on failure
    """
    try:
        output_dir = f"user_uploads/{user_id}/thumbnails"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        output_path = os.path.join(
            output_dir,
            f"optimized_{Path(frame_path).stem}_{timestamp}.jpg"
        )
        
        result_path = optimize_thumbnail_with_openai(
            input_image_path=frame_path,
            output_image_path=output_path,
            video_title=video_title,
            custom_prompt_instructions=prompt
        )
        
        if not result_path or not os.path.exists(result_path):
            logging.error(f"Failed to generate thumbnail for {frame_path}")
            return None
        
        evaluation = evaluate_thumbnail_with_gemini(
            image_path=result_path,
            prompt=prompt,
            video_title=video_title,
            video_description=video_description
        )
        
        if not evaluation:
            logging.warning(f"Failed to evaluate thumbnail {result_path}")
            return None
        
        return {
            "original_frame": frame_path,
            "optimized_thumbnail": result_path,
            "prompt": prompt,
            "evaluation": evaluation,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logging.error(f"Error in process_single_thumbnail: {e}")
        return None

def evaluate_thumbnail_with_gemini(
    image_path: str,
    prompt: str,
    video_title: str,
    video_description: str
) -> Optional[ThumbnailEvaluation]:
    """
    Evaluate a thumbnail using Gemini for comprehensive analysis.
    
    Args:
        image_path: Path to the thumbnail image
        prompt: The prompt used to generate the thumbnail
        video_title: Video title for context
        video_description: Video description for context
        
    Returns:
        ThumbnailEvaluation object or None on failure
    """
    try:
        evaluation_prompt = f"""
        Please analyze this YouTube thumbnail and provide a detailed evaluation based on the following criteria:
        
        1. Mobile Legibility:
           - Is the main subject clearly visible on small screens?
           - Is there enough contrast for elements to be distinguishable?
        
        2. Color and Contrast:
           - Are the colors vibrant and eye-catching?
           - Is there sufficient contrast between elements?
           - Does the color scheme avoid YouTube UI colors (red, white, black dominance)?
           - Is the color palette cohesive and optimized for platform differentiation?
        
        3. Curiosity Gap:
           - Does the thumbnail create intrigue or curiosity?
           - Would it make someone want to click to learn more?
           - Does it tease content without revealing everything?
        
        4. Emotional Alignment:
           - Does the thumbnail match the emotional tone of the title and description?
           - Title: {video_title}
           - Description: {video_description}
        
        5. Content Relevance:
           - Is the thumbnail semantically relevant to the video content?
           - Does it accurately represent what the video is about?
           - Is there a clear connection between visual elements and the title/topic?
        
        6. Text Visibility:
           - If there's text, is it large and readable on small screens?
           - Is there sufficient contrast between text and background?
           - Is the text placement optimal and limited to 3-5 impactful words?
        
        7. Composition:
           - Is the main subject properly framed and positioned?
           - Does it follow the rule of thirds?
           - Are important elements away from the bottom right corner (timestamp area)?
           - Is there cinematic depth of field and professional lighting?
        
        8. Emotional Impact:
           - What emotion does this thumbnail primarily evoke?
           - Is this appropriate for the video content?
           - How strong is the emotional resonance?
        
        9. Uniqueness & Trend Fit:
           - How original is this thumbnail compared to competitors?
           - Does it avoid generic stock poses and angles?
           - Does it incorporate current platform trends where appropriate?
        
        10. Seasonal or Trend Alignment:
           - Does the thumbnail reflect current seasons or trends if applicable?
           - Are there visual cues that connect to timely events or themes?
        
        Please provide:
        1. A score from 1-10 for each category
        2. Specific feedback for each category
        3. Overall impression
        4. Specific suggestions for improvement
        5. A total score out of 100 (sum of all categories)
        6. Estimated CTR uplift range (e.g., "5-10%", "10-15%", "15%+")
        7. Confidence level in this evaluation (Low, Medium, High)
        8. Margin Compliance Violations - Identify and report ALL of the following violations:
           - Any overlaid text graphics that are cut off or not SUFFICIENTLY readable
           - Any text overlays containing MORE than 7 words (excessive text violation)
           - Any inclusion of hashtags (# symbols) in the thumbnail image (hashtag violation)
        
        Return your response as a JSON object matching the ThumbnailEvaluation schema.
        """                    
        
        response = generate_content_from_file(
            file_path=image_path,
            prompt=evaluation_prompt,
            model_name="gemini-2.5-pro-preview-05-06",
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ThumbnailEvaluation
            }
        )
        
        if not response:
            logging.error("No response from Gemini for thumbnail evaluation")
            return None
            
        try:
            evaluation = response.parsed    
            return evaluation
        except Exception as e:
            logging.error(f"Failed to parse evaluation response: {e}")
            return None 
            
    except Exception as e:
        logging.error(f"Error in evaluate_thumbnail_with_gemini: {e}")
        return None

def get_suggested_thumbnail_times(
    video_file_path: str,
    num_suggestions: int = 5,
    model_name: str = "gemini-2.5-pro-preview-05-06"
) -> Optional[SuggestedThumbnails]:
    """
    Get suggested timestamps for optimal thumbnail frames using Gemini.
    
    Args:
        video_file_path: Path to the video file
        num_suggestions: Number of thumbnail suggestions to get
        model_name: Gemini model to use
        
    Returns:
        SuggestedThumbnails object or None on failure
    """
    prompt = f"""
    Analyze the provided video to identify the best {num_suggestions} moments for YouTube thumbnails.
    
    CRITICAL: These selected frames will be used as reference images for AI image generation to create the final optimized thumbnails. Therefore, visual clarity and detail definition are absolutely essential.
    
    For each moment, provide the exact timestamp in seconds (e.g., 45.67).
    Also, provide a concise description explaining why this frame would make a highly click-worthy thumbnail.
    
    Prioritize frames with EXCEPTIONAL visual quality and clarity:
    
    **FACE VISIBILITY REQUIREMENTS (if faces are present):**
    - Faces must be clearly visible, well-lit, and sharply defined
    - Face should occupy sufficient frame space (not tiny/distant)
    - Eyes should be clearly visible and preferably looking toward camera
    - Facial expressions should be distinct and engaging (surprise, joy, concentration, etc.)
    - Avoid faces that are partially obscured, in shadow, or at extreme angles
    
    **TECHNICAL CLARITY REQUIREMENTS:**
    - Frame must be sharp with high detail definition (essential for AI reference)
    - Excellent lighting with good contrast between subject and background
    - Clear distinction of edges, textures, and fine details
    - Avoid motion blur, camera shake, or focus issues
    - Rich color saturation and proper exposure
    
    **COMPOSITION AND ENGAGEMENT:**
    - Strong subject focus (face, product, key action, dramatic moment)
    - Compelling emotion, intrigue, or peak action moment
    - Visually appealing composition with clear focal points
    - Good contrast and color separation for AI processing
    - Minimal visual noise or distracting background elements
    
    **ADDITIONAL CONSIDERATIONS:**
    - Branding elements if present and clearly visible
    - Text elements only if they're crisp and legible
    - Moments that accurately represent video content while maximizing curiosity
    - Peak emotional or action moments that tell a story
    
    Remember: Since these frames will guide AI image generation, every visual element must be crystal clear and well-defined. Prefer slightly static moments over action if it means significantly better clarity.

    Return these suggestions formatted STRICTLY as a JSON object matching the SuggestedThumbnails schema.
    """

    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": SuggestedThumbnails,
    }

    logging.info(f"Requesting thumbnail suggestions for video: {video_file_path}")
    response = generate_content_from_file(
        file_path=video_file_path,
        prompt=prompt,
        model_name=model_name,
        generation_config=generation_config
    )

    if not response:
        logging.error("Failed to get response from Gemini for thumbnail suggestions")
        return None

    try:
        parsed_response: SuggestedThumbnails = response.parsed
        logging.info(f"Successfully parsed {len(parsed_response.suggested_scenes)} thumbnail suggestions")
        return parsed_response
    except (ValidationError, json.JSONDecodeError) as e:
        logging.error(f"Error parsing thumbnail suggestions: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing thumbnail suggestions: {e}")
        return None

def _calculate_frame_sharpness(image_path: str) -> float:
    """
    Calculates the sharpness of an image using the variance of the Laplacian.
    A higher value means a sharper image.

    Args:
        image_path: Path to the image file.

    Returns:
        The sharpness score (float). Returns 0.0 if the image cannot be read or is invalid.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Could not read image for sharpness calculation: {image_path}")
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute the Laplacian of the grayscale image and return the variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    except Exception as e:
        logging.error(f"Error calculating sharpness for {image_path}: {e}")
        return 0.0

def extract_frames_at_suggested_times(
    video_file_path: str,
    num_suggestions_for_llm: int = 5,
    output_directory: str = "extracted_thumbnails",
    base_filename_prefix: Optional[str] = None,
    search_window_seconds: float = 0.5,
    num_candidate_frames: int = 10
) -> List[str]:
    """
    Gets thumbnail suggestions, then for each suggestion, analyzes a small window of frames
    to pick the sharpest one using Laplacian variance, and saves it.

    Args:
        video_file_path: Path to the local video file.
        num_suggestions_for_llm: Number of thumbnail suggestions to request from the LLM.
        output_directory: Directory where extracted thumbnail JPEGs will be saved.
        base_filename_prefix: Optional prefix for output filenames. Video stem if None.
        search_window_seconds: Half-width of the window to search for sharp frames.
        num_candidate_frames: How many frames to extract and evaluate within the search window.

    Returns:
        A list of absolute paths to the successfully extracted and saved sharp thumbnail image files.
    """
    if not os.path.exists(video_file_path):
        logging.error(f"Video file not found for frame extraction: {video_file_path}")
        return []

    logging.info(f"Getting thumbnail suggestions for: {video_file_path}")
    suggested_thumbnails_data: SuggestedThumbnails = get_suggested_thumbnail_times(
        video_file_path=video_file_path,
        num_suggestions=num_suggestions_for_llm,
    )

    if not suggested_thumbnails_data or not suggested_thumbnails_data.suggested_scenes:
        logging.warning(f"No thumbnail suggestions received from LLM for {video_file_path}")
        return []

    output_path_obj = Path(output_directory).resolve()
    output_path_obj.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_path_obj}")

    extracted_final_frame_paths = []
    video_path_obj = Path(video_file_path)
    prefix = base_filename_prefix if base_filename_prefix else video_path_obj.stem

    for i, scene in enumerate(suggested_thumbnails_data.suggested_scenes):
        llm_timestamp = scene.timestamp_seconds
        if llm_timestamp < 0:
            logging.warning(f"Skipping invalid (negative) LLM timestamp: {llm_timestamp}s for scene {i+1}")
            continue

        logging.info(f"Processing LLM suggestion {i+1}: timestamp {llm_timestamp:.2f}s - {scene.description}")

        best_frame_path = None
        highest_combined_score = 0
        
        # Define the start and end of the search window
        window_start_time = max(0, llm_timestamp - search_window_seconds)
        window_end_time = llm_timestamp + search_window_seconds 
        
        if window_start_time >= window_end_time or num_candidate_frames <= 0:
            logging.warning(f"Invalid window or zero candidates for timestamp {llm_timestamp}, using exact time")
            candidate_timestamps = [llm_timestamp]
        else:
            candidate_timestamps = np.linspace(window_start_time, window_end_time, num_candidate_frames)

        # Create a temporary directory for candidate frames
        with tempfile.TemporaryDirectory(prefix=f"candidates_scene_{i+1}_") as temp_candidate_dir:
            temp_candidate_path_obj = Path(temp_candidate_dir)
            logging.info(f"Extracting {len(candidate_timestamps)} candidate frames for scene {i+1}")

            candidate_frame_files = []
            for cand_idx, current_timestamp in enumerate(candidate_timestamps):
                temp_frame_filename = temp_candidate_path_obj / f"candidate_{cand_idx:03d}_time_{current_timestamp:.3f}s.jpg"
                
                try:
                    cmd = [
                        "ffmpeg", "-nostdin",
                        "-ss", str(current_timestamp),
                        "-i", str(video_path_obj.resolve()),
                        "-frames:v", "1",
                        "-q:v", "2",
                        "-y",
                        str(temp_frame_filename)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)

                    if result.returncode == 0 and temp_frame_filename.exists() and temp_frame_filename.stat().st_size > 0:
                        candidate_frame_files.append(temp_frame_filename)
                    else:
                        logging.warning(f"ffmpeg failed for candidate at {current_timestamp:.3f}s")
                except subprocess.TimeoutExpired:
                    logging.error(f"ffmpeg timed out extracting candidate at {current_timestamp:.3f}s")
                except Exception as e:
                    logging.error(f"Error extracting candidate frame at {current_timestamp:.3f}s: {e}")
            
            if not candidate_frame_files:
                logging.warning(f"No candidate frames extracted for scene {i+1} at {llm_timestamp:.2f}s")
                continue

            logging.info(f"Analyzing {len(candidate_frame_files)} frames for sharpness and face quality")
            for cand_frame_path_obj in candidate_frame_files:
                sharpness = _calculate_frame_sharpness(str(cand_frame_path_obj))
                
                # Analyze faces in the frame
                face_analysis = analyze_frame_faces(str(cand_frame_path_obj))
                
                # Calculate combined score balancing sharpness and face quality
                combined_score = calculate_combined_frame_score(sharpness, face_analysis)

                if face_analysis:
                    logging.debug(f"Candidate {cand_frame_path_obj.name}: sharpness={sharpness:.2f}, "
                                f"face_score={face_analysis['overall_score']:.2f}, combined={combined_score:.2f}")
                else:
                    logging.debug(f"Candidate {cand_frame_path_obj.name}: sharpness={sharpness:.2f}, "
                                f"no faces, combined={combined_score:.2f}")
                
                if combined_score > highest_combined_score:
                    highest_combined_score = combined_score
                    best_frame_path = str(cand_frame_path_obj)

            if best_frame_path:
                logging.info(f"Best score for scene {i+1}: {highest_combined_score:.2f} from {Path(best_frame_path).name}")
                
                # Construct final output path
                timestamp_str = f"{Path(best_frame_path).stem.split('_time_')[-1].replace('s','').replace('.', '_')}"
                sane_description = ''.join(filter(str.isalnum, scene.description.replace(" ", "_")[:30]))
                output_filename = f"{prefix}_llm_sharp_thumb_{i+1:02d}_at_{timestamp_str}s_{sane_description}.jpg"
                final_output_file_path = output_path_obj / output_filename

                # Copy the best frame to final destination
                shutil.copy2(best_frame_path, final_output_file_path)
                if final_output_file_path.exists():
                    logging.info(f"Successfully saved best frame: {final_output_file_path}")
                    extracted_final_frame_paths.append(str(final_output_file_path))
                else:
                    logging.error(f"Failed to copy best frame to {final_output_file_path}")
            else:
                logging.warning(f"Could not determine best frame for scene {i+1}")

    if not extracted_final_frame_paths:
        logging.warning(f"No sharp frames extracted for {video_file_path}")
    
    return extracted_final_frame_paths


def optimize_thumbnail_with_openai(
    input_image_path: str,
    output_image_path: str,
    video_title: Optional[str] = None,
    custom_prompt_instructions: Optional[str] = None,
    model_name: str = "gpt-image-1",
    output_size: str = "1024x1024"
) -> Optional[str]:
    """
    Optimizes an image for a YouTube thumbnail using OpenAI's DALL-E image editing.

    Args:
        input_image_path: Path to the local input image file.
        output_image_path: Path where the optimized thumbnail image will be saved.
        video_title: Optional text (e.g., video title) for overlay.
        custom_prompt_instructions: Optional specific instructions for AI editor.
        model_name: The OpenAI model to use.
        output_size: The size of the generated image.

    Returns:
        Absolute path to the saved optimized image file, or None if error occurs.
    """
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY is not set in environment variables")
        return None

    if not os.path.exists(input_image_path):
        logging.error(f"Input image not found: {input_image_path}")
        return None

    try:
        client_obj = OpenAI()

        prompt = f"""
        
        IMAGE EDITING INSTRUCTIONS:
        {custom_prompt_instructions}
        
        ADDITIONAL INSTRUCTIONS:
        
        Thoroughly analyze the provided image, which is a frame from a video.
        Your task is to transform it into a highly engaging and click-compelling YouTube thumbnail based on the provided instructions.
        Preserve the main subject and the core essence of the original scene faithfully.
        Enhancements should make the thumbnail vibrant, clear, and professional for a YouTube audience.
        {f"If appropriate for the image, consider overlaying the text '{video_title}'. CRITICAL TEXT PLACEMENT REQUIREMENTS: The text MUST be positioned well within the image boundaries - at least 10% away from ALL edges (top, bottom, left, right) to ensure it's never cut off or cropped. The text MUST be bold, with EXTREMELY HIGH CONTRAST against its background to ensure absolute readability. It should be stylishly designed and strategically placed in the central 80% of the image for maximum impact and guaranteed visibility. Only add text if it significantly enhances the thumbnail and does not obscure critical visual elements. If the text cannot be made FULLY and perfectly clear, legible, and positioned safely away from all edges, do not add it." if video_title else ""}
        Boost colors and contrast to make the image pop, but maintain a natural look.
        Ensure the main focal point is exceptionally clear and sharp.
        Consider YouTube best practices for thumbnails: strong emotion, intrigue, clear subject, rule of thirds. DO NOT INCLUDE ANY HASHTAGS.
        The final image should be polished, high-resolution, and immediately grab attention on a crowded YouTube feed.
        Do not add any watermarks or signatures.
        
        CRITICAL: 
            1. The appearance of any people or characters in the original image MUST remain UNCHANGED. Do NOT alter facial features, expressions, body shape, or clothing. The goal is to enhance the *surrounding* visual elements and the overall composition, not the individuals themselves.
            2. TEXT POSITIONING: ANY AND ALL TEXT MUST BE ENTIRELY WITHIN THE SAFE ZONE - positioned at least 10% away from ALL image edges (top, bottom, left, right). The text should occupy only the central 80% area of the image to guarantee it's never cut off, cropped, or truncated. NO TEXT should touch or come close to any edge.
            3. TEXT READABILITY: ALL TEXT MUST BE ENTIRELY AND CLEARLY READABLE, with EXTREMELY HIGH CONTRAST against its background. The text must be large enough to read clearly even when the thumbnail is viewed at small sizes.
            4. YOU MAY ONLY ADD TEXT SPECIFICALLY MENTIONED IN THE IMAGE EDITING INSTRUCTIONS.
            
        The enhancements should primarily focus on visual elements such as colors, contrast, sharpness, and potentially overlaying text to grab the viewer's attention, similar to how a human thumbnail designer would enhance a thumbnail without distorting the core content.
        The goal is enhancement, not a radical transformation of the original image's subject or characters.
        """

        logging.info(f"OpenAI image optimization for '{Path(input_image_path).name}'")

        with open(input_image_path, "rb") as image_file_rb:
            response = client_obj.images.edit(
                model=model_name,
                image=image_file_rb,
                quality="high",
                prompt=prompt,
                n=1,
            )

        if not response.data or not response.data[0].b64_json:
            logging.error("OpenAI API did not return image data")
            return None

        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Ensure output directory exists
        output_image_path_obj = Path(output_image_path)
        output_image_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Check file size and compress if needed to stay under YouTube's 2MB limit
        file_size_mb = len(image_bytes) / (1024 * 1024)
        logging.info(f"Generated image size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 2.0:
            logging.warning(f"Image size ({file_size_mb:.2f} MB) exceeds YouTube's 2MB limit. Compressing...")
            
            from PIL import Image
            import io
            
            # Load image from bytes
            img = Image.open(io.BytesIO(image_bytes))
            
            # Start with quality 85 and reduce until under 2MB
            quality = 85
            compressed_bytes = None
            
            while quality > 20:
                output_buffer = io.BytesIO()
                img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
                compressed_bytes = output_buffer.getvalue()
                compressed_size_mb = len(compressed_bytes) / (1024 * 1024)
                
                if compressed_size_mb <= 2.0:
                    logging.info(f"Compressed to {compressed_size_mb:.2f} MB with quality {quality}")
                    image_bytes = compressed_bytes
                    # Update output path to .jpg since we converted to JPEG
                    if output_image_path_obj.suffix.lower() == '.png':
                        output_image_path_obj = output_image_path_obj.with_suffix('.jpg')
                    break
                
                quality -= 10
            
            if len(image_bytes) / (1024 * 1024) > 2.0:
                logging.warning("Could not compress image below 2MB while maintaining reasonable quality")

        # Write the final file
        with open(output_image_path_obj, "wb") as f:
            f.write(image_bytes)
        
        # Verify the actual file size
        actual_size_mb = os.path.getsize(output_image_path_obj) / (1024 * 1024)
        resolved_output_path = str(output_image_path_obj.resolve())
        logging.info(f"Successfully saved optimized thumbnail to: {resolved_output_path} (Size: {actual_size_mb:.2f} MB)")
        
        return resolved_output_path

    except openai.APIError as e:
        error_details = f"OpenAI API error: Status {e.status_code if hasattr(e, 'status_code') else 'unknown'}"
        if hasattr(e, 'message') and e.message:
            error_details += f", Message: {e.message}"
        logging.error(error_details)
        return None
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI thumbnail optimization: {e}")
        return None


# --- Sieve API related constants and functions ---
SIEVE_API_KEY = os.getenv("SIEVE_API_KEY", "")
if not SIEVE_API_KEY:
    logging.warning("SIEVE_API_KEY is not set in environment variables")

BASE_URL = "https://mango.sievedata.com/v2"
PUSH_ENDPOINT = f"{BASE_URL}/push"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/jobs"
POLL_INTERVAL_SECONDS = 10

def submit_youtube_download_job(youtube_url: str, api_key: str) -> str | None:
    """
    Submits a job to the Sieve API to download a YouTube video.

    Args:
        youtube_url: The URL of the YouTube video to download.
        api_key: Your Sieve API key.

    Returns:
        The job ID if submission was successful, otherwise None.
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
    payload = {
        "function": "sieve/youtube-downloader",
        "inputs": {
            "url": youtube_url,
            "download_type": "video",
            "resolution": "lowest-available",
            "include_audio": True,
            "start_time": 0,
            "end_time": -1,
            "include_metadata": False,
            "include_subtitles": False,
            "video_format": "mp4",
            "audio_format": "mp3"
        }
    }

    try:
        logging.info(f"Submitting download job for URL: {youtube_url}")
        response = requests.post(PUSH_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()

        response_data = response.json()
        job_id = response_data.get("id")

        if job_id:
            logging.info(f"Job submitted successfully. Job ID: {job_id}")
            return job_id
        else:
            logging.error(f"Job submission failed. No 'id' in response: {response_data}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error submitting job: {e}")
        if e.response is not None:
            logging.error(f"Response status: {e.response.status_code}")
            logging.error(f"Response content: {e.response.text}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON response: {response.text}")
        return None


def poll_job_status(job_id: str, api_key: str) -> dict | None:
    """
    Polls the Sieve API for the status of a specific job until it completes or fails.

    Args:
        job_id: The ID of the job to poll.
        api_key: Your Sieve API key.

    Returns:
        The final job data dictionary if the job finished successfully,
        or None if the job failed or an error occurred during polling.
    """
    headers = {
        "X-API-Key": api_key,
    }
    status_url = f"{JOB_STATUS_ENDPOINT}/{job_id}"

    logging.info(f"Polling status for job ID: {job_id}")

    while True:
        try:
            response = requests.get(status_url, headers=headers)
            response.raise_for_status()

            job_data = response.json()
            status = job_data.get("status")

            logging.info(f"Job '{job_id}' status: {status}")

            if status == "finished":
                logging.info(f"Job '{job_id}' finished successfully")
                logging.info(f"Job Output: {json.dumps(job_data.get('outputs'), indent=2)}")
                return job_data
            elif status == "error":
                logging.error(f"Job '{job_id}' failed")
                logging.error(f"Job Error Details: {json.dumps(job_data.get('error'), indent=2)}")
                return None
            elif status in ["starting", "processing", "queued"]:
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                logging.warning(f"Job '{job_id}' has unknown status: {status}")
                time.sleep(POLL_INTERVAL_SECONDS)

        except requests.exceptions.RequestException as e:
            logging.error(f"Error polling job status for '{job_id}': {e}")
            if e.response is not None:
                logging.error(f"Response status: {e.response.status_code}")
                logging.error(f"Response content: {e.response.text}")
            time.sleep(POLL_INTERVAL_SECONDS * 2)
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON response during polling: {response.text}")
            time.sleep(POLL_INTERVAL_SECONDS * 2)
        except KeyboardInterrupt:
            logging.info("Polling interrupted by user")
            return None

def download_video_from_url(video_url: str, output_path: str) -> bool:
    """
    Downloads a video file from a given URL to a specified local path.

    Args:
        video_url: The URL of the video file to download.
        output_path: The local file path (including filename) to save the video.

    Returns:
        True if download was successful, False otherwise.
    """
    logging.info(f"Downloading video from: {video_url}")
    try:
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"Video downloaded successfully to: {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download video: {e}")
        return False
    except Exception as e:
        logging.error(f"An error occurred during video download: {e}")
        return False

def extract_thumbnails_ffmpeg(
    video_path: str,
    output_dir: str,
    num_thumbnails: int = 10,
    base_filename: str = "thumb"
) -> list[str] | None:
    """
    Extracts representative thumbnails from a video file using multiple approaches with fallbacks.
    
    Uses three methods in order of preference:
    1. Global thumbnail filter - Best quality but may fail on some videos
    2. Distributed thumbnail sampling - Uses the thumbnail filter on segments
    3. Fixed timestamp sampling - Simple and reliable fallback
    
    Args:
        video_path: Path to the input video file.
        output_dir: Directory where thumbnails will be saved.
        num_thumbnails: The desired number of thumbnails to extract.
        base_filename: The base name for the output thumbnail files.
        
    Returns:
        A list of paths to the generated thumbnail files, or None if all extraction methods failed.
    """
    if not Path(video_path).is_file():
        logging.error(f"Video file not found at: {video_path}")
        return None

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        logging.error("ffmpeg or ffprobe not found. Please ensure ffmpeg is installed")
        return None

    if num_thumbnails <= 0:
        logging.warning("Number of thumbnails must be positive. Defaulting to 1")
        num_thumbnails = 1

    # Create/clean output directory
    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    
    # Clean up any existing thumbnail files
    for old_file in output_path_obj.glob(f"{base_filename}_*.jpg"):
        try:
            old_file.unlink()
        except Exception as e:
            logging.debug(f"Could not delete old file {old_file}: {e}")
    
    # Try global thumbnail filter first
    logging.info("Trying global thumbnail filter approach...")
    extracted_paths = _extract_global_thumbnails(video_path, output_dir, num_thumbnails, base_filename)
    if extracted_paths and len(extracted_paths) == num_thumbnails:
        logging.info(f"Global thumbnail filter succeeded: {len(extracted_paths)} thumbnails")
        return extracted_paths
    
    # Try distributed thumbnail filter
    if not extracted_paths or len(extracted_paths) < num_thumbnails:
        logging.info("Trying distributed thumbnail approach...")
        extracted_paths = _extract_distributed_thumbnails(video_path, output_dir, num_thumbnails, base_filename)
        if extracted_paths and len(extracted_paths) == num_thumbnails:
            logging.info(f"Distributed thumbnail method succeeded: {len(extracted_paths)} thumbnails")
            return extracted_paths
    
    # Try fixed timestamp fallback
    if not extracted_paths or len(extracted_paths) < num_thumbnails:
        logging.info("Trying fixed timestamp fallback method...")
        extracted_paths = _extract_fixed_timestamps(video_path, output_dir, num_thumbnails, base_filename)
        if extracted_paths:
            logging.info(f"Fixed timestamp method extracted {len(extracted_paths)} thumbnails")
            return extracted_paths
    
    logging.error("All thumbnail extraction methods failed")
    return None


def _extract_global_thumbnails(
    video_path: str,
    output_dir: str,
    num_thumbnails: int,
    base_filename: str
) -> list[str] | None:
    """
    Extracts thumbnails using ffmpeg's global thumbnail filter.
    """
    output_path_obj = Path(output_dir)

    try:
        # Get video duration
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())

        if duration <= 0:
            raise ValueError(f"Invalid video duration: {duration}")

        # Calculate timestamps
        interval = duration / (num_thumbnails + 1)
        timestamps = [interval * (i + 1) for i in range(num_thumbnails)]

        extracted_paths = []
        for i, timestamp in enumerate(timestamps):
            output_file = output_path_obj / f"{base_filename}_{i + 1:03d}.jpg"

            window_start = max(0, timestamp - 1.5)
            window_duration = min(3.0, duration - window_start)

            cmd = [
                "ffmpeg", "-nostdin",
                "-ss", str(window_start),
                "-t", str(window_duration),
                "-i", video_path,
                "-vf", "thumbnail=1,scale=640:-1",
                "-frames:v", "1",
                "-q:v", "2",
                "-y",
                str(output_file)
            ]

            logging.info(f"Extracting thumbnail {i + 1}/{num_thumbnails} at {timestamp:.2f}s")
            subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

            if output_file.exists():
                extracted_paths.append(str(output_file))

        return extracted_paths if extracted_paths else None

    except (subprocess.CalledProcessError, ValueError) as e:
        logging.warning(f"Global thumbnail extraction failed: {e}")
        return None
    except Exception as e:
        logging.warning(f"Unexpected error in global extraction: {e}")
        return None

def _extract_distributed_thumbnails(
    video_path: str,
    output_dir: str,
    num_thumbnails: int,
    base_filename: str
) -> list[str] | None:
    """
    Extracts thumbnails by dividing the video into segments.
    """
    output_path_obj = Path(output_dir)
    
    try:
        # Get video duration
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        stream_data = json.loads(result.stdout)['streams'][0]
        
        duration = float(stream_data.get('duration', 0))
        if duration <= 0:
            raise ValueError(f"Invalid video duration: {duration}")
            
        # Calculate segment points
        segment_duration = duration / num_thumbnails
        segment_points = [i * segment_duration for i in range(num_thumbnails)]
        
        extracted_paths = []
        
        for i, start_time in enumerate(segment_points):
            output_file = output_path_obj / f"{base_filename}_{i+1:03d}.jpg"
            
            cmd = [
                "ffmpeg", "-nostdin",
                "-ss", str(start_time),
                "-t", str(segment_duration),
                "-i", video_path,
                "-vf", "thumbnail=1",
                "-frames:v", "1",
                "-q:v", "2",
                "-y",
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            
            if result.returncode == 0 and output_file.exists():
                extracted_paths.append(str(output_file))
            else:
                logging.warning(f"Failed to extract thumbnail for segment {i+1}")
                
        return extracted_paths if extracted_paths else None
        
    except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError) as e:
        logging.warning(f"Distributed thumbnail extraction failed: {e}")
        return None
    except Exception as e:
        logging.warning(f"Unexpected error in distributed extraction: {e}")
        return None

def _extract_fixed_timestamps(
    video_path: str,
    output_dir: str,
    num_thumbnails: int,
    base_filename: str
) -> list[str] | None:
    """
    Extracts thumbnails at fixed intervals.
    """
    output_path_obj = Path(output_dir)
    
    try:
        # Get video duration
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        
        if duration <= 0:
            raise ValueError(f"Invalid video duration: {duration}")
        
        # Calculate timestamps
        interval = duration / (num_thumbnails + 1)
        timestamps = [interval * (i + 1) for i in range(num_thumbnails)]
        
        extracted_paths = []
        
        for i, timestamp in enumerate(timestamps):
            output_file = output_path_obj / f"{base_filename}_{i+1:03d}.jpg"
            
            cmd = [
                "ffmpeg", "-nostdin",
                "-ss", str(timestamp),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                "-y",
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            
            if result.returncode == 0 and output_file.exists():
                extracted_paths.append(str(output_file))
            else:
                logging.warning(f"Failed to extract thumbnail at {timestamp:.2f}s")
        
        return extracted_paths if extracted_paths else None
        
    except subprocess.CalledProcessError as e:
        logging.warning(f"Fixed timestamp extraction failed: {e}")
        return None
    except Exception as e:
        logging.warning(f"Unexpected error in fixed timestamp extraction: {e}")
        return None

def clean_thumbnail_directory(output_dir, base_filename=None):
    """
    Thoroughly clean a thumbnail directory.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0

    count = 0
    if base_filename:
        pattern = f"{base_filename}_*.jpg"
        logging.info(f"Cleaning up files matching pattern: {pattern}")
        for file in output_path.glob(pattern):
            try:
                file.unlink()
                count += 1
            except Exception as e:
                logging.debug(f"Could not delete {file}: {e}")
    else:
        logging.info(f"Cleaning up ALL jpg files in {output_dir}")
        for file in output_path.glob("*.jpg"):
            try:
                file.unlink()
                count += 1
            except Exception as e:
                logging.debug(f"Could not delete {file}: {e}")

    logging.info(f"Deleted {count} files from {output_dir}")
    return count

def download_youtube_video_and_extract_frames(
    video_id: str,
    output_path: str,
    thumbnail_output_dir: str = "thumbnails"
) -> Optional[List[str]]:
    """
    Downloads a YouTube video using the Sieve API and extracts suggested frames.
    
    Args:
        video_id: The ID of the YouTube video to download.
        output_path: The local file path (including filename) to save the video.
        thumbnail_output_dir: Directory where thumbnails will be saved.
        
    Returns:
        A list of paths to the generated thumbnail files if successful, None otherwise.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    logging.info(f"Starting YouTube video download for: {video_url}")
    
    # Create thumbnail output directory
    os.makedirs(thumbnail_output_dir, exist_ok=True)
    
    # Clean up previous video file
    if os.path.exists(output_path):
        logging.info(f"Removing previous video file: {output_path}")
        try:
            os.remove(output_path)
        except OSError as e:
            logging.error(f"Error removing existing file: {e}")
            return None
    
    # Submit download job to Sieve
    submitted_job_id = submit_youtube_download_job(video_url, SIEVE_API_KEY)
    
    if not submitted_job_id:
        logging.error("Failed to submit YouTube download job to Sieve")
        return None
    
    # Poll for job completion
    final_job_data = poll_job_status(submitted_job_id, SIEVE_API_KEY)
    if not final_job_data:
        logging.error("YouTube download job failed or was interrupted")
        return None
    
    logging.info("Download process completed")
    
    # Extract video URL from job output
    download_url = None
    outputs = final_job_data.get("outputs", [])
    
    if isinstance(outputs, list) and len(outputs) > 0:
        output_data = outputs[0]
        download_url = output_data.get("data", {}).get("url")
    
    if not download_url:
        logging.error("Could not find video URL in Sieve job output")
        return None
    
    logging.info(f"Video URL: {download_url}")

    if not download_video_from_url(download_url, output_path):
        logging.error("Failed to download video from Sieve URL")
        return None
    
    logging.info(f"Successfully downloaded video to: {output_path}")
    
    # Extract frames using suggested times
    try:
        extracted_frames = extract_frames_at_suggested_times(video_file_path=output_path)
        
        if extracted_frames:
            logging.info(f"Successfully extracted {len(extracted_frames)} frames")
            return extracted_frames
        else:
            logging.warning("No frames were extracted")
            return []
            
    except Exception as e:
        logging.error(f"Error extracting frames: {e}")
        return None

def detect_faces(path):
    """Detects faces in an image using Google Cloud Vision API."""
    for attempt in range(3):
        try:
            with open(path, "rb") as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.face_detection(image=image)
            faces = response.face_annotations
            return faces
        except Exception as e:
            logging.warning(f"Face detection failed on attempt {attempt + 1}: {e}")
            time.sleep(2 ** (attempt + 1))

    return []

def calculate_face_prominence_score(face_data, image_width=1280, image_height=720):
    """
    Calculate overall face prominence score (0-1) based on multiple factors.
    
    Args:
        face_data: Google Vision API face detection result
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Dict with detailed scoring breakdown
    """
    try:
        # 1. Size scoring
        bounding_poly = face_data.bounding_poly.vertices
        face_width = bounding_poly[1].x - bounding_poly[0].x
        face_height = bounding_poly[2].y - bounding_poly[1].y
        face_area = face_width * face_height

        size_percentage = face_area / (image_width * image_height)
        size_score = min(1.0, size_percentage * 100)

        # 2. Direction scoring
        pan_score = max(0, 1 - abs(face_data.pan_angle) / 15.0)
        tilt_score = max(0, 1 - abs(face_data.tilt_angle) / 20.0)
        direction_score = (pan_score + tilt_score) / 2

        # 3. Quality scoring
        quality_score = (face_data.detection_confidence + face_data.landmarking_confidence) / 2

        blur_bonus = 0.1 if face_data.blurred_likelihood.name == "VERY_UNLIKELY" else 0
        exposure_bonus = 0.1 if face_data.under_exposed_likelihood.name == "VERY_UNLIKELY" else 0
        quality_score = min(1.0, quality_score + blur_bonus + exposure_bonus)

        # 4. Position scoring
        face_center_x = (bounding_poly[0].x + bounding_poly[1].x) / 2
        face_center_y = (bounding_poly[0].y + bounding_poly[2].y) / 2

        center_distance = ((face_center_x - image_width/2)**2 + (face_center_y - image_height/2)**2)**0.5
        max_distance = (image_width**2 + image_height**2)**0.5 / 2
        position_score = 1 - (center_distance / max_distance)

        # 5. Expression scoring
        expression_score = 0.5
        if face_data.joy_likelihood.name in ["LIKELY", "VERY_LIKELY"]:
            expression_score = 0.8
        elif face_data.surprise_likelihood.name in ["LIKELY", "VERY_LIKELY"]:
            expression_score = 0.7
        elif face_data.sorrow_likelihood.name in ["LIKELY", "VERY_LIKELY"]:
            expression_score = 0.3
        elif face_data.anger_likelihood.name in ["LIKELY", "VERY_LIKELY"]:
            expression_score = 0.4

        # Weighted combination
        final_score = (
            size_score * 0.35 +
            direction_score * 0.30 +
            quality_score * 0.20 +
            position_score * 0.10 +
            expression_score * 0.05
        )

        return {
            'overall_score': final_score,
            'size_score': size_score,
            'direction_score': direction_score,
            'quality_score': quality_score,
            'position_score': position_score,
            'expression_score': expression_score,
            'face_area_pixels': face_area,
            'face_percentage': size_percentage * 100,
            'looking_at_camera': direction_score > 0.6,
            'highly_visible': size_score > 0.3,
            'pan_angle': face_data.pan_angle,
            'tilt_angle': face_data.tilt_angle,
            'detection_confidence': face_data.detection_confidence
        }

    except Exception as e:
        logging.error(f"Error calculating face prominence score: {e}")
        return None

def analyze_frame_faces(image_path):
    """
    Analyze all faces in a frame and return the best face score.
    
    Returns:
        Dict with best face analysis or None if no faces detected
    """
    try:
        faces = detect_faces(image_path)
        
        if not faces:
            return None
            
        # Get image dimensions
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        height, width = img.shape[:2]
        
        best_face_score = 0
        best_face_analysis = None
        
        for face in faces:
            face_analysis = calculate_face_prominence_score(face, width, height)
            if face_analysis and face_analysis['overall_score'] > best_face_score:
                best_face_score = face_analysis['overall_score']
                best_face_analysis = face_analysis
        
        return best_face_analysis
        
    except Exception as e:
        logging.error(f"Error analyzing faces in {image_path}: {e}")
        return None

def calculate_combined_frame_score(
    sharpness,
    face_analysis=None,
    sharpness_weight=0.6,
    face_weight=0.4
):
    """
    Calculate combined score balancing sharpness and face quality.
    
    Args:
        sharpness: Frame sharpness score
        face_analysis: Face analysis result (can be None)
        sharpness_weight: Weight for sharpness (default 60%)
        face_weight: Weight for face quality (default 40%)
        
    Returns:
        Combined score normalized to sharpness scale
    """
    # Normalize sharpness to 0-1 scale (typical range 0-2000)
    normalized_sharpness = min(1.0, sharpness / 2000.0)
    
    if face_analysis is None:
        # No face detected - use pure sharpness
        return sharpness
    
    # Face detected - combine scores
    face_score = face_analysis['overall_score']
    
    # Calculate weighted combination
    combined_normalized = (normalized_sharpness * sharpness_weight) + (face_score * face_weight)
    
    # Scale back to sharpness range for consistent comparison
    combined_score = combined_normalized * 2000.0
    
    return combined_score


# --- Example Usage ---
if __name__ == "__main__":
    def format_evaluation_results(evaluation: ThumbnailEvaluation) -> str:
        """Format a ThumbnailEvaluation object into a readable string."""
        metrics = [
            'mobile_legibility',
            'color_contrast',
            'curiosity_gap',
            'emotional_alignment',
            'content_relevance',
            'text_visibility',
            'composition',
            'emotional_impact'
        ]
        
        result = []
        result.append(f"\nOverall Score: {evaluation.overall_score}/10")
        result.append(f"Overall Impression: {evaluation.overall_impression}")
        
        result.append("\nDetailed Metrics:")
        for metric in metrics:
            metric_value = getattr(evaluation, metric)
            result.append(f"\n{metric.replace('_', ' ').title()}: {metric_value.score}/10")
            result.append(f"  Feedback: {metric_value.feedback}")
        
        if evaluation.improvement_suggestions:
            result.append("\nImprovement Suggestions:")
            for i, suggestion in enumerate(evaluation.improvement_suggestions, 1):
                result.append(f"  {i}. {suggestion}")
        
        return '\n'.join(result)

    # Example usage code would go here
    logging.info("Thumbnail optimizer module loaded successfully")
