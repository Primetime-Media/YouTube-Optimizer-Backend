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
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import List, Optional, Dict, Any
from google.genai.types import GenerateContentResponse, FileState
import cv2
import numpy as np
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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
    
    """
    # Allow extra fields to be stored but not affect validation
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "score": 8,
                "feedback": "The text is clearly visible against the background.",
                "additional_notes": "This is an example of an extra field that can be included"
            }
        }
    )
    """

class ThumbnailEvaluation(BaseModel):
    """Comprehensive evaluation of a YouTube thumbnail."""
    mobile_legibility: EvaluationMetric = Field(..., description="Evaluation of how well the thumbnail works on mobile devices")
    color_contrast: EvaluationMetric = Field(..., description="Assessment of color usage and contrast in the thumbnail")
    curiosity_gap: EvaluationMetric = Field(..., description="Evaluation of how well the thumbnail creates curiosity")
    emotional_alignment: EvaluationMetric = Field(..., description="How well the emotional tone matches the video content")
    content_relevance: EvaluationMetric = Field(..., description="Relevance of the thumbnail to the actual video content")
    text_visibility: EvaluationMetric = Field(..., description="Readability and effectiveness of any text in the thumbnail")
    composition: EvaluationMetric = Field(..., description="Overall composition and visual balance of the thumbnail")
    emotional_impact: EvaluationMetric = Field(..., description="Strength of emotional response the thumbnail evokes")
    overall_score: int = Field(..., ge=1, le=10, description="Overall score for the thumbnail (1-10)")
    overall_impression: str = Field(..., description="General impression and summary of the thumbnail's effectiveness")
    improvement_suggestions: List[str] = Field(default_factory=list, description="List of specific suggestions for improving the thumbnail")

def generate_content_from_file(
    file_path: str,
    prompt: str,
    model_name: str = "gemini-2.0-flash-001",
    generation_config: Optional[Dict[str, Any]] = None
) -> GenerateContentResponse | None:
    client = None
    uploaded_file_name = None
    # Max time to wait for file processing (e.g., 3 minutes)
    MAX_PROCESSING_TIME_SECONDS = 180
    POLL_INTERVAL_SECONDS = 5

    try:
        if not GEMINI_API_KEY:
            logging.error("GEMINI_API_KEY is not set in environment variables or provided.")
            return None

        client = genai.Client(api_key=GEMINI_API_KEY)

        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None
        
        logging.info(f"Uploading file: {file_path} for model {model_name}")
        uploaded_file = client.files.upload(file=file_path) 
        uploaded_file_name = uploaded_file.name
        logging.info(f"File uploaded successfully: {uploaded_file_name} (URI: {uploaded_file.uri}). Waiting for it to become active...")

        # Poll for file to become active
        time_waited = 0
        while time_waited < MAX_PROCESSING_TIME_SECONDS:
            processed_file = client.files.get(name=uploaded_file_name)
            current_state = processed_file.state
            logging.info(f"File '{uploaded_file_name}' state: {current_state}")

            if current_state == FileState.ACTIVE:
                logging.info(f"File '{uploaded_file_name}' is now ACTIVE.")
                break
            elif current_state == FileState.FAILED:
                logging.error(f"File '{uploaded_file_name}' processing FAILED. Error: {processed_file.error}")
                # Attempt to delete the failed file
                try:
                    client.files.delete(name=uploaded_file_name)
                    logging.info(f"Successfully deleted failed file: {uploaded_file_name}")
                except Exception as del_err:
                    logging.warning(f"Could not delete failed file {uploaded_file_name}: {del_err}")
                return None
            elif current_state != FileState.PROCESSING:
                 logging.warning(f"File '{uploaded_file_name}' is in an unexpected state: {str(current_state)}. Will continue polling.")

            time.sleep(POLL_INTERVAL_SECONDS)
            time_waited += POLL_INTERVAL_SECONDS
        else: # Loop exited due to timeout
            logging.error(f"Timeout: File '{uploaded_file_name}' did not become ACTIVE within {MAX_PROCESSING_TIME_SECONDS} seconds. Last state: {current_state}")
            # Attempt to delete the file if it's still there
            try:
                client.files.delete(name=uploaded_file_name)
                logging.info(f"Successfully deleted timed-out file: {uploaded_file_name}")
            except Exception as del_err:
                logging.warning(f"Could not delete timed-out file {uploaded_file_name}: {del_err}")
            return None

        logging.info(f"Generating content with Gemini model '{model_name}'...")
        if generation_config:
            logging.info(f"Using generation config: {generation_config}")
        
        response = client.models.generate_content(
            model=model_name,
            contents=[uploaded_file, prompt],
            config=generation_config
        )

        if uploaded_file_name:
            logging.info(f"Attempting to delete uploaded file: {uploaded_file_name}")
            try:
                client.files.delete(name=uploaded_file_name)
                logging.info(f"Successfully deleted file: {uploaded_file_name} from the service.")
            except Exception as del_e:
                logging.warning(f"Could not delete file {uploaded_file_name} from service after processing: {del_e}")
        
        return response
        
    except FileNotFoundError:
        logging.error(f"Upload failed: File not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred with the Gemini API or file processing: {e}")
        if client and uploaded_file_name:
            try:
                logging.warning(f"Attempting to clean up uploaded file due to error: {uploaded_file_name}")
                client.files.delete(name=uploaded_file_name)
                logging.info(f"Successfully cleaned up/deleted file: {uploaded_file_name} from the service.")
            except Exception as cleanup_e:
                logging.error(f"Error during file cleanup after an exception: {cleanup_e}")
        return None

def get_competitor_thumbnail_descriptions(
    competitor_analytics_data: Dict[str, Any],
    model_name: str = "gemini-2.0-flash",
    max_workers: int = 5
) -> List[CompetitorThumbnailDescription]:
    """
    Generate thumbnail descriptions for competitor thumbnails in parallel.
    
    Args:
        competitor_analytics_data: Dictionary containing competitor video data with thumbnails          
        model_name: Name of the Gemini model to use for analysis
        max_workers: Maximum number of concurrent thumbnail analysis tasks
        
    Returns:
        List of dictionaries containing the analysis for each competitor thumbnail
    """
    if not competitor_analytics_data or not competitor_analytics_data.get("competitor_videos"):
        return {}
    
    competitor_videos = competitor_analytics_data["competitor_videos"]
    
    def process_single_thumbnail(video_data: Dict[str, Any]) -> Optional[CompetitorThumbnailDescription]:
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
        
        if video_data.get("days_since_upload") is not None and video_data.get("views_per_day") is not None:
            metrics.extend([
                f"Days Since Upload: {video_data['days_since_upload']}",
                f"Views Per Day: {video_data['views_per_day']:,.2f}"
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
            "video_id": {video_id},
            "description": "Detailed analysis of the thumbnail",
            "strengths": ["strength 1", "strength 2", ...],
            "weaknesses": ["weakness 1", "weakness 2", ...],
            "suggested_improvements": ["suggestion 1", "suggestion 2", ...]
        }}
        """
        
        try:
            # Download the thumbnail image
            response = requests.get(thumbnail_url, timeout=10)
            response.raise_for_status()
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            try:
                # Generate content using Gemini
                generation_config = {
                    "response_mime_type": "application/json",
                    "response_schema": CompetitorThumbnailDescription
                }
                
                response = generate_content_from_file(
                    file_path=temp_file_path,
                    prompt=prompt,
                    model_name=model_name,
                    generation_config=generation_config
                )
                
                if not response:
                    return None
                
                try:
                    parsed_response: CompetitorThumbnailDescription = response.parsed

                    if not parsed_response.video_id:
                        parsed_response.video_id = video_id
                    
                    video_data['thumbnail_descriptions'] = parsed_response
                    logging.info(f"Successfully parsed thumbnail suggestions.")
                    return video_data
                except ValidationError as ve:
                    logging.error(f"Pydantic validation error parsing thumbnail suggestions: {ve}")
                    return None
                except json.JSONDecodeError as jde:
                    logging.error(f"JSON decoding error parsing thumbnail suggestions: {jde}")
                    return None
                except Exception as e:
                    logging.error(f"An unexpected error occurred while parsing thumbnail suggestions: {e}")
                    return None
                
            finally:
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logging.warning(f"Could not delete temporary file {temp_file_path}: {e}")
                    
        except Exception as e:
            logging.error(f"Error processing thumbnail for video {video_id}: {e}")
            return None
    
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
                logging.error(f"Error processing thumbnail for video {video_id}: {e}")
    
    #for result in results:
     #   competitor_analytics_data['competitors'][result['video_id']]['thumbnail_descriptions'] = result

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
    try:
        print('do_thumbnail_optimization')
        competitor_analytics_data = get_competitor_thumbnail_descriptions(competitor_analytics_data)    

        local_video_filename = f"{video_id}.mp4"

        #extracted_frames = download_youtube_video_and_extract_frames(video_id, local_video_filename)
        
        extracted_frames = [
            '/Users/jasonramirez/Documents/youtube-optimizer/backend/extracted_thumbnails/orig_y8yDBm7PvbM_llm_sharp_thumb_01_at_0_167s_Manstandingheroicallyontop.jpg',
            '/Users/jasonramirez/Documents/youtube-optimizer/backend/extracted_thumbnails/orig_y8yDBm7PvbM_llm_sharp_thumb_02_at_0_731s_Cowboymakesawhipsoundsol.jpg',
            '/Users/jasonramirez/Documents/youtube-optimizer/backend/extracted_thumbnails/orig_y8yDBm7PvbM_llm_sharp_thumb_03_at_8_660s_Manlookssurprisedbythefor.jpg',
            '/Users/jasonramirez/Documents/youtube-optimizer/backend/extracted_thumbnails/orig_y8yDBm7PvbM_llm_sharp_thumb_04_at_37_619s_Manputshisfingeronhishea.jpg',
            '/Users/jasonramirez/Documents/youtube-optimizer/backend/extracted_thumbnails/orig_y8yDBm7PvbM_llm_sharp_thumb_05_at_51_141s_Thesonicboomwiththefists.jpg'
        ]

        if not extracted_frames:
            logging.error("No frames were extracted from the video")
            return []

        def get_video_thumbnail_description(video_data: dict) -> str:
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
                """

                #color_prompt_instructions = Do not use YouTube color schema (red, white, black)

                user_prompt = f"""
                {system_prompt}

                ANALYSIS CONTEXT:
                Video Details:
                - Title: {original_title}
                - Category: {category_name}
                - Top Tags: {', '.join(original_tags[:5])}

                Competitor Thumbnail Insights:
                {competitor_insights}

                DETAILED TRANSFORMATION INSTRUCTIONS:

                1. FRAME COMPOSITION ANALYSIS:
                - Current strengths to preserve: [Analyze and list]
                - Areas for improvement: [Detail specific changes]
                - Focal point optimization: [Specify]
                - Rule of thirds application: [Specify]
                - Negative space usage: [Recommendations]

                2. VISUAL ELEMENTS:
                - Main subject enhancement: [Specific adjustments]
                - Background treatment: [Blur/Simplify/Enhance]
                - Foreground elements: [Add/Remove/Modify]
                - Depth creation: [Layering techniques]
                - Visual hierarchy: [How to guide the eye]

                3. COLOR & LIGHTING:
                - Color scheme: [Primary/secondary colors]
                - Contrast adjustments: [Specific values]
                - Lighting direction: [Key/fill/backlight]
                - Mood enhancement: [Color grading specifics]
                - Competitor color analysis: [How to differentiate]

                4. TEXT & TYPOGRAPHY (if applicable):
                - Text content: [Exact wording]
                - Font selection: [Specific font names]
                - Size and weight: [Specific values]
                - Color and effects: [Hex codes and effects]
                - Positioning: [Exact coordinates if possible]
                - Drop shadow/outline: [Specific settings]

                5. VISUAL EFFECTS:
                - Depth of field: [Specific blur amounts]
                - Glow/Highlight: [Areas and intensity]
                - Texture overlay: [Type and opacity]
                - Edge treatment: [Vignette/border specifics]
                - Animation potential: [Subtle movement hints]

                6. EMOTIONAL IMPACT:
                - Primary emotion: [Selected from triggers]
                - Color psychology: [Specific color-emotion pairs]
                - Facial expressions: [Specific emotions to show]
                - Body language: [Pose adjustments]
                - Visual metaphors: [Elements to include]

                7. MOBILE OPTIMIZATION:
                - Readability check: [Font size adjustments]
                - Thumbnail crop: [Safe zone definition]
                - Element sizing: [Minimum size requirements]
                - Touch target areas: [Important element placement]

                8. COMPETITIVE DIFFERENTIATION:
                - Competitor strengths: [What to learn from]
                - Competitor weaknesses: [How to improve]
                - Unique angles: [How to stand out]
                - Trend incorporation: [Current YouTube trends]

                TRANSFORMATION PROMPT:
                [Generate a highly detailed, specific transformation prompt that incorporates all the above elements. Be extremely specific about visual treatments, colors, positioning, and emotional impact. The prompt should be ready to use with an image generation model.]

                Return ONLY the transformation prompt, nothing else. Be extremely specific and detailed in your instructions.
                """
                
                # Generate the transformation prompt
                response = generate_content_from_file(
                    file_path=frame_path,
                    model_name="gemini-2.5-pro-preview-05-06",
                    prompt=user_prompt
                )
                
                # Clean and return the generated prompt
                prompt = response.text.strip()
                if not prompt:
                    raise ValueError("Empty prompt generated")
                    
                return prompt
                
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
                    generate_thumbnail_transformation_prompt(frame_path, original_title, category_name, original_tags, transcript, competitor_insights),
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
                        results.append({
                            'original_frame': frame_path,
                            'optimized_thumbnail': result,
                            #'prompt': generate_thumbnail_prompt(frame_path)
                        })
                except Exception as e:
                    logging.error(f"Error processing frame {frame_path}: {str(e)}")
            
            # TODO: generate_thumbnail_prompt in process_single_thumbnail function
            # TODO: Return top thumbnail
        
        
        return results
    except Exception as e:
        logging.error(f"Error processing frames: {str(e)}")
        return None

def process_single_thumbnail(
    frame_path: str,
    prompt: str,
    video_title: str,
    video_description: str,
    user_id: int
) -> Optional[str]:
    """
    Process a single thumbnail frame with OpenAI's GPT-4.1 Image Model
    
    Args:
        frame_path: Path to the original frame
        prompt: Custom prompt for this frame
        video_title: Video title for naming
        user_id: User ID for output directory
        
    Returns:
        Path to the optimized thumbnail or None if failed
    """
    try:
        # Create output directory
        output_dir = f"user_uploads/{user_id}/thumbnails"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output path
        timestamp = int(time.time())
        output_path = os.path.join(
            output_dir,
            f"optimized_{os.path.splitext(os.path.basename(frame_path))[0]}_{timestamp}.png"
        )
        
        # Optimize the thumbnail
        result_path= optimize_thumbnail_with_openai(
            input_image_path=frame_path,
            output_image_path=output_path,
            video_title=video_title,
            custom_prompt_instructions=prompt
        )
        
        if not result_path or not os.path.exists(result_path):
            logging.error(f"Failed to generate thumbnail for {frame_path}")
            return None
        
        # Evaluate the thumbnail
        evaluation = evaluate_thumbnail_with_gemini(
            image_path=result_path,
            prompt=prompt,
            video_title=video_title,
            video_description=video_description
        )
        
        if not evaluation:
            logging.warning(f"Failed to evaluate thumbnail {result_path}")
            evaluation = {"error": "Evaluation failed"}
        
        # Prepare the result
        result = {
            "original_frame": frame_path,
            "optimized_thumbnail": result_path,
            "prompt": prompt,
            "evaluation": evaluation,
            "timestamp": timestamp
        }
        
        return result
        
    except Exception as e:
        logging.error(f"Error in process_single_thumbnail: {str(e)}")
        return None

def evaluate_thumbnail_with_gemini(image_path: str, prompt: str, video_title: str, video_description: str) -> dict:
    """
    Evaluate a thumbnail using Gemini for comprehensive analysis.
    
    Args:
        image_path: Path to the thumbnail image
        prompt: The prompt used to generate the thumbnail
        video_title: Video title for context
        video_description: Video description for context
        
    Returns:
        Dictionary containing evaluation results and suggestions
    """
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        evaluation_prompt = f"""
        Please analyze this YouTube thumbnail and provide a detailed evaluation based on the following criteria:
        
        1. Mobile Legibility:
           - Is the main subject clearly visible on small screens?
           - Is there enough contrast for elements to be distinguishable?
        
        2. Color and Contrast:
           - Are the colors vibrant and eye-catching?
           - Is there sufficient contrast between elements?
           - Does the color scheme avoid YouTube UI colors (red, white, black dominance)?
        
        3. Curiosity Gap:
           - Does the thumbnail create intrigue or curiosity?
           - Would it make someone want to click to learn more?
        
        4. Emotional Alignment:
           - Does the thumbnail match the emotional tone of the title and description?
           Title: {video_title}
           Description: {video_description}
        
        5. Content Relevance:
           - Is the thumbnail relevant to the video content?
           - Does it accurately represent what the video is about?
        
        6. Text Visibility:
           - If there's text, is it large and readable on small screens?
           - Is there sufficient contrast between text and background?
        
        7. Composition:
           - Is the main subject properly framed and positioned?
           - Does it follow the rule of thirds?
           - Are important elements away from the bottom right corner (timestamp area)?
        
        8. Emotional Impact:
           - What emotion does this thumbnail primarily evoke?
           - Is this appropriate for the video content?
        
        Please provide:
        1. A score from 1-10 for each category
        2. Specific feedback for each category
        3. Overall impression
        4. Specific suggestions for improvement
        
        Return your response as a JSON object with this structure:
        {{
            "mobile_legibility": {{"score": 0-10, "feedback": ""}},
            "color_contrast": {{"score": 0-10, "feedback": ""}},
            "curiosity_gap": {{"score": 0-10, "feedback": ""}},
            "emotional_alignment": {{"score": 0-10, "feedback": ""}},
            "content_relevance": {{"score": 0-10, "feedback": ""}},
            "text_visibility": {{"score": 0-10, "feedback": ""}},
            "composition": {{"score": 0-10, "feedback": ""}},
            "emotional_impact": {{"score": 0-10, "feedback": ""}},
            "overall_score": 0-10,
            "overall_impression": "",
            "improvement_suggestions": ["improvement suggestion 1", ...]
        }}
        """                    
        
        response = generate_content_from_file(
            file_path=image_path,
            prompt=evaluation_prompt,
            model_name="gemini-2.5-pro-preview-05-06",
            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": ThumbnailEvaluation
            }
        )
        
        try:
            evaluation = response.parsed    
            return evaluation
        except Exception as e:
            logging.error("Failed to parse Gemini evaluation response: " + str(e))
            return None 
            
    except Exception as e:
        logging.error(f"Error in evaluate_thumbnail_with_gemini: {str(e)}")
        return None

def get_suggested_thumbnail_times(
    video_file_path: str,
    num_suggestions: int = 5,
    model_name: str = "gemini-2.0-flash"
) -> Optional[SuggestedThumbnails]:
    prompt = f"""
    Analyze the provided video to identify the best {num_suggestions} moments for YouTube thumbnails.
    For each moment, provide the exact timestamp in seconds (e.g., 45.67).
    Also, provide a concise description explaining why this frame would make a highly click-worthy thumbnail.
    Consider YouTube best practices:
    - Clear subject focus (face, product, key action)
    - Strong emotion or intrigue
    - Visually appealing composition, color, and contrast
    - Minimal or impactful text (if any naturally occurs in the frame)
    - Branding elements if present and relevant
    - Avoid blurry, dark, or cluttered frames.
    - The thumbnail should accurately represent the video content while maximizing curiosity and click-through rate.

    Return these suggestions formatted STRICTLY as a JSON object matching the following Pydantic schema:
    {{
        "suggested_scenes": [
            {{
                "timestamp_seconds": <float>,
                "description": "<string>"
            }},
            ...
        ]
    }}
    Ensure the output is ONLY the JSON object, with no other text before or after it.
    """

    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": SuggestedThumbnails,
    }

    logging.info(f"Requesting thumbnail suggestions for video: {video_file_path} with model {model_name}")
    response = generate_content_from_file(
        file_path=video_file_path,
        prompt=prompt,
        model_name=model_name,
        generation_config=generation_config
    )

    if not response:
        logging.error("Failed to get a response from generate_content_from_file for thumbnail suggestions.")
        return None

    try:
        parsed_response: SuggestedThumbnails = response.parsed
        logging.info(f"Successfully parsed {len(parsed_response.suggested_scenes)} thumbnail suggestions.")
        return parsed_response
    except ValidationError as ve:
        logging.error(f"Pydantic validation error parsing thumbnail suggestions: {ve}")
        return None
    except json.JSONDecodeError as jde:
        logging.error(f"JSON decoding error parsing thumbnail suggestions: {jde}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while parsing thumbnail suggestions: {e}")
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
        
        # Compute the Laplacian of the grayscale image and then return the variance
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
        search_window_seconds: Half-width of the window to search for sharp frames (e.g., 0.5 means +/- 0.5s).
        num_candidate_frames: How many frames to extract and evaluate within the search window.

    Returns:
        A list of absolute paths to the successfully extracted and saved sharp thumbnail image files.
    """
    if not os.path.exists(video_file_path):
        logging.error(f"Video file not found for frame extraction: {video_file_path}")
        return []

    logging.info(f"Attempting to get thumbnail suggestions for: {video_file_path}")
    suggested_thumbnails_data: SuggestedThumbnails = get_suggested_thumbnail_times(
        video_file_path=video_file_path,
        num_suggestions=num_suggestions_for_llm
    )

    if not suggested_thumbnails_data or not suggested_thumbnails_data.suggested_scenes:
        logging.warning(f"No thumbnail suggestions received from LLM for {video_file_path}.")
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
        highest_sharpness = -1.0
        
        # Define the start and end of the search window
        window_start_time = max(0, llm_timestamp - search_window_seconds)
        window_end_time = llm_timestamp + search_window_seconds 
        # Ensure window_end_time does not exceed video duration (approximate for now, ffmpeg handles exact)
        
        if window_start_time >= window_end_time or num_candidate_frames <= 0:
            logging.warning(f"Invalid window or zero candidates for timestamp {llm_timestamp}, falling back to exact time.")
            candidate_timestamps = [llm_timestamp]
        else:
            candidate_timestamps = np.linspace(window_start_time, window_end_time, num_candidate_frames)

        # Create a temporary directory for candidate frames for this specific scene
        with tempfile.TemporaryDirectory(prefix=f"candidates_scene_{i+1}_") as temp_candidate_dir:
            temp_candidate_path_obj = Path(temp_candidate_dir)
            logging.info(f"Extracting {len(candidate_timestamps)} candidate frames for scene {i+1} into {temp_candidate_path_obj}")

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
                        logging.warning(f"ffmpeg failed for candidate frame at {current_timestamp:.3f}s. stderr: {result.stderr}")
                except subprocess.TimeoutExpired:
                    logging.error(f"ffmpeg command timed out extracting candidate at {current_timestamp:.3f}s.")
                except Exception as e_ffmpeg:
                    logging.error(f"Error extracting candidate frame at {current_timestamp:.3f}s: {e_ffmpeg}")
            
            if not candidate_frame_files:
                logging.warning(f"No candidate frames were successfully extracted for scene {i+1} at LLM time {llm_timestamp:.2f}s. Skipping this suggestion.")
                continue

            logging.info(f"Analyzing {len(candidate_frame_files)} candidate frames for sharpness for scene {i+1}...")
            for cand_frame_path_obj in candidate_frame_files:
                sharpness = _calculate_frame_sharpness(str(cand_frame_path_obj))
                logging.debug(f"Candidate {cand_frame_path_obj.name}: sharpness = {sharpness:.2f}")
                if sharpness > highest_sharpness:
                    highest_sharpness = sharpness
                    best_frame_path = str(cand_frame_path_obj) # Store the path of the temp file

            if best_frame_path:
                logging.info(f"Best sharpness for scene {i+1} (LLM time {llm_timestamp:.2f}s) is {highest_sharpness:.2f} from {Path(best_frame_path).name}")
                
                # Construct final output path
                timestamp_str = f"{Path(best_frame_path).stem.split('_time_')[-1].replace('s','').replace('.', '_')}" # from temp filename like candidate_001_time_123.456s
                sane_description = ''.join(filter(str.isalnum, scene.description.replace(" ", "_")[:30]))
                output_filename = f"{prefix}_llm_sharp_thumb_{i+1:02d}_at_{timestamp_str}s_{sane_description}.jpg"
                final_output_file_path = output_path_obj / output_filename

                # Copy the best temporary frame to the final destination
                shutil.copy2(best_frame_path, final_output_file_path) # shutil.copy2 preserves metadata
                if final_output_file_path.exists():
                    logging.info(f"Successfully saved best frame: {final_output_file_path}")
                    extracted_final_frame_paths.append(str(final_output_file_path))
                else:
                    logging.error(f"Failed to copy best frame to final destination: {final_output_file_path}")
            else:
                logging.warning(f"Could not determine the best frame for scene {i+1} (LLM time {llm_timestamp:.2f}s). No frame will be saved.")

    if not extracted_final_frame_paths:
        logging.warning(f"No sharp frames were successfully extracted for {video_file_path} based on LLM suggestions and sharpness analysis.")
    
    return extracted_final_frame_paths

from openai import OpenAI
import base64
import openai 


def optimize_thumbnail_with_openai(
    input_image_path: str,
    output_image_path: str,
    video_title: Optional[str] = None,
    custom_prompt_instructions: Optional[str] = None,
    model_name: str = "gpt-image-1",
    output_size: str = "1024x1024" # DALL-E 2 edit supports "256x256", "512x512", "1024x1024"
) -> Optional[str]:
    """
    Optimizes an image for a YouTube thumbnail using OpenAI's DALL-E image editing.

    Args:
        input_image_path: Path to the local input image file (e.g., a sharp frame).
        output_image_path: Path where the optimized thumbnail image (PNG) will be saved.
        video_title: Optional text (e.g., video title) to suggest for an overlay on the thumbnail.
        custom_prompt_instructions: Optional additional specific instructions for the AI editor.
        model_name: The OpenAI model to use (e.g., "dall-e-2").
        output_size: The size of the generated image. 

    Returns:
        Absolute path to the saved optimized image file, or None if an error occurs.
    """
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY is not set in environment variables. Cannot optimize with OpenAI.")
        return None

    if not os.path.exists(input_image_path):
        logging.error(f"Input image for OpenAI optimization not found: {input_image_path}")
        return None

    try:
        client = OpenAI()

        prompt = f"""{custom_prompt_instructions}
        Thoroughly analyze the provided image, which is a frame from a video.
        Your task is to transform it into a highly engaging and click-compelling YouTube thumbnail.
        Preserve the main subject and the core essence of the original scene faithfully.
        Enhancements should make the thumbnail vibrant, clear, and professional for a YouTube audience.
        {f"If appropriate for the image, consider overlaying the text '{video_title}'. The text MUST be bold, with EXTREMELY HIGH CONTRAST against its background to ensure absolute readability. It should be stylishly designed and strategically placed for maximum impact. Only add text if it significantly enhances the thumbnail and does not obscure critical visual elements. If the text cannot be made perfectly clear and legible, do not add it." if video_title else ""}
        Boost colors and contrast to make the image pop, but maintain a natural look.
        Ensure the main focal point is exceptionally clear and sharp.
        Consider YouTube best practices for thumbnails: strong emotion, intrigue, clear subject, rule of thirds.
        The final image should be polished, high-resolution, and immediately grab attention on a crowded YouTube feed.
        Do not add any watermarks or signatures.
        CRITICAL: The appearance of any people or characters in the original image MUST remain UNCHANGED. Do NOT alter facial features, expressions, body shape, or clothing. The goal is to enhance the *surrounding* visual elements and the overall composition, not the individuals themselves.
        The enhancements should primarily focus on visual elements such as colors, contrast, sharpness, and potentially overlaying text to grab the viewer's attention, similar to how a human thumbnail designer would enhance a thumbnail without distorting the core content.
        The goal is enhancement, not a radical transformation of the original image's subject or characters.
        """

        logging.info(f"OpenAI image optimization prompt for '{Path(input_image_path).name}':\n{prompt}")

        with open(input_image_path, "rb") as image_file_rb:
            response = client.images.edit(
                model=model_name,
                image=image_file_rb,
                quality="high",
                prompt=prompt,
                n=1,
            )

        if not response.data or not response.data[0].b64_json:
            logging.error("OpenAI API did not return image data for thumbnail optimization.")
            return None

        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Ensure output directory exists
        output_image_path_obj = Path(output_image_path)
        output_image_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_image_path_obj, "wb") as f:
            f.write(image_bytes)
        
        resolved_output_path = str(output_image_path_obj.resolve())
        logging.info(f"Successfully optimized thumbnail with OpenAI saved to: {resolved_output_path}")
        return resolved_output_path

    except openai.APIError as e:
        # More detailed API error logging
        error_details = f"OpenAI API error: Status Code: {e.status_code}, Type: {e.type}"
        if hasattr(e, 'message') and e.message:
            error_details += f", Message: {e.message}"
        if hasattr(e, 'code') and e.code:
            error_details += f", Code: {e.code}"
        if hasattr(e, 'param') and e.param:
            error_details += f", Param: {e.param}"
        logging.error(error_details)
        if hasattr(e, 'response') and e.response and hasattr(e.response, 'content') and e.response.content:
            try:
                # Attempt to decode and log JSON error content if available
                error_content = e.response.content.decode('utf-8')
                logging.error(f"OpenAI API error response content: {error_content}")
            except UnicodeDecodeError:
                logging.error("OpenAI API error response content (non-UTF-8): [binary data]")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during OpenAI thumbnail optimization: {e}", exc_info=True)
        return None


# --- Sieve API related constants and functions ---
SIEVE_API_KEY = os.getenv("SIEVE_API_KEY", "YOUR_SIEVE_API_KEY_HERE") # Replace with your key or set env var
if SIEVE_API_KEY == "YOUR_SIEVE_API_KEY_HERE":
    logging.error("SIEVE_API_KEY is not set in environment variables")

BASE_URL = "https://mango.sievedata.com/v2"
PUSH_ENDPOINT = f"{BASE_URL}/push"
JOB_STATUS_ENDPOINT = f"{BASE_URL}/jobs"
POLL_INTERVAL_SECONDS = 10 # How often to check the job status

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
            "download_type": "video", # Or "audio"
            "resolution": "highest-available",
            "include_audio": True,
            "start_time": 0,
            "end_time": -1, # -1 means download until the end
            "include_metadata": False, # Set to True if you need metadata
            # "metadata_fields": "title,thumbnail,description,tags,duration", # Uncomment if include_metadata is True
            "include_subtitles": False,
            # "subtitle_languages": "en", # Uncomment if include_subtitles is True
            "video_format": "mp4",
            "audio_format": "mp3" # Relevant if download_type is "audio" or include_audio is True
        }
    }

    try:
        logging.info(f"Submitting download job for URL: {youtube_url}")
        response = requests.post(PUSH_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()
        job_id = response_data.get("id")

        if job_id:
            logging.info(f"Job submitted successfully. Job ID: {job_id}")
            return job_id
        else:
            logging.error(f"Job submission failed. No 'id' found in response: {response_data}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error submitting job: {e}")
        if e.response is not None:
            logging.error(f"Response status code: {e.response.status_code}")
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
                logging.info(f"Job '{job_id}' finished successfully.")
                # The 'outputs' field usually contains the results (e.g., download URL)
                logging.info(f"Job Output: {json.dumps(job_data.get('outputs'), indent=2)}")
                return job_data
            elif status == "error":
                logging.error(f"Job '{job_id}' failed.")
                logging.error(f"Job Error Details: {json.dumps(job_data.get('error'), indent=2)}")
                return None # Indicate failure
            elif status in ["starting", "processing", "queued"]:
                # Job is still running, wait and poll again
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                logging.warning(f"Job '{job_id}' has an unknown status: {status}")
                time.sleep(POLL_INTERVAL_SECONDS) # Still wait, might be a transient state

        except requests.exceptions.RequestException as e:
            logging.error(f"Error polling job status for '{job_id}': {e}")
            if e.response is not None:
                logging.error(f"Response status code: {e.response.status_code}")
                logging.error(f"Response content: {e.response.text}")
            # Decide if you want to retry or give up after polling errors
            logging.info("Waiting before retrying poll...")
            time.sleep(POLL_INTERVAL_SECONDS * 2) # Wait longer after an error
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON response during polling: {response.text}")
            logging.info("Waiting before retrying poll...")
            time.sleep(POLL_INTERVAL_SECONDS * 2)
        except KeyboardInterrupt:
             logging.info("Polling interrupted by user.")
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
    logging.info(f"Attempting to download video from: {video_url}")
    try:
        # Use stream=True to handle potentially large files efficiently
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status() # Check for download errors
            # Ensure the directory exists
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

def extract_thumbnails_ffmpeg(video_path: str, output_dir: str, num_thumbnails: int = 10, base_filename: str = "thumb") -> list[str] | None:
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
        logging.error("ffmpeg or ffprobe command not found. Please ensure ffmpeg is installed and in your system's PATH.")
        return None

    if num_thumbnails <= 0:
        logging.warning("Number of thumbnails must be positive. Defaulting to 1.")
        num_thumbnails = 1

    # Create/clean output directory
    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    
    # Clean up any existing thumbnail files with this base_filename
    for old_file in output_path_obj.glob(f"{base_filename}_*.jpg"):
        old_file.unlink()
    
    # --- Approach 1: Global thumbnail filter ---
    # This approach uses ffmpeg's thumbnail filter to select the best frames globally
    logging.info("Trying global thumbnail filter approach...")
    extracted_paths = _extract_global_thumbnails(video_path, output_dir, num_thumbnails, base_filename)
    if extracted_paths and len(extracted_paths) == num_thumbnails:
        logging.info(f"Global thumbnail filter succeeded, extracted {len(extracted_paths)} thumbnails")
        return extracted_paths
    
    # --- Approach 2: Distributed thumbnail filter ---
    # If global method failed, try using the thumbnail filter on distributed segments
    if not extracted_paths or len(extracted_paths) < num_thumbnails:
        logging.info("Global method failed, trying distributed thumbnail approach...")
        extracted_paths = _extract_distributed_thumbnails(video_path, output_dir, num_thumbnails, base_filename)
        if extracted_paths and len(extracted_paths) == num_thumbnails:
            logging.info(f"Distributed thumbnail method succeeded, extracted {len(extracted_paths)} thumbnails")
            return extracted_paths
    
    # --- Approach 3: Fixed timestamp fallback ---
    # Last resort: extract frames at fixed intervals
    if not extracted_paths or len(extracted_paths) < num_thumbnails:
        logging.info("Trying fixed timestamp fallback method...")
        extracted_paths = _extract_fixed_timestamps(video_path, output_dir, num_thumbnails, base_filename)
        if extracted_paths:
            logging.info(f"Fixed timestamp method extracted {len(extracted_paths)} thumbnails")
            return extracted_paths
    
    # If we get here, all methods failed
    logging.error("All thumbnail extraction methods failed")
    return None


def _extract_global_thumbnails(video_path: str, output_dir: str, num_thumbnails: int, base_filename: str) -> list[str] | None:
    """
    Extracts thumbnails using ffmpeg's global thumbnail filter.
    This tries to select the most representative frames from the entire video.
    """
    output_path_obj = Path(output_dir)

    # First, get video duration to calculate frame positions
    try:
        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())

        if duration <= 0:
            raise ValueError(f"Invalid video duration: {duration}")

        # Calculate timestamps - evenly distributed but avoiding start/end
        interval = duration / (num_thumbnails + 1)
        timestamps = [interval * (i + 1) for i in range(num_thumbnails)]

        # Extract each frame using a scene detection filter PLUS thumbnail
        extracted_paths = []
        for i, timestamp in enumerate(timestamps):
            output_file = output_path_obj / f"{base_filename}_{i + 1:03d}.jpg"

            # Use a 3-second window around the timestamp
            window_start = max(0, timestamp - 1.5)
            window_duration = min(3.0, duration - window_start)

            # Command to get the most representative frame in this 3-second window
            cmd = [
                "ffmpeg",
                "-ss", str(window_start),  # Start position
                "-t", str(window_duration),  # Duration to analyze
                "-i", video_path,
                "-vf", "thumbnail=1,scale=640:-1",  # Find 1 representative frame in this window
                "-frames:v", "1",  # One output frame only
                "-q:v", "2",
                "-y",
                str(output_file)
            ]

            logging.info(f"Extracting thumbnail {i + 1}/{num_thumbnails} at timestamp {timestamp:.2f}s")
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

def _extract_distributed_thumbnails(video_path: str, output_dir: str, num_thumbnails: int, base_filename: str) -> list[str] | None:
    """
    Extracts thumbnails by dividing the video into segments and using the thumbnail filter
    on each segment. This is more likely to succeed than the global method.
    """
    output_path_obj = Path(output_dir)
    
    try:
        # Get video duration
        probe_cmd = [
            "ffprobe",
            "-v", "error",
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
        
        # Process each segment
        for i, start_time in enumerate(segment_points):
            output_file = output_path_obj / f"{base_filename}_{i+1:03d}.jpg"
            
            # Extract one representative frame from this segment using thumbnail filter
            cmd = [
                "ffmpeg", 
                "-ss", str(start_time),  # Start at segment beginning
                "-t", str(segment_duration),  # Process only this segment duration
                "-i", video_path,
                "-vf", "thumbnail=1",  # Get 1 representative frame from segment
                "-frames:v", "1",
                "-q:v", "2",
                "-y",
                str(output_file)
            ]
            
            segment_result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            
            if segment_result.returncode == 0 and output_file.exists():
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

def _extract_fixed_timestamps(video_path: str, output_dir: str, num_thumbnails: int, base_filename: str) -> list[str] | None:
    """
    Extracts thumbnails at fixed intervals throughout the video.
    This is the most reliable but least intelligent method.
    """
    output_path_obj = Path(output_dir)
    
    try:
        # Get video duration
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        
        if duration <= 0:
            raise ValueError(f"Invalid video duration: {duration}")
        
        # Calculate timestamps (skip the very beginning and end)
        interval = duration / (num_thumbnails + 1)
        timestamps = [interval * (i + 1) for i in range(num_thumbnails)]
        
        extracted_paths = []
        
        # Extract a frame at each timestamp
        for i, timestamp in enumerate(timestamps):
            output_file = output_path_obj / f"{base_filename}_{i+1:03d}.jpg"
            
            cmd = [
                "ffmpeg",
                "-ss", str(timestamp),  # Seek to timestamp
                "-i", video_path,
                "-frames:v", "1",       # Extract exactly one frame
                "-q:v", "2",
                "-y",
                str(output_file)
            ]
            
            ts_result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            
            if ts_result.returncode == 0 and output_file.exists():
                extracted_paths.append(str(output_file))
            else:
                logging.warning(f"Failed to extract thumbnail at timestamp {timestamp:.2f}s")
        
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
    If base_filename is provided, only delete files matching that pattern.
    Otherwise, delete all jpg files in the directory.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0

    count = 0
    if base_filename:
        pattern = f"{base_filename}_*.jpg"
        logging.info(f"Cleaning up files matching pattern: {pattern}")
        for file in output_path.glob(pattern):
            file.unlink()
            count += 1
    else:
        logging.info(f"Cleaning up ALL jpg files in {output_dir}")
        for file in output_path.glob("*.jpg"):
            file.unlink()
            count += 1

    logging.info(f"Deleted {count} existing files from {output_dir}")
    return count

def download_youtube_video_and_extract_frames(video_id: str, output_path: str, thumbnail_output_dir: str = "thumbnails") -> Optional[List[str]]:
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
    
    # Create thumbnail output directory if it doesn't exist
    os.makedirs(thumbnail_output_dir, exist_ok=True)
    
    # Clean up previous video file if it exists
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
    
    logging.info("--- Download Process Completed ---")
    
    # Extract video URL from job output
    video_url = None
    outputs = final_job_data.get("outputs", [])
    
    if isinstance(outputs, list) and len(outputs) > 0:
        output_data = outputs[0]
        video_url = output_data.get("data", {}).get("url")
    
    if not video_url:
        logging.error("Could not find video URL in Sieve job output")
        logging.error(f"Full Sieve Output: {json.dumps(final_job_data.get('outputs'), indent=2)}")
        return None
    
    logging.info(f"Video Sieve Output URL: {video_url}")
    
    # Download the video
    if not download_video_from_url(video_url, output_path):
        logging.error("Failed to download video from Sieve URL")
        return None
    
    logging.info(f"Successfully downloaded video to: {output_path}")
    
    # Extract frames using the suggested times method
    try:
        extracted_frames = extract_frames_at_suggested_times(video_file_path=output_path)
        
        if extracted_frames:
            logging.info(f"Successfully extracted {len(extracted_frames)} frames using suggested times")
            return extracted_frames
        else:
            logging.warning("No frames were extracted using suggested times")
            return []
            
    except Exception as e:
        logging.error(f"Error extracting frames: {str(e)}")
        return None
        
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

    thumbnail_optimization_results = do_thumbnail_optimization(
        video_id="y8yDBm7PvbM",
        original_title="Testing What Happens If You Jump On A Moving Train",
        original_description="""Sometimes you gotta go full Tom Cruise to really teach the science. Now go grow your brain even more and get 2 FREE boxes at: https://crunchlabs.com/HotAirBalloon
        Get your CrunchLabs box today:
        Build Box for kids click here: https://crunchlabs.com/HotAirBalloon
        Hack Pack for teens and adults click here: https://crunchlabs.com/HotAirBalloonHP


        Thanks to these folks for providing some of the music in the video:
        Ponder- ‪@Pondermusic‬ 
        Laura Shigihara - ‪@supershigi‬ 
        Andrew Applepie -   / andrewapplepie  
        Blue Wednesday -   / bluewednesday  
        Danijel Zambo - https://open.spotify.com/intl-de/arti...

        PLATINUM TICKET INSTANT WIN GAME
        NO PURCHASE NECESSARY. Promotion starts on 06/1/2024 & ends on 05/31/25, subject to monthly entry deadlines. Open to legal residents of the 50 U.S. & D.C., 18+. 1 prize per month: each month is its own separate promotion. For the first 2-3 months, winner may be notified via phone call instead of winning game piece. If a monthly prize is unclaimed/forfeited, it will be awarded via 2nd chance drawing. See Official Rules at crunchlabs.com/sweepstakes for full details on eligibility requirements, how to enter, free method of entry, prize claim procedure, prize description and limitations. Void where prohibited.

        PLATINUM DIPLOMA SWEEPSTAKES
        NO PURCHASE NECESSARY. Ends 4/30/25, Open to legal residents of the 50 U.S. & D.C., 14+. Visit https://www.crunchlabs.com/pages/crun... for Official Rules including full details on eligibility requirements, how to enter, entry periods, free method of entry, entry limits, prize claim procedure, prize description and limitations. Void where prohibited.
        """,
        original_tags=[],
        transcript="",
        competitor_analytics_data={},
        category_name="",
        user_id=1
    )

    # Run the optimization
    thumbnail_optimization_results = do_thumbnail_optimization(
        video_id="y8yDBm7PvbM",
        original_title="Testing What Happens If You Jump On A Moving Train",
        original_description="""Sometimes you gotta go full Tom Cruise to really teach the science...""",
        original_tags=[],
        transcript="",
        competitor_analytics_data={},
        category_name="",
        user_id=1
    )

    if thumbnail_optimization_results:
        # Create output directory
        output_dir = Path("thumbnail_optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        raw_output = output_dir / f"thumbnail_optimization_raw_{timestamp}.json"
        with open(raw_output, 'w') as f:
            json.dump(thumbnail_optimization_results, f, indent=2, default=str)
        print(f"\nSaved raw results to: {raw_output}")

        # Process and save formatted results
        formatted_results = []
        frames = thumbnail_optimization_results.get('frames', [])
        
        for i, frame in enumerate(frames, 1):
            frame_result = {
                "frame_number": i,
                "original_frame": frame.get('original_frame'),
                "optimized_thumbnail": frame.get('optimized_thumbnail'),
                "prompt": frame.get('prompt'),
                "evaluation": format_evaluation_results(frame.get('evaluation')) if frame.get('evaluation') else None
            }
            formatted_results.append(frame_result)

        # Save formatted results
        formatted_output = output_dir / f"thumbnail_optimization_formatted_{timestamp}.json"
        with open(formatted_output, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        print(f"Saved formatted results to: {formatted_output}")

        # Print first thumbnail's evaluation
        if formatted_results and formatted_results[0].get('evaluation'):
            print("\nFirst Thumbnail Evaluation:")
            print(formatted_results[0]['evaluation'])
    else:
        print("Thumbnail optimization failed")

    

    video_id = "aKq8bkY5eTU" # Example URL
    local_video_filename = "test_video.mp4"
    thumbnail_output_dir = "thumbnails"

    #download_youtube_video_and_extract_frames(video_id, local_video_filename, thumbnail_output_dir)

    #extracted_frames = extract_frames_at_suggested_times("test_video.mp4")

    extracted_final_frame_paths = [
        '/Users/jasonramirez/Documents/youtube-optimizer/backend/services/extracted_thumbnails/orig_test_video_llm_sharp_thumb_01_at_1_370s_MansurroundedbysnakesCapt.jpg',
        '/Users/jasonramirez/Documents/youtube-optimizer/backend/services/extracted_thumbnails/orig_test_video_llm_sharp_thumb_02_at_34_350s_ManholdingasteakDrawsatt.jpg',
        '/Users/jasonramirez/Documents/youtube-optimizer/backend/services/extracted_thumbnails/orig_test_video_llm_sharp_thumb_03_at_51_340s_Twolionssurroundingsleeping.jpg',
        '/Users/jasonramirez/Documents/youtube-optimizer/backend/services/extracted_thumbnails/orig_test_video_llm_sharp_thumb_04_at_37_040s_Drivingonadangerousnarrow.jpg',
        '/Users/jasonramirez/Documents/youtube-optimizer/backend/services/extracted_thumbnails/orig_test_video_llm_sharp_thumb_05_at_76_122s_Sharkapproachingthescreen.jpg'
    ]

    optimize_thumbnail_with_openai(
        extracted_final_frame_paths[0],
        extracted_final_frame_paths[0].replace(".jpg","_output.jpg")
    )

    if False:
        # Clean up previous files to avoid confusion
        if os.path.exists(local_video_filename):
            logging.info(f"Removing previous video file: {local_video_filename}")
            os.remove(local_video_filename)

        # Create thumbnail directory if needed
        Path(thumbnail_output_dir).mkdir(parents=True, exist_ok=True)

        # Method 1: Direct download from URL if available
        video_url = "https://sieve-prod-us-central1-persistent-bucket.storage.googleapis.com/b14c32c0-74bd-47c9-ae62-1d578af7ae21/d0678f2d-edc0-48cc-b770-cd07fd5480b6/ff7361c9-2ed6-42bf-962d-38c46211c893/tmp_o1d2u8m.mp4?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=abhi-admin%40sieve-grapefruit.iam.gserviceaccount.com%2F20250502%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250502T131017Z&X-Goog-Expires=172800&X-Goog-SignedHeaders=host&x-goog-signature=2e1addc14cc5c8b16a8fc6a3014dbcb2b2cfc1e352fa53b86501335b8d4470dccb136a422749198d82ca1a1027ad090ae684491749d766db8e8425b8c42f880468e5826eea82c78bdb633f2c413ae50c4c5fe01524eebd67fa40f0256bb5f59d7822d1c5add854adf8f4d215d97380a6b69ffeb1970f211e4afbfca69513e066e589172e50824768b027c461d2a67bad397958ffad8f0ffed16d861d6cb7adbba842f58612788aef427dc3036167430b8e43f02d84e409d27b9fb0163e9a2f86c997786b9effbbf93e2588babeebd12ac0a6b7441cc9de88aa930210b0841fd63ddf6468fb6374c8362b416722fcecfcb62e6c5e6477bfb9f2acb77a24155ea4"

        if video_url:
            if download_video_from_url(video_url, local_video_filename):
                extracted_thumbnails = extract_thumbnails_ffmpeg(
                    video_path=local_video_filename,
                    output_dir=thumbnail_output_dir,
                    num_thumbnails=10
                )

                if extracted_thumbnails:
                    logging.info("--- Thumbnail Extraction Completed ---")
                    logging.info(f"Generated {len(extracted_thumbnails)} thumbnail candidates")
                    logging.info(f"Thumbnails saved to: {thumbnail_output_dir}")
                    logging.info(f"Thumbnail files: {extracted_thumbnails}")
                else:
                    logging.error("--- Thumbnail Extraction Failed ---")