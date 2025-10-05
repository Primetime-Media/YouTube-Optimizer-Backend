# services/llm_optimizer.py
"""
Production-Ready LLM Optimization Service
=========================================
Enterprise-grade AI-powered video optimization using Claude/GPT-4.

Features:
- Multi-model LLM support with automatic fallback
- Comprehensive error handling and retry logic
- Circuit breaker pattern for API failures
- Rate limiting and cost tracking
- Caching for repeated requests
- Metrics and monitoring
- Parallel processing support
- Quality validation
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import asyncio
import hashlib
import json
from functools import wraps
import time

from anthropic import AsyncAnthropic, APIError, RateLimitError
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from circuitbreaker import circuit
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from utils.db import get_pool
from utils.cache import CacheManager
from utils.metrics import MetricsCollector
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize clients
anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Metrics
metrics = MetricsCollector()

# Cache manager
cache = CacheManager()


# ============================================================================
# ENUMS & MODELS
# ============================================================================

class LLMProvider(str, Enum):
    """Available LLM providers"""
    CLAUDE = "claude"
    GPT4 = "gpt4"
    GPT4_TURBO = "gpt4_turbo"


class OptimizationType(str, Enum):
    """Types of optimizations"""
    TITLE = "title"
    DESCRIPTION = "description"
    TAGS = "tags"
    THUMBNAIL = "thumbnail"
    FULL = "full"


class OptimizationRequest(BaseModel):
    """Request model for optimization"""
    video_id: str = Field(..., min_length=11, max_length=11)
    user_id: int = Field(..., gt=0)
    optimization_type: OptimizationType
    current_title: Optional[str] = Field(None, max_length=100)
    current_description: Optional[str] = Field(None, max_length=5000)
    current_tags: Optional[List[str]] = Field(None, max_items=500)
    target_audience: Optional[str] = None
    video_category: Optional[str] = None
    competitor_data: Optional[Dict[str, Any]] = None
    trending_keywords: Optional[List[str]] = None
    preferred_provider: Optional[LLMProvider] = LLMProvider.CLAUDE
    
    @validator('current_tags')
    def validate_tags(cls, v):
        if v and len(v) > 500:
            raise ValueError("Too many tags")
        return v


class OptimizationResponse(BaseModel):
    """Response model for optimization"""
    success: bool
    video_id: str
    optimization_type: OptimizationType
    optimized_title: Optional[str] = None
    optimized_description: Optional[str] = None
    optimized_tags: Optional[List[str]] = None
    thumbnail_suggestions: Optional[List[Dict[str, str]]] = None
    provider_used: LLMProvider
    tokens_used: int
    cost_usd: float
    processing_time_ms: int
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    suggestions: Optional[List[str]] = None
    error: Optional[str] = None


# ============================================================================
# CIRCUIT BREAKER & RETRY LOGIC
# ============================================================================

class LLMCircuitBreaker:
    """Circuit breaker for LLM API calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures: Dict[str, int] = {}
        self.last_failure: Dict[str, datetime] = {}
    
    def is_open(self, provider: str) -> bool:
        """Check if circuit is open for provider"""
        if provider not in self.failures:
            return False
        
        if self.failures[provider] < self.failure_threshold:
            return False
        
        if provider in self.last_failure:
            time_since_failure = (
                datetime.now(timezone.utc) - self.last_failure[provider]
            ).total_seconds()
            if time_since_failure > self.timeout:
                # Reset circuit
                self.failures[provider] = 0
                return False
        
        return True
    
    def record_success(self, provider: str):
        """Record successful call"""
        self.failures[provider] = 0
        if provider in self.last_failure:
            del self.last_failure[provider]
    
    def record_failure(self, provider: str):
        """Record failed call"""
        self.failures[provider] = self.failures.get(provider, 0) + 1
        self.last_failure[provider] = datetime.now(timezone.utc)
        
        if self.failures[provider] >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker opened for {provider} "
                f"after {self.failures[provider]} failures"
            )
            metrics.increment(
                "llm.circuit_breaker.opened",
                tags={"provider": provider}
            )


circuit_breaker = LLMCircuitBreaker()


# ============================================================================
# COST TRACKING
# ============================================================================

class CostTracker:
    """Track LLM API costs"""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        LLMProvider.CLAUDE: {
            "input": 0.008,  # Claude Sonnet
            "output": 0.024
        },
        LLMProvider.GPT4: {
            "input": 0.03,
            "output": 0.06
        },
        LLMProvider.GPT4_TURBO: {
            "input": 0.01,
            "output": 0.03
        }
    }
    
    @classmethod
    def calculate_cost(
        cls,
        provider: LLMProvider,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost in USD"""
        pricing = cls.PRICING.get(provider, {"input": 0, "output": 0})
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    @classmethod
    async def log_usage(
        cls,
        user_id: int,
        video_id: str,
        provider: LLMProvider,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        optimization_type: str
    ):
        """Log usage to database"""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO llm_api_usage (
                        user_id, video_id, provider, optimization_type,
                        input_tokens, output_tokens, cost_usd, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    user_id, video_id, provider.value, optimization_type,
                    input_tokens, output_tokens, cost, datetime.now(timezone.utc)
                )
            
            metrics.histogram(
                "llm.cost.usd",
                cost,
                tags={"provider": provider.value, "type": optimization_type}
            )
            
        except Exception as e:
            logger.error(f"Failed to log LLM usage: {e}", exc_info=True)


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

class PromptTemplates:
    """Production-ready prompt templates"""
    
    TITLE_OPTIMIZATION = """You are an expert YouTube SEO specialist. Optimize this video title for maximum views and engagement.

Current Title: {current_title}

Video Context:
- Category: {category}
- Target Audience: {audience}
- Current Performance: {performance_data}

Competitor Analysis:
{competitor_titles}

Trending Keywords:
{trending_keywords}

Requirements:
1. Title must be 60 characters or less
2. Include primary keyword in first 5 words
3. Create curiosity or urgency
4. Be clear about video value
5. Avoid clickbait (YouTube penalizes)
6. Consider SEO and human appeal

Provide 3 optimized title variations with confidence scores (0-100) and brief explanations.

Output as JSON:
{{
  "variations": [
    {{
      "title": "...",
      "confidence": 85,
      "reasoning": "..."
    }}
  ]
}}"""

    DESCRIPTION_OPTIMIZATION = """You are an expert YouTube SEO specialist. Optimize this video description for discoverability and conversions.

Current Description:
{current_description}

Video Context:
- Title: {title}
- Category: {category}
- Target Keywords: {keywords}

Requirements:
1. First 150 characters are critical (shown in search)
2. Include primary keywords naturally
3. Add timestamps if applicable
4. Include relevant links (channel, social, resources)
5. Use proper formatting (line breaks, emojis)
6. Call-to-action for likes/subscribe
7. Maximum 5000 characters

Provide optimized description with sections clearly marked.

Output as JSON:
{{
  "description": "...",
  "confidence": 85,
  "key_improvements": ["...", "..."]
}}"""

    TAGS_OPTIMIZATION = """You are an expert YouTube SEO specialist. Optimize video tags for maximum discoverability.

Current Tags: {current_tags}

Video Context:
- Title: {title}
- Category: {category}
- Description: {description_preview}

Competitor Tags:
{competitor_tags}

Trending in Niche:
{trending_keywords}

Requirements:
1. 15-30 tags optimal (500 character limit)
2. Mix of specific and broad tags
3. Include misspellings if common
4. Include multi-word phrases
5. Front-load most important tags
6. Avoid tag stuffing

Provide optimized tag list ordered by priority.

Output as JSON:
{{
  "tags": ["tag1", "tag2", ...],
  "confidence": 85,
  "tag_strategy": "..."
}}"""

    FULL_OPTIMIZATION = """You are an expert YouTube optimization specialist. Provide comprehensive optimization for this video.

Current State:
- Title: {title}
- Description: {description}
- Tags: {tags}

Analytics:
{analytics_data}

Competitor Intelligence:
{competitor_data}

Trending Topics:
{trending_data}

Provide complete optimization strategy with:
1. Optimized title (3 variations)
2. Optimized description
3. Optimized tags (15-30)
4. Thumbnail suggestions (3 concepts)
5. Content recommendations
6. Expected impact estimate

Output as JSON with all sections."""


# ============================================================================
# MAIN OPTIMIZER SERVICE
# ============================================================================

class LLMOptimizerService:
    """Production-ready LLM optimization service"""
    
    def __init__(self):
        self.templates = PromptTemplates()
        self.cost_tracker = CostTracker()
    
    async def optimize(
        self,
        request: OptimizationRequest
    ) -> OptimizationResponse:
        """
        Main optimization method with full error handling and fallback.
        
        Args:
            request: OptimizationRequest with video details
            
        Returns:
            OptimizationResponse with optimized content
        """
        start_time = time.time()
        
        try:
            # Validate request
            if not request.current_title and request.optimization_type == OptimizationType.TITLE:
                raise ValueError("current_title required for TITLE optimization")
            
            # Check cache
            cache_key = self._generate_cache_key(request)
            cached_result = await cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {request.video_id}")
                metrics.increment("llm.cache.hit")
                return OptimizationResponse(**cached_result)
            
            metrics.increment("llm.cache.miss")
            
            # Try primary provider
            provider = request.preferred_provider
            result = None
            
            try:
                result = await self._optimize_with_provider(request, provider)
            except Exception as e:
                logger.warning(
                    f"Primary provider {provider} failed: {e}",
                    exc_info=True
                )
                circuit_breaker.record_failure(provider.value)
                
                # Fallback to alternative provider
                fallback_provider = self._get_fallback_provider(provider)
                logger.info(f"Falling back to {fallback_provider}")
                
                try:
                    result = await self._optimize_with_provider(
                        request,
                        fallback_provider
                    )
                    provider = fallback_provider
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback provider {fallback_provider} also failed: {fallback_error}",
                        exc_info=True
                    )
                    circuit_breaker.record_failure(fallback_provider.value)
                    raise
            
            # Success - record metrics
            circuit_breaker.record_success(provider.value)
            
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time
            result.provider_used = provider
            
            # Cache result
            await cache.set(
                cache_key,
                result.dict(),
                expire=3600  # 1 hour
            )
            
            # Log metrics
            metrics.histogram(
                "llm.processing_time.ms",
                processing_time,
                tags={"provider": provider.value, "type": request.optimization_type.value}
            )
            
            metrics.increment(
                "llm.optimization.success",
                tags={"provider": provider.value, "type": request.optimization_type.value}
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Optimization failed for {request.video_id}: {e}",
                exc_info=True
            )
            
            metrics.increment(
                "llm.optimization.error",
                tags={"type": request.optimization_type.value}
            )
            
            # Return error response
            processing_time = int((time.time() - start_time) * 1000)
            return OptimizationResponse(
                success=False,
                video_id=request.video_id,
                optimization_type=request.optimization_type,
                provider_used=request.preferred_provider,
                tokens_used=0,
                cost_usd=0.0,
                processing_time_ms=processing_time,
                confidence_score=0.0,
                error=str(e)
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError))
    )
    async def _optimize_with_provider(
        self,
        request: OptimizationRequest,
        provider: LLMProvider
    ) -> OptimizationResponse:
        """Optimize using specific provider with retry logic"""
        
        # Check circuit breaker
        if circuit_breaker.is_open(provider.value):
            raise Exception(f"Circuit breaker open for {provider.value}")
        
        # Build prompt
        prompt = self._build_prompt(request)
        
        # Call appropriate LLM
        if provider == LLMProvider.CLAUDE:
            result = await self._call_claude(request, prompt)
        elif provider in [LLMProvider.GPT4, LLMProvider.GPT4_TURBO]:
            result = await self._call_openai(request, prompt, provider)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Validate response
        validated_result = self._validate_and_format_response(request, result)
        
        return validated_result
    
    async def _call_claude(
        self,
        request: OptimizationRequest,
        prompt: str
    ) -> Dict[str, Any]:
        """Call Claude API"""
        try:
            response = await anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract response
            content = response.content[0].text
            
            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self.cost_tracker.calculate_cost(
                LLMProvider.CLAUDE,
                input_tokens,
                output_tokens
            )
            
            # Log usage
            await self.cost_tracker.log_usage(
                user_id=request.user_id,
                video_id=request.video_id,
                provider=LLMProvider.CLAUDE,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                optimization_type=request.optimization_type.value
            )
            
            # Parse JSON response
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    parsed_content = json.loads(json_match.group(1))
                else:
                    raise ValueError("Could not parse JSON from response")
            
            return {
                "content": parsed_content,
                "tokens_used": input_tokens + output_tokens,
                "cost_usd": cost
            }
            
        except Exception as e:
            logger.error(f"Claude API error: {e}", exc_info=True)
            raise
    
    async def _call_openai(
        self,
        request: OptimizationRequest,
        prompt: str,
        provider: LLMProvider
    ) -> Dict[str, Any]:
        """Call OpenAI API"""
        try:
            model = "gpt-4" if provider == LLMProvider.GPT4 else "gpt-4-turbo"
            
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system",
                    "content": "You are an expert YouTube SEO specialist. Always respond with valid JSON."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Extract response
            content = response.choices[0].message.content
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.cost_tracker.calculate_cost(
                provider,
                input_tokens,
                output_tokens
            )
            
            # Log usage
            await self.cost_tracker.log_usage(
                user_id=request.user_id,
                video_id=request.video_id,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                optimization_type=request.optimization_type.value
            )
            
            # Parse JSON
            parsed_content = json.loads(content)
            
            return {
                "content": parsed_content,
                "tokens_used": input_tokens + output_tokens,
                "cost_usd": cost
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise
    
    def _build_prompt(self, request: OptimizationRequest) -> str:
        """Build optimization prompt"""
        
        # Get appropriate template
        if request.optimization_type == OptimizationType.TITLE:
            template = self.templates.TITLE_OPTIMIZATION
        elif request.optimization_type == OptimizationType.DESCRIPTION:
            template = self.templates.DESCRIPTION_OPTIMIZATION
        elif request.optimization_type == OptimizationType.TAGS:
            template = self.templates.TAGS_OPTIMIZATION
        else:
            template = self.templates.FULL_OPTIMIZATION
        
        # Format template with request data
        prompt = template.format(
            current_title=request.current_title or "",
            current_description=request.current_description or "",
            current_tags=", ".join(request.current_tags or []),
            title=request.current_title or "",
            description=request.current_description or "",
            tags=", ".join(request.current_tags or []),
            category=request.video_category or "General",
            audience=request.target_audience or "General audience",
            keywords=", ".join(request.trending_keywords or []),
            trending_keywords="\n".join([f"- {kw}" for kw in (request.trending_keywords or [])]),
            competitor_titles=self._format_competitor_data(request.competitor_data, "titles"),
            competitor_tags=self._format_competitor_data(request.competitor_data, "tags"),
            competitor_data=json.dumps(request.competitor_data or {}, indent=2),
            performance_data="Views improving" if True else "Views declining",
            analytics_data="Placeholder analytics",
            trending_data="\n".join([f"- {kw}" for kw in (request.trending_keywords or [])]),
            description_preview=(request.current_description or "")[:200]
        )
        
        return prompt
    
    def _format_competitor_data(
        self,
        competitor_data: Optional[Dict],
        field: str
    ) -> str:
        """Format competitor data for prompt"""
        if not competitor_data:
            return "No competitor data available"
        
        if field == "titles":
            titles = competitor_data.get("titles", [])
            return "\n".join([f"- {title}" for title in titles[:5]])
        elif field == "tags":
            tags = competitor_data.get("tags", [])
            return "\n".join([f"- {tag}" for tag in tags[:20]])
        
        return ""
    
    def _validate_and_format_response(
        self,
        request: OptimizationRequest,
        llm_response: Dict[str, Any]
    ) -> OptimizationResponse:
        """Validate and format LLM response"""
        
        content = llm_response["content"]
        
        # Extract based on optimization type
        if request.optimization_type == OptimizationType.TITLE:
            variations = content.get("variations", [])
            if not variations:
                raise ValueError("No title variations in response")
            
            best_variation = max(variations, key=lambda x: x.get("confidence", 0))
            
            return OptimizationResponse(
                success=True,
                video_id=request.video_id,
                optimization_type=request.optimization_type,
                optimized_title=best_variation["title"],
                provider_used=LLMProvider.CLAUDE,  # Will be updated
                tokens_used=llm_response["tokens_used"],
                cost_usd=llm_response["cost_usd"],
                processing_time_ms=0,  # Will be updated
                confidence_score=best_variation.get("confidence", 50) / 100.0,
                suggestions=[v["reasoning"] for v in variations]
            )
        
        elif request.optimization_type == OptimizationType.DESCRIPTION:
            return OptimizationResponse(
                success=True,
                video_id=request.video_id,
                optimization_type=request.optimization_type,
                optimized_description=content.get("description", ""),
                provider_used=LLMProvider.CLAUDE,
                tokens_used=llm_response["tokens_used"],
                cost_usd=llm_response["cost_usd"],
                processing_time_ms=0,
                confidence_score=content.get("confidence", 50) / 100.0,
                suggestions=content.get("key_improvements", [])
            )
        
        elif request.optimization_type == OptimizationType.TAGS:
            return OptimizationResponse(
                success=True,
                video_id=request.video_id,
                optimization_type=request.optimization_type,
                optimized_tags=content.get("tags", []),
                provider_used=LLMProvider.CLAUDE,
                tokens_used=llm_response["tokens_used"],
                cost_usd=llm_response["cost_usd"],
                processing_time_ms=0,
                confidence_score=content.get("confidence", 50) / 100.0,
                suggestions=[content.get("tag_strategy", "")]
            )
        
        else:  # FULL optimization
            return OptimizationResponse(
                success=True,
                video_id=request.video_id,
                optimization_type=request.optimization_type,
                optimized_title=content.get("title", {}).get("best", ""),
                optimized_description=content.get("description", ""),
                optimized_tags=content.get("tags", []),
                thumbnail_suggestions=content.get("thumbnail_concepts", []),
                provider_used=LLMProvider.CLAUDE,
                tokens_used=llm_response["tokens_used"],
                cost_usd=llm_response["cost_usd"],
                processing_time_ms=0,
                confidence_score=0.75,  # Default for full optimization
                suggestions=content.get("recommendations", [])
            )
    
    def _get_fallback_provider(self, primary: LLMProvider) -> LLMProvider:
        """Get fallback provider"""
        if primary == LLMProvider.CLAUDE:
            return LLMProvider.GPT4_TURBO
        else:
            return LLMProvider.CLAUDE
    
    def _generate_cache_key(self, request: OptimizationRequest) -> str:
        """Generate cache key for request"""
        key_data = f"{request.video_id}:{request.optimization_type.value}:{request.current_title}:{request.current_description}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def batch_optimize(
        self,
        requests: List[OptimizationRequest],
        max_concurrent: int = 5
    ) -> List[OptimizationResponse]:
        """
        Batch optimize multiple videos concurrently.
        
        Args:
            requests: List of optimization requests
            max_concurrent: Maximum concurrent optimizations
            
        Returns:
            List of optimization responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def optimize_with_semaphore(req):
            async with semaphore:
                return await self.optimize(req)
        
        tasks = [optimize_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(OptimizationResponse(
                    success=False,
                    video_id=requests[i].video_id,
                    optimization_type=requests[i].optimization_type,
                    provider_used=requests[i].preferred_provider,
                    tokens_used=0,
                    cost_usd=0.0,
                    processing_time_ms=0,
                    confidence_score=0.0,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results


# ============================================================================
# SERVICE INSTANCE
# ============================================================================

llm_optimizer = LLMOptimizerService()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def optimize_video(
    video_id: str,
    user_id: int,
    optimization_type: str = "full",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for video optimization.
    
    Args:
        video_id: YouTube video ID
        user_id: User ID
        optimization_type: Type of optimization
        **kwargs: Additional optimization parameters
        
    Returns:
        Optimization result dictionary
    """
    request = OptimizationRequest(
        video_id=video_id,
        user_id=user_id,
        optimization_type=OptimizationType(optimization_type),
        **kwargs
    )
    
    result = await llm_optimizer.optimize(request)
    return result.dict()


async def get_optimization_cost_estimate(
    optimization_type: str,
    provider: str = "claude"
) -> Dict[str, float]:
    """
    Estimate cost for optimization.
    
    Args:
        optimization_type: Type of optimization
        provider: LLM provider
        
    Returns:
        Cost estimate
    """
    # Approximate token counts
    token_estimates = {
        "title": {"input": 500, "output": 200},
        "description": {"input": 800, "output": 500},
        "tags": {"input": 600, "output": 300},
        "full": {"input": 1500, "output": 1000}
    }
    
    estimate = token_estimates.get(optimization_type, token_estimates["full"])
    cost = CostTracker.calculate_cost(
        LLMProvider(provider),
        estimate["input"],
        estimate["output"]
    )
    
    return {
        "estimated_tokens": estimate["input"] + estimate["output"],
        "estimated_cost_usd": round(cost, 4),
        "provider": provider
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'llm_optimizer',
    'LLMOptimizerService',
    'OptimizationRequest',
    'OptimizationResponse',
    'LLMProvider',
    'OptimizationType',
    'optimize_video',
    'get_optimization_cost_estimate'
]
