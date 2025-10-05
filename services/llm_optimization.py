# services/optimizer.py
"""
Production-Ready Video Optimizer Service
=========================================
Orchestrates complete video optimization workflow.

Features:
- End-to-end optimization pipeline
- Multi-stage optimization (title, description, tags, thumbnail)
- A/B testing support
- Rollback capability
- Performance tracking
- Rate limiting and cost management
- Comprehensive error handling and logging
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field, validator

from services.llm_optimizer import (
    llm_optimizer,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationType,
    LLMProvider
)
from services.youtube import (
    update_video_metadata,
    get_video_details,
    get_video_analytics
)
from utils.db import get_pool
from utils.metrics import MetricsCollector
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Initialize metrics
metrics = MetricsCollector()


# ============================================================================
# ENUMS & MODELS
# ============================================================================

class OptimizationStage(str, Enum):
    """Stages of optimization process"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"
    APPLICATION = "application"
    MONITORING = "monitoring"
    ROLLBACK = "rollback"


class OptimizationStrategy(str, Enum):
    """Optimization strategies"""
    CONSERVATIVE = "conservative"  # Only high-confidence changes
    BALANCED = "balanced"  # Default strategy
    AGGRESSIVE = "aggressive"  # Try all suggestions
    AB_TEST = "ab_test"  # Create A/B test variations


class VideoOptimizationStatus(str, Enum):
    """Status of video optimization"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class BaselineMetrics:
    """Baseline performance metrics before optimization"""
    views: int
    likes: int
    comments: int
    watch_time: int
    ctr: float
    avg_view_duration: int
    revenue: float
    measured_at: datetime


class VideoOptimizationRequest(BaseModel):
    """Complete video optimization request"""
    video_id: str = Field(..., min_length=11, max_length=11)
    user_id: int = Field(..., gt=0)
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    optimization_types: List[OptimizationType] = Field(
        default=[OptimizationType.TITLE, OptimizationType.DESCRIPTION, OptimizationType.TAGS]
    )
    auto_apply: bool = False  # If True, apply changes automatically
    require_approval: bool = True  # If True, wait for user approval
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_ab_test: bool = False
    rollback_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Rollback if performance drops below this % of baseline"
    )
    monitoring_days: int = Field(default=7, ge=1, le=30)
    
    @validator('optimization_types')
    def validate_optimization_types(cls, v):
        if not v:
            raise ValueError("At least one optimization type required")
        if len(v) > len(set(v)):
            raise ValueError("Duplicate optimization types")
        return v


class OptimizationResult(BaseModel):
    """Complete optimization result"""
    optimization_id: int
    video_id: str
    user_id: int
    status: VideoOptimizationStatus
    strategy: OptimizationStrategy
    
    # Original values
    original_title: str
    original_description: str
    original_tags: List[str]
    
    # Optimized values
    optimized_title: Optional[str] = None
    optimized_description: Optional[str] = None
    optimized_tags: Optional[List[str]] = None
    
    # Baseline metrics
    baseline_views: int
    baseline_ctr: float
    baseline_revenue: float
    
    # Performance after optimization
    current_views: Optional[int] = None
    current_ctr: Optional[float] = None
    current_revenue: Optional[float] = None
    views_increase_pct: Optional[float] = None
    revenue_increase: Optional[float] = None
    
    # Metadata
    confidence_score: float
    total_cost_usd: float
    applied_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    # Status info
    stage: OptimizationStage
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None


# ============================================================================
# OPTIMIZER SERVICE
# ============================================================================

class VideoOptimizerService:
    """Production-ready video optimization orchestrator"""
    
    def __init__(self):
        self.max_concurrent_optimizations = 5
        self._optimization_semaphore = asyncio.Semaphore(self.max_concurrent_optimizations)
    
    async def optimize_video(
        self,
        request: VideoOptimizationRequest
    ) -> OptimizationResult:
        """
        Complete video optimization workflow.
        
        Stages:
        1. Analysis - Gather video data and baseline metrics
        2. Generation - Generate optimized content using LLM
        3. Validation - Validate optimized content
        4. Application - Apply changes to YouTube (if auto_apply)
        5. Monitoring - Track performance
        6. Rollback - Revert if performance drops
        
        Args:
            request: VideoOptimizationRequest
            
        Returns:
            OptimizationResult
        """
        async with self._optimization_semaphore:
            optimization_id = None
            
            try:
                # Stage 1: Analysis
                logger.info(f"Starting optimization for video {request.video_id}")
                metrics.increment("optimizer.optimization.started")
                
                video_data, baseline_metrics = await self._analyze_video(
                    request.video_id,
                    request.user_id
                )
                
                # Create optimization record
                optimization_id = await self._create_optimization_record(
                    request,
                    video_data,
                    baseline_metrics
                )
                
                # Stage 2: Generation
                await self._update_optimization_stage(
                    optimization_id,
                    OptimizationStage.GENERATION
                )
                
                optimized_content = await self._generate_optimizations(
                    request,
                    video_data,
                    baseline_metrics
                )
                
                # Stage 3: Validation
                await self._update_optimization_stage(
                    optimization_id,
                    OptimizationStage.VALIDATION
                )
                
                validated_content = await self._validate_optimizations(
                    video_data,
                    optimized_content,
                    request.confidence_threshold
                )
                
                # Store generated optimizations
                await self._store_optimizations(
                    optimization_id,
                    validated_content
                )
                
                # Stage 4: Application (if auto_apply)
                if request.auto_apply and not request.require_approval:
                    await self._update_optimization_stage(
                        optimization_id,
                        OptimizationStage.APPLICATION
                    )
                    
                    await self._apply_optimizations(
                        request.video_id,
                        request.user_id,
                        validated_content
                    )
                    
                    # Mark as applied
                    await self._mark_optimization_applied(optimization_id)
                    
                    # Stage 5: Start monitoring
                    await self._update_optimization_stage(
                        optimization_id,
                        OptimizationStage.MONITORING
                    )
                    
                    # Schedule performance monitoring
                    await self._schedule_performance_monitoring(
                        optimization_id,
                        request.monitoring_days,
                        request.rollback_threshold
                    )
                
                # Mark as completed
                await self._mark_optimization_completed(optimization_id)
                
                # Get final result
                result = await self._get_optimization_result(optimization_id)
                
                logger.info(f"Optimization completed for video {request.video_id}")
                metrics.increment("optimizer.optimization.completed")
                
                return result
                
            except Exception as e:
                logger.error(
                    f"Optimization failed for video {request.video_id}: {e}",
                    exc_info=True
                )
                metrics.increment("optimizer.optimization.failed")
                
                if optimization_id:
                    await self._mark_optimization_failed(
                        optimization_id,
                        str(e)
                    )
                
                raise
    
    async def _analyze_video(
        self,
        video_id: str,
        user_id: int
    ) -> Tuple[Dict[str, Any], BaselineMetrics]:
        """Analyze video and gather baseline metrics"""
        try:
            # Get video details
            video_data = await get_video_details(video_id, user_id)
            if not video_data:
                raise ValueError(f"Video not found: {video_id}")
            
            # Get baseline analytics
            analytics = await get_video_analytics(video_id, user_id)
            
            baseline = BaselineMetrics(
                views=analytics.get('views', 0),
                likes=analytics.get('likes', 0),
                comments=analytics.get('comments', 0),
                watch_time=analytics.get('watch_time', 0),
                ctr=analytics.get('ctr', 0.0),
                avg_view_duration=analytics.get('avg_view_duration', 0),
                revenue=analytics.get('revenue', 0.0),
                measured_at=datetime.now(timezone.utc)
            )
            
            logger.info(
                f"Baseline metrics for {video_id}: "
                f"{baseline.views} views, {baseline.ctr:.2%} CTR, "
                f"${baseline.revenue:.2f} revenue"
            )
            
            return video_data, baseline
            
        except Exception as e:
            logger.error(f"Failed to analyze video {video_id}: {e}", exc_info=True)
            raise
    
    async def _generate_optimizations(
        self,
        request: VideoOptimizationRequest,
        video_data: Dict[str, Any],
        baseline_metrics: BaselineMetrics
    ) -> Dict[str, Any]:
        """Generate optimized content using LLM"""
        try:
            optimizations = {}
            total_cost = 0.0
            
            # Optimize each requested type
            for opt_type in request.optimization_types:
                llm_request = OptimizationRequest(
                    video_id=request.video_id,
                    user_id=request.user_id,
                    optimization_type=opt_type,
                    current_title=video_data.get('title'),
                    current_description=video_data.get('description'),
                    current_tags=video_data.get('tags', []),
                    video_category=video_data.get('category'),
                    target_audience=video_data.get('target_audience')
                )
                
                result = await llm_optimizer.optimize(llm_request)
                
                if not result.success:
                    logger.warning(
                        f"Optimization failed for {opt_type}: {result.error}"
                    )
                    continue
                
                optimizations[opt_type.value] = {
                    'content': self._extract_content(result, opt_type),
                    'confidence': result.confidence_score,
                    'cost': result.cost_usd,
                    'provider': result.provider_used.value,
                    'suggestions': result.suggestions
                }
                
                total_cost += result.cost_usd
            
            optimizations['total_cost'] = total_cost
            
            logger.info(
                f"Generated {len(optimizations)} optimizations "
                f"for {request.video_id}, cost: ${total_cost:.4f}"
            )
            
            return optimizations
            
        except Exception as e:
            logger.error(
                f"Failed to generate optimizations for {request.video_id}: {e}",
                exc_info=True
            )
            raise
    
    def _extract_content(
        self,
        result: OptimizationResponse,
        opt_type: OptimizationType
    ) -> Any:
        """Extract optimized content from LLM response"""
        if opt_type == OptimizationType.TITLE:
            return result.optimized_title
        elif opt_type == OptimizationType.DESCRIPTION:
            return result.optimized_description
        elif opt_type == OptimizationType.TAGS:
            return result.optimized_tags
        elif opt_type == OptimizationType.THUMBNAIL:
            return result.thumbnail_suggestions
        else:
            return {
                'title': result.optimized_title,
                'description': result.optimized_description,
                'tags': result.optimized_tags,
                'thumbnail': result.thumbnail_suggestions
            }
    
    async def _validate_optimizations(
        self,
        video_data: Dict[str, Any],
        optimizations: Dict[str, Any],
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Validate and filter optimizations based on confidence"""
        validated = {}
        
        for opt_type, opt_data in optimizations.items():
            if opt_type == 'total_cost':
                validated[opt_type] = opt_data
                continue
            
            # Check confidence threshold
            if opt_data['confidence'] < confidence_threshold:
                logger.warning(
                    f"Skipping {opt_type} optimization: "
                    f"confidence {opt_data['confidence']:.2f} < "
                    f"threshold {confidence_threshold:.2f}"
                )
                continue
            
            # Additional validation rules
            if opt_type == 'title':
                if not self._validate_title(opt_data['content']):
                    logger.warning(f"Title validation failed: {opt_data['content']}")
                    continue
            
            elif opt_type == 'description':
                if not self._validate_description(opt_data['content']):
                    logger.warning("Description validation failed")
                    continue
            
            elif opt_type == 'tags':
                if not self._validate_tags(opt_data['content']):
                    logger.warning("Tags validation failed")
                    continue
            
            validated[opt_type] = opt_data
        
        logger.info(
            f"Validated {len(validated)} optimizations "
            f"(filtered {len(optimizations) - len(validated) - 1})"
        )
        
        return validated
    
    def _validate_title(self, title: str) -> bool:
        """Validate title meets YouTube requirements"""
        if not title or not isinstance(title, str):
            return False
        if len(title) > 100:
            return False
        if len(title.strip()) < 5:
            return False
        return True
    
    def _validate_description(self, description: str) -> bool:
        """Validate description meets YouTube requirements"""
        if not description or not isinstance(description, str):
            return False
        if len(description) > 5000:
            return False
        return True
    
    def _validate_tags(self, tags: List[str]) -> bool:
        """Validate tags meet YouTube requirements"""
        if not tags or not isinstance(tags, list):
            return False
        if len(tags) > 500:
            return False
        # Check total character count
        total_chars = sum(len(tag) for tag in tags)
        if total_chars > 500:
            return False
        return True
    
    async def _apply_optimizations(
        self,
        video_id: str,
        user_id: int,
        optimizations: Dict[str, Any]
    ) -> bool:
        """Apply optimizations to YouTube video"""
        try:
            # Build update payload
            update_data = {}
            
            if 'title' in optimizations:
                update_data['title'] = optimizations['title']['content']
            
            if 'description' in optimizations:
                update_data['description'] = optimizations['description']['content']
            
            if 'tags' in optimizations:
                update_data['tags'] = optimizations['tags']['content']
            
            if not update_data:
                logger.warning(f"No optimizations to apply for {video_id}")
                return False
            
            # Apply to YouTube
            success = await update_video_metadata(
                video_id=video_id,
                user_id=user_id,
                **update_data
            )
            
            if success:
                logger.info(f"Successfully applied optimizations to {video_id}")
                metrics.increment("optimizer.optimizations.applied")
            else:
                logger.error(f"Failed to apply optimizations to {video_id}")
                metrics.increment("optimizer.optimizations.application_failed")
            
            return success
            
        except Exception as e:
            logger.error(
                f"Failed to apply optimizations to {video_id}: {e}",
                exc_info=True
            )
            return False
    
    async def _create_optimization_record(
        self,
        request: VideoOptimizationRequest,
        video_data: Dict[str, Any],
        baseline: BaselineMetrics
    ) -> int:
        """Create optimization record in database"""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                optimization_id = await conn.fetchval(
                    """
                    INSERT INTO video_optimizations (
                        video_id, user_id, status, strategy, stage,
                        original_title, original_description, original_tags,
                        baseline_views, baseline_ctr, baseline_revenue,
                        confidence_threshold, auto_apply, rollback_threshold,
                        monitoring_days, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
                    ) RETURNING id
                    """,
                    request.video_id,
                    request.user_id,
                    VideoOptimizationStatus.IN_PROGRESS.value,
                    request.strategy.value,
                    OptimizationStage.ANALYSIS.value,
                    video_data.get('title'),
                    video_data.get('description'),
                    video_data.get('tags', []),
                    baseline.views,
                    baseline.ctr,
                    baseline.revenue,
                    request.confidence_threshold,
                    request.auto_apply,
                    request.rollback_threshold,
                    request.monitoring_days,
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc)
                )
            
            logger.info(f"Created optimization record: {optimization_id}")
            return optimization_id
            
        except Exception as e:
            logger.error(f"Failed to create optimization record: {e}", exc_info=True)
            raise
    
    async def _update_optimization_stage(
        self,
        optimization_id: int,
        stage: OptimizationStage
    ):
        """Update optimization stage"""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE video_optimizations
                    SET stage = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    stage.value,
                    datetime.now(timezone.utc),
                    optimization_id
                )
            
            logger.debug(f"Updated optimization {optimization_id} to stage {stage.value}")
            
        except Exception as e:
            logger.error(
                f"Failed to update optimization stage {optimization_id}: {e}",
                exc_info=True
            )
    
    async def _store_optimizations(
        self,
        optimization_id: int,
        optimizations: Dict[str, Any]
    ):
        """Store generated optimizations"""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE video_optimizations
                    SET 
                        optimized_title = $1,
                        optimized_description = $2,
                        optimized_tags = $3,
                        confidence_score = $4,
                        total_cost_usd = $5,
                        updated_at = $6
                    WHERE id = $7
                    """,
                    optimizations.get('title', {}).get('content'),
                    optimizations.get('description', {}).get('content'),
                    optimizations.get('tags', {}).get('content'),
                    max([opt.get('confidence', 0) for opt in optimizations.values() if isinstance(opt, dict)]),
                    optimizations.get('total_cost', 0.0),
                    datetime.now(timezone.utc),
                    optimization_id
                )
            
            logger.info(f"Stored optimizations for {optimization_id}")
            
        except Exception as e:
            logger.error(
                f"Failed to store optimizations {optimization_id}: {e}",
                exc_info=True
            )
    
    async def _mark_optimization_applied(self, optimization_id: int):
        """Mark optimization as applied"""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE video_optimizations
                    SET applied_at = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    datetime.now(timezone.utc),
                    datetime.now(timezone.utc),
                    optimization_id
                )
        except Exception as e:
            logger.error(f"Failed to mark optimization applied: {e}", exc_info=True)
    
    async def _mark_optimization_completed(self, optimization_id: int):
        """Mark optimization as completed"""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE video_optimizations
                    SET status = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    VideoOptimizationStatus.COMPLETED.value,
                    datetime.now(timezone.utc),
                    optimization_id
                )
        except Exception as e:
            logger.error(f"Failed to mark optimization completed: {e}", exc_info=True)
    
    async def _mark_optimization_failed(
        self,
        optimization_id: int,
        error_message: str
    ):
        """Mark optimization as failed"""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE video_optimizations
                    SET status = $1, error_message = $2, updated_at = $3
                    WHERE id = $4
                    """,
                    VideoOptimizationStatus.FAILED.value,
                    error_message,
                    datetime.now(timezone.utc),
                    optimization_id
                )
        except Exception as e:
            logger.error(f"Failed to mark optimization failed: {e}", exc_info=True)
    
    async def _schedule_performance_monitoring(
        self,
        optimization_id: int,
        monitoring_days: int,
        rollback_threshold: float
    ):
        """Schedule performance monitoring job"""
        # This would integrate with your scheduler service
        # For now, just log
        logger.info(
            f"Scheduled {monitoring_days} days of monitoring "
            f"for optimization {optimization_id}"
        )
        # TODO: Integrate with scheduler_routes to create monitoring job
    
    async def _get_optimization_result(
        self,
        optimization_id: int
    ) -> OptimizationResult:
        """Get complete optimization result"""
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT * FROM video_optimizations WHERE id = $1",
                    optimization_id
                )
            
            if not result:
                raise ValueError(f"Optimization not found: {optimization_id}")
            
            return OptimizationResult(
                optimization_id=result['id'],
                video_id=result['video_id'],
                user_id=result['user_id'],
                status=VideoOptimizationStatus(result['status']),
                strategy=OptimizationStrategy(result['strategy']),
                original_title=result['original_title'],
                original_description=result['original_description'],
                original_tags=result['original_tags'],
                optimized_title=result.get('optimized_title'),
                optimized_description=result.get('optimized_description'),
                optimized_tags=result.get('optimized_tags'),
                baseline_views=result['baseline_views'],
                baseline_ctr=result['baseline_ctr'],
                baseline_revenue=result['baseline_revenue'],
                current_views=result.get('current_views'),
                current_ctr=result.get('current_ctr'),
                current_revenue=result.get('current_revenue'),
                views_increase_pct=result.get('views_increase_pct'),
                revenue_increase=result.get('revenue_increase'),
                confidence_score=result.get('confidence_score', 0.0),
                total_cost_usd=result.get('total_cost_usd', 0.0),
                applied_at=result.get('applied_at'),
                created_at=result['created_at'],
                updated_at=result['updated_at'],
                stage=OptimizationStage(result['stage']),
                error_message=result.get('error_message'),
                rollback_reason=result.get('rollback_reason')
            )
            
        except Exception as e:
            logger.error(
                f"Failed to get optimization result {optimization_id}: {e}",
                exc_info=True
            )
            raise
    
    async def check_and_rollback(
        self,
        optimization_id: int
    ) -> bool:
        """
        Check optimization performance and rollback if necessary.
        
        Called by monitoring job to check if optimization should be rolled back.
        
        Args:
            optimization_id: Optimization to check
            
        Returns:
            True if rolled back, False otherwise
        """
        try:
            # Get optimization details
            pool = await get_pool()
            async with pool.acquire() as conn:
                opt = await conn.fetchrow(
                    "SELECT * FROM video_optimizations WHERE id = $1",
                    optimization_id
                )
            
            if not opt or not opt.get('applied_at'):
                logger.warning(f"Optimization {optimization_id} not applied, skipping rollback check")
                return False
            
            # Get current analytics
            current_analytics = await get_video_analytics(
                opt['video_id'],
                opt['user_id']
            )
            
            current_views = current_analytics.get('views', 0)
            current_ctr = current_analytics.get('ctr', 0.0)
            current_revenue = current_analytics.get('revenue', 0.0)
            
            # Calculate performance change
            baseline_views = opt['baseline_views']
            baseline_ctr = opt['baseline_ctr']
            baseline_revenue = opt['baseline_revenue']
            
            # Check if performance dropped below threshold
            rollback_threshold = opt['rollback_threshold']
            
            views_ratio = current_views / baseline_views if baseline_views > 0 else 1.0
            ctr_ratio = current_ctr / baseline_ctr if baseline_ctr > 0 else 1.0
            revenue_ratio = current_revenue / baseline_revenue if baseline_revenue > 0 else 1.0
            
            should_rollback = (
                views_ratio < rollback_threshold or
                ctr_ratio < rollback_threshold or
                revenue_ratio < rollback_threshold
            )
            
            if should_rollback:
                logger.warning(
                    f"Performance drop detected for optimization {optimization_id}: "
                    f"views={views_ratio:.2%}, ctr={ctr_ratio:.2%}, revenue={revenue_ratio:.2%}"
                )
                
                # Perform rollback
                success = await self._perform_rollback(optimization_id, opt)
                
                if success:
                    metrics.increment("optimizer.rollback.executed")
                    return True
            else:
                logger.info(
                    f"Optimization {optimization_id} performing well: "
                    f"views={views_ratio:.2%}, ctr={ctr_ratio:.2%}, revenue={revenue_ratio:.2%}"
                )
            
            # Update current metrics
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE video_optimizations
                    SET current_views = $1, current_ctr = $2, current_revenue = $3,
                        views_increase_pct = $4, revenue_increase = $5,
                        updated_at = $6
                    WHERE id = $7
                    """,
                    current_views, current_ctr, current_revenue,
                    (views_ratio - 1.0) * 100,
                    current_revenue - baseline_revenue,
                    datetime.now(timezone.utc),
                    optimization_id
                )
            
            return False
            
        except Exception as e:
            logger.error(
                f"Failed to check rollback for optimization {optimization_id}: {e}",
                exc_info=True
            )
            return False
    
    async def _perform_rollback(
        self,
        optimization_id: int,
        optimization_data: Dict[str, Any]
    ) -> bool:
        """Rollback optimization to original values"""
        try:
            logger.info(f"Rolling back optimization {optimization_id}")
            
            # Restore original values
            success = await update_video_metadata(
                video_id=optimization_data['video_id'],
                user_id=optimization_data['user_id'],
                title=optimization_data['original_title'],
                description=optimization_data['original_description'],
                tags=optimization_data['original_tags']
            )
            
            if success:
                # Update optimization record
                pool = await get_pool()
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE video_optimizations
                        SET status = $1, stage = $2,
                            rollback_reason = $3, updated_at = $4
                        WHERE id = $5
                        """,
                        VideoOptimizationStatus.ROLLED_BACK.value,
                        OptimizationStage.ROLLBACK.value,
                        "Performance dropped below rollback threshold",
                        datetime.now(timezone.utc),
                        optimization_id
                    )
                
                logger.info(f"Successfully rolled back optimization {optimization_id}")
                return True
            else:
                logger.error(f"Failed to rollback optimization {optimization_id}")
                return False
                
        except Exception as e:
            logger.error(
                f"Failed to perform rollback for {optimization_id}: {e}",
                exc_info=True
            )
            return False
    
    async def batch_optimize(
        self,
        requests: List[VideoOptimizationRequest]
    ) -> List[OptimizationResult]:
        """
        Optimize multiple videos concurrently.
        
        Args:
            requests: List of optimization requests
            
        Returns:
            List of optimization results
        """
        tasks = [self.optimize_video(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Batch optimization failed for video {requests[i].video_id}: {result}"
                )
                # Create error result
                # This would need proper error handling
            else:
                processed_results.append(result)
        
        return processed_results


# ============================================================================
# SERVICE INSTANCE
# ============================================================================

video_optimizer = VideoOptimizerService()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def optimize_single_video(
    video_id: str,
    user_id: int,
    strategy: str = "balanced",
    auto_apply: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for single video optimization.
    
    Args:
        video_id: YouTube video ID
        user_id: User ID
        strategy: Optimization strategy
        auto_apply: Whether to apply automatically
        
    Returns:
        Optimization result dictionary
    """
    request = VideoOptimizationRequest(
        video_id=video_id,
        user_id=user_id,
        strategy=OptimizationStrategy(strategy),
        auto_apply=auto_apply
    )
    
    result = await video_optimizer.optimize_video(request)
    return result.dict()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'video_optimizer',
    'VideoOptimizerService',
    'VideoOptimizationRequest',
    'OptimizationResult',
    'OptimizationStrategy',
    'VideoOptimizationStatus',
    'OptimizationStage',
    'optimize_single_video'
]
