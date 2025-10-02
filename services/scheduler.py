import logging
import asyncio
from datetime import datetime, timedelta
from utils.db import get_connection
from typing import Optional, Dict, Any

# Import with error handling
try:
    from services.channel import (
        get_channel_data,
        create_optimization,
        generate_channel_optimization,
        get_optimization_status,
        apply_channel_optimization
    )
    from services.optimizer import apply_optimization_to_youtube_channel
    CHANNEL_SERVICES_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import channel services: {e}")
    CHANNEL_SERVICES_AVAILABLE = False

logger = logging.getLogger(__name__)


def initialize_scheduler():
    """
    Initialize the scheduler system.
    For Cloud Run compatibility, we don't use APScheduler.
    """
    logger.info("Scheduler system initialized for Cloud Run")
    if not CHANNEL_SERVICES_AVAILABLE:
        logger.warning("Channel optimization services not available")


def setup_monthly_optimization_for_channel(channel_id: int, auto_apply: bool = True) -> Dict[str, Any]:
    """
    Create a monthly optimization schedule for a channel.
    
    Args:
        channel_id: The database ID of the channel
        auto_
