# health_routes.py
"""
Health Check Routes - FIXED VERSION
====================================
✅ FIXED: Removed redundant get_db_connection() function
✅ All functionality preserved
"""

from fastapi import APIRouter, HTTPException
from utils.db import get_connection
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

@router.get("/")
async def root():
    """Root endpoint - basic API status"""
    return {"message": "YouTube Optimizer API is running", "status": "healthy"}

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Quick database connectivity check
        conn = get_connection()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")
