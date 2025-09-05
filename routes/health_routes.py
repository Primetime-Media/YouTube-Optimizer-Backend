# health_routes.py
from fastapi import APIRouter, HTTPException
from utils.db import get_connection
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

def get_db_connection():
    """Database connection function"""
    try:
        return get_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

@router.get("/")
async def root():
    """Root endpoint - basic API status"""
    return {"message": "YouTube Optimizer API is running", "status": "healthy"}

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Quick database connectivity check
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database connection failed")
