import os
from psycopg2_pool import ThreadSafeConnectionPool
from dotenv import load_dotenv
import logging
import threading
import atexit

load_dotenv()

logger = logging.getLogger(__name__)

# Export the context manager for easy importing
__all__ = ['get_connection', 'return_connection', 'DatabaseConnection', 'close_connection_pool']

# Global connection pool
_connection_pool = None
_pool_lock = threading.Lock()

def get_connection_pool():
    """Get or create the connection pool."""
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                try:
                    database_url = os.getenv('DATABASE_URL')
                    if not database_url:
                        raise ValueError("DATABASE_URL environment variable not set")
                    
                    logger.info(f"Creating connection pool for database at {database_url}")
                    
                    # Create connection pool with configuration
                    _connection_pool = ThreadSafeConnectionPool(
                        minconn=1,      # Minimum connections in pool
                        maxconn=100,    # High limit - handles unlimited concurrent users
                        idle_timeout=30,  # Close idle connections quickly (30 seconds)
                        dsn=database_url
                    )
                    
                    # Register cleanup function
                    atexit.register(close_connection_pool)
                    
                    logger.info("Connection pool created successfully")
                except Exception as e:
                    logger.error(f"Error creating connection pool: {e}")
                    raise
    
    return _connection_pool

def get_connection():
    """Get a connection from the pool."""
    try:
        pool = get_connection_pool()
        conn = pool.getconn()
        return conn
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}")
        raise

def return_connection(conn):
    """Return a connection to the pool."""
    try:
        if conn and not conn.closed:
            pool = get_connection_pool()
            pool.putconn(conn)
        else:
            logger.warning("Attempted to return closed or invalid connection")
    except Exception as e:
        logger.error(f"Error returning connection to pool: {e}")

def close_connection_pool():
    """Close the connection pool and all connections."""
    global _connection_pool
    
    if _connection_pool:
        try:
            _connection_pool.clear()
            logger.info("Connection pool closed successfully")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")
        finally:
            _connection_pool = None

class DatabaseConnection:
    """Context manager for database connections."""
    
    def __init__(self):
        self.conn = None
    
    def __enter__(self):
        self.conn = get_connection()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            return_connection(self.conn)
            self.conn = None