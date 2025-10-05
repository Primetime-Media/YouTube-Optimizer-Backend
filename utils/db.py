# utils/db.py
"""
Production-Ready Database Utilities
Provides connection pooling, transaction management, and error handling
"""

import psycopg2
from psycopg2 import pool, extras, sql
from psycopg2.extensions import ISOLATION_LEVEL_READ_COMMITTED
from typing import Optional, Dict, List, Any, Callable
from contextlib import contextmanager
import logging
from functools import wraps
import time

from config import settings

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

# Connection pool configuration
MIN_CONNECTIONS = 2
MAX_CONNECTIONS = 20
CONNECTION_TIMEOUT = 30  # seconds
QUERY_TIMEOUT = 60  # seconds

# Global connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


# ============================================================================
# CONNECTION POOL MANAGEMENT
# ============================================================================

def initialize_connection_pool():
    """
    Initialize the database connection pool
    
    Should be called once at application startup
    """
    global _connection_pool
    
    if _connection_pool is not None:
        logger.warning("Connection pool already initialized")
        return
    
    try:
        _connection_pool = pool.ThreadedConnectionPool(
            MIN_CONNECTIONS,
            MAX_CONNECTIONS,
            host=settings.database_host,
            port=settings.database_port,
            database=settings.database_name,
            user=settings.database_user,
            password=settings.database_password.get_secret_value(),
            connect_timeout=CONNECTION_TIMEOUT,
            options=f'-c statement_timeout={QUERY_TIMEOUT * 1000}'  # milliseconds
        )
        
        logger.info(
            f"Database connection pool initialized: "
            f"{MIN_CONNECTIONS}-{MAX_CONNECTIONS} connections"
        )
        
        # Test connection
        test_conn = _connection_pool.getconn()
        try:
            with test_conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            logger.info("Database connection test successful")
        finally:
            _connection_pool.putconn(test_conn)
            
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}", exc_info=True)
        raise


def close_connection_pool():
    """
    Close all connections in the pool
    
    Should be called at application shutdown
    """
    global _connection_pool
    
    if _connection_pool is None:
        logger.warning("Connection pool not initialized")
        return
    
    try:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Database connection pool closed")
    except Exception as e:
        logger.error(f"Error closing connection pool: {e}", exc_info=True)


def get_connection():
    """
    Get a connection from the pool
    
    Returns:
        Database connection
        
    Raises:
        RuntimeError: If pool not initialized
        psycopg2.Error: On connection failure
    """
    global _connection_pool
    
    if _connection_pool is None:
        raise RuntimeError(
            "Connection pool not initialized. "
            "Call initialize_connection_pool() first."
        )
    
    try:
        conn = _connection_pool.getconn()
        conn.set_isolation_level(ISOLATION_LEVEL_READ_COMMITTED)
        return conn
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}", exc_info=True)
        raise


def return_connection(conn):
    """
    Return a connection to the pool
    
    Args:
        conn: Connection to return
    """
    global _connection_pool
    
    if _connection_pool is None:
        logger.warning("Cannot return connection - pool not initialized")
        return
    
    try:
        _connection_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Error returning connection to pool: {e}", exc_info=True)


@contextmanager
def get_db_connection():
    """
    Context manager for database connections
    
    Automatically returns connection to pool when done
    
    Usage:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users")
    """
    conn = None
    try:
        conn = get_connection()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error in context manager: {e}", exc_info=True)
        raise
    finally:
        if conn:
            return_connection(conn)


@contextmanager
def get_db_cursor(commit: bool = False):
    """
    Context manager for database cursor
    
    Args:
        commit: Whether to commit transaction on success
        
    Usage:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute("INSERT INTO users ...")
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
        yield cursor
        
        if commit:
            conn.commit()
            
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error in cursor context manager: {e}", exc_info=True)
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            return_connection(conn)


# ============================================================================
# TRANSACTION MANAGEMENT
# ============================================================================

@contextmanager
def transaction():
    """
    Context manager for database transactions
    
    Automatically commits on success, rollsback on error
    
    Usage:
        with transaction() as conn:
            with conn.cursor() as cursor:
                cursor.execute("INSERT ...")
                cursor.execute("UPDATE ...")
        # Auto-commits here if no exceptions
    """
    conn = None
    try:
        conn = get_connection()
        yield conn
        conn.commit()
        logger.debug("Transaction committed successfully")
    except Exception as e:
        if conn:
            conn.rollback()
            logger.warning(f"Transaction rolled back due to error: {e}")
        raise
    finally:
        if conn:
            return_connection(conn)


def retry_on_deadlock(max_retries: int = 3, delay: float = 0.1):
    """
    Decorator to retry database operations on deadlock
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries (seconds)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except psycopg2.extensions.TransactionRollbackError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Deadlock detected in {func.__name__}, "
                            f"retry {attempt + 1}/{max_retries}"
                        )
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(
                            f"Max retries exceeded for {func.__name__} "
                            f"after {max_retries} attempts"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# QUERY HELPERS
# ============================================================================

def execute_query(
    query: str,
    params: tuple = None,
    fetch: str = None
) -> Optional[Any]:
    """
    Execute a database query with automatic connection management
    
    Args:
        query: SQL query string
        params: Query parameters tuple
        fetch: Fetch mode ('one', 'all', or None for no fetch)
        
    Returns:
        Query results based on fetch mode, or None
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
            cursor.execute(query, params or ())
            
            if fetch == 'one':
                return cursor.fetchone()
            elif fetch == 'all':
                return cursor.fetchall()
            elif fetch is None:
                conn.commit()
                return None
            else:
                raise ValueError(f"Invalid fetch mode: {fetch}")
                
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            return_connection(conn)


def bulk_insert(
    table: str,
    columns: List[str],
    values: List[tuple],
    on_conflict: Optional[str] = None
) -> int:
    """
    Perform bulk insert operation
    
    Args:
        table: Table name
        columns: List of column names
        values: List of value tuples
        on_conflict: Optional ON CONFLICT clause
        
    Returns:
        Number of rows inserted
    """
    if not values:
        return 0
    
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # Build query
            cols_sql = sql.SQL(', ').join(map(sql.Identifier, columns))
            placeholders = sql.SQL(', ').join(
                sql.SQL('({})').format(
                    sql.SQL(', ').join(sql.Placeholder() for _ in columns)
                )
                for _ in range(len(values))
            )
            
            query = sql.SQL(
                "INSERT INTO {table} ({columns}) VALUES {placeholders}"
            ).format(
                table=sql.Identifier(table),
                columns=cols_sql,
                placeholders=placeholders
            )
            
            if on_conflict:
                query = sql.SQL("{query} {conflict}").format(
                    query=query,
                    conflict=sql.SQL(on_conflict)
                )
            
            # Flatten values for execute
            flat_values = [item for sublist in values for item in sublist]
            
            cursor.execute(query, flat_values)
            rows_inserted = cursor.rowcount
            conn.commit()
            
            logger.debug(f"Bulk inserted {rows_inserted} rows into {table}")
            return rows_inserted
            
    except Exception as e:
        logger.error(f"Error in bulk insert: {e}", exc_info=True)
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            return_connection(conn)


def upsert(
    table: str,
    data: Dict[str, Any],
    conflict_columns: List[str],
    update_columns: Optional[List[str]] = None
) -> bool:
    """
    Insert or update record (UPSERT)
    
    Args:
        table: Table name
        data: Dict of column: value pairs
        conflict_columns: Columns to check for conflicts
        update_columns: Columns to update on conflict (defaults to all)
        
    Returns:
        True if successful
    """
    if update_columns is None:
        update_columns = [k for k in data.keys() if k not in conflict_columns]
    
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # Build column and value lists
            columns = list(data.keys())
            values = [data[col] for col in columns]
            
            # Build SQL
            cols_sql = sql.SQL(', ').join(map(sql.Identifier, columns))
            vals_sql = sql.SQL(', ').join(sql.Placeholder() for _ in columns)
            
            conflict_sql = sql.SQL(', ').join(
                map(sql.Identifier, conflict_columns)
            )
            
            update_sql = sql.SQL(', ').join(
                sql.SQL("{col} = EXCLUDED.{col}").format(
                    col=sql.Identifier(col)
                )
                for col in update_columns
            )
            
            query = sql.SQL(
                "INSERT INTO {table} ({columns}) "
                "VALUES ({values}) "
                "ON CONFLICT ({conflict}) "
                "DO UPDATE SET {updates}"
            ).format(
                table=sql.Identifier(table),
                columns=cols_sql,
                values=vals_sql,
                conflict=conflict_sql,
                updates=update_sql
            )
            
            cursor.execute(query, values)
            conn.commit()
            
            logger.debug(f"Upserted record in {table}")
            return True
            
    except Exception as e:
        logger.error(f"Error in upsert: {e}", exc_info=True)
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            return_connection(conn)


# ============================================================================
# HEALTH CHECK
# ============================================================================

def check_database_health() -> Dict[str, Any]:
    """
    Check database health and connection pool status
    
    Returns:
        Dict with health status information
    """
    global _connection_pool
    
    health = {
        'healthy': False,
        'pool_initialized': _connection_pool is not None,
        'can_connect': False,
        'response_time_ms': None,
        'error': None
    }
    
    if _connection_pool is None:
        health['error'] = "Connection pool not initialized"
        return health
    
    conn = None
    try:
        start_time = time.time()
        conn = get_connection()
        
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        
        response_time = (time.time() - start_time) * 1000  # milliseconds
        
        health.update({
            'healthy': True,
            'can_connect': True,
            'response_time_ms': round(response_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health['error'] = str(e)
    finally:
        if conn:
            return_connection(conn)
    
    return health


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_database():
    """Initialize database connection pool (call at startup)"""
    try:
        initialize_connection_pool()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise


def cleanup_database():
    """Cleanup database resources (call at shutdown)"""
    try:
        close_connection_pool()
        logger.info("Database cleanup completed")
    except Exception as e:
        logger.error(f"Error during database cleanup: {e}", exc_info=True)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'get_connection',
    'return_connection',
    'get_db_connection',
    'get_db_cursor',
    'transaction',
    'retry_on_deadlock',
    'execute_query',
    'bulk_insert',
    'upsert',
    'check_database_health',
    'init_database',
    'cleanup_database',
    'initialize_connection_pool',
    'close_connection_pool',
]
