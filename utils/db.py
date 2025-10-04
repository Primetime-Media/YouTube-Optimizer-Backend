# utils/db.py
"""
Production-Ready Database Utilities
===================================
Provides connection pooling, transaction management, retry logic,
and comprehensive error handling for PostgreSQL database operations.
"""

import psycopg2
from psycopg2 import pool, extras, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT, ISOLATION_LEVEL_READ_COMMITTED
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple, Callable
import logging
import time
from functools import wraps
import threading

logger = logging.getLogger(__name__)

# Global connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()
_pool_config: Dict[str, Any] = {}


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class ConnectionPoolExhausted(DatabaseError):
    """Raised when connection pool is exhausted."""
    pass


class TransactionError(DatabaseError):
    """Raised when transaction operations fail."""
    pass


class QueryTimeoutError(DatabaseError):
    """Raised when query execution times out."""
    pass


def init_db_pool(
    host: str = "localhost",
    port: int = 5432,
    database: str = "youtube_optimizer",
    user: str = "postgres",
    password: str = "",
    min_connections: int = 2,
    max_connections: int = 10,
    pool_size: int = 10,
    max_overflow: int = 20,
    pool_timeout: int = 30,
    **kwargs
) -> None:
    """
    Initialize the database connection pool.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        min_connections: Minimum number of connections in pool
        max_connections: Maximum number of connections in pool
        pool_size: Alias for max_connections
        max_overflow: Additional connections beyond pool_size
        pool_timeout: Connection timeout in seconds
        **kwargs: Additional connection parameters
    """
    global _connection_pool, _pool_config
    
    with _pool_lock:
        if _connection_pool is not None:
            logger.warning("Connection pool already exists. Closing existing pool.")
            close_db_pool()
        
        try:
            # Calculate actual pool size
            actual_max = max_connections or (pool_size + max_overflow)
            actual_min = min(min_connections, actual_max)
            
            # Store configuration for reconnection
            _pool_config = {
                'host': host,
                'port': port,
                'database': database,
                'user': user,
                'password': password,
                'minconn': actual_min,
                'maxconn': actual_max,
                'connect_timeout': pool_timeout,
                'options': kwargs.get('options', '-c statement_timeout=30000'),  # 30s default timeout
            }
            
            # Add any additional kwargs
            for key, value in kwargs.items():
                if key not in _pool_config:
                    _pool_config[key] = value
            
            # Create connection pool
            _connection_pool = pool.ThreadedConnectionPool(**_pool_config)
            
            logger.info(
                f"Database connection pool initialized: "
                f"{actual_min}-{actual_max} connections to "
                f"{host}:{port}/{database}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}", exc_info=True)
            raise DatabaseError(f"Database pool initialization failed: {e}")


def close_db_pool() -> None:
    """Close all connections in the pool."""
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            try:
                _connection_pool.closeall()
                logger.info("Database connection pool closed")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
            finally:
                _connection_pool = None


def get_connection(timeout: int = 10):
    """
    Get a connection from the pool with timeout and retry logic.
    
    Args:
        timeout: Maximum time to wait for a connection
        
    Returns:
        psycopg2 connection object
        
    Raises:
        ConnectionPoolExhausted: If no connection available within timeout
        DatabaseError: If connection pool not initialized
    """
    global _connection_pool
    
    if _connection_pool is None:
        raise DatabaseError("Database connection pool not initialized. Call init_db_pool() first.")
    
    start_time = time.time()
    retry_count = 0
    max_retries = 3
    
    while time.time() - start_time < timeout:
        try:
            conn = _connection_pool.getconn()
            
            if conn is None:
                raise ConnectionPoolExhausted("No connection available in pool")
            
            # Verify connection is alive
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return conn
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                # Connection is dead, return to pool and retry
                logger.warning(f"Dead connection detected, reconnecting...")
                _connection_pool.putconn(conn, close=True)
                retry_count += 1
                
                if retry_count >= max_retries:
                    raise DatabaseError("Failed to get valid connection after retries")
                
                time.sleep(0.1 * retry_count)  # Exponential backoff
                continue
                
        except pool.PoolError as e:
            logger.warning(f"Pool error getting connection: {e}")
            time.sleep(0.1)
            continue
    
    raise ConnectionPoolExhausted(f"Could not get connection within {timeout} seconds")


def return_connection(conn, close: bool = False) -> None:
    """
    Return a connection to the pool.
    
    Args:
        conn: Connection to return
        close: If True, close the connection instead of returning to pool
    """
    global _connection_pool
    
    if _connection_pool is not None and conn is not None:
        try:
            _connection_pool.putconn(conn, close=close)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")


@contextmanager
def get_db_connection(auto_commit: bool = False, timeout: int = 10):
    """
    Context manager for database connections with automatic cleanup.
    
    Args:
        auto_commit: If True, set isolation level to autocommit
        timeout: Connection timeout in seconds
        
    Yields:
        Database connection
        
    Example:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users")
    """
    conn = None
    try:
        conn = get_connection(timeout=timeout)
        
        if auto_commit:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        else:
            conn.set_isolation_level(ISOLATION_LEVEL_READ_COMMITTED)
        
        yield conn
        
    except Exception as e:
        if conn is not None:
            try:
                conn.rollback()
            except:
                pass
        logger.error(f"Database operation failed: {e}", exc_info=True)
        raise
        
    finally:
        if conn is not None:
            return_connection(conn)


@contextmanager
def transaction(conn=None, auto_close: bool = True):
    """
    Context manager for database transactions with automatic rollback.
    
    Args:
        conn: Optional existing connection, creates new if None
        auto_close: If True, return connection to pool after transaction
        
    Yields:
        Database connection
        
    Example:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO users VALUES (%s)", (user_id,))
                cur.execute("UPDATE stats SET count = count + 1")
            # Automatically commits if no exception
    """
    new_conn = False
    if conn is None:
        conn = get_connection()
        new_conn = True
    
    try:
        yield conn
        conn.commit()
        logger.debug("Transaction committed successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction rolled back due to error: {e}", exc_info=True)
        raise TransactionError(f"Transaction failed: {e}")
        
    finally:
        if new_conn and auto_close:
            return_connection(conn)


def retry_on_db_error(max_retries: int = 3, backoff_factor: float = 0.5):
    """
    Decorator to retry database operations on transient errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
        
    Example:
        @retry_on_db_error(max_retries=3)
        def get_user(user_id):
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                    return cur.fetchone()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"Database error on attempt {attempt + 1}/{max_retries}, "
                            f"retrying in {wait_time}s: {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        
                except Exception as e:
                    # Don't retry on non-transient errors
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            raise DatabaseError(f"Operation failed after {max_retries} retries: {last_exception}")
        
        return wrapper
    return decorator


def test_db_connection() -> bool:
    """
    Test database connectivity.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result is not None and result[0] == 1
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def execute_query(
    query: str,
    params: Optional[Tuple] = None,
    fetch: str = "none",
    timeout: int = 30
) -> Any:
    """
    Execute a database query with proper error handling.
    
    Args:
        query: SQL query to execute
        params: Query parameters
        fetch: 'one', 'all', 'none' - how to fetch results
        timeout: Query timeout in seconds
        
    Returns:
        Query results based on fetch parameter
        
    Example:
        result = execute_query(
            "SELECT * FROM users WHERE id = %s",
            params=(user_id,),
            fetch='one'
        )
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            try:
                # Set statement timeout
                cur.execute(f"SET statement_timeout = {timeout * 1000}")
                
                # Execute query
                cur.execute(query, params)
                
                # Fetch results
                if fetch == 'one':
                    return cur.fetchone()
                elif fetch == 'all':
                    return cur.fetchall()
                else:
                    return None
                    
            except psycopg2.extensions.QueryCanceledError:
                raise QueryTimeoutError(f"Query exceeded timeout of {timeout}s")


def bulk_insert(
    table: str,
    columns: List[str],
    values: List[Tuple],
    batch_size: int = 1000
) -> int:
    """
    Perform bulk insert with batching.
    
    Args:
        table: Table name
        columns: Column names
        values: List of value tuples
        batch_size: Number of rows per batch
        
    Returns:
        Number of rows inserted
        
    Example:
        rows_inserted = bulk_insert(
            'users',
            ['id', 'name', 'email'],
            [(1, 'John', 'john@example.com'), (2, 'Jane', 'jane@example.com')]
        )
    """
    if not values:
        return 0
    
    total_inserted = 0
    
    with transaction() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                
                # Build insert query
                columns_str = ', '.join(columns)
                placeholders = ', '.join(['%s'] * len(columns))
                query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
                
                # Execute batch
                extras.execute_batch(cur, query, batch)
                total_inserted += len(batch)
                
                logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch)} rows")
    
    logger.info(f"Bulk insert completed: {total_inserted} rows inserted into {table}")
    return total_inserted


def get_table_info(table_name: str) -> Dict[str, Any]:
    """
    Get information about a table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        Dictionary with table information
    """
    query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
    """
    
    columns = execute_query(query, params=(table_name,), fetch='all')
    
    return {
        'table_name': table_name,
        'columns': [dict(col) for col in columns] if columns else []
    }


def get_pool_status() -> Dict[str, Any]:
    """
    Get current connection pool status.
    
    Returns:
        Dictionary with pool statistics
    """
    global _connection_pool
    
    if _connection_pool is None:
        return {'status': 'not_initialized'}
    
    try:
        # Try to get pool stats (these methods may not be available in all versions)
        return {
            'status': 'active',
            'minconn': _pool_config.get('minconn', 0),
            'maxconn': _pool_config.get('maxconn', 0),
            'host': _pool_config.get('host', 'unknown'),
            'database': _pool_config.get('database', 'unknown')
        }
    except Exception as e:
        logger.error(f"Error getting pool status: {e}")
        return {'status': 'error', 'message': str(e)}


# Health check function
def health_check() -> Dict[str, Any]:
    """
    Comprehensive database health check.
    
    Returns:
        Dictionary with health status
    """
    health = {
        'healthy': False,
        'pool_status': 'unknown',
        'connection_test': False,
        'response_time_ms': None,
        'error': None
    }
    
    try:
        # Check pool status
        pool_status = get_pool_status()
        health['pool_status'] = pool_status.get('status', 'unknown')
        
        # Test connection with timing
        start_time = time.time()
        connection_ok = test_db_connection()
        response_time = (time.time() - start_time) * 1000
        
        health['connection_test'] = connection_ok
        health['response_time_ms'] = round(response_time, 2)
        health['healthy'] = connection_ok and pool_status.get('status') == 'active'
        
    except Exception as e:
        health['error'] = str(e)
        logger.error(f"Health check failed: {e}")
    
    return health
