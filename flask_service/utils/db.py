import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

def get_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        print(f"Connecting to database at {database_url}")
        conn = psycopg2.connect(database_url)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise