import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Connection Pool
connection_pool = pool.SimpleConnectionPool(
    1, 10,  # Min and max connections
    host=os.environ['PGHOST'],
    database=os.environ['PGDATABASE'],
    user=os.environ['PGUSER'],
    password=os.environ['PGPASSWORD'],
    port=os.environ['PGPORT']
)

def get_db_connection():
    """Get a connection from the pool."""
    try:
        return connection_pool.getconn()
    except psycopg2.Error as e:
        logging.error(f"Error getting connection from pool: {e}")
        raise

def release_db_connection(conn):
    """Release a connection back to the pool."""
    try:
        connection_pool.putconn(conn)
    except psycopg2.Error as e:
        logging.error(f"Error releasing connection back to pool: {e}")
        raise

def safe_execute(query: str, params=None, fetch: bool = False, commit: bool = False):
    """Safely execute a query with connection management."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor if fetch else None) as cur:
            cur.execute(query, params)
            if commit:
                conn.commit()
            if fetch:
                return cur.fetchall()
    except psycopg2.Error as e:
        logging.error(f"Database error executing query: {e}")
        conn.rollback()
        raise
    except Exception as e:
        logging.error(f"Unexpected error executing query: {e}")
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)

def init_db():
    """Initialize database tables."""
    logging.info("Initializing database tables...")
    queries = [
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            dataset_name VARCHAR(100) NOT NULL,
            hyperparameters JSONB NOT NULL,
            metrics JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            size INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    for query in queries:
        safe_execute(query, commit=True)

def save_training_run(model_name: str, dataset_name: str, hyperparameters: dict, metrics: dict) -> int:
    """Save training run to database."""
    query = """
        INSERT INTO training_runs (model_name, dataset_name, hyperparameters, metrics)
        VALUES (%s, %s, %s, %s)
        RETURNING run_id
    """
    params = (model_name, dataset_name, json.dumps(hyperparameters), json.dumps(metrics))
    result = safe_execute(query, params=params, fetch=True, commit=True)
    return result[0]['run_id']

def get_training_runs() -> list:
    """Fetch all training runs."""
    query = "SELECT * FROM training_runs ORDER BY created_at DESC"
    return safe_execute(query, fetch=True)

def save_dataset(name: str, description: str, size: int) -> int:
    """Save dataset information."""
    query = """
        INSERT INTO datasets (name, description, size)
        VALUES (%s, %s, %s)
        RETURNING dataset_id
    """
    params = (name, description, size)
    result = safe_execute(query, params=params, fetch=True, commit=True)
    return result[0]['dataset_id']

def get_datasets() -> list:
    """Fetch all datasets."""
    query = "SELECT * FROM datasets ORDER BY created_at DESC"
    return safe_execute(query, fetch=True)