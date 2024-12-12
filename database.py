import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime

def get_db_connection():
    """Create database connection using environment variables"""
    return psycopg2.connect(
        host=os.environ.get('PGHOST'),
        database=os.environ.get('PGDATABASE'),
        user=os.environ.get('PGUSER'),
        password=os.environ.get('PGPASSWORD'),
        port=os.environ.get('PGPORT')
    )

def init_db():
    """Initialize database tables"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create tables if they don't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            dataset_name VARCHAR(100) NOT NULL,
            hyperparameters JSONB NOT NULL,
            metrics JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            size INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def save_training_run(model_name, dataset_name, hyperparameters, metrics):
    """Save training run to database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO training_runs (model_name, dataset_name, hyperparameters, metrics)
        VALUES (%s, %s, %s, %s)
        RETURNING run_id
    """, (model_name, dataset_name, json.dumps(hyperparameters), json.dumps(metrics)))
    
    run_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return run_id

def get_training_runs():
    """Fetch all training runs"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("SELECT * FROM training_runs ORDER BY created_at DESC")
    runs = cur.fetchall()
    
    cur.close()
    conn.close()
    return runs

def save_dataset(name, description, size):
    """Save dataset information"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO datasets (name, description, size)
        VALUES (%s, %s, %s)
        RETURNING dataset_id
    """, (name, description, size))
    
    dataset_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return dataset_id

def get_datasets():
    """Fetch all datasets"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("SELECT * FROM datasets ORDER BY created_at DESC")
    datasets = cur.fetchall()
    
    cur.close()
    conn.close()
    return datasets
