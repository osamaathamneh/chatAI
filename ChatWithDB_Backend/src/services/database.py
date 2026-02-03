"""
Database initialization and helper functions.
"""
import sqlite3
import logging
from pathlib import Path
from src.api.config import DB_PATH

def init_db():
    """Initialize SQLite database for query history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create table only if it doesn't exist (preserves existing data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                database_url TEXT NOT NULL,
                catalog_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                query TEXT NOT NULL,
                generated_sql TEXT,
                generation_time REAL,
                success BOOLEAN,
                error_message TEXT,
                request_limit INTEGER,
                iteration_limit INTEGER,
                temperature REAL,
                actual_iterations INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                base_url TEXT
            )
        """)
        
        # Migrate existing tables - add new columns if they don't exist
        try:
            cursor.execute("ALTER TABLE query_history ADD COLUMN actual_iterations INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE query_history ADD COLUMN input_tokens INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE query_history ADD COLUMN output_tokens INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE query_history ADD COLUMN total_tokens INTEGER")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE query_history ADD COLUMN base_url TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        conn.commit()
        conn.close()
        logging.info(f"✅ Database initialized successfully at {DB_PATH}")
    except Exception as e:
        logging.error(f"❌ Failed to initialize database: {str(e)}")
        raise
