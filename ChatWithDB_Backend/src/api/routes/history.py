"""
Query history routes.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import sqlite3
import logging
from src.api.config import DB_PATH
from src.api.dependencies import get_database_url

router = APIRouter(prefix="/api", tags=["history"])

@router.get("/history")
async def get_history(database: Optional[str] = None, limit: int = 50):
    """Get query history, optionally filtered by database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if database:
            # Convert identifier to actual URL for comparison
            db_url = get_database_url(database)
            cursor.execute("""
                SELECT * FROM query_history 
                WHERE database_url = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (db_url, limit))
        else:
            cursor.execute("""
                SELECT * FROM query_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        history = [dict(row) for row in rows]
        conn.close()
        
        return {"success": True, "history": history}
    except Exception as e:
        logging.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/examples")
async def get_examples(database: Optional[str] = None):
    """Get example queries based on successful history for a database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if database:
            # Convert identifier to actual URL for comparison
            db_url = get_database_url(database)
            # Get successful queries for this database
            cursor.execute("""
                SELECT DISTINCT query, generated_sql, generation_time 
                FROM query_history 
                WHERE database_url = ? AND success = 1
                ORDER BY timestamp DESC 
                LIMIT 3
            """, (db_url,))
        else:
            # Get successful queries from all databases
            cursor.execute("""
                SELECT DISTINCT query, generated_sql, generation_time, database_url 
                FROM query_history 
                WHERE success = 1
                ORDER BY timestamp DESC 
                LIMIT 3
            """)
        
        rows = cursor.fetchall()
        examples = [dict(row) for row in rows]
        conn.close()
        
        # If no history, return default examples based on database
        if not examples:
            if database and "salespoint" in database.lower():
                examples = [
                    {"query": "Get all prepaid activations from 2025-07-31 to 2025-08-31"},
                    {"query": "List the total number of new cards sold and the total number of cards revalued for each location in 2023"},
                    {"query": "What is the average number of new cards sold on days where the visitor count was greater than 500 in 2024?"}
                ]
            else:
                examples = [
                    {"query": "Which games generate the most revenue?"},
                    {"query": "Show top 10 customers by total purchases"},
                    {"query": "Get monthly revenue for the last 6 months"}
                ]
        
        return {"success": True, "examples": examples}
    except Exception as e:
        logging.error(f"Error fetching examples: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/{history_id}")
async def delete_history_item(history_id: int):
    """Delete a specific history item"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM query_history WHERE id = ?", (history_id,))
        conn.commit()
        conn.close()
        return {"success": True, "message": "History item deleted"}
    except Exception as e:
        logging.error(f"Error deleting history item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history")
async def clear_history():
    """Clear all query history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM query_history")
        conn.commit()
        conn.close()
        return {"success": True, "message": "All history cleared"}
    except Exception as e:
        logging.error(f"Error clearing history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
