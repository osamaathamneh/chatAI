"""
Template routes.
"""
from fastapi import APIRouter, HTTPException
import csv
import os
import logging
from src.api.config import TEMPLATE_CSV_PATH

router = APIRouter(prefix="/api", tags=["templates"])

@router.get("/templates")
async def get_templates():
    """Get all available query templates"""
    try:
        # Read directly from CSV file to get user_query and true_sql
        templates = []
        csv_path = TEMPLATE_CSV_PATH
        
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_query = row.get('user_query', '').strip()
                    true_sql = row.get('true_sql', '').strip()
                    if user_query and true_sql:
                        templates.append({
                            'user_query': user_query,
                            'true_sql': true_sql
                        })
        
        return {
            "success": True,
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        logging.error(f"Error fetching templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
