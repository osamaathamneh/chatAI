"""
Catalog and database management routes.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import psycopg
import pgai.semantic_catalog as sc
import logging
from src.api.config import CATALOG_NAME, CATALOG_DB, TARGET_DB
from src.api.dependencies import get_database_url

router = APIRouter(prefix="/api", tags=["catalog"])

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "catalog_name": CATALOG_NAME}

@router.get("/databases")
async def list_databases():
    """List available databases"""
    try:
        # Return database identifiers (not full connection strings for security)
        # The backend will use environment variables for actual connections
        databases = [
            {"value": "salespoint", "label": "SalesPoint DB", "url": TARGET_DB},
            {"value": "report", "label": "Report DB", "url": CATALOG_DB}
        ]
        return {"success": True, "databases": databases}
    except Exception as e:
        logging.error(f"Error listing databases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/catalogs")
async def list_catalogs(database: Optional[str] = None):
    """List semantic catalogs for a specific database"""
    try:
        # Use environment variables for security
        db_url = get_database_url(database)
        async with await psycopg.AsyncConnection.connect(db_url) as con:
            catalogs = await sc.list_semantic_catalogs(con)
            catalog_list = [
                {"value": cat.name, "label": cat.name, "id": cat.id, "database": db_url}
                for cat in catalogs
            ]
            return {"success": True, "catalogs": catalog_list}
    except Exception as e:
        logging.error(f"Error listing catalogs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
