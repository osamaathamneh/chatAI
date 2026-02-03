"""
Semantic search routes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import psycopg
import pgai.semantic_catalog as sc
import logging
from src.api.config import CATALOG_NAME
from src.api.dependencies import get_database_url

router = APIRouter(prefix="/api", tags=["search"])

class SearchRequest(BaseModel):
    query: str
    render: bool = False
    limit: int = 10
    catalog_db: Optional[str] = None
    catalog_name: Optional[str] = None

@router.post("/search")
async def search_catalog(request: SearchRequest):
    """Search the semantic catalog with natural language"""
    try:
        # Use environment variables for security (ignore URLs from frontend)
        catalog_db = get_database_url(request.catalog_db)
        catalog_name = request.catalog_name or CATALOG_NAME
        
        async with await psycopg.AsyncConnection.connect(catalog_db) as con:
            catalog = await sc.from_name(con, catalog_name)
            
            # Get the first available embedding config
            embeddings = await catalog.list_embeddings(con)
            if not embeddings or len(embeddings) == 0:
                raise HTTPException(status_code=500, detail="No embedding configurations found")
            
            embed_config = embeddings[0][0]
            
            # Perform semantic search on objects, SQL examples, and facts
            obj_matches = await catalog.search_objects(
                con, embedding_name=embed_config, query=request.query, limit=request.limit
            )
            sql_matches = await catalog.search_sql_examples(
                con, embedding_name=embed_config, query=request.query, limit=request.limit
            )
            fact_matches = await catalog.search_facts(
                con, embedding_name=embed_config, query=request.query, limit=request.limit
            )
            
            # Format results
            formatted_results = []
            
            # Add object matches
            for m in obj_matches:
                formatted_results.append({
                    "id": m.id,
                    "item": ".".join(m.objnames),
                    "description": m.description or "No description",
                    "type": "object"
                })
            
            # Add SQL example matches
            for m in sql_matches:
                formatted_results.append({
                    "id": m.id,
                    "item": m.sql[:100] + "..." if len(m.sql) > 100 else m.sql,
                    "description": m.description or "No description",
                    "type": "sql_example"
                })
            
            # Add fact matches
            for m in fact_matches:
                formatted_results.append({
                    "id": m.id,
                    "item": "Fact",
                    "description": m.description or "No description",
                    "type": "fact"
                })
            
            return {
                "success": True,
                "query": request.query,
                "results": formatted_results
            }
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
