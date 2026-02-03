"""
Configuration and debug routes.
"""
from fastapi import APIRouter
import os
import logging

router = APIRouter(prefix="/api", tags=["config"])

@router.get("/debug/openai-models")
async def debug_openai_models():
    """Debug endpoint to see raw OpenAI API response"""
    try:
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "No API key found"}
        
        client = openai.OpenAI(api_key=api_key)
        models_response = client.models.list()
        
        all_models = []
        for model in models_response.data:
            all_models.append({
                "id": model.id,
                "owned_by": model.owned_by if hasattr(model, 'owned_by') else "unknown",
                "created": model.created if hasattr(model, 'created') else 0
            })
        
        # Sort by ID for readability
        all_models.sort(key=lambda x: x['id'])
        
        return {
            "total_models": len(all_models),
            "models": all_models
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/config")
async def get_config():
    """Get available configuration options"""
    return {
        "defaults": {
            "database": "postgresql://postgres:password@localhost:5432/salespoint",
            "catalog": "salespoint_production",
            "model": "openai:gpt-5",
            "temperature": 0.0,
            "request_limit": 10,
            "iteration_limit": 10,
            "provider": "openai"
        }
    }
