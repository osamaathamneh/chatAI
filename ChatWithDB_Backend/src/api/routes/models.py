"""
Model listing routes for different providers.
"""
from fastapi import APIRouter, HTTPException
import httpx
import logging
from src.api.config import OLLAMA_BASE_URL, OPENROUTER_BASE_URL, OPENROUTER_API_KEY

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("/ollama")
async def list_ollama_models():
    """List available Ollama models (excluding embedding models)"""
    try:
        # Try to fetch models from Ollama API
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch Ollama models")
            
            data = response.json()
            models = []
            
            # Parse Ollama API response
            for model in data.get("models", []):
                model_name = model.get("name", "")
                # Exclude embedding models
                if "embedding" not in model_name.lower():
                    models.append({
                        "value": f"openai:{model_name}",
                        "label": model_name,
                        "provider": "ollama"
                    })
            
            logging.info(f"Found {len(models)} Ollama models from {OLLAMA_BASE_URL}")
            return {"success": True, "models": models}
    except httpx.TimeoutException:
        raise HTTPException(status_code=500, detail="Ollama request timed out")
    except httpx.ConnectError:
        raise HTTPException(status_code=500, detail=f"Cannot connect to Ollama server at {OLLAMA_BASE_URL}")
    except Exception as e:
        logging.error(f"Error listing Ollama models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/openai")
async def list_openai_models():
    """List curated OpenAI models based on pydantic_ai KnownModelName"""
    try:
        # Curated list of 6 commonly used OpenAI models
        # Based on https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.KnownModelName
        # GPT-5 is set as default (first in list)
        
        models = [
            {
                "value": "openai:gpt-5",
                "label": "GPT-5",
                "provider": "openai",
                "id": "gpt-5",
                "description": "Latest and most capable model"
            },
            {
                "value": "openai:gpt-4o",
                "label": "GPT-4o",
                "provider": "openai",
                "id": "gpt-4o",
                "description": "Fast, multimodal flagship model"
            },
            {
                "value": "openai:gpt-4o-mini",
                "label": "GPT-4o Mini",
                "provider": "openai",
                "id": "gpt-4o-mini",
                "description": "Affordable and intelligent small model"
            },
            {
                "value": "openai:gpt-4-turbo",
                "label": "GPT-4 Turbo",
                "provider": "openai",
                "id": "gpt-4-turbo",
                "description": "Previous high-intelligence model"
            },
            {
                "value": "openai:gpt-4",
                "label": "GPT-4",
                "provider": "openai",
                "id": "gpt-4",
                "description": "Classic GPT-4 model"
            },
            {
                "value": "openai:gpt-3.5-turbo",
                "label": "GPT-3.5 Turbo",
                "provider": "openai",
                "id": "gpt-3.5-turbo",
                "description": "Fast and inexpensive model"
            }
        ]
        
        logging.info(f"Returning {len(models)} curated OpenAI models (default: {models[0]['id']})")
        return {"success": True, "models": models}
            
    except Exception as e:
        logging.error(f"Error in list_openai_models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/openrouter")
async def list_openrouter_models():
    """List available OpenRouter models"""
    try:
        # Curated list of OpenRouter models that support tool/function calling
        # gpt-oss models are listed FIRST as defaults (from original project)
        # Note: Model IDs verified at https://openrouter.ai/models
        models = [
            {
                "value": "openai:openai/gpt-oss-20b",
                "label": "GPT OSS 20B",
                "provider": "openrouter",
                "id": "openai/gpt-oss-20b",
                "description": "Open source GPT model - 20B parameters (Default)"
            },
            {
                "value": "openai:openai/gpt-oss-120b",
                "label": "GPT OSS 120B",
                "provider": "openrouter",
                "id": "openai/gpt-oss-120b",
                "description": "Open source GPT model - 120B parameters"
            },
            {
                "value": "openai:gpt-4o",
                "label": "GPT-4o (via OpenRouter)",
                "provider": "openrouter",
                "id": "gpt-4o",
                "description": "Fast, multimodal flagship model"
            },
            {
                "value": "openai:gpt-4o-mini",
                "label": "GPT-4o Mini (via OpenRouter)",
                "provider": "openrouter",
                "id": "gpt-4o-mini",
                "description": "Affordable and intelligent small model"
            },
            {
                "value": "openai:gpt-4-turbo",
                "label": "GPT-4 Turbo (via OpenRouter)",
                "provider": "openrouter",
                "id": "gpt-4-turbo",
                "description": "Previous high-intelligence model"
            },
            {
                "value": "anthropic:claude-3-opus",
                "label": "Claude 3 Opus (via OpenRouter)",
                "provider": "openrouter",
                "id": "claude-3-opus",
                "description": "Most capable Claude model"
            },
            {
                "value": "anthropic:claude-3-sonnet",
                "label": "Claude 3 Sonnet (via OpenRouter)",
                "provider": "openrouter",
                "id": "claude-3-sonnet",
                "description": "Balanced Claude model"
            }
        ]
        
        logging.info(f"Returning {len(models)} OpenRouter models")
        return {"success": True, "models": models}
            
    except Exception as e:
        logging.error(f"Error in list_openrouter_models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
