"""
Provider configuration utilities.
Handles environment variable setup for different LLM providers.
"""
import os
import logging
from typing import Optional, Tuple
from src.api.config import (
    OLLAMA_BASE_URL,
    OPENROUTER_BASE_URL,
    OPENROUTER_API_KEY
)

def configure_provider(provider: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Configure environment variables for the specified provider.
    
    Args:
        provider: Provider name ("openai", "ollama", or "openrouter")
    
    Returns:
        Tuple of (original_base_url, original_api_key) to restore later
    
    Raises:
        ValueError: If provider configuration is invalid
        HTTPException: If required API keys are missing
    """
    original_base_url = os.environ.get("OPENAI_BASE_URL")
    original_api_key = os.environ.get("OPENAI_API_KEY")
    
    if provider == "ollama":
        # Set Ollama base URL for this request
        ollama_base_url = f"{OLLAMA_BASE_URL}/v1"
        os.environ["OPENAI_BASE_URL"] = ollama_base_url
        logging.info(f"Using Ollama provider with base URL: {ollama_base_url}")
    elif provider == "openrouter":
        # Set OpenRouter base URL and API key
        os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
        if OPENROUTER_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
            logging.info(f"Using OpenRouter provider with base URL: {OPENROUTER_BASE_URL}")
        else:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please add it to your .env file.")
    elif provider == "openai":
        # For OpenAI, unset the base URL (or keep default)
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]
        logging.info("Using OpenAI provider with default base URL")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return original_base_url, original_api_key

def restore_provider(original_base_url: Optional[str], original_api_key: Optional[str], provider: str):
    """
    Restore original environment variables after provider usage.
    
    Args:
        original_base_url: Original OPENAI_BASE_URL value
        original_api_key: Original OPENAI_API_KEY value
        provider: Provider name used (for cleanup logic)
    """
    if original_base_url is not None:
        os.environ["OPENAI_BASE_URL"] = original_base_url
    elif "OPENAI_BASE_URL" in os.environ:
        del os.environ["OPENAI_BASE_URL"]
    
    if original_api_key is not None:
        os.environ["OPENAI_API_KEY"] = original_api_key
    elif "OPENAI_API_KEY" in os.environ and provider == "ollama":
        # Only delete if we set a placeholder
        if os.environ.get("OPENAI_API_KEY") == "ollama":
            del os.environ["OPENAI_API_KEY"]

def get_provider_base_url(provider: str) -> Optional[str]:
    """
    Get the base URL for a provider without modifying environment.
    
    Args:
        provider: Provider name
    
    Returns:
        Base URL string or None
    """
    if provider == "ollama":
        return OLLAMA_BASE_URL
    elif provider == "openrouter":
        return OPENROUTER_BASE_URL
    elif provider == "openai":
        return None
    else:
        return None
