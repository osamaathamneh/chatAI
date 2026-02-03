#!/usr/bin/env python3
"""
Build FAISS vector store from data/templates/templates_with_prompt_fewshot.csv

This script:
1. Loads templates from data/templates/templates_with_prompt_fewshot.csv
2. Generates embeddings for all user_query entries using Ollama
3. Creates FAISS index for similarity search
4. Saves index and metadata to disk
"""
import os
import sys
import csv
import json
import numpy as np
import logging

# Suppress FAISS loader INFO messages (AVX512/AVX2 warnings are just informational)
logging.getLogger('faiss.loader').setLevel(logging.WARNING)

import faiss
import httpx
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Fix Windows event loop policy
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()

# Configuration
CSV_FILE = "data/templates/templates_with_prompt_fewshot.csv"
INDEX_FILE = "data/templates/template_vectors.faiss"
METADATA_FILE = "data/templates/template_vectors_metadata.json"

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "hf.co/bartowski/granite-embedding-30m-english-GGUF:Q4_K_M"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def find_embedding_model(base_url: str, preferred_model: str) -> str:
    """
    Find the embedding model from available Ollama models.
    
    Args:
        base_url: Ollama base URL (without /v1)
        preferred_model: Preferred model name to use
        
    Returns:
        Model name that exists in Ollama, or preferred_model if not found
    """
    try:
        # Remove /v1 if present to get base URL
        api_base = base_url.rstrip('/').replace('/v1', '')
        tags_url = f"{api_base}/api/tags"
        
        logger.info(f"Checking available Ollama models at {tags_url}")
        
        # Use httpx which is already in requirements
        with httpx.Client(timeout=10.0, verify=False) as client:
            response = client.get(tags_url)
            
            if response.status_code == 200:
                data = response.json()
                available_models = [model.get("name", "") for model in data.get("models", [])]
                
                logger.info(f"Found {len(available_models)} available models")
                if available_models:
                    logger.info(f"Available models: {', '.join(available_models[:10])}")
                
                # First, try exact match with preferred model
                if preferred_model in available_models:
                    logger.info(f"Using exact match: {preferred_model}")
                    return preferred_model
                
                # Try to find embedding models containing "granite" or "embedding"
                embedding_models = [
                    model for model in available_models 
                    if "granite" in model.lower() or "embedding" in model.lower()
                ]
                
                if embedding_models:
                    # Prefer models matching the preferred name pattern
                    matching = [m for m in embedding_models if "granite" in m.lower() and "embedding" in m.lower()]
                    if matching:
                        selected = matching[0]
                        logger.info(f"Found embedding model: {selected}")
                        return selected
                    else:
                        selected = embedding_models[0]
                        logger.info(f"Using first available embedding model: {selected}")
                        return selected
                
                # If no embedding models found, check if preferred model name matches any available model
                # (sometimes Ollama stores models with shortened names)
                preferred_parts = preferred_model.lower().replace(':', '-').split('/')
                for model in available_models:
                    model_lower = model.lower()
                    # Check if any part of the preferred model name matches
                    if any(part in model_lower for part in preferred_parts if len(part) > 5):
                        logger.info(f"Found potential match: {model} (preferred: {preferred_model})")
                        return model
                
                # If no embedding models found, log available models
                logger.warning(f"No embedding models found. Available models: {', '.join(available_models[:10])}")
                logger.warning(f"Will try to use preferred model: {preferred_model}")
                logger.warning(f"If this fails, you may need to pull the model: ollama pull {preferred_model}")
                return preferred_model
            else:
                logger.warning(f"Failed to fetch Ollama models (status {response.status_code}), using preferred model")
                return preferred_model
    except httpx.TimeoutException:
        logger.warning(f"Timeout checking Ollama models, using preferred model: {preferred_model}")
        return preferred_model
    except httpx.ConnectError as e:
        logger.warning(f"Cannot connect to Ollama at {tags_url}: {e}, using preferred model: {preferred_model}")
        return preferred_model
    except Exception as e:
        logger.warning(f"Error checking Ollama models: {e}, using preferred model: {preferred_model}")
        return preferred_model


def pull_ollama_model(base_url: str, model_name: str) -> bool:
    """
    Attempt to pull an Ollama model via API.
    
    Args:
        base_url: Ollama base URL (without /v1)
        model_name: Name of the model to pull
        
    Returns:
        True if pull was initiated successfully, False otherwise
    """
    try:
        api_base = base_url.rstrip('/').replace('/v1', '')
        pull_url = f"{api_base}/api/pull"
        
        logger.info(f"Attempting to pull model '{model_name}' from Ollama...")
        
        with httpx.Client(timeout=300.0, verify=False) as client:
            response = client.post(
                pull_url,
                json={"name": model_name},
                timeout=300.0
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully initiated pull for model '{model_name}'")
                logger.info("   This may take several minutes. Please wait...")
                # Note: The pull is asynchronous, so we return True but the model may not be immediately available
                return True
            else:
                logger.warning(f"Failed to pull model (status {response.status_code}): {response.text}")
                return False
    except Exception as e:
        logger.warning(f"Error pulling model: {e}")
        return False


def test_embedding_model(client: OpenAI, model_name: str) -> bool:
    """
    Test if the embedding model is available by making a test call.
    
    Args:
        client: OpenAI-compatible client configured for Ollama
        model_name: Name of the embedding model to test
        
    Returns:
        True if model is available, False otherwise
    """
    try:
        logger.info(f"Testing embedding model '{model_name}'...")
        response = client.embeddings.create(
            model=model_name,
            input="test"
        )
        logger.info(f"✅ Model '{model_name}' is available (dimension: {len(response.data[0].embedding)})")
        return True
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "404" in error_msg:
            logger.error(f"❌ Model '{model_name}' is not available on this Ollama instance.")
            logger.error(f"   Please pull the model first: ollama pull {model_name}")
        else:
            logger.error(f"❌ Error testing model '{model_name}': {error_msg}")
        return False


def get_embedding(text: str, client: OpenAI, model_name: str) -> List[float]:
    """
    Generate embedding for text using Ollama API
    
    Args:
        text: Text to embed
        client: OpenAI-compatible client configured for Ollama
        model_name: Name of the embedding model to use
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        # Ollama uses OpenAI-compatible API
        response = client.embeddings.create(
            model=model_name,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error generating embedding with model '{model_name}': {error_msg}")
        # If model not found, provide helpful error message
        if "not found" in error_msg.lower() or "404" in error_msg:
            logger.error(f"Model '{model_name}' not found. Please ensure the model is pulled in Ollama.")
            logger.error(f"Try running: ollama pull {model_name}")
        raise


def load_templates(csv_path: str) -> List[Dict[str, str]]:
    """
    Load templates from CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of template dictionaries
    """
    templates = []
    logger.info(f"Loading templates from {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            user_query = row.get('user_query', '').strip()
            true_sql = row.get('true_sql', '').strip()
            few_shot_1 = row.get('few_shot_example_1', '').strip()
            few_shot_2 = row.get('few_shot_example_2', '').strip()
            
            if user_query and true_sql:
                templates.append({
                    'index': idx,
                    'user_query': user_query,
                    'true_sql': true_sql,
                    'few_shot_example_1': few_shot_1,
                    'few_shot_example_2': few_shot_2
                })
    
    logger.info(f"Loaded {len(templates)} templates")
    return templates


def build_vector_store(templates: List[Dict[str, str]], client: OpenAI, model_name: str) -> tuple:
    """
    Build FAISS index from templates
    
    Args:
        templates: List of template dictionaries
        client: OpenAI-compatible client for embeddings
        model_name: Name of the embedding model to use
        
    Returns:
        Tuple of (FAISS index, metadata list)
    """
    logger.info("Generating embeddings for templates...")
    
    # Generate embeddings
    embeddings = []
    metadata = []
    
    for i, template in enumerate(templates):
        if (i + 1) % 10 == 0:
            logger.info(f"Processing template {i + 1}/{len(templates)}")
        
        # Generate embedding for user_query
        embedding = get_embedding(template['user_query'], client, model_name)
        embeddings.append(embedding)
        
        # Store metadata
        metadata.append({
            'index': template['index'],
            'user_query': template['user_query'],
            'true_sql': template['true_sql'],
            'few_shot_example_1': template['few_shot_example_1'],
            'few_shot_example_2': template['few_shot_example_2']
        })
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings_array)
    
    # Get dimension
    dimension = embeddings_array.shape[1]
    logger.info(f"Embedding dimension: {dimension}")
    
    # Create FAISS index (using inner product for cosine similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to index
    index.add(embeddings_array)
    logger.info(f"FAISS index created with {index.ntotal} vectors")
    
    return index, metadata


def save_vector_store(index: faiss.Index, metadata: List[Dict[str, Any]], embedding_model: str):
    """
    Save FAISS index and metadata to disk
    
    Args:
        index: FAISS index
        metadata: List of metadata dictionaries
        embedding_model: Name of the embedding model used
    """
    logger.info(f"Saving FAISS index to {INDEX_FILE}")
    faiss.write_index(index, INDEX_FILE)
    
    logger.info(f"Saving metadata to {METADATA_FILE}")
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': metadata,
            'csv_file': CSV_FILE,
            'embedding_model': embedding_model,
            'dimension': index.d
        }, f, indent=2, ensure_ascii=False)
    
    logger.info("Vector store saved successfully")


def main():
    """Main function"""
    logger.info("="*60)
    logger.info("Template Vector Store Builder")
    logger.info("="*60)
    
    try:
        # Initialize Ollama client (OpenAI-compatible)
        # Strip trailing slashes to avoid double slashes
        base_url = OLLAMA_BASE_URL.rstrip('/')
        ollama_base_url = f"{base_url}/v1" if not base_url.endswith('/v1') else base_url
        logger.info(f"Connecting to Ollama at {ollama_base_url}")
        
        # Find the correct embedding model name from available Ollama models
        embedding_model = find_embedding_model(OLLAMA_BASE_URL, EMBEDDING_MODEL)
        logger.info(f"Using embedding model: {embedding_model}")
        
        client = OpenAI(
            base_url=ollama_base_url,
            api_key="ollama"  # Placeholder for Ollama
        )
        
        # Test if the embedding model is available before processing
        if not test_embedding_model(client, embedding_model):
            logger.warning("="*60)
            logger.warning("⚠️  Embedding model is not available")
            logger.warning("="*60)
            
            # Try to pull the model
            logger.info("Attempting to pull the model automatically...")
            if pull_ollama_model(OLLAMA_BASE_URL, embedding_model):
                logger.info("Waiting 10 seconds for model to be pulled...")
                import time
                time.sleep(10)
                
                # Test again after pull attempt
                if test_embedding_model(client, embedding_model):
                    logger.info("✅ Model is now available!")
                else:
                    logger.error("="*60)
                    logger.error("❌ Model pull initiated but model is still not available")
                    logger.error("="*60)
                    logger.error("The model pull may take several minutes.")
                    logger.error("Please wait and try again later, or pull manually:")
                    logger.error(f"  ollama pull {embedding_model}")
                    logger.error("="*60)
                    sys.exit(1)
            else:
                logger.error("="*60)
                logger.error("❌ Cannot proceed: Embedding model is not available")
                logger.error("="*60)
                logger.error("Please ensure the embedding model is pulled on your Ollama instance.")
                logger.error(f"Run: ollama pull {embedding_model}")
                logger.error("="*60)
                sys.exit(1)
        
        # Load templates
        templates = load_templates(CSV_FILE)
        
        if len(templates) == 0:
            raise ValueError("No templates found in CSV file")
        
        # Build vector store
        index, metadata = build_vector_store(templates, client, embedding_model)
        
        # Save to disk
        save_vector_store(index, metadata, embedding_model)
        
        logger.info("="*60)
        logger.info("✅ Vector store built successfully!")
        logger.info(f"  Index file: {INDEX_FILE}")
        logger.info(f"  Metadata file: {METADATA_FILE}")
        logger.info(f"  Templates indexed: {len(metadata)}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Error building vector store: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

