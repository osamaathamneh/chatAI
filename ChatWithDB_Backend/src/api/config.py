"""
Configuration module for the application.
Loads environment variables and provides configuration constants.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix for Windows asyncio with psycopg
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Database configuration
CATALOG_DB = os.getenv("CATALOG_DB", "postgresql://postgres:password@localhost:5432/salespoint")
TARGET_DB = os.getenv("TARGET_DB", CATALOG_DB)
CATALOG_NAME = os.getenv("CATALOG_NAME", "salespoint_production")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai:gpt-5")

# Configure OpenRouter integration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Configure Ollama integration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Template CSV path
TEMPLATE_CSV_PATH = "data/templates/templates_with_prompt_fewshot.csv"

# Database path for query history
DB_PATH = Path("query_history.db")

# Debug: Log loaded configuration (mask password for security)
def mask_password(url):
    """Mask password in database URL for logging"""
    if url and "://" in url:
        try:
            parts = url.split("://")
            if "@" in parts[1]:
                auth_and_rest = parts[1].split("@")
                if ":" in auth_and_rest[0]:
                    user_pass = auth_and_rest[0].split(":")
                    return f"{parts[0]}://{user_pass[0]}:****@{auth_and_rest[1]}"
        except:
            pass
    return url

# Log configuration on import
logging.info(f"Environment loaded - CATALOG_DB: {mask_password(CATALOG_DB)}")
logging.info(f"Environment loaded - TARGET_DB: {mask_password(TARGET_DB)}")
logging.info(f"Environment loaded - CATALOG_NAME: {CATALOG_NAME}")
logging.info(f"Environment loaded - OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set'}")
logging.info(f"Environment loaded - OPENROUTER_API_KEY: {'Set' if OPENROUTER_API_KEY else 'Not Set'}")
logging.info(f"Environment loaded - OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
