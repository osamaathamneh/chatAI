"""
Shared dependencies for API routes.
"""
from src.api.config import (
    CATALOG_DB,
    TARGET_DB,
    CATALOG_NAME,
    DEFAULT_MODEL,
    TEMPLATE_CSV_PATH,
    DB_PATH
)
from src.services.database import init_db
from src.services.template_matcher import QueryTemplateMatcher
from src.services.llm_template_matcher import LLMTemplateMatcher
from typing import Optional
import logging

# Initialize template matchers
query_matcher = None
llm_matcher = None

def init_query_matcher():
    """Initialize the query template matcher (static)"""
    global query_matcher
    try:
        query_matcher = QueryTemplateMatcher(TEMPLATE_CSV_PATH)
        logging.info(f"✅ Static query template matcher initialized with {len(query_matcher.templates)} templates")
    except Exception as e:
        logging.warning(f"⚠️ Failed to initialize static query matcher: {str(e)}. Static template matching will be disabled.")
        query_matcher = None

def init_llm_matcher():
    """Initialize the LLM-based template matcher"""
    global llm_matcher
    try:
        llm_matcher = LLMTemplateMatcher(TEMPLATE_CSV_PATH)
        logging.info(f"✅ LLM template matcher initialized with {llm_matcher.get_templates_count()} templates (dynamic model selection)")
    except Exception as e:
        logging.warning(f"⚠️ Failed to initialize LLM matcher: {str(e)}. LLM template matching will be disabled.")
        llm_matcher = None

# Initialize matchers on import
init_query_matcher()
init_llm_matcher()

def get_database_url(db_identifier: Optional[str] = None) -> str:
    """
    Map database identifier to actual connection URL from environment variables.
    This prevents exposing credentials in frontend.
    """
    from typing import Optional
    import re
    import logging
    
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
    
    if not db_identifier:
        return TARGET_DB
    
    # For backwards compatibility, if full URL is passed, ignore it and use env vars
    if db_identifier.startswith("postgresql://"):
        logging.warning("Ignoring database URL from request, using environment variables for security")
        return TARGET_DB
    
    # Get the base connection from env (user:password@host:port)
    # Extract from TARGET_DB and rebuild with different database name
    target_match = re.match(r'(postgresql://[^/]+)/(.+)', TARGET_DB)
    if target_match:
        base_url = target_match.group(1)
        
        # Map identifier to database name
        db_map = {
            "salespoint": f"{base_url}/salespoint",
            "report": f"{base_url}/report",
            "default": TARGET_DB
        }
        
        result = db_map.get(db_identifier, TARGET_DB)
        logging.info(f"Mapped database '{db_identifier}' to: {mask_password(result)}")
        return result
    
    # Fallback to TARGET_DB if parsing fails
    return TARGET_DB
