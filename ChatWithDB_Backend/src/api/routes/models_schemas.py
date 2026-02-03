"""
Pydantic models/schemas for API requests.
"""
from pydantic import BaseModel
from typing import Optional
from src.api.config import DEFAULT_MODEL

class SearchRequest(BaseModel):
    query: str
    render: bool = False
    limit: int = 10
    catalog_db: Optional[str] = None
    catalog_name: Optional[str] = None

class GenerateSQLRequest(BaseModel):
    query: str
    model: str = DEFAULT_MODEL
    request_limit: int = 5
    iteration_limit: int = 5
    target_db: Optional[str] = None
    catalog_db: Optional[str] = None
    catalog_name: Optional[str] = None
    temperature: float = 0.3
    provider: str = "openai"  # "openai", "ollama", or "openrouter"

class ExecuteSQLRequest(BaseModel):
    sql: str
    target_db: Optional[str] = None

class SmartGenerateSQLRequest(BaseModel):
    query: str
    model: str = DEFAULT_MODEL
    request_limit: int = 5
    iteration_limit: int = 5
    target_db: Optional[str] = None
    catalog_db: Optional[str] = None
    catalog_name: Optional[str] = None
    temperature: float = 0.3
    provider: str = "openai"  # "openai", "ollama", or "openrouter"
    use_templates: bool = True  # Whether to try template matching first
    similarity_threshold: float = 0.6  # Minimum similarity score for template matching
    auto_execute: bool = False  # Whether to automatically execute the generated SQL
    use_llm_matching: bool = False  # Whether to use LLM for template matching (vs static regex)

class FeedbackLikeRequest(BaseModel):
    user_query: str
    true_sql: str
    model: str
    provider: str
    feedback: str = "like"  # "like" or "dislike"

class FeedbackDislikeRequest(BaseModel):
    user_query: str
    generated_sql: str
    model: str
    provider: str

class RegenerateSQLRequest(BaseModel):
    user_query: str
    model: str
    provider: str
    base_url: Optional[str] = None
