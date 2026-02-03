"""
LLM-Based Query Template Matcher
Uses AI to find similar queries and intelligently substitute parameters
Uses FAISS vector store with Ollama embeddings for semantic similarity search
"""
import os
import sys
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)

# Suppress FAISS loader INFO messages (AVX512/AVX2 warnings are just informational)
logging.getLogger('faiss.loader').setLevel(logging.WARNING)

import csv
import re
import json
import random
import numpy as np
import faiss
import psycopg
import pgai.semantic_catalog as sc
import asyncio
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from openai import OpenAI

# Fix Windows event loop policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger(__name__)


class LLMTemplateMatcher:
    """Uses LLM to match queries to templates and substitute parameters"""
    
    def __init__(self, csv_path: str = "data/templates/templates_with_prompt_fewshot.csv", top_k: int = 5):
        """
        Initialize LLM Template Matcher
        
        Args:
            csv_path: Path to CSV file with templates (default: data/templates/templates_with_prompt_fewshot.csv)
            top_k: Number of top similar queries to retrieve (default: 5)
        """
        self.csv_path = csv_path
        self.top_k = top_k
        self.templates: List[Dict[str, str]] = []
        self.vector_index: Optional[faiss.Index] = None
        self.vector_metadata: List[Dict[str, Any]] = []
        self.embedding_client: Optional[OpenAI] = None
        
        # Load templates
        self._load_templates()
        
        # Initialize embedding client FIRST (needed for vector store building)
        self._initialize_embedding_client()
        
        # Initialize vector store (requires embedding client if building on-demand)
        self._initialize_vector_store()
        
        # LLM is always available (we use direct OpenAI client)
        self.llm_available = True
    
    def _load_templates(self):
        """Load templates from CSV file with few-shot examples"""
        try:
            self.templates = []  # Clear existing templates
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    user_query = row.get('user_query', '').strip()
                    true_sql = row.get('true_sql', '').strip()
                    few_shot_example_1 = row.get('few_shot_example_1', '').strip()
                    few_shot_example_2 = row.get('few_shot_example_2', '').strip()
                    feedback = row.get('feedback', '').strip()  # Read feedback column if present
                    
                    if user_query and true_sql:
                        self.templates.append({
                            'index': idx,
                            'user_query': user_query,
                            'true_sql': true_sql,
                            'few_shot_example_1': few_shot_example_1,
                            'few_shot_example_2': few_shot_example_2,
                            'feedback': feedback  # Store feedback for potential future use
                        })
            
            logger.info(f"Loaded {len(self.templates)} templates for LLM matching")
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            raise
    
    def _initialize_embedding_client(self):
        """Initialize Ollama embedding client"""
        try:
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_base_url = f"{ollama_base_url}/v1" if not ollama_base_url.endswith('/v1') else ollama_base_url
            
            self.embedding_client = OpenAI(
                base_url=ollama_base_url,
                api_key="ollama"  # Placeholder for Ollama
            )
            logger.info(f"Initialized Ollama embedding client at {ollama_base_url}")
        except Exception as e:
            logger.error(f"Error initializing embedding client: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize FAISS vector store (load from disk or build if missing)"""
        index_file = "data/templates/template_vectors.faiss"
        metadata_file = "data/templates/template_vectors_metadata.json"
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self._load_vector_store(index_file, metadata_file)
                logger.info("Loaded vector store from disk")
            except Exception as e:
                logger.warning(f"Error loading vector store: {e}. Will rebuild.")
                self._build_vector_store()
        else:
            logger.info("Vector store not found. Building new one...")
            self._build_vector_store()
    
    def _load_vector_store(self, index_file: str, metadata_file: str):
        """Load FAISS index and metadata from disk"""
        self.vector_index = faiss.read_index(index_file)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vector_metadata = data['metadata']
        
        logger.info(f"Loaded vector store: {self.vector_index.ntotal} vectors")
    
    def _build_vector_store(self):
        """Build FAISS vector store from templates"""
        self._build_vector_store_on_demand()
    
    def _add_template_to_vector_store(self, user_query: str, true_sql: str, few_shot_example_1: str = "", few_shot_example_2: str = "", feedback: str = ""):
        """
        Add a single new template to the existing vector store without rebuilding everything.
        This is much more efficient than rebuilding the entire vector store.
        """
        if self.vector_index is None or len(self.vector_metadata) == 0:
            logger.warning("Vector store not initialized. Building from scratch...")
            self._rebuild_vector_store()
            return
        
        logger.info("Adding new template to vector store (incremental update)...")
        
        # Generate embedding for the new template
        new_embedding = self._get_embedding(user_query)
        new_embedding_array = np.array([new_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(new_embedding_array)
        
        # Get the next index (should be the current count)
        new_index = len(self.vector_metadata)
        
        # Add to FAISS index
        self.vector_index.add(new_embedding_array)
        
        # Add metadata
        self.vector_metadata.append({
            'index': new_index,
            'user_query': user_query,
            'true_sql': true_sql,
            'few_shot_example_1': few_shot_example_1,
            'few_shot_example_2': few_shot_example_2,
            'feedback': feedback
        })
        
        logger.info(f"Successfully added template {new_index} to vector store (total: {self.vector_index.ntotal})")
        
        # Save updated vector store to disk
        self._save_vector_store_to_disk()
    
    def _rebuild_vector_store(self):
        """Rebuild vector store after templates are updated (full rebuild)"""
        logger.info("Rebuilding vector store (full rebuild)...")
        # Clear existing vector store
        self.vector_index = None
        self.vector_metadata = []
        
        # Reload templates
        self._load_templates()
        
        # Build vector store
        self._build_vector_store_on_demand()
        
        # Save to disk
        self._save_vector_store_to_disk()
        
        logger.info(f"Vector store rebuilt with {len(self.templates)} templates")
    
    def _save_vector_store_to_disk(self):
        """Save FAISS index and metadata to disk"""
        if self.vector_index is None or len(self.vector_metadata) == 0:
            logger.warning("Vector store not built, cannot save")
            return
        
        index_file = "data/templates/template_vectors.faiss"
        metadata_file = "data/templates/template_vectors_metadata.json"
        
        try:
            logger.info(f"Saving FAISS index to {index_file}")
            faiss.write_index(self.vector_index, index_file)
            
            logger.info(f"Saving metadata to {metadata_file}")
            import json
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': self.vector_metadata,
                    'csv_file': self.csv_path,
                    'embedding_model': "hf.co/bartowski/granite-embedding-30m-english-GGUF:Q4_K_M",
                    'dimension': self.vector_index.d
                }, f, indent=2, ensure_ascii=False)
            
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Ollama API
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not self.embedding_client:
            raise RuntimeError("Embedding client not initialized")
        
        try:
            response = self.embedding_client.embeddings.create(
                model="hf.co/bartowski/granite-embedding-30m-english-GGUF:Q4_K_M",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _retrieve_similar_queries(self, user_query: str, k: int) -> List[Dict[str, Any]]:
        """
        Embed user query and retrieve top K similar templates
        
        Args:
            user_query: User's natural language query
            k: Number of similar queries to retrieve
            
        Returns:
            List of template dictionaries with similarity scores
        """
        if not self.vector_index or len(self.vector_metadata) == 0:
            logger.warning("Vector store not loaded. Building on-demand...")
            # Build vector store on-demand
            self._build_vector_store_on_demand()
        
        # Generate embedding for user query
        query_embedding = self._get_embedding(user_query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search for top K similar vectors
        k = min(k, self.vector_index.ntotal)
        distances, indices = self.vector_index.search(query_vector, k)
        
        # Retrieve templates
        similar_templates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.vector_metadata):
                template = self.vector_metadata[idx].copy()
                template['similarity_score'] = float(distances[0][i])  # Cosine similarity
                similar_templates.append(template)
        
        logger.info(f"Retrieved {len(similar_templates)} similar templates")
        return similar_templates
    
    def _build_vector_store_on_demand(self):
        """Build vector store on-demand if not available"""
        logger.info("Building vector store on-demand...")
        
        # Generate embeddings for all templates
        embeddings = []
        metadata = []
        
        for template in self.templates:
            embedding = self._get_embedding(template['user_query'])
            embeddings.append(embedding)
            metadata.append({
                'index': template['index'],
                'user_query': template['user_query'],
                'true_sql': template['true_sql'],
                'few_shot_example_1': template['few_shot_example_1'],
                'few_shot_example_2': template['few_shot_example_2'],
                'feedback': template.get('feedback', '')  # Include feedback in metadata
            })
        
        # Convert to numpy array and normalize
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)
        self.vector_index.add(embeddings_array)
        self.vector_metadata = metadata
        
        logger.info(f"Built vector store: {self.vector_index.ntotal} vectors")
    
    async def _call_llm_direct(self, prompt: str, model: str, provider: str, base_url: Optional[str] = None, max_tokens: int = 1000) -> str:
        """
        Call LLM directly using OpenAI-compatible client (like test_llm_runpod.py)
        
        Args:
            prompt: Complete prompt to send
            model: Model name (e.g., "gpt-oss:20b", "gpt-4o-mini")
            provider: Provider name ("openai", "ollama", "openrouter")
            base_url: Base URL for the provider (required for ollama, openrouter)
            max_tokens: Maximum tokens for response (default: 1000)
        
        Returns:
            Raw response text from LLM
        """
        # Save original environment variables
        original_base_url = os.environ.get("OPENAI_BASE_URL")
        original_api_key = os.environ.get("OPENAI_API_KEY")
        
        try:
            # Configure client based on provider
            if provider == "ollama":
                if base_url:
                    api_base_url = f"{base_url}/v1" if not base_url.endswith('/v1') else base_url
                else:
                    api_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    api_base_url = f"{api_base_url}/v1" if not api_base_url.endswith('/v1') else api_base_url
                api_key = "ollama"  # Placeholder for Ollama
                logger.info(f"Calling Ollama LLM at {api_base_url}")
            
            elif provider == "openrouter":
                api_base_url = "https://openrouter.ai/api/v1"
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY not found")
                logger.info("Calling OpenRouter LLM")
            
            elif provider == "openai":
                api_base_url = None  # Use default OpenAI endpoint
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found")
                logger.info("Calling OpenAI LLM")
            
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            # Validate model name
            if not model or not model.strip():
                raise ValueError(f"Model name is required but got: '{model}'")
            
            # Extract model name only if it starts with a known provider prefix
            # This prevents double-extraction when model is already extracted (e.g., "gpt-oss:20b")
            model_for_call = model.strip()
            
            # Known provider prefixes that should be removed
            provider_prefixes = ["openai:", "ollama:", "openrouter:"]
            model_lower = model_for_call.lower()
            
            # Only extract if model starts with a provider prefix
            for prefix in provider_prefixes:
                if model_lower.startswith(prefix):
                    model_for_call = model_for_call[len(prefix):].strip()
                    break
            
            if not model_for_call:
                raise ValueError(f"Invalid model name after extraction: '{model}'")
            
            logger.info(f"Calling LLM with model: {model_for_call}, provider: {provider}")
            
            # Create OpenAI client
            client = OpenAI(
                base_url=api_base_url,
                api_key=api_key
            )
            
            # Call LLM (single user message, not system/user split)
            # Wrap blocking call in asyncio for async compatibility
            # Capture variables for lambda closure
            prompt_for_call = prompt
            max_tokens_for_call = max_tokens
            
            def _make_llm_call():
                return client.chat.completions.create(
                    model=model_for_call,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_for_call
                        }
                    ],
                    temperature=0.2,  # Low temperature for consistent output
                    max_tokens=max_tokens_for_call
                )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _make_llm_call)
            
            # Extract response content
            result = response.choices[0].message.content
            if not result:
                raise ValueError("Empty response from LLM")
            
            return result.strip()
            
        finally:
            # Restore original environment variables
            if original_base_url:
                os.environ["OPENAI_BASE_URL"] = original_base_url
            elif "OPENAI_BASE_URL" in os.environ and provider != "openai":
                del os.environ["OPENAI_BASE_URL"]
            
            if original_api_key:
                os.environ["OPENAI_API_KEY"] = original_api_key
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the LLM agent"""
        prompt = """You are an expert in SQL, semantic similarity, and template-based SQL generation.

        Your role is to map a natural language user query to the closest SQL template from a predefined list and then bind new parameters into that template without altering its structure.

        The template list is an array of dictionaries, each with:
        - "user_query": natural language description of the query
        - "true_sql": the canonical SQL template to execute

        CRITICAL INSTRUCTIONS - YOU MUST FOLLOW EXACTLY:

        1. Receive one user_input (natural language text from the user).
        2. Compare this user_input with the list of example "user_query" values in the template list above.
        3. Compute a matching_score between 0.0 and 1.0 (e.g., 0.95) for the BEST single template.
        4. If the best matching_score is ≥ 0.90, select that template.
        5. Extract the new parameters from the user_input (such as dates, statuses, price plan, limits, etc.).
        6. Bind these new parameters into the selected template's "true_sql" without modifying:
        - the SQL structure,
        - table names,
        - column names,
        - or the order of conditions and clauses.
        7. If the matching_score is < 0.90, do NOT generate SQL and leave "binded_sql" empty.
        You SHOULD still return the best-matching template's "user_query" and "true_sql" in the corresponding fields when there is a clear best match.

        OUTPUT FORMAT (MANDATORY):
        You MUST return ONLY a JSON object with these EXACT 5 fields. NO other text, NO markdown, NO explanations:

        {
        "matching_score": 0.95,
        "user_input": "Get all prepaid activations from 2025-05-01 to 2025-08-31",
        "matched_user_query": "Get all prepaid activations from 2025-07-31 to 2025-08-31",
        "matched_true_sql": "SELECT * FROM mobile_prepaid_request WHERE create_date >= '2025-07-31 00:00:00' AND create_date <= '2025-08-31 23:59:59';",
        "binded_sql": "SELECT * FROM mobile_prepaid_request WHERE create_date >= '2025-05-01 00:00:00' AND create_date <= '2025-08-31 23:59:59';"
        }

        ABSOLUTELY NO OTHER FORMAT IS ACCEPTABLE. DO NOT add markdown code blocks, explanations, or any other text.

        FIELD RULES (VERY IMPORTANT):

        "user_input":
        - MUST always be exactly the new user query text you received.
        - Copy it as-is, without any changes.

        "matched_user_query":
        - MUST always represent the BEST-MATCHING template "user_query" from the template list.
        - When a template is selected (matching_score > 0), "matched_user_query" MUST be exactly one of the "user_query" values from the template list, copied character by character.
        - It MUST NOT be the same as "user_input" unless the user_input is literally identical to that template's "user_query".
        - If there is no meaningful match at all (very low similarity, e.g. around 0.4), you MAY set "matched_user_query" to an empty string "".

        "matched_true_sql":
        - When "matched_user_query" is not empty, "matched_true_sql" MUST be exactly the corresponding "true_sql" from the SAME dictionary in the template list.
        - Copy it character by character from that template.
        - If "matched_user_query" is an empty string, "matched_true_sql" MUST also be an empty string "".

        "binded_sql":
        - If matching_score ≥ 0.90:
        - "binded_sql" MUST be derived from the selected template's "true_sql".
        - You MUST keep the same structure and only change parameter values (dates, numbers, simple string filters).
        - You are NOT allowed to invent a brand new SQL that is not based on any "true_sql" from the template list.
        - If matching_score < 0.90:
        - "binded_sql" MUST be an empty string "" (do NOT generate SQL).
        - You MUST still set "matched_user_query" and "matched_true_sql" to the best-matching template from the list when there is a clear best match.
        - For very low similarity (e.g. ~0.4), you MAY set both "matched_user_query" and "matched_true_sql" to empty strings.

        Consistency rule:
        - Whenever "binded_sql" is not empty:
        - "matched_user_query" MUST be the "user_query" of the SAME template whose "true_sql" you used.
        - "matched_true_sql" MUST be exactly that template's "true_sql".
        - If you cannot clearly identify which template you used for "binded_sql", you MUST set:
        - "matched_user_query": ""
        - "matched_true_sql": ""
        - "binded_sql": ""

        FINAL REMINDER:
        Return ONLY the JSON object with the 5 required fields: matching_score, user_input, matched_user_query, matched_true_sql, binded_sql
        NO markdown formatting (no ```json), NO additional text, NO explanations.
        Just the raw JSON object starting with { and ending with }."""
        
        return prompt
    
    def _build_prompt(self, user_query: str, similar_templates: List[Dict[str, Any]], k: int) -> str:
        """
        Build structured prompt with 7 blocks
        
        Args:
            user_query: User's natural language query
            similar_templates: List of top K similar templates
            k: Number of similar templates retrieved
            
        Returns:
            Complete prompt string
        """
        # Block 1: System
        system_block = """You are an expert in SQL, semantic similarity, and template-based SQL generation.

Your role is to map a natural language user query to the closest SQL template from a predefined list and then bind new parameters into that template without altering its structure."""

        # Block 2: Thinking Budget (determine based on SQL length)
        # Check if SQLs are short (< 250 chars) or long (>= 250 chars)
        sql_lengths = [len(t.get('true_sql', '')) for t in similar_templates[:2]]  # Check top 2
        avg_sql_length = sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0
        
        if avg_sql_length < 250:
            thinking_budget = """Option A (fast): Do NOT THINK TOO MUCH. Return the final JSON only."""
        else:
            thinking_budget = """Option B (quality): THINK Deeply. Return the final JSON only."""

        # Block 3: Output Format
        output_format = """Always return output in this exact JSON format and nothing else:

{
"matching_score": 0.0,
"user_input": "",
"matched_user_query": "",
"matched_true_sql": "",
"binded_sql": ""
}

You MUST always return valid JSON and nothing else."""

        # Block 4: Decision Policy
        decision_policy = """- Compare the user_input with the provided Templates.
- Select exactly ONE best-matching template.
- Set matching_score ∈ [0.0, 1.0].

- If matching_score ≥ 0.90:
  - matched_user_query MUST be copied exactly from the selected template.
  - matched_true_sql MUST be copied exactly from the same template.
  - binded_sql MUST be derived from matched_true_sql by changing ONLY parameter values
    (dates, numbers, simple strings).
  - SQL structure, table names, column names, and clause order MUST NOT change.

- If matching_score < 0.90:
  - binded_sql MUST be "".
  - matched_user_query and matched_true_sql MAY be "" if similarity is very low.

Consistency:
- When binded_sql is not empty, all fields MUST refer to the SAME template.
- If a valid template cannot be clearly identified, set all template-related fields to ""."""

        # Block 5: Templates (top K similar)
        templates_array = []
        for template in similar_templates:
            templates_array.append({
                "user_query": template['user_query'],
                "true_sql": template['true_sql']
            })
        
        templates_block = f"""The following list of dictionaries contains a user query ("user_query") and its corresponding SQL ("true_sql"):

{json.dumps(templates_array, indent=2)}"""

        # Block 6: Few-shot Examples
        # Matched cases: few_shot_example_1 and few_shot_example_2 from top 2 matches
        few_shot_block = "LIST OF SQL EXAMPLES\n\nMATCHED CASES:\n\n"
        
        # Get top 2 matches for few-shot examples
        top_2 = similar_templates[:2]
        example_num = 1
        for template in top_2:
            if template.get('few_shot_example_1'):
                few_shot_block += f"EXAMPLE {example_num}:\n{template['few_shot_example_1']}\n\n"
                example_num += 1
            if template.get('few_shot_example_2'):
                few_shot_block += f"EXAMPLE {example_num}:\n{template['few_shot_example_2']}\n\n"
                example_num += 1
        
        # Non-matched cases: 2 random examples from templates NOT in top K
        few_shot_block += "NOT-MATCHED CASES:\n\n"
        
        # Get templates NOT in top K
        # similar_templates come from vector_metadata which has 'index' field
        # We need to match against self.templates which also has 'index' field
        top_k_indices = {t.get('index') for t in similar_templates}
        non_top_k_templates = [t for t in self.templates if t.get('index') not in top_k_indices]
        
        # Select 2 random non-matched examples
        if len(non_top_k_templates) >= 2:
            random_non_matched = random.sample(non_top_k_templates, 2)
        else:
            random_non_matched = non_top_k_templates[:2] if non_top_k_templates else []
        
        for i, template in enumerate(random_non_matched[:2], 1):
            few_shot_block += f"""EXAMPLE {i}:

User Input: {template['user_query']}

Expected Output:

{{
"matching_score": {random.uniform(0.4, 0.6):.2f},
"user_input": "{template['user_query']}",
"matched_user_query": "",
"matched_true_sql": "",
"binded_sql": ""
}}

"""

        # Block 7: Live User Input
        user_input_block = f"""USER INPUT:

{user_query}"""

        # Combine all blocks
        prompt = f"""{system_block}

{thinking_budget}

{output_format}

{decision_policy}

{templates_block}

{few_shot_block}

{user_input_block}"""

        return prompt
    
    def _check_exact_match(self, user_query: str) -> Optional[Tuple[str, str]]:
        """
        Check if user query exactly matches any template query
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            Tuple of (template_query, true_sql) if exact match found, None otherwise
        """
        # Normalize: strip whitespace and normalize multiple spaces
        import re
        user_query_normalized = re.sub(r'\s+', ' ', user_query.strip())
        
        for template in self.templates:
            template_query_normalized = re.sub(r'\s+', ' ', template['user_query'].strip())
            
            # Exact match (case-insensitive comparison)
            if user_query_normalized.lower() == template_query_normalized.lower():
                logger.info(f"✅ EXACT MATCH found (without LLM): {template_query_normalized}")
                return (template['user_query'], template['true_sql'])
        
        return None
    
    def _format_templates_for_prompt(
        self, 
        filtered_templates: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format templates as a JSON array for the prompt
        
        Args:
            filtered_templates: Optional list of filtered templates. 
                               If None, uses all templates.
        
        Returns:
            JSON formatted string with templates
        """
        templates_to_use = filtered_templates if filtered_templates is not None else self.templates
        
        # Create list of dictionaries with only user_query and true_sql
        template_array = []
        for template in templates_to_use:
            template_array.append({
                "user_query": template['user_query'],
                "true_sql": template['true_sql']
            })
        
        # Return as formatted JSON string
        return json.dumps(template_array, indent=2)
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response with new structure validation"""
        try:
            # Try to extract JSON from response (might have markdown code blocks)
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if response_text:
                    response_text = response_text.group(1)
            elif "```" in response_text:
                response_text = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if response_text:
                    response_text = response_text.group(1)
            
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            # Parse JSON
            result = json.loads(response_text.strip())
            
            # Validate required fields for new format
            if not isinstance(result, dict):
                logger.error("Response is not a dictionary")
                return None
            
            required_fields = ['matching_score', 'user_input', 'matched_user_query', 'matched_true_sql', 'binded_sql']
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                logger.error(f"Missing required fields: {missing_fields}")
                logger.error(f"Received fields: {list(result.keys())}")
                logger.error(f"Response excerpt: {str(result)[:200]}")
                return None
            
            # Validate types
            if not isinstance(result['matching_score'], (int, float)):
                logger.error(f"matching_score must be a number, got {type(result['matching_score'])}")
                return None
            
            if not isinstance(result['user_input'], str):
                logger.error(f"user_input must be a string, got {type(result['user_input'])}")
                return None
            
            if not isinstance(result['matched_user_query'], str):
                logger.error(f"matched_user_query must be a string, got {type(result['matched_user_query'])}")
                return None
            
            if not isinstance(result['matched_true_sql'], str):
                logger.error(f"matched_true_sql must be a string, got {type(result['matched_true_sql'])}")
                return None
            
            if not isinstance(result['binded_sql'], str):
                logger.error(f"binded_sql must be a string, got {type(result['binded_sql'])}")
                return None
            
            # Validate matching_score range
            if not (0.0 <= result['matching_score'] <= 1.0):
                logger.warning(f"matching_score {result['matching_score']} outside [0.0, 1.0] range")
                result['matching_score'] = max(0.0, min(1.0, result['matching_score']))
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
    
    async def match_query(
        self, 
        user_query: str,
        model: str,
        provider: str,
        base_url: Optional[str] = None,
        catalog_db_url: Optional[str] = None,
        catalog_name: Optional[str] = None,
        confidence_threshold: float = 0.9
    ) -> Optional[Dict[str, Any]]:
        """
        Match user query to templates using 2-tier approach:
        1. Exact match
        2. LLM matching with ALL templates and parameter binding
        
        Args:
            user_query: The user's natural language query
            model: Model name to use (e.g., "openai:gpt-4o-mini", "ollama:llama3.2")
            provider: Provider name ("openai", "ollama", "openrouter")
            base_url: Base URL for the provider (required for ollama, openrouter)
            catalog_db_url: Not used (kept for backward compatibility)
            catalog_name: Not used (kept for backward compatibility)
            confidence_threshold: Minimum confidence (0.0-1.0) to accept match (default 0.9)
        
        Returns:
            Dictionary with structure:
            {
                "matching_score": float,     # 0.0-1.0
                "user_input": str,           # Original query
                "matched_user_query": str,   # Template query matched (or "")
                "matched_true_sql": str,     # Template SQL (or "")
                "binded_sql": str            # Generated SQL (or "" if score < threshold)
            }
            or None if error occurs
        """
        try:
            # TIER 1: Check for exact match without using LLM
            exact_match = self._check_exact_match(user_query)
            if exact_match:
                template_query, true_sql = exact_match
                print("\n" + "="*80)
                print("⚡ EXACT MATCH (Tier 1 - No LLM needed)")
                print("="*80)
                print(f"User Query: {user_query}")
                print(f"Matched Template: {template_query}")
                print(f"SQL: {true_sql[:100]}...")
                print("Status: SUCCESS - Using exact match (bypassed LLM)")
                print("="*80 + "\n")
                logger.info(f"⚡ Tier 1: Exact match found - returning SQL without LLM")
                return {
                    "matching_score": 1.0,
                    "user_input": user_query,
                    "matched_user_query": template_query,
                    "matched_true_sql": true_sql,
                    "binded_sql": true_sql,
                    "match_type": "exact"  # Distinguish exact matches
                }
            
            # TIER 2: LLM matching with embedding-based retrieval
            logger.info(f"🤖 Tier 2: Using embedding-based LLM matching")
            
            # Retrieve top K similar queries using FAISS
            logger.info(f"Retrieving top {self.top_k} similar templates...")
            similar_templates = self._retrieve_similar_queries(user_query, self.top_k)
            
            if not similar_templates or len(similar_templates) == 0:
                logger.warning("No similar templates found, falling back to PGAI")
                return None
            
            logger.info(f"Found {len(similar_templates)} similar templates")
            
            # Calculate average SQL length for max_tokens determination
            sql_lengths = [len(t.get('true_sql', '')) for t in similar_templates[:2]]  # Check top 2
            avg_sql_length = sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0
            
            # Set max_tokens based on SQL length: 1100 if < 250 chars, else 3000
            max_tokens = 1100 if avg_sql_length < 250 else 3000
            logger.info(f"Average SQL length: {avg_sql_length:.1f} chars, using max_tokens={max_tokens}")
            
            # Build structured prompt with 7 blocks
            prompt = self._build_prompt(user_query, similar_templates, self.top_k)
            
            # Print prompt preview
            print("\n" + "="*80)
            print("PROMPT PREVIEW:")
            print("="*80)
            # Print first 2000 chars + last 1000 chars
            if len(prompt) > 3000:
                print(prompt[:2000])
                print(f"\n... [{len(prompt) - 3000} characters] ...\n")
                print(prompt[-1000:])
            else:
                print(prompt)
            print("="*80 + "\n")
            
            # Call LLM directly (async call)
            logger.info(f"Calling LLM to match query: {user_query[:50]}...")
            try:
                # Extract model name (remove provider prefix if present, e.g., "ollama:gpt-oss:20b" -> "gpt-oss:20b")
                model_name = model.split(":", 1)[-1] if ":" in model else model
                response_text = await self._call_llm_direct(prompt, model_name, provider, base_url, max_tokens=max_tokens)
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                print("\n" + "="*80)
                print("⚠️  FALLBACK: Error calling LLM")
                print("="*80)
                print(f"Error: {str(e)}")
                print("Will fall back to alternative method (PGAI API)")
                print("="*80 + "\n")
                return None
            
            # Print full LLM response
            print("\n" + "="*80)
            print("LLM RESPONSE:")
            print("="*80)
            print(response_text)
            print("="*80 + "\n")
            
            logger.debug(f"LLM raw response: {response_text[:500]}")
            
            # Parse JSON response
            match_result = self._parse_llm_response(response_text)
            
            if not match_result:
                print("\n" + "="*80)
                print("⚠️  FALLBACK: Failed to parse LLM response")
                print("="*80)
                print("Reason: LLM returned invalid JSON or missing required fields")
                print("Required fields: matching_score, user_input, matched_user_query, matched_true_sql, binded_sql")
                print("Will fall back to alternative method (PGAI API)")
                print("="*80 + "\n")
                logger.error("Failed to parse LLM response as JSON")
                return None
            
            # Extract fields from new format
            matching_score = match_result.get('matching_score', 0.0)
            user_input = match_result.get('user_input', '')
            matched_user_query = match_result.get('matched_user_query', '')
            matched_true_sql = match_result.get('matched_true_sql', '')
            binded_sql = match_result.get('binded_sql', '')
            
            logger.info(f"LLM Match Result: matching_score={matching_score:.2f}")
            logger.info(f"  user_input: {user_input[:50]}...")
            logger.info(f"  matched_user_query: {matched_user_query[:50] if matched_user_query else '(empty)'}...")
            logger.info(f"  binded_sql: {'Present' if binded_sql else 'Empty'}")
            
            # Print match result summary
            print("\n" + "="*80)
            print("✅ LLM MATCH RESULT")
            print("="*80)
            print(f"Matching Score: {matching_score:.2f}")
            print(f"Confidence Threshold: {confidence_threshold:.2f}")
            print(f"User Input: {user_input[:80]}...")
            print(f"Matched Template: {matched_user_query[:80] if matched_user_query else '(empty)'}...")
            
            if binded_sql:
                print(f"✅ SQL Generated: {binded_sql[:100]}...")
                print("Status: SUCCESS - Using LLM result")
            else:
                print("⚠️  SQL Generated: (empty)")
                print(f"Status: LOW CONFIDENCE (score {matching_score:.2f} < {confidence_threshold:.2f})")
                print("Will fall back to alternative method (PGAI API)")
            print("="*80 + "\n")
            
            # Add match_type to distinguish LLM-based template matching
            match_result["match_type"] = "llm-template-embedding"
            
            # Return the full result dictionary
            logger.info(f"✨ LLM analysis complete with score {matching_score:.2f}")
            if binded_sql:
                logger.info(f"Generated SQL: {binded_sql[:100]}...")
            else:
                logger.info(f"No SQL generated (score < {confidence_threshold} or no match)")
            
            return match_result
                
        except Exception as e:
            print("\n" + "="*80)
            print("⚠️  FALLBACK: Exception during LLM matching")
            print("="*80)
            print(f"Error: {str(e)}")
            print("Will fall back to alternative method (PGAI API)")
            print("="*80 + "\n")
            logger.error(f"Error in LLM template matching: {str(e)}", exc_info=True)
            return None
    
    def get_templates_count(self) -> int:
        """Get number of loaded templates"""
        return len(self.templates)
    
    async def _retrieve_relevant_tables(
        self, 
        user_query: str, 
        catalog_db_url: str,
        catalog_name: str,
        limit: int = 10
    ) -> List[str]:
        """
        Retrieve relevant tables using PGAI semantic search
        
        Args:
            user_query: The user's natural language query
            catalog_db_url: Database URL for the semantic catalog
            catalog_name: Name of the semantic catalog
            limit: Maximum number of objects to retrieve
            
        Returns:
            List of table names relevant to the query
        """
        try:
            async with await psycopg.AsyncConnection.connect(catalog_db_url) as con:
                # Get semantic catalog
                catalog = await sc.from_name(con, catalog_name)
                
                # Get first available embedding config
                embeddings = await catalog.list_embeddings(con)
                if not embeddings or len(embeddings) == 0:
                    logger.warning("No embedding configurations found in semantic catalog")
                    return []
                
                embed_config = embeddings[0][0]
                logger.info(f"Using embedding config: {embed_config}")
                
                # Search for relevant database objects
                obj_matches = await catalog.search_objects(
                    con, 
                    embedding_name=embed_config, 
                    query=user_query, 
                    limit=limit
                )
                
                # Extract table names from matches - ONLY process table objects, not columns
                table_names = []
                table_names_set = set()  # Use set to avoid duplicates
                
                for match in obj_matches:
                    # Debug: print the full objnames structure
                    logger.info(f"PGAI object: {match.objnames}, type: {match.objtype if hasattr(match, 'objtype') else 'unknown'}")
                    
                    # ONLY process table-level objects (2 elements: schema + table)
                    # Skip column-level objects (3+ elements: schema + table + column + ...)
                    if match.objnames and len(match.objnames) == 2:
                        # This is a table: ['public', 'mobile_prepaid_request']
                        schema = match.objnames[0]
                        table_name = match.objnames[1]
                        
                        # Verify it's actually a table (not a view or other object type)
                        obj_type = match.objtype if hasattr(match, 'objtype') else 'unknown'
                        logger.info(f"  → Schema: {schema}, Table: {table_name}, Type: {obj_type}")
                        
                        # Add to set to avoid duplicates
                        if table_name not in table_names_set:
                            table_names_set.add(table_name)
                            table_names.append(table_name)
                            logger.info(f"✓ Added table: {table_name}")
                    else:
                        # This is a column or other object - skip it
                        logger.debug(f"  → Skipping non-table object (length={len(match.objnames)})")
                
                logger.info(f"Retrieved {len(table_names)} unique relevant tables from PGAI: {table_names}")
                return table_names
                
        except Exception as e:
            logger.error(f"Error retrieving tables from PGAI: {str(e)}")
            return []
    
    def _filter_templates_by_tables(
        self, 
        relevant_tables: List[str]
    ) -> List[Dict[str, str]]:
        """
        Filter templates based on relevant tables
        
        Args:
            relevant_tables: List of table names from PGAI search
            
        Returns:
            List of templates that use any of the relevant tables
        """
        if not relevant_tables:
            return []
        
        # Normalize table names to lowercase for comparison
        relevant_tables_lower = [t.lower() for t in relevant_tables]
        logger.info(f"Filtering templates - Relevant tables from PGAI: {relevant_tables_lower}")
        
        filtered_templates = []
        for template in self.templates:
            related_tables_str = template.get('related_tables', '')
            if not related_tables_str:
                continue
            
            # Split comma-separated table names and normalize
            template_tables = [t.strip().lower() for t in related_tables_str.split(',')]
            
            # Check if any template table matches any relevant table
            for template_table in template_tables:
                if template_table in relevant_tables_lower:
                    filtered_templates.append(template)
                    logger.info(f"✓ Matched template: '{template['user_query'][:50]}...' (table: {template_table})")
                    break  # Avoid duplicates
        
        if not filtered_templates:
            logger.warning(f"No templates matched! PGAI tables: {relevant_tables_lower}")
            logger.warning(f"Sample template tables from CSV: {[self.templates[i]['related_tables'] for i in range(min(3, len(self.templates)))]}")
        
        logger.info(f"Filtered to {len(filtered_templates)} templates based on relevant tables")
        return filtered_templates


# Example usage
async def main():
    """Test the LLM template matcher"""
    matcher = LLMTemplateMatcher("salespoint_testing_samples_solved.csv")
    
    test_queries = [
        "Get all prepaid activations from 2025-07-31 to 2025-08-31",  # Should be exact match
        "Get all completed requests from 31/1 to 5/12",  # Should match and substitute dates
        "Show me prepaid activations from yesterday",
        "Get best performance dealers top 5",
    ]
    
    # Example: using OpenAI
    model = "openai:gpt-4o-mini"
    provider = "openai"
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Test Query: {query}")
        print('='*80)
        
        result = await matcher.match_query(query, model=model, provider=provider)
        
        if result:
            template, confidence, sql, reasoning = result
            print(f"✅ Match Found! (Confidence: {confidence:.0%})")
            print(f"Template: {template}")
            print(f"Generated SQL: {sql}")
            print(f"Reasoning: {reasoning}")
        else:
            print("❌ No match found - will use PGAI API")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
