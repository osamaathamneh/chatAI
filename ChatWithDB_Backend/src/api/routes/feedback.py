"""
Feedback routes for template management.
"""
import os
import json
import csv
import logging
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict
from src.api.config import (
    CATALOG_NAME, OLLAMA_BASE_URL, OPENROUTER_BASE_URL, 
    OPENROUTER_API_KEY, TEMPLATE_CSV_PATH
)
from src.api.dependencies import get_database_url, query_matcher, llm_matcher
from src.api.routes.models_schemas import (
    FeedbackLikeRequest, FeedbackDislikeRequest, RegenerateSQLRequest
)
from src.api.routes.sql_generation import generate_sql
import psycopg
import pgai.semantic_catalog as sc
from pydantic_ai.usage import UsageLimits

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

def _ensure_feedback_column_in_csv(csv_path: str):
    """Ensure CSV file has feedback column in header"""
    try:
        # Check if file is writable
        if os.path.exists(csv_path):
            if not os.access(csv_path, os.W_OK):
                raise PermissionError(
                    f"CSV file '{csv_path}' is not writable. "
                    "Please check file permissions or close the file if it's open in another program."
                )
        
        # Read existing file
        rows = []
        has_feedback_column = False
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_feedback_column = 'feedback' in fieldnames
            
            for row in reader:
                rows.append(row)
        
        # If feedback column is missing, add it
        if not has_feedback_column:
            logging.info(f"Adding 'feedback' column to {csv_path}")
            fieldnames = list(fieldnames) + ['feedback']
            
            # Write back with feedback column
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    row['feedback'] = ''  # Empty for existing rows
                    writer.writerow(row)
            
            logging.info(f"Successfully added 'feedback' column to {csv_path}")
    except FileNotFoundError:
        # File doesn't exist yet, will be created when first row is added
        pass
    except Exception as e:
        logging.warning(f"Error ensuring feedback column: {str(e)}")

def _append_to_csv(user_query: str, true_sql: str, few_shot_1: str, few_shot_2: str, feedback: str, csv_path: str = TEMPLATE_CSV_PATH):
    """Append a new template row to CSV file"""
    try:
        # Check file permissions before attempting to write
        if os.path.exists(csv_path):
            if not os.access(csv_path, os.W_OK):
                raise PermissionError(
                    f"CSV file '{csv_path}' is not writable. "
                    "Possible causes:\n"
                    "1. File is open in Excel or another program - please close it\n"
                    "2. File is set to read-only - check file properties\n"
                    "3. Insufficient permissions - run as administrator or check file permissions"
                )
        
        # Ensure feedback column exists (if file exists)
        if os.path.exists(csv_path):
            _ensure_feedback_column_in_csv(csv_path)
        
        # Define standard fieldnames
        fieldnames = ['user_query', 'true_sql', 'few_shot_example_1', 'few_shot_example_2', 'feedback']
        
        file_exists = os.path.exists(csv_path)
        if file_exists:
            # Read existing header to preserve column order
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                # Use existing fieldnames if they contain all required fields
                if set(fieldnames).issubset(set(existing_fieldnames)):
                    fieldnames = list(existing_fieldnames)
        
        # Append row (or create file if it doesn't exist)
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            # Create row dict with all fieldnames, filling missing ones with empty string
            row_dict = {}
            for field in fieldnames:
                if field == 'user_query':
                    row_dict[field] = user_query
                elif field == 'true_sql':
                    row_dict[field] = true_sql
                elif field == 'few_shot_example_1':
                    row_dict[field] = few_shot_1
                elif field == 'few_shot_example_2':
                    row_dict[field] = few_shot_2
                elif field == 'feedback':
                    row_dict[field] = feedback
                else:
                    row_dict[field] = ''  # For any extra columns
            
            writer.writerow(row_dict)
        
        logging.info(f"Successfully appended template to {csv_path}")
    except PermissionError as e:
        logging.error(f"Permission denied when writing to CSV: {str(e)}")
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )
    except Exception as e:
        logging.error(f"Error appending to CSV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save template to CSV: {str(e)}"
        )

async def _generate_few_shot_examples(user_query: str, true_sql: str, model: str, provider: str, base_url: Optional[str] = None) -> Dict[str, str]:
    """
    Generate few-shot examples using LLM.
    Always uses OpenRouter with gpt-oss-120b for few-shot generation,
    regardless of the provider/model used for SQL generation.
    """
    if not llm_matcher:
        raise HTTPException(status_code=500, detail="LLM matcher not initialized")
    
    # Always use OpenRouter with gpt-oss-120b for few-shot generation
    few_shot_provider = "openrouter"
    few_shot_model = "openai/gpt-oss-120b"  # OpenRouter model ID format
    few_shot_base_url = OPENROUTER_BASE_URL
    
    # Check if OpenRouter API key is available
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY not found. Please configure it in your .env file to use few-shot generation."
        )
    
    logging.info(f"Generating few-shot examples using OpenRouter with model '{few_shot_model}' (ignoring user selection: {provider}/{model})")
    
    # Few-shot generation prompt (from user specification)
    prompt = """You are an expert in NL-to-SQL template matching and parameter binding.

Your task:

Given ONE row containing:

- user_query (canonical natural-language query)

- true_sql (canonical SQL template for user_query)

You must generate TWO prompt-ready few-shot examples:

- few_shot_example_1

- few_shot_example_2

Definitions:

1) few_shot_example_1

- A paraphrase of user_query with the SAME intent.

- NO value changes (same time window, same filters).

- binded_sql MUST be IDENTICAL to matched_true_sql.

2) few_shot_example_2  (IMPORTANT – read carefully)

- A paraphrase that changes EXACTLY ONE value compared to user_query.

- The intent MUST remain the same (same table/domain).

- Prefer changing a TIME-RELATED value when possible.

- The binded_sql MUST reflect ONLY that changed value.

Allowed single-value changes (choose ONE only):

- Time range:

  - "last month" → "five months"

  - "last week" → "last 2 weeks"

  - "yesterday" → "last 3 days"

  - "from 2025-08-01 to 2025-08-31" → "from 2025-07-01 to 2025-08-31"

- Numeric values:

  - "top 10" → "top 20"

  - "limit 50" → "limit 100"

- Status / category (only if time is not present):

  - "completed requests" → "failed requests"

SQL binding rules:

- DO NOT change table names unless required by the changed value.

- Preserve the original SQL structure as much as possible.

- Examples:

  - INTERVAL '1 month' → INTERVAL '5 months'

  - CURRENT_DATE - INTERVAL '7 day' → CURRENT_DATE - INTERVAL '14 day'

  - Explicit date literals must be updated consistently (start and end).

What NOT to do:

- ❌ Do NOT change more than one value.

- ❌ Do NOT change intent (e.g., prepaid → postpaid).

- ❌ Do NOT invent new filters not implied by the user input.

- ❌ Do NOT reformat SQL (keep it one line).

Output format rules (STRICT):

1) Return ONLY a valid JSON object (no markdown, no extra text).

2) JSON must have exactly these keys:

   - "few_shot_example_1"

   - "few_shot_example_2"

3) Each few_shot_example value MUST be a single multi-line string in exactly this style:

User Input: <text>

Expected Output:

        {

        "matching_score": <number between 0.90 and 0.99 with 2 decimals>,

        "user_input": "<same as User Input text>",

        "matched_user_query": "<the provided user_query>",

        "matched_true_sql": "<the provided true_sql>",

        "binded_sql": "<SQL after binding; for example_1 it must equal matched_true_sql>"

        }

4) Use double quotes everywhere.

5) Keep SQL on ONE LINE.

6) matching_score:

   - few_shot_example_1: choose a value between 0.94 and 0.99

   - few_shot_example_2: choose a value between 0.90 and 0.98

Input (one row):

user_query: """ + user_query + """

true_sql: """ + true_sql + """

*******************************************"""
    
    try:
        # Call LLM directly using OpenRouter with gpt-oss-120b
        # Use higher max_tokens for few-shot examples (they can be long with SQL queries)
        response_text = await llm_matcher._call_llm_direct(
            prompt=prompt,
            model=few_shot_model,
            provider=few_shot_provider,
            base_url=few_shot_base_url,
            max_tokens=10000  # Increased for longer few-shot examples with SQL
        )
        
        # Preserve original response for error recovery
        original_response_text = response_text
        
        # Parse JSON response
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        
        # Try to extract JSON from markdown code blocks
        if response_text.startswith('```'):
            # Extract JSON from markdown
            lines = response_text.split('\n')
            json_start = None
            json_end = None
            for i, line in enumerate(lines):
                if line.strip().startswith('{') and json_start is None:
                    json_start = i
                if line.strip().startswith('}') and json_start is not None:
                    json_end = i + 1
                    break
            if json_start is not None and json_end is not None:
                response_text = '\n'.join(lines[json_start:json_end])
        
        # Handle leading non-JSON characters (e.g., ".", "Here is", etc.)
        # Find the first occurrence of '{' which should be the start of JSON
        first_brace = response_text.find('{')
        if first_brace > 0:
            # There are characters before the JSON, strip them
            response_text = response_text[first_brace:]
            logging.info(f"Stripped {first_brace} leading characters before JSON")
        
        # Try to find JSON object boundaries if response is truncated
        if not response_text.strip().endswith('}'):
            # Response might be truncated, try to find the last complete JSON structure
            last_brace = response_text.rfind('}')
            if last_brace > 0:
                # Try to extract up to the last complete brace
                potential_json = response_text[:last_brace + 1]
                # Check if it starts with {
                first_brace_in_potential = potential_json.find('{')
                if first_brace_in_potential >= 0:
                    response_text = potential_json[first_brace_in_potential:]
                    logging.warning("Response appeared truncated, attempting to parse partial JSON")
        
        result = json.loads(response_text)
        
        return {
            'few_shot_example_1': result.get('few_shot_example_1', ''),
            'few_shot_example_2': result.get('few_shot_example_2', '')
        }
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response as JSON: {str(e)}")
        # Log more of the response to help debug
        response_preview = response_text[:2000] if len(response_text) > 2000 else response_text
        logging.error(f"Response preview ({len(response_text)} chars total): {response_preview}")
        if len(response_text) > 2000:
            logging.error(f"Response ends with: ...{response_text[-200:]}")
        
        # Try recovery: find first { and last } and extract JSON from original response
        logging.info("Attempting to recover JSON by finding first '{' and last '}'...")
        # Use original response for recovery (should always be available in JSONDecodeError handler)
        try:
            recovery_text = original_response_text.strip()
        except NameError:
            # Fallback to current response_text if original not available (shouldn't happen)
            recovery_text = response_text.strip()
        first_brace = recovery_text.find('{')
        last_brace = recovery_text.rfind('}')
        
        if first_brace >= 0 and last_brace > first_brace:
            try:
                recovered_json = recovery_text[first_brace:last_brace + 1]
                result = json.loads(recovered_json)
                logging.info("Successfully recovered JSON after initial parse failure")
                return {
                    'few_shot_example_1': result.get('few_shot_example_1', ''),
                    'few_shot_example_2': result.get('few_shot_example_2', '')
                }
            except json.JSONDecodeError as recovery_error:
                logging.error(f"Recovery attempt also failed: {str(recovery_error)}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to parse few-shot examples: {str(e)}. Response may be truncated or malformed."
        )
    except Exception as e:
        logging.error(f"Error generating few-shot examples: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating few-shot examples: {str(e)}")

@router.post("/like")
async def feedback_like(request: FeedbackLikeRequest):
    """Handle like feedback - add template to CSV with few-shot examples"""
    try:
        logging.info(f"Processing like feedback for query: {request.user_query[:50]}...")
        
        # Validate inputs
        if not request.user_query or not request.user_query.strip():
            raise HTTPException(status_code=400, detail="user_query is required")
        if not request.true_sql or not request.true_sql.strip():
            raise HTTPException(status_code=400, detail="true_sql is required")
        if not request.model or not request.model.strip():
            raise HTTPException(status_code=400, detail="model is required")
        if not request.provider or not request.provider.strip():
            raise HTTPException(status_code=400, detail="provider is required")
        
        logging.info(f"Feedback request - Model: {request.model}, Provider: {request.provider}")
        
        # Determine base URL based on provider
        base_url = None
        if request.provider == "ollama":
            base_url = OLLAMA_BASE_URL
        elif request.provider == "openrouter":
            base_url = OPENROUTER_BASE_URL
        
        # Generate few-shot examples
        logging.info("Generating few-shot examples...")
        # Pass model as-is, let _generate_few_shot_examples handle extraction
        few_shot_examples = await _generate_few_shot_examples(
            user_query=request.user_query,
            true_sql=request.true_sql,
            model=request.model,
            provider=request.provider,
            base_url=base_url
        )
        
        # Append to CSV
        logging.info("Appending template to CSV...")
        _append_to_csv(
            user_query=request.user_query,
            true_sql=request.true_sql,
            few_shot_1=few_shot_examples['few_shot_example_1'],
            few_shot_2=few_shot_examples['few_shot_example_2'],
            feedback=request.feedback,
            csv_path=TEMPLATE_CSV_PATH
        )
        
        # Reload templates and add new template to vector store incrementally
        logging.info("Updating template matchers...")
        if llm_matcher:
            # Reload templates to get updated list
            llm_matcher._load_templates()
            # Add new template to vector store incrementally (much faster than full rebuild)
            logging.info("Adding new template to vector store incrementally...")
            llm_matcher._add_template_to_vector_store(
                user_query=request.user_query,
                true_sql=request.true_sql,
                few_shot_example_1=few_shot_examples['few_shot_example_1'],
                few_shot_example_2=few_shot_examples['few_shot_example_2'],
                feedback=request.feedback
            )
        
        if query_matcher:
            query_matcher.reload_templates()
        
        logging.info("Feedback processed successfully")
        return {
            "success": True,
            "message": "Template added successfully",
            "feedback": request.feedback
        }
    except Exception as e:
        logging.error(f"Error processing like feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dislike")
async def feedback_dislike(request: FeedbackDislikeRequest):
    """Handle dislike feedback - return suggested SQL regeneration"""
    try:
        logging.info(f"Processing dislike feedback for query: {request.user_query[:50]}...")
        
        # Determine base URL based on provider
        base_url = None
        if request.provider == "ollama":
            base_url = OLLAMA_BASE_URL
        elif request.provider == "openrouter":
            base_url = OPENROUTER_BASE_URL
        
        # Return the generated SQL as a suggestion (user will provide corrected version)
        return {
            "success": True,
            "message": "Please provide the correct SQL",
            "generated_sql": request.generated_sql
        }
    except Exception as e:
        logging.error(f"Error processing dislike feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/regenerate-sql")
async def regenerate_sql(request: RegenerateSQLRequest):
    """Regenerate SQL using PGAI with iterative generation (iteration_limit=10, temperature=0.3)"""
    try:
        logging.info(f"Regenerating SQL for query: {request.user_query[:50]}...")
        
        # Use PGAI API for regeneration (same as smart_generate_sql but with higher iteration limit)
        # Get default database and catalog from environment
        target_db = get_database_url(None)
        catalog_db = get_database_url(None)
        catalog_name = CATALOG_NAME
        
        # Handle provider-specific base_url and API key (same as smart_generate_sql)
        original_base_url = os.environ.get("OPENAI_BASE_URL")
        original_api_key = os.environ.get("OPENAI_API_KEY")
        
        try:
            if request.provider == "ollama":
                ollama_base_url = f"{OLLAMA_BASE_URL}/v1"
                os.environ["OPENAI_BASE_URL"] = ollama_base_url
                logging.info(f"Using Ollama provider with base URL: {ollama_base_url}")
            elif request.provider == "openrouter":
                os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
                if OPENROUTER_API_KEY:
                    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
                    logging.info(f"Using OpenRouter provider with base URL: {OPENROUTER_BASE_URL}")
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="OPENROUTER_API_KEY not found in environment variables."
                    )
            else:
                # For OpenAI, unset the base URL (or keep default)
                if "OPENAI_BASE_URL" in os.environ:
                    del os.environ["OPENAI_BASE_URL"]
                logging.info("Using OpenAI provider with default base URL")
            
            async with await psycopg.AsyncConnection.connect(catalog_db) as catalog_con:
                async with await psycopg.AsyncConnection.connect(target_db) as target_con:
                    # Load the semantic catalog
                    catalog = await sc.from_name(catalog_con, catalog_name)
                    
                    # Get embedding configuration
                    embeddings = await catalog.list_embeddings(catalog_con)
                    if not embeddings or len(embeddings) == 0:
                        raise HTTPException(status_code=500, detail="No embedding configurations found")
                    
                    embed_config = embeddings[0][0]
                    
                    # Use PGAI with iteration_limit=10 and temperature=0.3 for regeneration
                    model_settings = {"temperature": 0.3}
                    usage_limits = UsageLimits(request_limit=5)  # Standard request limit
                    
                    sql_response = await catalog.generate_sql(
                        catalog_con=catalog_con,
                        target_con=target_con,
                        embedding_name=embed_config,
                        prompt=request.user_query,
                        model=request.model,
                        model_settings=model_settings,
                        usage_limits=usage_limits,
                        iteration_limit=10  # Maximum iterations for better results
                    )
                    
                    sql = sql_response.sql_statement
                    
                    return {
                        "success": True,
                        "sql": sql
                    }
        finally:
            # Restore original environment variables
            if original_base_url is not None:
                os.environ["OPENAI_BASE_URL"] = original_base_url
            elif "OPENAI_BASE_URL" in os.environ:
                del os.environ["OPENAI_BASE_URL"]
            
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            elif "OPENAI_API_KEY" in os.environ and request.provider == "ollama":
                # Only delete if we set a placeholder
                if os.environ.get("OPENAI_API_KEY") == "ollama":
                    del os.environ["OPENAI_API_KEY"]
                    
    except Exception as e:
        logging.error(f"Error regenerating SQL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
