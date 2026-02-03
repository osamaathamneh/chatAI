"""
SQL generation routes.
"""
import os
import time
import sqlite3
import logging
from fastapi import APIRouter, HTTPException
import psycopg
import pgai.semantic_catalog as sc
from pydantic_ai.usage import UsageLimits
from src.api.config import (
    CATALOG_NAME, OLLAMA_BASE_URL, OPENROUTER_BASE_URL, 
    OPENROUTER_API_KEY, DB_PATH
)
from src.api.dependencies import get_database_url, query_matcher, llm_matcher
from src.api.routes.models_schemas import (
    GenerateSQLRequest, SmartGenerateSQLRequest, ExecuteSQLRequest
)

router = APIRouter(prefix="/api", tags=["sql"])

@router.post("/generate-sql")
async def generate_sql(request: GenerateSQLRequest):
    """Generate SQL from natural language query using Python API"""
    start_time = time.time()
    success = False
    
    # Debug logging
    logging.info(f"Received request - temperature: {request.temperature}, iteration_limit: {request.iteration_limit}, request_limit: {request.request_limit}")
    sql_result = None
    error_message = None
    
    try:
        # Use environment variables for security (ignore URLs from frontend)
        target_db = get_database_url(request.target_db)
        catalog_db = get_database_url(request.catalog_db)
        catalog_name = request.catalog_name or CATALOG_NAME
        
        # Handle provider-specific base_url and API key
        original_base_url = os.environ.get("OPENAI_BASE_URL")
        original_api_key = os.environ.get("OPENAI_API_KEY")
        try:
            if request.provider == "ollama":
                # Set Ollama base URL for this request
                ollama_base_url = f"{OLLAMA_BASE_URL}/v1"
                os.environ["OPENAI_BASE_URL"] = ollama_base_url
                logging.info(f"Using Ollama provider with base URL: {ollama_base_url}")
            elif request.provider == "openrouter":
                # Set OpenRouter base URL and API key
                os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
                if OPENROUTER_API_KEY:
                    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
                    logging.info(f"Using OpenRouter provider with base URL: {OPENROUTER_BASE_URL}")
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="OPENROUTER_API_KEY not found in environment variables. Please add it to your .env file."
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
                    
                    # Generate SQL using the Python API with temperature control
                    model_settings = {"temperature": request.temperature}
                    usage_limits = UsageLimits(request_limit=request.request_limit)
                    
                    try:
                        sql_response = await catalog.generate_sql(
                            catalog_con=catalog_con,
                            target_con=target_con,
                            embedding_name=embed_config,
                            prompt=request.query,
                            model=request.model,
                            model_settings=model_settings,
                            usage_limits=usage_limits,
                            iteration_limit=request.iteration_limit
                        )
                    except Exception as sql_gen_error:
                        error_str = str(sql_gen_error)
                        # Check if the error is about tools not being supported
                        if ("does not support tools" in error_str.lower() or 
                            "does not support function" in error_str.lower() or
                            "no endpoints found that support tool use" in error_str.lower() or
                            "tool use" in error_str.lower()):
                            model_name = request.model.split(":")[-1] if ":" in request.model else request.model
                            logging.error(f"Model {model_name} doesn't support tools/function calling")
                            raise HTTPException(
                                status_code=400,
                                detail=f"The model '{model_name}' does not support tool calling, which is required for SQL generation. "
                                       f"Please use a model that supports function calling."
                            )
                        # Re-raise other errors
                        raise
                    
                    # Extract the SQL statement from the response
                    sql_result = sql_response.sql_statement
                    success = True
                    
                    # Calculate actual usage metrics
                    actual_iterations = len(sql_response.messages)
                    
                    # Calculate generation time
                    generation_time = time.time() - start_time
                    
                    # Save to history database
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO query_history 
                            (database_url, catalog_name, provider, model, query, generated_sql, generation_time, success, 
                             request_limit, iteration_limit, temperature, actual_iterations, 
                             input_tokens, output_tokens, total_tokens, base_url)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (target_db, catalog_name, request.provider, request.model, request.query, sql_result, 
                              generation_time, True, request.request_limit, request.iteration_limit, request.temperature,
                              actual_iterations, sql_response.usage.input_tokens, 
                              sql_response.usage.output_tokens, sql_response.usage.total_tokens,
                              OLLAMA_BASE_URL if request.provider == "ollama" else None))
                        conn.commit()
                        conn.close()
                    except Exception as db_error:
                        logging.error(f"Failed to save to history: {str(db_error)}")
                    
                    return {
                        "success": True,
                        "query": request.query,
                        "sql": sql_result,
                        "generation_time": round(generation_time, 2),
                        "actual_iterations": actual_iterations,
                        "token_usage": {
                            "input_tokens": sql_response.usage.input_tokens,
                            "output_tokens": sql_response.usage.output_tokens,
                            "total_tokens": sql_response.usage.total_tokens
                        },
                        "explanation": None
                    }
        finally:
            # Restore original base URL and API key
            if original_base_url is not None:
                os.environ["OPENAI_BASE_URL"] = original_base_url
            elif "OPENAI_BASE_URL" in os.environ:
                del os.environ["OPENAI_BASE_URL"]
            
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
            elif request.provider == "openrouter" and "OPENAI_API_KEY" in os.environ:
                # Restore original key if we changed it for OpenRouter
                if original_api_key:
                    os.environ["OPENAI_API_KEY"] = original_api_key
                
    except Exception as e:
        error_message = str(e)
        generation_time = time.time() - start_time
        
        # Save failed attempt to history
        try:
            target_db = get_database_url(request.target_db)
            catalog_name = request.catalog_name or CATALOG_NAME
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_history 
                (database_url, catalog_name, provider, model, query, generation_time, success, error_message,
                 request_limit, iteration_limit, temperature, actual_iterations, 
                 input_tokens, output_tokens, total_tokens, base_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (target_db, catalog_name, request.provider, request.model, request.query, generation_time, False, 
                  error_message, request.request_limit, request.iteration_limit, request.temperature,
                  None, None, None, None, OLLAMA_BASE_URL if request.provider == "ollama" else None))
            conn.commit()
            conn.close()
        except Exception as db_error:
            logging.error(f"Failed to save error to history: {str(db_error)}")
        
        logging.error(f"SQL generation error: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

@router.post("/smart-generate-sql")
async def smart_generate_sql(request: SmartGenerateSQLRequest):
    """
    Smart SQL generation that tries template matching first, then falls back to PGAI API
    Optionally executes the SQL and returns results
    """
    start_time = time.time()
    
    try:
        # Try template matching first if enabled
        if request.use_templates:
            match_result = None
            
            # Choose between LLM and static matching
            if request.use_llm_matching and llm_matcher:
                logging.info("Using LLM-based template matching...")
                
                # Determine base URL based on provider
                base_url = None
                if request.provider == "ollama":
                    base_url = OLLAMA_BASE_URL
                elif request.provider == "openrouter":
                    base_url = OPENROUTER_BASE_URL
                
                # Get catalog DB URL
                catalog_db_url = get_database_url(request.catalog_db) if request.catalog_db else None
                
                match_result = await llm_matcher.match_query(
                    user_query=request.query,
                    model=request.model,
                    provider=request.provider,
                    base_url=base_url,
                    catalog_db_url=catalog_db_url,
                    catalog_name=request.catalog_name,
                    confidence_threshold=request.similarity_threshold
                )
                
                # Handle new dictionary format from LLM matcher
                if match_result:
                    matching_score = match_result.get("matching_score", 0.0)
                    user_input = match_result.get("user_input", "")
                    matched_user_query = match_result.get("matched_user_query", "")
                    matched_true_sql = match_result.get("matched_true_sql", "")
                    generated_sql = match_result.get("binded_sql", "")
                    match_type = match_result.get("match_type", "llm-template")  # "exact" or "llm-template"
                    
                    # Only use if we have generated SQL (score >= threshold)
                    if generated_sql:
                        # Create a mock template object for compatibility
                        class MockTemplate:
                            def __init__(self, query):
                                self.user_query = query
                        
                        template = MockTemplate(matched_user_query)
                        # Store additional metadata for response
                        template.matched_true_sql = matched_true_sql
                        template.user_input = user_input
                        template.match_type = match_type
                        match_result = (template, matching_score, generated_sql)
                        logging.info(f"LLM matched with score {matching_score:.2f} (type: {match_type})")
                    else:
                        # No SQL generated (score < threshold), fall back to PGAI
                        match_result = None
                        logging.info(f"LLM score {matching_score:.2f} below threshold or no SQL generated, falling back to PGAI")
            
            elif query_matcher:
                logging.info("Using static regex-based template matching...")
                match_result = query_matcher.match_query(request.query, threshold=request.similarity_threshold)
            
            if match_result:
                template, similarity_score, generated_sql = match_result
                generation_time = time.time() - start_time
                
                logging.info(f"✨ Template match found! Score: {similarity_score:.2f}")
                logging.info(f"Template: {template.user_query}")
                logging.info(f"Generated SQL: {generated_sql}")
                
                # Save to history
                try:
                    target_db = get_database_url(request.target_db)
                    catalog_name = request.catalog_name or CATALOG_NAME
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO query_history 
                        (database_url, catalog_name, provider, model, query, generated_sql, generation_time, success, 
                         request_limit, iteration_limit, temperature, actual_iterations, 
                         input_tokens, output_tokens, total_tokens, base_url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (target_db, catalog_name, "template", "template-matching", request.query, generated_sql,
                          generation_time, True, 0, 0, 0, 0, 0, 0, 0,
                          OLLAMA_BASE_URL if request.provider == "ollama" else None))
                    conn.commit()
                    conn.close()
                except Exception as db_error:
                    logging.error(f"Failed to save to history: {str(db_error)}")
                
                # Determine method and explanation based on match type
                if hasattr(template, 'match_type'):
                    match_type = template.match_type
                    if match_type == "exact":
                        method = "template-exact"
                        explanation = f"📋 Exact template match (100% similarity) - SQL from verified template library"
                    elif match_type == "llm-template":
                        method = "template-llm"
                        explanation = f"🎯 AI-assisted template match (similarity: {similarity_score:.0%}) - Template-based with parameter substitution"
                    else:
                        method = "template"
                        explanation = f"Matched to predefined template (similarity: {similarity_score:.0%})"
                else:
                    # Fallback for static regex matcher
                    method = "template"
                    explanation = f"Matched to predefined template (similarity: {similarity_score:.0%})"
                
                result = {
                    "success": True,
                    "query": request.query,
                    "sql": generated_sql,
                    "generation_time": round(generation_time, 3),
                    "method": method,
                    "template_query": template.user_query,
                    "similarity_score": round(similarity_score, 2),
                    "actual_iterations": 0,
                    "token_usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0
                    },
                    "explanation": explanation
                }
                
                # Add new fields if available (from LLM matcher)
                if hasattr(template, 'matched_true_sql'):
                    result["matched_true_sql"] = template.matched_true_sql
                if hasattr(template, 'user_input'):
                    result["user_input"] = template.user_input
                
                # Execute SQL if requested
                if request.auto_execute:
                    try:
                        exec_start = time.time()
                        exec_request = ExecuteSQLRequest(
                            sql=generated_sql,
                            target_db=request.target_db
                        )
                        exec_result = await execute_sql(exec_request)
                        exec_time = time.time() - exec_start
                        
                        result["execution"] = exec_result
                        result["execution_time"] = round(exec_time, 3)
                        result["executed"] = True
                        
                        logging.info(f"✅ SQL executed successfully in {exec_time:.3f}s - {exec_result.get('row_count', 0)} rows")
                    except Exception as exec_error:
                        logging.error(f"SQL execution failed: {str(exec_error)}")
                        result["execution"] = {
                            "success": False,
                            "error": str(exec_error)
                        }
                        result["executed"] = True
                else:
                    result["executed"] = False
                
                return result
        
        # No template match or templates disabled - use PGAI API
        logging.info("No template match found or templates disabled. Using PGAI API...")
        
        # Convert to regular GenerateSQLRequest and call existing function
        gen_request = GenerateSQLRequest(
            query=request.query,
            model=request.model,
            request_limit=request.request_limit,
            iteration_limit=request.iteration_limit,
            target_db=request.target_db,
            catalog_db=request.catalog_db,
            catalog_name=request.catalog_name,
            temperature=request.temperature,
            provider=request.provider
        )
        
        result = await generate_sql(gen_request)
        result["method"] = "pgai"
        result["explanation"] = "🤖 AI generated from scratch using database schema (no template match found)"
        
        # Execute SQL if requested
        if request.auto_execute:
            try:
                exec_start = time.time()
                exec_request = ExecuteSQLRequest(
                    sql=result["sql"],
                    target_db=request.target_db
                )
                exec_result = await execute_sql(exec_request)
                exec_time = time.time() - exec_start
                
                result["execution"] = exec_result
                result["execution_time"] = round(exec_time, 3)
                result["executed"] = True
                
                logging.info(f"✅ SQL executed successfully in {exec_time:.3f}s - {exec_result.get('row_count', 0)} rows")
            except Exception as exec_error:
                logging.error(f"SQL execution failed: {str(exec_error)}")
                result["execution"] = {
                    "success": False,
                    "error": str(exec_error)
                }
                result["executed"] = True
        else:
            result["executed"] = False
        
        return result
        
    except Exception as e:
        logging.error(f"Smart SQL generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute-sql")
async def execute_sql(request: ExecuteSQLRequest):
    """Execute SQL query and return results"""
    try:
        # Use environment variables for security (ignore URLs from frontend)
        target_db = get_database_url(request.target_db)
        
        async with await psycopg.AsyncConnection.connect(target_db) as con:
            async with con.cursor() as cur:
                await cur.execute(request.sql)
                
                # Get column names
                columns = [desc[0] for desc in cur.description] if cur.description else []
                
                # Fetch results
                rows = await cur.fetchall()
                
                # Format results
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                return {
                    "success": True,
                    "columns": columns,
                    "rows": results,
                    "row_count": len(results)
                }
    except Exception as e:
        logging.error(f"SQL execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
