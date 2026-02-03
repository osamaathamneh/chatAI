"""
Evaluation routes for model comparison.
"""
import os
import json
import csv
import io
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.api.config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY
from src.api.routes.sql_generation import generate_sql
from src.api.routes.models_schemas import GenerateSQLRequest
from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

class ModelInfo(BaseModel):
    value: str
    provider: str

class EvaluationRequest(BaseModel):
    queries: List[str]
    true_sqls: Optional[List[Optional[str]]] = None
    models: List[ModelInfo]
    baseline_model: Optional[str] = None
    database: str
    catalog: str
    temperature: float = 0.3
    request_limit: int = 5
    iteration_limit: int = 5

def save_evaluation_results(request: EvaluationRequest, results: dict):
    """Save evaluation results to a folder with timestamp"""
    try:
        # Create evaluations folder if it doesn't exist
        evaluations_dir = Path("evaluations")
        evaluations_dir.mkdir(exist_ok=True)
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_folder = evaluations_dir / timestamp
        eval_folder.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "database": request.database,
            "catalog": request.catalog,
            "temperature": request.temperature,
            "models": [{"value": m.value, "provider": m.provider} for m in request.models],
            "queries_count": len(request.queries),
            "has_true_sqls": len(request.true_sqls) > 0 if request.true_sqls else False,
            "baseline_model": request.baseline_model
        }
        
        with open(eval_folder / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Save full results as JSON
        with open(eval_folder / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        # Save summary as CSV
        if results.get("summary"):
            with open(eval_folder / "summary.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Provider", "Model", "Accuracy (%)", "Avg Speed (s)", "Total Time (s)", 
                                "Avg Iterations", "Temperature", "Success Rate (%)"])
                for item in results["summary"]:
                    writer.writerow([
                        item["provider"],
                        item["model_name"],
                        f"{item['accuracy']:.2f}",
                        f"{item['avg_speed']:.2f}",
                        f"{item['total_time']:.2f}",
                        f"{item['avg_iterations']:.1f}",
                        f"{item['temperature']:.1f}",
                        f"{item['success_rate']:.1f}"
                    ])
        
        # Save detailed results as text report
        report = "SQL Generation Model Evaluation Report\n"
        report += "=" * 60 + "\n\n"
        report += f"Timestamp: {timestamp}\n"
        report += f"Database: {request.database}\n"
        report += f"Catalog: {request.catalog}\n"
        report += f"Temperature: {request.temperature}\n\n"
        
        report += "Summary:\n"
        report += "-" * 60 + "\n"
        for s in results["summary"]:
            report += f"Model: {s['model_name']} ({s['provider']})\n"
            report += f"  Accuracy: {s['accuracy']:.2f}%\n"
            report += f"  Avg Speed: {s['avg_speed']:.2f}s\n"
            report += f"  Success Rate: {s['success_rate']:.1f}%\n"
            report += f"  Avg Iterations: {s['avg_iterations']:.1f}\n\n"
        
        report += "\n\nDetailed Results:\n"
        report += "=" * 60 + "\n"
        for idx, qr in enumerate(results["detailed"], 1):
            report += f"\nQuery {idx}: {qr['query']}\n"
            report += "-" * 60 + "\n"
            if qr.get("true_sql"):
                report += f"True SQL: {qr['true_sql']}\n\n"
            for r in qr["results"]:
                report += f"  Model: {r['model']}\n"
                report += f"    Success: {'Yes' if r['success'] else 'No'}\n"
                report += f"    Accuracy: {r['accuracy_score']:.2f}%\n"
                report += f"    Speed: {r['generation_time']:.2f}s\n"
                report += f"    Iterations: {r['iterations']}\n"
                if r.get("generated_sql"):
                    report += f"    Generated SQL: {r['generated_sql']}\n"
                if r.get("error"):
                    report += f"    Error: {r['error']}\n"
                report += "\n"
        
        with open(eval_folder / "report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        logging.info(f"Evaluation results saved to {eval_folder}")
        
    except Exception as e:
        logging.error(f"Error saving evaluation results: {str(e)}")
        raise

async def judge_sql_accuracy(query: str, generated_sql: str, reference_sql: str) -> float:
    """Use Claude 3.5 Sonnet as judge to score SQL accuracy"""
    try:
        # Set up OpenRouter with Claude
        original_base_url = os.environ.get("OPENAI_BASE_URL")
        original_api_key = os.environ.get("OPENAI_API_KEY")
        
        try:
            os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
            if OPENROUTER_API_KEY:
                os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
            else:
                logging.warning("OPENROUTER_API_KEY not found, using default accuracy scoring")
                return 50.0  # Default score if no judge available
            
            judge_model = "openai:anthropic/claude-3.5-sonnet"
            
            system_prompt = """You are an expert SQL judge. Your task is to evaluate how well a generated SQL query matches the intent and structure of a reference SQL query for a given natural language question.

Score from 0-100 based on:
- Semantic equivalence (60%): Does it achieve the same goal?
- Structural similarity (30%): Similar approach and logic?
- Syntax correctness (10%): Valid SQL syntax?

Return ONLY a number between 0-100."""

            user_prompt = f"""Natural Language Query: {query}

Reference SQL:
{reference_sql}

Generated SQL:
{generated_sql}

Score (0-100):"""

            response = await model_request(
                model=judge_model,
                messages=[
                    ModelRequest(
                        parts=[
                            SystemPromptPart(content=system_prompt),
                            UserPromptPart(content=user_prompt)
                        ]
                    )
                ],
                model_settings={"temperature": 0.1},
                model_request_parameters=ModelRequestParameters(allow_text_output=True)
            )
            
            # Extract score from response
            score_text = ""
            for part in response.parts:
                if hasattr(part, 'content'):
                    score_text += part.content
            
            # Parse score
            score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return min(100.0, max(0.0, score))
            else:
                logging.warning(f"Could not parse judge score: {score_text}")
                return 50.0
                
        finally:
            # Restore original settings
            if original_base_url is not None:
                os.environ["OPENAI_BASE_URL"] = original_base_url
            elif "OPENAI_BASE_URL" in os.environ:
                del os.environ["OPENAI_BASE_URL"]
            
            if original_api_key is not None:
                os.environ["OPENAI_API_KEY"] = original_api_key
                
    except Exception as e:
        logging.error(f"Judge error: {str(e)}")
        return 50.0  # Default score on error

@router.get("/template")
async def download_evaluation_template(format: str = "csv"):
    """Download evaluation template (CSV or Excel)"""
    try:
        if format == "csv":
            # Create CSV template
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["user_query", "true_sql"])
            writer.writerow(["Show all customers from 2024", "SELECT * FROM customers WHERE year = 2024"])
            writer.writerow(["List top 10 products by revenue", "SELECT product_name, revenue FROM products ORDER BY revenue DESC LIMIT 10"])
            writer.writerow(["Get total sales by month", ""])
            
            content = output.getvalue()
            return StreamingResponse(
                io.BytesIO(content.encode()),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=evaluation_template.csv"}
            )
        elif format == "excel":
            # For Excel, we'll return CSV with .xlsx extension
            # In production, you'd use openpyxl or xlsxwriter
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["user_query", "true_sql"])
            writer.writerow(["Show all customers from 2024", "SELECT * FROM customers WHERE year = 2024"])
            writer.writerow(["List top 10 products by revenue", "SELECT product_name, revenue FROM products ORDER BY revenue DESC LIMIT 10"])
            writer.writerow(["Get total sales by month", ""])
            
            content = output.getvalue()
            return StreamingResponse(
                io.BytesIO(content.encode()),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=evaluation_template.xlsx"}
            )
    except Exception as e:
        logging.error(f"Error generating template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_evaluation_file(file: UploadFile = File(...)):
    """Upload and parse evaluation file (CSV or Excel)"""
    try:
        content = await file.read()
        
        # Parse CSV
        if file.filename.endswith('.csv'):
            text_content = content.decode('utf-8')
            reader = csv.DictReader(io.StringIO(text_content))
            rows = list(reader)
        else:
            # For Excel files, try to parse as CSV (in production, use openpyxl)
            text_content = content.decode('utf-8')
            reader = csv.DictReader(io.StringIO(text_content))
            rows = list(reader)
        
        queries = []
        true_sqls = []
        
        for row in rows:
            query = row.get('user_query', '').strip()
            true_sql = row.get('true_sql', '').strip()
            
            if query:
                queries.append(query)
                # Only append true_sql if it's not empty, otherwise skip it
                if true_sql:
                    true_sqls.append(true_sql)
        
        return {
            "success": True,
            "queries": queries,
            "true_sqls": true_sqls,
            "count": len(queries)
        }
    except Exception as e:
        logging.error(f"Error parsing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")

@router.post("/test")
async def test_evaluation_request(request: EvaluationRequest):
    """Test endpoint to validate request structure"""
    return {
        "success": True,
        "received": {
            "queries_count": len(request.queries),
            "models_count": len(request.models),
            "database": request.database,
            "catalog": request.catalog,
            "has_true_sqls": request.true_sqls is not None and len(request.true_sqls) > 0,
            "baseline_model": request.baseline_model,
            "temperature": request.temperature
        }
    }

@router.post("/run")
async def run_evaluation(request: EvaluationRequest):
    """Run evaluation across multiple models"""
    try:
        logging.info(f"Received evaluation request: {len(request.queries)} queries, {len(request.models)} models")
        logging.info(f"Database: {request.database}, Catalog: {request.catalog}")
        logging.info(f"Models: {[m.value for m in request.models]}")
        logging.info(f"Temperature: {request.temperature}, Baseline: {request.baseline_model}")
        
        results = {
            "summary": [],
            "detailed": []
        }
        
        # Ensure true_sqls is a list and filter out None values
        true_sqls = [sql for sql in (request.true_sqls or []) if sql is not None and sql.strip()]
        
        # Process each query
        for query_idx, query in enumerate(request.queries):
            true_sql = true_sqls[query_idx] if query_idx < len(true_sqls) else None
            query_results = {
                "query": query,
                "true_sql": true_sql,
                "results": []
            }
            
            # Test each model
            baseline_sql = None
            for model_info in request.models:
                model_value = model_info.value
                provider = model_info.provider
                
                # Generate SQL with this model
                try:
                    gen_request = GenerateSQLRequest(
                        query=query,
                        model=model_value,
                        request_limit=request.request_limit,
                        iteration_limit=request.iteration_limit,
                        target_db=request.database,
                        catalog_db=request.database,
                        catalog_name=request.catalog,
                        temperature=request.temperature,
                        provider=provider
                    )
                    
                    gen_result = await generate_sql(gen_request)
                    
                    # Store baseline if this is the baseline model and no true SQL
                    if request.baseline_model and model_value == request.baseline_model and not true_sql:
                        baseline_sql = gen_result["sql"]
                    
                    # Calculate accuracy using judge model
                    reference_sql = true_sql if true_sql else baseline_sql
                    accuracy_score = 0.0
                    
                    if reference_sql and gen_result["sql"]:
                        accuracy_score = await judge_sql_accuracy(
                            query=query,
                            generated_sql=gen_result["sql"],
                            reference_sql=reference_sql
                        )
                    elif gen_result["success"]:
                        # If no reference but generation succeeded, give partial credit
                        accuracy_score = 50.0
                    
                    query_results["results"].append({
                        "model": model_value,
                        "provider": provider,
                        "success": gen_result["success"],
                        "generated_sql": gen_result["sql"],
                        "generation_time": gen_result["generation_time"],
                        "iterations": gen_result.get("actual_iterations", 1),
                        "accuracy_score": accuracy_score,
                        "error": None
                    })
                    
                except Exception as e:
                    logging.error(f"Error generating SQL for {model_value}: {str(e)}")
                    query_results["results"].append({
                        "model": model_value,
                        "provider": provider,
                        "success": False,
                        "generated_sql": None,
                        "generation_time": 0,
                        "iterations": 0,
                        "accuracy_score": 0.0,
                        "error": str(e)
                    })
            
            results["detailed"].append(query_results)
        
        # Calculate summary statistics
        model_stats = {}
        for model_info in request.models:
            model_value = model_info.value
            model_name = model_value.split(":")[-1] if ":" in model_value else model_value
            
            model_data = {
                "model_name": model_name,
                "provider": model_info.provider,
                "total_queries": len(request.queries),
                "successful_queries": 0,
                "total_time": 0,
                "total_iterations": 0,
                "total_accuracy": 0,
                "temperature": request.temperature
            }
            
            for query_result in results["detailed"]:
                for result in query_result["results"]:
                    if result["model"] == model_value:
                        if result["success"]:
                            model_data["successful_queries"] += 1
                        model_data["total_time"] += result["generation_time"]
                        model_data["total_iterations"] += result["iterations"]
                        model_data["total_accuracy"] += result["accuracy_score"]
            
            num_queries = len(request.queries)
            model_stats[model_value] = {
                "provider": model_data["provider"],
                "model_name": model_name,
                "accuracy": model_data["total_accuracy"] / num_queries if num_queries > 0 else 0,
                "avg_speed": model_data["total_time"] / num_queries if num_queries > 0 else 0,
                "total_time": model_data["total_time"],
                "avg_iterations": model_data["total_iterations"] / num_queries if num_queries > 0 else 0,
                "temperature": request.temperature,
                "success_rate": (model_data["successful_queries"] / num_queries * 100) if num_queries > 0 else 0
            }
        
        results["summary"] = list(model_stats.values())
        
        # Save results to folder
        try:
            save_evaluation_results(request, results)
        except Exception as save_error:
            logging.error(f"Failed to save evaluation results: {str(save_error)}")
        
        return results
        
    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/saved")
async def list_saved_evaluations():
    """List all saved evaluation results"""
    try:
        evaluations_dir = Path("evaluations")
        if not evaluations_dir.exists():
            return {"success": True, "evaluations": []}
        
        evaluations = []
        for eval_folder in sorted(evaluations_dir.iterdir(), reverse=True):
            if eval_folder.is_dir():
                metadata_file = eval_folder / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        evaluations.append({
                            "id": eval_folder.name,
                            "timestamp": metadata.get("timestamp"),
                            "database": metadata.get("database"),
                            "catalog": metadata.get("catalog"),
                            "models_count": len(metadata.get("models", [])),
                            "queries_count": metadata.get("queries_count")
                        })
        
        return {"success": True, "evaluations": evaluations}
    except Exception as e:
        logging.error(f"Error listing saved evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/saved/{evaluation_id}")
async def get_saved_evaluation(evaluation_id: str):
    """Retrieve a specific saved evaluation"""
    try:
        eval_folder = Path("evaluations") / evaluation_id
        if not eval_folder.exists():
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        results_file = eval_folder / "results.json"
        if not results_file.exists():
            raise HTTPException(status_code=404, detail="Results file not found")
        
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        return {"success": True, "results": results}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error retrieving saved evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/saved/{evaluation_id}/download/{file_type}")
async def download_saved_evaluation_file(evaluation_id: str, file_type: str):
    """Download a specific file from saved evaluation (summary.csv, report.txt, or results.json)"""
    try:
        eval_folder = Path("evaluations") / evaluation_id
        if not eval_folder.exists():
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        file_mapping = {
            "csv": ("summary.csv", "text/csv"),
            "report": ("report.txt", "text/plain"),
            "json": ("results.json", "application/json")
        }
        
        if file_type not in file_mapping:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        filename, content_type = file_mapping[file_type]
        file_path = eval_folder / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        
        with open(file_path, "rb") as f:
            content = f.read()
        
        return StreamingResponse(
            io.BytesIO(content),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error downloading evaluation file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
