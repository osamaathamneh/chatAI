"""
Main FastAPI application.
"""
import os
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Windows event loop policy for psycopg compatibility
if os.name == 'nt':  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize database
from src.services.database import init_db
init_db()

# Create FastAPI app
app = FastAPI(title="Semantic SQL Generation API")

# Custom validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logging.error(f"Validation error: {errors}")
    try:
        body = await request.body()
        logging.error(f"Request body: {body}")
    except:
        pass
    return JSONResponse(
        status_code=422,
        content={
            "detail": f"Validation error: {errors}",
            "errors": errors
        }
    )

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from src.api.routes import (
    semantic_search,
    sql_generation,
    feedback,
    catalog_management,
    models,
    history,
    templates,
    config,
    evaluation
)

app.include_router(semantic_search.router)
app.include_router(sql_generation.router)
app.include_router(feedback.router)
app.include_router(catalog_management.router)
app.include_router(models.router)
app.include_router(history.router)
app.include_router(templates.router)
app.include_router(config.router)
app.include_router(evaluation.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
