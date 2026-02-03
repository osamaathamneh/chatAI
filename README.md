# LLMonDBPGAI Backend

A backend API for semantic SQL generation using PostgreSQL AI (pgai). This project enables natural language to SQL conversion by leveraging semantic catalogs, vector embeddings, and large language models.

## What is This Project?

This project provides a REST API backend that uses pgai (PostgreSQL AI) to:
- Generate SQL queries from natural language prompts
- Search database schemas semantically
- Manage semantic catalogs with vector embeddings
- Support multiple LLM providers (OpenAI, OpenRouter, Ollama)
  - OpenRouter includes GPT OSS 20B/120B, GPT-4o, GPT-4o-mini, GPT-4-turbo, Claude 3 Opus/Sonnet
- Template-based SQL generation with intelligent parameter binding
- Query history tracking and evaluation tools

The system works by:
1. Creating a semantic catalog with embedding configurations
2. Generating natural language descriptions of database objects
3. Importing these descriptions into the catalog with vector embeddings
4. Using semantic search to find relevant schema information
5. Generating SQL queries using LLMs with context from the semantic catalog

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- pgai installed. to install directly: (`pip install pgai[vectorizer-worker,semantic-catalog]`) or with a requirements.txt later.
- (Optional) Ollama for local embeddings

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important:** If you encounter issues with SQL generation, you may need to reinstall pydantic-ai:

```bash
pip uninstall -y pydantic-ai
pip install pydantic-ai==1.0.10
```

### 4. Verify pgai Installation

```bash
pgai --version
```

### 5. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your configuration:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
- Database connection URLs
- API keys (OpenAI, OpenRouter)
- Catalog name
- Model preferences

### 6. (Optional) Build Template Vector Store

**If you plan to use template matching features**, build the vector store before starting the app:

```bash
python src/scripts/build_template_vector_store.py
```

**Prerequisites:**
- Ollama must be running (`ollama serve`)
- `data/templates/templates_with_prompt_fewshot.csv` must exist

**Note:** This step is optional but recommended. The app will build it automatically if missing, but running it first provides better error handling and faster startup. See [Build Template Vector Store](#build-template-vector-store-recommended-before-first-run) for details.

## Semantic Catalog Setup

### Using OpenAI

1. **Set up environment variables** in `.env`:
   ```
   TARGET_DB=postgresql://postgres:password@localhost:5432/your_database
   CATALOG_DB=postgresql://postgres:password@localhost:5432/your_database
   OPENAI_API_KEY=your-secret-key
   CATALOG_NAME=production_test
   DEFAULT_MODEL=openai:gpt-5
   ```

2. **Create the semantic catalog:**
   ```bash
   pgai semantic-catalog create \
     --catalog-db-url postgresql://postgres:password@localhost:5432/your_database \
     --catalog-name "production_test" \
     --embed-config "production_test_embeddings" \
     --model text-embedding-3-small
   ```

   **Windows Event Loop Fix:** If you encounter this error on Windows:
   ```
   psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async mode
   ```
   
   Add this code to `venv\Lib\site-packages\pgai\cli.py`:
   ```python
   if sys.platform.startswith("win"):
       asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
   ```

3. **Example of expected output:**
   ```
   created "production_test" semantic catalog with id: 4
   ```

### Using Ollama

1. **Ensure Ollama is running** on `http://localhost:11434`

2. **Create the semantic catalog:**
   ```bash
   pgai semantic-catalog create \
     --catalog-name production_test \
     --embed-config production_test_embeddings \
     --base-url "http://localhost:11434/" \
     --provider "ollama" \
     --model "hf.co/bartowski/granite-embedding-30m-english-GGUF:Q4_K_M" \
     --vector-dimensions "384"
   ```

## Database Description Generation

You have two options for generating database descriptions: using the `pgai` command directly or using the custom Python script.

### Using pgai Command

**When to use:** Small to medium databases, quick one-time runs, simple filtering with regex patterns.

**Command:**
```bash
pgai semantic-catalog describe \
  -f descriptions.yaml \
  -m openai:gpt-4o-mini
```

**Options:**
- `-d, --db-url`: Database connection URL
- `-m, --model`: LLM model (format: `provider:model`)
- `-f, --yaml-file`: Output YAML file path
- `--include-schema`: Regex pattern to include schemas
- `--exclude-schema`: Regex pattern to exclude schemas
- `--include-table`: Regex pattern to include tables
- `--exclude-table`: Regex pattern to exclude tables
- `--sample-size`: Number of sample rows to retrieve
- `--batch-size`: Number of objects to process per LLM request
- `--dry-run`: List objects without describing them

**Example:**
```bash
pgai semantic-catalog describe \
  -f descriptions.yaml \
  -m openai:gpt-4o-mini \
  --include-schema "public" \
  --exclude-table "temp_.*"
```

### Using generate_descriptions.py Script

**When to use:** Large databases, need CSV-based filtering, want resume capability, need automatic retries, have very wide tables.

**Command:**
```bash
python src/scripts/generate_descriptions.py \
  --db-url "postgresql://postgres:password@localhost:5432/your_database" \
  --primary-tables-csv "data/catalog/primary_tables.csv" \
  --model "openai:gpt-4o-mini" \
  --chunks-dir "data/catalog/yaml_chunks" \
  --final-out "data/catalog/descriptions_final.yaml"
```

**Options:**
- `--db-url`: Database connection URL (required)
- `--primary-tables-csv`: CSV file with primary tables list
- `--table-column`: Name of CSV column containing table names (default: "Table Name")
- `--primary-column`: Name of CSV column indicating primary tables (default: "Is Primary")
- `--model`: LLM model (default: from `PGAI_MODEL` env var or `openai:gpt-4o-mini`)
- `--base-out`: Existing/partial YAML file to seed from (default: `descriptions.yaml`)
- `--final-out`: Merged final YAML path (default: `descriptions.final.yaml`)
- `--chunks-dir`: Directory for per-table YAMLs (default: `data/catalog/yaml_chunks`)
- `--max-retries`: Retries per chunk (default: 2)
- `--dry-run`: Only print plan and exit

**CSV Format:**
The CSV file should have columns for table names and a boolean indicating if the table is primary:
```csv
Table Name,Is Primary
users,TRUE
orders,TRUE
temp_data,FALSE
```

### Comparison Table

| Feature | pgai semantic-catalog describe Command | generate_descriptions.py Script |
|---------|----------------------------------------|----------------------------------|
| CSV file support | ❌ No — only regex patterns | ✅ Yes — `--primary-tables-csv` |
| Processing strategy | All objects in one run | One table at a time (chunked) |
| Resume capability | ❌ No — must restart | ✅ Yes — skips already processed tables |
| Error handling | ❌ Fails completely on error | ✅ Retries with `--max-retries` |
| Tool calls limit handling | ❌ May fail on wide tables | ✅ Writes stubs for problematic tables |
| Output management | Single file | Per-table chunks + merged final file |
| Progress tracking | Basic | Shows `[X/Y] table_name` progress |
| Cost control | `--request-limit`, `--total-tokens-limit` | Per-table processing reduces risk |
| Filtering | Regex only (`--include-table`, etc.) | CSV-based + regex schemas |
| Dry run | ✅ `--dry-run` | ✅ `--dry-run` |
| Batch processing | ✅ `--batch-size` | ✅ Passes through to command |
| Sample rows | ✅ `--sample-size` | ✅ Passes through to command |
| Logging | ✅ `--log-file`, `--log-level` | Basic print statements |
| Append mode | ✅ `--append` | ❌ No (uses merge instead) |
| Wide table detection | ❌ No | ✅ Detects and handles wide tables |
| YAML merging | ❌ No | ✅ Merges multiple YAML files |
| Dependency | Just pgai | pgai + psycopg2-binary + PyYAML |

## Importing Data into Catalog

After generating descriptions, import them into the semantic catalog:

```bash
pgai semantic-catalog import \
  -f descriptions.yaml \
  --catalog-name production_test
```

**Options:**
- `-f, --yaml-file`: Path to YAML file with descriptions
- `-n, --catalog-name`: Name of the semantic catalog
- `-c, --catalog-db-url`: Database URL (if different from default)
- `-e, --embed-config`: Specific embedding configuration to use
- `-b, --batch-size`: Number of embeddings per batch

**What it does:**
- Reads catalog items (tables, views, procedures) from the YAML file
- Imports them into the semantic catalog
- Generates vector embeddings for semantic search

**Example:**
```bash
pgai semantic-catalog import \
  -f descriptions_final.yaml \
  --catalog-name production_test \
  --embed-config production_test_embeddings
```

## Searching the Catalog

Search the semantic catalog using natural language:

```bash
pgai semantic-catalog search \
  -p "Get all prepaid activations from 2025-07-31 to 2025-08-31" \
  --catalog-name production_test
```

**Options:**
- `-p, --prompt`: Natural language query (required)
- `-n, --catalog-name`: Name of the semantic catalog
- `-c, --catalog-db-url`: Database URL (if different from default)
- `-e, --embed-config`: Embedding configuration to use
- `-s, --sample-size`: Number of sample rows per table/view (default: 3)
- `--render`: Render the prompts for matches

**What it does:**
- Performs semantic search across database objects, SQL examples, and facts
- Ranks results by semantic similarity
- Displays schema information and sample data for matching objects

**Example with rendering:**
```bash
pgai semantic-catalog search \
  -p "How are orders related to customers?" \
  --catalog-name production_test \
  --render \
  --sample-size 5
```

## Generating SQL

Generate SQL queries from natural language:

```bash
pgai semantic-catalog generate-sql \
  --prompt "Get all prepaid activations from 2025-07-31 to 2025-08-31" \
  --model openai:gpt-4o-mini \
  --catalog-name production_test \
  --iteration-limit 15
```

**Options:**
- `-p, --prompt`: Natural language description of the SQL query (required)
- `-m, --model`: LLM model (format: `provider:model`, default: `openai:gpt-4.1`)
- `-n, --catalog-name`: Name of the semantic catalog (default: `default`)
- `-d, --db-url`: Target database URL
- `-c, --catalog-db-url`: Catalog database URL
- `-e, --embed-config`: Embedding configuration to use
- `--iteration-limit`: Maximum refinement attempts (default: 5)
- `-s, --sample-size`: Sample rows per table/view (default: 3)
- `--request-limit`: Maximum LLM requests allowed
- `--total-tokens-limit`: Maximum total tokens allowed
- `--print-messages`: Print LLM messages
- `--print-usage`: Print token usage metrics
- `--save-final-prompt`: Save final prompt to file

**What it does:**
- Uses AI to generate SQL based on the natural language prompt
- Leverages semantic catalog context (schema, examples, facts)
- Validates SQL against the target database
- Iteratively refines the query if needed

**Examples:**
```bash
# Simple query
pgai semantic-catalog generate-sql \
  --prompt "Find all users who signed up last month" \
  --catalog-name production_test

# With specific model and iteration limit
pgai semantic-catalog generate-sql \
  --prompt "Count orders by product category for Q1 2023" \
  --model openai:gpt-5 \
  --catalog-name production_test \
  --iteration-limit 10

# Save prompt for debugging
pgai semantic-catalog generate-sql \
  --prompt "Find inactive customers" \
  --catalog-name production_test \
  --save-final-prompt debug_prompt.txt
```

## Running the Application

### Prerequisites Before Starting

**Before starting the app for the first time, ensure:**

1. ✅ Environment variables are configured (`.env` file exists)
2. ✅ Semantic catalog is created and populated (see [Semantic Catalog Setup](#semantic-catalog-setup))
3. ✅ **If using template matching:** Vector store is built (see [Build Template Vector Store](#build-template-vector-store-recommended-before-first-run))

### Start the API Server

```bash
python run.py
```

Or directly:

```bash
python app.py
```

The server will start on `http://localhost:8000`

### API Endpoints

**Core Endpoints:**
- `GET /api/health` - Health check
- `POST /api/search` - Semantic search in catalog
- `POST /api/generate-sql` - Generate SQL from natural language
- `POST /api/smart-generate-sql` - Smart SQL generation with template matching
- `POST /api/execute-sql` - Execute SQL query and return results

**Model Management:**
- `GET /api/models/ollama` - List available Ollama models
- `GET /api/models/openai` - List available OpenAI models
- `GET /api/models/openrouter` - List available OpenRouter models

**Catalog & Database:**
- `GET /api/catalogs` - List semantic catalogs
- `GET /api/databases` - List available databases

**History & Templates:**
- `GET /api/history` - Get query history
- `DELETE /api/history/{id}` - Delete specific history item
- `DELETE /api/history` - Clear all history
- `GET /api/examples` - Get example queries
- `GET /api/templates` - Get all query templates

**Feedback:**
- `POST /api/feedback/like` - Add template with few-shot examples
- `POST /api/feedback/dislike` - Handle dislike feedback
- `POST /api/feedback/regenerate-sql` - Regenerate SQL with higher iteration limit

**Evaluation:**
- `GET /api/evaluation/template` - Download evaluation template (CSV/Excel)
- `POST /api/evaluation/upload` - Upload evaluation file
- `POST /api/evaluation/run` - Run model evaluation
- `GET /api/evaluation/saved` - List saved evaluations
- `GET /api/evaluation/saved/{id}` - Get specific evaluation
- `GET /api/evaluation/saved/{id}/download/{type}` - Download evaluation files

**Configuration:**
- `GET /api/config` - Get configuration options
- `GET /api/debug/openai-models` - Debug OpenAI models

See the interactive API documentation at `http://localhost:8000/docs` for full details, request/response schemas, and testing capabilities.

### Build Template Vector Store (Recommended Before First Run)

**⚠️ Important:** If you plan to use template matching features (smart SQL generation), run this script **before starting the app for the first time**.

```bash
python src/scripts/build_template_vector_store.py
```

**Why run it first:**
- ✅ **Faster app startup** - Avoids building the vector store during app initialization
- ✅ **Better error handling** - Validates Ollama setup and embedding model availability before starting the server
- ✅ **Clearer feedback** - Provides detailed logging about the build process
- ✅ **One-time setup** - Build once, use many times (only rebuild when templates change)

**What it does:**
- Reads `data/templates/templates_with_prompt_fewshot.csv`
- Generates embeddings for all templates using Ollama
- Creates `data/templates/template_vectors.faiss` (FAISS index)
- Creates `data/templates/template_vectors_metadata.json` (metadata)

**Prerequisites:**
- Ollama must be running (`ollama serve`)
- Embedding model must be available (script will try to pull it automatically if missing)
- `data/templates/templates_with_prompt_fewshot.csv` must exist

**When to rebuild:**
- After updating templates in the CSV file
- If you delete the vector store files manually

**Note:** The app will automatically build the vector store if missing, but running the script first is **strongly recommended** for better performance and error handling. If Ollama isn't running or the model is missing, the app startup may fail or template matching will be disabled.

## Architecture Diagram

```mermaid
graph TB
    subgraph "Setup Phase"
        DB[(PostgreSQL Database)]
        PGAI[pgai CLI]
        YAML[YAML Descriptions]
        CATALOG[Semantic Catalog]
        
        DB -->|1. Describe Schema| PGAI
        PGAI -->|2. Generate| YAML
        YAML -->|3. Import| CATALOG
        CATALOG -->|4. Create Embeddings| VECTORS[Vector Embeddings]
    end
    
    subgraph "Runtime Phase"
        USER[User Query]
        API[FastAPI Backend]
        SEARCH[Semantic Search]
        LLM[LLM Provider]
        SQL[Generated SQL]
        RESULT[Query Results]
        
        USER -->|Natural Language| API
        API -->|Search Catalog| SEARCH
        SEARCH -->|Context| LLM
        LLM -->|Generate| SQL
        SQL -->|Execute| DB
        DB -->|Return| RESULT
        RESULT -->|Response| USER
    end
    
    CATALOG -.->|Provides Context| SEARCH
    VECTORS -.->|Enables| SEARCH
```

**Visual Diagrams:**

Additional detailed diagrams are available in the `diagrams/images/` folder:
- **PGAI Diagram** - Overview of the PostgreSQL AI integration
- **Matching Template Approach** - Template matching workflow
- **Flow SQL Generation LLM Matching** - Complete SQL generation flow with LLM matching

The source DrawIO file is available in `diagrams/diagrams/Diagrams.drawio` (not tracked in git).

## Project Structure

```
LLMonDBPGAI_Backend/
├── src/
│   ├── api/
│   │   ├── config.py              # Configuration and environment variables
│   │   ├── dependencies.py        # Shared dependencies and utilities
│   │   └── routes/                 # API route handlers
│   │       ├── semantic_search.py  # Search endpoints
│   │       ├── sql_generation.py   # SQL generation endpoints
│   │       ├── feedback.py         # Feedback endpoints
│   │       ├── catalog_management.py # Catalog management endpoints
│   │       ├── models.py           # Model listing endpoints
│   │       ├── history.py          # Query history endpoints
│   │       ├── templates.py        # Template endpoints
│   │       ├── config.py           # Configuration endpoints
│   │       ├── evaluation.py      # Evaluation endpoints
│   │       └── models_schemas.py   # Pydantic request/response models
│   ├── services/
│   │   ├── database.py            # Database initialization
│   │   ├── template_matcher.py    # Static template matcher
│   │   └── llm_template_matcher.py # LLM-based template matcher
│   ├── utils/
│   │   └── providers.py           # Provider configuration utilities
│   └── scripts/
│       ├── generate_descriptions.py # Database description generator
│       └── build_template_vector_store.py # Vector store builder
├── app.py                         # FastAPI application (main entry point)
├── run.py                         # Application entry point
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── data/                          # Data files directory (not in git)
│   ├── templates/                 # SQL query templates & vector store
│   │   ├── templates_with_prompt_fewshot.csv.template  # Template file (in git)
│   │   ├── templates_with_prompt_fewshot.csv           # Your data (not in git)
│   │   ├── template_vectors.faiss                      # Generated (not in git)
│   │   └── template_vectors_metadata.json              # Generated (not in git)
│   └── catalog/                   # Database descriptions & metadata
│       ├── primary_tables.csv.template                 # Template file (in git)
│       ├── primary_tables.csv                         # Your data (not in git)
│       ├── descriptions*.yaml                         # Generated (not in git)
│       └── yaml_chunks/                               # Generated (not in git)
└── README.md                      # This file
```

## Data Files Setup

The project uses data files for catalog management and template matching. These files are **not included in the repository** to keep it clean, but template files are provided to help you get started.

### Required Data Files

1. **`data/catalog/primary_tables.csv`** - Defines which database tables are considered "primary" for description generation
2. **`data/templates/templates_with_prompt_fewshot.csv`** - Contains SQL query templates for template-based SQL generation

### Initial Setup

1. **Create Primary Tables CSV:**
   ```bash
   # Copy the template
   cp data/catalog/primary_tables.csv.template data/catalog/primary_tables.csv
   
   # Edit the file and add your table names
   # Format: Table Name,Is Primary
   # Example:
   # users,TRUE
   # orders,TRUE
   # temp_data,FALSE
   ```

2. **Create Templates CSV:**
   ```bash
   # Copy the template
   cp data/templates/templates_with_prompt_fewshot.csv.template data/templates/templates_with_prompt_fewshot.csv
   
   # Edit the file and add your query templates
   # Format: user_query,true_sql,few_shot_example_1,few_shot_example_2,feedback
   # Example:
   # Get all users,SELECT * FROM users;,"User Input: Show all users
   # Expected Output: {...}","User Input: List users
   # Expected Output: {...}",
   ```

### Generating Data Files

#### Generate Catalog Descriptions

After setting up `primary_tables.csv`, generate database descriptions:

```bash
python src/scripts/generate_descriptions.py \
  --db-url "postgresql://postgres:password@localhost:5432/your_database" \
  --primary-tables-csv "data/catalog/primary_tables.csv" \
  --model "openai:gpt-4o-mini" \
  --chunks-dir "data/catalog/yaml_chunks" \
  --final-out "data/catalog/descriptions_final.yaml"
```

**What this does:**
- Reads your `primary_tables.csv` to filter which tables to process
- Generates YAML descriptions for each table using the LLM
- Saves per-table chunks in `data/catalog/yaml_chunks/`
- Merges all chunks into `data/catalog/descriptions_final.yaml`

**Note:** This script handles large databases, retries on errors, and can resume if interrupted.

#### Build Template Vector Store

**⚠️ Run this before starting the app for the first time!**

After setting up `templates_with_prompt_fewshot.csv`, build the FAISS vector store for fast template matching:

```bash
python src/scripts/build_template_vector_store.py
```

**What this does:**
- Reads `data/templates/templates_with_prompt_fewshot.csv`
- Generates embeddings for all templates using Ollama
- Creates `data/templates/template_vectors.faiss` (FAISS index)
- Creates `data/templates/template_vectors_metadata.json` (metadata)

**Prerequisites:**
- Ollama must be running (`ollama serve`)
- Embedding model must be available (script will try to pull it automatically if missing)

**Why run it first:**
- Faster app startup (avoids building during initialization)
- Better error messages if Ollama/model is not configured
- Validates your setup before starting the API server
- One-time setup - build once, use many times

**When to rebuild:**
- After updating templates in the CSV file
- If you delete the vector store files manually

**Note:** The app will automatically build the vector store if missing, but running the script first is **strongly recommended** for better performance and error handling. If Ollama isn't running or the model is missing, the app startup may fail or template matching will be disabled.

## Troubleshooting

### Windows Event Loop Error

If you see `psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop'`, add this to `venv\Lib\site-packages\pgai\cli.py`:

```python
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

### pydantic-ai Version Issues

If SQL generation fails with iteration limit errors, reinstall pydantic-ai:

```bash
pip uninstall -y pydantic-ai
pip install pydantic-ai==1.0.10
```

### Wide Tables Hitting Tool Calls Limit

For tables with 100+ columns, use `generate_descriptions.py` script which handles this automatically by writing stubs for problematic tables.
