# generate_descriptions.py
# Chunked runner for: pgai semantic-catalog describe
# - Runs per table (best) or per schema (fallback)
# - Retries transient errors
# - Writes stubs for ultra-wide tables that still hit OpenAI tool_calls limit
# - Merges all YAMLs into --final-out

import os, sys, argparse, subprocess, shlex, time, csv
from pathlib import Path

# ---------- Utilities ----------

def get_pgai_executable():
    """Find pgai executable, checking venv first."""
    # Check if we're in a venv
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        if sys.platform == 'win32':
            pgai_path = Path(venv_path) / 'Scripts' / 'pgai.exe'
        else:
            pgai_path = Path(venv_path) / 'bin' / 'pgai'
        if pgai_path.exists():
            return str(pgai_path)
    
    # Check relative to current Python executable
    python_dir = Path(sys.executable).parent
    if sys.platform == 'win32':
        pgai_path = python_dir / 'pgai.exe'
    else:
        pgai_path = python_dir / 'pgai'
    if pgai_path.exists():
        return str(pgai_path)
    
    # Fallback to system PATH
    return "pgai"

PGAI_CMD = get_pgai_executable()

def run(cmd:list, text=True, capture_output=True):
    p = subprocess.run(cmd, text=text, capture_output=capture_output)
    return p.returncode, (p.stdout or ""), (p.stderr or "")

def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)

def detect_pgai_flags():
    rc, out, err = run([PGAI_CMD,"semantic-catalog","describe","--help"])
    if rc != 0:
        die("Could not run 'pgai ... --help'. Ensure 'pgai' is installed and on PATH inside your venv.")
    help_text = out + "\n" + err
    flags = {
        "include_schema": ("--include-schema" in help_text) or ("--schemas" in help_text),
        "include_table":  ("--include-table"  in help_text),
        "batch_size":     ("--batch-size"    in help_text),
        "sample_size":    ("--sample-size"   in help_text),
        "dry_run":        ("--dry-run"       in help_text),
    }
    flags["schemas_kw"] = "--include-schema" if "--include-schema" in help_text else ("--schemas" if "--schemas" in help_text else None)
    return flags

def is_tool_calls_array_too_long(s:str)->bool:
    s = (s or "").lower()
    return ("array too long" in s) and ("tool" in s)

def load_primary_tables_from_csv(csv_path:Path, table_column:str="Table Name", primary_column:str="Is Primary"):
    """
    Load primary table names from CSV file.
    
    Args:
        csv_path: Path to CSV file
        table_column: Name of the column containing table names (default: "Table Name")
        primary_column: Name of the column indicating if table is primary (default: "Is Primary")
    
    Returns:
        Set of primary table names (lowercased)
    """
    if not csv_path.exists():
        die(f"Primary tables CSV file not found: {csv_path}")
    
    primary_tables = set()
    try:
        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Try to find the actual column names (case-insensitive)
            fieldnames = reader.fieldnames or []
            table_col = None
            primary_col = None
            
            for col in fieldnames:
                if col.strip().lower() == table_column.lower():
                    table_col = col
                if col.strip().lower() == primary_column.lower():
                    primary_col = col
            
            if not table_col:
                # Try common variations
                for col in fieldnames:
                    if 'table' in col.lower() and 'name' in col.lower():
                        table_col = col
                        break
                if not table_col:
                    die(f"Could not find table name column. Available columns: {', '.join(fieldnames)}")
            
            if not primary_col:
                # Try common variations
                for col in fieldnames:
                    if 'primary' in col.lower() or 'is_primary' in col.lower():
                        primary_col = col
                        break
                if not primary_col:
                    die(f"Could not find primary column. Available columns: {', '.join(fieldnames)}")
            
            for row in reader:
                table_name = row.get(table_col, '').strip()
                is_primary = row.get(primary_col, '').strip().upper()
                if table_name and is_primary == 'TRUE':
                    primary_tables.add(table_name.lower())
    except Exception as e:
        die(f"Error reading primary tables CSV: {e}")
    
    return primary_tables

# ---------- DB access ----------

def get_tables(db_url, include_schemas=None, exclude_schemas=None):
    try:
        import psycopg2
    except Exception:
        die("Missing dependency 'psycopg2-binary'. Install with: pip install psycopg2-binary")

    exclude_schemas = exclude_schemas or set()
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()

    parts = []
    params = []

    if include_schemas:
        parts.append("c.table_schema = ANY(%s)")
        params.append(include_schemas)

    if exclude_schemas:
        parts.append("c.table_schema <> ALL(%s)")
        params.append(list(exclude_schemas))

    where = ""
    if parts:
        where = "AND " + " AND ".join(parts)

    sql = f"""
        SELECT c.table_schema, c.table_name, COUNT(*) AS col_count
        FROM information_schema.columns c
        JOIN information_schema.tables t 
          ON c.table_schema = t.table_schema 
          AND c.table_name = t.table_name
        WHERE t.table_type='BASE TABLE' {where}
        GROUP BY 1,2
        ORDER BY c.table_schema, c.table_name;
    """

    cur.execute(sql, params if params else None)
    rows = cur.fetchall()
    cur.close(); conn.close()

    return [{"schema": r[0], "table": r[1], "cols": int(r[2])} for r in rows]

# ---------- YAML helpers ----------

def load_yaml_tables(path:Path):
    """Return set of 'schema.table' already present in YAML (best effort)."""
    if not path.exists():
        return set()
    try:
        import yaml
    except Exception:
        die("Missing dependency 'PyYAML'. Install with: pip install PyYAML")

    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Could not parse YAML {path}: {e}. Treating as empty.")
            return set()

    described = set()

    if isinstance(data, dict) and "tables" in data and isinstance(data["tables"], dict):
        for k in data["tables"].keys():
            described.add(str(k).lower())

    # Fallback: deep scan for {schema, table} dicts
    def walk(node):
        if isinstance(node, dict):
            if "schema" in node and "table" in node:
                described.add(f"{node['schema']}.{node['table']}".lower())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    if isinstance(data, (dict, list)):
        walk(data)

    return described

def deep_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = deep_merge(out[k], v) if k in out else v
        return out
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    return b

def merge_yaml_files(src_files, dest_file:Path):
    """Merge multi-document YAML files, keeping header and all table/view documents."""
    import yaml
    header = None
    all_objects = []  # Will store all table/view documents
    seen_objects = set()  # Track seen (schema, name) pairs to avoid duplicates
    
    for p in src_files:
        pth = Path(p)
        if not pth.exists():
            continue
        with pth.open("r", encoding="utf-8") as f:
            try:
                # Use safe_load_all to handle multi-document YAML files
                documents = list(yaml.safe_load_all(f))
                
                for doc in documents:
                    if not doc:  # Skip empty documents
                        continue
                    
                    doc_type = doc.get('type')
                    
                    # Capture header (only once)
                    if doc_type == 'header' and header is None:
                        header = doc
                    
                    # Collect table/view documents
                    elif doc_type in ('table', 'view'):
                        schema = doc.get('schema')
                        name = doc.get('name')
                        obj_id = (schema, name)
                        
                        # Avoid duplicates
                        if obj_id not in seen_objects:
                            all_objects.append(doc)
                            seen_objects.add(obj_id)
                
            except Exception as e:
                print(f"[WARN] Skipping unreadable YAML {pth}: {e}")
                continue
    
    # Write merged file
    with dest_file.open("w", encoding="utf-8") as f:
        # Write header first
        if header:
            yaml.safe_dump(header, f, allow_unicode=True, sort_keys=False, explicit_start=True, explicit_end=True)
        
        # Write all table/view documents
        for obj in all_objects:
            yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False, explicit_start=True, explicit_end=True)

def write_table_stub_yaml(path:Path, schema:str, table:str):
    import yaml
    stub = {
        "tables": {
            f"{schema}.{table}": {
                "schema": schema,
                "table": table,
                "description": f"{table.replace('_',' ').title()} table. (Auto-stubbed to bypass tool_calls limit; refine later.)"
            }
        }
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(stub, f, allow_unicode=True, sort_keys=False)

# ---------- pgai runners ----------

def run_describe_for_table(db_url, model, flags, schema, table, out_path, batch_size=None, sample_size=None):
    cmd = [PGAI_CMD,"semantic-catalog","describe","-d", db_url, "-f", str(out_path), "-m", model]
    if flags["include_schema"] and flags["schemas_kw"]:
        cmd += [flags["schemas_kw"], schema]
    if flags["include_table"]:
        cmd += ["--include-table", table]
    if flags["batch_size"] and batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if flags["sample_size"] and sample_size:
        cmd += ["--sample-size", str(sample_size)]
    print(">", " ".join(shlex.quote(c) for c in cmd))
    return run(cmd)

def run_describe_for_schema(db_url, model, flags, schema, out_path, batch_size=None, sample_size=None):
    cmd = [PGAI_CMD,"semantic-catalog","describe","-d", db_url, "-f", str(out_path), "-m", model]
    if flags["include_schema"] and flags["schemas_kw"]:
        cmd += [flags["schemas_kw"], schema]
    if flags["batch_size"] and batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if flags["sample_size"] and sample_size:
        cmd += ["--sample-size", str(sample_size)]
    print(">", " ".join(shlex.quote(c) for c in cmd))
    return run(cmd)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Resume/complete pgai semantic-catalog describe in safe chunks.")
    ap.add_argument("--db-url", required=True, help="Postgres URL (e.g., postgresql://user:pass@host:5432/db)")
    ap.add_argument("--base-out", default="descriptions.yaml", help="Existing/partial YAML file to seed from")
    ap.add_argument("--final-out", default="descriptions.final.yaml", help="Merged final YAML path")
    ap.add_argument("--model", default=os.getenv("PGAI_MODEL","openai:gpt-4o-mini"), help="Model id for pgai (e.g., openai:gpt-4o-mini)")
    ap.add_argument("--include-schemas", nargs="*", default=None, help="Only process these schemas (default: all non-system)")
    ap.add_argument("--exclude-schemas", nargs="*", default=["pg_catalog","information_schema"], help="Schemas to skip")
    ap.add_argument("--batch-size", type=int, default=1, help="pgai --batch-size if supported")
    ap.add_argument("--sample-size", type=int, default=1, help="pgai --sample-size if supported")
    ap.add_argument("--max-retries", type=int, default=2, help="Retries per chunk")
    ap.add_argument("--wide-threshold", type=int, default=120, help="Columns >= this are considered very wide")
    ap.add_argument("--chunks-dir", default="data/catalog/yaml_chunks", help="Where per-table YAMLs are written")
    ap.add_argument("--primary-tables-csv", default=None, help="CSV file with primary tables list (filters to only process primary tables)")
    ap.add_argument("--table-column", default="Table Name", help="Name of CSV column containing table names (default: 'Table Name')")
    ap.add_argument("--primary-column", default="Is Primary", help="Name of CSV column indicating primary tables (default: 'Is Primary')")
    ap.add_argument("--dry-run", action="store_true", help="Only print plan and exit")
    args = ap.parse_args()

    flags = detect_pgai_flags()
    print("Detected pgai flags:", flags)

    chunks_dir = Path(args.chunks_dir); chunks_dir.mkdir(exist_ok=True)

    # Load primary tables filter if CSV provided
    primary_tables = None
    if args.primary_tables_csv:
        primary_tables = load_primary_tables_from_csv(
            Path(args.primary_tables_csv),
            table_column=args.table_column,
            primary_column=args.primary_column
        )
        print(f"Loaded {len(primary_tables)} primary tables from CSV: {args.primary_tables_csv}")

    all_tables = get_tables(args.db_url, include_schemas=args.include_schemas, exclude_schemas=set(args.exclude_schemas))
    
    # Filter for primary tables if specified
    if primary_tables:
        all_tables = [t for t in all_tables if t['table'].lower() in primary_tables]
        print(f"Filtered to {len(all_tables)} primary tables")
    
    base_out = Path(args.base_out)
    already = load_yaml_tables(base_out)

    print(f"Tables found: {len(all_tables)}")
    print(f"Already in YAML: {len(already)}")

    work = [t for t in all_tables if f"{t['schema']}.{t['table']}".lower() not in already]
    print(f"Remaining to process: {len(work)}")

    if args.dry_run:
        for t in work[:20]:
            print(f" - {t['schema']}.{t['table']} (cols={t['cols']})")
        if len(work) > 20:
            print(f"... and {len(work)-20} more")
        sys.exit(0)

    chunk_outputs = []
    if base_out.exists():
        chunk_outputs.append(base_out)

    # Best path: per-table
    if flags["include_table"]:
        for i, t in enumerate(work, 1):
            schema, table, cols = t["schema"], t["table"], t["cols"]
            out_path = chunks_dir / f"{schema}.{table}.yaml"
            if out_path.exists():
                chunk_outputs.append(out_path)
                continue

            print(f"[{i}/{len(work)}] {schema}.{table} (cols={cols})")
            hit_array_cap = False

            for attempt in range(1, args.max_retries+1):
                rc, out, err = run_describe_for_table(
                    args.db_url, args.model, flags, schema, table, out_path,
                    batch_size=args.batch_size, sample_size=args.sample_size
                )
                if rc == 0:
                    chunk_outputs.append(out_path)
                    break

                body = (out or "") + "\n" + (err or "")
                if is_tool_calls_array_too_long(body):
                    print("   -> Hit OpenAI 128 tool_calls cap on this table.")
                    hit_array_cap = True
                    if attempt == args.max_retries:
                        print("   -> Writing stub YAML for table to move on.")
                        write_table_stub_yaml(out_path, schema, table)
                        chunk_outputs.append(out_path)
                        break
                else:
                    print(f"   -> Error (attempt {attempt}/{args.max_retries}). Retrying...\n{body}")
                    time.sleep(2.0)

            time.sleep(0.5)  # gentle pacing

    else:
        # Fallback: per-schema
        schemas = sorted(set(t["schema"] for t in work))
        for i, schema in enumerate(schemas, 1):
            out_path = chunks_dir / f"{schema}.yaml"
            if out_path.exists():
                chunk_outputs.append(out_path)
                continue

            print(f"[{i}/{len(schemas)}] Schema {schema}")
            for attempt in range(1, args.max_retries+1):
                rc, out, err = run_describe_for_schema(
                    args.db_url, args.model, flags, schema, out_path,
                    batch_size=args.batch_size, sample_size=args.sample_size
                )
                if rc == 0:
                    chunk_outputs.append(out_path)
                    break

                body = (out or "") + "\n" + (err or "")
                if is_tool_calls_array_too_long(body):
                    print("   -> Still hit 128 tool_calls cap even per schema. Consider upgrading pgai or switching model.")
                    # No smaller unit to split into without --include-table
                    break
                else:
                    print(f"   -> Error (attempt {attempt}/{args.max_retries}). Retrying...\n{body}")
                    time.sleep(2.0)

    # Merge
    final_out = Path(args.final_out)
    print("\nMerging YAMLs into:", final_out)
    merge_yaml_files(chunk_outputs, final_out)
    
    # Count what was merged
    import yaml
    with final_out.open('r', encoding='utf-8') as f:
        docs = list(yaml.safe_load_all(f))
    tables = [d for d in docs if d and d.get('type') == 'table']
    views = [d for d in docs if d and d.get('type') == 'view']
    
    print("\n" + "="*60)
    print("✅ Done!")
    print("="*60)
    print(f"Merged file: {final_out.resolve()}")
    print(f"Source files: {len(chunk_outputs)}")
    print(f"Tables described: {len(tables)}")
    print(f"Views included: {len(views)}")
    print("="*60)

if __name__ == "__main__":
    main()
