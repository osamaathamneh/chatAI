#!/usr/bin/env python3
"""
Startup script for Semantic SQL Generation API
"""
import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists"""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ Error: .env file not found!")
        print("\n📝 Please create a .env file with the following content:")
        print("""
OPENAI_API_KEY=your-openai-api-key-here
TARGET_DB=postgresql://postgres:password@localhost:5432/salespoint
CATALOG_DB=postgresql://postgres:password@localhost:5432/salespoint
CATALOG_NAME=salespoint_production
DEFAULT_MODEL=openai:gpt-4o-mini
        """)
        return False
    return True

def main():
    print("🔍 Semantic SQL Generation API")
    print("=" * 50)
    
    # Check prerequisites
    if not check_env_file():
        sys.exit(1)
    
    print("✅ Configuration files found")
    print("\n🚀 Starting API server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📍 API docs will be available at: http://localhost:8000/docs")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    # Start the server
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

