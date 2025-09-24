#!/usr/bin/env python3
"""
Simple script to start the FastAPI server with proper configuration
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Start the FastAPI server"""
    # Check if OpenAI API key is configured
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Please set it in your .env file or environment")
        print("   The server will start but API calls will fail without it\n")

    print("🚀 Starting SignGPT Crew Server...")
    print("📝 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Interactive API explorer at: http://localhost:8000/redoc")
    print("❤️  Health check at: http://localhost:8000/health")
    print("\n" + "="*50)

    # Start the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )


if __name__ == "__main__":
    main()
