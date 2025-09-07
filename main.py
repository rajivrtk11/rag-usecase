#!/usr/bin/env python3

import os
import sys
import asyncio
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_system import RAGSystem
# Import the app instance directly instead of create_app function
from src.api import app

async def main():
    """Main entry point for the RAG system."""
    if len(sys.argv) < 2:
        print("Usage: python main.py [initialize|serve|query]")
        return

    command = sys.argv[1]
    
    if command == "initialize":
        # Initialize the RAG system
        rag_system = RAGSystem()
        success = await rag_system.initialize()
        if success:
            print("RAG system initialized successfully!")
        else:
            print("Failed to initialize RAG system!")
            return 1
            
    elif command == "serve":
        # Start the API server
        # Import uvicorn here to avoid issues when not needed
        import uvicorn
        uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
        
    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python main.py query <query_text>")
            return
            
        query_text = " ".join(sys.argv[2:])
        rag_system = RAGSystem()
        await rag_system.initialize()
        
        results = await rag_system.query(query_text)
        print("Query Results:")
        print(results)
        
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [initialize|serve|query]")

if __name__ == "__main__":
    asyncio.run(main())