"""
Main application entry point for the Vector-Enhanced Temporal Knowledge Graph System.
"""

import asyncio
import logging
import sys
from pathlib import Path

import click
import uvicorn
from rich.console import Console
from rich.logging import RichHandler

from .config import get_config
from .api import create_app
from .vector_store import initialize_vector_store
from .embeddings import embedding_manager


# Setup rich console for beautiful output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Vector-Enhanced Temporal Knowledge Graph System CLI."""
    pass


@cli.command()
@click.option('--host', default=None, help='Host to bind the server to')
@click.option('--port', default=None, type=int, help='Port to bind the server to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--workers', default=None, type=int, help='Number of worker processes')
def serve(host, port, reload, workers):
    """Start the API server."""
    config = get_config()
    
    # Override config with CLI options
    server_host = host or config.api.host
    server_port = port or config.api.port
    server_reload = reload or config.api.reload
    server_workers = workers or config.api.workers
    
    console.print(f"🚀 Starting server on {server_host}:{server_port}")
    
    # Create the FastAPI app
    app = create_app()
    
    # Run the server
    uvicorn.run(
        app,
        host=server_host,
        port=server_port,
        reload=server_reload,
        workers=server_workers if not server_reload else 1,  # Workers don't work with reload
        log_level=config.logging.level.lower()
    )


@cli.command()
async def initialize():
    """Initialize the system components."""
    console.print("🔧 Initializing system components...")
    
    try:
        # Initialize embedding manager
        console.print("📊 Initializing embedding providers...")
        embedding_success = await embedding_manager.initialize()
        
        if embedding_success:
            console.print("✅ Embedding providers initialized successfully")
            available_providers = embedding_manager.get_available_providers()
            console.print(f"Available providers: {[p.value for p in available_providers]}")
        else:
            console.print("❌ Failed to initialize embedding providers")
            return
        
        # Initialize vector store
        console.print("🗂️  Initializing vector store...")
        vector_success = await initialize_vector_store()
        
        if vector_success:
            console.print("✅ Vector store initialized successfully")
            collections = await vector_store.list_collections()
            console.print(f"Available collections: {[c.name for c in collections]}")
        else:
            console.print("❌ Failed to initialize vector store")
            return
        
        console.print("🎉 System initialization completed successfully!")
        
    except Exception as e:
        console.print(f"💥 System initialization failed: {e}")
        logger.exception("Full error details:")


@cli.command()
@click.option('--query', prompt='Enter your search query', help='Search query')
@click.option('--method', type=click.Choice(['semantic', 'hybrid', 'temporal']), default='hybrid', help='Search method')
@click.option('--max-results', default=10, help='Maximum number of results')
async def search(query, method, max_results):
    """Perform a search query."""
    console.print(f"🔍 Searching: '{query}' using {method} method")
    
    try:
        # Import here to avoid issues if system not initialized
        from .models import SearchQuery, QueryContext, SearchMethod
        
        # Create search query
        search_method = SearchMethod(method.upper())
        context = QueryContext(max_results=max_results)
        search_query = SearchQuery(
            query=query,
            search_method=search_method,
            context=context
        )
        
        # Perform search based on method
        if method == 'semantic':
            from .semantic_search import perform_semantic_search
            response = await perform_semantic_search(search_query)
        elif method == 'hybrid':
            from .hybrid_search import perform_hybrid_search  
            response = await perform_hybrid_search(search_query)
        elif method == 'temporal':
            from .temporal_search import perform_temporal_search
            results = await perform_temporal_search(search_query)
            # Convert to response-like format
            response = type('Response', (), {
                'results': results,
                'total_results': len(results),
                'search_time_ms': 0
            })()
        
        # Display results
        console.print(f"\n📋 Found {response.total_results} results in {getattr(response, 'search_time_ms', 0):.2f}ms")
        
        for i, result in enumerate(response.results[:max_results], 1):
            console.print(f"\n{i}. [bold]Score: {result.similarity_score:.3f}[/bold]")
            console.print(f"   Content: {result.content[:200]}{'...' if len(result.content) > 200 else ''}")
            console.print(f"   Domain: {result.metadata.domain.value}")
            console.print(f"   Type: {result.metadata.content_type.value}")
        
    except Exception as e:
        console.print(f"💥 Search failed: {e}")
        logger.exception("Full error details:")


@cli.command()
async def stats():
    """Show system statistics."""
    console.print("📊 System Statistics")
    
    try:
        # Embedding stats
        embedding_stats = embedding_manager.get_stats()
        console.print(f"\n🔤 Embedding Manager:")
        console.print(f"  Total Requests: {embedding_stats.get('requests_total', 0)}")
        console.print(f"  Successful: {embedding_stats.get('requests_successful', 0)}")
        console.print(f"  Failed: {embedding_stats.get('requests_failed', 0)}")
        console.print(f"  Success Rate: {embedding_stats.get('success_rate', 0):.2%}")
        
        # Vector store stats
        collections = await vector_store.list_collections()
        console.print(f"\n🗂️  Vector Store:")
        console.print(f"  Collections: {len(collections)}")
        for collection in collections:
            console.print(f"    - {collection.name}: {collection.vector_count} vectors")
        
        # Knowledge graph stats
        from .knowledge_graph import temporal_knowledge_graph
        graph_stats = temporal_knowledge_graph.get_graph_stats()
        console.print(f"\n🕸️  Knowledge Graph:")
        console.print(f"  Nodes: {graph_stats.get('nodes', 0)}")
        console.print(f"  Edges: {graph_stats.get('edges', 0)}")
        console.print(f"  Entities: {graph_stats.get('entities', 0)}")
        console.print(f"  Facts: {graph_stats.get('facts', 0)}")
        console.print(f"  Events: {graph_stats.get('events', 0)}")
        
    except Exception as e:
        console.print(f"💥 Failed to get stats: {e}")
        logger.exception("Full error details:")


@cli.command()
@click.option('--collection', help='Collection to reindex (all if not specified)')
async def reindex(collection):
    """Reindex vector collections."""
    if collection:
        console.print(f"🔄 Reindexing collection: {collection}")
        collections_to_reindex = [collection]
    else:
        console.print("🔄 Reindexing all collections...")
        collections = await vector_store.list_collections()
        collections_to_reindex = [c.name for c in collections]
    
    try:
        for collection_name in collections_to_reindex:
            console.print(f"  Reindexing {collection_name}...")
            success = await vector_store.reindex_collection(collection_name)
            
            if success:
                console.print(f"  ✅ {collection_name} reindexed successfully")
            else:
                console.print(f"  ❌ Failed to reindex {collection_name}")
        
        console.print("🎉 Reindexing completed!")
        
    except Exception as e:
        console.print(f"💥 Reindexing failed: {e}")
        logger.exception("Full error details:")


@cli.command()
def config():
    """Show current configuration."""
    console.print("⚙️  Current Configuration")
    
    try:
        config = get_config()
        config_dict = config.dict()
        
        # Display key configuration sections
        console.print(f"\n🔤 Embedding Configuration:")
        console.print(f"  Primary Provider: {config.embedding.primary_provider.value}")
        console.print(f"  Fallback Provider: {config.embedding.fallback_provider.value}")
        console.print(f"  Local Provider: {config.embedding.local_provider.value}")
        console.print(f"  Batch Size: {config.embedding.batch_size}")
        
        console.print(f"\n🗂️  Vector Store Configuration:")
        console.print(f"  Provider: {config.vector_store.provider}")
        console.print(f"  Directory: {config.vector_store.persist_directory}")
        console.print(f"  Collections: {len(config.vector_store.collections)}")
        
        console.print(f"\n🔍 Search Configuration:")
        console.print(f"  Similarity Threshold: {config.search.similarity_threshold}")
        console.print(f"  Max Results: {config.search.max_results}")
        console.print(f"  Hybrid Weights: {config.search.hybrid_weights}")
        
        console.print(f"\n🕐 Temporal Configuration:")
        console.print(f"  Enabled: {config.temporal.enable_temporal_search}")
        console.print(f"  Decay Factor: {config.temporal.temporal_decay_factor}")
        console.print(f"  Max Range Days: {config.temporal.max_temporal_range_days}")
        
        console.print(f"\n🌐 API Configuration:")
        console.print(f"  Host: {config.api.host}")
        console.print(f"  Port: {config.api.port}")
        console.print(f"  Workers: {config.api.workers}")
        
    except Exception as e:
        console.print(f"💥 Failed to show configuration: {e}")
        logger.exception("Full error details:")


def main():
    """Main entry point."""
    # Handle async commands
    async_commands = {'initialize', 'search', 'stats', 'reindex'}
    
    if len(sys.argv) > 1 and sys.argv[1] in async_commands:
        # Run async command
        command_name = sys.argv[1]
        
        if command_name == 'initialize':
            asyncio.run(initialize())
        elif command_name == 'search':
            # Parse search arguments manually for async
            query = input("Enter your search query: ") if '--query' not in sys.argv else None
            method = 'hybrid'  # Default
            max_results = 10   # Default
            
            # Simple argument parsing
            for i, arg in enumerate(sys.argv):
                if arg == '--method' and i + 1 < len(sys.argv):
                    method = sys.argv[i + 1]
                elif arg == '--max-results' and i + 1 < len(sys.argv):
                    max_results = int(sys.argv[i + 1])
                elif arg == '--query' and i + 1 < len(sys.argv):
                    query = sys.argv[i + 1]
            
            if query is None:
                query = input("Enter your search query: ")
            
            asyncio.run(search(query, method, max_results))
        elif command_name == 'stats':
            asyncio.run(stats())
        elif command_name == 'reindex':
            collection = None
            if '--collection' in sys.argv:
                idx = sys.argv.index('--collection')
                if idx + 1 < len(sys.argv):
                    collection = sys.argv[idx + 1]
            asyncio.run(reindex(collection))
    else:
        # Run regular CLI
        cli()


if __name__ == "__main__":
    main()