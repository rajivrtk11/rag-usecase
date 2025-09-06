"""
FastAPI endpoints for the Vector-Enhanced Temporal Knowledge Graph System.

This module provides RESTful API endpoints for vector search operations,
collection management, and system monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import get_config
from .models import (
    SearchQuery,
    SearchResponse,
    APIResponse,
    CollectionInfo,
    SystemStats,
    VectorMetadata,
    DocumentChunk,
    AtomicFact,
    TemporalEvent,
    ContentType,
    DomainType,
    SearchMethod
)
from .vector_store import vector_store, initialize_vector_store
from .embeddings import embedding_manager
from .semantic_search import semantic_search_engine, perform_semantic_search
from .hybrid_search import hybrid_search_coordinator, perform_hybrid_search
from .temporal_search import temporal_search_engine, perform_temporal_search
from .knowledge_graph import temporal_knowledge_graph


logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vector-Enhanced Temporal Knowledge Graph System",
    description="A comprehensive RAG system with vector search, temporal reasoning, and hybrid retrieval",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_system_initialized = False
_system_stats = {
    'startup_time': None,
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0
}


async def get_system_dependency():
    """Dependency to ensure system is initialized."""
    global _system_initialized
    if not _system_initialized:
        raise HTTPException(
            status_code=503,
            detail="System is not initialized. Call /system/initialize first."
        )
    return True


def track_request():
    """Dependency to track API requests."""
    _system_stats['total_requests'] += 1


def create_api_response(success: bool, data: Any = None, error: str = None) -> APIResponse:
    """Create standardized API response."""
    if success:
        _system_stats['successful_requests'] += 1
    else:
        _system_stats['failed_requests'] += 1
    
    return APIResponse(
        success=success,
        data=data,
        error=error
    )


# System Management Endpoints

@app.post("/system/initialize", response_model=APIResponse)
async def initialize_system():
    """Initialize the vector-enhanced temporal knowledge graph system."""
    global _system_initialized, _system_stats
    
    try:
        logger.info("Initializing system...")
        
        # Initialize embedding manager
        logger.info("Initializing embedding manager...")
        embedding_init = await embedding_manager.initialize()
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_init = await initialize_vector_store()
        
        if not embedding_init or not vector_init:
            raise Exception("Failed to initialize core components")
        
        _system_initialized = True
        _system_stats['startup_time'] = datetime.utcnow()
        
        logger.info("System initialization completed successfully")
        
        return create_api_response(
            success=True,
            data={
                "message": "System initialized successfully",
                "embedding_providers": embedding_manager.get_available_providers(),
                "vector_collections": [info.name for info in await vector_store.list_collections()]
            }
        )
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return create_api_response(
            success=False,
            error=f"System initialization failed: {str(e)}"
        )


@app.get("/system/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check embedding providers
        embedding_health = await embedding_manager.health_check()
        
        # Check vector store
        collections = await vector_store.list_collections()
        
        health_status = {
            "status": "healthy" if _system_initialized else "not_initialized",
            "embedding_providers": {
                provider.value: status 
                for provider, status in embedding_health.items()
            },
            "vector_collections": len(collections),
            "uptime_seconds": (
                (datetime.utcnow() - _system_stats['startup_time']).total_seconds()
                if _system_stats['startup_time'] else 0
            )
        }
        
        return create_api_response(success=True, data=health_status)
        
    except Exception as e:
        return create_api_response(
            success=False,
            error=f"Health check failed: {str(e)}"
        )


@app.get("/system/stats", response_model=APIResponse)
async def get_system_stats():
    """Get system statistics."""
    try:
        # Collect stats from all components
        stats = {
            "api_stats": _system_stats.copy(),
            "embedding_stats": embedding_manager.get_stats(),
            "semantic_search_stats": semantic_search_engine.get_search_stats(),
            "hybrid_search_stats": hybrid_search_coordinator.get_search_stats(),
            "temporal_search_stats": temporal_search_engine.get_temporal_search_stats(),
            "knowledge_graph_stats": temporal_knowledge_graph.get_graph_stats()
        }
        
        return create_api_response(success=True, data=stats)
        
    except Exception as e:
        return create_api_response(
            success=False,
            error=f"Failed to get system stats: {str(e)}"
        )


# Search Endpoints

@app.post("/search/semantic", response_model=SearchResponse, dependencies=[Depends(track_request)])
async def semantic_search(search_query: SearchQuery, _: bool = Depends(get_system_dependency)):
    """Perform semantic vector search."""
    try:
        logger.info(f"Semantic search query: {search_query.query}")
        
        response = await perform_semantic_search(search_query)
        
        logger.info(f"Semantic search completed: {len(response.results)} results in {response.search_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(e)}"
        )


@app.post("/search/hybrid", response_model=SearchResponse, dependencies=[Depends(track_request)])
async def hybrid_search(search_query: SearchQuery, _: bool = Depends(get_system_dependency)):
    """Perform hybrid search combining vector, graph, and lexical methods."""
    try:
        logger.info(f"Hybrid search query: {search_query.query}")
        
        response = await perform_hybrid_search(search_query)
        
        logger.info(f"Hybrid search completed: {len(response.results)} results in {response.search_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid search failed: {str(e)}"
        )


@app.post("/search/temporal", response_model=APIResponse, dependencies=[Depends(track_request)])
async def temporal_search(
    search_query: SearchQuery,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    _: bool = Depends(get_system_dependency)
):
    """Perform temporal-aware search."""
    try:
        logger.info(f"Temporal search query: {search_query.query}")
        
        # Build temporal context
        temporal_context = {}
        if start_date:
            temporal_context['start_date'] = start_date
        if end_date:
            temporal_context['end_date'] = end_date
        
        results = await perform_temporal_search(
            search_query, 
            temporal_context if temporal_context else None
        )
        
        logger.info(f"Temporal search completed: {len(results)} results")
        
        return create_api_response(
            success=True,
            data={
                "query": search_query.query,
                "results": [result.dict() for result in results],
                "temporal_context": temporal_context
            }
        )
        
    except Exception as e:
        logger.error(f"Temporal search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Temporal search failed: {str(e)}"
        )


# Vector Collection Management Endpoints

@app.get("/vectors/collections", response_model=APIResponse, dependencies=[Depends(track_request)])
async def list_collections(_: bool = Depends(get_system_dependency)):
    """List all vector collections."""
    try:
        collections = await vector_store.list_collections()
        
        return create_api_response(
            success=True,
            data=[collection.dict() for collection in collections]
        )
        
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list collections: {str(e)}"
        )


@app.get("/vectors/collections/{collection_name}", response_model=APIResponse, dependencies=[Depends(track_request)])
async def get_collection_info(collection_name: str, _: bool = Depends(get_system_dependency)):
    """Get information about a specific collection."""
    try:
        collection_info = await vector_store.get_collection_info(collection_name)
        
        if not collection_info:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found"
            )
        
        return create_api_response(
            success=True,
            data=collection_info.dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get collection info: {str(e)}"
        )


@app.post("/vectors/collections/{collection_name}", response_model=APIResponse, dependencies=[Depends(track_request)])
async def create_collection(
    collection_name: str,
    description: Optional[str] = None,
    _: bool = Depends(get_system_dependency)
):
    """Create a new vector collection."""
    try:
        success = await vector_store.create_collection(
            name=collection_name,
            description=description
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to create collection '{collection_name}'"
            )
        
        return create_api_response(
            success=True,
            data={"message": f"Collection '{collection_name}' created successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create collection: {str(e)}"
        )


@app.delete("/vectors/collections/{collection_name}", response_model=APIResponse, dependencies=[Depends(track_request)])
async def delete_collection(collection_name: str, _: bool = Depends(get_system_dependency)):
    """Delete a vector collection."""
    try:
        success = await vector_store.delete_collection(collection_name)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found or could not be deleted"
            )
        
        return create_api_response(
            success=True,
            data={"message": f"Collection '{collection_name}' deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection: {str(e)}"
        )


@app.post("/vectors/reindex", response_model=APIResponse, dependencies=[Depends(track_request)])
async def reindex_collections(
    collection_names: Optional[List[str]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: bool = Depends(get_system_dependency)
):
    """Reindex vector collections."""
    try:
        if collection_names is None:
            # Reindex all collections
            collections = await vector_store.list_collections()
            collection_names = [col.name for col in collections]
        
        # Start reindexing in background
        async def reindex_task():
            for collection_name in collection_names:
                try:
                    logger.info(f"Reindexing collection: {collection_name}")
                    success = await vector_store.reindex_collection(collection_name)
                    if success:
                        logger.info(f"Successfully reindexed collection: {collection_name}")
                    else:
                        logger.error(f"Failed to reindex collection: {collection_name}")
                except Exception as e:
                    logger.error(f"Error reindexing collection {collection_name}: {e}")
        
        # Add to background tasks
        background_tasks.add_task(reindex_task)
        
        return create_api_response(
            success=True,
            data={
                "message": f"Reindexing started for {len(collection_names)} collections",
                "collections": collection_names
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start reindexing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start reindexing: {str(e)}"
        )


# Content Management Endpoints

@app.post("/content/chunks", response_model=APIResponse, dependencies=[Depends(track_request)])
async def add_document_chunks(
    chunks: List[DocumentChunk],
    collection_name: str = "document_chunks",
    _: bool = Depends(get_system_dependency)
):
    """Add document chunks to vector collection."""
    try:
        logger.info(f"Adding {len(chunks)} document chunks to collection '{collection_name}'")
        
        # Prepare data for vector store
        texts = [chunk.content for chunk in chunks]
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Create vector metadata
            metadata = VectorMetadata(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content_type=ContentType.CHUNK,
                domain=chunk.domain,
                confidence_score=chunk.confidence_score or 0.8,
                source=chunk.source,
                title=chunk.title,
                creation_date=chunk.creation_date,
                last_updated=chunk.last_modified,
                valid_from=chunk.valid_from,
                valid_to=chunk.valid_to
            )
            
            metadatas.append(metadata)
            ids.append(str(chunk.chunk_id))
        
        # Add to vector store
        success = await vector_store.add_vectors(
            collection_name=collection_name,
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add chunks to vector store"
            )
        
        logger.info(f"Successfully added {len(chunks)} chunks")
        
        return create_api_response(
            success=True,
            data={
                "message": f"Added {len(chunks)} document chunks",
                "collection": collection_name,
                "chunk_ids": ids
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add document chunks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add document chunks: {str(e)}"
        )


@app.post("/content/facts", response_model=APIResponse, dependencies=[Depends(track_request)])
async def add_atomic_facts(
    facts: List[AtomicFact],
    _: bool = Depends(get_system_dependency)
):
    """Add atomic facts to knowledge graph and vector store."""
    try:
        logger.info(f"Adding {len(facts)} atomic facts")
        
        success_count = 0
        
        for fact in facts:
            # Add to knowledge graph
            graph_success = temporal_knowledge_graph.add_fact(fact)
            
            if graph_success:
                # Add to vector store
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}"
                
                metadata = VectorMetadata(
                    chunk_id=fact.fact_id,
                    document_id=fact.source_document_id,
                    content_type=ContentType.FACT,
                    domain=fact.domain,
                    confidence_score=fact.confidence,
                    source=f"fact_extraction_{fact.extraction_method}",
                    valid_from=fact.valid_from,
                    valid_to=fact.valid_to
                )
                
                vector_success = await vector_store.add_vectors(
                    collection_name="atomic_facts",
                    texts=[fact_text],
                    metadatas=[metadata],
                    ids=[str(fact.fact_id)]
                )
                
                if vector_success:
                    success_count += 1
        
        logger.info(f"Successfully added {success_count}/{len(facts)} facts")
        
        return create_api_response(
            success=True,
            data={
                "message": f"Added {success_count}/{len(facts)} atomic facts",
                "success_count": success_count,
                "total_count": len(facts)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to add atomic facts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add atomic facts: {str(e)}"
        )


@app.post("/content/events", response_model=APIResponse, dependencies=[Depends(track_request)])
async def add_temporal_events(
    events: List[TemporalEvent],
    _: bool = Depends(get_system_dependency)
):
    """Add temporal events to knowledge graph and vector store."""
    try:
        logger.info(f"Adding {len(events)} temporal events")
        
        success_count = 0
        
        for event in events:
            # Add to knowledge graph
            graph_success = temporal_knowledge_graph.add_event(event)
            
            if graph_success:
                # Add to vector store
                event_text = f"{event.event_type}: {event.description}"
                
                metadata = VectorMetadata(
                    chunk_id=event.event_id,
                    document_id=uuid.uuid4(),  # Events don't have documents
                    content_type=ContentType.EVENT,
                    domain=event.domain,
                    confidence_score=event.confidence,
                    source="temporal_event",
                    valid_from=event.start_time,
                    valid_to=event.end_time
                )
                
                vector_success = await vector_store.add_vectors(
                    collection_name="temporal_events",
                    texts=[event_text],
                    metadatas=[metadata],
                    ids=[str(event.event_id)]
                )
                
                if vector_success:
                    success_count += 1
        
        logger.info(f"Successfully added {success_count}/{len(events)} events")
        
        return create_api_response(
            success=True,
            data={
                "message": f"Added {success_count}/{len(events)} temporal events",
                "success_count": success_count,
                "total_count": len(events)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to add temporal events: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add temporal events: {str(e)}"
        )


# Knowledge Graph Endpoints

@app.get("/graph/entities/{entity_name}/neighbors", response_model=APIResponse, dependencies=[Depends(track_request)])
async def get_entity_neighbors(
    entity_name: str,
    depth: int = Query(1, ge=1, le=5),
    relationship_filter: Optional[str] = None,
    _: bool = Depends(get_system_dependency)
):
    """Get neighbors of an entity in the knowledge graph."""
    try:
        neighbors = temporal_knowledge_graph.get_entity_neighbors(
            entity_name=entity_name,
            depth=depth,
            relationship_filter=relationship_filter
        )
        
        return create_api_response(
            success=True,
            data={
                "entity": entity_name,
                "depth": depth,
                "neighbors": neighbors,
                "count": len(neighbors)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get entity neighbors: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get entity neighbors: {str(e)}"
        )


@app.get("/graph/paths", response_model=APIResponse, dependencies=[Depends(track_request)])
async def find_entity_paths(
    start_entity: str = Query(..., description="Start entity name"),
    end_entity: str = Query(..., description="End entity name"),
    max_depth: Optional[int] = Query(None, ge=1, le=10),
    _: bool = Depends(get_system_dependency)
):
    """Find paths between two entities in the knowledge graph."""
    try:
        paths = temporal_knowledge_graph.find_entity_paths(
            start_entity=start_entity,
            end_entity=end_entity,
            max_depth=max_depth
        )
        
        return create_api_response(
            success=True,
            data={
                "start_entity": start_entity,
                "end_entity": end_entity,
                "paths": paths,
                "path_count": len(paths)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to find entity paths: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find entity paths: {str(e)}"
        )


# Configuration endpoint
@app.get("/config", response_model=APIResponse)
async def get_configuration():
    """Get system configuration."""
    try:
        config = get_config()
        
        # Convert to dictionary and remove sensitive information
        config_dict = config.dict()
        
        # Remove API keys and sensitive data
        if 'embedding' in config_dict and 'providers' in config_dict['embedding']:
            for provider_name in config_dict['embedding']['providers']:
                if 'api_key_env' in config_dict['embedding']['providers'][provider_name]:
                    config_dict['embedding']['providers'][provider_name]['api_key_env'] = "***"
        
        return create_api_response(
            success=True,
            data=config_dict
        )
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get configuration: {str(e)}"
        )


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    return app


def run_server():
    """Run the FastAPI server."""
    config = get_config()
    
    uvicorn.run(
        "src.api:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.logging.level.lower()
    )


if __name__ == "__main__":
    run_server()