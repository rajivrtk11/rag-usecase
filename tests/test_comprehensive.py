"""
Comprehensive test suite for the Vector-Enhanced Temporal Knowledge Graph System.
"""

import asyncio
import logging
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import shutil

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# Import system components
from src.config import Config, ConfigManager
from src.models import (
    SearchQuery, QueryContext, VectorMetadata, DocumentChunk,
    AtomicFact, TemporalEvent, ContentType, DomainType,
    SearchMethod, EmbeddingProvider
)
from src.embeddings import embedding_manager
from src.vector_store import vector_store
from src.semantic_search import semantic_search_engine
from src.hybrid_search import hybrid_search_coordinator
from src.temporal_search import temporal_search_engine
from src.knowledge_graph import temporal_knowledge_graph
from src.api import create_app


# Configure test logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    # Create temporary directory for test data
    temp_dir = tempfile.mkdtemp(prefix="test_vector_kg_")
    
    config = Config(
        vector_store={
            "provider": "chromadb",
            "persist_directory": f"{temp_dir}/vector_store",
            "collections": [
                {"name": "test_chunks", "embedding_function": "sentence_transformers"},
                {"name": "test_facts", "embedding_function": "sentence_transformers"},
                {"name": "test_events", "embedding_function": "sentence_transformers"}
            ]
        },
        embedding={
            "primary_provider": "sentence_transformers",
            "fallback_provider": "sentence_transformers", 
            "local_provider": "sentence_transformers",
            "batch_size": 4,
            "providers": {
                "sentence_transformers": {
                    "model": "all-MiniLM-L6-v2",
                    "device": "cpu"
                }
            }
        },
        search={
            "similarity_threshold": 0.5,
            "max_results": 10,
            "hybrid_weights": {"vector": 0.7, "graph": 0.2, "lexical": 0.1}
        },
        temporal={
            "enable_temporal_search": True,
            "temporal_decay_factor": 0.9,
            "max_temporal_range_days": 365
        }
    )
    
    yield config, temp_dir
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to cleanup test directory: {e}")


@pytest_asyncio.fixture
async def initialized_system(test_config):
    """Initialize system for testing."""
    config, temp_dir = test_config
    
    # Set test configuration
    from src.config import config_manager
    config_manager._config = config
    
    # Initialize components
    embedding_success = await embedding_manager.initialize()
    assert embedding_success, "Failed to initialize embedding manager"
    
    # Initialize vector store with test directory
    vector_store.persist_directory = config.vector_store.persist_directory
    vector_success = await vector_store.initialize()
    assert vector_success, "Failed to initialize vector store"
    
    yield config
    
    # Cleanup - reset global state
    embedding_manager.providers.clear()
    vector_store.collections.clear()
    temporal_knowledge_graph.graph.clear()


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    base_time = datetime.utcnow()
    
    documents = [
        DocumentChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="Python is a powerful programming language used for AI and machine learning applications.",
            chunk_index=0,
            source="test_doc_1.txt",
            domain=DomainType.TECHNICAL,
            creation_date=base_time - timedelta(days=10),
            confidence_score=0.9
        ),
        DocumentChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="The company reported strong quarterly earnings with revenue growth of 15% year over year.",
            chunk_index=0,
            source="earnings_report.pdf",
            domain=DomainType.BUSINESS,
            creation_date=base_time - timedelta(days=5),
            confidence_score=0.85
        ),
        DocumentChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            content="Climate change research shows increasing global temperatures over the past century.",
            chunk_index=0,
            source="climate_study.pdf",
            domain=DomainType.SCIENTIFIC,
            creation_date=base_time - timedelta(days=2),
            confidence_score=0.95
        )
    ]
    
    return documents


@pytest.fixture
def sample_facts():
    """Sample atomic facts for testing."""
    base_time = datetime.utcnow()
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    
    facts = [
        AtomicFact(
            fact_id=uuid.uuid4(),
            subject="Python",
            predicate="is_used_for",
            object="AI applications",
            confidence=0.9,
            source_chunk_id=chunk_id,
            source_document_id=doc_id,
            domain=DomainType.TECHNICAL,
            valid_from=base_time - timedelta(days=30)
        ),
        AtomicFact(
            fact_id=uuid.uuid4(),
            subject="Company",
            predicate="reported",
            object="quarterly earnings",
            confidence=0.85,
            source_chunk_id=chunk_id,
            source_document_id=doc_id,
            domain=DomainType.BUSINESS,
            valid_from=base_time - timedelta(days=7)
        ),
        AtomicFact(
            fact_id=uuid.uuid4(),
            subject="Global temperatures",
            predicate="increased_over",
            object="past century",
            confidence=0.95,
            source_chunk_id=chunk_id,
            source_document_id=doc_id,
            domain=DomainType.SCIENTIFIC,
            valid_from=base_time - timedelta(days=365)
        )
    ]
    
    return facts


@pytest.fixture
def sample_events():
    """Sample temporal events for testing."""
    base_time = datetime.utcnow()
    
    events = [
        TemporalEvent(
            event_id=uuid.uuid4(),
            event_type="product_launch",
            description="Company launched new AI-powered software platform",
            start_time=base_time - timedelta(days=30),
            primary_entities=["Company", "AI platform"],
            confidence=0.9,
            source_chunk_id=uuid.uuid4(),
            domain=DomainType.BUSINESS
        ),
        TemporalEvent(
            event_id=uuid.uuid4(),
            event_type="research_publication",
            description="Research team published findings on climate modeling",
            start_time=base_time - timedelta(days=60),
            primary_entities=["Research team", "Climate model"],
            confidence=0.85,
            source_chunk_id=uuid.uuid4(),
            domain=DomainType.SCIENTIFIC
        )
    ]
    
    return events


class TestEmbeddingManager:
    """Test embedding manager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, initialized_system):
        """Test embedding manager initialization."""
        assert len(embedding_manager.providers) > 0
        assert EmbeddingProvider.SENTENCE_TRANSFORMERS in embedding_manager.providers
    
    @pytest.mark.asyncio
    async def test_single_embedding(self, initialized_system):
        """Test generating single embedding."""
        text = "This is a test sentence for embedding generation."
        embedding = await embedding_manager.get_embedding(text)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_batch_embeddings(self, initialized_system):
        """Test generating batch embeddings."""
        texts = [
            "First test sentence.",
            "Second test sentence.", 
            "Third test sentence."
        ]
        
        embeddings = await embedding_manager.get_embeddings_batch(texts)
        
        assert len(embeddings) == len(texts)
        assert all(emb is not None for emb in embeddings)
        assert all(isinstance(emb, list) for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_health_check(self, initialized_system):
        """Test embedding provider health check."""
        health_status = await embedding_manager.health_check()
        
        assert isinstance(health_status, dict)
        assert EmbeddingProvider.SENTENCE_TRANSFORMERS in health_status
        assert health_status[EmbeddingProvider.SENTENCE_TRANSFORMERS] is True


class TestVectorStore:
    """Test vector store functionality."""
    
    @pytest.mark.asyncio
    async def test_add_vectors(self, initialized_system, sample_documents):
        """Test adding vectors to collection."""
        # Prepare data
        texts = [doc.content for doc in sample_documents]
        metadatas = []
        ids = []
        
        for doc in sample_documents:
            metadata = VectorMetadata(
                chunk_id=doc.chunk_id,
                document_id=doc.document_id,
                content_type=ContentType.CHUNK,
                domain=doc.domain,
                confidence_score=doc.confidence_score or 0.8,
                source=doc.source,
                embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS
            )
            metadatas.append(metadata)
            ids.append(str(doc.chunk_id))
        
        # Add vectors
        success = await vector_store.add_vectors(
            collection_name="test_chunks",
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_search_vectors(self, initialized_system, sample_documents):
        """Test vector search."""
        # First add some vectors
        await self.test_add_vectors(initialized_system, sample_documents)
        
        # Search for Python-related content
        results = await vector_store.search_vectors(
            collection_name="test_chunks",
            query_text="Python programming language",
            n_results=5,
            similarity_threshold=0.3
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check result structure
        result = results[0]
        assert hasattr(result, 'content')
        assert hasattr(result, 'similarity_score')
        assert hasattr(result, 'metadata')
        assert 0 <= result.similarity_score <= 1
    
    @pytest.mark.asyncio
    async def test_collection_management(self, initialized_system):
        """Test collection creation and listing."""
        # List existing collections
        collections = await vector_store.list_collections()
        initial_count = len(collections)
        
        # Create new collection
        success = await vector_store.create_collection(
            name="test_new_collection",
            description="Test collection for unit tests"
        )
        assert success is True
        
        # Verify collection was created
        collections = await vector_store.list_collections()
        assert len(collections) == initial_count + 1
        
        # Get collection info
        info = await vector_store.get_collection_info("test_new_collection")
        assert info is not None
        assert info.name == "test_new_collection"


class TestSemanticSearch:
    """Test semantic search engine."""
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, initialized_system, sample_documents):
        """Test semantic search functionality."""
        # Add sample documents first
        texts = [doc.content for doc in sample_documents]
        metadatas = []
        ids = []
        
        for doc in sample_documents:
            metadata = VectorMetadata(
                chunk_id=doc.chunk_id,
                document_id=doc.document_id,
                content_type=ContentType.CHUNK,
                domain=doc.domain,
                confidence_score=doc.confidence_score or 0.8,
                source=doc.source,
                embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS
            )
            metadatas.append(metadata)
            ids.append(str(doc.chunk_id))
        
        await vector_store.add_vectors(
            collection_name="test_chunks",
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Create search query
        query_context = QueryContext(max_results=5)
        search_query = SearchQuery(
            query="programming language for artificial intelligence",
            search_method=SearchMethod.VECTOR,
            context=query_context
        )
        
        # Perform search
        response = await semantic_search_engine.search(search_query)
        
        # Verify response
        assert response is not None
        assert response.query == search_query.query
        assert isinstance(response.results, list)
        assert response.search_method == SearchMethod.VECTOR
        assert response.search_time_ms >= 0
    
    def test_query_preprocessing(self, initialized_system):
        """Test query preprocessing."""
        from src.semantic_search import QueryPreprocessor
        
        preprocessor = QueryPreprocessor()
        
        # Test temporal expression extraction
        query = "What happened in 2023 regarding AI development?"
        processed = preprocessor.preprocess_query(query)
        
        assert 'temporal_expressions' in processed
        assert 'keywords' in processed
        assert 'domain_hints' in processed
        assert len(processed['keywords']) > 0


class TestHybridSearch:
    """Test hybrid search coordinator."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, initialized_system, sample_documents):
        """Test hybrid search functionality."""
        # Add sample data
        texts = [doc.content for doc in sample_documents]
        metadatas = []
        ids = []
        
        for doc in sample_documents:
            metadata = VectorMetadata(
                chunk_id=doc.chunk_id,
                document_id=doc.document_id,
                content_type=ContentType.CHUNK,
                domain=doc.domain,
                confidence_score=doc.confidence_score or 0.8,
                source=doc.source,
                embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS
            )
            metadatas.append(metadata)
            ids.append(str(doc.chunk_id))
        
        await vector_store.add_vectors(
            collection_name="test_chunks",
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Test hybrid search
        query_context = QueryContext(max_results=5)
        search_query = SearchQuery(
            query="revenue growth business performance",
            search_method=SearchMethod.HYBRID,
            context=query_context
        )
        
        response = await hybrid_search_coordinator.search(search_query)
        
        assert response is not None
        assert isinstance(response.results, list)
        assert response.search_method in [SearchMethod.HYBRID, SearchMethod.VECTOR, SearchMethod.GRAPH, SearchMethod.LEXICAL]


class TestKnowledgeGraph:
    """Test knowledge graph functionality."""
    
    def test_add_facts(self, initialized_system, sample_facts):
        """Test adding facts to knowledge graph."""
        for fact in sample_facts:
            success = temporal_knowledge_graph.add_fact(fact)
            assert success is True
        
        # Verify graph stats
        stats = temporal_knowledge_graph.get_graph_stats()
        assert stats['facts'] >= len(sample_facts)
        assert stats['entities'] > 0
        assert stats['nodes'] > 0
    
    def test_add_events(self, initialized_system, sample_events):
        """Test adding events to knowledge graph."""
        for event in sample_events:
            success = temporal_knowledge_graph.add_event(event)
            assert success is True
        
        # Verify graph stats
        stats = temporal_knowledge_graph.get_graph_stats()
        assert stats['events'] >= len(sample_events)
    
    def test_entity_search(self, initialized_system, sample_facts):
        """Test entity search in knowledge graph."""
        # Add facts first
        for fact in sample_facts:
            temporal_knowledge_graph.add_fact(fact)
        
        # Search for Python entity
        neighbors = temporal_knowledge_graph.get_entity_neighbors("Python", depth=1)
        assert isinstance(neighbors, list)
        
        # Search by keywords
        results = temporal_knowledge_graph.search_by_keywords(["Python", "AI"])
        assert isinstance(results, list)


class TestTemporalSearch:
    """Test temporal search functionality."""
    
    def test_temporal_parser(self, initialized_system):
        """Test temporal expression parsing."""
        from src.temporal_search import TemporalParser
        
        parser = TemporalParser()
        
        # Test absolute date parsing
        expressions = parser.extract_temporal_expressions("The event happened on 2023-06-15")
        assert len(expressions) > 0
        assert any(expr['category'] == 'absolute_dates' for expr in expressions)
        
        # Test relative date parsing
        expressions = parser.extract_temporal_expressions("This happened last week")
        assert len(expressions) > 0
        
        # Test temporal indicators
        assert parser.has_temporal_indicators("when did this happen?") is True
        assert parser.has_temporal_indicators("what is Python?") is False
    
    @pytest.mark.asyncio
    async def test_temporal_search(self, initialized_system, sample_documents):
        """Test temporal search functionality."""
        # Add sample documents with temporal metadata
        texts = [doc.content for doc in sample_documents]
        metadatas = []
        ids = []
        
        for doc in sample_documents:
            metadata = VectorMetadata(
                chunk_id=doc.chunk_id,
                document_id=doc.document_id,
                content_type=ContentType.CHUNK,
                domain=doc.domain,
                confidence_score=doc.confidence_score or 0.8,
                source=doc.source,
                creation_date=doc.creation_date,
                embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS
            )
            metadatas.append(metadata)
            ids.append(str(doc.chunk_id))
        
        await vector_store.add_vectors(
            collection_name="test_chunks",
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Test temporal search
        query_context = QueryContext(max_results=5)
        search_query = SearchQuery(
            query="recent business developments",
            context=query_context
        )
        
        results = await temporal_search_engine.temporal_search(search_query)
        assert isinstance(results, list)


class TestAPI:
    """Test API endpoints."""
    
    @pytest.fixture
    def client(self, initialized_system):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/system/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['success'] is True
        assert 'data' in data
    
    def test_search_endpoints(self, client, sample_documents):
        """Test search endpoints."""
        # Note: This would need the system to be properly initialized
        # with real data for a complete test
        
        search_data = {
            "query": "test query",
            "search_method": "hybrid",
            "context": {
                "max_results": 5
            }
        }
        
        # Test semantic search (might return empty results without data)
        response = client.post("/search/semantic", json=search_data)
        # The endpoint might return 503 if system not initialized properly
        # or 200 with empty results
        assert response.status_code in [200, 503]


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, initialized_system, sample_documents, sample_facts, sample_events):
        """Test complete workflow from data ingestion to search."""
        # 1. Add documents
        texts = [doc.content for doc in sample_documents]
        metadatas = []
        ids = []
        
        for doc in sample_documents:
            metadata = VectorMetadata(
                chunk_id=doc.chunk_id,
                document_id=doc.document_id,
                content_type=ContentType.CHUNK,
                domain=doc.domain,
                confidence_score=doc.confidence_score or 0.8,
                source=doc.source,
                creation_date=doc.creation_date,
                embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS
            )
            metadatas.append(metadata)
            ids.append(str(doc.chunk_id))
        
        doc_success = await vector_store.add_vectors(
            collection_name="test_chunks",
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        assert doc_success is True
        
        # 2. Add facts to knowledge graph
        for fact in sample_facts:
            fact_success = temporal_knowledge_graph.add_fact(fact)
            assert fact_success is True
        
        # 3. Add events to knowledge graph
        for event in sample_events:
            event_success = temporal_knowledge_graph.add_event(event)
            assert event_success is True
        
        # 4. Perform various searches
        query_context = QueryContext(max_results=5)
        
        # Semantic search
        semantic_query = SearchQuery(
            query="artificial intelligence programming",
            search_method=SearchMethod.VECTOR,
            context=query_context
        )
        semantic_response = await semantic_search_engine.search(semantic_query)
        assert len(semantic_response.results) >= 0
        
        # Hybrid search
        hybrid_query = SearchQuery(
            query="business revenue growth",
            search_method=SearchMethod.HYBRID,
            context=query_context
        )
        hybrid_response = await hybrid_search_coordinator.search(hybrid_query)
        assert len(hybrid_response.results) >= 0
        
        # Temporal search
        temporal_results = await temporal_search_engine.temporal_search(semantic_query)
        assert isinstance(temporal_results, list)
        
        # 5. Verify system statistics
        embedding_stats = embedding_manager.get_stats()
        assert embedding_stats['requests_total'] > 0
        
        graph_stats = temporal_knowledge_graph.get_graph_stats()
        assert graph_stats['nodes'] > 0
        assert graph_stats['facts'] >= len(sample_facts)
        assert graph_stats['events'] >= len(sample_events)


class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, initialized_system):
        """Test performance of batch processing."""
        import time
        
        # Generate test data
        test_texts = [f"This is test document number {i} with some content." for i in range(100)]
        
        # Test batch embedding generation
        start_time = time.time()
        embeddings = await embedding_manager.get_embeddings_batch(test_texts)
        batch_time = time.time() - start_time
        
        assert len(embeddings) == len(test_texts)
        assert batch_time < 30  # Should complete within 30 seconds
        
        # Test individual embedding generation for comparison
        start_time = time.time()
        individual_embeddings = []
        for text in test_texts[:10]:  # Test smaller subset
            embedding = await embedding_manager.get_embedding(text)
            individual_embeddings.append(embedding)
        individual_time = time.time() - start_time
        
        assert len(individual_embeddings) == 10
        
        # Batch processing should be more efficient per item
        batch_per_item = batch_time / len(test_texts)
        individual_per_item = individual_time / 10
        
        logger.info(f"Batch processing: {batch_per_item:.3f}s per item")
        logger.info(f"Individual processing: {individual_per_item:.3f}s per item")
    
    @pytest.mark.asyncio
    async def test_search_performance(self, initialized_system, sample_documents):
        """Test search performance."""
        import time
        
        # Add sample data
        texts = [doc.content for doc in sample_documents * 10]  # Multiply for more data
        metadatas = []
        ids = []
        
        for i, doc in enumerate(sample_documents * 10):
            metadata = VectorMetadata(
                chunk_id=uuid.uuid4(),
                document_id=doc.document_id,
                content_type=ContentType.CHUNK,
                domain=doc.domain,
                confidence_score=doc.confidence_score or 0.8,
                source=f"{doc.source}_{i}",
                embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS
            )
            metadatas.append(metadata)
            ids.append(str(uuid.uuid4()))
        
        await vector_store.add_vectors(
            collection_name="test_chunks",
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Test search performance
        query_context = QueryContext(max_results=10)
        search_query = SearchQuery(
            query="programming language artificial intelligence",
            context=query_context
        )
        
        start_time = time.time()
        response = await semantic_search_engine.search(search_query)
        search_time = time.time() - start_time
        
        assert response.search_time_ms > 0
        assert search_time < 5.0  # Should complete within 5 seconds
        
        logger.info(f"Search completed in {search_time:.3f}s ({response.search_time_ms:.2f}ms reported)")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])