"""
Data models and schemas for the Vector-Enhanced Temporal Knowledge Graph System.

This module defines Pydantic models for vector metadata, search results,
configuration schemas, and API request/response models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class DomainType(str, Enum):
    """Supported knowledge domains."""
    GENERAL = "general"
    TECHNICAL = "technical"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    MEDICAL = "medical"
    LEGAL = "legal"


class ContentType(str, Enum):
    """Types of content that can be vectorized."""
    CHUNK = "chunk"
    FACT = "fact"
    EVENT = "event"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"


class SearchMethod(str, Enum):
    """Available search methods."""
    VECTOR = "vector"
    GRAPH = "graph"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    GOOGLE = "google"
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class VectorMetadata(BaseModel):
    """Metadata associated with vector embeddings."""
    
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content_type: ContentType
    domain: DomainType
    temporal_context: Optional[Dict[str, Any]] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    vector_version: str = "1.0"
    embedding_provider: EmbeddingProvider
    
    # Temporal fields
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    creation_date: Optional[datetime] = None
    
    # Content fields
    title: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class VectorSearchResult(BaseModel):
    """Result from vector similarity search."""
    
    content_id: UUID
    content: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    metadata: VectorMetadata
    search_method: SearchMethod
    ranking_factors: Dict[str, float] = Field(default_factory=dict)
    
    # Additional context
    highlighted_text: Optional[str] = None
    related_entities: List[str] = Field(default_factory=list)
    temporal_relevance: Optional[float] = None


class QueryContext(BaseModel):
    """Context information for search queries."""
    
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    domain_filter: Optional[DomainType] = None
    temporal_filter: Optional[Dict[str, datetime]] = None
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=50, ge=1, le=1000)
    
    # Search preferences
    preferred_content_types: List[ContentType] = Field(default_factory=list)
    exclude_sources: List[str] = Field(default_factory=list)
    language_preference: str = "en"


class SearchQuery(BaseModel):
    """Search query with parameters."""
    
    query: str = Field(min_length=1, max_length=2000)
    search_method: SearchMethod = SearchMethod.HYBRID
    context: QueryContext = Field(default_factory=QueryContext)
    
    # Search method specific parameters
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    graph_depth: Optional[int] = Field(None, ge=1, le=5)
    enable_query_expansion: bool = True
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v, values):
        if v is not None and v < 0.3:
            raise ValueError('Similarity threshold should be at least 0.3 for meaningful results')
        return v


class HybridSearchWeights(BaseModel):
    """Weights for different search methods in hybrid search."""
    
    vector: float = Field(default=0.6, ge=0.0, le=1.0)
    graph: float = Field(default=0.3, ge=0.0, le=1.0)
    lexical: float = Field(default=0.1, ge=0.0, le=1.0)
    
    @validator('lexical')
    def validate_weights_sum_to_one(cls, v, values):
        total = v + values.get('vector', 0) + values.get('graph', 0)
        if abs(total - 1.0) > 0.01:
            raise ValueError('Search weights must sum to 1.0')
        return v


class SearchResponse(BaseModel):
    """Response from search operations."""
    
    query: str
    results: List[VectorSearchResult]
    total_results: int
    search_time_ms: float
    search_method: SearchMethod
    
    # Aggregated information
    domains_found: List[DomainType] = Field(default_factory=list)
    content_types_found: List[ContentType] = Field(default_factory=list)
    temporal_range: Optional[Dict[str, datetime]] = None
    confidence_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query_id: UUID = Field(default_factory=uuid4)


class DocumentChunk(BaseModel):
    """A chunk of document content for processing."""
    
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    chunk_index: int
    
    # Metadata
    title: Optional[str] = None
    source: str
    domain: DomainType = DomainType.GENERAL
    
    # Temporal information
    creation_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    
    # Processing metadata
    processed: bool = False
    processing_version: str = "1.0"
    confidence_score: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: lambda v: str(v)
        }


class AtomicFact(BaseModel):
    """An atomic fact extracted from content."""
    
    fact_id: UUID = Field(default_factory=uuid4)
    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Source information
    source_chunk_id: UUID
    source_document_id: UUID
    extraction_method: str = "nlp"
    
    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    temporal_context: Optional[str] = None
    
    # Metadata
    domain: DomainType = DomainType.GENERAL
    verified: bool = False
    verification_source: Optional[str] = None


class TemporalEvent(BaseModel):
    """A temporal event in the knowledge graph."""
    
    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    description: str
    
    # Temporal information
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[str] = None
    
    # Entities involved
    primary_entities: List[str] = Field(default_factory=list)
    secondary_entities: List[str] = Field(default_factory=list)
    
    # Source and confidence
    source_chunk_id: UUID
    confidence: float = Field(ge=0.0, le=1.0)
    domain: DomainType = DomainType.GENERAL


class CollectionInfo(BaseModel):
    """Information about a vector collection."""
    
    name: str
    description: Optional[str] = None
    embedding_function: EmbeddingProvider
    vector_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    metadata_schema: Dict[str, Any] = Field(default_factory=dict)
    
    # Statistics
    average_similarity: Optional[float] = None
    content_type_distribution: Dict[ContentType, int] = Field(default_factory=dict)
    domain_distribution: Dict[DomainType, int] = Field(default_factory=dict)


class SystemStats(BaseModel):
    """System performance and usage statistics."""
    
    total_vectors: int = 0
    total_documents: int = 0
    total_facts: int = 0
    total_events: int = 0
    
    # Performance metrics
    average_query_time_ms: float = 0.0
    queries_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Usage statistics
    most_queried_domains: List[str] = Field(default_factory=list)
    popular_search_methods: Dict[SearchMethod, int] = Field(default_factory=dict)
    
    # System health
    last_backup: Optional[datetime] = None
    vector_store_size_mb: float = 0.0
    uptime_seconds: int = 0


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""
    
    provider: EmbeddingProvider
    model: str
    api_key_env: Optional[str] = None
    batch_size: int = Field(default=32, ge=1, le=128)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    
    # Provider-specific settings
    device: str = "cpu"  # For local models
    normalize_embeddings: bool = True
    timeout_seconds: int = 30


class VectorStoreConfig(BaseModel):
    """Configuration for vector storage."""
    
    provider: str = "chromadb"
    persist_directory: str = "./data/vector_store"
    collections: List[Dict[str, str]] = Field(default_factory=list)
    
    # Performance settings
    batch_size: int = Field(default=100, ge=1, le=1000)
    index_type: str = "hnsw"
    distance_function: str = "cosine"


class APIResponse(BaseModel):
    """Standard API response wrapper."""
    
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: UUID = Field(default_factory=uuid4)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }