# Developer Guide

Technical documentation for developers working on the Vector-Enhanced Temporal Knowledge Graph System

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                      │
├─────────────────────────────────────────────────────────────┤
│                Query Processing Engine                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Vector    │  │   Graph     │  │   Hybrid Search     │ │
│  │   Search    │  │   Search    │  │   Coordinator       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                Vector Store Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  ChromaDB   │  │  Embedding  │  │   Metadata Store    │ │
│  │ Collections │  │  Manager    │  │   (In-memory)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                Knowledge Graph Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  NetworkX   │  │  Temporal   │  │   Entity Resolution │ │
│  │   Graph     │  │ Validation  │  │   & Canonicalization│ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Models (`src/models.py`)

Pydantic models defining data structures:

- **VectorMetadata**: Metadata for vector embeddings
- **VectorSearchResult**: Search result structure
- **SearchQuery**: Search query parameters
- **SearchResponse**: Search response format
- **DocumentChunk**: Document content chunks
- **AtomicFact**: Structured facts
- **TemporalEvent**: Time-based events

### 2. Configuration (`src/config.py`)

Configuration management system:

- **ConfigManager**: Loads and manages configuration
- **Configuration models**: Pydantic models for config sections
- **Environment variable resolution**: Secure handling of secrets

### 3. Embeddings (`src/embeddings.py`)

Multi-provider embedding system:

- **EmbeddingProviderBase**: Abstract base class
- **Provider implementations**: Google, OpenAI, SentenceTransformers
- **EmbeddingManager**: Orchestrates providers with fallback

### 4. Vector Store (`src/vector_store.py`)

Vector database integration:

- **ChromaDBVectorStore**: ChromaDB implementation
- **Collection management**: Create, list, delete collections
- **CRUD operations**: Add, search, update, delete vectors

### 5. Semantic Search (`src/semantic_search.py`)

Vector-based search engine:

- **QueryPreprocessor**: Query analysis and normalization
- **QueryExpander**: Query expansion with synonyms
- **ResultRanker**: Multi-factor result ranking
- **SemanticSearchEngine**: Main search coordinator

### 6. Knowledge Graph (`src/knowledge_graph.py`)

Graph-based knowledge representation:

- **TemporalKnowledgeGraph**: NetworkX-based implementation
- **Fact management**: Add and query atomic facts
- **Event management**: Temporal event handling
- **Graph traversal**: Path finding and neighbor search

### 7. Hybrid Search (`src/hybrid_search.py`)

Multi-modal search coordination:

- **QueryRouter**: Determines optimal search method
- **ResultFuser**: Combines results from different methods
- **LexicalSearchEngine**: TF-IDF based search
- **HybridSearchCoordinator**: Main hybrid search engine

### 8. Temporal Search (`src/temporal_search.py`)

Time-aware search capabilities:

- **TemporalParser**: Extracts temporal expressions
- **TemporalIndexer**: Indexes content with temporal info
- **TemporalSearchEngine**: Time-aware search engine

### 9. API (`src/api.py`)

FastAPI-based REST API:

- **System endpoints**: Initialization, health, stats
- **Search endpoints**: Semantic, hybrid, temporal search
- **Management endpoints**: Collections, content management
- **Graph endpoints**: Entity and relationship queries

### 10. Main Application (`src/main.py`)

Command-line interface and application entry point:

- **CLI commands**: Initialize, search, stats, etc.
- **Server startup**: FastAPI server management
- **Async support**: Proper async/await handling

## Data Flow

### Search Request Flow

1. **API Layer**: Receives HTTP request
2. **Query Processing**: Validates and processes query
3. **Search Method Selection**: Determines optimal search approach
4. **Execution**: Performs search using appropriate components
5. **Result Processing**: Ranks and formats results
6. **Response**: Returns JSON response

### Data Ingestion Flow

1. **API Layer**: Receives content via API
2. **Validation**: Validates data structure
3. **Embedding Generation**: Creates vector embeddings
4. **Storage**: Stores vectors in ChromaDB
5. **Graph Update**: Updates knowledge graph
6. **Metadata**: Stores metadata for retrieval

## Extending the System

### Adding New Embedding Providers

1. Create a new class inheriting from `EmbeddingProviderBase`
2. Implement `initialize()`, `embed_text()`, and `embed_batch()` methods
3. Add provider configuration to `config.yaml`
4. Register provider in `EmbeddingManager`

```python
class NewEmbeddingProvider(EmbeddingProviderBase):
    async def initialize(self) -> bool:
        # Initialize provider client
        pass
    
    async def embed_text(self, text: str) -> Optional[List[float]]:
        # Generate single embedding
        pass
    
    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        # Generate batch embeddings
        pass
```

### Adding New Search Methods

1. Create new search method in appropriate module
2. Add method to `SearchMethod` enum in `models.py`
3. Update `QueryRouter` to detect when to use the method
4. Add result processing and ranking logic

### Adding New Content Types

1. Add new `ContentType` enum value
2. Update data models to handle new content type
3. Add processing logic in relevant components
4. Update API endpoints for new content type

## Testing Strategy

### Unit Tests (`tests/test_comprehensive.py`)

- **Component isolation**: Test individual components
- **Mock dependencies**: Use mocks for external services
- **Edge cases**: Test boundary conditions
- **Error handling**: Verify proper error responses

### Integration Tests

- **Component interaction**: Test component communication
- **Data flow**: Verify complete data processing pipelines
- **API endpoints**: Test HTTP interface
- **Performance**: Measure response times and throughput

### Performance Tests

- **Load testing**: Simulate concurrent users
- **Stress testing**: Test system limits
- **Memory usage**: Monitor memory consumption
- **Scalability**: Test with increasing data volumes

### Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_comprehensive.py::TestEmbeddingManager -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance tests
pytest tests/test_comprehensive.py::TestPerformance -v
```

## Development Workflow

### Setting Up Development Environment

1. **Clone repository**:
   ```bash
   git clone <repository-url>
   cd rag-impl-1
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies**:
   ```bash
   pip install black isort flake8 mypy pytest pytest-asyncio
   ```

### Code Quality

#### Formatting (Black)
```bash
black src/ tests/
```

#### Import Sorting (isort)
```bash
isort src/ tests/
```

#### Linting (Flake8)
```bash
flake8 src/ tests/
```

#### Type Checking (MyPy)
```bash
mypy src/
```

### Development Commands

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Format code
black src/ tests/
isort src/ tests/

# Check types
mypy src/

# Lint code
flake8 src/ tests/
```

## API Development

### Adding New Endpoints

1. **Define endpoint in `src/api.py`**:
   ```python
   @app.post("/new/endpoint", response_model=APIResponse)
   async def new_endpoint(request_data: RequestModel):
       # Implementation
       pass
   ```

2. **Add request/response models** in `src/models.py`
3. **Implement business logic** in appropriate modules
4. **Add tests** in `tests/test_comprehensive.py`

### API Documentation

The API is self-documenting using FastAPI's built-in Swagger UI:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Performance Optimization

### Caching Strategy

1. **Result Caching**: Cache search results with TTL
2. **Embedding Caching**: Cache generated embeddings
3. **Configuration Caching**: Cache parsed configurations

### Memory Management

1. **Object pooling**: Reuse objects where possible
2. **Lazy loading**: Load components only when needed
3. **Memory profiling**: Monitor memory usage patterns

### Concurrency

1. **Async/await**: Use async patterns for I/O operations
2. **Thread pools**: Use thread pools for CPU-bound tasks
3. **Connection pooling**: Reuse database connections

## Monitoring and Observability

### Logging

The system uses structured logging with different levels:

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about system operation
- **WARNING**: Warning conditions
- **ERROR**: Error conditions
- **CRITICAL**: Critical conditions

### Metrics

Key metrics collected:

- **Request rates**: Requests per second
- **Response times**: Latency measurements
- **Error rates**: Failed request percentage
- **Resource usage**: CPU, memory, disk usage

### Health Checks

Regular health checks for system components:

- **Embedding providers**: API connectivity
- **Vector store**: Database connectivity
- **Knowledge graph**: Graph integrity
- **API endpoints**: Service availability

## Deployment Considerations

### Containerization

Dockerfile example:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "src.main", "serve"]
```

### Kubernetes Deployment

Example deployment manifest:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vector-kg-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vector-kg
  template:
    metadata:
      labels:
        app: vector-kg
    spec:
      containers:
      - name: vector-kg
        image: vector-kg:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: vector-kg-secrets
        volumeMounts:
        - name: data
          mountPath: /app/data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: vector-kg-pvc
```

### Scaling Strategies

1. **Horizontal scaling**: Multiple API instances
2. **Vertical scaling**: Larger instances for heavy workloads
3. **Database scaling**: Vector store optimization
4. **Caching layers**: External caching for high-traffic scenarios

## Security Considerations

### Authentication

Implement authentication for production deployments:

```python
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("EXPECTED_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.get("/protected-endpoint")
async def protected_endpoint(api_key: str = Depends(verify_api_key)):
    # Implementation
    pass
```

### Data Protection

1. **Encryption at rest**: Encrypt sensitive data
2. **Encryption in transit**: Use HTTPS/TLS
3. **Access controls**: Implement role-based access
4. **Audit logging**: Track data access and modifications

### Input Validation

All API inputs are validated using Pydantic models:

```python
class SearchQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    search_method: SearchMethod = SearchMethod.HYBRID
    # ... other fields with validation
```

## Troubleshooting

### Common Issues

#### 1. Embedding Provider Failures

**Symptoms**: Search returns empty results, slow performance

**Solutions**:
- Check API keys in environment variables
- Verify internet connectivity
- Check provider rate limits
- Enable fallback providers

#### 2. Vector Store Issues

**Symptoms**: Database errors, slow queries

**Solutions**:
- Check disk space
- Verify file permissions
- Restart vector store
- Reindex collections

#### 3. Memory Issues

**Symptoms**: High memory usage, slow performance

**Solutions**:
- Reduce cache sizes
- Limit batch sizes
- Add more memory
- Optimize data structures

### Debugging Tools

1. **Logging**: Enable DEBUG level logging
2. **Profiling**: Use Python profiling tools
3. **Monitoring**: Check system metrics
4. **Tracing**: Implement request tracing

### Performance Profiling

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()
# ... code to profile ...
profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## Contributing

### Code Style

Follow these guidelines:

1. **PEP 8**: Adhere to Python style guide
2. **Type hints**: Use type hints for all functions
3. **Docstrings**: Document public APIs
4. **Naming**: Use descriptive variable and function names

### Pull Request Process

1. **Fork repository**
2. **Create feature branch**
3. **Implement changes**
4. **Add tests**
5. **Update documentation**
6. **Submit pull request**

### Code Review Guidelines

1. **Functionality**: Does it work as intended?
2. **Tests**: Are there adequate tests?
3. **Documentation**: Is it properly documented?
4. **Style**: Does it follow coding standards?
5. **Performance**: Are there performance considerations?

## Versioning

The system follows semantic versioning:

- **MAJOR**: Breaking changes
- **MINOR**: New features
- **PATCH**: Bug fixes

### Release Process

1. **Update version** in `src/__init__.py`
2. **Update changelog**
3. **Tag release** in Git
4. **Publish package** (if applicable)

## Changelog

### v1.0.0 (Initial Release)
- Core vector search functionality
- Multi-provider embedding support
- Knowledge graph integration
- Temporal search capabilities
- Hybrid search coordination
- REST API with FastAPI
- Comprehensive documentation

### Future Enhancements
- Distributed vector storage
- Advanced graph algorithms
- Real-time indexing
- Multi-language support
- Plugin architecture

This developer guide provides comprehensive information for developers working with the Vector-Enhanced Temporal Knowledge Graph System. It covers architecture, components, development workflows, and best practices for extending and maintaining the system.