# Configuration Guide

Detailed configuration options for the Vector-Enhanced Temporal Knowledge Graph System

## Configuration File Structure

The system uses YAML configuration files located at `config/config.yaml`. A complete example configuration is provided in `config/config.example.yaml`.

## Vector Store Configuration

Configure the vector database and collections:

```yaml
vector_store:
  provider: "chromadb"
  persist_directory: "./data/vector_store"
  collections:
    - name: "document_chunks"
      embedding_function: "google"
    - name: "atomic_facts" 
      embedding_function: "google"
    - name: "temporal_events"
      embedding_function: "google"
    - name: "domain_entities"
      embedding_function: "google"
```

### Vector Store Options

- **provider**: Vector database provider (currently only "chromadb" supported)
- **persist_directory**: Directory for persistent storage
- **collections**: List of vector collections with their embedding functions

## Embedding Configuration

Configure embedding providers and their settings:

```yaml
embedding:
  primary_provider: "google"
  fallback_provider: "openai"
  local_provider: "sentence_transformers"
  batch_size: 32
  max_tokens: 512
  
  providers:
    google:
      model: "text-embedding-004"
      api_key_env: "GOOGLE_API_KEY"
    openai:
      model: "text-embedding-3-small"
      api_key_env: "OPENAI_API_KEY"
    sentence_transformers:
      model: "all-MiniLM-L6-v2"
      device: "cpu"
```

### Provider Configuration

#### Google AI
```yaml
google:
  model: "text-embedding-004"
  api_key_env: "GOOGLE_API_KEY"  # Environment variable name
```

#### OpenAI
```yaml
openai:
  model: "text-embedding-3-small"
  api_key_env: "OPENAI_API_KEY"
```

#### Sentence Transformers (Local)
```yaml
sentence_transformers:
  model: "all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda" for GPU
```

### Embedding Parameters

- **primary_provider**: Primary embedding provider
- **fallback_provider**: Fallback if primary fails
- **local_provider**: Local provider for offline use
- **batch_size**: Number of texts to embed in batch
- **max_tokens**: Maximum tokens per text

## Search Configuration

Configure search behavior and ranking:

```yaml
search:
  similarity_threshold: 0.7
  max_results: 50
  hybrid_weights:
    vector: 0.6
    graph: 0.3
    lexical: 0.1
  
  ranking:
    temporal_weight: 0.2
    confidence_weight: 0.3
    diversity_factor: 0.1
```

### Search Parameters

- **similarity_threshold**: Minimum similarity for vector search results
- **max_results**: Maximum results per search
- **hybrid_weights**: Weights for different search methods in hybrid search

### Ranking Factors

- **temporal_weight**: Weight for temporal relevance
- **confidence_weight**: Weight for confidence scores
- **diversity_factor**: Factor for result diversity

## Temporal Configuration

Configure temporal search features:

```yaml
temporal:
  enable_temporal_search: true
  temporal_decay_factor: 0.95
  max_temporal_range_days: 365
  temporal_resolution: "day"
```

### Temporal Parameters

- **enable_temporal_search**: Enable/disable temporal features
- **temporal_decay_factor**: Decay factor for temporal relevance (0.0-1.0)
- **max_temporal_range_days**: Maximum temporal range in days
- **temporal_resolution**: Granularity ("day", "hour", "minute")

## Knowledge Graph Configuration

Configure knowledge graph behavior:

```yaml
graph:
  backend: "networkx"
  max_depth: 3
  relationship_weights: true
  
knowledge_domains:
  - name: "general"
    confidence_threshold: 0.8
  - name: "technical"
    confidence_threshold: 0.9
  - name: "business"
    confidence_threshold: 0.85
```

### Graph Parameters

- **backend**: Graph database backend (currently only "networkx" supported)
- **max_depth**: Maximum depth for graph traversal
- **relationship_weights**: Enable/disable relationship weights

### Knowledge Domains

Define knowledge domains with confidence thresholds:

- **name**: Domain name
- **confidence_threshold**: Minimum confidence for domain-specific results

## API Configuration

Configure the API server:

```yaml
api:
  host: "localhost"
  port: 8000
  workers: 1
  reload: true
```

### API Parameters

- **host**: Host to bind the server to
- **port**: Port to listen on
- **workers**: Number of worker processes
- **reload**: Enable auto-reload for development

## Logging Configuration

Configure logging behavior:

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/app.log"
```

### Logging Parameters

- **level**: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
- **format**: Log message format
- **file**: Log file path

## Performance Configuration

Configure performance and optimization settings:

```yaml
performance:
  vector_cache_size: 1000
  query_timeout_seconds: 30
  max_concurrent_queries: 50
  enable_metrics: true
```

### Performance Parameters

- **vector_cache_size**: Size of vector result cache
- **query_timeout_seconds**: Query timeout in seconds
- **max_concurrent_queries**: Maximum concurrent queries
- **enable_metrics**: Enable/disable performance metrics collection

## Environment Variables

The system uses environment variables for sensitive configuration:

### Required Variables

```bash
# API Keys (set one or more based on providers used)
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key

# Optional overrides
CONFIG_PATH=./config/config.yaml
LOG_LEVEL=INFO
```

### Setting API Keys

1. **Google AI**:
   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```

2. **OpenAI**:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Configuration Examples

### Development Configuration

```yaml
vector_store:
  provider: "chromadb"
  persist_directory: "./data/vector_store_dev"
  collections:
    - name: "document_chunks"
      embedding_function: "sentence_transformers"

embedding:
  primary_provider: "sentence_transformers"
  fallback_provider: "sentence_transformers"
  local_provider: "sentence_transformers"
  batch_size: 8
  max_tokens: 512
  
  providers:
    sentence_transformers:
      model: "all-MiniLM-L6-v2"
      device: "cpu"

search:
  similarity_threshold: 0.5
  max_results: 20
  hybrid_weights:
    vector: 0.7
    graph: 0.2
    lexical: 0.1

api:
  host: "localhost"
  port: 8000
  workers: 1
  reload: true

logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/app_dev.log"

performance:
  vector_cache_size: 100
  query_timeout_seconds: 60
  max_concurrent_queries: 10
  enable_metrics: true
```

### Production Configuration

```yaml
vector_store:
  provider: "chromadb"
  persist_directory: "/var/lib/vector-store"
  collections:
    - name: "document_chunks"
      embedding_function: "google"
    - name: "atomic_facts"
      embedding_function: "google"

embedding:
  primary_provider: "google"
  fallback_provider: "openai"
  local_provider: "sentence_transformers"
  batch_size: 64
  max_tokens: 1024
  
  providers:
    google:
      model: "text-embedding-004"
      api_key_env: "GOOGLE_API_KEY"
    openai:
      model: "text-embedding-3-small"
      api_key_env: "OPENAI_API_KEY"
    sentence_transformers:
      model: "all-MiniLM-L6-v2"
      device: "cuda"

search:
  similarity_threshold: 0.75
  max_results: 100
  hybrid_weights:
    vector: 0.6
    graph: 0.3
    lexical: 0.1

temporal:
  enable_temporal_search: true
  temporal_decay_factor: 0.9
  max_temporal_range_days: 730
  temporal_resolution: "day"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "/var/log/vector-kg/app.log"

performance:
  vector_cache_size: 10000
  query_timeout_seconds: 30
  max_concurrent_queries: 200
  enable_metrics: true
```

### Enterprise Configuration

```yaml
vector_store:
  provider: "chromadb"
  persist_directory: "/mnt/vector-store"
  collections:
    - name: "hr_policies"
      embedding_function: "openai"
    - name: "technical_docs"
      embedding_function: "google"
    - name: "business_strategy"
      embedding_function: "sentence_transformers"

embedding:
  primary_provider: "google"
  fallback_provider: "openai"
  local_provider: "sentence_transformers"
  batch_size: 128
  max_tokens: 2048
  
  providers:
    google:
      model: "text-embedding-004"
      api_key_env: "GOOGLE_API_KEY"
    openai:
      model: "text-embedding-3-large"
      api_key_env: "OPENAI_API_KEY"
    sentence_transformers:
      model: "all-mpnet-base-v2"
      device: "cuda"

search:
  similarity_threshold: 0.8
  max_results: 200
  hybrid_weights:
    vector: 0.5
    graph: 0.4
    lexical: 0.1

  ranking:
    temporal_weight: 0.3
    confidence_weight: 0.4
    diversity_factor: 0.2

temporal:
  enable_temporal_search: true
  temporal_decay_factor: 0.85
  max_temporal_range_days: 1825  # 5 years
  temporal_resolution: "hour"

graph:
  backend: "networkx"
  max_depth: 5
  relationship_weights: true

knowledge_domains:
  - name: "hr_policies"
    confidence_threshold: 0.95
  - name: "technical_docs"
    confidence_threshold: 0.9
  - name: "business_strategy"
    confidence_threshold: 0.85
  - name: "legal"
    confidence_threshold: 0.98
  - name: "compliance"
    confidence_threshold: 0.97

api:
  host: "0.0.0.0"
  port: 8080
  workers: 8
  reload: false

logging:
  level: "WARNING"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "/var/log/vector-kg/enterprise.log"

performance:
  vector_cache_size: 50000
  query_timeout_seconds: 15
  max_concurrent_queries: 500
  enable_metrics: true
```

## Configuration Validation

The system validates configuration at startup and provides helpful error messages for invalid configurations.

### Common Configuration Errors

1. **Missing API Keys**: Ensure required API keys are set in environment variables
2. **Invalid Provider Names**: Check spelling of provider names
3. **Invalid Paths**: Verify directory paths exist and are writable
4. **Invalid Values**: Ensure numeric values are within valid ranges

### Configuration Reload

The system supports configuration reload without restart:

```bash
# Send SIGHUP to reload configuration
kill -HUP <process_id>
```

Or through the API:

```bash
curl -X POST http://localhost:8000/system/reload-config
```

## Best Practices

### Security

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive configuration
3. **Restrict file permissions** on configuration files
4. **Use HTTPS** in production deployments

### Performance

1. **Tune batch sizes** based on available memory
2. **Use GPU** for local embeddings when available
3. **Adjust cache sizes** based on usage patterns
4. **Monitor resource usage** and adjust accordingly

### Maintenance

1. **Regular backups** of vector stores
2. **Monitor logs** for errors and warnings
3. **Update providers** to latest versions
4. **Review configuration** periodically for optimization

### Scalability

1. **Start with defaults** and tune based on usage
2. **Scale workers** based on CPU cores
3. **Increase cache sizes** for high-traffic systems
4. **Consider sharding** for very large datasets

This configuration guide provides a comprehensive overview of all available configuration options. Start with the example configuration and adjust based on your specific requirements.