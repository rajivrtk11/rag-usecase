# User Guide

Vector-Enhanced Temporal Knowledge Graph System User Guide

## Overview

The Vector-Enhanced Temporal Knowledge Graph System is a comprehensive RAG (Retrieval-Augmented Generation) system that combines vector search, knowledge graphs, and temporal reasoning to provide intelligent information retrieval and analysis.

## Key Features

- **Multi-Modal Search**: Semantic vector search, graph-based search, and lexical search
- **Temporal Reasoning**: Time-aware search and temporal relationship analysis
- **Hybrid Retrieval**: Intelligent combination of different search methods
- **Multi-Provider Embeddings**: Support for Google, OpenAI, and local embedding models
- **Knowledge Graphs**: Structured representation of facts, entities, and relationships
- **High Performance**: Sub-2 second response times with efficient caching

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-impl-1

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/config.example.yaml config/config.yaml
cp .env.example .env
```

### 2. Configuration

Edit `config/config.yaml` to customize system behavior:

```yaml
embedding:
  primary_provider: "google"  # or "openai" or "sentence_transformers"
  
search:
  similarity_threshold: 0.7
  max_results: 50
  
temporal:
  enable_temporal_search: true
```

Set up environment variables in `.env`:

```bash
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Initialize the System

```bash
# Using CLI
python -m src.main initialize

# Or start the API server and call initialization endpoint
python -m src.main serve
# Then POST to http://localhost:8000/system/initialize
```

### 4. Add Your Data

#### Add Document Chunks

```bash
# Using the API
curl -X POST http://localhost:8000/content/chunks \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {
        "content": "Your document content here...",
        "source": "document.pdf",
        "domain": "technical"
      }
    ]
  }'
```

#### Add Structured Facts

```bash
curl -X POST http://localhost:8000/content/facts \
  -H "Content-Type: application/json" \
  -d '{
    "facts": [
      {
        "subject": "Python",
        "predicate": "is_used_for", 
        "object": "machine learning",
        "confidence": 0.9
      }
    ]
  }'
```

### 5. Search Your Data

#### Semantic Search

```bash
curl -X POST http://localhost:8000/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "context": {"max_results": 10}
  }'
```

#### Hybrid Search (Recommended)

```bash
curl -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence trends",
    "search_method": "hybrid",
    "context": {
      "max_results": 20,
      "domain_filter": "technical"
    }
  }'
```

## Search Methods

### 1. Semantic Search

Uses vector embeddings to find semantically similar content.

**Best for:**
- Natural language queries
- Concept-based search
- Finding related ideas even with different wording

**Example:**
```json
{
  "query": "natural language processing techniques",
  "search_method": "vector"
}
```

### 2. Graph Search

Searches through knowledge graph relationships.

**Best for:**
- Finding connections between entities
- Relationship-based queries
- Exploring entity neighborhoods

**Example:**
- "What is connected to Python in the knowledge graph?"
- "Find relationships between Machine Learning and Data Science"

### 3. Lexical Search

Traditional keyword-based search using TF-IDF.

**Best for:**
- Exact keyword matches
- Specific term searches
- Complement to semantic search

### 4. Hybrid Search (Recommended)

Intelligently combines all search methods based on query analysis.

**Benefits:**
- Automatic method selection
- Best of all worlds
- Improved accuracy and coverage

## Advanced Features

### Temporal Search

Search with time awareness and temporal constraints.

```json
{
  "query": "quarterly earnings reports",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z"
}
```

**Temporal Queries:**
- "What happened in 2023?"
- "Recent developments in AI"
- "Events before the product launch"

### Domain-Specific Search

Filter results by knowledge domain.

```json
{
  "query": "revenue analysis",
  "context": {
    "domain_filter": "business"
  }
}
```

**Available Domains:**
- `general`
- `technical`
- `business`
- `scientific`
- `medical`
- `legal`

### Confidence-Based Filtering

Filter results by confidence scores.

```json
{
  "query": "machine learning",
  "context": {
    "confidence_threshold": 0.8
  }
}
```

### Query Expansion

Automatically expand queries with synonyms and related terms.

```json
{
  "query": "AI applications",
  "enable_query_expansion": true
}
```

## CLI Usage

The system includes a powerful command-line interface.

### Initialize System

```bash
python -m src.main initialize
```

### Start API Server

```bash
python -m src.main serve --host 0.0.0.0 --port 8000
```

### Interactive Search

```bash
python -m src.main search
# Enter query when prompted
```

### Search with Parameters

```bash
python -m src.main search \
  --query "machine learning algorithms" \
  --method hybrid \
  --max-results 15
```

### View System Statistics

```bash
python -m src.main stats
```

### Reindex Collections

```bash
# Reindex all collections
python -m src.main reindex

# Reindex specific collection
python -m src.main reindex --collection document_chunks
```

### View Configuration

```bash
python -m src.main config
```

## Performance Optimization

### Embedding Providers

Choose the right embedding provider for your needs:

**Google (text-embedding-004):**
- Highest quality
- Requires API key
- Rate limits apply

**OpenAI (text-embedding-3-small):**
- High quality
- Requires API key
- Good performance/cost ratio

**SentenceTransformers (local):**
- No API required
- Runs locally
- Good for development/testing

### Caching

The system includes intelligent caching:

- **Result Cache**: Caches search results (5-minute TTL)
- **Embedding Cache**: Caches generated embeddings
- **Vector Cache**: Configurable vector cache size

### Batch Processing

Use batch operations for better performance:

```python
# Add multiple documents at once
chunks = [chunk1, chunk2, chunk3, ...]
response = requests.post('/content/chunks', json={'chunks': chunks})
```

### Configuration Tuning

Optimize configuration for your use case:

```yaml
performance:
  vector_cache_size: 2000      # Increase for more caching
  query_timeout_seconds: 60    # Increase for complex queries
  max_concurrent_queries: 100  # Increase for higher load

search:
  similarity_threshold: 0.6    # Lower for more results
  max_results: 100             # Adjust based on needs

embedding:
  batch_size: 64               # Increase for better throughput
```

## Monitoring and Maintenance

### Health Monitoring

Check system health regularly:

```bash
curl http://localhost:8000/system/health
```

### Performance Metrics

Monitor system performance:

```bash
curl http://localhost:8000/system/stats
```

### Log Analysis

Monitor application logs:

```bash
tail -f logs/app.log
```

### Collection Maintenance

Regularly maintain vector collections:

```bash
# Reindex collections for optimal performance
python -m src.main reindex

# Check collection statistics
curl http://localhost:8000/vectors/collections
```

## Troubleshooting

### Common Issues

#### 1. System Not Initialized

**Error:** "System is not initialized"

**Solution:**
```bash
python -m src.main initialize
# or
curl -X POST http://localhost:8000/system/initialize
```

#### 2. Embedding Provider Issues

**Error:** "No embedding providers available"

**Solutions:**
- Check API keys in `.env` file
- Verify internet connection for cloud providers
- Ensure SentenceTransformers is installed for local fallback

#### 3. Search Returns No Results

**Possible causes:**
- Similarity threshold too high
- No data indexed
- Query not matching content

**Solutions:**
- Lower similarity threshold
- Add more diverse content
- Try different search methods

#### 4. Performance Issues

**Symptoms:**
- Slow search responses
- High memory usage
- Timeouts

**Solutions:**
- Optimize configuration
- Increase cache sizes
- Use local embeddings for development
- Implement proper indexing

### Debug Mode

Enable debug logging:

```yaml
logging:
  level: "DEBUG"
```

### Testing

Run comprehensive tests:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/test_comprehensive.py::TestEmbeddingManager -v
pytest tests/test_comprehensive.py::TestPerformance -v
```

## Best Practices

### Data Preparation

1. **Clean your data**: Remove irrelevant content, normalize text
2. **Proper chunking**: Split documents into meaningful chunks
3. **Add metadata**: Include domain, confidence, temporal information
4. **Quality control**: Verify data before indexing

### Query Optimization

1. **Use specific queries**: More specific queries yield better results
2. **Leverage domains**: Filter by domain for better precision
3. **Combine methods**: Use hybrid search for best results
4. **Iterate and refine**: Analyze results and adjust queries

### System Configuration

1. **Match providers to use case**: Choose appropriate embedding providers
2. **Tune thresholds**: Adjust similarity and confidence thresholds
3. **Monitor performance**: Regular performance monitoring and optimization
4. **Plan for scale**: Configure system for expected load

### Security Considerations

1. **API Keys**: Keep API keys secure, rotate regularly
2. **Access Control**: Implement authentication in production
3. **Data Privacy**: Consider data sensitivity in embedding choices
4. **Network Security**: Use HTTPS in production deployments

## Advanced Use Cases

### Enterprise Knowledge Management

```yaml
# Configuration for enterprise use
knowledge_domains:
  - name: "hr_policies"
    confidence_threshold: 0.95
  - name: "technical_docs" 
    confidence_threshold: 0.9
  - name: "business_strategy"
    confidence_threshold: 0.85

search:
  max_results: 100
  hybrid_weights:
    vector: 0.5
    graph: 0.4
    lexical: 0.1
```

### Research and Development

```yaml
# Configuration for R&D use
temporal:
  enable_temporal_search: true
  temporal_decay_factor: 0.8  # Favor recent research
  
search:
  similarity_threshold: 0.6   # Cast wider net
  hybrid_weights:
    vector: 0.7               # Emphasize semantic similarity
    graph: 0.2
    lexical: 0.1
```

### Customer Support

```yaml
# Configuration for customer support
search:
  similarity_threshold: 0.8   # High precision
  max_results: 20
  
performance:
  query_timeout_seconds: 5    # Fast responses
  max_concurrent_queries: 200 # High concurrency
```

## Migration and Updates

### Upgrading

1. **Backup data**: Always backup vector stores and configuration
2. **Test in staging**: Test upgrades in non-production environment
3. **Gradual rollout**: Implement changes gradually
4. **Monitor metrics**: Watch for performance impacts

### Data Migration

```bash
# Export existing data
python scripts/export_data.py

# Upgrade system
pip install -r requirements.txt --upgrade

# Re-initialize
python -m src.main initialize

# Import data
python scripts/import_data.py
```

## Support and Community

### Documentation

- [API Reference](api_documentation.md)
- [Configuration Guide](configuration.md)
- [Developer Documentation](developer_guide.md)

### Getting Help

1. Check documentation and troubleshooting guide
2. Review system logs for error details
3. Test with minimal configuration
4. Create reproducible test case

### Contributing

1. Follow coding standards (Black, isort, mypy)
2. Add comprehensive tests
3. Update documentation
4. Submit pull requests with clear descriptions

## Conclusion

The Vector-Enhanced Temporal Knowledge Graph System provides a powerful platform for intelligent information retrieval. By combining multiple search methods, temporal reasoning, and flexible configuration, it can adapt to a wide variety of use cases from enterprise knowledge management to research applications.

Start with the basic setup, experiment with different search methods, and gradually optimize the configuration for your specific needs. The system's modular design allows for incremental adoption and continuous improvement.