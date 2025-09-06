# API Documentation

Vector-Enhanced Temporal Knowledge Graph System API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production deployments, implement appropriate authentication mechanisms.

## Response Format

All API endpoints return responses in the following format:

```json
{
  "success": boolean,
  "data": any | null,
  "error": string | null,
  "timestamp": "ISO 8601 datetime",
  "request_id": "UUID"
}
```

## System Management

### Initialize System

Initialize all system components including embedding providers and vector stores.

**POST** `/system/initialize`

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "System initialized successfully",
    "embedding_providers": ["google", "openai", "sentence_transformers"],
    "vector_collections": ["document_chunks", "atomic_facts", "temporal_events"]
  }
}
```

### Health Check

Check system health and component status.

**GET** `/system/health`

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "embedding_providers": {
      "google": true,
      "openai": false,
      "sentence_transformers": true
    },
    "vector_collections": 4,
    "uptime_seconds": 3600
  }
}
```

### System Statistics

Get comprehensive system statistics and performance metrics.

**GET** `/system/stats`

**Response:**
```json
{
  "success": true,
  "data": {
    "api_stats": {
      "total_requests": 150,
      "successful_requests": 145,
      "failed_requests": 5
    },
    "embedding_stats": {
      "requests_total": 50,
      "success_rate": 0.96
    },
    "semantic_search_stats": {...},
    "hybrid_search_stats": {...}
  }
}
```

## Search Operations

### Semantic Search

Perform semantic vector search using embeddings.

**POST** `/search/semantic`

**Request Body:**
```json
{
  "query": "artificial intelligence machine learning",
  "search_method": "vector",
  "context": {
    "max_results": 10,
    "domain_filter": "technical",
    "confidence_threshold": 0.7
  },
  "similarity_threshold": 0.8
}
```

**Response:**
```json
{
  "query": "artificial intelligence machine learning",
  "results": [
    {
      "content_id": "uuid",
      "content": "Machine learning is a subset of AI...",
      "similarity_score": 0.92,
      "metadata": {
        "domain": "technical",
        "confidence_score": 0.85,
        "source": "ml_textbook.pdf"
      },
      "search_method": "vector",
      "ranking_factors": {
        "similarity": 0.92,
        "confidence": 0.85
      }
    }
  ],
  "total_results": 1,
  "search_time_ms": 245.6,
  "search_method": "vector"
}
```

### Hybrid Search

Combine vector, graph, and lexical search methods.

**POST** `/search/hybrid`

**Request Body:**
```json
{
  "query": "company revenue growth trends",
  "search_method": "hybrid",
  "context": {
    "max_results": 20,
    "domain_filter": "business"
  },
  "enable_query_expansion": true
}
```

### Temporal Search

Perform time-aware search with temporal constraints.

**POST** `/search/temporal`

**Query Parameters:**
- `start_date` (optional): ISO 8601 datetime
- `end_date` (optional): ISO 8601 datetime

**Request Body:**
```json
{
  "query": "quarterly earnings report",
  "context": {
    "max_results": 15
  }
}
```

## Vector Collection Management

### List Collections

Get all available vector collections.

**GET** `/vectors/collections`

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "name": "document_chunks",
      "vector_count": 1250,
      "embedding_function": "google",
      "last_updated": "2024-01-06T10:30:00Z"
    }
  ]
}
```

### Get Collection Info

Get detailed information about a specific collection.

**GET** `/vectors/collections/{collection_name}`

**Response:**
```json
{
  "success": true,
  "data": {
    "name": "document_chunks",
    "description": "Document chunks collection",
    "vector_count": 1250,
    "embedding_function": "google",
    "content_type_distribution": {
      "chunk": 1200,
      "fact": 50
    }
  }
}
```

### Create Collection

Create a new vector collection.

**POST** `/vectors/collections/{collection_name}`

**Query Parameters:**
- `description` (optional): Collection description

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Collection 'new_collection' created successfully"
  }
}
```

### Delete Collection

Delete a vector collection.

**DELETE** `/vectors/collections/{collection_name}`

### Reindex Collections

Rebuild embeddings for vector collections.

**POST** `/vectors/reindex`

**Request Body:**
```json
{
  "collection_names": ["document_chunks", "atomic_facts"]
}
```

## Content Management

### Add Document Chunks

Add document chunks to the vector store.

**POST** `/content/chunks`

**Query Parameters:**
- `collection_name` (optional): Target collection, defaults to "document_chunks"

**Request Body:**
```json
{
  "chunks": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid", 
      "content": "This is document content...",
      "chunk_index": 0,
      "source": "document.pdf",
      "domain": "technical",
      "confidence_score": 0.9
    }
  ]
}
```

### Add Atomic Facts

Add structured facts to the knowledge graph and vector store.

**POST** `/content/facts`

**Request Body:**
```json
{
  "facts": [
    {
      "fact_id": "uuid",
      "subject": "Python",
      "predicate": "is_used_for",
      "object": "machine learning",
      "confidence": 0.95,
      "source_chunk_id": "uuid",
      "source_document_id": "uuid",
      "domain": "technical"
    }
  ]
}
```

### Add Temporal Events

Add temporal events to the knowledge graph.

**POST** `/content/events`

**Request Body:**
```json
{
  "events": [
    {
      "event_id": "uuid",
      "event_type": "product_launch",
      "description": "Company launched new AI platform",
      "start_time": "2024-01-15T09:00:00Z",
      "primary_entities": ["Company", "AI Platform"],
      "confidence": 0.9,
      "domain": "business"
    }
  ]
}
```

## Knowledge Graph Operations

### Get Entity Neighbors

Retrieve neighboring entities in the knowledge graph.

**GET** `/graph/entities/{entity_name}/neighbors`

**Query Parameters:**
- `depth`: Search depth (1-5)
- `relationship_filter` (optional): Filter by relationship type

**Response:**
```json
{
  "success": true,
  "data": {
    "entity": "Python",
    "depth": 2,
    "neighbors": [
      {
        "name": "Machine Learning",
        "type": "entity",
        "distance": 1,
        "relationships": [
          {
            "predicate": "is_used_for",
            "confidence": 0.95
          }
        ]
      }
    ],
    "count": 1
  }
}
```

### Find Entity Paths

Find paths between two entities in the knowledge graph.

**GET** `/graph/paths`

**Query Parameters:**
- `start_entity`: Starting entity name
- `end_entity`: Target entity name  
- `max_depth` (optional): Maximum path depth

**Response:**
```json
{
  "success": true,
  "data": {
    "start_entity": "Python",
    "end_entity": "Data Science",
    "paths": [
      ["entity_python", "entity_machine_learning", "entity_data_science"]
    ],
    "path_count": 1
  }
}
```

## Configuration

### Get Configuration

Retrieve current system configuration (sensitive values hidden).

**GET** `/config`

**Response:**
```json
{
  "success": true,
  "data": {
    "embedding": {
      "primary_provider": "google",
      "batch_size": 32
    },
    "search": {
      "similarity_threshold": 0.7,
      "max_results": 50
    },
    "temporal": {
      "enable_temporal_search": true
    }
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "data": null,
  "error": "Detailed error message",
  "timestamp": "2024-01-06T10:30:00Z",
  "request_id": "uuid"
}
```

### Common HTTP Status Codes

- `200 OK`: Successful operation
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: System error
- `503 Service Unavailable`: System not initialized

## Rate Limits

Current implementation does not enforce rate limits. For production use:

- Implement rate limiting per API key/user
- Consider request throttling for resource-intensive operations
- Monitor system performance and adjust limits accordingly

## SDK Examples

### Python

```python
import requests

# Initialize system
response = requests.post('http://localhost:8000/system/initialize')
print(response.json())

# Perform search
search_data = {
    "query": "machine learning algorithms",
    "search_method": "hybrid",
    "context": {"max_results": 10}
}

response = requests.post(
    'http://localhost:8000/search/hybrid',
    json=search_data
)

results = response.json()
for result in results['results']:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
```

### JavaScript

```javascript
// Initialize system
const initResponse = await fetch('http://localhost:8000/system/initialize', {
    method: 'POST'
});
const initData = await initResponse.json();

// Perform search
const searchData = {
    query: 'artificial intelligence',
    search_method: 'semantic',
    context: { max_results: 5 }
};

const searchResponse = await fetch('http://localhost:8000/search/semantic', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(searchData)
});

const results = await searchResponse.json();
console.log(results);
```

### cURL

```bash
# Initialize system
curl -X POST http://localhost:8000/system/initialize

# Search
curl -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "revenue growth analysis",
    "context": {"max_results": 10}
  }'

# Get system stats
curl http://localhost:8000/system/stats
```