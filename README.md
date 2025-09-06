# Vector-Enhanced Temporal Knowledge Graph System

A comprehensive RAG system with vector search capabilities, temporal reasoning, and hybrid retrieval.

## Features

- **Vector Search**: ChromaDB integration with multi-provider embeddings
- **Temporal Reasoning**: Time-aware knowledge graph and vector search
- **Hybrid Retrieval**: Combined vector, graph, and lexical search
- **Multi-Domain Support**: Domain-specific embeddings and search strategies
- **High Performance**: Sub-2 second response times with scalable architecture

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up configuration:
```bash
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your settings
```

3. Run the system:
```bash
python -m src.main
```

## Architecture

```
API Layer (FastAPI)
├── Query Processing Engine
│   ├── Vector Search
│   ├── Graph Search  
│   └── Hybrid Search Coordinator
├── Vector Store Layer
│   ├── ChromaDB Collections
│   ├── Embedding Manager
│   └── Metadata Store
└── Knowledge Graph Layer
    ├── NetworkX Graph
    ├── Temporal Validation
    └── Entity Resolution
```

## API Endpoints

- `POST /search/semantic` - Semantic vector search
- `POST /search/hybrid` - Combined vector + graph search  
- `GET /vectors/collections` - Vector collection management
- `POST /vectors/reindex` - Reindex vector collections

## Configuration

See `config/config.example.yaml` for configuration options.

## Testing

```bash
pytest tests/
```

## Development

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```