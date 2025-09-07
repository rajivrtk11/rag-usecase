# RAG System for PDF Question Answering

A Retrieval-Augmented Generation (RAG) system that can read PDF documents and answer questions about their content using Google AI services.

## Features

- PDF text extraction and chunking
- Vector embeddings using Google AI
- ChromaDB for vector storage
- Question answering using Google Gemini
- RESTful API for easy integration
- Command-line interface for direct usage

## Prerequisites

- Python 3.8+
- Google AI API key (for embeddings and question answering)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

### Command Line Interface

1. Initialize the system:
   ```bash
   python main.py initialize
   ```

2. Start the API server:
   ```bash
   python main.py serve
   ```

3. Query directly from command line:
   ```bash
   python main.py query "Your question here"
   ```

### API Endpoints

1. Upload a PDF:
   ```bash
   curl -X POST "http://localhost:8000/upload-pdf/" -F "file=@path/to/document.pdf"
   ```

2. Query the system:
   ```bash
   curl -X POST "http://localhost:8000/query/" -F "query_text=Your question here"
   ```

3. Health check:
   ```bash
   curl "http://localhost:8000/health/"
   ```

## System Architecture

The system consists of several components:

1. **PDF Processor**: Extracts text from PDF files and splits it into manageable chunks
2. **Embedding Provider**: Generates vector embeddings using Google AI
3. **Vector Store**: Stores document chunks and their embeddings using ChromaDB
4. **LLM Provider**: Generates answers using Google Gemini
5. **RAG System**: Orchestrates the entire process
6. **API**: Provides RESTful endpoints for interaction

## Configuration

The system can be configured using environment variables:

- `GOOGLE_API_KEY`: Your Google AI API key
- `CHROMA_PERSIST_DIRECTORY`: Directory for ChromaDB persistence (default: ./data/chroma)
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

## License

This project is licensed under the MIT License.