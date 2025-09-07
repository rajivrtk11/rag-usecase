import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid
from src.config import settings
from src.embeddings import embedding_provider
import asyncio

class VectorStore:
    """Vector store using ChromaDB for storing and retrieving document embeddings."""
    
    def __init__(self):
        self.persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        self.client = None
        self.collection = None
        
    async def initialize(self) -> bool:
        """Initialize the vector store."""
        try:
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"description": "Document chunks for RAG system"}
            )
            
            return True
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            return False
    
    async def add_documents(self, documents: List[Dict]) -> bool:
        """Add documents to the vector store."""
        if not self.collection:
            print("Vector store not initialized")
            return False
            
        try:
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                ids.append(doc.get("id", str(uuid.uuid4())))
                texts.append(doc["text"])
                metadatas.append({
                    "source": doc.get("source", ""),
                    "chunk_index": doc.get("chunk_index", 0)
                })
            
            # Generate embeddings
            embeddings = []
            for text in texts:
                embedding = await embedding_provider.get_embedding(text)
                if embedding:
                    embeddings.append(embedding)
                else:
                    print(f"Failed to generate embedding for document")
                    return False
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            return True
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            return False
    
    async def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar documents."""
        if not self.collection:
            print("Vector store not initialized")
            return []
            
        try:
            # Generate query embedding
            query_embedding = await embedding_provider.get_embedding(query)
            if not query_embedding:
                print("Failed to generate query embedding")
                return []
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
                
            return formatted_results
        except Exception as e:
            print(f"Error searching vector store: {str(e)}")
            return []