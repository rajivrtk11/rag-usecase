"""
Vector store implementation using ChromaDB.

This module provides a comprehensive vector store manager with CRUD operations,
collection management, and integration with the embedding system.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from .config import get_config
from .embeddings import embedding_manager, get_embedding, get_embeddings_batch
from .models import (
    VectorMetadata, 
    VectorSearchResult, 
    CollectionInfo, 
    ContentType,
    DomainType,
    EmbeddingProvider,
    SearchMethod
)


logger = logging.getLogger(__name__)


class ChromaDBVectorStore:
    """ChromaDB-based vector store implementation."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize ChromaDB vector store."""
        self.config = get_config()
        self.persist_directory = persist_directory or self.config.vector_store.persist_directory
        self.client = None
        self.collections = {}
        self._initialized = False
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize ChromaDB client and collections."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding manager if not already done
            if not embedding_manager.providers:
                await embedding_manager.initialize()
            
            # Create or get collections
            await self._setup_collections()
            
            self._initialized = True
            logger.info(f"ChromaDB vector store initialized at {self.persist_directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB vector store: {e}")
            return False
    
    async def _setup_collections(self):
        """Set up vector collections based on configuration."""
        for collection_config in self.config.vector_store.collections:
            collection_name = collection_config["name"]
            embedding_function_name = collection_config["embedding_function"]
            
            try:
                # Create custom embedding function that uses our embedding manager
                embedding_function = CustomEmbeddingFunction(embedding_function_name)
                
                # Get or create collection
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    metadata={"description": f"Collection for {collection_name}"}
                )
                
                self.collections[collection_name] = collection
                logger.info(f"Set up collection: {collection_name}")
                
            except Exception as e:
                logger.error(f"Failed to setup collection {collection_name}: {e}")
    
    async def add_vectors(
        self, 
        collection_name: str,
        texts: List[str],
        metadatas: List[VectorMetadata],
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add vectors to a collection."""
        if not self._initialized:
            await self.initialize()
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return False
        
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]
            
            # Convert metadata objects to dictionaries
            metadata_dicts = []
            for metadata in metadatas:
                metadata_dict = metadata.dict()
                # Convert datetime objects to ISO format strings
                for key, value in metadata_dict.items():
                    if isinstance(value, datetime):
                        metadata_dict[key] = value.isoformat()
                    elif isinstance(value, uuid.UUID):
                        metadata_dict[key] = str(value)
                    elif value is None:
                        metadata_dict[key] = ""
                metadata_dicts.append(metadata_dict)
            
            collection = self.collections[collection_name]
            
            # Add documents to collection
            collection.add(
                documents=texts,
                metadatas=metadata_dicts,
                ids=ids
            )
            
            logger.info(f"Added {len(texts)} vectors to collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to collection {collection_name}: {e}")
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in a collection."""
        if not self._initialized:
            await self.initialize()
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return []
        
        try:
            collection = self.collections[collection_name]
            
            # Perform similarity search
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=metadata_filter
            )
            
            # Convert results to VectorSearchResult objects
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    # Extract data for this result
                    content_id = results["ids"][0][i]
                    content = results["documents"][0][i]
                    distance = results["distances"][0][i] if results.get("distances") else None
                    metadata_dict = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    # Convert distance to similarity score (cosine distance -> cosine similarity)
                    similarity_score = 1.0 - distance if distance is not None else 0.0
                    
                    # Apply similarity threshold filter
                    if similarity_threshold and similarity_score < similarity_threshold:
                        continue
                    
                    # Reconstruct metadata object
                    try:
                        # Convert string values back to appropriate types
                        processed_metadata = self._process_metadata_from_db(metadata_dict)
                        vector_metadata = VectorMetadata(**processed_metadata)
                    except Exception as e:
                        logger.warning(f"Failed to reconstruct metadata for result {content_id}: {e}")
                        # Create minimal metadata
                        vector_metadata = VectorMetadata(
                            document_id=uuid.uuid4(),
                            content_type=ContentType.CHUNK,
                            domain=DomainType.GENERAL,
                            confidence_score=0.5,
                            embedding_provider=EmbeddingProvider.GOOGLE
                        )
                    
                    # Create search result
                    search_result = VectorSearchResult(
                        content_id=uuid.UUID(content_id),
                        content=content,
                        similarity_score=similarity_score,
                        metadata=vector_metadata,
                        search_method=SearchMethod.VECTOR,
                        ranking_factors={"similarity": similarity_score}
                    )
                    
                    search_results.append(search_result)
            
            logger.debug(f"Found {len(search_results)} results in collection {collection_name}")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search collection {collection_name}: {e}")
            return []
    
    async def update_vectors(
        self,
        collection_name: str,
        ids: List[str],
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[VectorMetadata]] = None
    ) -> bool:
        """Update existing vectors in a collection."""
        if not self._initialized:
            await self.initialize()
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return False
        
        try:
            collection = self.collections[collection_name]
            
            update_data = {"ids": ids}
            
            if texts is not None:
                update_data["documents"] = texts
            
            if metadatas is not None:
                metadata_dicts = []
                for metadata in metadatas:
                    metadata_dict = metadata.dict()
                    # Convert datetime objects to ISO format strings
                    for key, value in metadata_dict.items():
                        if isinstance(value, datetime):
                            metadata_dict[key] = value.isoformat()
                        elif isinstance(value, uuid.UUID):
                            metadata_dict[key] = str(value)
                        elif value is None:
                            metadata_dict[key] = ""
                    metadata_dicts.append(metadata_dict)
                
                update_data["metadatas"] = metadata_dicts
            
            # Update documents in collection
            collection.update(**update_data)
            
            logger.info(f"Updated {len(ids)} vectors in collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update vectors in collection {collection_name}: {e}")
            return False
    
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> bool:
        """Delete vectors from a collection."""
        if not self._initialized:
            await self.initialize()
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return False
        
        try:
            collection = self.collections[collection_name]
            
            # Delete documents from collection
            collection.delete(ids=ids)
            
            logger.info(f"Deleted {len(ids)} vectors from collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from collection {collection_name}: {e}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """Get information about a collection."""
        if not self._initialized:
            await self.initialize()
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return None
        
        try:
            collection = self.collections[collection_name]
            
            # Get collection count
            count_result = collection.count()
            
            # Get collection metadata
            collection_metadata = collection.metadata or {}
            
            # Create collection info
            info = CollectionInfo(
                name=collection_name,
                description=collection_metadata.get("description"),
                embedding_function=EmbeddingProvider.GOOGLE,  # Default, should be configurable
                vector_count=count_result,
                last_updated=datetime.utcnow()
            )
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get info for collection {collection_name}: {e}")
            return None
    
    async def list_collections(self) -> List[CollectionInfo]:
        """List all collections."""
        if not self._initialized:
            await self.initialize()
        
        collections_info = []
        
        for collection_name in self.collections.keys():
            info = await self.get_collection_info(collection_name)
            if info:
                collections_info.append(info)
        
        return collections_info
    
    async def create_collection(
        self, 
        name: str, 
        embedding_provider: EmbeddingProvider = EmbeddingProvider.GOOGLE,
        description: Optional[str] = None
    ) -> bool:
        """Create a new collection."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create custom embedding function
            embedding_function = CustomEmbeddingFunction(embedding_provider.value)
            
            # Create collection
            collection = self.client.create_collection(
                name=name,
                embedding_function=embedding_function,
                metadata={"description": description or f"Collection {name}"}
            )
            
            self.collections[name] = collection
            logger.info(f"Created new collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            return False
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if not self._initialized:
            await self.initialize()
        
        try:
            self.client.delete_collection(name=name)
            
            if name in self.collections:
                del self.collections[name]
            
            logger.info(f"Deleted collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            return False
    
    def _process_metadata_from_db(self, metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata dictionary from database to convert types."""
        processed = metadata_dict.copy()
        
        # Convert string representations back to proper types
        datetime_fields = ["last_updated", "valid_from", "valid_to", "creation_date"]
        uuid_fields = ["chunk_id", "document_id"]
        
        for field in datetime_fields:
            if field in processed and processed[field] and processed[field] != "":
                try:
                    processed[field] = datetime.fromisoformat(processed[field])
                except (ValueError, TypeError):
                    processed[field] = None
        
        for field in uuid_fields:
            if field in processed and processed[field] and processed[field] != "":
                try:
                    processed[field] = uuid.UUID(processed[field])
                except (ValueError, TypeError):
                    # Generate new UUID if invalid
                    processed[field] = uuid.uuid4()
        
        # Ensure required fields have values
        if "chunk_id" not in processed or not processed["chunk_id"]:
            processed["chunk_id"] = uuid.uuid4()
        
        if "document_id" not in processed or not processed["document_id"]:
            processed["document_id"] = uuid.uuid4()
        
        # Convert enum strings back to enums
        if "content_type" in processed:
            try:
                processed["content_type"] = ContentType(processed["content_type"])
            except (ValueError, TypeError):
                processed["content_type"] = ContentType.CHUNK
        
        if "domain" in processed:
            try:
                processed["domain"] = DomainType(processed["domain"])
            except (ValueError, TypeError):
                processed["domain"] = DomainType.GENERAL
        
        if "embedding_provider" in processed:
            try:
                processed["embedding_provider"] = EmbeddingProvider(processed["embedding_provider"])
            except (ValueError, TypeError):
                processed["embedding_provider"] = EmbeddingProvider.GOOGLE
        
        # Set defaults for missing required fields
        defaults = {
            "confidence_score": 0.5,
            "vector_version": "1.0",
            "content_type": ContentType.CHUNK,
            "domain": DomainType.GENERAL,
            "embedding_provider": EmbeddingProvider.GOOGLE,
            "last_updated": datetime.utcnow(),
            "tags": []
        }
        
        for key, default_value in defaults.items():
            if key not in processed or processed[key] is None:
                processed[key] = default_value
        
        return processed
    
    async def reindex_collection(self, collection_name: str) -> bool:
        """Reindex a collection (rebuild embeddings)."""
        if not self._initialized:
            await self.initialize()
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return False
        
        try:
            collection = self.collections[collection_name]
            
            # Get all documents from collection
            all_data = collection.get()
            
            if not all_data["ids"]:
                logger.info(f"Collection {collection_name} is empty, nothing to reindex")
                return True
            
            logger.info(f"Reindexing {len(all_data['ids'])} documents in collection {collection_name}")
            
            # Delete and recreate collection
            self.client.delete_collection(name=collection_name)
            
            # Find the embedding provider for this collection
            embedding_provider = EmbeddingProvider.GOOGLE  # Default
            for collection_config in self.config.vector_store.collections:
                if collection_config["name"] == collection_name:
                    embedding_provider = EmbeddingProvider(collection_config["embedding_function"])
                    break
            
            # Recreate collection
            embedding_function = CustomEmbeddingFunction(embedding_provider.value)
            new_collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            
            self.collections[collection_name] = new_collection
            
            # Re-add all documents
            new_collection.add(
                documents=all_data["documents"],
                metadatas=all_data["metadatas"],
                ids=all_data["ids"]
            )
            
            logger.info(f"Successfully reindexed collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reindex collection {collection_name}: {e}")
            return False


class CustomEmbeddingFunction:
    """Custom embedding function that integrates with our embedding manager."""
    
    def __init__(self, provider_name: str):
        """Initialize with provider name."""
        self.provider_name = provider_name
        try:
            self.provider = EmbeddingProvider(provider_name)
        except ValueError:
            self.provider = EmbeddingProvider.GOOGLE
            logger.warning(f"Invalid provider {provider_name}, using Google as default")
    
    def __call__(self, input_texts: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts."""
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                embeddings = loop.run_until_complete(
                    get_embeddings_batch(input_texts, self.provider)
                )
            finally:
                loop.close()
            
            # Handle None results
            result = []
            for embedding in embeddings:
                if embedding is not None:
                    result.append(embedding)
                else:
                    # Generate zero embedding as fallback
                    result.append([0.0] * 384)  # Default embedding size
            
            return result
            
        except Exception as e:
            logger.error(f"Embedding function failed: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 384 for _ in input_texts]


# Global vector store instance
vector_store = ChromaDBVectorStore()


async def initialize_vector_store() -> bool:
    """Initialize the global vector store."""
    return await vector_store.initialize()