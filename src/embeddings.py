"""
Multi-provider embedding support with fallback strategies.

This module provides a unified interface for generating embeddings using
multiple providers (Google, OpenAI, SentenceTransformers) with automatic
fallback and error handling.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import get_config, get_api_key
from .models import EmbeddingProvider


logger = logging.getLogger(__name__)


class EmbeddingProviderBase(ABC):
    """Base class for embedding providers."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the embedding provider."""
        self.model_name = model_name
        self.kwargs = kwargs
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider. Returns True if successful."""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized


class GoogleEmbeddingProvider(EmbeddingProviderBase):
    """Google AI embedding provider."""
    
    def __init__(self, model_name: str = "text-embedding-004", **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = None
    
    async def initialize(self) -> bool:
        """Initialize Google AI client."""
        try:
            import google.generativeai as genai
            
            api_key = get_api_key("google")
            if not api_key:
                logger.warning("Google API key not found")
                return False
            
            genai.configure(api_key=api_key)
            self.client = genai
            self._initialized = True
            
            logger.info(f"Initialized Google embedding provider with model {self.model_name}")
            return True
            
        except ImportError:
            logger.error("Google AI library not installed. Install with: pip install google-ai-generativelanguage")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Google embedding provider: {e}")
            return False
    
    async def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text using Google AI."""
        if not self._initialized:
            return None
            
        try:
            result = self.client.embed_content(
                model=f"models/{self.model_name}",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Google embedding generation failed: {e}")
            return None
    
    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts."""
        if not self._initialized:
            return [None] * len(texts)
        
        results = []
        batch_size = get_config().embedding.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                embedding = await self.embed_text(text)
                batch_results.append(embedding)
            
            results.extend(batch_results)
            
            # Add small delay to respect rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return results


class OpenAIEmbeddingProvider(EmbeddingProviderBase):
    """OpenAI embedding provider."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = None
    
    async def initialize(self) -> bool:
        """Initialize OpenAI client."""
        try:
            import openai
            
            api_key = get_api_key("openai")
            if not api_key:
                logger.warning("OpenAI API key not found")
                return False
            
            self.client = openai.OpenAI(api_key=api_key)
            self._initialized = True
            
            logger.info(f"Initialized OpenAI embedding provider with model {self.model_name}")
            return True
            
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding provider: {e}")
            return False
    
    async def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text using OpenAI."""
        if not self._initialized:
            return None
            
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            return None
    
    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts."""
        if not self._initialized:
            return [None] * len(texts)
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = []
            for data_point in response.data:
                embeddings.append(data_point.embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding generation failed: {e}")
            return [None] * len(texts)


class SentenceTransformersProvider(EmbeddingProviderBase):
    """Local SentenceTransformers embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        super().__init__(model_name, **kwargs)
        self.model = None
        self.device = kwargs.get("device", "cpu")
    
    async def initialize(self) -> bool:
        """Initialize SentenceTransformer model."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._initialized = True
            
            logger.info(f"Initialized SentenceTransformer provider with model {self.model_name} on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer provider: {e}")
            return False
    
    async def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text using SentenceTransformers."""
        if not self._initialized:
            return None
            
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"SentenceTransformer embedding generation failed: {e}")
            return None
    
    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts."""
        if not self._initialized:
            return [None] * len(texts)
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            logger.error(f"SentenceTransformer batch embedding generation failed: {e}")
            return [None] * len(texts)


class EmbeddingManager:
    """Manager for multiple embedding providers with fallback support."""
    
    def __init__(self):
        """Initialize the embedding manager."""
        self.providers: Dict[EmbeddingProvider, EmbeddingProviderBase] = {}
        self.config = get_config()
        self._initialization_status: Dict[EmbeddingProvider, bool] = {}
        
        # Performance tracking
        self.stats = {
            'requests_total': 0,
            'requests_successful': 0,
            'requests_failed': 0,
            'fallback_used': 0,
            'provider_usage': {provider.value: 0 for provider in EmbeddingProvider}
        }
    
    async def initialize(self) -> bool:
        """Initialize all configured embedding providers."""
        success = False
        
        # Initialize providers based on configuration
        provider_configs = self.config.embedding.providers
        
        for provider_name, provider_config in provider_configs.items():
            try:
                provider_enum = EmbeddingProvider(provider_name)
                
                if provider_enum == EmbeddingProvider.GOOGLE:
                    provider = GoogleEmbeddingProvider(
                        model_name=provider_config.model,
                        **provider_config.dict(exclude={'model'})
                    )
                elif provider_enum == EmbeddingProvider.OPENAI:
                    provider = OpenAIEmbeddingProvider(
                        model_name=provider_config.model,
                        **provider_config.dict(exclude={'model'})
                    )
                elif provider_enum == EmbeddingProvider.SENTENCE_TRANSFORMERS:
                    provider = SentenceTransformersProvider(
                        model_name=provider_config.model,
                        device=provider_config.device
                    )
                else:
                    logger.warning(f"Unknown provider: {provider_name}")
                    continue
                
                # Initialize the provider
                init_success = await provider.initialize()
                self._initialization_status[provider_enum] = init_success
                
                if init_success:
                    self.providers[provider_enum] = provider
                    success = True
                    logger.info(f"Successfully initialized {provider_name} provider")
                else:
                    logger.warning(f"Failed to initialize {provider_name} provider")
                    
            except Exception as e:
                logger.error(f"Error initializing provider {provider_name}: {e}")
        
        if not success:
            logger.error("No embedding providers could be initialized")
        
        return success
    
    async def get_embedding(self, text: str, preferred_provider: Optional[EmbeddingProvider] = None) -> Optional[List[float]]:
        """Get embedding for text with fallback support."""
        if not text or not text.strip():
            return None
        
        self.stats['requests_total'] += 1
        
        # Determine provider order
        provider_order = self._get_provider_order(preferred_provider)
        
        for provider in provider_order:
            if provider not in self.providers:
                continue
            
            try:
                start_time = time.time()
                embedding = await self.providers[provider].embed_text(text.strip())
                
                if embedding is not None:
                    self.stats['requests_successful'] += 1
                    self.stats['provider_usage'][provider.value] += 1
                    
                    # Log fallback usage
                    if provider != provider_order[0]:
                        self.stats['fallback_used'] += 1
                        logger.info(f"Used fallback provider {provider.value}")
                    
                    elapsed_time = time.time() - start_time
                    logger.debug(f"Generated embedding using {provider.value} in {elapsed_time:.3f}s")
                    
                    return embedding
                    
            except Exception as e:
                logger.warning(f"Provider {provider.value} failed: {e}")
                continue
        
        self.stats['requests_failed'] += 1
        logger.error("All embedding providers failed")
        return None
    
    async def get_embeddings_batch(
        self, 
        texts: List[str], 
        preferred_provider: Optional[EmbeddingProvider] = None
    ) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts with fallback support."""
        if not texts:
            return []
        
        # Filter out empty texts but maintain indices
        text_indices = []
        filtered_texts = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                text_indices.append(i)
                filtered_texts.append(text.strip())
        
        if not filtered_texts:
            return [None] * len(texts)
        
        self.stats['requests_total'] += len(filtered_texts)
        
        # Determine provider order
        provider_order = self._get_provider_order(preferred_provider)
        
        for provider in provider_order:
            if provider not in self.providers:
                continue
            
            try:
                start_time = time.time()
                embeddings = await self.providers[provider].embed_batch(filtered_texts)
                
                # Check if we got valid results
                valid_count = sum(1 for emb in embeddings if emb is not None)
                
                if valid_count > 0:
                    self.stats['requests_successful'] += valid_count
                    self.stats['requests_failed'] += len(filtered_texts) - valid_count
                    self.stats['provider_usage'][provider.value] += valid_count
                    
                    # Log fallback usage
                    if provider != provider_order[0]:
                        self.stats['fallback_used'] += valid_count
                        logger.info(f"Used fallback provider {provider.value} for batch")
                    
                    elapsed_time = time.time() - start_time
                    logger.debug(f"Generated {valid_count} embeddings using {provider.value} in {elapsed_time:.3f}s")
                    
                    # Reconstruct full results array
                    results = [None] * len(texts)
                    for i, embedding in enumerate(embeddings):
                        original_index = text_indices[i]
                        results[original_index] = embedding
                    
                    return results
                    
            except Exception as e:
                logger.warning(f"Provider {provider.value} batch failed: {e}")
                continue
        
        self.stats['requests_failed'] += len(filtered_texts)
        logger.error("All embedding providers failed for batch")
        return [None] * len(texts)
    
    def _get_provider_order(self, preferred_provider: Optional[EmbeddingProvider] = None) -> List[EmbeddingProvider]:
        """Get the order of providers to try."""
        if preferred_provider and preferred_provider in self.providers:
            order = [preferred_provider]
        else:
            # Use configured primary provider first
            order = [self.config.embedding.primary_provider]
        
        # Add fallback providers
        if self.config.embedding.fallback_provider not in order:
            order.append(self.config.embedding.fallback_provider)
        
        if self.config.embedding.local_provider not in order:
            order.append(self.config.embedding.local_provider)
        
        # Filter to only include initialized providers
        return [p for p in order if p in self.providers]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics."""
        total_requests = self.stats['requests_total']
        
        stats = self.stats.copy()
        stats['success_rate'] = (
            self.stats['requests_successful'] / total_requests 
            if total_requests > 0 else 0.0
        )
        stats['fallback_rate'] = (
            self.stats['fallback_used'] / total_requests 
            if total_requests > 0 else 0.0
        )
        stats['initialized_providers'] = [
            provider.value for provider, status in self._initialization_status.items() 
            if status
        ]
        
        return stats
    
    def get_available_providers(self) -> List[EmbeddingProvider]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    async def health_check(self) -> Dict[EmbeddingProvider, bool]:
        """Check health of all providers."""
        results = {}
        
        test_text = "This is a test embedding."
        
        for provider_enum, provider in self.providers.items():
            try:
                embedding = await provider.embed_text(test_text)
                results[provider_enum] = embedding is not None
            except Exception:
                results[provider_enum] = False
        
        return results


# Global embedding manager instance
embedding_manager = EmbeddingManager()


async def get_embedding(text: str, preferred_provider: Optional[EmbeddingProvider] = None) -> Optional[List[float]]:
    """Convenience function to get embedding for text."""
    return await embedding_manager.get_embedding(text, preferred_provider)


async def get_embeddings_batch(
    texts: List[str], 
    preferred_provider: Optional[EmbeddingProvider] = None
) -> List[Optional[List[float]]]:
    """Convenience function to get embeddings for multiple texts."""
    return await embedding_manager.get_embeddings_batch(texts, preferred_provider)