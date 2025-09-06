"""
Hybrid search coordinator integrating vector, graph, and lexical search.

This module orchestrates different search modes, handles result fusion,
and provides intelligent query routing for optimal search performance.
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import get_config
from .models import (
    SearchQuery,
    SearchResponse,
    VectorSearchResult,
    QueryContext,
    SearchMethod,
    HybridSearchWeights,
    ContentType,
    DomainType
)
from .semantic_search import semantic_search_engine
from .knowledge_graph import temporal_knowledge_graph


logger = logging.getLogger(__name__)


class LexicalSearchEngine:
    """Simple lexical/keyword-based search engine."""
    
    def __init__(self):
        """Initialize lexical search engine."""
        self.config = get_config()
        self.document_index = {}  # Document ID to content mapping
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.documents = []
        self._index_built = False
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for lexical search."""
        try:
            self.documents = documents
            self.document_index = {doc['id']: doc for doc in documents}
            
            # Extract text content for TF-IDF
            texts = [doc.get('content', '') for doc in documents]
            
            if texts:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                self._index_built = True
                
                logger.info(f"Indexed {len(documents)} documents for lexical search")
            
        except Exception as e:
            logger.error(f"Failed to index documents for lexical search: {e}")
            self._index_built = False
    
    def search(self, query: str, max_results: int = 50) -> List[VectorSearchResult]:
        """Perform lexical search using TF-IDF."""
        if not self._index_built or not self.documents:
            return []
        
        try:
            # Transform query using the fitted vectorizer
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarity_scores)[::-1][:max_results]
            
            results = []
            for idx in top_indices:
                if similarity_scores[idx] > 0.0:  # Only include non-zero similarities
                    doc = self.documents[idx]
                    
                    # Create search result
                    result = self._create_lexical_result(doc, similarity_scores[idx], query)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Lexical search failed: {e}")
            return []
    
    def _create_lexical_result(
        self, 
        document: Dict[str, Any], 
        similarity_score: float,
        query: str
    ) -> VectorSearchResult:
        """Create VectorSearchResult from lexical search."""
        # Import here to avoid circular imports
        from .models import VectorMetadata, EmbeddingProvider
        import uuid
        
        # Create metadata
        metadata = VectorMetadata(
            document_id=uuid.UUID(document.get('document_id', str(uuid.uuid4()))),
            content_type=ContentType.CHUNK,
            domain=DomainType(document.get('domain', 'general')),
            confidence_score=document.get('confidence', 0.7),
            embedding_provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            source=document.get('source', 'lexical_search')
        )
        
        # Highlight matching terms in content
        highlighted_content = self._highlight_terms(document.get('content', ''), query)
        
        return VectorSearchResult(
            content_id=uuid.UUID(document.get('id', str(uuid.uuid4()))),
            content=document.get('content', ''),
            similarity_score=similarity_score,
            metadata=metadata,
            search_method=SearchMethod.LEXICAL,
            ranking_factors={'tfidf_score': similarity_score, 'lexical_match': 1.0},
            highlighted_text=highlighted_content
        )
    
    def _highlight_terms(self, content: str, query: str) -> str:
        """Highlight query terms in content."""
        try:
            # Simple highlighting - wrap matching words with markers
            query_words = query.lower().split()
            highlighted = content
            
            for word in query_words:
                if len(word) > 2:  # Only highlight longer words
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    highlighted = pattern.sub(f'**{word}**', highlighted)
            
            return highlighted
            
        except Exception as e:
            logger.warning(f"Failed to highlight terms: {e}")
            return content


class QueryRouter:
    """Routes queries to appropriate search methods based on query analysis."""
    
    def __init__(self):
        """Initialize query router."""
        self.config = get_config()
        
        # Query patterns for different search methods
        self.vector_patterns = [
            r'\b(?:similar|like|related|semantic|meaning)\b',
            r'\b(?:concept|idea|notion|understanding)\b'
        ]
        
        self.graph_patterns = [
            r'\b(?:relationship|connection|link|association)\b',
            r'\b(?:path|route|journey|sequence)\b',
            r'\b(?:between|connects|relates|links)\b'
        ]
        
        self.temporal_patterns = [
            r'\b(?:when|time|date|period|during|before|after)\b',
            r'\b(?:history|timeline|chronology|sequence)\b',
            r'\b\d{4}\b',  # Years
            r'\b(?:yesterday|today|tomorrow|recent|latest)\b'
        ]
        
        self.lexical_patterns = [
            r'\b(?:exact|specific|precise|literal)\b',
            r'"[^"]+"',  # Quoted strings
            r'\b(?:contains|includes|has|with)\b'
        ]
    
    def determine_search_method(self, search_query: SearchQuery) -> SearchMethod:
        """Determine the best search method for a query."""
        # If method is explicitly specified, use it
        if search_query.search_method != SearchMethod.HYBRID:
            return search_query.search_method
        
        query_text = search_query.query.lower()
        
        # Score different search methods
        scores = {
            SearchMethod.VECTOR: 0,
            SearchMethod.GRAPH: 0,
            SearchMethod.LEXICAL: 0
        }
        
        # Check vector search patterns
        for pattern in self.vector_patterns:
            if re.search(pattern, query_text):
                scores[SearchMethod.VECTOR] += 1
        
        # Check graph search patterns
        for pattern in self.graph_patterns:
            if re.search(pattern, query_text):
                scores[SearchMethod.GRAPH] += 1
        
        # Check lexical search patterns
        for pattern in self.lexical_patterns:
            if re.search(pattern, query_text):
                scores[SearchMethod.LEXICAL] += 1
        
        # Temporal queries favor vector + graph combination
        temporal_score = 0
        for pattern in self.temporal_patterns:
            if re.search(pattern, query_text):
                temporal_score += 1
        
        if temporal_score > 0:
            scores[SearchMethod.VECTOR] += temporal_score * 0.5
            scores[SearchMethod.GRAPH] += temporal_score * 0.5
        
        # Check for entity references (favor graph search)
        if self._has_entity_references(query_text):
            scores[SearchMethod.GRAPH] += 2
        
        # Check query length and complexity
        word_count = len(query_text.split())
        if word_count <= 3:
            # Short queries favor lexical search
            scores[SearchMethod.LEXICAL] += 1
        elif word_count >= 10:
            # Long queries favor vector search
            scores[SearchMethod.VECTOR] += 1
        
        # Determine best method
        if max(scores.values()) == 0:
            return SearchMethod.HYBRID  # Default to hybrid if no clear winner
        
        best_method = max(scores.keys(), key=lambda k: scores[k])
        
        # Use hybrid if multiple methods have similar scores
        max_score = max(scores.values())
        high_scoring_methods = [method for method, score in scores.items() if score >= max_score - 1]
        
        if len(high_scoring_methods) > 1:
            return SearchMethod.HYBRID
        
        return best_method
    
    def _has_entity_references(self, query: str) -> bool:
        """Check if query contains potential entity references."""
        # Look for capitalized words or quoted strings
        capitalized_pattern = r'\b[A-Z][a-z]+\b'
        quoted_pattern = r'"[^"]+"'
        
        return bool(re.search(capitalized_pattern, query) or re.search(quoted_pattern, query))


class ResultFuser:
    """Fuses results from different search methods."""
    
    def __init__(self):
        """Initialize result fuser."""
        self.config = get_config()
    
    def fuse_results(
        self,
        vector_results: List[VectorSearchResult],
        graph_results: List[VectorSearchResult],
        lexical_results: List[VectorSearchResult],
        weights: HybridSearchWeights,
        max_results: int
    ) -> List[VectorSearchResult]:
        """Fuse results from different search methods."""
        try:
            # Combine all results
            all_results = []
            
            # Add weighted scores to vector results
            for result in vector_results:
                weighted_result = result.copy(deep=True)
                weighted_result.ranking_factors['fusion_score'] = (
                    result.similarity_score * weights.vector
                )
                weighted_result.ranking_factors['method_weight'] = weights.vector
                all_results.append(weighted_result)
            
            # Add weighted scores to graph results
            for result in graph_results:
                weighted_result = result.copy(deep=True)
                weighted_result.ranking_factors['fusion_score'] = (
                    result.similarity_score * weights.graph
                )
                weighted_result.ranking_factors['method_weight'] = weights.graph
                all_results.append(weighted_result)
            
            # Add weighted scores to lexical results
            for result in lexical_results:
                weighted_result = result.copy(deep=True)
                weighted_result.ranking_factors['fusion_score'] = (
                    result.similarity_score * weights.lexical
                )
                weighted_result.ranking_factors['method_weight'] = weights.lexical
                all_results.append(weighted_result)
            
            # Remove duplicates based on content similarity
            unique_results = self._remove_duplicates(all_results)
            
            # Sort by fusion score
            unique_results.sort(
                key=lambda r: r.ranking_factors.get('fusion_score', 0),
                reverse=True
            )
            
            # Apply additional ranking factors
            final_results = self._apply_hybrid_ranking(unique_results)
            
            return final_results[:max_results]
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            return []
    
    def _remove_duplicates(self, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """Remove duplicate results based on content similarity."""
        if len(results) <= 1:
            return results
        
        unique_results = []
        similarity_threshold = 0.85
        
        for result in results:
            is_duplicate = False
            
            for existing_result in unique_results:
                # Check content similarity
                similarity = self._calculate_text_similarity(
                    result.content, existing_result.content
                )
                
                if similarity > similarity_threshold:
                    # Keep the result with higher score
                    existing_score = existing_result.ranking_factors.get('fusion_score', 0)
                    current_score = result.ranking_factors.get('fusion_score', 0)
                    
                    if current_score > existing_score:
                        # Replace existing result
                        unique_results.remove(existing_result)
                        unique_results.append(result)
                    
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _apply_hybrid_ranking(self, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """Apply additional ranking factors for hybrid results."""
        for result in results:
            # Boost results that appear in multiple methods
            method_diversity = len(set([
                factor.split('_')[0] for factor in result.ranking_factors.keys()
                if factor.endswith('_score') or factor.endswith('_match')
            ]))
            
            if method_diversity > 1:
                result.ranking_factors['diversity_boost'] = 0.1
                current_score = result.ranking_factors.get('fusion_score', 0)
                result.ranking_factors['fusion_score'] = min(current_score + 0.1, 1.0)
        
        return results


class HybridSearchCoordinator:
    """Main coordinator for hybrid search operations."""
    
    def __init__(self):
        """Initialize hybrid search coordinator."""
        self.config = get_config()
        self.query_router = QueryRouter()
        self.result_fuser = ResultFuser()
        self.lexical_engine = LexicalSearchEngine()
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'method_usage': defaultdict(int),
            'average_response_times': defaultdict(float),
            'fusion_efficiency': 0.0
        }
    
    async def search(self, search_query: SearchQuery) -> SearchResponse:
        """Perform hybrid search coordinating multiple search methods."""
        start_time = time.time()
        
        try:
            # Determine optimal search strategy
            optimal_method = self.query_router.determine_search_method(search_query)
            
            # Initialize results containers
            vector_results = []
            graph_results = []
            lexical_results = []
            
            # Get hybrid search weights
            weights = HybridSearchWeights(**self.config.search.hybrid_weights)
            
            # Execute searches based on determined method
            if optimal_method == SearchMethod.HYBRID or optimal_method == SearchMethod.VECTOR:
                vector_results = await self._perform_vector_search(search_query)
            
            if optimal_method == SearchMethod.HYBRID or optimal_method == SearchMethod.GRAPH:
                graph_results = await self._perform_graph_search(search_query)
            
            if optimal_method == SearchMethod.HYBRID or optimal_method == SearchMethod.LEXICAL:
                lexical_results = await self._perform_lexical_search(search_query)
            
            # Fuse results if hybrid search
            if optimal_method == SearchMethod.HYBRID:
                final_results = self.result_fuser.fuse_results(
                    vector_results,
                    graph_results,
                    lexical_results,
                    weights,
                    search_query.context.max_results
                )
                search_method = SearchMethod.HYBRID
            else:
                # Use results from single method
                if optimal_method == SearchMethod.VECTOR:
                    final_results = vector_results[:search_query.context.max_results]
                elif optimal_method == SearchMethod.GRAPH:
                    final_results = graph_results[:search_query.context.max_results]
                else:
                    final_results = lexical_results[:search_query.context.max_results]
                
                search_method = optimal_method
            
            # Calculate response time and update statistics
            response_time = (time.time() - start_time) * 1000
            self._update_stats(search_method, response_time, len(final_results))
            
            # Create search response
            response = SearchResponse(
                query=search_query.query,
                results=final_results,
                total_results=len(final_results),
                search_time_ms=response_time,
                search_method=search_method,
                domains_found=list(set(r.metadata.domain for r in final_results)),
                content_types_found=list(set(r.metadata.content_type for r in final_results))
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            
            # Return empty response on error
            response_time = (time.time() - start_time) * 1000
            return SearchResponse(
                query=search_query.query,
                results=[],
                total_results=0,
                search_time_ms=response_time,
                search_method=SearchMethod.HYBRID
            )
    
    async def _perform_vector_search(self, search_query: SearchQuery) -> List[VectorSearchResult]:
        """Perform vector search."""
        try:
            response = await semantic_search_engine.search(search_query)
            return response.results
        except Exception as e:
            logger.error(f"Vector search failed in hybrid coordinator: {e}")
            return []
    
    async def _perform_graph_search(self, search_query: SearchQuery) -> List[VectorSearchResult]:
        """Perform graph-based search."""
        try:
            # Extract keywords for graph search
            keywords = search_query.query.split()
            
            # Search knowledge graph
            results = temporal_knowledge_graph.search_by_keywords(
                keywords,
                search_entities=True,
                search_descriptions=True
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Graph search failed in hybrid coordinator: {e}")
            return []
    
    async def _perform_lexical_search(self, search_query: SearchQuery) -> List[VectorSearchResult]:
        """Perform lexical search."""
        try:
            # Note: In a real implementation, you would need to index documents first
            # This is a placeholder that returns empty results
            results = self.lexical_engine.search(
                search_query.query,
                max_results=search_query.context.max_results
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Lexical search failed in hybrid coordinator: {e}")
            return []
    
    def _update_stats(self, method: SearchMethod, response_time: float, result_count: int):
        """Update search statistics."""
        self.search_stats['total_searches'] += 1
        self.search_stats['method_usage'][method.value] += 1
        
        # Update average response time for this method
        current_avg = self.search_stats['average_response_times'][method.value]
        method_count = self.search_stats['method_usage'][method.value]
        
        new_avg = ((current_avg * (method_count - 1)) + response_time) / method_count
        self.search_stats['average_response_times'][method.value] = new_avg
        
        # Update fusion efficiency (results per search)
        total_results = sum(
            self.search_stats['method_usage'][m] for m in self.search_stats['method_usage']
        )
        self.search_stats['fusion_efficiency'] = result_count / max(total_results, 1)
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search coordinator statistics."""
        return dict(self.search_stats)
    
    async def index_documents_for_lexical_search(self, documents: List[Dict[str, Any]]):
        """Index documents for lexical search."""
        try:
            self.lexical_engine.index_documents(documents)
            logger.info(f"Indexed {len(documents)} documents for lexical search")
        except Exception as e:
            logger.error(f"Failed to index documents for lexical search: {e}")


# Global hybrid search coordinator instance
hybrid_search_coordinator = HybridSearchCoordinator()


async def perform_hybrid_search(search_query: SearchQuery) -> SearchResponse:
    """Convenience function to perform hybrid search."""
    return await hybrid_search_coordinator.search(search_query)