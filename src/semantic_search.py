"""
Semantic search engine with query vectorization and intelligent ranking.

This module provides comprehensive semantic search capabilities including
query preprocessing, vector similarity search, result ranking, and
performance optimization.
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import get_config
from .embeddings import get_embedding, get_embeddings_batch
from .models import (
    VectorSearchResult,
    SearchQuery,
    SearchResponse,
    QueryContext,
    SearchMethod,
    ContentType,
    DomainType,
    VectorMetadata,
    EmbeddingProvider
)
from .vector_store import vector_store


logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """Handles query preprocessing and normalization."""
    
    def __init__(self):
        """Initialize query preprocessor."""
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Temporal expressions patterns
        self.temporal_patterns = [
            r'\b(?:yesterday|today|tomorrow)\b',
            r'\b(?:last|next|this)\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
            r'\b(?:before|after|since|until|during)\s+\d{4}\b',
            r'\b(?:in|on|at)\s+\d{4}\b'
        ]
        
        # Domain-specific keywords
        self.domain_keywords = {
            DomainType.TECHNICAL: {'api', 'database', 'server', 'code', 'software', 'system', 'algorithm', 'programming'},
            DomainType.BUSINESS: {'revenue', 'profit', 'market', 'sales', 'customer', 'strategy', 'business', 'finance'},
            DomainType.SCIENTIFIC: {'research', 'study', 'experiment', 'hypothesis', 'data', 'analysis', 'method', 'result'},
            DomainType.MEDICAL: {'patient', 'treatment', 'diagnosis', 'symptom', 'medical', 'health', 'doctor', 'medicine'},
            DomainType.LEGAL: {'law', 'legal', 'court', 'contract', 'regulation', 'compliance', 'attorney', 'lawsuit'}
        }
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """Preprocess query and extract relevant information."""
        processed = {
            'original_query': query,
            'cleaned_query': self._clean_query(query),
            'keywords': self._extract_keywords(query),
            'temporal_expressions': self._extract_temporal_expressions(query),
            'domain_hints': self._detect_domain_hints(query),
            'intent': self._classify_intent(query),
            'entities': self._extract_entities(query)
        }
        
        return processed
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text."""
        # Convert to lowercase
        cleaned = query.lower().strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep useful punctuation
        cleaned = re.sub(r'[^\w\s\-\'\".,!?]', ' ', cleaned)
        
        # Remove multiple punctuation marks
        cleaned = re.sub(r'[.,!?]{2,}', '.', cleaned)
        
        return cleaned.strip()
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        cleaned = self._clean_query(query)
        words = cleaned.split()
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word.lower() not in self.stop_words and len(word) > 2
        ]
        
        return keywords
    
    def _extract_temporal_expressions(self, query: str) -> List[str]:
        """Extract temporal expressions from query."""
        expressions = []
        
        for pattern in self.temporal_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            expressions.extend([match.group() for match in matches])
        
        return expressions
    
    def _detect_domain_hints(self, query: str) -> List[DomainType]:
        """Detect domain hints from query keywords."""
        query_lower = query.lower()
        domain_scores = defaultdict(int)
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    domain_scores[domain] += 1
        
        # Return domains with at least one matching keyword
        return [domain for domain, score in domain_scores.items() if score > 0]
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent."""
        query_lower = query.lower()
        
        # Question patterns
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        
        # Search patterns
        if any(word in query_lower for word in ['find', 'search', 'look for', 'show me']):
            return 'search'
        
        # Temporal patterns
        if any(word in query_lower for word in ['before', 'after', 'since', 'until', 'during']):
            return 'temporal'
        
        # Comparison patterns
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        
        # Definition patterns
        if any(word in query_lower for word in ['define', 'definition', 'what is', 'explain']):
            return 'definition'
        
        return 'general'
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entity names from query."""
        # Simple entity extraction - look for capitalized words and quoted strings
        entities = []
        
        # Quoted strings
        quoted_matches = re.finditer(r'"([^"]+)"', query)
        entities.extend([match.group(1) for match in quoted_matches])
        
        # Capitalized words (potential proper nouns)
        capitalized_matches = re.finditer(r'\b[A-Z][a-z]+\b', query)
        entities.extend([match.group() for match in capitalized_matches])
        
        return list(set(entities))


class QueryExpander:
    """Handles query expansion for better matching."""
    
    def __init__(self):
        """Initialize query expander."""
        # Synonym mappings for common terms
        self.synonyms = {
            'find': ['search', 'locate', 'discover', 'identify'],
            'show': ['display', 'present', 'reveal', 'demonstrate'],
            'create': ['make', 'build', 'generate', 'produce'],
            'remove': ['delete', 'eliminate', 'erase', 'clear'],
            'update': ['modify', 'change', 'edit', 'revise'],
            'error': ['bug', 'issue', 'problem', 'fault'],
            'method': ['function', 'procedure', 'approach', 'technique'],
            'data': ['information', 'content', 'records', 'details']
        }
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with synonyms and related terms."""
        expanded_queries = [query]
        
        # Simple synonym expansion
        words = query.lower().split()
        
        for word in words:
            if word in self.synonyms:
                # Create variations with synonyms
                for synonym in self.synonyms[word][:max_expansions]:
                    expanded_query = query.replace(word, synonym)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        return expanded_queries


class ResultRanker:
    """Handles result ranking and scoring."""
    
    def __init__(self):
        """Initialize result ranker."""
        self.config = get_config()
        
        # Ranking weights from configuration
        self.weights = self.config.search.ranking
    
    def rank_results(
        self, 
        results: List[VectorSearchResult],
        query_context: QueryContext,
        query_info: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Rank search results based on multiple factors."""
        if not results:
            return results
        
        # Calculate ranking scores for each result
        scored_results = []
        
        for result in results:
            ranking_score = self._calculate_ranking_score(result, query_context, query_info)
            
            # Update ranking factors in result
            result.ranking_factors.update({
                'final_score': ranking_score,
                'similarity_weight': self.weights.get('similarity_weight', 0.4),
                'temporal_weight': self.weights.get('temporal_weight', 0.2),
                'confidence_weight': self.weights.get('confidence_weight', 0.3),
                'diversity_weight': self.weights.get('diversity_factor', 0.1)
            })
            
            scored_results.append((ranking_score, result))
        
        # Sort by ranking score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Apply diversity filtering
        final_results = self._apply_diversity_filter([result for _, result in scored_results])
        
        return final_results
    
    def _calculate_ranking_score(
        self, 
        result: VectorSearchResult,
        query_context: QueryContext,
        query_info: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive ranking score."""
        score = 0.0
        
        # Base similarity score
        similarity_score = result.similarity_score
        score += similarity_score * self.weights.get('similarity_weight', 0.4)
        
        # Confidence score from metadata
        confidence_score = result.metadata.confidence_score
        score += confidence_score * self.weights.get('confidence_weight', 0.3)
        
        # Temporal relevance
        temporal_score = self._calculate_temporal_relevance(result, query_context, query_info)
        score += temporal_score * self.weights.get('temporal_weight', 0.2)
        
        # Domain matching
        domain_score = self._calculate_domain_relevance(result, query_context, query_info)
        score += domain_score * 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_temporal_relevance(
        self, 
        result: VectorSearchResult,
        query_context: QueryContext,
        query_info: Dict[str, Any]
    ) -> float:
        """Calculate temporal relevance score."""
        if not self.config.temporal.enable_temporal_search:
            return 1.0
        
        # Check if query has temporal context
        temporal_expressions = query_info.get('temporal_expressions', [])
        if not temporal_expressions and not query_context.temporal_filter:
            return 1.0
        
        # Get temporal information from result metadata
        result_date = None
        metadata = result.metadata
        
        if metadata.valid_from:
            result_date = metadata.valid_from
        elif metadata.creation_date:
            result_date = metadata.creation_date
        elif metadata.last_updated:
            result_date = metadata.last_updated
        
        if not result_date:
            return 0.5  # Neutral score for undated content
        
        # Calculate temporal decay based on recency
        now = datetime.utcnow()
        time_diff = (now - result_date).days
        
        # Apply temporal decay
        decay_factor = self.config.temporal.temporal_decay_factor
        max_age_days = self.config.temporal.max_temporal_range_days
        
        if time_diff > max_age_days:
            return 0.1  # Very old content gets low score
        
        # Calculate decay score
        temporal_score = decay_factor ** (time_diff / 30)  # Decay over 30-day periods
        
        return temporal_score
    
    def _calculate_domain_relevance(
        self, 
        result: VectorSearchResult,
        query_context: QueryContext,
        query_info: Dict[str, Any]
    ) -> float:
        """Calculate domain relevance score."""
        # Domain filter from context
        if query_context.domain_filter:
            if result.metadata.domain == query_context.domain_filter:
                return 1.0
            else:
                return 0.1  # Penalize mismatched domains
        
        # Domain hints from query
        domain_hints = query_info.get('domain_hints', [])
        if domain_hints:
            if result.metadata.domain in domain_hints:
                return 1.0
            else:
                return 0.8  # Slight penalty for non-matching domains
        
        return 1.0  # No domain preference
    
    def _apply_diversity_filter(self, results: List[VectorSearchResult]) -> List[VectorSearchResult]:
        """Apply diversity filtering to avoid too similar results."""
        if len(results) <= 1:
            return results
        
        diversity_threshold = 0.9  # Similarity threshold for duplicate detection
        filtered_results = []
        
        for result in results:
            is_duplicate = False
            
            for existing_result in filtered_results:
                # Simple content similarity check
                content_similarity = self._calculate_content_similarity(
                    result.content, existing_result.content
                )
                
                if content_similarity > diversity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity between two texts."""
        if not content1 or not content2:
            return 0.0
        
        # Simple character-based similarity
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        # Calculate Jaccard similarity of words
        words1 = set(content1_lower.split())
        words2 = set(content2_lower.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class SemanticSearchEngine:
    """Main semantic search engine."""
    
    def __init__(self):
        """Initialize semantic search engine."""
        self.config = get_config()
        self.preprocessor = QueryPreprocessor()
        self.expander = QueryExpander()
        self.ranker = ResultRanker()
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'average_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Simple result cache
        self._result_cache: Dict[str, Tuple[datetime, List[VectorSearchResult]]] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    async def search(self, search_query: SearchQuery) -> SearchResponse:
        """Perform semantic search."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(search_query)
            cached_results = self._get_cached_results(cache_key)
            
            if cached_results is not None:
                self.search_stats['cache_hits'] += 1
                response_time = (time.time() - start_time) * 1000
                
                return SearchResponse(
                    query=search_query.query,
                    results=cached_results,
                    total_results=len(cached_results),
                    search_time_ms=response_time,
                    search_method=SearchMethod.VECTOR
                )
            
            self.search_stats['cache_misses'] += 1
            
            # Preprocess query
            query_info = self.preprocessor.preprocess_query(search_query.query)
            
            # Perform vector search across relevant collections
            all_results = await self._perform_vector_search(search_query, query_info)
            
            # Rank results
            ranked_results = self.ranker.rank_results(all_results, search_query.context, query_info)
            
            # Apply result limits
            max_results = search_query.context.max_results
            final_results = ranked_results[:max_results]
            
            # Cache results
            self._cache_results(cache_key, final_results)
            
            # Update statistics
            response_time = (time.time() - start_time) * 1000
            self.search_stats['total_searches'] += 1
            self.search_stats['average_response_time'] = (
                (self.search_stats['average_response_time'] * (self.search_stats['total_searches'] - 1) + response_time) /
                self.search_stats['total_searches']
            )
            
            # Create response
            response = SearchResponse(
                query=search_query.query,
                results=final_results,
                total_results=len(final_results),
                search_time_ms=response_time,
                search_method=SearchMethod.VECTOR,
                domains_found=list(set(result.metadata.domain for result in final_results)),
                content_types_found=list(set(result.metadata.content_type for result in final_results))
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            
            # Return empty response on error
            response_time = (time.time() - start_time) * 1000
            return SearchResponse(
                query=search_query.query,
                results=[],
                total_results=0,
                search_time_ms=response_time,
                search_method=SearchMethod.VECTOR
            )
    
    async def _perform_vector_search(
        self, 
        search_query: SearchQuery, 
        query_info: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Perform vector search across collections."""
        all_results = []
        
        # Determine which collections to search
        collections_to_search = self._determine_collections(search_query, query_info)
        
        # Prepare query for expansion if enabled
        queries_to_search = [search_query.query]
        if search_query.enable_query_expansion:
            expanded_queries = self.expander.expand_query(search_query.query)
            queries_to_search.extend(expanded_queries[:2])  # Limit expansions
        
        # Search each collection with each query variant
        search_tasks = []
        
        for collection_name in collections_to_search:
            for query_text in queries_to_search:
                task = self._search_collection(
                    collection_name, 
                    query_text, 
                    search_query,
                    query_info
                )
                search_tasks.append(task)
        
        # Execute searches in parallel
        if search_tasks:
            search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine results from all searches
            for search_results in search_results_list:
                if isinstance(search_results, list):
                    all_results.extend(search_results)
                elif isinstance(search_results, Exception):
                    logger.warning(f"Collection search failed: {search_results}")
        
        # Remove duplicates based on content_id
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            if result.content_id not in seen_ids:
                seen_ids.add(result.content_id)
                unique_results.append(result)
        
        return unique_results
    
    async def _search_collection(
        self,
        collection_name: str,
        query_text: str,
        search_query: SearchQuery,
        query_info: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Search a specific collection."""
        try:
            # Build metadata filter
            metadata_filter = self._build_metadata_filter(search_query, query_info)
            
            # Determine similarity threshold
            similarity_threshold = (
                search_query.similarity_threshold or 
                self.config.search.similarity_threshold
            )
            
            # Perform vector search
            results = await vector_store.search_vectors(
                collection_name=collection_name,
                query_text=query_text,
                n_results=search_query.context.max_results * 2,  # Get more for better ranking
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filter
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search collection {collection_name}: {e}")
            return []
    
    def _determine_collections(
        self, 
        search_query: SearchQuery, 
        query_info: Dict[str, Any]
    ) -> List[str]:
        """Determine which collections to search based on query context."""
        collections = []
        
        # Default collections
        default_collections = ["document_chunks", "atomic_facts"]
        
        # Add temporal events collection if temporal query
        if (query_info.get('temporal_expressions') or 
            query_context.temporal_filter or
            query_info.get('intent') == 'temporal'):
            collections.append("temporal_events")
        
        # Add entity collection if entity-focused query
        if query_info.get('entities'):
            collections.append("domain_entities")
        
        # Filter by content type preferences
        preferred_types = search_query.context.preferred_content_types
        if preferred_types:
            type_to_collection = {
                ContentType.CHUNK: "document_chunks",
                ContentType.FACT: "atomic_facts",
                ContentType.EVENT: "temporal_events",
                ContentType.ENTITY: "domain_entities"
            }
            
            collections = [
                type_to_collection[content_type] 
                for content_type in preferred_types
                if content_type in type_to_collection
            ]
        
        # Use default collections if none specified
        if not collections:
            collections = default_collections
        
        return collections
    
    def _build_metadata_filter(
        self, 
        search_query: SearchQuery, 
        query_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build metadata filter for search."""
        filters = {}
        
        # Domain filter
        if search_query.context.domain_filter:
            filters["domain"] = search_query.context.domain_filter.value
        
        # Confidence threshold filter
        if search_query.context.confidence_threshold > 0:
            filters["confidence_score"] = {"$gte": search_query.context.confidence_threshold}
        
        # Temporal filters
        if search_query.context.temporal_filter:
            temporal_filter = search_query.context.temporal_filter
            
            if "start_date" in temporal_filter:
                filters["valid_from"] = {"$gte": temporal_filter["start_date"].isoformat()}
            
            if "end_date" in temporal_filter:
                filters["valid_to"] = {"$lte": temporal_filter["end_date"].isoformat()}
        
        # Source exclusion filter
        if search_query.context.exclude_sources:
            filters["source"] = {"$nin": search_query.context.exclude_sources}
        
        return filters if filters else None
    
    def _generate_cache_key(self, search_query: SearchQuery) -> str:
        """Generate cache key for search query."""
        # Create a hash of the search parameters
        import hashlib
        
        key_parts = [
            search_query.query,
            str(search_query.search_method.value),
            str(search_query.context.dict())
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_results(self, cache_key: str) -> Optional[List[VectorSearchResult]]:
        """Get cached results if still valid."""
        if cache_key in self._result_cache:
            cache_time, results = self._result_cache[cache_key]
            
            if datetime.utcnow() - cache_time < self._cache_ttl:
                return results
            else:
                # Remove expired cache entry
                del self._result_cache[cache_key]
        
        return None
    
    def _cache_results(self, cache_key: str, results: List[VectorSearchResult]):
        """Cache search results."""
        # Limit cache size
        if len(self._result_cache) >= self.config.performance.vector_cache_size:
            # Remove oldest entry
            oldest_key = min(self._result_cache.keys(), 
                           key=lambda k: self._result_cache[k][0])
            del self._result_cache[oldest_key]
        
        self._result_cache[cache_key] = (datetime.utcnow(), results)
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return self.search_stats.copy()
    
    def clear_cache(self):
        """Clear result cache."""
        self._result_cache.clear()
        logger.info("Search cache cleared")


# Global semantic search engine instance
semantic_search_engine = SemanticSearchEngine()


async def perform_semantic_search(search_query: SearchQuery) -> SearchResponse:
    """Convenience function to perform semantic search."""
    return await semantic_search_engine.search(search_query)