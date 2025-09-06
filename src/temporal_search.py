"""
Temporal vector features and time-aware search capabilities.

This module provides temporal indexing, time-aware vector search,
and temporal context embedding for the knowledge graph system.
"""

import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
from dateutil.parser import parse as parse_date

from .config import get_config
from .models import (
    VectorSearchResult,
    SearchQuery,
    QueryContext,
    VectorMetadata,
    TemporalEvent,
    AtomicFact,
    ContentType,
    DomainType
)
from .vector_store import vector_store
from .knowledge_graph import temporal_knowledge_graph


logger = logging.getLogger(__name__)


class TemporalParser:
    """Parser for temporal expressions in text."""
    
    def __init__(self):
        """Initialize temporal parser."""
        # Common temporal patterns
        self.temporal_patterns = {
            'absolute_dates': [
                r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{2}-\d{2}-\d{4}\b',  # MM-DD-YYYY
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # M/D/YY or MM/DD/YYYY
            ],
            'relative_dates': [
                r'\b(?:yesterday|today|tomorrow)\b',
                r'\b(?:last|next|this)\s+(?:week|month|year|quarter)\b',
                r'\b(?:last|next|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|from now)\b',
            ],
            'temporal_modifiers': [
                r'\b(?:before|after|since|until|during|within|by)\b',
                r'\b(?:early|late|mid)\s+\d{4}\b',
                r'\b(?:beginning|end|start|finish)\s+of\s+\d{4}\b',
            ],
            'named_periods': [
                r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
                r'\b(?:spring|summer|fall|autumn|winter)\s+\d{4}\b',
                r'\bq[1-4]\s+\d{4}\b',  # Q1 2023
            ]
        }
        
        # Temporal keywords that indicate time-sensitive queries
        self.temporal_keywords = {
            'when', 'time', 'date', 'period', 'during', 'timeline', 'chronology',
            'history', 'historical', 'recent', 'latest', 'current', 'past', 'future',
            'before', 'after', 'since', 'until', 'while', 'meanwhile', 'simultaneous'
        }
    
    def extract_temporal_expressions(self, text: str) -> List[Dict[str, Any]]:
        """Extract temporal expressions from text."""
        expressions = []
        text_lower = text.lower()
        
        # Extract different types of temporal expressions
        for category, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    expressions.append({
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'category': category,
                        'normalized': self._normalize_temporal_expression(match.group(), category)
                    })
        
        return expressions
    
    def _normalize_temporal_expression(self, expression: str, category: str) -> Optional[Dict[str, Any]]:
        """Normalize temporal expression to standard format."""
        try:
            if category == 'absolute_dates':
                # Try to parse as date
                parsed_date = parse_date(expression)
                return {
                    'type': 'absolute',
                    'datetime': parsed_date,
                    'precision': self._determine_precision(expression)
                }
            
            elif category == 'relative_dates':
                return self._parse_relative_date(expression)
            
            elif category == 'temporal_modifiers':
                return {
                    'type': 'modifier',
                    'modifier': expression.lower(),
                    'requires_context': True
                }
            
            elif category == 'named_periods':
                return self._parse_named_period(expression)
            
        except Exception as e:
            logger.debug(f"Failed to normalize temporal expression '{expression}': {e}")
            return None
    
    def _determine_precision(self, date_string: str) -> str:
        """Determine the precision of a date string."""
        if re.match(r'\d{4}-\d{2}-\d{2}', date_string):
            return 'day'
        elif re.match(r'\d{4}-\d{2}', date_string):
            return 'month'
        elif re.match(r'\d{4}', date_string):
            return 'year'
        else:
            return 'day'  # Default
    
    def _parse_relative_date(self, expression: str) -> Dict[str, Any]:
        """Parse relative date expressions."""
        now = datetime.utcnow()
        expression_lower = expression.lower()
        
        if 'yesterday' in expression_lower:
            return {
                'type': 'relative',
                'datetime': now - timedelta(days=1),
                'precision': 'day'
            }
        elif 'today' in expression_lower:
            return {
                'type': 'relative',
                'datetime': now,
                'precision': 'day'
            }
        elif 'tomorrow' in expression_lower:
            return {
                'type': 'relative',
                'datetime': now + timedelta(days=1),
                'precision': 'day'
            }
        elif 'last week' in expression_lower:
            return {
                'type': 'relative',
                'datetime': now - timedelta(weeks=1),
                'precision': 'week'
            }
        elif 'last month' in expression_lower:
            return {
                'type': 'relative',
                'datetime': now - timedelta(days=30),
                'precision': 'month'
            }
        elif 'last year' in expression_lower:
            return {
                'type': 'relative',
                'datetime': now - timedelta(days=365),
                'precision': 'year'
            }
        
        # Parse "X days/weeks/months/years ago" patterns
        ago_pattern = r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago'
        match = re.search(ago_pattern, expression_lower)
        if match:
            amount = int(match.group(1))
            unit = match.group(2).rstrip('s')  # Remove plural
            
            if unit == 'day':
                delta = timedelta(days=amount)
            elif unit == 'week':
                delta = timedelta(weeks=amount)
            elif unit == 'month':
                delta = timedelta(days=amount * 30)
            elif unit == 'year':
                delta = timedelta(days=amount * 365)
            else:
                delta = timedelta(days=amount)
            
            return {
                'type': 'relative',
                'datetime': now - delta,
                'precision': unit
            }
        
        return {
            'type': 'relative',
            'datetime': now,
            'precision': 'day',
            'uncertain': True
        }
    
    def _parse_named_period(self, expression: str) -> Dict[str, Any]:
        """Parse named time periods."""
        expression_lower = expression.lower()
        
        # Extract year
        year_match = re.search(r'\d{4}', expression)
        if year_match:
            year = int(year_match.group())
            
            # Month names
            month_names = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            
            for month_name, month_num in month_names.items():
                if month_name in expression_lower:
                    return {
                        'type': 'absolute',
                        'datetime': datetime(year, month_num, 1),
                        'precision': 'month'
                    }
            
            # Seasons
            if 'spring' in expression_lower:
                return {
                    'type': 'absolute',
                    'datetime': datetime(year, 3, 1),  # March
                    'precision': 'season'
                }
            elif 'summer' in expression_lower:
                return {
                    'type': 'absolute',
                    'datetime': datetime(year, 6, 1),  # June
                    'precision': 'season'
                }
            elif any(season in expression_lower for season in ['fall', 'autumn']):
                return {
                    'type': 'absolute',
                    'datetime': datetime(year, 9, 1),  # September
                    'precision': 'season'
                }
            elif 'winter' in expression_lower:
                return {
                    'type': 'absolute',
                    'datetime': datetime(year, 12, 1),  # December
                    'precision': 'season'
                }
            
            # Quarters
            quarter_match = re.search(r'q([1-4])', expression_lower)
            if quarter_match:
                quarter = int(quarter_match.group(1))
                month = (quarter - 1) * 3 + 1
                return {
                    'type': 'absolute',
                    'datetime': datetime(year, month, 1),
                    'precision': 'quarter'
                }
        
        return {
            'type': 'named',
            'expression': expression,
            'uncertain': True
        }
    
    def has_temporal_indicators(self, text: str) -> bool:
        """Check if text has temporal indicators."""
        text_lower = text.lower()
        
        # Check for temporal keywords
        for keyword in self.temporal_keywords:
            if keyword in text_lower:
                return True
        
        # Check for temporal patterns
        for category, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
        
        return False


class TemporalIndexer:
    """Indexes content with temporal information for efficient time-aware search."""
    
    def __init__(self):
        """Initialize temporal indexer."""
        self.config = get_config()
        self.temporal_parser = TemporalParser()
        
        # Temporal indexes
        self.date_index = defaultdict(list)  # Date -> content IDs
        self.period_index = defaultdict(list)  # Period -> content IDs
        self.event_timeline = []  # Chronologically ordered events
        
        # Statistics
        self.stats = {
            'indexed_items': 0,
            'temporal_items': 0,
            'date_ranges': 0
        }
    
    def index_content(
        self, 
        content_id: str, 
        content: str, 
        metadata: VectorMetadata
    ) -> Dict[str, Any]:
        """Index content with temporal information."""
        try:
            # Extract temporal expressions from content
            temporal_expressions = self.temporal_parser.extract_temporal_expressions(content)
            
            # Process metadata temporal information
            temporal_metadata = self._extract_metadata_temporal_info(metadata)
            
            # Create temporal index entry
            temporal_info = {
                'content_id': content_id,
                'expressions': temporal_expressions,
                'metadata_temporal': temporal_metadata,
                'has_temporal': len(temporal_expressions) > 0 or temporal_metadata is not None
            }
            
            # Add to appropriate indexes
            if temporal_info['has_temporal']:
                self._add_to_temporal_indexes(content_id, temporal_expressions, temporal_metadata)
                self.stats['temporal_items'] += 1
            
            self.stats['indexed_items'] += 1
            
            return temporal_info
            
        except Exception as e:
            logger.error(f"Failed to index temporal content {content_id}: {e}")
            return {'content_id': content_id, 'has_temporal': False, 'error': str(e)}
    
    def _extract_metadata_temporal_info(self, metadata: VectorMetadata) -> Optional[Dict[str, Any]]:
        """Extract temporal information from metadata."""
        temporal_info = {}
        
        if metadata.valid_from:
            temporal_info['valid_from'] = metadata.valid_from
        
        if metadata.valid_to:
            temporal_info['valid_to'] = metadata.valid_to
        
        if metadata.creation_date:
            temporal_info['creation_date'] = metadata.creation_date
        
        if metadata.last_updated:
            temporal_info['last_updated'] = metadata.last_updated
        
        if metadata.temporal_context:
            temporal_info['context'] = metadata.temporal_context
        
        return temporal_info if temporal_info else None
    
    def _add_to_temporal_indexes(
        self, 
        content_id: str, 
        expressions: List[Dict[str, Any]], 
        metadata_temporal: Optional[Dict[str, Any]]
    ):
        """Add content to temporal indexes."""
        # Index by expressions
        for expr in expressions:
            normalized = expr.get('normalized')
            if normalized and normalized.get('datetime'):
                date_key = normalized['datetime'].strftime('%Y-%m-%d')
                self.date_index[date_key].append(content_id)
                
                # Add to period indexes based on precision
                precision = normalized.get('precision', 'day')
                if precision == 'month':
                    month_key = normalized['datetime'].strftime('%Y-%m')
                    self.period_index[f"month_{month_key}"].append(content_id)
                elif precision == 'year':
                    year_key = normalized['datetime'].strftime('%Y')
                    self.period_index[f"year_{year_key}"].append(content_id)
        
        # Index by metadata temporal information
        if metadata_temporal:
            if 'valid_from' in metadata_temporal:
                date_key = metadata_temporal['valid_from'].strftime('%Y-%m-%d')
                self.date_index[date_key].append(content_id)
            
            if 'creation_date' in metadata_temporal:
                date_key = metadata_temporal['creation_date'].strftime('%Y-%m-%d')
                self.date_index[date_key].append(content_id)
    
    def search_by_temporal_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[str]:
        """Search content by temporal range."""
        matching_content_ids = set()
        
        # Search date index
        current_date = start_date
        while current_date <= end_date:
            date_key = current_date.strftime('%Y-%m-%d')
            if date_key in self.date_index:
                matching_content_ids.update(self.date_index[date_key])
            current_date += timedelta(days=1)
        
        return list(matching_content_ids)
    
    def get_temporal_stats(self) -> Dict[str, Any]:
        """Get temporal indexer statistics."""
        stats = self.stats.copy()
        stats['unique_dates'] = len(self.date_index)
        stats['unique_periods'] = len(self.period_index)
        stats['temporal_coverage'] = (
            self.stats['temporal_items'] / max(self.stats['indexed_items'], 1)
        )
        return stats


class TemporalSearchEngine:
    """Engine for time-aware vector search."""
    
    def __init__(self):
        """Initialize temporal search engine."""
        self.config = get_config()
        self.temporal_parser = TemporalParser()
        self.temporal_indexer = TemporalIndexer()
        
        # Temporal search cache
        self.search_cache = {}
        self.cache_ttl = timedelta(minutes=10)
    
    async def temporal_search(
        self, 
        search_query: SearchQuery,
        temporal_context: Optional[Dict[str, datetime]] = None
    ) -> List[VectorSearchResult]:
        """Perform time-aware vector search."""
        try:
            # Extract temporal information from query
            query_temporal_info = self._analyze_query_temporal_context(
                search_query.query, temporal_context
            )
            
            # Determine search strategy based on temporal context
            if query_temporal_info['has_temporal']:
                # Time-constrained search
                results = await self._time_constrained_search(search_query, query_temporal_info)
            else:
                # Regular search with temporal ranking
                results = await self._temporal_ranked_search(search_query)
            
            # Apply temporal ranking
            ranked_results = self._apply_temporal_ranking(results, query_temporal_info)
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
            return []
    
    def _analyze_query_temporal_context(
        self, 
        query: str, 
        explicit_context: Optional[Dict[str, datetime]] = None
    ) -> Dict[str, Any]:
        """Analyze temporal context in query."""
        temporal_info = {
            'has_temporal': False,
            'explicit_context': explicit_context,
            'query_expressions': [],
            'temporal_focus': None
        }
        
        # Check for explicit temporal context
        if explicit_context:
            temporal_info['has_temporal'] = True
            temporal_info['temporal_focus'] = 'explicit'
        
        # Extract temporal expressions from query
        expressions = self.temporal_parser.extract_temporal_expressions(query)
        if expressions:
            temporal_info['has_temporal'] = True
            temporal_info['query_expressions'] = expressions
            temporal_info['temporal_focus'] = 'query_based'
        
        # Check for temporal keywords
        if self.temporal_parser.has_temporal_indicators(query):
            temporal_info['has_temporal'] = True
            if not temporal_info['temporal_focus']:
                temporal_info['temporal_focus'] = 'temporal_keywords'
        
        return temporal_info
    
    async def _time_constrained_search(
        self, 
        search_query: SearchQuery, 
        temporal_info: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Perform search with temporal constraints."""
        results = []
        
        # Build temporal filter
        temporal_filter = self._build_temporal_filter(temporal_info)
        
        # Update search query context with temporal filter
        modified_query = search_query.copy(deep=True)
        modified_query.context.temporal_filter = temporal_filter
        
        # Search vector store with temporal constraints
        for collection_name in ['document_chunks', 'atomic_facts', 'temporal_events']:
            try:
                collection_results = await vector_store.search_vectors(
                    collection_name=collection_name,
                    query_text=search_query.query,
                    n_results=search_query.context.max_results,
                    similarity_threshold=search_query.similarity_threshold,
                    metadata_filter=self._convert_temporal_filter_to_metadata_filter(temporal_filter)
                )
                results.extend(collection_results)
            except Exception as e:
                logger.warning(f"Failed to search collection {collection_name} with temporal constraints: {e}")
        
        return results
    
    async def _temporal_ranked_search(self, search_query: SearchQuery) -> List[VectorSearchResult]:
        """Perform regular search with temporal ranking applied."""
        # Import to avoid circular dependency
        from .semantic_search import semantic_search_engine
        
        response = await semantic_search_engine.search(search_query)
        return response.results
    
    def _build_temporal_filter(self, temporal_info: Dict[str, Any]) -> Dict[str, datetime]:
        """Build temporal filter from temporal information."""
        temporal_filter = {}
        
        # Handle explicit context
        if temporal_info['explicit_context']:
            temporal_filter.update(temporal_info['explicit_context'])
        
        # Handle query expressions
        for expression in temporal_info['query_expressions']:
            normalized = expression.get('normalized')
            if normalized and normalized.get('datetime'):
                dt = normalized['datetime']
                precision = normalized.get('precision', 'day')
                
                # Create date range based on precision
                if precision == 'day':
                    temporal_filter['start_date'] = dt
                    temporal_filter['end_date'] = dt + timedelta(days=1)
                elif precision == 'month':
                    temporal_filter['start_date'] = dt
                    # Last day of month
                    if dt.month == 12:
                        end_date = datetime(dt.year + 1, 1, 1)
                    else:
                        end_date = datetime(dt.year, dt.month + 1, 1)
                    temporal_filter['end_date'] = end_date
                elif precision == 'year':
                    temporal_filter['start_date'] = dt
                    temporal_filter['end_date'] = datetime(dt.year + 1, 1, 1)
        
        return temporal_filter
    
    def _convert_temporal_filter_to_metadata_filter(
        self, 
        temporal_filter: Dict[str, datetime]
    ) -> Dict[str, Any]:
        """Convert temporal filter to metadata filter for vector search."""
        metadata_filter = {}
        
        if 'start_date' in temporal_filter:
            metadata_filter['valid_from'] = {'$gte': temporal_filter['start_date'].isoformat()}
        
        if 'end_date' in temporal_filter:
            metadata_filter['valid_to'] = {'$lte': temporal_filter['end_date'].isoformat()}
        
        return metadata_filter
    
    def _apply_temporal_ranking(
        self, 
        results: List[VectorSearchResult], 
        temporal_info: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Apply temporal ranking to search results."""
        if not results or not self.config.temporal.enable_temporal_search:
            return results
        
        ranked_results = []
        
        for result in results:
            # Calculate temporal relevance score
            temporal_score = self._calculate_temporal_relevance(result, temporal_info)
            
            # Update ranking factors
            result.ranking_factors['temporal_relevance'] = temporal_score
            result.temporal_relevance = temporal_score
            
            # Adjust overall similarity score
            temporal_weight = self.config.search.ranking.get('temporal_weight', 0.2)
            adjusted_score = (
                result.similarity_score * (1 - temporal_weight) +
                temporal_score * temporal_weight
            )
            
            # Create new result with adjusted score
            adjusted_result = result.copy(deep=True)
            adjusted_result.similarity_score = adjusted_score
            adjusted_result.ranking_factors['temporal_adjusted'] = adjusted_score
            
            ranked_results.append(adjusted_result)
        
        # Sort by adjusted score
        ranked_results.sort(key=lambda r: r.similarity_score, reverse=True)
        
        return ranked_results
    
    def _calculate_temporal_relevance(
        self, 
        result: VectorSearchResult, 
        temporal_info: Dict[str, Any]
    ) -> float:
        """Calculate temporal relevance score for a result."""
        if not temporal_info['has_temporal']:
            return 1.0  # No temporal context, all results equally relevant
        
        # Get temporal information from result metadata
        metadata = result.metadata
        result_dates = []
        
        if metadata.valid_from:
            result_dates.append(metadata.valid_from)
        
        if metadata.creation_date:
            result_dates.append(metadata.creation_date)
        
        if metadata.last_updated:
            result_dates.append(metadata.last_updated)
        
        if not result_dates:
            return 0.5  # Neutral score for undated content
        
        # Calculate relevance based on temporal context
        relevance_scores = []
        
        # Check against explicit temporal context
        if temporal_info['explicit_context']:
            for result_date in result_dates:
                score = self._calculate_date_relevance(
                    result_date, temporal_info['explicit_context']
                )
                relevance_scores.append(score)
        
        # Check against query temporal expressions
        for expression in temporal_info['query_expressions']:
            normalized = expression.get('normalized')
            if normalized and normalized.get('datetime'):
                target_date = normalized['datetime']
                for result_date in result_dates:
                    score = self._calculate_date_proximity_score(result_date, target_date)
                    relevance_scores.append(score)
        
        # Apply temporal decay for recency
        if temporal_info['temporal_focus'] == 'temporal_keywords':
            for result_date in result_dates:
                decay_score = self._calculate_temporal_decay_score(result_date)
                relevance_scores.append(decay_score)
        
        # Return maximum relevance score
        return max(relevance_scores) if relevance_scores else 0.5
    
    def _calculate_date_relevance(
        self, 
        result_date: datetime, 
        temporal_context: Dict[str, datetime]
    ) -> float:
        """Calculate relevance of result date to temporal context."""
        start_date = temporal_context.get('start_date')
        end_date = temporal_context.get('end_date')
        
        if start_date and end_date:
            # Check if result date falls within range
            if start_date <= result_date <= end_date:
                return 1.0
            else:
                # Calculate proximity to range
                if result_date < start_date:
                    days_diff = (start_date - result_date).days
                else:
                    days_diff = (result_date - end_date).days
                
                # Apply distance penalty
                max_distance_days = 365  # 1 year
                proximity_score = max(0, 1 - (days_diff / max_distance_days))
                return proximity_score
        
        elif start_date:
            # Only start date specified
            if result_date >= start_date:
                return 1.0
            else:
                days_diff = (start_date - result_date).days
                proximity_score = max(0, 1 - (days_diff / 365))
                return proximity_score
        
        elif end_date:
            # Only end date specified
            if result_date <= end_date:
                return 1.0
            else:
                days_diff = (result_date - end_date).days
                proximity_score = max(0, 1 - (days_diff / 365))
                return proximity_score
        
        return 0.5  # No specific date context
    
    def _calculate_date_proximity_score(self, result_date: datetime, target_date: datetime) -> float:
        """Calculate proximity score between two dates."""
        days_diff = abs((result_date - target_date).days)
        
        # Perfect match
        if days_diff == 0:
            return 1.0
        
        # Apply distance decay
        max_distance_days = 365 * 2  # 2 years
        proximity_score = max(0, 1 - (days_diff / max_distance_days))
        
        return proximity_score
    
    def _calculate_temporal_decay_score(self, result_date: datetime) -> float:
        """Calculate temporal decay score based on recency."""
        now = datetime.utcnow()
        days_old = (now - result_date).days
        
        # Apply exponential decay
        decay_factor = self.config.temporal.temporal_decay_factor
        max_age_days = self.config.temporal.max_temporal_range_days
        
        if days_old > max_age_days:
            return 0.1  # Minimum score for very old content
        
        # Calculate decay score
        decay_score = decay_factor ** (days_old / 30)  # Decay over 30-day periods
        
        return decay_score
    
    def get_temporal_search_stats(self) -> Dict[str, Any]:
        """Get temporal search engine statistics."""
        return {
            'indexer_stats': self.temporal_indexer.get_temporal_stats(),
            'cache_size': len(self.search_cache),
            'temporal_config': {
                'decay_factor': self.config.temporal.temporal_decay_factor,
                'max_range_days': self.config.temporal.max_temporal_range_days,
                'resolution': self.config.temporal.temporal_resolution
            }
        }


# Global temporal search engine instance
temporal_search_engine = TemporalSearchEngine()


async def perform_temporal_search(
    search_query: SearchQuery,
    temporal_context: Optional[Dict[str, datetime]] = None
) -> List[VectorSearchResult]:
    """Convenience function to perform temporal search."""
    return await temporal_search_engine.temporal_search(search_query, temporal_context)