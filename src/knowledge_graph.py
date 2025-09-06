"""
Knowledge graph implementation using NetworkX.

This module provides graph-based search capabilities, temporal validation,
and entity resolution for the knowledge graph system.
"""

import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import networkx as nx

from .config import get_config
from .models import (
    AtomicFact,
    TemporalEvent,
    VectorSearchResult,
    SearchMethod,
    ContentType,
    DomainType,
    VectorMetadata,
    EmbeddingProvider
)


logger = logging.getLogger(__name__)


class TemporalKnowledgeGraph:
    """Temporal knowledge graph implementation using NetworkX."""
    
    def __init__(self):
        """Initialize temporal knowledge graph."""
        self.config = get_config()
        self.graph = nx.MultiDiGraph()  # Directed multigraph for temporal relationships
        self.entity_index = {}  # Entity ID to node mapping
        self.temporal_index = {}  # Time-based index for events
        
        # Statistics
        self.stats = {
            'nodes': 0,
            'edges': 0,
            'entities': 0,
            'facts': 0,
            'events': 0
        }
    
    def add_fact(self, fact: AtomicFact) -> bool:
        """Add an atomic fact to the knowledge graph."""
        try:
            # Create nodes for subject and object if they don't exist
            subject_node = self._ensure_entity_node(fact.subject, fact.domain)
            object_node = self._ensure_entity_node(fact.object, fact.domain)
            
            # Create edge with fact information
            edge_data = {
                'fact_id': str(fact.fact_id),
                'predicate': fact.predicate,
                'confidence': fact.confidence,
                'source_chunk_id': str(fact.source_chunk_id),
                'source_document_id': str(fact.source_document_id),
                'extraction_method': fact.extraction_method,
                'domain': fact.domain.value,
                'verified': fact.verified,
                'type': 'fact'
            }
            
            # Add temporal information if available
            if fact.valid_from:
                edge_data['valid_from'] = fact.valid_from.isoformat()
            if fact.valid_to:
                edge_data['valid_to'] = fact.valid_to.isoformat()
            if fact.temporal_context:
                edge_data['temporal_context'] = fact.temporal_context
            
            # Add edge to graph
            self.graph.add_edge(subject_node, object_node, **edge_data)
            
            self.stats['facts'] += 1
            self._update_graph_stats()
            
            logger.debug(f"Added fact: {fact.subject} -> {fact.predicate} -> {fact.object}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add fact to graph: {e}")
            return False
    
    def add_event(self, event: TemporalEvent) -> bool:
        """Add a temporal event to the knowledge graph."""
        try:
            # Create event node
            event_node = f"event_{event.event_id}"
            
            event_data = {
                'type': 'event',
                'event_id': str(event.event_id),
                'event_type': event.event_type,
                'description': event.description,
                'start_time': event.start_time.isoformat(),
                'confidence': event.confidence,
                'domain': event.domain.value,
                'source_chunk_id': str(event.source_chunk_id)
            }
            
            if event.end_time:
                event_data['end_time'] = event.end_time.isoformat()
            if event.duration:
                event_data['duration'] = event.duration
            
            self.graph.add_node(event_node, **event_data)
            
            # Connect event to primary entities
            for entity_name in event.primary_entities:
                entity_node = self._ensure_entity_node(entity_name, event.domain)
                self.graph.add_edge(entity_node, event_node, 
                                  relationship='participates_in',
                                  type='event_participation',
                                  role='primary')
            
            # Connect event to secondary entities
            for entity_name in event.secondary_entities:
                entity_node = self._ensure_entity_node(entity_name, event.domain)
                self.graph.add_edge(entity_node, event_node,
                                  relationship='associated_with',
                                  type='event_association',
                                  role='secondary')
            
            # Add to temporal index
            time_key = event.start_time.strftime('%Y-%m-%d')
            if time_key not in self.temporal_index:
                self.temporal_index[time_key] = []
            self.temporal_index[time_key].append(event_node)
            
            self.stats['events'] += 1
            self._update_graph_stats()
            
            logger.debug(f"Added event: {event.description}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add event to graph: {e}")
            return False
    
    def _ensure_entity_node(self, entity_name: str, domain: DomainType) -> str:
        """Ensure entity node exists in graph."""
        # Normalize entity name
        normalized_name = self._normalize_entity_name(entity_name)
        node_id = f"entity_{normalized_name}"
        
        if not self.graph.has_node(node_id):
            node_data = {
                'type': 'entity',
                'name': entity_name,
                'normalized_name': normalized_name,
                'domain': domain.value,
                'aliases': [entity_name],
                'confidence': 1.0
            }
            
            self.graph.add_node(node_id, **node_data)
            self.entity_index[normalized_name] = node_id
            self.stats['entities'] += 1
        else:
            # Update aliases if needed
            node_data = self.graph.nodes[node_id]
            if entity_name not in node_data.get('aliases', []):
                node_data['aliases'].append(entity_name)
        
        return node_id
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for consistent indexing."""
        # Convert to lowercase and remove extra whitespace
        normalized = name.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes = ['the ', 'a ', 'an ']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        # Remove punctuation
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        # Replace spaces with underscores
        normalized = '_'.join(normalized.split())
        
        return normalized
    
    def find_entity_paths(
        self, 
        start_entity: str, 
        end_entity: str, 
        max_depth: Optional[int] = None
    ) -> List[List[str]]:
        """Find paths between two entities."""
        max_depth = max_depth or self.config.graph.max_depth
        
        try:
            # Normalize entity names
            start_normalized = self._normalize_entity_name(start_entity)
            end_normalized = self._normalize_entity_name(end_entity)
            
            # Get node IDs
            start_node = self.entity_index.get(start_normalized)
            end_node = self.entity_index.get(end_normalized)
            
            if not start_node or not end_node:
                return []
            
            # Find all simple paths up to max_depth
            try:
                paths = list(nx.all_simple_paths(
                    self.graph, 
                    start_node, 
                    end_node, 
                    cutoff=max_depth
                ))
                return paths
            except nx.NetworkXNoPath:
                return []
            
        except Exception as e:
            logger.error(f"Failed to find paths between {start_entity} and {end_entity}: {e}")
            return []
    
    def get_entity_neighbors(
        self, 
        entity_name: str, 
        depth: int = 1,
        relationship_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get neighboring entities and their relationships."""
        try:
            normalized_name = self._normalize_entity_name(entity_name)
            entity_node = self.entity_index.get(normalized_name)
            
            if not entity_node:
                return []
            
            neighbors = []
            visited = {entity_node}
            queue = deque([(entity_node, 0)])
            
            while queue:
                current_node, current_depth = queue.popleft()
                
                if current_depth >= depth:
                    continue
                
                # Get all neighbors (both incoming and outgoing)
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        # Get edge data
                        edge_data = self.graph.get_edge_data(current_node, neighbor)
                        
                        # Apply relationship filter if specified
                        if relationship_filter:
                            has_matching_relationship = False
                            for edge_key, edge_attrs in edge_data.items():
                                if edge_attrs.get('predicate') == relationship_filter:
                                    has_matching_relationship = True
                                    break
                            
                            if not has_matching_relationship:
                                continue
                        
                        # Add neighbor info
                        neighbor_data = self.graph.nodes.get(neighbor, {})
                        neighbor_info = {
                            'node_id': neighbor,
                            'name': neighbor_data.get('name', neighbor),
                            'type': neighbor_data.get('type', 'unknown'),
                            'domain': neighbor_data.get('domain', 'general'),
                            'distance': current_depth + 1,
                            'relationships': []
                        }
                        
                        # Add relationship details
                        for edge_key, edge_attrs in edge_data.items():
                            relationship = {
                                'predicate': edge_attrs.get('predicate', 'related_to'),
                                'confidence': edge_attrs.get('confidence', 0.5),
                                'type': edge_attrs.get('type', 'unknown')
                            }
                            neighbor_info['relationships'].append(relationship)
                        
                        neighbors.append(neighbor_info)
                        visited.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Failed to get neighbors for entity {entity_name}: {e}")
            return []
    
    def search_by_temporal_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Search for facts and events within a temporal range."""
        results = []
        
        try:
            # Search through all nodes and edges for temporal information
            for node_id, node_data in self.graph.nodes(data=True):
                if node_data.get('type') == 'event':
                    event_start = node_data.get('start_time')
                    if event_start:
                        event_start_dt = datetime.fromisoformat(event_start)
                        if start_date <= event_start_dt <= end_date:
                            results.append({
                                'type': 'event',
                                'node_id': node_id,
                                'data': node_data,
                                'temporal_match': event_start_dt
                            })
            
            # Search edges (facts) for temporal information
            for source, target, edge_data in self.graph.edges(data=True):
                if edge_data.get('type') == 'fact':
                    valid_from = edge_data.get('valid_from')
                    valid_to = edge_data.get('valid_to')
                    
                    if valid_from:
                        valid_from_dt = datetime.fromisoformat(valid_from)
                        
                        # Check if fact is valid within the range
                        fact_end = datetime.fromisoformat(valid_to) if valid_to else datetime.utcnow()
                        
                        if (valid_from_dt <= end_date and fact_end >= start_date):
                            results.append({
                                'type': 'fact',
                                'source': source,
                                'target': target,
                                'data': edge_data,
                                'temporal_match': valid_from_dt
                            })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by temporal range: {e}")
            return []
    
    def search_by_keywords(
        self, 
        keywords: List[str],
        search_entities: bool = True,
        search_descriptions: bool = True
    ) -> List[VectorSearchResult]:
        """Search graph by keywords."""
        results = []
        
        try:
            keywords_lower = [kw.lower() for kw in keywords]
            
            # Search entity names and aliases
            if search_entities:
                for node_id, node_data in self.graph.nodes(data=True):
                    if node_data.get('type') == 'entity':
                        name = node_data.get('name', '').lower()
                        aliases = [alias.lower() for alias in node_data.get('aliases', [])]
                        
                        # Check if any keyword matches
                        for keyword in keywords_lower:
                            if (keyword in name or 
                                any(keyword in alias for alias in aliases)):
                                
                                # Create search result
                                result = self._create_graph_search_result(
                                    node_id, node_data, keyword, 'entity_match'
                                )
                                results.append(result)
                                break
            
            # Search event descriptions
            if search_descriptions:
                for node_id, node_data in self.graph.nodes(data=True):
                    if node_data.get('type') == 'event':
                        description = node_data.get('description', '').lower()
                        event_type = node_data.get('event_type', '').lower()
                        
                        for keyword in keywords_lower:
                            if keyword in description or keyword in event_type:
                                result = self._create_graph_search_result(
                                    node_id, node_data, keyword, 'event_match'
                                )
                                results.append(result)
                                break
                
                # Search fact predicates
                for source, target, edge_data in self.graph.edges(data=True):
                    if edge_data.get('type') == 'fact':
                        predicate = edge_data.get('predicate', '').lower()
                        
                        for keyword in keywords_lower:
                            if keyword in predicate:
                                result = self._create_graph_search_result(
                                    f"{source}->{target}", edge_data, keyword, 'fact_match'
                                )
                                results.append(result)
                                break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by keywords: {e}")
            return []
    
    def _create_graph_search_result(
        self, 
        item_id: str, 
        item_data: Dict[str, Any], 
        matched_keyword: str,
        match_type: str
    ) -> VectorSearchResult:
        """Create a VectorSearchResult from graph search."""
        # Generate content based on item type
        if item_data.get('type') == 'entity':
            content = f"Entity: {item_data.get('name', item_id)}"
            if item_data.get('aliases'):
                content += f" (aliases: {', '.join(item_data['aliases'])})"
        
        elif item_data.get('type') == 'event':
            content = f"Event: {item_data.get('description', item_id)}"
            if item_data.get('start_time'):
                content += f" (Time: {item_data['start_time']})"
        
        elif item_data.get('type') == 'fact':
            content = f"Fact: {item_data.get('predicate', 'related_to')}"
        
        else:
            content = str(item_data)
        
        # Create metadata
        domain_str = item_data.get('domain', 'general')
        try:
            domain = DomainType(domain_str)
        except ValueError:
            domain = DomainType.GENERAL
        
        metadata = VectorMetadata(
            document_id=uuid.uuid4(),
            content_type=ContentType.ENTITY,
            domain=domain,
            confidence_score=item_data.get('confidence', 0.8),
            embedding_provider=EmbeddingProvider.GOOGLE,
            source=f"knowledge_graph_{match_type}"
        )
        
        # Calculate similarity score based on keyword match
        similarity_score = 0.9 if matched_keyword in content.lower() else 0.7
        
        return VectorSearchResult(
            content_id=uuid.uuid4(),
            content=content,
            similarity_score=similarity_score,
            metadata=metadata,
            search_method=SearchMethod.GRAPH,
            ranking_factors={'keyword_match': similarity_score, 'graph_relevance': 1.0}
        )
    
    def get_subgraph(
        self, 
        center_entity: str, 
        radius: int = 2,
        include_temporal: bool = True
    ) -> nx.Graph:
        """Extract subgraph around a central entity."""
        try:
            normalized_name = self._normalize_entity_name(center_entity)
            center_node = self.entity_index.get(normalized_name)
            
            if not center_node:
                return nx.Graph()
            
            # Get all nodes within radius
            subgraph_nodes = set([center_node])
            current_level = {center_node}
            
            for level in range(radius):
                next_level = set()
                for node in current_level:
                    neighbors = set(self.graph.neighbors(node))
                    next_level.update(neighbors)
                
                subgraph_nodes.update(next_level)
                current_level = next_level
            
            # Create subgraph
            subgraph = self.graph.subgraph(subgraph_nodes).copy()
            
            # Filter temporal information if requested
            if not include_temporal:
                nodes_to_remove = []
                for node_id, node_data in subgraph.nodes(data=True):
                    if node_data.get('type') == 'event':
                        nodes_to_remove.append(node_id)
                
                subgraph.remove_nodes_from(nodes_to_remove)
            
            return subgraph
            
        except Exception as e:
            logger.error(f"Failed to get subgraph for entity {center_entity}: {e}")
            return nx.Graph()
    
    def _update_graph_stats(self):
        """Update graph statistics."""
        self.stats['nodes'] = self.graph.number_of_nodes()
        self.stats['edges'] = self.graph.number_of_edges()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        self._update_graph_stats()
        
        # Add additional statistics
        stats = self.stats.copy()
        
        # Calculate graph metrics
        if self.stats['nodes'] > 0:
            stats['density'] = nx.density(self.graph)
            
            # Get largest connected component
            if self.graph.is_directed():
                largest_component = max(
                    nx.weakly_connected_components(self.graph), 
                    key=len, 
                    default=set()
                )
            else:
                largest_component = max(
                    nx.connected_components(self.graph), 
                    key=len, 
                    default=set()
                )
            
            stats['largest_component_size'] = len(largest_component)
            stats['component_ratio'] = len(largest_component) / self.stats['nodes']
        
        # Domain distribution
        domain_counts = defaultdict(int)
        for node_id, node_data in self.graph.nodes(data=True):
            domain = node_data.get('domain', 'general')
            domain_counts[domain] += 1
        
        stats['domain_distribution'] = dict(domain_counts)
        
        return stats


# Global temporal knowledge graph instance
temporal_knowledge_graph = TemporalKnowledgeGraph()