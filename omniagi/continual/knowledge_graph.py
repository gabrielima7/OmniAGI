"""
Knowledge Graph - Dynamic knowledge representation.

A graph structure for representing concepts, facts,
and their relationships.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = structlog.get_logger()


@dataclass
class Node:
    """A node in the knowledge graph."""
    
    id: str
    label: str
    node_type: str  # concept, fact, procedure, entity
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class Edge:
    """An edge (relationship) in the knowledge graph."""
    
    id: str
    source_id: str
    target_id: str
    relation: str  # is_a, has_part, causes, requires, etc.
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    bidirectional: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Edge":
        return cls(**data)


class KnowledgeGraph:
    """
    Dynamic knowledge graph for continual learning.
    
    Represents knowledge as a graph of:
    - Nodes: concepts, facts, procedures, entities
    - Edges: relationships between nodes
    
    Features:
    - Incremental updates
    - Conflict resolution
    - Subgraph queries
    - Path finding
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize knowledge graph.
        
        Args:
            storage_path: Path for persistent storage.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        
        # Indexes for fast lookup
        self._outgoing: dict[str, list[str]] = {}  # node_id -> edge_ids
        self._incoming: dict[str, list[str]] = {}  # node_id -> edge_ids
        self._by_type: dict[str, list[str]] = {}  # type -> node_ids
        
        if self.storage_path:
            self._load()
    
    def _load(self) -> None:
        """Load graph from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            for node_data in data.get("nodes", []):
                node = Node.from_dict(node_data)
                self._nodes[node.id] = node
                self._index_node(node)
            
            for edge_data in data.get("edges", []):
                edge = Edge.from_dict(edge_data)
                self._edges[edge.id] = edge
                self._index_edge(edge)
            
            logger.info("Knowledge graph loaded", nodes=len(self._nodes), edges=len(self._edges))
        except Exception as e:
            logger.error("Failed to load graph", error=str(e))
    
    def _save(self) -> None:
        """Save graph to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _index_node(self, node: Node) -> None:
        """Add node to indexes."""
        if node.node_type not in self._by_type:
            self._by_type[node.node_type] = []
        if node.id not in self._by_type[node.node_type]:
            self._by_type[node.node_type].append(node.id)
    
    def _index_edge(self, edge: Edge) -> None:
        """Add edge to indexes."""
        if edge.source_id not in self._outgoing:
            self._outgoing[edge.source_id] = []
        self._outgoing[edge.source_id].append(edge.id)
        
        if edge.target_id not in self._incoming:
            self._incoming[edge.target_id] = []
        self._incoming[edge.target_id].append(edge.id)
    
    def add_node(
        self,
        label: str,
        node_type: str,
        node_id: str | None = None,
        confidence: float = 1.0,
        **properties,
    ) -> Node:
        """Add a node to the graph."""
        import uuid
        
        node_id = node_id or str(uuid.uuid4())
        
        # Check for existing node with same label
        existing = self.find_node(label, node_type)
        if existing:
            # Update existing
            existing.properties.update(properties)
            existing.confidence = max(existing.confidence, confidence)
            existing.updated_at = datetime.now()
            self._save()
            return existing
        
        node = Node(
            id=node_id,
            label=label,
            node_type=node_type,
            properties=properties,
            confidence=confidence,
        )
        
        self._nodes[node_id] = node
        self._index_node(node)
        self._save()
        
        return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        edge_id: str | None = None,
        weight: float = 1.0,
        bidirectional: bool = False,
        **properties,
    ) -> Edge | None:
        """Add an edge to the graph."""
        import uuid
        
        if source_id not in self._nodes or target_id not in self._nodes:
            logger.warning("Cannot add edge: source or target not found")
            return None
        
        edge_id = edge_id or str(uuid.uuid4())
        
        edge = Edge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            properties=properties,
            weight=weight,
            bidirectional=bidirectional,
        )
        
        self._edges[edge_id] = edge
        self._index_edge(edge)
        self._save()
        
        return edge
    
    def find_node(self, label: str, node_type: str | None = None) -> Node | None:
        """Find a node by label."""
        for node in self._nodes.values():
            if node.label.lower() == label.lower():
                if node_type is None or node.node_type == node_type:
                    return node
        return None
    
    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        """Get all nodes of a type."""
        ids = self._by_type.get(node_type, [])
        return [self._nodes[id] for id in ids if id in self._nodes]
    
    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        direction: str = "both",  # out, in, both
    ) -> list[tuple[Node, Edge]]:
        """Get neighboring nodes and their edges."""
        results = []
        
        if direction in ("out", "both"):
            for edge_id in self._outgoing.get(node_id, []):
                edge = self._edges.get(edge_id)
                if edge and (relation is None or edge.relation == relation):
                    target = self._nodes.get(edge.target_id)
                    if target:
                        results.append((target, edge))
        
        if direction in ("in", "both"):
            for edge_id in self._incoming.get(node_id, []):
                edge = self._edges.get(edge_id)
                if edge and (relation is None or edge.relation == relation):
                    source = self._nodes.get(edge.source_id)
                    if source:
                        results.append((source, edge))
        
        return results
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[tuple[Node, Edge]] | None:
        """Find a path between two nodes."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        
        # BFS
        from collections import deque
        
        queue = deque([(source_id, [])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_id:
                return path
            
            if len(path) >= max_depth:
                continue
            
            for neighbor, edge in self.get_neighbors(current_id, direction="out"):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [(neighbor, edge)]))
        
        return None
    
    def query(
        self,
        node_type: str | None = None,
        relation: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[Node]:
        """Query nodes with filters."""
        results = []
        
        candidates = (
            self.get_nodes_by_type(node_type)
            if node_type else list(self._nodes.values())
        )
        
        for node in candidates:
            if node.confidence < min_confidence:
                continue
            
            if relation:
                # Check if has any edge with this relation
                neighbors = self.get_neighbors(node.id, relation=relation)
                if not neighbors:
                    continue
            
            results.append(node)
        
        return results
    
    def to_context(self, node_id: str, depth: int = 2) -> str:
        """Generate context string from a node and neighbors."""
        node = self._nodes.get(node_id)
        if not node:
            return ""
        
        lines = [f"## {node.label} ({node.node_type})"]
        
        if node.properties:
            for k, v in node.properties.items():
                lines.append(f"- {k}: {v}")
        
        if depth > 0:
            neighbors = self.get_neighbors(node_id)
            if neighbors:
                lines.append("\n### Relacionados:")
                for neighbor, edge in neighbors[:5]:
                    lines.append(f"- {edge.relation} → {neighbor.label}")
        
        return "\n".join(lines)
    
    def merge(self, other: "KnowledgeGraph") -> dict[str, int]:
        """Merge another graph into this one."""
        stats = {"nodes_added": 0, "edges_added": 0, "conflicts": 0}
        
        for node in other._nodes.values():
            existing = self.find_node(node.label, node.node_type)
            if existing:
                # Merge properties
                existing.properties.update(node.properties)
                existing.confidence = max(existing.confidence, node.confidence)
                stats["conflicts"] += 1
            else:
                self._nodes[node.id] = node
                self._index_node(node)
                stats["nodes_added"] += 1
        
        for edge in other._edges.values():
            if edge.id not in self._edges:
                self._edges[edge.id] = edge
                self._index_edge(edge)
                stats["edges_added"] += 1
        
        self._save()
        return stats
    
    def __len__(self) -> int:
        return len(self._nodes)
