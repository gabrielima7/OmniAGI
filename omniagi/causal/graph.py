"""
Causal Graph - Directed Acyclic Graph for causal relationships.

Represents causal relationships between variables using
Structural Causal Models (SCM).
"""

from __future__ import annotations

import json
import logging

try:
    import structlog
except ImportError:
    structlog = None
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from collections import deque

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


@dataclass
class CausalNode:
    """A variable in the causal graph."""
    
    id: str
    name: str
    node_type: str  # treatment, outcome, confounder, mediator, collider
    description: str = ""
    observed: bool = True  # Is this variable observable?
    value: Any = None  # Current value if known
    domain: list[Any] | None = None  # Possible values
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CausalNode":
        return cls(**data)


@dataclass
class CausalEdge:
    """A causal relationship between variables."""
    
    source_id: str
    target_id: str
    mechanism: str = ""  # Description of the causal mechanism
    strength: float = 1.0  # Effect strength (0-1)
    confidence: float = 1.0  # Confidence in this relationship
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CausalEdge":
        return cls(**data)


class CausalGraph:
    """
    Causal graph for representing causal relationships.
    
    Based on Structural Causal Models (SCM). Supports:
    - DAG structure (no cycles)
    - Interventions (do-calculus)
    - Counterfactual queries
    - Confounding detection
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize causal graph.
        
        Args:
            storage_path: Path for persistent storage.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._nodes: dict[str, CausalNode] = {}
        self._edges: list[CausalEdge] = []
        
        # Indexes
        self._parents: dict[str, list[str]] = {}  # node -> parent nodes
        self._children: dict[str, list[str]] = {}  # node -> child nodes
        
        if self.storage_path:
            self._load()
    
    def _load(self) -> None:
        """Load from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            for node_data in data.get("nodes", []):
                node = CausalNode.from_dict(node_data)
                self._nodes[node.id] = node
            
            for edge_data in data.get("edges", []):
                edge = CausalEdge.from_dict(edge_data)
                self._edges.append(edge)
                self._index_edge(edge)
        except Exception as e:
            logger.error("Failed to load causal graph", error=str(e))
    
    def _save(self) -> None:
        """Save to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _index_edge(self, edge: CausalEdge) -> None:
        """Add edge to indexes."""
        if edge.source_id not in self._children:
            self._children[edge.source_id] = []
        self._children[edge.source_id].append(edge.target_id)
        
        if edge.target_id not in self._parents:
            self._parents[edge.target_id] = []
        self._parents[edge.target_id].append(edge.source_id)
    
    def add_node(
        self,
        name: str,
        node_type: str = "observed",
        node_id: str | None = None,
        **kwargs,
    ) -> CausalNode:
        """Add a node to the graph."""
        import uuid
        
        node_id = node_id or str(uuid.uuid4())
        
        node = CausalNode(
            id=node_id,
            name=name,
            node_type=node_type,
            **kwargs,
        )
        
        self._nodes[node_id] = node
        self._save()
        return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        mechanism: str = "",
        strength: float = 1.0,
    ) -> CausalEdge | None:
        """
        Add a causal edge (source causes target).
        
        Returns None if would create a cycle.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        
        # Check for cycles
        if self._would_create_cycle(source_id, target_id):
            logger.warning("Cannot add edge: would create cycle")
            return None
        
        edge = CausalEdge(
            source_id=source_id,
            target_id=target_id,
            mechanism=mechanism,
            strength=strength,
        )
        
        self._edges.append(edge)
        self._index_edge(edge)
        self._save()
        return edge
    
    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding edge would create a cycle."""
        # BFS from target to see if we can reach source
        visited = set()
        queue = deque([target])
        
        while queue:
            current = queue.popleft()
            if current == source:
                return True
            
            if current in visited:
                continue
            visited.add(current)
            
            queue.extend(self._children.get(current, []))
        
        return False
    
    def get_parents(self, node_id: str) -> list[CausalNode]:
        """Get parent nodes (direct causes)."""
        parent_ids = self._parents.get(node_id, [])
        return [self._nodes[id] for id in parent_ids if id in self._nodes]
    
    def get_children(self, node_id: str) -> list[CausalNode]:
        """Get child nodes (direct effects)."""
        child_ids = self._children.get(node_id, [])
        return [self._nodes[id] for id in child_ids if id in self._nodes]
    
    def get_ancestors(self, node_id: str) -> set[str]:
        """Get all ancestor nodes (all causes)."""
        ancestors = set()
        queue = deque(self._parents.get(node_id, []))
        
        while queue:
            current = queue.popleft()
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self._parents.get(current, []))
        
        return ancestors
    
    def get_descendants(self, node_id: str) -> set[str]:
        """Get all descendant nodes (all effects)."""
        descendants = set()
        queue = deque(self._children.get(node_id, []))
        
        while queue:
            current = queue.popleft()
            if current not in descendants:
                descendants.add(current)
                queue.extend(self._children.get(current, []))
        
        return descendants
    
    def find_confounders(self, treatment_id: str, outcome_id: str) -> list[CausalNode]:
        """
        Find confounding variables between treatment and outcome.
        
        A confounder is a common cause of both treatment and outcome.
        """
        treatment_ancestors = self.get_ancestors(treatment_id)
        outcome_ancestors = self.get_ancestors(outcome_id)
        
        # Confounders are in both ancestor sets
        confounder_ids = treatment_ancestors & outcome_ancestors
        
        return [self._nodes[id] for id in confounder_ids if id in self._nodes]
    
    def find_mediators(self, treatment_id: str, outcome_id: str) -> list[CausalNode]:
        """
        Find mediating variables between treatment and outcome.
        
        A mediator is on a causal path from treatment to outcome.
        """
        treatment_descendants = self.get_descendants(treatment_id)
        outcome_ancestors = self.get_ancestors(outcome_id)
        
        # Mediators are descendants of treatment AND ancestors of outcome
        mediator_ids = treatment_descendants & outcome_ancestors
        
        return [self._nodes[id] for id in mediator_ids if id in self._nodes]
    
    def intervene(self, node_id: str, value: Any) -> "CausalGraph":
        """
        Perform an intervention (do-operation).
        
        Creates a new graph where the node is set to value
        and all incoming edges are removed.
        
        Args:
            node_id: Node to intervene on.
            value: Value to set.
            
        Returns:
            New CausalGraph with intervention applied.
        """
        new_graph = CausalGraph()
        
        # Copy nodes
        for node in self._nodes.values():
            new_node = CausalNode(
                id=node.id,
                name=node.name,
                node_type=node.node_type,
                description=node.description,
                observed=node.observed,
                value=value if node.id == node_id else node.value,
                domain=node.domain,
            )
            new_graph._nodes[new_node.id] = new_node
        
        # Copy edges, excluding incoming to intervened node
        for edge in self._edges:
            if edge.target_id != node_id:
                new_edge = CausalEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    mechanism=edge.mechanism,
                    strength=edge.strength,
                )
                new_graph._edges.append(new_edge)
                new_graph._index_edge(new_edge)
        
        return new_graph
    
    def topological_order(self) -> list[str]:
        """Get nodes in topological order."""
        in_degree = {id: 0 for id in self._nodes}
        
        for edge in self._edges:
            in_degree[edge.target_id] += 1
        
        queue = deque([id for id, deg in in_degree.items() if deg == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for child_id in self._children.get(node_id, []):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)
        
        return result
    
    def to_mermaid(self) -> str:
        """Export to Mermaid diagram format."""
        lines = ["graph TD"]
        
        for node in self._nodes.values():
            label = node.name.replace(" ", "_")
            lines.append(f"    {node.id}[{label}]")
        
        for edge in self._edges:
            lines.append(f"    {edge.source_id} --> {edge.target_id}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self._nodes)
