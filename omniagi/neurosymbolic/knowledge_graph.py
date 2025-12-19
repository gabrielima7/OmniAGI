"""
Knowledge Graph Neural Module.

Combines knowledge graph embeddings with logical constraints
for structured reasoning and inference.

Key features:
1. Entity/Relation embeddings
2. Link prediction with logical rules
3. Graph-based reasoning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch")


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    id: str
    name: str
    entity_type: str = "default"
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Relation:
    """A relation type in the knowledge graph."""
    id: str
    name: str
    symmetric: bool = False
    transitive: bool = False
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Triple:
    """A triple (head, relation, tail) in the knowledge graph."""
    head: Entity
    relation: Relation
    tail: Entity
    confidence: float = 1.0
    
    def __str__(self):
        return f"({self.head.name}, {self.relation.name}, {self.tail.name})"


class KnowledgeGraphNeural(nn.Module if TORCH_AVAILABLE else object):
    """
    Neural Knowledge Graph with logical constraints.
    
    Learns embeddings for entities and relations while
    respecting logical properties like transitivity.
    
    Args:
        embedding_dim: Dimension of embeddings
        margin: Margin for loss function
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        margin: float = 1.0,
    ):
        _check_torch()
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Entity and relation registries
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.triples: List[Triple] = []
        
        # Entity to index mapping
        self.entity_to_idx: Dict[str, int] = {}
        self.relation_to_idx: Dict[str, int] = {}
        
        # Embeddings (initialized when entities are added)
        self.entity_embeddings = None
        self.relation_embeddings = None
        
        # Scoring function parameters
        self.score_fc = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        
        # Logical constraint weights
        self.transitivity_weight = nn.Parameter(torch.tensor(1.0))
        self.symmetry_weight = nn.Parameter(torch.tensor(1.0))
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        if entity.id not in self.entities:
            self.entities[entity.id] = entity
            self.entity_to_idx[entity.id] = len(self.entity_to_idx)
            self._rebuild_embeddings()
    
    def add_relation(self, relation: Relation) -> None:
        """Add a relation type to the graph."""
        if relation.id not in self.relations:
            self.relations[relation.id] = relation
            self.relation_to_idx[relation.id] = len(self.relation_to_idx)
            self._rebuild_embeddings()
    
    def add_triple(self, triple: Triple) -> None:
        """Add a triple to the graph."""
        self.add_entity(triple.head)
        self.add_entity(triple.tail)
        self.add_relation(triple.relation)
        self.triples.append(triple)
    
    def _rebuild_embeddings(self) -> None:
        """Rebuild embedding matrices."""
        num_entities = len(self.entities)
        num_relations = len(self.relations)
        
        if num_entities > 0:
            self.entity_embeddings = nn.Embedding(
                num_entities, self.embedding_dim
            )
            nn.init.xavier_uniform_(self.entity_embeddings.weight)
        
        if num_relations > 0:
            self.relation_embeddings = nn.Embedding(
                num_relations, self.embedding_dim
            )
            nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def get_entity_embedding(self, entity_id: str) -> torch.Tensor:
        """Get embedding for an entity."""
        if entity_id not in self.entity_to_idx:
            raise KeyError(f"Unknown entity: {entity_id}")
        idx = self.entity_to_idx[entity_id]
        return self.entity_embeddings(torch.tensor(idx))
    
    def get_relation_embedding(self, relation_id: str) -> torch.Tensor:
        """Get embedding for a relation."""
        if relation_id not in self.relation_to_idx:
            raise KeyError(f"Unknown relation: {relation_id}")
        idx = self.relation_to_idx[relation_id]
        return self.relation_embeddings(torch.tensor(idx))
    
    def score_triple(self, triple: Triple) -> torch.Tensor:
        """
        Score a triple for plausibility.
        
        Returns:
            Score in [0, 1] indicating likelihood
        """
        head_emb = self.get_entity_embedding(triple.head.id)
        rel_emb = self.get_relation_embedding(triple.relation.id)
        tail_emb = self.get_entity_embedding(triple.tail.id)
        
        # Concatenate and score
        combined = torch.cat([head_emb, rel_emb, tail_emb])
        score = torch.sigmoid(self.score_fc(combined.unsqueeze(0)))
        
        return score.squeeze()
    
    def forward(
        self,
        head_id: str,
        relation_id: str,
        tail_id: str,
    ) -> torch.Tensor:
        """
        Score a triple by IDs.
        
        Returns:
            Plausibility score
        """
        head = self.entities[head_id]
        relation = self.relations[relation_id]
        tail = self.entities[tail_id]
        
        triple = Triple(head=head, relation=relation, tail=tail)
        return self.score_triple(triple)
    
    def predict_tail(
        self,
        head_id: str,
        relation_id: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Predict most likely tail entities.
        
        Returns:
            List of (entity_id, score) tuples
        """
        head = self.entities[head_id]
        relation = self.relations[relation_id]
        
        scores = []
        for tail_id, tail in self.entities.items():
            if tail_id == head_id:
                continue
            triple = Triple(head=head, relation=relation, tail=tail)
            with torch.no_grad():
                score = self.score_triple(triple)
            scores.append((tail_id, score.item()))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def predict_relation(
        self,
        head_id: str,
        tail_id: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Predict most likely relations between entities.
        
        Returns:
            List of (relation_id, score) tuples
        """
        head = self.entities[head_id]
        tail = self.entities[tail_id]
        
        scores = []
        for rel_id, relation in self.relations.items():
            triple = Triple(head=head, relation=relation, tail=tail)
            with torch.no_grad():
                score = self.score_triple(triple)
            scores.append((rel_id, score.item()))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def compute_loss(
        self,
        positive_triples: List[Triple],
        negative_triples: List[Triple],
    ) -> torch.Tensor:
        """
        Compute training loss with logical constraints.
        
        Uses margin ranking loss with additional
        penalties for logical constraint violations.
        """
        loss = torch.tensor(0.0)
        
        # Margin ranking loss
        for pos in positive_triples:
            pos_score = self.score_triple(pos)
            for neg in negative_triples:
                neg_score = self.score_triple(neg)
                loss = loss + F.relu(self.margin - pos_score + neg_score)
        
        # Logical constraint losses
        for triple in positive_triples:
            relation = triple.relation
            
            # Transitivity constraint
            if relation.transitive:
                trans_loss = self._transitivity_loss(triple)
                loss = loss + self.transitivity_weight * trans_loss
            
            # Symmetry constraint
            if relation.symmetric:
                sym_loss = self._symmetry_loss(triple)
                loss = loss + self.symmetry_weight * sym_loss
        
        return loss
    
    def _transitivity_loss(self, triple: Triple) -> torch.Tensor:
        """
        Transitivity constraint: (a,r,b) and (b,r,c) implies (a,r,c)
        """
        # Find triples that could form transitive chains
        loss = torch.tensor(0.0)
        
        for other in self.triples:
            if (other.relation.id == triple.relation.id and 
                other.head.id == triple.tail.id):
                # We have (a,r,b) and (b,r,c), check (a,r,c)
                implied = Triple(
                    head=triple.head,
                    relation=triple.relation,
                    tail=other.tail,
                )
                implied_score = self.score_triple(implied)
                # Should be high
                loss = loss + F.relu(0.5 - implied_score)
        
        return loss
    
    def _symmetry_loss(self, triple: Triple) -> torch.Tensor:
        """
        Symmetry constraint: (a,r,b) implies (b,r,a)
        """
        # Score the symmetric triple
        symmetric = Triple(
            head=triple.tail,
            relation=triple.relation,
            tail=triple.head,
        )
        
        orig_score = self.score_triple(triple)
        sym_score = self.score_triple(symmetric)
        
        # Scores should be similar
        return (orig_score - sym_score).abs()
    
    def infer(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Answer a query using the knowledge graph.
        
        Args:
            query: Query string like "head relation ?"
            
        Returns:
            Dictionary with answers
        """
        parts = query.strip().split()
        
        if len(parts) == 3 and parts[2] == "?":
            # Tail prediction
            head_id = parts[0]
            rel_id = parts[1]
            predictions = self.predict_tail(head_id, rel_id)
            return {
                "type": "tail_prediction",
                "query": query,
                "answers": [
                    {"entity": self.entities[eid].name, "score": score}
                    for eid, score in predictions
                ],
            }
        
        elif len(parts) == 3 and parts[1] == "?":
            # Relation prediction
            head_id = parts[0]
            tail_id = parts[2]
            predictions = self.predict_relation(head_id, tail_id)
            return {
                "type": "relation_prediction",
                "query": query,
                "answers": [
                    {"relation": self.relations[rid].name, "score": score}
                    for rid, score in predictions
                ],
            }
        
        return {"type": "unknown", "query": query, "answers": []}
    
    def get_stats(self) -> Dict[str, int]:
        """Get knowledge graph statistics."""
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "triples": len(self.triples),
        }
