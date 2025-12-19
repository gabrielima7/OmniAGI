"""
Differentiable Reasoning Module.

Enables gradient-based learning of logical inference,
bridging neural networks and symbolic reasoning.

Key components:
1. Soft Unification - differentiable variable binding
2. Neural Prover - backward chaining with gradients
3. Rule Learning - acquire rules from data
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
class Term:
    """A term in first-order logic."""
    name: str
    is_variable: bool = False
    embedding: Optional[torch.Tensor] = None
    
    def __str__(self):
        return f"?{self.name}" if self.is_variable else self.name


@dataclass
class Atom:
    """An atomic formula (predicate with terms)."""
    predicate: str
    terms: List[Term]
    confidence: float = 1.0
    
    def __str__(self):
        args = ", ".join(str(t) for t in self.terms)
        return f"{self.predicate}({args})"


@dataclass
class Rule:
    """A logical rule: head :- body."""
    name: str
    head: Atom
    body: List[Atom]
    weight: float = 1.0
    
    def __str__(self):
        body_str = ", ".join(str(a) for a in self.body)
        return f"{self.head} :- {body_str}"


class SoftUnification(nn.Module if TORCH_AVAILABLE else object):
    """
    Soft/Differentiable Unification.
    
    Instead of hard unification that returns True/False,
    soft unification returns a similarity score that can
    be backpropagated through.
    
    Uses embedding similarity for entity matching.
    """
    
    def __init__(self, embedding_dim: int = 64):
        _check_torch()
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(1000, embedding_dim)
        self.entity_to_idx: Dict[str, int] = {}
        self.next_idx = 0
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def get_entity_embedding(self, entity: str) -> torch.Tensor:
        """Get or create embedding for an entity."""
        if entity not in self.entity_to_idx:
            self.entity_to_idx[entity] = self.next_idx
            self.next_idx += 1
        
        idx = self.entity_to_idx[entity]
        return self.entity_embeddings(torch.tensor(idx))
    
    def forward(
        self,
        term1: Term,
        term2: Term,
    ) -> torch.Tensor:
        """
        Compute soft unification score between two terms.
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            Similarity score in [0, 1]
        """
        # Variables unify with anything
        if term1.is_variable or term2.is_variable:
            return torch.tensor(1.0)
        
        # Same constant
        if term1.name == term2.name:
            return torch.tensor(1.0)
        
        # Use embedding similarity
        emb1 = self.get_entity_embedding(term1.name)
        emb2 = self.get_entity_embedding(term2.name)
        
        # Cosine similarity
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return torch.sigmoid(sim / self.temperature)
    
    def unify_atoms(
        self,
        atom1: Atom,
        atom2: Atom,
    ) -> torch.Tensor:
        """
        Compute soft unification score between two atoms.
        
        Returns:
            Similarity score in [0, 1]
        """
        # Predicates must match
        if atom1.predicate != atom2.predicate:
            return torch.tensor(0.0)
        
        if len(atom1.terms) != len(atom2.terms):
            return torch.tensor(0.0)
        
        # Product of term unifications
        score = torch.tensor(1.0)
        for t1, t2 in zip(atom1.terms, atom2.terms):
            score = score * self.forward(t1, t2)
        
        return score


class NeuralProver(nn.Module if TORCH_AVAILABLE else object):
    """
    Neural Theorem Prover.
    
    Performs differentiable backward chaining for
    logical inference. Allows learning from data.
    
    Based on Neural Theorem Provers (NTPs).
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        max_depth: int = 5,
    ):
        _check_torch()
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_depth = max_depth
        
        # Soft unification
        self.unifier = SoftUnification(embedding_dim)
        
        # Rule weights
        self.rule_weights = nn.ParameterDict()
        
        # Knowledge base
        self.facts: List[Atom] = []
        self.rules: List[Rule] = []
    
    def add_fact(self, atom: Atom) -> None:
        """Add a fact to the knowledge base."""
        self.facts.append(atom)
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the knowledge base."""
        self.rules.append(rule)
        self.rule_weights[rule.name] = nn.Parameter(
            torch.tensor(rule.weight)
        )
    
    def forward(
        self,
        query: Atom,
        depth: int = 0,
    ) -> torch.Tensor:
        """
        Prove a query using backward chaining.
        
        Args:
            query: Atom to prove
            depth: Current recursion depth
            
        Returns:
            Proof score in [0, 1]
        """
        if depth >= self.max_depth:
            return torch.tensor(0.0)
        
        # Try to unify with facts
        fact_scores = []
        for fact in self.facts:
            score = self.unifier.unify_atoms(query, fact)
            fact_scores.append(score * fact.confidence)
        
        # Try to apply rules
        rule_scores = []
        for rule in self.rules:
            # Unify query with rule head
            head_score = self.unifier.unify_atoms(query, rule.head)
            
            if head_score > 0.1:  # Threshold for efficiency
                # Prove body
                body_score = torch.tensor(1.0)
                for body_atom in rule.body:
                    subproof = self.forward(body_atom, depth + 1)
                    body_score = body_score * subproof
                
                # Combine with rule weight
                weight = torch.sigmoid(self.rule_weights[rule.name])
                rule_scores.append(head_score * body_score * weight)
        
        # Combine all proof paths (soft OR)
        all_scores = fact_scores + rule_scores
        
        if not all_scores:
            return torch.tensor(0.0)
        
        # Noisy-OR combination
        stacked = torch.stack(all_scores)
        result = 1.0 - torch.prod(1.0 - stacked)
        
        return result
    
    def prove(self, query: Atom) -> Tuple[float, List[str]]:
        """
        Prove a query and return explanation.
        
        Returns:
            (score, explanation_steps)
        """
        with torch.no_grad():
            score = self.forward(query)
        
        explanation = self._explain(query)
        
        return score.item(), explanation
    
    def _explain(self, query: Atom) -> List[str]:
        """Generate explanation for a query."""
        explanations = []
        
        # Check facts
        for fact in self.facts:
            if fact.predicate == query.predicate:
                explanations.append(f"Fact: {fact}")
        
        # Check applicable rules
        for rule in self.rules:
            if rule.head.predicate == query.predicate:
                explanations.append(f"Rule: {rule}")
        
        return explanations


class DifferentiableReasoner(nn.Module if TORCH_AVAILABLE else object):
    """
    Complete differentiable reasoning system.
    
    Combines soft unification, neural proving, and
    rule learning for end-to-end trainable reasoning.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        _check_torch()
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Neural prover
        self.prover = NeuralProver(embedding_dim)
        
        # Rule generator (neural -> symbolic)
        self.rule_generator = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        
        # Confidence scorer
        self.confidence_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def add_knowledge(
        self,
        facts: List[Atom] = None,
        rules: List[Rule] = None,
    ) -> None:
        """Add facts and rules to the reasoner."""
        if facts:
            for fact in facts:
                self.prover.add_fact(fact)
        
        if rules:
            for rule in rules:
                self.prover.add_rule(rule)
    
    def forward(self, query: Atom) -> torch.Tensor:
        """Prove a query."""
        return self.prover.forward(query)
    
    def reason(
        self,
        query_predicate: str,
        query_args: List[str],
    ) -> Dict[str, Any]:
        """
        Perform reasoning on a query.
        
        Args:
            query_predicate: Predicate name
            query_args: Argument values
            
        Returns:
            Dictionary with score and explanation
        """
        # Build query atom
        terms = [Term(name=arg) for arg in query_args]
        query = Atom(predicate=query_predicate, terms=terms)
        
        # Prove
        score, explanation = self.prover.prove(query)
        
        return {
            "query": str(query),
            "score": score,
            "is_true": score > 0.5,
            "confidence": score,
            "explanation": explanation,
        }
    
    def learn_from_example(
        self,
        positive_examples: List[Atom],
        negative_examples: List[Atom],
        epochs: int = 100,
        lr: float = 0.01,
    ) -> float:
        """
        Learn rule weights from examples.
        
        Returns:
            Final loss value
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        final_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            loss = torch.tensor(0.0)
            
            # Positive examples should have high scores
            for pos in positive_examples:
                score = self.forward(pos)
                loss = loss + F.binary_cross_entropy(score, torch.tensor(1.0))
            
            # Negative examples should have low scores
            for neg in negative_examples:
                score = self.forward(neg)
                loss = loss + F.binary_cross_entropy(score, torch.tensor(0.0))
            
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
        
        return final_loss
