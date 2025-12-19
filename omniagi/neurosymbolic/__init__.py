"""
Neuro-Symbolic AI Module.

Advanced integration of neural networks and symbolic reasoning
for AGI-level intelligence.

Components:
1. Logical Neural Networks (LNN) - differentiable logic
2. Differentiable Reasoning - gradient-based inference
3. Knowledge Graph Neural - graph embeddings with logic
"""

from omniagi.neurosymbolic.neural_logic import (
    LogicalNeuron,
    LNN,
    LogicalOperator,
    create_lnn_from_rules,
)
from omniagi.neurosymbolic.differentiable_reasoning import (
    DifferentiableReasoner,
    SoftUnification,
    NeuralProver,
)
from omniagi.neurosymbolic.knowledge_graph import (
    KnowledgeGraphNeural,
    Entity,
    Relation,
    Triple,
)

__all__ = [
    # LNN
    "LogicalNeuron",
    "LNN",
    "LogicalOperator",
    "create_lnn_from_rules",
    # Differentiable Reasoning
    "DifferentiableReasoner",
    "SoftUnification",
    "NeuralProver",
    # Knowledge Graph
    "KnowledgeGraphNeural",
    "Entity",
    "Relation",
    "Triple",
]
