"""
Continual Learning module - Avoid catastrophic forgetting.

Enables continuous knowledge accumulation without
forgetting previously learned information.
"""

from omniagi.continual.consolidator import MemoryConsolidator
from omniagi.continual.knowledge_graph import KnowledgeGraph, Node, Edge
from omniagi.continual.curriculum import CurriculumLearner

__all__ = [
    "MemoryConsolidator",
    "KnowledgeGraph",
    "Node",
    "Edge",
    "CurriculumLearner",
]
