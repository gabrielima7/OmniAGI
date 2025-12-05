"""
Meta-Learning module - Learning to learn.

Adapts learning strategies based on past experiences
for rapid adaptation to new domains and tasks.
"""

from omniagi.meta.strategy import StrategyBank, Strategy
from omniagi.meta.adapter import StrategyAdapter
from omniagi.meta.learner import MetaLearner

__all__ = [
    "StrategyBank",
    "Strategy",
    "StrategyAdapter",
    "MetaLearner",
]
