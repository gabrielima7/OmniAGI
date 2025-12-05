"""
Ouroboros - Self-improvement system.

The serpent that eats its own tail - a system that improves its own code.
"""

from omniagi.ouroboros.analyzer import CodeAnalyzer
from omniagi.ouroboros.critic import CriticAgent
from omniagi.ouroboros.refactor import Refactorer
from omniagi.ouroboros.loop import OuroborosLoop

__all__ = [
    "CodeAnalyzer",
    "CriticAgent",
    "Refactorer",
    "OuroborosLoop",
]
