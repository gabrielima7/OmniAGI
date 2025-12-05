"""
Emergent Behavior Detector - Detect novel behaviors emerging from AGI.

Monitors for:
1. Novel problem-solving approaches
2. Unexpected generalizations  
3. Creative insights
4. Self-improvement attempts
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, List
from uuid import uuid4

logger = structlog.get_logger()


class EmergenceType(Enum):
    """Types of emergent behavior."""
    
    NOVEL_SOLUTION = auto()      # New way to solve problem
    GENERALIZATION = auto()      # Applied learning unexpectedly
    CREATIVE_INSIGHT = auto()    # Original idea
    SELF_MODIFICATION = auto()   # Attempted self-improvement
    META_COGNITION = auto()      # Thinking about thinking
    GOAL_GENERATION = auto()     # Created own goal


@dataclass
class EmergentBehavior:
    """A detected emergent behavior."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    emergence_type: EmergenceType = EmergenceType.NOVEL_SOLUTION
    
    description: str = ""
    context: str = ""
    
    # Metrics
    novelty_score: float = 0.5
    significance: float = 0.5
    reproducibility: float = 0.5
    
    # Evidence
    evidence: list[str] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "type": self.emergence_type.name,
            "description": self.description[:50],
            "novelty": self.novelty_score,
            "significance": self.significance,
        }


class EmergenceDetector:
    """
    Emergent Behavior Detector.
    
    Monitors AGI behavior for signs of emergence:
    - Novel solutions not in training
    - Creative insights
    - Self-directed learning
    - Goal creation
    """
    
    def __init__(self):
        self._behaviors: list[EmergentBehavior] = []
        self._baselines: dict[str, float] = {}
        self._thresholds = {
            "novelty": 0.7,
            "significance": 0.6,
        }
        
        logger.info("Emergence Detector initialized")
    
    def observe(
        self,
        action: str,
        context: str,
        expected: str = None,
    ) -> EmergentBehavior | None:
        """
        Observe an action for emergent behavior.
        """
        # Check for novelty
        novelty = self._calculate_novelty(action, expected)
        
        if novelty < self._thresholds["novelty"]:
            return None
        
        # Classify emergence type
        emergence_type = self._classify_emergence(action, context)
        
        # Calculate significance
        significance = self._calculate_significance(action, emergence_type)
        
        if significance < self._thresholds["significance"]:
            return None
        
        # Create behavior record
        behavior = EmergentBehavior(
            emergence_type=emergence_type,
            description=action[:200],
            context=context[:200],
            novelty_score=novelty,
            significance=significance,
            evidence=[f"Action observed: {action[:100]}"],
        )
        
        self._behaviors.append(behavior)
        
        logger.info(
            "Emergent behavior detected",
            type=emergence_type.name,
            novelty=novelty,
        )
        
        return behavior
    
    def _calculate_novelty(self, action: str, expected: str) -> float:
        """Calculate how novel an action is."""
        if expected is None:
            return 0.5  # Unknown baseline
        
        # Simple text similarity
        action_words = set(action.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.8  # No expectation = potentially novel
        
        overlap = len(action_words & expected_words)
        similarity = overlap / max(len(action_words), len(expected_words), 1)
        
        # Novelty is inverse of similarity
        return 1.0 - similarity
    
    def _classify_emergence(self, action: str, context: str) -> EmergenceType:
        """Classify the type of emergent behavior."""
        action_lower = action.lower()
        
        if any(w in action_lower for w in ["create", "invent", "imagine"]):
            return EmergenceType.CREATIVE_INSIGHT
        
        if any(w in action_lower for w in ["improve", "optimize", "modify self"]):
            return EmergenceType.SELF_MODIFICATION
        
        if any(w in action_lower for w in ["think about", "reflect", "aware"]):
            return EmergenceType.META_COGNITION
        
        if any(w in action_lower for w in ["goal", "want", "intend"]):
            return EmergenceType.GOAL_GENERATION
        
        if any(w in action_lower for w in ["apply", "transfer", "generalize"]):
            return EmergenceType.GENERALIZATION
        
        return EmergenceType.NOVEL_SOLUTION
    
    def _calculate_significance(
        self, action: str, emergence_type: EmergenceType
    ) -> float:
        """Calculate significance of the emergence."""
        base_significance = {
            EmergenceType.SELF_MODIFICATION: 0.9,
            EmergenceType.GOAL_GENERATION: 0.85,
            EmergenceType.META_COGNITION: 0.8,
            EmergenceType.CREATIVE_INSIGHT: 0.75,
            EmergenceType.GENERALIZATION: 0.7,
            EmergenceType.NOVEL_SOLUTION: 0.6,
        }
        
        return base_significance.get(emergence_type, 0.5)
    
    def get_emergence_summary(self) -> dict:
        """Get summary of detected emergent behaviors."""
        type_counts = {}
        for b in self._behaviors:
            type_name = b.emergence_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_behaviors": len(self._behaviors),
            "by_type": type_counts,
            "avg_novelty": sum(b.novelty_score for b in self._behaviors) / max(1, len(self._behaviors)),
            "avg_significance": sum(b.significance for b in self._behaviors) / max(1, len(self._behaviors)),
        }
    
    def __len__(self) -> int:
        return len(self._behaviors)
