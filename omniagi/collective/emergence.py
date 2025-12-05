"""
Emergence Detector - Detecting emergent behaviors in collective systems.

Monitors for emergent patterns that indicate capabilities
beyond individual agents - a key sign of superintelligence.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = structlog.get_logger()


class PatternType(Enum):
    """Types of emergent patterns."""
    
    COORDINATION = auto()   # Agents coordinating without explicit instruction
    INNOVATION = auto()     # Novel solutions not in training
    ABSTRACTION = auto()    # Higher-level conceptual understanding
    TRANSFER = auto()       # Skills transferring across domains
    COLLECTIVE_MEMORY = auto()  # Shared memory emergence
    SPECIALIZATION = auto()  # Spontaneous role differentiation
    SYNERGY = auto()        # Performance > sum of parts


class SignificanceLevel(Enum):
    """How significant is the emergence."""
    
    NOISE = auto()          # Probably random
    WEAK = auto()           # Slight indication
    MODERATE = auto()       # Notable pattern
    STRONG = auto()         # Clear emergence
    BREAKTHROUGH = auto()   # Major capability gain


@dataclass
class EmergentPattern:
    """A detected emergent pattern."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    pattern_type: PatternType = PatternType.COORDINATION
    significance: SignificanceLevel = SignificanceLevel.WEAK
    
    description: str = ""
    evidence: list[str] = field(default_factory=list)
    agents_involved: list[str] = field(default_factory=list)
    
    # Metrics
    confidence: float = 0.0
    performance_gain: float = 0.0  # vs individual baseline
    
    # Timestamps
    first_detected: str = field(default_factory=lambda: datetime.now().isoformat())
    last_observed: str = field(default_factory=lambda: datetime.now().isoformat())
    observation_count: int = 1
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.pattern_type.name,
            "significance": self.significance.name,
            "description": self.description,
            "evidence": self.evidence[:5],
            "confidence": self.confidence,
            "performance_gain": self.performance_gain,
            "observations": self.observation_count,
        }


@dataclass
class EmergenceMetrics:
    """Metrics for tracking emergence."""
    
    collective_performance: float = 0.0
    individual_baseline: float = 0.0
    synergy_score: float = 0.0
    coordination_index: float = 0.0
    innovation_rate: float = 0.0
    
    def calculate_synergy(self) -> float:
        """Calculate synergy (collective vs sum of individuals)."""
        if self.individual_baseline == 0:
            return 0.0
        self.synergy_score = (
            self.collective_performance - self.individual_baseline
        ) / self.individual_baseline
        return self.synergy_score


class EmergenceDetector:
    """
    Detects emergent behaviors in collective AI systems.
    
    Emergence is a key indicator of superintelligence -
    when the collective exhibits capabilities that
    individual agents don't possess.
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._patterns: dict[str, EmergentPattern] = {}
        self._metrics = EmergenceMetrics()
        self._observations: list[dict] = []
        
        # Thresholds for detection
        self._thresholds = {
            "synergy_weak": 0.1,
            "synergy_moderate": 0.25,
            "synergy_strong": 0.5,
            "coordination_weak": 0.3,
            "coordination_strong": 0.7,
            "innovation_threshold": 0.2,
        }
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Emergence Detector initialized")
    
    def record_observation(
        self,
        agent_performances: dict[str, float],
        collective_performance: float,
        task_domain: str = "general",
        context: dict = None,
    ) -> None:
        """Record an observation for emergence detection."""
        # Calculate baseline (sum of individuals)
        individual_sum = sum(agent_performances.values())
        num_agents = len(agent_performances)
        
        # Update metrics
        self._metrics.individual_baseline = individual_sum
        self._metrics.collective_performance = collective_performance
        self._metrics.calculate_synergy()
        
        observation = {
            "timestamp": datetime.now().isoformat(),
            "agents": list(agent_performances.keys()),
            "individual_sum": individual_sum,
            "collective": collective_performance,
            "synergy": self._metrics.synergy_score,
            "domain": task_domain,
        }
        
        self._observations.append(observation)
        
        # Check for patterns
        self._detect_patterns(observation, context)
        self._save()
    
    def _detect_patterns(self, observation: dict, context: dict = None) -> None:
        """Detect emergent patterns from observation."""
        synergy = observation.get("synergy", 0)
        agents = observation.get("agents", [])
        
        # Synergy detection
        if synergy > self._thresholds["synergy_strong"]:
            self._register_pattern(
                PatternType.SYNERGY,
                SignificanceLevel.STRONG,
                f"Strong synergy detected: {synergy:.2f}",
                agents,
                synergy,
            )
        elif synergy > self._thresholds["synergy_moderate"]:
            self._register_pattern(
                PatternType.SYNERGY,
                SignificanceLevel.MODERATE,
                f"Moderate synergy: {synergy:.2f}",
                agents,
                synergy,
            )
        elif synergy > self._thresholds["synergy_weak"]:
            self._register_pattern(
                PatternType.SYNERGY,
                SignificanceLevel.WEAK,
                f"Weak synergy: {synergy:.2f}",
                agents,
                synergy,
            )
        
        # Coordination detection
        if len(agents) >= 3 and synergy > 0:
            self._metrics.coordination_index = min(1.0, synergy * 2)
            
            if self._metrics.coordination_index > self._thresholds["coordination_strong"]:
                self._register_pattern(
                    PatternType.COORDINATION,
                    SignificanceLevel.STRONG,
                    "High coordination between agents",
                    agents,
                    self._metrics.coordination_index,
                )
    
    def _register_pattern(
        self,
        pattern_type: PatternType,
        significance: SignificanceLevel,
        description: str,
        agents: list[str],
        metric: float,
    ) -> EmergentPattern:
        """Register or update an emergent pattern."""
        # Check for existing pattern of same type
        for pattern in self._patterns.values():
            if pattern.pattern_type == pattern_type:
                # Update existing
                pattern.last_observed = datetime.now().isoformat()
                pattern.observation_count += 1
                pattern.confidence = min(1.0, pattern.confidence + 0.05)
                pattern.performance_gain = max(pattern.performance_gain, metric)
                
                if metric > pattern.confidence:
                    pattern.significance = significance
                
                return pattern
        
        # Create new pattern
        pattern = EmergentPattern(
            pattern_type=pattern_type,
            significance=significance,
            description=description,
            agents_involved=agents,
            confidence=0.3 + min(0.5, metric),
            performance_gain=metric,
        )
        pattern.evidence.append(description)
        
        self._patterns[pattern.id] = pattern
        
        logger.info(
            "Emergent pattern detected",
            type=pattern_type.name,
            significance=significance.name,
        )
        
        return pattern
    
    def detect_innovation(
        self,
        solution: str,
        known_solutions: list[str],
        performance: float,
    ) -> EmergentPattern | None:
        """Detect innovative solutions."""
        # Check if solution is novel
        is_novel = solution not in known_solutions
        
        if not is_novel:
            return None
        
        # Check performance threshold
        if performance < self._thresholds["innovation_threshold"]:
            return None
        
        self._metrics.innovation_rate += 0.1
        
        significance = (
            SignificanceLevel.STRONG if performance > 0.8
            else SignificanceLevel.MODERATE if performance > 0.5
            else SignificanceLevel.WEAK
        )
        
        pattern = self._register_pattern(
            PatternType.INNOVATION,
            significance,
            f"Novel solution with performance {performance:.2f}",
            [],
            performance,
        )
        
        return pattern
    
    def detect_transfer(
        self,
        source_domain: str,
        target_domain: str,
        transfer_performance: float,
    ) -> EmergentPattern | None:
        """Detect cross-domain transfer."""
        if transfer_performance < 0.3:
            return None
        
        significance = (
            SignificanceLevel.BREAKTHROUGH if transfer_performance > 0.9
            else SignificanceLevel.STRONG if transfer_performance > 0.7
            else SignificanceLevel.MODERATE if transfer_performance > 0.5
            else SignificanceLevel.WEAK
        )
        
        return self._register_pattern(
            PatternType.TRANSFER,
            significance,
            f"Transfer from {source_domain} to {target_domain}: {transfer_performance:.2f}",
            [],
            transfer_performance,
        )
    
    def detect_specialization(
        self,
        agent_roles: dict[str, str],
        performance_by_role: dict[str, float],
    ) -> EmergentPattern | None:
        """Detect spontaneous specialization."""
        unique_roles = set(agent_roles.values())
        
        if len(unique_roles) < 2:
            return None
        
        # Check if specialization improves performance
        avg_performance = sum(performance_by_role.values()) / len(performance_by_role)
        
        if avg_performance < 0.5:
            return None
        
        return self._register_pattern(
            PatternType.SPECIALIZATION,
            SignificanceLevel.MODERATE,
            f"Specialization into {len(unique_roles)} roles",
            list(agent_roles.keys()),
            avg_performance,
        )
    
    def get_patterns(
        self,
        pattern_type: PatternType = None,
        min_significance: SignificanceLevel = None,
    ) -> list[EmergentPattern]:
        """Get detected patterns with filters."""
        patterns = list(self._patterns.values())
        
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        if min_significance:
            patterns = [
                p for p in patterns
                if p.significance.value >= min_significance.value
            ]
        
        return sorted(patterns, key=lambda p: p.confidence, reverse=True)
    
    def get_emergence_score(self) -> float:
        """Calculate overall emergence score."""
        if not self._patterns:
            return 0.0
        
        # Weighted sum of pattern confidences
        weights = {
            PatternType.SYNERGY: 1.0,
            PatternType.COORDINATION: 0.8,
            PatternType.INNOVATION: 1.2,
            PatternType.TRANSFER: 1.1,
            PatternType.ABSTRACTION: 1.3,
            PatternType.SPECIALIZATION: 0.7,
            PatternType.COLLECTIVE_MEMORY: 0.9,
        }
        
        total = 0.0
        weight_sum = 0.0
        
        for pattern in self._patterns.values():
            weight = weights.get(pattern.pattern_type, 1.0)
            total += pattern.confidence * weight
            weight_sum += weight
        
        return total / weight_sum if weight_sum > 0 else 0.0
    
    def is_superintelligent(self, threshold: float = 0.7) -> bool:
        """Check if system shows superintelligent emergence."""
        score = self.get_emergence_score()
        
        # Also check for specific breakthrough patterns
        has_breakthrough = any(
            p.significance == SignificanceLevel.BREAKTHROUGH
            for p in self._patterns.values()
        )
        
        return score >= threshold or has_breakthrough
    
    def get_stats(self) -> dict:
        """Get emergence statistics."""
        return {
            "patterns_detected": len(self._patterns),
            "emergence_score": self.get_emergence_score(),
            "is_superintelligent": self.is_superintelligent(),
            "synergy_score": self._metrics.synergy_score,
            "coordination_index": self._metrics.coordination_index,
            "innovation_rate": self._metrics.innovation_rate,
            "observations": len(self._observations),
            "by_type": {
                pt.name: sum(1 for p in self._patterns.values() if p.pattern_type == pt)
                for pt in PatternType
            },
        }
    
    def __len__(self) -> int:
        return len(self._patterns)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "patterns": {k: v.to_dict() for k, v in self._patterns.items()},
                "observations": self._observations[-100:],
                "metrics": {
                    "synergy": self._metrics.synergy_score,
                    "coordination": self._metrics.coordination_index,
                    "innovation": self._metrics.innovation_rate,
                },
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        # Simplified load
        pass
