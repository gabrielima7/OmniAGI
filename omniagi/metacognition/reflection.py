"""
Self-Reflection Engine - Analyze own reasoning and performance.

Enables the AGI to:
1. Introspect its own thought processes
2. Identify cognitive biases and errors
3. Quantify uncertainty in its conclusions
4. Improve through self-analysis

Essential for true AGI - genuine self-awareness.
"""

from __future__ import annotations

import json
import logging

try:
    import structlog
except ImportError:
    structlog = None
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


class ReflectionType(Enum):
    """Types of self-reflection."""
    
    PERFORMANCE = auto()    # How well did I do?
    PROCESS = auto()        # How did I arrive at this?
    BIAS = auto()           # What biases affected me?
    UNCERTAINTY = auto()    # How confident am I?
    IMPROVEMENT = auto()    # How can I do better?


class CognitiveBias(Enum):
    """Known cognitive biases to check for."""
    
    CONFIRMATION = auto()     # Seeking confirming evidence
    ANCHORING = auto()        # Over-relying on first info
    AVAILABILITY = auto()     # Judging by ease of recall
    OVERCONFIDENCE = auto()   # Too certain
    RECENCY = auto()          # Over-weighting recent info
    SUNK_COST = auto()        # Continuing due to past effort


@dataclass
class Reflection:
    """A self-reflection result."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    reflection_type: ReflectionType = ReflectionType.PERFORMANCE
    
    # What was reflected on
    target: str = ""  # What action/decision was analyzed
    context: str = ""
    
    # Analysis results
    observations: list[str] = field(default_factory=list)
    biases_detected: list[CognitiveBias] = field(default_factory=list)
    uncertainty_level: float = 0.5  # 0 = certain, 1 = uncertain
    
    # Conclusions
    lessons: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    
    # Meta
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "type": self.reflection_type.name,
            "target": self.target[:50],
            "observations": len(self.observations),
            "biases": [b.name for b in self.biases_detected],
            "uncertainty": self.uncertainty_level,
        }


@dataclass
class UncertaintyEstimate:
    """Estimate of uncertainty about a conclusion."""
    
    conclusion: str
    confidence: float  # 0-1
    
    # Sources of uncertainty
    data_uncertainty: float = 0.0      # Uncertain input data
    model_uncertainty: float = 0.0     # Model limitations
    reasoning_uncertainty: float = 0.0  # Logical steps uncertain
    
    # Evidence
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    
    @property
    def overall_uncertainty(self) -> float:
        return 1 - self.confidence


class SelfReflectionEngine:
    """
    Self-Reflection Engine for True AGI.
    
    Provides introspective capabilities:
    1. Analyze reasoning processes
    2. Detect cognitive biases
    3. Estimate uncertainty
    4. Generate self-improvement insights
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._reflections: list[Reflection] = []
        self._bias_patterns: dict[CognitiveBias, list] = {b: [] for b in CognitiveBias}
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Self-Reflection Engine initialized")
    
    def reflect_on_decision(
        self,
        decision: str,
        reasoning: str,
        outcome: str = None,
        was_correct: bool = None,
    ) -> Reflection:
        """
        Reflect on a decision that was made.
        """
        reflection = Reflection(
            reflection_type=ReflectionType.PERFORMANCE,
            target=decision,
            context=reasoning,
        )
        
        # Analyze the reasoning
        observations = self._analyze_reasoning(reasoning)
        reflection.observations = observations
        
        # Check for biases
        biases = self._detect_biases(reasoning, decision)
        reflection.biases_detected = biases
        
        # Estimate uncertainty
        reflection.uncertainty_level = self._estimate_uncertainty(reasoning)
        
        # Generate lessons if outcome known
        if outcome is not None or was_correct is not None:
            lessons = self._extract_lessons(decision, reasoning, outcome, was_correct)
            reflection.lessons = lessons
        
        # Suggest improvements
        improvements = self._suggest_improvements(reflection)
        reflection.improvements = improvements
        
        self._reflections.append(reflection)
        self._save()
        
        logger.debug("Reflection completed", type=reflection.reflection_type.name)
        return reflection
    
    def estimate_uncertainty(
        self,
        conclusion: str,
        evidence: list[str] = None,
        reasoning_chain: list[str] = None,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty about a conclusion.
        """
        estimate = UncertaintyEstimate(conclusion=conclusion, confidence=0.5)
        
        # Analyze evidence
        if evidence:
            supporting = []
            contradicting = []
            
            for e in evidence:
                if self._supports_conclusion(e, conclusion):
                    supporting.append(e)
                elif self._contradicts_conclusion(e, conclusion):
                    contradicting.append(e)
            
            estimate.supporting_evidence = supporting
            estimate.contradicting_evidence = contradicting
            
            # Adjust confidence based on evidence balance
            if supporting and not contradicting:
                estimate.confidence = 0.8
            elif contradicting and not supporting:
                estimate.confidence = 0.2
            elif supporting and contradicting:
                ratio = len(supporting) / (len(supporting) + len(contradicting))
                estimate.confidence = 0.3 + 0.4 * ratio
            
            estimate.data_uncertainty = 0.3 if not evidence else 0.1
        
        # Analyze reasoning chain
        if reasoning_chain:
            chain_length = len(reasoning_chain)
            # Longer chains = more uncertainty
            estimate.reasoning_uncertainty = min(0.5, chain_length * 0.05)
            estimate.confidence *= (1 - estimate.reasoning_uncertainty)
        
        return estimate
    
    def check_for_bias(
        self,
        reasoning: str,
        decision: str,
    ) -> list[tuple[CognitiveBias, float, str]]:
        """
        Check for cognitive biases.
        
        Returns list of (bias, confidence, explanation).
        """
        detected = []
        
        # Check confirmation bias
        if self._check_confirmation_bias(reasoning):
            detected.append((
                CognitiveBias.CONFIRMATION,
                0.7,
                "Only considered confirming evidence"
            ))
        
        # Check overconfidence
        if self._check_overconfidence(reasoning, decision):
            detected.append((
                CognitiveBias.OVERCONFIDENCE,
                0.6,
                "Certainty not supported by evidence"
            ))
        
        # Check anchoring
        if self._check_anchoring(reasoning):
            detected.append((
                CognitiveBias.ANCHORING,
                0.5,
                "Over-reliance on initial information"
            ))
        
        # Check recency bias
        if self._check_recency_bias(reasoning):
            detected.append((
                CognitiveBias.RECENCY,
                0.5,
                "Over-weighting recent events"
            ))
        
        return detected
    
    def generate_self_improvement(self) -> list[str]:
        """
        Generate self-improvement suggestions based on reflection history.
        """
        suggestions = []
        
        # Analyze bias patterns
        total_biases = sum(len(v) for v in self._bias_patterns.values())
        if total_biases > 0:
            most_common = max(
                self._bias_patterns.items(),
                key=lambda x: len(x[1])
            )[0]
            suggestions.append(
                f"Work on reducing {most_common.name.lower()} bias - detected {len(self._bias_patterns[most_common])} times"
            )
        
        # Analyze uncertainty
        avg_uncertainty = sum(
            r.uncertainty_level for r in self._reflections
        ) / max(1, len(self._reflections))
        
        if avg_uncertainty > 0.6:
            suggestions.append(
                "Seek more evidence before making conclusions - high average uncertainty"
            )
        
        # Analyze lessons
        all_lessons = []
        for r in self._reflections:
            all_lessons.extend(r.lessons)
        
        if all_lessons:
            suggestions.append(f"Key lessons to remember: {all_lessons[-3:]}")
        
        return suggestions
    
    def _analyze_reasoning(self, reasoning: str) -> list[str]:
        """Analyze a reasoning string."""
        observations = []
        
        reasoning_lower = reasoning.lower()
        
        # Check for structured reasoning
        if "therefore" in reasoning_lower or "because" in reasoning_lower:
            observations.append("Uses logical connectives")
        
        # Check for uncertainty acknowledgment
        if any(w in reasoning_lower for w in ["might", "possibly", "uncertain"]):
            observations.append("Acknowledges uncertainty")
        
        # Check for evidence citation
        if any(w in reasoning_lower for w in ["evidence", "data", "shows"]):
            observations.append("References evidence")
        
        # Check for alternatives considered
        if any(w in reasoning_lower for w in ["alternatively", "however", "but"]):
            observations.append("Considers alternatives")
        
        return observations
    
    def _detect_biases(
        self, reasoning: str, decision: str
    ) -> list[CognitiveBias]:
        """Detect biases in reasoning."""
        biases = []
        bias_checks = self.check_for_bias(reasoning, decision)
        
        for bias, confidence, _ in bias_checks:
            if confidence > 0.5:
                biases.append(bias)
                self._bias_patterns[bias].append({
                    "reasoning": reasoning[:100],
                    "decision": decision[:50],
                })
        
        return biases
    
    def _estimate_uncertainty(self, reasoning: str) -> float:
        """Estimate uncertainty from reasoning text."""
        uncertainty = 0.5
        reasoning_lower = reasoning.lower()
        
        # Certain language decreases uncertainty
        certain_words = ["definitely", "certainly", "clearly", "obviously"]
        if any(w in reasoning_lower for w in certain_words):
            uncertainty -= 0.2
        
        # Uncertain language increases uncertainty
        uncertain_words = ["maybe", "possibly", "might", "perhaps", "unclear"]
        if any(w in reasoning_lower for w in uncertain_words):
            uncertainty += 0.2
        
        return max(0, min(1, uncertainty))
    
    def _extract_lessons(
        self,
        decision: str,
        reasoning: str,
        outcome: str,
        was_correct: bool,
    ) -> list[str]:
        """Extract lessons from a decision outcome."""
        lessons = []
        
        if was_correct:
            lessons.append(f"Approach worked: {decision[:30]}")
        else:
            lessons.append(f"Need different approach for: {decision[:30]}")
            if outcome:
                lessons.append(f"Better outcome needed: {outcome[:50]}")
        
        return lessons
    
    def _suggest_improvements(self, reflection: Reflection) -> list[str]:
        """Suggest improvements based on reflection."""
        improvements = []
        
        if reflection.biases_detected:
            for bias in reflection.biases_detected:
                if bias == CognitiveBias.CONFIRMATION:
                    improvements.append("Actively seek disconfirming evidence")
                elif bias == CognitiveBias.OVERCONFIDENCE:
                    improvements.append("Express uncertainty more explicitly")
                elif bias == CognitiveBias.ANCHORING:
                    improvements.append("Consider multiple starting points")
        
        if reflection.uncertainty_level > 0.7:
            improvements.append("Gather more evidence before deciding")
        
        return improvements
    
    def _check_confirmation_bias(self, reasoning: str) -> bool:
        """Check for confirmation bias."""
        # Simple heuristic: no "however", "but", "although"
        counterargument_words = ["however", "but", "although", "despite", "contrary"]
        return not any(w in reasoning.lower() for w in counterargument_words)
    
    def _check_overconfidence(self, reasoning: str, decision: str) -> bool:
        """Check for overconfidence."""
        certain_words = ["definitely", "certainly", "obvious", "clearly", "100%"]
        uncertain_words = ["evidence", "data", "shows", "because"]
        
        is_certain = any(w in reasoning.lower() for w in certain_words)
        has_evidence = any(w in reasoning.lower() for w in uncertain_words)
        
        return is_certain and not has_evidence
    
    def _check_anchoring(self, reasoning: str) -> bool:
        """Check for anchoring bias."""
        # Check if first-mentioned info is over-weighted
        return "first" in reasoning.lower() or "initial" in reasoning.lower()
    
    def _check_recency_bias(self, reasoning: str) -> bool:
        """Check for recency bias."""
        recency_words = ["recently", "just", "latest", "new"]
        return any(w in reasoning.lower() for w in recency_words)
    
    def _supports_conclusion(self, evidence: str, conclusion: str) -> bool:
        """Check if evidence supports conclusion."""
        # Simple word overlap check
        evidence_words = set(evidence.lower().split())
        conclusion_words = set(conclusion.lower().split())
        overlap = len(evidence_words & conclusion_words)
        return overlap > 2
    
    def _contradicts_conclusion(self, evidence: str, conclusion: str) -> bool:
        """Check if evidence contradicts conclusion."""
        contradiction_words = ["not", "never", "false", "wrong", "incorrect"]
        return any(w in evidence.lower() for w in contradiction_words)
    
    def __len__(self) -> int:
        return len(self._reflections)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "reflections": len(self._reflections),
                "biases": {b.name: len(v) for b, v in self._bias_patterns.items()},
            }, f, indent=2)
    
    def _load(self) -> None:
        pass
