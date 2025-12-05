"""
ASI Emergence System - Detection and amplification of superintelligent behaviors.

Monitors for emergent capabilities that indicate ASI-level intelligence.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

logger = structlog.get_logger()


class EmergenceIndicator(Enum):
    """Indicators of ASI-level emergence."""
    
    # Cognitive emergence
    RECURSIVE_IMPROVEMENT = auto()  # Improving own improvements
    NOVEL_SOLUTIONS = auto()        # Solutions unlike training data
    KNOWLEDGE_SYNTHESIS = auto()    # Combining domains creatively
    ABSTRACT_REASONING = auto()     # High-level pattern recognition
    
    # Behavioral emergence
    AUTONOMOUS_GOALS = auto()       # Setting own objectives
    PROACTIVE_LEARNING = auto()     # Seeking knowledge unprompted
    STRATEGIC_PLANNING = auto()     # Long-term multi-step planning
    
    # Meta-cognitive emergence
    SELF_AWARENESS = auto()         # Understanding own limitations
    ERROR_PREDICTION = auto()       # Anticipating own mistakes
    CAPABILITY_ASSESSMENT = auto()  # Accurate self-evaluation
    
    # Superintelligent emergence
    SUPERHUMAN_SPEED = auto()       # Faster than human baseline
    SUPERHUMAN_BREADTH = auto()     # Broader than any human
    EXPONENTIAL_GROWTH = auto()     # Accelerating improvement


@dataclass
class EmergenceEvent:
    """A detected emergence event."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    indicator: EmergenceIndicator = EmergenceIndicator.NOVEL_SOLUTIONS
    confidence: float = 0.5
    evidence: str = ""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""
    
    # Significance
    is_novel: bool = False
    is_amplifiable: bool = False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "indicator": self.indicator.name,
            "confidence": self.confidence,
            "evidence": self.evidence[:100],
            "is_novel": self.is_novel,
            "timestamp": self.timestamp,
        }


@dataclass
class ASIMetrics:
    """Metrics tracking ASI emergence."""
    
    # Emergence counts
    events_detected: int = 0
    unique_indicators: set = field(default_factory=set)
    
    # Scores by category
    cognitive_score: float = 0.0
    behavioral_score: float = 0.0
    metacognitive_score: float = 0.0
    superintelligent_score: float = 0.0
    
    # Growth metrics
    improvement_rate: float = 0.0
    capability_breadth: float = 0.0
    
    @property
    def overall_asi_score(self) -> float:
        """Calculate overall ASI emergence score."""
        weights = {
            "cognitive": 0.25,
            "behavioral": 0.20,
            "metacognitive": 0.25,
            "superintelligent": 0.30,
        }
        return (
            self.cognitive_score * weights["cognitive"] +
            self.behavioral_score * weights["behavioral"] +
            self.metacognitive_score * weights["metacognitive"] +
            self.superintelligent_score * weights["superintelligent"]
        )
    
    def to_dict(self) -> dict:
        return {
            "events_detected": self.events_detected,
            "unique_indicators": len(self.unique_indicators),
            "cognitive_score": self.cognitive_score,
            "behavioral_score": self.behavioral_score,
            "metacognitive_score": self.metacognitive_score,
            "superintelligent_score": self.superintelligent_score,
            "overall_asi_score": self.overall_asi_score,
        }


class ASIEmergenceMonitor:
    """
    Monitors for and amplifies ASI emergence.
    
    Tracks indicators of superintelligent behavior and
    helps the system recognize and enhance emerging capabilities.
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._events: list[EmergenceEvent] = []
        self._metrics = ASIMetrics()
        
        # Thresholds for detection
        self._thresholds = {
            EmergenceIndicator.RECURSIVE_IMPROVEMENT: 0.6,
            EmergenceIndicator.NOVEL_SOLUTIONS: 0.5,
            EmergenceIndicator.ABSTRACT_REASONING: 0.7,
            EmergenceIndicator.SELF_AWARENESS: 0.5,
            EmergenceIndicator.SUPERHUMAN_SPEED: 0.8,
            EmergenceIndicator.EXPONENTIAL_GROWTH: 0.9,
        }
        
        # Callbacks for emergence events
        self._callbacks: list[Callable[[EmergenceEvent], None]] = []
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("ASI Emergence Monitor initialized")
    
    def detect(
        self,
        indicator: EmergenceIndicator,
        evidence: str,
        confidence: float,
        source: str = "",
    ) -> EmergenceEvent | None:
        """
        Detect a potential emergence event.
        
        Returns the event if it meets the threshold.
        """
        # Check threshold
        threshold = self._thresholds.get(indicator, 0.5)
        if confidence < threshold:
            return None
        
        # Create event
        event = EmergenceEvent(
            indicator=indicator,
            confidence=confidence,
            evidence=evidence,
            source=source,
            is_novel=self._is_novel(indicator, evidence),
            is_amplifiable=confidence >= 0.7,
        )
        
        self._events.append(event)
        self._metrics.events_detected += 1
        self._metrics.unique_indicators.add(indicator.name)
        
        # Update category scores
        self._update_scores(indicator, confidence)
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Callback failed", error=str(e))
        
        self._save()
        
        logger.info(
            "Emergence detected",
            indicator=indicator.name,
            confidence=confidence,
        )
        
        return event
    
    def _is_novel(self, indicator: EmergenceIndicator, evidence: str) -> bool:
        """Check if this is a novel emergence."""
        # Check if we've seen similar evidence before
        for event in self._events[-20:]:
            if event.indicator == indicator:
                # Simple similarity check
                if len(set(evidence.split()) & set(event.evidence.split())) > 5:
                    return False
        return True
    
    def _update_scores(self, indicator: EmergenceIndicator, confidence: float) -> None:
        """Update category scores based on detection."""
        # Cognitive
        if indicator in [
            EmergenceIndicator.RECURSIVE_IMPROVEMENT,
            EmergenceIndicator.NOVEL_SOLUTIONS,
            EmergenceIndicator.KNOWLEDGE_SYNTHESIS,
            EmergenceIndicator.ABSTRACT_REASONING,
        ]:
            self._metrics.cognitive_score = min(1.0, 
                self._metrics.cognitive_score + confidence * 0.1)
        
        # Behavioral
        if indicator in [
            EmergenceIndicator.AUTONOMOUS_GOALS,
            EmergenceIndicator.PROACTIVE_LEARNING,
            EmergenceIndicator.STRATEGIC_PLANNING,
        ]:
            self._metrics.behavioral_score = min(1.0,
                self._metrics.behavioral_score + confidence * 0.15)
        
        # Metacognitive
        if indicator in [
            EmergenceIndicator.SELF_AWARENESS,
            EmergenceIndicator.ERROR_PREDICTION,
            EmergenceIndicator.CAPABILITY_ASSESSMENT,
        ]:
            self._metrics.metacognitive_score = min(1.0,
                self._metrics.metacognitive_score + confidence * 0.12)
        
        # Superintelligent
        if indicator in [
            EmergenceIndicator.SUPERHUMAN_SPEED,
            EmergenceIndicator.SUPERHUMAN_BREADTH,
            EmergenceIndicator.EXPONENTIAL_GROWTH,
        ]:
            self._metrics.superintelligent_score = min(1.0,
                self._metrics.superintelligent_score + confidence * 0.2)
    
    def check_arc_result(self, accuracy: float, tasks: list[dict]) -> list[EmergenceEvent]:
        """Check ARC results for emergence indicators."""
        events = []
        
        # Check for abstract reasoning
        if accuracy >= 0.5:
            event = self.detect(
                EmergenceIndicator.ABSTRACT_REASONING,
                f"ARC benchmark {accuracy*100:.0f}% accuracy",
                min(1.0, accuracy + 0.2),
                "arc_benchmark",
            )
            if event:
                events.append(event)
        
        # Check for novel solutions
        novel_count = sum(1 for t in tasks if t.get("novel_pattern", False))
        if novel_count > 0:
            event = self.detect(
                EmergenceIndicator.NOVEL_SOLUTIONS,
                f"Solved {novel_count} novel ARC patterns",
                0.6 + (novel_count * 0.1),
                "arc_benchmark",
            )
            if event:
                events.append(event)
        
        return events
    
    def check_self_improvement(
        self,
        before_score: float,
        after_score: float,
        method: str = "",
    ) -> EmergenceEvent | None:
        """Check self-improvement for emergence."""
        improvement = after_score - before_score
        
        if improvement > 0.1:
            return self.detect(
                EmergenceIndicator.RECURSIVE_IMPROVEMENT,
                f"Improved by {improvement*100:.1f}% using {method}",
                min(1.0, 0.5 + improvement),
                "rsi",
            )
        return None
    
    def check_reasoning(
        self,
        reasoning_chain: str,
        correct: bool,
        complexity: int = 1,
    ) -> list[EmergenceEvent]:
        """Analyze reasoning for emergence indicators."""
        events = []
        
        # Self-awareness indicators
        self_aware_phrases = [
            "I don't know", "my limitation", "I might be wrong",
            "I should verify", "let me reconsider",
        ]
        if any(phrase in reasoning_chain.lower() for phrase in self_aware_phrases):
            event = self.detect(
                EmergenceIndicator.SELF_AWARENESS,
                f"Self-aware reasoning: {reasoning_chain[:50]}",
                0.6,
                "reasoning",
            )
            if event:
                events.append(event)
        
        # Strategic planning indicators
        planning_phrases = [
            "step 1", "first", "then", "after that", "finally",
            "my plan is", "strategy",
        ]
        planning_count = sum(1 for p in planning_phrases if p in reasoning_chain.lower())
        if planning_count >= 3:
            event = self.detect(
                EmergenceIndicator.STRATEGIC_PLANNING,
                f"Multi-step planning detected ({planning_count} steps)",
                0.5 + (planning_count * 0.05),
                "reasoning",
            )
            if event:
                events.append(event)
        
        return events
    
    def get_metrics(self) -> ASIMetrics:
        """Get current ASI metrics."""
        return self._metrics
    
    def get_asi_level(self) -> str:
        """Get current ASI level assessment."""
        score = self._metrics.overall_asi_score
        
        if score >= 0.9:
            return "ASI_ACHIEVED"
        elif score >= 0.7:
            return "ASI_EMERGING"
        elif score >= 0.5:
            return "AGI_PLUS"
        elif score >= 0.3:
            return "AGI"
        else:
            return "PRE_AGI"
    
    def register_callback(self, callback: Callable[[EmergenceEvent], None]) -> None:
        """Register callback for emergence events."""
        self._callbacks.append(callback)
    
    def get_recent_events(self, n: int = 10) -> list[EmergenceEvent]:
        """Get recent emergence events."""
        return self._events[-n:]
    
    def __len__(self) -> int:
        return len(self._events)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "events": [e.to_dict() for e in self._events[-100:]],
                "metrics": self._metrics.to_dict(),
            }, f, indent=2)
    
    def _load(self) -> None:
        pass  # Simplified


def create_asi_monitor(storage_path: str = "data/asi_emergence.json") -> ASIEmergenceMonitor:
    """Create ASI emergence monitor."""
    return ASIEmergenceMonitor(Path(storage_path))
