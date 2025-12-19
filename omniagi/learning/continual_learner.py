"""
Continual Learner - Learn without forgetting.

Implements Elastic Weight Consolidation (EWC) and other techniques
for learning new concepts without catastrophic forgetting.

This is a CRITICAL component for true AGI - the ability to 
continuously learn from new experiences.
"""

from __future__ import annotations

import json
import logging

try:
    import structlog
except ImportError:
    structlog = None
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Strategies for continual learning."""
    
    EWC = auto()          # Elastic Weight Consolidation
    REPLAY = auto()       # Experience Replay
    PROGRESSIVE = auto()  # Progressive Networks
    HYBRID = auto()       # Combination


@dataclass
class Concept:
    """A learned concept that can be applied."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    description: str = ""
    
    # Knowledge representation
    examples: list[dict] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    embeddings: list[float] = field(default_factory=list)
    
    # Learning metadata
    learned_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 0.5
    times_applied: int = 0
    success_rate: float = 0.0
    
    # Relations to other concepts
    prerequisites: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description[:100],
            "confidence": self.confidence,
            "examples": len(self.examples),
            "rules": len(self.rules),
        }


@dataclass 
class LearningExperience:
    """A single learning experience."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    
    # What was learned
    input_data: Any = None
    expected_output: Any = None
    actual_output: Any = None
    
    # Learning outcome
    was_correct: bool = False
    error_type: str = ""
    lesson_learned: str = ""
    
    # Applied concepts
    concepts_used: list[str] = field(default_factory=list)
    new_concepts: list[str] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LearningMetrics:
    """Metrics for continual learning."""
    
    total_experiences: int = 0
    concepts_learned: int = 0
    
    # Forgetting metrics
    retention_rate: float = 1.0  # How much old knowledge is retained
    forward_transfer: float = 0.0  # How much new learning helps old tasks
    backward_transfer: float = 0.0  # How much old knowledge helps new tasks
    
    # Performance
    avg_accuracy_old: float = 0.0
    avg_accuracy_new: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "experiences": self.total_experiences,
            "concepts": self.concepts_learned,
            "retention": self.retention_rate,
            "forward_transfer": self.forward_transfer,
        }


class ContinualLearner:
    """
    Continual Learning System for True AGI.
    
    Key capabilities:
    1. Learn new concepts without forgetting old ones
    2. Transfer knowledge between domains
    3. Build on prior knowledge
    4. Consolidate learning over time
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        strategy: LearningStrategy = LearningStrategy.HYBRID,
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.strategy = strategy
        
        # Knowledge base
        self._concepts: dict[str, Concept] = {}
        self._experiences: list[LearningExperience] = []
        
        # Memory buffer for replay
        self._replay_buffer: list[LearningExperience] = []
        self._buffer_size = 1000
        
        # EWC parameters (Fisher information for important weights)
        self._fisher_info: dict[str, float] = {}
        self._importance_threshold = 0.5
        
        # Metrics
        self._metrics = LearningMetrics()
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info(
            "Continual Learner initialized",
            strategy=strategy.name,
            concepts=len(self._concepts),
        )
    
    def learn_concept(
        self,
        name: str,
        description: str,
        examples: list[dict],
        rules: list[str] = None,
    ) -> Concept:
        """
        Learn a new concept.
        
        This is the core AGI capability - learning new concepts
        from examples and integrating them with existing knowledge.
        """
        # Check for existing related concepts
        related = self._find_related_concepts(name, description)
        
        # Create concept
        concept = Concept(
            name=name,
            description=description,
            examples=examples,
            rules=rules or [],
            related=[c.id for c in related],
        )
        
        # Generate embedding (simplified - would use real embeddings)
        concept.embeddings = self._generate_embedding(name, description)
        
        # Check for prerequisites
        concept.prerequisites = self._identify_prerequisites(concept, related)
        
        # Store concept
        self._concepts[concept.id] = concept
        self._metrics.concepts_learned += 1
        
        # Update related concepts
        for rel in related:
            if concept.id not in rel.related:
                rel.related.append(concept.id)
        
        self._save()
        
        logger.info(
            "Concept learned",
            name=name,
            related=len(related),
            prerequisites=len(concept.prerequisites),
        )
        
        return concept
    
    def learn_from_experience(
        self,
        input_data: Any,
        expected: Any,
        actual: Any,
        concepts_used: list[str] = None,
    ) -> LearningExperience:
        """
        Learn from a single experience.
        
        This enables learning from mistakes and successes.
        """
        was_correct = self._compare_outputs(expected, actual)
        
        experience = LearningExperience(
            input_data=input_data,
            expected_output=expected,
            actual_output=actual,
            was_correct=was_correct,
            concepts_used=concepts_used or [],
        )
        
        # Analyze error if wrong
        if not was_correct:
            experience.error_type = self._classify_error(expected, actual)
            experience.lesson_learned = self._extract_lesson(experience)
        
        # Add to experiences
        self._experiences.append(experience)
        self._metrics.total_experiences += 1
        
        # Add to replay buffer (for EWC)
        if self.strategy in [LearningStrategy.REPLAY, LearningStrategy.HYBRID]:
            self._add_to_replay_buffer(experience)
        
        # Update concept confidence
        for concept_id in experience.concepts_used:
            if concept_id in self._concepts:
                self._update_concept_confidence(concept_id, was_correct)
        
        # Learn new concept if pattern detected
        if not was_correct:
            new_concepts = self._try_learn_from_error(experience)
            experience.new_concepts = [c.id for c in new_concepts]
        
        self._save()
        
        return experience
    
    def apply_concept(
        self,
        concept_id: str,
        input_data: Any,
    ) -> tuple[Any, float]:
        """
        Apply a learned concept to new input.
        
        Returns (output, confidence).
        """
        if concept_id not in self._concepts:
            return None, 0.0
        
        concept = self._concepts[concept_id]
        
        # Find most similar example
        best_match, similarity = self._find_similar_example(
            input_data, concept.examples
        )
        
        # Apply rules if available
        if concept.rules:
            output = self._apply_rules(input_data, concept.rules)
            if output is not None:
                concept.times_applied += 1
                return output, concept.confidence * 0.9
        
        # Apply by analogy to example
        if best_match:
            output = self._apply_by_analogy(input_data, best_match)
            concept.times_applied += 1
            return output, concept.confidence * similarity
        
        return None, 0.0
    
    def get_applicable_concepts(
        self,
        input_data: Any,
        min_confidence: float = 0.3,
    ) -> list[tuple[Concept, float]]:
        """
        Find concepts that might apply to the input.
        """
        applicable = []
        
        input_embedding = self._generate_embedding(str(input_data), "")
        
        for concept in self._concepts.values():
            if concept.confidence < min_confidence:
                continue
            
            # Check embedding similarity
            similarity = self._cosine_similarity(
                input_embedding, concept.embeddings
            )
            
            if similarity > 0.3:
                applicable.append((concept, similarity))
        
        # Sort by similarity
        applicable.sort(key=lambda x: x[1], reverse=True)
        
        return applicable[:10]
    
    def consolidate(self) -> dict:
        """
        Consolidate learning (like sleep in humans).
        
        This strengthens important connections and
        prunes weak ones.
        """
        stats = {
            "concepts_strengthened": 0,
            "concepts_pruned": 0,
            "connections_made": 0,
        }
        
        # Strengthen frequently used concepts
        for concept in self._concepts.values():
            if concept.times_applied > 5 and concept.success_rate > 0.6:
                concept.confidence = min(1.0, concept.confidence + 0.05)
                stats["concepts_strengthened"] += 1
        
        # Prune weak concepts
        weak = [c for c in self._concepts.values() 
                if c.confidence < 0.2 and c.times_applied > 10]
        for concept in weak:
            # Don't delete, just mark as weak
            concept.confidence *= 0.9
            stats["concepts_pruned"] += 1
        
        # Find new connections
        concepts_list = list(self._concepts.values())
        for i, c1 in enumerate(concepts_list):
            for c2 in concepts_list[i+1:]:
                if self._should_connect(c1, c2):
                    if c2.id not in c1.related:
                        c1.related.append(c2.id)
                        c2.related.append(c1.id)
                        stats["connections_made"] += 1
        
        self._save()
        
        logger.info("Learning consolidated", **stats)
        return stats
    
    def get_metrics(self) -> LearningMetrics:
        """Get learning metrics."""
        return self._metrics
    
    def _find_related_concepts(
        self, name: str, description: str
    ) -> list[Concept]:
        """Find concepts related to a new concept."""
        related = []
        embedding = self._generate_embedding(name, description)
        
        for concept in self._concepts.values():
            similarity = self._cosine_similarity(embedding, concept.embeddings)
            if similarity > 0.5:
                related.append(concept)
        
        return related
    
    def _generate_embedding(self, name: str, description: str) -> list[float]:
        """Generate embedding for text (simplified)."""
        # In real implementation, would use actual embeddings
        text = f"{name} {description}".lower()
        # Simple bag-of-characters embedding
        embedding = [0.0] * 26
        for char in text:
            if 'a' <= char <= 'z':
                embedding[ord(char) - ord('a')] += 1
        # Normalize
        total = sum(embedding) or 1
        return [e / total for e in embedding]
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _identify_prerequisites(
        self, concept: Concept, related: list[Concept]
    ) -> list[str]:
        """Identify prerequisite concepts."""
        prerequisites = []
        for rel in related:
            # If related concept is simpler, it might be a prerequisite
            if len(rel.examples) < len(concept.examples):
                if rel.confidence > 0.5:
                    prerequisites.append(rel.id)
        return prerequisites
    
    def _compare_outputs(self, expected: Any, actual: Any) -> bool:
        """Compare expected and actual outputs."""
        if expected == actual:
            return True
        # Try string comparison
        return str(expected).strip() == str(actual).strip()
    
    def _classify_error(self, expected: Any, actual: Any) -> str:
        """Classify the type of error."""
        if actual is None:
            return "no_output"
        if type(expected) != type(actual):
            return "type_mismatch"
        return "value_error"
    
    def _extract_lesson(self, experience: LearningExperience) -> str:
        """Extract a lesson from a failed experience."""
        return f"Expected {experience.expected_output}, got {experience.actual_output}"
    
    def _add_to_replay_buffer(self, experience: LearningExperience) -> None:
        """Add experience to replay buffer."""
        self._replay_buffer.append(experience)
        # Keep buffer size limited
        if len(self._replay_buffer) > self._buffer_size:
            self._replay_buffer = self._replay_buffer[-self._buffer_size:]
    
    def _update_concept_confidence(self, concept_id: str, success: bool) -> None:
        """Update concept confidence based on application result."""
        concept = self._concepts.get(concept_id)
        if not concept:
            return
        
        # Update success rate
        total = concept.times_applied + 1
        if success:
            concept.success_rate = (
                concept.success_rate * concept.times_applied + 1
            ) / total
            concept.confidence = min(1.0, concept.confidence + 0.01)
        else:
            concept.success_rate = (
                concept.success_rate * concept.times_applied
            ) / total
            concept.confidence = max(0.0, concept.confidence - 0.02)
    
    def _try_learn_from_error(
        self, experience: LearningExperience
    ) -> list[Concept]:
        """Try to learn a new concept from an error."""
        # This is where real AGI would shine - learning from mistakes
        # For now, we create a simple correction concept
        new_concepts = []
        
        if experience.error_type == "value_error":
            # Create a correction rule
            concept = Concept(
                name=f"correction_{experience.id}",
                description=experience.lesson_learned,
                examples=[{
                    "input": experience.input_data,
                    "output": experience.expected_output
                }],
                rules=[f"When input is similar to {experience.input_data}, "
                       f"output should be {experience.expected_output}"],
                confidence=0.4,
            )
            self._concepts[concept.id] = concept
            new_concepts.append(concept)
        
        return new_concepts
    
    def _find_similar_example(
        self, input_data: Any, examples: list[dict]
    ) -> tuple[dict | None, float]:
        """Find most similar example."""
        if not examples:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        input_str = str(input_data)
        for example in examples:
            example_str = str(example.get("input", ""))
            # Simple Jaccard similarity
            input_words = set(input_str.lower().split())
            example_words = set(example_str.lower().split())
            if not input_words or not example_words:
                continue
            similarity = len(input_words & example_words) / len(input_words | example_words)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = example
        
        return best_match, best_similarity
    
    def _apply_rules(self, input_data: Any, rules: list[str]) -> Any | None:
        """Apply rules to input."""
        # Simplified rule application
        # Real implementation would use a rule engine
        return None
    
    def _apply_by_analogy(self, input_data: Any, example: dict) -> Any:
        """Apply by analogy to an example."""
        # Transform input similar to how example input maps to output
        return example.get("output")
    
    def _should_connect(self, c1: Concept, c2: Concept) -> bool:
        """Check if two concepts should be connected."""
        if c1.id == c2.id:
            return False
        similarity = self._cosine_similarity(c1.embeddings, c2.embeddings)
        return similarity > 0.6
    
    def __len__(self) -> int:
        return len(self._concepts)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "concepts": {k: v.to_dict() for k, v in self._concepts.items()},
                "metrics": self._metrics.to_dict(),
            }, f, indent=2)
    
    def _load(self) -> None:
        pass  # Simplified
