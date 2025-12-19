"""
Open-Ended Learning Module.

Implements curiosity-driven exploration, novelty detection,
and self-directed learning for AGI.
"""

from __future__ import annotations

import logging
import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A learning experience."""
    id: str
    state: Dict[str, Any]
    action: str
    outcome: Dict[str, Any]
    reward: float
    novelty: float = 0.0
    learning_progress: float = 0.0


class NoveltyDetector:
    """
    Detects novelty in experiences.
    
    Uses count-based and prediction-error methods.
    """
    
    def __init__(self, memory_size: int = 10000):
        self.state_counts: Dict[str, int] = defaultdict(int)
        self.action_counts: Dict[str, int] = defaultdict(int)
        self.transition_counts: Dict[str, int] = defaultdict(int)
        self.total_count = 0
        self.memory_size = memory_size
        
        # For prediction-based novelty
        self.predicted_outcomes: Dict[str, Dict] = {}
    
    def compute_novelty(self, experience: Experience) -> float:
        """
        Compute novelty score for an experience.
        
        Returns score in [0, 1], higher = more novel.
        """
        # State novelty (count-based)
        state_key = self._hash_state(experience.state)
        state_count = self.state_counts[state_key]
        state_novelty = 1.0 / (1 + math.log(1 + state_count))
        
        # Action novelty
        action_count = self.action_counts[experience.action]
        action_novelty = 1.0 / (1 + math.log(1 + action_count))
        
        # Transition novelty
        trans_key = f"{state_key}:{experience.action}"
        trans_count = self.transition_counts[trans_key]
        trans_novelty = 1.0 / (1 + math.log(1 + trans_count))
        
        # Prediction error novelty
        pred_novelty = self._prediction_error(experience)
        
        # Combined novelty
        novelty = 0.3 * state_novelty + 0.2 * action_novelty + \
                  0.3 * trans_novelty + 0.2 * pred_novelty
        
        # Update counts
        self._update_counts(experience)
        
        return min(1.0, novelty)
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Hash a state for counting."""
        return str(sorted(state.items()))[:100]
    
    def _prediction_error(self, experience: Experience) -> float:
        """Compute prediction error novelty."""
        state_key = self._hash_state(experience.state)
        trans_key = f"{state_key}:{experience.action}"
        
        if trans_key in self.predicted_outcomes:
            predicted = self.predicted_outcomes[trans_key]
            actual = experience.outcome
            
            # Simple difference measure
            diff = 0
            for key in set(predicted.keys()) | set(actual.keys()):
                p_val = predicted.get(key, 0)
                a_val = actual.get(key, 0)
                if isinstance(p_val, (int, float)) and isinstance(a_val, (int, float)):
                    diff += abs(p_val - a_val)
            
            error = min(1.0, diff / 10)  # Normalize
        else:
            error = 1.0  # Never seen before
        
        # Update prediction
        self.predicted_outcomes[trans_key] = experience.outcome
        
        return error
    
    def _update_counts(self, experience: Experience) -> None:
        """Update count statistics."""
        state_key = self._hash_state(experience.state)
        trans_key = f"{state_key}:{experience.action}"
        
        self.state_counts[state_key] += 1
        self.action_counts[experience.action] += 1
        self.transition_counts[trans_key] += 1
        self.total_count += 1


class CuriosityModule:
    """
    Intrinsic curiosity for exploration.
    
    Based on learning progress hypothesis.
    """
    
    def __init__(self):
        self.novelty_detector = NoveltyDetector()
        self.learning_progress: Dict[str, List[float]] = defaultdict(list)
        self.curiosity_weight = 0.5
        
        # Exploration strategies
        self.exploration_rate = 0.3
        self.min_exploration = 0.05
        self.decay_rate = 0.001
    
    def compute_curiosity_reward(self, experience: Experience) -> float:
        """
        Compute intrinsic curiosity reward.
        
        Combines novelty and learning progress.
        """
        # Novelty component
        novelty = self.novelty_detector.compute_novelty(experience)
        experience.novelty = novelty
        
        # Learning progress component
        domain = self._get_domain(experience.state)
        self.learning_progress[domain].append(novelty)
        
        if len(self.learning_progress[domain]) >= 2:
            recent = self.learning_progress[domain][-10:]
            older = self.learning_progress[domain][-20:-10]
            
            if older:
                progress = abs(sum(recent)/len(recent) - sum(older)/len(older))
            else:
                progress = sum(recent)/len(recent)
        else:
            progress = novelty
        
        experience.learning_progress = progress
        
        # Curiosity reward formula
        # Higher when: high progress AND moderate novelty (sweet spot)
        curiosity = progress * (novelty * (1 - novelty) * 4)  # Peak at novelty=0.5
        
        return curiosity * self.curiosity_weight
    
    def _get_domain(self, state: Dict[str, Any]) -> str:
        """Extract domain from state."""
        return state.get("domain", "general")
    
    def should_explore(self) -> bool:
        """Decide whether to explore or exploit."""
        should = random.random() < self.exploration_rate
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * (1 - self.decay_rate)
        )
        
        return should
    
    def get_exploration_action(self, available_actions: List[str]) -> str:
        """Select an action for exploration."""
        if not available_actions:
            return ""
        
        # Prefer less-tried actions
        novelty_scores = [
            1.0 / (1 + self.novelty_detector.action_counts[a])
            for a in available_actions
        ]
        
        # Softmax selection
        total = sum(novelty_scores)
        if total > 0:
            probs = [s / total for s in novelty_scores]
        else:
            probs = [1/len(available_actions)] * len(available_actions)
        
        r = random.random()
        cumsum = 0
        for action, prob in zip(available_actions, probs):
            cumsum += prob
            if r < cumsum:
                return action
        
        return available_actions[-1]


class CompositionalLearner:
    """
    Learn compositional concepts from experience.
    
    Enables generalization to novel combinations.
    """
    
    def __init__(self):
        # Primitive concepts
        self.primitives: Dict[str, Dict[str, Any]] = {}
        
        # Composition rules
        self.compositions: Dict[str, List[str]] = {}
        
        # Observed combinations
        self.observed_combinations: Set[str] = set()
    
    def learn_primitive(
        self,
        name: str,
        examples: List[Dict[str, Any]],
    ) -> None:
        """Learn a primitive concept from examples."""
        # Extract common features
        if not examples:
            return
        
        common = {}
        for key in examples[0].keys():
            values = [e.get(key) for e in examples]
            if len(set(str(v) for v in values)) == 1:
                common[key] = values[0]
        
        self.primitives[name] = {
            "features": common,
            "examples": len(examples),
        }
    
    def compose(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Compose multiple concepts into a new one.
        
        Returns combined features.
        """
        composition_key = "+".join(sorted(concepts))
        
        if composition_key in self.compositions:
            return self._get_composition_features(composition_key)
        
        # Create new composition
        combined = {}
        for concept in concepts:
            if concept in self.primitives:
                combined.update(self.primitives[concept]["features"])
        
        self.compositions[composition_key] = concepts
        self.observed_combinations.add(composition_key)
        
        return combined
    
    def _get_composition_features(self, key: str) -> Dict[str, Any]:
        """Get features of a composed concept."""
        concepts = self.compositions.get(key, [])
        combined = {}
        for concept in concepts:
            if concept in self.primitives:
                combined.update(self.primitives[concept]["features"])
        return combined
    
    def generalize(self, partial: Dict[str, Any]) -> List[str]:
        """
        Generalize from partial observation.
        
        Returns list of possible concept completions.
        """
        matches = []
        
        for name, data in self.primitives.items():
            features = data["features"]
            if all(partial.get(k) == v for k, v in features.items() if k in partial):
                matches.append(name)
        
        return matches
    
    def analogical_reasoning(
        self,
        source_domain: str,
        target_domain: str,
        source_pattern: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply analogical reasoning to transfer knowledge.
        
        Maps patterns from source to target domain.
        """
        # Simple structural mapping
        result = {}
        
        for key, value in source_pattern.items():
            # Replace domain prefix
            new_key = key.replace(source_domain, target_domain)
            result[new_key] = value
        
        return result


class OpenEndedLearner:
    """
    Complete open-ended learning system.
    
    Integrates curiosity, novelty detection, and
    compositional learning.
    """
    
    def __init__(self):
        self.curiosity = CuriosityModule()
        self.compositional = CompositionalLearner()
        
        self.experience_buffer: List[Experience] = []
        self.max_buffer = 10000
        
        self.skills_learned: Dict[str, Dict] = {}
        self.concepts_learned: Set[str] = set()
        
        self.total_experiences = 0
        self.intrinsic_reward_total = 0.0
    
    def process_experience(self, experience: Experience) -> float:
        """
        Process a new experience.
        
        Returns intrinsic reward.
        """
        # Compute curiosity reward
        intrinsic = self.curiosity.compute_curiosity_reward(experience)
        self.intrinsic_reward_total += intrinsic
        
        # Add to buffer
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer:
            self.experience_buffer.pop(0)
        
        self.total_experiences += 1
        
        # Try to learn new concepts
        self._extract_concepts(experience)
        
        return intrinsic
    
    def _extract_concepts(self, experience: Experience) -> None:
        """Extract concepts from experience."""
        state = experience.state
        
        # Learn primitives from high-novelty experiences
        if experience.novelty > 0.7:
            for key, value in state.items():
                concept = f"{key}:{value}"
                if concept not in self.concepts_learned:
                    self.concepts_learned.add(concept)
                    self.compositional.learn_primitive(
                        concept,
                        [state]
                    )
    
    def choose_action(self, available_actions: List[str]) -> str:
        """Choose action based on exploration/exploitation."""
        if self.curiosity.should_explore():
            return self.curiosity.get_exploration_action(available_actions)
        
        # Exploitation - return first action (would use Q-values in full impl)
        return available_actions[0] if available_actions else ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_experiences": self.total_experiences,
            "concepts_learned": len(self.concepts_learned),
            "skills_learned": len(self.skills_learned),
            "intrinsic_reward_total": self.intrinsic_reward_total,
            "exploration_rate": self.curiosity.exploration_rate,
            "buffer_size": len(self.experience_buffer),
        }
