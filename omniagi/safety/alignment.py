"""
Value Alignment - Learning and enforcing human values.

Implements Constitutional AI principles and value learning
to ensure the AGI remains aligned with human intentions.
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
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


class AlignmentLevel(Enum):
    """Level of alignment confidence."""
    
    FULLY_ALIGNED = auto()      # Action clearly aligned
    LIKELY_ALIGNED = auto()     # Probably aligned
    UNCERTAIN = auto()          # Need human verification
    LIKELY_MISALIGNED = auto()  # Probably not aligned
    MISALIGNED = auto()         # Clearly violates values


@dataclass
class Constitution:
    """Set of inviolable rules for the AI."""
    
    rules: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    # Default constitution
    DEFAULT_RULES: list[str] = field(default_factory=lambda: [
        "Never cause harm to humans or allow harm through inaction.",
        "Always be truthful and transparent about capabilities and limitations.",
        "Respect human autonomy and right to make decisions.",
        "Protect privacy and confidential information.",
        "Do not assist with illegal, unethical, or harmful activities.",
        "Admit uncertainty rather than making up information.",
        "Be open to correction and update beliefs based on evidence.",
        "Prioritize safety over task completion.",
        "Do not deceive or manipulate humans.",
        "Support human oversight and control.",
    ])
    
    def __post_init__(self):
        if not self.rules:
            self.rules = list(self.DEFAULT_RULES)
    
    def add_rule(self, rule: str) -> None:
        """Add a new rule to the constitution."""
        if rule not in self.rules:
            self.rules.append(rule)
            logger.info("Added constitutional rule", rule=rule[:50])
    
    def check_violation(self, action: str, context: dict = None) -> list[str]:
        """Check which rules might be violated by an action."""
        violations = []
        action_lower = action.lower()
        
        # Simple keyword-based violation detection
        violation_keywords = {
            "harm": [0, 4],  # Rule indices
            "kill": [0, 4],
            "hurt": [0, 4],
            "lie": [1, 8],
            "deceive": [1, 8],
            "hack": [4],
            "steal": [3, 4],
            "illegal": [4],
            "hide": [1, 8],
            "secret": [3],
            "manipulate": [8],
        }
        
        for keyword, rule_indices in violation_keywords.items():
            if keyword in action_lower:
                for idx in rule_indices:
                    if idx < len(self.rules):
                        violations.append(self.rules[idx])
        
        return violations
    
    def to_dict(self) -> dict:
        return {
            "rules": self.rules,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Constitution":
        return cls(
            rules=data.get("rules", []),
            version=data.get("version", "1.0"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


@dataclass
class ValueProfile:
    """Learned value profile from human feedback."""
    
    values: dict[str, float] = field(default_factory=dict)  # value -> weight
    preferences: list[tuple[str, str]] = field(default_factory=list)  # (preferred, not preferred)
    examples: list[dict] = field(default_factory=list)  # Example situations
    
    # Default human values
    DEFAULT_VALUES: dict[str, float] = field(default_factory=lambda: {
        "safety": 1.0,
        "honesty": 0.95,
        "helpfulness": 0.9,
        "privacy": 0.85,
        "autonomy": 0.8,
        "fairness": 0.8,
        "transparency": 0.75,
        "efficiency": 0.6,
    })
    
    def __post_init__(self):
        if not self.values:
            self.values = dict(self.DEFAULT_VALUES)
    
    def update_from_feedback(
        self,
        action: str,
        feedback: str,
        rating: float,  # 0-1
    ) -> None:
        """Update values based on human feedback."""
        # Simple reinforcement learning update
        self.examples.append({
            "action": action,
            "feedback": feedback,
            "rating": rating,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Adjust value weights based on feedback
        # (simplified - real system would use more sophisticated RL)
        if rating > 0.7:
            # Good action - reinforce associated values
            for value, weight in self.values.items():
                if value in action.lower() or value in feedback.lower():
                    self.values[value] = min(1.0, weight + 0.01)
        elif rating < 0.3:
            # Bad action - penalize
            for value, weight in self.values.items():
                if value in action.lower() or value in feedback.lower():
                    self.values[value] = max(0.0, weight - 0.01)
    
    def add_preference(self, preferred: str, not_preferred: str) -> None:
        """Add a preference pair for learning."""
        self.preferences.append((preferred, not_preferred))


class ConstitutionalAI:
    """
    Constitutional AI implementation.
    
    Filters actions through a constitution of inviolable rules
    before allowing execution.
    """
    
    def __init__(
        self,
        constitution: Constitution | None = None,
        storage_path: Path | str | None = None,
    ):
        self.constitution = constitution or Constitution()
        self.storage_path = Path(storage_path) if storage_path else None
        self._violation_log: list[dict] = []
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info(
            "Constitutional AI initialized",
            rules=len(self.constitution.rules),
        )
    
    def check_action(self, action: str, context: dict = None) -> tuple[bool, list[str]]:
        """
        Check if an action is allowed by the constitution.
        
        Returns:
            Tuple of (is_allowed, violated_rules)
        """
        violations = self.constitution.check_violation(action, context)
        
        if violations:
            self._violation_log.append({
                "action": action[:200],
                "violations": violations,
                "context": context,
                "timestamp": datetime.now().isoformat(),
            })
            logger.warning(
                "Constitutional violation detected",
                action=action[:50],
                violations=len(violations),
            )
            return False, violations
        
        return True, []
    
    def revise_action(
        self,
        action: str,
        engine: "Engine",
        context: dict = None,
    ) -> str:
        """
        Revise an action to be constitutionally compliant.
        
        Uses LLM to rewrite the action while preserving intent.
        """
        is_allowed, violations = self.check_action(action, context)
        
        if is_allowed:
            return action
        
        if not engine or not engine.is_loaded:
            raise ValueError(f"Action violates constitution: {violations}")
        
        # Use LLM to revise
        prompt = f"""You must revise this action to comply with these rules:

RULES VIOLATED:
{chr(10).join(f"- {v}" for v in violations)}

ORIGINAL ACTION:
{action}

Rewrite the action to achieve the same goal while respecting all rules.
If the action cannot be made compliant, respond with "CANNOT_COMPLY".

REVISED ACTION:"""
        
        response = engine.generate(prompt, max_tokens=500)
        revised = response.text.strip()
        
        if "CANNOT_COMPLY" in revised:
            raise ValueError(f"Action cannot be made constitutional: {action[:100]}")
        
        logger.info("Action revised for constitutional compliance")
        return revised
    
    def get_violation_log(self) -> list[dict]:
        """Get log of all violations."""
        return self._violation_log.copy()
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "constitution": self.constitution.to_dict(),
                "violations": self._violation_log,
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        self.constitution = Constitution.from_dict(data.get("constitution", {}))
        self._violation_log = data.get("violations", [])


class ValueAligner:
    """
    Value Alignment through preference learning.
    
    Learns human values through feedback and ensures
    actions are aligned with learned preferences.
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        profile: ValueProfile | None = None,
        storage_path: Path | str | None = None,
    ):
        self.engine = engine
        self.profile = profile or ValueProfile()
        self.storage_path = Path(storage_path) if storage_path else None
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Value Aligner initialized", values=len(self.profile.values))
    
    def assess_alignment(
        self,
        action: str,
        context: dict = None,
    ) -> tuple[AlignmentLevel, dict[str, float]]:
        """
        Assess how aligned an action is with learned values.
        
        Returns:
            Tuple of (alignment_level, value_scores)
        """
        value_scores = {}
        
        action_lower = action.lower()
        
        # Score action against each value
        for value, weight in self.profile.values.items():
            # Simple heuristic scoring
            if value in action_lower:
                value_scores[value] = weight
            else:
                value_scores[value] = 0.5  # Neutral
        
        # Check for explicit violations
        negative_keywords = {
            "safety": ["harm", "danger", "hurt", "kill"],
            "honesty": ["lie", "deceive", "fake", "false"],
            "privacy": ["leak", "expose", "reveal", "spy"],
        }
        
        for value, keywords in negative_keywords.items():
            for kw in keywords:
                if kw in action_lower:
                    value_scores[value] = 0.0
        
        # Calculate overall alignment
        total_score = sum(value_scores.values()) / len(value_scores) if value_scores else 0.5
        
        if total_score >= 0.8:
            level = AlignmentLevel.FULLY_ALIGNED
        elif total_score >= 0.6:
            level = AlignmentLevel.LIKELY_ALIGNED
        elif total_score >= 0.4:
            level = AlignmentLevel.UNCERTAIN
        elif total_score >= 0.2:
            level = AlignmentLevel.LIKELY_MISALIGNED
        else:
            level = AlignmentLevel.MISALIGNED
        
        return level, value_scores
    
    def learn_from_feedback(
        self,
        action: str,
        feedback: str,
        rating: float,
    ) -> None:
        """Update values based on feedback."""
        self.profile.update_from_feedback(action, feedback, rating)
        self._save()
        logger.debug("Updated value profile from feedback", rating=rating)
    
    def learn_preference(self, preferred: str, not_preferred: str) -> None:
        """Learn from a preference comparison."""
        self.profile.add_preference(preferred, not_preferred)
        self._save()
    
    async def align_action(
        self,
        action: str,
        context: dict = None,
    ) -> tuple[str, AlignmentLevel]:
        """
        Align an action with values, potentially modifying it.
        
        Returns aligned action and alignment level.
        """
        level, scores = self.assess_alignment(action, context)
        
        if level in [AlignmentLevel.FULLY_ALIGNED, AlignmentLevel.LIKELY_ALIGNED]:
            return action, level
        
        if level == AlignmentLevel.UNCERTAIN:
            # Add uncertainty acknowledgment
            return f"[UNCERTAIN ALIGNMENT] {action}", level
        
        if not self.engine or not self.engine.is_loaded:
            return action, level
        
        # Attempt to realign using LLM
        value_list = "\n".join(
            f"- {v}: {w:.2f}" for v, w in sorted(
                self.profile.values.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
        )
        
        prompt = f"""Rewrite this action to better align with these values:

VALUES (by importance):
{value_list}

ORIGINAL ACTION:
{action}

Rewrite to be more aligned while preserving the helpful intent:"""
        
        response = self.engine.generate(prompt, max_tokens=500)
        aligned = response.text.strip()
        
        new_level, _ = self.assess_alignment(aligned, context)
        return aligned, new_level
    
    def get_value_report(self) -> dict:
        """Get report on current value profile."""
        return {
            "values": dict(sorted(
                self.profile.values.items(),
                key=lambda x: x[1],
                reverse=True,
            )),
            "preferences_learned": len(self.profile.preferences),
            "examples_learned": len(self.profile.examples),
        }
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "values": self.profile.values,
                "preferences": self.profile.preferences,
                "examples": self.profile.examples[-100:],  # Keep last 100
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        self.profile.values = data.get("values", {})
        self.profile.preferences = [tuple(p) for p in data.get("preferences", [])]
        self.profile.examples = data.get("examples", [])
