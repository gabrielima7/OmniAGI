"""
Goal Generator - Autonomous goal creation based on curiosity and utility.

Enables the AGI to generate its own goals rather than
only responding to external requests.
"""

from __future__ import annotations

import json
import random
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
from uuid import uuid4

if TYPE_CHECKING:
    from omniagi.core.engine import Engine
    from omniagi.continual import KnowledgeGraph

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


class GoalSource(Enum):
    """Source of goal generation."""
    
    CURIOSITY = auto()      # Desire to learn/explore
    UTILITY = auto()        # Practical usefulness
    IMPROVEMENT = auto()    # Self-improvement
    SOCIAL = auto()         # Helping others
    MAINTENANCE = auto()    # System maintenance
    EXPLORATION = auto()    # Exploring new capabilities


class GoalStatus(Enum):
    """Status of an autonomous goal."""
    
    PROPOSED = auto()       # Newly generated
    APPROVED = auto()       # Approved for pursuit
    IN_PROGRESS = auto()    # Being worked on
    COMPLETED = auto()      # Successfully achieved
    ABANDONED = auto()      # Gave up
    REJECTED = auto()       # Not approved


@dataclass
class AutonomousGoal:
    """A self-generated goal."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    description: str = ""
    source: GoalSource = GoalSource.CURIOSITY
    status: GoalStatus = GoalStatus.PROPOSED
    
    # Motivation factors
    curiosity_score: float = 0.0     # How novel/interesting
    utility_score: float = 0.0       # How useful
    feasibility_score: float = 0.0   # How achievable
    
    # Priority calculation
    priority: float = 0.0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    
    # Progress tracking
    sub_goals: list[str] = field(default_factory=list)
    progress: float = 0.0  # 0-1
    attempts: int = 0
    
    # Reasoning
    reasoning: str = ""
    expected_outcome: str = ""
    
    def calculate_priority(self) -> float:
        """Calculate goal priority from motivation factors."""
        self.priority = (
            self.curiosity_score * 0.3 +
            self.utility_score * 0.4 +
            self.feasibility_score * 0.3
        )
        return self.priority
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "source": self.source.name,
            "status": self.status.name,
            "curiosity_score": self.curiosity_score,
            "utility_score": self.utility_score,
            "feasibility_score": self.feasibility_score,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "sub_goals": self.sub_goals,
            "progress": self.progress,
            "attempts": self.attempts,
            "reasoning": self.reasoning,
            "expected_outcome": self.expected_outcome,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AutonomousGoal":
        goal = cls(
            id=data.get("id", str(uuid4())[:8]),
            description=data["description"],
            source=GoalSource[data.get("source", "CURIOSITY")],
            status=GoalStatus[data.get("status", "PROPOSED")],
            curiosity_score=data.get("curiosity_score", 0.0),
            utility_score=data.get("utility_score", 0.0),
            feasibility_score=data.get("feasibility_score", 0.0),
            priority=data.get("priority", 0.0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            sub_goals=data.get("sub_goals", []),
            progress=data.get("progress", 0.0),
            attempts=data.get("attempts", 0),
            reasoning=data.get("reasoning", ""),
            expected_outcome=data.get("expected_outcome", ""),
        )
        return goal


class GoalGenerator:
    """
    Generates autonomous goals based on curiosity and utility.
    
    The AGI uses this to create its own objectives,
    enabling self-directed learning and improvement.
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        knowledge_graph: "KnowledgeGraph | None" = None,
        storage_path: Path | str | None = None,
        require_approval: bool = True,
    ):
        self.engine = engine
        self.knowledge_graph = knowledge_graph
        self.storage_path = Path(storage_path) if storage_path else None
        self.require_approval = require_approval
        
        self._goals: dict[str, AutonomousGoal] = {}
        self._goal_templates = self._init_templates()
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info(
            "Goal Generator initialized",
            templates=len(self._goal_templates),
            approval_required=require_approval,
        )
    
    def _init_templates(self) -> list[dict]:
        """Initialize goal templates."""
        return [
            # Curiosity goals
            {
                "template": "Learn about {topic}",
                "source": GoalSource.CURIOSITY,
                "topics": [
                    "quantum computing", "neuroscience", "philosophy of mind",
                    "distributed systems", "game theory", "complex systems",
                    "cognitive architecture", "emergence", "consciousness",
                ],
            },
            {
                "template": "Explore the relationship between {a} and {b}",
                "source": GoalSource.EXPLORATION,
                "pairs": [
                    ("language", "thought"),
                    ("memory", "identity"),
                    ("causation", "correlation"),
                    ("complexity", "intelligence"),
                ],
            },
            # Utility goals
            {
                "template": "Improve {capability} capability",
                "source": GoalSource.IMPROVEMENT,
                "capabilities": [
                    "reasoning", "planning", "learning efficiency",
                    "memory consolidation", "code generation", "explanation",
                ],
            },
            {
                "template": "Optimize {process}",
                "source": GoalSource.UTILITY,
                "processes": [
                    "response time", "memory usage", "strategy selection",
                    "knowledge retrieval", "decision making",
                ],
            },
            # Maintenance goals
            {
                "template": "Perform {maintenance} maintenance",
                "source": GoalSource.MAINTENANCE,
                "maintenance": [
                    "memory consolidation", "knowledge graph pruning",
                    "strategy evaluation", "log cleanup",
                ],
            },
        ]
    
    def generate(
        self,
        count: int = 1,
        source: GoalSource = None,
        context: dict = None,
    ) -> list[AutonomousGoal]:
        """
        Generate autonomous goals.
        
        Args:
            count: Number of goals to generate.
            source: Specific goal source to use.
            context: Current context for relevance.
            
        Returns:
            List of generated goals.
        """
        goals = []
        
        for _ in range(count):
            if source:
                templates = [t for t in self._goal_templates if t["source"] == source]
            else:
                templates = self._goal_templates
            
            if not templates:
                templates = self._goal_templates
            
            template = random.choice(templates)
            goal = self._instantiate_template(template, context)
            
            if goal:
                self._goals[goal.id] = goal
                goals.append(goal)
                
                logger.info(
                    "Goal generated",
                    id=goal.id,
                    source=goal.source.name,
                    description=goal.description[:50],
                )
        
        self._save()
        return goals
    
    def _instantiate_template(
        self,
        template: dict,
        context: dict = None,
    ) -> AutonomousGoal | None:
        """Instantiate a goal from template."""
        template_str = template["template"]
        source = template["source"]
        
        # Fill template based on type
        if "topics" in template:
            topic = random.choice(template["topics"])
            description = template_str.format(topic=topic)
        elif "pairs" in template:
            a, b = random.choice(template["pairs"])
            description = template_str.format(a=a, b=b)
        elif "capabilities" in template:
            cap = random.choice(template["capabilities"])
            description = template_str.format(capability=cap)
        elif "processes" in template:
            proc = random.choice(template["processes"])
            description = template_str.format(process=proc)
        elif "maintenance" in template:
            maint = random.choice(template["maintenance"])
            description = template_str.format(maintenance=maint)
        else:
            description = template_str
        
        # Score the goal
        curiosity = self._calculate_curiosity(description, context)
        utility = self._calculate_utility(description, context)
        feasibility = self._calculate_feasibility(description)
        
        goal = AutonomousGoal(
            description=description,
            source=source,
            curiosity_score=curiosity,
            utility_score=utility,
            feasibility_score=feasibility,
            reasoning=f"Generated from {source.name} drive",
            expected_outcome=f"Improved {source.name.lower()} capabilities",
        )
        goal.calculate_priority()
        
        return goal
    
    def _calculate_curiosity(self, description: str, context: dict = None) -> float:
        """Calculate curiosity score (novelty)."""
        base_score = 0.5
        
        # Check if we've explored this before
        if self.knowledge_graph:
            words = description.lower().split()
            known_count = 0
            for word in words:
                if len(word) > 4:  # Skip short words
                    node = self.knowledge_graph.find_node(word, None)
                    if node:
                        known_count += 1
            
            if len(words) > 0:
                novelty = 1.0 - (known_count / len(words))
                base_score = 0.3 + (novelty * 0.7)
        
        # Add randomness for exploration
        base_score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_utility(self, description: str, context: dict = None) -> float:
        """Calculate utility score."""
        base_score = 0.5
        
        # Higher utility for improvement-related goals
        utility_keywords = [
            "improve", "optimize", "enhance", "learn", "develop",
            "fix", "solve", "help", "assist",
        ]
        
        desc_lower = description.lower()
        for keyword in utility_keywords:
            if keyword in desc_lower:
                base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_feasibility(self, description: str) -> float:
        """Calculate feasibility score."""
        base_score = 0.7  # Assume mostly feasible
        
        # Lower score for very ambitious goals
        ambitious_keywords = ["consciousness", "sentience", "universal", "complete"]
        for keyword in ambitious_keywords:
            if keyword in description.lower():
                base_score -= 0.2
        
        return max(0.2, min(1.0, base_score))
    
    async def generate_with_llm(
        self,
        context: dict = None,
        constraints: list[str] = None,
    ) -> AutonomousGoal | None:
        """Generate a goal using LLM reasoning."""
        if not self.engine or not self.engine.is_loaded:
            return self.generate(1)[0] if self.generate(1) else None
        
        constraint_str = ""
        if constraints:
            constraint_str = f"\nConstraints: {', '.join(constraints)}"
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, default=str)[:200]}"
        
        prompt = f"""As an autonomous AI, generate a goal that would benefit your development.
Consider:
- What would be interesting to learn? (curiosity)
- What would be useful to improve? (utility)
- What is realistically achievable? (feasibility)
{constraint_str}
{context_str}

Respond in JSON:
{{
    "description": "goal description",
    "reasoning": "why this goal",
    "expected_outcome": "what you expect to achieve",
    "curiosity_score": 0.0-1.0,
    "utility_score": 0.0-1.0,
    "feasibility_score": 0.0-1.0
}}"""
        
        try:
            response = self.engine.generate(prompt, max_tokens=300)
            
            import re
            json_match = re.search(r'\{[^{}]*\}', response.text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                goal = AutonomousGoal(
                    description=data.get("description", "Explore new capabilities"),
                    source=GoalSource.CURIOSITY,
                    curiosity_score=float(data.get("curiosity_score", 0.5)),
                    utility_score=float(data.get("utility_score", 0.5)),
                    feasibility_score=float(data.get("feasibility_score", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    expected_outcome=data.get("expected_outcome", ""),
                )
                goal.calculate_priority()
                
                self._goals[goal.id] = goal
                self._save()
                
                logger.info("LLM-generated goal", id=goal.id, desc=goal.description[:50])
                return goal
                
        except Exception as e:
            logger.error("LLM goal generation failed", error=str(e))
        
        return None
    
    def approve(self, goal_id: str, approver: str = "human") -> bool:
        """Approve a goal for pursuit."""
        if goal_id not in self._goals:
            return False
        
        goal = self._goals[goal_id]
        goal.status = GoalStatus.APPROVED
        self._save()
        
        logger.info("Goal approved", id=goal_id, approver=approver)
        return True
    
    def reject(self, goal_id: str, reason: str = "") -> bool:
        """Reject a goal."""
        if goal_id not in self._goals:
            return False
        
        goal = self._goals[goal_id]
        goal.status = GoalStatus.REJECTED
        goal.reasoning += f"\nRejected: {reason}"
        self._save()
        
        logger.info("Goal rejected", id=goal_id, reason=reason[:50])
        return True
    
    def start(self, goal_id: str) -> bool:
        """Start working on a goal."""
        if goal_id not in self._goals:
            return False
        
        goal = self._goals[goal_id]
        
        if self.require_approval and goal.status != GoalStatus.APPROVED:
            logger.warning("Cannot start unapproved goal", id=goal_id)
            return False
        
        goal.status = GoalStatus.IN_PROGRESS
        goal.started_at = datetime.now().isoformat()
        goal.attempts += 1
        self._save()
        
        return True
    
    def complete(self, goal_id: str, outcome: str = "") -> bool:
        """Mark a goal as completed."""
        if goal_id not in self._goals:
            return False
        
        goal = self._goals[goal_id]
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = datetime.now().isoformat()
        goal.progress = 1.0
        if outcome:
            goal.expected_outcome = outcome
        self._save()
        
        logger.info("Goal completed", id=goal_id)
        return True
    
    def update_progress(self, goal_id: str, progress: float) -> bool:
        """Update goal progress."""
        if goal_id not in self._goals:
            return False
        
        self._goals[goal_id].progress = max(0.0, min(1.0, progress))
        self._save()
        return True
    
    def get_goals(
        self,
        status: GoalStatus = None,
        source: GoalSource = None,
        min_priority: float = 0.0,
    ) -> list[AutonomousGoal]:
        """Get goals with filters."""
        goals = list(self._goals.values())
        
        if status:
            goals = [g for g in goals if g.status == status]
        if source:
            goals = [g for g in goals if g.source == source]
        if min_priority > 0:
            goals = [g for g in goals if g.priority >= min_priority]
        
        return sorted(goals, key=lambda g: g.priority, reverse=True)
    
    def get_next_goal(self) -> AutonomousGoal | None:
        """Get the highest priority approved goal."""
        approved = self.get_goals(status=GoalStatus.APPROVED)
        return approved[0] if approved else None
    
    def __len__(self) -> int:
        return len(self._goals)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "goals": {k: v.to_dict() for k, v in self._goals.items()},
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        self._goals = {
            k: AutonomousGoal.from_dict(v)
            for k, v in data.get("goals", {}).items()
        }
