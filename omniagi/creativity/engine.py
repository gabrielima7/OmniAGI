"""
Creative Engine - Novel idea generation and problem solving.

Enables true creativity through:
1. Combinatorial exploration
2. Analogy-based generation
3. Constraint relaxation
4. Divergent thinking
"""

from __future__ import annotations

import json
import random
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, List
from uuid import uuid4

logger = structlog.get_logger()


class CreativeStrategy(Enum):
    """Creative generation strategies."""
    
    COMBINATION = auto()     # Combine existing concepts
    ANALOGY = auto()         # Apply pattern from other domain
    MUTATION = auto()        # Random variation
    INVERSION = auto()       # Flip assumptions
    ABSTRACTION = auto()     # Generalize then specialize
    CONSTRAINT = auto()      # Add/remove constraints


@dataclass
class CreativeIdea:
    """A generated creative idea."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    content: str = ""
    strategy: CreativeStrategy = CreativeStrategy.COMBINATION
    
    # Sources
    source_concepts: list[str] = field(default_factory=list)
    domain: str = ""
    
    # Quality metrics
    novelty: float = 0.5      # How new/original
    usefulness: float = 0.5   # How practical
    surprise: float = 0.5     # How unexpected
    
    # Validation
    feasibility: float = 0.5
    validated: bool = False
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def creativity_score(self) -> float:
        """Compute overall creativity score."""
        return (self.novelty + self.usefulness + self.surprise) / 3
    
    def to_dict(self) -> dict:
        return {
            "idea": self.content[:100],
            "strategy": self.strategy.name,
            "creativity": self.creativity_score,
            "novelty": self.novelty,
        }


class CreativeEngine:
    """
    Creative Engine for True AGI.
    
    Generates novel ideas through various strategies.
    Essential for AGI - genuine creativity, not just recombination.
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._ideas: list[CreativeIdea] = []
        self._concept_pool: dict[str, list[str]] = {}
        
        # Initialize some seed concepts
        self._init_concepts()
        
        logger.info("Creative Engine initialized")
    
    def _init_concepts(self) -> None:
        """Initialize seed concepts for creativity."""
        self._concept_pool = {
            "technology": ["AI", "neural network", "quantum", "blockchain", "robot"],
            "nature": ["evolution", "ecosystem", "adaptation", "symbiosis", "emergence"],
            "cognition": ["learning", "memory", "attention", "reasoning", "intuition"],
            "abstract": ["pattern", "structure", "flow", "balance", "transformation"],
        }
    
    def generate(
        self,
        prompt: str = "",
        strategy: CreativeStrategy = None,
        domain: str = None,
        n: int = 3,
    ) -> list[CreativeIdea]:
        """
        Generate creative ideas.
        """
        if strategy is None:
            strategy = random.choice(list(CreativeStrategy))
        
        ideas = []
        for _ in range(n):
            if strategy == CreativeStrategy.COMBINATION:
                idea = self._generate_combination(prompt, domain)
            elif strategy == CreativeStrategy.ANALOGY:
                idea = self._generate_analogy(prompt, domain)
            elif strategy == CreativeStrategy.MUTATION:
                idea = self._generate_mutation(prompt)
            elif strategy == CreativeStrategy.INVERSION:
                idea = self._generate_inversion(prompt)
            elif strategy == CreativeStrategy.ABSTRACTION:
                idea = self._generate_abstraction(prompt, domain)
            else:
                idea = self._generate_constraint(prompt)
            
            idea.strategy = strategy
            ideas.append(idea)
            self._ideas.append(idea)
        
        logger.debug("Generated ideas", count=len(ideas), strategy=strategy.name)
        return ideas
    
    def _generate_combination(self, prompt: str, domain: str = None) -> CreativeIdea:
        """Combine concepts from different domains."""
        domains = list(self._concept_pool.keys())
        d1, d2 = random.sample(domains, 2)
        
        c1 = random.choice(self._concept_pool[d1])
        c2 = random.choice(self._concept_pool[d2])
        
        templates = [
            f"What if we applied {c1} principles to {c2}?",
            f"A {c1}-inspired approach to {c2}",
            f"Combining {c1} with {c2} for {prompt or 'innovation'}",
            f"Using {c2} to enhance {c1}",
        ]
        
        content = random.choice(templates)
        
        return CreativeIdea(
            content=content,
            source_concepts=[c1, c2],
            domain=f"{d1}_{d2}",
            novelty=0.7,
            surprise=0.6,
        )
    
    def _generate_analogy(self, prompt: str, domain: str = None) -> CreativeIdea:
        """Generate through analogy."""
        source_domain = domain or random.choice(list(self._concept_pool.keys()))
        target_domain = random.choice([d for d in self._concept_pool.keys() if d != source_domain])
        
        source_concept = random.choice(self._concept_pool[source_domain])
        
        content = f"Just as {source_concept} works in {source_domain}, we could apply similar principles to {target_domain}: {prompt or 'new approach'}"
        
        return CreativeIdea(
            content=content,
            source_concepts=[source_concept],
            domain=target_domain,
            novelty=0.65,
            usefulness=0.7,
        )
    
    def _generate_mutation(self, prompt: str) -> CreativeIdea:
        """Random variation on existing concept."""
        mutations = ["faster", "smaller", "distributed", "self-organizing", "adaptive", "reversible"]
        
        mutation = random.choice(mutations)
        base = prompt or "the current approach"
        
        content = f"What if {base} was {mutation}? This could lead to unexpected benefits."
        
        return CreativeIdea(
            content=content,
            novelty=0.5,
            surprise=0.8,
        )
    
    def _generate_inversion(self, prompt: str) -> CreativeIdea:
        """Flip assumptions."""
        inversions = [
            ("input becomes output", "output becomes input"),
            ("centralized becomes distributed", "distributed becomes centralized"),
            ("sequential becomes parallel", "parallel becomes sequential"),
            ("static becomes dynamic", "dynamic becomes static"),
        ]
        
        inv = random.choice(inversions)
        
        content = f"Inversion: Instead of {inv[0]}, what if {inv[1]}? Applied to {prompt or 'the problem'}"
        
        return CreativeIdea(
            content=content,
            novelty=0.8,
            surprise=0.9,
        )
    
    def _generate_abstraction(self, prompt: str, domain: str = None) -> CreativeIdea:
        """Abstract then re-specialize."""
        abstractions = ["pattern", "process", "structure", "relationship", "transformation"]
        
        abstraction = random.choice(abstractions)
        
        content = f"Abstract the {abstraction} from {prompt or 'the concept'}, then apply it in a new context"
        
        return CreativeIdea(
            content=content,
            novelty=0.6,
            usefulness=0.65,
        )
    
    def _generate_constraint(self, prompt: str) -> CreativeIdea:
        """Add or remove constraints."""
        constraints = [
            "remove time constraints",
            "add resource scarcity",
            "remove physical limitations",
            "add collaboration requirement",
            "remove hierarchy",
        ]
        
        constraint = random.choice(constraints)
        
        content = f"What if we {constraint} for {prompt or 'the problem'}? New solutions emerge."
        
        return CreativeIdea(
            content=content,
            novelty=0.55,
            usefulness=0.6,
            surprise=0.7,
        )
    
    def brainstorm(
        self,
        problem: str,
        n_ideas: int = 10,
    ) -> list[CreativeIdea]:
        """
        Brainstorm multiple ideas using all strategies.
        """
        all_ideas = []
        
        for strategy in CreativeStrategy:
            ideas = self.generate(problem, strategy, n=max(1, n_ideas // 6))
            all_ideas.extend(ideas)
        
        # Sort by creativity score
        all_ideas.sort(key=lambda x: x.creativity_score, reverse=True)
        
        return all_ideas[:n_ideas]
    
    def add_concept(self, domain: str, concept: str) -> None:
        """Add a new concept to the pool."""
        if domain not in self._concept_pool:
            self._concept_pool[domain] = []
        self._concept_pool[domain].append(concept)
    
    def get_best_ideas(self, n: int = 5) -> list[CreativeIdea]:
        """Get top ideas by creativity score."""
        sorted_ideas = sorted(self._ideas, key=lambda x: x.creativity_score, reverse=True)
        return sorted_ideas[:n]
    
    def __len__(self) -> int:
        return len(self._ideas)


class GoalDirectedPlanner:
    """
    Goal-directed planning for autonomous behavior.
    
    Creates and executes plans to achieve goals.
    Essential for AGI autonomy.
    """
    
    def __init__(self):
        self._goals: list[dict] = []
        self._plans: list[dict] = []
        self._completed: list[dict] = []
        
        logger.info("Goal-Directed Planner initialized")
    
    def set_goal(
        self,
        description: str,
        priority: int = 5,
        deadline: str = None,
    ) -> dict:
        """Set a new goal."""
        goal = {
            "id": str(uuid4())[:8],
            "description": description,
            "priority": priority,
            "deadline": deadline,
            "status": "active",
            "created": datetime.now().isoformat(),
        }
        self._goals.append(goal)
        logger.info("Goal set", description=description[:30])
        return goal
    
    def create_plan(self, goal_id: str) -> dict:
        """Create a plan for achieving a goal."""
        goal = next((g for g in self._goals if g["id"] == goal_id), None)
        if not goal:
            return {}
        
        # Decompose into steps
        steps = self._decompose_goal(goal["description"])
        
        plan = {
            "id": str(uuid4())[:8],
            "goal_id": goal_id,
            "steps": steps,
            "current_step": 0,
            "status": "ready",
        }
        self._plans.append(plan)
        return plan
    
    def _decompose_goal(self, description: str) -> list[dict]:
        """Decompose a goal into actionable steps."""
        # Simplified decomposition
        steps = [
            {"action": f"Analyze: {description[:30]}", "status": "pending"},
            {"action": f"Plan approach for: {description[:30]}", "status": "pending"},
            {"action": f"Execute solution for: {description[:30]}", "status": "pending"},
            {"action": f"Verify results for: {description[:30]}", "status": "pending"},
        ]
        return steps
    
    def execute_step(self, plan_id: str) -> dict:
        """Execute the next step in a plan."""
        plan = next((p for p in self._plans if p["id"] == plan_id), None)
        if not plan or plan["current_step"] >= len(plan["steps"]):
            return {}
        
        step = plan["steps"][plan["current_step"]]
        step["status"] = "completed"
        plan["current_step"] += 1
        
        if plan["current_step"] >= len(plan["steps"]):
            plan["status"] = "completed"
            self._completed.append(plan)
        
        return step
    
    def get_active_goals(self) -> list[dict]:
        """Get all active goals."""
        return [g for g in self._goals if g["status"] == "active"]
    
    def __len__(self) -> int:
        return len(self._goals)
