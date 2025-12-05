"""
Hierarchical Planner - Long-term goal planning.

Decomposes high-level goals into subgoals and concrete actions,
with dynamic replanning based on world state changes.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, TYPE_CHECKING
from heapq import heappush, heappop

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

from omniagi.world.state import WorldState
from omniagi.world.simulator import MentalSimulator, Prediction

logger = structlog.get_logger()


class GoalStatus(Enum):
    """Status of a goal."""
    
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    BLOCKED = auto()


class GoalPriority(Enum):
    """Priority levels for goals."""
    
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Goal:
    """A goal to achieve."""
    
    id: str
    description: str
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    
    # Hierarchy
    parent_id: str | None = None
    subgoal_ids: list[str] = field(default_factory=list)
    
    # Success criteria
    success_conditions: list[str] = field(default_factory=list)
    
    # Progress
    progress: float = 0.0  # 0.0 - 1.0
    attempts: int = 0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    deadline: datetime | None = None
    completed_at: datetime | None = None
    
    # Context
    context: dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: "Goal") -> bool:
        """For priority queue ordering."""
        return self.priority.value < other.priority.value


@dataclass
class PlanStep:
    """A step in a plan."""
    
    step_id: str
    action: str
    actor_id: str
    target_id: str | None = None
    
    preconditions: list[str] = field(default_factory=list)
    expected_effects: list[str] = field(default_factory=list)
    
    executed: bool = False
    success: bool | None = None


@dataclass
class Plan:
    """A plan to achieve a goal."""
    
    goal_id: str
    steps: list[PlanStep] = field(default_factory=list)
    
    current_step: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    total_estimated_probability: float = 1.0
    
    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)
    
    @property
    def next_step(self) -> PlanStep | None:
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None


PLANNING_PROMPT = '''Você é um planejador estratégico. Crie um plano para alcançar o objetivo.

## Objetivo:
{goal_description}

## Estado Atual do Mundo:
{world_context}

## Condições de Sucesso:
{success_conditions}

## Instruções:
1. Decomponha o objetivo em passos concretos
2. Ordene os passos logicamente
3. Identifique precondições e efeitos esperados

Responda no formato:
PASSO 1:
Ação: [ação concreta]
Precondições: [o que precisa estar verdade antes]
Efeitos: [resultado esperado]

PASSO 2:
...

RESUMO:
Total de passos: [N]
Probabilidade de sucesso: [0.0-1.0]
'''


class HierarchicalPlanner:
    """
    Hierarchical planner for complex goal achievement.
    
    Features:
    - Goal decomposition into subgoals
    - Plan generation with simulation
    - Dynamic replanning
    - Priority-based execution
    """
    
    def __init__(
        self,
        world_state: WorldState | None = None,
        simulator: MentalSimulator | None = None,
        engine: "Engine | None" = None,
    ):
        """
        Initialize planner.
        
        Args:
            world_state: Current world state.
            simulator: Mental simulator for plan evaluation.
            engine: LLM engine for planning.
        """
        self.world = world_state or WorldState()
        self.simulator = simulator or MentalSimulator(self.world, engine)
        self.engine = engine
        
        self._goals: dict[str, Goal] = {}
        self._plans: dict[str, Plan] = {}
        self._goal_queue: list[Goal] = []  # Priority queue
        
        self._goal_counter = 0
    
    def create_goal(
        self,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_id: str | None = None,
        success_conditions: list[str] | None = None,
        deadline: datetime | None = None,
        **context,
    ) -> Goal:
        """
        Create a new goal.
        
        Args:
            description: What to achieve.
            priority: Goal priority.
            parent_id: Parent goal (for hierarchy).
            success_conditions: How to know it's achieved.
            deadline: Optional deadline.
            context: Additional context.
            
        Returns:
            The created Goal.
        """
        goal_id = f"goal_{self._goal_counter}"
        self._goal_counter += 1
        
        goal = Goal(
            id=goal_id,
            description=description,
            priority=priority,
            parent_id=parent_id,
            success_conditions=success_conditions or [],
            deadline=deadline,
            context=context,
        )
        
        self._goals[goal_id] = goal
        heappush(self._goal_queue, goal)
        
        # Link to parent
        if parent_id and parent_id in self._goals:
            self._goals[parent_id].subgoal_ids.append(goal_id)
        
        logger.info("Goal created", id=goal_id, description=description[:50])
        return goal
    
    def decompose_goal(self, goal_id: str) -> list[Goal]:
        """
        Decompose a goal into subgoals.
        
        Uses LLM to break down complex goals.
        """
        goal = self._goals.get(goal_id)
        if not goal:
            return []
        
        if not self.engine or not self.engine.is_loaded:
            # Simple decomposition: just mark as ready
            return []
        
        from omniagi.core.engine import GenerationConfig
        
        prompt = f'''Decomponha este objetivo em subobjetivos menores e concretos:

Objetivo: {goal.description}

Liste 2-5 subobjetivos necessários para alcançar este objetivo.
Cada subobjetivo deve ser específico e verificável.

Formato:
1. [subobjetivo]
2. [subobjetivo]
'''
        
        response = self.engine.generate(
            prompt,
            GenerationConfig(max_tokens=512, temperature=0.3),
        )
        
        # Parse subgoals
        subgoals = []
        for line in response.text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                desc = line.lstrip("0123456789.-) ").strip()
                if desc:
                    subgoal = self.create_goal(
                        description=desc,
                        priority=goal.priority,
                        parent_id=goal_id,
                    )
                    subgoals.append(subgoal)
        
        return subgoals
    
    def plan_for_goal(self, goal_id: str) -> Plan | None:
        """
        Generate a plan to achieve a goal.
        
        Args:
            goal_id: The goal to plan for.
            
        Returns:
            A Plan with steps, or None if planning fails.
        """
        goal = self._goals.get(goal_id)
        if not goal:
            return None
        
        # Check for existing plan
        if goal_id in self._plans:
            return self._plans[goal_id]
        
        steps = []
        
        if self.engine and self.engine.is_loaded:
            # Use LLM for planning
            steps = self._llm_plan(goal)
        else:
            # Simple default plan
            steps = [
                PlanStep(
                    step_id=f"{goal_id}_step_0",
                    action=f"Execute: {goal.description}",
                    actor_id="self",
                    expected_effects=[f"Goal achieved: {goal.description}"],
                ),
            ]
        
        # Simulate and evaluate plan
        total_prob = 1.0
        for step in steps:
            prediction = self.simulator.simulate_action(
                step.action,
                step.actor_id,
                step.target_id,
            )
            total_prob *= prediction.probability
        
        plan = Plan(
            goal_id=goal_id,
            steps=steps,
            total_estimated_probability=total_prob,
        )
        
        self._plans[goal_id] = plan
        
        logger.info(
            "Plan created",
            goal=goal_id,
            steps=len(steps),
            probability=f"{total_prob:.0%}",
        )
        
        return plan
    
    def _llm_plan(self, goal: Goal) -> list[PlanStep]:
        """Use LLM to generate plan steps."""
        from omniagi.core.engine import GenerationConfig
        
        prompt = PLANNING_PROMPT.format(
            goal_description=goal.description,
            world_context=self.world.to_context(),
            success_conditions="\n".join(
                f"- {c}" for c in goal.success_conditions
            ) or "Nenhuma condição específica.",
        )
        
        response = self.engine.generate(
            prompt,
            GenerationConfig(max_tokens=1024, temperature=0.3),
        )
        
        return self._parse_plan_response(response.text, goal.id)
    
    def _parse_plan_response(self, response: str, goal_id: str) -> list[PlanStep]:
        """Parse LLM plan response into steps."""
        steps = []
        current_step = None
        step_count = 0
        
        for line in response.split("\n"):
            line = line.strip()
            
            if line.startswith("PASSO") or line.startswith("STEP"):
                if current_step:
                    steps.append(current_step)
                
                current_step = PlanStep(
                    step_id=f"{goal_id}_step_{step_count}",
                    action="",
                    actor_id="self",
                )
                step_count += 1
                
            elif current_step:
                if line.startswith("Ação:") or line.startswith("Action:"):
                    current_step.action = line.split(":", 1)[1].strip()
                elif line.startswith("Precondições:") or line.startswith("Preconditions:"):
                    current_step.preconditions = [line.split(":", 1)[1].strip()]
                elif line.startswith("Efeitos:") or line.startswith("Effects:"):
                    current_step.expected_effects = [line.split(":", 1)[1].strip()]
        
        if current_step and current_step.action:
            steps.append(current_step)
        
        return steps
    
    def get_next_goal(self) -> Goal | None:
        """Get the next goal to work on."""
        while self._goal_queue:
            goal = heappop(self._goal_queue)
            
            if goal.status in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                goal.status = GoalStatus.ACTIVE
                heappush(self._goal_queue, goal)  # Put back
                return goal
        
        return None
    
    def execute_next_step(self, goal_id: str) -> dict[str, Any]:
        """
        Execute the next step in a goal's plan.
        
        Returns execution result.
        """
        plan = self._plans.get(goal_id)
        if not plan:
            plan = self.plan_for_goal(goal_id)
        
        if not plan or plan.is_complete:
            return {"status": "complete", "message": "Plan complete or not found"}
        
        step = plan.next_step
        if not step:
            return {"status": "error", "message": "No next step"}
        
        result = {
            "step_id": step.step_id,
            "action": step.action,
            "status": "pending",
        }
        
        # Simulate before execute
        prediction = self.simulator.simulate_action(
            step.action,
            step.actor_id,
            step.target_id,
        )
        
        result["prediction"] = {
            "probability": prediction.probability,
            "effects": prediction.predicted_effects,
            "risks": prediction.risks,
        }
        
        # Mark as executed (actual execution would happen externally)
        step.executed = True
        plan.current_step += 1
        
        result["status"] = "executed"
        
        # Update goal progress
        goal = self._goals.get(goal_id)
        if goal:
            goal.progress = plan.current_step / max(1, len(plan.steps))
            goal.attempts += 1
            
            if plan.is_complete:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = datetime.now()
        
        return result
    
    def replan(self, goal_id: str, reason: str = "") -> Plan | None:
        """
        Replan for a goal due to changed conditions.
        
        Args:
            goal_id: Goal to replan.
            reason: Why replanning is needed.
            
        Returns:
            New plan.
        """
        # Remove old plan
        if goal_id in self._plans:
            del self._plans[goal_id]
        
        logger.info("Replanning", goal=goal_id, reason=reason)
        
        return self.plan_for_goal(goal_id)
    
    def get_goal_tree(self, root_id: str | None = None) -> dict[str, Any]:
        """Get goal hierarchy as a tree structure."""
        def build_tree(goal_id: str) -> dict:
            goal = self._goals.get(goal_id)
            if not goal:
                return {}
            
            return {
                "id": goal.id,
                "description": goal.description,
                "status": goal.status.name,
                "progress": f"{goal.progress:.0%}",
                "subgoals": [
                    build_tree(sub_id) for sub_id in goal.subgoal_ids
                ],
            }
        
        if root_id:
            return build_tree(root_id)
        
        # Return all top-level goals
        top_level = [
            g for g in self._goals.values()
            if g.parent_id is None
        ]
        
        return {
            "goals": [build_tree(g.id) for g in top_level],
        }
    
    def get_statistics(self) -> dict[str, Any]:
        """Get planning statistics."""
        total = len(self._goals)
        by_status = {}
        
        for goal in self._goals.values():
            status = goal.status.name
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total_goals": total,
            "by_status": by_status,
            "total_plans": len(self._plans),
            "completed_rate": by_status.get("COMPLETED", 0) / max(1, total),
        }
