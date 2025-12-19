"""
Advanced Autonomy Module.

Implements goal decomposition, intrinsic motivation, and
long-term planning for true AGI autonomy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = auto()
    ACTIVE = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABANDONED = auto()


class GoalPriority(Enum):
    """Priority levels for goals."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Goal:
    """A goal with decomposition capabilities."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    description: str = ""
    
    # Hierarchy
    parent_id: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    
    # Status
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM
    progress: float = 0.0  # 0.0 to 1.0
    
    # Constraints
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    
    # Timing
    deadline: Optional[datetime] = None
    estimated_time_hours: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf goal (no subgoals)."""
        return len(self.subgoals) == 0
    
    def is_complete(self) -> bool:
        return self.status == GoalStatus.COMPLETED


class GoalDecompositionEngine:
    """
    Decomposes high-level goals into actionable subgoals.
    
    Uses hierarchical task networks (HTN) principles.
    """
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.decomposition_rules: Dict[str, Callable] = {}
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default decomposition rules."""
        self.decomposition_rules = {
            "learn": self._decompose_learn,
            "create": self._decompose_create,
            "solve": self._decompose_solve,
            "analyze": self._decompose_analyze,
            "improve": self._decompose_improve,
        }
    
    def add_goal(self, goal: Goal) -> str:
        """Add a goal to the system."""
        self.goals[goal.id] = goal
        return goal.id
    
    def decompose(self, goal_id: str, depth: int = 3) -> List[Goal]:
        """
        Decompose a goal into subgoals.
        
        Args:
            goal_id: ID of goal to decompose
            depth: Maximum decomposition depth
            
        Returns:
            List of generated subgoals
        """
        if depth <= 0:
            return []
        
        goal = self.goals.get(goal_id)
        if not goal:
            return []
        
        # Find applicable decomposition rule
        subgoals = []
        for keyword, rule in self.decomposition_rules.items():
            if keyword in goal.name.lower():
                subgoals = rule(goal)
                break
        
        if not subgoals:
            subgoals = self._generic_decompose(goal)
        
        # Register subgoals
        for sg in subgoals:
            sg.parent_id = goal_id
            self.goals[sg.id] = sg
            goal.subgoals.append(sg.id)
            
            # Recursive decomposition
            if not sg.is_leaf():
                self.decompose(sg.id, depth - 1)
        
        return subgoals
    
    def _decompose_learn(self, goal: Goal) -> List[Goal]:
        """Decompose a learning goal."""
        return [
            Goal(name=f"Research {goal.description}", priority=GoalPriority.HIGH),
            Goal(name=f"Study fundamentals", priority=GoalPriority.HIGH),
            Goal(name=f"Practice examples", priority=GoalPriority.MEDIUM),
            Goal(name=f"Apply knowledge", priority=GoalPriority.MEDIUM),
            Goal(name=f"Evaluate understanding", priority=GoalPriority.LOW),
        ]
    
    def _decompose_create(self, goal: Goal) -> List[Goal]:
        """Decompose a creation goal."""
        return [
            Goal(name="Define requirements", priority=GoalPriority.CRITICAL),
            Goal(name="Design solution", priority=GoalPriority.HIGH),
            Goal(name="Implement core", priority=GoalPriority.HIGH),
            Goal(name="Test and iterate", priority=GoalPriority.MEDIUM),
            Goal(name="Document", priority=GoalPriority.LOW),
        ]
    
    def _decompose_solve(self, goal: Goal) -> List[Goal]:
        """Decompose a problem-solving goal."""
        return [
            Goal(name="Understand problem", priority=GoalPriority.CRITICAL),
            Goal(name="Gather information", priority=GoalPriority.HIGH),
            Goal(name="Generate hypotheses", priority=GoalPriority.HIGH),
            Goal(name="Test solutions", priority=GoalPriority.MEDIUM),
            Goal(name="Verify solution", priority=GoalPriority.MEDIUM),
        ]
    
    def _decompose_analyze(self, goal: Goal) -> List[Goal]:
        """Decompose an analysis goal."""
        return [
            Goal(name="Collect data", priority=GoalPriority.HIGH),
            Goal(name="Process data", priority=GoalPriority.MEDIUM),
            Goal(name="Identify patterns", priority=GoalPriority.HIGH),
            Goal(name="Draw conclusions", priority=GoalPriority.MEDIUM),
        ]
    
    def _decompose_improve(self, goal: Goal) -> List[Goal]:
        """Decompose an improvement goal."""
        return [
            Goal(name="Assess current state", priority=GoalPriority.HIGH),
            Goal(name="Identify weaknesses", priority=GoalPriority.HIGH),
            Goal(name="Generate improvements", priority=GoalPriority.MEDIUM),
            Goal(name="Implement changes", priority=GoalPriority.MEDIUM),
            Goal(name="Measure impact", priority=GoalPriority.LOW),
        ]
    
    def _generic_decompose(self, goal: Goal) -> List[Goal]:
        """Generic decomposition for any goal."""
        return [
            Goal(name=f"Plan: {goal.name}", priority=GoalPriority.HIGH),
            Goal(name=f"Execute: {goal.name}", priority=GoalPriority.MEDIUM),
            Goal(name=f"Verify: {goal.name}", priority=GoalPriority.LOW),
        ]
    
    def get_executable_goals(self) -> List[Goal]:
        """Get all leaf goals that can be executed."""
        return [
            g for g in self.goals.values()
            if g.is_leaf() and g.status == GoalStatus.PENDING
        ]
    
    def update_progress(self, goal_id: str, progress: float) -> None:
        """Update goal progress and propagate to parents."""
        goal = self.goals.get(goal_id)
        if not goal:
            return
        
        goal.progress = min(1.0, max(0.0, progress))
        
        if goal.progress >= 1.0:
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = datetime.now()
        
        # Propagate to parent
        if goal.parent_id:
            parent = self.goals.get(goal.parent_id)
            if parent and parent.subgoals:
                child_progress = sum(
                    self.goals[sg].progress 
                    for sg in parent.subgoals
                    if sg in self.goals
                ) / len(parent.subgoals)
                self.update_progress(parent.id, child_progress)


@dataclass
class MotivationalDrive:
    """An intrinsic motivational drive."""
    name: str
    intensity: float = 0.5  # 0.0 to 1.0
    satisfaction: float = 0.5
    decay_rate: float = 0.01


class IntrinsicMotivationSystem:
    """
    Implements intrinsic motivation for autonomous behavior.
    
    Based on cognitive science models of motivation.
    """
    
    def __init__(self):
        self.drives: Dict[str, MotivationalDrive] = {
            "curiosity": MotivationalDrive("curiosity", 0.8, 0.5, 0.02),
            "competence": MotivationalDrive("competence", 0.7, 0.5, 0.01),
            "autonomy": MotivationalDrive("autonomy", 0.6, 0.5, 0.01),
            "social": MotivationalDrive("social", 0.4, 0.5, 0.005),
            "self_improvement": MotivationalDrive("self_improvement", 0.9, 0.5, 0.02),
        }
        self.reward_history: List[float] = []
    
    def get_motivation_score(self) -> float:
        """Calculate overall motivation score."""
        total = sum(
            d.intensity * (1 - d.satisfaction)
            for d in self.drives.values()
        )
        return min(1.0, total / len(self.drives))
    
    def satisfy_drive(self, drive_name: str, amount: float) -> None:
        """Satisfy a motivational drive."""
        if drive_name in self.drives:
            drive = self.drives[drive_name]
            drive.satisfaction = min(1.0, drive.satisfaction + amount)
    
    def update(self) -> None:
        """Update drives over time (decay satisfaction)."""
        for drive in self.drives.values():
            drive.satisfaction = max(0.0, drive.satisfaction - drive.decay_rate)
    
    def get_priority_drive(self) -> str:
        """Get the drive with highest priority (lowest satisfaction)."""
        return min(
            self.drives.keys(),
            key=lambda d: self.drives[d].satisfaction
        )
    
    def compute_intrinsic_reward(self, novelty: float, competence: float) -> float:
        """
        Compute intrinsic reward based on novelty and competence.
        
        Based on the learning progress hypothesis.
        """
        # Optimal challenge: not too easy, not too hard
        challenge_reward = 4 * competence * (1 - competence)
        
        # Novelty reward
        novelty_reward = novelty * self.drives["curiosity"].intensity
        
        total = 0.5 * challenge_reward + 0.5 * novelty_reward
        self.reward_history.append(total)
        
        return total


class LongTermPlanner:
    """
    Long-term planning with temporal reasoning.
    
    Uses hierarchical planning and plan repair.
    """
    
    def __init__(self, goal_engine: GoalDecompositionEngine):
        self.goal_engine = goal_engine
        self.current_plan: List[str] = []  # List of goal IDs
        self.plan_history: List[List[str]] = []
    
    def create_plan(self, top_goal: Goal) -> List[Goal]:
        """
        Create a long-term plan for a top-level goal.
        
        Returns ordered list of executable goals.
        """
        # Add and decompose top goal
        goal_id = self.goal_engine.add_goal(top_goal)
        self.goal_engine.decompose(goal_id, depth=4)
        
        # Get executable goals in priority order
        executable = self.goal_engine.get_executable_goals()
        
        # Sort by priority and dependencies
        plan = sorted(executable, key=lambda g: g.priority.value)
        
        self.current_plan = [g.id for g in plan]
        self.plan_history.append(self.current_plan.copy())
        
        return plan
    
    def get_next_goal(self) -> Optional[Goal]:
        """Get the next goal to execute."""
        for goal_id in self.current_plan:
            goal = self.goal_engine.goals.get(goal_id)
            if goal and goal.status == GoalStatus.PENDING:
                goal.status = GoalStatus.ACTIVE
                return goal
        return None
    
    def repair_plan(self, failed_goal_id: str) -> List[Goal]:
        """
        Repair the plan when a goal fails.
        
        Generates alternative approaches.
        """
        failed_goal = self.goal_engine.goals.get(failed_goal_id)
        if not failed_goal:
            return []
        
        failed_goal.status = GoalStatus.FAILED
        
        # Create alternative goal
        alt_goal = Goal(
            name=f"Alternative: {failed_goal.name}",
            description=f"Alternative approach to: {failed_goal.description}",
            parent_id=failed_goal.parent_id,
            priority=failed_goal.priority,
        )
        
        self.goal_engine.add_goal(alt_goal)
        
        # Replace failed goal in plan
        if failed_goal_id in self.current_plan:
            idx = self.current_plan.index(failed_goal_id)
            self.current_plan[idx] = alt_goal.id
        
        return [alt_goal]
    
    def estimate_completion_time(self) -> float:
        """Estimate total time to complete the plan in hours."""
        return sum(
            self.goal_engine.goals[gid].estimated_time_hours
            for gid in self.current_plan
            if gid in self.goal_engine.goals
        )


class AdvancedAutonomySystem:
    """
    Complete advanced autonomy system.
    
    Integrates goal decomposition, motivation, and planning.
    """
    
    def __init__(self):
        self.goal_engine = GoalDecompositionEngine()
        self.motivation = IntrinsicMotivationSystem()
        self.planner = LongTermPlanner(self.goal_engine)
        
        self.active_goal: Optional[Goal] = None
        self.goals_completed = 0
        self.goals_failed = 0
    
    def set_objective(self, objective: str, description: str = "") -> List[Goal]:
        """
        Set a high-level objective and generate a plan.
        
        Returns list of executable goals.
        """
        top_goal = Goal(
            name=objective,
            description=description,
            priority=GoalPriority.HIGH,
        )
        
        plan = self.planner.create_plan(top_goal)
        logger.info(f"Created plan with {len(plan)} goals for: {objective}")
        
        return plan
    
    def step(self) -> Optional[Goal]:
        """
        Execute one step of autonomous behavior.
        
        Returns the goal to work on.
        """
        # Update motivation
        self.motivation.update()
        
        # Check motivation level
        if self.motivation.get_motivation_score() < 0.2:
            # Low motivation - satisfy priority drive
            priority_drive = self.motivation.get_priority_drive()
            logger.info(f"Low motivation, prioritizing drive: {priority_drive}")
        
        # Get next goal
        self.active_goal = self.planner.get_next_goal()
        
        return self.active_goal
    
    def report_success(self, novelty: float = 0.5, difficulty: float = 0.5):
        """Report successful goal completion."""
        if self.active_goal:
            self.goal_engine.update_progress(self.active_goal.id, 1.0)
            self.goals_completed += 1
            
            # Intrinsic reward
            reward = self.motivation.compute_intrinsic_reward(novelty, difficulty)
            self.motivation.satisfy_drive("competence", 0.1)
            self.motivation.satisfy_drive("curiosity", novelty * 0.2)
            
            self.active_goal = None
    
    def report_failure(self):
        """Report goal failure and trigger plan repair."""
        if self.active_goal:
            self.planner.repair_plan(self.active_goal.id)
            self.goals_failed += 1
            self.active_goal = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get autonomy system status."""
        return {
            "total_goals": len(self.goal_engine.goals),
            "completed": self.goals_completed,
            "failed": self.goals_failed,
            "motivation": self.motivation.get_motivation_score(),
            "priority_drive": self.motivation.get_priority_drive(),
            "plan_length": len(self.planner.current_plan),
            "estimated_hours": self.planner.estimate_completion_time(),
        }
