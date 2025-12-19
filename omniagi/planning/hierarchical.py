"""
Hierarchical Planning Module.

Implements hierarchical task networks, temporal reasoning,
and plan repair for long-term AGI planning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from uuid import uuid4
import heapq

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a planning task."""
    PENDING = auto()
    READY = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    BLOCKED = auto()


@dataclass
class PlanningTask:
    """A task in the hierarchical plan."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    
    # Hierarchy
    parent_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    
    # Method - how to accomplish
    method: Optional[str] = None
    
    # Preconditions and effects
    preconditions: Set[str] = field(default_factory=set)
    effects: Set[str] = field(default_factory=set)
    
    # Temporal constraints
    earliest_start: Optional[datetime] = None
    latest_finish: Optional[datetime] = None
    duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10, lower = higher priority
    
    def is_primitive(self) -> bool:
        """Check if this is a primitive (executable) task."""
        return len(self.subtasks) == 0


@dataclass
class PlanningMethod:
    """A method for decomposing a task."""
    name: str
    task_type: str
    subtasks: List[str]
    preconditions: Set[str] = field(default_factory=set)
    
    def applicable(self, state: Set[str]) -> bool:
        """Check if method is applicable in current state."""
        return self.preconditions.issubset(state)


class HierarchicalTaskNetwork:
    """
    Hierarchical Task Network (HTN) planner.
    
    Decomposes abstract tasks into primitive actions
    using methods.
    """
    
    def __init__(self):
        self.tasks: Dict[str, PlanningTask] = {}
        self.methods: Dict[str, List[PlanningMethod]] = {}
        self.state: Set[str] = set()
        
        self._init_default_methods()
    
    def _init_default_methods(self):
        """Initialize default planning methods."""
        self.methods = {
            "solve_problem": [
                PlanningMethod(
                    "analytical_solve",
                    "solve_problem",
                    ["understand", "analyze", "synthesize", "verify"],
                    {"problem_defined"},
                ),
                PlanningMethod(
                    "creative_solve", 
                    "solve_problem",
                    ["brainstorm", "prototype", "iterate", "finalize"],
                    {"problem_defined"},
                ),
            ],
            "learn_topic": [
                PlanningMethod(
                    "systematic_learn",
                    "learn_topic",
                    ["research", "study", "practice", "apply"],
                    set(),
                ),
            ],
            "build_system": [
                PlanningMethod(
                    "iterative_build",
                    "build_system",
                    ["design", "implement_core", "implement_features", "test", "deploy"],
                    {"requirements_known"},
                ),
            ],
        }
    
    def add_task(self, task: PlanningTask) -> None:
        """Add a task to the network."""
        self.tasks[task.id] = task
    
    def decompose(self, task_id: str) -> List[PlanningTask]:
        """
        Decompose a task using applicable methods.
        
        Returns list of generated subtasks.
        """
        task = self.tasks.get(task_id)
        if not task:
            return []
        
        # Find applicable methods
        task_type = task.name.lower().replace(" ", "_")
        methods = self.methods.get(task_type, [])
        
        applicable = [m for m in methods if m.applicable(self.state)]
        
        if not applicable:
            # Try generic decomposition
            return self._generic_decompose(task)
        
        # Use first applicable method
        method = applicable[0]
        task.method = method.name
        
        subtasks = []
        for i, subtask_name in enumerate(method.subtasks):
            subtask = PlanningTask(
                name=subtask_name,
                parent_id=task_id,
                priority=task.priority,
            )
            self.tasks[subtask.id] = subtask
            task.subtasks.append(subtask.id)
            subtasks.append(subtask)
        
        return subtasks
    
    def _generic_decompose(self, task: PlanningTask) -> List[PlanningTask]:
        """Generic decomposition fallback."""
        subtasks = []
        for name in ["plan", "execute", "verify"]:
            st = PlanningTask(name=f"{name}_{task.name}", parent_id=task.id)
            self.tasks[st.id] = st
            task.subtasks.append(st.id)
            subtasks.append(st)
        return subtasks
    
    def plan(self, goal_task: PlanningTask, max_depth: int = 5) -> List[PlanningTask]:
        """
        Create a complete plan for a goal.
        
        Returns ordered list of primitive tasks.
        """
        self.add_task(goal_task)
        
        # Decompose recursively
        to_decompose = [goal_task.id]
        
        for _ in range(max_depth):
            next_level = []
            for task_id in to_decompose:
                task = self.tasks[task_id]
                if not task.is_primitive():
                    subtasks = self.decompose(task_id)
                    next_level.extend([st.id for st in subtasks])
                elif not task.subtasks:
                    # Already primitive
                    pass
            
            if not next_level:
                break
            to_decompose = next_level
        
        # Get all primitive tasks in order
        primitives = [
            t for t in self.tasks.values()
            if t.is_primitive()
        ]
        
        return sorted(primitives, key=lambda t: t.priority)


class TemporalReasoner:
    """
    Temporal reasoning for planning.
    
    Handles temporal constraints and scheduling.
    """
    
    def __init__(self):
        self.constraints: List[Tuple[str, str, str, timedelta]] = []  # (task1, relation, task2, gap)
        self.schedule: Dict[str, Tuple[datetime, datetime]] = {}
    
    def add_constraint(
        self,
        task1_id: str,
        relation: str,  # "before", "after", "during", "meets"
        task2_id: str,
        min_gap: timedelta = timedelta(),
    ) -> None:
        """Add a temporal constraint."""
        self.constraints.append((task1_id, relation, task2_id, min_gap))
    
    def schedule_tasks(
        self,
        tasks: List[PlanningTask],
        start_time: datetime,
    ) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Schedule tasks respecting constraints.
        
        Returns mapping of task_id to (start, end) times.
        """
        schedule = {}
        current_time = start_time
        
        # Sort by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)
        
        for task in sorted_tasks:
            # Check constraints
            task_start = current_time
            
            for t1, rel, t2, gap in self.constraints:
                if t2 == task.id and t1 in schedule:
                    other_end = schedule[t1][1]
                    if rel == "after":
                        task_start = max(task_start, other_end + gap)
            
            task_end = task_start + task.duration
            schedule[task.id] = (task_start, task_end)
            current_time = task_end
        
        self.schedule = schedule
        return schedule
    
    def check_deadline(
        self,
        task_id: str,
        deadline: datetime,
    ) -> Tuple[bool, str]:
        """Check if a task can meet its deadline."""
        if task_id not in self.schedule:
            return False, "Task not scheduled"
        
        _, end_time = self.schedule[task_id]
        
        if end_time <= deadline:
            return True, f"Task completes at {end_time}"
        else:
            slack = end_time - deadline
            return False, f"Task misses deadline by {slack}"


class PlanRepairModule:
    """
    Repairs plans when execution fails.
    
    Implements replanning and workarounds.
    """
    
    def __init__(self, htn: HierarchicalTaskNetwork):
        self.htn = htn
        self.repair_history: List[Dict] = []
    
    def diagnose_failure(
        self,
        failed_task: PlanningTask,
        error: str,
    ) -> Dict[str, Any]:
        """Diagnose why a task failed."""
        diagnosis = {
            "task_id": failed_task.id,
            "task_name": failed_task.name,
            "error": error,
            "missing_preconditions": [],
            "suggested_repairs": [],
        }
        
        # Check preconditions
        missing = failed_task.preconditions - self.htn.state
        diagnosis["missing_preconditions"] = list(missing)
        
        # Suggest repairs
        if missing:
            diagnosis["suggested_repairs"].append("achieve_preconditions")
        
        diagnosis["suggested_repairs"].extend([
            "retry_with_alternative",
            "skip_and_compensate",
            "escalate_to_parent",
        ])
        
        return diagnosis
    
    def repair(
        self,
        failed_task: PlanningTask,
        repair_type: str,
    ) -> List[PlanningTask]:
        """
        Repair the plan after a failure.
        
        Returns new tasks to add to plan.
        """
        new_tasks = []
        
        if repair_type == "achieve_preconditions":
            # Add tasks to achieve missing preconditions
            missing = failed_task.preconditions - self.htn.state
            for precond in missing:
                task = PlanningTask(
                    name=f"achieve_{precond}",
                    effects={precond},
                    priority=failed_task.priority - 1,
                )
                self.htn.add_task(task)
                new_tasks.append(task)
        
        elif repair_type == "retry_with_alternative":
            # Create alternative task
            alt_task = PlanningTask(
                name=f"alt_{failed_task.name}",
                parent_id=failed_task.parent_id,
                preconditions=failed_task.preconditions,
                effects=failed_task.effects,
                priority=failed_task.priority,
            )
            self.htn.add_task(alt_task)
            new_tasks.append(alt_task)
        
        elif repair_type == "skip_and_compensate":
            # Skip task and add compensation
            comp_task = PlanningTask(
                name=f"compensate_{failed_task.name}",
                effects=failed_task.effects,
                priority=failed_task.priority + 1,
            )
            self.htn.add_task(comp_task)
            new_tasks.append(comp_task)
        
        # Record repair
        self.repair_history.append({
            "failed_task": failed_task.id,
            "repair_type": repair_type,
            "new_tasks": [t.id for t in new_tasks],
        })
        
        return new_tasks


class AdvancedPlanner:
    """
    Complete advanced planning system.
    
    Integrates HTN, temporal reasoning, and plan repair.
    """
    
    def __init__(self):
        self.htn = HierarchicalTaskNetwork()
        self.temporal = TemporalReasoner()
        self.repair = PlanRepairModule(self.htn)
        
        self.current_plan: List[PlanningTask] = []
        self.execution_index = 0
    
    def create_plan(
        self,
        goal: str,
        deadline: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Create a complete plan for a goal.
        
        Returns plan with schedule.
        """
        # Create goal task
        goal_task = PlanningTask(
            name=goal,
            latest_finish=deadline,
        )
        
        # Generate plan
        self.current_plan = self.htn.plan(goal_task)
        
        # Schedule
        schedule = self.temporal.schedule_tasks(
            self.current_plan,
            datetime.now(),
        )
        
        # Check deadline
        deadline_ok = True
        if deadline:
            deadline_ok, _ = self.temporal.check_deadline(goal_task.id, deadline)
        
        return {
            "goal": goal,
            "tasks": len(self.current_plan),
            "schedule": {
                tid: (s.isoformat(), e.isoformat())
                for tid, (s, e) in schedule.items()
            },
            "deadline_feasible": deadline_ok,
        }
    
    def get_next_task(self) -> Optional[PlanningTask]:
        """Get the next task to execute."""
        while self.execution_index < len(self.current_plan):
            task = self.current_plan[self.execution_index]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.EXECUTING
                return task
            self.execution_index += 1
        return None
    
    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id in self.htn.tasks:
            self.htn.tasks[task_id].status = TaskStatus.COMPLETED
            # Update state with effects
            self.htn.state.update(self.htn.tasks[task_id].effects)
    
    def handle_failure(self, task_id: str, error: str) -> List[PlanningTask]:
        """Handle task failure with plan repair."""
        task = self.htn.tasks.get(task_id)
        if not task:
            return []
        
        task.status = TaskStatus.FAILED
        
        # Diagnose and repair
        diagnosis = self.repair.diagnose_failure(task, error)
        
        # Try first suggested repair
        if diagnosis["suggested_repairs"]:
            repair_type = diagnosis["suggested_repairs"][0]
            return self.repair.repair(task, repair_type)
        
        return []
