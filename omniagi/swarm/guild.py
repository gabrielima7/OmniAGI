"""
Guild - Multi-agent coordination and orchestration.
"""

from __future__ import annotations

import asyncio
import structlog
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from enum import Enum, auto

from omniagi.core.engine import Engine, GenerationConfig
from omniagi.agent.base import Agent
from omniagi.agent.persona import Persona
from omniagi.swarm.shared_brain import SharedBrain

logger = structlog.get_logger()


class AgentRole(Enum):
    """Predefined agent roles in the guild."""
    
    COORDINATOR = auto()   # Plans and delegates
    RESEARCHER = auto()    # Gathers information
    DEVELOPER = auto()     # Writes code
    REVIEWER = auto()      # Reviews and validates
    EXECUTOR = auto()      # Runs tasks


@dataclass
class GuildMember:
    """A member of the guild."""
    
    id: str
    role: AgentRole
    agent: Agent
    busy: bool = False
    tasks_completed: int = 0


@dataclass
class Task:
    """A task to be executed by the guild."""
    
    id: str
    description: str
    assigned_to: str | None = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Guild:
    """
    A guild of specialized agents working together.
    
    The guild orchestrates multiple agents with different specializations
    to solve complex tasks collaboratively.
    
    Architecture:
    - One Coordinator agent that plans and delegates
    - Multiple specialist agents (Researcher, Developer, etc.)
    - Shared brain for model access
    - Task queue for work distribution
    """
    
    def __init__(
        self,
        engine: Engine,
        name: str = "OmniGuild",
    ):
        """
        Initialize the guild.
        
        Args:
            engine: The inference engine.
            name: Guild name.
        """
        self.name = name
        self.shared_brain = SharedBrain(engine)
        self._members: dict[str, GuildMember] = {}
        self._tasks: dict[str, Task] = {}
        self._task_counter = 0
        
        logger.info("Guild created", name=name)
    
    def add_member(
        self,
        role: AgentRole,
        persona: Persona | None = None,
    ) -> GuildMember:
        """
        Add a new member to the guild.
        
        Args:
            role: The agent's role.
            persona: Custom persona (uses default for role if None).
            
        Returns:
            The created GuildMember.
        """
        member_id = f"{role.name.lower()}_{len(self._members)}"
        
        # Use role-specific persona if not provided
        if persona is None:
            persona = self._get_default_persona(role)
        
        # Create agent with shared engine
        agent = Agent(
            engine=self.shared_brain.engine,
            persona=persona,
        )
        
        member = GuildMember(
            id=member_id,
            role=role,
            agent=agent,
        )
        
        self._members[member_id] = member
        logger.info("Member added", member_id=member_id, role=role.name)
        
        return member
    
    def _get_default_persona(self, role: AgentRole) -> Persona:
        """Get default persona for a role."""
        match role:
            case AgentRole.COORDINATOR:
                return Persona.manager()
            case AgentRole.RESEARCHER:
                return Persona.researcher()
            case AgentRole.DEVELOPER:
                return Persona.developer()
            case AgentRole.REVIEWER:
                return Persona(
                    name="Reviewer",
                    role="Code Reviewer",
                    description="An expert at reviewing code and identifying issues.",
                    traits=["thorough", "critical", "constructive"],
                    expertise=["code review", "best practices", "testing"],
                )
            case _:
                return Persona.default()
    
    def get_member(self, member_id: str) -> GuildMember | None:
        """Get a member by ID."""
        return self._members.get(member_id)
    
    def get_members_by_role(self, role: AgentRole) -> list[GuildMember]:
        """Get all members with a specific role."""
        return [m for m in self._members.values() if m.role == role]
    
    def get_available_member(self, role: AgentRole | None = None) -> GuildMember | None:
        """Get an available (not busy) member."""
        for member in self._members.values():
            if not member.busy:
                if role is None or member.role == role:
                    return member
        return None
    
    async def assign_task(
        self,
        description: str,
        role: AgentRole | None = None,
        member_id: str | None = None,
    ) -> Task:
        """
        Assign a task to a guild member.
        
        Args:
            description: Task description.
            role: Preferred role for the task.
            member_id: Specific member to assign to.
            
        Returns:
            The created Task.
        """
        task_id = f"task_{self._task_counter}"
        self._task_counter += 1
        
        task = Task(id=task_id, description=description)
        self._tasks[task_id] = task
        
        # Find assignee
        if member_id:
            member = self._members.get(member_id)
        else:
            member = self.get_available_member(role)
        
        if member is None:
            task.status = "pending"
            logger.warning("No available member for task", task_id=task_id)
            return task
        
        task.assigned_to = member.id
        task.status = "assigned"
        
        logger.info("Task assigned", task_id=task_id, member=member.id)
        return task
    
    async def execute_task(self, task_id: str) -> Task:
        """
        Execute a task.
        
        Args:
            task_id: The task to execute.
            
        Returns:
            The completed Task.
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        if not task.assigned_to:
            raise ValueError(f"Task not assigned: {task_id}")
        
        member = self._members.get(task.assigned_to)
        if not member:
            raise ValueError(f"Member not found: {task.assigned_to}")
        
        member.busy = True
        task.status = "in_progress"
        
        try:
            logger.info("Executing task", task_id=task_id, member=member.id)
            result = await member.agent.run(task.description)
            
            task.result = result
            task.status = "completed"
            member.tasks_completed += 1
            
            logger.info("Task completed", task_id=task_id)
            
        except Exception as e:
            task.status = "failed"
            task.result = str(e)
            logger.error("Task failed", task_id=task_id, error=str(e))
        
        finally:
            member.busy = False
        
        return task
    
    async def solve(self, problem: str) -> str:
        """
        Collaboratively solve a complex problem.
        
        The coordinator breaks down the problem, delegates to specialists,
        and synthesizes the results.
        
        Args:
            problem: The problem to solve.
            
        Returns:
            The solution.
        """
        # Ensure we have a coordinator
        coordinators = self.get_members_by_role(AgentRole.COORDINATOR)
        if not coordinators:
            self.add_member(AgentRole.COORDINATOR)
            coordinators = self.get_members_by_role(AgentRole.COORDINATOR)
        
        coordinator = coordinators[0]
        
        # Phase 1: Planning
        logger.info("Phase 1: Planning")
        plan_prompt = (
            f"You are coordinating a team to solve this problem:\n\n{problem}\n\n"
            f"Break this down into subtasks. For each subtask, specify:\n"
            f"1. What needs to be done\n"
            f"2. What type of specialist should do it (researcher, developer, reviewer)\n"
            f"Format as a numbered list."
        )
        
        plan = await coordinator.agent.run(plan_prompt)
        logger.info("Plan created")
        
        # For MVP, return the plan directly
        # Full implementation would parse plan and delegate to specialists
        
        return f"## Plan\n{plan}\n\n[Guild execution of subtasks not yet implemented]"
    
    def get_stats(self) -> dict[str, Any]:
        """Get guild statistics."""
        return {
            "name": self.name,
            "members": len(self._members),
            "total_tasks": len(self._tasks),
            "completed_tasks": sum(
                1 for t in self._tasks.values() if t.status == "completed"
            ),
            "shared_brain_stats": self.shared_brain.get_stats(),
        }
