"""
Life Daemon - Autonomous life cycle management.

The daemon runs continuously, allowing the AI to:
- Wake up periodically
- Introspect on what to do
- Execute autonomous actions
- Rest and consolidate knowledge
"""

from __future__ import annotations

import asyncio
import structlog
from enum import Enum, auto
from datetime import datetime
from typing import Callable, Awaitable, Any

from omniagi.core.config import get_config
from omniagi.agent.base import Agent
from omniagi.daemon.introspection import Introspector
from omniagi.daemon.scheduler import TaskScheduler

logger = structlog.get_logger()


class DaemonState(Enum):
    """States of the life daemon."""
    
    DORMANT = auto()       # Not running
    WAKING = auto()        # Starting up
    INTROSPECTING = auto() # Deciding what to do
    ACTING = auto()        # Executing an action
    RESTING = auto()       # Between cycles
    SHUTTING_DOWN = auto() # Graceful shutdown


class LifeDaemon:
    """
    The Life Daemon manages the autonomous life cycle of OmniAGI.
    
    It operates in a continuous loop:
    1. Wake up
    2. Introspect (what should I do?)
    3. Act (execute the chosen action)
    4. Rest (wait before next cycle)
    
    The daemon can be configured to:
    - Run continuously or on a schedule
    - Limit actions per cycle
    - Prioritize different types of activities
    """
    
    def __init__(
        self,
        agent: Agent,
        introspector: Introspector | None = None,
        scheduler: TaskScheduler | None = None,
    ):
        """
        Initialize the life daemon.
        
        Args:
            agent: The agent instance to control.
            introspector: Custom introspector (uses default if None).
            scheduler: Custom scheduler (uses default if None).
        """
        self.agent = agent
        self.introspector = introspector or Introspector()
        self.scheduler = scheduler or TaskScheduler()
        self.config = get_config()
        
        self._state = DaemonState.DORMANT
        self._running = False
        self._cycle_count = 0
        self._last_action_time: datetime | None = None
        
        # Callbacks
        self._on_state_change: list[Callable[[DaemonState], None]] = []
        self._on_action: list[Callable[[str, Any], Awaitable[None]]] = []
        
        logger.info("Life Daemon initialized")
    
    @property
    def state(self) -> DaemonState:
        """Get the current daemon state."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if the daemon is running."""
        return self._running
    
    @property
    def cycle_count(self) -> int:
        """Get the number of completed cycles."""
        return self._cycle_count
    
    def _set_state(self, new_state: DaemonState) -> None:
        """Set state and notify callbacks."""
        old_state = self._state
        self._state = new_state
        
        logger.debug("Daemon state change", old=old_state.name, new=new_state.name)
        
        for callback in self._on_state_change:
            try:
                callback(new_state)
            except Exception as e:
                logger.warning("State callback error", error=str(e))
    
    def on_state_change(self, callback: Callable[[DaemonState], None]) -> None:
        """Register a callback for state changes."""
        self._on_state_change.append(callback)
    
    def on_action(self, callback: Callable[[str, Any], Awaitable[None]]) -> None:
        """Register a callback for actions."""
        self._on_action.append(callback)
    
    async def start(self) -> None:
        """Start the life daemon."""
        if self._running:
            logger.warning("Daemon already running")
            return
        
        if not self.config.daemon.enabled:
            logger.warning("Daemon is disabled in configuration")
            return
        
        self._running = True
        self._set_state(DaemonState.WAKING)
        
        logger.info("Life Daemon starting")
        
        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("Daemon loop cancelled")
        finally:
            self._running = False
            self._set_state(DaemonState.DORMANT)
    
    async def stop(self) -> None:
        """Stop the life daemon gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping Life Daemon")
        self._set_state(DaemonState.SHUTTING_DOWN)
        self._running = False
    
    async def _run_loop(self) -> None:
        """Main daemon loop."""
        interval = self.config.daemon.introspection_interval
        
        while self._running:
            try:
                # Introspect
                self._set_state(DaemonState.INTROSPECTING)
                actions = await self._introspect()
                
                if actions:
                    # Act
                    self._set_state(DaemonState.ACTING)
                    await self._execute_actions(actions)
                    self._last_action_time = datetime.now()
                
                # Rest
                self._set_state(DaemonState.RESTING)
                self._cycle_count += 1
                
                logger.info(
                    "Cycle completed",
                    cycle=self._cycle_count,
                    actions_taken=len(actions),
                )
                
                # Wait before next cycle
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error("Error in daemon loop", error=str(e))
                await asyncio.sleep(30)  # Brief pause on error
    
    async def _introspect(self) -> list[dict[str, Any]]:
        """Determine what actions to take."""
        max_actions = self.config.daemon.max_actions_per_cycle
        
        # Get scheduled tasks
        scheduled = self.scheduler.get_due_tasks()
        
        # Get introspection suggestions
        suggestions = await self.introspector.suggest_actions(
            agent=self.agent,
            scheduled_tasks=scheduled,
            max_suggestions=max_actions,
        )
        
        return suggestions[:max_actions]
    
    async def _execute_actions(self, actions: list[dict[str, Any]]) -> None:
        """Execute a list of actions."""
        for action in actions:
            action_type = action.get("type", "unknown")
            action_data = action.get("data", {})
            
            logger.info("Executing action", type=action_type)
            
            try:
                result = await self._execute_single_action(action_type, action_data)
                
                # Notify callbacks
                for callback in self._on_action:
                    await callback(action_type, result)
                    
            except Exception as e:
                logger.error(
                    "Action failed",
                    type=action_type,
                    error=str(e),
                )
    
    async def _execute_single_action(
        self,
        action_type: str,
        data: dict[str, Any],
    ) -> Any:
        """Execute a single action."""
        match action_type:
            case "study":
                # Self-study action
                topic = data.get("topic", "general knowledge")
                return await self.agent.run(
                    f"Learn about {topic} and summarize key points."
                )
            
            case "reflect":
                # Self-reflection
                return await self.agent.run(
                    "Reflect on recent interactions and identify areas for improvement."
                )
            
            case "maintain":
                # System maintenance
                return await self.agent.run(
                    "Review system health and suggest optimizations."
                )
            
            case "task":
                # Execute a specific task
                task_prompt = data.get("prompt", "")
                return await self.agent.run(task_prompt)
            
            case _:
                logger.warning("Unknown action type", type=action_type)
                return None
    
    async def run_once(self) -> list[Any]:
        """Run a single daemon cycle manually."""
        logger.info("Running single daemon cycle")
        
        self._set_state(DaemonState.INTROSPECTING)
        actions = await self._introspect()
        
        results = []
        if actions:
            self._set_state(DaemonState.ACTING)
            for action in actions:
                result = await self._execute_single_action(
                    action.get("type", "unknown"),
                    action.get("data", {}),
                )
                results.append(result)
        
        self._set_state(DaemonState.DORMANT)
        return results
