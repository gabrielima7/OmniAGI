"""
Life Daemon - Autonomous life cycle with AGI integration.

The daemon runs continuously, coordinating all AGI pillars:
- Ouroboros: Self-improvement
- Meta-Learning: Adaptive strategies
- Continual Learning: Memory consolidation
- Causal Reasoning: Understanding causation
- World Model: Simulation and planning
"""

from __future__ import annotations

import asyncio
import structlog
from enum import Enum, auto
from datetime import datetime
from typing import Callable, Awaitable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.daemon.brain import AGIBrain

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
    LEARNING = auto()      # Consolidating knowledge
    IMPROVING = auto()     # Self-improvement cycle
    PLANNING = auto()      # Long-term planning
    RESTING = auto()       # Between cycles
    SHUTTING_DOWN = auto() # Graceful shutdown


class LifeDaemon:
    """
    AGI Life Daemon - Autonomous cognitive life cycle.
    
    Integrates all AGI pillars into a continuous cycle:
    1. Wake up
    2. Introspect (what should I do?)
    3. Plan (using World Model)
    4. Act (execute with Meta-Learning)
    5. Learn (Continual Learning)
    6. Improve (Ouroboros self-improvement)
    7. Rest (prepare for next cycle)
    """
    
    def __init__(
        self,
        agent: Agent,
        brain: "AGIBrain | None" = None,
        introspector: Introspector | None = None,
        scheduler: TaskScheduler | None = None,
    ):
        """
        Initialize the AGI life daemon.
        
        Args:
            agent: The agent instance to control.
            brain: AGI Brain with all pillars (created if None).
            introspector: Custom introspector (uses default if None).
            scheduler: Custom scheduler (uses default if None).
        """
        self.agent = agent
        self.introspector = introspector or Introspector()
        self.scheduler = scheduler or TaskScheduler()
        self.config = get_config()
        
        # AGI Brain integration
        self._brain = brain
        self._brain_initialized = False
        
        self._state = DaemonState.DORMANT
        self._running = False
        self._cycle_count = 0
        self._last_action_time: datetime | None = None
        self._last_consolidation: datetime | None = None
        self._last_improvement: datetime | None = None
        
        # Callbacks
        self._on_state_change: list[Callable[[DaemonState], None]] = []
        self._on_action: list[Callable[[str, Any], Awaitable[None]]] = []
        
        logger.info("AGI Life Daemon initialized")
    
    def _init_brain(self) -> None:
        """Lazily initialize AGI Brain."""
        if self._brain_initialized:
            return
        
        if self._brain is None:
            from omniagi.daemon.brain import AGIBrain
            self._brain = AGIBrain(
                agent=self.agent,
                engine=self.agent.engine,
                work_dir=str(self.config.data_dir),
            )
        
        self._brain_initialized = True
        logger.info("AGI Brain connected to Life Daemon")
    
    @property
    def brain(self) -> "AGIBrain":
        """Get the AGI Brain, initializing if needed."""
        self._init_brain()
        return self._brain  # type: ignore
    
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
        """Start the AGI life daemon."""
        if self._running:
            logger.warning("Daemon already running")
            return
        
        if not self.config.daemon.enabled:
            logger.warning("Daemon is disabled in configuration")
            return
        
        self._running = True
        self._set_state(DaemonState.WAKING)
        self._init_brain()
        
        logger.info("AGI Life Daemon starting with full cognitive capabilities")
        
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
        
        logger.info("Stopping AGI Life Daemon")
        self._set_state(DaemonState.SHUTTING_DOWN)
        
        # Final consolidation before shutdown
        if self._brain_initialized:
            await self.brain.consolidate()
        
        self._running = False
    
    async def _run_loop(self) -> None:
        """Main AGI daemon loop."""
        interval = self.config.daemon.introspection_interval
        consolidation_interval = 3600  # 1 hour
        improvement_interval = 86400   # 24 hours
        
        while self._running:
            try:
                cycle_start = datetime.now()
                
                # 1. Introspect - Decide what to do
                self._set_state(DaemonState.INTROSPECTING)
                actions = await self._introspect()
                
                # 2. Plan - Use World Model if needed
                if actions:
                    self._set_state(DaemonState.PLANNING)
                    actions = await self._plan_actions(actions)
                
                # 3. Act - Execute with cognitive enhancement
                if actions:
                    self._set_state(DaemonState.ACTING)
                    await self._execute_actions(actions)
                    self._last_action_time = datetime.now()
                
                # 4. Learn - Periodic memory consolidation
                should_consolidate = (
                    self._last_consolidation is None or
                    (datetime.now() - self._last_consolidation).seconds > consolidation_interval
                )
                if should_consolidate:
                    self._set_state(DaemonState.LEARNING)
                    await self._consolidate_learning()
                    self._last_consolidation = datetime.now()
                
                # 5. Improve - Periodic self-improvement
                should_improve = (
                    self._last_improvement is None or
                    (datetime.now() - self._last_improvement).seconds > improvement_interval
                )
                if should_improve and self._cycle_count > 0:
                    self._set_state(DaemonState.IMPROVING)
                    await self._self_improve()
                    self._last_improvement = datetime.now()
                
                # 6. Rest
                self._set_state(DaemonState.RESTING)
                self._cycle_count += 1
                
                cycle_duration = (datetime.now() - cycle_start).seconds
                
                logger.info(
                    "AGI Cycle completed",
                    cycle=self._cycle_count,
                    actions_taken=len(actions),
                    duration_seconds=cycle_duration,
                    brain_status=self.brain.get_status()["metrics"] if self._brain_initialized else {},
                )
                
                # Wait before next cycle
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error("Error in AGI daemon loop", error=str(e))
                await asyncio.sleep(30)
    
    async def _introspect(self) -> list[dict[str, Any]]:
        """Determine what actions to take using Meta-Learning."""
        max_actions = self.config.daemon.max_actions_per_cycle
        
        # Get scheduled tasks
        scheduled = self.scheduler.get_due_tasks()
        
        # Get introspection suggestions
        suggestions = await self.introspector.suggest_actions(
            agent=self.agent,
            scheduled_tasks=scheduled,
            max_suggestions=max_actions,
        )
        
        # Enhance with Meta-Learning if available
        if self._brain_initialized:
            for suggestion in suggestions:
                approach = self.brain.meta_learner.suggest_approach(
                    task_description=suggestion.get("description", ""),
                    domain="daemon",
                    task_type=suggestion.get("type", "general"),
                )
                suggestion["meta_approach"] = approach
        
        return suggestions[:max_actions]
    
    async def _plan_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Plan actions using World Model."""
        if not self._brain_initialized:
            return actions
        
        planned_actions = []
        
        for action in actions:
            # Simulate action first
            action_desc = action.get("description", str(action.get("type", "")))
            prediction = self.brain.simulator.simulate_action(
                action=action_desc,
                actor_id="self",
                use_llm=False,  # Quick simulation
            )
            
            # Only include if probability is acceptable
            if prediction.probability >= 0.3:
                action["prediction"] = {
                    "probability": prediction.probability,
                    "effects": prediction.predicted_effects[:2],
                    "risks": prediction.risks[:1],
                }
                planned_actions.append(action)
            else:
                logger.info(
                    "Action skipped due to low probability",
                    action=action_desc[:50],
                    probability=prediction.probability,
                )
        
        return planned_actions
    
    async def _execute_actions(self, actions: list[dict[str, Any]]) -> None:
        """Execute actions with cognitive enhancement."""
        for action in actions:
            action_type = action.get("type", "unknown")
            action_data = action.get("data", {})
            
            logger.info("Executing AGI action", type=action_type)
            
            try:
                result = await self._execute_single_action(action_type, action_data)
                
                # Record decision for explainability
                if self._brain_initialized:
                    self.brain.decision_explainer.record_decision(
                        decision=f"Executed {action_type}",
                        action_taken=str(result)[:100] if result else "completed",
                        inputs=action_data,
                        causes=action.get("prediction", {}).get("effects", []),
                    )
                
                # Notify callbacks
                for callback in self._on_action:
                    await callback(action_type, result)
                    
            except Exception as e:
                logger.error("AGI Action failed", type=action_type, error=str(e))
    
    async def _execute_single_action(
        self,
        action_type: str,
        data: dict[str, Any],
    ) -> Any:
        """Execute a single action with AGI enhancement."""
        match action_type:
            case "think":
                # Full AGI thinking process
                query = data.get("query", "What should I focus on?")
                return await self.brain.think(query)
            
            case "study":
                # Enhanced self-study
                topic = data.get("topic", "general knowledge")
                result = await self.brain.think(f"Learn about {topic} and summarize key points.")
                
                # Add to knowledge graph
                self.brain.knowledge_graph.add_node(
                    name=topic,
                    node_type="concept",
                    learned=True,
                )
                return result
            
            case "reflect":
                # Self-reflection with causal analysis
                reflection = await self.agent.run(
                    "Reflect on recent interactions and identify areas for improvement."
                )
                
                # Record causal insights
                self.brain.causal_graph.add_node(
                    name="reflection",
                    node_type="event",
                )
                return reflection
            
            case "plan":
                # Hierarchical planning
                goal_desc = data.get("goal", "Improve capabilities")
                goal = self.brain.planner.create_goal(description=goal_desc)
                plan = self.brain.planner.plan_for_goal(goal.id)
                return {"goal_id": goal.id, "plan_steps": len(plan.steps) if plan else 0}
            
            case "improve":
                # Self-improvement
                return await self.brain.self_improve()
            
            case "task":
                # Execute with full cognitive enhancement
                task_prompt = data.get("prompt", "")
                return await self.brain.think(task_prompt)
            
            case _:
                # Fallback to basic agent
                logger.warning("Unknown action type, using basic agent", type=action_type)
                return await self.agent.run(str(data))
    
    async def _consolidate_learning(self) -> None:
        """Run memory consolidation."""
        if not self._brain_initialized:
            return
        
        stats = await self.brain.consolidate()
        logger.info("Memory consolidation complete", **stats)
    
    async def _self_improve(self) -> None:
        """Run self-improvement cycle."""
        if not self._brain_initialized:
            return
        
        # Only improve if we have a valid work directory
        try:
            result = await self.brain.self_improve()
            logger.info("Self-improvement cycle complete", **result)
        except Exception as e:
            logger.warning("Self-improvement skipped", error=str(e))
    
    async def run_once(self) -> dict[str, Any]:
        """Run a single AGI daemon cycle manually."""
        logger.info("Running single AGI daemon cycle")
        self._init_brain()
        
        results = {"actions": [], "consolidation": None, "status": {}}
        
        self._set_state(DaemonState.INTROSPECTING)
        actions = await self._introspect()
        
        if actions:
            self._set_state(DaemonState.PLANNING)
            actions = await self._plan_actions(actions)
            
            self._set_state(DaemonState.ACTING)
            for action in actions:
                result = await self._execute_single_action(
                    action.get("type", "unknown"),
                    action.get("data", {}),
                )
                results["actions"].append({
                    "type": action.get("type"),
                    "result": str(result)[:200] if result else None,
                })
        
        self._set_state(DaemonState.LEARNING)
        results["consolidation"] = await self.brain.consolidate()
        
        results["status"] = self.brain.get_status()
        
        self._set_state(DaemonState.DORMANT)
        return results
    
    def get_brain_status(self) -> dict[str, Any]:
        """Get AGI Brain status."""
        if not self._brain_initialized:
            return {"initialized": False}
        return self.brain.get_status()

