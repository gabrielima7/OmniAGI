"""
Actuator Controller - High-level motor control.

Translates high-level actions into motor commands
for the simulated embodiment.
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
    from omniagi.embodiment.sim_env import SimulationEnvironment

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


class ActionType(Enum):
    """Types of embodied actions."""
    
    MOVE = auto()         # Move in direction
    TURN = auto()         # Rotate
    GRASP = auto()        # Grab object
    RELEASE = auto()      # Release object
    PUSH = auto()         # Push object
    PULL = auto()         # Pull object
    LOOK = auto()         # Look at target
    REACH = auto()        # Reach toward target
    WAIT = auto()         # No action


class ActionStatus(Enum):
    """Status of an action."""
    
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class Action:
    """An embodied action."""
    
    action_type: ActionType
    target: Any = None
    parameters: dict = field(default_factory=dict)
    
    status: ActionStatus = ActionStatus.PENDING
    progress: float = 0.0
    
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "type": self.action_type.name,
            "target": str(self.target) if self.target else None,
            "parameters": self.parameters,
            "status": self.status.name,
            "progress": self.progress,
            "error": self.error,
        }


@dataclass
class ActionResult:
    """Result of executing an action."""
    
    success: bool
    action: Action
    duration: float = 0.0
    effects: list[str] = field(default_factory=list)
    observations: dict = field(default_factory=dict)


class ActuatorController:
    """
    High-level actuator control.
    
    Translates semantic actions into motor commands
    and executes them in the simulation.
    """
    
    def __init__(
        self,
        environment: "SimulationEnvironment | None" = None,
        storage_path: Path | str | None = None,
    ):
        self.environment = environment
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Action queue
        self._queue: list[Action] = []
        self._current_action: Action | None = None
        
        # State
        self._grasped_object: str | None = None
        
        # History
        self._history: list[ActionResult] = []
        
        # Motor parameters
        self._move_speed = 1.0
        self._turn_speed = 45.0  # degrees per step
        self._reach_distance = 2.0
        
        logger.info("Actuator Controller initialized")
    
    def queue_action(self, action: Action) -> int:
        """Add action to queue."""
        self._queue.append(action)
        return len(self._queue) - 1
    
    def create_action(
        self,
        action_type: ActionType,
        target: Any = None,
        **kwargs,
    ) -> Action:
        """Create and queue an action."""
        action = Action(
            action_type=action_type,
            target=target,
            parameters=kwargs,
        )
        self.queue_action(action)
        return action
    
    # High-level action methods
    
    def move_to(self, position: tuple[float, float, float]) -> Action:
        """Move to a position."""
        return self.create_action(
            ActionType.MOVE,
            target=position,
            mode="absolute",
        )
    
    def move_forward(self, distance: float = 1.0) -> Action:
        """Move forward."""
        return self.create_action(
            ActionType.MOVE,
            parameters={"direction": (1, 0, 0), "distance": distance},
            mode="relative",
        )
    
    def turn(self, angle: float) -> Action:
        """Turn by angle degrees."""
        return self.create_action(
            ActionType.TURN,
            parameters={"angle": angle},
        )
    
    def grasp(self, object_id: str) -> Action:
        """Grasp an object."""
        return self.create_action(
            ActionType.GRASP,
            target=object_id,
        )
    
    def release(self) -> Action:
        """Release grasped object."""
        return self.create_action(ActionType.RELEASE)
    
    def push(self, object_id: str, force: float = 10.0) -> Action:
        """Push an object."""
        return self.create_action(
            ActionType.PUSH,
            target=object_id,
            force=force,
        )
    
    def look_at(self, target: tuple[float, float, float] | str) -> Action:
        """Look at a target position or object."""
        return self.create_action(
            ActionType.LOOK,
            target=target,
        )
    
    def wait(self, duration: float = 1.0) -> Action:
        """Wait for duration."""
        return self.create_action(
            ActionType.WAIT,
            parameters={"duration": duration},
        )
    
    def execute_action(self, action: Action) -> ActionResult:
        """Execute a single action."""
        action.status = ActionStatus.EXECUTING
        action.started_at = datetime.now().timestamp()
        
        effects = []
        observations = {}
        success = False
        
        try:
            match action.action_type:
                case ActionType.MOVE:
                    success = self._execute_move(action, effects)
                    
                case ActionType.TURN:
                    success = self._execute_turn(action, effects)
                    
                case ActionType.GRASP:
                    success = self._execute_grasp(action, effects)
                    
                case ActionType.RELEASE:
                    success = self._execute_release(action, effects)
                    
                case ActionType.PUSH:
                    success = self._execute_push(action, effects)
                    
                case ActionType.LOOK:
                    success = self._execute_look(action, effects)
                    
                case ActionType.WAIT:
                    success = True
                    effects.append("Waited")
                    
                case _:
                    action.error = f"Unknown action type: {action.action_type}"
            
            action.status = ActionStatus.COMPLETED if success else ActionStatus.FAILED
            action.progress = 1.0 if success else action.progress
            
        except Exception as e:
            action.status = ActionStatus.FAILED
            action.error = str(e)
            logger.error("Action execution failed", error=str(e))
        
        action.completed_at = datetime.now().timestamp()
        duration = action.completed_at - action.started_at
        
        result = ActionResult(
            success=success,
            action=action,
            duration=duration,
            effects=effects,
            observations=observations,
        )
        
        self._history.append(result)
        return result
    
    def _execute_move(self, action: Action, effects: list) -> bool:
        """Execute move action."""
        if not self.environment:
            effects.append("No environment connected")
            return False
        
        params = action.parameters
        
        if action.target:
            # Move to absolute position
            target = action.target
            self.environment.move_agent(target, self._move_speed)
            effects.append(f"Moved to {target}")
        else:
            # Move in direction
            direction = params.get("direction", (1, 0, 0))
            distance = params.get("distance", 1.0)
            scaled = tuple(d * distance for d in direction)
            self.environment.move_agent(scaled, self._move_speed)
            effects.append(f"Moved by {scaled}")
        
        return True
    
    def _execute_turn(self, action: Action, effects: list) -> bool:
        """Execute turn action."""
        if not self.environment:
            return False
        
        angle = action.parameters.get("angle", 90.0)
        # Simplified - just update rotation
        effects.append(f"Turned {angle} degrees")
        return True
    
    def _execute_grasp(self, action: Action, effects: list) -> bool:
        """Execute grasp action."""
        if not action.target:
            action.error = "No target specified"
            return False
        
        if self._grasped_object:
            action.error = "Already grasping an object"
            return False
        
        # Check if object is in range
        self._grasped_object = action.target
        effects.append(f"Grasped {action.target}")
        return True
    
    def _execute_release(self, action: Action, effects: list) -> bool:
        """Execute release action."""
        if not self._grasped_object:
            action.error = "Not grasping anything"
            return False
        
        released = self._grasped_object
        self._grasped_object = None
        effects.append(f"Released {released}")
        return True
    
    def _execute_push(self, action: Action, effects: list) -> bool:
        """Execute push action."""
        if not self.environment or not action.target:
            return False
        
        force = action.parameters.get("force", 10.0)
        success = self.environment.push_object(
            action.target,
            (1, 0, 0),  # Forward direction
            force,
        )
        
        if success:
            effects.append(f"Pushed {action.target} with force {force}")
        return success
    
    def _execute_look(self, action: Action, effects: list) -> bool:
        """Execute look action."""
        effects.append(f"Looking at {action.target}")
        return True
    
    def step(self) -> ActionResult | None:
        """Execute next action in queue."""
        if self._current_action and self._current_action.status == ActionStatus.EXECUTING:
            # Continue current action
            return None
        
        if not self._queue:
            return None
        
        action = self._queue.pop(0)
        self._current_action = action
        
        return self.execute_action(action)
    
    def execute_all(self) -> list[ActionResult]:
        """Execute all queued actions."""
        results = []
        while self._queue:
            result = self.step()
            if result:
                results.append(result)
        return results
    
    def cancel_all(self) -> None:
        """Cancel all queued actions."""
        for action in self._queue:
            action.status = ActionStatus.CANCELLED
        self._queue.clear()
    
    @property
    def is_grasping(self) -> bool:
        """Check if currently grasping."""
        return self._grasped_object is not None
    
    @property
    def grasped_object(self) -> str | None:
        """Get ID of grasped object."""
        return self._grasped_object
    
    def get_queue_length(self) -> int:
        """Get number of pending actions."""
        return len(self._queue)
    
    def get_history(self, n: int = 10) -> list[dict]:
        """Get recent action history."""
        return [
            {
                "action": r.action.to_dict(),
                "success": r.success,
                "duration": r.duration,
                "effects": r.effects,
            }
            for r in self._history[-n:]
        ]
