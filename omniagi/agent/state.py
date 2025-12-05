"""
Agent state management with finite state machine.
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


class AgentState(Enum):
    """Possible states an agent can be in."""
    
    IDLE = auto()           # Waiting for input
    THINKING = auto()       # Processing/reasoning
    ACTING = auto()         # Executing a tool
    WAITING = auto()        # Waiting for external result
    ERROR = auto()          # In error state
    TERMINATED = auto()     # Finished execution


@dataclass
class StateTransition:
    """Records a state transition."""
    
    from_state: AgentState
    to_state: AgentState
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentStateManager:
    """
    Manages agent state transitions with history tracking.
    
    Implements a finite state machine pattern for agent lifecycle.
    """
    
    # Valid state transitions
    VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
        AgentState.IDLE: {AgentState.THINKING, AgentState.TERMINATED},
        AgentState.THINKING: {AgentState.ACTING, AgentState.IDLE, AgentState.ERROR},
        AgentState.ACTING: {AgentState.WAITING, AgentState.THINKING, AgentState.ERROR},
        AgentState.WAITING: {AgentState.THINKING, AgentState.ERROR},
        AgentState.ERROR: {AgentState.IDLE, AgentState.TERMINATED},
        AgentState.TERMINATED: set(),  # Terminal state
    }
    
    def __init__(self, initial_state: AgentState = AgentState.IDLE):
        self._current_state = initial_state
        self._history: list[StateTransition] = []
        self._callbacks: dict[AgentState, list[Callable[[StateTransition], None]]] = {}
    
    @property
    def current_state(self) -> AgentState:
        """Get the current agent state."""
        return self._current_state
    
    @property
    def history(self) -> list[StateTransition]:
        """Get the state transition history."""
        return self._history.copy()
    
    @property
    def is_terminal(self) -> bool:
        """Check if the agent is in a terminal state."""
        return self._current_state == AgentState.TERMINATED
    
    def can_transition_to(self, target: AgentState) -> bool:
        """Check if a transition to the target state is valid."""
        return target in self.VALID_TRANSITIONS.get(self._current_state, set())
    
    def transition_to(
        self,
        target: AgentState,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        """
        Transition to a new state.
        
        Args:
            target: The target state.
            reason: Reason for the transition.
            metadata: Additional context.
            
        Returns:
            The recorded StateTransition.
            
        Raises:
            ValueError: If the transition is invalid.
        """
        if not self.can_transition_to(target):
            raise ValueError(
                f"Invalid transition: {self._current_state} -> {target}"
            )
        
        transition = StateTransition(
            from_state=self._current_state,
            to_state=target,
            reason=reason,
            metadata=metadata or {},
        )
        
        self._current_state = target
        self._history.append(transition)
        
        # Fire callbacks
        for callback in self._callbacks.get(target, []):
            try:
                callback(transition)
            except Exception:
                pass  # Don't let callback errors affect state management
        
        return transition
    
    def on_enter(
        self,
        state: AgentState,
        callback: Callable[[StateTransition], None],
    ) -> None:
        """Register a callback for when entering a state."""
        if state not in self._callbacks:
            self._callbacks[state] = []
        self._callbacks[state].append(callback)
    
    def reset(self) -> None:
        """Reset to initial state, clearing history."""
        self._current_state = AgentState.IDLE
        self._history.clear()
    
    def get_time_in_state(self) -> float:
        """Get seconds spent in current state."""
        if not self._history:
            return 0.0
        
        last_transition = self._history[-1]
        delta = datetime.now() - last_transition.timestamp
        return delta.total_seconds()
