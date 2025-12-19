"""Agents module."""

from omniagi.agents.loop import (
    AgentLoop,
    ReactAgent,
    AgentState,
    AgentStep,
    Observation,
    Action,
    Evaluation,
)

from omniagi.agents.multi_agent import (
    MultiAgentSystem,
    SpecializedAgent,
    CoordinatorAgent,
    ResearcherAgent,
    ReasonerAgent,
    CoderAgent,
    CriticAgent,
    AgentRole,
    AgentMessage,
    AgentTask,
)

__all__ = [
    # Loop
    "AgentLoop",
    "ReactAgent",
    "AgentState",
    "AgentStep",
    "Observation",
    "Action",
    "Evaluation",
    # Multi-agent
    "MultiAgentSystem",
    "SpecializedAgent",
    "CoordinatorAgent",
    "ResearcherAgent",
    "ReasonerAgent",
    "CoderAgent",
    "CriticAgent",
    "AgentRole",
    "AgentMessage",
    "AgentTask",
]
