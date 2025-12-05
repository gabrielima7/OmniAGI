"""
Introspection - Self-analysis and decision making.
"""

from __future__ import annotations

import structlog
from typing import Any, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from omniagi.agent.base import Agent

logger = structlog.get_logger()


class Introspector:
    """
    Handles agent introspection - deciding what to do autonomously.
    
    The introspector analyzes:
    - Current state and context
    - Past experiences and lessons
    - Scheduled tasks
    - Resource availability
    
    And suggests appropriate actions.
    """
    
    def __init__(self):
        """Initialize the introspector."""
        self._last_study_time: datetime | None = None
        self._last_reflect_time: datetime | None = None
        self._study_topics = [
            "software architecture patterns",
            "machine learning advances",
            "security best practices",
            "code optimization techniques",
            "problem-solving strategies",
        ]
        self._topic_index = 0
    
    async def suggest_actions(
        self,
        agent: "Agent",
        scheduled_tasks: list[dict[str, Any]] | None = None,
        max_suggestions: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Suggest actions for the daemon to take.
        
        Args:
            agent: The agent instance.
            scheduled_tasks: Pre-scheduled tasks.
            max_suggestions: Maximum suggestions to return.
            
        Returns:
            List of action dictionaries with 'type' and 'data'.
        """
        suggestions = []
        
        # Add scheduled tasks first
        if scheduled_tasks:
            for task in scheduled_tasks[:max_suggestions]:
                suggestions.append({
                    "type": "task",
                    "data": task,
                    "priority": task.get("priority", 5),
                })
        
        remaining = max_suggestions - len(suggestions)
        if remaining <= 0:
            return self._sort_by_priority(suggestions)
        
        # Check if it's time for self-study
        if self._should_study():
            topic = self._get_next_study_topic()
            suggestions.append({
                "type": "study",
                "data": {"topic": topic},
                "priority": 3,
            })
            self._last_study_time = datetime.now()
            remaining -= 1
        
        # Check if it's time for reflection
        if remaining > 0 and self._should_reflect():
            suggestions.append({
                "type": "reflect",
                "data": {},
                "priority": 2,
            })
            self._last_reflect_time = datetime.now()
            remaining -= 1
        
        # Maintenance if nothing else
        if remaining > 0 and not suggestions:
            suggestions.append({
                "type": "maintain",
                "data": {},
                "priority": 1,
            })
        
        return self._sort_by_priority(suggestions)[:max_suggestions]
    
    def _should_study(self) -> bool:
        """Check if it's time for self-study."""
        if self._last_study_time is None:
            return True
        
        # Study every 6 hours
        elapsed = datetime.now() - self._last_study_time
        return elapsed > timedelta(hours=6)
    
    def _should_reflect(self) -> bool:
        """Check if it's time for reflection."""
        if self._last_reflect_time is None:
            return True
        
        # Reflect every 12 hours
        elapsed = datetime.now() - self._last_reflect_time
        return elapsed > timedelta(hours=12)
    
    def _get_next_study_topic(self) -> str:
        """Get the next topic to study."""
        topic = self._study_topics[self._topic_index]
        self._topic_index = (self._topic_index + 1) % len(self._study_topics)
        return topic
    
    def _sort_by_priority(
        self,
        suggestions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sort suggestions by priority (higher first)."""
        return sorted(
            suggestions,
            key=lambda x: x.get("priority", 0),
            reverse=True,
        )
    
    async def analyze_state(self, agent: "Agent") -> dict[str, Any]:
        """
        Analyze the current agent state.
        
        Returns various metrics about the agent's state.
        """
        return {
            "conversation_length": len(agent.messages),
            "tools_available": len(agent.tools),
            "persona": agent.persona.name,
            "last_study": self._last_study_time.isoformat() if self._last_study_time else None,
            "last_reflect": self._last_reflect_time.isoformat() if self._last_reflect_time else None,
        }
