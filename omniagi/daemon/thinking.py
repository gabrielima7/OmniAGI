"""
Background Thinking Daemon - Continuous consciousness.

Enables the AGI to think between interactions,
process memories, generate goals, and improve itself
in the background.
"""

from __future__ import annotations

import asyncio
import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING
from uuid import uuid4
import threading
import time

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

logger = structlog.get_logger()


class ThinkingMode(Enum):
    """Modes of background thinking."""
    
    IDLE = auto()           # Minimal processing
    REFLECTION = auto()     # Review past experiences
    PLANNING = auto()       # Generate future plans
    CONSOLIDATION = auto()  # Memory consolidation
    IMPROVEMENT = auto()    # Self-improvement
    EXPLORATION = auto()    # Curiosity-driven thinking


@dataclass
class Thought:
    """A single thought produced by background thinking."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    mode: ThinkingMode = ThinkingMode.REFLECTION
    content: str = ""
    importance: float = 0.5  # 0-1
    
    # Connections
    related_memories: list[str] = field(default_factory=list)
    generated_goals: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode.name,
            "content": self.content[:200],
            "importance": self.importance,
            "insights": self.insights[:3],
            "created_at": self.created_at,
        }


@dataclass
class ThinkingCycle:
    """Summary of a thinking cycle."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: str | None = None
    
    thoughts_generated: int = 0
    memories_consolidated: int = 0
    goals_generated: int = 0
    improvements_proposed: int = 0
    
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thoughts": self.thoughts_generated,
            "goals": self.goals_generated,
            "duration": self.duration_seconds,
        }


class BackgroundThinkingDaemon:
    """
    Continuous consciousness through background thinking.
    
    This is a critical AGI component that enables:
    - Thinking between user interactions
    - Memory consolidation during idle time
    - Spontaneous goal generation
    - Continuous self-improvement
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        storage_path: Path | str | None = None,
        think_interval: float = 30.0,  # Seconds between think cycles
    ):
        self.engine = engine
        self.storage_path = Path(storage_path) if storage_path else None
        self.think_interval = think_interval
        
        # State
        self._running = False
        self._current_mode = ThinkingMode.IDLE
        self._thread: threading.Thread | None = None
        
        # History
        self._thoughts: list[Thought] = []
        self._cycles: list[ThinkingCycle] = []
        
        # Integration hooks
        self._memory_hook: Callable | None = None
        self._goal_hook: Callable | None = None
        self._improvement_hook: Callable | None = None
        
        # Thinking prompts by mode
        self._prompts = {
            ThinkingMode.REFLECTION: """Reflect on recent experiences and extract insights.
What patterns do you notice? What could be improved?
Think deeply about what you've learned.""",
            
            ThinkingMode.PLANNING: """Consider your goals and responsibilities.
What should you focus on next? What opportunities exist?
Generate a plan for future actions.""",
            
            ThinkingMode.CONSOLIDATION: """Review and organize your memories.
Identify important information to remember.
Create connections between related concepts.""",
            
            ThinkingMode.IMPROVEMENT: """Analyze your own capabilities.
Where are you weak? How could you improve?
Propose specific enhancements to yourself.""",
            
            ThinkingMode.EXPLORATION: """Explore interesting ideas and questions.
What are you curious about? What mysteries exist?
Follow your curiosity to new understanding.""",
        }
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Background Thinking Daemon initialized")
    
    def start(self) -> bool:
        """Start the background thinking daemon."""
        if self._running:
            logger.warning("Daemon already running")
            return False
        
        self._running = True
        self._thread = threading.Thread(target=self._think_loop, daemon=True)
        self._thread.start()
        
        logger.info("Background thinking started")
        return True
    
    def stop(self) -> None:
        """Stop the background thinking daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._thread = None
        
        logger.info("Background thinking stopped")
    
    def _think_loop(self) -> None:
        """Main thinking loop."""
        while self._running:
            try:
                # Wait for interval (but check running flag periodically)
                for _ in range(int(self.think_interval)):
                    if not self._running:
                        return
                    time.sleep(1)
                
                # Run a thinking cycle
                self._run_cycle()
                
            except Exception as e:
                logger.error("Thinking cycle failed", error=str(e))
                time.sleep(5)  # Brief pause on error
    
    def _run_cycle(self) -> ThinkingCycle:
        """Run a single thinking cycle."""
        cycle = ThinkingCycle()
        start_time = time.time()
        
        # Choose mode based on various factors
        mode = self._choose_mode()
        self._current_mode = mode
        
        # Generate thoughts
        thoughts = self._generate_thoughts(mode)
        cycle.thoughts_generated = len(thoughts)
        
        # Process thoughts based on mode
        if mode == ThinkingMode.CONSOLIDATION:
            cycle.memories_consolidated = self._consolidate_memories(thoughts)
        elif mode == ThinkingMode.PLANNING:
            cycle.goals_generated = self._generate_goals(thoughts)
        elif mode == ThinkingMode.IMPROVEMENT:
            cycle.improvements_proposed = self._propose_improvements(thoughts)
        
        # Store thoughts
        self._thoughts.extend(thoughts)
        
        # Finalize cycle
        cycle.ended_at = datetime.now().isoformat()
        cycle.duration_seconds = time.time() - start_time
        self._cycles.append(cycle)
        
        self._save()
        
        logger.info(
            "Thinking cycle completed",
            mode=mode.name,
            thoughts=cycle.thoughts_generated,
            duration=f"{cycle.duration_seconds:.1f}s",
        )
        
        return cycle
    
    def _choose_mode(self) -> ThinkingMode:
        """Choose thinking mode based on current needs."""
        # Simple round-robin with bias toward reflection
        modes = [
            ThinkingMode.REFLECTION,
            ThinkingMode.REFLECTION,
            ThinkingMode.PLANNING,
            ThinkingMode.CONSOLIDATION,
            ThinkingMode.IMPROVEMENT,
            ThinkingMode.EXPLORATION,
        ]
        
        cycle_num = len(self._cycles)
        return modes[cycle_num % len(modes)]
    
    def _generate_thoughts(self, mode: ThinkingMode) -> list[Thought]:
        """Generate thoughts using LLM."""
        thoughts = []
        
        if not self.engine or not self.engine.is_loaded:
            # Generate placeholder thoughts without LLM
            thought = Thought(
                mode=mode,
                content=f"Thinking in {mode.name} mode... (no LLM available)",
                importance=0.3,
            )
            thoughts.append(thought)
            return thoughts
        
        # Generate with LLM
        prompt = self._prompts.get(mode, "Think freely about anything interesting.")
        
        try:
            start = time.time()
            response = self.engine.generate(prompt, max_tokens=300)
            elapsed = (time.time() - start) * 1000
            
            # Parse response into thought
            thought = Thought(
                mode=mode,
                content=response.text,
                importance=self._assess_importance(response.text),
                processing_time_ms=elapsed,
            )
            
            # Extract insights
            thought.insights = self._extract_insights(response.text)
            
            thoughts.append(thought)
            
        except Exception as e:
            logger.error("Thought generation failed", error=str(e))
        
        return thoughts
    
    def _assess_importance(self, content: str) -> float:
        """Assess importance of a thought."""
        # Simple heuristic based on content
        importance = 0.5
        
        important_keywords = [
            "important", "critical", "key", "insight", "realize",
            "understand", "must", "should", "goal", "improve",
        ]
        
        content_lower = content.lower()
        for keyword in important_keywords:
            if keyword in content_lower:
                importance += 0.05
        
        return min(1.0, importance)
    
    def _extract_insights(self, content: str) -> list[str]:
        """Extract key insights from thought content."""
        insights = []
        
        # Simple extraction - sentences with key phrases
        sentences = content.split(".")
        insight_phrases = ["I realize", "I understand", "This means", "The key"]
        
        for sentence in sentences:
            for phrase in insight_phrases:
                if phrase.lower() in sentence.lower():
                    insights.append(sentence.strip())
                    break
        
        return insights[:5]
    
    def _consolidate_memories(self, thoughts: list[Thought]) -> int:
        """Consolidate memories based on thoughts."""
        if not self._memory_hook:
            return 0
        
        count = 0
        for thought in thoughts:
            if thought.importance > 0.6:
                try:
                    self._memory_hook(thought.content, thought.insights)
                    count += 1
                except Exception:
                    pass
        
        return count
    
    def _generate_goals(self, thoughts: list[Thought]) -> int:
        """Generate goals from planning thoughts."""
        if not self._goal_hook:
            return 0
        
        count = 0
        for thought in thoughts:
            for insight in thought.insights:
                if "should" in insight.lower() or "will" in insight.lower():
                    try:
                        self._goal_hook(insight)
                        thought.generated_goals.append(insight)
                        count += 1
                    except Exception:
                        pass
        
        return count
    
    def _propose_improvements(self, thoughts: list[Thought]) -> int:
        """Propose self-improvements from thoughts."""
        if not self._improvement_hook:
            return 0
        
        count = 0
        for thought in thoughts:
            if "improve" in thought.content.lower() or "better" in thought.content.lower():
                try:
                    self._improvement_hook(thought.content)
                    count += 1
                except Exception:
                    pass
        
        return count
    
    def think_now(self, mode: ThinkingMode = None) -> Thought | None:
        """Trigger immediate thinking (useful for testing)."""
        if mode:
            self._current_mode = mode
        else:
            mode = self._choose_mode()
        
        thoughts = self._generate_thoughts(mode)
        self._thoughts.extend(thoughts)
        self._save()
        
        return thoughts[0] if thoughts else None
    
    def set_hooks(
        self,
        memory_hook: Callable = None,
        goal_hook: Callable = None,
        improvement_hook: Callable = None,
    ) -> None:
        """Set integration hooks."""
        if memory_hook:
            self._memory_hook = memory_hook
        if goal_hook:
            self._goal_hook = goal_hook
        if improvement_hook:
            self._improvement_hook = improvement_hook
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def current_mode(self) -> ThinkingMode:
        return self._current_mode
    
    def get_recent_thoughts(self, n: int = 10) -> list[Thought]:
        """Get recent thoughts."""
        return self._thoughts[-n:]
    
    def get_important_insights(self, min_importance: float = 0.6) -> list[str]:
        """Get all important insights."""
        insights = []
        for thought in self._thoughts:
            if thought.importance >= min_importance:
                insights.extend(thought.insights)
        return insights
    
    def get_stats(self) -> dict:
        """Get daemon statistics."""
        total_thoughts = len(self._thoughts)
        total_cycles = len(self._cycles)
        
        return {
            "running": self._running,
            "current_mode": self._current_mode.name,
            "total_thoughts": total_thoughts,
            "total_cycles": total_cycles,
            "avg_thoughts_per_cycle": total_thoughts / max(1, total_cycles),
            "think_interval": self.think_interval,
            "insights_count": sum(len(t.insights) for t in self._thoughts),
        }
    
    def __len__(self) -> int:
        return len(self._thoughts)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "thoughts": [t.to_dict() for t in self._thoughts[-100:]],
                "cycles": [c.to_dict() for c in self._cycles[-50:]],
                "stats": self.get_stats(),
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        # Simplified load
        pass
