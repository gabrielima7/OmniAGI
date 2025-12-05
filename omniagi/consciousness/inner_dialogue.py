"""
Inner Dialogue - Internal monologue system for AGI.

Implements internal self-talk that:
1. Plans before acting
2. Debates alternatives
3. Self-corrects
4. Builds narrative coherence
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, List
from uuid import uuid4

logger = structlog.get_logger()


class DialogueType(Enum):
    """Types of inner dialogue."""
    
    PLANNING = auto()      # What should I do?
    REASONING = auto()     # Why is this true?
    DEBATE = auto()        # Pros vs cons
    CORRECTION = auto()    # I was wrong because...
    REFLECTION = auto()    # What did I learn?
    NARRATIVE = auto()     # The story so far...


@dataclass
class InnerVoice:
    """A voice in the inner dialogue."""
    
    name: str
    perspective: str  # e.g., "analytical", "creative", "cautious"
    weight: float = 1.0
    
    def speak(self, topic: str) -> str:
        """Generate speech from this perspective."""
        return f"[{self.name}]: From a {self.perspective} view, {topic}"


@dataclass
class DialogueTurn:
    """A turn in the inner dialogue."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    voice: str = ""
    content: str = ""
    dialogue_type: DialogueType = DialogueType.REASONING
    
    # Reasoning
    supports: list[str] = field(default_factory=list)
    opposes: list[str] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class InnerDialogue:
    """
    Inner Dialogue System.
    
    Creates internal monologue for:
    - Planning and decision-making
    - Self-debate and evaluation
    - Error correction
    - Narrative building
    """
    
    def __init__(self):
        # Default voices
        self._voices: dict[str, InnerVoice] = {
            "analyst": InnerVoice("Analyst", "analytical"),
            "creative": InnerVoice("Creative", "creative"),
            "critic": InnerVoice("Critic", "critical"),
            "integrator": InnerVoice("Integrator", "holistic"),
        }
        
        self._dialogue_history: list[DialogueTurn] = []
        self._current_topic: str = ""
        
        logger.info("Inner Dialogue initialized", voices=len(self._voices))
    
    def start_dialogue(self, topic: str) -> None:
        """Start a new inner dialogue on a topic."""
        self._current_topic = topic
        self._add_turn(
            "integrator",
            f"Let's think about: {topic}",
            DialogueType.PLANNING
        )
    
    def deliberate(self, question: str) -> list[DialogueTurn]:
        """
        Deliberate on a question using multiple voices.
        
        Returns the dialogue turns.
        """
        turns = []
        
        # Each voice contributes
        analyst = self._add_turn(
            "analyst",
            f"Analyzing: {question}. Key factors to consider...",
            DialogueType.REASONING
        )
        turns.append(analyst)
        
        creative = self._add_turn(
            "creative",
            f"Alternative approach: What if we think about this differently?",
            DialogueType.REASONING
        )
        turns.append(creative)
        
        critic = self._add_turn(
            "critic",
            f"Potential issues: We should consider what could go wrong.",
            DialogueType.DEBATE,
            opposes=[analyst.id]
        )
        turns.append(critic)
        
        integrator = self._add_turn(
            "integrator",
            f"Synthesizing: Combining these perspectives...",
            DialogueType.REASONING,
            supports=[analyst.id, creative.id]
        )
        turns.append(integrator)
        
        return turns
    
    def self_correct(self, error: str, correction: str) -> DialogueTurn:
        """Generate self-correction dialogue."""
        turn = self._add_turn(
            "critic",
            f"I made an error: {error}. The correction is: {correction}",
            DialogueType.CORRECTION
        )
        
        # Add learning
        self._add_turn(
            "integrator",
            f"Learning from this: I should remember this for next time.",
            DialogueType.REFLECTION
        )
        
        return turn
    
    def build_narrative(self) -> str:
        """Build a narrative from recent dialogue."""
        if not self._dialogue_history:
            return "No dialogue yet."
        
        recent = self._dialogue_history[-5:]
        narrative = "Internal Narrative:\n"
        
        for turn in recent:
            narrative += f"  - {turn.content[:50]}...\n"
        
        return narrative
    
    def get_conclusion(self) -> str:
        """Get the current conclusion from dialogue."""
        integrator_turns = [
            t for t in self._dialogue_history
            if t.voice == "integrator"
        ]
        
        if integrator_turns:
            return integrator_turns[-1].content
        return "No conclusion yet."
    
    def _add_turn(
        self,
        voice: str,
        content: str,
        dialogue_type: DialogueType,
        supports: list[str] = None,
        opposes: list[str] = None,
    ) -> DialogueTurn:
        """Add a turn to the dialogue."""
        turn = DialogueTurn(
            voice=voice,
            content=content,
            dialogue_type=dialogue_type,
            supports=supports or [],
            opposes=opposes or [],
        )
        self._dialogue_history.append(turn)
        return turn
    
    def __len__(self) -> int:
        return len(self._dialogue_history)


class AttentionController:
    """
    Attention Control System.
    
    Manages what the AGI focuses on:
    - Priority-based attention
    - Distraction filtering
    - Focus maintenance
    """
    
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self._focus_stack: list[dict] = []
        self._attention_weights: dict[str, float] = {}
        
        logger.info("Attention Controller initialized")
    
    def focus_on(self, target: str, priority: float = 0.5) -> bool:
        """Focus attention on a target."""
        item = {
            "target": target,
            "priority": priority,
            "focused_at": datetime.now().isoformat(),
        }
        
        if len(self._focus_stack) >= self.capacity:
            # Remove lowest priority
            self._focus_stack.sort(key=lambda x: x["priority"])
            if priority > self._focus_stack[0]["priority"]:
                self._focus_stack.pop(0)
            else:
                return False
        
        self._focus_stack.append(item)
        self._attention_weights[target] = priority
        return True
    
    def get_focus(self) -> str | None:
        """Get current primary focus."""
        if not self._focus_stack:
            return None
        
        self._focus_stack.sort(key=lambda x: x["priority"], reverse=True)
        return self._focus_stack[0]["target"]
    
    def release_focus(self, target: str) -> bool:
        """Release focus from a target."""
        self._focus_stack = [
            f for f in self._focus_stack if f["target"] != target
        ]
        self._attention_weights.pop(target, None)
        return True
    
    def get_attention_distribution(self) -> dict[str, float]:
        """Get current attention distribution."""
        total = sum(self._attention_weights.values()) or 1
        return {k: v / total for k, v in self._attention_weights.items()}


class StreamOfConsciousness:
    """
    Stream of Consciousness Generator.
    
    Creates continuous internal experience.
    """
    
    def __init__(self, llm_pipeline=None):
        self._stream: list[str] = []
        self._llm = llm_pipeline
        self._running = False
        
        logger.info("Stream of Consciousness initialized")
    
    def generate(self, seed: str = "") -> str:
        """Generate a stream of consciousness segment."""
        if self._llm:
            from rwkv.utils import PIPELINE_ARGS
            args = PIPELINE_ARGS(temperature=0.9, top_p=0.95)
            
            prompt = f"Stream of consciousness: {seed}\nThinking: "
            output = self._llm.generate(prompt, token_count=50, args=args)
            
            self._stream.append(output)
            return output
        
        # Fallback without LLM
        thought = f"Thinking about {seed}..."
        self._stream.append(thought)
        return thought
    
    def get_recent(self, n: int = 5) -> list[str]:
        """Get recent stream segments."""
        return self._stream[-n:]
    
    def __len__(self) -> int:
        return len(self._stream)
