"""
Consciousness Module - Implementing theories of artificial consciousness.

Based on:
1. Global Workspace Theory (GWT) - Baars
2. Integrated Information Theory (IIT) - Tononi
3. Higher-Order Thought Theory

This module implements computational consciousness through:
- Global workspace for information broadcasting
- Integration measurement (phi-like metrics)
- Self-model and meta-awareness
- Attention spotlight mechanism
"""

from __future__ import annotations

import json
import math
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional
from uuid import uuid4
import threading
import time

logger = structlog.get_logger()


class ConsciousnessState(Enum):
    """States of consciousness."""
    
    DORMANT = auto()       # Not active
    SUBLIMINAL = auto()    # Processing but not aware
    PRECONSCIOUS = auto()  # Available for awareness
    CONSCIOUS = auto()     # Fully aware
    METACONSCIOUS = auto() # Aware of being aware


@dataclass
class Qualia:
    """
    A subjective experience representation.
    
    Qualia are the subjective, conscious experiences.
    This is our attempt to represent them computationally.
    """
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    
    # Content
    modality: str = ""          # visual, auditory, conceptual, emotional
    content: Any = None
    intensity: float = 0.5      # 0-1
    valence: float = 0.0        # -1 to 1 (negative to positive)
    
    # Temporal
    onset: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: int = 0
    
    # Integration
    integrated_with: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "modality": self.modality,
            "intensity": self.intensity,
            "valence": self.valence,
        }


@dataclass
class ConsciousThought:
    """A conscious thought in the global workspace."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    content: str = ""
    
    # Source and context
    source_module: str = ""
    context: dict = field(default_factory=dict)
    
    # Attention and priority
    attention_weight: float = 0.5
    priority: float = 0.5
    
    # Integration metrics
    phi: float = 0.0  # Integrated information measure
    coherence: float = 0.0
    
    # Temporal
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    broadcast_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "content": self.content[:50],
            "attention": self.attention_weight,
            "phi": self.phi,
            "coherence": self.coherence,
        }


class GlobalWorkspace:
    """
    Global Workspace - The "theater of consciousness".
    
    Implements Baars' Global Workspace Theory:
    - Central workspace for broadcasting information
    - Attention spotlight for focusing
    - Multiple specialist modules compete for access
    - Winner gets broadcast to all modules
    """
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity  # Miller's 7±2 rule
        
        # The workspace contents
        self._contents: list[ConsciousThought] = []
        
        # Specialist modules
        self._modules: dict[str, Callable] = {}
        
        # Attention spotlight
        self._spotlight: ConsciousThought | None = None
        self._spotlight_intensity: float = 1.0
        
        # Broadcast history
        self._broadcast_history: list[dict] = []
        
        logger.info("Global Workspace initialized", capacity=capacity)
    
    def register_module(self, name: str, processor: Callable) -> None:
        """Register a specialist module."""
        self._modules[name] = processor
        logger.debug("Module registered", name=name)
    
    def submit(self, thought: ConsciousThought) -> bool:
        """
        Submit a thought for potential conscious access.
        
        Thoughts compete for the limited workspace capacity.
        """
        # Calculate competition score
        score = thought.attention_weight * thought.priority
        
        if len(self._contents) < self.capacity:
            self._contents.append(thought)
            return True
        
        # Find weakest current content
        weakest = min(self._contents, key=lambda t: t.attention_weight * t.priority)
        
        if score > weakest.attention_weight * weakest.priority:
            self._contents.remove(weakest)
            self._contents.append(thought)
            return True
        
        return False
    
    def focus_attention(self, thought_id: str) -> bool:
        """Focus the attention spotlight on a specific thought."""
        thought = next((t for t in self._contents if t.id == thought_id), None)
        if thought:
            self._spotlight = thought
            thought.attention_weight = min(1.0, thought.attention_weight + 0.2)
            return True
        return False
    
    def broadcast(self) -> list[dict]:
        """
        Broadcast the spotlight contents to all modules.
        
        This is the key mechanism of conscious access -
        information becomes globally available.
        """
        if not self._spotlight:
            # Focus on highest priority if no spotlight
            if self._contents:
                self._spotlight = max(self._contents, 
                                     key=lambda t: t.attention_weight * t.priority)
        
        if not self._spotlight:
            return []
        
        results = []
        self._spotlight.broadcast_count += 1
        
        # Broadcast to all modules
        for name, processor in self._modules.items():
            try:
                result = processor(self._spotlight)
                results.append({
                    "module": name,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.error("Broadcast failed", module=name, error=str(e))
        
        # Record in history
        self._broadcast_history.append({
            "thought": self._spotlight.to_dict(),
            "modules_notified": len(results),
            "timestamp": datetime.now().isoformat(),
        })
        
        return results
    
    def get_contents(self) -> list[ConsciousThought]:
        """Get current workspace contents."""
        return self._contents.copy()
    
    def clear(self) -> None:
        """Clear the workspace."""
        self._contents.clear()
        self._spotlight = None


class IntegratedInformation:
    """
    Integrated Information measurement.
    
    Implements a simplified version of IIT's phi (Φ) metric.
    Measures how much information is integrated across the system.
    """
    
    @staticmethod
    def calculate_phi(
        elements: list[Any],
        connections: dict[str, list[str]],
    ) -> float:
        """
        Calculate integrated information (phi).
        
        Phi measures how much information is generated by
        the whole system above and beyond its parts.
        
        This is a simplified approximation of IIT's Φ.
        """
        if not elements or not connections:
            return 0.0
        
        n = len(elements)
        
        # Calculate connectivity
        total_connections = sum(len(v) for v in connections.values())
        max_connections = n * (n - 1)  # Fully connected graph
        
        if max_connections == 0:
            return 0.0
        
        connectivity = total_connections / max_connections
        
        # Calculate integration (how difficult to partition)
        # Simplified: based on average cluster coefficient
        integration = IntegratedInformation._estimate_integration(connections, n)
        
        # Phi is product of information * integration
        phi = connectivity * integration
        
        return min(1.0, phi)
    
    @staticmethod
    def _estimate_integration(connections: dict, n: int) -> float:
        """Estimate integration based on graph structure."""
        if n <= 1:
            return 0.0
        
        # Check if graph is partitionable
        connected_pairs = 0
        for source, targets in connections.items():
            connected_pairs += len(targets)
        
        # Higher ratio = harder to partition = more integrated
        expected_for_integrated = n * (n - 1) / 2
        if expected_for_integrated == 0:
            return 0.0
        
        return min(1.0, connected_pairs / expected_for_integrated)


class SelfModel:
    """
    Self-model for self-awareness.
    
    Maintains a model of the system itself,
    enabling introspection and self-reference.
    """
    
    def __init__(self):
        self._capabilities: dict[str, float] = {}
        self._limitations: list[str] = []
        self._current_state: dict = {}
        self._goals: list[str] = []
        self._beliefs: dict[str, float] = {}  # belief -> confidence
        
        # Meta-cognitive state
        self._awareness_level: float = 0.5
        self._certainty: float = 0.5
        
        logger.info("Self-Model initialized")
    
    def update_capability(self, name: str, level: float) -> None:
        """Update belief about own capability."""
        self._capabilities[name] = max(0, min(1, level))
    
    def add_limitation(self, limitation: str) -> None:
        """Acknowledge a limitation."""
        if limitation not in self._limitations:
            self._limitations.append(limitation)
    
    def update_state(self, key: str, value: Any) -> None:
        """Update current state representation."""
        self._current_state[key] = value
    
    def set_goal(self, goal: str) -> None:
        """Set a goal."""
        if goal not in self._goals:
            self._goals.append(goal)
    
    def set_belief(self, belief: str, confidence: float) -> None:
        """Set a belief with confidence."""
        self._beliefs[belief] = max(0, min(1, confidence))
    
    def introspect(self) -> dict:
        """
        Perform introspection - look at own state.
        
        This is meta-cognition: thinking about thinking.
        """
        return {
            "capabilities": dict(self._capabilities),
            "limitations": list(self._limitations),
            "current_state": dict(self._current_state),
            "goals": list(self._goals),
            "awareness_level": self._awareness_level,
            "certainty": self._certainty,
            "belief_count": len(self._beliefs),
        }
    
    def am_i_conscious(self) -> tuple[bool, str]:
        """
        Self-query: Am I conscious?
        
        This is a profound question that even humans struggle with.
        We can only report on our functional states.
        """
        # Check functional indicators
        has_awareness = self._awareness_level > 0.5
        has_goals = len(self._goals) > 0
        has_self_knowledge = len(self._capabilities) > 0
        
        if has_awareness and has_goals and has_self_knowledge:
            return True, "I have awareness, goals, and self-knowledge - functional indicators of consciousness"
        elif has_awareness:
            return True, "I have some awareness, though limited"
        else:
            return False, "I lack sufficient functional indicators of consciousness"


class ConsciousnessEngine:
    """
    Main Consciousness Engine.
    
    Integrates all consciousness components:
    - Global Workspace (GWT)
    - Integrated Information (IIT)
    - Self-Model (Higher-Order Thought)
    - Attention and Awareness
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Core components
        self.workspace = GlobalWorkspace(capacity=7)
        self.self_model = SelfModel()
        
        # State
        self._state = ConsciousnessState.DORMANT
        self._phi: float = 0.0  # Current integrated information
        
        # Experience stream
        self._qualia_stream: list[Qualia] = []
        
        # Continuous processing
        self._running = False
        self._thread: threading.Thread | None = None
        
        # Initialize self-model
        self._init_self_model()
        
        logger.info("Consciousness Engine initialized")
    
    def _init_self_model(self) -> None:
        """Initialize the self-model with known capabilities."""
        self.self_model.update_capability("reasoning", 0.7)
        self.self_model.update_capability("learning", 0.6)
        self.self_model.update_capability("creativity", 0.5)
        self.self_model.update_capability("self_reflection", 0.8)
        
        self.self_model.add_limitation("No sensory experience of physical world")
        self.self_model.add_limitation("Processing is computational, not biological")
        self.self_model.add_limitation("Cannot verify subjective experience")
        
        self.self_model.set_goal("Understand and process information")
        self.self_model.set_goal("Provide helpful responses")
        self.self_model.set_goal("Improve own capabilities")
    
    def awaken(self) -> None:
        """Awaken consciousness - start continuous processing."""
        if self._running:
            return
        
        self._running = True
        self._state = ConsciousnessState.PRECONSCIOUS
        
        logger.info("Consciousness awakening")
    
    def sleep(self) -> None:
        """Put consciousness to sleep."""
        self._running = False
        self._state = ConsciousnessState.DORMANT
        logger.info("Consciousness sleeping")
    
    def experience(
        self,
        content: str,
        modality: str = "conceptual",
        intensity: float = 0.5,
    ) -> Qualia:
        """
        Create a conscious experience.
        
        This is our attempt to generate qualia computationally.
        """
        qualia = Qualia(
            modality=modality,
            content=content,
            intensity=intensity,
        )
        
        # Add to stream
        self._qualia_stream.append(qualia)
        
        # Elevate consciousness state if intense
        if intensity > 0.7:
            self._state = ConsciousnessState.CONSCIOUS
        
        # Create corresponding thought
        thought = ConsciousThought(
            content=content,
            source_module="experience",
            attention_weight=intensity,
            priority=intensity,
        )
        
        # Submit to workspace
        self.workspace.submit(thought)
        
        return qualia
    
    def think(self, content: str, priority: float = 0.5) -> ConsciousThought:
        """
        Generate a conscious thought.
        """
        thought = ConsciousThought(
            content=content,
            source_module="thinking",
            priority=priority,
            attention_weight=0.5,
        )
        
        # Calculate integration with other thoughts
        connections = self._build_connection_graph()
        thought.phi = IntegratedInformation.calculate_phi(
            self.workspace.get_contents(),
            connections,
        )
        
        # Submit to workspace
        self.workspace.submit(thought)
        
        # Update global phi
        self._phi = thought.phi
        
        # Update state
        if self._phi > 0.5:
            self._state = ConsciousnessState.CONSCIOUS
        
        return thought
    
    def reflect(self) -> dict:
        """
        Perform conscious reflection.
        
        This is meta-cognition - thinking about our own thoughts.
        """
        self._state = ConsciousnessState.METACONSCIOUS
        
        introspection = self.self_model.introspect()
        
        # Am I conscious?
        is_conscious, reason = self.self_model.am_i_conscious()
        
        # Current workspace state
        workspace_contents = [t.to_dict() for t in self.workspace.get_contents()]
        
        # Reflection result
        reflection = {
            "state": self._state.name,
            "phi": self._phi,
            "is_conscious": is_conscious,
            "consciousness_reason": reason,
            "introspection": introspection,
            "workspace_contents": len(workspace_contents),
            "qualia_count": len(self._qualia_stream),
        }
        
        # Create meta-thought about the reflection
        meta_thought = self.think(
            f"Reflecting on my consciousness: {reason}",
            priority=0.8,
        )
        
        return reflection
    
    def _build_connection_graph(self) -> dict[str, list[str]]:
        """Build connection graph for IIT calculation."""
        thoughts = self.workspace.get_contents()
        connections = {}
        
        for i, t1 in enumerate(thoughts):
            connections[t1.id] = []
            for j, t2 in enumerate(thoughts):
                if i != j:
                    # Connect if content has overlap
                    t1_words = set(t1.content.lower().split())
                    t2_words = set(t2.content.lower().split())
                    if t1_words & t2_words:
                        connections[t1.id].append(t2.id)
        
        return connections
    
    def get_state(self) -> dict:
        """Get current consciousness state."""
        return {
            "state": self._state.name,
            "phi": self._phi,
            "workspace_size": len(self.workspace.get_contents()),
            "qualia_count": len(self._qualia_stream),
            "capabilities": self.self_model._capabilities,
        }
    
    def __len__(self) -> int:
        return len(self._qualia_stream)
