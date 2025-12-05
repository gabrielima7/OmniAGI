"""
Memory Consolidator - Sleep-inspired memory consolidation.

Replays and consolidates important experiences to prevent
catastrophic forgetting, similar to how the brain consolidates
memories during sleep.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from collections import deque

logger = structlog.get_logger()


@dataclass
class Memory:
    """A memory to be consolidated."""
    
    id: str
    content: str
    importance: float  # 0.0 - 1.0
    category: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 1
    consolidated: bool = False
    compressed_form: str | None = None
    links: list[str] = field(default_factory=list)  # Related memory IDs
    
    @property
    def age_hours(self) -> float:
        """Get age of this memory in hours."""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    @property
    def recency_score(self) -> float:
        """Score based on how recently accessed."""
        hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        return max(0, 1 - hours_since_access / 168)  # Decay over 1 week
    
    @property
    def priority_score(self) -> float:
        """Overall priority for consolidation."""
        return (
            0.4 * self.importance +
            0.3 * self.recency_score +
            0.2 * min(1, self.access_count / 10) +
            0.1 * (0 if self.consolidated else 1)
        )
    
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["last_accessed"] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


class MemoryConsolidator:
    """
    Consolidates memories to prevent catastrophic forgetting.
    
    Inspired by the brain's sleep-based memory consolidation:
    - Replays important memories
    - Compresses and abstracts information
    - Creates links between related memories
    - Prunes less important memories
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        max_memories: int = 10000,
        consolidation_threshold: float = 0.5,
    ):
        """
        Initialize consolidator.
        
        Args:
            storage_path: Path for persistent storage.
            max_memories: Maximum memories to retain.
            consolidation_threshold: Minimum priority for retention.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_memories = max_memories
        self.consolidation_threshold = consolidation_threshold
        
        self._memories: dict[str, Memory] = {}
        self._recent_queue: deque[str] = deque(maxlen=100)
        self._consolidation_count = 0
        
        if self.storage_path:
            self._load()
    
    def _load(self) -> None:
        """Load memories from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            for item in data:
                memory = Memory.from_dict(item)
                self._memories[memory.id] = memory
            
            logger.info("Memories loaded", count=len(self._memories))
        except Exception as e:
            logger.error("Failed to load memories", error=str(e))
    
    def _save(self) -> None:
        """Save memories to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [m.to_dict() for m in self._memories.values()]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add(
        self,
        content: str,
        importance: float = 0.5,
        category: str = "general",
        memory_id: str | None = None,
    ) -> Memory:
        """
        Add a new memory.
        
        Args:
            content: The memory content.
            importance: How important (0-1).
            category: Memory category.
            memory_id: Optional specific ID.
            
        Returns:
            The created Memory.
        """
        import uuid
        
        memory_id = memory_id or str(uuid.uuid4())
        
        memory = Memory(
            id=memory_id,
            content=content,
            importance=importance,
            category=category,
        )
        
        self._memories[memory_id] = memory
        self._recent_queue.append(memory_id)
        
        # Auto-consolidate if we're at capacity
        if len(self._memories) > self.max_memories:
            self.consolidate()
        
        self._save()
        return memory
    
    def access(self, memory_id: str) -> Memory | None:
        """Access a memory (updates access stats)."""
        memory = self._memories.get(memory_id)
        if memory:
            memory.last_accessed = datetime.now()
            memory.access_count += 1
            self._recent_queue.append(memory_id)
        return memory
    
    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        """Search memories by content."""
        query_lower = query.lower()
        results = []
        
        for memory in self._memories.values():
            if category and memory.category != category:
                continue
            
            if query_lower in memory.content.lower():
                results.append(memory)
        
        # Sort by priority
        results.sort(key=lambda m: m.priority_score, reverse=True)
        return results[:limit]
    
    def consolidate(self) -> dict[str, int]:
        """
        Perform memory consolidation.
        
        This is like a "sleep cycle" that:
        1. Replays important memories
        2. Compresses old memories
        3. Creates links between related memories
        4. Prunes low-priority memories
        
        Returns:
            Stats about the consolidation.
        """
        logger.info("Starting memory consolidation")
        self._consolidation_count += 1
        
        stats = {
            "replayed": 0,
            "compressed": 0,
            "linked": 0,
            "pruned": 0,
        }
        
        # 1. Replay - mark important recent as consolidated
        for memory_id in self._recent_queue:
            memory = self._memories.get(memory_id)
            if memory and memory.importance > 0.7:
                memory.consolidated = True
                stats["replayed"] += 1
        
        # 2. Compress old memories
        cutoff = datetime.now() - timedelta(days=7)
        for memory in self._memories.values():
            if memory.created_at < cutoff and not memory.compressed_form:
                memory.compressed_form = self._compress(memory.content)
                stats["compressed"] += 1
        
        # 3. Create links between related memories
        stats["linked"] = self._create_links()
        
        # 4. Prune low-priority memories
        if len(self._memories) > self.max_memories * 0.9:
            stats["pruned"] = self._prune()
        
        self._save()
        
        logger.info("Consolidation complete", **stats)
        return stats
    
    def _compress(self, content: str) -> str:
        """Compress memory content to essential form."""
        # Simple compression: first 200 chars + key sentences
        if len(content) <= 200:
            return content
        
        sentences = content.split(". ")
        key_sentences = sentences[:2] + sentences[-1:]
        return ". ".join(key_sentences)[:500]
    
    def _create_links(self) -> int:
        """Create links between related memories."""
        links_created = 0
        memories = list(self._memories.values())
        
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                # Simple keyword overlap
                if mem1.category == mem2.category:
                    words1 = set(mem1.content.lower().split())
                    words2 = set(mem2.content.lower().split())
                    overlap = len(words1 & words2)
                    
                    if overlap > 5 and mem2.id not in mem1.links:
                        mem1.links.append(mem2.id)
                        mem2.links.append(mem1.id)
                        links_created += 1
        
        return links_created
    
    def _prune(self) -> int:
        """Prune low-priority memories."""
        # Sort by priority
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.priority_score,
        )
        
        # Remove lowest priority until under threshold
        target = int(self.max_memories * 0.8)
        to_remove = len(self._memories) - target
        
        pruned = 0
        for memory in sorted_memories:
            if pruned >= to_remove:
                break
            if memory.priority_score < self.consolidation_threshold:
                del self._memories[memory.id]
                pruned += 1
        
        return pruned
    
    def get_context(self, limit: int = 5) -> str:
        """Get consolidated context for prompts."""
        # Get most important memories
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.priority_score,
            reverse=True,
        )[:limit]
        
        lines = ["## Conhecimento Consolidado"]
        for mem in sorted_memories:
            content = mem.compressed_form or mem.content[:200]
            lines.append(f"- [{mem.category}] {content}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self._memories)
