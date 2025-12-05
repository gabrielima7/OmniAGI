"""
Working memory - Short-term context management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections import deque


@dataclass
class MemoryItem:
    """A single item in working memory."""
    
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def age_seconds(self) -> float:
        """Get age of this memory in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


class WorkingMemory:
    """
    Short-term working memory for the agent.
    
    Maintains a fixed-size buffer of recent context items,
    with optional importance-based retention.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Initialize working memory.
        
        Args:
            max_size: Maximum number of items to retain.
        """
        self.max_size = max_size
        self._items: deque[MemoryItem] = deque(maxlen=max_size)
    
    def add(
        self,
        content: str,
        importance: float = 0.5,
        **metadata,
    ) -> None:
        """Add an item to working memory."""
        item = MemoryItem(
            content=content,
            importance=importance,
            metadata=metadata,
        )
        self._items.append(item)
    
    def get_recent(self, n: int | None = None) -> list[MemoryItem]:
        """Get the n most recent items."""
        items = list(self._items)
        if n is not None:
            items = items[-n:]
        return items
    
    def get_important(self, threshold: float = 0.7) -> list[MemoryItem]:
        """Get items above importance threshold."""
        return [item for item in self._items if item.importance >= threshold]
    
    def search(self, query: str) -> list[MemoryItem]:
        """Simple keyword search in memory contents."""
        query = query.lower()
        return [
            item for item in self._items
            if query in item.content.lower()
        ]
    
    def clear(self) -> None:
        """Clear all items from working memory."""
        self._items.clear()
    
    def to_context_string(self) -> str:
        """Convert recent memory to context string for prompts."""
        if not self._items:
            return ""
        
        lines = ["## Recent Context"]
        for item in self._items:
            age = item.age_seconds()
            if age < 60:
                age_str = f"{int(age)}s ago"
            elif age < 3600:
                age_str = f"{int(age / 60)}m ago"
            else:
                age_str = f"{int(age / 3600)}h ago"
            
            lines.append(f"- [{age_str}] {item.content[:200]}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __bool__(self) -> bool:
        return len(self._items) > 0
