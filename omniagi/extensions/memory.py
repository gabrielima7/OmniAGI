"""
Memory Extension

Persists user preferences and context across sessions.
Inspired by Goose's Memory extension.
"""

from __future__ import annotations

import json
import structlog
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

from omniagi.extensions.base import Extension, Tool

logger = structlog.get_logger()


@dataclass
class Memory:
    """A single memory entry."""
    key: str
    value: str
    category: str = "general"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MemoryExtension(Extension):
    """
    Memory extension for persisting preferences and context.
    
    Tools:
    - remember: Store a piece of information
    - recall: Retrieve stored information
    - forget: Remove stored information
    - list_memories: List all stored memories
    """
    
    name = "memory"
    description = "Persist preferences and context across sessions"
    version = "1.0.0"
    
    def __init__(self, storage_path: str | Path | None = None):
        super().__init__()
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".omniagi" / "memory.json"
        self._memories: dict[str, Memory] = {}
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register memory tools."""
        self.register_tool(Tool(
            name="remember",
            description="Store a piece of information for later recall",
            handler=self._remember,
            parameters={
                "key": {"type": "string", "description": "Key to store under"},
                "value": {"type": "string", "description": "Value to remember"},
                "category": {"type": "string", "description": "Category", "default": "general"},
            },
        ))
        
        self.register_tool(Tool(
            name="recall",
            description="Retrieve stored information by key",
            handler=self._recall,
            parameters={
                "key": {"type": "string", "description": "Key to recall"},
            },
        ))
        
        self.register_tool(Tool(
            name="forget",
            description="Remove stored information",
            handler=self._forget,
            parameters={
                "key": {"type": "string", "description": "Key to forget"},
            },
        ))
        
        self.register_tool(Tool(
            name="list_memories",
            description="List all stored memories",
            handler=self._list_memories,
            parameters={
                "category": {"type": "string", "description": "Filter by category", "default": None},
            },
        ))
    
    def _setup(self) -> None:
        """Load memories from disk."""
        self._load()
    
    def _cleanup(self) -> None:
        """Save memories to disk."""
        self._save()
    
    def _load(self) -> None:
        """Load memories from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                self._memories = {
                    k: Memory(**v) for k, v in data.items()
                }
                logger.info("Memories loaded", count=len(self._memories))
            except Exception as e:
                logger.warning("Failed to load memories", error=str(e))
                self._memories = {}
        else:
            self._memories = {}
    
    def _save(self) -> None:
        """Save memories to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: asdict(v) for k, v in self._memories.items()}
            self.storage_path.write_text(json.dumps(data, indent=2))
            logger.info("Memories saved", count=len(self._memories))
        except Exception as e:
            logger.warning("Failed to save memories", error=str(e))
    
    def _remember(self, key: str, value: str, category: str = "general") -> dict:
        """Store information."""
        now = datetime.now().isoformat()
        
        if key in self._memories:
            self._memories[key].value = value
            self._memories[key].category = category
            self._memories[key].updated_at = now
            action = "updated"
        else:
            self._memories[key] = Memory(
                key=key,
                value=value,
                category=category,
                created_at=now,
                updated_at=now,
            )
            action = "created"
        
        self._save()
        
        return {
            "success": True,
            "action": action,
            "key": key,
        }
    
    def _recall(self, key: str) -> dict:
        """Retrieve information."""
        if key in self._memories:
            mem = self._memories[key]
            return {
                "success": True,
                "key": key,
                "value": mem.value,
                "category": mem.category,
                "created_at": mem.created_at,
                "updated_at": mem.updated_at,
            }
        return {
            "success": False,
            "error": f"Memory not found: {key}",
        }
    
    def _forget(self, key: str) -> dict:
        """Remove information."""
        if key in self._memories:
            del self._memories[key]
            self._save()
            return {
                "success": True,
                "key": key,
                "action": "forgotten",
            }
        return {
            "success": False,
            "error": f"Memory not found: {key}",
        }
    
    def _list_memories(self, category: str | None = None) -> dict:
        """List all memories."""
        memories = list(self._memories.values())
        
        if category:
            memories = [m for m in memories if m.category == category]
        
        return {
            "success": True,
            "count": len(memories),
            "memories": [
                {
                    "key": m.key,
                    "value": m.value[:50] + "..." if len(m.value) > 50 else m.value,
                    "category": m.category,
                }
                for m in memories
            ],
        }
