"""
Persistent Memory System.

Implements long-term memory storage for AGI experiences,
knowledge, and learned patterns.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    memory_type: str  # "episodic", "semantic", "procedural"
    importance: float  # 0.0 - 1.0
    created_at: str
    last_accessed: str
    access_count: int = 0
    metadata: Dict[str, Any] = None
    embedding: List[float] = None


class PersistentMemory:
    """
    SQLite-based persistent memory system.
    
    Stores memories that persist across sessions.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.home() / ".omniagi" / "memory.db")
        
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    embedding TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type 
                ON memories(memory_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance 
                ON memories(importance DESC)
            """)
            
            conn.commit()
    
    def store(self, memory: Memory) -> bool:
        """Store a memory."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, content, memory_type, importance, created_at, 
                     last_accessed, access_count, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    memory.content,
                    memory.memory_type,
                    memory.importance,
                    memory.created_at,
                    memory.last_accessed,
                    memory.access_count,
                    json.dumps(memory.metadata) if memory.metadata else None,
                    json.dumps(memory.embedding) if memory.embedding else None,
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", 
                    (memory_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Update access info
                    conn.execute("""
                        UPDATE memories 
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE id = ?
                    """, (datetime.now().isoformat(), memory_id))
                    conn.commit()
                    
                    return self._row_to_memory(row)
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return None
    
    def search(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Search memories by content."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if memory_type:
                    cursor = conn.execute("""
                        SELECT * FROM memories 
                        WHERE content LIKE ? AND memory_type = ?
                        ORDER BY importance DESC, access_count DESC
                        LIMIT ?
                    """, (f"%{query}%", memory_type, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM memories 
                        WHERE content LIKE ?
                        ORDER BY importance DESC, access_count DESC
                        LIMIT ?
                    """, (f"%{query}%", limit))
                
                return [self._row_to_memory(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def get_recent(self, limit: int = 10) -> List[Memory]:
        """Get most recent memories."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM memories 
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                return [self._row_to_memory(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get recent: {e}")
            return []
    
    def get_important(self, limit: int = 10) -> List[Memory]:
        """Get most important memories."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM memories 
                    ORDER BY importance DESC
                    LIMIT ?
                """, (limit,))
                return [self._row_to_memory(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get important: {e}")
            return []
    
    def consolidate(self, min_access: int = 3):
        """Consolidate memories (increase importance of frequently accessed)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Increase importance of frequently accessed memories
                conn.execute("""
                    UPDATE memories 
                    SET importance = MIN(1.0, importance + 0.1)
                    WHERE access_count >= ?
                """, (min_access,))
                
                # Decay rarely accessed memories
                conn.execute("""
                    UPDATE memories 
                    SET importance = MAX(0.1, importance - 0.05)
                    WHERE access_count < ?
                    AND importance > 0.3
                """, (min_access,))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to consolidate: {e}")
    
    def forget(self, threshold: float = 0.1) -> int:
        """Forget low-importance memories."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE importance < ?",
                    (threshold,)
                )
                count = cursor.rowcount
                conn.commit()
                return count
        except Exception as e:
            logger.error(f"Failed to forget: {e}")
            return 0
    
    def count(self) -> int:
        """Count total memories."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memories")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count: {e}")
            return 0
    
    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            importance=row[3],
            created_at=row[4],
            last_accessed=row[5],
            access_count=row[6],
            metadata=json.loads(row[7]) if row[7] else None,
            embedding=json.loads(row[8]) if row[8] else None,
        )


class WorkingMemory:
    """
    Short-term working memory.
    
    Maintains current context and recent information.
    """
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[Dict[str, Any]] = []
        self.focus: Optional[Dict[str, Any]] = None
    
    def add(self, item: Dict[str, Any]) -> None:
        """Add item to working memory."""
        self.items.append(item)
        
        # Maintain capacity (Miller's 7±2)
        while len(self.items) > self.capacity:
            self.items.pop(0)
    
    def get_context(self) -> List[Dict[str, Any]]:
        """Get current context."""
        return self.items.copy()
    
    def set_focus(self, item: Dict[str, Any]) -> None:
        """Set current focus."""
        self.focus = item
    
    def get_focus(self) -> Optional[Dict[str, Any]]:
        """Get current focus."""
        return self.focus
    
    def clear(self) -> None:
        """Clear working memory."""
        self.items = []
        self.focus = None


class MemorySystem:
    """
    Complete memory system integrating short and long-term memory.
    """
    
    def __init__(self, db_path: str = None):
        self.persistent = PersistentMemory(db_path)
        self.working = WorkingMemory()
        self._counter = 0
    
    def remember(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Store a new memory."""
        self._counter += 1
        now = datetime.now().isoformat()
        
        memory = Memory(
            id=f"mem_{self._counter}_{now[:10]}",
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=now,
            last_accessed=now,
            access_count=1,
            metadata=metadata,
        )
        
        self.persistent.store(memory)
        
        # Also add to working memory
        self.working.add({
            "id": memory.id,
            "content": content,
            "type": memory_type,
        })
        
        return memory.id
    
    def recall(self, query: str, limit: int = 5) -> List[Memory]:
        """Recall memories matching query."""
        return self.persistent.search(query, limit=limit)
    
    def get_context(self) -> Dict[str, Any]:
        """Get current memory context."""
        return {
            "working_memory": self.working.get_context(),
            "focus": self.working.get_focus(),
            "total_memories": self.persistent.count(),
            "recent": [m.content for m in self.persistent.get_recent(3)],
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_memories": self.persistent.count(),
            "working_memory_items": len(self.working.items),
            "db_path": self.persistent.db_path,
        }
