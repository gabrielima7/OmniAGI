"""
Long-Term Agenda - Managing goals over extended time periods.

Enables the AGI to maintain and pursue goals over days,
weeks, or longer with proper prioritization.
"""

from __future__ import annotations

import json
import logging

try:
    import structlog
except ImportError:
    structlog = None
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


class TimeHorizon(Enum):
    """Time horizon for agenda items."""
    
    IMMEDIATE = auto()   # Within minutes
    SHORT = auto()       # Today
    MEDIUM = auto()      # This week
    LONG = auto()        # This month
    EXTENDED = auto()    # Longer term


class AgendaStatus(Enum):
    """Status of agenda items."""
    
    PENDING = auto()
    SCHEDULED = auto()
    IN_PROGRESS = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    DEFERRED = auto()


@dataclass
class AgendaItem:
    """A single item in the long-term agenda."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    title: str = ""
    description: str = ""
    
    # Timing
    horizon: TimeHorizon = TimeHorizon.MEDIUM
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    due_date: str | None = None
    scheduled_for: str | None = None
    
    # Priority and status
    priority: float = 0.5  # 0-1
    status: AgendaStatus = AgendaStatus.PENDING
    
    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    
    # Progress
    progress: float = 0.0
    sub_items: list[str] = field(default_factory=list)
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    goal_id: str | None = None  # Link to autonomous goal
    
    def is_actionable(self) -> bool:
        """Check if item can be worked on now."""
        if self.status in [AgendaStatus.COMPLETED, AgendaStatus.BLOCKED]:
            return False
        if self.depends_on:
            return False  # Has unmet dependencies
        return True
    
    def is_overdue(self) -> bool:
        """Check if past due date."""
        if not self.due_date:
            return False
        return datetime.now().isoformat() > self.due_date
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "horizon": self.horizon.name,
            "created_at": self.created_at,
            "due_date": self.due_date,
            "scheduled_for": self.scheduled_for,
            "priority": self.priority,
            "status": self.status.name,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "progress": self.progress,
            "sub_items": self.sub_items,
            "tags": self.tags,
            "goal_id": self.goal_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgendaItem":
        return cls(
            id=data.get("id", str(uuid4())[:8]),
            title=data["title"],
            description=data.get("description", ""),
            horizon=TimeHorizon[data.get("horizon", "MEDIUM")],
            created_at=data.get("created_at", datetime.now().isoformat()),
            due_date=data.get("due_date"),
            scheduled_for=data.get("scheduled_for"),
            priority=data.get("priority", 0.5),
            status=AgendaStatus[data.get("status", "PENDING")],
            depends_on=data.get("depends_on", []),
            blocks=data.get("blocks", []),
            progress=data.get("progress", 0.0),
            sub_items=data.get("sub_items", []),
            tags=data.get("tags", []),
            goal_id=data.get("goal_id"),
        )


class LongTermAgenda:
    """
    Long-term agenda for managing goals across time.
    
    Organizes work into time horizons and manages
    dependencies between items.
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._items: dict[str, AgendaItem] = {}
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Long-Term Agenda initialized", items=len(self._items))
    
    def add(
        self,
        title: str,
        description: str = "",
        horizon: TimeHorizon = TimeHorizon.MEDIUM,
        priority: float = 0.5,
        due_date: str = None,
        depends_on: list[str] = None,
        tags: list[str] = None,
        goal_id: str = None,
    ) -> AgendaItem:
        """Add an item to the agenda."""
        item = AgendaItem(
            title=title,
            description=description,
            horizon=horizon,
            priority=priority,
            due_date=due_date,
            depends_on=depends_on or [],
            tags=tags or [],
            goal_id=goal_id,
        )
        
        # Update blocks for dependencies
        for dep_id in item.depends_on:
            if dep_id in self._items:
                self._items[dep_id].blocks.append(item.id)
        
        self._items[item.id] = item
        self._save()
        
        logger.info("Agenda item added", id=item.id, title=title[:30])
        return item
    
    def remove(self, item_id: str) -> bool:
        """Remove an item from the agenda."""
        if item_id not in self._items:
            return False
        
        item = self._items[item_id]
        
        # Remove from dependencies
        for dep_id in item.depends_on:
            if dep_id in self._items:
                self._items[dep_id].blocks.remove(item_id)
        
        # Unblock dependent items
        for blocked_id in item.blocks:
            if blocked_id in self._items:
                self._items[blocked_id].depends_on.remove(item_id)
        
        del self._items[item_id]
        self._save()
        return True
    
    def schedule(
        self,
        item_id: str,
        scheduled_for: str | datetime,
    ) -> bool:
        """Schedule an item for a specific time."""
        if item_id not in self._items:
            return False
        
        if isinstance(scheduled_for, datetime):
            scheduled_for = scheduled_for.isoformat()
        
        self._items[item_id].scheduled_for = scheduled_for
        self._items[item_id].status = AgendaStatus.SCHEDULED
        self._save()
        return True
    
    def start(self, item_id: str) -> bool:
        """Start working on an item."""
        if item_id not in self._items:
            return False
        
        item = self._items[item_id]
        if not item.is_actionable():
            return False
        
        item.status = AgendaStatus.IN_PROGRESS
        self._save()
        return True
    
    def complete(self, item_id: str) -> bool:
        """Mark an item as completed."""
        if item_id not in self._items:
            return False
        
        item = self._items[item_id]
        item.status = AgendaStatus.COMPLETED
        item.progress = 1.0
        
        # Unblock dependent items
        for blocked_id in item.blocks:
            if blocked_id in self._items:
                self._items[blocked_id].depends_on.remove(item_id)
        
        self._save()
        logger.info("Agenda item completed", id=item_id)
        return True
    
    def update_progress(self, item_id: str, progress: float) -> bool:
        """Update progress on an item."""
        if item_id not in self._items:
            return False
        
        self._items[item_id].progress = max(0.0, min(1.0, progress))
        
        if progress >= 1.0:
            return self.complete(item_id)
        
        self._save()
        return True
    
    def defer(self, item_id: str, new_horizon: TimeHorizon = None) -> bool:
        """Defer an item to later."""
        if item_id not in self._items:
            return False
        
        item = self._items[item_id]
        item.status = AgendaStatus.DEFERRED
        if new_horizon:
            item.horizon = new_horizon
        
        self._save()
        return True
    
    def get_actionable(self, limit: int = 10) -> list[AgendaItem]:
        """Get actionable items sorted by priority."""
        actionable = [
            item for item in self._items.values()
            if item.is_actionable()
        ]
        
        # Sort by priority and overdue status
        actionable.sort(
            key=lambda x: (x.is_overdue(), x.priority),
            reverse=True,
        )
        
        return actionable[:limit]
    
    def get_by_horizon(self, horizon: TimeHorizon) -> list[AgendaItem]:
        """Get items for a specific time horizon."""
        return [
            item for item in self._items.values()
            if item.horizon == horizon and item.status != AgendaStatus.COMPLETED
        ]
    
    def get_overdue(self) -> list[AgendaItem]:
        """Get overdue items."""
        return [
            item for item in self._items.values()
            if item.is_overdue() and item.status != AgendaStatus.COMPLETED
        ]
    
    def get_scheduled_today(self) -> list[AgendaItem]:
        """Get items scheduled for today."""
        today = datetime.now().date().isoformat()
        return [
            item for item in self._items.values()
            if item.scheduled_for and item.scheduled_for.startswith(today)
        ]
    
    def prioritize(self) -> list[AgendaItem]:
        """
        Get prioritized list of all non-complete items.
        
        Priority factors:
        1. Overdue items first
        2. Then by horizon (immediate > short > medium > long)
        3. Then by priority score
        4. Then by dependencies (items that unblock others)
        """
        active = [
            item for item in self._items.values()
            if item.status not in [AgendaStatus.COMPLETED, AgendaStatus.DEFERRED]
        ]
        
        def priority_key(item: AgendaItem) -> tuple:
            return (
                item.is_overdue(),
                -item.horizon.value,  # Lower horizon value = more urgent
                len(item.blocks),     # Unblocks more = higher priority
                item.priority,
            )
        
        active.sort(key=priority_key, reverse=True)
        return active
    
    def get_next_action(self) -> AgendaItem | None:
        """Get the single most important actionable item."""
        actionable = self.get_actionable(1)
        return actionable[0] if actionable else None
    
    def get_summary(self) -> dict:
        """Get agenda summary."""
        by_status = {s.name: 0 for s in AgendaStatus}
        by_horizon = {h.name: 0 for h in TimeHorizon}
        
        for item in self._items.values():
            by_status[item.status.name] += 1
            if item.status != AgendaStatus.COMPLETED:
                by_horizon[item.horizon.name] += 1
        
        return {
            "total_items": len(self._items),
            "by_status": by_status,
            "by_horizon": by_horizon,
            "overdue": len(self.get_overdue()),
            "actionable": len(self.get_actionable()),
            "scheduled_today": len(self.get_scheduled_today()),
        }
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "items": {k: v.to_dict() for k, v in self._items.items()},
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        self._items = {
            k: AgendaItem.from_dict(v)
            for k, v in data.get("items", {}).items()
        }
