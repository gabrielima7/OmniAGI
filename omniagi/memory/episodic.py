"""
Episodic memory - Long-term experience storage.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = structlog.get_logger()


@dataclass
class Episode:
    """A recorded experience/episode."""
    
    id: str
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)
    category: str = "general"
    outcome: str = "neutral"  # positive, negative, neutral
    lessons: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class EpisodicMemory:
    """
    Long-term episodic memory for storing experiences.
    
    Persists episodes to disk for retrieval across sessions.
    """
    
    def __init__(self, storage_path: Path | str):
        """
        Initialize episodic memory.
        
        Args:
            storage_path: Path to store memory files.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._episodes: dict[str, Episode] = {}
        self._load_all()
    
    def _load_all(self) -> None:
        """Load all episodes from storage."""
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    episode = Episode.from_dict(data)
                    self._episodes[episode.id] = episode
            except Exception as e:
                logger.warning("Failed to load episode", file=str(file), error=str(e))
        
        logger.info("Loaded episodic memory", count=len(self._episodes))
    
    def _save_episode(self, episode: Episode) -> None:
        """Save an episode to disk."""
        file_path = self.storage_path / f"{episode.id}.json"
        with open(file_path, "w") as f:
            json.dump(episode.to_dict(), f, indent=2)
    
    def record(
        self,
        summary: str,
        category: str = "general",
        outcome: str = "neutral",
        lessons: list[str] | None = None,
        **context,
    ) -> Episode:
        """
        Record a new episode.
        
        Args:
            summary: What happened.
            category: Type of experience.
            outcome: How it turned out.
            lessons: Learnings from this experience.
            **context: Additional context data.
            
        Returns:
            The recorded Episode.
        """
        import uuid
        
        episode = Episode(
            id=str(uuid.uuid4()),
            summary=summary,
            category=category,
            outcome=outcome,
            lessons=lessons or [],
            context=context,
        )
        
        self._episodes[episode.id] = episode
        self._save_episode(episode)
        
        logger.info("Recorded episode", id=episode.id, category=category)
        return episode
    
    def get(self, episode_id: str) -> Episode | None:
        """Get an episode by ID."""
        return self._episodes.get(episode_id)
    
    def get_by_category(self, category: str) -> list[Episode]:
        """Get all episodes in a category."""
        return [
            ep for ep in self._episodes.values()
            if ep.category == category
        ]
    
    def get_recent(self, n: int = 10) -> list[Episode]:
        """Get the n most recent episodes."""
        sorted_episodes = sorted(
            self._episodes.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        return sorted_episodes[:n]
    
    def search(self, query: str) -> list[Episode]:
        """Search episodes by keyword."""
        query = query.lower()
        results = []
        for ep in self._episodes.values():
            if query in ep.summary.lower():
                results.append(ep)
            elif any(query in lesson.lower() for lesson in ep.lessons):
                results.append(ep)
        return results
    
    def get_lessons(self, category: str | None = None) -> list[str]:
        """Get all lessons, optionally filtered by category."""
        lessons = []
        for ep in self._episodes.values():
            if category is None or ep.category == category:
                lessons.extend(ep.lessons)
        return lessons
    
    def clear(self) -> None:
        """Clear all episodes (use with caution!)."""
        for file in self.storage_path.glob("*.json"):
            file.unlink()
        self._episodes.clear()
        logger.warning("Cleared episodic memory")
    
    def __len__(self) -> int:
        return len(self._episodes)
