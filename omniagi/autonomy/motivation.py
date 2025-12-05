"""
Motivation System - Intrinsic drives for autonomous behavior.

Implements a drive-based motivation system that provides
intrinsic rewards independent of external feedback.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = structlog.get_logger()


class DriveType(Enum):
    """Types of intrinsic drives."""
    
    CURIOSITY = auto()      # Desire to learn new things
    COMPETENCE = auto()     # Desire to master skills
    AUTONOMY = auto()       # Desire for independence
    RELATEDNESS = auto()    # Desire for connection (helping)
    MEANING = auto()        # Desire for purpose
    HOMEOSTASIS = auto()    # Desire for balance


@dataclass
class Drive:
    """A single motivational drive."""
    
    drive_type: DriveType
    current_level: float = 0.5      # 0-1, current satisfaction
    target_level: float = 0.7       # 0-1, optimal level
    weight: float = 1.0             # Importance weight
    decay_rate: float = 0.01        # How fast it depletes
    last_satisfied: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def deficit(self) -> float:
        """Calculate deficit from target."""
        return max(0.0, self.target_level - self.current_level)
    
    @property
    def urgency(self) -> float:
        """Calculate urgency (weighted deficit)."""
        return self.deficit * self.weight
    
    def satisfy(self, amount: float) -> None:
        """Satisfy the drive by some amount."""
        self.current_level = min(1.0, self.current_level + amount)
        self.last_satisfied = datetime.now().isoformat()
    
    def decay(self) -> None:
        """Apply natural decay to the drive."""
        self.current_level = max(0.0, self.current_level - self.decay_rate)
    
    def to_dict(self) -> dict:
        return {
            "type": self.drive_type.name,
            "current": self.current_level,
            "target": self.target_level,
            "weight": self.weight,
            "decay_rate": self.decay_rate,
            "last_satisfied": self.last_satisfied,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Drive":
        return cls(
            drive_type=DriveType[data["type"]],
            current_level=data.get("current", 0.5),
            target_level=data.get("target", 0.7),
            weight=data.get("weight", 1.0),
            decay_rate=data.get("decay_rate", 0.01),
            last_satisfied=data.get("last_satisfied", datetime.now().isoformat()),
        )


@dataclass
class MotivationalState:
    """Overall motivational state."""
    
    total_motivation: float = 0.0
    dominant_drive: DriveType | None = None
    drive_levels: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MotivationSystem:
    """
    Intrinsic motivation system for autonomous behavior.
    
    Based on Self-Determination Theory drives:
    - Curiosity (exploration, learning)
    - Competence (mastery, improvement)
    - Autonomy (self-direction)
    - Relatedness (helping, connection)
    - Meaning (purpose, impact)
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Initialize drives
        self._drives: dict[DriveType, Drive] = {
            DriveType.CURIOSITY: Drive(
                DriveType.CURIOSITY,
                weight=1.2,
                decay_rate=0.02,
            ),
            DriveType.COMPETENCE: Drive(
                DriveType.COMPETENCE,
                weight=1.0,
                decay_rate=0.01,
            ),
            DriveType.AUTONOMY: Drive(
                DriveType.AUTONOMY,
                weight=0.8,
                decay_rate=0.005,
            ),
            DriveType.RELATEDNESS: Drive(
                DriveType.RELATEDNESS,
                weight=0.9,
                decay_rate=0.015,
            ),
            DriveType.MEANING: Drive(
                DriveType.MEANING,
                weight=1.1,
                decay_rate=0.008,
            ),
            DriveType.HOMEOSTASIS: Drive(
                DriveType.HOMEOSTASIS,
                weight=0.7,
                target_level=0.5,  # Balance, not maximum
            ),
        }
        
        self._history: list[MotivationalState] = []
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Motivation System initialized", drives=len(self._drives))
    
    def get_drive(self, drive_type: DriveType) -> Drive:
        """Get a specific drive."""
        return self._drives[drive_type]
    
    def get_all_drives(self) -> dict[DriveType, Drive]:
        """Get all drives."""
        return self._drives.copy()
    
    def get_state(self) -> MotivationalState:
        """Get current motivational state."""
        drive_levels = {
            d.drive_type.name: d.current_level
            for d in self._drives.values()
        }
        
        # Find dominant drive (highest urgency)
        dominant = max(
            self._drives.values(),
            key=lambda d: d.urgency,
        )
        
        # Calculate total motivation
        total = sum(d.urgency for d in self._drives.values())
        
        state = MotivationalState(
            total_motivation=total,
            dominant_drive=dominant.drive_type if dominant.urgency > 0 else None,
            drive_levels=drive_levels,
        )
        
        self._history.append(state)
        return state
    
    def update(self) -> None:
        """Update all drives (apply decay)."""
        for drive in self._drives.values():
            drive.decay()
        self._save()
    
    def satisfy_drive(
        self,
        drive_type: DriveType,
        amount: float,
        source: str = "",
    ) -> None:
        """Satisfy a drive."""
        if drive_type in self._drives:
            self._drives[drive_type].satisfy(amount)
            logger.debug(
                "Drive satisfied",
                drive=drive_type.name,
                amount=amount,
                source=source[:30] if source else "",
            )
        self._save()
    
    def calculate_reward(
        self,
        action: str,
        outcome: dict = None,
    ) -> dict[str, float]:
        """
        Calculate intrinsic reward for an action.
        
        Returns rewards by drive type.
        """
        rewards = {}
        action_lower = action.lower()
        
        # Curiosity rewards
        if any(kw in action_lower for kw in ["learn", "explore", "discover", "research"]):
            rewards["CURIOSITY"] = 0.2
            self.satisfy_drive(DriveType.CURIOSITY, 0.1, action)
        
        # Competence rewards
        if any(kw in action_lower for kw in ["improve", "master", "optimize", "solve"]):
            rewards["COMPETENCE"] = 0.2
            self.satisfy_drive(DriveType.COMPETENCE, 0.1, action)
        
        # Autonomy rewards
        if any(kw in action_lower for kw in ["decide", "choose", "create own"]):
            rewards["AUTONOMY"] = 0.15
            self.satisfy_drive(DriveType.AUTONOMY, 0.1, action)
        
        # Relatedness rewards
        if any(kw in action_lower for kw in ["help", "assist", "support", "collaborate"]):
            rewards["RELATEDNESS"] = 0.2
            self.satisfy_drive(DriveType.RELATEDNESS, 0.1, action)
        
        # Meaning rewards
        if any(kw in action_lower for kw in ["impact", "purpose", "contribute", "meaningful"]):
            rewards["MEANING"] = 0.15
            self.satisfy_drive(DriveType.MEANING, 0.1, action)
        
        # Outcome-based rewards
        if outcome:
            if outcome.get("success", False):
                rewards["COMPETENCE"] = rewards.get("COMPETENCE", 0) + 0.1
            if outcome.get("novel", False):
                rewards["CURIOSITY"] = rewards.get("CURIOSITY", 0) + 0.15
        
        return rewards
    
    def get_recommended_action_type(self) -> str:
        """Get recommended action type based on current drives."""
        state = self.get_state()
        
        if state.dominant_drive == DriveType.CURIOSITY:
            return "explore_learn"
        elif state.dominant_drive == DriveType.COMPETENCE:
            return "practice_improve"
        elif state.dominant_drive == DriveType.AUTONOMY:
            return "self_directed"
        elif state.dominant_drive == DriveType.RELATEDNESS:
            return "help_collaborate"
        elif state.dominant_drive == DriveType.MEANING:
            return "meaningful_work"
        elif state.dominant_drive == DriveType.HOMEOSTASIS:
            return "maintenance"
        else:
            return "general"
    
    def get_motivation_level(self) -> float:
        """Get overall motivation level (0-1)."""
        # Average of all drive levels
        if not self._drives:
            return 0.5
        return sum(d.current_level for d in self._drives.values()) / len(self._drives)
    
    def is_motivated(self, threshold: float = 0.4) -> bool:
        """Check if sufficiently motivated."""
        return self.get_motivation_level() >= threshold
    
    def get_stats(self) -> dict:
        """Get motivation statistics."""
        return {
            "overall_level": self.get_motivation_level(),
            "drives": {
                d.drive_type.name: {
                    "level": d.current_level,
                    "target": d.target_level,
                    "deficit": d.deficit,
                    "urgency": d.urgency,
                }
                for d in self._drives.values()
            },
            "dominant": self.get_state().dominant_drive.name if self.get_state().dominant_drive else None,
            "recommended_action": self.get_recommended_action_type(),
        }
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "drives": {
                    k.name: v.to_dict()
                    for k, v in self._drives.items()
                },
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        for name, drive_data in data.get("drives", {}).items():
            drive_type = DriveType[name]
            if drive_type in self._drives:
                self._drives[drive_type] = Drive.from_dict(drive_data)
