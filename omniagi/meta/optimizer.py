"""
Meta-Learning Optimizer - Learn how to learn better.

This enables the AGI to:
1. Optimize its own learning strategies
2. Select best approaches per task type
3. Improve learning efficiency over time

Critical for AGI - humans get better at learning itself.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List
from uuid import uuid4

logger = structlog.get_logger()


class StrategyType(Enum):
    """Types of learning strategies."""
    
    ANALOGY = auto()        # Learn by comparison
    DECOMPOSITION = auto()  # Break down problem
    ABSTRACTION = auto()    # Generalize patterns
    REHEARSAL = auto()      # Repeat practice
    ELABORATION = auto()    # Enrich with context
    TRIAL_ERROR = auto()    # Experiment
    TEACHING = auto()       # Explain to learn


@dataclass
class LearningStrategy:
    """A meta-learning strategy."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    strategy_type: StrategyType = StrategyType.ANALOGY
    
    # When to use
    applicable_domains: list[str] = field(default_factory=list)
    applicable_task_types: list[str] = field(default_factory=list)
    
    # Performance metrics
    times_used: int = 0
    success_count: int = 0
    avg_learning_speed: float = 0.0  # Time to learn
    avg_retention: float = 0.0       # How well remembered
    
    # Parameters
    parameters: dict = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.times_used == 0:
            return 0.0
        return self.success_count / self.times_used
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.strategy_type.name,
            "success_rate": self.success_rate,
            "times_used": self.times_used,
        }


@dataclass
class MetaLearningExperience:
    """Record of a meta-learning experience."""
    
    task_type: str
    domain: str
    strategy_used: str
    
    success: bool
    learning_time: float  # seconds
    retention_score: float  # 0-1
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetaLearner:
    """
    Meta-Learning System for AGI.
    
    Learns how to learn more effectively by:
    1. Tracking which strategies work for which tasks
    2. Optimizing strategy selection
    3. Refining strategy parameters
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Available strategies
        self._strategies: dict[str, LearningStrategy] = {}
        
        # Experience history
        self._experiences: list[MetaLearningExperience] = []
        
        # Task-strategy mappings (learned over time)
        self._task_strategy_scores: dict[str, dict[str, float]] = {}
        
        # Initialize default strategies
        self._init_default_strategies()
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info(
            "Meta-Learner initialized",
            strategies=len(self._strategies)
        )
    
    def _init_default_strategies(self) -> None:
        """Initialize default learning strategies."""
        defaults = [
            LearningStrategy(
                name="analogy_learning",
                strategy_type=StrategyType.ANALOGY,
                applicable_task_types=["pattern_recognition", "classification"],
                parameters={"similarity_threshold": 0.6},
            ),
            LearningStrategy(
                name="divide_conquer",
                strategy_type=StrategyType.DECOMPOSITION,
                applicable_task_types=["problem_solving", "planning"],
                parameters={"max_depth": 5},
            ),
            LearningStrategy(
                name="pattern_abstraction",
                strategy_type=StrategyType.ABSTRACTION,
                applicable_task_types=["generalization", "rule_learning"],
                parameters={"abstraction_level": 2},
            ),
            LearningStrategy(
                name="spaced_rehearsal",
                strategy_type=StrategyType.REHEARSAL,
                applicable_task_types=["memorization", "skill_acquisition"],
                parameters={"interval_multiplier": 2.0},
            ),
            LearningStrategy(
                name="active_experimentation",
                strategy_type=StrategyType.TRIAL_ERROR,
                applicable_task_types=["exploration", "optimization"],
                parameters={"exploration_rate": 0.3},
            ),
        ]
        
        for strategy in defaults:
            self._strategies[strategy.id] = strategy
    
    def select_strategy(
        self,
        task_type: str,
        domain: str = "",
    ) -> LearningStrategy:
        """
        Select the best learning strategy for a task.
        
        Uses historical performance to make selection.
        """
        candidates = []
        
        for strategy in self._strategies.values():
            score = self._compute_strategy_score(strategy, task_type, domain)
            candidates.append((strategy, score))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            selected = candidates[0][0]
            logger.debug(
                "Strategy selected",
                strategy=selected.name,
                task_type=task_type,
            )
            return selected
        
        # Fallback to first strategy
        return list(self._strategies.values())[0]
    
    def _compute_strategy_score(
        self,
        strategy: LearningStrategy,
        task_type: str,
        domain: str,
    ) -> float:
        """Compute strategy fitness score for task."""
        score = 0.5  # Base score
        
        # Boost if task type matches
        if task_type in strategy.applicable_task_types:
            score += 0.2
        
        # Boost if domain matches
        if domain in strategy.applicable_domains:
            score += 0.1
        
        # Add historical performance
        key = f"{task_type}_{domain}"
        if key in self._task_strategy_scores:
            historical = self._task_strategy_scores[key].get(strategy.id, 0.5)
            score = 0.6 * score + 0.4 * historical
        
        # Factor in general success rate
        score = 0.7 * score + 0.3 * strategy.success_rate
        
        return score
    
    def record_experience(
        self,
        task_type: str,
        domain: str,
        strategy_id: str,
        success: bool,
        learning_time: float = 1.0,
        retention_score: float = 0.5,
    ) -> None:
        """
        Record a learning experience.
        
        Updates strategy performance metrics.
        """
        experience = MetaLearningExperience(
            task_type=task_type,
            domain=domain,
            strategy_used=strategy_id,
            success=success,
            learning_time=learning_time,
            retention_score=retention_score,
        )
        self._experiences.append(experience)
        
        # Update strategy stats
        if strategy_id in self._strategies:
            strategy = self._strategies[strategy_id]
            strategy.times_used += 1
            if success:
                strategy.success_count += 1
            
            # Update averages
            n = strategy.times_used
            strategy.avg_learning_speed = (
                strategy.avg_learning_speed * (n - 1) + learning_time
            ) / n
            strategy.avg_retention = (
                strategy.avg_retention * (n - 1) + retention_score
            ) / n
        
        # Update task-strategy mapping
        key = f"{task_type}_{domain}"
        if key not in self._task_strategy_scores:
            self._task_strategy_scores[key] = {}
        
        current = self._task_strategy_scores[key].get(strategy_id, 0.5)
        # Exponential moving average
        alpha = 0.3
        new_score = 1.0 if success else 0.0
        self._task_strategy_scores[key][strategy_id] = (
            alpha * new_score + (1 - alpha) * current
        )
        
        self._save()
    
    def optimize_strategies(self) -> dict:
        """
        Optimize all strategies based on experience.
        
        Adjusts parameters to improve performance.
        """
        optimizations = {
            "strategies_optimized": 0,
            "parameters_adjusted": 0,
        }
        
        for strategy in self._strategies.values():
            if strategy.times_used < 5:
                continue  # Not enough data
            
            # Adjust based on performance
            if strategy.success_rate < 0.3:
                # Strategy performing poorly, try adjusting parameters
                if strategy.strategy_type == StrategyType.ANALOGY:
                    # Lower threshold for more matches
                    current = strategy.parameters.get("similarity_threshold", 0.6)
                    strategy.parameters["similarity_threshold"] = max(0.3, current - 0.1)
                    optimizations["parameters_adjusted"] += 1
                
                elif strategy.strategy_type == StrategyType.TRIAL_ERROR:
                    # Increase exploration
                    current = strategy.parameters.get("exploration_rate", 0.3)
                    strategy.parameters["exploration_rate"] = min(0.7, current + 0.1)
                    optimizations["parameters_adjusted"] += 1
            
            elif strategy.success_rate > 0.8:
                # Strategy doing well, fine-tune
                if strategy.strategy_type == StrategyType.ANALOGY:
                    # Can be more selective
                    current = strategy.parameters.get("similarity_threshold", 0.6)
                    strategy.parameters["similarity_threshold"] = min(0.9, current + 0.05)
                    optimizations["parameters_adjusted"] += 1
            
            optimizations["strategies_optimized"] += 1
        
        self._save()
        
        logger.info("Strategies optimized", **optimizations)
        return optimizations
    
    def get_best_strategies(self, n: int = 5) -> list[LearningStrategy]:
        """Get top performing strategies."""
        sorted_strategies = sorted(
            self._strategies.values(),
            key=lambda s: s.success_rate * (1 + s.times_used / 100),
            reverse=True,
        )
        return sorted_strategies[:n]
    
    def add_strategy(
        self,
        name: str,
        strategy_type: StrategyType,
        task_types: list[str] = None,
        parameters: dict = None,
    ) -> LearningStrategy:
        """Add a new learning strategy."""
        strategy = LearningStrategy(
            name=name,
            strategy_type=strategy_type,
            applicable_task_types=task_types or [],
            parameters=parameters or {},
        )
        self._strategies[strategy.id] = strategy
        self._save()
        return strategy
    
    def get_stats(self) -> dict:
        """Get meta-learning statistics."""
        return {
            "strategies": len(self._strategies),
            "experiences": len(self._experiences),
            "avg_success_rate": sum(
                s.success_rate for s in self._strategies.values()
            ) / max(1, len(self._strategies)),
        }
    
    def __len__(self) -> int:
        return len(self._strategies)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "strategies": [s.to_dict() for s in self._strategies.values()],
                "experiences": len(self._experiences),
            }, f, indent=2)
    
    def _load(self) -> None:
        pass
