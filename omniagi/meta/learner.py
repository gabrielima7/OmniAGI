"""
Meta-Learner - Learning to learn.

Optimizes learning strategies and hyperparameters
based on past performance.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

from omniagi.meta.strategy import Strategy, StrategyBank
from omniagi.meta.adapter import StrategyAdapter

logger = structlog.get_logger()


@dataclass
class LearningExperience:
    """Record of a learning experience."""
    
    id: str
    task_description: str
    domain: str
    task_type: str
    
    strategy_used: str | None
    prompt_used: str
    
    success: bool
    quality_score: float
    duration_seconds: float
    
    # What we learned
    lessons: list[str] = field(default_factory=list)
    
    # Hyperparameters used
    temperature: float = 0.7
    max_tokens: int = 512
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimalConfig:
    """Optimal configuration learned for a task type."""
    
    domain: str
    task_type: str
    
    # Best hyperparameters
    temperature: float
    max_tokens: int
    
    # Best strategy
    best_strategy_id: str | None
    
    # Confidence based on sample size
    confidence: float
    samples: int


class MetaLearner:
    """
    Meta-learner that optimizes learning strategies.
    
    Tracks performance across different:
    - Task types
    - Domains
    - Strategies
    - Hyperparameters
    
    And learns optimal configurations over time.
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        strategy_bank: StrategyBank | None = None,
    ):
        """
        Initialize meta-learner.
        
        Args:
            engine: LLM engine.
            strategy_bank: Bank of strategies.
        """
        self.engine = engine
        self.bank = strategy_bank or StrategyBank()
        self.adapter = StrategyAdapter(engine, self.bank)
        
        self._experiences: list[LearningExperience] = []
        self._optimal_configs: dict[tuple[str, str], OptimalConfig] = {}
        
        self._experience_counter = 0
    
    def record_experience(
        self,
        task_description: str,
        domain: str,
        task_type: str,
        strategy_id: str | None,
        prompt_used: str,
        success: bool,
        quality_score: float,
        duration_seconds: float,
        lessons: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LearningExperience:
        """
        Record a learning experience.
        
        Args:
            task_description: What was the task.
            domain: Task domain.
            task_type: Type of task.
            strategy_id: Strategy used (if any).
            prompt_used: The prompt that was used.
            success: Whether it succeeded.
            quality_score: Quality of the result (0-1).
            duration_seconds: Time taken.
            lessons: What was learned.
            temperature: Temperature used.
            max_tokens: Max tokens used.
            
        Returns:
            The recorded experience.
        """
        exp = LearningExperience(
            id=f"exp_{self._experience_counter}",
            task_description=task_description,
            domain=domain,
            task_type=task_type,
            strategy_used=strategy_id,
            prompt_used=prompt_used,
            success=success,
            quality_score=quality_score,
            duration_seconds=duration_seconds,
            lessons=lessons or [],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        self._experience_counter += 1
        self._experiences.append(exp)
        
        # Update strategy performance
        if strategy_id:
            self.bank.record_result(strategy_id, success, quality_score)
        
        # Update optimal config
        self._update_optimal_config(exp)
        
        logger.info(
            "Experience recorded",
            id=exp.id,
            domain=domain,
            success=success,
        )
        
        return exp
    
    def _update_optimal_config(self, exp: LearningExperience) -> None:
        """Update optimal configuration based on new experience."""
        key = (exp.domain, exp.task_type)
        
        # Get all experiences for this domain/type
        relevant = [
            e for e in self._experiences
            if e.domain == exp.domain and e.task_type == exp.task_type
        ]
        
        if len(relevant) < 3:
            # Not enough data yet
            return
        
        # Find best performing configuration
        successful = [e for e in relevant if e.success]
        
        if not successful:
            return
        
        # Calculate average best config
        avg_temp = sum(e.temperature for e in successful) / len(successful)
        avg_tokens = int(sum(e.max_tokens for e in successful) / len(successful))
        
        # Find most successful strategy
        strategy_counts: dict[str, tuple[int, float]] = {}
        for e in successful:
            if e.strategy_used:
                if e.strategy_used not in strategy_counts:
                    strategy_counts[e.strategy_used] = (0, 0.0)
                count, total = strategy_counts[e.strategy_used]
                strategy_counts[e.strategy_used] = (
                    count + 1,
                    total + e.quality_score,
                )
        
        best_strategy = None
        if strategy_counts:
            best_strategy = max(
                strategy_counts.keys(),
                key=lambda s: strategy_counts[s][1] / strategy_counts[s][0],
            )
        
        # Calculate confidence based on sample size
        confidence = min(0.95, 0.5 + 0.1 * len(relevant))
        
        self._optimal_configs[key] = OptimalConfig(
            domain=exp.domain,
            task_type=exp.task_type,
            temperature=round(avg_temp, 2),
            max_tokens=avg_tokens,
            best_strategy_id=best_strategy,
            confidence=confidence,
            samples=len(relevant),
        )
    
    def get_optimal_config(
        self,
        domain: str,
        task_type: str,
    ) -> OptimalConfig | None:
        """Get the optimal configuration for a task type."""
        return self._optimal_configs.get((domain, task_type))
    
    def suggest_approach(
        self,
        task_description: str,
        domain: str,
        task_type: str,
        context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Suggest the best approach for a task.
        
        Args:
            task_description: What needs to be done.
            domain: Task domain.
            task_type: Type of task.
            context: Additional context.
            
        Returns:
            Dictionary with suggested strategy, config, and prompt.
        """
        context = context or {}
        context["task_description"] = task_description
        
        # Get optimal config if available
        optimal = self.get_optimal_config(domain, task_type)
        
        # Find and adapt best strategy
        adapted = self.adapter.find_and_adapt(domain, task_type, context)
        
        suggestion = {
            "domain": domain,
            "task_type": task_type,
            "strategy": None,
            "prompt": None,
            "temperature": 0.7,
            "max_tokens": 512,
            "confidence": 0.5,
        }
        
        if adapted:
            suggestion["strategy"] = adapted.original_strategy.id
            suggestion["prompt"] = adapted.adapted_prompt
            suggestion["confidence"] = adapted.confidence
        
        if optimal:
            suggestion["temperature"] = optimal.temperature
            suggestion["max_tokens"] = optimal.max_tokens
            
            # Override strategy if optimal has better confidence
            if optimal.confidence > suggestion["confidence"]:
                suggestion["strategy"] = optimal.best_strategy_id
                suggestion["confidence"] = optimal.confidence
        
        logger.info(
            "Approach suggested",
            domain=domain,
            task_type=task_type,
            strategy=suggestion["strategy"],
            confidence=suggestion["confidence"],
        )
        
        return suggestion
    
    def extract_lessons(
        self,
        experience_ids: list[str] | None = None,
    ) -> list[str]:
        """
        Extract lessons from experiences.
        
        Args:
            experience_ids: Specific experiences (or all if None).
            
        Returns:
            List of lessons learned.
        """
        if experience_ids:
            experiences = [
                e for e in self._experiences
                if e.id in experience_ids
            ]
        else:
            experiences = self._experiences
        
        all_lessons = []
        for exp in experiences:
            all_lessons.extend(exp.lessons)
        
        # Deduplicate and return
        return list(set(all_lessons))
    
    def get_stats(self) -> dict[str, Any]:
        """Get meta-learning statistics."""
        total = len(self._experiences)
        successful = sum(1 for e in self._experiences if e.success)
        
        return {
            "total_experiences": total,
            "success_rate": successful / max(1, total),
            "domains_covered": len(set(e.domain for e in self._experiences)),
            "strategies_tried": len(set(
                e.strategy_used for e in self._experiences
                if e.strategy_used
            )),
            "optimal_configs": len(self._optimal_configs),
            "total_lessons": len(self.extract_lessons()),
        }
