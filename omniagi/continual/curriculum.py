"""
Curriculum Learning - Progressive skill acquisition.

Orders learning tasks by difficulty for optimal skill building.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = structlog.get_logger()


@dataclass
class Skill:
    """A skill to be learned."""
    
    id: str
    name: str
    domain: str
    difficulty: float  # 0.0 - 1.0
    prerequisites: list[str] = field(default_factory=list)
    
    # Progress
    attempts: int = 0
    successes: int = 0
    mastery_level: float = 0.0  # 0.0 - 1.0
    last_practiced: datetime | None = None
    
    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts
    
    def record_attempt(self, success: bool) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
        
        # Update mastery based on recent performance
        self.mastery_level = 0.7 * self.mastery_level + 0.3 * (1.0 if success else 0.0)
        self.last_practiced = datetime.now()


@dataclass
class LearningTask:
    """A task for skill development."""
    
    id: str
    skill_id: str
    description: str
    difficulty: float
    example_input: str
    expected_output: str | None = None
    hints: list[str] = field(default_factory=list)


class CurriculumLearner:
    """
    Curriculum-based learning for progressive skill acquisition.
    
    Orders tasks by difficulty and prerequisites to ensure
    optimal learning progression.
    """
    
    def __init__(self):
        """Initialize curriculum learner."""
        self._skills: dict[str, Skill] = {}
        self._tasks: dict[str, LearningTask] = {}
        self._current_focus: str | None = None
    
    def add_skill(
        self,
        name: str,
        domain: str,
        difficulty: float,
        skill_id: str | None = None,
        prerequisites: list[str] | None = None,
    ) -> Skill:
        """Add a skill to the curriculum."""
        import uuid
        
        skill_id = skill_id or str(uuid.uuid4())
        
        skill = Skill(
            id=skill_id,
            name=name,
            domain=domain,
            difficulty=difficulty,
            prerequisites=prerequisites or [],
        )
        
        self._skills[skill_id] = skill
        return skill
    
    def add_task(
        self,
        skill_id: str,
        description: str,
        difficulty: float,
        example_input: str,
        expected_output: str | None = None,
        hints: list[str] | None = None,
    ) -> LearningTask | None:
        """Add a task for a skill."""
        if skill_id not in self._skills:
            logger.warning("Skill not found", skill_id=skill_id)
            return None
        
        import uuid
        task_id = str(uuid.uuid4())
        
        task = LearningTask(
            id=task_id,
            skill_id=skill_id,
            description=description,
            difficulty=difficulty,
            example_input=example_input,
            expected_output=expected_output,
            hints=hints or [],
        )
        
        self._tasks[task_id] = task
        return task
    
    def get_next_skill(self) -> Skill | None:
        """
        Get the next skill to focus on.
        
        Considers:
        - Prerequisites met
        - Appropriate difficulty
        - Not yet mastered
        """
        available = []
        
        for skill in self._skills.values():
            # Skip mastered skills
            if skill.mastery_level > 0.9:
                continue
            
            # Check prerequisites
            prereqs_met = all(
                self._skills.get(p) and self._skills[p].mastery_level > 0.7
                for p in skill.prerequisites
            )
            
            if not prereqs_met:
                continue
            
            available.append(skill)
        
        if not available:
            return None
        
        # Sort by optimal learning order
        # Prefer: low mastery, appropriate difficulty
        available.sort(key=lambda s: (s.mastery_level, s.difficulty))
        
        return available[0]
    
    def get_task_for_skill(self, skill_id: str) -> LearningTask | None:
        """Get an appropriate task for a skill."""
        skill = self._skills.get(skill_id)
        if not skill:
            return None
        
        # Get tasks for this skill
        tasks = [t for t in self._tasks.values() if t.skill_id == skill_id]
        
        if not tasks:
            return None
        
        # Choose task based on current mastery
        # Lower mastery -> easier tasks
        target_difficulty = skill.mastery_level * 0.8 + 0.2
        
        # Find closest match
        tasks.sort(key=lambda t: abs(t.difficulty - target_difficulty))
        
        return tasks[0]
    
    def record_practice(self, skill_id: str, success: bool) -> None:
        """Record practice results."""
        skill = self._skills.get(skill_id)
        if skill:
            skill.record_attempt(success)
            logger.info(
                "Practice recorded",
                skill=skill.name,
                success=success,
                mastery=f"{skill.mastery_level:.2f}",
            )
    
    def get_curriculum(self) -> list[Skill]:
        """Get the full curriculum ordered by learning path."""
        # Topological sort based on prerequisites
        ordered = []
        visited = set()
        
        def visit(skill_id: str):
            if skill_id in visited:
                return
            
            skill = self._skills.get(skill_id)
            if not skill:
                return
            
            for prereq in skill.prerequisites:
                visit(prereq)
            
            visited.add(skill_id)
            ordered.append(skill)
        
        for skill_id in self._skills:
            visit(skill_id)
        
        return ordered
    
    def get_progress(self) -> dict[str, Any]:
        """Get overall learning progress."""
        total_skills = len(self._skills)
        mastered = sum(1 for s in self._skills.values() if s.mastery_level > 0.9)
        in_progress = sum(
            1 for s in self._skills.values()
            if 0.1 < s.mastery_level <= 0.9
        )
        
        avg_mastery = (
            sum(s.mastery_level for s in self._skills.values()) / max(1, total_skills)
        )
        
        return {
            "total_skills": total_skills,
            "mastered": mastered,
            "in_progress": in_progress,
            "not_started": total_skills - mastered - in_progress,
            "average_mastery": round(avg_mastery, 2),
            "total_practice_attempts": sum(s.attempts for s in self._skills.values()),
        }
    
    def suggest_practice(self, duration_minutes: int = 30) -> list[Skill]:
        """Suggest skills to practice in a session."""
        suggestions = []
        remaining_time = duration_minutes
        
        while remaining_time > 0:
            skill = self.get_next_skill()
            if not skill or skill in suggestions:
                break
            
            suggestions.append(skill)
            # Estimate time based on difficulty
            remaining_time -= (10 + skill.difficulty * 20)
        
        return suggestions
