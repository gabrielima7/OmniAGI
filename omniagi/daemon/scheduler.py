"""
Task scheduler for the daemon.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from heapq import heappush, heappop

logger = structlog.get_logger()


@dataclass(order=True)
class ScheduledTask:
    """A task scheduled for future execution."""
    
    run_at: datetime
    task_id: str = field(compare=False)
    prompt: str = field(compare=False)
    priority: int = field(default=5, compare=False)
    recurring: bool = field(default=False, compare=False)
    interval_seconds: int = field(default=0, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


class TaskScheduler:
    """
    Scheduler for future tasks.
    
    Maintains a priority queue of scheduled tasks that
    the daemon should execute.
    """
    
    def __init__(self):
        """Initialize the scheduler."""
        self._tasks: list[ScheduledTask] = []
        self._task_lookup: dict[str, ScheduledTask] = {}
    
    def schedule(
        self,
        task_id: str,
        prompt: str,
        run_at: datetime | None = None,
        delay_seconds: int = 0,
        priority: int = 5,
        recurring: bool = False,
        interval_seconds: int = 0,
        **metadata,
    ) -> ScheduledTask:
        """
        Schedule a task for future execution.
        
        Args:
            task_id: Unique task identifier.
            prompt: The prompt to execute.
            run_at: Specific time to run (or None to use delay).
            delay_seconds: Delay from now in seconds.
            priority: Task priority (1-10, higher = more important).
            recurring: Whether to reschedule after completion.
            interval_seconds: Interval for recurring tasks.
            **metadata: Additional task metadata.
            
        Returns:
            The scheduled task.
        """
        if run_at is None:
            run_at = datetime.now() + timedelta(seconds=delay_seconds)
        
        task = ScheduledTask(
            run_at=run_at,
            task_id=task_id,
            prompt=prompt,
            priority=priority,
            recurring=recurring,
            interval_seconds=interval_seconds,
            metadata=metadata,
        )
        
        heappush(self._tasks, task)
        self._task_lookup[task_id] = task
        
        logger.info(
            "Task scheduled",
            task_id=task_id,
            run_at=run_at.isoformat(),
            recurring=recurring,
        )
        
        return task
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.
        
        Returns True if task was found and cancelled.
        """
        if task_id in self._task_lookup:
            # Mark as cancelled by removing from lookup
            del self._task_lookup[task_id]
            logger.info("Task cancelled", task_id=task_id)
            return True
        return False
    
    def get_due_tasks(self) -> list[dict[str, Any]]:
        """
        Get all tasks that are due for execution.
        
        Returns task data and removes non-recurring tasks from queue.
        """
        now = datetime.now()
        due_tasks = []
        
        while self._tasks and self._tasks[0].run_at <= now:
            task = heappop(self._tasks)
            
            # Skip cancelled tasks
            if task.task_id not in self._task_lookup:
                continue
            
            due_tasks.append({
                "task_id": task.task_id,
                "prompt": task.prompt,
                "priority": task.priority,
                **task.metadata,
            })
            
            # Reschedule recurring tasks
            if task.recurring and task.interval_seconds > 0:
                new_run_at = now + timedelta(seconds=task.interval_seconds)
                new_task = ScheduledTask(
                    run_at=new_run_at,
                    task_id=task.task_id,
                    prompt=task.prompt,
                    priority=task.priority,
                    recurring=True,
                    interval_seconds=task.interval_seconds,
                    metadata=task.metadata,
                )
                heappush(self._tasks, new_task)
                self._task_lookup[task.task_id] = new_task
            else:
                # Remove one-time tasks from lookup
                if task.task_id in self._task_lookup:
                    del self._task_lookup[task.task_id]
        
        return due_tasks
    
    def get_pending(self) -> list[ScheduledTask]:
        """Get all pending (not yet due) tasks."""
        return [
            task for task in self._tasks
            if task.task_id in self._task_lookup
        ]
    
    def clear(self) -> None:
        """Clear all scheduled tasks."""
        self._tasks.clear()
        self._task_lookup.clear()
        logger.info("All tasks cleared")
    
    def __len__(self) -> int:
        return len(self._task_lookup)
