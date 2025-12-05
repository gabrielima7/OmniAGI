"""
ARC Benchmark - Abstract Reasoning Corpus for AGI testing.

ARC is a key benchmark for measuring genuine reasoning
ability - the kind that separates AGI from narrow AI.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

logger = structlog.get_logger()


@dataclass
class ARCTask:
    """A single ARC task."""
    
    id: str
    train: list[dict]  # Training examples: [{"input": grid, "output": grid}]
    test: list[dict]   # Test examples
    
    @property
    def num_train_examples(self) -> int:
        return len(self.train)
    
    @property
    def grid_size(self) -> tuple[int, int]:
        if self.train:
            grid = self.train[0]["input"]
            return len(grid), len(grid[0]) if grid else 0
        return 0, 0
    
    def to_text(self) -> str:
        """Convert task to text format for LLM."""
        lines = ["TASK: Learn the pattern from training examples.\n"]
        
        for i, example in enumerate(self.train):
            lines.append(f"Training Example {i+1}:")
            lines.append("Input:")
            lines.append(self._grid_to_text(example["input"]))
            lines.append("Output:")
            lines.append(self._grid_to_text(example["output"]))
            lines.append("")
        
        lines.append("Test Input (predict the output):")
        if self.test:
            lines.append(self._grid_to_text(self.test[0]["input"]))
        
        return "\n".join(lines)
    
    def _grid_to_text(self, grid: list[list[int]]) -> str:
        """Convert grid to text representation."""
        symbols = ".1234567890"  # 0=black, 1-9=colors
        lines = []
        for row in grid:
            line = "".join(symbols[min(c, 10)] for c in row)
            lines.append(line)
        return "\n".join(lines)


@dataclass
class ARCResult:
    """Result of solving an ARC task."""
    
    task_id: str
    predicted: list[list[int]] | None
    actual: list[list[int]]
    correct: bool
    reasoning: str = ""
    time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "correct": self.correct,
            "time_ms": self.time_ms,
            "reasoning": self.reasoning[:200],
        }


class ARCBenchmark:
    """
    ARC (Abstraction and Reasoning Corpus) benchmark.
    
    Tests genuine reasoning ability - solving novel
    visual puzzles with minimal examples.
    
    Current SOTA: ~30% (humans: ~85%)
    This is a key gap for AGI.
    """
    
    def __init__(
        self,
        data_path: Path | str | None = None,
        storage_path: Path | str | None = None,
    ):
        self.data_path = Path(data_path) if data_path else None
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._tasks: dict[str, ARCTask] = {}
        self._results: list[ARCResult] = []
        
        # Load tasks if available
        if self.data_path:
            self._load_tasks()
        else:
            # Use built-in sample tasks
            self._init_sample_tasks()
        
        logger.info("ARC Benchmark initialized", tasks=len(self._tasks))
    
    def _init_sample_tasks(self) -> None:
        """Initialize with sample ARC-like tasks."""
        # Simple pattern completion
        self._tasks["sample_1"] = ARCTask(
            id="sample_1",
            train=[
                {
                    "input": [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
                    "output": [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
                },
                {
                    "input": [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
                    "output": [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
                },
            ],
            test=[
                {
                    "input": [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                    "output": [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
                },
            ],
        )
        
        # Copy pattern
        self._tasks["sample_2"] = ARCTask(
            id="sample_2",
            train=[
                {
                    "input": [[1, 0], [0, 0]],
                    "output": [[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
                },
            ],
            test=[
                {
                    "input": [[2, 0], [0, 2]],
                    "output": [[2, 0, 2, 0], [0, 2, 0, 2], [2, 0, 2, 0], [0, 2, 0, 2]],
                },
            ],
        )
        
        # Rotation
        self._tasks["sample_3"] = ARCTask(
            id="sample_3",
            train=[
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[3, 1], [4, 2]],
                },
                {
                    "input": [[5, 6], [7, 8]],
                    "output": [[7, 5], [8, 6]],
                },
            ],
            test=[
                {
                    "input": [[1, 1], [2, 2]],
                    "output": [[2, 1], [2, 1]],
                },
            ],
        )
    
    def _load_tasks(self) -> None:
        """Load ARC tasks from data directory."""
        if not self.data_path or not self.data_path.exists():
            self._init_sample_tasks()
            return
        
        # Load from standard ARC format
        for json_file in self.data_path.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                task = ARCTask(
                    id=json_file.stem,
                    train=data.get("train", []),
                    test=data.get("test", []),
                )
                self._tasks[task.id] = task
                
            except Exception as e:
                logger.warning("Failed to load task", file=str(json_file), error=str(e))
    
    def get_task(self, task_id: str) -> ARCTask | None:
        """Get a specific task."""
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> list[ARCTask]:
        """Get all tasks."""
        return list(self._tasks.values())
    
    def solve_with_engine(
        self,
        task_id: str,
        engine: "Engine",
    ) -> ARCResult:
        """
        Attempt to solve an ARC task using the LLM.
        
        This tests abstract reasoning capability.
        """
        if task_id not in self._tasks:
            raise ValueError(f"Unknown task: {task_id}")
        
        if not engine or not engine.is_loaded:
            raise RuntimeError("Engine not loaded")
        
        import time
        start = time.time()
        
        task = self._tasks[task_id]
        
        # Create prompt
        prompt = f"""{task.to_text()}

Analyze the pattern in the training examples.
Then predict the output for the test input.

Think step-by-step:
1. What transforms input to output in each example?
2. What is the underlying rule?
3. Apply the rule to the test input.

Output the predicted grid in the same format (one line per row, use . for 0):"""
        
        # Generate solution
        response = engine.generate(prompt, max_tokens=500)
        
        elapsed = (time.time() - start) * 1000
        
        # Parse response
        predicted = self._parse_grid_response(response.text)
        actual = task.test[0]["output"] if task.test else []
        
        correct = predicted == actual
        
        result = ARCResult(
            task_id=task_id,
            predicted=predicted,
            actual=actual,
            correct=correct,
            reasoning=response.text[:500],
            time_ms=elapsed,
        )
        
        self._results.append(result)
        self._save()
        
        logger.info(
            "ARC task attempted",
            task=task_id,
            correct=correct,
            time_ms=elapsed,
        )
        
        return result
    
    def _parse_grid_response(self, response: str) -> list[list[int]] | None:
        """Parse a grid from LLM response."""
        try:
            lines = response.strip().split("\n")
            grid = []
            
            symbol_map = {".": 0, "0": 0}
            for i in range(1, 10):
                symbol_map[str(i)] = i
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Try to parse as grid row
                row = []
                for char in line:
                    if char in symbol_map:
                        row.append(symbol_map[char])
                    elif char.isdigit():
                        row.append(int(char))
                
                if row:
                    grid.append(row)
            
            return grid if grid else None
            
        except Exception:
            return None
    
    def run_benchmark(
        self,
        engine: "Engine",
        task_ids: list[str] = None,
    ) -> dict:
        """
        Run benchmark on multiple tasks.
        
        Returns summary statistics.
        """
        if task_ids is None:
            task_ids = list(self._tasks.keys())
        
        results = []
        for task_id in task_ids:
            try:
                result = self.solve_with_engine(task_id, engine)
                results.append(result)
            except Exception as e:
                logger.error("Task failed", task=task_id, error=str(e))
        
        correct = sum(1 for r in results if r.correct)
        total = len(results)
        
        return {
            "total_tasks": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "avg_time_ms": sum(r.time_ms for r in results) / len(results) if results else 0,
            "results": [r.to_dict() for r in results],
        }
    
    def get_accuracy(self) -> float:
        """Get overall accuracy from results."""
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if r.correct) / len(self._results)
    
    def is_agi_level(self, threshold: float = 0.5) -> bool:
        """
        Check if performance indicates AGI-level reasoning.
        
        Human performance: ~85%
        Current SOTA: ~30%
        Threshold for "AGI-level": 50%+
        """
        return self.get_accuracy() >= threshold
    
    def get_stats(self) -> dict:
        """Get benchmark statistics."""
        return {
            "total_tasks": len(self._tasks),
            "tasks_attempted": len(self._results),
            "accuracy": self.get_accuracy(),
            "is_agi_level": self.is_agi_level(),
            "human_baseline": 0.85,
            "current_sota": 0.30,
        }
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "results": [r.to_dict() for r in self._results],
                "stats": self.get_stats(),
            }, f, indent=2)
