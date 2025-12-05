"""
ARC Benchmark - Abstraction and Reasoning Corpus.

ARC is considered the gold standard for measuring AGI progress.
Tasks require:
- Pattern recognition
- Spatial reasoning
- Rule abstraction
- Novel generalization
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import structlog

logger = structlog.get_logger()


@dataclass
class ARCTask:
    """A single ARC task."""
    
    task_id: str
    train_examples: list[dict]  # [{"input": [[...]], "output": [[...]]}]
    test_input: list[list[int]]
    test_output: list[list[int]]
    
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class ARCResult:
    """Result of solving an ARC task."""
    
    task_id: str
    predicted: list[list[int]]
    expected: list[list[int]]
    correct: bool
    reasoning: str = ""
    time_seconds: float = 0.0


class ARCGenerator:
    """
    Generate ARC-style tasks for testing.
    
    Since full ARC requires licensed data, we generate
    similar synthetic tasks that test the same capabilities.
    """
    
    # Colors: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=pink, 7=orange, 8=cyan, 9=brown
    
    def __init__(self):
        self._task_generators = [
            self._mirror_task,
            self._rotate_task,
            self._fill_task,
            self._pattern_continuation,
            self._color_swap_task,
            self._scale_task,
            self._count_task,
            self._symmetry_task,
        ]
        
        logger.info("ARC Generator initialized", tasks=len(self._task_generators))
    
    def generate_tasks(self, n: int = 10) -> list[ARCTask]:
        """Generate n ARC-style tasks."""
        tasks = []
        
        for i in range(n):
            generator = random.choice(self._task_generators)
            task = generator(f"synthetic_{i}")
            tasks.append(task)
        
        return tasks
    
    def _mirror_task(self, task_id: str) -> ARCTask:
        """Generate a horizontal mirror task."""
        size = random.randint(3, 5)
        
        def make_example():
            inp = [[random.randint(0, 5) for _ in range(size)] for _ in range(size)]
            out = [row[::-1] for row in inp]  # Horizontal mirror
            return {"input": inp, "output": out}
        
        train = [make_example() for _ in range(3)]
        test_inp = [[random.randint(0, 5) for _ in range(size)] for _ in range(size)]
        test_out = [row[::-1] for row in test_inp]
        
        return ARCTask(
            task_id=task_id,
            train_examples=train,
            test_input=test_inp,
            test_output=test_out,
            difficulty="easy",
        )
    
    def _rotate_task(self, task_id: str) -> ARCTask:
        """Generate a 90-degree rotation task."""
        size = random.randint(3, 5)
        
        def rotate_90(grid):
            return [[grid[size-1-j][i] for j in range(size)] for i in range(size)]
        
        def make_example():
            inp = [[random.randint(0, 5) for _ in range(size)] for _ in range(size)]
            out = rotate_90(inp)
            return {"input": inp, "output": out}
        
        train = [make_example() for _ in range(3)]
        test_inp = [[random.randint(0, 5) for _ in range(size)] for _ in range(size)]
        test_out = rotate_90(test_inp)
        
        return ARCTask(
            task_id=task_id,
            train_examples=train,
            test_input=test_inp,
            test_output=test_out,
            difficulty="easy",
        )
    
    def _fill_task(self, task_id: str) -> ARCTask:
        """Generate a fill-enclosed-area task."""
        size = 5
        
        def make_example():
            inp = [[0 for _ in range(size)] for _ in range(size)]
            # Draw a rectangle outline
            r1, c1 = random.randint(0, 1), random.randint(0, 1)
            r2, c2 = random.randint(3, 4), random.randint(3, 4)
            color = random.randint(1, 5)
            
            for r in range(r1, r2+1):
                inp[r][c1] = color
                inp[r][c2] = color
            for c in range(c1, c2+1):
                inp[r1][c] = color
                inp[r2][c] = color
            
            # Output fills the inside
            out = [row[:] for row in inp]
            fill_color = random.randint(1, 5)
            while fill_color == color:
                fill_color = random.randint(1, 5)
            
            for r in range(r1+1, r2):
                for c in range(c1+1, c2):
                    out[r][c] = fill_color
            
            return {"input": inp, "output": out}
        
        train = [make_example() for _ in range(3)]
        test_ex = make_example()
        
        return ARCTask(
            task_id=task_id,
            train_examples=train,
            test_input=test_ex["input"],
            test_output=test_ex["output"],
            difficulty="medium",
        )
    
    def _pattern_continuation(self, task_id: str) -> ARCTask:
        """Generate a pattern continuation task."""
        size = 7
        
        def make_example():
            inp = [[0 for _ in range(size)] for _ in range(size)]
            out = [[0 for _ in range(size)] for _ in range(size)]
            
            # Draw a simple pattern (diagonal)
            color = random.randint(1, 5)
            for i in range(size // 2):
                inp[i][i] = color
                out[i][i] = color
            
            # Continue pattern in output only
            for i in range(size // 2, size):
                out[i][i] = color
            
            return {"input": inp, "output": out}
        
        train = [make_example() for _ in range(3)]
        test_ex = make_example()
        
        return ARCTask(
            task_id=task_id,
            train_examples=train,
            test_input=test_ex["input"],
            test_output=test_ex["output"],
            difficulty="medium",
        )
    
    def _color_swap_task(self, task_id: str) -> ARCTask:
        """Generate a color swap task."""
        size = random.randint(4, 6)
        
        color1, color2 = random.sample(range(1, 6), 2)
        
        def make_example():
            inp = [[random.choice([0, color1, color2]) for _ in range(size)] for _ in range(size)]
            out = [[color2 if c == color1 else (color1 if c == color2 else c) for c in row] for row in inp]
            return {"input": inp, "output": out}
        
        train = [make_example() for _ in range(3)]
        test_ex = make_example()
        
        return ARCTask(
            task_id=task_id,
            train_examples=train,
            test_input=test_ex["input"],
            test_output=test_ex["output"],
            difficulty="easy",
        )
    
    def _scale_task(self, task_id: str) -> ARCTask:
        """Generate a 2x scaling task."""
        small_size = 3
        
        def make_example():
            inp = [[random.randint(0, 5) for _ in range(small_size)] for _ in range(small_size)]
            # Scale 2x
            out = []
            for row in inp:
                scaled_row = []
                for c in row:
                    scaled_row.extend([c, c])
                out.append(scaled_row)
                out.append(scaled_row[:])
            return {"input": inp, "output": out}
        
        train = [make_example() for _ in range(3)]
        test_ex = make_example()
        
        return ARCTask(
            task_id=task_id,
            train_examples=train,
            test_input=test_ex["input"],
            test_output=test_ex["output"],
            difficulty="easy",
        )
    
    def _count_task(self, task_id: str) -> ARCTask:
        """Generate a counting task - output count of colored cells."""
        size = 5
        
        def make_example():
            color = random.randint(1, 5)
            inp = [[0 for _ in range(size)] for _ in range(size)]
            count = random.randint(1, 9)
            
            positions = random.sample([(r, c) for r in range(size) for c in range(size)], count)
            for r, c in positions:
                inp[r][c] = color
            
            # Output: 1xN where N is the count
            out = [[color] * count]
            
            return {"input": inp, "output": out}
        
        train = [make_example() for _ in range(3)]
        test_ex = make_example()
        
        return ARCTask(
            task_id=task_id,
            train_examples=train,
            test_input=test_ex["input"],
            test_output=test_ex["output"],
            difficulty="medium",
        )
    
    def _symmetry_task(self, task_id: str) -> ARCTask:
        """Generate a symmetry detection task."""
        size = 5
        
        def make_example():
            half = size // 2
            inp = [[random.randint(0, 5) for _ in range(half)] for _ in range(size)]
            
            # Complete with horizontal symmetry
            out = [row + row[::-1] + [row[0]] for row in inp]
            
            return {"input": [row + [0] * (half + 1) for row in inp], "output": out}
        
        train = [make_example() for _ in range(3)]
        test_ex = make_example()
        
        return ARCTask(
            task_id=task_id,
            train_examples=train,
            test_input=test_ex["input"],
            test_output=test_ex["output"],
            difficulty="medium",
        )


class ARCBenchmark:
    """
    ARC Benchmark runner.
    
    Tests AGI capabilities on abstract reasoning tasks.
    """
    
    def __init__(self, llm_pipeline=None):
        self._generator = ARCGenerator()
        self._pipeline = llm_pipeline
        self._results: list[ARCResult] = []
        
        logger.info("ARC Benchmark initialized")
    
    def set_pipeline(self, pipeline):
        """Set the LLM pipeline for reasoning."""
        self._pipeline = pipeline
    
    def run_benchmark(self, n_tasks: int = 10) -> dict:
        """
        Run ARC benchmark with n tasks.
        
        Returns summary statistics.
        """
        import time
        
        tasks = self._generator.generate_tasks(n_tasks)
        self._results = []
        
        for task in tasks:
            start = time.time()
            result = self._solve_task(task)
            result.time_seconds = time.time() - start
            self._results.append(result)
            
            logger.debug(
                "Task completed",
                task_id=task.task_id,
                correct=result.correct,
            )
        
        return self._compute_summary()
    
    def _solve_task(self, task: ARCTask) -> ARCResult:
        """Attempt to solve a single ARC task."""
        # Build prompt for the LLM
        prompt = self._build_prompt(task)
        
        if self._pipeline is None:
            # Without LLM, use heuristic solver
            predicted = self._heuristic_solve(task)
        else:
            # Use LLM to reason about the task
            predicted = self._llm_solve(task, prompt)
        
        correct = predicted == task.test_output
        
        return ARCResult(
            task_id=task.task_id,
            predicted=predicted,
            expected=task.test_output,
            correct=correct,
        )
    
    def _build_prompt(self, task: ARCTask) -> str:
        """Build a prompt for the LLM to solve the task."""
        prompt = "You are solving an abstract reasoning task. Study the examples and predict the output.\n\n"
        
        for i, ex in enumerate(task.train_examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Input:\n{self._grid_to_str(ex['input'])}\n"
            prompt += f"Output:\n{self._grid_to_str(ex['output'])}\n\n"
        
        prompt += "Now predict the output for:\n"
        prompt += f"Input:\n{self._grid_to_str(task.test_input)}\n"
        prompt += "Output:\n"
        
        return prompt
    
    def _grid_to_str(self, grid: list[list[int]]) -> str:
        """Convert grid to string representation."""
        return "\n".join(" ".join(str(c) for c in row) for row in grid)
    
    def _heuristic_solve(self, task: ARCTask) -> list[list[int]]:
        """Simple heuristic solver without LLM."""
        # Try to detect simple transformations
        ex = task.train_examples[0]
        inp, out = ex["input"], ex["output"]
        
        # Check if it's a mirror
        if inp == [row[::-1] for row in out]:
            return [row[::-1] for row in task.test_input]
        
        # Check if same size - might be color swap
        if len(inp) == len(out) and len(inp[0]) == len(out[0]):
            # Just return input as fallback
            return task.test_input
        
        # Fallback: return input unchanged
        return task.test_input
    
    def _llm_solve(self, task: ARCTask, prompt: str) -> list[list[int]]:
        """Use LLM to solve the task."""
        from rwkv.utils import PIPELINE_ARGS
        
        args = PIPELINE_ARGS(temperature=0.3, top_p=0.8)
        
        try:
            response = self._pipeline.generate(prompt, token_count=200, args=args)
            
            # Parse the response to extract grid
            return self._parse_grid_response(response, task.test_output)
        except Exception as e:
            logger.warning(f"LLM solve failed: {e}")
            return self._heuristic_solve(task)
    
    def _parse_grid_response(
        self, response: str, expected: list[list[int]]
    ) -> list[list[int]]:
        """Parse LLM response to extract grid."""
        # Try to find numbers in the response
        lines = response.strip().split("\n")
        grid = []
        
        for line in lines:
            row = []
            for char in line.split():
                try:
                    row.append(int(char))
                except ValueError:
                    continue
            if row:
                grid.append(row)
        
        # Validate dimensions match expected
        if grid and len(grid) == len(expected):
            if all(len(r) == len(expected[0]) for r in grid):
                return grid
        
        # Fallback to expected (wrong answer but right dimensions)
        return [[0] * len(expected[0]) for _ in range(len(expected))]
    
    def _compute_summary(self) -> dict:
        """Compute benchmark summary statistics."""
        total = len(self._results)
        correct = sum(1 for r in self._results if r.correct)
        
        by_difficulty = {}
        for r in self._results:
            # We don't have difficulty in result, so just count
            pass
        
        avg_time = sum(r.time_seconds for r in self._results) / max(1, total)
        
        return {
            "total_tasks": total,
            "correct": correct,
            "accuracy": round(correct / max(1, total) * 100, 1),
            "avg_time_per_task": round(avg_time, 3),
            "agi_threshold": 50,  # 50% is considered AGI-level
            "agi_achieved": correct / max(1, total) >= 0.5,
        }
    
    def get_detailed_results(self) -> list[dict]:
        """Get detailed results for each task."""
        return [
            {
                "task_id": r.task_id,
                "correct": r.correct,
                "time_s": round(r.time_seconds, 3),
            }
            for r in self._results
        ]
