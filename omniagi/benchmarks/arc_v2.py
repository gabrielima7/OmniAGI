"""
ARC-AGI Benchmark Solver.

Implements a solver for the Abstraction and Reasoning Corpus (ARC),
the gold standard benchmark for measuring AGI capabilities.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    np = None


@dataclass
class ARCTask:
    """An ARC task with training and test examples."""
    task_id: str
    train_examples: List[Tuple[List[List[int]], List[List[int]]]]  # (input, output)
    test_examples: List[Tuple[List[List[int]], Optional[List[List[int]]]]]
    
    @property
    def num_train(self) -> int:
        return len(self.train_examples)
    
    @property
    def num_test(self) -> int:
        return len(self.test_examples)


@dataclass
class ARCPrediction:
    """A prediction for an ARC task."""
    task_id: str
    test_index: int
    predicted_output: List[List[int]]
    confidence: float
    reasoning: str


class PatternDetector:
    """
    Detects patterns in ARC grids.
    
    Identifies transformations like rotation, reflection,
    color mapping, scaling, etc.
    """
    
    def __init__(self):
        self.detected_patterns: List[str] = []
    
    def analyze(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[str, Any]:
        """Analyze input-output pair to detect patterns."""
        patterns = {}
        
        in_h, in_w = len(input_grid), len(input_grid[0]) if input_grid else 0
        out_h, out_w = len(output_grid), len(output_grid[0]) if output_grid else 0
        
        # Size relationship
        if in_h == out_h and in_w == out_w:
            patterns['same_size'] = True
        elif out_h == in_h * 2 and out_w == in_w * 2:
            patterns['doubled'] = True
        elif out_h == in_h // 2 and out_w == in_w // 2:
            patterns['halved'] = True
        
        # Color analysis
        in_colors = set(c for row in input_grid for c in row)
        out_colors = set(c for row in output_grid for c in row)
        
        patterns['in_colors'] = in_colors
        patterns['out_colors'] = out_colors
        patterns['color_mapping'] = self._find_color_mapping(input_grid, output_grid)
        
        # Transformation detection
        patterns['is_rotation'] = self._check_rotation(input_grid, output_grid)
        patterns['is_reflection'] = self._check_reflection(input_grid, output_grid)
        patterns['is_fill'] = self._check_fill(input_grid, output_grid)
        
        return patterns
    
    def _find_color_mapping(self, in_grid: List[List[int]], out_grid: List[List[int]]) -> Dict[int, int]:
        """Find color mapping between input and output."""
        mapping = {}
        
        in_h, in_w = len(in_grid), len(in_grid[0]) if in_grid else 0
        out_h, out_w = len(out_grid), len(out_grid[0]) if out_grid else 0
        
        if in_h == out_h and in_w == out_w:
            for i in range(in_h):
                for j in range(in_w):
                    in_c = in_grid[i][j]
                    out_c = out_grid[i][j]
                    if in_c not in mapping:
                        mapping[in_c] = out_c
        
        return mapping
    
    def _check_rotation(self, in_grid: List[List[int]], out_grid: List[List[int]]) -> Optional[int]:
        """Check if output is a rotation of input."""
        for angle in [90, 180, 270]:
            rotated = self._rotate(in_grid, angle)
            if rotated == out_grid:
                return angle
        return None
    
    def _check_reflection(self, in_grid: List[List[int]], out_grid: List[List[int]]) -> Optional[str]:
        """Check if output is a reflection of input."""
        # Horizontal flip
        h_flip = [row[::-1] for row in in_grid]
        if h_flip == out_grid:
            return 'horizontal'
        
        # Vertical flip
        v_flip = in_grid[::-1]
        if v_flip == out_grid:
            return 'vertical'
        
        return None
    
    def _check_fill(self, in_grid: List[List[int]], out_grid: List[List[int]]) -> bool:
        """Check if transformation involves flood fill."""
        # Simple heuristic: same grid but some regions filled
        if len(in_grid) != len(out_grid):
            return False
        
        changes = 0
        for i in range(len(in_grid)):
            for j in range(len(in_grid[0])):
                if in_grid[i][j] != out_grid[i][j]:
                    changes += 1
        
        # If changes are clustered, likely a fill
        return 0 < changes < len(in_grid) * len(in_grid[0]) * 0.5
    
    def _rotate(self, grid: List[List[int]], angle: int) -> List[List[int]]:
        """Rotate grid by angle (90, 180, 270)."""
        if angle == 90:
            return [list(row) for row in zip(*grid[::-1])]
        elif angle == 180:
            return [row[::-1] for row in grid[::-1]]
        elif angle == 270:
            return [list(row) for row in zip(*grid)][::-1]
        return grid


class AbstractionEngine:
    """
    Creates abstract representations of ARC patterns.
    
    Extracts the underlying rule from examples.
    """
    
    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.learned_rules: List[Dict] = []
    
    def abstract(self, task: ARCTask) -> Dict[str, Any]:
        """Create abstract representation of task pattern."""
        all_patterns = []
        
        for in_grid, out_grid in task.train_examples:
            patterns = self.pattern_detector.analyze(in_grid, out_grid)
            all_patterns.append(patterns)
        
        # Find consistent patterns across all examples
        consistent = self._find_consistent_patterns(all_patterns)
        
        # Create abstract rule
        rule = {
            "task_id": task.task_id,
            "consistent_patterns": consistent,
            "transformation_type": self._infer_transformation_type(consistent),
        }
        
        self.learned_rules.append(rule)
        return rule
    
    def _find_consistent_patterns(self, all_patterns: List[Dict]) -> Dict[str, Any]:
        """Find patterns consistent across all examples."""
        if not all_patterns:
            return {}
        
        consistent = {}
        
        # Check each pattern type
        keys = all_patterns[0].keys()
        for key in keys:
            values = [p.get(key) for p in all_patterns]
            
            # Check if all values are the same
            if all(v == values[0] for v in values):
                consistent[key] = values[0]
        
        return consistent
    
    def _infer_transformation_type(self, patterns: Dict) -> str:
        """Infer the type of transformation."""
        if patterns.get('is_rotation'):
            return f"rotation_{patterns['is_rotation']}"
        if patterns.get('is_reflection'):
            return f"reflection_{patterns['is_reflection']}"
        if patterns.get('color_mapping'):
            return "color_mapping"
        if patterns.get('doubled'):
            return "scale_up"
        if patterns.get('halved'):
            return "scale_down"
        if patterns.get('is_fill'):
            return "fill"
        return "unknown"


class ARCSolverV2:
    """
    Solves ARC tasks using abstraction and pattern matching.
    """
    
    def __init__(self):
        self.abstraction = AbstractionEngine()
        self.pattern_detector = PatternDetector()
        self.solved_count = 0
        self.total_count = 0
    
    def solve(self, task: ARCTask) -> List[ARCPrediction]:
        """Solve an ARC task."""
        self.total_count += 1
        
        # Abstract the pattern from training examples
        rule = self.abstraction.abstract(task)
        
        predictions = []
        for i, (test_input, test_output) in enumerate(task.test_examples):
            # Apply learned transformation
            predicted = self._apply_transformation(test_input, rule)
            
            # Check if correct (for validation)
            confidence = 0.5
            if test_output:
                if predicted == test_output:
                    confidence = 1.0
                    self.solved_count += 1
                else:
                    # Partial match
                    confidence = self._compute_similarity(predicted, test_output)
            
            predictions.append(ARCPrediction(
                task_id=task.task_id,
                test_index=i,
                predicted_output=predicted,
                confidence=confidence,
                reasoning=f"Applied {rule['transformation_type']}",
            ))
        
        return predictions
    
    def _apply_transformation(self, input_grid: List[List[int]], rule: Dict) -> List[List[int]]:
        """Apply the learned transformation to input."""
        trans_type = rule.get('transformation_type', 'unknown')
        
        if trans_type.startswith('rotation_'):
            angle = int(trans_type.split('_')[1])
            return self.pattern_detector._rotate(input_grid, angle)
        
        if trans_type == 'reflection_horizontal':
            return [row[::-1] for row in input_grid]
        
        if trans_type == 'reflection_vertical':
            return input_grid[::-1]
        
        if trans_type == 'color_mapping':
            mapping = rule['consistent_patterns'].get('color_mapping', {})
            return [[mapping.get(c, c) for c in row] for row in input_grid]
        
        if trans_type == 'scale_up':
            return self._scale_grid(input_grid, 2)
        
        if trans_type == 'scale_down':
            return self._scale_grid(input_grid, 0.5)
        
        # Default: return input unchanged
        return input_grid
    
    def _scale_grid(self, grid: List[List[int]], factor: float) -> List[List[int]]:
        """Scale grid by factor."""
        h, w = len(grid), len(grid[0]) if grid else 0
        new_h, new_w = int(h * factor), int(w * factor)
        
        result = []
        for i in range(new_h):
            row = []
            for j in range(new_w):
                row.append(grid[int(i / factor)][int(j / factor)])
            result.append(row)
        return result
    
    def _compute_similarity(self, pred: List[List[int]], actual: List[List[int]]) -> float:
        """Compute similarity between prediction and actual."""
        if len(pred) != len(actual):
            return 0.0
        
        if not pred or not actual:
            return 0.0
        
        if len(pred[0]) != len(actual[0]):
            return 0.0
        
        matches = 0
        total = 0
        for i in range(len(pred)):
            for j in range(len(pred[0])):
                total += 1
                if pred[i][j] == actual[i][j]:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def get_accuracy(self) -> float:
        """Get overall accuracy."""
        return self.solved_count / self.total_count if self.total_count > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "solved": self.solved_count,
            "total": self.total_count,
            "accuracy": self.get_accuracy(),
            "rules_learned": len(self.abstraction.learned_rules),
        }


class ARCBenchmarkV2:
    """
    Complete ARC-AGI benchmark runner.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.solver = ARCSolverV2()
        self.data_dir = Path(data_dir) if data_dir else None
        self.tasks: List[ARCTask] = []
        self.results: List[Dict] = []
    
    def load_task(self, task_data: Dict) -> ARCTask:
        """Load a single ARC task from JSON data."""
        train = [(ex['input'], ex['output']) for ex in task_data.get('train', [])]
        test = [(ex['input'], ex.get('output')) for ex in task_data.get('test', [])]
        
        return ARCTask(
            task_id=task_data.get('id', 'unknown'),
            train_examples=train,
            test_examples=test,
        )
    
    def add_sample_tasks(self) -> None:
        """Add sample ARC-like tasks for testing."""
        # Task 1: Color swap
        self.tasks.append(ARCTask(
            task_id="color_swap",
            train_examples=[
                ([[1, 2], [2, 1]], [[2, 1], [1, 2]]),
                ([[1, 1], [2, 2]], [[2, 2], [1, 1]]),
            ],
            test_examples=[
                ([[1, 2, 1], [2, 1, 2]], [[2, 1, 2], [1, 2, 1]]),
            ],
        ))
        
        # Task 2: Horizontal flip
        self.tasks.append(ARCTask(
            task_id="h_flip",
            train_examples=[
                ([[1, 2, 3]], [[3, 2, 1]]),
                ([[4, 5]], [[5, 4]]),
            ],
            test_examples=[
                ([[1, 2, 3, 4]], [[4, 3, 2, 1]]),
            ],
        ))
        
        # Task 3: Scale up 2x
        self.tasks.append(ARCTask(
            task_id="scale_2x",
            train_examples=[
                ([[1]], [[1, 1], [1, 1]]),
                ([[2, 3]], [[2, 2, 3, 3], [2, 2, 3, 3]]),
            ],
            test_examples=[
                ([[1, 2], [3, 4]], [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),
            ],
        ))
    
    def run(self) -> Dict[str, Any]:
        """Run the benchmark on all tasks."""
        if not self.tasks:
            self.add_sample_tasks()
        
        for task in self.tasks:
            predictions = self.solver.solve(task)
            
            for pred in predictions:
                self.results.append({
                    "task_id": pred.task_id,
                    "confidence": pred.confidence,
                    "correct": pred.confidence == 1.0,
                    "reasoning": pred.reasoning,
                })
        
        return self.solver.get_stats()
    
    def report(self) -> str:
        """Generate benchmark report."""
        stats = self.solver.get_stats()
        
        report = ["=" * 50]
        report.append("ARC-AGI BENCHMARK REPORT")
        report.append("=" * 50)
        report.append(f"Tasks Solved: {stats['solved']}/{stats['total']}")
        report.append(f"Accuracy: {stats['accuracy']*100:.1f}%")
        report.append(f"Rules Learned: {stats['rules_learned']}")
        report.append("")
        report.append("Results by Task:")
        
        for result in self.results:
            status = "✅" if result['correct'] else "❌"
            report.append(f"  {status} {result['task_id']}: {result['reasoning']}")
        
        return "\n".join(report)
