"""
Advanced ARC Solver with Program Synthesis.

Uses DSL-based program synthesis and more primitives
for improved ARC task solving.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from itertools import product
import copy

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class Grid:
    """Wrapper for ARC grids with utility methods."""
    
    def __init__(self, data: List[List[int]]):
        self.data = [row[:] for row in data]
        self.height = len(data)
        self.width = len(data[0]) if data else 0
    
    def __eq__(self, other):
        if isinstance(other, Grid):
            return self.data == other.data
        if isinstance(other, list):
            return self.data == other
        return False
    
    def copy(self) -> "Grid":
        return Grid([row[:] for row in self.data])
    
    def get(self, r: int, c: int) -> int:
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.data[r][c]
        return 0
    
    def set(self, r: int, c: int, val: int) -> None:
        if 0 <= r < self.height and 0 <= c < self.width:
            self.data[r][c] = val
    
    def colors(self) -> set:
        return set(c for row in self.data for c in row)
    
    def to_list(self) -> List[List[int]]:
        return self.data


# DSL Primitives
class DSL:
    """Domain Specific Language for ARC transformations."""
    
    @staticmethod
    def identity(grid: Grid) -> Grid:
        return grid.copy()
    
    @staticmethod
    def rotate_90(grid: Grid) -> Grid:
        return Grid([list(row) for row in zip(*grid.data[::-1])])
    
    @staticmethod
    def rotate_180(grid: Grid) -> Grid:
        return Grid([row[::-1] for row in grid.data[::-1]])
    
    @staticmethod
    def rotate_270(grid: Grid) -> Grid:
        return Grid([list(row) for row in zip(*grid.data)][::-1])
    
    @staticmethod
    def flip_horizontal(grid: Grid) -> Grid:
        return Grid([row[::-1] for row in grid.data])
    
    @staticmethod
    def flip_vertical(grid: Grid) -> Grid:
        return Grid(grid.data[::-1])
    
    @staticmethod
    def transpose(grid: Grid) -> Grid:
        return Grid([list(row) for row in zip(*grid.data)])
    
    @staticmethod
    def scale_2x(grid: Grid) -> Grid:
        result = []
        for row in grid.data:
            new_row = []
            for c in row:
                new_row.extend([c, c])
            result.append(new_row)
            result.append(new_row[:])
        return Grid(result)
    
    @staticmethod
    def scale_3x(grid: Grid) -> Grid:
        result = []
        for row in grid.data:
            new_row = []
            for c in row:
                new_row.extend([c, c, c])
            for _ in range(3):
                result.append(new_row[:])
        return Grid(result)
    
    @staticmethod
    def crop_to_nonzero(grid: Grid) -> Grid:
        """Crop to bounding box of non-zero elements."""
        min_r, max_r, min_c, max_c = grid.height, 0, grid.width, 0
        
        for r in range(grid.height):
            for c in range(grid.width):
                if grid.data[r][c] != 0:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
        
        if max_r < min_r:
            return Grid([[0]])
        
        return Grid([row[min_c:max_c+1] for row in grid.data[min_r:max_r+1]])
    
    @staticmethod
    def fill_color(grid: Grid, from_color: int, to_color: int) -> Grid:
        """Replace one color with another."""
        result = grid.copy()
        for r in range(result.height):
            for c in range(result.width):
                if result.data[r][c] == from_color:
                    result.data[r][c] = to_color
        return result
    
    @staticmethod
    def swap_colors(grid: Grid, c1: int, c2: int) -> Grid:
        """Swap two colors."""
        result = grid.copy()
        for r in range(result.height):
            for c in range(result.width):
                if result.data[r][c] == c1:
                    result.data[r][c] = c2
                elif result.data[r][c] == c2:
                    result.data[r][c] = c1
        return result
    
    @staticmethod
    def fill_enclosed(grid: Grid, fill_color: int) -> Grid:
        """Fill enclosed areas with a color."""
        result = grid.copy()
        
        # Find enclosed regions using flood fill from edges
        visited = [[False] * grid.width for _ in range(grid.height)]
        
        def flood_fill(r, c):
            if r < 0 or r >= grid.height or c < 0 or c >= grid.width:
                return
            if visited[r][c] or grid.data[r][c] != 0:
                return
            visited[r][c] = True
            flood_fill(r+1, c)
            flood_fill(r-1, c)
            flood_fill(r, c+1)
            flood_fill(r, c-1)
        
        # Flood fill from all edges
        for r in range(grid.height):
            flood_fill(r, 0)
            flood_fill(r, grid.width - 1)
        for c in range(grid.width):
            flood_fill(0, c)
            flood_fill(grid.height - 1, c)
        
        # Fill unvisited zeros
        for r in range(result.height):
            for c in range(result.width):
                if result.data[r][c] == 0 and not visited[r][c]:
                    result.data[r][c] = fill_color
        
        return result
    
    @staticmethod
    def outline(grid: Grid, outline_color: int) -> Grid:
        """Add outline around non-zero regions."""
        result = Grid([[0] * grid.width for _ in range(grid.height)])
        
        for r in range(grid.height):
            for c in range(grid.width):
                if grid.data[r][c] != 0:
                    result.data[r][c] = grid.data[r][c]
                else:
                    # Check if adjacent to non-zero
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid.height and 0 <= nc < grid.width:
                            if grid.data[nr][nc] != 0:
                                result.data[r][c] = outline_color
                                break
        
        return result
    
    @staticmethod
    def tile_2x2(grid: Grid) -> Grid:
        """Tile grid in 2x2 pattern."""
        result = []
        for _ in range(2):
            for row in grid.data:
                result.append(row * 2)
        return Grid(result)
    
    @staticmethod
    def extract_unique_colors(grid: Grid) -> List[int]:
        """Get unique colors in order of appearance."""
        seen = set()
        colors = []
        for row in grid.data:
            for c in row:
                if c not in seen:
                    seen.add(c)
                    colors.append(c)
        return colors


class AdvancedARCSolver:
    """
    Advanced ARC solver using program synthesis.
    
    Tries combinations of DSL primitives to find solutions.
    """
    
    def __init__(self):
        # All available primitives
        self.primitives = {
            "identity": DSL.identity,
            "rotate_90": DSL.rotate_90,
            "rotate_180": DSL.rotate_180,
            "rotate_270": DSL.rotate_270,
            "flip_h": DSL.flip_horizontal,
            "flip_v": DSL.flip_vertical,
            "transpose": DSL.transpose,
            "scale_2x": DSL.scale_2x,
            "scale_3x": DSL.scale_3x,
            "crop": DSL.crop_to_nonzero,
            "tile_2x2": DSL.tile_2x2,
        }
        
        self.solved = 0
        self.total = 0
    
    def solve(
        self,
        train_examples: List[Tuple[List[List[int]], List[List[int]]]],
        test_input: List[List[int]],
    ) -> Tuple[List[List[int]], float, str]:
        """Solve an ARC task."""
        self.total += 1
        
        # Try single primitives
        for name, func in self.primitives.items():
            if self._test_primitive(name, func, train_examples):
                result = func(Grid(test_input)).to_list()
                self.solved += 1
                return result, 0.95, f"Program: {name}"
        
        # Try pairs of primitives
        for name1, func1 in self.primitives.items():
            for name2, func2 in self.primitives.items():
                program_name = f"{name1} -> {name2}"
                
                def compose(g):
                    return func2(func1(g))
                
                if self._test_primitive(program_name, compose, train_examples):
                    result = compose(Grid(test_input)).to_list()
                    self.solved += 1
                    return result, 0.9, f"Program: {program_name}"
        
        # Try color operations
        colors = self._analyze_colors(train_examples)
        for c1, c2 in colors.get("swaps", []):
            def color_swap(g, a=c1, b=c2):
                return DSL.swap_colors(g, a, b)
            
            if self._test_primitive(f"swap_{c1}_{c2}", color_swap, train_examples):
                result = color_swap(Grid(test_input)).to_list()
                self.solved += 1
                return result, 0.85, f"Program: swap_colors({c1}, {c2})"
        
        # Fallback
        return test_input, 0.1, "identity (no solution found)"
    
    def _test_primitive(
        self,
        name: str,
        func: Callable,
        examples: List[Tuple],
    ) -> bool:
        """Test if primitive works on all examples."""
        for inp, out in examples:
            try:
                result = func(Grid(inp))
                if result.to_list() != out:
                    return False
            except Exception:
                return False
        return True
    
    def _analyze_colors(self, examples: List[Tuple]) -> Dict[str, Any]:
        """Analyze color patterns."""
        swaps = []
        
        for inp, out in examples:
            in_colors = set(c for row in inp for c in row)
            out_colors = set(c for row in out for c in row)
            
            # Look for potential swaps
            for c1 in in_colors:
                for c2 in in_colors:
                    if c1 < c2:
                        swaps.append((c1, c2))
        
        return {"swaps": list(set(swaps))[:5]}  # Limit
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "solved": self.solved,
            "total": self.total,
            "accuracy": self.solved / self.total if self.total > 0 else 0,
        }


def run_benchmark(data_dir: str = "data/arc") -> Dict[str, Any]:
    """Run improved benchmark."""
    import json
    from pathlib import Path
    
    solver = AdvancedARCSolver()
    
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob("*.json"))[:100]  # Test on 100 tasks
    
    print("=" * 60)
    print("🧠 ADVANCED ARC SOLVER")
    print("=" * 60)
    
    correct = 0
    
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'train' not in data:
                continue
            
            train = [(ex['input'], ex['output']) for ex in data['train']]
            
            for test_ex in data.get('test', []):
                test_input = test_ex['input']
                test_output = test_ex.get('output')
                
                predicted, conf, reason = solver.solve(train, test_input)
                
                if test_output and predicted == test_output:
                    correct += 1
                    print(f"✅ {json_file.stem}: {reason}")
                
        except Exception as e:
            pass
    
    stats = solver.get_stats()
    print(f"\n📊 Results: {correct}/{stats['total']} ({stats['accuracy']*100:.1f}%)")
    
    return stats


if __name__ == "__main__":
    run_benchmark()
