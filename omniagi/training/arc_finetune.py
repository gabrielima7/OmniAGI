"""
ARC Fine-Tuning Dataset Generator.

Creates training examples specifically for ARC-style reasoning tasks.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import structlog

from omniagi.training.finetune import FinetuneDataset, FinetuneExample

logger = structlog.get_logger()


def grid_to_text(grid: list[list[int]]) -> str:
    """Convert grid to text representation."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def create_arc_finetuning_dataset() -> FinetuneDataset:
    """
    Create a comprehensive dataset for ARC fine-tuning.
    
    Includes examples for:
    - Mirror/reflection
    - Rotation
    - Color swapping
    - Pattern continuation
    - Scaling
    - Counting
    """
    dataset = FinetuneDataset("arc_training")
    
    # 1. MIRROR EXAMPLES
    for i in range(50):
        size = random.randint(3, 5)
        inp = [[random.randint(0, 5) for _ in range(size)] for _ in range(size)]
        out = [row[::-1] for row in inp]  # Horizontal mirror
        
        prompt = f"""Task: Look at the examples and predict the output.

Example 1:
Input:
1 2 3
4 5 6
7 8 9
Output:
3 2 1
6 5 4
9 8 7

Example 2:
Input:
1 0 2
3 1 4
Output:
2 0 1
4 1 3

Now predict:
Input:
{grid_to_text(inp)}
Output:
"""
        completion = grid_to_text(out)
        dataset.add_example(prompt, completion, "mirror")
    
    # 2. ROTATION EXAMPLES (90 degrees clockwise)
    for i in range(50):
        size = random.randint(3, 4)
        inp = [[random.randint(0, 5) for _ in range(size)] for _ in range(size)]
        out = [[inp[size-1-j][i] for j in range(size)] for i in range(size)]
        
        prompt = f"""Task: Rotate the grid 90 degrees clockwise.

Example 1:
Input:
1 2
3 4
Output:
3 1
4 2

Example 2:
Input:
1 2 3
4 5 6
7 8 9
Output:
7 4 1
8 5 2
9 6 3

Now predict:
Input:
{grid_to_text(inp)}
Output:
"""
        completion = grid_to_text(out)
        dataset.add_example(prompt, completion, "rotation")
    
    # 3. COLOR SWAP EXAMPLES
    for i in range(50):
        size = random.randint(3, 5)
        c1, c2 = random.sample(range(1, 6), 2)
        inp = [[random.choice([0, c1, c2]) for _ in range(size)] for _ in range(size)]
        out = [[c2 if c == c1 else (c1 if c == c2 else c) for c in row] for row in inp]
        
        prompt = f"""Task: Swap colors {c1} and {c2} in the grid.

Example 1 (swap 1 and 2):
Input:
1 0 2
2 1 0
Output:
2 0 1
1 2 0

Now swap colors {c1} and {c2}:
Input:
{grid_to_text(inp)}
Output:
"""
        completion = grid_to_text(out)
        dataset.add_example(prompt, completion, "color_swap")
    
    # 4. SCALING EXAMPLES (2x)
    for i in range(30):
        size = 2
        inp = [[random.randint(0, 5) for _ in range(size)] for _ in range(size)]
        out = []
        for row in inp:
            scaled_row = []
            for c in row:
                scaled_row.extend([c, c])
            out.append(scaled_row)
            out.append(scaled_row[:])
        
        prompt = f"""Task: Scale the grid 2x (each cell becomes 2x2).

Example:
Input:
1 2
3 4
Output:
1 1 2 2
1 1 2 2
3 3 4 4
3 3 4 4

Now scale 2x:
Input:
{grid_to_text(inp)}
Output:
"""
        completion = grid_to_text(out)
        dataset.add_example(prompt, completion, "scaling")
    
    # 5. COUNTING EXAMPLES
    for i in range(30):
        size = 4
        color = random.randint(1, 5)
        count = random.randint(1, 8)
        inp = [[0 for _ in range(size)] for _ in range(size)]
        positions = random.sample([(r, c) for r in range(size) for c in range(size)], count)
        for r, c in positions:
            inp[r][c] = color
        out = [[color] * count]
        
        prompt = f"""Task: Count colored cells and output them in a row.

Example:
Input:
0 1 0
1 0 1
0 0 0
Output:
1 1 1

Count the colored cells:
Input:
{grid_to_text(inp)}
Output:
"""
        completion = grid_to_text(out)
        dataset.add_example(prompt, completion, "counting")
    
    # 6. PATTERN REASONING
    for i in range(40):
        # Simple diagonal pattern
        size = 5
        inp = [[0 for _ in range(size)] for _ in range(size)]
        out = [[0 for _ in range(size)] for _ in range(size)]
        color = random.randint(1, 5)
        
        # Partial diagonal in input
        for j in range(3):
            inp[j][j] = color
            out[j][j] = color
        
        # Complete diagonal in output
        for j in range(3, size):
            out[j][j] = color
        
        prompt = f"""Task: Continue the pattern.

Example:
Input:
1 0 0 0
0 1 0 0
0 0 0 0
0 0 0 0
Output:
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1

Continue the pattern:
Input:
{grid_to_text(inp)}
Output:
"""
        completion = grid_to_text(out)
        dataset.add_example(prompt, completion, "pattern")
    
    logger.info(
        "ARC fine-tuning dataset created",
        total_examples=len(dataset),
    )
    
    return dataset


def create_arc_reasoning_dataset() -> FinetuneDataset:
    """
    Create a dataset that teaches step-by-step reasoning for ARC.
    """
    dataset = FinetuneDataset("arc_reasoning")
    
    # Teach how to analyze ARC problems
    examples = [
        (
            """Analyze this ARC task:
Input: 1 2 3
Output: 3 2 1

What is the transformation?""",
            """Step 1: Compare input and output sizes. Both are 1x3.
Step 2: Look at element positions. 
- Input[0] = 1, Output[2] = 1
- Input[1] = 2, Output[1] = 2  
- Input[2] = 3, Output[0] = 3
Step 3: The elements are reversed.
Rule: Horizontal mirror (reverse each row)."""
        ),
        (
            """Analyze this ARC task:
Input:
1 0
0 1
Output:
0 1
1 0

What is the transformation?""",
            """Step 1: Compare sizes. Both 2x2.
Step 2: Compare elements by position.
- (0,0): 1 → 0, (0,1): 0 → 1
- (1,0): 0 → 1, (1,1): 1 → 0
Step 3: Each 1 becomes 0, each 0 becomes 1.
Rule: Invert colors (swap 0 and 1)."""
        ),
        (
            """Given this transformation rule: "Rotate 90 degrees clockwise"
Apply to:
1 2
3 4

What is the output?""",
            """Step 1: 90 degree clockwise rotation formula:
- New[row][col] = Old[n-1-col][row] where n is grid size
Step 2: For 2x2 grid:
- New[0][0] = Old[1][0] = 3
- New[0][1] = Old[0][0] = 1
- New[1][0] = Old[1][1] = 4
- New[1][1] = Old[0][1] = 2
Output:
3 1
4 2"""
        ),
    ]
    
    for prompt, completion in examples:
        dataset.add_example(prompt, completion, "reasoning")
    
    return dataset


def export_arc_datasets():
    """Export all ARC datasets for training."""
    
    # Create datasets
    arc_data = create_arc_finetuning_dataset()
    reasoning_data = create_arc_reasoning_dataset()
    
    # Save
    arc_data.save()
    reasoning_data.save()
    
    # Export to JSONL
    arc_path = arc_data.export_to_jsonl()
    reasoning_path = reasoning_data.export_to_jsonl()
    
    return {
        "arc_training": {
            "count": len(arc_data),
            "path": str(arc_path),
        },
        "arc_reasoning": {
            "count": len(reasoning_data),
            "path": str(reasoning_path),
        },
    }
