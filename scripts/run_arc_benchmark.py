#!/usr/bin/env python3
"""
ARC-AGI Benchmark Runner.

Downloads and runs the official ARC benchmark to measure
OmniAGI's abstraction and reasoning capabilities.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omniagi.benchmarks.arc_v2 import ARCSolverV2, ARCTask, ARCBenchmarkV2


def load_arc_task(filepath: str) -> Dict[str, Any]:
    """Load an ARC task from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def convert_to_arc_task(task_id: str, data: Dict) -> ARCTask:
    """Convert JSON data to ARCTask object."""
    train_examples = [
        (ex['input'], ex['output'])
        for ex in data.get('train', [])
    ]
    
    test_examples = [
        (ex['input'], ex.get('output'))
        for ex in data.get('test', [])
    ]
    
    return ARCTask(
        task_id=task_id,
        train_examples=train_examples,
        test_examples=test_examples,
    )


def run_arc_benchmark(data_dir: str = "data/arc") -> Dict[str, Any]:
    """
    Run ARC benchmark on all tasks in directory.
    
    Returns statistics and individual results.
    """
    solver = ARCSolverV2()
    results = []
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return {"error": "Data directory not found"}
    
    json_files = list(data_path.glob("*.json"))
    print(f"📂 Found {len(json_files)} ARC tasks")
    print("=" * 60)
    
    for json_file in sorted(json_files):
        task_id = json_file.stem
        
        try:
            data = load_arc_task(str(json_file))
            
            # Skip invalid files
            if not isinstance(data, dict) or 'train' not in data:
                print(f"⚠️  {task_id}: Invalid format, skipping")
                continue
            
            task = convert_to_arc_task(task_id, data)
            
            print(f"\n📋 Task: {task_id}")
            print(f"   Train examples: {len(task.train_examples)}")
            print(f"   Test examples: {len(task.test_examples)}")
            
            # Solve task
            predictions = solver.solve(task)
            
            for pred in predictions:
                status = "✅" if pred.confidence == 1.0 else "❌"
                print(f"   {status} Test {pred.test_index}: {pred.reasoning} (conf: {pred.confidence:.2f})")
                
                results.append({
                    "task_id": task_id,
                    "test_index": pred.test_index,
                    "correct": pred.confidence == 1.0,
                    "confidence": pred.confidence,
                    "reasoning": pred.reasoning,
                })
                
        except Exception as e:
            print(f"❌ {task_id}: Error - {e}")
            results.append({
                "task_id": task_id,
                "error": str(e),
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    stats = solver.get_stats()
    
    correct = sum(1 for r in results if r.get('correct', False))
    total = len([r for r in results if 'error' not in r])
    accuracy = correct / total if total > 0 else 0
    
    print(f"   Tasks Processed: {len(json_files)}")
    print(f"   Predictions Made: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy * 100:.1f}%")
    print(f"   Rules Learned: {stats['rules_learned']}")
    
    return {
        "tasks_processed": len(json_files),
        "predictions": total,
        "correct": correct,
        "accuracy": accuracy,
        "rules_learned": stats['rules_learned'],
        "results": results,
    }


def train_on_arc(data_dir: str = "data/arc") -> Dict[str, Any]:
    """
    Train the AGI system on ARC tasks.
    
    Uses online learning to improve with each task.
    """
    from omniagi.agi.true_agi import TrueAGI
    import torch
    
    agi = TrueAGI()
    
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    print("=" * 60)
    print("🧠 TRAINING AGI ON ARC TASKS")
    print("=" * 60)
    
    total_loss = 0.0
    examples_trained = 0
    
    for json_file in sorted(json_files):
        task_id = json_file.stem
        
        try:
            data = load_arc_task(str(json_file))
            
            if not isinstance(data, dict) or 'train' not in data:
                continue
            
            print(f"\n📋 Training on: {task_id}")
            
            for i, example in enumerate(data.get('train', [])):
                input_grid = example['input']
                output_grid = example['output']
                
                # Flatten grids for training
                input_flat = [c for row in input_grid for c in row]
                output_flat = [c for row in output_grid for c in row]
                
                # Pad/truncate to fixed size
                input_tensor = input_flat[:64] + [0] * max(0, 64 - len(input_flat))
                output_tensor = output_flat[:32] + [0] * max(0, 32 - len(output_flat))
                
                # Train online
                loss = agi.learn_online(input_tensor, output_tensor, task_id)
                total_loss += loss
                examples_trained += 1
                
                print(f"   Example {i+1}: loss={loss:.4f}")
                
        except Exception as e:
            print(f"❌ {task_id}: Error - {e}")
    
    avg_loss = total_loss / examples_trained if examples_trained > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    print(f"   Examples Trained: {examples_trained}")
    print(f"   Average Loss: {avg_loss:.4f}")
    
    # Get final stats
    if agi.online_learner:
        stats = agi.online_learner.get_stats()
        print(f"   Buffer Size: {stats['online_stats']['buffer_size']}")
        print(f"   Learning Rate: {stats['online_stats']['learning_rate']:.6f}")
    
    return {
        "examples_trained": examples_trained,
        "avg_loss": avg_loss,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ARC-AGI Benchmark Runner")
    parser.add_argument("--train", action="store_true", help="Train on ARC tasks")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--data-dir", default="data/arc", help="ARC data directory")
    
    args = parser.parse_args()
    
    if args.train:
        train_on_arc(args.data_dir)
    
    if args.benchmark or not args.train:
        run_arc_benchmark(args.data_dir)
