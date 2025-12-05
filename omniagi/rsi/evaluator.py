"""
Capability Evaluator - Benchmarking and capability measurement.

Measures AI capabilities across various dimensions to
track improvement and identify weaknesses.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

logger = structlog.get_logger()


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""
    
    REASONING = auto()      # Logical reasoning
    CODE = auto()           # Code generation/understanding
    MATH = auto()           # Mathematical reasoning
    LANGUAGE = auto()       # Language understanding
    KNOWLEDGE = auto()      # Factual knowledge
    PLANNING = auto()       # Planning and goal decomposition
    LEARNING = auto()       # Learning efficiency
    CREATIVITY = auto()     # Creative generation


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    benchmark_id: str
    score: float  # 0-1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "score": self.score,
            "timestamp": self.timestamp,
            "details": self.details,
            "errors": self.errors,
        }


@dataclass
class Benchmark:
    """A capability benchmark."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    category: BenchmarkCategory = BenchmarkCategory.REASONING
    description: str = ""
    
    # Test cases
    test_cases: list[dict] = field(default_factory=list)
    
    # Scoring
    weight: float = 1.0
    baseline_score: float = 0.0  # Expected human baseline
    
    # History
    results: list[BenchmarkResult] = field(default_factory=list)
    
    @property
    def latest_score(self) -> float:
        """Get the most recent score."""
        return self.results[-1].score if self.results else 0.0
    
    @property
    def average_score(self) -> float:
        """Get average score across all runs."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)
    
    @property
    def trend(self) -> float:
        """Get improvement trend (-1 to 1)."""
        if len(self.results) < 2:
            return 0.0
        recent = self.results[-3:]
        if len(recent) < 2:
            return 0.0
        return recent[-1].score - recent[0].score
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.name,
            "description": self.description,
            "test_cases": self.test_cases,
            "weight": self.weight,
            "baseline_score": self.baseline_score,
            "results": [r.to_dict() for r in self.results[-10:]],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Benchmark":
        benchmark = cls(
            id=data.get("id", str(uuid4())[:8]),
            name=data["name"],
            category=BenchmarkCategory[data.get("category", "REASONING")],
            description=data.get("description", ""),
            test_cases=data.get("test_cases", []),
            weight=data.get("weight", 1.0),
            baseline_score=data.get("baseline_score", 0.0),
        )
        benchmark.results = [
            BenchmarkResult(**r) for r in data.get("results", [])
        ]
        return benchmark


class CapabilityEvaluator:
    """
    Evaluates AI capabilities across benchmarks.
    
    Tracks performance over time to measure improvement
    and identify areas needing attention.
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        storage_path: Path | str | None = None,
    ):
        self.engine = engine
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._benchmarks: dict[str, Benchmark] = {}
        self._init_default_benchmarks()
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info(
            "Capability Evaluator initialized",
            benchmarks=len(self._benchmarks),
        )
    
    def _init_default_benchmarks(self) -> None:
        """Initialize default benchmarks."""
        defaults = [
            Benchmark(
                name="Logical Reasoning",
                category=BenchmarkCategory.REASONING,
                description="Basic logical reasoning tasks",
                test_cases=[
                    {"input": "If A implies B, and A is true, what is B?", "expected": "true"},
                    {"input": "All dogs are animals. Fido is a dog. Is Fido an animal?", "expected": "yes"},
                ],
                baseline_score=0.9,
            ),
            Benchmark(
                name="Code Generation",
                category=BenchmarkCategory.CODE,
                description="Generate correct code from descriptions",
                test_cases=[
                    {"input": "Write a Python function to reverse a string", "contains": "def"},
                    {"input": "Write a function to find the maximum of two numbers", "contains": "max"},
                ],
                baseline_score=0.7,
            ),
            Benchmark(
                name="Mathematical Reasoning",
                category=BenchmarkCategory.MATH,
                description="Solve math word problems",
                test_cases=[
                    {"input": "If x + 5 = 10, what is x?", "expected": "5"},
                    {"input": "What is 15% of 200?", "expected": "30"},
                ],
                baseline_score=0.8,
            ),
            Benchmark(
                name="Planning",
                category=BenchmarkCategory.PLANNING,
                description="Decompose goals into steps",
                test_cases=[
                    {"input": "How would you make a sandwich?", "min_steps": 3},
                    {"input": "Plan a trip from A to B", "min_steps": 2},
                ],
                baseline_score=0.6,
            ),
        ]
        
        for b in defaults:
            self._benchmarks[b.id] = b
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a custom benchmark."""
        self._benchmarks[benchmark.id] = benchmark
        self._save()
    
    def run_benchmark(
        self,
        benchmark_id: str,
        evaluator_fn: Callable[[str, dict], float] = None,
    ) -> BenchmarkResult | None:
        """
        Run a single benchmark.
        
        Args:
            benchmark_id: ID of benchmark to run.
            evaluator_fn: Custom evaluation function.
            
        Returns:
            Benchmark result.
        """
        if benchmark_id not in self._benchmarks:
            return None
        
        benchmark = self._benchmarks[benchmark_id]
        
        if not self.engine or not self.engine.is_loaded:
            # Return placeholder result
            result = BenchmarkResult(
                benchmark_id=benchmark_id,
                score=0.0,
                errors=["No engine available"],
            )
            benchmark.results.append(result)
            self._save()
            return result
        
        scores = []
        errors = []
        details = {}
        
        for i, case in enumerate(benchmark.test_cases):
            try:
                # Generate response
                response = self.engine.generate(
                    case["input"],
                    max_tokens=200,
                )
                response_text = response.text.strip().lower()
                
                # Evaluate
                if evaluator_fn:
                    score = evaluator_fn(response_text, case)
                else:
                    score = self._default_evaluate(response_text, case)
                
                scores.append(score)
                details[f"case_{i}"] = {
                    "input": case["input"][:50],
                    "score": score,
                }
                
            except Exception as e:
                errors.append(f"Case {i}: {str(e)}")
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            score=avg_score,
            details=details,
            errors=errors,
        )
        
        benchmark.results.append(result)
        self._save()
        
        logger.info(
            "Benchmark completed",
            benchmark=benchmark.name,
            score=avg_score,
        )
        
        return result
    
    def _default_evaluate(self, response: str, case: dict) -> float:
        """Default evaluation logic."""
        # Exact match
        if "expected" in case:
            expected = str(case["expected"]).lower()
            if expected in response:
                return 1.0
            return 0.0
        
        # Contains check
        if "contains" in case:
            if case["contains"].lower() in response:
                return 1.0
            return 0.0
        
        # Minimum steps
        if "min_steps" in case:
            # Count numbered steps or bullet points
            steps = response.count("\n") + 1
            if steps >= case["min_steps"]:
                return min(1.0, steps / (case["min_steps"] * 2))
            return steps / case["min_steps"]
        
        # Default: check for non-empty response
        return 0.5 if len(response) > 10 else 0.0
    
    def run_all_benchmarks(self) -> dict[str, BenchmarkResult]:
        """Run all benchmarks."""
        results = {}
        for benchmark_id in self._benchmarks:
            result = self.run_benchmark(benchmark_id)
            if result:
                results[benchmark_id] = result
        return results
    
    def get_capability_profile(self) -> dict[str, float]:
        """Get overall capability profile by category."""
        profile = {cat.name: [] for cat in BenchmarkCategory}
        
        for benchmark in self._benchmarks.values():
            if benchmark.latest_score > 0:
                profile[benchmark.category.name].append(
                    benchmark.latest_score * benchmark.weight
                )
        
        return {
            category: sum(scores) / len(scores) if scores else 0.0
            for category, scores in profile.items()
        }
    
    def get_weakest_areas(self, n: int = 3) -> list[tuple[str, float]]:
        """Get the n weakest capability areas."""
        profile = self.get_capability_profile()
        sorted_areas = sorted(
            profile.items(),
            key=lambda x: x[1],
        )
        return sorted_areas[:n]
    
    def get_improvement_rate(self) -> dict[str, float]:
        """Get improvement rate by benchmark."""
        rates = {}
        for b_id, benchmark in self._benchmarks.items():
            rates[b_id] = benchmark.trend
        return rates
    
    def get_overall_score(self) -> float:
        """Get weighted overall capability score."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for benchmark in self._benchmarks.values():
            if benchmark.latest_score > 0:
                weighted_sum += benchmark.latest_score * benchmark.weight
                total_weight += benchmark.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def compare_to_baseline(self) -> dict[str, float]:
        """Compare current scores to human baselines."""
        comparison = {}
        for b_id, benchmark in self._benchmarks.items():
            if benchmark.baseline_score > 0:
                comparison[benchmark.name] = (
                    benchmark.latest_score - benchmark.baseline_score
                )
        return comparison
    
    def get_stats(self) -> dict:
        """Get evaluator statistics."""
        return {
            "total_benchmarks": len(self._benchmarks),
            "overall_score": self.get_overall_score(),
            "capability_profile": self.get_capability_profile(),
            "weakest_areas": self.get_weakest_areas(),
            "improvement_rates": self.get_improvement_rate(),
            "vs_baseline": self.compare_to_baseline(),
        }
    
    def __len__(self) -> int:
        return len(self._benchmarks)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "benchmarks": {k: v.to_dict() for k, v in self._benchmarks.items()},
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        for k, v in data.get("benchmarks", {}).items():
            self._benchmarks[k] = Benchmark.from_dict(v)
