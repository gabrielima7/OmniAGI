"""Benchmarks module."""

try:
    from omniagi.benchmarks.arc import ARCBenchmark, ARCGenerator, ARCTask
except ImportError:
    ARCBenchmark = None
    ARCGenerator = None
    ARCTask = None

try:
    from omniagi.benchmarks.arc_v2 import ARCBenchmarkV2, ARCSolverV2
except ImportError:
    ARCBenchmarkV2 = None
    ARCSolverV2 = None

__all__ = [
    "ARCBenchmark",
    "ARCGenerator",
    "ARCTask",
    "ARCBenchmarkV2",
    "ARCSolverV2",
]

