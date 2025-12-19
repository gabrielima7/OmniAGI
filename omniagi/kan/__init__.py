"""
Kolmogorov-Arnold Networks (KAN) Module.

Efficient implementation for interpretable neural networks
that can achieve AGI-level pattern recognition with
symbolic extraction capabilities.
"""

from omniagi.kan.efficient_kan import (
    KANLayer,
    EfficientKAN,
    RadialBasisKAN,
)
from omniagi.kan.symbolic_kan import (
    SymbolicKAN,
    extract_symbolic_formula,
)

__all__ = [
    "KANLayer",
    "EfficientKAN",
    "RadialBasisKAN",
    "SymbolicKAN",
    "extract_symbolic_formula",
]
