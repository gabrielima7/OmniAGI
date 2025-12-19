"""
Symbolic KAN - Extract symbolic formulas from trained KAN networks.

This module enables interpretability by converting learned KAN
activation functions into human-readable mathematical expressions.

Key capabilities:
1. Extract symbolic formulas from KAN layers
2. Simplify using SymPy
3. Provide interpretable explanations
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field

# Check for dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import sympy as sp
    from sympy import symbols, simplify, expand, N
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

logger = logging.getLogger(__name__)


@dataclass
class SymbolicFormula:
    """A symbolic mathematical formula extracted from KAN."""
    
    expression: str  # Human-readable expression
    sympy_expr: object = None  # SymPy expression object
    variables: List[str] = field(default_factory=list)
    complexity: int = 0  # Number of operations
    accuracy: float = 0.0  # How well it fits the learned function
    
    def __str__(self):
        return self.expression
    
    def evaluate(self, **kwargs) -> float:
        """Evaluate the formula with given variable values."""
        if self.sympy_expr is None:
            raise ValueError("No SymPy expression available")
        return float(self.sympy_expr.subs(kwargs))


# Common basis functions for symbolic fitting
SYMBOLIC_BASIS = [
    ("constant", lambda x: 1),
    ("linear", lambda x: x),
    ("quadratic", lambda x: x**2),
    ("cubic", lambda x: x**3),
    ("sqrt", lambda x: sp.sqrt(sp.Abs(x)) if SYMPY_AVAILABLE else x**0.5),
    ("sin", lambda x: sp.sin(x) if SYMPY_AVAILABLE else None),
    ("cos", lambda x: sp.cos(x) if SYMPY_AVAILABLE else None),
    ("exp", lambda x: sp.exp(x) if SYMPY_AVAILABLE else None),
    ("log", lambda x: sp.log(sp.Abs(x) + 1e-6) if SYMPY_AVAILABLE else None),
    ("tanh", lambda x: sp.tanh(x) if SYMPY_AVAILABLE else None),
    ("sigmoid", lambda x: 1 / (1 + sp.exp(-x)) if SYMPY_AVAILABLE else None),
]


def _check_sympy():
    """Check if SymPy is available."""
    if not SYMPY_AVAILABLE:
        raise ImportError(
            "SymPy is required for symbolic extraction. "
            "Install with: pip install sympy"
        )


def fit_symbolic_function(
    x_samples: "torch.Tensor",
    y_samples: "torch.Tensor",
    basis_functions: List[Tuple[str, Callable]] = None,
    max_complexity: int = 5,
) -> SymbolicFormula:
    """
    Fit a symbolic function to numerical samples.
    
    Uses least squares to find the best linear combination
    of basis functions.
    
    Args:
        x_samples: Input samples, shape (N,)
        y_samples: Output samples, shape (N,)
        basis_functions: List of (name, function) tuples
        max_complexity: Maximum number of basis functions to use
        
    Returns:
        SymbolicFormula with the best fit
    """
    _check_sympy()
    
    if basis_functions is None:
        basis_functions = SYMBOLIC_BASIS[:max_complexity]
    
    x_np = x_samples.detach().cpu().numpy()
    y_np = y_samples.detach().cpu().numpy()
    
    # Create symbolic variable
    x_sym = sp.Symbol('x')
    
    # Evaluate basis functions at sample points
    import numpy as np
    
    basis_values = []
    valid_basis = []
    
    for name, func in basis_functions:
        try:
            # Evaluate symbolically then convert to numeric
            expr = func(x_sym)
            if expr is None:
                continue
            
            # Convert to numpy function
            f_np = sp.lambdify(x_sym, expr, modules=['numpy'])
            values = f_np(x_np)
            
            if np.isfinite(values).all():
                basis_values.append(values)
                valid_basis.append((name, expr))
        except Exception:
            continue
    
    if len(basis_values) == 0:
        return SymbolicFormula(
            expression="f(x) = 0",
            sympy_expr=sp.Integer(0),
            variables=['x'],
            complexity=0,
            accuracy=0.0,
        )
    
    # Solve least squares: A @ coeffs = y
    A = np.column_stack(basis_values)
    coeffs, residuals, _, _ = np.linalg.lstsq(A, y_np, rcond=None)
    
    # Build symbolic expression
    result_expr = sp.Integer(0)
    terms = []
    
    for i, (name, expr) in enumerate(valid_basis):
        coeff = coeffs[i]
        if abs(coeff) > 1e-6:
            result_expr += coeff * expr
            terms.append(f"{coeff:.4g}*{name}")
    
    # Simplify
    result_expr = simplify(result_expr)
    
    # Compute accuracy
    f_result = sp.lambdify(x_sym, result_expr, modules=['numpy'])
    y_pred = f_result(x_np)
    ss_res = np.sum((y_np - y_pred) ** 2)
    ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return SymbolicFormula(
        expression=f"f(x) = {result_expr}",
        sympy_expr=result_expr,
        variables=['x'],
        complexity=len(valid_basis),
        accuracy=max(0.0, r2),
    )


def extract_symbolic_formula(
    kan_layer: "nn.Module",
    input_idx: int = 0,
    output_idx: int = 0,
    num_samples: int = 100,
) -> SymbolicFormula:
    """
    Extract symbolic formula from a KAN layer.
    
    Samples the activation function and fits a symbolic expression.
    
    Args:
        kan_layer: A KANLayer instance
        input_idx: Which input feature to analyze
        output_idx: Which output feature to analyze
        num_samples: Number of samples for fitting
        
    Returns:
        SymbolicFormula representing the learned function
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    _check_sympy()
    
    # Create sample inputs
    x_samples = torch.linspace(-1, 1, num_samples)
    
    # Create input tensor with zeros except for the target feature
    batch = torch.zeros(num_samples, kan_layer.in_features)
    batch[:, input_idx] = x_samples
    
    # Forward pass
    with torch.no_grad():
        output = kan_layer(batch)
    
    y_samples = output[:, output_idx]
    
    # Fit symbolic function
    formula = fit_symbolic_function(x_samples, y_samples)
    formula.variables = [f'x_{input_idx}']
    
    return formula


class SymbolicKAN(nn.Module if TORCH_AVAILABLE else object):
    """
    KAN network with built-in symbolic extraction.
    
    Wraps an EfficientKAN and provides methods for
    extracting interpretable formulas.
    """
    
    def __init__(self, base_kan: "nn.Module"):
        if TORCH_AVAILABLE:
            super().__init__()
        self.base_kan = base_kan
        self._extracted_formulas: Dict[str, SymbolicFormula] = {}
    
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through base KAN."""
        return self.base_kan(x)
    
    def extract_all_formulas(
        self,
        num_samples: int = 100,
    ) -> Dict[str, SymbolicFormula]:
        """
        Extract symbolic formulas for all input-output pairs.
        
        Returns:
            Dictionary mapping "layer_i_j" to SymbolicFormula
        """
        formulas = {}
        
        # Iterate through layers
        if hasattr(self.base_kan, 'layers'):
            layers = self.base_kan.layers
        elif hasattr(self.base_kan, 'children'):
            layers = [m for m in self.base_kan.children() 
                     if hasattr(m, 'in_features')]
        else:
            return formulas
        
        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, 'in_features'):
                continue
            
            for in_idx in range(layer.in_features):
                for out_idx in range(layer.out_features):
                    key = f"layer{layer_idx}_in{in_idx}_out{out_idx}"
                    try:
                        formula = extract_symbolic_formula(
                            layer, in_idx, out_idx, num_samples
                        )
                        formulas[key] = formula
                    except Exception as e:
                        logger.warning(f"Failed to extract {key}: {e}")
        
        self._extracted_formulas = formulas
        return formulas
    
    def get_interpretation(self) -> str:
        """
        Get human-readable interpretation of the network.
        
        Returns:
            String describing what the network has learned
        """
        if not self._extracted_formulas:
            self.extract_all_formulas()
        
        lines = ["KAN Network Interpretation:", "=" * 40]
        
        for key, formula in self._extracted_formulas.items():
            if formula.accuracy > 0.8:  # Only show good fits
                lines.append(f"\n{key}:")
                lines.append(f"  {formula.expression}")
                lines.append(f"  Accuracy: {formula.accuracy:.2%}")
        
        return "\n".join(lines)
    
    def to_sympy(self) -> Dict[str, object]:
        """
        Get SymPy expressions for all extracted formulas.
        
        Returns:
            Dictionary of SymPy expressions
        """
        if not self._extracted_formulas:
            self.extract_all_formulas()
        
        return {
            k: v.sympy_expr 
            for k, v in self._extracted_formulas.items()
        }
