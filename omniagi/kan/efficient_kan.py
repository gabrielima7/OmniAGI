"""
Efficient Kolmogorov-Arnold Networks (KAN) Implementation.

Based on the Kolmogorov-Arnold representation theorem:
f(x) = Σ Φ_q(Σ φ_{q,p}(x_p))

This implementation uses B-spline basis functions for efficiency,
avoiding the memory-intensive expansion of intermediate variables.

Key optimizations:
1. Matrix multiplication for B-spline computation
2. L1 regularization on weights (not activations)
3. Optional Radial Basis Functions for 3x speedup
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple, List

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    torch = None

logger = logging.getLogger(__name__)


def _check_torch():
    """Check if PyTorch is available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for KAN. Install with: pip install torch"
        )


class BSplineBasis:
    """
    B-spline basis functions for efficient KAN.
    
    Uses de Boor's algorithm for stable computation.
    """
    
    def __init__(
        self,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        # Create grid
        self.num_basis = grid_size + spline_order
        self._create_grid()
    
    def _create_grid(self):
        """Create uniform B-spline grid."""
        _check_torch()
        
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - self.spline_order * h,
            self.grid_range[1] + self.spline_order * h,
            self.num_basis + self.spline_order + 1,
        )
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis at points x.
        
        Args:
            x: Input points, shape (batch, features)
            
        Returns:
            Basis values, shape (batch, features, num_basis)
        """
        _check_torch()
        
        # Ensure grid is on same device
        grid = self.grid.to(x.device)
        
        # Initialize with order 0 B-splines
        x_expanded = x.unsqueeze(-1)  # (batch, features, 1)
        
        # Compute all basis functions using Cox-de Boor recursion
        bases = []
        for i in range(self.num_basis):
            # B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1], else 0
            left = grid[i]
            right = grid[i + 1]
            basis = ((x_expanded >= left) & (x_expanded < right)).float()
            bases.append(basis)
        
        bases = torch.cat(bases, dim=-1)  # (batch, features, num_basis)
        
        # Recursively compute higher order B-splines
        for k in range(1, self.spline_order + 1):
            new_bases = []
            for i in range(self.num_basis):
                # Left term
                denom_left = grid[i + k] - grid[i]
                if denom_left > 1e-8:
                    left_term = (x_expanded - grid[i]) / denom_left * bases[..., i:i+1]
                else:
                    left_term = torch.zeros_like(bases[..., i:i+1])
                
                # Right term
                if i + 1 < self.num_basis:
                    denom_right = grid[i + k + 1] - grid[i + 1]
                    if denom_right > 1e-8:
                        right_term = (grid[i + k + 1] - x_expanded) / denom_right * bases[..., i+1:i+2]
                    else:
                        right_term = torch.zeros_like(bases[..., i:i+1])
                else:
                    right_term = torch.zeros_like(bases[..., i:i+1])
                
                new_bases.append(left_term + right_term)
            
            bases = torch.cat(new_bases, dim=-1)
        
        return bases


class KANLayer(nn.Module if TORCH_AVAILABLE else object):
    """
    Efficient Kolmogorov-Arnold Network Layer.
    
    Uses B-spline basis functions with learnable coefficients.
    This formulation allows efficient matrix multiplication
    instead of expensive per-activation computation.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        grid_size: Number of grid intervals for B-splines
        spline_order: Order of B-spline (default 3 = cubic)
        use_bias: Whether to add bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        use_bias: bool = True,
    ):
        _check_torch()
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # B-spline basis
        self.basis = BSplineBasis(grid_size, spline_order)
        self.num_basis = self.basis.num_basis
        
        # Learnable spline coefficients: (out, in, num_basis)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.num_basis) * 0.1
        )
        
        # Base weight for residual connection
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )
        
        # Scale parameter
        self.scale = nn.Parameter(torch.ones(out_features))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.xavier_uniform_(
            self.spline_weight.view(self.out_features, -1)
        ).view_as(self.spline_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.
        
        Args:
            x: Input tensor, shape (batch, in_features)
            
        Returns:
            Output tensor, shape (batch, out_features)
        """
        batch_size = x.shape[0]
        
        # Normalize input to [-1, 1]
        x_norm = torch.tanh(x)
        
        # Compute B-spline basis: (batch, in_features, num_basis)
        basis_vals = self.basis.evaluate(x_norm)
        
        # Spline contribution: efficient matrix multiplication
        # (batch, in, num_basis) @ (out, in, num_basis)^T -> (batch, out)
        spline_out = torch.einsum(
            'bik,oik->bo',
            basis_vals,
            self.spline_weight
        )
        
        # Base (linear) contribution
        base_out = F.linear(x, self.base_weight)
        
        # Combine with learned scale
        output = self.scale * (spline_out + base_out)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def regularization_loss(self, l1_weight: float = 1e-4) -> torch.Tensor:
        """Compute L1 regularization on spline weights."""
        return l1_weight * torch.mean(torch.abs(self.spline_weight))


class EfficientKAN(nn.Module if TORCH_AVAILABLE else object):
    """
    Complete Efficient Kolmogorov-Arnold Network.
    
    Multi-layer KAN with residual connections and layer normalization
    for stable training.
    
    Args:
        layer_sizes: List of layer dimensions [input, hidden..., output]
        grid_size: Grid size for B-splines
        spline_order: B-spline order
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        _check_torch()
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        
        # Build layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.layers.append(
                KANLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
            )
            if i < self.num_layers - 1:
                self.norms.append(nn.LayerNorm(layer_sizes[i + 1]))
        
        logger.info(
            f"EfficientKAN created: {layer_sizes}, "
            f"grid={grid_size}, order={spline_order}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.norms):
                x = self.norms[i](x)
                x = F.silu(x)  # SiLU activation between layers
        return x
    
    def regularization_loss(self, l1_weight: float = 1e-4) -> torch.Tensor:
        """Total regularization loss."""
        total = torch.tensor(0.0)
        for layer in self.layers:
            total = total + layer.regularization_loss(l1_weight)
        return total


class RadialBasisKAN(nn.Module if TORCH_AVAILABLE else object):
    """
    Fast KAN using Radial Basis Functions.
    
    3x faster than B-spline KAN by using RBF kernels
    instead of B-spline basis functions.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_centers: Number of RBF centers
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_centers: int = 8,
    ):
        _check_torch()
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        
        # RBF centers: uniformly distributed in [-1, 1]
        centers = torch.linspace(-1, 1, num_centers)
        self.register_buffer('centers', centers)
        
        # Learnable width
        self.log_sigma = nn.Parameter(torch.zeros(1))
        
        # Learnable coefficients: (out, in, num_centers)
        self.coefficients = nn.Parameter(
            torch.randn(out_features, in_features, num_centers) * 0.1
        )
        
        # Base linear
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1
        )
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using RBF kernel."""
        # Normalize input
        x = self.input_norm(x)
        
        # Clamp to [-1, 1]
        x_clamped = torch.clamp(x, -1, 1)
        
        # Compute RBF values: exp(-||x - c||^2 / (2 * sigma^2))
        sigma = torch.exp(self.log_sigma) + 0.1
        
        # (batch, in, 1) - (num_centers,) -> (batch, in, num_centers)
        diff = x_clamped.unsqueeze(-1) - self.centers
        rbf = torch.exp(-diff ** 2 / (2 * sigma ** 2))
        
        # Apply coefficients: (batch, in, centers) @ (out, in, centers)
        rbf_out = torch.einsum('bic,oic->bo', rbf, self.coefficients)
        
        # Base linear
        base_out = F.linear(x, self.base_weight)
        
        return rbf_out + base_out + self.bias


# Convenience function for creating KAN networks
def create_kan(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    fast_mode: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Create a KAN network.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        fast_mode: Use RadialBasisKAN (3x faster) instead of B-spline
        **kwargs: Additional arguments for KAN layers
        
    Returns:
        KAN network module
    """
    _check_torch()
    
    layer_sizes = [input_dim] + hidden_dims + [output_dim]
    
    if fast_mode:
        # Build with RadialBasisKAN layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(
                RadialBasisKAN(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    **kwargs
                )
            )
            if i < len(layer_sizes) - 2:
                layers.append(nn.SiLU())
        return nn.Sequential(*layers)
    else:
        return EfficientKAN(layer_sizes, **kwargs)
