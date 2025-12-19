"""
Logical Neural Networks (LNN).

A neuro-symbolic framework where each neuron corresponds to
a component of a formula in weighted real-valued logic.

Key features:
1. Full logical interpretability
2. End-to-end differentiability
3. Handles inconsistent/incomplete knowledge
4. Tight bounds on truth values

Based on IBM's LNN research.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class LogicalOperator(Enum):
    """Logical operators for LNN."""
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    IFF = auto()
    ATOM = auto()  # Atomic proposition


@dataclass
class TruthBounds:
    """
    Bounds on truth value in real-valued logic.
    
    LNN uses [lower, upper] bounds to represent
    uncertainty and incomplete information.
    """
    lower: float = 0.0
    upper: float = 1.0
    
    def __post_init__(self):
        self.lower = max(0.0, min(1.0, self.lower))
        self.upper = max(self.lower, min(1.0, self.upper))
    
    @property
    def is_true(self) -> bool:
        return self.lower >= 0.5
    
    @property
    def is_false(self) -> bool:
        return self.upper < 0.5
    
    @property
    def is_unknown(self) -> bool:
        return self.lower < 0.5 <= self.upper
    
    @property
    def point_estimate(self) -> float:
        return (self.lower + self.upper) / 2


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch")


class LogicalNeuron(nn.Module if TORCH_AVAILABLE else object):
    """
    A single logical neuron in the LNN.
    
    Implements real-valued logic operations with learnable
    parameters for soft bounds propagation.
    
    Args:
        operator: The logical operator this neuron implements
        num_inputs: Number of input connections
        alpha: Learnable tightness parameter
    """
    
    def __init__(
        self,
        operator: LogicalOperator,
        num_inputs: int = 2,
        alpha: float = 1.0,
    ):
        _check_torch()
        super().__init__()
        
        self.operator = operator
        self.num_inputs = num_inputs
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
        # Bias for atoms
        if operator == LogicalOperator.ATOM:
            self.bias = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_parameter('bias', None)
        
        # Weights for weighted operators
        if operator in [LogicalOperator.AND, LogicalOperator.OR]:
            self.weights = nn.Parameter(torch.ones(num_inputs))
        else:
            self.register_parameter('weights', None)
    
    def forward(
        self,
        *inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing truth bounds.
        
        Args:
            *inputs: Pairs of (lower, upper) bounds for each input
            
        Returns:
            (lower, upper) bounds for the output
        """
        if self.operator == LogicalOperator.ATOM:
            # Atomic proposition - just return bias as point
            val = torch.sigmoid(self.bias)
            return val, val
        
        elif self.operator == LogicalOperator.NOT:
            # NOT(x) = 1 - x, bounds swap
            lower, upper = inputs[0]
            return 1 - upper, 1 - lower
        
        elif self.operator == LogicalOperator.AND:
            # Lukasiewicz AND: max(0, sum(x_i) - (n-1))
            return self._lukasiewicz_and(inputs)
        
        elif self.operator == LogicalOperator.OR:
            # Lukasiewicz OR: min(1, sum(x_i))
            return self._lukasiewicz_or(inputs)
        
        elif self.operator == LogicalOperator.IMPLIES:
            # A -> B = NOT(A) OR B
            a_lower, a_upper = inputs[0]
            b_lower, b_upper = inputs[1]
            
            # NOT(A)
            not_a_lower, not_a_upper = 1 - a_upper, 1 - a_lower
            
            # OR with B
            lower = torch.clamp(not_a_lower + b_lower, 0, 1)
            upper = torch.clamp(not_a_upper + b_upper, 0, 1)
            return lower, upper
        
        else:
            # Default: pass through first input
            return inputs[0]
    
    def _lukasiewicz_and(
        self,
        inputs: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lukasiewicz t-norm for AND."""
        n = len(inputs)
        
        # Apply weights
        weights = F.softmax(self.weights[:n], dim=0)
        
        # Sum of lower bounds
        lower_sum = sum(w * inp[0] for w, inp in zip(weights, inputs))
        upper_sum = sum(w * inp[1] for w, inp in zip(weights, inputs))
        
        # Lukasiewicz: max(0, sum - (n-1))
        threshold = (n - 1) / n
        lower = torch.clamp(lower_sum - threshold, 0, 1)
        upper = torch.clamp(upper_sum - threshold, 0, 1)
        
        return lower, upper
    
    def _lukasiewicz_or(
        self,
        inputs: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lukasiewicz t-conorm for OR."""
        n = len(inputs)
        
        weights = F.softmax(self.weights[:n], dim=0)
        
        lower_sum = sum(w * inp[0] for w, inp in zip(weights, inputs))
        upper_sum = sum(w * inp[1] for w, inp in zip(weights, inputs))
        
        # Lukasiewicz OR: min(1, sum)
        lower = torch.clamp(lower_sum, 0, 1)
        upper = torch.clamp(upper_sum, 0, 1)
        
        return lower, upper


class LNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Complete Logical Neural Network.
    
    A network of logical neurons that can represent
    and reason over first-order logic formulas.
    
    Example:
        lnn = LNN()
        lnn.add_predicate("mortal", 1)
        lnn.add_predicate("human", 1)
        lnn.add_rule(
            "mortal_rule",
            antecedents=["human(X)"],
            consequent="mortal(X)"
        )
        lnn.set_fact("human", "socrates", 1.0)
        result = lnn.infer("mortal", "socrates")
    """
    
    def __init__(self):
        _check_torch()
        super().__init__()
        
        # Knowledge base
        self.predicates: Dict[str, int] = {}  # name -> arity
        self.facts: Dict[str, Dict[str, TruthBounds]] = {}  # pred -> {args -> bounds}
        self.rules: Dict[str, LogicalNeuron] = {}
        
        # Neuron modules
        self.neurons = nn.ModuleDict()
        
        logger.info("LNN initialized")
    
    def add_predicate(self, name: str, arity: int = 1) -> None:
        """Add a predicate to the knowledge base."""
        self.predicates[name] = arity
        self.facts[name] = {}
        
        # Create atomic neuron for this predicate
        self.neurons[name] = LogicalNeuron(
            LogicalOperator.ATOM,
            num_inputs=0,
        )
    
    def set_fact(
        self,
        predicate: str,
        args: str,
        value: float,
        uncertainty: float = 0.0,
    ) -> None:
        """
        Set a fact in the knowledge base.
        
        Args:
            predicate: Predicate name
            args: Arguments as string (e.g., "socrates")
            value: Truth value (0 to 1)
            uncertainty: Uncertainty range
        """
        if predicate not in self.predicates:
            self.add_predicate(predicate, 1)
        
        self.facts[predicate][args] = TruthBounds(
            lower=value - uncertainty,
            upper=value + uncertainty,
        )
    
    def add_rule(
        self,
        name: str,
        antecedents: List[str],
        consequent: str,
    ) -> None:
        """
        Add a logical rule.
        
        Args:
            name: Rule name
            antecedents: List of antecedent predicates
            consequent: Consequent predicate
        """
        # Create implication neuron
        rule_neuron = LogicalNeuron(
            LogicalOperator.IMPLIES,
            num_inputs=len(antecedents) + 1,
        )
        
        self.rules[name] = {
            "neuron": rule_neuron,
            "antecedents": antecedents,
            "consequent": consequent,
        }
        self.neurons[f"rule_{name}"] = rule_neuron
    
    def forward(
        self,
        query: str,
        args: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for inference.
        
        Args:
            query: Predicate to query
            args: Arguments for the predicate
            
        Returns:
            (lower, upper) truth bounds
        """
        # Check direct facts
        if query in self.facts and args in self.facts[query]:
            bounds = self.facts[query][args]
            return (
                torch.tensor(bounds.lower),
                torch.tensor(bounds.upper),
            )
        
        # Try rule-based inference
        for rule_name, rule in self.rules.items():
            if query in rule["consequent"]:
                # Try to prove antecedents
                ant_bounds = []
                for ant in rule["antecedents"]:
                    # Parse predicate
                    pred = ant.split("(")[0]
                    ant_args = args  # Simplified: use same args
                    
                    lower, upper = self.forward(pred, ant_args)
                    ant_bounds.append((lower, upper))
                
                if ant_bounds:
                    # Combine with AND
                    and_neuron = LogicalNeuron(
                        LogicalOperator.AND,
                        num_inputs=len(ant_bounds),
                    )
                    and_lower, and_upper = and_neuron(*ant_bounds)
                    return and_lower, and_upper
        
        # Unknown
        return torch.tensor(0.0), torch.tensor(1.0)
    
    def infer(self, predicate: str, args: str) -> TruthBounds:
        """
        Infer truth bounds for a query.
        
        Args:
            predicate: Predicate name
            args: Arguments
            
        Returns:
            TruthBounds for the query
        """
        with torch.no_grad():
            lower, upper = self.forward(predicate, args)
        
        return TruthBounds(
            lower=lower.item(),
            upper=upper.item(),
        )
    
    def explain(self, predicate: str, args: str) -> List[str]:
        """
        Explain why a query has certain truth bounds.
        
        Returns:
            List of explanation steps
        """
        explanations = []
        
        # Check facts
        if predicate in self.facts and args in self.facts[predicate]:
            bounds = self.facts[predicate][args]
            explanations.append(
                f"Direct fact: {predicate}({args}) = [{bounds.lower:.2f}, {bounds.upper:.2f}]"
            )
        
        # Check rules
        for rule_name, rule in self.rules.items():
            if predicate in rule["consequent"]:
                explanations.append(
                    f"Rule '{rule_name}': {' AND '.join(rule['antecedents'])} -> {rule['consequent']}"
                )
        
        return explanations if explanations else ["No explanation available"]


def create_lnn_from_rules(rules: List[Dict]) -> LNN:
    """
    Create an LNN from a list of rule definitions.
    
    Args:
        rules: List of {name, antecedents, consequent} dicts
        
    Returns:
        Configured LNN
    """
    lnn = LNN()
    
    for rule in rules:
        # Extract predicates
        for ant in rule.get("antecedents", []):
            pred = ant.split("(")[0]
            lnn.add_predicate(pred, 1)
        
        consequent = rule.get("consequent", "")
        pred = consequent.split("(")[0]
        lnn.add_predicate(pred, 1)
        
        # Add rule
        lnn.add_rule(
            name=rule.get("name", f"rule_{len(lnn.rules)}"),
            antecedents=rule.get("antecedents", []),
            consequent=consequent,
        )
    
    return lnn
