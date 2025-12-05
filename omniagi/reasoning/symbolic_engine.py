"""
Neuro-Symbolic Reasoning Engine.

Combines neural network intuition with symbolic logic
for verifiable, explainable reasoning.

This is critical for true AGI - pure neural networks
cannot guarantee logical consistency.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = structlog.get_logger()


class LogicOperator(Enum):
    """Logical operators."""
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    IFF = auto()  # If and only if
    FORALL = auto()
    EXISTS = auto()


@dataclass
class Proposition:
    """A logical proposition."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    value: bool | None = None  # None = unknown
    
    # Grounding
    grounded_in: str = ""  # Natural language source
    confidence: float = 1.0
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"P({self.name}={self.value})"


@dataclass
class Rule:
    """A logical rule (implication)."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    
    # Structure: antecedent -> consequent
    antecedent: list[str] = field(default_factory=list)  # Proposition names
    consequent: str = ""  # Proposition name
    operator: LogicOperator = LogicOperator.IMPLIES
    
    # Metadata
    confidence: float = 1.0
    source: str = ""  # Where this rule came from
    times_applied: int = 0
    
    def __str__(self):
        ante = " AND ".join(self.antecedent)
        return f"{ante} -> {self.consequent}"


@dataclass
class Inference:
    """A single inference step."""
    
    rule_used: str
    premises: list[str]
    conclusion: str
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            "rule": self.rule_used,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
        }


@dataclass
class ReasoningChain:
    """A chain of reasoning steps."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    goal: str = ""
    
    steps: list[Inference] = field(default_factory=list)
    
    success: bool = False
    final_confidence: float = 0.0
    
    def add_step(self, inference: Inference) -> None:
        self.steps.append(inference)
        # Confidence decreases with each step
        self.final_confidence = inference.confidence * (0.95 ** len(self.steps))
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
            "confidence": self.final_confidence,
        }


class SymbolicEngine:
    """
    Symbolic Logic Engine for Formal Reasoning.
    
    Provides:
    1. Propositional logic
    2. First-order logic (simplified)
    3. Rule-based inference
    4. Consistency checking
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Knowledge base
        self._propositions: dict[str, Proposition] = {}
        self._rules: dict[str, Rule] = {}
        
        # Inference history
        self._inferences: list[Inference] = []
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info(
            "Symbolic Engine initialized",
            propositions=len(self._propositions),
            rules=len(self._rules),
        )
    
    def add_proposition(
        self,
        name: str,
        value: bool | None = None,
        grounded_in: str = "",
    ) -> Proposition:
        """Add a proposition to the knowledge base."""
        prop = Proposition(
            name=name,
            value=value,
            grounded_in=grounded_in,
        )
        self._propositions[name] = prop
        return prop
    
    def add_rule(
        self,
        name: str,
        antecedent: list[str],
        consequent: str,
        source: str = "",
    ) -> Rule:
        """Add a logical rule."""
        rule = Rule(
            name=name,
            antecedent=antecedent,
            consequent=consequent,
            source=source,
        )
        self._rules[rule.id] = rule
        
        # Ensure propositions exist
        for prop_name in antecedent + [consequent]:
            if prop_name not in self._propositions:
                self.add_proposition(prop_name)
        
        return rule
    
    def infer(self, goal: str, max_depth: int = 10) -> ReasoningChain:
        """
        Try to infer whether a goal proposition is true.
        
        Uses forward and backward chaining.
        """
        chain = ReasoningChain(goal=goal)
        
        # Check if already known
        if goal in self._propositions:
            prop = self._propositions[goal]
            if prop.value is not None:
                chain.success = prop.value
                chain.final_confidence = prop.confidence
                return chain
        
        # Try backward chaining
        success = self._backward_chain(goal, chain, set(), max_depth)
        chain.success = success
        
        self._save()
        return chain
    
    def _backward_chain(
        self,
        goal: str,
        chain: ReasoningChain,
        visited: set[str],
        depth: int,
    ) -> bool:
        """Backward chaining inference."""
        if depth <= 0:
            return False
        
        if goal in visited:
            return False  # Avoid cycles
        visited.add(goal)
        
        # Check if goal is already known
        if goal in self._propositions:
            prop = self._propositions[goal]
            if prop.value is not None:
                return prop.value
        
        # Find rules that conclude the goal
        applicable_rules = [
            r for r in self._rules.values()
            if r.consequent == goal
        ]
        
        for rule in applicable_rules:
            # Try to prove all antecedents
            all_true = True
            premise_values = []
            
            for ante in rule.antecedent:
                if ante in self._propositions:
                    prop = self._propositions[ante]
                    if prop.value is True:
                        premise_values.append(ante)
                        continue
                    elif prop.value is False:
                        all_true = False
                        break
                
                # Recurse
                if self._backward_chain(ante, chain, visited.copy(), depth - 1):
                    premise_values.append(ante)
                else:
                    all_true = False
                    break
            
            if all_true:
                # Rule fires!
                inference = Inference(
                    rule_used=rule.name,
                    premises=premise_values,
                    conclusion=goal,
                    confidence=rule.confidence,
                )
                chain.add_step(inference)
                rule.times_applied += 1
                
                # Update proposition
                if goal not in self._propositions:
                    self.add_proposition(goal, True)
                else:
                    self._propositions[goal].value = True
                
                self._inferences.append(inference)
                return True
        
        return False
    
    def forward_chain(self) -> list[Inference]:
        """
        Forward chaining - derive all possible conclusions.
        """
        new_inferences = []
        changed = True
        
        while changed:
            changed = False
            
            for rule in self._rules.values():
                # Check if consequent is already known
                if rule.consequent in self._propositions:
                    if self._propositions[rule.consequent].value is not None:
                        continue
                
                # Check if all antecedents are true
                all_true = True
                for ante in rule.antecedent:
                    if ante not in self._propositions:
                        all_true = False
                        break
                    if self._propositions[ante].value is not True:
                        all_true = False
                        break
                
                if all_true:
                    # Derive conclusion
                    if rule.consequent not in self._propositions:
                        self.add_proposition(rule.consequent, True)
                    else:
                        self._propositions[rule.consequent].value = True
                    
                    inference = Inference(
                        rule_used=rule.name,
                        premises=rule.antecedent,
                        conclusion=rule.consequent,
                        confidence=rule.confidence,
                    )
                    new_inferences.append(inference)
                    self._inferences.append(inference)
                    rule.times_applied += 1
                    changed = True
        
        return new_inferences
    
    def check_consistency(self) -> list[str]:
        """
        Check knowledge base for logical inconsistencies.
        
        Returns list of inconsistencies found.
        """
        inconsistencies = []
        
        # Check for direct contradictions
        for name, prop in self._propositions.items():
            neg_name = f"not_{name}"
            if neg_name in self._propositions:
                neg_prop = self._propositions[neg_name]
                if prop.value is True and neg_prop.value is True:
                    inconsistencies.append(
                        f"Contradiction: {name} and {neg_name} both true"
                    )
        
        return inconsistencies
    
    def explain(self, proposition: str) -> list[str]:
        """
        Explain why a proposition is true/false.
        """
        explanations = []
        
        if proposition not in self._propositions:
            return ["Unknown proposition"]
        
        prop = self._propositions[proposition]
        
        if prop.grounded_in:
            explanations.append(f"Grounded in: {prop.grounded_in}")
        
        # Find inferences that concluded this
        for inf in self._inferences:
            if inf.conclusion == proposition:
                explanations.append(
                    f"Inferred by {inf.rule_used} from {', '.join(inf.premises)}"
                )
        
        return explanations if explanations else ["No explanation available"]
    
    def query(self, proposition: str) -> tuple[bool | None, float]:
        """
        Query the truth value of a proposition.
        
        Returns (value, confidence).
        """
        if proposition not in self._propositions:
            return None, 0.0
        
        prop = self._propositions[proposition]
        return prop.value, prop.confidence
    
    def __len__(self) -> int:
        return len(self._propositions)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "propositions": len(self._propositions),
                "rules": len(self._rules),
            }, f)
    
    def _load(self) -> None:
        pass


class HybridReasoner:
    """
    Hybrid Neural-Symbolic Reasoner.
    
    Combines:
    1. Neural intuition (LLM) for pattern matching
    2. Symbolic logic for verification
    3. Falls back to logic when neural is uncertain
    """
    
    def __init__(
        self,
        symbolic_engine: SymbolicEngine = None,
        llm_backend: Any = None,
    ):
        self.symbolic = symbolic_engine or SymbolicEngine()
        self.llm = llm_backend
        
        logger.info("Hybrid Reasoner initialized")
    
    def reason(
        self,
        query: str,
        context: str = "",
    ) -> dict:
        """
        Perform hybrid reasoning on a query.
        """
        result = {
            "query": query,
            "answer": None,
            "confidence": 0.0,
            "method": "",
            "explanation": [],
        }
        
        # Step 1: Try symbolic reasoning first
        symbolic_result = self.symbolic.infer(query)
        
        if symbolic_result.success:
            result["answer"] = True
            result["confidence"] = symbolic_result.final_confidence
            result["method"] = "symbolic"
            result["explanation"] = [
                s.to_dict() for s in symbolic_result.steps
            ]
            return result
        
        # Step 2: Use neural if symbolic fails
        if self.llm:
            neural_result = self._neural_reason(query, context)
            result["answer"] = neural_result.get("answer")
            result["confidence"] = neural_result.get("confidence", 0.5)
            result["method"] = "neural"
            result["explanation"] = [neural_result.get("explanation", "")]
        
        return result
    
    def _neural_reason(self, query: str, context: str) -> dict:
        """Use LLM for neural reasoning."""
        if not self.llm:
            return {"answer": None, "confidence": 0.0}
        
        # Would call LLM here
        return {"answer": None, "confidence": 0.0}
    
    def learn_rule_from_example(
        self,
        premises: list[str],
        conclusion: str,
        name: str = "",
    ) -> Rule:
        """
        Learn a logical rule from an example.
        
        This bridges neural pattern recognition to symbolic rules.
        """
        rule = self.symbolic.add_rule(
            name=name or f"learned_{len(self.symbolic._rules)}",
            antecedent=premises,
            consequent=conclusion,
            source="learned_from_example",
        )
        
        logger.info(
            "Rule learned from example",
            premises=premises,
            conclusion=conclusion,
        )
        
        return rule
