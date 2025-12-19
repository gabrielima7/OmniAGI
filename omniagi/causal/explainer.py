"""
Decision Explainer - Explain decisions causally.

Provides causal explanations for agent decisions,
enabling transparency and debuggability.
"""

from __future__ import annotations

import logging

try:
    import structlog
except ImportError:
    structlog = None
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from omniagi.causal.graph import CausalGraph, CausalNode

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Record of a decision made by the agent."""
    
    id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # What was decided
    decision: str
    action_taken: str
    
    # Why it was decided
    inputs: dict[str, Any] = field(default_factory=dict)
    reasoning_steps: list[str] = field(default_factory=list)
    
    # Causal factors
    causes: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    
    # Outcome
    outcome: str | None = None
    success: bool | None = None


@dataclass
class Explanation:
    """A causal explanation for a decision."""
    
    decision_id: str
    summary: str
    
    # Causal chain
    because: list[str]  # Direct causes
    led_to: list[str]  # Effects
    
    # Counterfactual
    if_not: str  # What would have happened otherwise
    
    # Confidence
    confidence: float


class DecisionExplainer:
    """
    Explains agent decisions causally.
    
    Tracks decisions, builds causal models, and provides
    explanations for why decisions were made.
    """
    
    def __init__(self, causal_graph: CausalGraph | None = None):
        """
        Initialize explainer.
        
        Args:
            causal_graph: Optional causal graph for domain knowledge.
        """
        self.causal_graph = causal_graph or CausalGraph()
        self._decisions: dict[str, DecisionRecord] = {}
        self._decision_counter = 0
    
    def record_decision(
        self,
        decision: str,
        action_taken: str,
        inputs: dict[str, Any] | None = None,
        reasoning_steps: list[str] | None = None,
        causes: list[str] | None = None,
    ) -> DecisionRecord:
        """
        Record a decision for later explanation.
        
        Args:
            decision: What was decided.
            action_taken: What action was taken.
            inputs: Input data that influenced the decision.
            reasoning_steps: Steps in the reasoning process.
            causes: Known causal factors.
            
        Returns:
            The recorded decision.
        """
        import uuid
        
        decision_id = f"dec_{self._decision_counter}"
        self._decision_counter += 1
        
        record = DecisionRecord(
            id=decision_id,
            decision=decision,
            action_taken=action_taken,
            inputs=inputs or {},
            reasoning_steps=reasoning_steps or [],
            causes=causes or [],
        )
        
        self._decisions[decision_id] = record
        
        logger.info("Decision recorded", id=decision_id, decision=decision[:50])
        return record
    
    def record_outcome(
        self,
        decision_id: str,
        outcome: str,
        success: bool,
        effects: list[str] | None = None,
    ) -> None:
        """Record the outcome of a decision."""
        record = self._decisions.get(decision_id)
        if record:
            record.outcome = outcome
            record.success = success
            record.effects = effects or []
    
    def explain(self, decision_id: str) -> Explanation:
        """
        Generate a causal explanation for a decision.
        
        Args:
            decision_id: ID of the decision to explain.
            
        Returns:
            Explanation with causal reasoning.
        """
        record = self._decisions.get(decision_id)
        
        if not record:
            return Explanation(
                decision_id=decision_id,
                summary="Decisão não encontrada.",
                because=[],
                led_to=[],
                if_not="Desconhecido.",
                confidence=0.0,
            )
        
        # Build "because" chain from causes and inputs
        because = []
        
        for cause in record.causes:
            because.append(f"Porque {cause}")
        
        for key, value in record.inputs.items():
            if value:  # Only add meaningful inputs
                because.append(f"{key} = {value}")
        
        if record.reasoning_steps:
            because.extend([f"Passo: {step}" for step in record.reasoning_steps[:3]])
        
        # Build "led to" chain from effects
        led_to = []
        if record.effects:
            led_to = [f"Resultou em: {e}" for e in record.effects]
        if record.outcome:
            led_to.append(f"Resultado final: {record.outcome}")
        
        # Generate counterfactual
        if_not = self._generate_counterfactual(record)
        
        # Calculate confidence
        confidence = 0.7
        if record.success is not None:
            confidence = 0.9 if record.success else 0.5
        
        # Summary
        summary = f'A decisão "{record.decision}" foi tomada e a ação "{record.action_taken}" foi executada.'
        if record.outcome:
            summary += f" O resultado foi: {record.outcome}"
        
        return Explanation(
            decision_id=decision_id,
            summary=summary,
            because=because[:5],  # Limit to top 5 causes
            led_to=led_to[:3],  # Limit to top 3 effects
            if_not=if_not,
            confidence=confidence,
        )
    
    def _generate_counterfactual(self, record: DecisionRecord) -> str:
        """Generate a counterfactual explanation."""
        if not record.causes:
            return "Sem informação suficiente para análise contrafactual."
        
        primary_cause = record.causes[0]
        
        if record.success:
            return (
                f"Se não fosse por '{primary_cause}', "
                f"provavelmente não teríamos tomado esta decisão ou "
                f"o resultado poderia ter sido diferente."
            )
        else:
            return (
                f"Se '{primary_cause}' fosse diferente, "
                f"poderíamos ter evitado o resultado negativo."
            )
    
    def explain_why(self, decision_id: str) -> str:
        """Get a simple "why" explanation."""
        explanation = self.explain(decision_id)
        
        if not explanation.because:
            return "Não há informação suficiente para explicar esta decisão."
        
        reasons = " → ".join(explanation.because[:3])
        return f"Esta decisão foi tomada porque: {reasons}"
    
    def explain_chain(
        self,
        decision_ids: list[str],
    ) -> str:
        """Explain a chain of decisions."""
        lines = ["## Cadeia de Decisões\n"]
        
        for i, dec_id in enumerate(decision_ids):
            record = self._decisions.get(dec_id)
            if not record:
                continue
            
            lines.append(f"### {i+1}. {record.decision}")
            lines.append(f"Ação: {record.action_taken}")
            
            if record.causes:
                lines.append(f"Causas: {', '.join(record.causes[:2])}")
            
            if record.outcome:
                lines.append(f"Resultado: {record.outcome}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def get_decision_graph(self) -> CausalGraph:
        """
        Build a causal graph from recorded decisions.
        
        Shows how decisions led to outcomes.
        """
        graph = CausalGraph()
        
        for record in self._decisions.values():
            # Add decision node
            dec_node = graph.add_node(
                name=record.decision[:30],
                node_type="decision",
                node_id=record.id,
            )
            
            # Add cause nodes and edges
            for cause in record.causes:
                cause_node = graph.add_node(
                    name=cause[:30],
                    node_type="cause",
                )
                graph.add_edge(cause_node.id, dec_node.id, "caused")
            
            # Add outcome node and edge
            if record.outcome:
                outcome_node = graph.add_node(
                    name=record.outcome[:30],
                    node_type="outcome",
                )
                graph.add_edge(dec_node.id, outcome_node.id, "led_to")
        
        return graph
    
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about decisions."""
        total = len(self._decisions)
        with_outcome = sum(1 for d in self._decisions.values() if d.outcome)
        successful = sum(1 for d in self._decisions.values() if d.success)
        
        return {
            "total_decisions": total,
            "with_outcome": with_outcome,
            "successful": successful,
            "success_rate": successful / max(1, with_outcome),
        }
