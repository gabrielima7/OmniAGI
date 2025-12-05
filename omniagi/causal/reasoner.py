"""
Causal Reasoner - Inference engine for causal reasoning.

Performs causal inference including:
- Effect estimation
- Counterfactual reasoning
- Intervention analysis
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

from omniagi.causal.graph import CausalGraph, CausalNode

logger = structlog.get_logger()


@dataclass
class CausalQuery:
    """A causal query to answer."""
    
    query_type: str  # effect, counterfactual, intervention
    treatment: str  # Variable to manipulate
    outcome: str  # Variable to observe
    condition: dict[str, Any] | None = None  # Conditioning variables
    intervention_value: Any = None  # Value for do(X=x)


@dataclass
class CausalAnswer:
    """Answer to a causal query."""
    
    query: CausalQuery
    answer: str
    confidence: float
    reasoning: list[str]
    confounders: list[str]
    mediators: list[str]


CAUSAL_REASONING_PROMPT = '''Você é um especialista em raciocínio causal. Analise a seguinte questão:

## Grafo Causal:
```mermaid
{graph_mermaid}
```

## Questão:
Tipo: {query_type}
Tratamento (causa): {treatment}
Resultado (efeito): {outcome}
{condition_text}

## Variáveis Confundidoras Identificadas:
{confounders}

## Variáveis Mediadoras:
{mediators}

## Instruções:
1. Analise as relações causais no grafo
2. Identifique caminhos causais diretos e indiretos
3. Considere possíveis confundidores
4. Responda à questão com raciocínio causal

Responda no formato:
ANÁLISE: [análise detalhada das relações causais]
CAMINHOS: [caminhos causais relevantes]
CONCLUSÃO: [resposta à questão]
CONFIANÇA: [0.0-1.0]
'''


class CausalReasoner:
    """
    Engine for causal reasoning.
    
    Combines causal graphs with LLM reasoning to answer
    causal questions.
    """
    
    def __init__(
        self,
        graph: CausalGraph | None = None,
        engine: "Engine | None" = None,
    ):
        """
        Initialize causal reasoner.
        
        Args:
            graph: The causal graph to reason over.
            engine: LLM engine for complex reasoning.
        """
        self.graph = graph or CausalGraph()
        self.engine = engine
    
    def query(self, query: CausalQuery) -> CausalAnswer:
        """
        Answer a causal query.
        
        Args:
            query: The causal query.
            
        Returns:
            CausalAnswer with reasoning.
        """
        # Get structural information
        confounders = [
            n.name for n in self.graph.find_confounders(query.treatment, query.outcome)
        ]
        mediators = [
            n.name for n in self.graph.find_mediators(query.treatment, query.outcome)
        ]
        
        # Basic structural analysis
        reasoning = []
        
        treatment_node = self.graph._nodes.get(query.treatment)
        outcome_node = self.graph._nodes.get(query.outcome)
        
        if not treatment_node or not outcome_node:
            return CausalAnswer(
                query=query,
                answer="Variáveis não encontradas no grafo causal.",
                confidence=0.0,
                reasoning=["Nós não existem no grafo."],
                confounders=[],
                mediators=[],
            )
        
        # Check for direct effect
        is_direct = outcome_node.id in self.graph._children.get(treatment_node.id, [])
        
        if is_direct:
            reasoning.append(f"{treatment_node.name} tem efeito DIRETO em {outcome_node.name}")
        
        # Check for indirect effect
        if mediators:
            reasoning.append(
                f"Efeito INDIRETO via: {', '.join(mediators)}"
            )
        
        # Check for confounding
        if confounders:
            reasoning.append(
                f"ATENÇÃO: Confundidores detectados: {', '.join(confounders)}. "
                "A associação observada pode não ser causal."
            )
        
        # Use LLM for deeper reasoning if available
        answer = ""
        confidence = 0.5
        
        if self.engine and self.engine.is_loaded:
            llm_result = self._llm_reason(query, confounders, mediators)
            answer = llm_result.get("answer", "")
            confidence = llm_result.get("confidence", 0.5)
            reasoning.extend(llm_result.get("reasoning", []))
        else:
            # Simple answer without LLM
            if is_direct and not confounders:
                answer = f"Sim, {treatment_node.name} causa {outcome_node.name} diretamente."
                confidence = 0.8
            elif mediators and not confounders:
                answer = f"{treatment_node.name} afeta {outcome_node.name} indiretamente via {mediators[0]}."
                confidence = 0.7
            elif confounders:
                answer = (
                    f"A relação entre {treatment_node.name} e {outcome_node.name} "
                    f"pode ser confundida por {confounders[0]}. Cautela é necessária."
                )
                confidence = 0.4
            else:
                answer = f"Não há caminho causal claro de {treatment_node.name} para {outcome_node.name}."
                confidence = 0.3
        
        return CausalAnswer(
            query=query,
            answer=answer,
            confidence=confidence,
            reasoning=reasoning,
            confounders=confounders,
            mediators=mediators,
        )
    
    def _llm_reason(
        self,
        query: CausalQuery,
        confounders: list[str],
        mediators: list[str],
    ) -> dict[str, Any]:
        """Use LLM for causal reasoning."""
        from omniagi.core.engine import GenerationConfig
        
        condition_text = ""
        if query.condition:
            condition_text = "Condições: " + ", ".join(
                f"{k}={v}" for k, v in query.condition.items()
            )
        
        prompt = CAUSAL_REASONING_PROMPT.format(
            graph_mermaid=self.graph.to_mermaid(),
            query_type=query.query_type,
            treatment=query.treatment,
            outcome=query.outcome,
            condition_text=condition_text,
            confounders=", ".join(confounders) or "Nenhum",
            mediators=", ".join(mediators) or "Nenhum",
        )
        
        response = self.engine.generate(
            prompt,
            GenerationConfig(max_tokens=1024, temperature=0.3),
        )
        
        return self._parse_causal_response(response.text)
    
    def _parse_causal_response(self, response: str) -> dict[str, Any]:
        """Parse LLM causal reasoning response."""
        result = {
            "answer": "",
            "confidence": 0.5,
            "reasoning": [],
        }
        
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("ANÁLISE:"):
                current_section = "analysis"
                result["reasoning"].append(line.replace("ANÁLISE:", "").strip())
            elif line.startswith("CAMINHOS:"):
                current_section = "paths"
                result["reasoning"].append(line.replace("CAMINHOS:", "").strip())
            elif line.startswith("CONCLUSÃO:"):
                result["answer"] = line.replace("CONCLUSÃO:", "").strip()
            elif line.startswith("CONFIANÇA:"):
                try:
                    result["confidence"] = float(
                        line.replace("CONFIANÇA:", "").strip()
                    )
                except ValueError:
                    pass
            elif current_section and line:
                result["reasoning"].append(line)
        
        return result
    
    def estimate_effect(
        self,
        treatment: str,
        outcome: str,
        treatment_value: Any = 1,
        control_value: Any = 0,
    ) -> dict[str, Any]:
        """
        Estimate the causal effect of treatment on outcome.
        
        Uses do-calculus conceptually.
        """
        query = CausalQuery(
            query_type="effect",
            treatment=treatment,
            outcome=outcome,
            intervention_value=treatment_value,
        )
        
        answer = self.query(query)
        
        # Create intervened graph
        treated_graph = self.graph.intervene(treatment, treatment_value)
        control_graph = self.graph.intervene(treatment, control_value)
        
        return {
            "treatment": treatment,
            "outcome": outcome,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "confounders": answer.confounders,
            "needs_adjustment": len(answer.confounders) > 0,
            "adjustment_set": answer.confounders,
        }
    
    def counterfactual(
        self,
        factual: dict[str, Any],
        intervention: dict[str, Any],
        outcome: str,
    ) -> dict[str, Any]:
        """
        Answer a counterfactual question.
        
        "Given that X happened, what would Y have been if we had done Z?"
        
        Args:
            factual: What actually happened {var: value}.
            intervention: What if we had done {var: value}.
            outcome: What would have happened to this variable?
            
        Returns:
            Counterfactual result.
        """
        # Build counterfactual query
        treatment = list(intervention.keys())[0]
        
        query = CausalQuery(
            query_type="counterfactual",
            treatment=treatment,
            outcome=outcome,
            condition=factual,
            intervention_value=intervention[treatment],
        )
        
        answer = self.query(query)
        
        return {
            "factual": factual,
            "intervention": intervention,
            "outcome": outcome,
            "counterfactual_answer": answer.answer,
            "confidence": answer.confidence,
            "reasoning": answer.reasoning,
        }
