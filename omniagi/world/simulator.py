"""
Mental Simulator - Imagine consequences before acting.

Simulates actions in an internal world model to predict
outcomes without actually executing them.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

from omniagi.world.state import WorldState, Entity

logger = structlog.get_logger()


@dataclass
class Prediction:
    """A predicted outcome from simulation."""
    
    action: str
    actor_id: str
    target_id: str | None
    
    predicted_state: dict[str, Any]
    predicted_effects: list[str]
    
    probability: float  # 0.0 - 1.0
    confidence: float
    
    risks: list[str] = field(default_factory=list)
    benefits: list[str] = field(default_factory=list)
    
    simulated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimulationResult:
    """Result of a multi-step simulation."""
    
    initial_state: WorldState
    final_state: WorldState
    steps: list[Prediction]
    
    total_probability: float
    expected_value: float
    
    trajectory: list[str]  # Description of what happened


SIMULATION_PROMPT = '''Você é um simulador de consequências. Dado o estado atual e uma ação proposta, preveja o que acontecerá.

## Estado Atual:
{world_context}

## Ação Proposta:
Ator: {actor}
Ação: {action}
Alvo: {target}

## Instruções:
1. Simule o que aconteceria se esta ação fosse executada
2. Considere efeitos diretos e indiretos
3. Identifique riscos e benefícios
4. Estime a probabilidade de sucesso

Responda no formato:
EFEITOS_DIRETOS:
- [efeito 1]
- [efeito 2]

EFEITOS_INDIRETOS:
- [efeito secundário]

RISCOS:
- [risco potencial]

BENEFÍCIOS:
- [benefício esperado]

PROBABILIDADE_SUCESSO: [0.0-1.0]
'''


class MentalSimulator:
    """
    Simulates actions in an internal world model.
    
    "Imagines" consequences before executing actions,
    enabling safer and more informed decision-making.
    
    Based on the concept of mental simulation from cognitive science.
    """
    
    def __init__(
        self,
        world_state: WorldState | None = None,
        engine: "Engine | None" = None,
        max_depth: int = 5,
    ):
        """
        Initialize simulator.
        
        Args:
            world_state: The world state to simulate against.
            engine: LLM engine for complex predictions.
            max_depth: Maximum simulation depth.
        """
        self.world = world_state or WorldState()
        self.engine = engine
        self.max_depth = max_depth
        
        self._simulation_cache: dict[str, Prediction] = {}
    
    def simulate_action(
        self,
        action: str,
        actor_id: str,
        target_id: str | None = None,
        use_llm: bool = True,
    ) -> Prediction:
        """
        Simulate a single action and predict outcomes.
        
        Args:
            action: The action to simulate.
            actor_id: Who is performing the action.
            target_id: Target of the action (if any).
            use_llm: Use LLM for richer predictions.
            
        Returns:
            Prediction with expected outcomes.
        """
        # Check cache
        cache_key = f"{action}:{actor_id}:{target_id}"
        if cache_key in self._simulation_cache:
            return self._simulation_cache[cache_key]
        
        actor = self.world.get_entity(actor_id)
        target = self.world.get_entity(target_id) if target_id else None
        
        if not actor:
            return Prediction(
                action=action,
                actor_id=actor_id,
                target_id=target_id,
                predicted_state={},
                predicted_effects=["Ator não encontrado"],
                probability=0.0,
                confidence=0.0,
            )
        
        # Create simulation world (clone)
        sim_world = self.world.clone()
        
        # Basic effect prediction
        effects = []
        risks = []
        benefits = []
        probability = 0.7  # Default
        
        # Simple rule-based prediction
        if "move" in action.lower() or "go" in action.lower():
            effects.append(f"{actor.name} muda de localização")
            benefits.append("Novo contexto, novas oportunidades")
            
        elif "create" in action.lower() or "make" in action.lower():
            effects.append(f"Novo item criado")
            benefits.append("Recurso disponível")
            risks.append("Pode consumir recursos")
            
        elif "delete" in action.lower() or "remove" in action.lower():
            effects.append(f"Item removido")
            risks.append("Ação irreversível")
            probability = 0.5  # Higher risk
            
        elif "modify" in action.lower() or "change" in action.lower():
            effects.append(f"Estado modificado")
            if target:
                effects.append(f"{target.name} será alterado")
        
        # Use LLM for richer predictions
        if use_llm and self.engine and self.engine.is_loaded:
            llm_result = self._llm_simulate(action, actor, target)
            effects.extend(llm_result.get("effects", []))
            risks.extend(llm_result.get("risks", []))
            benefits.extend(llm_result.get("benefits", []))
            probability = llm_result.get("probability", probability)
        
        # Apply simulated effects to clone
        if target_id:
            sim_world.apply_action(
                action=action,
                actor_id=actor_id,
                target_id=target_id,
                effects={"simulated": True, "action": action},
            )
        
        prediction = Prediction(
            action=action,
            actor_id=actor_id,
            target_id=target_id,
            predicted_state=sim_world._entities.get(target_id, Entity("", "", "")).state if target_id else {},
            predicted_effects=effects,
            probability=probability,
            confidence=0.7 if use_llm else 0.5,
            risks=risks,
            benefits=benefits,
        )
        
        # Cache result
        self._simulation_cache[cache_key] = prediction
        
        logger.info(
            "Action simulated",
            action=action,
            probability=probability,
            effects=len(effects),
        )
        
        return prediction
    
    def _llm_simulate(
        self,
        action: str,
        actor: Entity,
        target: Entity | None,
    ) -> dict[str, Any]:
        """Use LLM for richer simulation."""
        from omniagi.core.engine import GenerationConfig
        
        prompt = SIMULATION_PROMPT.format(
            world_context=self.world.to_context(actor.id),
            actor=actor.name,
            action=action,
            target=target.name if target else "Nenhum",
        )
        
        response = self.engine.generate(
            prompt,
            GenerationConfig(max_tokens=512, temperature=0.4),
        )
        
        return self._parse_simulation_response(response.text)
    
    def _parse_simulation_response(self, response: str) -> dict[str, Any]:
        """Parse LLM simulation response."""
        result = {
            "effects": [],
            "risks": [],
            "benefits": [],
            "probability": 0.7,
        }
        
        current_section = None
        
        for line in response.split("\n"):
            line = line.strip()
            
            if line.startswith("EFEITOS"):
                current_section = "effects"
            elif line.startswith("RISCOS"):
                current_section = "risks"
            elif line.startswith("BENEFÍCIOS"):
                current_section = "benefits"
            elif line.startswith("PROBABILIDADE"):
                try:
                    result["probability"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
            elif line.startswith("-") and current_section:
                result[current_section].append(line.lstrip("- "))
        
        return result
    
    def simulate_sequence(
        self,
        actions: list[tuple[str, str, str | None]],  # (action, actor, target)
    ) -> SimulationResult:
        """
        Simulate a sequence of actions.
        
        Args:
            actions: List of (action, actor_id, target_id) tuples.
            
        Returns:
            SimulationResult with trajectory.
        """
        initial_state = self.world.clone()
        sim_world = self.world.clone()
        
        steps = []
        trajectory = []
        total_prob = 1.0
        
        for action, actor_id, target_id in actions[:self.max_depth]:
            # Temporarily use sim_world
            original_world = self.world
            self.world = sim_world
            
            prediction = self.simulate_action(action, actor_id, target_id)
            steps.append(prediction)
            
            # Restore original
            self.world = original_world
            
            # Update simulation world
            if target_id:
                sim_world.apply_action(action, actor_id, target_id)
            
            total_prob *= prediction.probability
            trajectory.append(
                f"{action} ({prediction.probability:.0%}): {', '.join(prediction.predicted_effects[:2])}"
            )
        
        return SimulationResult(
            initial_state=initial_state,
            final_state=sim_world,
            steps=steps,
            total_probability=total_prob,
            expected_value=total_prob * len(steps),  # Simple heuristic
            trajectory=trajectory,
        )
    
    def lookahead(
        self,
        current_entity_id: str,
        possible_actions: list[str],
        depth: int = 3,
    ) -> list[tuple[str, float, list[str]]]:
        """
        Look ahead to evaluate possible actions.
        
        Returns actions sorted by expected value.
        """
        evaluations = []
        
        for action in possible_actions:
            prediction = self.simulate_action(action, current_entity_id)
            
            score = (
                prediction.probability * 0.5 +
                len(prediction.benefits) * 0.1 -
                len(prediction.risks) * 0.15
            )
            
            evaluations.append((
                action,
                score,
                prediction.predicted_effects,
            ))
        
        # Sort by score descending
        evaluations.sort(key=lambda x: x[1], reverse=True)
        
        return evaluations
    
    def what_if(
        self,
        condition: str,
        entity_id: str,
    ) -> dict[str, Any]:
        """
        Answer a "what if" question.
        
        Args:
            condition: The hypothetical condition.
            entity_id: The entity to focus on.
            
        Returns:
            Dictionary with hypothetical analysis.
        """
        entity = self.world.get_entity(entity_id)
        if not entity:
            return {"error": "Entity not found"}
        
        # Simulate the condition as an action
        prediction = self.simulate_action(
            action=condition,
            actor_id=entity_id,
            use_llm=True,
        )
        
        return {
            "condition": condition,
            "entity": entity.name,
            "likely_effects": prediction.predicted_effects,
            "risks": prediction.risks,
            "benefits": prediction.benefits,
            "probability": prediction.probability,
        }
