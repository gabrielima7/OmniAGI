"""
AGI Brain - Unified integration of all AGI pillars.

This module connects all cognitive capabilities into a cohesive
autonomous intelligence that can learn, reason, plan, and improve.
"""

from __future__ import annotations

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

from omniagi.core.config import get_config
from omniagi.agent.base import Agent

# AGI Pillars
from omniagi.ouroboros import OuroborosLoop, CodeAnalyzer
from omniagi.meta import MetaLearner, StrategyBank
from omniagi.continual import MemoryConsolidator, KnowledgeGraph
from omniagi.causal import CausalGraph, CausalReasoner, DecisionExplainer
from omniagi.world import WorldState, MentalSimulator, HierarchicalPlanner

logger = structlog.get_logger()


@dataclass
class CognitiveState:
    """Current cognitive state of the AGI."""
    
    attention_focus: str | None = None
    current_goal: str | None = None
    emotional_state: str = "neutral"  # For future affect modeling
    energy_level: float = 1.0  # Computational budget
    
    last_thought: str = ""
    pending_decisions: list[str] = field(default_factory=list)
    
    # Metrics
    decisions_made: int = 0
    goals_completed: int = 0
    self_improvements: int = 0


class AGIBrain:
    """
    Unified AGI Brain integrating all cognitive pillars.
    
    This is the "consciousness" layer that coordinates:
    - Ouroboros: Self-improvement
    - Meta-Learning: Learn how to learn
    - Continual Learning: Never forget
    - Causal Reasoning: Understand why
    - World Model: Simulate and plan
    """
    
    def __init__(
        self,
        agent: Agent,
        engine: "Engine | None" = None,
        work_dir: str = ".",
    ):
        """
        Initialize the AGI Brain.
        
        Args:
            agent: Base agent for execution.
            engine: LLM engine.
            work_dir: Working directory for self-improvement.
        """
        self.agent = agent
        self.engine = engine or agent.engine
        self.config = get_config()
        
        # Initialize cognitive state
        self.state = CognitiveState()
        
        # Initialize AGI Pillars
        self._init_pillars(work_dir)
        
        logger.info("AGI Brain initialized with all pillars")
    
    def _init_pillars(self, work_dir: str) -> None:
        """Initialize all AGI pillars."""
        from pathlib import Path
        
        data_dir = Path(self.config.data_dir) / "agi"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 🐍 Ouroboros - Self-improvement
        self.ouroboros = OuroborosLoop(
            engine=self.engine,
            work_dir=work_dir,
            require_approval=True,  # Safety first
        )
        
        # 🧠 Meta-Learning
        self.strategy_bank = StrategyBank(
            storage_path=data_dir / "strategies.json"
        )
        self.meta_learner = MetaLearner(
            engine=self.engine,
            strategy_bank=self.strategy_bank,
        )
        
        # 📚 Continual Learning
        self.memory_consolidator = MemoryConsolidator(
            storage_path=data_dir / "consolidated_memory.json"
        )
        self.knowledge_graph = KnowledgeGraph(
            storage_path=data_dir / "knowledge.json"
        )
        
        # ⚡ Causal Reasoning
        self.causal_graph = CausalGraph(
            storage_path=data_dir / "causal_model.json"
        )
        self.causal_reasoner = CausalReasoner(
            graph=self.causal_graph,
            engine=self.engine,
        )
        self.decision_explainer = DecisionExplainer(
            causal_graph=self.causal_graph
        )
        
        # 🌍 World Model
        self.world_state = WorldState(
            storage_path=data_dir / "world_state.json"
        )
        self.simulator = MentalSimulator(
            world_state=self.world_state,
            engine=self.engine,
        )
        self.planner = HierarchicalPlanner(
            world_state=self.world_state,
            simulator=self.simulator,
            engine=self.engine,
        )
    
    async def think(self, input_text: str) -> dict[str, Any]:
        """
        Main thinking process integrating all pillars.
        
        Args:
            input_text: The input to process.
            
        Returns:
            Comprehensive response with reasoning.
        """
        start_time = datetime.now()
        self.state.attention_focus = input_text[:100]
        
        result = {
            "input": input_text,
            "response": "",
            "reasoning": [],
            "decisions": [],
            "plan": None,
            "meta": {},
        }
        
        try:
            # 1. Update world model with input
            self._update_world_from_input(input_text)
            
            # 2. Get meta-learning suggestion for approach
            approach = self.meta_learner.suggest_approach(
                task_description=input_text,
                domain=self._detect_domain(input_text),
                task_type=self._detect_task_type(input_text),
            )
            result["meta"]["approach"] = approach
            result["reasoning"].append(f"Estratégia: {approach.get('strategy', 'default')}")
            
            # 3. Simulate potential responses
            if self.engine and self.engine.is_loaded:
                prediction = self.simulator.simulate_action(
                    action=f"respond to: {input_text[:50]}",
                    actor_id="self",
                    use_llm=True,
                )
                result["reasoning"].append(f"Previsão: {prediction.predicted_effects[:2]}")
            
            # 4. Plan if needed
            if self._needs_planning(input_text):
                goal = self.planner.create_goal(
                    description=input_text,
                    priority=self._assess_priority(input_text),
                )
                plan = self.planner.plan_for_goal(goal.id)
                result["plan"] = {
                    "goal_id": goal.id,
                    "steps": len(plan.steps) if plan else 0,
                }
            
            # 5. Execute through agent with enhanced prompt
            enhanced_prompt = self._enhance_prompt(input_text, approach, result)
            response = await self.agent.run(enhanced_prompt)
            result["response"] = response
            
            # 6. Record decision for explainability
            decision = self.decision_explainer.record_decision(
                decision=f"Responded to: {input_text[:50]}",
                action_taken="Generated response",
                inputs={"input": input_text[:100]},
                reasoning_steps=result["reasoning"],
                causes=[approach.get("strategy", "default_strategy")],
            )
            result["decisions"].append(decision.id)
            
            # 7. Learn from this interaction
            await self._learn_from_interaction(input_text, response, approach)
            
            # 8. Update state
            self.state.decisions_made += 1
            self.state.last_thought = input_text[:100]
            
        except Exception as e:
            logger.error("Thinking error", error=str(e))
            result["error"] = str(e)
            result["response"] = await self.agent.run(input_text)  # Fallback
        
        result["meta"]["thinking_time_ms"] = (
            datetime.now() - start_time
        ).total_seconds() * 1000
        
        return result
    
    def _update_world_from_input(self, input_text: str) -> None:
        """Update world state based on input."""
        # Add self if not present
        if not self.world_state.get_entity("self"):
            self.world_state.add_entity(
                name="OmniAGI",
                entity_type="agent",
                entity_id="self",
                state={"status": "active"},
            )
        
        # Add user if interacting
        if not self.world_state.get_entity("user"):
            self.world_state.add_entity(
                name="User",
                entity_type="agent",
                entity_id="user",
            )
            self.world_state.add_relation("user", "self", "interacting_with")
        
        # Update last interaction
        self.world_state.update_entity(
            "self",
            state={"last_input": input_text[:100], "last_activity": datetime.now().isoformat()},
        )
    
    def _detect_domain(self, text: str) -> str:
        """Detect the domain of the input."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["código", "code", "python", "função", "bug"]):
            return "coding"
        elif any(kw in text_lower for kw in ["pesquise", "search", "encontre", "what is"]):
            return "research"
        elif any(kw in text_lower for kw in ["explique", "explain", "como", "how"]):
            return "explanation"
        elif any(kw in text_lower for kw in ["escreva", "write", "crie", "create"]):
            return "creation"
        return "general"
    
    def _detect_task_type(self, text: str) -> str:
        """Detect the task type."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["debug", "fix", "erro", "error"]):
            return "debug"
        elif any(kw in text_lower for kw in ["refatore", "refactor", "melhore", "improve"]):
            return "refactor"
        elif any(kw in text_lower for kw in ["explique", "explain"]):
            return "explain"
        elif any(kw in text_lower for kw in ["planeje", "plan"]):
            return "plan"
        return "general"
    
    def _needs_planning(self, text: str) -> bool:
        """Determine if the task requires planning."""
        keywords = ["planeje", "plan", "crie projeto", "build", "desenvolva", "implement"]
        return any(kw in text.lower() for kw in keywords) or len(text) > 200
    
    def _assess_priority(self, text: str) -> Any:
        """Assess priority of the task."""
        from omniagi.world.planner import GoalPriority
        
        if any(kw in text.lower() for kw in ["urgente", "urgent", "crítico", "critical"]):
            return GoalPriority.CRITICAL
        elif any(kw in text.lower() for kw in ["importante", "important"]):
            return GoalPriority.HIGH
        return GoalPriority.MEDIUM
    
    def _enhance_prompt(
        self,
        original: str,
        approach: dict,
        context: dict,
    ) -> str:
        """Enhance prompt with cognitive context."""
        enhanced = original
        
        # Add consolidated knowledge context
        if len(self.memory_consolidator) > 0:
            knowledge_context = self.memory_consolidator.get_context(limit=3)
            enhanced = f"{knowledge_context}\n\n{enhanced}"
        
        # Add strategy if available
        if approach.get("prompt"):
            enhanced = f"[Estratégia: {approach.get('strategy', 'default')}]\n\n{enhanced}"
        
        return enhanced
    
    async def _learn_from_interaction(
        self,
        input_text: str,
        response: str,
        approach: dict,
    ) -> None:
        """Learn from the interaction."""
        # Add to consolidated memory
        self.memory_consolidator.add(
            content=f"Q: {input_text[:100]} A: {response[:200]}",
            importance=0.5,
            category=self._detect_domain(input_text),
        )
        
        # Update knowledge graph
        domain = self._detect_domain(input_text)
        domain_node = self.knowledge_graph.find_node(domain, "domain")
        if not domain_node:
            domain_node = self.knowledge_graph.add_node(
                name=domain,
                node_type="domain",
            )
        
        # Record meta-learning experience
        self.meta_learner.record_experience(
            task_description=input_text[:100],
            domain=domain,
            task_type=self._detect_task_type(input_text),
            strategy_id=approach.get("strategy"),
            prompt_used=input_text,
            success=True,  # Would be determined by feedback
            quality_score=0.7,  # Would be determined by evaluation
            duration_seconds=1.0,
        )
    
    async def consolidate(self) -> dict[str, int]:
        """
        Run memory consolidation (like sleep).
        
        Should be called periodically to consolidate learning.
        """
        logger.info("Running memory consolidation...")
        
        stats = self.memory_consolidator.consolidate()
        
        # Also update knowledge graph links
        # (would do more sophisticated processing here)
        
        return stats
    
    async def self_improve(self, target_dir: str | None = None) -> dict[str, Any]:
        """
        Run self-improvement cycle using Ouroboros.
        
        Args:
            target_dir: Directory to improve (None for default).
            
        Returns:
            Improvement results.
        """
        logger.info("Running self-improvement cycle...")
        
        results = await self.ouroboros.improve_directory(
            directory=target_dir,
            max_files=5,
        )
        
        successful = sum(1 for r in results if r.state.name == "COMPLETE")
        self.state.self_improvements += successful
        
        return {
            "files_analyzed": len(results),
            "successful_improvements": successful,
            "details": [
                {
                    "file": r.file_path,
                    "state": r.state.name,
                    "changes": len(r.refactor_result.changes_made) if r.refactor_result else 0,
                }
                for r in results
            ],
        }
    
    def explain_decision(self, decision_id: str) -> str:
        """Get explanation for a decision."""
        explanation = self.decision_explainer.explain(decision_id)
        return explanation.summary
    
    def what_if(self, condition: str) -> dict[str, Any]:
        """Answer a counterfactual question."""
        return self.simulator.what_if(condition, "self")
    
    def get_status(self) -> dict[str, Any]:
        """Get current cognitive status."""
        return {
            "state": {
                "focus": self.state.attention_focus,
                "goal": self.state.current_goal,
                "energy": self.state.energy_level,
                "last_thought": self.state.last_thought,
            },
            "metrics": {
                "decisions_made": self.state.decisions_made,
                "goals_completed": self.state.goals_completed,
                "self_improvements": self.state.self_improvements,
            },
            "pillars": {
                "memories": len(self.memory_consolidator),
                "knowledge_nodes": len(self.knowledge_graph),
                "strategies": len(self.strategy_bank),
                "world_entities": len(self.world_state),
                "goals": len(self.planner._goals),
            },
        }
