"""
Strategy Adapter - Adapt strategies to new domains.

Transfers and modifies strategies for use in different contexts.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

from omniagi.meta.strategy import Strategy, StrategyBank

logger = structlog.get_logger()


@dataclass
class AdaptedPrompt:
    """A strategy prompt adapted for a specific task."""
    
    original_strategy: Strategy
    adapted_prompt: str
    context: dict[str, str]
    confidence: float  # How confident we are this adaptation will work


ADAPTATION_PROMPT = '''Você é um especialista em adaptar estratégias de resolução de problemas.

## Estratégia Original:
Nome: {strategy_name}
Domínio: {strategy_domain}
Tipo: {strategy_type}
Template:
{strategy_template}

## Nova Tarefa:
Domínio: {target_domain}
Tipo: {target_type}
Contexto: {target_context}

## Instruções:
Adapte a estratégia original para o novo contexto. Mantenha a estrutura
e princípios, mas ajuste a linguagem e exemplos para o novo domínio.

Responda com o prompt adaptado:
'''


class StrategyAdapter:
    """
    Adapts strategies for use in new domains and contexts.
    
    Features:
    - Cross-domain transfer
    - Context-aware adaptation
    - Few-shot example generation
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        strategy_bank: StrategyBank | None = None,
    ):
        """
        Initialize the adapter.
        
        Args:
            engine: LLM engine for adaptations.
            strategy_bank: Bank of available strategies.
        """
        self.engine = engine
        self.bank = strategy_bank or StrategyBank()
    
    def adapt(
        self,
        strategy: Strategy,
        target_domain: str,
        target_type: str,
        context: dict[str, str],
    ) -> AdaptedPrompt:
        """
        Adapt a strategy for a new context.
        
        Args:
            strategy: The strategy to adapt.
            target_domain: Target domain.
            target_type: Target task type.
            context: Context variables for the prompt.
            
        Returns:
            AdaptedPrompt with the adapted content.
        """
        # If same domain and type, just fill in the template
        if strategy.domain == target_domain and strategy.task_type == target_type:
            adapted = self._fill_template(strategy.prompt_template, context)
            return AdaptedPrompt(
                original_strategy=strategy,
                adapted_prompt=adapted,
                context=context,
                confidence=0.9,
            )
        
        # Otherwise, use LLM to adapt
        if self.engine and self.engine.is_loaded:
            adapted = self._llm_adapt(strategy, target_domain, target_type, context)
            return AdaptedPrompt(
                original_strategy=strategy,
                adapted_prompt=adapted,
                context=context,
                confidence=0.7,
            )
        
        # Fallback: simple template fill
        adapted = self._fill_template(strategy.prompt_template, context)
        return AdaptedPrompt(
            original_strategy=strategy,
            adapted_prompt=adapted,
            context=context,
            confidence=0.5,
        )
    
    def find_and_adapt(
        self,
        target_domain: str,
        target_type: str,
        context: dict[str, str],
    ) -> AdaptedPrompt | None:
        """
        Find the best strategy and adapt it for the task.
        
        Args:
            target_domain: Target domain.
            target_type: Target task type.
            context: Context variables.
            
        Returns:
            AdaptedPrompt or None if no strategy found.
        """
        # First, try exact match
        strategy = self.bank.get_best(target_domain, target_type)
        
        if strategy:
            return self.adapt(strategy, target_domain, target_type, context)
        
        # Try just domain match
        strategies = self.bank.find(domain=target_domain, limit=1)
        if strategies:
            return self.adapt(strategies[0], target_domain, target_type, context)
        
        # Try just task type match
        strategies = self.bank.find(task_type=target_type, limit=1)
        if strategies:
            return self.adapt(strategies[0], target_domain, target_type, context)
        
        # Use any strategy with high effectiveness
        strategies = self.bank.find(min_effectiveness=0.7, limit=1)
        if strategies:
            return self.adapt(strategies[0], target_domain, target_type, context)
        
        return None
    
    def _fill_template(
        self,
        template: str,
        context: dict[str, str],
    ) -> str:
        """Fill a template with context variables."""
        result = template
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result
    
    def _llm_adapt(
        self,
        strategy: Strategy,
        target_domain: str,
        target_type: str,
        context: dict[str, str],
    ) -> str:
        """Use LLM to adapt the strategy."""
        from omniagi.core.engine import GenerationConfig
        
        prompt = ADAPTATION_PROMPT.format(
            strategy_name=strategy.name,
            strategy_domain=strategy.domain,
            strategy_type=strategy.task_type,
            strategy_template=strategy.prompt_template,
            target_domain=target_domain,
            target_type=target_type,
            target_context="\n".join(f"- {k}: {v}" for k, v in context.items()),
        )
        
        response = self.engine.generate(
            prompt,
            GenerationConfig(max_tokens=1024, temperature=0.4),
        )
        
        # Fill in the context
        adapted = self._fill_template(response.text, context)
        
        return adapted
    
    def generate_examples(
        self,
        strategy: Strategy,
        domain: str,
        n_examples: int = 3,
    ) -> list[dict[str, str]]:
        """
        Generate few-shot examples for a strategy.
        
        Args:
            strategy: The strategy to generate examples for.
            domain: Domain for examples.
            n_examples: Number of examples.
            
        Returns:
            List of example input/output pairs.
        """
        if not self.engine or not self.engine.is_loaded:
            return strategy.examples[:n_examples]
        
        from omniagi.core.engine import GenerationConfig
        
        prompt = f'''Gere {n_examples} exemplos de uso da estratégia "{strategy.name}" no domínio "{domain}".

Estratégia:
{strategy.description}

Para cada exemplo, forneça:
- INPUT: A entrada/problema
- OUTPUT: O resultado esperado

Exemplos:'''
        
        response = self.engine.generate(
            prompt,
            GenerationConfig(max_tokens=1024, temperature=0.7),
        )
        
        # Parse examples from response
        examples = []
        current_input = ""
        current_output = ""
        
        for line in response.text.split("\n"):
            if line.strip().startswith("INPUT:"):
                current_input = line.replace("INPUT:", "").strip()
            elif line.strip().startswith("OUTPUT:"):
                current_output = line.replace("OUTPUT:", "").strip()
                if current_input and current_output:
                    examples.append({
                        "input": current_input,
                        "output": current_output,
                    })
                    current_input = ""
                    current_output = ""
        
        return examples[:n_examples]
