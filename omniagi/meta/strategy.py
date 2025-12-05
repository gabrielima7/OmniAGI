"""
Strategy Bank - Store and retrieve successful strategies.

Maintains a collection of problem-solving strategies
indexed by domain, task type, and effectiveness.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = structlog.get_logger()


@dataclass
class Strategy:
    """A reusable problem-solving strategy."""
    
    id: str
    name: str
    domain: str  # coding, research, writing, etc.
    task_type: str  # debug, refactor, explain, etc.
    
    # Strategy content
    description: str
    prompt_template: str
    
    # Performance metrics
    uses: int = 0
    successes: int = 0
    failures: int = 0
    avg_quality_score: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime | None = None
    tags: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses
    
    @property
    def effectiveness_score(self) -> float:
        """Calculate overall effectiveness score."""
        if self.uses == 0:
            return 0.5  # Default for unused strategies
        
        # Combine success rate and quality
        success_weight = 0.6
        quality_weight = 0.4
        
        return (
            success_weight * self.success_rate +
            quality_weight * self.avg_quality_score
        )
    
    def record_use(self, success: bool, quality_score: float = 0.0) -> None:
        """Record a use of this strategy."""
        self.uses += 1
        self.last_used = datetime.now()
        
        if success:
            self.successes += 1
        else:
            self.failures += 1
        
        # Update running average of quality
        if quality_score > 0:
            total_quality = self.avg_quality_score * (self.uses - 1) + quality_score
            self.avg_quality_score = total_quality / self.uses
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["last_used"] = self.last_used.isoformat() if self.last_used else None
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Strategy":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data["last_used"]:
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)


class StrategyBank:
    """
    Bank of problem-solving strategies.
    
    Stores, indexes, and retrieves strategies based on:
    - Domain (coding, research, etc.)
    - Task type (debug, explain, etc.)
    - Effectiveness metrics
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize strategy bank.
        
        Args:
            storage_path: Path for persistent storage.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._strategies: dict[str, Strategy] = {}
        
        # Indexes for fast lookup
        self._by_domain: dict[str, list[str]] = {}
        self._by_task_type: dict[str, list[str]] = {}
        
        if self.storage_path:
            self._load()
        
        # Add default strategies if empty
        if not self._strategies:
            self._add_defaults()
    
    def _load(self) -> None:
        """Load strategies from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            for item in data:
                strategy = Strategy.from_dict(item)
                self._strategies[strategy.id] = strategy
                self._index(strategy)
            
            logger.info("Loaded strategies", count=len(self._strategies))
        except Exception as e:
            logger.error("Failed to load strategies", error=str(e))
    
    def _save(self) -> None:
        """Save strategies to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [s.to_dict() for s in self._strategies.values()]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _index(self, strategy: Strategy) -> None:
        """Add strategy to indexes."""
        if strategy.domain not in self._by_domain:
            self._by_domain[strategy.domain] = []
        if strategy.id not in self._by_domain[strategy.domain]:
            self._by_domain[strategy.domain].append(strategy.id)
        
        if strategy.task_type not in self._by_task_type:
            self._by_task_type[strategy.task_type] = []
        if strategy.id not in self._by_task_type[strategy.task_type]:
            self._by_task_type[strategy.task_type].append(strategy.id)
    
    def add(self, strategy: Strategy) -> None:
        """Add a strategy to the bank."""
        self._strategies[strategy.id] = strategy
        self._index(strategy)
        self._save()
        
        logger.info("Strategy added", id=strategy.id, name=strategy.name)
    
    def get(self, strategy_id: str) -> Strategy | None:
        """Get a strategy by ID."""
        return self._strategies.get(strategy_id)
    
    def find(
        self,
        domain: str | None = None,
        task_type: str | None = None,
        min_effectiveness: float = 0.0,
        limit: int = 5,
    ) -> list[Strategy]:
        """
        Find strategies matching criteria.
        
        Args:
            domain: Filter by domain.
            task_type: Filter by task type.
            min_effectiveness: Minimum effectiveness score.
            limit: Maximum results.
            
        Returns:
            List of matching strategies, sorted by effectiveness.
        """
        candidates = set(self._strategies.keys())
        
        # Filter by domain
        if domain and domain in self._by_domain:
            candidates &= set(self._by_domain[domain])
        
        # Filter by task type
        if task_type and task_type in self._by_task_type:
            candidates &= set(self._by_task_type[task_type])
        
        # Get strategies and filter by effectiveness
        strategies = [
            self._strategies[id] for id in candidates
            if self._strategies[id].effectiveness_score >= min_effectiveness
        ]
        
        # Sort by effectiveness
        strategies.sort(key=lambda s: s.effectiveness_score, reverse=True)
        
        return strategies[:limit]
    
    def get_best(
        self,
        domain: str,
        task_type: str,
    ) -> Strategy | None:
        """Get the best strategy for a domain/task combination."""
        strategies = self.find(domain=domain, task_type=task_type, limit=1)
        return strategies[0] if strategies else None
    
    def record_result(
        self,
        strategy_id: str,
        success: bool,
        quality_score: float = 0.0,
    ) -> None:
        """Record the result of using a strategy."""
        strategy = self._strategies.get(strategy_id)
        if strategy:
            strategy.record_use(success, quality_score)
            self._save()
    
    def _add_defaults(self) -> None:
        """Add default strategies."""
        defaults = [
            Strategy(
                id="debug_systematic",
                name="Systematic Debugging",
                domain="coding",
                task_type="debug",
                description="Step-by-step debugging approach",
                prompt_template="""Analise o bug sistematicamente:

1. **Reproduzir**: {reprodution_steps}
2. **Isolar**: Identifique a menor porção de código que causa o problema
3. **Hipóteses**: Liste 3 possíveis causas
4. **Testar**: Verifique cada hipótese
5. **Corrigir**: Aplique a correção mínima necessária
6. **Validar**: Confirme que o bug foi resolvido

Código com bug:
```python
{code}
```

Erro observado:
{error}
""",
                tags=["debug", "systematic", "step-by-step"],
            ),
            Strategy(
                id="refactor_solid",
                name="SOLID Refactoring",
                domain="coding",
                task_type="refactor",
                description="Refactoring using SOLID principles",
                prompt_template="""Refatore o código seguindo princípios SOLID:

- **S**ingle Responsibility: Uma razão para mudar
- **O**pen/Closed: Aberto para extensão, fechado para modificação
- **L**iskov Substitution: Subtipos substituíveis
- **I**nterface Segregation: Interfaces específicas
- **D**ependency Inversion: Depender de abstrações

Código original:
```python
{code}
```

Foco em: {focus_area}
""",
                tags=["refactor", "solid", "clean-code"],
            ),
            Strategy(
                id="explain_feynman",
                name="Feynman Explanation",
                domain="general",
                task_type="explain",
                description="Explain like teaching a beginner",
                prompt_template="""Explique usando a técnica Feynman:

1. **Simplifique**: Use linguagem simples, sem jargão
2. **Analogias**: Compare com conceitos cotidianos
3. **Identifique gaps**: Onde você travou?
4. **Refine**: Simplifique ainda mais

Tópico: {topic}
Público: {audience}
""",
                tags=["explain", "feynman", "teaching"],
            ),
            Strategy(
                id="research_structured",
                name="Structured Research",
                domain="research",
                task_type="research",
                description="Systematic research approach",
                prompt_template="""Pesquise de forma estruturada:

1. **Definir questão**: {question}
2. **Fontes primárias**: Papers, documentação oficial
3. **Fontes secundárias**: Blogs, tutoriais
4. **Síntese**: Combine informações
5. **Validação**: Verifique consistência
6. **Conclusão**: Responda à questão original

Formato de saída: {output_format}
""",
                tags=["research", "systematic", "academic"],
            ),
        ]
        
        for strategy in defaults:
            self._strategies[strategy.id] = strategy
            self._index(strategy)
        
        self._save()
        logger.info("Added default strategies", count=len(defaults))
    
    def domains(self) -> list[str]:
        """Get all available domains."""
        return list(self._by_domain.keys())
    
    def task_types(self) -> list[str]:
        """Get all available task types."""
        return list(self._by_task_type.keys())
    
    def __len__(self) -> int:
        return len(self._strategies)
