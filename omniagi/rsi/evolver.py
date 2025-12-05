"""
Agent Evolver - Evolutionary improvement of agents.

Implements genetic/evolutionary approaches to improve
agent prompts, strategies, and configurations.
"""

from __future__ import annotations

import json
import random
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

logger = structlog.get_logger()


class EvolutionTarget(Enum):
    """What can be evolved."""
    
    PROMPT = auto()         # System prompts
    STRATEGY = auto()       # Problem-solving strategies
    HYPERPARAMETER = auto() # Configuration parameters
    AGENT = auto()          # Full agent configurations


@dataclass
class Individual:
    """An individual in the evolutionary population."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    target: EvolutionTarget = EvolutionTarget.PROMPT
    
    # Genetic material
    genes: dict = field(default_factory=dict)
    
    # Fitness
    fitness: float = 0.0
    evaluated: bool = False
    
    # Lineage
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target": self.target.name,
            "genes": self.genes,
            "fitness": self.fitness,
            "evaluated": self.evaluated,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Individual":
        return cls(
            id=data.get("id", str(uuid4())[:8]),
            target=EvolutionTarget[data.get("target", "PROMPT")],
            genes=data.get("genes", {}),
            fitness=data.get("fitness", 0.0),
            evaluated=data.get("evaluated", False),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


@dataclass
class Evolution:
    """Record of an evolution run."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    target: EvolutionTarget = EvolutionTarget.PROMPT
    
    # Configuration
    population_size: int = 10
    generations: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_count: int = 2
    
    # State
    current_generation: int = 0
    best_fitness: float = 0.0
    best_individual_id: str | None = None
    
    # History
    fitness_history: list[float] = field(default_factory=list)
    
    # Metadata
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target": self.target.name,
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elite_count": self.elite_count,
            "current_generation": self.current_generation,
            "best_fitness": self.best_fitness,
            "best_individual_id": self.best_individual_id,
            "fitness_history": self.fitness_history,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class AgentEvolver:
    """
    Evolutionary agent improvement system.
    
    Uses genetic algorithms to evolve prompts,
    strategies, and configurations for better performance.
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        storage_path: Path | str | None = None,
    ):
        self.engine = engine
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._populations: dict[str, list[Individual]] = {}
        self._evolutions: dict[str, Evolution] = {}
        self._best_individuals: dict[EvolutionTarget, Individual] = {}
        
        # Gene templates
        self._prompt_genes = {
            "role": ["helpful assistant", "expert problem solver", "careful analyst"],
            "style": ["concise", "detailed", "step-by-step"],
            "approach": ["systematic", "creative", "balanced"],
            "tone": ["professional", "friendly", "neutral"],
        }
        
        self._strategy_genes = {
            "decomposition": ["hierarchical", "sequential", "parallel"],
            "search": ["depth-first", "breadth-first", "best-first"],
            "validation": ["strict", "relaxed", "adaptive"],
        }
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Agent Evolver initialized")
    
    def create_evolution(
        self,
        target: EvolutionTarget,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.1,
    ) -> Evolution:
        """Create a new evolution run."""
        evolution = Evolution(
            target=target,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
        )
        
        # Initialize population
        population = self._initialize_population(target, population_size)
        
        self._evolutions[evolution.id] = evolution
        self._populations[evolution.id] = population
        self._save()
        
        logger.info(
            "Evolution created",
            id=evolution.id,
            target=target.name,
            population=population_size,
        )
        
        return evolution
    
    def _initialize_population(
        self,
        target: EvolutionTarget,
        size: int,
    ) -> list[Individual]:
        """Initialize population with random individuals."""
        population = []
        
        for _ in range(size):
            genes = self._random_genes(target)
            individual = Individual(
                target=target,
                genes=genes,
                generation=0,
            )
            population.append(individual)
        
        return population
    
    def _random_genes(self, target: EvolutionTarget) -> dict:
        """Generate random genes for a target type."""
        if target == EvolutionTarget.PROMPT:
            return {
                key: random.choice(values)
                for key, values in self._prompt_genes.items()
            }
        elif target == EvolutionTarget.STRATEGY:
            return {
                key: random.choice(values)
                for key, values in self._strategy_genes.items()
            }
        elif target == EvolutionTarget.HYPERPARAMETER:
            return {
                "temperature": random.uniform(0.1, 1.0),
                "max_tokens": random.choice([256, 512, 1024]),
                "top_p": random.uniform(0.7, 1.0),
            }
        else:
            return {}
    
    def evaluate_population(
        self,
        evolution_id: str,
        fitness_fn: callable = None,
    ) -> list[float]:
        """Evaluate fitness of all individuals in population."""
        if evolution_id not in self._evolutions:
            return []
        
        evolution = self._evolutions[evolution_id]
        population = self._populations.get(evolution_id, [])
        
        fitness_scores = []
        
        for individual in population:
            if individual.evaluated:
                fitness_scores.append(individual.fitness)
                continue
            
            if fitness_fn:
                fitness = fitness_fn(individual.genes)
            else:
                fitness = self._default_fitness(individual)
            
            individual.fitness = fitness
            individual.evaluated = True
            fitness_scores.append(fitness)
        
        # Update best
        best = max(population, key=lambda x: x.fitness)
        if best.fitness > evolution.best_fitness:
            evolution.best_fitness = best.fitness
            evolution.best_individual_id = best.id
            self._best_individuals[evolution.target] = best
        
        evolution.fitness_history.append(max(fitness_scores))
        self._save()
        
        return fitness_scores
    
    def _default_fitness(self, individual: Individual) -> float:
        """Default fitness function (random for testing)."""
        # In production, this would evaluate actual performance
        base = 0.5
        
        # Bonus for certain gene combinations
        genes = individual.genes
        
        if genes.get("style") == "step-by-step":
            base += 0.1
        if genes.get("approach") == "systematic":
            base += 0.1
        
        # Add some randomness
        base += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base))
    
    def evolve_generation(self, evolution_id: str) -> list[Individual]:
        """Evolve to the next generation."""
        if evolution_id not in self._evolutions:
            return []
        
        evolution = self._evolutions[evolution_id]
        population = self._populations.get(evolution_id, [])
        
        if not population:
            return []
        
        # Ensure population is evaluated
        if not all(ind.evaluated for ind in population):
            self.evaluate_population(evolution_id)
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Create new population
        new_population = []
        
        # Keep elites
        for i in range(min(evolution.elite_count, len(population))):
            elite = population[i]
            new_individual = Individual(
                target=elite.target,
                genes=elite.genes.copy(),
                generation=evolution.current_generation + 1,
                parent_ids=[elite.id],
            )
            new_population.append(new_individual)
        
        # Fill rest with crossover and mutation
        while len(new_population) < evolution.population_size:
            # Selection (tournament)
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)
            
            # Crossover
            if random.random() < evolution.crossover_rate:
                child_genes = self._crossover(parent1.genes, parent2.genes)
            else:
                child_genes = parent1.genes.copy()
            
            # Mutation
            if random.random() < evolution.mutation_rate:
                child_genes = self._mutate(child_genes, evolution.target)
            
            child = Individual(
                target=evolution.target,
                genes=child_genes,
                generation=evolution.current_generation + 1,
                parent_ids=[parent1.id, parent2.id],
            )
            new_population.append(child)
        
        # Update state
        evolution.current_generation += 1
        self._populations[evolution_id] = new_population
        self._save()
        
        logger.info(
            "Evolution advanced",
            id=evolution_id,
            generation=evolution.current_generation,
        )
        
        return new_population
    
    def _tournament_select(
        self,
        population: list[Individual],
        tournament_size: int = 3,
    ) -> Individual:
        """Tournament selection."""
        tournament = random.sample(
            population,
            min(tournament_size, len(population)),
        )
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, genes1: dict, genes2: dict) -> dict:
        """Uniform crossover of genes."""
        child = {}
        all_keys = set(genes1.keys()) | set(genes2.keys())
        
        for key in all_keys:
            if key in genes1 and key in genes2:
                child[key] = random.choice([genes1[key], genes2[key]])
            elif key in genes1:
                child[key] = genes1[key]
            else:
                child[key] = genes2[key]
        
        return child
    
    def _mutate(self, genes: dict, target: EvolutionTarget) -> dict:
        """Mutate genes."""
        mutated = genes.copy()
        
        # Pick a random gene to mutate
        if not genes:
            return mutated
        
        key = random.choice(list(genes.keys()))
        
        # Get possible values for this gene
        if target == EvolutionTarget.PROMPT:
            if key in self._prompt_genes:
                mutated[key] = random.choice(self._prompt_genes[key])
        elif target == EvolutionTarget.STRATEGY:
            if key in self._strategy_genes:
                mutated[key] = random.choice(self._strategy_genes[key])
        elif target == EvolutionTarget.HYPERPARAMETER:
            if key == "temperature":
                mutated[key] = max(0.1, min(1.0, genes[key] + random.uniform(-0.2, 0.2)))
            elif key == "max_tokens":
                mutated[key] = random.choice([256, 512, 1024])
            elif key == "top_p":
                mutated[key] = max(0.5, min(1.0, genes[key] + random.uniform(-0.1, 0.1)))
        
        return mutated
    
    def run_evolution(
        self,
        evolution_id: str,
        fitness_fn: callable = None,
    ) -> Individual | None:
        """Run complete evolution and return best individual."""
        if evolution_id not in self._evolutions:
            return None
        
        evolution = self._evolutions[evolution_id]
        
        while evolution.current_generation < evolution.generations:
            self.evaluate_population(evolution_id, fitness_fn)
            self.evolve_generation(evolution_id)
        
        # Final evaluation
        self.evaluate_population(evolution_id, fitness_fn)
        
        evolution.completed_at = datetime.now().isoformat()
        self._save()
        
        # Return best
        population = self._populations.get(evolution_id, [])
        if population:
            return max(population, key=lambda x: x.fitness)
        return None
    
    def get_best_individual(
        self,
        target: EvolutionTarget,
    ) -> Individual | None:
        """Get best individual for a target type."""
        return self._best_individuals.get(target)
    
    def apply_best_genes(self, target: EvolutionTarget) -> dict:
        """Get the best genes to apply."""
        best = self.get_best_individual(target)
        return best.genes if best else {}
    
    def get_evolution_stats(self, evolution_id: str) -> dict:
        """Get statistics for an evolution run."""
        if evolution_id not in self._evolutions:
            return {}
        
        evolution = self._evolutions[evolution_id]
        population = self._populations.get(evolution_id, [])
        
        fitness_values = [ind.fitness for ind in population if ind.evaluated]
        
        return {
            "id": evolution_id,
            "target": evolution.target.name,
            "current_generation": evolution.current_generation,
            "total_generations": evolution.generations,
            "population_size": len(population),
            "best_fitness": evolution.best_fitness,
            "avg_fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0,
            "fitness_history": evolution.fitness_history,
            "completed": evolution.completed_at is not None,
        }
    
    def __len__(self) -> int:
        return len(self._evolutions)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "evolutions": {k: v.to_dict() for k, v in self._evolutions.items()},
                "populations": {
                    k: [ind.to_dict() for ind in v]
                    for k, v in self._populations.items()
                },
                "best": {
                    k.name: v.to_dict()
                    for k, v in self._best_individuals.items()
                },
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        
        for k, v in data.get("evolutions", {}).items():
            self._evolutions[k] = Evolution(**{
                key: val for key, val in v.items()
                if key != "target"
            })
            self._evolutions[k].target = EvolutionTarget[v["target"]]
        
        for k, v in data.get("populations", {}).items():
            self._populations[k] = [Individual.from_dict(ind) for ind in v]
        
        for k, v in data.get("best", {}).items():
            self._best_individuals[EvolutionTarget[k]] = Individual.from_dict(v)
