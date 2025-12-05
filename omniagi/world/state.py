"""
World State - Structured representation of the environment.

Represents the current state of the world with entities,
relations, and properties for mental simulation.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from copy import deepcopy

logger = structlog.get_logger()


@dataclass
class Entity:
    """An entity in the world."""
    
    id: str
    name: str
    entity_type: str  # agent, object, location, concept
    properties: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        return cls(**data)


@dataclass
class Relation:
    """A relation between entities."""
    
    source_id: str
    target_id: str
    relation_type: str  # at, has, knows, affects, etc.
    properties: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        return cls(**data)


@dataclass
class StateSnapshot:
    """A snapshot of the world state at a point in time."""
    
    timestamp: datetime
    entities: dict[str, Entity]
    relations: list[Relation]
    metadata: dict[str, Any] = field(default_factory=dict)


class WorldState:
    """
    Representation of the world state.
    
    Maintains a structured model of:
    - Entities (agents, objects, locations)
    - Relations between entities
    - Properties and states
    
    Supports:
    - State updates
    - Querying
    - Snapshots for simulation
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize world state.
        
        Args:
            storage_path: Path for persistent storage.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._history: list[StateSnapshot] = []
        self._current_time: datetime = datetime.now()
        
        if self.storage_path:
            self._load()
    
    def _load(self) -> None:
        """Load from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            
            for entity_data in data.get("entities", []):
                entity = Entity.from_dict(entity_data)
                self._entities[entity.id] = entity
            
            for rel_data in data.get("relations", []):
                self._relations.append(Relation.from_dict(rel_data))
            
            logger.info("World state loaded", entities=len(self._entities))
        except Exception as e:
            logger.error("Failed to load world state", error=str(e))
    
    def _save(self) -> None:
        """Save to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations],
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_entity(
        self,
        name: str,
        entity_type: str,
        entity_id: str | None = None,
        properties: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
    ) -> Entity:
        """Add an entity to the world."""
        import uuid
        
        entity_id = entity_id or str(uuid.uuid4())
        
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            state=state or {},
        )
        
        self._entities[entity_id] = entity
        self._save()
        return entity
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        **properties,
    ) -> Relation | None:
        """Add a relation between entities."""
        if source_id not in self._entities or target_id not in self._entities:
            return None
        
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties,
        )
        
        self._relations.append(relation)
        self._save()
        return relation
    
    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self._entities.get(entity_id)
    
    def find_entities(
        self,
        entity_type: str | None = None,
        **properties,
    ) -> list[Entity]:
        """Find entities matching criteria."""
        results = []
        
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            
            matches = True
            for key, value in properties.items():
                if entity.properties.get(key) != value and entity.state.get(key) != value:
                    matches = False
                    break
            
            if matches:
                results.append(entity)
        
        return results
    
    def get_relations(
        self,
        entity_id: str,
        relation_type: str | None = None,
        direction: str = "both",  # source, target, both
    ) -> list[tuple[Entity, Relation]]:
        """Get relations for an entity."""
        results = []
        
        for relation in self._relations:
            if relation_type and relation.relation_type != relation_type:
                continue
            
            if direction in ("source", "both") and relation.source_id == entity_id:
                target = self._entities.get(relation.target_id)
                if target:
                    results.append((target, relation))
            
            if direction in ("target", "both") and relation.target_id == entity_id:
                source = self._entities.get(relation.source_id)
                if source:
                    results.append((source, relation))
        
        return results
    
    def update_entity(
        self,
        entity_id: str,
        properties: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
    ) -> Entity | None:
        """Update an entity's properties or state."""
        entity = self._entities.get(entity_id)
        if not entity:
            return None
        
        if properties:
            entity.properties.update(properties)
        if state:
            entity.state.update(state)
        
        self._save()
        return entity
    
    def snapshot(self, metadata: dict[str, Any] | None = None) -> StateSnapshot:
        """Create a snapshot of current state."""
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            entities=deepcopy(self._entities),
            relations=deepcopy(self._relations),
            metadata=metadata or {},
        )
        
        self._history.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(self._history) > 100:
            self._history = self._history[-100:]
        
        return snapshot
    
    def restore(self, snapshot: StateSnapshot) -> None:
        """Restore state from a snapshot."""
        self._entities = deepcopy(snapshot.entities)
        self._relations = deepcopy(snapshot.relations)
        self._save()
    
    def clone(self) -> "WorldState":
        """Create a clone of this world state."""
        clone = WorldState()
        clone._entities = deepcopy(self._entities)
        clone._relations = deepcopy(self._relations)
        return clone
    
    def to_context(self, focus_entity_id: str | None = None) -> str:
        """Generate context string for prompts."""
        lines = ["## Estado do Mundo\n"]
        
        if focus_entity_id:
            entity = self._entities.get(focus_entity_id)
            if entity:
                lines.append(f"### Foco: {entity.name} ({entity.entity_type})")
                for k, v in entity.state.items():
                    lines.append(f"- {k}: {v}")
                
                relations = self.get_relations(focus_entity_id)
                if relations:
                    lines.append("\n### Relações:")
                    for other, rel in relations[:5]:
                        lines.append(f"- {rel.relation_type} → {other.name}")
        else:
            # General summary
            by_type: dict[str, list[Entity]] = {}
            for entity in self._entities.values():
                if entity.entity_type not in by_type:
                    by_type[entity.entity_type] = []
                by_type[entity.entity_type].append(entity)
            
            for etype, entities in by_type.items():
                lines.append(f"\n### {etype.title()}s ({len(entities)})")
                for e in entities[:3]:
                    lines.append(f"- {e.name}")
        
        return "\n".join(lines)
    
    def apply_action(
        self,
        action: str,
        actor_id: str,
        target_id: str | None = None,
        effects: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Apply an action to the world state.
        
        Returns a dictionary describing what changed.
        """
        changes = {"action": action, "actor": actor_id, "changes": []}
        
        actor = self._entities.get(actor_id)
        if not actor:
            return changes
        
        # Record action in actor's state
        actor.state["last_action"] = action
        actor.state["last_action_time"] = datetime.now().isoformat()
        changes["changes"].append(f"{actor.name} performed {action}")
        
        # Apply effects
        if effects and target_id:
            target = self._entities.get(target_id)
            if target:
                for key, value in effects.items():
                    target.state[key] = value
                    changes["changes"].append(f"{target.name}.{key} = {value}")
        
        self._save()
        return changes
    
    def __len__(self) -> int:
        return len(self._entities)
