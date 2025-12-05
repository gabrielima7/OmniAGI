"""
Transfer Learning - Apply knowledge across domains.

Enables the AGI to:
1. Map knowledge from one domain to another
2. Find structural analogies
3. Transfer learned concepts to new problems

Critical for true AGI - ability to generalize.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

logger = structlog.get_logger()


class TransferType(Enum):
    """Types of knowledge transfer."""
    
    DIRECT = auto()         # Direct mapping
    ANALOGICAL = auto()     # Structural analogy
    ABSTRACTION = auto()    # Abstract then apply
    COMPOSITION = auto()    # Combine multiple


@dataclass
class DomainMapping:
    """Mapping between two domains."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    source_domain: str = ""
    target_domain: str = ""
    
    # Mappings
    concept_mappings: dict[str, str] = field(default_factory=dict)
    relation_mappings: dict[str, str] = field(default_factory=dict)
    
    # Quality
    confidence: float = 0.5
    times_used: int = 0
    success_rate: float = 0.0
    
    def map_concept(self, source_concept: str) -> str | None:
        """Map a concept from source to target domain."""
        return self.concept_mappings.get(source_concept)
    
    def to_dict(self) -> dict:
        return {
            "source": self.source_domain,
            "target": self.target_domain,
            "mappings": len(self.concept_mappings),
            "confidence": self.confidence,
        }


@dataclass
class TransferResult:
    """Result of a knowledge transfer."""
    
    success: bool = False
    transfer_type: TransferType = TransferType.DIRECT
    
    source_knowledge: str = ""
    transferred_knowledge: str = ""
    
    confidence: float = 0.0
    explanation: str = ""
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "type": self.transfer_type.name,
            "confidence": self.confidence,
            "explanation": self.explanation[:100],
        }


class TransferLearner:
    """
    Transfer Learning System for True AGI.
    
    Enables applying knowledge learned in one domain
    to solve problems in another domain.
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Domain mappings
        self._mappings: dict[str, DomainMapping] = {}
        
        # Knowledge base (simplified)
        self._domain_knowledge: dict[str, dict] = {}
        
        # Transfer history
        self._history: list[TransferResult] = []
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Transfer Learner initialized")
    
    def register_domain(
        self,
        name: str,
        concepts: list[str],
        relations: list[tuple[str, str, str]] = None,
    ) -> None:
        """
        Register a domain with its concepts and relations.
        
        Args:
            name: Domain name
            concepts: List of concept names
            relations: List of (concept1, relation, concept2) tuples
        """
        self._domain_knowledge[name] = {
            "concepts": set(concepts),
            "relations": relations or [],
        }
        
        logger.info("Domain registered", name=name, concepts=len(concepts))
    
    def create_mapping(
        self,
        source: str,
        target: str,
        concept_mappings: dict[str, str],
        relation_mappings: dict[str, str] = None,
    ) -> DomainMapping:
        """
        Create a mapping between two domains.
        """
        mapping = DomainMapping(
            source_domain=source,
            target_domain=target,
            concept_mappings=concept_mappings,
            relation_mappings=relation_mappings or {},
        )
        
        key = f"{source}_to_{target}"
        self._mappings[key] = mapping
        
        self._save()
        return mapping
    
    def transfer(
        self,
        knowledge: str,
        source_domain: str,
        target_domain: str,
        transfer_type: TransferType = TransferType.DIRECT,
    ) -> TransferResult:
        """
        Transfer knowledge from source to target domain.
        """
        result = TransferResult(
            transfer_type=transfer_type,
            source_knowledge=knowledge,
        )
        
        # Get mapping
        key = f"{source_domain}_to_{target_domain}"
        mapping = self._mappings.get(key)
        
        if not mapping:
            # Try to create automatic mapping
            mapping = self._auto_create_mapping(source_domain, target_domain)
        
        if not mapping:
            result.explanation = "No mapping found between domains"
            return result
        
        # Apply transfer
        if transfer_type == TransferType.DIRECT:
            result = self._direct_transfer(knowledge, mapping)
        elif transfer_type == TransferType.ANALOGICAL:
            result = self._analogical_transfer(knowledge, mapping)
        elif transfer_type == TransferType.ABSTRACTION:
            result = self._abstraction_transfer(knowledge, mapping)
        
        # Update mapping stats
        mapping.times_used += 1
        if result.success:
            mapping.success_rate = (
                mapping.success_rate * (mapping.times_used - 1) + 1
            ) / mapping.times_used
        
        self._history.append(result)
        self._save()
        
        return result
    
    def _direct_transfer(
        self,
        knowledge: str,
        mapping: DomainMapping,
    ) -> TransferResult:
        """Direct concept mapping transfer."""
        result = TransferResult(
            transfer_type=TransferType.DIRECT,
            source_knowledge=knowledge,
        )
        
        transferred = knowledge
        mappings_applied = 0
        
        # Replace concepts
        for source_concept, target_concept in mapping.concept_mappings.items():
            if source_concept.lower() in knowledge.lower():
                transferred = transferred.replace(source_concept, target_concept)
                mappings_applied += 1
        
        if mappings_applied > 0:
            result.success = True
            result.transferred_knowledge = transferred
            result.confidence = min(0.9, 0.5 + mappings_applied * 0.1)
            result.explanation = f"Applied {mappings_applied} concept mappings"
        else:
            result.explanation = "No mappings could be applied"
        
        return result
    
    def _analogical_transfer(
        self,
        knowledge: str,
        mapping: DomainMapping,
    ) -> TransferResult:
        """Transfer via structural analogy."""
        result = TransferResult(
            transfer_type=TransferType.ANALOGICAL,
            source_knowledge=knowledge,
        )
        
        # Find structural patterns
        # Pattern: X is-a Y → map to target domain
        if " is " in knowledge or " are " in knowledge:
            parts = knowledge.replace(" is ", "|").replace(" are ", "|").split("|")
            if len(parts) >= 2:
                subject = parts[0].strip()
                predicate = parts[1].strip()
                
                # Map both parts
                mapped_subject = mapping.concept_mappings.get(subject, subject)
                mapped_predicate = mapping.concept_mappings.get(predicate, predicate)
                
                result.transferred_knowledge = f"{mapped_subject} is {mapped_predicate}"
                result.success = mapped_subject != subject or mapped_predicate != predicate
                result.confidence = 0.7
                result.explanation = "Applied structural analogy"
        
        return result
    
    def _abstraction_transfer(
        self,
        knowledge: str,
        mapping: DomainMapping,
    ) -> TransferResult:
        """Transfer via abstraction."""
        result = TransferResult(
            transfer_type=TransferType.ABSTRACTION,
            source_knowledge=knowledge,
        )
        
        # Abstract the knowledge
        abstract = self._abstract_knowledge(knowledge)
        
        # Apply to target domain
        concrete = self._concretize(abstract, mapping.target_domain)
        
        if concrete != abstract:
            result.success = True
            result.transferred_knowledge = concrete
            result.confidence = 0.6
            result.explanation = f"Abstracted to '{abstract}' then applied to target"
        
        return result
    
    def _auto_create_mapping(
        self,
        source: str,
        target: str,
    ) -> DomainMapping | None:
        """Try to automatically create a domain mapping."""
        if source not in self._domain_knowledge or target not in self._domain_knowledge:
            return None
        
        source_concepts = self._domain_knowledge[source]["concepts"]
        target_concepts = self._domain_knowledge[target]["concepts"]
        
        # Simple string similarity mapping
        concept_mappings = {}
        for sc in source_concepts:
            best_match = None
            best_score = 0
            for tc in target_concepts:
                score = self._similarity(sc, tc)
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = tc
            if best_match:
                concept_mappings[sc] = best_match
        
        if concept_mappings:
            return self.create_mapping(source, target, concept_mappings)
        
        return None
    
    def _abstract_knowledge(self, knowledge: str) -> str:
        """Abstract knowledge to general form."""
        # Replace specific terms with placeholders
        abstract = knowledge
        replacements = [
            ("is a", "ISA"),
            ("has", "HAS"),
            ("can", "CAN"),
        ]
        for old, new in replacements:
            abstract = abstract.replace(old, new)
        return abstract
    
    def _concretize(self, abstract: str, domain: str) -> str:
        """Apply abstract knowledge to concrete domain."""
        concrete = abstract
        # Reverse abstraction
        concrete = concrete.replace("ISA", "is a")
        concrete = concrete.replace("HAS", "has")
        concrete = concrete.replace("CAN", "can")
        return concrete
    
    def _similarity(self, a: str, b: str) -> float:
        """Compute similarity between strings."""
        a_set = set(a.lower())
        b_set = set(b.lower())
        if not a_set or not b_set:
            return 0.0
        intersection = len(a_set & b_set)
        union = len(a_set | b_set)
        return intersection / union
    
    def get_stats(self) -> dict:
        """Get transfer learning statistics."""
        successful = sum(1 for r in self._history if r.success)
        total = len(self._history)
        
        return {
            "domains": len(self._domain_knowledge),
            "mappings": len(self._mappings),
            "transfers": total,
            "success_rate": successful / total if total > 0 else 0.0,
        }
    
    def __len__(self) -> int:
        return len(self._mappings)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "mappings": len(self._mappings),
                "domains": list(self._domain_knowledge.keys()),
            }, f)
    
    def _load(self) -> None:
        pass
