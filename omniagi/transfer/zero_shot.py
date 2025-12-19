"""
Zero-Shot Transfer Learning.

Implements task embedding, analogical transfer, and
concept abstraction for generalizing to new tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class Task:
    """A task description for transfer learning."""
    id: str
    name: str
    description: str
    input_type: str
    output_type: str
    examples: List[Tuple[Any, Any]] = field(default_factory=list)
    embedding: Optional[Any] = None


@dataclass
class Concept:
    """An abstract concept that can transfer across domains."""
    id: str
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str]] = field(default_factory=list)
    instances: List[str] = field(default_factory=list)


class TaskEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Encodes tasks into a shared embedding space.
    
    Similar tasks will have similar embeddings.
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128):
        if not TORCH_AVAILABLE:
            return
        
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Text encoder for task descriptions
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # Example encoder
        self.example_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
        )
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text description to embedding."""
        if not TORCH_AVAILABLE:
            return None
        
        # Simple bag-of-words encoding
        words = text.lower().split()[:50]  # Limit length
        word_ids = [hash(w) % 10000 for w in words]
        
        if not word_ids:
            return torch.zeros(self.embed_dim)
        
        word_tensor = torch.tensor(word_ids)
        embeddings = self.word_embed(word_tensor)
        pooled = embeddings.mean(dim=0)
        
        return self.encoder(pooled)
    
    def encode_task(self, task: Task) -> torch.Tensor:
        """Encode a complete task."""
        if not TORCH_AVAILABLE:
            return None
        
        # Encode description
        desc_embed = self.encode_text(task.description)
        
        # Encode examples if available
        if task.examples:
            example_embeds = []
            for inp, out in task.examples[:5]:
                inp_str = str(inp)[:100]
                out_str = str(out)[:100]
                inp_embed = self.encode_text(inp_str)
                out_embed = self.encode_text(out_str)
                combined = torch.cat([inp_embed, out_embed])
                example_embeds.append(self.example_encoder(combined))
            
            example_embed = torch.stack(example_embeds).mean(dim=0)
            task_embed = (desc_embed + example_embed) / 2
        else:
            task_embed = desc_embed
        
        task.embedding = task_embed
        return task_embed


class ConceptAbstractionLayer:
    """
    Abstracts concrete instances into general concepts.
    
    Enables transfer by operating on concepts rather than specifics.
    """
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.instance_to_concept: Dict[str, str] = {}
        self.concept_hierarchy: Dict[str, List[str]] = defaultdict(list)  # parent -> children
    
    def create_concept(
        self,
        name: str,
        attributes: Dict[str, Any] = None,
        parent: Optional[str] = None,
    ) -> Concept:
        """Create a new concept."""
        concept = Concept(
            id=f"concept_{len(self.concepts)}",
            name=name,
            attributes=attributes or {},
        )
        
        self.concepts[concept.id] = concept
        
        if parent and parent in self.concepts:
            self.concept_hierarchy[parent].append(concept.id)
        
        return concept
    
    def abstract(self, instance: Any, instance_name: str) -> Optional[Concept]:
        """
        Abstract an instance into a concept.
        
        Returns the concept that best matches the instance.
        """
        # Check if already mapped
        if instance_name in self.instance_to_concept:
            return self.concepts.get(self.instance_to_concept[instance_name])
        
        # Find or create matching concept
        instance_attrs = self._extract_attributes(instance)
        
        best_match = None
        best_score = 0.0
        
        for concept in self.concepts.values():
            score = self._concept_match_score(instance_attrs, concept.attributes)
            if score > best_score:
                best_score = score
                best_match = concept
        
        if best_match and best_score > 0.5:
            best_match.instances.append(instance_name)
            self.instance_to_concept[instance_name] = best_match.id
            return best_match
        
        # Create new concept
        new_concept = self.create_concept(
            name=f"concept_for_{instance_name}",
            attributes=instance_attrs,
        )
        new_concept.instances.append(instance_name)
        self.instance_to_concept[instance_name] = new_concept.id
        
        return new_concept
    
    def _extract_attributes(self, instance: Any) -> Dict[str, Any]:
        """Extract attributes from an instance."""
        attrs = {}
        
        if isinstance(instance, dict):
            attrs = instance.copy()
        elif hasattr(instance, '__dict__'):
            attrs = instance.__dict__.copy()
        else:
            attrs = {"type": type(instance).__name__, "value": str(instance)[:100]}
        
        return attrs
    
    def _concept_match_score(self, attrs1: Dict, attrs2: Dict) -> float:
        """Compute similarity between attribute sets."""
        if not attrs1 or not attrs2:
            return 0.0
        
        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for k in common_keys if attrs1[k] == attrs2[k])
        total = len(set(attrs1.keys()) | set(attrs2.keys()))
        
        return matches / total


class AnalogicalTransfer:
    """
    Transfers knowledge between domains using analogical reasoning.
    
    Maps structure from source domain to target domain.
    """
    
    def __init__(self):
        self.mappings: Dict[str, Dict[str, str]] = {}
        self.transfer_history: List[Dict] = []
    
    def create_mapping(
        self,
        source_domain: str,
        target_domain: str,
        correspondences: Dict[str, str],
    ) -> None:
        """Create an analogical mapping between domains."""
        key = f"{source_domain}:{target_domain}"
        self.mappings[key] = correspondences
    
    def transfer(
        self,
        source_domain: str,
        target_domain: str,
        source_knowledge: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transfer knowledge from source to target domain.
        
        Uses established mappings to translate concepts.
        """
        key = f"{source_domain}:{target_domain}"
        
        if key not in self.mappings:
            # Attempt to infer mapping
            mapping = self._infer_mapping(source_domain, target_domain, source_knowledge)
            self.mappings[key] = mapping
        else:
            mapping = self.mappings[key]
        
        # Apply mapping
        transferred = {}
        for src_key, src_value in source_knowledge.items():
            # Map key
            tgt_key = mapping.get(src_key, src_key)
            
            # Map value if it's a mapped concept
            if isinstance(src_value, str) and src_value in mapping:
                tgt_value = mapping[src_value]
            else:
                tgt_value = src_value
            
            transferred[tgt_key] = tgt_value
        
        self.transfer_history.append({
            "source": source_domain,
            "target": target_domain,
            "items_transferred": len(transferred),
        })
        
        return transferred
    
    def _infer_mapping(
        self,
        source: str,
        target: str,
        knowledge: Dict[str, Any],
    ) -> Dict[str, str]:
        """Infer mapping between domains based on structure."""
        # Simple structural mapping
        mapping = {}
        
        # Map domain names
        mapping[source] = target
        
        # Attempt to map based on naming conventions
        for key in knowledge.keys():
            if source.lower() in key.lower():
                mapped_key = key.replace(source, target).replace(source.lower(), target.lower())
                mapping[key] = mapped_key
        
        return mapping
    
    def find_analogy(
        self,
        source_a: Any,
        source_b: Any,
        target_a: Any,
    ) -> Any:
        """
        Complete analogy: A is to B as C is to ?
        
        Returns the analogical completion.
        """
        # Extract relationship between source_a and source_b
        relationship = self._extract_relationship(source_a, source_b)
        
        # Apply same relationship to target_a
        return self._apply_relationship(target_a, relationship)
    
    def _extract_relationship(self, a: Any, b: Any) -> Dict[str, Any]:
        """Extract the relationship between two items."""
        rel = {"type": "unknown", "transform": None}
        
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if b == a * 2:
                rel = {"type": "multiply", "factor": 2}
            elif b == a + 1:
                rel = {"type": "add", "value": 1}
            elif b == a - 1:
                rel = {"type": "subtract", "value": 1}
        elif isinstance(a, str) and isinstance(b, str):
            if b == a.upper():
                rel = {"type": "uppercase"}
            elif b == a.lower():
                rel = {"type": "lowercase"}
            elif b == a[::-1]:
                rel = {"type": "reverse"}
        
        return rel
    
    def _apply_relationship(self, target: Any, relationship: Dict) -> Any:
        """Apply a relationship to a target."""
        rel_type = relationship.get("type", "unknown")
        
        if rel_type == "multiply":
            return target * relationship["factor"]
        elif rel_type == "add":
            return target + relationship["value"]
        elif rel_type == "subtract":
            return target - relationship["value"]
        elif rel_type == "uppercase":
            return str(target).upper()
        elif rel_type == "lowercase":
            return str(target).lower()
        elif rel_type == "reverse":
            return str(target)[::-1]
        
        return target


class ZeroShotTransferSystem:
    """
    Complete zero-shot transfer learning system.
    
    Enables applying knowledge to completely new tasks.
    """
    
    def __init__(self):
        if TORCH_AVAILABLE:
            self.task_encoder = TaskEncoder()
        else:
            self.task_encoder = None
        
        self.concept_layer = ConceptAbstractionLayer()
        self.analogical = AnalogicalTransfer()
        
        self.known_tasks: Dict[str, Task] = {}
        self.task_solutions: Dict[str, Callable] = {}
    
    def register_task(
        self,
        task: Task,
        solution: Optional[Callable] = None,
    ) -> None:
        """Register a known task with optional solution."""
        # Encode task
        if self.task_encoder:
            self.task_encoder.encode_task(task)
        
        self.known_tasks[task.id] = task
        
        if solution:
            self.task_solutions[task.id] = solution
    
    def find_similar_task(self, new_task: Task) -> Optional[Task]:
        """Find the most similar known task."""
        if not self.task_encoder or not self.known_tasks:
            return None
        
        # Encode new task
        new_embed = self.task_encoder.encode_task(new_task)
        
        if new_embed is None:
            return None
        
        best_task = None
        best_sim = -float('inf')
        
        for task in self.known_tasks.values():
            if task.embedding is not None:
                sim = F.cosine_similarity(
                    new_embed.unsqueeze(0),
                    task.embedding.unsqueeze(0),
                ).item()
                
                if sim > best_sim:
                    best_sim = sim
                    best_task = task
        
        return best_task
    
    def zero_shot_solve(self, new_task: Task, input_data: Any) -> Any:
        """
        Solve a new task using zero-shot transfer.
        
        Finds similar task and adapts its solution.
        """
        # Find similar known task
        similar = self.find_similar_task(new_task)
        
        if similar and similar.id in self.task_solutions:
            # Use analogical transfer to adapt solution
            solution = self.task_solutions[similar.id]
            
            # Abstract input
            input_concept = self.concept_layer.abstract(input_data, str(input_data))
            
            # Apply solution with adaptation
            try:
                result = solution(input_data)
                return result
            except Exception:
                pass
        
        # Fallback: use analogical reasoning from examples
        if new_task.examples:
            # Try to infer pattern from examples
            return self._infer_from_examples(new_task, input_data)
        
        return None
    
    def _infer_from_examples(self, task: Task, new_input: Any) -> Any:
        """Infer output from task examples using analogy."""
        if len(task.examples) < 2:
            return None
        
        # Find pattern in examples
        (in1, out1), (in2, out2) = task.examples[:2]
        
        # Create analogy: in1:out1 :: new_input:?
        return self.analogical.find_analogy(in1, out1, new_input)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "known_tasks": len(self.known_tasks),
            "solutions_registered": len(self.task_solutions),
            "concepts_abstracted": len(self.concept_layer.concepts),
            "transfers_performed": len(self.analogical.transfer_history),
        }
