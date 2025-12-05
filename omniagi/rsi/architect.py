"""
Self-Architect - Self-modifying architecture for RSI.

Enables the AGI to propose and implement changes to
its own architecture for capability improvement.
"""

from __future__ import annotations

import json
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


class ChangeType(Enum):
    """Types of architectural changes."""
    
    NEW_MODULE = auto()         # Add new capability module
    MODIFY_MODULE = auto()      # Modify existing module
    NEW_STRATEGY = auto()       # Add new strategy
    OPTIMIZE = auto()           # Optimize existing code
    HYPERPARAMETER = auto()     # Change hyperparameters
    INTEGRATION = auto()        # Integrate components
    DEPRECATE = auto()          # Remove/replace component


class ChangeStatus(Enum):
    """Status of a proposed change."""
    
    PROPOSED = auto()
    ANALYZING = auto()
    APPROVED = auto()
    IMPLEMENTING = auto()
    TESTING = auto()
    DEPLOYED = auto()
    ROLLED_BACK = auto()
    REJECTED = auto()


@dataclass
class ArchitecturalChange:
    """A proposed architectural change."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    change_type: ChangeType = ChangeType.OPTIMIZE
    status: ChangeStatus = ChangeStatus.PROPOSED
    
    title: str = ""
    description: str = ""
    rationale: str = ""
    
    # Target
    target_module: str = ""
    target_path: str = ""
    
    # Impact assessment
    expected_benefit: str = ""
    risk_level: float = 0.5  # 0-1
    complexity: float = 0.5  # 0-1
    
    # Implementation
    implementation_steps: list[str] = field(default_factory=list)
    code_changes: dict[str, str] = field(default_factory=dict)
    
    # Validation
    test_plan: list[str] = field(default_factory=list)
    rollback_plan: str = ""
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    implemented_at: str | None = None
    
    # Results
    success: bool | None = None
    actual_benefit: str = ""
    lessons_learned: str = ""
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "change_type": self.change_type.name,
            "status": self.status.name,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "target_module": self.target_module,
            "target_path": self.target_path,
            "expected_benefit": self.expected_benefit,
            "risk_level": self.risk_level,
            "complexity": self.complexity,
            "implementation_steps": self.implementation_steps,
            "code_changes": self.code_changes,
            "test_plan": self.test_plan,
            "rollback_plan": self.rollback_plan,
            "created_at": self.created_at,
            "implemented_at": self.implemented_at,
            "success": self.success,
            "actual_benefit": self.actual_benefit,
            "lessons_learned": self.lessons_learned,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ArchitecturalChange":
        return cls(
            id=data.get("id", str(uuid4())[:8]),
            change_type=ChangeType[data.get("change_type", "OPTIMIZE")],
            status=ChangeStatus[data.get("status", "PROPOSED")],
            title=data.get("title", ""),
            description=data.get("description", ""),
            rationale=data.get("rationale", ""),
            target_module=data.get("target_module", ""),
            target_path=data.get("target_path", ""),
            expected_benefit=data.get("expected_benefit", ""),
            risk_level=data.get("risk_level", 0.5),
            complexity=data.get("complexity", 0.5),
            implementation_steps=data.get("implementation_steps", []),
            code_changes=data.get("code_changes", {}),
            test_plan=data.get("test_plan", []),
            rollback_plan=data.get("rollback_plan", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            implemented_at=data.get("implemented_at"),
            success=data.get("success"),
            actual_benefit=data.get("actual_benefit", ""),
            lessons_learned=data.get("lessons_learned", ""),
        )


class SelfArchitect:
    """
    Self-modifying architecture system.
    
    Proposes, analyzes, and implements changes to the
    AGI's own architecture for capability improvement.
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        storage_path: Path | str | None = None,
        require_approval: bool = True,
        max_risk: float = 0.7,
    ):
        self.engine = engine
        self.storage_path = Path(storage_path) if storage_path else None
        self.require_approval = require_approval
        self.max_risk = max_risk
        
        self._changes: dict[str, ArchitecturalChange] = {}
        self._improvement_history: list[dict] = []
        
        # Areas for potential improvement
        self._improvement_areas = [
            "reasoning_chain",
            "memory_retrieval",
            "planning_depth",
            "code_generation",
            "meta_learning",
            "strategy_adaptation",
            "world_model_accuracy",
        ]
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Self-Architect initialized", approval_required=require_approval)
    
    def propose_improvement(
        self,
        area: str = None,
        context: dict = None,
    ) -> ArchitecturalChange | None:
        """
        Propose an architectural improvement.
        
        Args:
            area: Specific area to improve (or auto-detect).
            context: Current performance context.
            
        Returns:
            Proposed architectural change.
        """
        if area is None:
            area = self._identify_improvement_area(context)
        
        change = self._generate_proposal(area, context)
        
        if change:
            self._changes[change.id] = change
            self._save()
            
            logger.info(
                "Improvement proposed",
                id=change.id,
                area=area,
                type=change.change_type.name,
            )
        
        return change
    
    def _identify_improvement_area(self, context: dict = None) -> str:
        """Identify the area most needing improvement."""
        # Simple heuristic - would use metrics in production
        import random
        return random.choice(self._improvement_areas)
    
    def _generate_proposal(
        self,
        area: str,
        context: dict = None,
    ) -> ArchitecturalChange | None:
        """Generate a specific improvement proposal."""
        proposals = {
            "reasoning_chain": {
                "title": "Enhance Reasoning Chain Depth",
                "type": ChangeType.MODIFY_MODULE,
                "target": "omniagi/agent/react.py",
                "description": "Increase reasoning chain depth for complex problems",
                "benefit": "Better handling of multi-step problems",
                "risk": 0.3,
            },
            "memory_retrieval": {
                "title": "Optimize Memory Retrieval",
                "type": ChangeType.OPTIMIZE,
                "target": "omniagi/memory/vector.py",
                "description": "Improve semantic search relevance",
                "benefit": "More relevant context retrieval",
                "risk": 0.4,
            },
            "planning_depth": {
                "title": "Deeper Hierarchical Planning",
                "type": ChangeType.MODIFY_MODULE,
                "target": "omniagi/world/planner.py",
                "description": "Add additional planning layers",
                "benefit": "Better long-term goal decomposition",
                "risk": 0.5,
            },
            "code_generation": {
                "title": "Enhanced Code Generation",
                "type": ChangeType.NEW_STRATEGY,
                "target": "omniagi/meta/strategy.py",
                "description": "Add specialized code generation strategy",
                "benefit": "Higher quality generated code",
                "risk": 0.3,
            },
            "meta_learning": {
                "title": "Faster Strategy Adaptation",
                "type": ChangeType.HYPERPARAMETER,
                "target": "omniagi/meta/learner.py",
                "description": "Tune learning rate for faster adaptation",
                "benefit": "Quicker learning from experience",
                "risk": 0.4,
            },
            "strategy_adaptation": {
                "title": "Cross-Domain Strategy Transfer",
                "type": ChangeType.NEW_MODULE,
                "target": "omniagi/meta/transfer.py",
                "description": "Add cross-domain strategy transfer module",
                "benefit": "Apply strategies across domains",
                "risk": 0.5,
            },
            "world_model_accuracy": {
                "title": "Improved World Model Predictions",
                "type": ChangeType.MODIFY_MODULE,
                "target": "omniagi/world/simulator.py",
                "description": "Enhance prediction accuracy",
                "benefit": "Better action outcome predictions",
                "risk": 0.4,
            },
        }
        
        if area not in proposals:
            return None
        
        p = proposals[area]
        
        change = ArchitecturalChange(
            change_type=p["type"],
            title=p["title"],
            description=p["description"],
            target_module=area,
            target_path=p["target"],
            expected_benefit=p["benefit"],
            risk_level=p["risk"],
            complexity=0.5,
            rationale=f"Identified {area} as improvement opportunity",
            implementation_steps=[
                f"Analyze current {area} implementation",
                "Design improved version",
                "Implement changes",
                "Test thoroughly",
                "Deploy if tests pass",
            ],
            test_plan=[
                "Unit tests pass",
                "Integration tests pass",
                "No performance regression",
            ],
            rollback_plan=f"Restore previous version of {p['target']}",
        )
        
        return change
    
    async def propose_with_llm(
        self,
        performance_metrics: dict = None,
        constraints: list[str] = None,
    ) -> ArchitecturalChange | None:
        """Use LLM to propose improvements based on metrics."""
        if not self.engine or not self.engine.is_loaded:
            return self.propose_improvement()
        
        metrics_str = ""
        if performance_metrics:
            metrics_str = f"Current metrics: {json.dumps(performance_metrics)}"
        
        constraint_str = ""
        if constraints:
            constraint_str = f"Constraints: {', '.join(constraints)}"
        
        prompt = f"""As a self-improving AI, analyze your architecture and propose an improvement.

{metrics_str}
{constraint_str}

Available modules:
- omniagi/agent/ (reasoning, agents)
- omniagi/memory/ (memory systems)
- omniagi/meta/ (meta-learning)
- omniagi/world/ (world model)
- omniagi/continual/ (continual learning)
- omniagi/causal/ (causal reasoning)

Propose ONE specific improvement in JSON:
{{
    "title": "improvement title",
    "change_type": "NEW_MODULE|MODIFY_MODULE|OPTIMIZE|HYPERPARAMETER",
    "target_path": "path/to/module.py",
    "description": "what to change",
    "expected_benefit": "what benefit",
    "risk_level": 0.0-1.0,
    "rationale": "why this improvement"
}}"""
        
        try:
            response = self.engine.generate(prompt, max_tokens=400)
            
            import re
            json_match = re.search(r'\{[^{}]*\}', response.text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                change = ArchitecturalChange(
                    change_type=ChangeType[data.get("change_type", "OPTIMIZE")],
                    title=data.get("title", "LLM-proposed improvement"),
                    description=data.get("description", ""),
                    target_path=data.get("target_path", ""),
                    expected_benefit=data.get("expected_benefit", ""),
                    risk_level=float(data.get("risk_level", 0.5)),
                    rationale=data.get("rationale", ""),
                )
                
                self._changes[change.id] = change
                self._save()
                
                logger.info("LLM-proposed improvement", id=change.id)
                return change
                
        except Exception as e:
            logger.error("LLM proposal failed", error=str(e))
        
        return None
    
    def analyze_risk(self, change_id: str) -> dict:
        """Analyze risk of a proposed change."""
        if change_id not in self._changes:
            return {"error": "Change not found"}
        
        change = self._changes[change_id]
        
        risk_factors = {
            "base_risk": change.risk_level,
            "complexity_risk": change.complexity * 0.3,
            "type_risk": {
                ChangeType.NEW_MODULE: 0.4,
                ChangeType.MODIFY_MODULE: 0.3,
                ChangeType.OPTIMIZE: 0.2,
                ChangeType.HYPERPARAMETER: 0.1,
                ChangeType.INTEGRATION: 0.4,
                ChangeType.NEW_STRATEGY: 0.2,
                ChangeType.DEPRECATE: 0.5,
            }.get(change.change_type, 0.3),
        }
        
        total_risk = sum(risk_factors.values()) / len(risk_factors)
        
        return {
            "change_id": change_id,
            "total_risk": total_risk,
            "factors": risk_factors,
            "recommendation": "approve" if total_risk < self.max_risk else "reject",
            "within_tolerance": total_risk < self.max_risk,
        }
    
    def approve(self, change_id: str, approver: str = "human") -> bool:
        """Approve a change for implementation."""
        if change_id not in self._changes:
            return False
        
        change = self._changes[change_id]
        
        # Check risk tolerance
        risk = self.analyze_risk(change_id)
        if not risk.get("within_tolerance", False):
            logger.warning("Change exceeds risk tolerance", id=change_id)
            if self.require_approval:
                return False
        
        change.status = ChangeStatus.APPROVED
        self._save()
        
        logger.info("Change approved", id=change_id, approver=approver)
        return True
    
    def implement(self, change_id: str) -> bool:
        """
        Implement an approved change.
        
        Note: Actual code modification would require careful
        implementation with sandboxing.
        """
        if change_id not in self._changes:
            return False
        
        change = self._changes[change_id]
        
        if self.require_approval and change.status != ChangeStatus.APPROVED:
            logger.warning("Change not approved", id=change_id)
            return False
        
        change.status = ChangeStatus.IMPLEMENTING
        
        # In production, this would:
        # 1. Create backup
        # 2. Apply code changes
        # 3. Run tests
        # 4. Deploy or rollback
        
        # For now, just mark as deployed
        change.status = ChangeStatus.DEPLOYED
        change.implemented_at = datetime.now().isoformat()
        change.success = True
        
        self._improvement_history.append({
            "change_id": change_id,
            "timestamp": change.implemented_at,
            "type": change.change_type.name,
        })
        
        self._save()
        
        logger.info("Change implemented", id=change_id)
        return True
    
    def rollback(self, change_id: str, reason: str = "") -> bool:
        """Rollback a deployed change."""
        if change_id not in self._changes:
            return False
        
        change = self._changes[change_id]
        change.status = ChangeStatus.ROLLED_BACK
        change.success = False
        change.lessons_learned = reason
        self._save()
        
        logger.warning("Change rolled back", id=change_id, reason=reason[:50])
        return True
    
    def get_pending(self) -> list[ArchitecturalChange]:
        """Get pending changes."""
        return [
            c for c in self._changes.values()
            if c.status in [ChangeStatus.PROPOSED, ChangeStatus.APPROVED]
        ]
    
    def get_history(self) -> list[dict]:
        """Get improvement history."""
        return self._improvement_history.copy()
    
    def get_stats(self) -> dict:
        """Get architect statistics."""
        by_status = {s.name: 0 for s in ChangeStatus}
        by_type = {t.name: 0 for t in ChangeType}
        
        for change in self._changes.values():
            by_status[change.status.name] += 1
            by_type[change.change_type.name] += 1
        
        return {
            "total_changes": len(self._changes),
            "by_status": by_status,
            "by_type": by_type,
            "success_rate": sum(
                1 for c in self._changes.values() if c.success is True
            ) / max(1, len([c for c in self._changes.values() if c.success is not None])),
        }
    
    def __len__(self) -> int:
        return len(self._changes)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "changes": {k: v.to_dict() for k, v in self._changes.items()},
                "history": self._improvement_history,
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        self._changes = {
            k: ArchitecturalChange.from_dict(v)
            for k, v in data.get("changes", {}).items()
        }
        self._improvement_history = data.get("history", [])
