"""
HiveMind - Collective intelligence through agent coordination.

Enables multiple AI agents to share knowledge, coordinate
decisions, and achieve emergent superintelligence.
"""

from __future__ import annotations

import json
import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

logger = structlog.get_logger()


class ConsensusMethod(Enum):
    """Methods for reaching consensus."""
    
    MAJORITY = auto()       # Simple majority
    WEIGHTED = auto()       # Weighted by competence
    UNANIMOUS = auto()      # All must agree
    EXPERT = auto()         # Defer to expert
    DELIBERATIVE = auto()   # Discuss until consensus


class AgentRole(Enum):
    """Roles in the collective."""
    
    GENERALIST = auto()     # General purpose
    SPECIALIST = auto()     # Domain expert
    CRITIC = auto()         # Evaluates proposals
    COORDINATOR = auto()    # Coordinates decisions
    LEARNER = auto()        # Focuses on learning


@dataclass
class CollectiveAgent:
    """An agent in the collective."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    role: AgentRole = AgentRole.GENERALIST
    
    # Competencies
    domains: list[str] = field(default_factory=list)
    competence_scores: dict[str, float] = field(default_factory=dict)
    
    # State
    active: bool = True
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Performance
    proposals_made: int = 0
    proposals_accepted: int = 0
    
    @property
    def acceptance_rate(self) -> float:
        if self.proposals_made == 0:
            return 0.0
        return self.proposals_accepted / self.proposals_made
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.name,
            "domains": self.domains,
            "competence": self.competence_scores,
            "active": self.active,
            "acceptance_rate": self.acceptance_rate,
        }


@dataclass
class Proposal:
    """A proposal for collective decision."""
    
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    author_id: str = ""
    content: str = ""
    domain: str = "general"
    
    # Voting
    votes: dict[str, bool] = field(default_factory=dict)  # agent_id -> vote
    weights: dict[str, float] = field(default_factory=dict)
    
    # Discussion
    comments: list[dict] = field(default_factory=list)
    
    # Status
    resolved: bool = False
    accepted: bool = False
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_vote(self, agent_id: str, vote: bool, weight: float = 1.0) -> None:
        self.votes[agent_id] = vote
        self.weights[agent_id] = weight
    
    def get_result(self, method: ConsensusMethod = ConsensusMethod.WEIGHTED) -> bool:
        """Calculate result based on consensus method."""
        if not self.votes:
            return False
        
        if method == ConsensusMethod.MAJORITY:
            yes = sum(1 for v in self.votes.values() if v)
            return yes > len(self.votes) / 2
        
        elif method == ConsensusMethod.WEIGHTED:
            yes_weight = sum(
                self.weights.get(aid, 1.0)
                for aid, v in self.votes.items() if v
            )
            total_weight = sum(self.weights.values()) or len(self.votes)
            return yes_weight > total_weight / 2
        
        elif method == ConsensusMethod.UNANIMOUS:
            return all(self.votes.values())
        
        return False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "author": self.author_id,
            "content": self.content[:200],
            "domain": self.domain,
            "votes": len(self.votes),
            "resolved": self.resolved,
            "accepted": self.accepted,
        }


@dataclass
class SharedKnowledge:
    """Knowledge shared across the collective."""
    
    key: str
    value: Any
    confidence: float = 1.0
    source_agents: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0


class ConsensusProtocol:
    """Protocol for reaching collective decisions."""
    
    def __init__(
        self,
        method: ConsensusMethod = ConsensusMethod.WEIGHTED,
        timeout: float = 60.0,
        min_participants: int = 2,
    ):
        self.method = method
        self.timeout = timeout
        self.min_participants = min_participants
    
    def validate_quorum(self, participants: int) -> bool:
        """Check if enough participants for decision."""
        return participants >= self.min_participants
    
    def calculate_result(self, proposal: Proposal) -> bool:
        """Calculate consensus result."""
        return proposal.get_result(self.method)


class HiveMind:
    """
    Collective superintelligence system.
    
    Coordinates multiple agents to share knowledge,
    make collective decisions, and achieve emergent
    capabilities beyond individual agents.
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED,
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        
        # Agents
        self._agents: dict[str, CollectiveAgent] = {}
        
        # Consensus
        self._protocol = ConsensusProtocol(method=consensus_method)
        self._proposals: dict[str, Proposal] = {}
        
        # Shared knowledge
        self._knowledge: dict[str, SharedKnowledge] = {}
        
        # Global state
        self._collective_state: dict[str, Any] = {}
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("HiveMind initialized", method=consensus_method.name)
    
    def register_agent(
        self,
        name: str,
        role: AgentRole = AgentRole.GENERALIST,
        domains: list[str] = None,
    ) -> CollectiveAgent:
        """Register a new agent in the collective."""
        agent = CollectiveAgent(
            name=name,
            role=role,
            domains=domains or [],
        )
        
        self._agents[agent.id] = agent
        self._save()
        
        logger.info("Agent registered", id=agent.id, name=name, role=role.name)
        return agent
    
    def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent."""
        if agent_id in self._agents:
            self._agents[agent_id].active = False
            self._save()
            return True
        return False
    
    def get_active_agents(self) -> list[CollectiveAgent]:
        """Get all active agents."""
        return [a for a in self._agents.values() if a.active]
    
    def get_experts(self, domain: str) -> list[CollectiveAgent]:
        """Get agents with expertise in domain."""
        experts = []
        for agent in self.get_active_agents():
            if domain in agent.domains:
                experts.append(agent)
            elif agent.role == AgentRole.SPECIALIST and domain in agent.competence_scores:
                experts.append(agent)
        return sorted(experts, key=lambda a: a.competence_scores.get(domain, 0), reverse=True)
    
    def propose(
        self,
        content: str,
        author_id: str,
        domain: str = "general",
    ) -> Proposal:
        """Create a proposal for collective decision."""
        if author_id not in self._agents:
            raise ValueError(f"Unknown agent: {author_id}")
        
        proposal = Proposal(
            author_id=author_id,
            content=content,
            domain=domain,
        )
        
        self._proposals[proposal.id] = proposal
        self._agents[author_id].proposals_made += 1
        self._save()
        
        logger.info("Proposal created", id=proposal.id, domain=domain)
        return proposal
    
    def vote(
        self,
        proposal_id: str,
        agent_id: str,
        vote: bool,
        comment: str = None,
    ) -> bool:
        """Vote on a proposal."""
        if proposal_id not in self._proposals:
            return False
        if agent_id not in self._agents:
            return False
        
        proposal = self._proposals[proposal_id]
        agent = self._agents[agent_id]
        
        if proposal.resolved:
            return False
        
        # Calculate weight based on competence
        weight = agent.competence_scores.get(proposal.domain, 0.5)
        if agent.role == AgentRole.SPECIALIST and proposal.domain in agent.domains:
            weight *= 1.5
        
        proposal.add_vote(agent_id, vote, weight)
        
        if comment:
            proposal.comments.append({
                "agent_id": agent_id,
                "comment": comment,
                "timestamp": datetime.now().isoformat(),
            })
        
        self._save()
        return True
    
    def resolve_proposal(self, proposal_id: str) -> bool:
        """Resolve a proposal based on votes."""
        if proposal_id not in self._proposals:
            return False
        
        proposal = self._proposals[proposal_id]
        
        if not self._protocol.validate_quorum(len(proposal.votes)):
            logger.warning("Quorum not met", proposal=proposal_id)
            return False
        
        proposal.accepted = self._protocol.calculate_result(proposal)
        proposal.resolved = True
        
        # Update author stats
        if proposal.accepted:
            self._agents[proposal.author_id].proposals_accepted += 1
        
        self._save()
        
        logger.info(
            "Proposal resolved",
            id=proposal_id,
            accepted=proposal.accepted,
            votes=len(proposal.votes),
        )
        
        return proposal.accepted
    
    def share_knowledge(
        self,
        key: str,
        value: Any,
        agent_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Share knowledge with the collective."""
        if key in self._knowledge:
            # Update existing knowledge
            existing = self._knowledge[key]
            
            # Merge confidence
            new_confidence = (existing.confidence + confidence) / 2
            existing.confidence = new_confidence
            existing.value = value
            existing.updated_at = datetime.now().isoformat()
            
            if agent_id not in existing.source_agents:
                existing.source_agents.append(agent_id)
        else:
            # New knowledge
            self._knowledge[key] = SharedKnowledge(
                key=key,
                value=value,
                confidence=confidence,
                source_agents=[agent_id],
            )
        
        self._save()
        logger.debug("Knowledge shared", key=key)
    
    def query_knowledge(self, key: str) -> Any | None:
        """Query shared knowledge."""
        if key in self._knowledge:
            self._knowledge[key].access_count += 1
            return self._knowledge[key].value
        return None
    
    def search_knowledge(self, pattern: str) -> list[SharedKnowledge]:
        """Search knowledge by pattern."""
        results = []
        for key, knowledge in self._knowledge.items():
            if pattern.lower() in key.lower():
                results.append(knowledge)
        return sorted(results, key=lambda k: k.confidence, reverse=True)
    
    def broadcast(
        self,
        message: str,
        sender_id: str,
        target_roles: list[AgentRole] = None,
    ) -> int:
        """Broadcast message to agents."""
        recipients = 0
        
        for agent in self.get_active_agents():
            if agent.id == sender_id:
                continue
            
            if target_roles and agent.role not in target_roles:
                continue
            
            # In production, this would actually notify agent
            recipients += 1
        
        logger.info(
            "Broadcast sent",
            sender=sender_id,
            recipients=recipients,
        )
        
        return recipients
    
    def get_collective_decision(
        self,
        question: str,
        domain: str = "general",
    ) -> dict:
        """
        Get collective decision on a question.
        
        This aggregates opinions from available agents
        weighted by their expertise.
        """
        experts = self.get_experts(domain)
        active = self.get_active_agents()
        
        return {
            "question": question,
            "domain": domain,
            "experts_available": len(experts),
            "active_agents": len(active),
            "needs_proposal": True,
            "suggested_method": (
                ConsensusMethod.EXPERT.name if experts
                else ConsensusMethod.WEIGHTED.name
            ),
        }
    
    def update_competence(
        self,
        agent_id: str,
        domain: str,
        score: float,
    ) -> None:
        """Update agent competence in a domain."""
        if agent_id in self._agents:
            self._agents[agent_id].competence_scores[domain] = max(0, min(1, score))
            self._save()
    
    def get_stats(self) -> dict:
        """Get hivemind statistics."""
        active = self.get_active_agents()
        
        return {
            "total_agents": len(self._agents),
            "active_agents": len(active),
            "knowledge_items": len(self._knowledge),
            "proposals": len(self._proposals),
            "resolved_proposals": sum(1 for p in self._proposals.values() if p.resolved),
            "acceptance_rate": (
                sum(1 for p in self._proposals.values() if p.accepted) /
                max(1, sum(1 for p in self._proposals.values() if p.resolved))
            ),
            "by_role": {
                role.name: sum(1 for a in active if a.role == role)
                for role in AgentRole
            },
        }
    
    def __len__(self) -> int:
        return len(self._agents)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "agents": {k: v.to_dict() for k, v in self._agents.items()},
                "proposals": {k: v.to_dict() for k, v in self._proposals.items()},
                "knowledge_keys": list(self._knowledge.keys())[:100],
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        # Simplified load - full implementation would restore all state
        pass
