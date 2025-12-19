"""
Multi-Agent System.

Implements specialized agents that work together:
- Coordinator: Orchestrates other agents
- Researcher: Searches and retrieves information
- Reasoner: Logical reasoning and analysis
- Coder: Generates and evaluates code
- Critic: Reviews and improves outputs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles for specialized agents."""
    COORDINATOR = auto()
    RESEARCHER = auto()
    REASONER = auto()
    CODER = auto()
    CRITIC = auto()
    MEMORY = auto()


@dataclass
class AgentMessage:
    """Message between agents."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    from_agent: str = ""
    to_agent: str = ""
    content: str = ""
    message_type: str = "request"  # request, response, broadcast
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """A task assigned to an agent."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    description: str = ""
    assigned_to: str = ""
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None
    priority: int = 5  # 1-10


class SpecializedAgent:
    """Base class for specialized agents."""
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        llm_func: Callable[[str, int], str],
    ):
        self.name = name
        self.role = role
        self.llm = llm_func
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
    
    def receive(self, message: AgentMessage):
        """Receive a message."""
        self.inbox.append(message)
    
    def send(self, to_agent: str, content: str, msg_type: str = "response") -> AgentMessage:
        """Send a message to another agent."""
        msg = AgentMessage(
            from_agent=self.name,
            to_agent=to_agent,
            content=content,
            message_type=msg_type,
        )
        self.outbox.append(msg)
        return msg
    
    def process_task(self, task: AgentTask) -> str:
        """Process a task. Override in subclasses."""
        raise NotImplementedError
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return f"You are a {self.role.name} agent. Your name is {self.name}."


class ResearcherAgent(SpecializedAgent):
    """Agent specialized in research and information retrieval."""
    
    def __init__(self, name: str, llm_func: Callable, memory_func: Optional[Callable] = None):
        super().__init__(name, AgentRole.RESEARCHER, llm_func)
        self.memory_func = memory_func
    
    def process_task(self, task: AgentTask) -> str:
        """Research a topic."""
        # Search memory if available
        context = ""
        if self.memory_func:
            try:
                memories = self.memory_func(task.description, 5)
                if memories:
                    context = "\n".join(f"- {m.content if hasattr(m, 'content') else str(m)}" for m in memories[:5])
            except:
                pass
        
        prompt = f"""You are a research specialist. Find relevant information about:

Topic: {task.description}

{"Relevant information from memory:" + chr(10) + context if context else ""}

Provide a comprehensive research summary with key facts:"""
        
        return self.llm(prompt, 500)


class ReasonerAgent(SpecializedAgent):
    """Agent specialized in logical reasoning and analysis."""
    
    def __init__(self, name: str, llm_func: Callable):
        super().__init__(name, AgentRole.REASONER, llm_func)
    
    def process_task(self, task: AgentTask) -> str:
        """Reason about a problem."""
        prompt = f"""You are a logical reasoning specialist. Analyze this carefully:

Problem: {task.description}

Think step by step:
1. Identify the key elements
2. Find relationships and patterns
3. Apply logical reasoning
4. Draw conclusions

Analysis:"""
        
        return self.llm(prompt, 500)


class CoderAgent(SpecializedAgent):
    """Agent specialized in code generation and analysis."""
    
    def __init__(self, name: str, llm_func: Callable):
        super().__init__(name, AgentRole.CODER, llm_func)
    
    def process_task(self, task: AgentTask) -> str:
        """Generate or analyze code."""
        prompt = f"""You are a coding specialist. Write clean, efficient code for:

Task: {task.description}

Provide:
1. Code solution
2. Brief explanation
3. Any important notes

Solution:"""
        
        return self.llm(prompt, 600)


class CriticAgent(SpecializedAgent):
    """Agent specialized in reviewing and improving outputs."""
    
    def __init__(self, name: str, llm_func: Callable):
        super().__init__(name, AgentRole.CRITIC, llm_func)
    
    def process_task(self, task: AgentTask) -> str:
        """Critique and improve content."""
        prompt = f"""You are a critical reviewer. Evaluate this carefully:

Content to review: {task.description}

Provide:
1. Strengths (what's good)
2. Weaknesses (what needs improvement)
3. Specific suggestions for improvement
4. Improved version if applicable

Review:"""
        
        return self.llm(prompt, 500)


class CoordinatorAgent(SpecializedAgent):
    """Agent that coordinates other agents."""
    
    def __init__(self, name: str, llm_func: Callable):
        super().__init__(name, AgentRole.COORDINATOR, llm_func)
        self.agents: Dict[str, SpecializedAgent] = {}
    
    def register_agent(self, agent: SpecializedAgent):
        """Register an agent."""
        self.agents[agent.name] = agent
    
    def delegate_task(self, task: AgentTask) -> str:
        """Delegate task to appropriate agent."""
        # Decide which agent should handle this
        agent_name = self._select_agent(task.description)
        
        if agent_name in self.agents:
            task.assigned_to = agent_name
            task.status = "in_progress"
            
            result = self.agents[agent_name].process_task(task)
            task.status = "completed"
            task.result = result
            
            return result
        else:
            return self.llm(task.description, 400)
    
    def _select_agent(self, task_description: str) -> str:
        """Select the best agent for a task."""
        desc_lower = task_description.lower()
        
        # Simple rule-based selection
        if any(w in desc_lower for w in ['code', 'program', 'function', 'script', 'python']):
            return "coder"
        elif any(w in desc_lower for w in ['research', 'find', 'search', 'information', 'what is']):
            return "researcher"
        elif any(w in desc_lower for w in ['analyze', 'reason', 'why', 'logic', 'prove']):
            return "reasoner"
        elif any(w in desc_lower for w in ['review', 'improve', 'critique', 'evaluate']):
            return "critic"
        
        return "reasoner"  # Default
    
    def process_task(self, task: AgentTask) -> str:
        """Coordinate task execution."""
        return self.delegate_task(task)


class MultiAgentSystem:
    """
    Complete Multi-Agent System.
    
    Multiple specialized agents working together.
    """
    
    def __init__(self, llm_func: Callable[[str, int], str], memory_func: Optional[Callable] = None):
        self.llm = llm_func
        self.memory_func = memory_func
        
        # Create specialized agents
        self.coordinator = CoordinatorAgent("coordinator", llm_func)
        self.researcher = ResearcherAgent("researcher", llm_func, memory_func)
        self.reasoner = ReasonerAgent("reasoner", llm_func)
        self.coder = CoderAgent("coder", llm_func)
        self.critic = CriticAgent("critic", llm_func)
        
        # Register agents with coordinator
        self.coordinator.register_agent(self.researcher)
        self.coordinator.register_agent(self.reasoner)
        self.coordinator.register_agent(self.coder)
        self.coordinator.register_agent(self.critic)
        
        self.task_history: List[AgentTask] = []
    
    def process(self, query: str, with_critique: bool = True) -> Dict[str, Any]:
        """Process a query using multiple agents."""
        results = {}
        
        # Create task
        task = AgentTask(description=query)
        
        # Coordinator delegates to appropriate agent
        initial_result = self.coordinator.delegate_task(task)
        results["initial"] = {
            "agent": task.assigned_to,
            "result": initial_result,
        }
        
        # Optional critique phase
        if with_critique:
            critique_task = AgentTask(
                description=f"Review and improve this response:\n{initial_result[:500]}"
            )
            critique_result = self.critic.process_task(critique_task)
            results["critique"] = critique_result
        
        self.task_history.append(task)
        
        return {
            "query": query,
            "primary_agent": task.assigned_to,
            "results": results,
            "task_id": task.id,
        }
    
    def collaborative_solve(self, problem: str) -> Dict[str, Any]:
        """Solve a problem collaboratively using all agents."""
        results = {}
        
        # Step 1: Researcher gathers information
        research_task = AgentTask(description=f"Research background information for: {problem}")
        research = self.researcher.process_task(research_task)
        results["research"] = research
        
        # Step 2: Reasoner analyzes
        reason_task = AgentTask(
            description=f"Problem: {problem}\n\nResearch findings: {research[:500]}\n\nAnalyze and reason about this."
        )
        reasoning = self.reasoner.process_task(reason_task)
        results["reasoning"] = reasoning
        
        # Step 3: If code is needed, coder helps
        if any(w in problem.lower() for w in ['code', 'implement', 'program', 'script']):
            code_task = AgentTask(
                description=f"Based on this analysis: {reasoning[:300]}\n\nImplement a solution for: {problem}"
            )
            code = self.coder.process_task(code_task)
            results["code"] = code
        
        # Step 4: Critic reviews final result
        final_content = "\n\n".join(f"{k}: {v[:200]}" for k, v in results.items())
        critique_task = AgentTask(description=f"Review this collaborative solution:\n{final_content}")
        critique = self.critic.process_task(critique_task)
        results["critique"] = critique
        
        return {
            "problem": problem,
            "agents_used": list(results.keys()),
            "results": results,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_tasks": len(self.task_history),
            "agents": ["coordinator", "researcher", "reasoner", "coder", "critic"],
            "memory_connected": self.memory_func is not None,
        }
