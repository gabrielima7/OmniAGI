"""
Autonomous Agent Loop.

Implements a complete agent cycle:
OBSERVE → THINK → ACT → EVALUATE → LEARN → REPEAT

This enables the AGI to work autonomously on tasks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent states."""
    IDLE = auto()
    OBSERVING = auto()
    THINKING = auto()
    ACTING = auto()
    EVALUATING = auto()
    LEARNING = auto()
    WAITING = auto()
    ERROR = auto()


@dataclass
class Observation:
    """What the agent observes from environment."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""  # "user", "environment", "internal", "memory"
    content: Dict[str, Any] = field(default_factory=dict)
    raw_input: str = ""


@dataclass
class Action:
    """An action the agent can take."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    action_type: str = ""  # "respond", "query", "execute", "learn", "wait"
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    success: bool = False


@dataclass
class Evaluation:
    """Evaluation of an action's outcome."""
    action_id: str = ""
    success: bool = False
    score: float = 0.0
    feedback: str = ""
    lessons: List[str] = field(default_factory=list)


@dataclass
class AgentStep:
    """A single step in the agent loop."""
    step_number: int
    observation: Observation
    thought: str
    action: Action
    evaluation: Optional[Evaluation] = None
    duration_ms: float = 0.0


class ActionExecutor:
    """Executes actions in the environment."""
    
    def __init__(self):
        self.action_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default action handlers."""
        self.action_handlers = {
            "respond": self._handle_respond,
            "query": self._handle_query,
            "remember": self._handle_remember,
            "search": self._handle_search,
            "calculate": self._handle_calculate,
            "wait": self._handle_wait,
        }
    
    def register_handler(self, action_type: str, handler: Callable):
        """Register a custom action handler."""
        self.action_handlers[action_type] = handler
    
    def execute(self, action: Action) -> Action:
        """Execute an action."""
        handler = self.action_handlers.get(action.action_type)
        
        if handler:
            try:
                result = handler(action.parameters)
                action.result = str(result)
                action.success = True
            except Exception as e:
                action.result = f"Error: {e}"
                action.success = False
        else:
            action.result = f"Unknown action type: {action.action_type}"
            action.success = False
        
        return action
    
    def _handle_respond(self, params: Dict) -> str:
        """Handle respond action."""
        return params.get("message", "")
    
    def _handle_query(self, params: Dict) -> str:
        """Handle query action."""
        return f"Query: {params.get('query', '')}"
    
    def _handle_remember(self, params: Dict) -> str:
        """Handle remember action."""
        return f"Remembered: {params.get('content', '')}"
    
    def _handle_search(self, params: Dict) -> str:
        """Handle search action."""
        return f"Search results for: {params.get('query', '')}"
    
    def _handle_calculate(self, params: Dict) -> str:
        """Handle calculate action."""
        expr = params.get("expression", "")
        try:
            # Safe eval
            if any(x in expr for x in ['import', 'exec', 'eval', '__']):
                return "Invalid expression"
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
        except:
            return "Calculation failed"
    
    def _handle_wait(self, params: Dict) -> str:
        """Handle wait action."""
        duration = params.get("seconds", 1)
        time.sleep(min(duration, 5))  # Max 5 seconds
        return f"Waited {duration} seconds"


class AgentLoop:
    """
    Autonomous Agent Loop.
    
    Implements continuous OBSERVE → THINK → ACT → EVALUATE cycle.
    """
    
    def __init__(
        self,
        llm_func: Callable[[str, int], str],
        memory_func: Optional[Callable] = None,
    ):
        self.llm = llm_func
        self.memory_func = memory_func
        
        self.executor = ActionExecutor()
        self.state = AgentState.IDLE
        self.history: List[AgentStep] = []
        self.step_count = 0
        self.max_steps = 10
        
        # Goals and context
        self.current_goal: Optional[str] = None
        self.context: Dict[str, Any] = {}
    
    def set_goal(self, goal: str):
        """Set the agent's current goal."""
        self.current_goal = goal
        self.context["goal"] = goal
        self.context["start_time"] = datetime.now().isoformat()
    
    def observe(self, input_data: str = "", source: str = "user") -> Observation:
        """Observe the environment."""
        self.state = AgentState.OBSERVING
        
        obs = Observation(
            source=source,
            raw_input=input_data,
            content={
                "input": input_data,
                "goal": self.current_goal,
                "step": self.step_count,
                "history_length": len(self.history),
            }
        )
        
        # Add memory context if available
        if self.memory_func and input_data:
            try:
                memories = self.memory_func(input_data, 3)
                obs.content["memories"] = [str(m) for m in memories]
            except:
                pass
        
        return obs
    
    def think(self, observation: Observation) -> Tuple[str, Action]:
        """Think about what to do based on observation."""
        self.state = AgentState.THINKING
        
        # Build context for thinking
        context_parts = []
        
        if self.current_goal:
            context_parts.append(f"Goal: {self.current_goal}")
        
        if observation.raw_input:
            context_parts.append(f"Input: {observation.raw_input}")
        
        if observation.content.get("memories"):
            context_parts.append(f"Relevant memories: {observation.content['memories'][:2]}")
        
        if self.history:
            last_step = self.history[-1]
            context_parts.append(f"Last action: {last_step.action.action_type} -> {last_step.action.result[:50] if last_step.action.result else 'pending'}")
        
        context_str = "\n".join(context_parts)
        
        # Ask LLM to decide
        prompt = f"""You are an autonomous agent. Based on the current situation, decide what to do.

{context_str}

Available actions:
- respond: Send a message (use for answering questions)
- query: Ask a follow-up question
- remember: Store information in memory
- calculate: Do math calculation
- wait: Wait before next action

Think step by step, then choose ONE action.

Thought process:
1."""
        
        thought = self.llm(prompt, 300)
        
        # Parse action from thought
        action = self._parse_action(thought, observation)
        
        return thought, action
    
    def _parse_action(self, thought: str, observation: Observation) -> Action:
        """Parse action from LLM thought."""
        thought_lower = thought.lower()
        
        # Default to respond
        action_type = "respond"
        params = {}
        
        if "calculate" in thought_lower or "math" in thought_lower:
            action_type = "calculate"
            # Extract expression
            import re
            numbers = re.findall(r'\d+', observation.raw_input or thought)
            if len(numbers) >= 2:
                params["expression"] = f"{numbers[0]} + {numbers[1]}"
        
        elif "remember" in thought_lower or "store" in thought_lower:
            action_type = "remember"
            params["content"] = observation.raw_input
        
        elif "wait" in thought_lower or "pause" in thought_lower:
            action_type = "wait"
            params["seconds"] = 1
        
        elif "question" in thought_lower or "query" in thought_lower:
            action_type = "query"
            params["query"] = "What more information do you need?"
        
        else:
            # Default: respond with the thought
            action_type = "respond"
            params["message"] = thought.split(":")[-1].strip()[:500]
        
        return Action(action_type=action_type, parameters=params)
    
    def act(self, action: Action) -> Action:
        """Execute the action."""
        self.state = AgentState.ACTING
        return self.executor.execute(action)
    
    def evaluate(self, action: Action, observation: Observation) -> Evaluation:
        """Evaluate the action's outcome."""
        self.state = AgentState.EVALUATING
        
        # Simple evaluation
        success = action.success
        score = 1.0 if success else 0.0
        
        # Check if goal was achieved
        if self.current_goal and action.result:
            if any(word in action.result.lower() for word in self.current_goal.lower().split()):
                score += 0.5
        
        feedback = "Action completed successfully" if success else "Action failed"
        
        lessons = []
        if not success:
            lessons.append(f"Failed action: {action.action_type}")
        
        return Evaluation(
            action_id=action.id,
            success=success,
            score=min(score, 1.0),
            feedback=feedback,
            lessons=lessons,
        )
    
    def step(self, input_data: str = "") -> AgentStep:
        """Execute one complete agent step."""
        start = time.time()
        self.step_count += 1
        
        # OBSERVE
        observation = self.observe(input_data)
        
        # THINK
        thought, action = self.think(observation)
        
        # ACT
        action = self.act(action)
        
        # EVALUATE
        evaluation = self.evaluate(action, observation)
        
        duration = (time.time() - start) * 1000
        
        step = AgentStep(
            step_number=self.step_count,
            observation=observation,
            thought=thought[:500],
            action=action,
            evaluation=evaluation,
            duration_ms=duration,
        )
        
        self.history.append(step)
        self.state = AgentState.IDLE
        
        return step
    
    def run(self, goal: str, max_steps: int = 5) -> List[AgentStep]:
        """Run agent loop until goal is achieved or max steps reached."""
        self.set_goal(goal)
        steps = []
        
        # Initial step with goal
        step = self.step(goal)
        steps.append(step)
        
        for _ in range(max_steps - 1):
            if step.evaluation and step.evaluation.score >= 0.9:
                break  # Goal likely achieved
            
            # Continue with empty input (agent drives itself)
            step = self.step("")
            steps.append(step)
        
        return steps
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "state": self.state.name,
            "goal": self.current_goal,
            "steps_completed": self.step_count,
            "last_action": self.history[-1].action.action_type if self.history else None,
        }


class ReactAgent(AgentLoop):
    """
    ReAct (Reasoning + Acting) Agent.
    
    Interleaves reasoning and acting for better performance.
    """
    
    def think(self, observation: Observation) -> Tuple[str, Action]:
        """ReAct-style thinking with explicit reasoning."""
        self.state = AgentState.THINKING
        
        # Build ReAct prompt
        context_parts = []
        
        if self.current_goal:
            context_parts.append(f"Goal: {self.current_goal}")
        
        if observation.raw_input:
            context_parts.append(f"Observation: {observation.raw_input}")
        
        # Add recent history
        if self.history:
            context_parts.append("\nRecent history:")
            for step in self.history[-3:]:
                context_parts.append(f"  Thought: {step.thought[:100]}...")
                context_parts.append(f"  Action: {step.action.action_type}")
                context_parts.append(f"  Result: {step.action.result[:50] if step.action.result else 'N/A'}")
        
        context_str = "\n".join(context_parts)
        
        prompt = f"""You are a ReAct agent. Use this format:

Thought: [your reasoning about what to do]
Action: [one of: respond, calculate, remember, query, wait]
Action Input: [the input for the action]

{context_str}

Thought:"""
        
        response = self.llm(prompt, 400)
        
        # Parse ReAct format
        thought, action = self._parse_react_response(response, observation)
        
        return thought, action
    
    def _parse_react_response(self, response: str, observation: Observation) -> Tuple[str, Action]:
        """Parse ReAct format response."""
        lines = response.split('\n')
        
        thought = ""
        action_type = "respond"
        action_input = ""
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith("thought:"):
                thought = line[8:].strip()
            elif line.lower().startswith("action:"):
                action_type = line[7:].strip().lower()
                if action_type not in ["respond", "calculate", "remember", "query", "wait"]:
                    action_type = "respond"
            elif line.lower().startswith("action input:"):
                action_input = line[13:].strip()
        
        # Build action
        params = {}
        if action_type == "respond":
            params["message"] = action_input or thought
        elif action_type == "calculate":
            params["expression"] = action_input
        elif action_type == "remember":
            params["content"] = action_input
        elif action_type == "query":
            params["query"] = action_input
        elif action_type == "wait":
            params["seconds"] = 1
        
        return thought, Action(action_type=action_type, parameters=params)
