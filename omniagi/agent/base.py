"""
Base Agent class - the foundation of all autonomous agents.
"""

from __future__ import annotations

import json
import structlog
from dataclasses import dataclass, field
from typing import Any, Iterator, TYPE_CHECKING

from omniagi.core.engine import Engine, GenerationConfig, GenerationOutput
from omniagi.agent.persona import Persona
from omniagi.agent.state import AgentState, AgentStateManager

if TYPE_CHECKING:
    from omniagi.tools.base import Tool
    from omniagi.memory.working import WorkingMemory

logger = structlog.get_logger()


@dataclass
class Message:
    """A message in the conversation."""
    
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: str | None = None  # For tool messages
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM API."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolCall:
    """A tool invocation by the agent."""
    
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentResponse:
    """Response from agent execution."""
    
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    generation_output: GenerationOutput | None = None
    finished: bool = True


class Agent:
    """
    Base autonomous agent class.
    
    An agent combines:
    - A persona that defines its identity
    - An engine for LLM inference
    - Tools for taking actions
    - Memory for context management
    - State machine for lifecycle
    """
    
    def __init__(
        self,
        engine: Engine,
        persona: Persona | None = None,
        tools: list["Tool"] | None = None,
        max_iterations: int = 10,
    ):
        """
        Initialize an agent.
        
        Args:
            engine: The LLM inference engine.
            persona: Agent's persona/identity.
            tools: Available tools for the agent.
            max_iterations: Max reasoning loops before stopping.
        """
        self.engine = engine
        self.persona = persona or Persona.default()
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        
        self.state_manager = AgentStateManager()
        self.messages: list[Message] = []
        
        # Initialize with system message
        self._add_system_message()
        
        logger.info(
            "Agent initialized",
            persona=self.persona.name,
            tools=list(self.tools.keys()),
        )
    
    def _add_system_message(self) -> None:
        """Add the system message based on persona and tools."""
        system_content = self.persona.get_system_prompt()
        
        if self.tools:
            tools_desc = self._format_tools_description()
            system_content += f"\n\n## Available Tools\n\n{tools_desc}"
            system_content += "\n\nTo use a tool, respond with a JSON object:"
            system_content += '\n```json\n{"tool": "tool_name", "args": {...}}\n```'
        
        self.messages.append(Message(role="system", content=system_content))
    
    def _format_tools_description(self) -> str:
        """Format tools description for the system prompt."""
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"### {name}")
            lines.append(f"{tool.description}")
            if tool.parameters:
                lines.append("Parameters:")
                for param_name, param_info in tool.parameters.items():
                    required = "(required)" if param_info.get("required") else "(optional)"
                    lines.append(f"  - {param_name}: {param_info.get('description', '')} {required}")
            lines.append("")
        return "\n".join(lines)
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content, **kwargs))
    
    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message("user", content)
    
    def _parse_tool_call(self, content: str) -> ToolCall | None:
        """Try to parse a tool call from the response."""
        # Look for JSON in the response
        import re
        
        # Try to find JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "tool" in data:
                    import uuid
                    return ToolCall(
                        id=str(uuid.uuid4()),
                        name=data["tool"],
                        arguments=data.get("args", {}),
                    )
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON
        try:
            # Find first { and last }
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                if "tool" in data:
                    import uuid
                    return ToolCall(
                        id=str(uuid.uuid4()),
                        name=data["tool"],
                        arguments=data.get("args", {}),
                    )
        except json.JSONDecodeError:
            pass
        
        return None
    
    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool and return the result."""
        tool = self.tools.get(tool_call.name)
        if not tool:
            return f"Error: Unknown tool '{tool_call.name}'"
        
        try:
            self.state_manager.transition_to(AgentState.ACTING, f"Executing {tool_call.name}")
            result = await tool.execute(**tool_call.arguments)
            return str(result)
        except Exception as e:
            logger.error("Tool execution failed", tool=tool_call.name, error=str(e))
            return f"Error executing {tool_call.name}: {str(e)}"
    
    def think(self, config: GenerationConfig | None = None) -> AgentResponse:
        """
        Run one step of agent reasoning (synchronous).
        
        Returns:
            AgentResponse with content and any tool calls.
        """
        self.state_manager.transition_to(AgentState.THINKING)
        
        messages_for_llm = [m.to_dict() for m in self.messages]
        output = self.engine.chat(messages_for_llm, config)
        
        content = output.text.strip()
        tool_calls = []
        
        # Check if the response contains a tool call
        tool_call = self._parse_tool_call(content)
        if tool_call and tool_call.name in self.tools:
            tool_calls.append(tool_call)
        
        self.add_message("assistant", content)
        
        self.state_manager.transition_to(AgentState.IDLE)
        
        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            generation_output=output,
            finished=len(tool_calls) == 0,
        )
    
    async def run(
        self,
        user_input: str,
        config: GenerationConfig | None = None,
    ) -> str:
        """
        Run the agent with the given input until completion.
        
        This implements a ReAct-style loop:
        1. Think (generate response)
        2. If tool call, execute and add result
        3. Repeat until no more tool calls or max iterations
        
        Args:
            user_input: The user's input message.
            config: Generation configuration.
            
        Returns:
            The final response content.
        """
        self.add_user_message(user_input)
        
        for i in range(self.max_iterations):
            logger.info("Agent iteration", iteration=i + 1)
            
            response = self.think(config)
            
            if response.finished or not response.tool_calls:
                return response.content
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                logger.info("Executing tool", tool=tool_call.name)
                result = await self._execute_tool(tool_call)
                
                self.add_message(
                    "tool",
                    result,
                    name=tool_call.name,
                    tool_call_id=tool_call.id,
                )
        
        logger.warning("Max iterations reached")
        return response.content
    
    def clear_history(self, keep_system: bool = True) -> None:
        """Clear conversation history."""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages = []
            self._add_system_message()
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for memory."""
        user_messages = [m.content for m in self.messages if m.role == "user"]
        assistant_messages = [m.content for m in self.messages if m.role == "assistant"]
        
        return (
            f"Conversation with {len(user_messages)} user messages and "
            f"{len(assistant_messages)} assistant responses."
        )
