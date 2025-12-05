"""
Base Tool interface and utilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolResultStatus(Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    DENIED = "denied"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    
    status: ToolResultStatus
    output: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success(cls, output: Any, **metadata) -> "ToolResult":
        """Create a successful result."""
        return cls(
            status=ToolResultStatus.SUCCESS,
            output=output,
            metadata=metadata,
        )
    
    @classmethod
    def error(cls, error: str, **metadata) -> "ToolResult":
        """Create an error result."""
        return cls(
            status=ToolResultStatus.ERROR,
            output=None,
            error=error,
            metadata=metadata,
        )
    
    def __str__(self) -> str:
        if self.status == ToolResultStatus.SUCCESS:
            return str(self.output)
        return f"Error: {self.error}"


class Tool(ABC):
    """
    Abstract base class for all tools.
    
    Tools are the actions an agent can take to interact with
    the world. Each tool has a name, description, and parameters.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this tool does."""
        pass
    
    @property
    def parameters(self) -> dict[str, Any]:
        """
        JSON Schema-style description of parameters.
        
        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        return {}
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments.
        
        Args:
            **kwargs: Tool-specific arguments.
            
        Returns:
            ToolResult with success/error status and output.
        """
        pass
    
    def validate_args(self, **kwargs) -> list[str]:
        """
        Validate the provided arguments.
        
        Returns:
            List of validation error messages, empty if valid.
        """
        errors = []
        for name, info in self.parameters.items():
            if info.get("required") and name not in kwargs:
                errors.append(f"Missing required parameter: {name}")
        return errors
    
    def __repr__(self) -> str:
        return f"Tool(name={self.name})"
