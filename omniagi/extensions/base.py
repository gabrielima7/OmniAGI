"""
Extension Base Classes

Inspired by Goose's MCP extension architecture.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

# Make structlog optional
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)



@dataclass
class Tool:
    """
    Represents a tool that an extension provides.
    
    Tools are functions that the AI agent can call to perform actions.
    """
    name: str
    description: str
    handler: Callable[..., Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        if self.handler is None:
            raise NotImplementedError(f"Tool {self.name} has no handler")
        return self.handler(**kwargs)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_confirmation": self.requires_confirmation,
        }


class Extension(ABC):
    """
    Base class for OmniAGI extensions.
    
    Extensions provide tools and capabilities to the AGI system.
    Inspired by Goose's MCP extension architecture.
    """
    
    name: str = "base"
    description: str = "Base extension"
    version: str = "1.0.0"
    enabled: bool = True
    
    def __init__(self):
        self._tools: list[Tool] = []
        self._active = False
        logger.info(f"Extension initialized", name=self.name)
    
    @property
    def tools(self) -> list[Tool]:
        """Get list of tools provided by this extension."""
        return self._tools
    
    @property
    def is_active(self) -> bool:
        """Check if extension is active."""
        return self._active
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool with this extension."""
        self._tools.append(tool)
        logger.debug(f"Tool registered", extension=self.name, tool=tool.name)
    
    def activate(self) -> None:
        """Activate the extension."""
        self._active = True
        self._setup()
        logger.info(f"Extension activated", name=self.name)
    
    def deactivate(self) -> None:
        """Deactivate the extension."""
        self._cleanup()
        self._active = False
        logger.info(f"Extension deactivated", name=self.name)
    
    def _setup(self) -> None:
        """Override to perform setup when activated."""
        pass
    
    def _cleanup(self) -> None:
        """Override to perform cleanup when deactivated."""
        pass
    
    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool not found: {name}")
        return tool.execute(**kwargs)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "enabled": self.enabled,
            "active": self._active,
            "tools": [t.to_dict() for t in self._tools],
        }


class ExtensionManager:
    """
    Manages all extensions in the OmniAGI system.
    
    Handles discovery, loading, and lifecycle of extensions.
    """
    
    def __init__(self):
        self._extensions: dict[str, Extension] = {}
        self._enabled: set[str] = set()
        logger.info("ExtensionManager initialized")
    
    def register(self, extension: Extension) -> None:
        """Register an extension."""
        self._extensions[extension.name] = extension
        logger.info(f"Extension registered", name=extension.name)
    
    def unregister(self, name: str) -> None:
        """Unregister an extension."""
        if name in self._extensions:
            if name in self._enabled:
                self.disable(name)
            del self._extensions[name]
            logger.info(f"Extension unregistered", name=name)
    
    def enable(self, name: str) -> None:
        """Enable an extension."""
        if name not in self._extensions:
            raise ValueError(f"Extension not found: {name}")
        
        ext = self._extensions[name]
        ext.activate()
        self._enabled.add(name)
        logger.info(f"Extension enabled", name=name)
    
    def disable(self, name: str) -> None:
        """Disable an extension."""
        if name in self._enabled:
            self._extensions[name].deactivate()
            self._enabled.remove(name)
            logger.info(f"Extension disabled", name=name)
    
    def get(self, name: str) -> Extension | None:
        """Get an extension by name."""
        return self._extensions.get(name)
    
    def list_extensions(self) -> list[Extension]:
        """List all registered extensions."""
        return list(self._extensions.values())
    
    def list_enabled(self) -> list[Extension]:
        """List all enabled extensions."""
        return [self._extensions[n] for n in self._enabled]
    
    def list_tools(self) -> list[Tool]:
        """List all tools from enabled extensions."""
        tools = []
        for name in self._enabled:
            tools.extend(self._extensions[name].tools)
        return tools
    
    def get_tool(self, name: str) -> tuple[Extension, Tool] | None:
        """Find a tool by name across all enabled extensions."""
        for ext_name in self._enabled:
            ext = self._extensions[ext_name]
            tool = ext.get_tool(name)
            if tool:
                return ext, tool
        return None
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        result = self.get_tool(name)
        if result is None:
            raise ValueError(f"Tool not found: {name}")
        ext, tool = result
        return tool.execute(**kwargs)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "extensions": {n: e.to_dict() for n, e in self._extensions.items()},
            "enabled": list(self._enabled),
        }


# Global extension manager instance
_manager: ExtensionManager | None = None


def get_extension_manager() -> ExtensionManager:
    """Get the global extension manager instance."""
    global _manager
    if _manager is None:
        _manager = ExtensionManager()
    return _manager
