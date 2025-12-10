"""
MCP Protocol Data Structures

Based on the Model Context Protocol specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class MCPMethod(str, Enum):
    """MCP method types."""
    INITIALIZE = "initialize"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"


@dataclass
class MCPTool:
    """
    Represents a tool in MCP format.
    
    Compatible with Goose and other MCP servers.
    """
    name: str
    description: str
    inputSchema: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MCPTool":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            inputSchema=data.get("inputSchema", {}),
        )


@dataclass
class MCPResource:
    """Represents a resource in MCP format."""
    uri: str
    name: str
    description: str = ""
    mimeType: str = "text/plain"
    
    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mimeType,
        }


@dataclass
class MCPPrompt:
    """Represents a prompt template in MCP format."""
    name: str
    description: str = ""
    arguments: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


@dataclass
class MCPRequest:
    """
    MCP JSON-RPC request.
    """
    method: str
    params: dict = field(default_factory=dict)
    id: int | str = 1
    jsonrpc: str = "2.0"
    
    def to_dict(self) -> dict:
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MCPRequest":
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", 1),
            method=data["method"],
            params=data.get("params", {}),
        )


@dataclass
class MCPResponse:
    """
    MCP JSON-RPC response.
    """
    result: Any = None
    error: dict | None = None
    id: int | str = 1
    jsonrpc: str = "2.0"
    
    @property
    def success(self) -> bool:
        return self.error is None
    
    def to_dict(self) -> dict:
        d = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "MCPResponse":
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id", 1),
            result=data.get("result"),
            error=data.get("error"),
        )
    
    @classmethod
    def error_response(cls, code: int, message: str, id: int | str = 1) -> "MCPResponse":
        return cls(
            id=id,
            error={"code": code, "message": message},
        )


@dataclass
class MCPCapabilities:
    """Server capabilities."""
    tools: bool = True
    resources: bool = False
    prompts: bool = False
    
    def to_dict(self) -> dict:
        caps = {}
        if self.tools:
            caps["tools"] = {}
        if self.resources:
            caps["resources"] = {}
        if self.prompts:
            caps["prompts"] = {}
        return caps
