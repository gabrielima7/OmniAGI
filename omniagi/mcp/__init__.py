"""
OmniAGI MCP (Model Context Protocol) Support

Enables interoperability with Goose and other MCP-compatible tools.
"""

from omniagi.mcp.server import MCPServer
from omniagi.mcp.client import MCPClient
from omniagi.mcp.protocol import MCPRequest, MCPResponse, MCPTool

__all__ = [
    "MCPServer",
    "MCPClient",
    "MCPRequest",
    "MCPResponse",
    "MCPTool",
]
