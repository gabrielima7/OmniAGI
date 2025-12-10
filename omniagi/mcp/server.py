"""
MCP Server Implementation

Allows OmniAGI to be used as an MCP server for Goose and other clients.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from omniagi.mcp.protocol import (
    MCPRequest, MCPResponse, MCPTool, MCPCapabilities, MCPMethod
)
from omniagi.extensions.base import Extension, ExtensionManager, get_extension_manager

# Make structlog optional
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    logger = logging.getLogger(__name__)



class MCPServer:
    """
    MCP Server that exposes OmniAGI tools to external clients.
    
    This allows Goose and other MCP-compatible tools to use OmniAGI's
    capabilities as extensions.
    """
    
    def __init__(
        self,
        name: str = "omniagi",
        version: str = "1.0.0",
        extension_manager: ExtensionManager | None = None,
    ):
        self.name = name
        self.version = version
        self._ext_manager = extension_manager or get_extension_manager()
        self._capabilities = MCPCapabilities(tools=True)
        self._initialized = False
        
        logger.info("MCP Server created", name=name, version=version)
    
    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle an incoming MCP request.
        
        Args:
            request: The MCP request to handle.
            
        Returns:
            MCPResponse with the result or error.
        """
        logger.debug("Handling MCP request", method=request.method)
        
        handlers = {
            MCPMethod.INITIALIZE.value: self._handle_initialize,
            MCPMethod.LIST_TOOLS.value: self._handle_list_tools,
            MCPMethod.CALL_TOOL.value: self._handle_call_tool,
        }
        
        handler = handlers.get(request.method)
        if handler is None:
            return MCPResponse.error_response(
                code=-32601,
                message=f"Method not found: {request.method}",
                id=request.id,
            )
        
        try:
            result = handler(request.params)
            return MCPResponse(result=result, id=request.id)
        except Exception as e:
            logger.exception("Error handling MCP request")
            return MCPResponse.error_response(
                code=-32603,
                message=str(e),
                id=request.id,
            )
    
    def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request."""
        self._initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
            "capabilities": self._capabilities.to_dict(),
        }
    
    def _handle_list_tools(self, params: dict) -> dict:
        """Handle tools/list request."""
        tools = []
        
        for ext in self._ext_manager.list_enabled():
            for tool in ext.tools:
                mcp_tool = MCPTool(
                    name=f"{ext.name}.{tool.name}",
                    description=tool.description,
                    inputSchema={
                        "type": "object",
                        "properties": tool.parameters,
                    },
                )
                tools.append(mcp_tool.to_dict())
        
        return {"tools": tools}
    
    def _handle_call_tool(self, params: dict) -> dict:
        """Handle tools/call request."""
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        # Parse extension.tool format
        if "." in name:
            ext_name, tool_name = name.split(".", 1)
            ext = self._ext_manager.get(ext_name)
            if ext and ext.is_active:
                tool = ext.get_tool(tool_name)
                if tool:
                    result = tool.execute(**arguments)
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result) if isinstance(result, dict) else str(result),
                            }
                        ],
                    }
        
        # Try global tool lookup
        result = self._ext_manager.get_tool(name)
        if result:
            ext, tool = result
            output = tool.execute(**arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(output) if isinstance(output, dict) else str(output),
                    }
                ],
            }
        
        raise ValueError(f"Tool not found: {name}")
    
    def handle_json(self, json_str: str) -> str:
        """
        Handle a JSON-RPC request string.
        
        Args:
            json_str: JSON-RPC request as string.
            
        Returns:
            JSON-RPC response as string.
        """
        try:
            data = json.loads(json_str)
            request = MCPRequest.from_dict(data)
            response = self.handle_request(request)
            return json.dumps(response.to_dict())
        except json.JSONDecodeError as e:
            return json.dumps(MCPResponse.error_response(
                code=-32700,
                message=f"Parse error: {e}",
            ).to_dict())
