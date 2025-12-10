"""
MCP Client Implementation

Allows OmniAGI to connect to external MCP servers.
"""

from __future__ import annotations

import json
import subprocess
import structlog
from typing import Any
from pathlib import Path

from omniagi.mcp.protocol import MCPRequest, MCPResponse, MCPTool

logger = structlog.get_logger()


class MCPClient:
    """
    MCP Client for connecting to external MCP servers.
    
    Supports:
    - stdio: Local server via stdin/stdout
    - sse: Server-sent events (HTTP)
    """
    
    def __init__(self, name: str = "omniagi-client"):
        self.name = name
        self._process: subprocess.Popen | None = None
        self._tools: list[MCPTool] = []
        self._server_info: dict = {}
        self._connected = False
        self._request_id = 0
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def tools(self) -> list[MCPTool]:
        return self._tools
    
    def connect_stdio(self, command: str | list[str], cwd: str | Path | None = None) -> bool:
        """
        Connect to an MCP server via stdio.
        
        Args:
            command: Command to start the server.
            cwd: Working directory.
            
        Returns:
            True if connected successfully.
        """
        if isinstance(command, str):
            command = command.split()
        
        logger.info("Connecting to MCP server", command=command)
        
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
                text=True,
            )
            
            # Initialize
            response = self._send_request(MCPRequest(
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {
                        "name": self.name,
                        "version": "1.0.0",
                    },
                    "capabilities": {},
                },
            ))
            
            if response.success:
                self._server_info = response.result.get("serverInfo", {})
                self._connected = True
                
                # List tools
                self._refresh_tools()
                
                logger.info(
                    "Connected to MCP server",
                    server=self._server_info.get("name"),
                    tools=len(self._tools),
                )
                return True
            else:
                logger.error("Failed to initialize MCP connection", error=response.error)
                return False
                
        except Exception as e:
            logger.exception("Failed to connect to MCP server")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._process:
            self._process.terminate()
            self._process = None
        self._connected = False
        self._tools = []
        logger.info("Disconnected from MCP server")
    
    def _send_request(self, request: MCPRequest) -> MCPResponse:
        """Send a request to the server."""
        if not self._process:
            return MCPResponse.error_response(-32600, "Not connected")
        
        self._request_id += 1
        request.id = self._request_id
        
        try:
            request_json = json.dumps(request.to_dict()) + "\n"
            self._process.stdin.write(request_json)
            self._process.stdin.flush()
            
            response_line = self._process.stdout.readline()
            if not response_line:
                return MCPResponse.error_response(-32600, "No response")
            
            data = json.loads(response_line)
            return MCPResponse.from_dict(data)
            
        except Exception as e:
            return MCPResponse.error_response(-32603, str(e))
    
    def _refresh_tools(self) -> None:
        """Refresh the list of available tools."""
        response = self._send_request(MCPRequest(method="tools/list"))
        
        if response.success:
            tools_data = response.result.get("tools", [])
            self._tools = [MCPTool.from_dict(t) for t in tools_data]
    
    def list_tools(self) -> list[MCPTool]:
        """Get list of available tools."""
        if self._connected:
            self._refresh_tools()
        return self._tools
    
    def call_tool(self, name: str, **arguments) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Tool name.
            **arguments: Tool arguments.
            
        Returns:
            Tool result.
        """
        if not self._connected:
            raise RuntimeError("Not connected to MCP server")
        
        response = self._send_request(MCPRequest(
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments,
            },
        ))
        
        if response.success:
            content = response.result.get("content", [])
            if content and content[0].get("type") == "text":
                text = content[0].get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            return content
        else:
            raise RuntimeError(f"Tool call failed: {response.error}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
