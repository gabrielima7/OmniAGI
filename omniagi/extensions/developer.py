"""
Developer Extension

Provides development tools like shell execution, file operations, etc.
Inspired by Goose's Developer extension.
"""

from __future__ import annotations

import os
import subprocess
import structlog
from pathlib import Path

from omniagi.extensions.base import Extension, Tool

logger = structlog.get_logger()


class DeveloperExtension(Extension):
    """
    Developer extension with tools for software development.
    
    Tools:
    - shell: Execute shell commands
    - read_file: Read file contents
    - write_file: Write to file
    - list_dir: List directory contents
    - search_files: Search for files by pattern
    """
    
    name = "developer"
    description = "Development tools for shell, files, and code"
    version = "1.0.0"
    
    def __init__(self, working_dir: str | Path | None = None):
        super().__init__()
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self._register_tools()
    
    def _register_tools(self) -> None:
        """Register all developer tools."""
        self.register_tool(Tool(
            name="shell",
            description="Execute a shell command and return the output",
            handler=self._shell,
            parameters={
                "command": {"type": "string", "description": "Command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
            },
            requires_confirmation=True,
        ))
        
        self.register_tool(Tool(
            name="read_file",
            description="Read the contents of a file",
            handler=self._read_file,
            parameters={
                "path": {"type": "string", "description": "Path to the file"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
            },
        ))
        
        self.register_tool(Tool(
            name="write_file",
            description="Write content to a file",
            handler=self._write_file,
            parameters={
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
            },
            requires_confirmation=True,
        ))
        
        self.register_tool(Tool(
            name="list_dir",
            description="List contents of a directory",
            handler=self._list_dir,
            parameters={
                "path": {"type": "string", "description": "Directory path", "default": "."},
                "recursive": {"type": "boolean", "description": "List recursively", "default": False},
            },
        ))
        
        self.register_tool(Tool(
            name="search_files",
            description="Search for files matching a pattern",
            handler=self._search_files,
            parameters={
                "pattern": {"type": "string", "description": "Glob pattern to match"},
                "path": {"type": "string", "description": "Directory to search", "default": "."},
            },
        ))
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to working directory."""
        p = Path(path)
        if not p.is_absolute():
            p = self.working_dir / p
        return p.resolve()
    
    def _shell(self, command: str, timeout: int = 30) -> dict:
        """Execute a shell command."""
        logger.info("Executing shell command", command=command)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    def _read_file(self, path: str, encoding: str = "utf-8") -> dict:
        """Read file contents."""
        resolved = self._resolve_path(path)
        logger.info("Reading file", path=str(resolved))
        
        try:
            content = resolved.read_text(encoding=encoding)
            return {
                "success": True,
                "content": content,
                "path": str(resolved),
                "size": len(content),
            }
        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _write_file(self, path: str, content: str, encoding: str = "utf-8") -> dict:
        """Write content to file."""
        resolved = self._resolve_path(path)
        logger.info("Writing file", path=str(resolved))
        
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding=encoding)
            return {
                "success": True,
                "path": str(resolved),
                "size": len(content),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _list_dir(self, path: str = ".", recursive: bool = False) -> dict:
        """List directory contents."""
        resolved = self._resolve_path(path)
        logger.info("Listing directory", path=str(resolved))
        
        try:
            if not resolved.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}
            
            if recursive:
                items = [str(p.relative_to(resolved)) for p in resolved.rglob("*")]
            else:
                items = [p.name for p in resolved.iterdir()]
            
            return {
                "success": True,
                "path": str(resolved),
                "items": sorted(items),
                "count": len(items),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _search_files(self, pattern: str, path: str = ".") -> dict:
        """Search for files matching pattern."""
        resolved = self._resolve_path(path)
        logger.info("Searching files", pattern=pattern, path=str(resolved))
        
        try:
            matches = list(resolved.glob(pattern))
            return {
                "success": True,
                "pattern": pattern,
                "matches": [str(m.relative_to(resolved)) for m in matches],
                "count": len(matches),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
