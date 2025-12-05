"""
Filesystem tools for reading, writing, and managing files.
"""

from __future__ import annotations

import os
import aiofiles
import structlog
from pathlib import Path
from typing import Any

from omniagi.tools.base import Tool, ToolResult
from omniagi.core.config import get_config

logger = structlog.get_logger()


def _is_path_allowed(path: Path) -> bool:
    """Check if a path is within allowed directories."""
    config = get_config()
    path = path.resolve()
    
    for allowed in config.security.allowed_paths:
        allowed_path = Path(allowed).resolve()
        try:
            path.relative_to(allowed_path)
            return True
        except ValueError:
            continue
    
    return False


class ReadFileTool(Tool):
    """Tool for reading file contents."""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read the contents of a file at the given path."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "path": {
                "description": "Path to the file to read",
                "type": "string",
                "required": True,
            },
            "encoding": {
                "description": "File encoding (default: utf-8)",
                "type": "string",
                "required": False,
            },
        }
    
    async def execute(self, path: str, encoding: str = "utf-8") -> ToolResult:
        try:
            file_path = Path(path).resolve()
            
            if not _is_path_allowed(file_path):
                return ToolResult.error(f"Access denied: {path}")
            
            if not file_path.exists():
                return ToolResult.error(f"File not found: {path}")
            
            if not file_path.is_file():
                return ToolResult.error(f"Not a file: {path}")
            
            async with aiofiles.open(file_path, "r", encoding=encoding) as f:
                content = await f.read()
            
            logger.info("File read", path=str(file_path), size=len(content))
            return ToolResult.success(content, path=str(file_path))
            
        except Exception as e:
            logger.error("Failed to read file", path=path, error=str(e))
            return ToolResult.error(str(e))


class WriteFileTool(Tool):
    """Tool for writing content to files."""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Write content to a file. Creates the file and parent directories if they don't exist."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "path": {
                "description": "Path to the file to write",
                "type": "string",
                "required": True,
            },
            "content": {
                "description": "Content to write to the file",
                "type": "string",
                "required": True,
            },
            "encoding": {
                "description": "File encoding (default: utf-8)",
                "type": "string",
                "required": False,
            },
        }
    
    async def execute(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> ToolResult:
        try:
            file_path = Path(path).resolve()
            
            if not _is_path_allowed(file_path):
                return ToolResult.error(f"Access denied: {path}")
            
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(file_path, "w", encoding=encoding) as f:
                await f.write(content)
            
            logger.info("File written", path=str(file_path), size=len(content))
            return ToolResult.success(
                f"Successfully wrote {len(content)} bytes to {path}",
                path=str(file_path),
            )
            
        except Exception as e:
            logger.error("Failed to write file", path=path, error=str(e))
            return ToolResult.error(str(e))


class ListDirectoryTool(Tool):
    """Tool for listing directory contents."""
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return "List the contents of a directory."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "path": {
                "description": "Path to the directory to list",
                "type": "string",
                "required": True,
            },
            "recursive": {
                "description": "Whether to list recursively (default: false)",
                "type": "boolean",
                "required": False,
            },
        }
    
    async def execute(self, path: str, recursive: bool = False) -> ToolResult:
        try:
            dir_path = Path(path).resolve()
            
            if not _is_path_allowed(dir_path):
                return ToolResult.error(f"Access denied: {path}")
            
            if not dir_path.exists():
                return ToolResult.error(f"Directory not found: {path}")
            
            if not dir_path.is_dir():
                return ToolResult.error(f"Not a directory: {path}")
            
            entries = []
            if recursive:
                for item in dir_path.rglob("*"):
                    rel_path = item.relative_to(dir_path)
                    entry_type = "dir" if item.is_dir() else "file"
                    entries.append(f"[{entry_type}] {rel_path}")
            else:
                for item in dir_path.iterdir():
                    entry_type = "dir" if item.is_dir() else "file"
                    entries.append(f"[{entry_type}] {item.name}")
            
            result = "\n".join(sorted(entries))
            logger.info("Directory listed", path=str(dir_path), count=len(entries))
            return ToolResult.success(result, path=str(dir_path), count=len(entries))
            
        except Exception as e:
            logger.error("Failed to list directory", path=path, error=str(e))
            return ToolResult.error(str(e))


class DeleteFileTool(Tool):
    """Tool for deleting files."""
    
    @property
    def name(self) -> str:
        return "delete_file"
    
    @property
    def description(self) -> str:
        return "Delete a file at the given path. Use with caution!"
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "path": {
                "description": "Path to the file to delete",
                "type": "string",
                "required": True,
            },
        }
    
    async def execute(self, path: str) -> ToolResult:
        try:
            file_path = Path(path).resolve()
            
            if not _is_path_allowed(file_path):
                return ToolResult.error(f"Access denied: {path}")
            
            if not file_path.exists():
                return ToolResult.error(f"File not found: {path}")
            
            if not file_path.is_file():
                return ToolResult.error(f"Not a file: {path}")
            
            file_path.unlink()
            
            logger.info("File deleted", path=str(file_path))
            return ToolResult.success(f"Successfully deleted {path}")
            
        except Exception as e:
            logger.error("Failed to delete file", path=path, error=str(e))
            return ToolResult.error(str(e))
