"""
Sandboxed Python code execution tool.
"""

from __future__ import annotations

import asyncio
import sys
import io
import traceback
import structlog
from typing import Any
from contextlib import redirect_stdout, redirect_stderr

from omniagi.tools.base import Tool, ToolResult, ToolResultStatus
from omniagi.core.config import get_config

logger = structlog.get_logger()


# Imports that are always blocked for security
ALWAYS_BLOCKED = {
    "os.system",
    "os.popen",
    "os.spawn",
    "os.exec",
    "subprocess",
    "shutil.rmtree",
    "shutil.move",
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",  # Use built-in open with restrictions
}


class RestrictedBuiltins:
    """Provide restricted builtins for sandboxed execution."""
    
    def __init__(self, allowed_paths: list[str]):
        self.allowed_paths = [str(p) for p in allowed_paths]
        self._original_open = open
    
    def _safe_open(self, path, mode="r", *args, **kwargs):
        """Restricted open that only allows reading from allowed paths."""
        from pathlib import Path
        
        resolved = Path(path).resolve()
        
        # Only allow read mode
        if "w" in mode or "a" in mode or "x" in mode:
            raise PermissionError(f"Write access denied: {path}")
        
        # Check allowed paths
        for allowed in self.allowed_paths:
            try:
                resolved.relative_to(Path(allowed).resolve())
                return self._original_open(path, mode, *args, **kwargs)
            except ValueError:
                continue
        
        raise PermissionError(f"Access denied: {path}")
    
    def get_builtins(self) -> dict:
        """Get the restricted builtins dict."""
        import builtins
        
        restricted = dict(vars(builtins))
        
        # Replace dangerous functions
        restricted["open"] = self._safe_open
        restricted["__import__"] = self._restricted_import
        restricted["eval"] = lambda *args, **kwargs: None
        restricted["exec"] = lambda *args, **kwargs: None
        restricted["compile"] = lambda *args, **kwargs: None
        
        return restricted
    
    def _restricted_import(self, name, *args, **kwargs):
        """Restricted import that blocks dangerous modules."""
        blocked_prefixes = ["os", "subprocess", "shutil", "socket", "ctypes"]
        
        for prefix in blocked_prefixes:
            if name == prefix or name.startswith(f"{prefix}."):
                raise ImportError(f"Import of '{name}' is not allowed")
        
        return __builtins__["__import__"](name, *args, **kwargs)


class PythonExecutorTool(Tool):
    """
    Tool for executing Python code in a sandboxed environment.
    
    Features:
    - Timeout enforcement
    - Restricted builtins
    - Blocked dangerous imports
    - Output capture
    """
    
    @property
    def name(self) -> str:
        return "python_executor"
    
    @property
    def description(self) -> str:
        return (
            "Execute Python code in a sandboxed environment. "
            "Returns stdout output. Limited to safe operations."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "code": {
                "description": "Python code to execute",
                "type": "string",
                "required": True,
            },
            "timeout": {
                "description": "Maximum execution time in seconds (default: 30)",
                "type": "integer",
                "required": False,
            },
        }
    
    async def execute(self, code: str, timeout: int | None = None) -> ToolResult:
        config = get_config()
        timeout = timeout or config.security.execution_timeout
        
        if not config.security.sandbox_enabled:
            return ToolResult.error("Sandbox is disabled in configuration")
        
        logger.info("Executing Python code", code_length=len(code), timeout=timeout)
        
        # Run in a separate thread to allow timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._execute_sync, code, config),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("Code execution timed out", timeout=timeout)
            return ToolResult(
                status=ToolResultStatus.TIMEOUT,
                output=None,
                error=f"Execution timed out after {timeout} seconds",
            )
    
    def _execute_sync(self, code: str, config) -> ToolResult:
        """Execute code synchronously in a restricted environment."""
        
        # Prepare restricted environment
        restricted = RestrictedBuiltins(config.security.allowed_paths)
        
        # Global namespace for execution
        globals_dict = {
            "__builtins__": restricted.get_builtins(),
            "__name__": "__sandbox__",
        }
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, globals_dict)
            
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            output = stdout
            if stderr:
                output += f"\n[stderr]\n{stderr}"
            
            return ToolResult.success(output.strip() or "(no output)")
            
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error("Code execution failed", error=str(e))
            return ToolResult.error(f"{type(e).__name__}: {e}\n{error_msg}")
