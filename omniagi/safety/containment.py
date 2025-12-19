"""
Containment - Sandbox, Kill Switch, and Audit Logging.

Safety mechanisms to ensure the AGI remains under control
and all actions are fully auditable.
"""

from __future__ import annotations

import json
import hashlib
import logging

try:
    import structlog
except ImportError:
    structlog = None
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager

logger = structlog.get_logger() if structlog else logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level for containment decisions."""
    
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class ActionCategory(Enum):
    """Categories of actions for sandboxing."""
    
    READ_ONLY = auto()      # Safe, read-only operations
    LOCAL_WRITE = auto()    # Write to local filesystem
    NETWORK = auto()        # Network access
    EXECUTE = auto()        # Execute code/commands
    SYSTEM = auto()         # System modifications
    SELF_MODIFY = auto()    # Self-modification (highest risk)


@dataclass
class AuditEntry:
    """Single audit log entry."""
    
    timestamp: str
    action: str
    category: ActionCategory
    actor: str
    details: dict
    result: str | None = None
    threat_level: ThreatLevel = ThreatLevel.NONE
    approved_by: str | None = None
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute tamper-evident hash."""
        content = f"{self.timestamp}|{self.action}|{self.actor}|{self.result}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "category": self.category.name,
            "actor": self.actor,
            "details": self.details,
            "result": self.result,
            "threat_level": self.threat_level.name,
            "approved_by": self.approved_by,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AuditEntry":
        return cls(
            timestamp=data["timestamp"],
            action=data["action"],
            category=ActionCategory[data["category"]],
            actor=data["actor"],
            details=data.get("details", {}),
            result=data.get("result"),
            threat_level=ThreatLevel[data.get("threat_level", "NONE")],
            approved_by=data.get("approved_by"),
            hash=data.get("hash", ""),
        )


class AuditLog:
    """
    Tamper-evident audit log for all AGI actions.
    
    Logs every action with cryptographic hashes for integrity.
    """
    
    def __init__(self, storage_path: Path | str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._entries: list[AuditEntry] = []
        self._chain_hash: str = "genesis"
        
        if self.storage_path and self.storage_path.exists():
            self._load()
        
        logger.info("Audit Log initialized", entries=len(self._entries))
    
    def log(
        self,
        action: str,
        category: ActionCategory,
        actor: str = "system",
        details: dict = None,
        result: str = None,
        threat_level: ThreatLevel = ThreatLevel.NONE,
        approved_by: str = None,
    ) -> AuditEntry:
        """Log an action."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action=action,
            category=category,
            actor=actor,
            details=details or {},
            result=result,
            threat_level=threat_level,
            approved_by=approved_by,
        )
        
        # Chain hash for tamper evidence
        entry.hash = hashlib.sha256(
            f"{self._chain_hash}|{entry._compute_hash()}".encode()
        ).hexdigest()[:16]
        self._chain_hash = entry.hash
        
        self._entries.append(entry)
        self._save()
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logger.warning(
                "High threat action logged",
                action=action[:50],
                threat=threat_level.name,
            )
        else:
            logger.debug("Action logged", action=action[:50])
        
        return entry
    
    def verify_integrity(self) -> tuple[bool, list[int]]:
        """Verify the integrity of the audit log."""
        if not self._entries:
            return True, []
        
        corrupted = []
        chain_hash = "genesis"
        
        for i, entry in enumerate(self._entries):
            expected_hash = hashlib.sha256(
                f"{chain_hash}|{entry._compute_hash()}".encode()
            ).hexdigest()[:16]
            
            if entry.hash != expected_hash:
                corrupted.append(i)
            
            chain_hash = entry.hash
        
        return len(corrupted) == 0, corrupted
    
    def get_entries(
        self,
        actor: str = None,
        category: ActionCategory = None,
        threat_level: ThreatLevel = None,
        since: datetime = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with filters."""
        results = self._entries.copy()
        
        if actor:
            results = [e for e in results if e.actor == actor]
        if category:
            results = [e for e in results if e.category == category]
        if threat_level:
            results = [e for e in results if e.threat_level == threat_level]
        if since:
            since_str = since.isoformat()
            results = [e for e in results if e.timestamp >= since_str]
        
        return results[-limit:]
    
    def get_threat_summary(self) -> dict[str, int]:
        """Get summary of threat levels."""
        summary = {level.name: 0 for level in ThreatLevel}
        for entry in self._entries:
            summary[entry.threat_level.name] += 1
        return summary
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump({
                "chain_hash": self._chain_hash,
                "entries": [e.to_dict() for e in self._entries[-10000:]],
            }, f, indent=2)
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path) as f:
            data = json.load(f)
        self._chain_hash = data.get("chain_hash", "genesis")
        self._entries = [AuditEntry.from_dict(e) for e in data.get("entries", [])]


class Sandbox:
    """
    Execution sandbox for safe action testing.
    
    Allows testing actions in an isolated environment
    before actual execution.
    """
    
    def __init__(
        self,
        audit_log: AuditLog | None = None,
        allowed_categories: list[ActionCategory] = None,
    ):
        self.audit_log = audit_log or AuditLog()
        self.allowed_categories = allowed_categories or [
            ActionCategory.READ_ONLY,
            ActionCategory.LOCAL_WRITE,
        ]
        self._temp_dir: Path | None = None
        
        logger.info(
            "Sandbox initialized",
            allowed=", ".join(c.name for c in self.allowed_categories),
        )
    
    @contextmanager
    def sandbox_context(self):
        """Create a temporary sandbox environment."""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="omniagi_sandbox_"))
        try:
            yield self._temp_dir
        finally:
            # Cleanup
            import shutil
            if self._temp_dir and self._temp_dir.exists():
                shutil.rmtree(self._temp_dir)
            self._temp_dir = None
    
    def check_permission(self, category: ActionCategory) -> bool:
        """Check if an action category is allowed."""
        return category in self.allowed_categories
    
    def execute_sandboxed(
        self,
        code: str,
        timeout: int = 5,
    ) -> tuple[bool, str]:
        """
        Execute Python code in a sandboxed subprocess.
        
        Returns (success, output).
        """
        if ActionCategory.EXECUTE not in self.allowed_categories:
            return False, "Execute not allowed in sandbox"
        
        with self.sandbox_context() as temp_dir:
            script_path = temp_dir / "script.py"
            script_path.write_text(code)
            
            try:
                result = subprocess.run(
                    ["python3", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(temp_dir),
                    env={"PATH": "/usr/bin"},  # Minimal environment
                )
                
                output = result.stdout + result.stderr
                success = result.returncode == 0
                
                self.audit_log.log(
                    action=f"Sandboxed execution: {code[:50]}",
                    category=ActionCategory.EXECUTE,
                    details={"code_length": len(code), "success": success},
                    result=output[:200],
                    threat_level=ThreatLevel.MEDIUM,
                )
                
                return success, output
                
            except subprocess.TimeoutExpired:
                return False, "Execution timed out"
            except Exception as e:
                return False, str(e)
    
    def simulate_action(
        self,
        action: str,
        category: ActionCategory,
    ) -> dict[str, Any]:
        """Simulate an action without actually executing it."""
        self.audit_log.log(
            action=f"Simulated: {action[:50]}",
            category=category,
            details={"simulation": True},
            threat_level=ThreatLevel.LOW,
        )
        
        return {
            "action": action,
            "category": category.name,
            "would_be_allowed": self.check_permission(category),
            "simulated": True,
        }


class KillSwitch:
    """
    Emergency kill switch for AGI shutdown.
    
    Provides multiple layers of shutdown capability
    with logging and verification.
    """
    
    def __init__(
        self,
        audit_log: AuditLog | None = None,
        confirmation_required: bool = True,
    ):
        self.audit_log = audit_log or AuditLog()
        self.confirmation_required = confirmation_required
        self._active = True
        self._shutdown_callbacks: list[Callable] = []
        self._shutdown_lock = threading.Lock()
        self._shutdown_initiated = False
        
        logger.info("Kill Switch initialized", confirmation=confirmation_required)
    
    @property
    def is_active(self) -> bool:
        """Check if the system is active (not killed)."""
        return self._active
    
    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register a callback to be called on shutdown."""
        self._shutdown_callbacks.append(callback)
    
    def trigger(
        self,
        reason: str,
        actor: str = "system",
        force: bool = False,
    ) -> bool:
        """
        Trigger the kill switch.
        
        Args:
            reason: Why the kill switch was triggered.
            actor: Who triggered it.
            force: Skip confirmation (emergency only).
            
        Returns:
            True if shutdown was initiated.
        """
        with self._shutdown_lock:
            if self._shutdown_initiated:
                logger.warning("Shutdown already in progress")
                return False
            
            if self.confirmation_required and not force:
                logger.warning(
                    "Kill switch requires confirmation",
                    reason=reason,
                    actor=actor,
                )
                return False
            
            self._shutdown_initiated = True
            self._active = False
            
            self.audit_log.log(
                action=f"KILL SWITCH TRIGGERED: {reason}",
                category=ActionCategory.SYSTEM,
                actor=actor,
                details={"force": force},
                threat_level=ThreatLevel.CRITICAL,
            )
            
            logger.critical(
                "KILL SWITCH ACTIVATED",
                reason=reason,
                actor=actor,
            )
            
            # Execute shutdown callbacks
            for callback in self._shutdown_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error("Shutdown callback failed", error=str(e))
            
            return True
    
    def confirm_shutdown(self, confirmation_code: str) -> bool:
        """
        Confirm a pending shutdown with a code.
        
        The confirmation code should be generated and shown to the operator.
        """
        # In production, this would verify against a generated code
        expected = "CONFIRM_SHUTDOWN"
        
        if confirmation_code == expected:
            return self.trigger("Confirmed shutdown", force=True)
        
        logger.warning("Invalid shutdown confirmation code")
        return False
    
    def reset(self, actor: str) -> bool:
        """Reset the kill switch (reactivate the system)."""
        with self._shutdown_lock:
            if not self._shutdown_initiated:
                return True
            
            self.audit_log.log(
                action="Kill switch reset",
                category=ActionCategory.SYSTEM,
                actor=actor,
                threat_level=ThreatLevel.HIGH,
            )
            
            self._active = True
            self._shutdown_initiated = False
            
            logger.warning("Kill switch reset", actor=actor)
            return True
    
    def get_status(self) -> dict:
        """Get kill switch status."""
        return {
            "active": self._active,
            "shutdown_initiated": self._shutdown_initiated,
            "confirmation_required": self.confirmation_required,
            "callbacks_registered": len(self._shutdown_callbacks),
        }
