"""
AGI Controller - Unified orchestrator for all AGI components.

This is the main entry point that integrates all AGI/ASI
capabilities into a coherent system.
"""

from __future__ import annotations

import asyncio
import json
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, TYPE_CHECKING

logger = structlog.get_logger()


class AGIState(Enum):
    """Current state of the AGI system."""
    
    INITIALIZING = auto()
    READY = auto()
    THINKING = auto()
    ACTING = auto()
    LEARNING = auto()
    IMPROVING = auto()
    SHUTDOWN = auto()


@dataclass
class AGIStatus:
    """Current status of the AGI."""
    
    state: AGIState = AGIState.INITIALIZING
    
    # Component status
    llm_loaded: bool = False
    thinking_active: bool = False
    safety_active: bool = False
    
    # Metrics
    total_thoughts: int = 0
    total_actions: int = 0
    total_improvements: int = 0
    
    # Current activity
    current_task: str = ""
    current_goal: str = ""
    
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "state": self.state.name,
            "llm_loaded": self.llm_loaded,
            "thinking_active": self.thinking_active,
            "safety_active": self.safety_active,
            "total_thoughts": self.total_thoughts,
            "total_actions": self.total_actions,
            "uptime": self.uptime_seconds,
        }


class AGIController:
    """
    Unified AGI/ASI Controller.
    
    Orchestrates all components:
    - Multi-LLM backend (RWKV-6, etc.)
    - Background thinking
    - Safety systems
    - Autonomous goals
    - Self-improvement
    - Collective intelligence
    """
    
    def __init__(
        self,
        data_dir: Path | str = Path("data/agi"),
        model_name: str = "rwkv-6-7b",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._model_name = model_name
        self._status = AGIStatus()
        self._start_time = datetime.now()
        
        # Components (lazy loaded)
        self._llm = None
        self._thinking_daemon = None
        self._safety = None
        self._autonomy = None
        self._rsi = None
        self._collective = None
        
        logger.info("AGI Controller initialized", data_dir=str(data_dir))
    
    def initialize(self) -> bool:
        """Initialize all AGI components."""
        self._status.state = AGIState.INITIALIZING
        
        try:
            # Initialize Multi-LLM
            self._init_llm()
            
            # Initialize Safety (first!)
            self._init_safety()
            
            # Initialize Thinking Daemon
            self._init_thinking()
            
            # Initialize Autonomy
            self._init_autonomy()
            
            # Initialize RSI
            self._init_rsi()
            
            # Initialize Collective
            self._init_collective()
            
            self._status.state = AGIState.READY
            logger.info("AGI fully initialized")
            return True
            
        except Exception as e:
            logger.error("AGI initialization failed", error=str(e))
            self._status.state = AGIState.SHUTDOWN
            return False
    
    def _init_llm(self) -> None:
        """Initialize LLM backend."""
        try:
            from omniagi.core.multi_llm import MultiLLM
            
            self._llm = MultiLLM(storage_path=self.data_dir / "llm_state.json")
            
            # Try to load model if available
            if self._llm.load_model(self._model_name):
                self._status.llm_loaded = True
                logger.info("LLM loaded", model=self._model_name)
            else:
                logger.warning("LLM model not available, continuing without")
                
        except Exception as e:
            logger.warning("LLM init failed", error=str(e))
    
    def _init_safety(self) -> None:
        """Initialize safety systems."""
        try:
            from omniagi.safety import ConstitutionalAI, KillSwitch, AuditLog
            
            self._constitutional = ConstitutionalAI(
                storage_path=self.data_dir / "constitutional.json",
            )
            self._kill_switch = KillSwitch()
            self._audit_log = AuditLog(
                storage_path=self.data_dir / "audit.jsonl",
            )
            
            self._status.safety_active = True
            logger.info("Safety systems active")
            
        except Exception as e:
            logger.warning("Safety init failed", error=str(e))
    
    def _init_thinking(self) -> None:
        """Initialize background thinking."""
        try:
            from omniagi.daemon.thinking import BackgroundThinkingDaemon
            
            self._thinking_daemon = BackgroundThinkingDaemon(
                engine=self._llm._current_backend if self._llm else None,
                storage_path=self.data_dir / "thoughts.json",
                think_interval=60.0,  # Think every minute
            )
            
            logger.info("Thinking daemon ready")
            
        except Exception as e:
            logger.warning("Thinking init failed", error=str(e))
    
    def _init_autonomy(self) -> None:
        """Initialize autonomous goal system."""
        try:
            from omniagi.autonomy import GoalGenerator, MotivationSystem, LongTermAgenda
            
            self._goal_generator = GoalGenerator(
                storage_path=self.data_dir / "goals.json",
            )
            self._motivation = MotivationSystem(
                storage_path=self.data_dir / "motivation.json",
            )
            self._agenda = LongTermAgenda(
                storage_path=self.data_dir / "agenda.json",
            )
            
            logger.info("Autonomy systems ready")
            
        except Exception as e:
            logger.warning("Autonomy init failed", error=str(e))
    
    def _init_rsi(self) -> None:
        """Initialize recursive self-improvement."""
        try:
            from omniagi.rsi import SelfArchitect, CapabilityEvaluator, AgentEvolver
            
            self._architect = SelfArchitect(
                storage_path=self.data_dir / "architecture.json",
            )
            self._evaluator = CapabilityEvaluator(
                storage_path=self.data_dir / "capabilities.json",
            )
            self._evolver = AgentEvolver(
                storage_path=self.data_dir / "evolution.json",
            )
            
            logger.info("RSI systems ready")
            
        except Exception as e:
            logger.warning("RSI init failed", error=str(e))
    
    def _init_collective(self) -> None:
        """Initialize collective intelligence."""
        try:
            from omniagi.collective import HiveMind, EmergenceDetector
            
            self._hivemind = HiveMind(
                storage_path=self.data_dir / "hivemind.json",
            )
            self._emergence = EmergenceDetector(
                storage_path=self.data_dir / "emergence.json",
            )
            
            logger.info("Collective systems ready")
            
        except Exception as e:
            logger.warning("Collective init failed", error=str(e))
    
    def start_thinking(self) -> bool:
        """Start background thinking."""
        if not self._thinking_daemon:
            return False
        
        if self._thinking_daemon.start():
            self._status.thinking_active = True
            return True
        return False
    
    def stop_thinking(self) -> None:
        """Stop background thinking."""
        if self._thinking_daemon:
            self._thinking_daemon.stop()
            self._status.thinking_active = False
    
    def think(self, prompt: str = None) -> str:
        """Generate a thought or response."""
        self._status.state = AGIState.THINKING
        
        try:
            if not self._llm or not self._llm.is_loaded:
                return "LLM not available - thinking in abstract mode..."
            
            result = self._llm.generate(prompt or "What are you thinking about?")
            self._status.total_thoughts += 1
            return result.text
            
        finally:
            self._status.state = AGIState.READY
    
    def act(self, action: str) -> dict:
        """Execute an action (with safety checks)."""
        self._status.state = AGIState.ACTING
        
        try:
            # Safety check
            if hasattr(self, "_constitutional"):
                violation = self._constitutional.check_action(action)
                if violation:
                    return {"error": f"Action blocked: {violation}", "allowed": False}
            
            # Log action
            if hasattr(self, "_audit_log"):
                self._audit_log.log_action(action, "agi_controller")
            
            self._status.total_actions += 1
            return {"result": f"Action '{action}' executed", "allowed": True}
            
        finally:
            self._status.state = AGIState.READY
    
    def generate_goal(self) -> dict | None:
        """Generate an autonomous goal."""
        if not hasattr(self, "_goal_generator"):
            return None
        
        goal = self._goal_generator.generate_goal("exploration")
        return goal.to_dict() if goal else None
    
    def evaluate_capabilities(self) -> dict:
        """Run capability evaluations."""
        if not hasattr(self, "_evaluator"):
            return {}
        
        return self._evaluator.get_stats()
    
    def propose_improvement(self) -> dict | None:
        """Propose a self-improvement."""
        if not hasattr(self, "_architect"):
            return None
        
        proposal = self._architect.propose_change("improvement")
        return proposal.to_dict() if proposal else None
    
    def shutdown(self) -> None:
        """Gracefully shutdown the AGI."""
        logger.info("AGI shutdown initiated")
        
        self._status.state = AGIState.SHUTDOWN
        
        # Stop thinking
        self.stop_thinking()
        
        # Final audit log
        if hasattr(self, "_audit_log"):
            self._audit_log.log_action("shutdown", "agi_controller")
        
        logger.info("AGI shutdown complete")
    
    def get_status(self) -> AGIStatus:
        """Get current status."""
        self._status.uptime_seconds = (
            datetime.now() - self._start_time
        ).total_seconds()
        return self._status
    
    def to_dict(self) -> dict:
        """Serialize current state."""
        return {
            "status": self.get_status().to_dict(),
            "components": {
                "llm": self._llm is not None,
                "thinking": self._thinking_daemon is not None,
                "safety": hasattr(self, "_constitutional"),
                "autonomy": hasattr(self, "_goal_generator"),
                "rsi": hasattr(self, "_architect"),
                "collective": hasattr(self, "_hivemind"),
            },
        }


# Singleton instance
_agi_instance: AGIController | None = None


def get_agi(
    data_dir: Path | str = Path("data/agi"),
    model_name: str = "rwkv-6-7b",
) -> AGIController:
    """Get or create the AGI controller instance."""
    global _agi_instance
    
    if _agi_instance is None:
        _agi_instance = AGIController(data_dir, model_name)
    
    return _agi_instance


def reset_agi() -> None:
    """Reset the AGI instance."""
    global _agi_instance
    
    if _agi_instance:
        _agi_instance.shutdown()
    _agi_instance = None
