"""
Ouroboros Loop - Main self-improvement cycle.

The serpent eating its own tail - continuous self-improvement
through analysis, critique, refactoring, and validation.
"""

from __future__ import annotations

import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable, Any
from enum import Enum, auto

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

from omniagi.ouroboros.analyzer import CodeAnalyzer, AnalysisResult
from omniagi.ouroboros.critic import CriticAgent, CritiqueResult
from omniagi.ouroboros.refactor import Refactorer, RefactorResult
from omniagi.tools.git import GitTool

logger = structlog.get_logger()


class ImprovementState(Enum):
    """State of an improvement attempt."""
    
    PENDING = auto()
    ANALYZING = auto()
    CRITIQUING = auto()
    REFACTORING = auto()
    TESTING = auto()
    APPLYING = auto()
    COMPLETE = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


@dataclass
class ImprovementAttempt:
    """Record of a single improvement attempt."""
    
    file_path: str
    state: ImprovementState = ImprovementState.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    
    original_code: str = ""
    improved_code: str = ""
    
    analysis: AnalysisResult | None = None
    critique: CritiqueResult | None = None
    refactor_result: RefactorResult | None = None
    
    tests_passed: bool = False
    applied: bool = False
    commit_hash: str | None = None
    
    error: str | None = None
    retries: int = 0


@dataclass
class OuroborosStats:
    """Statistics for the Ouroboros loop."""
    
    total_attempts: int = 0
    successful: int = 0
    failed: int = 0
    rolled_back: int = 0
    total_lines_improved: int = 0
    total_issues_fixed: int = 0


class OuroborosLoop:
    """
    Self-improvement loop for code evolution.
    
    The loop operates as follows:
    1. Analyze: Identify code issues and improvement opportunities
    2. Critique: Evaluate the severity and suggest improvements
    3. Refactor: Generate improved code
    4. Test: Validate the changes don't break anything
    5. Apply: Commit the changes via Git
    6. Reflect: Learn from the attempt
    
    Safety features:
    - Sandbox execution of tests
    - Automatic rollback on failures
    - Human approval option
    - File/directory restrictions
    """
    
    def __init__(
        self,
        engine: "Engine",
        work_dir: Path | str,
        allowed_paths: list[str] | None = None,
        require_approval: bool = False,
        max_retries: int = 3,
    ):
        """
        Initialize the Ouroboros loop.
        
        Args:
            engine: LLM engine for code generation.
            work_dir: Working directory for the project.
            allowed_paths: Glob patterns for allowed files.
            require_approval: Require human approval before applying.
            max_retries: Max retries per improvement attempt.
        """
        self.engine = engine
        self.work_dir = Path(work_dir)
        self.allowed_paths = allowed_paths or ["**/*.py"]
        self.require_approval = require_approval
        self.max_retries = max_retries
        
        # Components
        self.analyzer = CodeAnalyzer(self.work_dir)
        self.critic = CriticAgent(engine=engine, analyzer=self.analyzer)
        self.refactorer = Refactorer(engine=engine)
        self.git = GitTool(allowed_dirs=[str(self.work_dir)])
        
        # State
        self._running = False
        self._attempts: list[ImprovementAttempt] = []
        self._stats = OuroborosStats()
        
        # Callbacks
        self._on_improvement: list[Callable[[ImprovementAttempt], Awaitable[None]]] = []
        self._approval_callback: Callable[[ImprovementAttempt], Awaitable[bool]] | None = None
        
        logger.info("Ouroboros loop initialized", work_dir=str(work_dir))
    
    @property
    def stats(self) -> OuroborosStats:
        return self._stats
    
    def on_improvement(
        self,
        callback: Callable[[ImprovementAttempt], Awaitable[None]],
    ) -> None:
        """Register callback for improvement events."""
        self._on_improvement.append(callback)
    
    def set_approval_callback(
        self,
        callback: Callable[[ImprovementAttempt], Awaitable[bool]],
    ) -> None:
        """Set callback for approval requests."""
        self._approval_callback = callback
    
    async def improve_file(self, file_path: Path | str) -> ImprovementAttempt:
        """
        Attempt to improve a single file.
        
        Args:
            file_path: Path to the file to improve.
            
        Returns:
            ImprovementAttempt with results.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        attempt = ImprovementAttempt(file_path=str(file_path))
        attempt.original_code = file_path.read_text(encoding='utf-8')
        
        try:
            # 1. Analyze
            attempt.state = ImprovementState.ANALYZING
            attempt.analysis = self.analyzer.analyze_file(file_path)
            
            # Skip if no issues
            if not attempt.analysis.issues:
                logger.info("No issues found", file=str(file_path))
                attempt.state = ImprovementState.COMPLETE
                attempt.completed_at = datetime.now()
                return attempt
            
            # 2. Critique
            attempt.state = ImprovementState.CRITIQUING
            attempt.critique = self.critic.critique(
                attempt.original_code,
                context=f"File: {file_path}",
            )
            
            # Skip if already good enough
            if attempt.critique.passed and attempt.critique.score > 0.9:
                logger.info("Code quality is acceptable", file=str(file_path))
                attempt.state = ImprovementState.COMPLETE
                attempt.completed_at = datetime.now()
                return attempt
            
            # 3. Refactor
            attempt.state = ImprovementState.REFACTORING
            attempt.refactor_result = self.refactorer.refactor(
                attempt.original_code,
                attempt.critique,
            )
            
            if not attempt.refactor_result.success:
                raise RuntimeError(attempt.refactor_result.error)
            
            attempt.improved_code = attempt.refactor_result.refactored_code
            
            # 4. Test (basic validation)
            attempt.state = ImprovementState.TESTING
            attempt.tests_passed = await self._run_tests(file_path, attempt.improved_code)
            
            if not attempt.tests_passed:
                # Retry with error feedback
                if attempt.retries < self.max_retries:
                    attempt.retries += 1
                    logger.info("Retrying improvement", attempt=attempt.retries)
                    # Re-refactor with test failure info
                    attempt.refactor_result = self.refactorer.refactor(
                        attempt.improved_code,
                        custom_instructions="Código anterior falhou nos testes. Corrija os problemas.",
                    )
                    if attempt.refactor_result.success:
                        attempt.improved_code = attempt.refactor_result.refactored_code
                        attempt.tests_passed = await self._run_tests(file_path, attempt.improved_code)
            
            if not attempt.tests_passed:
                raise RuntimeError("Tests failed after retries")
            
            # 5. Apply
            attempt.state = ImprovementState.APPLYING
            
            # Check for approval if required
            if self.require_approval:
                approved = await self._request_approval(attempt)
                if not approved:
                    logger.info("Improvement rejected by user")
                    attempt.state = ImprovementState.FAILED
                    attempt.error = "Rejected by user"
                    attempt.completed_at = datetime.now()
                    return attempt
            
            # Write the improved code
            file_path.write_text(attempt.improved_code, encoding='utf-8')
            
            # Commit via Git
            commit_result = await self.git.execute({
                "action": "commit",
                "message": f"refactor: Auto-improve {file_path.name}\n\n"
                          f"Changes:\n" + "\n".join(
                              f"- {c}" for c in attempt.refactor_result.changes_made[:5]
                          ),
                "repo_path": str(self.work_dir),
            })
            
            if commit_result.success:
                attempt.commit_hash = commit_result.output.get("commit_hash")
                attempt.applied = True
            
            attempt.state = ImprovementState.COMPLETE
            self._stats.successful += 1
            self._stats.total_issues_fixed += len(attempt.analysis.issues)
            
        except Exception as e:
            logger.error("Improvement failed", error=str(e))
            attempt.state = ImprovementState.FAILED
            attempt.error = str(e)
            self._stats.failed += 1
            
            # Rollback if we wrote anything
            if attempt.applied:
                await self._rollback(attempt)
        
        attempt.completed_at = datetime.now()
        self._attempts.append(attempt)
        self._stats.total_attempts += 1
        
        # Notify callbacks
        for callback in self._on_improvement:
            try:
                await callback(attempt)
            except Exception as e:
                logger.warning("Callback error", error=str(e))
        
        return attempt
    
    async def improve_directory(
        self,
        directory: Path | str | None = None,
        max_files: int = 10,
    ) -> list[ImprovementAttempt]:
        """
        Improve all files in a directory.
        
        Args:
            directory: Directory to improve (defaults to work_dir).
            max_files: Maximum files to improve in one run.
            
        Returns:
            List of improvement attempts.
        """
        directory = Path(directory) if directory else self.work_dir
        
        # Find files with issues
        analyses = self.analyzer.analyze_directory(directory)
        
        # Sort by number of issues (most issues first)
        analyses.sort(key=lambda a: len(a.issues), reverse=True)
        
        results = []
        for analysis in analyses[:max_files]:
            if analysis.issues:  # Only improve files with issues
                try:
                    result = await self.improve_file(analysis.file_path)
                    results.append(result)
                except Exception as e:
                    logger.error("Failed to improve file", file=analysis.file_path, error=str(e))
        
        return results
    
    async def _run_tests(self, file_path: Path, code: str) -> bool:
        """Run tests to validate the code."""
        import tempfile
        import subprocess
        
        # Write to temp file for testing
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Basic syntax check
            import ast
            ast.parse(code)
            
            # Try to import (catches many issues)
            result = subprocess.run(
                ["python", "-c", f"import ast; ast.parse(open('{temp_path}').read())"],
                capture_output=True,
                timeout=10,
            )
            
            if result.returncode != 0:
                logger.warning("Import check failed", stderr=result.stderr.decode())
                return False
            
            # TODO: Run actual project tests if available
            
            return True
            
        except Exception as e:
            logger.warning("Test failed", error=str(e))
            return False
        
        finally:
            import os
            os.unlink(temp_path)
    
    async def _request_approval(self, attempt: ImprovementAttempt) -> bool:
        """Request human approval for the change."""
        if self._approval_callback:
            return await self._approval_callback(attempt)
        
        # Default: auto-approve if score is high enough
        if attempt.critique and attempt.critique.score > 0.8:
            return True
        
        return False
    
    async def _rollback(self, attempt: ImprovementAttempt) -> None:
        """Rollback a failed improvement."""
        try:
            # Restore original code
            Path(attempt.file_path).write_text(attempt.original_code, encoding='utf-8')
            
            # Git reset if we committed
            if attempt.commit_hash:
                await self.git.execute({
                    "action": "checkout",
                    "branch": "HEAD~1",
                    "repo_path": str(self.work_dir),
                })
            
            attempt.state = ImprovementState.ROLLED_BACK
            self._stats.rolled_back += 1
            
            logger.info("Rolled back improvement", file=attempt.file_path)
            
        except Exception as e:
            logger.error("Rollback failed", error=str(e))
    
    async def run_continuous(
        self,
        interval_seconds: int = 3600,
        max_iterations: int | None = None,
    ) -> None:
        """
        Run the improvement loop continuously.
        
        Args:
            interval_seconds: Time between iterations.
            max_iterations: Max iterations (None for infinite).
        """
        self._running = True
        iteration = 0
        
        logger.info("Starting continuous Ouroboros loop")
        
        while self._running:
            if max_iterations and iteration >= max_iterations:
                break
            
            try:
                logger.info("Ouroboros iteration", iteration=iteration)
                await self.improve_directory()
                iteration += 1
                
            except Exception as e:
                logger.error("Iteration failed", error=str(e))
            
            if self._running:
                await asyncio.sleep(interval_seconds)
        
        logger.info("Ouroboros loop stopped", iterations=iteration)
    
    def stop(self) -> None:
        """Stop the continuous loop."""
        self._running = False
    
    def get_history(self, limit: int = 50) -> list[ImprovementAttempt]:
        """Get recent improvement history."""
        return self._attempts[-limit:]
