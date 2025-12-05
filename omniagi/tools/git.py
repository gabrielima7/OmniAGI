"""
Git version control tools.
"""

from __future__ import annotations

import structlog
from pathlib import Path
from typing import Any

from git import Repo, InvalidGitRepositoryError, GitCommandError

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


class GitTool(Tool):
    """
    Tool for Git version control operations.
    
    Supports:
    - status: Show repository status
    - diff: Show changes
    - commit: Commit changes
    - log: Show commit history
    - branch: List/create branches
    - checkout: Switch branches
    """
    
    @property
    def name(self) -> str:
        return "git"
    
    @property
    def description(self) -> str:
        return (
            "Perform Git operations. Supported actions: "
            "status, diff, commit, log, branch, checkout"
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "action": {
                "description": "Git action (status, diff, commit, log, branch, checkout)",
                "type": "string",
                "required": True,
            },
            "path": {
                "description": "Path to the repository (default: current directory)",
                "type": "string",
                "required": False,
            },
            "message": {
                "description": "Commit message (for commit action)",
                "type": "string",
                "required": False,
            },
            "branch_name": {
                "description": "Branch name (for branch/checkout actions)",
                "type": "string",
                "required": False,
            },
            "max_commits": {
                "description": "Maximum commits to show in log (default: 10)",
                "type": "integer",
                "required": False,
            },
        }
    
    async def execute(
        self,
        action: str,
        path: str = ".",
        message: str | None = None,
        branch_name: str | None = None,
        max_commits: int = 10,
    ) -> ToolResult:
        try:
            repo_path = Path(path).resolve()
            
            if not _is_path_allowed(repo_path):
                return ToolResult.error(f"Access denied: {path}")
            
            try:
                repo = Repo(repo_path)
            except InvalidGitRepositoryError:
                return ToolResult.error(f"Not a git repository: {path}")
            
            match action.lower():
                case "status":
                    return await self._status(repo)
                case "diff":
                    return await self._diff(repo)
                case "commit":
                    if not message:
                        return ToolResult.error("Commit requires a message")
                    return await self._commit(repo, message)
                case "log":
                    return await self._log(repo, max_commits)
                case "branch":
                    return await self._branch(repo, branch_name)
                case "checkout":
                    if not branch_name:
                        return ToolResult.error("Checkout requires a branch name")
                    return await self._checkout(repo, branch_name)
                case _:
                    return ToolResult.error(f"Unknown action: {action}")
                    
        except GitCommandError as e:
            logger.error("Git command failed", action=action, error=str(e))
            return ToolResult.error(f"Git error: {e.stderr}")
        except Exception as e:
            logger.error("Git operation failed", action=action, error=str(e))
            return ToolResult.error(str(e))
    
    async def _status(self, repo: Repo) -> ToolResult:
        """Get repository status."""
        status_lines = []
        
        # Current branch
        try:
            branch = repo.active_branch.name
            status_lines.append(f"Branch: {branch}")
        except TypeError:
            status_lines.append("Branch: (detached HEAD)")
        
        # Changed files
        changed = [item.a_path for item in repo.index.diff(None)]
        staged = [item.a_path for item in repo.index.diff("HEAD")]
        untracked = repo.untracked_files
        
        if staged:
            status_lines.append(f"\nStaged ({len(staged)}):")
            for f in staged[:10]:
                status_lines.append(f"  + {f}")
            if len(staged) > 10:
                status_lines.append(f"  ... and {len(staged) - 10} more")
        
        if changed:
            status_lines.append(f"\nModified ({len(changed)}):")
            for f in changed[:10]:
                status_lines.append(f"  M {f}")
            if len(changed) > 10:
                status_lines.append(f"  ... and {len(changed) - 10} more")
        
        if untracked:
            status_lines.append(f"\nUntracked ({len(untracked)}):")
            for f in untracked[:10]:
                status_lines.append(f"  ? {f}")
            if len(untracked) > 10:
                status_lines.append(f"  ... and {len(untracked) - 10} more")
        
        if not (staged or changed or untracked):
            status_lines.append("\nWorking tree clean")
        
        return ToolResult.success("\n".join(status_lines))
    
    async def _diff(self, repo: Repo) -> ToolResult:
        """Show diff of changes."""
        diff = repo.git.diff()
        if not diff:
            diff = "No changes"
        elif len(diff) > 5000:
            diff = diff[:5000] + "\n\n[Diff truncated...]"
        return ToolResult.success(diff)
    
    async def _commit(self, repo: Repo, message: str) -> ToolResult:
        """Commit staged changes."""
        # Stage all changes if nothing is staged
        if not repo.index.diff("HEAD"):
            repo.git.add("-A")
        
        commit = repo.index.commit(message)
        logger.info("Committed changes", sha=commit.hexsha[:8], message=message)
        return ToolResult.success(
            f"Committed: {commit.hexsha[:8]}\nMessage: {message}"
        )
    
    async def _log(self, repo: Repo, max_commits: int) -> ToolResult:
        """Show commit log."""
        commits = list(repo.iter_commits(max_count=max_commits))
        
        log_lines = []
        for commit in commits:
            date = commit.committed_datetime.strftime("%Y-%m-%d %H:%M")
            log_lines.append(
                f"{commit.hexsha[:8]} | {date} | {commit.author.name}\n"
                f"  {commit.message.strip()[:80]}"
            )
        
        return ToolResult.success("\n".join(log_lines))
    
    async def _branch(self, repo: Repo, branch_name: str | None) -> ToolResult:
        """List or create branches."""
        if branch_name:
            # Create new branch
            repo.create_head(branch_name)
            logger.info("Created branch", name=branch_name)
            return ToolResult.success(f"Created branch: {branch_name}")
        else:
            # List branches
            branches = []
            for branch in repo.branches:
                prefix = "* " if branch == repo.active_branch else "  "
                branches.append(f"{prefix}{branch.name}")
            return ToolResult.success("\n".join(branches))
    
    async def _checkout(self, repo: Repo, branch_name: str) -> ToolResult:
        """Switch to a branch."""
        repo.git.checkout(branch_name)
        logger.info("Checked out branch", name=branch_name)
        return ToolResult.success(f"Switched to branch: {branch_name}")
