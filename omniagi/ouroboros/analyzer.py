"""
Code Analyzer - AST-based code analysis for self-improvement.

Analyzes Python source code to identify:
- Code smells and anti-patterns
- Complexity metrics
- Improvement opportunities
"""

from __future__ import annotations

import ast
import structlog
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = structlog.get_logger()


@dataclass
class CodeIssue:
    """An identified issue in code."""
    
    file: str
    line: int
    column: int
    issue_type: str
    severity: str  # low, medium, high, critical
    message: str
    suggestion: str | None = None
    code_snippet: str | None = None


@dataclass
class CodeMetrics:
    """Metrics for a code file or function."""
    
    lines_of_code: int = 0
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    num_functions: int = 0
    num_classes: int = 0
    max_nesting_depth: int = 0
    maintainability_index: float = 100.0


@dataclass
class AnalysisResult:
    """Result of code analysis."""
    
    file_path: str
    metrics: CodeMetrics
    issues: list[CodeIssue] = field(default_factory=list)
    ast_tree: ast.AST | None = None


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate complexity metrics."""
    
    def __init__(self):
        self.complexity = 1
        self.cognitive_complexity = 0
        self.nesting_depth = 0
        self.max_nesting = 0
        self.num_functions = 0
        self.num_classes = 0
    
    def _increase_nesting(self):
        self.nesting_depth += 1
        self.max_nesting = max(self.max_nesting, self.nesting_depth)
    
    def _decrease_nesting(self):
        self.nesting_depth -= 1
    
    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.cognitive_complexity += 1 + self.nesting_depth
        self._increase_nesting()
        self.generic_visit(node)
        self._decrease_nesting()
    
    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.cognitive_complexity += 1 + self.nesting_depth
        self._increase_nesting()
        self.generic_visit(node)
        self._decrease_nesting()
    
    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.cognitive_complexity += 1 + self.nesting_depth
        self._increase_nesting()
        self.generic_visit(node)
        self._decrease_nesting()
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.cognitive_complexity += 1 + self.nesting_depth
        self.generic_visit(node)
    
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each 'and'/'or' adds to complexity
        self.complexity += len(node.values) - 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.num_functions += 1
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.num_functions += 1
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.num_classes += 1
        self.generic_visit(node)


class CodeSmellDetector(ast.NodeVisitor):
    """Detect code smells and anti-patterns."""
    
    def __init__(self, source_lines: list[str]):
        self.source_lines = source_lines
        self.issues: list[CodeIssue] = []
        self.current_function: str | None = None
    
    def _add_issue(
        self,
        node: ast.AST,
        issue_type: str,
        severity: str,
        message: str,
        suggestion: str | None = None,
    ) -> None:
        line = getattr(node, 'lineno', 0)
        col = getattr(node, 'col_offset', 0)
        
        snippet = None
        if 0 < line <= len(self.source_lines):
            snippet = self.source_lines[line - 1].strip()
        
        self.issues.append(CodeIssue(
            file="",  # Will be set by analyzer
            line=line,
            column=col,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggestion=suggestion,
            code_snippet=snippet,
        ))
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.current_function = node.name
        
        # Check for too many arguments
        num_args = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
        if num_args > 5:
            self._add_issue(
                node, "too_many_arguments", "medium",
                f"Function '{node.name}' has {num_args} arguments (max recommended: 5)",
                "Consider grouping related arguments into a dataclass or dict"
            )
        
        # Check for long functions
        end_line = getattr(node, 'end_lineno', node.lineno)
        func_length = end_line - node.lineno
        if func_length > 50:
            self._add_issue(
                node, "long_function", "medium",
                f"Function '{node.name}' is {func_length} lines long",
                "Consider breaking into smaller functions"
            )
        
        # Check for missing docstring
        if not ast.get_docstring(node):
            self._add_issue(
                node, "missing_docstring", "low",
                f"Function '{node.name}' has no docstring",
                "Add a docstring describing the function's purpose"
            )
        
        self.generic_visit(node)
        self.current_function = None
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Check for missing class docstring
        if not ast.get_docstring(node):
            self._add_issue(
                node, "missing_docstring", "low",
                f"Class '{node.name}' has no docstring",
                "Add a docstring describing the class"
            )
        
        # Check for too many methods
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(methods) > 20:
            self._add_issue(
                node, "god_class", "high",
                f"Class '{node.name}' has {len(methods)} methods",
                "Consider splitting into smaller, focused classes"
            )
        
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try) -> None:
        # Check for bare except
        for handler in node.handlers:
            if handler.type is None:
                self._add_issue(
                    handler, "bare_except", "high",
                    "Bare 'except:' clause catches all exceptions",
                    "Specify the exception type, e.g., 'except Exception:'"
                )
        
        self.generic_visit(node)
    
    def visit_Compare(self, node: ast.Compare) -> None:
        # Check for 'is' comparison with literals
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)):
                if isinstance(comp, (ast.Constant, ast.List, ast.Dict, ast.Set)):
                    self._add_issue(
                        node, "identity_comparison", "medium",
                        "Using 'is' to compare with a literal",
                        "Use '==' for value comparison"
                    )
        
        self.generic_visit(node)


class CodeAnalyzer:
    """
    Analyzes Python source code for improvement opportunities.
    
    Features:
    - AST parsing and analysis
    - Complexity metrics
    - Code smell detection
    - Improvement suggestions
    """
    
    def __init__(self, base_path: Path | str | None = None):
        """
        Initialize analyzer.
        
        Args:
            base_path: Base path for relative file resolution.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def analyze_file(self, file_path: Path | str) -> AnalysisResult:
        """
        Analyze a single Python file.
        
        Args:
            file_path: Path to the Python file.
            
        Returns:
            AnalysisResult with metrics and issues.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        source = file_path.read_text(encoding='utf-8')
        source_lines = source.splitlines()
        
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            logger.error("Syntax error in file", file=str(file_path), error=str(e))
            return AnalysisResult(
                file_path=str(file_path),
                metrics=CodeMetrics(),
                issues=[CodeIssue(
                    file=str(file_path),
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    issue_type="syntax_error",
                    severity="critical",
                    message=str(e.msg),
                )],
            )
        
        # Calculate metrics
        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(tree)
        
        loc = len([l for l in source_lines if l.strip() and not l.strip().startswith('#')])
        
        # Calculate maintainability index (simplified Halstead-based)
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        import math
        cc = complexity_visitor.complexity
        mi = max(0, 171 - 5.2 * math.log(max(1, loc)) - 0.23 * cc - 16.2 * math.log(max(1, loc)))
        mi = min(100, mi * 100 / 171)  # Normalize to 0-100
        
        metrics = CodeMetrics(
            lines_of_code=loc,
            cyclomatic_complexity=complexity_visitor.complexity,
            cognitive_complexity=complexity_visitor.cognitive_complexity,
            num_functions=complexity_visitor.num_functions,
            num_classes=complexity_visitor.num_classes,
            max_nesting_depth=complexity_visitor.max_nesting,
            maintainability_index=round(mi, 2),
        )
        
        # Detect code smells
        smell_detector = CodeSmellDetector(source_lines)
        smell_detector.visit(tree)
        
        # Set file path in issues
        for issue in smell_detector.issues:
            issue.file = str(file_path)
        
        # Add complexity-based issues
        if metrics.cyclomatic_complexity > 10:
            smell_detector.issues.append(CodeIssue(
                file=str(file_path),
                line=1,
                column=0,
                issue_type="high_complexity",
                severity="high",
                message=f"File has cyclomatic complexity of {metrics.cyclomatic_complexity}",
                suggestion="Consider breaking down complex logic into smaller functions",
            ))
        
        if metrics.max_nesting_depth > 4:
            smell_detector.issues.append(CodeIssue(
                file=str(file_path),
                line=1,
                column=0,
                issue_type="deep_nesting",
                severity="medium",
                message=f"Maximum nesting depth is {metrics.max_nesting_depth}",
                suggestion="Use early returns or extract nested logic into functions",
            ))
        
        logger.info(
            "File analyzed",
            file=str(file_path),
            loc=loc,
            complexity=metrics.cyclomatic_complexity,
            issues=len(smell_detector.issues),
        )
        
        return AnalysisResult(
            file_path=str(file_path),
            metrics=metrics,
            issues=smell_detector.issues,
            ast_tree=tree,
        )
    
    def analyze_directory(
        self,
        directory: Path | str,
        pattern: str = "**/*.py",
        exclude: list[str] | None = None,
    ) -> list[AnalysisResult]:
        """
        Analyze all Python files in a directory.
        
        Args:
            directory: Directory to analyze.
            pattern: Glob pattern for files.
            exclude: Patterns to exclude.
            
        Returns:
            List of AnalysisResult for each file.
        """
        directory = Path(directory)
        exclude = exclude or ["**/test_*", "**/__pycache__/*", "**/.*"]
        
        results = []
        for file_path in directory.glob(pattern):
            # Check exclusions
            if any(file_path.match(ex) for ex in exclude):
                continue
            
            try:
                result = self.analyze_file(file_path)
                results.append(result)
            except Exception as e:
                logger.warning("Failed to analyze file", file=str(file_path), error=str(e))
        
        return results
    
    def get_summary(self, results: list[AnalysisResult]) -> dict[str, Any]:
        """Get summary statistics for multiple analysis results."""
        total_loc = sum(r.metrics.lines_of_code for r in results)
        total_issues = sum(len(r.issues) for r in results)
        avg_complexity = sum(r.metrics.cyclomatic_complexity for r in results) / max(1, len(results))
        avg_maintainability = sum(r.metrics.maintainability_index for r in results) / max(1, len(results))
        
        issues_by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for result in results:
            for issue in result.issues:
                issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1
        
        return {
            "files_analyzed": len(results),
            "total_lines_of_code": total_loc,
            "total_issues": total_issues,
            "average_complexity": round(avg_complexity, 2),
            "average_maintainability": round(avg_maintainability, 2),
            "issues_by_severity": issues_by_severity,
        }
