"""
Critic Agent - Self-critique and code evaluation.

Evaluates generated code and provides structured feedback
for improvement iterations.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

from omniagi.ouroboros.analyzer import CodeAnalyzer, AnalysisResult

logger = structlog.get_logger()


@dataclass
class CritiqueResult:
    """Result of code critique."""
    
    score: float  # 0.0 - 1.0
    passed: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    analysis: AnalysisResult | None = None
    reasoning: str = ""


CRITIQUE_PROMPT = '''Você é um revisor de código expert. Analise o seguinte código e forneça uma crítica estruturada.

## Código para Revisar:
```python
{code}
```

## Contexto:
{context}

## Análise Estática:
- Complexidade Ciclomática: {complexity}
- Índice de Manutenibilidade: {maintainability}
- Issues Detectados: {num_issues}

## Issues Automáticos:
{issues}

## Tarefa:
Avalie o código nos seguintes critérios (0-10 cada):
1. **Correção**: O código faz o que deveria?
2. **Clareza**: O código é fácil de entender?
3. **Eficiência**: O código é performático?
4. **Segurança**: O código tem vulnerabilidades?
5. **Testabilidade**: O código é fácil de testar?

Responda no formato:
CORREÇÃO: X/10 - [justificativa]
CLAREZA: X/10 - [justificativa]
EFICIÊNCIA: X/10 - [justificativa]
SEGURANÇA: X/10 - [justificativa]
TESTABILIDADE: X/10 - [justificativa]

SCORE_FINAL: X.X
PASSOU: SIM/NÃO

SUGESTÕES:
1. [sugestão específica]
2. [sugestão específica]
...
'''


class CriticAgent:
    """
    Agent that critiques code and provides improvement feedback.
    
    Uses a combination of:
    - Static analysis (CodeAnalyzer)
    - LLM-based review
    - Best practices validation
    """
    
    def __init__(
        self,
        engine: "Engine | None" = None,
        analyzer: CodeAnalyzer | None = None,
        min_score: float = 0.7,
    ):
        """
        Initialize the critic agent.
        
        Args:
            engine: LLM engine for reviews.
            analyzer: Code analyzer for static analysis.
            min_score: Minimum score to pass (0.0-1.0).
        """
        self.engine = engine
        self.analyzer = analyzer or CodeAnalyzer()
        self.min_score = min_score
    
    def critique(
        self,
        code: str,
        context: str = "",
        use_llm: bool = True,
    ) -> CritiqueResult:
        """
        Critique a piece of code.
        
        Args:
            code: The code to critique.
            context: Additional context about the code.
            use_llm: Whether to use LLM for deeper analysis.
            
        Returns:
            CritiqueResult with score, issues, and suggestions.
        """
        # First, do static analysis
        issues = []
        suggestions = []
        
        try:
            # Parse and analyze the code
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            analysis = self.analyzer.analyze_file(temp_path)
            
            import os
            os.unlink(temp_path)
            
            # Extract issues
            for issue in analysis.issues:
                issues.append(f"[{issue.severity.upper()}] {issue.message}")
                if issue.suggestion:
                    suggestions.append(issue.suggestion)
            
        except Exception as e:
            logger.warning("Static analysis failed", error=str(e))
            analysis = None
        
        # Calculate base score from static analysis
        base_score = 1.0
        if analysis:
            # Penalize based on issues
            for issue in analysis.issues:
                if issue.severity == "critical":
                    base_score -= 0.3
                elif issue.severity == "high":
                    base_score -= 0.15
                elif issue.severity == "medium":
                    base_score -= 0.05
                else:
                    base_score -= 0.02
            
            # Penalize based on complexity
            if analysis.metrics.cyclomatic_complexity > 15:
                base_score -= 0.1
            elif analysis.metrics.cyclomatic_complexity > 10:
                base_score -= 0.05
            
            # Bonus for good maintainability
            if analysis.metrics.maintainability_index > 80:
                base_score += 0.1
            
            base_score = max(0.0, min(1.0, base_score))
        
        # LLM-based critique
        llm_score = None
        reasoning = ""
        
        if use_llm and self.engine and self.engine.is_loaded:
            try:
                llm_result = self._llm_critique(code, context, analysis)
                llm_score = llm_result.get("score", base_score)
                reasoning = llm_result.get("reasoning", "")
                suggestions.extend(llm_result.get("suggestions", []))
            except Exception as e:
                logger.warning("LLM critique failed", error=str(e))
        
        # Combine scores
        if llm_score is not None:
            final_score = (base_score + llm_score) / 2
        else:
            final_score = base_score
        
        passed = final_score >= self.min_score
        
        logger.info(
            "Code critique complete",
            score=round(final_score, 2),
            passed=passed,
            num_issues=len(issues),
        )
        
        return CritiqueResult(
            score=round(final_score, 2),
            passed=passed,
            issues=issues,
            suggestions=list(set(suggestions)),  # Remove duplicates
            analysis=analysis,
            reasoning=reasoning,
        )
    
    def _llm_critique(
        self,
        code: str,
        context: str,
        analysis: AnalysisResult | None,
    ) -> dict[str, Any]:
        """Use LLM for deeper code analysis."""
        issues_text = ""
        complexity = 0
        maintainability = 0
        num_issues = 0
        
        if analysis:
            complexity = analysis.metrics.cyclomatic_complexity
            maintainability = analysis.metrics.maintainability_index
            num_issues = len(analysis.issues)
            issues_text = "\n".join(
                f"- [{i.severity}] L{i.line}: {i.message}"
                for i in analysis.issues[:10]  # Limit to 10
            )
        
        prompt = CRITIQUE_PROMPT.format(
            code=code[:3000],  # Limit code size
            context=context or "Nenhum contexto adicional fornecido.",
            complexity=complexity,
            maintainability=maintainability,
            num_issues=num_issues,
            issues=issues_text or "Nenhum issue automático detectado.",
        )
        
        from omniagi.core.engine import GenerationConfig
        
        response = self.engine.generate(
            prompt,
            GenerationConfig(max_tokens=1024, temperature=0.3),
        )
        
        return self._parse_critique_response(response.text)
    
    def _parse_critique_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM critique response."""
        result = {
            "score": 0.7,
            "reasoning": response,
            "suggestions": [],
        }
        
        lines = response.split("\n")
        
        scores = []
        in_suggestions = False
        
        for line in lines:
            line = line.strip()
            
            # Extract scores
            for criterion in ["CORREÇÃO", "CLAREZA", "EFICIÊNCIA", "SEGURANÇA", "TESTABILIDADE"]:
                if line.startswith(criterion):
                    try:
                        score_part = line.split(":")[1].split("/")[0].strip()
                        scores.append(float(score_part) / 10)
                    except (IndexError, ValueError):
                        pass
            
            # Extract final score
            if line.startswith("SCORE_FINAL"):
                try:
                    result["score"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
            
            # Extract suggestions
            if line.startswith("SUGESTÕES"):
                in_suggestions = True
                continue
            
            if in_suggestions and line.startswith(("1.", "2.", "3.", "4.", "5.", "-", "*")):
                suggestion = line.lstrip("0123456789.-*) ").strip()
                if suggestion:
                    result["suggestions"].append(suggestion)
        
        # Calculate average if we have individual scores
        if scores:
            result["score"] = sum(scores) / len(scores)
        
        return result
    
    def quick_check(self, code: str) -> bool:
        """Quick pass/fail check without full critique."""
        try:
            # Just try to parse
            import ast
            ast.parse(code)
            
            # Basic checks
            if len(code) > 10000:  # Too long
                return False
            
            if code.count("    " * 10) > 0:  # Too nested
                return False
            
            return True
        except SyntaxError:
            return False
