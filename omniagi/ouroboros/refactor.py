"""
Refactorer - Code improvement and refactoring.

Generates improved versions of code based on critique feedback.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omniagi.core.engine import Engine

from omniagi.ouroboros.critic import CritiqueResult

logger = structlog.get_logger()


@dataclass
class RefactorResult:
    """Result of code refactoring."""
    
    original_code: str
    refactored_code: str
    changes_made: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None


REFACTOR_PROMPT = '''Você é um engenheiro de software expert. Refatore o código abaixo para resolver os problemas identificados.

## Código Original:
```python
{original_code}
```

## Problemas Identificados:
{issues}

## Sugestões de Melhoria:
{suggestions}

## Instruções:
1. Mantenha a mesma funcionalidade
2. Melhore a legibilidade e manutenibilidade
3. Aplique as sugestões quando apropriado
4. Adicione docstrings se faltarem
5. Simplifique lógica complexa
6. Use type hints quando possível

## Responda com APENAS o código refatorado:
```python
<código refatorado aqui>
```

## Após o código, liste as mudanças feitas:
MUDANÇAS:
1. <descrição da mudança>
2. <descrição da mudança>
'''


FIX_ERROR_PROMPT = '''O código abaixo tem um erro. Corrija-o mantendo a funcionalidade original.

## Código com Erro:
```python
{code}
```

## Erro:
{error}

## Responda com APENAS o código corrigido:
```python
<código corrigido>
```
'''


class Refactorer:
    """
    Refactors code based on critique feedback.
    
    Uses LLM to generate improved versions of code
    while maintaining functionality.
    """
    
    def __init__(self, engine: "Engine | None" = None):
        """
        Initialize refactorer.
        
        Args:
            engine: LLM engine for code generation.
        """
        self.engine = engine
    
    def refactor(
        self,
        code: str,
        critique: CritiqueResult | None = None,
        custom_instructions: str = "",
    ) -> RefactorResult:
        """
        Refactor code based on critique.
        
        Args:
            code: Original code to refactor.
            critique: Critique result with issues and suggestions.
            custom_instructions: Additional refactoring instructions.
            
        Returns:
            RefactorResult with refactored code.
        """
        if not self.engine or not self.engine.is_loaded:
            logger.warning("No engine available for refactoring")
            return RefactorResult(
                original_code=code,
                refactored_code=code,
                success=False,
                error="No LLM engine available",
            )
        
        issues = ""
        suggestions = ""
        
        if critique:
            issues = "\n".join(f"- {issue}" for issue in critique.issues)
            suggestions = "\n".join(f"- {s}" for s in critique.suggestions)
        
        if custom_instructions:
            suggestions += f"\n\nInstruções Adicionais:\n{custom_instructions}"
        
        prompt = REFACTOR_PROMPT.format(
            original_code=code,
            issues=issues or "Nenhum problema crítico identificado.",
            suggestions=suggestions or "Melhorar legibilidade e adicionar docstrings.",
        )
        
        try:
            from omniagi.core.engine import GenerationConfig
            
            response = self.engine.generate(
                prompt,
                GenerationConfig(
                    max_tokens=2048,
                    temperature=0.2,  # Lower for more deterministic refactoring
                    stop=["```\n\n", "MUDANÇAS:"],
                ),
            )
            
            refactored, changes = self._parse_refactor_response(response.text, code)
            
            # Validate the refactored code
            try:
                import ast
                ast.parse(refactored)
            except SyntaxError as e:
                logger.warning("Refactored code has syntax error", error=str(e))
                # Try to fix the error
                fixed = self.fix_error(refactored, str(e))
                if fixed:
                    refactored = fixed
                else:
                    return RefactorResult(
                        original_code=code,
                        refactored_code=code,
                        success=False,
                        error=f"Refactored code has syntax error: {e}",
                    )
            
            logger.info("Code refactored", changes=len(changes))
            
            return RefactorResult(
                original_code=code,
                refactored_code=refactored,
                changes_made=changes,
                success=True,
            )
            
        except Exception as e:
            logger.error("Refactoring failed", error=str(e))
            return RefactorResult(
                original_code=code,
                refactored_code=code,
                success=False,
                error=str(e),
            )
    
    def fix_error(self, code: str, error: str) -> str | None:
        """
        Try to fix an error in code.
        
        Args:
            code: Code with error.
            error: Error message.
            
        Returns:
            Fixed code or None if fix failed.
        """
        if not self.engine or not self.engine.is_loaded:
            return None
        
        prompt = FIX_ERROR_PROMPT.format(code=code, error=error)
        
        try:
            from omniagi.core.engine import GenerationConfig
            
            response = self.engine.generate(
                prompt,
                GenerationConfig(max_tokens=2048, temperature=0.1),
            )
            
            fixed = self._extract_code_block(response.text)
            
            # Verify fix
            import ast
            ast.parse(fixed)
            
            return fixed
            
        except Exception as e:
            logger.warning("Failed to fix error", error=str(e))
            return None
    
    def _parse_refactor_response(
        self,
        response: str,
        original: str,
    ) -> tuple[str, list[str]]:
        """Parse the refactoring response."""
        refactored = self._extract_code_block(response)
        
        if not refactored:
            refactored = original
        
        # Extract changes
        changes = []
        lines = response.split("\n")
        in_changes = False
        
        for line in lines:
            if "MUDANÇAS" in line or "CHANGES" in line:
                in_changes = True
                continue
            
            if in_changes and line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-")):
                change = line.strip().lstrip("0123456789.-) ").strip()
                if change:
                    changes.append(change)
        
        return refactored, changes
    
    def _extract_code_block(self, text: str) -> str:
        """Extract Python code from markdown code block."""
        import re
        
        # Try to find ```python ... ``` block
        pattern = r"```python\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Try plain ``` block
        pattern = r"```\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return ""
    
    def improve_incrementally(
        self,
        code: str,
        improvements: list[str],
    ) -> RefactorResult:
        """
        Apply improvements one at a time for more control.
        
        Args:
            code: Original code.
            improvements: List of specific improvements to make.
            
        Returns:
            RefactorResult with all improvements applied.
        """
        current_code = code
        all_changes = []
        
        for improvement in improvements:
            result = self.refactor(
                current_code,
                custom_instructions=improvement,
            )
            
            if result.success:
                current_code = result.refactored_code
                all_changes.extend(result.changes_made)
            else:
                logger.warning("Improvement failed", improvement=improvement)
        
        return RefactorResult(
            original_code=code,
            refactored_code=current_code,
            changes_made=all_changes,
            success=current_code != code,
        )
