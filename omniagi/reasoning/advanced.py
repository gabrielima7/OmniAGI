"""
Advanced Reasoning Module.

Implements state-of-the-art reasoning techniques:
1. RAG (Retrieval-Augmented Generation)
2. Chain-of-Thought (CoT)
3. Self-Critique / Reflexion
4. Tool Use
5. Multi-Response Voting (Self-Consistency)
6. Hybrid ARC Solver
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result of advanced reasoning."""
    answer: str
    confidence: float
    reasoning_steps: List[str]
    method_used: str
    tools_called: List[str] = field(default_factory=list)
    critiques: List[str] = field(default_factory=list)
    context_used: List[str] = field(default_factory=list)


class ChainOfThought:
    """
    Chain-of-Thought reasoning.
    
    Makes the model think step-by-step for better accuracy.
    """
    
    def __init__(self, llm_func: Callable[[str, int], str]):
        self.llm = llm_func
    
    def reason(self, question: str, max_tokens: int = 500) -> ReasoningResult:
        """Apply chain-of-thought reasoning."""
        prompt = f"""Think through this step by step.

Question: {question}

Let me solve this carefully:

Step 1: First, I need to understand what's being asked.
"""
        
        response = self.llm(prompt, max_tokens)
        
        # Extract steps
        steps = self._extract_steps(response)
        answer = self._extract_answer(response)
        
        return ReasoningResult(
            answer=answer,
            confidence=0.8,
            reasoning_steps=steps,
            method_used="chain_of_thought",
        )
    
    def _extract_steps(self, response: str) -> List[str]:
        """Extract reasoning steps."""
        steps = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Step') or line.startswith('1.') or line.startswith('2.'):
                steps.append(line)
        return steps if steps else [response[:200]]
    
    def _extract_answer(self, response: str) -> str:
        """Extract final answer."""
        # Look for explicit answer markers
        markers = ['answer is', 'therefore', 'so,', 'finally,', 'conclusion:']
        lower = response.lower()
        
        for marker in markers:
            if marker in lower:
                idx = lower.index(marker)
                return response[idx:idx+200].strip()
        
        # Return last sentence
        sentences = response.split('.')
        return sentences[-1].strip() if sentences else response


class SelfCritique:
    """
    Self-Critique / Reflexion.
    
    The model evaluates and improves its own answers.
    """
    
    def __init__(self, llm_func: Callable[[str, int], str]):
        self.llm = llm_func
        self.max_iterations = 3
    
    def reason(self, question: str, max_tokens: int = 300) -> ReasoningResult:
        """Apply self-critique reasoning."""
        critiques = []
        
        # Initial answer
        answer = self.llm(f"Question: {question}\nAnswer:", max_tokens)
        
        for i in range(self.max_iterations):
            # Critique the answer
            critique_prompt = f"""Question: {question}
Answer given: {answer}

Is this answer correct and complete? 
If there are any errors or missing information, explain what's wrong.
If it's correct, say "CORRECT".

Evaluation:"""
            
            critique = self.llm(critique_prompt, 200)
            critiques.append(critique)
            
            # Check if answer is good
            if "CORRECT" in critique.upper() or "correct" in critique.lower()[:50]:
                break
            
            # Improve the answer
            improve_prompt = f"""Question: {question}
Previous answer: {answer}
Critique: {critique}

Provide an improved answer that addresses the critique:"""
            
            answer = self.llm(improve_prompt, max_tokens)
        
        return ReasoningResult(
            answer=answer,
            confidence=0.7 + (0.1 * len(critiques)),
            reasoning_steps=[f"Iteration {i+1}: {c[:100]}" for i, c in enumerate(critiques)],
            method_used="self_critique",
            critiques=critiques,
        )


class SelfConsistency:
    """
    Self-Consistency Decoding.
    
    Generates multiple answers and votes for the best one.
    """
    
    def __init__(self, llm_func: Callable[[str, int], str], num_samples: int = 5):
        self.llm = llm_func
        self.num_samples = num_samples
    
    def reason(self, question: str, max_tokens: int = 200) -> ReasoningResult:
        """Apply self-consistency reasoning."""
        responses = []
        
        # Generate multiple responses
        for i in range(self.num_samples):
            prompt = f"""Question: {question}

Think step by step and give a clear answer.

Answer:"""
            response = self.llm(prompt, max_tokens)
            responses.append(response)
        
        # Extract answers and vote
        extracted = [self._extract_key(r) for r in responses]
        counter = Counter(extracted)
        best_answer, count = counter.most_common(1)[0]
        
        # Find full response for best answer
        best_full = responses[extracted.index(best_answer)]
        
        confidence = count / self.num_samples
        
        return ReasoningResult(
            answer=best_full,
            confidence=confidence,
            reasoning_steps=[f"Sample {i+1}: {r[:50]}..." for i, r in enumerate(responses)],
            method_used="self_consistency",
        )
    
    def _extract_key(self, response: str) -> str:
        """Extract key answer for comparison."""
        # Get first sentence or first 50 chars
        first_sentence = response.split('.')[0].strip()
        return first_sentence[:50].lower()


class ToolUse:
    """
    Tool Use / Function Calling.
    
    Enables the AGI to use external tools for better accuracy.
    """
    
    def __init__(self, llm_func: Callable[[str, int], str]):
        self.llm = llm_func
        self.tools: Dict[str, Callable] = {
            "calculator": self._calculator,
            "search_memory": self._search_memory,
            "get_date": self._get_date,
            "python_eval": self._python_eval,
        }
        self.memory_search_func: Optional[Callable] = None
    
    def set_memory_search(self, func: Callable[[str], List[str]]):
        """Set memory search function."""
        self.memory_search_func = func
    
    def reason(self, question: str, max_tokens: int = 300) -> ReasoningResult:
        """Apply tool-augmented reasoning."""
        tools_called = []
        context = []
        
        # Decide which tools to use
        tool_prompt = f"""Question: {question}

Available tools:
- calculator: for math calculations
- search_memory: to recall past information  
- get_date: to get current date
- python_eval: to run Python code

Which tools would help answer this? List them separated by commas, or say "none".

Tools needed:"""
        
        tool_decision = self.llm(tool_prompt, 50)
        
        # Parse tools
        requested_tools = []
        for tool_name in self.tools.keys():
            if tool_name in tool_decision.lower():
                requested_tools.append(tool_name)
        
        # Execute tools
        tool_results = {}
        for tool_name in requested_tools:
            try:
                result = self.tools[tool_name](question)
                tool_results[tool_name] = result
                tools_called.append(tool_name)
                context.append(f"{tool_name}: {result}")
            except Exception as e:
                logger.warning(f"Tool {tool_name} failed: {e}")
        
        # Generate answer with tool context
        if tool_results:
            answer_prompt = f"""Question: {question}

Tool results:
{chr(10).join(f'- {k}: {v}' for k, v in tool_results.items())}

Using this information, provide a complete answer:"""
        else:
            answer_prompt = f"Question: {question}\nAnswer:"
        
        answer = self.llm(answer_prompt, max_tokens)
        
        return ReasoningResult(
            answer=answer,
            confidence=0.85 if tool_results else 0.7,
            reasoning_steps=[f"Used {len(tools_called)} tools"],
            method_used="tool_use",
            tools_called=tools_called,
            context_used=context,
        )
    
    def _calculator(self, question: str) -> str:
        """Simple calculator."""
        # Extract numbers and operation
        numbers = re.findall(r'-?\d+\.?\d*', question)
        if len(numbers) >= 2:
            a, b = float(numbers[0]), float(numbers[1])
            if '+' in question or 'plus' in question or 'add' in question:
                return str(a + b)
            elif '-' in question or 'minus' in question or 'subtract' in question:
                return str(a - b)
            elif '*' in question or 'times' in question or 'multiply' in question:
                return str(a * b)
            elif '/' in question or 'divided' in question:
                return str(a / b) if b != 0 else "undefined"
        return "Could not calculate"
    
    def _search_memory(self, question: str) -> str:
        """Search memory for relevant info."""
        if self.memory_search_func:
            results = self.memory_search_func(question)
            return "; ".join(results[:3]) if results else "No relevant memories"
        return "Memory not available"
    
    def _get_date(self, question: str) -> str:
        """Get current date."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M")
    
    def _python_eval(self, question: str) -> str:
        """Safely evaluate Python expression."""
        # Extract Python code
        code_match = re.search(r'`([^`]+)`', question)
        if code_match:
            code = code_match.group(1)
            try:
                # Very restricted eval
                allowed = {'abs', 'round', 'min', 'max', 'sum', 'len'}
                if any(f in code for f in ['import', 'exec', 'eval', 'open', '__']):
                    return "Code not allowed"
                result = eval(code, {"__builtins__": {}}, {})
                return str(result)
            except:
                return "Evaluation failed"
        return "No code found"


class RAGReasoner:
    """
    Retrieval-Augmented Generation.
    
    Uses memory to provide context for better answers.
    """
    
    def __init__(self, llm_func: Callable[[str, int], str]):
        self.llm = llm_func
        self.memory_search_func: Optional[Callable] = None
    
    def set_memory_search(self, func: Callable[[str, int], List[Any]]):
        """Set memory search function."""
        self.memory_search_func = func
    
    def reason(self, question: str, max_tokens: int = 400) -> ReasoningResult:
        """Apply RAG reasoning."""
        context_used = []
        
        # Retrieve relevant memories
        if self.memory_search_func:
            memories = self.memory_search_func(question, 5)
            for mem in memories:
                if hasattr(mem, 'content'):
                    context_used.append(mem.content)
                else:
                    context_used.append(str(mem))
        
        # Build prompt with context
        if context_used:
            context_str = "\n".join(f"- {c}" for c in context_used[:5])
            prompt = f"""Use the following context to answer the question.

Context (from memory):
{context_str}

Question: {question}

Answer based on the context and your knowledge:"""
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        answer = self.llm(prompt, max_tokens)
        
        return ReasoningResult(
            answer=answer,
            confidence=0.85 if context_used else 0.7,
            reasoning_steps=[f"Retrieved {len(context_used)} memories"],
            method_used="rag",
            context_used=context_used,
        )


class HybridARCSolver:
    """
    Hybrid ARC Solver using LLM + DSL.
    
    Uses LLM to understand pattern, then generates DSL code.
    """
    
    def __init__(self, llm_func: Callable[[str, int], str]):
        self.llm = llm_func
        self.primitives = [
            "rotate_90", "rotate_180", "rotate_270",
            "flip_horizontal", "flip_vertical",
            "crop", "fill", "copy", "scale",
        ]
    
    def solve(self, input_grid: List[List[int]], output_grid: List[List[int]] = None) -> Dict[str, Any]:
        """Solve ARC task using hybrid approach."""
        # Convert grid to string representation
        grid_str = self._grid_to_str(input_grid)
        
        # Ask LLM to describe the pattern
        describe_prompt = f"""Look at this grid:
{grid_str}

Describe the pattern you see. What transformation would you apply?
Available operations: {', '.join(self.primitives)}

Pattern description:"""
        
        description = self.llm(describe_prompt, 200)
        
        # Extract suggested operation
        suggested_op = None
        for prim in self.primitives:
            if prim.replace('_', ' ') in description.lower() or prim in description.lower():
                suggested_op = prim
                break
        
        # Apply operation
        if suggested_op:
            result = self._apply_primitive(input_grid, suggested_op)
        else:
            result = input_grid
        
        return {
            "input": input_grid,
            "output": result,
            "operation": suggested_op or "identity",
            "description": description[:200],
        }
    
    def _grid_to_str(self, grid: List[List[int]]) -> str:
        """Convert grid to string."""
        return '\n'.join(' '.join(str(c) for c in row) for row in grid)
    
    def _apply_primitive(self, grid: List[List[int]], op: str) -> List[List[int]]:
        """Apply a primitive operation."""
        import copy
        grid = copy.deepcopy(grid)
        
        if op == "rotate_90":
            return [list(row) for row in zip(*grid[::-1])]
        elif op == "rotate_180":
            return [row[::-1] for row in grid[::-1]]
        elif op == "rotate_270":
            return [list(row) for row in zip(*grid)][::-1]
        elif op == "flip_horizontal":
            return [row[::-1] for row in grid]
        elif op == "flip_vertical":
            return grid[::-1]
        
        return grid


class AdvancedReasoner:
    """
    Complete Advanced Reasoning System.
    
    Integrates all reasoning techniques and selects the best approach.
    """
    
    def __init__(self, llm_func: Callable[[str, int], str]):
        self.llm = llm_func
        
        # Initialize all reasoning modules
        self.cot = ChainOfThought(llm_func)
        self.critique = SelfCritique(llm_func)
        self.consistency = SelfConsistency(llm_func, num_samples=3)
        self.tools = ToolUse(llm_func)
        self.rag = RAGReasoner(llm_func)
        self.arc = HybridARCSolver(llm_func)
    
    def set_memory(self, memory_system):
        """Connect memory system for RAG and tools."""
        if memory_system:
            self.rag.set_memory_search(memory_system.recall)
            self.tools.set_memory_search(
                lambda q: [m.content for m in memory_system.recall(q, 3)]
            )
    
    def reason(
        self, 
        question: str, 
        method: str = "auto",
        max_tokens: int = 400,
    ) -> ReasoningResult:
        """
        Apply advanced reasoning.
        
        Methods:
        - auto: automatically select best method
        - cot: chain of thought
        - critique: self-critique
        - consistency: self-consistency voting
        - tools: tool-augmented
        - rag: retrieval-augmented
        """
        if method == "auto":
            method = self._select_method(question)
        
        if method == "cot":
            return self.cot.reason(question, max_tokens)
        elif method == "critique":
            return self.critique.reason(question, max_tokens)
        elif method == "consistency":
            return self.consistency.reason(question, max_tokens)
        elif method == "tools":
            return self.tools.reason(question, max_tokens)
        elif method == "rag":
            return self.rag.reason(question, max_tokens)
        else:
            # Default to CoT
            return self.cot.reason(question, max_tokens)
    
    def _select_method(self, question: str) -> str:
        """Automatically select best reasoning method."""
        q_lower = question.lower()
        
        # Math/calculation → tools
        if any(w in q_lower for w in ['calculate', 'compute', '+', '-', '*', '/', 'how much', 'how many']):
            return "tools"
        
        # Memory-related → RAG
        if any(w in q_lower for w in ['remember', 'recall', 'previously', 'earlier', 'last time']):
            return "rag"
        
        # Complex reasoning → CoT
        if any(w in q_lower for w in ['why', 'how', 'explain', 'step by step', 'prove']):
            return "cot"
        
        # Verification/critique
        if any(w in q_lower for w in ['check', 'verify', 'is it true', 'correct']):
            return "critique"
        
        # Default to consistency for factual questions
        return "cot"
    
    def solve_arc(self, input_grid: List[List[int]]) -> Dict[str, Any]:
        """Solve ARC task."""
        return self.arc.solve(input_grid)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics."""
        return {
            "methods_available": ["cot", "critique", "consistency", "tools", "rag", "arc"],
            "rag_connected": self.rag.memory_search_func is not None,
            "tools_available": list(self.tools.tools.keys()),
        }
