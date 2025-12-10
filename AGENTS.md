# AGENTS.md

> Instructions for AI coding agents working on OmniAGI

## Project Overview

OmniAGI is an **Artificial General Intelligence framework** with:
- Consciousness engine (GWT + IIT)
- Hybrid reasoning (Neural + Symbolic + Algorithmic)
- RAG memory system
- Multi-modal processing
- Self-improvement capabilities

## Dev Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Set Python path
export PYTHONPATH=/path/to/OmniAGI

# Install dependencies
pip install -e .
```

## Project Structure

```
omniagi/
├── consciousness/    # Consciousness engine (GWT + IIT)
├── reasoning/        # Symbolic reasoning
├── benchmarks/       # ARC solver, math tests
├── memory/           # RAG + episodic memory
├── multimodal/       # Text/image embeddings
├── meta/             # Meta-learning
├── metacognition/    # Self-reflection
├── safety/           # Constitutional AI
├── creativity/       # Idea generation
├── autonomy/         # Goal generation
├── collective/       # HiveMind swarm
├── ouroboros/        # Self-improvement
├── extensions/       # Goose-style extensions
├── mcp/              # Model Context Protocol
├── tools/            # Tool implementations
├── core/             # Engine, config, multi-LLM
├── cli/              # Command-line interface
└── daemon/           # Background services
```

## Code Conventions

- **Python 3.11+** required
- Use **type hints** everywhere
- Use **dataclasses** for data structures
- Use **structlog** for logging
- Follow **PEP 8** style guide
- Maximum line length: **100 characters**

## Testing Instructions

```bash
# Quick test - verify all modules import
python -m omniagi.cli.main test

# Test specific module
python -c "from omniagi.consciousness import ConsciousnessEngine; print('OK')"

# Test math solver
python -c "
from omniagi.benchmarks.arc_solver import ChainOfThoughtSolver
s = ChainOfThoughtSolver()
assert s.solve('sum', '25+37').answer == '62'
print('Math OK')
"

# Test RAG
python -c "
from omniagi.memory.rag import RAGSystem
r = RAGSystem('test')
r.initialize()
print('RAG OK')
"
```

## Key Files to Understand

| File | Purpose |
|------|---------|
| `omniagi/consciousness/engine.py` | Main consciousness loop |
| `omniagi/core/engine.py` | LLM inference engine |
| `omniagi/core/multi_llm.py` | Multi-LLM orchestration |
| `omniagi/benchmarks/arc_solver.py` | Math/puzzle solver |
| `omniagi/memory/rag.py` | RAG system with ChromaDB |
| `omniagi/cli/main.py` | CLI entry point |

## Making Changes

1. **Before changing**: Run tests to ensure baseline works
2. **After changing**: Run tests again to verify nothing broke
3. **New modules**: Add `__init__.py` with exports
4. **Dependencies**: Add to `pyproject.toml`

## Common Tasks

### Add a new tool
```python
# In omniagi/tools/your_tool.py
from dataclasses import dataclass

@dataclass
class YourTool:
    name: str = "your_tool"
    
    def execute(self, **kwargs) -> str:
        # Implementation
        return result
```

### Add a new extension
```python
# In omniagi/extensions/your_extension.py
from omniagi.extensions.base import Extension, Tool

class YourExtension(Extension):
    name = "your_extension"
    tools = [
        Tool("tool_name", "Tool description"),
    ]
```

## LLM Backend

OmniAGI uses **RWKV-6** as the default LLM:
- Models in `models/rwkv/`
- 1.6B (3GB) - lightweight
- 3B (6GB) - recommended
- 7B (14GB) - best quality

```python
# Load RWKV
from rwkv.model import RWKV
model = RWKV(model="models/rwkv/rwkv-6-3b.pth", strategy="cuda fp16")
```

## PR Instructions

- **Title format**: `[module] Brief description`
- **Always run tests** before committing
- **Update AGENTS.md** if adding new modules
- **Document breaking changes** in commit message

## Important Notes

> [!CAUTION]
> The `llama_cpp` import is optional. Don't add hard dependencies on it.

> [!TIP]
> Use `structlog` for logging, not `print()` or `logging`.

> [!NOTE]
> RWKV models require significant GPU memory. Use hybrid strategy if limited VRAM.
