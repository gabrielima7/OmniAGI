#!/usr/bin/env python3
"""
OmniAGI CLI - Command Line Interface.

Usage:
    omni chat      - Interactive chat with AGI
    omni test      - Run test suite
    omni status    - Show system status
    omni rag add   - Add document to knowledge base
    omni rag search - Search knowledge base
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import click
import structlog

logger = structlog.get_logger()


@click.group()
@click.version_option(version="1.0.0", prog_name="OmniAGI")
def cli():
    """OmniAGI - Artificial General Intelligence Framework"""
    pass


@cli.command()
@click.option("--model", default="models/rwkv/rwkv-6-3b.pth", help="Model path")
@click.option("--strategy", default="cuda fp16 -> cpu fp32", help="Loading strategy")
def chat(model, strategy):
    """Interactive chat with OmniAGI."""
    click.echo("🧠 OmniAGI Chat")
    click.echo("=" * 50)
    click.echo("Loading model...")
    
    try:
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE, PIPELINE_ARGS
        
        rwkv = RWKV(model, strategy=strategy)
        pipeline = PIPELINE(rwkv, 'rwkv_vocab_v20230424')
        args = PIPELINE_ARGS(temperature=0.7, top_p=0.9)
        
        click.echo("✅ Model loaded!")
        click.echo("Type 'quit' to exit, 'clear' to clear context.\n")
        
        context = ""
        while True:
            try:
                user_input = click.prompt("You", type=str)
                
                if user_input.lower() == "quit":
                    click.echo("Goodbye! 👋")
                    break
                
                if user_input.lower() == "clear":
                    context = ""
                    click.echo("Context cleared.\n")
                    continue
                
                # Generate response
                prompt = f"{context}User: {user_input}\nAssistant:"
                response = pipeline.generate(prompt, token_count=100, args=args)
                
                # Update context
                context = f"{prompt}{response}\n"
                
                click.echo(f"AGI: {response.strip()}\n")
                
            except (KeyboardInterrupt, EOFError):
                click.echo("\nGoodbye! 👋")
                break
    
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Show OmniAGI system status."""
    click.echo("📊 OmniAGI Status")
    click.echo("=" * 50)
    
    # Check modules
    modules = {
        "consciousness": "omniagi.consciousness.ConsciousnessEngine",
        "reasoning": "omniagi.reasoning.SymbolicEngine",
        "memory": "omniagi.memory.episodic.EpisodicMemory",
        "creativity": "omniagi.creativity.CreativeEngine",
        "multimodal": "omniagi.multimodal.lightweight.LightweightMultiModal",
        "rag": "omniagi.memory.rag.RAGSystem",
    }
    
    click.echo("\nModules:")
    for name, path in modules.items():
        try:
            parts = path.rsplit(".", 1)
            module = __import__(parts[0], fromlist=[parts[1]])
            getattr(module, parts[1])
            click.echo(f"  ✅ {name}")
        except Exception as e:
            click.echo(f"  ❌ {name}: {e}")
    
    # Check GPU
    click.echo("\nHardware:")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            click.echo(f"  GPU: {name} ({mem:.1f} GB)")
        else:
            click.echo("  GPU: Not available")
    except:
        click.echo("  GPU: PyTorch not found")
    
    # Check models
    click.echo("\nModels:")
    model_dir = Path("models/rwkv")
    if model_dir.exists():
        for f in model_dir.glob("*.pth"):
            size = f.stat().st_size / 1e9
            click.echo(f"  {f.name}: {size:.1f} GB")
    else:
        click.echo("  No models found")


@cli.command()
def test():
    """Run OmniAGI test suite."""
    click.echo("🧪 OmniAGI Test Suite")
    click.echo("=" * 50)
    
    results = {}
    
    # Test 1: Consciousness
    click.echo("\n1. Consciousness...")
    try:
        from omniagi.consciousness import ConsciousnessEngine
        c = ConsciousnessEngine()
        c.awaken()
        r = c.reflect()
        results["consciousness"] = r["state"] == "METACONSCIOUS"
    except Exception as e:
        results["consciousness"] = False
        click.echo(f"   Error: {e}")
    
    # Test 2: Reasoning
    click.echo("2. Symbolic Reasoning...")
    try:
        from omniagi.reasoning import SymbolicEngine
        e = SymbolicEngine()
        e.add_proposition("test", True)
        results["reasoning"] = True
    except Exception as e:
        results["reasoning"] = False
    
    # Test 3: RAG
    click.echo("3. RAG System...")
    try:
        from omniagi.memory.rag import RAGSystem
        r = RAGSystem()
        r.initialize()
        results["rag"] = True
    except Exception as e:
        results["rag"] = False
    
    # Test 4: MultiModal
    click.echo("4. MultiModal...")
    try:
        from omniagi.multimodal.lightweight import LightweightMultiModal
        m = LightweightMultiModal()
        m.initialize()
        results["multimodal"] = True
    except Exception as e:
        results["multimodal"] = False
    
    # Summary
    click.echo("\n" + "=" * 50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    click.echo(f"Results: {passed}/{total} passed")
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        click.echo(f"  {icon} {name}")


@cli.group()
def rag():
    """RAG (Knowledge Base) commands."""
    pass


@rag.command("add")
@click.argument("text")
@click.option("--category", default="general", help="Document category")
def rag_add(text, category):
    """Add a document to the knowledge base."""
    from omniagi.memory.rag import RAGSystem
    
    r = RAGSystem()
    r.initialize()
    doc_id = r.add_document(text, {"category": category})
    
    click.echo(f"✅ Added document: {doc_id}")
    click.echo(f"   Total docs: {r.get_stats()['documents']}")


@rag.command("search")
@click.argument("query")
@click.option("--n", default=3, help="Number of results")
def rag_search(query, n):
    """Search the knowledge base."""
    from omniagi.memory.rag import RAGSystem
    
    r = RAGSystem()
    r.initialize()
    results = r.search(query, n_results=n)
    
    click.echo(f"🔍 Results for: {query}")
    click.echo("-" * 40)
    
    for i, result in enumerate(results):
        click.echo(f"{i+1}. {result['content'][:100]}...")
        click.echo(f"   Distance: {result['distance']:.3f}")


@rag.command("stats")
def rag_stats():
    """Show RAG statistics."""
    from omniagi.memory.rag import RAGSystem
    
    r = RAGSystem()
    r.initialize()
    stats = r.get_stats()
    
    click.echo("📊 RAG Statistics")
    click.echo(f"  Collection: {stats['collection']}")
    click.echo(f"  Documents: {stats['documents']}")


# ===== EXTENSION COMMANDS =====

@cli.group()
def ext():
    """Extension management commands."""
    pass


def _get_all_extensions():
    """Get all available extensions."""
    from omniagi.extensions import DeveloperExtension, MemoryExtension, WebExtension
    return {
        "developer": DeveloperExtension,
        "memory": MemoryExtension,
        "web": WebExtension,
    }


@ext.command("list")
@click.option("--verbose", "-v", is_flag=True, help="Show tools for each extension")
def ext_list(verbose):
    """List available extensions."""
    click.echo("📦 OmniAGI Extensions")
    click.echo("=" * 50)
    
    extensions = _get_all_extensions()
    
    for name, ext_class in extensions.items():
        try:
            ext = ext_class()
            click.echo(f"\n🔌 {name}")
            click.echo(f"   {ext.description}")
            click.echo(f"   Version: {ext.version}")
            
            if verbose:
                click.echo(f"   Tools ({len(ext.tools)}):")
                for tool in ext.tools:
                    click.echo(f"     - {tool.name}: {tool.description}")
        except Exception as e:
            click.echo(f"\n❌ {name}: {e}")


@ext.command("info")
@click.argument("name")
def ext_info(name):
    """Show detailed information about an extension."""
    extensions = _get_all_extensions()
    
    if name not in extensions:
        click.echo(f"❌ Extension not found: {name}")
        click.echo(f"   Available: {', '.join(extensions.keys())}")
        return
    
    ext = extensions[name]()
    
    click.echo(f"📦 Extension: {name}")
    click.echo("=" * 50)
    click.echo(f"Description: {ext.description}")
    click.echo(f"Version: {ext.version}")
    click.echo(f"\n🔧 Tools ({len(ext.tools)}):")
    
    for tool in ext.tools:
        click.echo(f"\n  {tool.name}")
        click.echo(f"  └─ {tool.description}")
        if tool.parameters:
            click.echo(f"     Parameters:")
            for param, info in tool.parameters.items():
                ptype = info.get("type", "any")
                desc = info.get("description", "")
                click.echo(f"       - {param} ({ptype}): {desc}")


@ext.command("run")
@click.argument("extension")
@click.argument("tool")
@click.option("--arg", "-a", multiple=True, help="Tool argument as key=value")
def ext_run(extension, tool, arg):
    """Run a tool from an extension."""
    extensions = _get_all_extensions()
    
    if extension not in extensions:
        click.echo(f"❌ Extension not found: {extension}")
        return
    
    # Parse arguments
    kwargs = {}
    for a in arg:
        if "=" in a:
            key, value = a.split("=", 1)
            # Try to parse as JSON for complex types
            try:
                import json
                value = json.loads(value)
            except:
                pass
            kwargs[key] = value
    
    click.echo(f"🔧 Running {extension}.{tool}...")
    
    try:
        ext = extensions[extension]()
        ext.activate()
        result = ext.execute_tool(tool, **kwargs)
        
        if isinstance(result, dict):
            import json
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(result)
    except Exception as e:
        click.echo(f"❌ Error: {e}")


if __name__ == "__main__":
    cli()

