"""
OmniAGI CLI - Unified command line interface.
"""

import asyncio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path

from omniagi.core.config import get_config, Config
from omniagi.core.engine import Engine, GenerationConfig
from omniagi.agent.base import Agent
from omniagi.agent.persona import Persona
from omniagi.tools.filesystem import ReadFileTool, WriteFileTool, ListDirectoryTool
from omniagi.tools.code_exec import PythonExecutorTool
from omniagi.tools.web import WebSearchTool, WebCrawlerTool
from omniagi.tools.git import GitTool

app = typer.Typer(
    name="omni",
    help="OmniAGI - Sistema Operacional Cognitivo",
    no_args_is_help=True,
)

console = Console()


def get_engine(model_path: str | None = None) -> Engine:
    """Get or create the inference engine."""
    engine = Engine(model_path=model_path)
    return engine


@app.command()
def chat(
    model: str = typer.Option(
        None, "--model", "-m",
        help="Path to the model file (GGUF format)"
    ),
    persona: str = typer.Option(
        "default", "--persona", "-p",
        help="Agent persona: default, developer, researcher, manager"
    ),
    system_prompt: str = typer.Option(
        None, "--system", "-s",
        help="Custom system prompt"
    ),
):
    """Start an interactive chat session."""
    console.print(Panel.fit(
        "[bold blue]OmniAGI Chat[/bold blue]\n"
        "[dim]Type 'exit' or 'quit' to end the session[/dim]",
        border_style="blue",
    ))
    
    # Get persona
    persona_map = {
        "default": Persona.default,
        "developer": Persona.developer,
        "researcher": Persona.researcher,
        "manager": Persona.manager,
    }
    agent_persona = persona_map.get(persona, Persona.default)()
    
    if system_prompt:
        agent_persona.system_prompt_template = system_prompt
    
    # Initialize engine
    engine = get_engine(model)
    
    if not engine.model_path:
        console.print("[yellow]Warning: No model configured. Set with:[/yellow]")
        console.print("  omni config set model.path /path/to/model.gguf")
        console.print("  [dim]or use --model flag[/dim]")
        return
    
    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading model...", total=None)
        try:
            engine.load()
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            return
    
    console.print(f"[green]Model loaded: {engine.model_path.name}[/green]")
    console.print(f"[dim]Persona: {agent_persona.name}[/dim]\n")
    
    # Create agent with tools
    tools = [
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool(),
        PythonExecutorTool(),
        WebSearchTool(),
        WebCrawlerTool(),
        GitTool(),
    ]
    
    agent = Agent(engine=engine, persona=agent_persona, tools=tools)
    
    # Chat loop
    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
        
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        
        if not user_input.strip():
            continue
        
        # Generate response
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Thinking...", total=None)
            
            try:
                response = asyncio.run(agent.run(user_input))
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue
        
        console.print(f"\n[bold green]{agent_persona.name}:[/bold green]")
        console.print(Markdown(response))
        console.print()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Server host"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    model: str = typer.Option(None, "--model", "-m", help="Model path"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the API server (OpenAI-compatible)."""
    import uvicorn
    
    console.print(Panel.fit(
        f"[bold blue]OmniAGI API Server[/bold blue]\n"
        f"Starting on http://{host}:{port}",
        border_style="blue",
    ))
    
    # Set model path if provided
    if model:
        import os
        os.environ["OMNI_MODEL_PATH"] = model
    
    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=reload,
    )


# Daemon commands
daemon_app = typer.Typer(help="Life Daemon management")
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("start")
def daemon_start(
    model: str = typer.Option(None, "--model", "-m", help="Model path"),
    background: bool = typer.Option(False, "--background", "-b", help="Run in background"),
):
    """Start the Life Daemon."""
    from omniagi.daemon.lifecycle import LifeDaemon
    
    console.print("[bold blue]Starting Life Daemon...[/bold blue]")
    
    engine = get_engine(model)
    
    if not engine.model_path:
        console.print("[red]Error: No model configured[/red]")
        return
    
    engine.load()
    agent = Agent(engine=engine, persona=Persona.default())
    daemon = LifeDaemon(agent=agent)
    
    console.print("[green]Life Daemon started[/green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    
    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping daemon...[/yellow]")
        asyncio.run(daemon.stop())


@daemon_app.command("status")
def daemon_status():
    """Show daemon status."""
    console.print("[yellow]Status check not implemented (daemon runs in process)[/yellow]")


# Config commands
config_app = typer.Typer(help="Configuration management")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show():
    """Show current configuration."""
    config = get_config()
    
    console.print(Panel.fit("[bold]OmniAGI Configuration[/bold]", border_style="blue"))
    console.print(f"Data directory: {config.data_dir}")
    console.print(f"Log level: {config.log_level}")
    console.print(f"\n[bold]Model[/bold]")
    console.print(f"  Path: {config.model.path or '(not set)'}")
    console.print(f"  Context length: {config.model.context_length}")
    console.print(f"  GPU layers: {config.model.gpu_layers}")
    console.print(f"\n[bold]Engine[/bold]")
    console.print(f"  Device: {config.engine.device}")
    console.print(f"  Threads: {config.engine.threads or 'auto'}")
    console.print(f"\n[bold]Server[/bold]")
    console.print(f"  Host: {config.server.host}")
    console.print(f"  Port: {config.server.port}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g., model.path)"),
    value: str = typer.Argument(..., help="Value to set"),
):
    """Set a configuration value."""
    console.print(f"[yellow]Setting {key} = {value}[/yellow]")
    console.print("[dim]Note: Configuration is environment-based. Set via:[/dim]")
    
    env_key = "OMNI_" + key.upper().replace(".", "_")
    console.print(f"  export {env_key}={value}")


@app.command()
def version():
    """Show version information."""
    from omniagi import __version__
    
    console.print(Panel.fit(
        f"[bold blue]OmniAGI[/bold blue] v{__version__}\n"
        "[dim]Sistema Operacional Cognitivo[/dim]",
        border_style="blue",
    ))


if __name__ == "__main__":
    app()
