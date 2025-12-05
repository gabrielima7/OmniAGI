#!/usr/bin/env python3
"""
OmniAGI - Main Entry Point

Start and interact with the AGI system.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="OmniAGI - Artificial General Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  omniagi start          Start the AGI system
  omniagi think          Generate a thought
  omniagi status         Show system status
  omniagi benchmark      Run capability benchmarks
  omniagi improve        Propose self-improvement
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Start command
    start = subparsers.add_parser("start", help="Start the AGI system")
    start.add_argument("--model", default="rwkv-6-7b", help="Model to use")
    start.add_argument("--thinking", action="store_true", help="Enable background thinking")
    
    # Think command
    think = subparsers.add_parser("think", help="Generate a thought")
    think.add_argument("prompt", nargs="?", default="What are you thinking about?")
    
    # Status command
    subparsers.add_parser("status", help="Show system status")
    
    # Benchmark command
    benchmark = subparsers.add_parser("benchmark", help="Run capability benchmarks")
    benchmark.add_argument("--arc", action="store_true", help="Run ARC benchmark")
    
    # Improve command
    subparsers.add_parser("improve", help="Propose self-improvement")
    
    # Setup command
    setup = subparsers.add_parser("setup", help="Setup RWKV model")
    setup.add_argument("--model", default="rwkv-6-7b", help="Model to download")
    
    args = parser.parse_args()
    
    if args.command == "start":
        cmd_start(args)
    elif args.command == "think":
        cmd_think(args)
    elif args.command == "status":
        cmd_status()
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "improve":
        cmd_improve()
    elif args.command == "setup":
        cmd_setup(args)


def cmd_start(args):
    """Start the AGI system."""
    print("🧠 Starting OmniAGI...")
    
    try:
        from omniagi.agi_controller import get_agi
        
        agi = get_agi(model_name=args.model)
        if agi.initialize():
            print("✅ AGI initialized successfully")
            
            if args.thinking:
                if agi.start_thinking():
                    print("💭 Background thinking enabled")
            
            # Print status
            status = agi.get_status()
            print(f"\n📊 Status:")
            print(f"   State: {status.state.name}")
            print(f"   LLM: {'✅' if status.llm_loaded else '❌'}")
            print(f"   Safety: {'✅' if status.safety_active else '❌'}")
            print(f"   Thinking: {'✅' if status.thinking_active else '❌'}")
            
        else:
            print("❌ AGI initialization failed")
            sys.exit(1)
            
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -e .[rwkv]")
        sys.exit(1)


def cmd_think(args):
    """Generate a thought."""
    try:
        from omniagi.agi_controller import get_agi
        
        agi = get_agi()
        agi.initialize()
        
        print(f"💭 Thinking about: {args.prompt}\n")
        response = agi.think(args.prompt)
        print(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def cmd_status():
    """Show system status."""
    try:
        from omniagi.agi_controller import get_agi
        
        agi = get_agi()
        status = agi.get_status()
        data = agi.to_dict()
        
        print("📊 OmniAGI Status")
        print("=" * 40)
        print(f"State: {status.state.name}")
        print(f"Uptime: {status.uptime_seconds:.0f}s")
        print()
        print("Components:")
        for name, active in data["components"].items():
            icon = "✅" if active else "❌"
            print(f"  {icon} {name}")
        print()
        print(f"Total Thoughts: {status.total_thoughts}")
        print(f"Total Actions: {status.total_actions}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def cmd_benchmark(args):
    """Run benchmarks."""
    try:
        if args.arc:
            from omniagi.rsi.arc_benchmark import ARCBenchmark
            
            print("🧪 Running ARC Benchmark...")
            arc = ARCBenchmark()
            
            # Show available tasks
            tasks = arc.get_all_tasks()
            print(f"   Tasks available: {len(tasks)}")
            
            stats = arc.get_stats()
            print(f"\n📊 ARC Stats:")
            print(f"   Current Accuracy: {stats['accuracy']*100:.1f}%")
            print(f"   Human Baseline: {stats['human_baseline']*100:.1f}%")
            print(f"   AGI Threshold: 50%")
            print(f"   Is AGI Level: {'✅' if stats['is_agi_level'] else '❌'}")
        else:
            from omniagi.rsi.evaluator import CapabilityEvaluator
            
            print("🧪 Running Capability Benchmark...")
            evaluator = CapabilityEvaluator()
            stats = evaluator.get_stats()
            
            print(f"\n📊 Capability Profile:")
            for category, score in stats.get("capability_profile", {}).items():
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                print(f"   {category}: {bar} {score*100:.0f}%")
            
            print(f"\n⚠️ Weakest Areas:")
            for area, score in stats.get("weakest_areas", []):
                print(f"   - {area}: {score*100:.0f}%")
                
    except Exception as e:
        print(f"❌ Error: {e}")


def cmd_improve():
    """Propose self-improvement."""
    try:
        from omniagi.agi_controller import get_agi
        
        agi = get_agi()
        agi.initialize()
        
        print("🔧 Proposing self-improvement...\n")
        proposal = agi.propose_improvement()
        
        if proposal:
            print(f"📝 Improvement Proposal:")
            print(f"   Type: {proposal.get('change_type', 'unknown')}")
            print(f"   Description: {proposal.get('description', 'N/A')}")
            print(f"   Impact: {proposal.get('impact', 0)*100:.0f}%")
            print(f"   Risk: {proposal.get('risk', 0)*100:.0f}%")
        else:
            print("No improvements proposed at this time.")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def cmd_setup(args):
    """Setup RWKV model."""
    import subprocess
    import sys
    
    script = Path(__file__).parent.parent / "scripts" / "setup_rwkv.py"
    subprocess.run([sys.executable, str(script)])


if __name__ == "__main__":
    main()
