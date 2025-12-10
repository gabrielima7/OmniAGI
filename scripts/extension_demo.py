#!/usr/bin/env python3
"""
OmniAGI Extension Demo - Solve Real Problems

This script demonstrates the OmniAGI extension system by solving
practical problems using DeveloperExtension, MemoryExtension, and WebExtension.

Usage:
    python scripts/extension_demo.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omniagi.extensions.developer import DeveloperExtension
from omniagi.extensions.memory import MemoryExtension
from omniagi.extensions.web import WebExtension


def main():
    print("=" * 60)
    print("🧠 OMNIAGI EXTENSION DEMO")
    print("=" * 60)
    
    # Initialize extensions
    dev = DeveloperExtension()
    dev.activate()
    
    mem = MemoryExtension()
    mem.activate()
    
    web = WebExtension()
    
    # ===== CHALLENGE 1: Analyze Project =====
    print("\n📁 TASK 1: Project Analysis")
    print("-" * 40)
    
    result = dev.execute_tool("search_files", pattern="**/*.py", path="omniagi")
    print(f"   Python files: {result.get('count', 0)}")
    
    result = dev.execute_tool("shell", command="wc -l omniagi/**/*.py 2>/dev/null | tail -1 || echo '0'")
    print(f"   Lines of code: {result.get('stdout', '').strip()}")
    
    # ===== CHALLENGE 2: Read and Analyze Files =====
    print("\n📄 TASK 2: Extension Analysis")
    print("-" * 40)
    
    for ext in ["developer", "memory", "web"]:
        result = dev.execute_tool("read_file", path=f"omniagi/extensions/{ext}.py")
        if result.get("success"):
            content = result.get("content", "")
            classes = content.count("class ")
            functions = content.count("def ")
            print(f"   {ext}.py: {classes} classes, {functions} functions")
    
    # ===== CHALLENGE 3: Memory Operations =====
    print("\n🧠 TASK 3: Memory System")
    print("-" * 40)
    
    # Store data
    mem.execute_tool("remember", key="demo_run", value="success", category="demo")
    mem.execute_tool("remember", key="timestamp", value="2024", category="demo")
    
    # Recall
    result = mem.execute_tool("recall", key="demo_run")
    print(f"   Stored: demo_run = {result.get('value')}")
    
    result = mem.execute_tool("list_memories", category="demo")
    print(f"   Total memories in 'demo': {result.get('count')}")
    
    # Cleanup
    mem.execute_tool("forget", key="demo_run")
    mem.execute_tool("forget", key="timestamp")
    print("   Cleaned up demo memories")
    
    # ===== CHALLENGE 4: Shell Commands =====
    print("\n⚡ TASK 4: System Info")
    print("-" * 40)
    
    commands = [
        ("Python", "python3 --version 2>&1 | head -1"),
        ("Git branch", "git branch --show-current"),
        ("Last commit", "git log -1 --oneline"),
    ]
    
    for name, cmd in commands:
        result = dev.execute_tool("shell", command=cmd)
        output = result.get("stdout", "").strip() or "N/A"
        print(f"   {name}: {output}")
    
    # ===== CHALLENGE 5: Directory Operations =====
    print("\n📂 TASK 5: Directory Scan")
    print("-" * 40)
    
    result = dev.execute_tool("list_dir", path="omniagi/extensions")
    items = result.get("items", [])
    print(f"   Extensions directory: {len(items)} items")
    for item in items:
        if item.endswith(".py") and not item.startswith("__"):
            print(f"     📄 {item}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("✅ ALL TASKS COMPLETED!")
    print("=" * 60)
    print(f"\nTools used:")
    print(f"  🔧 DeveloperExtension: shell, read_file, list_dir, search_files")
    print(f"  🧠 MemoryExtension: remember, recall, list_memories, forget")
    print(f"  🌐 WebExtension: Available (http_get, scrape_text, etc.)")


if __name__ == "__main__":
    main()
