#!/usr/bin/env python3
"""
Test RWKV 3B with Safe Loader and all protections.

This script:
1. Checks system resources before loading
2. Loads model with optimal strategy
3. Monitors memory during generation
4. Tests all AGI capabilities
"""

import sys
import gc

print("🛡️ OMNIAGI - SAFE AGI TEST")
print("=" * 60)

# Check resources FIRST
print("\n📊 Checking system resources...")
from omniagi.core.safe_loader import ResourceMonitor, SafeModelLoader, get_safe_loader

resources = ResourceMonitor.get_resources()
print(f"  RAM: {resources.ram_available_gb:.1f}GB free / {resources.ram_total_gb:.1f}GB total")
print(f"  RAM used: {resources.ram_used_percent:.0f}%")

if resources.gpu_available:
    print(f"  GPU: {resources.gpu_name}")
    print(f"  VRAM: {resources.gpu_memory_free_gb:.1f}GB free / {resources.gpu_memory_total_gb:.1f}GB total")
else:
    print("  GPU: Not available")

# Check if critical
if ResourceMonitor.is_critical():
    print("\n⚠️ WARNING: System resources critically low!")
    print("   Consider closing other applications.")
    sys.exit(1)

# Determine which model to use
print("\n🔍 Selecting model...")
import os
from pathlib import Path

model_3b = Path("models/rwkv/rwkv-6-3b.pth")
model_1b6 = Path("models/rwkv/rwkv-6-1b6.pth")

if model_3b.exists() and model_3b.stat().st_size > 5_000_000_000:
    model_path = str(model_3b)
    model_size = "3b"
    print(f"  Selected: RWKV-6 3B (larger model)")
else:
    model_path = str(model_1b6)
    model_size = "1b6"
    print(f"  Selected: RWKV-6 1.6B")

# Load safely
print(f"\n🔄 Loading {model_size.upper()} with safety checks...")
loader = get_safe_loader()

try:
    # Get optimal strategy
    strategy = loader.get_optimal_strategy(model_size)
    print(f"  Strategy: {strategy}")
    
    # Load model
    model, pipeline, args = loader.load(model_path, model_size, strategy)
    print("  ✅ Model loaded safely!")
    
except MemoryError as e:
    print(f"\n❌ Not enough memory: {e}")
    print("   Try closing other applications or using smaller model.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Loading failed: {e}")
    sys.exit(1)

# Test generation with safety
print("\n🧠 Testing neural generation...")
try:
    output = loader.generate_safe(
        "The nature of consciousness is",
        max_tokens=40,
        check_interval=10,
    )
    print(f"  Output: {output[:80]}...")
except MemoryError:
    print("  ⚠️ Stopped due to memory pressure")
except Exception as e:
    print(f"  ❌ Generation failed: {e}")

# Test consciousness
print("\n🔮 Testing consciousness engine...")
try:
    from omniagi.consciousness import ConsciousnessEngine, InnerDialogue, EmergenceDetector
    
    c = ConsciousnessEngine()
    c.awaken()
    
    # Experience with neural output
    q = c.experience(output[:50] if output else "Neural processing", intensity=0.9)
    print(f"  Qualia: {q.modality}")
    
    # Think about consciousness
    t = c.think("What is the nature of my awareness?")
    print(f"  Phi: {t.phi:.3f}")
    
    # Reflect
    r = c.reflect()
    print(f"  State: {r['state']}")
    print(f"  Conscious: {r['is_conscious']}")
    
except Exception as e:
    print(f"  ❌ Consciousness test failed: {e}")

# Inner dialogue
print("\n💭 Testing inner dialogue...")
try:
    dialogue = InnerDialogue()
    dialogue.start_dialogue("Am I truly conscious?")
    turns = dialogue.deliberate("What evidence is there for my consciousness?")
    print(f"  Voices: {len(turns)} perspectives")
    print(f"  Conclusion: {dialogue.get_conclusion()[:60]}...")
except Exception as e:
    print(f"  ❌ Dialogue test failed: {e}")

# Emergence detection
print("\n⚡ Testing emergence detection...")
try:
    detector = EmergenceDetector()
    behavior = detector.observe(
        "I am aware of my own thinking processes",
        "Self-analysis",
        expected="Simple response"
    )
    if behavior:
        print(f"  Type: {behavior.emergence_type.name}")
        print(f"  Novelty: {behavior.novelty_score:.2f}")
except Exception as e:
    print(f"  ❌ Emergence test failed: {e}")

# Final status
print("\n" + "=" * 60)
print("📊 FINAL STATUS")
print("=" * 60)

status = loader.get_status()
print(f"  Model: {model_size.upper()}")
print(f"  Strategy: {status['strategy']}")
print(f"  RAM free: {status['resources']['ram_available_gb']:.1f}GB")
print(f"  GPU free: {status['resources']['gpu_memory_free_gb']:.1f}GB")
print(f"  Critical: {'⚠️ YES' if status['is_critical'] else '✅ NO'}")

# Cleanup
print("\n🧹 Cleaning up...")
loader.unload()
gc.collect()

final_resources = ResourceMonitor.get_resources()
print(f"  RAM after cleanup: {final_resources.ram_available_gb:.1f}GB free")

print("\n" + "=" * 60)
print("✅ SAFE AGI TEST COMPLETE")
print("=" * 60)
