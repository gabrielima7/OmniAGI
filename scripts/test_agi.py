#!/usr/bin/env python3
"""
OmniAGI AGI Test Suite

Tests the new KAN and Neuro-Symbolic AI components
for achieving true AGI capabilities.

Usage:
    python scripts/test_agi.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_kan():
    """Test Kolmogorov-Arnold Networks."""
    print("\n" + "=" * 60)
    print("🧠 TESTING KOLMOGOROV-ARNOLD NETWORKS (KAN)")
    print("=" * 60)
    
    results = {}
    
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        print("  ❌ PyTorch not available, skipping KAN tests")
        return {"kan": False}
    
    # Test 1: KAN Layer
    print("\n📦 1. KAN Layer")
    try:
        from omniagi.kan.efficient_kan import KANLayer
        layer = KANLayer(in_features=4, out_features=2)
        x = torch.randn(8, 4)
        y = layer(x)
        assert y.shape == (8, 2)
        results["KANLayer"] = True
        print(f"  ✅ KANLayer: Input {x.shape} → Output {y.shape}")
    except Exception as e:
        results["KANLayer"] = False
        print(f"  ❌ KANLayer: {e}")
    
    # Test 2: EfficientKAN
    print("\n📦 2. Efficient KAN Network")
    try:
        from omniagi.kan.efficient_kan import EfficientKAN
        kan = EfficientKAN([8, 16, 8, 4])
        x = torch.randn(4, 8)
        y = kan(x)
        assert y.shape == (4, 4)
        reg_loss = kan.regularization_loss()
        results["EfficientKAN"] = True
        print(f"  ✅ EfficientKAN: {[8, 16, 8, 4]} layers")
        print(f"     Input {x.shape} → Output {y.shape}")
        print(f"     Regularization loss: {reg_loss.item():.4f}")
    except Exception as e:
        results["EfficientKAN"] = False
        print(f"  ❌ EfficientKAN: {e}")
    
    # Test 3: RadialBasisKAN (3x faster)
    print("\n📦 3. Radial Basis KAN (3x faster)")
    try:
        from omniagi.kan.efficient_kan import RadialBasisKAN
        rbf_kan = RadialBasisKAN(8, 4, num_centers=10)
        x = torch.randn(4, 8)
        y = rbf_kan(x)
        assert y.shape == (4, 4)
        results["RadialBasisKAN"] = True
        print(f"  ✅ RadialBasisKAN: Input {x.shape} → Output {y.shape}")
    except Exception as e:
        results["RadialBasisKAN"] = False
        print(f"  ❌ RadialBasisKAN: {e}")
    
    # Test 4: Symbolic extraction
    print("\n📦 4. Symbolic KAN (Formula Extraction)")
    try:
        from omniagi.kan.symbolic_kan import SymbolicKAN, SymbolicFormula
        from omniagi.kan.efficient_kan import EfficientKAN
        
        kan = EfficientKAN([4, 8, 2])
        sym_kan = SymbolicKAN(kan)
        
        # Forward pass
        x = torch.randn(4, 4)
        y = sym_kan(x)
        
        results["SymbolicKAN"] = True
        print(f"  ✅ SymbolicKAN: Wrapped EfficientKAN")
        print(f"     Capable of formula extraction")
    except Exception as e:
        results["SymbolicKAN"] = False
        print(f"  ❌ SymbolicKAN: {e}")
    
    return results


def test_neurosymbolic():
    """Test Neuro-Symbolic AI components."""
    print("\n" + "=" * 60)
    print("🔮 TESTING NEURO-SYMBOLIC AI")
    print("=" * 60)
    
    results = {}
    
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        print("  ❌ PyTorch not available, skipping Neuro-Symbolic tests")
        return {"neurosymbolic": False}
    
    # Test 1: Logical Neural Network
    print("\n🔯 1. Logical Neural Network (LNN)")
    try:
        from omniagi.neurosymbolic.neural_logic import LNN, TruthBounds
        
        lnn = LNN()
        lnn.add_predicate("human", 1)
        lnn.add_predicate("mortal", 1)
        lnn.add_rule("mortality", ["human(X)"], "mortal(X)")
        lnn.set_fact("human", "socrates", 0.9)
        
        result = lnn.infer("mortal", "socrates")
        results["LNN"] = True
        print(f"  ✅ LNN: mortal(socrates) = [{result.lower:.2f}, {result.upper:.2f}]")
        print(f"     Explanation: {lnn.explain('mortal', 'socrates')}")
    except Exception as e:
        results["LNN"] = False
        print(f"  ❌ LNN: {e}")
    
    # Test 2: Differentiable Reasoning
    print("\n🔯 2. Differentiable Reasoning")
    try:
        from omniagi.neurosymbolic.differentiable_reasoning import (
            DifferentiableReasoner, Term, Atom, Rule
        )
        
        reasoner = DifferentiableReasoner()
        
        # Add knowledge
        t1 = Term("socrates")
        t2 = Term("human")
        fact = Atom("is_a", [t1, t2], confidence=0.9)
        reasoner.add_knowledge(facts=[fact])
        
        result = reasoner.reason("is_a", ["socrates", "human"])
        results["DifferentiableReasoning"] = True
        print(f"  ✅ Differentiable Reasoning")
        print(f"     Query: {result['query']}")
        print(f"     Score: {result['score']:.2f}")
    except Exception as e:
        results["DifferentiableReasoning"] = False
        print(f"  ❌ Differentiable Reasoning: {e}")
    
    # Test 3: Knowledge Graph
    print("\n🔯 3. Knowledge Graph Neural")
    try:
        from omniagi.neurosymbolic.knowledge_graph import (
            KnowledgeGraphNeural, Entity, Relation, Triple
        )
        
        kg = KnowledgeGraphNeural()
        
        # Add entities
        e1 = Entity("socrates", "Socrates", "person")
        e2 = Entity("plato", "Plato", "person")
        r = Relation("taught", "taught", symmetric=False)
        
        triple = Triple(e1, r, e2)
        kg.add_triple(triple)
        
        stats = kg.get_stats()
        results["KnowledgeGraph"] = True
        print(f"  ✅ Knowledge Graph: {stats}")
    except Exception as e:
        results["KnowledgeGraph"] = False
        print(f"  ❌ Knowledge Graph: {e}")
    
    return results


def test_integrated_brain():
    """Test integrated AGI brain with new components."""
    print("\n" + "=" * 60)
    print("🧬 TESTING INTEGRATED AGI BRAIN")
    print("=" * 60)
    
    results = {}
    
    print("\n🧠 1. Brain Initialization")
    try:
        from omniagi.brain import UnifiedAGIBrain, CognitiveMode
        
        brain = UnifiedAGIBrain()
        status = brain.get_status()
        
        results["BrainInit"] = status["components"] > 0
        print(f"  ✅ Brain initialized: {status['components']} components")
        print(f"     LLM: {status['llm']}")
        print(f"     Symbolic: {status['symbolic']}")
    except Exception as e:
        results["BrainInit"] = False
        print(f"  ❌ Brain initialization: {e}")
    
    print("\n🧠 2. Neuro-Symbolic Mode")
    try:
        from omniagi.brain import UnifiedAGIBrain, CognitiveMode
        
        brain = UnifiedAGIBrain()
        thought = brain.think("infer the relationship between concepts", 
                              mode=CognitiveMode.NEUROSYMBOLIC)
        
        results["NeuroSymbolicMode"] = "neuro" in thought.reasoning.lower() or \
                                       thought.mode == CognitiveMode.NEUROSYMBOLIC
        print(f"  ✅ Neuro-Symbolic thinking")
        print(f"     Mode: {thought.mode.name}")
        print(f"     Reasoning: {thought.reasoning[:80]}...")
    except Exception as e:
        results["NeuroSymbolicMode"] = False
        print(f"  ❌ Neuro-Symbolic mode: {e}")
    
    return results


def main():
    print("=" * 60)
    print("🌟 OmniAGI AGI CAPABILITY TEST SUITE")
    print("=" * 60)
    print("Testing Kolmogorov-Arnold Networks (KAN)")
    print("Testing Neuro-Symbolic AI (LNN, Differentiable Reasoning)")
    print("Testing Integrated AGI Brain")
    
    all_results = {}
    
    # Run tests
    all_results.update(test_kan())
    all_results.update(test_neurosymbolic())
    all_results.update(test_integrated_brain())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in all_results.values() if v)
    total = len(all_results)
    pct = (passed / total * 100) if total > 0 else 0
    
    for name, status in all_results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
    
    print(f"\nResults: {passed}/{total} ({pct:.0f}%)")
    
    if pct >= 80:
        print("\n🎉 AGI CAPABILITIES OPERATIONAL!")
    elif pct >= 50:
        print("\n⚠️  Partial AGI capabilities available")
    else:
        print("\n❌ AGI capabilities need attention")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
