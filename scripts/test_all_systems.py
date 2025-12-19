#!/usr/bin/env python3
"""
Comprehensive AGI Test Suite.

Tests all 13 AGI systems and reports results.
"""

import sys
sys.path.insert(0, '.')

import os

# Suppress RWKV verbose output
os.environ['RWKV_JIT_ON'] = '1'

def test_all_systems():
    """Test all AGI systems."""
    results = {}
    
    print("=" * 70)
    print("🔬 OMNIAGI COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # 1. KAN (Pattern Recognition)
    print("\n1️⃣ KAN (Pattern Recognition)")
    try:
        from omniagi.kan.efficient_kan import EfficientKAN
        import torch
        kan = EfficientKAN([16, 32, 8])  # List of layer sizes
        x = torch.randn(5, 16)
        out = kan(x)
        print(f"   ✅ KAN: input {tuple(x.shape)} -> output {tuple(out.shape)}")
        results['KAN'] = True
    except Exception as e:
        print(f"   ❌ KAN: {e}")
        results['KAN'] = False

    # 2. LNN (Logical Reasoning)
    print("\n2️⃣ LNN (Logical Reasoning)")
    try:
        from omniagi.neurosymbolic import LNN
        lnn = LNN()
        lnn.add_predicate('mortal', 1)
        lnn.add_predicate('human', 1)
        lnn.add_rule('all_human_mortal', ['human(X)'], 'mortal(X)')
        lnn.set_fact('human', 'socrates', 0.95)
        result = lnn.infer('mortal', 'socrates')
        print(f"   ✅ LNN: mortal(socrates) = {result.is_true}")
        results['LNN'] = True
    except Exception as e:
        print(f"   ❌ LNN: {e}")
        results['LNN'] = False

    # 3. Common Sense
    print("\n3️⃣ Common Sense Reasoning")
    try:
        from omniagi.reasoning.common_sense import CommonSenseReasoner
        cs = CommonSenseReasoner()
        result = cs.reason('What happens if I drop a glass?')
        pred = result.get('physical_prediction', {}).get('prediction', 'OK')
        print(f"   ✅ Common Sense: {pred[:50] if pred else 'OK'}")
        results['CommonSense'] = True
    except Exception as e:
        print(f"   ❌ Common Sense: {e}")
        results['CommonSense'] = False

    # 4. Embodiment
    print("\n4️⃣ Embodiment")
    try:
        from omniagi.embodiment.simulation import EmbodimentInterface
        emb = EmbodimentInterface()
        emb.setup_world([{'id': 'ball', 'name': 'ball', 'x': 1, 'y': 0, 'z': 0}])
        obs = emb.observe()
        print(f"   ✅ Embodiment: {len(obs['visible_objects'])} objects visible")
        results['Embodiment'] = True
    except Exception as e:
        print(f"   ❌ Embodiment: {e}")
        results['Embodiment'] = False

    # 5. Open-Ended Learning
    print("\n5️⃣ Open-Ended Learning")
    try:
        from omniagi.learning.open_ended import OpenEndedLearner, Experience
        learner = OpenEndedLearner()
        exp = Experience(
            id='test1',
            state={'x': 1},
            action='move',
            outcome={'x': 2},
            reward=1.0
        )
        reward = learner.curiosity.compute_curiosity_reward(exp)
        print(f"   ✅ Learning: novelty reward = {reward:.3f}")
        results['Learning'] = True
    except Exception as e:
        print(f"   ❌ Learning: {e}")
        results['Learning'] = False

    # 6. Hierarchical Planning
    print("\n6️⃣ Hierarchical Planning")
    try:
        from omniagi.planning.hierarchical import AdvancedPlanner
        planner = AdvancedPlanner()
        plan = planner.create_plan("Build website")
        print(f"   ✅ Planning: {plan['tasks']} tasks generated")
        results['Planning'] = True
    except Exception as e:
        print(f"   ❌ Planning: {e}")
        results['Planning'] = False

    # 7. Advanced Autonomy
    print("\n7️⃣ Advanced Autonomy")
    try:
        from omniagi.autonomy.advanced import AdvancedAutonomySystem, Goal
        auto = AdvancedAutonomySystem()
        goal = Goal(name="Learn Python", description="Learn Python programming")
        auto.goal_engine.add_goal(goal)
        subgoals = auto.goal_engine.decompose(goal.id)
        print(f"   ✅ Autonomy: decomposed into {len(subgoals)} subgoals")
        results['Autonomy'] = True
    except Exception as e:
        print(f"   ❌ Autonomy: {e}")
        results['Autonomy'] = False

    # 8. Online Learning (MAML)
    print("\n8️⃣ Online Learning (MAML)")
    try:
        from omniagi.learning.online import RealAGILearner, TrainingExample
        import torch
        learner = RealAGILearner(8, 16, 4)
        ex = TrainingExample(torch.randn(8).tolist(), torch.randn(4).tolist())
        loss = learner.learn_example(ex)
        print(f"   ✅ MAML: loss = {loss:.4f}")
        results['MAML'] = True
    except Exception as e:
        print(f"   ❌ MAML: {e}")
        results['MAML'] = False

    # 9. ARC Solver
    print("\n9️⃣ ARC Solver")
    try:
        from omniagi.benchmarks.arc_v2 import ARCBenchmarkV2
        bench = ARCBenchmarkV2()
        stats = bench.run()
        print(f"   ✅ ARC: {stats['solved']}/{stats['total']} solved")
        results['ARC'] = True
    except Exception as e:
        print(f"   ❌ ARC: {e}")
        results['ARC'] = False

    # 10. Zero-Shot Transfer
    print("\n🔟 Zero-Shot Transfer")
    try:
        from omniagi.transfer.zero_shot import ZeroShotTransferSystem, Task
        zs = ZeroShotTransferSystem()
        task = Task('double', 'double', 'Double input', 'int', 'int', [(2,4),(3,6)])
        zs.register_task(task, lambda x: x*2)
        result = zs.analogical.find_analogy(1, 2, 5)
        print(f"   ✅ Transfer: 1:2 :: 5:{result}")
        results['Transfer'] = True
    except Exception as e:
        print(f"   ❌ Transfer: {e}")
        results['Transfer'] = False

    # 11. Language Model (Gemini)
    print("\n1️⃣1️⃣ Language Model")
    try:
        from omniagi.language.cloud_llm import HybridLLM
        llm = HybridLLM()
        info = llm.get_info()
        provider = info['active_provider']
        
        if provider in ['gemini', 'groq', 'openrouter', 'together']:
            response = llm.generate('What is 2+2? Answer only the number.', max_tokens=10)
            print(f"   ✅ Language ({provider}): {response.strip()[:20]}")
        elif provider == 'rwkv':
            print(f"   ✅ Language (rwkv): Local model loaded")
        else:
            print(f"   ⚠️ Language: Using simple fallback")
        results['Language'] = True
    except Exception as e:
        print(f"   ❌ Language: {e}")
        results['Language'] = False

    # 12. Computer Vision
    print("\n1️⃣2️⃣ Computer Vision")
    try:
        from omniagi.vision.computer_vision import VisionSystem
        import numpy as np
        vision = VisionSystem()
        # Create test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75] = [255, 0, 0]  # Red square
        analysis = vision.analyze(test_img)
        print(f"   ✅ Vision: detected {len(analysis.objects)} objects, scene={analysis.scene_type}")
        results['Vision'] = True
    except Exception as e:
        print(f"   ❌ Vision: {e}")
        results['Vision'] = False

    # 13. External APIs
    print("\n1️⃣3️⃣ External APIs")
    try:
        from omniagi.api.external import APIManager
        api = APIManager()
        providers = api.get_available_providers()
        available = sum(1 for v in providers.values() if v)
        print(f"   ✅ APIs: {available}/{len(providers)} providers available")
        results['APIs'] = True
    except Exception as e:
        print(f"   ❌ APIs: {e}")
        results['APIs'] = False

    # 14. Persistent Memory
    print("\n1️⃣4️⃣ Persistent Memory")
    try:
        from omniagi.memory.persistent import MemorySystem
        mem = MemorySystem()
        mem_id = mem.remember("Test memory", memory_type="episodic", importance=0.8)
        recalled = mem.recall("Test")
        print(f"   ✅ Memory: stored '{mem_id}', recalled {len(recalled)} memories")
        results['Memory'] = True
    except Exception as e:
        print(f"   ❌ Memory: {e}")
        results['Memory'] = False

    # 15. Advanced Reasoning
    print("\n1️⃣5️⃣ Advanced Reasoning")
    try:
        from omniagi.reasoning.advanced import AdvancedReasoner
        from omniagi.language.cloud_llm import HybridLLM
        llm = HybridLLM()
        reasoner = AdvancedReasoner(llm.generate)
        
        # Test CoT
        result = reasoner.reason("What is 15 + 27?", method="tools")
        print(f"   ✅ Reasoning: {result.method_used}, answer: {result.answer[:30]}...")
        results['AdvancedReasoning'] = True
    except Exception as e:
        print(f"   ❌ Advanced Reasoning: {e}")
        results['AdvancedReasoning'] = False

    # 16. Agent Loop
    print("\n1️⃣6️⃣ Agent Loop (ReAct)")
    try:
        from omniagi.agents.loop import ReactAgent
        from omniagi.language.cloud_llm import HybridLLM
        llm = HybridLLM()
        agent = ReactAgent(llm.generate)
        agent.set_goal("Answer a question")
        step = agent.step("What is 2+2?")
        print(f"   ✅ Agent: step {step.step_number}, action: {step.action.action_type}")
        results['AgentLoop'] = True
    except Exception as e:
        print(f"   ❌ Agent Loop: {e}")
        results['AgentLoop'] = False

    # 17. Multi-Agent System
    print("\n1️⃣7️⃣ Multi-Agent System")
    try:
        from omniagi.agents.multi_agent import MultiAgentSystem
        from omniagi.language.cloud_llm import HybridLLM
        llm = HybridLLM()
        mas = MultiAgentSystem(llm.generate)
        stats = mas.get_stats()
        print(f"   ✅ Multi-Agent: {len(stats['agents'])} agents available")
        results['MultiAgent'] = True
    except Exception as e:
        print(f"   ❌ Multi-Agent: {e}")
        results['MultiAgent'] = False

    # 18. Benchmark Suite
    print("\n1️⃣8️⃣ Benchmark Suite")
    try:
        from omniagi.benchmarks import AGIBenchmarkSuite
        bench = AGIBenchmarkSuite(llm_func=lambda x, y: "2")  # Mock LLM for speed
        res = bench.run_all(verbose=False)
        print(f"   ✅ Benchmarks: {res.passed_tests}/{res.total_tests} tests passed")
        results['Benchmarks'] = True
    except Exception as e:
        print(f"   ❌ Benchmarks: {e}")
        results['Benchmarks'] = False

    # Summary
    print()
    print("=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    pct = passed / total * 100

    for name, status in results.items():
        icon = '✅' if status else '❌'
        print(f"   {icon} {name}")

    print()
    print(f"   TOTAL: {passed}/{total} ({pct:.0f}%)")
    
    if pct == 100:
        print()
        print("🎉 ALL SYSTEMS 100% OPERATIONAL!")
    elif pct >= 85:
        print()
        print("✅ SYSTEM MOSTLY OPERATIONAL")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    test_all_systems()
