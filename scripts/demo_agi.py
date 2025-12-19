#!/usr/bin/env python3
"""
OmniAGI Comprehensive Demonstration.

Demonstrates all AGI capabilities in action.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 70)
    print("🧠 OmniAGI - COMPREHENSIVE AGI DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Initialize AGI
    print("📦 Initializing True AGI System...")
    from omniagi.agi.true_agi import TrueAGI
    
    agi = TrueAGI()
    caps = agi.get_capabilities()
    print(f"   Systems Active: {caps['total_systems']}")
    print()
    
    # Demo 1: Logical Reasoning
    print("=" * 70)
    print("🔬 DEMO 1: LOGICAL REASONING (LNN)")
    print("=" * 70)
    
    if agi.lnn:
        agi.lnn.add_predicate("mortal", 1)
        agi.lnn.add_predicate("human", 1)
        agi.lnn.add_rule("mortality", ["human(X)"], "mortal(X)")
        agi.lnn.set_fact("human", "socrates", 0.95)
        
        result = agi.lnn.infer("mortal", "socrates")
        print(f"   Premise: Humans are mortal")
        print(f"   Fact: Socrates is human (confidence: 0.95)")
        print(f"   Inference: mortal(socrates) = [{result.lower:.2f}, {result.upper:.2f}]")
        print(f"   Conclusion: Socrates is mortal: {result.is_true}")
    else:
        print("   LNN not available")
    print()
    
    # Demo 2: Common Sense Reasoning
    print("=" * 70)
    print("🌍 DEMO 2: COMMON SENSE REASONING")
    print("=" * 70)
    
    if agi.common_sense:
        scenarios = [
            "What happens if I drop a glass?",
            "What happens if I put ice in hot water?",
            "Why is someone crying?",
        ]
        
        for scenario in scenarios:
            result = agi.common_sense.reason(scenario)
            if result.get('physical_prediction'):
                pred = result['physical_prediction']['prediction']
                print(f"   Q: {scenario}")
                print(f"   A: {pred}")
            elif result.get('social_inference'):
                emotion = result['social_inference']['emotion']
                print(f"   Q: {scenario}")
                print(f"   A: Likely emotion: {emotion}")
            print()
    else:
        print("   Common Sense not available")
    print()
    
    # Demo 3: Pattern Recognition (KAN)
    print("=" * 70)
    print("📊 DEMO 3: PATTERN RECOGNITION (KAN)")
    print("=" * 70)
    
    if agi.kan:
        import torch
        
        # Generate pattern data
        x = torch.randn(5, 64)
        patterns = agi.kan(x)
        
        print(f"   Input: 5 samples of 64 dimensions")
        print(f"   Output: Pattern activations of shape {patterns.shape}")
        print(f"   Mean activation: {patterns.mean().item():.4f}")
        print(f"   Pattern variance: {patterns.var().item():.4f}")
    else:
        print("   KAN not available")
    print()
    
    # Demo 4: Embodied Simulation
    print("=" * 70)
    print("🤖 DEMO 4: EMBODIED SIMULATION")
    print("=" * 70)
    
    if agi.embodiment:
        agi.embodiment.setup_world([
            {'id': 'ball', 'name': 'red ball', 'x': 2, 'y': 1, 'z': 0, 'mass': 0.5},
            {'id': 'cube', 'name': 'blue cube', 'x': -1, 'y': 0, 'z': 0, 'mass': 1.0},
        ])
        
        obs = agi.embodiment.observe()
        print(f"   World objects: {len(obs['visible_objects'])}")
        
        for obj in obs['visible_objects']:
            pos = obj['position']
            print(f"      - {obj['name']} at position ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        
        action_result = agi.embodiment.act('move', direction=[1, 0, 0], magnitude=1.0)
        print(f"\n   Action: Move right")
        print(f"   Result: {action_result['feedback']}")
        
        new_obs = agi.embodiment.observe()
        new_pos = new_obs['position']
        print(f"   New position: ({new_pos[0]:.1f}, {new_pos[1]:.1f}, {new_pos[2]:.1f})")
    else:
        print("   Embodiment not available")
    print()
    
    # Demo 5: Hierarchical Planning
    print("=" * 70)
    print("📋 DEMO 5: HIERARCHICAL PLANNING")
    print("=" * 70)
    
    if agi.planner:
        goal = "Build a website"
        plan = agi.planner.create_plan(goal)
        
        print(f"   Goal: {goal}")
        print(f"   Tasks generated: {plan['tasks']}")
        print(f"   Deadline feasible: {plan['deadline_feasible']}")
        
        next_task = agi.planner.get_next_task()
        if next_task:
            print(f"   Next task: {next_task.name}")
    else:
        print("   Planner not available")
    print()
    
    # Demo 6: Online Learning
    print("=" * 70)
    print("🎓 DEMO 6: ONLINE LEARNING (MAML)")
    print("=" * 70)
    
    if getattr(agi, 'online_learner', None):
        import torch
        
        print("   Training on 10 examples...")
        for i in range(10):
            loss = agi.learn_online(
                torch.randn(64).tolist(),
                torch.randn(32).tolist(),
                f"task_{i % 3}"
            )
        
        stats = agi.online_learner.get_stats()
        print(f"   Examples learned: {stats['total_examples']}")
        print(f"   Running loss: {stats['online_stats']['running_loss']:.4f}")
        print(f"   Learning rate: {stats['online_stats']['learning_rate']:.6f}")
    else:
        print("   Online Learner not available")
    print()
    
    # Demo 7: Zero-Shot Transfer
    print("=" * 70)
    print("🔄 DEMO 7: ZERO-SHOT TRANSFER")
    print("=" * 70)
    
    if getattr(agi, 'zero_shot', None):
        from omniagi.transfer.zero_shot import Task
        
        # Register a task
        task = Task(
            id="double",
            name="double",
            description="Double the input number",
            input_type="int",
            output_type="int",
            examples=[(2, 4), (3, 6), (5, 10)],
        )
        agi.zero_shot.register_task(task, lambda x: x * 2)
        
        # Test analogical reasoning
        result = agi.zero_shot.analogical.find_analogy(1, 2, 5)
        print(f"   Registered task: 'double'")
        print(f"   Analogy: 1 is to 2 as 5 is to ?")
        print(f"   Answer: {result}")
        
        stats = agi.zero_shot.get_stats()
        print(f"   Tasks known: {stats['known_tasks']}")
    else:
        print("   Zero-Shot Transfer not available")
    print()
    
    # Demo 8: Integrated Thinking
    print("=" * 70)
    print("🧠 DEMO 8: INTEGRATED THINKING")
    print("=" * 70)
    
    queries = [
        "What happens if I drop a heavy rock into water?",
        "Plan to learn machine learning",
        "What patterns exist in the sequence 1, 1, 2, 3, 5, 8?",
    ]
    
    for query in queries:
        thought = agi.think(query)
        print(f"\n   Query: {query}")
        print(f"   Systems used: {thought.systems_used}")
        print(f"   Response: {thought.integrated_response[:80]}...")
        print(f"   Confidence: {thought.confidence:.2f}")
        print(f"   Time: {thought.processing_time_ms}ms")
    print()
    
    # Final Evaluation
    print("=" * 70)
    print("📊 FINAL AGI EVALUATION")
    print("=" * 70)
    
    eval_result = agi.evaluate_agi_level()
    
    print(f"\n   🎯 AGI Level: {eval_result['level']}")
    print(f"   📈 Average Score: {eval_result['average']:.1f}%")
    print(f"   🔧 Systems Active: {eval_result['systems_active']}")
    print()
    print("   Capability Breakdown:")
    for cap, score in eval_result['scores'].items():
        bar = '█' * (score // 5) + '░' * (20 - score // 5)
        print(f"      {cap}: [{bar}] {score}%")
    
    print()
    print("=" * 70)
    if eval_result['average'] >= 95:
        print("🎉 TRUE AGI OPERATIONAL!")
    elif eval_result['average'] >= 80:
        print("✅ ADVANCED AGI OPERATIONAL!")
    else:
        print("⚠️ PROTO-AGI STATUS")
    print("=" * 70)


if __name__ == "__main__":
    main()
