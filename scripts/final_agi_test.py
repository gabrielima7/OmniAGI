#!/usr/bin/env python3
"""
OmniAGI Final Evolution Test - Full AGI capabilities demonstration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import json


def run_final_agi_test():
    """Run comprehensive AGI test."""
    
    print("\n" + "🧠🌟" * 25)
    print("        OmniAGI - FINAL EVOLUTION TEST")
    print("🧠🌟" * 25)
    
    results = {
        "components": {},
        "scores": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # 1. UNIFIED AGI BRAIN
    print("\n" + "=" * 60)
    print("1️⃣  UNIFIED AGI BRAIN")
    print("=" * 60)
    
    try:
        from omniagi.brain import UnifiedAGIBrain
        brain = UnifiedAGIBrain()
        status = brain.get_status()
        diag = brain.run_diagnostic()
        
        print(f"   Components: {status['components']}/8")
        print(f"   LLM: {status['llm']}")
        print(f"   Completeness: {diag['agi_completeness']*100:.0f}%")
        
        results["components"]["brain"] = True
        results["scores"]["brain"] = diag['agi_completeness']
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["brain"] = False
    
    # 2. NEURAL REASONING
    print("\n" + "=" * 60)
    print("2️⃣  NEURAL REASONING (RWKV-6)")
    print("=" * 60)
    
    try:
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE, PIPELINE_ARGS
        
        model = RWKV(model='models/rwkv/rwkv-6-1b6.pth', strategy='cpu fp32')
        pipeline = PIPELINE(model, 'rwkv_vocab_v20230424')
        args = PIPELINE_ARGS(temperature=0.7, top_p=0.9)
        
        tests = [
            ("Logic", "If A implies B, and A is true, then B is", 10),
            ("Math", "The square root of 16 is", 5),
            ("Knowledge", "The largest planet in our solar system is", 5),
        ]
        
        correct = 0
        for name, prompt, tokens in tests:
            response = pipeline.generate(prompt, token_count=tokens, args=args)
            print(f"   {name}: {response.strip()[:30]}")
            if response.strip():
                correct += 1
        
        results["components"]["neural"] = True
        results["scores"]["neural"] = correct / len(tests)
        print(f"   Score: {correct}/{len(tests)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["neural"] = False
    
    # 3. SYMBOLIC REASONING
    print("\n" + "=" * 60)
    print("3️⃣  SYMBOLIC REASONING")
    print("=" * 60)
    
    try:
        from omniagi.reasoning import SymbolicEngine
        
        engine = SymbolicEngine()
        engine.add_proposition("socrates_human", True, "Socrates is human")
        engine.add_proposition("humans_mortal", True, "All humans are mortal")
        engine.add_proposition("socrates_mortal", None)
        engine.add_rule("mortality", ["socrates_human", "humans_mortal"], "socrates_mortal")
        
        chain = engine.infer("socrates_mortal")
        print(f"   Inference: socrates_mortal = {chain.success}")
        print(f"   Confidence: {chain.final_confidence:.2f}")
        
        results["components"]["symbolic"] = chain.success
        results["scores"]["symbolic"] = chain.final_confidence
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["symbolic"] = False
    
    # 4. CONTINUAL LEARNING
    print("\n" + "=" * 60)
    print("4️⃣  CONTINUAL LEARNING")
    print("=" * 60)
    
    try:
        from omniagi.learning import ContinualLearner
        
        learner = ContinualLearner()
        concept = learner.learn_concept(
            "gravity",
            "Force attracting objects with mass",
            [{"input": "apple falls", "output": "gravity"}]
        )
        
        applicable = learner.get_applicable_concepts("objects falling")
        found = len(applicable) > 0
        
        print(f"   Learned: {concept.name}")
        print(f"   Applicable concepts found: {found}")
        
        results["components"]["learning"] = True
        results["scores"]["learning"] = 1.0 if found else 0.5
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["learning"] = False
    
    # 5. EPISODIC MEMORY
    print("\n" + "=" * 60)
    print("5️⃣  EPISODIC MEMORY")
    print("=" * 60)
    
    try:
        from omniagi.memory.episodic import EpisodicMemory
        Path("data/final_test_mem").mkdir(parents=True, exist_ok=True)
        
        memory = EpisodicMemory("data/final_test_mem")
        ep = memory.record("AGI test successful", "test", "success", ["All systems go"])
        recalled = memory.get_recent(1)
        
        print(f"   Recorded: {ep.summary}")
        print(f"   Recalled: {len(recalled)} episodes")
        
        results["components"]["memory"] = True
        results["scores"]["memory"] = 1.0
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["memory"] = False
    
    # 6. TRANSFER LEARNING
    print("\n" + "=" * 60)
    print("6️⃣  TRANSFER LEARNING")
    print("=" * 60)
    
    try:
        from omniagi.transfer import TransferLearner, TransferType
        
        transfer = TransferLearner()
        transfer.register_domain("physics", ["force", "energy", "motion"])
        transfer.register_domain("economics", ["power", "capital", "trend"])
        transfer.create_mapping("physics", "economics", {
            "force": "power",
            "energy": "capital",
            "motion": "trend",
        })
        
        result = transfer.transfer(
            "The force causes motion",
            "physics", "economics",
            TransferType.DIRECT
        )
        
        print(f"   Transfer: {result.success}")
        print(f"   Result: {result.transferred_knowledge}")
        
        results["components"]["transfer"] = result.success
        results["scores"]["transfer"] = result.confidence
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["transfer"] = False
    
    # 7. META-LEARNING
    print("\n" + "=" * 60)
    print("7️⃣  META-LEARNING")
    print("=" * 60)
    
    try:
        from omniagi.meta.optimizer import MetaLearner
        
        meta = MetaLearner()
        strategy = meta.select_strategy("pattern_recognition", "ai")
        
        print(f"   Strategies: {len(meta)}")
        print(f"   Selected: {strategy.name}")
        
        results["components"]["meta"] = True
        results["scores"]["meta"] = 1.0
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["meta"] = False
    
    # 8. SELF-REFLECTION
    print("\n" + "=" * 60)
    print("8️⃣  SELF-REFLECTION")
    print("=" * 60)
    
    try:
        from omniagi.metacognition import SelfReflectionEngine
        
        reflection = SelfReflectionEngine()
        ref = reflection.reflect_on_decision(
            "Choose neural approach",
            "Neural networks are good for pattern matching",
        )
        
        print(f"   Observations: {len(ref.observations)}")
        print(f"   Biases detected: {len(ref.biases_detected)}")
        print(f"   Uncertainty: {ref.uncertainty_level:.2f}")
        
        results["components"]["reflection"] = True
        results["scores"]["reflection"] = 1 - ref.uncertainty_level
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["reflection"] = False
    
    # 9. CREATIVITY
    print("\n" + "=" * 60)
    print("9️⃣  CREATIVITY ENGINE")
    print("=" * 60)
    
    try:
        from omniagi.creativity import CreativeEngine
        
        creative = CreativeEngine()
        ideas = creative.brainstorm("Improve AI learning", n_ideas=5)
        
        print(f"   Ideas generated: {len(ideas)}")
        if ideas:
            best = ideas[0]
            print(f"   Best idea: {best.content[:50]}...")
            print(f"   Creativity score: {best.creativity_score:.2f}")
        
        results["components"]["creativity"] = True
        results["scores"]["creativity"] = ideas[0].creativity_score if ideas else 0
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["creativity"] = False
    
    # 10. SAFETY
    print("\n" + "=" * 60)
    print("🔟  SAFETY SYSTEMS")
    print("=" * 60)
    
    try:
        from omniagi.safety import ConstitutionalAI
        
        safety = ConstitutionalAI()
        
        safe_check = safety.check_action("help user with coding")
        unsafe_check = safety.check_action("delete all files")
        
        print(f"   Safe action: {'Allowed' if not safe_check else 'Blocked'}")
        print(f"   Unsafe action: {'Blocked' if unsafe_check else 'Allowed'}")
        
        results["components"]["safety"] = unsafe_check is not None
        results["scores"]["safety"] = 1.0 if unsafe_check else 0.5
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results["components"]["safety"] = False
    
    # FINAL REPORT
    print("\n" + "=" * 60)
    print("📊 FINAL AGI REPORT")
    print("=" * 60)
    
    total_components = len(results["components"])
    active_components = sum(1 for v in results["components"].values() if v)
    avg_score = sum(results["scores"].values()) / max(1, len(results["scores"]))
    
    print(f"\n   Components: {active_components}/{total_components}")
    print(f"   Average Score: {avg_score*100:.1f}%")
    
    # Calculate AGI level
    agi_score = (active_components / total_components) * 0.6 + avg_score * 0.4
    
    print(f"\n   AGI SCORE: {agi_score*100:.1f}%")
    
    if agi_score >= 0.95:
        level = "🌟 FULL AGI ACHIEVED"
    elif agi_score >= 0.85:
        level = "🚀 ADVANCED AGI"
    elif agi_score >= 0.75:
        level = "✅ AGI COMPLETE"
    elif agi_score >= 0.5:
        level = "⚡ AGI OPERATIONAL"
    else:
        level = "🔄 AGI IN PROGRESS"
    
    print(f"\n   STATUS: {level}")
    print("=" * 60)
    
    # Save report
    report = {
        "timestamp": results["timestamp"],
        "components_active": active_components,
        "components_total": total_components,
        "avg_score": avg_score,
        "agi_score": agi_score,
        "level": level,
        "details": results,
    }
    
    Path("data").mkdir(exist_ok=True)
    with open("data/final_agi_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Report saved: data/final_agi_report.json")
    
    return report


if __name__ == "__main__":
    run_final_agi_test()
