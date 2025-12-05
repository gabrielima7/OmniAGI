#!/usr/bin/env python3
"""
OmniAGI - RWKV-6 Setup Script

Downloads and configures RWKV-6 for use with OmniAGI.
RWKV-6 Finch 7B requires ~6GB VRAM and is ideal for AGI.
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
MODELS_DIR = Path("models/rwkv")
RWKV_MODELS = {
    "rwkv-6-1b6": {
        "url": "https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth",
        "size": "3.2GB",
        "vram": "2GB",
    },
    "rwkv-6-3b": {
        "url": "https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth",
        "size": "6GB",
        "vram": "4GB",
    },
    "rwkv-6-7b": {
        "url": "https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/RWKV-x060-World-7B-v3-20241112-ctx4096.pth",
        "size": "14GB",
        "vram": "6GB",
    },
}


def check_dependencies():
    """Check and install dependencies."""
    print("📦 Checking dependencies...")
    
    deps = [
        ("rwkv", "rwkv"),
        ("torch", "torch"),
        ("tokenizers", "tokenizers"),
    ]
    
    missing = []
    for module, package in deps:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - will install")
            missing.append(package)
    
    if missing:
        print(f"\n📥 Installing: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *missing, "-q"
        ])
        print("  ✅ Dependencies installed")
    
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\n🖥️ Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ GPU: {gpu_name}")
            print(f"  ✅ VRAM: {vram:.1f}GB")
            return vram
        else:
            print("  ⚠️ No CUDA GPU detected - will use CPU")
            return 0
    except Exception as e:
        print(f"  ⚠️ GPU check failed: {e}")
        return 0


def recommend_model(vram_gb: float) -> str:
    """Recommend model based on VRAM."""
    if vram_gb >= 6:
        return "rwkv-6-7b"
    elif vram_gb >= 4:
        return "rwkv-6-3b"
    else:
        return "rwkv-6-1b6"


def download_model(model_name: str):
    """Download RWKV model."""
    if model_name not in RWKV_MODELS:
        print(f"❌ Unknown model: {model_name}")
        return None
    
    model_info = RWKV_MODELS[model_name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = model_info["url"].split("/")[-1]
    model_path = MODELS_DIR / filename
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return model_path
    
    print(f"\n📥 Downloading {model_name} ({model_info['size']})...")
    print(f"   Requires: {model_info['vram']} VRAM")
    print(f"   URL: {model_info['url']}")
    
    try:
        # Use wget or curl
        if os.system("which wget > /dev/null 2>&1") == 0:
            cmd = f'wget -c "{model_info["url"]}" -O "{model_path}"'
        else:
            cmd = f'curl -L -C - "{model_info["url"]}" -o "{model_path}"'
        
        print(f"   Running: {cmd[:80]}...")
        os.system(cmd)
        
        if model_path.exists():
            print(f"✅ Downloaded: {model_path}")
            return model_path
        else:
            print("❌ Download failed")
            return None
            
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None


def create_config(model_path: Path, vram_gb: float):
    """Create OmniAGI configuration for RWKV."""
    config_path = Path("config/rwkv.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    strategy = "cuda fp16" if vram_gb >= 4 else "cpu fp32"
    
    config = {
        "model_path": str(model_path),
        "strategy": strategy,
        "context_length": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "vocab": "rwkv_vocab_v20230424",
    }
    
    import json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📝 Config saved: {config_path}")
    return config_path


def test_model(model_path: Path, strategy: str = "cpu fp32"):
    """Quick test of the model."""
    print("\n🧪 Testing model...")
    
    try:
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE
        
        print("   Loading model...")
        model = RWKV(model=str(model_path), strategy=strategy)
        pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
        
        print("   Generating text...")
        result = pipeline.generate(
            "The meaning of life is",
            token_count=50,
            args={"temperature": 0.8, "top_p": 0.9},
        )
        
        print(f"\n💬 Test output:\n{result}")
        print("\n✅ Model works!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    print("=" * 50)
    print("🧠 OmniAGI - RWKV-6 Setup")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Check GPU
    vram = check_cuda()
    
    # Recommend model
    recommended = recommend_model(vram)
    print(f"\n📊 Recommended model: {recommended}")
    
    # Show options
    print("\n📋 Available models:")
    for name, info in RWKV_MODELS.items():
        marker = "⭐" if name == recommended else "  "
        print(f"  {marker} {name}: {info['size']} (needs {info['vram']} VRAM)")
    
    # Ask user
    print(f"\n🔽 Will download: {recommended}")
    print("   (Press Ctrl+C to cancel, or wait 5 seconds)")
    
    try:
        import time
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n❌ Cancelled")
        return
    
    # Download
    model_path = download_model(recommended)
    
    if model_path:
        # Create config
        strategy = "cuda fp16" if vram >= 4 else "cpu fp32"
        create_config(model_path, vram)
        
        # Test
        test_model(model_path, strategy)
        
        print("\n" + "=" * 50)
        print("✅ RWKV-6 setup complete!")
        print("=" * 50)
        print(f"\nTo use in OmniAGI:")
        print(f"  from omniagi.core.multi_llm import MultiLLM")
        print(f"  llm = MultiLLM()")
        print(f'  llm.load_model("rwkv-6-7b")')


if __name__ == "__main__":
    main()
