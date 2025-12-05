"""
Model Quantization - Reduce model size for faster inference.

RWKV supports multiple quantization strategies:
- fp16: Half precision (default on GPU)
- int8: 8-bit quantization (2x compression)
- int4/nf4: 4-bit quantization (4x compression)
"""

from __future__ import annotations

import gc
import os
import struct
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class RWKVQuantizer:
    """
    RWKV Model Quantizer.
    
    Provides optimized loading strategies for different hardware:
    - cuda fp16: Full GPU, fast
    - cuda fp16i8: GPU with int8 attention (less VRAM)
    - cuda fp16 -> cpu fp32: Hybrid GPU+CPU
    - cpu bf16: CPU with bfloat16
    - cpu fp32: CPU full precision (slow but universal)
    """
    
    # Strategy presets for different hardware
    PRESETS = {
        "gpu_fast": "cuda fp16",
        "gpu_efficient": "cuda fp16i8",  # int8 attention
        "gpu_low_vram": "cuda fp16 -> cpu fp32",
        "cpu_fast": "cpu bf16",
        "cpu_safe": "cpu fp32",
    }
    
    # Approximate VRAM usage (GB) for RWKV models
    VRAM_ESTIMATES = {
        "1b6": {"fp16": 3.0, "fp16i8": 2.0, "fp32": 6.0},
        "3b": {"fp16": 6.0, "fp16i8": 4.0, "fp32": 12.0},
        "7b": {"fp16": 14.0, "fp16i8": 9.0, "fp32": 28.0},
    }
    
    def __init__(self):
        self._model = None
        self._pipeline = None
        self._current_strategy = None
        
        logger.info("RWKV Quantizer initialized")
    
    def get_recommended_strategy(
        self,
        model_size: str,
        gpu_vram_gb: float = 0,
        ram_gb: float = 16,
    ) -> str:
        """
        Get recommended strategy based on hardware.
        
        Args:
            model_size: '1b6', '3b', or '7b'
            gpu_vram_gb: Available GPU VRAM in GB (0 = no GPU)
            ram_gb: Available RAM in GB
            
        Returns:
            Optimal loading strategy string
        """
        estimates = self.VRAM_ESTIMATES.get(model_size, self.VRAM_ESTIMATES["1b6"])
        
        if gpu_vram_gb >= estimates["fp16"] + 0.5:
            return self.PRESETS["gpu_fast"]
        
        if gpu_vram_gb >= estimates["fp16i8"] + 0.5:
            return self.PRESETS["gpu_efficient"]
        
        if gpu_vram_gb >= 2.0:
            return self.PRESETS["gpu_low_vram"]
        
        if ram_gb >= estimates["fp32"] + 2:
            return self.PRESETS["cpu_fast"]
        
        return self.PRESETS["cpu_safe"]
    
    def load_quantized(
        self,
        model_path: str,
        strategy: str = None,
        model_size: str = "3b",
    ) -> tuple:
        """
        Load model with quantization strategy.
        
        Returns: (model, pipeline, generation_args)
        """
        from omniagi.core.safe_loader import ResourceMonitor
        
        resources = ResourceMonitor.get_resources()
        
        if strategy is None:
            strategy = self.get_recommended_strategy(
                model_size,
                resources.gpu_memory_free_gb if resources.gpu_available else 0,
                resources.ram_available_gb,
            )
        
        logger.info(
            "Loading with quantization",
            strategy=strategy,
            model=model_size,
        )
        
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE, PIPELINE_ARGS
        
        self._model = RWKV(model=model_path, strategy=strategy)
        self._pipeline = PIPELINE(self._model, 'rwkv_vocab_v20230424')
        self._current_strategy = strategy
        
        args = PIPELINE_ARGS(temperature=0.7, top_p=0.9)
        
        return self._model, self._pipeline, args
    
    def benchmark_strategies(
        self,
        model_path: str,
        model_size: str = "3b",
    ) -> dict[str, dict]:
        """
        Benchmark different loading strategies.
        
        Returns dict with timing and memory for each strategy.
        """
        import time
        from omniagi.core.safe_loader import ResourceMonitor
        
        results = {}
        test_prompt = "Hello, how are you?"
        
        strategies_to_test = ["cuda fp16", "cuda fp16i8", "cuda fp16 -> cpu fp32"]
        
        for strategy in strategies_to_test:
            logger.info(f"Benchmarking: {strategy}")
            
            try:
                # Cleanup first
                self._model = None
                self._pipeline = None
                gc.collect()
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                
                # Time loading
                start_load = time.time()
                model, pipeline, args = self.load_quantized(
                    model_path, strategy, model_size
                )
                load_time = time.time() - start_load
                
                # Check memory
                resources = ResourceMonitor.get_resources()
                
                # Time inference
                start_inf = time.time()
                output = pipeline.generate(test_prompt, token_count=20, args=args)
                inf_time = time.time() - start_inf
                
                results[strategy] = {
                    "load_time_s": round(load_time, 2),
                    "inference_time_s": round(inf_time, 2),
                    "gpu_used_gb": round(resources.gpu_memory_total_gb - resources.gpu_memory_free_gb, 2),
                    "ram_used_percent": round(resources.ram_used_percent, 1),
                    "tokens_per_second": round(20 / inf_time, 1),
                    "status": "success",
                }
                
            except Exception as e:
                results[strategy] = {
                    "status": "failed",
                    "error": str(e)[:50],
                }
        
        # Cleanup
        self._model = None
        self._pipeline = None
        gc.collect()
        
        return results
    
    def unload(self):
        """Unload model and free memory."""
        self._model = None
        self._pipeline = None
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


# Global instance
_quantizer: RWKVQuantizer | None = None


def get_quantizer() -> RWKVQuantizer:
    """Get global quantizer instance."""
    global _quantizer
    if _quantizer is None:
        _quantizer = RWKVQuantizer()
    return _quantizer
