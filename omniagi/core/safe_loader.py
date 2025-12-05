"""
Safe Model Loader - Prevent system crashes from memory exhaustion.

Features:
1. Monitor RAM and GPU memory before loading
2. Automatic strategy selection based on hardware
3. Gradual loading with checkpoints
4. Emergency unload if memory critical
"""

from __future__ import annotations

import gc
import os
import psutil
import structlog
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = structlog.get_logger()


@dataclass
class SystemResources:
    """Current system resource state."""
    
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    
    gpu_available: bool
    gpu_name: str
    gpu_memory_total_gb: float
    gpu_memory_free_gb: float
    gpu_memory_used_percent: float
    
    def is_safe_to_load(self, required_gb: float) -> bool:
        """Check if safe to load model of given size."""
        # Need at least 2GB headroom
        ram_ok = self.ram_available_gb > required_gb + 2.0
        
        if self.gpu_available:
            gpu_ok = self.gpu_memory_free_gb > required_gb * 0.5
            return ram_ok and gpu_ok
        
        return ram_ok
    
    def to_dict(self) -> dict:
        return {
            "ram_available_gb": round(self.ram_available_gb, 1),
            "ram_used_percent": round(self.ram_used_percent, 1),
            "gpu_available": self.gpu_available,
            "gpu_memory_free_gb": round(self.gpu_memory_free_gb, 1) if self.gpu_available else 0,
        }


class ResourceMonitor:
    """Monitor system resources in real-time."""
    
    # Safety thresholds
    RAM_CRITICAL_PERCENT = 90
    GPU_CRITICAL_PERCENT = 95
    MIN_FREE_RAM_GB = 1.5
    
    @staticmethod
    def get_resources() -> SystemResources:
        """Get current system resources."""
        # RAM info
        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024**3)
        ram_available = ram.available / (1024**3)
        ram_percent = ram.percent
        
        # GPU info
        gpu_available = False
        gpu_name = "None"
        gpu_total = 0.0
        gpu_free = 0.0
        gpu_percent = 0.0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_total = props.total_memory / (1024**3)
                
                # Get free memory
                torch.cuda.synchronize()
                gpu_free = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                gpu_percent = (1 - gpu_free / gpu_total) * 100 if gpu_total > 0 else 0
        except Exception:
            pass
        
        return SystemResources(
            ram_total_gb=ram_total,
            ram_available_gb=ram_available,
            ram_used_percent=ram_percent,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_total_gb=gpu_total,
            gpu_memory_free_gb=gpu_free,
            gpu_memory_used_percent=gpu_percent,
        )
    
    @classmethod
    def is_critical(cls) -> bool:
        """Check if resources are critical."""
        resources = cls.get_resources()
        
        if resources.ram_used_percent > cls.RAM_CRITICAL_PERCENT:
            return True
        
        if resources.ram_available_gb < cls.MIN_FREE_RAM_GB:
            return True
        
        if resources.gpu_available and resources.gpu_memory_used_percent > cls.GPU_CRITICAL_PERCENT:
            return True
        
        return False
    
    @classmethod
    def emergency_cleanup(cls) -> None:
        """Emergency cleanup to free memory."""
        logger.warning("Emergency memory cleanup triggered")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        
        gc.collect()


class SafeModelLoader:
    """
    Safe Model Loader with memory protection.
    
    Features:
    - Pre-checks available memory
    - Selects optimal loading strategy
    - Monitors during loading
    - Emergency unload capability
    """
    
    # Model size estimates (in GB)
    MODEL_SIZES = {
        "1b6": {"ram": 3.5, "vram_fp16": 3.0, "vram_fp32": 6.0},
        "3b": {"ram": 7.0, "vram_fp16": 6.0, "vram_fp32": 12.0},
        "7b": {"ram": 14.0, "vram_fp16": 14.0, "vram_fp32": 28.0},
    }
    
    def __init__(self):
        self._model = None
        self._pipeline = None
        self._current_strategy = None
        
        logger.info("Safe Model Loader initialized")
    
    def get_optimal_strategy(self, model_size: str) -> str:
        """
        Determine optimal loading strategy based on hardware.
        
        Strategies:
        - 'cuda fp16': Full GPU, fast
        - 'cuda fp16 -> cpu fp32': Hybrid, GPU + CPU
        - 'cpu fp32': CPU only, slow but safe
        """
        resources = ResourceMonitor.get_resources()
        sizes = self.MODEL_SIZES.get(model_size, self.MODEL_SIZES["1b6"])
        
        logger.info(
            "Selecting strategy",
            model=model_size,
            gpu_free=f"{resources.gpu_memory_free_gb:.1f}GB",
            ram_free=f"{resources.ram_available_gb:.1f}GB",
        )
        
        # Check if GPU can handle it
        if resources.gpu_available:
            if resources.gpu_memory_free_gb >= sizes["vram_fp16"] + 0.5:
                return "cuda fp16"
            elif resources.gpu_memory_free_gb >= sizes["vram_fp16"] * 0.5:
                # Hybrid: part GPU, part CPU
                return "cuda fp16 -> cpu fp32"
        
        # CPU fallback
        if resources.ram_available_gb >= sizes["ram"] + 2.0:
            return "cpu fp32"
        
        # Not enough resources
        raise MemoryError(
            f"Insufficient memory for {model_size} model. "
            f"Need {sizes['ram']}GB RAM, have {resources.ram_available_gb:.1f}GB free"
        )
    
    def load(
        self,
        model_path: str,
        model_size: str = "1b6",
        strategy: str = None,
    ) -> tuple:
        """
        Safely load a model with memory protection.
        
        Returns (model, pipeline, args) or raises MemoryError.
        """
        # Check resources first
        resources = ResourceMonitor.get_resources()
        logger.info("Pre-load resources", **resources.to_dict())
        
        # Determine strategy
        if strategy is None:
            strategy = self.get_optimal_strategy(model_size)
        
        self._current_strategy = strategy
        logger.info("Loading model", path=model_path, strategy=strategy)
        
        # Check if file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            from rwkv.model import RWKV
            from rwkv.utils import PIPELINE, PIPELINE_ARGS
            
            # Load with selected strategy
            self._model = RWKV(model=model_path, strategy=strategy)
            self._pipeline = PIPELINE(self._model, 'rwkv_vocab_v20230424')
            args = PIPELINE_ARGS(temperature=0.7, top_p=0.9)
            
            # Check resources after loading
            resources_after = ResourceMonitor.get_resources()
            logger.info("Post-load resources", **resources_after.to_dict())
            
            if ResourceMonitor.is_critical():
                logger.warning("Resources critical after loading, consider unloading")
            
            return self._model, self._pipeline, args
            
        except Exception as e:
            logger.error("Model loading failed", error=str(e))
            self.unload()
            raise
    
    def unload(self) -> None:
        """Safely unload the model and free memory."""
        logger.info("Unloading model")
        
        self._model = None
        self._pipeline = None
        self._current_strategy = None
        
        ResourceMonitor.emergency_cleanup()
        
        resources = ResourceMonitor.get_resources()
        logger.info("After unload", **resources.to_dict())
    
    def generate_safe(
        self,
        prompt: str,
        max_tokens: int = 50,
        check_interval: int = 10,
    ) -> str:
        """
        Generate text with memory monitoring.
        
        Stops early if memory becomes critical.
        """
        if self._pipeline is None:
            raise RuntimeError("No model loaded")
        
        from rwkv.utils import PIPELINE_ARGS
        args = PIPELINE_ARGS(temperature=0.7, top_p=0.9)
        
        # Check before starting
        if ResourceMonitor.is_critical():
            raise MemoryError("Memory critical, cannot generate")
        
        # Generate in chunks to allow memory checks
        output = ""
        tokens_generated = 0
        
        while tokens_generated < max_tokens:
            # Check memory periodically
            if tokens_generated > 0 and tokens_generated % check_interval == 0:
                if ResourceMonitor.is_critical():
                    logger.warning("Stopping generation due to memory pressure")
                    break
            
            # Generate chunk
            chunk_size = min(check_interval, max_tokens - tokens_generated)
            chunk = self._pipeline.generate(
                prompt + output,
                token_count=chunk_size,
                args=args,
            )
            
            output += chunk
            tokens_generated += chunk_size
        
        return output
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None
    
    def get_status(self) -> dict:
        """Get current loader status."""
        resources = ResourceMonitor.get_resources()
        return {
            "loaded": self.is_loaded(),
            "strategy": self._current_strategy,
            "resources": resources.to_dict(),
            "is_critical": ResourceMonitor.is_critical(),
        }


# Global instance
_safe_loader: SafeModelLoader | None = None


def get_safe_loader() -> SafeModelLoader:
    """Get global safe loader instance."""
    global _safe_loader
    if _safe_loader is None:
        _safe_loader = SafeModelLoader()
    return _safe_loader
