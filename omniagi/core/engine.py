"""
LLM Inference Engine - Python wrapper for the Rust core or llama-cpp-python.
"""

from __future__ import annotations

import structlog
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Callable, Any

# Make llama_cpp optional
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False

from omniagi.core.config import get_config, Config

logger = structlog.get_logger()



@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    
    @classmethod
    def greedy(cls) -> "GenerationConfig":
        """Create a greedy decoding configuration."""
        return cls(temperature=0.0, top_p=1.0, top_k=0)
    
    @classmethod
    def creative(cls) -> "GenerationConfig":
        """Create a creative/sampling configuration."""
        return cls(temperature=1.0, top_p=0.95, top_k=50)


@dataclass
class GenerationOutput:
    """Output from text generation."""
    
    text: str
    tokens_generated: int
    prompt_tokens: int
    generation_time_ms: float
    tokens_per_second: float
    stopped_by_length: bool = False
    stopped_by_stop_sequence: bool = False
    stop_sequence: str | None = None


class Engine:
    """
    LLM Inference Engine.
    
    Provides a unified interface for running LLM inference,
    supporting both CPU and GPU backends.
    """
    
    def __init__(
        self,
        model_path: str | Path | None = None,
        config: Config | None = None,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model file. If None, uses config.
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self._model_path = Path(model_path) if model_path else self.config.model.path
        self._llm: Llama | None = None
        self._loaded = False
        
        logger.info(
            "Engine initialized",
            model_path=str(self._model_path),
            device=self.config.engine.device,
        )
    
    @property
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._loaded
    
    @property
    def model_path(self) -> Path | None:
        """Get the current model path."""
        return self._model_path
    
    def load(self, model_path: str | Path | None = None) -> None:
        """
        Load a model into memory.
        
        Args:
            model_path: Path to the model. Uses initialized path if None.
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama_cpp not installed. Install with: pip install llama-cpp-python")
        
        if model_path:
            self._model_path = Path(model_path)
        
        if not self._model_path:
            raise ValueError("No model path specified")
        
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")
        
        logger.info("Loading model", path=str(self._model_path))

        
        # Determine GPU layers
        n_gpu_layers = self.config.model.gpu_layers
        if self.config.engine.device == "cpu":
            n_gpu_layers = 0
        elif self.config.engine.device in ("cuda", "metal") or (
            self.config.engine.device == "auto" and n_gpu_layers == 0
        ):
            n_gpu_layers = -1  # All layers on GPU
        
        # Determine threads
        n_threads = self.config.engine.threads
        if n_threads == 0:
            import os
            n_threads = os.cpu_count() or 4
        
        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=self.config.model.context_length,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=self.config.engine.batch_size,
            verbose=self.config.log_level == "DEBUG",
        )
        
        self._loaded = True
        logger.info(
            "Model loaded",
            context_length=self.config.model.context_length,
            gpu_layers=n_gpu_layers,
            threads=n_threads,
        )
    
    def unload(self) -> None:
        """Unload the current model from memory."""
        if self._llm:
            del self._llm
            self._llm = None
        self._loaded = False
        logger.info("Model unloaded")
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationOutput:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            config: Generation configuration.
            
        Returns:
            GenerationOutput with the generated text and metadata.
        """
        if not self._loaded or not self._llm:
            raise RuntimeError("No model loaded. Call load() first.")
        
        config = config or GenerationConfig()
        
        import time
        start = time.perf_counter()
        
        output = self._llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repetition_penalty,
            stop=config.stop or None,
            echo=False,
        )
        
        elapsed = (time.perf_counter() - start) * 1000
        
        text = output["choices"][0]["text"]
        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        finish_reason = output["choices"][0].get("finish_reason", "")
        
        tokens_per_second = (
            (completion_tokens / elapsed) * 1000 if elapsed > 0 else 0
        )
        
        return GenerationOutput(
            text=text,
            tokens_generated=completion_tokens,
            prompt_tokens=prompt_tokens,
            generation_time_ms=elapsed,
            tokens_per_second=tokens_per_second,
            stopped_by_length=finish_reason == "length",
            stopped_by_stop_sequence=finish_reason == "stop",
        )
    
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """
        Generate text with streaming output.
        
        Args:
            prompt: The input prompt.
            config: Generation configuration.
            
        Yields:
            Text chunks as they are generated.
        """
        if not self._loaded or not self._llm:
            raise RuntimeError("No model loaded. Call load() first.")
        
        config = config or GenerationConfig()
        
        for output in self._llm(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repetition_penalty,
            stop=config.stop or None,
            stream=True,
        ):
            chunk = output["choices"][0]["text"]
            if chunk:
                yield chunk
    
    def chat(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> GenerationOutput:
        """
        Chat completion with message history.
        
        Args:
            messages: List of {"role": str, "content": str} messages.
            config: Generation configuration.
            
        Returns:
            GenerationOutput with the assistant's response.
        """
        if not self._loaded or not self._llm:
            raise RuntimeError("No model loaded. Call load() first.")
        
        config = config or GenerationConfig()
        
        import time
        start = time.perf_counter()
        
        output = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repetition_penalty,
            stop=config.stop or None,
        )
        
        elapsed = (time.perf_counter() - start) * 1000
        
        text = output["choices"][0]["message"]["content"]
        usage = output.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        tokens_per_second = (
            (completion_tokens / elapsed) * 1000 if elapsed > 0 else 0
        )
        
        return GenerationOutput(
            text=text,
            tokens_generated=completion_tokens,
            prompt_tokens=prompt_tokens,
            generation_time_ms=elapsed,
            tokens_per_second=tokens_per_second,
        )
    
    def embed(self, text: str) -> list[float]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        if not self._loaded or not self._llm:
            raise RuntimeError("No model loaded. Call load() first.")
        
        return self._llm.embed(text)
