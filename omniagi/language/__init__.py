"""Language model module."""

from omniagi.language.rwkv_model import (
    RWKVInterface,
    LanguageModelManager,
    SimpleLanguageModel,
    GenerationConfig,
    GenerationResult,
)

try:
    from omniagi.language.cloud_llm import (
        HybridLLM,
        GroqClient,
        TogetherClient,
        OpenRouterClient,
        OllamaClient,
        setup_instructions,
    )
    CLOUD_LLM_AVAILABLE = True
except ImportError:
    CLOUD_LLM_AVAILABLE = False
    HybridLLM = None

__all__ = [
    "RWKVInterface",
    "LanguageModelManager",
    "SimpleLanguageModel",
    "GenerationConfig",
    "GenerationResult",
    "HybridLLM",
    "GroqClient",
    "TogetherClient",
    "OpenRouterClient",
    "OllamaClient",
    "setup_instructions",
]
