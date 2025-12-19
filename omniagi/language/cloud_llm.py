"""
Cloud LLM Integration.

Provides access to powerful LLMs via free/cheap cloud APIs:
- Groq (free, very fast)
- Together AI (free tier)
- OpenRouter (many models)
- Ollama (local alternative)

This allows using 7B, 70B+ models without local hardware.
"""

from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    model: str
    tokens_used: int
    success: bool
    error: Optional[str] = None


class GroqClient:
    """
    Groq API client - FREE and VERY FAST.
    
    Provides access to:
    - LLaMA-3.1-70B (free)
    - LLaMA-3.1-8B (free)
    - Gemma-2-9B (free)
    - Mixtral-8x7B (free)
    
    Sign up at: https://console.groq.com
    Set GROQ_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.default_model = "llama-3.1-8b-instant"  # Fast and good
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> LLMResponse:
        """Generate text using Groq."""
        if not HTTPX_AVAILABLE:
            return LLMResponse("", "", 0, False, "httpx not available")
        
        if not self.api_key:
            return LLMResponse("", "", 0, False, "GROQ_API_KEY not set")
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.default_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    return LLMResponse(text, self.default_model, tokens, True)
                else:
                    return LLMResponse("", "", 0, False, response.text)
                    
        except Exception as e:
            return LLMResponse("", "", 0, False, str(e))


class TogetherClient:
    """
    Together AI client - Free tier available.
    
    Provides access to many open source models.
    Sign up at: https://together.ai
    Set TOGETHER_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.base_url = "https://api.together.xyz/v1"
        self.default_model = "meta-llama/Llama-3-8b-chat-hf"
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> LLMResponse:
        """Generate text using Together AI."""
        if not HTTPX_AVAILABLE:
            return LLMResponse("", "", 0, False, "httpx not available")
        
        if not self.api_key:
            return LLMResponse("", "", 0, False, "TOGETHER_API_KEY not set")
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.default_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    return LLMResponse(text, self.default_model, tokens, True)
                else:
                    return LLMResponse("", "", 0, False, response.text)
                    
        except Exception as e:
            return LLMResponse("", "", 0, False, str(e))


class OpenRouterClient:
    """
    OpenRouter client - Access to 100+ models.
    
    Free models available!
    Sign up at: https://openrouter.ai
    Set OPENROUTER_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        # Free models
        self.default_model = "meta-llama/llama-3.2-3b-instruct:free"
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> LLMResponse:
        """Generate text using OpenRouter."""
        if not HTTPX_AVAILABLE:
            return LLMResponse("", "", 0, False, "httpx not available")
        
        if not self.api_key:
            return LLMResponse("", "", 0, False, "OPENROUTER_API_KEY not set")
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.default_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    return LLMResponse(text, self.default_model, tokens, True)
                else:
                    return LLMResponse("", "", 0, False, response.text)
                    
        except Exception as e:
            return LLMResponse("", "", 0, False, str(e))


class GeminiClient:
    """
    Google Gemini API client.
    
    Use Google AI Studio for FREE API access!
    Get your key at: https://aistudio.google.com/apikey
    Set GOOGLE_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.default_model = "gemini-1.5-flash"  # Fast and capable
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> LLMResponse:
        """Generate text using Google Gemini."""
        if not HTTPX_AVAILABLE:
            return LLMResponse("", "", 0, False, "httpx not available")
        
        if not self.api_key:
            return LLMResponse("", "", 0, False, "GOOGLE_API_KEY not set")
        
        try:
            url = f"{self.base_url}/models/{self.default_model}:generateContent?key={self.api_key}"
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "maxOutputTokens": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0]["content"]["parts"][0]["text"]
                        tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
                        return LLMResponse(text, self.default_model, tokens, True)
                    return LLMResponse("", "", 0, False, "No candidates returned")
                else:
                    return LLMResponse("", "", 0, False, response.text)
                    
        except Exception as e:
            return LLMResponse("", "", 0, False, str(e))


class OllamaClient:
    """
    Ollama client - Run models locally with optimization.
    
    Ollama optimizes models to run on limited hardware.
    Install: curl -fsSL https://ollama.com/install.sh | sh
    """
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.default_model = "llama3.2:1b"  # Small but good
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        if not HTTPX_AVAILABLE:
            return False
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.host}/api/version")
                return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> LLMResponse:
        """Generate text using Ollama."""
        if not HTTPX_AVAILABLE:
            return LLMResponse("", "", 0, False, "httpx not available")
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.default_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    text = data.get("response", "")
                    return LLMResponse(text, self.default_model, 0, True)
                else:
                    return LLMResponse("", "", 0, False, response.text)
                    
        except Exception as e:
            return LLMResponse("", "", 0, False, str(e))


class HybridLLM:
    """
    Hybrid LLM that uses best available provider.
    
    Priority:
    1. Gemini (Google AI Studio - FREE!)
    2. Groq (FREE, fast)
    3. Together AI
    4. OpenRouter
    5. Ollama (local optimized)
    6. RWKV local
    7. Simple fallback
    """
    
    def __init__(self):
        self.gemini = GeminiClient()
        self.groq = GroqClient()
        self.together = TogetherClient()
        self.openrouter = OpenRouterClient()
        self.ollama = OllamaClient()
        
        # Try to import local RWKV
        self.rwkv = None
        try:
            from omniagi.language.rwkv_model import LanguageModelManager
            self.rwkv = LanguageModelManager()
        except:
            pass
        
        self.active_provider = self._detect_best_provider()
    
    def _detect_best_provider(self) -> str:
        """Detect best available provider."""
        # Check cloud APIs - Gemini first!
        if self.gemini.api_key:
            return "gemini"
        if self.groq.api_key:
            return "groq"
        if self.together.api_key:
            return "together"
        if self.openrouter.api_key:
            return "openrouter"
        
        # Check local options
        if self.ollama.is_available():
            return "ollama"
        
        if self.rwkv and self.rwkv.model_type == "rwkv":
            return "rwkv"
        
        return "simple"
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate text using best available provider."""
        
        if self.active_provider == "gemini":
            response = self.gemini.generate(prompt, max_tokens, temperature)
            if response.success:
                return response.text
        
        if self.active_provider == "groq":
            response = self.groq.generate(prompt, max_tokens, temperature)
            if response.success:
                return response.text
        
        if self.active_provider == "together":
            response = self.together.generate(prompt, max_tokens, temperature)
            if response.success:
                return response.text
        
        if self.active_provider == "openrouter":
            response = self.openrouter.generate(prompt, max_tokens, temperature)
            if response.success:
                return response.text
        
        if self.active_provider == "ollama":
            response = self.ollama.generate(prompt, max_tokens, temperature)
            if response.success:
                return response.text
        
        if self.rwkv:
            return self.rwkv.generate(prompt, max_tokens)
        
        return f"[No LLM available. Set GOOGLE_API_KEY for Gemini access]"
    
    def answer(self, question: str, context: str = "") -> str:
        """Answer a question."""
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer concisely:"
        else:
            prompt = f"Question: {question}\n\nAnswer concisely:"
        return self.generate(prompt, max_tokens=150)
    
    def reason(self, problem: str) -> str:
        """Reason through a problem step by step."""
        prompt = f"""Problem: {problem}

Let me solve this step by step:

1."""
        return self.generate(prompt, max_tokens=300)
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider info."""
        return {
            "active_provider": self.active_provider,
            "providers_available": {
                "gemini": bool(self.gemini.api_key),
                "groq": bool(self.groq.api_key),
                "together": bool(self.together.api_key),
                "openrouter": bool(self.openrouter.api_key),
                "ollama": self.ollama.is_available(),
                "rwkv": self.rwkv is not None and self.rwkv.model_type == "rwkv",
            },
            "model": self._get_current_model(),
        }
    
    def _get_current_model(self) -> str:
        """Get current model name."""
        if self.active_provider == "gemini":
            return self.gemini.default_model
        if self.active_provider == "groq":
            return self.groq.default_model
        if self.active_provider == "together":
            return self.together.default_model
        if self.active_provider == "openrouter":
            return self.openrouter.default_model
        if self.active_provider == "ollama":
            return self.ollama.default_model
        return "local"


def setup_instructions():
    """Print setup instructions for cloud APIs."""
    return """
╔══════════════════════════════════════════════════════════════╗
║          🚀 CLOUD LLM SETUP (FREE OPTIONS)                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Option 1: GROQ (Recommended - FREE & FAST)                  ║
║  ─────────────────────────────────────────                   ║
║  1. Go to: https://console.groq.com                          ║
║  2. Sign up (free)                                           ║
║  3. Create API key                                           ║
║  4. Run: export GROQ_API_KEY="your-key"                      ║
║                                                              ║
║  Models available: LLaMA-3.1-70B, Mixtral-8x7B (FREE!)       ║
║                                                              ║
║──────────────────────────────────────────────────────────────║
║                                                              ║
║  Option 2: Together AI (Free tier)                           ║
║  ─────────────────────────────────                           ║
║  1. Go to: https://together.ai                               ║
║  2. Sign up and get API key                                  ║
║  3. Run: export TOGETHER_API_KEY="your-key"                  ║
║                                                              ║
║──────────────────────────────────────────────────────────────║
║                                                              ║
║  Option 3: Ollama (Local optimized)                          ║
║  ────────────────────────────────                            ║
║  1. Install: curl -fsSL https://ollama.com/install.sh | sh   ║
║  2. Run: ollama pull llama3.2:1b                             ║
║  3. The AGI will detect it automatically                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(setup_instructions())
    
    llm = HybridLLM()
    print(f"\nActive provider: {llm.active_provider}")
    print(f"Info: {llm.get_info()}")
